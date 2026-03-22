/*
 * ternary_inference.c — Ternary weight CNN inference on CPU
 *
 * Weights are stored as int8 {-1, 0, +1} with a float alpha per layer.
 * Conv2d: w*x becomes sign(w)*alpha*x → additions/subtractions only.
 * This eliminates ALL floating-point multiplications in conv layers.
 *
 * Compile: gcc -O3 -march=native -ffast-math -shared -fPIC \
 *          -o libternary.so ternary_inference.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Ternary Conv2d ──────────────────────────────────────────────
 * weights: int8_t[out_ch][in_ch][kH][kW] in {-1, 0, +1}
 * alpha: float scaling factor (one per layer)
 * bias: float[out_ch] (can be NULL)
 * input: float[in_ch][H][W]
 * output: float[out_ch][outH][outW]
 *
 * For ternary: out[o][y][x] = alpha * sum_{i,ky,kx} sign(w) * in[i][y+ky][x+kx]
 * where sign(w) in {-1,0,+1}. Zeros are skipped entirely.
 */
void ternary_conv2d(
    const float *input,     /* [in_ch, H, W] */
    const int8_t *weights,  /* [out_ch, in_ch, kH, kW] */
    const float *bias,      /* [out_ch] or NULL */
    float alpha,
    float *output,          /* [out_ch, outH, outW] */
    int in_ch, int H, int W,
    int out_ch, int kH, int kW,
    int pad, int stride)
{
    int outH = (H + 2 * pad - kH) / stride + 1;
    int outW = (W + 2 * pad - kW) / stride + 1;

    for (int o = 0; o < out_ch; o++) {
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                float acc = 0.0f;
                for (int i = 0; i < in_ch; i++) {
                    for (int ky = 0; ky < kH; ky++) {
                        for (int kx = 0; kx < kW; kx++) {
                            int8_t w = weights[((o * in_ch + i) * kH + ky) * kW + kx];
                            if (w == 0) continue;  /* skip zeros — sparsity win */
                            int iy = oy * stride + ky - pad;
                            int ix = ox * stride + kx - pad;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                            float val = input[(i * H + iy) * W + ix];
                            /* No multiplication! Just add or subtract */
                            if (w > 0) acc += val;
                            else       acc -= val;
                        }
                    }
                }
                acc *= alpha;
                if (bias) acc += bias[o];
                output[(o * outH + oy) * outW + ox] = acc;
            }
        }
    }
}

/* ── BatchNorm (fused into scale+bias for inference) ─────────── */
void batchnorm2d(
    float *data,            /* [ch, H, W] — in-place */
    const float *gamma,     /* [ch] */
    const float *beta,      /* [ch] */
    const float *mean,      /* [ch] */
    const float *var,       /* [ch] */
    float eps,
    int ch, int H, int W)
{
    for (int c = 0; c < ch; c++) {
        float scale = gamma[c] / sqrtf(var[c] + eps);
        float shift = beta[c] - mean[c] * scale;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = (c * H + y) * W + x;
                data[idx] = data[idx] * scale + shift;
            }
        }
    }
}

/* ── ReLU (in-place) ─────────────────────────────────────────── */
void relu_inplace(float *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

/* ── MaxPool2d (2x2, stride 2) ───────────────────────────────── */
void maxpool2d(
    const float *input,     /* [ch, H, W] */
    float *output,          /* [ch, H/2, W/2] */
    int ch, int H, int W)
{
    int outH = H / 2;
    int outW = W / 2;
    for (int c = 0; c < ch; c++) {
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                int iy = oy * 2;
                int ix = ox * 2;
                float m = input[(c * H + iy) * W + ix];
                float v;
                v = input[(c * H + iy) * W + ix + 1]; if (v > m) m = v;
                v = input[(c * H + iy + 1) * W + ix]; if (v > m) m = v;
                v = input[(c * H + iy + 1) * W + ix + 1]; if (v > m) m = v;
                output[(c * outH + oy) * outW + ox] = m;
            }
        }
    }
}

/* ── Ternary Linear ──────────────────────────────────────────── */
void ternary_linear(
    const float *input,     /* [in_features] */
    const int8_t *weights,  /* [out_features, in_features] */
    const float *bias,      /* [out_features] or NULL */
    float alpha,
    float *output,          /* [out_features] */
    int in_features, int out_features)
{
    for (int o = 0; o < out_features; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in_features; i++) {
            int8_t w = weights[o * in_features + i];
            if (w == 0) continue;
            if (w > 0) acc += input[i];
            else       acc -= input[i];
        }
        acc *= alpha;
        if (bias) acc += bias[o];
        output[o] = acc;
    }
}

/* ── Full-precision Linear (for hybrid model's FC head) ──────── */
void fp_linear(
    const float *input,     /* [in_features] */
    const float *weights,   /* [out_features, in_features] */
    const float *bias,      /* [out_features] or NULL */
    float *output,          /* [out_features] */
    int in_features, int out_features)
{
    for (int o = 0; o < out_features; o++) {
        float acc = (bias) ? bias[o] : 0.0f;
        for (int i = 0; i < in_features; i++) {
            acc += weights[o * in_features + i] * input[i];
        }
        output[o] = acc;
    }
}

/* ── Full ternary CNN inference pipeline ─────────────────────────
 * Architecture: [1,C1,C2] conv + BN + ReLU + Pool -> FC -> out
 * For a 28x28 input with 2 conv layers:
 *   28 -> conv+pool -> 14 -> conv+pool -> 7
 *   flatten -> FC(C2*7*7, fc_out) -> ReLU -> FC(fc_out, 10)
 */

typedef struct {
    /* Conv layer 1 */
    int8_t *conv1_w;    float conv1_alpha;  float *conv1_bias;
    float *bn1_gamma, *bn1_beta, *bn1_mean, *bn1_var;
    int conv1_out_ch;

    /* Conv layer 2 */
    int8_t *conv2_w;    float conv2_alpha;  float *conv2_bias;
    float *bn2_gamma, *bn2_beta, *bn2_mean, *bn2_var;
    int conv2_out_ch;

    /* FC layer 1 */
    int8_t *fc1_w;      float fc1_alpha;    float *fc1_bias;
    float *fc1_fp_w;    /* non-NULL for hybrid (full-precision FC) */
    int fc1_out;

    /* FC layer 2 (output) */
    int8_t *fc2_w;      float fc2_alpha;    float *fc2_bias;
    float *fc2_fp_w;    /* non-NULL for hybrid */
    int fc2_out;

    int is_hybrid;      /* 1 = hybrid (FP fc), 0 = full ternary */
} TernaryCNNModel;

/* Allocate intermediate buffers and run full forward pass.
 * input: float[1][28][28], output: float[num_classes]
 * Returns pointer to output (caller-owned buf). */
void ternary_cnn_forward(
    const TernaryCNNModel *m,
    const float *input,     /* [1, 28, 28] */
    float *output)          /* [fc2_out] */
{
    /* Buffers — stack allocated for small models */
    float buf_a[64 * 28 * 28];  /* max channels * max spatial */
    float buf_b[64 * 28 * 28];

    int H = 28, W = 28;

    /* Conv1: [1, 28, 28] -> [C1, 28, 28] -> BN -> ReLU -> Pool -> [C1, 14, 14] */
    ternary_conv2d(input, m->conv1_w, m->conv1_bias, m->conv1_alpha,
                   buf_a, 1, H, W, m->conv1_out_ch, 3, 3, 1, 1);
    batchnorm2d(buf_a, m->bn1_gamma, m->bn1_beta, m->bn1_mean, m->bn1_var,
                1e-5f, m->conv1_out_ch, H, W);
    relu_inplace(buf_a, m->conv1_out_ch * H * W);
    maxpool2d(buf_a, buf_b, m->conv1_out_ch, H, W);
    H /= 2; W /= 2;  /* 14x14 */

    /* Conv2: [C1, 14, 14] -> [C2, 14, 14] -> BN -> ReLU -> Pool -> [C2, 7, 7] */
    ternary_conv2d(buf_b, m->conv2_w, m->conv2_bias, m->conv2_alpha,
                   buf_a, m->conv1_out_ch, H, W, m->conv2_out_ch, 3, 3, 1, 1);
    batchnorm2d(buf_a, m->bn2_gamma, m->bn2_beta, m->bn2_mean, m->bn2_var,
                1e-5f, m->conv2_out_ch, H, W);
    relu_inplace(buf_a, m->conv2_out_ch * H * W);
    maxpool2d(buf_a, buf_b, m->conv2_out_ch, H, W);
    H /= 2; W /= 2;  /* 7x7 */

    /* Flatten: [C2, 7, 7] -> [C2*49] — buf_b is already flat in memory */
    int flat_size = m->conv2_out_ch * H * W;

    /* FC1 -> ReLU */
    if (m->is_hybrid && m->fc1_fp_w) {
        fp_linear(buf_b, m->fc1_fp_w, m->fc1_bias, buf_a, flat_size, m->fc1_out);
    } else {
        ternary_linear(buf_b, m->fc1_w, m->fc1_bias, m->fc1_alpha,
                       buf_a, flat_size, m->fc1_out);
    }
    relu_inplace(buf_a, m->fc1_out);

    /* FC2 (output, no activation) */
    if (m->is_hybrid && m->fc2_fp_w) {
        fp_linear(buf_a, m->fc2_fp_w, m->fc2_bias, output, m->fc1_out, m->fc2_out);
    } else {
        ternary_linear(buf_a, m->fc2_w, m->fc2_bias, m->fc2_alpha,
                       output, m->fc1_out, m->fc2_out);
    }
}

/* ── Argmax ──────────────────────────────────────────────────── */
int argmax(const float *data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}
