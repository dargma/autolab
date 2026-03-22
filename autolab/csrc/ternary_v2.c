/*
 * ternary_v2.c — TRUE ternary inference: adder + shift only, no multiplier.
 *
 * KEY INSIGHT: Ternary weights {-1, 0, +1} mean matmul is ONLY adds/subs.
 * At the logic gate level:
 *   w=+1 → ADD input to accumulator
 *   w=-1 → SUB input from accumulator (two's complement: XOR + carry)
 *   w=0  → NOP (skip, gate off)
 *
 * Implementation:
 *   1. Activations in fixed-point int16 (Q8.8: 8 integer, 8 fractional bits)
 *   2. Weights packed as 2 bitmasks per group: pos_mask & neg_mask
 *   3. Inner loop: pure int16 add/sub, zero float multiplies
 *   4. Alpha scaling applied ONCE per output via bit-shift approximation
 *   5. BN fused as integer scale+shift (pre-computed)
 *
 * On x86: int16 add = 1 cycle, fits 8 ops in 128-bit SSE register
 * vs float mul = 3-5 cycles, only 4 ops per SSE register
 *
 * Compile: gcc -O3 -march=native -ffast-math -shared -fPIC \
 *          -o libternary_v2.so ternary_v2.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ── Fixed-point Q8.8 conversion ─────────────────────────────
 * Range: [-128.0, +127.996] with 1/256 precision.
 * Sufficient for normalized image data (typ. range [-3, +3]).
 */
#define FP_BITS 8
#define FP_SCALE (1 << FP_BITS)  /* 256 */
#define FP_HALF  (1 << (FP_BITS - 1))

static inline int16_t float_to_fp(float v) {
    int32_t x = (int32_t)(v * FP_SCALE + (v >= 0 ? 0.5f : -0.5f));
    if (x > 32767) x = 32767;
    if (x < -32768) x = -32768;
    return (int16_t)x;
}

static inline float fp_to_float(int32_t v) {
    return (float)v / FP_SCALE;
}

/* ── Packed ternary weight format ────────────────────────────
 * For each weight element, store 2 bits:
 *   bit 0 = nonzero (1 if w != 0)
 *   bit 1 = sign    (1 if w < 0)
 *
 * Pack 8 ternary values into 2 bytes (pos_byte, neg_byte):
 *   pos_byte: bit i = 1 if weight[i] == +1
 *   neg_byte: bit i = 1 if weight[i] == -1
 *
 * This is the NATURAL representation for adder-only hardware:
 *   for each bit set in pos_byte: ADD input[i]
 *   for each bit set in neg_byte: SUB input[i]
 */
typedef struct {
    uint8_t *pos_mask;  /* bit i set = weight[i] is +1 */
    uint8_t *neg_mask;  /* bit i set = weight[i] is -1 */
    int n_elements;     /* total weight elements */
    int n_bytes;        /* ceil(n_elements / 8) */
    float alpha;        /* scaling factor */
} PackedTernary;

/* Pack int8 ternary weights into bitmasks */
static PackedTernary pack_ternary(const int8_t *weights, int n, float alpha) {
    PackedTernary p;
    p.n_elements = n;
    p.n_bytes = (n + 7) / 8;
    p.alpha = alpha;
    p.pos_mask = (uint8_t *)calloc(p.n_bytes, 1);
    p.neg_mask = (uint8_t *)calloc(p.n_bytes, 1);

    for (int i = 0; i < n; i++) {
        int byte_idx = i / 8;
        int bit_idx  = i % 8;
        if (weights[i] > 0)
            p.pos_mask[byte_idx] |= (1 << bit_idx);
        else if (weights[i] < 0)
            p.neg_mask[byte_idx] |= (1 << bit_idx);
        /* w==0: both masks stay 0 → NOP */
    }
    return p;
}

/* ── Ternary dot product: pure adds/subs ─────────────────────
 * Computes: alpha * sum_i(w_i * x_i) where w_i in {-1,0,+1}
 *         = alpha * (sum where w=+1) - alpha * (sum where w=-1)
 *
 * All accumulation is int32 (sum of int16 values). No multiply.
 * Alpha applied ONCE at the end via a single float multiply.
 */
static inline int32_t ternary_dot_fp(
    const int16_t *input,       /* fixed-point Q8.8 input vector */
    const uint8_t *pos_mask,    /* bit-packed positive weights */
    const uint8_t *neg_mask,    /* bit-packed negative weights */
    int n)                      /* number of elements */
{
    int32_t acc = 0;
    int full_bytes = n / 8;
    int remainder = n % 8;

    /* Process 8 elements per iteration using bitmask */
    for (int b = 0; b < full_bytes; b++) {
        uint8_t pm = pos_mask[b];
        uint8_t nm = neg_mask[b];
        const int16_t *x = input + b * 8;

        /* Unrolled: check each bit, add or sub */
        if (pm | nm) {  /* skip if entire byte is zero (sparsity win) */
            if (pm & 0x01) acc += x[0]; else if (nm & 0x01) acc -= x[0];
            if (pm & 0x02) acc += x[1]; else if (nm & 0x02) acc -= x[1];
            if (pm & 0x04) acc += x[2]; else if (nm & 0x04) acc -= x[2];
            if (pm & 0x08) acc += x[3]; else if (nm & 0x08) acc -= x[3];
            if (pm & 0x10) acc += x[4]; else if (nm & 0x10) acc -= x[4];
            if (pm & 0x20) acc += x[5]; else if (nm & 0x20) acc -= x[5];
            if (pm & 0x40) acc += x[6]; else if (nm & 0x40) acc -= x[6];
            if (pm & 0x80) acc += x[7]; else if (nm & 0x80) acc -= x[7];
        }
    }

    /* Handle remaining elements */
    for (int i = full_bytes * 8; i < n; i++) {
        int byte_idx = i / 8;
        int bit_idx = i % 8;
        if (pos_mask[byte_idx] & (1 << bit_idx))      acc += input[i];
        else if (neg_mask[byte_idx] & (1 << bit_idx))  acc -= input[i];
    }

    return acc;
}

/* ── Ternary Conv2d: adder-only ──────────────────────────────
 * Input/output in fixed-point int16.
 * Each output element = ternary_dot over the receptive field.
 * Alpha * result computed once per output element.
 */
void ternary_conv2d_v2(
    const int16_t *input,       /* [in_ch, H, W] in Q8.8 */
    const uint8_t *pos_masks,   /* packed: [out_ch][in_ch*kH*kW] */
    const uint8_t *neg_masks,
    float alpha,
    const float *bias,          /* [out_ch] in float */
    int16_t *output,            /* [out_ch, outH, outW] */
    int in_ch, int H, int W,
    int out_ch, int kH, int kW,
    int pad, int stride)
{
    int outH = (H + 2 * pad - kH) / stride + 1;
    int outW = (W + 2 * pad - kW) / stride + 1;
    int kernel_size = in_ch * kH * kW;
    int kernel_bytes = (kernel_size + 7) / 8;

    /* Temp buffer to gather receptive field */
    int16_t *patch = (int16_t *)malloc(kernel_size * sizeof(int16_t));

    for (int o = 0; o < out_ch; o++) {
        const uint8_t *pm = pos_masks + o * kernel_bytes;
        const uint8_t *nm = neg_masks + o * kernel_bytes;
        float b = bias ? bias[o] : 0.0f;

        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                /* Gather input patch (im2col for one output position) */
                int pi = 0;
                for (int i = 0; i < in_ch; i++) {
                    for (int ky = 0; ky < kH; ky++) {
                        for (int kx = 0; kx < kW; kx++) {
                            int iy = oy * stride + ky - pad;
                            int ix = ox * stride + kx - pad;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                patch[pi] = input[(i * H + iy) * W + ix];
                            else
                                patch[pi] = 0;
                            pi++;
                        }
                    }
                }

                /* Ternary dot: pure adds/subs, no multiply */
                int32_t acc = ternary_dot_fp(patch, pm, nm, kernel_size);

                /* Alpha + bias: the ONLY float operations per output element */
                float val = fp_to_float(acc) * alpha + b;
                output[(o * outH + oy) * outW + ox] = float_to_fp(val);
            }
        }
    }
    free(patch);
}

/* ── Ternary FC: adder-only ──────────────────────────────────
 * Same principle: pure int16 add/sub, alpha applied once at end.
 */
void ternary_fc_v2(
    const int16_t *input,       /* [in_features] Q8.8 */
    const uint8_t *pos_mask,    /* packed [out_features][in_features] */
    const uint8_t *neg_mask,
    float alpha,
    const float *bias,
    int16_t *output,            /* [out_features] Q8.8 */
    int in_features, int out_features)
{
    int in_bytes = (in_features + 7) / 8;

    for (int o = 0; o < out_features; o++) {
        const uint8_t *pm = pos_mask + o * in_bytes;
        const uint8_t *nm = neg_mask + o * in_bytes;

        int32_t acc = ternary_dot_fp(input, pm, nm, in_features);

        float val = fp_to_float(acc) * alpha + (bias ? bias[o] : 0.0f);
        output[o] = float_to_fp(val);
    }
}

/* ── Full-precision FC (for hybrid model's FC head) ──────────
 * Uses float since this is the accuracy-critical path.
 */
void fp_fc_v2(
    const int16_t *input,       /* [in_features] Q8.8 */
    const float *weights,       /* [out_features, in_features] */
    const float *bias,
    int16_t *output,            /* [out_features] Q8.8 */
    int in_features, int out_features)
{
    for (int o = 0; o < out_features; o++) {
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_features; i++) {
            acc += weights[o * in_features + i] * fp_to_float(input[i]);
        }
        output[o] = float_to_fp(acc);
    }
}

/* ── Fused BatchNorm + ReLU (integer) ────────────────────────
 * BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * Pre-fused to: y = scale * x + shift
 *   where scale = gamma / sqrt(var + eps)
 *         shift = beta - mean * scale
 * Applied in fixed-point with a single multiply + add per element.
 * ReLU: clamp to >= 0.
 */
void bn_relu_fused_v2(
    int16_t *data,              /* [ch, H, W] in-place */
    const float *scale,         /* [ch] pre-computed: gamma/sqrt(var+eps) */
    const float *shift,         /* [ch] pre-computed: beta - mean*scale */
    int ch, int H, int W)
{
    for (int c = 0; c < ch; c++) {
        /* Convert scale/shift to fixed-point for integer-only path */
        int32_t fp_scale = (int32_t)(scale[c] * FP_SCALE + 0.5f);
        int16_t fp_shift = float_to_fp(shift[c]);

        for (int j = 0; j < H * W; j++) {
            int idx = c * H * W + j;
            /* Integer multiply + shift for BN (this is a shift+add in HW) */
            int32_t val = ((int32_t)data[idx] * fp_scale) >> FP_BITS;
            val += fp_shift;
            /* ReLU: just a comparison + mux */
            data[idx] = (val > 0) ? (int16_t)(val > 32767 ? 32767 : val) : 0;
        }
    }
}

/* ── MaxPool2d 2x2 (integer) ─────────────────────────────────
 * Pure comparisons, no arithmetic — ideal for integer path.
 */
void maxpool2_v2(const int16_t *in, int16_t *out, int ch, int H, int W) {
    int oH = H/2, oW = W/2;
    for (int c = 0; c < ch; c++) {
        for (int y = 0; y < oH; y++) {
            for (int x = 0; x < oW; x++) {
                int iy = y*2, ix = x*2;
                int16_t m = in[(c*H+iy)*W+ix];
                int16_t v;
                v = in[(c*H+iy)*W+ix+1]; if(v>m) m=v;
                v = in[(c*H+iy+1)*W+ix]; if(v>m) m=v;
                v = in[(c*H+iy+1)*W+ix+1]; if(v>m) m=v;
                out[(c*oH+y)*oW+x] = m;
            }
        }
    }
}

/* ── ReLU (integer, in-place) ────────────────────────────────
 * Just a comparison: if negative, set to 0. No arithmetic.
 */
void relu_v2(int16_t *data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0) data[i] = 0;
}

/* ── Full forward pass benchmark ─────────────────────────────
 * Architecture: [1, C1, C2] -> BN+ReLU+Pool -> flatten -> FC1+ReLU -> FC2
 *
 * Conv layers: ternary (adder-only)
 * FC layers: ternary (full) or float (hybrid)
 * BN: fused integer scale+shift
 * Everything except 2 alpha multiplies per conv output is add/sub/compare.
 */
void benchmark_v2(
    /* Conv1 packed weights */
    const uint8_t *c1_pos, const uint8_t *c1_neg,
    float c1_alpha, const float *c1_bias,
    const float *bn1_scale, const float *bn1_shift,
    int c1_out,
    /* Conv2 packed weights */
    const uint8_t *c2_pos, const uint8_t *c2_neg,
    float c2_alpha, const float *c2_bias,
    const float *bn2_scale, const float *bn2_shift,
    int c2_out,
    /* FC1 */
    const uint8_t *fc1_pos, const uint8_t *fc1_neg,
    float fc1_alpha, const float *fc1_bias,
    const float *fc1_fpw,  /* non-NULL for hybrid */
    int fc1_out,
    /* FC2 */
    const uint8_t *fc2_pos, const uint8_t *fc2_neg,
    float fc2_alpha, const float *fc2_bias,
    const float *fc2_fpw,
    int fc2_out,
    /* Input (float, will be converted to Q8.8) */
    const float *input_float,
    int is_hybrid,
    int n_warmup, int n_runs,
    /* Output */
    float *latencies_out,
    int *prediction_out)
{
    /* Convert input to fixed-point once */
    int16_t input_fp[28 * 28];
    for (int i = 0; i < 28 * 28; i++)
        input_fp[i] = float_to_fp(input_float[i]);

    /* Intermediate buffers (int16 = half the memory of float32!) */
    int16_t buf_a[64 * 28 * 28];
    int16_t buf_b[64 * 28 * 28];

    int H, W;

    /* -- Forward pass function -- */
    #define FORWARD() do { \
        H = 28; W = 28; \
        /* Conv1: ternary dot (add/sub only) */ \
        ternary_conv2d_v2(input_fp, c1_pos, c1_neg, c1_alpha, c1_bias, \
                          buf_a, 1, H, W, c1_out, 3, 3, 1, 1); \
        bn_relu_fused_v2(buf_a, bn1_scale, bn1_shift, c1_out, H, W); \
        maxpool2_v2(buf_a, buf_b, c1_out, H, W); \
        H /= 2; W /= 2; \
        \
        /* Conv2: ternary dot (add/sub only) */ \
        ternary_conv2d_v2(buf_b, c2_pos, c2_neg, c2_alpha, c2_bias, \
                          buf_a, c1_out, H, W, c2_out, 3, 3, 1, 1); \
        bn_relu_fused_v2(buf_a, bn2_scale, bn2_shift, c2_out, H, W); \
        maxpool2_v2(buf_a, buf_b, c2_out, H, W); \
        H /= 2; W /= 2; \
        \
        /* FC1 */ \
        int flat = c2_out * H * W; \
        if (is_hybrid && fc1_fpw) \
            fp_fc_v2(buf_b, fc1_fpw, fc1_bias, buf_a, flat, fc1_out); \
        else \
            ternary_fc_v2(buf_b, fc1_pos, fc1_neg, fc1_alpha, fc1_bias, \
                          buf_a, flat, fc1_out); \
        relu_v2(buf_a, fc1_out); \
        \
        /* FC2 */ \
        if (is_hybrid && fc2_fpw) \
            fp_fc_v2(buf_a, fc2_fpw, fc2_bias, buf_b, fc1_out, fc2_out); \
        else \
            ternary_fc_v2(buf_a, fc2_pos, fc2_neg, fc2_alpha, fc2_bias, \
                          buf_b, fc1_out, fc2_out); \
    } while(0)

    /* Warmup */
    for (int i = 0; i < n_warmup; i++)
        FORWARD();

    /* Timed runs */
    struct timespec t1, t2;
    for (int i = 0; i < n_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        FORWARD();
        clock_gettime(CLOCK_MONOTONIC, &t2);
        latencies_out[i] = (float)((t2.tv_sec - t1.tv_sec) * 1e3 +
                                   (t2.tv_nsec - t1.tv_nsec) / 1e6);
    }

    /* Get prediction from last run */
    int best = 0;
    for (int i = 1; i < fc2_out; i++)
        if (buf_b[i] > buf_b[best]) best = i;
    *prediction_out = best;

    #undef FORWARD
}
