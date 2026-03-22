/*
 * ternary_v3.c — Optimized adder-only ternary CNN.
 *
 * Key optimizations over v2:
 *   1. No im2col patch copy — direct accumulation from input
 *   2. Zero-skipping: explicit branch to skip ~35-45% of zero weights
 *   3. Cache-friendly output-stationary loop order
 *   4. Compiler hint: restrict pointers
 *   5. Branchless add/sub for non-zero weights (+1/-1 only)
 *
 * Compile: gcc -O3 -march=native -ffast-math -funroll-loops -shared -fPIC \
 *          -o libternary_v3.so ternary_v3.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define FP_BITS 8
#define FP_SCALE 256

static inline int16_t f2fp(float v) {
    int32_t x = (int32_t)(v * FP_SCALE);
    return (int16_t)(x > 32767 ? 32767 : (x < -32768 ? -32768 : x));
}

static inline float fp2f(int32_t v) { return (float)v * (1.0f / FP_SCALE); }

/* ── Direct ternary conv2d — no im2col, with zero-skipping ────
 * Accumulates directly from input, skipping the patch gather.
 * Weights in {-1, 0, +1}: zero weights skipped entirely (~35-45%),
 * non-zero weights use pure add/sub (no multiply).
 * One int32 accumulator per output element.
 */
static void tconv2d(
    const int16_t *restrict input,  /* [in_ch, H, W] */
    const int8_t  *restrict w,      /* [out_ch, in_ch, kH, kW] in {-1,0,+1} */
    float alpha,
    const float *restrict bias,
    int16_t *restrict output,       /* [out_ch, outH, outW] */
    int in_ch, int H, int W,
    int out_ch, int kH, int kW,
    int pad)
{
    int outH = H + 2*pad - kH + 1;
    int outW = W + 2*pad - kW + 1;

    for (int o = 0; o < out_ch; o++) {
        float b = bias[o];
        const int8_t *wo = w + o * in_ch * kH * kW;

        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                int32_t acc = 0;
                int wi = 0;

                for (int i = 0; i < in_ch; i++) {
                    const int16_t *xi = input + i * H * W;
                    for (int ky = 0; ky < kH; ky++) {
                        int iy = oy + ky - pad;
                        if (iy < 0 || iy >= H) { wi += kW; continue; }
                        for (int kx = 0; kx < kW; kx++, wi++) {
                            int8_t wv = wo[wi];
                            /* Zero-skipping: ~35-45% of ternary weights are 0 */
                            if (wv == 0) continue;
                            int ix = ox + kx - pad;
                            if (ix < 0 || ix >= W) continue;
                            /* wv is +1 or -1: pure add or sub, no multiply */
                            int32_t xval = (int32_t)xi[iy * W + ix];
                            acc += (wv > 0) ? xval : -xval;
                        }
                    }
                }

                /* Alpha + bias: only float ops per output */
                float val = fp2f(acc) * alpha + b;
                output[(o * outH + oy) * outW + ox] = f2fp(val);
            }
        }
    }
}

/* ── Fused BN + ReLU ─────────────────────────────────────── */
static void bn_relu(int16_t *data, const float *scale, const float *shift,
                    int ch, int HW) {
    for (int c = 0; c < ch; c++) {
        float s = scale[c], b = shift[c];
        int16_t *row = data + c * HW;
        for (int j = 0; j < HW; j++) {
            float v = fp2f(row[j]) * s + b;
            row[j] = (v > 0) ? f2fp(v) : 0;
        }
    }
}

/* ── MaxPool2d 2x2 ─────────────────────────────────────────── */
static void maxpool2(const int16_t *in, int16_t *out, int ch, int H, int W) {
    int oH = H/2, oW = W/2;
    for (int c = 0; c < ch; c++) {
        const int16_t *ic = in + c*H*W;
        int16_t *oc = out + c*oH*oW;
        for (int y = 0; y < oH; y++) {
            for (int x = 0; x < oW; x++) {
                int16_t a = ic[(y*2)*W + x*2];
                int16_t b = ic[(y*2)*W + x*2+1];
                int16_t c_ = ic[(y*2+1)*W + x*2];
                int16_t d = ic[(y*2+1)*W + x*2+1];
                int16_t m = a;
                if (b > m) m = b;
                if (c_ > m) m = c_;
                if (d > m) m = d;
                oc[y*oW+x] = m;
            }
        }
    }
}

/* ── Ternary FC ──────────────────────────────────────────── */
static void tfc(const int16_t *in, const int8_t *w, float alpha,
                const float *bias, int16_t *out, int n_in, int n_out) {
    for (int o = 0; o < n_out; o++) {
        int32_t acc = 0;
        const int8_t *wo = w + o * n_in;
        for (int i = 0; i < n_in; i++) {
            int8_t wv = wo[i];
            if (wv == 0) continue;  /* zero-skipping: skip ~35-45% of weights */
            int32_t xval = (int32_t)in[i];
            acc += (wv > 0) ? xval : -xval;  /* pure add/sub, no multiply */
        }
        out[o] = f2fp(fp2f(acc) * alpha + bias[o]);
    }
}

/* ── FP FC (hybrid) ──────────────────────────────────────── */
static void fpfc(const int16_t *in, const float *w, const float *bias,
                 int16_t *out, int n_in, int n_out) {
    for (int o = 0; o < n_out; o++) {
        float acc = bias[o];
        const float *wo = w + o * n_in;
        for (int i = 0; i < n_in; i++)
            acc += wo[i] * fp2f(in[i]);
        out[o] = f2fp(acc);
    }
}

static void relu16(int16_t *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

/* ── Benchmark: full forward in C, zero Python overhead ──── */
void bench_v3(
    /* Conv1 */
    const int8_t *c1w, float c1a, const float *c1b,
    const float *bn1s, const float *bn1sh, int c1_out,
    /* Conv2 */
    const int8_t *c2w, float c2a, const float *c2b,
    const float *bn2s, const float *bn2sh, int c2_out,
    /* FC1 */
    const int8_t *fc1w, float fc1a, const float *fc1b,
    const float *fc1fpw, int fc1_out,
    /* FC2 */
    const int8_t *fc2w, float fc2a, const float *fc2b,
    const float *fc2fpw, int fc2_out,
    /* Config */
    const float *input_f, int is_hybrid,
    int n_warmup, int n_runs,
    float *lats, int *pred)
{
    int16_t inp[28*28];
    for (int i = 0; i < 784; i++) inp[i] = f2fp(input_f[i]);

    int16_t a[64*28*28], b[64*28*28];

    #define FWD() do { \
        tconv2d(inp, c1w, c1a, c1b, a, 1,28,28, c1_out,3,3, 1); \
        bn_relu(a, bn1s, bn1sh, c1_out, 28*28); \
        maxpool2(a, b, c1_out, 28, 28); \
        tconv2d(b, c2w, c2a, c2b, a, c1_out,14,14, c2_out,3,3, 1); \
        bn_relu(a, bn2s, bn2sh, c2_out, 14*14); \
        maxpool2(a, b, c2_out, 14, 14); \
        int flat = c2_out * 49; \
        if (is_hybrid && fc1fpw) fpfc(b, fc1fpw, fc1b, a, flat, fc1_out); \
        else tfc(b, fc1w, fc1a, fc1b, a, flat, fc1_out); \
        relu16(a, fc1_out); \
        if (is_hybrid && fc2fpw) fpfc(a, fc2fpw, fc2b, b, fc1_out, fc2_out); \
        else tfc(a, fc2w, fc2a, fc2b, b, fc1_out, fc2_out); \
    } while(0)

    for (int i = 0; i < n_warmup; i++) FWD();

    struct timespec t1, t2;
    for (int i = 0; i < n_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        FWD();
        clock_gettime(CLOCK_MONOTONIC, &t2);
        lats[i] = (float)((t2.tv_sec-t1.tv_sec)*1e3 + (t2.tv_nsec-t1.tv_nsec)/1e6);
    }

    int best = 0;
    for (int i = 1; i < fc2_out; i++) if (b[i] > b[best]) best = i;
    *pred = best;
    #undef FWD
}
