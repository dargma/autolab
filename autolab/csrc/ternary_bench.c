/*
 * ternary_bench.c — Self-contained ternary CNN benchmark.
 *
 * Loads weights from binary files, runs N inferences, reports latency.
 * Single C call = full forward pass, zero Python overhead in hot loop.
 *
 * Compile: gcc -O3 -march=native -ffast-math -o ternary_bench ternary_bench.c -lm
 * Usage:   ./ternary_bench <weights_dir> <n_runs>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Ternary Conv2d ─────────────────────────────────────────── */
static void ternary_conv2d(
    const float *input, const signed char *weights, const float *bias,
    float alpha, float *output,
    int in_ch, int H, int W, int out_ch, int kH, int kW, int pad, int stride)
{
    int outH = (H + 2*pad - kH) / stride + 1;
    int outW = (W + 2*pad - kW) / stride + 1;
    for (int o = 0; o < out_ch; o++) {
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                float acc = 0.0f;
                for (int i = 0; i < in_ch; i++) {
                    for (int ky = 0; ky < kH; ky++) {
                        for (int kx = 0; kx < kW; kx++) {
                            signed char w = weights[((o*in_ch+i)*kH+ky)*kW+kx];
                            if (w == 0) continue;
                            int iy = oy*stride + ky - pad;
                            int ix = ox*stride + kx - pad;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                            float val = input[(i*H+iy)*W+ix];
                            if (w > 0) acc += val; else acc -= val;
                        }
                    }
                }
                output[(o*outH+oy)*outW+ox] = acc * alpha + bias[o];
            }
        }
    }
}

static void batchnorm_relu(float *data, const float *gamma, const float *beta,
                           const float *mean, const float *var, int ch, int H, int W)
{
    for (int c = 0; c < ch; c++) {
        float s = gamma[c] / sqrtf(var[c] + 1e-5f);
        float b = beta[c] - mean[c] * s;
        for (int j = 0; j < H*W; j++) {
            float v = data[c*H*W+j] * s + b;
            data[c*H*W+j] = v > 0 ? v : 0;  /* fused ReLU */
        }
    }
}

static void maxpool2(const float *in, float *out, int ch, int H, int W) {
    int oH = H/2, oW = W/2;
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oH; y++)
            for (int x = 0; x < oW; x++) {
                int iy = y*2, ix = x*2;
                float m = in[(c*H+iy)*W+ix];
                float v;
                v = in[(c*H+iy)*W+ix+1]; if(v>m) m=v;
                v = in[(c*H+iy+1)*W+ix]; if(v>m) m=v;
                v = in[(c*H+iy+1)*W+ix+1]; if(v>m) m=v;
                out[(c*oH+y)*oW+x] = m;
            }
}

static void ternary_fc(const float *in, const signed char *w, const float *bias,
                       float alpha, float *out, int n_in, int n_out) {
    for (int o = 0; o < n_out; o++) {
        float acc = 0;
        for (int i = 0; i < n_in; i++) {
            signed char wv = w[o*n_in+i];
            if (wv > 0) acc += in[i]; else if (wv < 0) acc -= in[i];
        }
        out[o] = acc * alpha + bias[o];
    }
}

static void fp_fc(const float *in, const float *w, const float *bias,
                  float *out, int n_in, int n_out) {
    for (int o = 0; o < n_out; o++) {
        float acc = bias[o];
        for (int i = 0; i < n_in; i++) acc += w[o*n_in+i] * in[i];
        out[o] = acc;
    }
}

static void relu(float *x, int n) { for(int i=0;i<n;i++) if(x[i]<0) x[i]=0; }

/* ── Model struct ───────────────────────────────────────────── */
typedef struct {
    int c1, c2, fc1_n, fc2_n;
    signed char *conv1_w, *conv2_w, *fc1_tw, *fc2_tw;
    float conv1_a, conv2_a, fc1_a, fc2_a;
    float *conv1_b, *conv2_b, *fc1_b, *fc2_b;
    float *bn1_g, *bn1_b, *bn1_m, *bn1_v;
    float *bn2_g, *bn2_b, *bn2_m, *bn2_v;
    float *fc1_fpw, *fc2_fpw; /* non-NULL = hybrid */
} Model;

static int forward(const Model *m, const float *input, float *logits) {
    float a[64*28*28], b[64*28*28];
    int H=28, W=28;

    ternary_conv2d(input, m->conv1_w, m->conv1_b, m->conv1_a, a, 1,H,W, m->c1,3,3,1,1);
    batchnorm_relu(a, m->bn1_g, m->bn1_b, m->bn1_m, m->bn1_v, m->c1,H,W);
    maxpool2(a,b,m->c1,H,W); H/=2; W/=2;

    ternary_conv2d(b, m->conv2_w, m->conv2_b, m->conv2_a, a, m->c1,H,W, m->c2,3,3,1,1);
    batchnorm_relu(a, m->bn2_g, m->bn2_b, m->bn2_m, m->bn2_v, m->c2,H,W);
    maxpool2(a,b,m->c2,H,W); H/=2; W/=2;

    int flat = m->c2*H*W;
    if (m->fc1_fpw) fp_fc(b, m->fc1_fpw, m->fc1_b, a, flat, m->fc1_n);
    else ternary_fc(b, m->fc1_tw, m->fc1_b, m->fc1_a, a, flat, m->fc1_n);
    relu(a, m->fc1_n);

    if (m->fc2_fpw) fp_fc(a, m->fc2_fpw, m->fc2_b, logits, m->fc1_n, m->fc2_n);
    else ternary_fc(a, m->fc2_tw, m->fc2_b, m->fc2_a, logits, m->fc1_n, m->fc2_n);

    int best=0;
    for(int i=1;i<m->fc2_n;i++) if(logits[i]>logits[best]) best=i;
    return best;
}

/* ── File I/O helpers ──────────────────────────────────────── */
static void *load_bin(const char *dir, const char *name, size_t bytes) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    void *buf = malloc(bytes);
    fread(buf, 1, bytes, f);
    fclose(f);
    return buf;
}

static float load_float(const char *dir, const char *name) {
    float v;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fread(&v, sizeof(float), 1, f);
    fclose(f);
    return v;
}

/* ── Benchmark entry point (called from Python via ctypes) ── */
void benchmark_forward(
    /* Conv1 */
    const signed char *c1w, float c1a, const float *c1b,
    const float *bn1g, const float *bn1b, const float *bn1m, const float *bn1v,
    int c1_out,
    /* Conv2 */
    const signed char *c2w, float c2a, const float *c2b,
    const float *bn2g, const float *bn2b, const float *bn2m, const float *bn2v,
    int c2_out,
    /* FC1 */
    const signed char *fc1tw, float fc1a, const float *fc1b,
    const float *fc1fpw, int fc1_out,
    /* FC2 */
    const signed char *fc2tw, float fc2a, const float *fc2b,
    const float *fc2fpw, int fc2_out,
    /* Input & config */
    const float *input,
    int n_warmup, int n_runs,
    /* Output */
    float *latencies_out)
{
    Model m;
    m.c1 = c1_out; m.c2 = c2_out; m.fc1_n = fc1_out; m.fc2_n = fc2_out;
    m.conv1_w = (signed char*)c1w; m.conv1_a = c1a; m.conv1_b = (float*)c1b;
    m.bn1_g = (float*)bn1g; m.bn1_b = (float*)bn1b;
    m.bn1_m = (float*)bn1m; m.bn1_v = (float*)bn1v;
    m.conv2_w = (signed char*)c2w; m.conv2_a = c2a; m.conv2_b = (float*)c2b;
    m.bn2_g = (float*)bn2g; m.bn2_b = (float*)bn2b;
    m.bn2_m = (float*)bn2m; m.bn2_v = (float*)bn2v;
    m.fc1_tw = (signed char*)fc1tw; m.fc1_a = fc1a; m.fc1_b = (float*)fc1b;
    m.fc1_fpw = (float*)fc1fpw;
    m.fc2_tw = (signed char*)fc2tw; m.fc2_a = fc2a; m.fc2_b = (float*)fc2b;
    m.fc2_fpw = (float*)fc2fpw;

    float logits[64];

    /* Warmup */
    for (int i = 0; i < n_warmup; i++)
        forward(&m, input, logits);

    /* Timed runs */
    struct timespec t1, t2;
    for (int i = 0; i < n_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        forward(&m, input, logits);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        latencies_out[i] = (t2.tv_sec - t1.tv_sec) * 1000.0f +
                           (t2.tv_nsec - t1.tv_nsec) / 1e6f;
    }
}
