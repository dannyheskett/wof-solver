// Forward pass of scripts/14_vision/decoder_net.py. A ReLU CNN (batch norm
// already folded into each conv) gives features at 1/8, 1/16 and 1/32 scale;
// the coarser two are projected, upsampled (nearest) and added to the 1/8
// ones, giving 45 x 80 tokens of width d. 77 learned queries then pass
// through the decoder layers (pre-norm self-attention among the queries,
// cross-attention to the tokens, a GELU MLP), a final LayerNorm and the three
// heads. GELU is the exact erf form, LayerNorm eps 1e-5, as in PyTorch.
#include "decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char DEC_CLASS_CHARS[DEC_CLASSES + 1] = "#_*ABCDEFGHIJKLMNOPQRSTUVWXYZ-&'.!?,~";
static const char CAT_CHARS[DEC_CAT_CHARS + 1] = "\0 ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&?!'.-/";

#define D_MODEL 128
#define HEADS 4
#define LAYERS 3
#define N_Q (DEC_SQUARES + DEC_CAT_LEN + 1)
#define MEAN 0.45f
#define STD 0.25f

typedef struct {
    const char *name;
    int ndim, dim[4];
    float *v;
} Tensor;

typedef struct {
    const float *w, *b;
    int cout, cin, k;
} Conv;

typedef struct {
    Conv a, b, skip;
    int has_skip, stride;
} Block;

// A linear layer with its weight stored transposed, [in][out], so y = xW^T + b
// accumulates along out (contiguous) and vectorises without reordering sums.
typedef struct {
    float *wt;
    const float *b;
    int in, out;
} Lin;

typedef struct {
    Lin q, k, v, o;
} Attn;

typedef struct {
    const float *n1w, *n1b, *n2w, *n2b, *n3w, *n3b;
    Attn self, cross;
    Lin fc, proj;
} DecLayer;

struct Decoder {
    Tensor *t;
    int nt;
    char *names;
    Conv stem, lat8, lat16, lat32;
    Block blocks[8];   // s1.0 s1.1 s2.0 s2.1 s3.0 s3.1 s4.0 s4.1
    const float *pos, *query, *qpos, *norm_w, *norm_b;
    DecLayer layer[LAYERS];
    Lin sq, cat, pr;
    float *owned[64];   // transposed weights, freed with the decoder
    int nowned;
};

// ------------------------------------------------------------------ load --

static const Tensor *find(const Decoder *d, const char *name) {
    for (int i = 0; i < d->nt; i++)
        if (!strcmp(d->t[i].name, name)) return &d->t[i];
    return NULL;
}

// The tensor called name with exactly these dimensions (unused ones 0), or NULL.
static const float *get(const Decoder *d, const char *name, int a, int b, int c, int e, int *ok) {
    const Tensor *t = find(d, name);
    int want[4] = {a, b, c, e}, n = 0;
    while (n < 4 && want[n]) n++;
    if (!t || t->ndim != n) { *ok = 0; return NULL; }
    for (int i = 0; i < n; i++)
        if (t->dim[i] != want[i]) { *ok = 0; return NULL; }
    return t->v;
}

static Conv conv(const Decoder *d, const char *name, int cout, int cin, int k, int *ok) {
    char w[64], b[64];
    snprintf(w, sizeof w, "%s.w", name);
    snprintf(b, sizeof b, "%s.b", name);
    Conv c = {get(d, w, cout, cin, k, k, ok), get(d, b, cout, 0, 0, 0, ok), cout, cin, k};
    return c;
}

// A Lin from rows [row0, row0 + out) of a PyTorch weight [*][in] and its bias.
static Lin lin(Decoder *d, const float *w, const float *b, int row0, int out, int in) {
    Lin L = {NULL, b ? b + row0 : NULL, in, out};
    if (!w || d->nowned >= 64) return L;
    L.wt = malloc((size_t)in * out * sizeof(float));
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++) L.wt[(size_t)i * out + o] = w[(size_t)(row0 + o) * in + i];
    d->owned[d->nowned++] = L.wt;
    return L;
}

Decoder *decoder_load(const char *path, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errlen, "%s: cannot open", path);
        return NULL;
    }
    Decoder *d = calloc(1, sizeof *d);
    char magic[4];
    uint32_t hdr[2];
    int ok = fread(magic, 4, 1, f) == 1 && fread(hdr, 4, 2, f) == 2 && !memcmp(magic, "WOFD", 4)
             && hdr[0] == 1 && hdr[1] < 1000;
    if (ok) {
        d->nt = (int)hdr[1];
        d->t = calloc((size_t)d->nt, sizeof *d->t);
        d->names = calloc((size_t)d->nt, 64);
    }
    for (int i = 0; ok && i < d->nt; i++) {
        Tensor *t = &d->t[i];
        uint32_t len, nd, dims[4] = {0};
        ok = fread(&len, 4, 1, f) == 1 && len < 64 && fread(d->names + 64 * i, 1, len, f) == len
             && fread(&nd, 4, 1, f) == 1 && nd >= 1 && nd <= 4 && fread(dims, 4, nd, f) == nd;
        if (!ok) break;
        t->name = d->names + 64 * i;
        t->ndim = (int)nd;
        size_t n = 1;
        for (uint32_t k = 0; k < nd; k++) t->dim[k] = (int)dims[k], n *= dims[k];
        t->v = malloc(n * sizeof(float));
        ok = t->v && fread(t->v, sizeof(float), n, f) == n;
    }
    fclose(f);
    if (!ok) {
        snprintf(err, errlen, "%s: not a version 1 WOFD decoder file", path);
        decoder_free(d);
        return NULL;
    }
    static const int widths[5] = {32, 48, 64, 96, 128};
    d->stem = conv(d, "stem", widths[0], 3, 3, &ok);
    for (int s = 0; s < 4; s++)
        for (int k = 0; k < 2; k++) {
            Block *B = &d->blocks[2 * s + k];
            int cin = k ? widths[s + 1] : widths[s], cout = widths[s + 1];
            char n[32];
            snprintf(n, sizeof n, "s%d.%d.a", s + 1, k);
            B->a = conv(d, n, cout, cin, 3, &ok);
            snprintf(n, sizeof n, "s%d.%d.b", s + 1, k);
            B->b = conv(d, n, cout, cout, 3, &ok);
            B->stride = k ? 1 : 2;
            B->has_skip = !k;
            if (!k) {
                snprintf(n, sizeof n, "s%d.%d.skip", s + 1, k);
                B->skip = conv(d, n, cout, cin, 1, &ok);
            }
        }
    d->lat8 = conv(d, "lat8", D_MODEL, widths[2], 1, &ok);
    d->lat16 = conv(d, "lat16", D_MODEL, widths[3], 1, &ok);
    d->lat32 = conv(d, "lat32", D_MODEL, widths[4], 1, &ok);
    d->pos = get(d, "pos", 45 * 80, D_MODEL, 0, 0, &ok);
    d->query = get(d, "query", N_Q, D_MODEL, 0, 0, &ok);
    d->qpos = get(d, "qpos", N_Q, D_MODEL, 0, 0, &ok);
    for (int l = 0; l < LAYERS; l++) {
        DecLayer *L = &d->layer[l];
        char n[48];
#define G(field, suffix, a, b)                              \
    snprintf(n, sizeof n, "layer%d.%s", l, suffix);         \
    L->field = get(d, n, a, b, 0, 0, &ok);
        G(n1w, "n1.w", D_MODEL, 0) G(n1b, "n1.b", D_MODEL, 0)
        G(n2w, "n2.w", D_MODEL, 0) G(n2b, "n2.b", D_MODEL, 0)
        G(n3w, "n3.w", D_MODEL, 0) G(n3b, "n3.b", D_MODEL, 0)
#undef G
        const char *an[2] = {"self_attn", "cross"};
        Attn *at[2] = {&L->self, &L->cross};
        for (int a = 0; a < 2; a++) {
            const float *iw, *ib, *ow, *ob;
            snprintf(n, sizeof n, "layer%d.%s.in_w", l, an[a]);
            iw = get(d, n, 3 * D_MODEL, D_MODEL, 0, 0, &ok);
            snprintf(n, sizeof n, "layer%d.%s.in_b", l, an[a]);
            ib = get(d, n, 3 * D_MODEL, 0, 0, 0, &ok);
            snprintf(n, sizeof n, "layer%d.%s.out_w", l, an[a]);
            ow = get(d, n, D_MODEL, D_MODEL, 0, 0, &ok);
            snprintf(n, sizeof n, "layer%d.%s.out_b", l, an[a]);
            ob = get(d, n, D_MODEL, 0, 0, 0, &ok);
            at[a]->q = lin(d, iw, ib, 0, D_MODEL, D_MODEL);
            at[a]->k = lin(d, iw, ib, D_MODEL, D_MODEL, D_MODEL);
            at[a]->v = lin(d, iw, ib, 2 * D_MODEL, D_MODEL, D_MODEL);
            at[a]->o = lin(d, ow, ob, 0, D_MODEL, D_MODEL);
        }
        const float *w, *b;
        snprintf(n, sizeof n, "layer%d.fc.w", l);
        w = get(d, n, 4 * D_MODEL, D_MODEL, 0, 0, &ok);
        snprintf(n, sizeof n, "layer%d.fc.b", l);
        b = get(d, n, 4 * D_MODEL, 0, 0, 0, &ok);
        L->fc = lin(d, w, b, 0, 4 * D_MODEL, D_MODEL);
        snprintf(n, sizeof n, "layer%d.proj.w", l);
        w = get(d, n, D_MODEL, 4 * D_MODEL, 0, 0, &ok);
        snprintf(n, sizeof n, "layer%d.proj.b", l);
        b = get(d, n, D_MODEL, 0, 0, 0, &ok);
        L->proj = lin(d, w, b, 0, D_MODEL, 4 * D_MODEL);
    }
    d->norm_w = get(d, "norm.w", D_MODEL, 0, 0, 0, &ok);
    d->norm_b = get(d, "norm.b", D_MODEL, 0, 0, 0, &ok);
    const float *hw = get(d, "sq_head.w", DEC_CLASSES, D_MODEL, 0, 0, &ok);
    d->sq = lin(d, hw, get(d, "sq_head.b", DEC_CLASSES, 0, 0, 0, &ok), 0, DEC_CLASSES, D_MODEL);
    hw = get(d, "cat_head.w", DEC_CAT_CHARS, D_MODEL, 0, 0, &ok);
    d->cat = lin(d, hw, get(d, "cat_head.b", DEC_CAT_CHARS, 0, 0, 0, &ok), 0, DEC_CAT_CHARS, D_MODEL);
    hw = get(d, "present_head.w", 1, D_MODEL, 0, 0, &ok);
    d->pr = lin(d, hw, get(d, "present_head.b", 1, 0, 0, 0, &ok), 0, 1, D_MODEL);
    if (!ok || d->nowned >= 64) {
        snprintf(err, errlen, "%s: a tensor is missing or the wrong shape", path);
        decoder_free(d);
        return NULL;
    }
    return d;
}

void decoder_free(Decoder *d) {
    if (!d) return;
    for (int i = 0; i < d->nt; i++) free(d->t[i].v);
    for (int i = 0; i < d->nowned; i++) free(d->owned[i]);
    free(d->t);
    free(d->names);
    free(d);
}

// ------------------------------------------------------------- letterbox --

// One axis of Pillow's BILINEAR resample (a triangle filter widened by the
// downscale factor), as ImagingResample computes its coefficients.
typedef struct {
    int *lo, *n;
    double *w;
    int ksize;
} Coeffs;

static Coeffs coeffs(int in, int out) {
    double scale = (double)in / out, fs = scale < 1 ? 1 : scale, support = fs;
    int ksize = (int)ceil(support) * 2 + 1;
    Coeffs c = {malloc(sizeof(int) * out), malloc(sizeof(int) * out), calloc((size_t)out * ksize, sizeof(double)), ksize};
    for (int i = 0; i < out; i++) {
        double center = (i + 0.5) * scale, ss = 1.0 / fs, total = 0;
        int lo = (int)(center - support + 0.5), hi = (int)(center + support + 0.5);
        if (lo < 0) lo = 0;
        if (hi > in) hi = in;
        int n = hi - lo;
        double *w = c.w + (size_t)i * ksize;
        for (int x = 0; x < n; x++) {
            double t = fabs((x + lo - center + 0.5) * ss);
            w[x] = t < 1 ? 1 - t : 0;
            total += w[x];
        }
        for (int x = 0; x < n && total; x++) w[x] /= total;
        c.lo[i] = lo, c.n[i] = n;
    }
    return c;
}

static void free_coeffs(Coeffs *c) { free(c->lo), free(c->n), free(c->w); }

static uint8_t clip8(double v) {
    v = floor(v + 0.5);
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

void decoder_letterbox(const uint8_t *rgb, int w, int h, uint8_t *out) {
    double s = fmin((double)DEC_IN_W / w, (double)DEC_IN_H / h);
    int nw = (int)nearbyint(w * s), nh = (int)nearbyint(h * s);   // ties to even, as Python's round
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    if (nw > DEC_IN_W) nw = DEC_IN_W;
    if (nh > DEC_IN_H) nh = DEC_IN_H;
    int ox = (DEC_IN_W - nw) / 2, oy = (DEC_IN_H - nh) / 2;
    memset(out, 0, (size_t)DEC_IN_W * DEC_IN_H * 3);
    // horizontal pass into an nw x h image, then vertical, rounding to bytes
    // between the two as Pillow does
    uint8_t *mid = malloc((size_t)nw * h * 3);
    Coeffs cx = coeffs(w, nw), cy = coeffs(h, nh);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < nw; x++)
            for (int ch = 0; ch < 3; ch++) {
                double acc = 0;
                const double *k = cx.w + (size_t)x * cx.ksize;
                for (int i = 0; i < cx.n[x]; i++) acc += k[i] * rgb[((size_t)y * w + cx.lo[x] + i) * 3 + ch];
                mid[((size_t)y * nw + x) * 3 + ch] = clip8(acc);
            }
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++)
            for (int ch = 0; ch < 3; ch++) {
                double acc = 0;
                const double *k = cy.w + (size_t)y * cy.ksize;
                for (int i = 0; i < cy.n[y]; i++) acc += k[i] * mid[((size_t)(cy.lo[y] + i) * nw + x) * 3 + ch];
                out[((size_t)(y + oy) * DEC_IN_W + x + ox) * 3 + ch] = clip8(acc);
            }
    free(mid);
    free_coeffs(&cx);
    free_coeffs(&cy);
}

// ------------------------------------------------------------------- CNN --

static int out_size(int n, int stride) { return (n - 1) / stride + 1; }

// out[cout][oh][ow] = conv(in[cin][h][w]) + bias, padding k/2, then ReLU if
// relu. For each input channel, cols holds k copies of it, column kx holding
// in[y][x * stride + kx - pad] (0 outside the image), so every tap's inner
// loop runs contiguously along an output row and vectorises at any stride.
static void conv2d(const Conv *c, const float *in, int h, int w, int stride, float *out, int relu,
                   float *cols) {
    int oh = out_size(h, stride), ow = out_size(w, stride), p = c->k / 2, k = c->k;
    size_t plane = (size_t)oh * ow;
    for (int o = 0; o < c->cout; o++)
        for (size_t i = 0; i < plane; i++) out[o * plane + i] = c->b[o];
    for (int i = 0; i < c->cin; i++) {
        const float *src = in + (size_t)i * h * w;
        for (int kx = 0; kx < k; kx++)
            for (int y = 0; y < h; y++) {
                float *dst = cols + ((size_t)kx * h + y) * ow;
                const float *row = src + (size_t)y * w;
                for (int x = 0; x < ow; x++) {
                    int ix = x * stride + kx - p;
                    dst[x] = ix >= 0 && ix < w ? row[ix] : 0;
                }
            }
        for (int o = 0; o < c->cout; o++) {
            float *dst = out + o * plane;
            const float *wk = c->w + ((size_t)o * c->cin + i) * k * k;
            for (int ky = 0; ky < k; ky++)
                for (int y = 0; y < oh; y++) {
                    int iy = y * stride + ky - p;
                    if (iy < 0 || iy >= h) continue;
                    float *drow = dst + (size_t)y * ow;
                    for (int kx = 0; kx < k; kx++) {
                        float wt = wk[ky * k + kx];
                        const float *crow = cols + ((size_t)kx * h + iy) * ow;
                        for (int x = 0; x < ow; x++) drow[x] += wt * crow[x];
                    }
                }
        }
    }
    if (relu)
        for (size_t i = 0; i < (size_t)c->cout * plane; i++) out[i] = out[i] > 0 ? out[i] : 0;
}

// A residual block: relu(b(relu(a(x))) + skip(x)).
static void block(const Block *B, const float *in, int h, int w, float *out, float *t1, float *t2,
                  float *cols) {
    int oh = out_size(h, B->stride), ow = out_size(w, B->stride);
    conv2d(&B->a, in, h, w, B->stride, t1, 1, cols);
    conv2d(&B->b, t1, oh, ow, 1, out, 0, cols);
    size_t n = (size_t)B->b.cout * oh * ow;
    if (B->has_skip) {
        conv2d(&B->skip, in, h, w, B->stride, t2, 0, cols);
        for (size_t i = 0; i < n; i++) out[i] += t2[i];
    } else {
        for (size_t i = 0; i < n; i++) out[i] += in[i];
    }
    for (size_t i = 0; i < n; i++) out[i] = out[i] > 0 ? out[i] : 0;
}

// dst[c][h][w] += src[c][sh][sw] upsampled to h x w, nearest (PyTorch's
// floor(i * in / out)).
static void add_nearest(float *dst, int h, int w, const float *src, int sh, int sw, int ch) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < h; y++) {
            int sy = (int)floor(y * ((double)sh / h));
            if (sy > sh - 1) sy = sh - 1;
            for (int x = 0; x < w; x++) {
                int sx = (int)floor(x * ((double)sw / w));
                if (sx > sw - 1) sx = sw - 1;
                dst[((size_t)c * h + y) * w + x] += src[((size_t)c * sh + sy) * sw + sx];
            }
        }
}

// ------------------------------------------------------------ attention --

// y[n][out] = x[n][in] W^T + b
static void linear(float *y, const float *x, const Lin *L, int n) {
    for (int r = 0; r < n; r++) {
        const float *xr = x + (size_t)r * L->in;
        float *yr = y + (size_t)r * L->out;
        for (int o = 0; o < L->out; o++) yr[o] = L->b ? L->b[o] : 0;
        for (int i = 0; i < L->in; i++) {
            const float xi = xr[i], *wr = L->wt + (size_t)i * L->out;
            for (int o = 0; o < L->out; o++) yr[o] += xi * wr[o];
        }
    }
}

static void layernorm(float *y, const float *x, const float *g, const float *b, int n) {
    for (int r = 0; r < n; r++) {
        const float *xr = x + (size_t)r * D_MODEL;
        float *yr = y + (size_t)r * D_MODEL;
        double mean = 0, var = 0;
        for (int i = 0; i < D_MODEL; i++) mean += xr[i];
        mean /= D_MODEL;
        for (int i = 0; i < D_MODEL; i++) var += (xr[i] - mean) * (xr[i] - mean);
        float inv = (float)(1.0 / sqrt(var / D_MODEL + 1e-5));
        for (int i = 0; i < D_MODEL; i++) yr[i] = (float)(xr[i] - mean) * inv * g[i] + b[i];
    }
}

// Multi-head attention as nn.MultiheadAttention: queries from qin (nq rows),
// keys from kin and values from vin (nk rows). out gets nq x D_MODEL. The
// keys are transposed to Kt[d][nk], so each score row accumulates along k.
static void attention(const Attn *A, const float *qin, int nq, const float *kin, const float *vin, int nk,
                      float *out, float *Q, float *K, float *Kt, float *V, float *ctx, float *score) {
    const int dh = D_MODEL / HEADS;
    linear(Q, qin, &A->q, nq);
    linear(K, kin, &A->k, nk);
    linear(V, vin, &A->v, nk);
    for (int k = 0; k < nk; k++)
        for (int i = 0; i < D_MODEL; i++) Kt[(size_t)i * nk + k] = K[(size_t)k * D_MODEL + i];
    float scale = 1.0f / sqrtf((float)dh);
    for (int hd = 0; hd < HEADS; hd++)
        for (int q = 0; q < nq; q++) {
            const float *qv = Q + (size_t)q * D_MODEL + hd * dh;
            for (int k = 0; k < nk; k++) score[k] = 0;
            for (int i = 0; i < dh; i++) {
                const float qi = qv[i], *kt = Kt + (size_t)(hd * dh + i) * nk;
                for (int k = 0; k < nk; k++) score[k] += qi * kt[k];
            }
            float mx = -INFINITY;
            for (int k = 0; k < nk; k++) {
                score[k] *= scale;
                if (score[k] > mx) mx = score[k];
            }
            double total = 0;
            for (int k = 0; k < nk; k++) score[k] = expf(score[k] - mx), total += score[k];
            float *c = ctx + (size_t)q * D_MODEL + hd * dh;
            for (int i = 0; i < dh; i++) c[i] = 0;
            for (int k = 0; k < nk; k++) {
                float p = (float)(score[k] / total);
                const float *vv = V + (size_t)k * D_MODEL + hd * dh;
                for (int i = 0; i < dh; i++) c[i] += p * vv[i];
            }
        }
    linear(out, ctx, &A->o, nq);
}

// ---------------------------------------------------------------- forward --

int decoder_square(int r, int c) {
    static const int start[4] = {0, 12, 26, 40}, first[4] = {1, 0, 0, 1}, width[4] = {12, 14, 14, 12};
    if (r < 0 || r > 3 || c < first[r] || c >= first[r] + width[r]) return -1;
    return start[r] + c - first[r];
}

int decoder_forward(const Decoder *d, const uint8_t *input, DecodeResult *out) {
    const int H = DEC_IN_H, W = DEC_IN_W;
    const int h2 = out_size(H, 2), w2 = out_size(W, 2), h4 = out_size(h2, 2), w4 = out_size(w2, 2);
    const int h8 = out_size(h4, 2), w8 = out_size(w4, 2), h16 = out_size(h8, 2), w16 = out_size(w8, 2);
    const int h32 = out_size(h16, 2), w32 = out_size(w16, 2), NT = h8 * w8;
    size_t big = (size_t)32 * h2 * w2;                 // the largest activation
    float *a = malloc(big * sizeof(float)), *b = malloc(big * sizeof(float));
    float *t1 = malloc(big * sizeof(float)), *t2 = malloc(big * sizeof(float));
    float *f8 = malloc((size_t)64 * h8 * w8 * sizeof(float));
    float *f16 = malloc((size_t)96 * h16 * w16 * sizeof(float));
    float *p8 = malloc((size_t)D_MODEL * NT * sizeof(float));
    float *p16 = malloc((size_t)D_MODEL * h16 * w16 * sizeof(float));
    float *mem = malloc((size_t)NT * D_MODEL * sizeof(float)), *mpos = malloc((size_t)NT * D_MODEL * sizeof(float));
    float *K = malloc((size_t)NT * D_MODEL * sizeof(float)), *V = malloc((size_t)NT * D_MODEL * sizeof(float));
    float *Kt = malloc((size_t)NT * D_MODEL * sizeof(float));
    float *cols = malloc((size_t)3 * H * w2 * sizeof(float));   // the widest conv input's column copies
    float *score = malloc((size_t)NT * sizeof(float));
    float q[N_Q * D_MODEL], hq[N_Q * D_MODEL], qp[N_Q * D_MODEL], Q[N_Q * D_MODEL], ctx[N_Q * D_MODEL],
        att[N_Q * D_MODEL];
    float *mlp = malloc((size_t)N_Q * 4 * D_MODEL * sizeof(float));
    if (!a || !b || !t1 || !t2 || !f8 || !f16 || !p8 || !p16 || !mem || !mpos || !K || !V || !Kt || !cols || !score
        || !mlp) {
        free(a), free(b), free(t1), free(t2), free(f8), free(f16), free(p8), free(p16);
        free(mem), free(mpos), free(K), free(V), free(Kt), free(cols), free(score), free(mlp);
        return -1;
    }
    // input: channels first, scaled to [0, 1] and normalised
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                a[((size_t)c * H + y) * W + x] = (input[((size_t)y * W + x) * 3 + c] / 255.0f - MEAN) / STD;
    conv2d(&d->stem, a, H, W, 2, b, 1, cols);
    block(&d->blocks[0], b, h2, w2, a, t1, t2, cols);
    block(&d->blocks[1], a, h4, w4, b, t1, t2, cols);
    block(&d->blocks[2], b, h4, w4, a, t1, t2, cols);
    block(&d->blocks[3], a, h8, w8, f8, t1, t2, cols);
    block(&d->blocks[4], f8, h8, w8, a, t1, t2, cols);
    block(&d->blocks[5], a, h16, w16, f16, t1, t2, cols);
    block(&d->blocks[6], f16, h16, w16, a, t1, t2, cols);
    block(&d->blocks[7], a, h32, w32, b, t1, t2, cols);      // b holds the 1/32 features
    conv2d(&d->lat32, b, h32, w32, 1, t1, 0, cols);
    conv2d(&d->lat16, f16, h16, w16, 1, p16, 0, cols);
    add_nearest(p16, h16, w16, t1, h32, w32, D_MODEL);
    conv2d(&d->lat8, f8, h8, w8, 1, p8, 0, cols);
    add_nearest(p8, h8, w8, p16, h16, w16, D_MODEL);
    // tokens: N x d, and the keys' input mem + pos
    for (int t = 0; t < NT; t++)
        for (int c = 0; c < D_MODEL; c++) {
            mem[(size_t)t * D_MODEL + c] = p8[(size_t)c * NT + t];
            mpos[(size_t)t * D_MODEL + c] = mem[(size_t)t * D_MODEL + c] + d->pos[(size_t)t * D_MODEL + c];
        }
    memcpy(q, d->query, sizeof q);
    for (int l = 0; l < LAYERS; l++) {
        const DecLayer *L = &d->layer[l];
        layernorm(hq, q, L->n1w, L->n1b, N_Q);
        for (int i = 0; i < N_Q * D_MODEL; i++) qp[i] = hq[i] + d->qpos[i];
        attention(&L->self, qp, N_Q, qp, hq, N_Q, att, Q, K, Kt, V, ctx, score);
        for (int i = 0; i < N_Q * D_MODEL; i++) q[i] += att[i];
        layernorm(hq, q, L->n2w, L->n2b, N_Q);
        for (int i = 0; i < N_Q * D_MODEL; i++) qp[i] = hq[i] + d->qpos[i];
        attention(&L->cross, qp, N_Q, mpos, mem, NT, att, Q, K, Kt, V, ctx, score);
        for (int i = 0; i < N_Q * D_MODEL; i++) q[i] += att[i];
        layernorm(hq, q, L->n3w, L->n3b, N_Q);
        linear(mlp, hq, &L->fc, N_Q);
        for (int i = 0; i < N_Q * 4 * D_MODEL; i++) mlp[i] = 0.5f * mlp[i] * (1.0f + erff(mlp[i] * 0.70710678118654752f));
        linear(att, mlp, &L->proj, N_Q);
        for (int i = 0; i < N_Q * D_MODEL; i++) q[i] += att[i];
    }
    layernorm(hq, q, d->norm_w, d->norm_b, N_Q);
    // heads
    float logits[DEC_CLASSES], cl[DEC_CAT_CHARS], pr;
    memset(out, 0, sizeof *out);
    for (int r = 0; r < 4; r++) {
        memset(out->grid[r], ' ', 14);
        out->grid[r][14] = 0;
    }
    for (int s = 0; s < DEC_SQUARES; s++) {
        linear(logits, hq + (size_t)s * D_MODEL, &d->sq, 1);
        float mx = logits[0];
        int best = 0;
        for (int k = 1; k < DEC_CLASSES; k++)
            if (logits[k] > mx) mx = logits[k], best = k;
        double total = 0;
        for (int k = 0; k < DEC_CLASSES; k++) total += exp(logits[k] - mx);
        for (int k = 0; k < DEC_CLASSES; k++) out->probs[s][k] = (float)(exp(logits[k] - mx) / total);
        out->conf[s] = out->probs[s][best];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 14; c++)
                if (decoder_square(r, c) == s) out->grid[r][c] = DEC_CLASS_CHARS[best];
    }
    int n = 0;
    for (int s = 0; s < DEC_CAT_LEN; s++) {
        linear(cl, hq + (size_t)(DEC_SQUARES + s) * D_MODEL, &d->cat, 1);
        int best = 0;
        for (int k = 1; k < DEC_CAT_CHARS; k++)
            if (cl[k] > cl[best]) best = k;
        if (best) out->category[n++] = CAT_CHARS[best];
    }
    while (n && out->category[n - 1] == ' ') n--;   // decode_category's strip()
    out->category[n] = 0;
    int lead = 0;
    while (out->category[lead] == ' ') lead++;
    memmove(out->category, out->category + lead, (size_t)(n - lead + 1));
    linear(&pr, hq + (size_t)(N_Q - 1) * D_MODEL, &d->pr, 1);
    out->present = 1.0f / (1.0f + expf(-pr));
    free(a), free(b), free(t1), free(t2), free(f8), free(f16), free(p8), free(p16);
    free(mem), free(mpos), free(K), free(V), free(Kt), free(cols), free(score), free(mlp);
    return 0;
}
