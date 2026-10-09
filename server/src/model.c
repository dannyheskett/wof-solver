// Forward pass of scripts/model.py: token + position embeddings, pre-norm
// blocks (LayerNorm, causal self-attention, LayerNorm, GELU MLP), a final
// LayerNorm and the output layer tied to the token embedding. Linear layers
// have no biases. GELU is the exact erf form, as in PyTorch's nn.GELU().
// Weights are float16, mapped from the file. F16C instructions widen them
// to float32.
#include "model.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_IN 4096   // the widest weight row linear() takes (4 x n_embd)

static int fail(char *err, size_t errlen, const char *msg, const char *path) {
    snprintf(err, errlen, "%s: %s", path, msg);
    return -1;
}

int model_load(Model *m, const char *path, char *err, size_t errlen) {
    memset(m, 0, sizeof *m);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(err, errlen, "cannot open", path);
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 28) {
        close(fd);
        return fail(err, errlen, "not a version 2 (float16) WOFM weights file", path);
    }
    // Map the file rather than copy it: its pages are the kernel's page cache,
    // which it can drop and reread under memory pressure instead of killing us.
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail(err, errlen, "cannot map", path);
    m->map = map, m->map_len = (size_t)st.st_size;
    const unsigned char *b = map;
    uint32_t h[6];
    memcpy(h, b + 4, sizeof h);
    if (memcmp(b, "WOFM", 4) || h[0] != 2) {
        model_free(m);
        return fail(err, errlen, "not a version 2 (float16) WOFM weights file", path);
    }
    m->vocab = (int)h[1], m->block = (int)h[2], m->n_layer = (int)h[3];
    m->n_head = (int)h[4], m->n_embd = (int)h[5];
    if (m->vocab > MODEL_MAX_VOCAB || m->vocab % 2 || m->n_layer > MODEL_MAX_LAYERS
        || m->n_embd % m->n_head || m->n_embd % 16 || 4 * m->n_embd > MAX_IN || m->block <= 0) {
        model_free(m);
        return fail(err, errlen, "unsupported model shape", path);
    }
    size_t V = (size_t)m->vocab, C = (size_t)m->n_embd, B = (size_t)m->block;
    size_t per_layer = 4 * C + 3 * C * C + C * C + 4 * C * C + 4 * C * C;
    size_t total = V * C + B * C + (size_t)m->n_layer * per_layer + 2 * C;
    if (m->map_len != 28 + V + total * sizeof(half)) {
        model_free(m);
        return fail(err, errlen, "wrong size for its shape", path);
    }
    memcpy(m->alphabet, b + 28, V);
    posix_madvise(map, m->map_len, POSIX_MADV_WILLNEED);
    const half *p = (const half *)(b + 28 + V);   // even offset: vocab is even
    m->tok = p, p += V * C;
    m->pos = p, p += B * C;
    for (int l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layer[l];
        L->ln1_w = p, p += C;
        L->ln1_b = p, p += C;
        L->qkv = p, p += 3 * C * C;
        L->proj = p, p += C * C;
        L->ln2_w = p, p += C;
        L->ln2_b = p, p += C;
        L->fc = p, p += 4 * C * C;
        L->out = p, p += 4 * C * C;
    }
    m->ln_w = p, p += C;
    m->ln_b = p;
    return 0;
}

void model_free(Model *m) {
    if (m->map) munmap(m->map, m->map_len);
    m->map = NULL;
}

int model_token(const Model *m, char c) {
    for (int i = 0; i < m->vocab; i++)
        if (m->alphabet[i] == c) return i;
    return -1;
}

Seq *seq_new(const Model *m) {
    size_t C = (size_t)m->n_embd, B = (size_t)m->block, L = (size_t)m->n_layer;
    Seq *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->k = malloc(L * B * C * sizeof(float));
    s->v = malloc(L * B * C * sizeof(float));
    s->x = malloc(B * C * sizeof(float));
    s->xn = malloc(B * C * sizeof(float));
    s->qkv = malloc(B * 3 * C * sizeof(float));
    s->y = malloc(B * C * sizeof(float));
    s->h = malloc(B * 4 * C * sizeof(float));
    s->att = malloc(B * sizeof(float));
    if (!s->k || !s->v || !s->x || !s->xn || !s->qkv || !s->y || !s->h || !s->att) {
        seq_free(s);
        return NULL;
    }
    return s;
}

void seq_free(Seq *s) {
    if (!s) return;
    free(s->k), free(s->v), free(s->x), free(s->xn), free(s->qkv), free(s->y), free(s->h), free(s->att);
    free(s);
}

// n is a multiple of 16. Sixteen running sums let the compiler use vector
// registers without reordering a single sum.
static float dot(const float *a, const float *b, int n) {
    float acc[16] = {0};
    for (int i = 0; i < n; i += 16)
        for (int j = 0; j < 16; j++) acc[j] += a[i + j] * b[i + j];
    float s = 0;
    for (int j = 0; j < 16; j++) s += acc[j];
    return s;
}

// float16 to float32, n a multiple of 8 (MAX_IN is defined at the top)
static void widen(float *dst, const half *src, int n) {
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i))));
}

static float h2f(half v) { return _cvtsh_ss(v); }

// y[t][o] = x[t] . w[o] for t < T: each weight row is widened once for all T rows.
static void linear(float *y, const float *x, const half *w, int T, int in, int out) {
    static float row[MAX_IN];
    for (int o = 0; o < out; o++) {
        widen(row, w + (size_t)o * in, in);
        for (int t = 0; t < T; t++) y[(size_t)t * out + o] = dot(x + (size_t)t * in, row, in);
    }
}

static void layernorm(float *y, const float *x, const half *w, const half *b, int C) {
    float mean = 0, var = 0;
    for (int i = 0; i < C; i++) mean += x[i];
    mean /= C;
    for (int i = 0; i < C; i++) var += (x[i] - mean) * (x[i] - mean);
    float r = 1.0f / sqrtf(var / C + 1e-5f);
    for (int i = 0; i < C; i++) y[i] = (x[i] - mean) * r * h2f(w[i]) + h2f(b[i]);
}

const float *model_feed(const Model *m, Seq *s, const int *tokens, int T) {
    const int C = m->n_embd, H = m->n_head, D = C / H, B = m->block, P = s->n;
    const float scale = 1.0f / sqrtf((float)D);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++)
            s->x[t * C + i] = h2f(m->tok[tokens[t] * C + i]) + h2f(m->pos[(P + t) * C + i]);

    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layer[l];
        float *K = s->k + (size_t)l * B * C, *V = s->v + (size_t)l * B * C;
        for (int t = 0; t < T; t++) layernorm(s->xn + t * C, s->x + t * C, L->ln1_w, L->ln1_b, C);
        linear(s->qkv, s->xn, L->qkv, T, C, 3 * C);
        for (int t = 0; t < T; t++) {
            memcpy(K + (size_t)(P + t) * C, s->qkv + (size_t)t * 3 * C + C, C * sizeof(float));
            memcpy(V + (size_t)(P + t) * C, s->qkv + (size_t)t * 3 * C + 2 * C, C * sizeof(float));
        }
        // causal attention: position P+t sees positions 0..P+t
        for (int t = 0; t < T; t++) {
            int n = P + t + 1;
            for (int hd = 0; hd < H; hd++) {
                const float *q = s->qkv + (size_t)t * 3 * C + hd * D;
                float mx = -INFINITY;
                for (int j = 0; j < n; j++) {
                    s->att[j] = dot(q, K + (size_t)j * C + hd * D, D) * scale;
                    if (s->att[j] > mx) mx = s->att[j];
                }
                float sum = 0;
                for (int j = 0; j < n; j++) sum += s->att[j] = expf(s->att[j] - mx);
                float *y = s->y + (size_t)t * C + hd * D;
                for (int i = 0; i < D; i++) y[i] = 0;
                for (int j = 0; j < n; j++) {
                    float a = s->att[j] / sum;
                    const float *v = V + (size_t)j * C + hd * D;
                    for (int i = 0; i < D; i++) y[i] += a * v[i];
                }
            }
        }
        linear(s->xn, s->y, L->proj, T, C, C);
        for (int i = 0; i < T * C; i++) s->x[i] += s->xn[i];

        for (int t = 0; t < T; t++) layernorm(s->xn + t * C, s->x + t * C, L->ln2_w, L->ln2_b, C);
        linear(s->h, s->xn, L->fc, T, C, 4 * C);
        for (int i = 0; i < T * 4 * C; i++) s->h[i] = 0.5f * s->h[i] * (1.0f + erff(s->h[i] * 0.70710678118654752f));
        linear(s->xn, s->h, L->out, T, 4 * C, C);
        for (int i = 0; i < T * C; i++) s->x[i] += s->xn[i];
    }
    s->n += T;
    layernorm(s->xn, s->x + (size_t)(T - 1) * C, m->ln_w, m->ln_b, C);
    linear(s->logits, s->xn, m->tok, 1, C, m->vocab);
    return s->logits;
}
