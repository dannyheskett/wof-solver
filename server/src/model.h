// The 59M-parameter transformer (scripts/model.py) in plain C: load the
// weights written by scripts/12_export_weights.py, feed tokens, read the logits
// of the last position. Single-threaded. Weights are float16 (about 118 MB),
// mapped from the file, and converted to float32 as they are used. All
// arithmetic, the activations and the key/value cache are float32.
#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

#define MODEL_MAX_LAYERS 32
#define MODEL_MAX_VOCAB 64

typedef uint16_t half;   // IEEE float16, as stored

typedef struct {
    const half *ln1_w, *ln1_b, *qkv, *proj, *ln2_w, *ln2_b, *fc, *out;
} Layer;

typedef struct {
    int vocab, block, n_layer, n_head, n_embd;
    char alphabet[MODEL_MAX_VOCAB + 1];   // a token's id is its index
    void *map;                            // the weights file, mapped read-only
    size_t map_len;
    const half *tok, *pos, *ln_w, *ln_b;
    Layer layer[MODEL_MAX_LAYERS];
} Model;

// One sequence being fed: its key/value cache and scratch space.
typedef struct {
    int n;                 // positions filled so far
    float *k, *v;          // [n_layer][block][n_embd]
    float *x, *xn, *qkv, *y, *h;
    float *att;            // [block]
    float logits[MODEL_MAX_VOCAB];
} Seq;

// Returns 0, or -1 with a message in err.
int model_load(Model *m, const char *path, char *err, size_t errlen);
void model_free(Model *m);

Seq *seq_new(const Model *m);
void seq_free(Seq *s);
static inline void seq_reset(Seq *s) { s->n = 0; }

// Feed n tokens after the ones already fed; returns the logits of the last,
// valid until the next call. The caller keeps s->n + n <= m->block.
const float *model_feed(const Model *m, Seq *s, const int *tokens, int n);

// The token id of character c, or -1.
int model_token(const Model *m, char c);

#endif
