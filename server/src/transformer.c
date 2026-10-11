// transformer-solve. A move's allowed shapes are a few fixed-length
// patterns, one set of allowed tokens per character; decoding takes the
// most likely allowed token at each step and stops when a pattern is
// complete (the model would write a newline there, which ends a move).
#include "transformer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PAT (TF_MAX_TEXT + 8)
#define MAX_ALTS 4

typedef struct {
    uint64_t set[MAX_PAT];   // allowed token ids per character
    int len;
} Pattern;

typedef struct {
    Pattern alt[MAX_ALTS];
    int n;
} Grammar;

static uint64_t tok_bit(const Model *m, char c) {
    int t = model_token(m, c);
    return t < 0 ? 0 : (uint64_t)1 << t;
}

static uint64_t letters_not_in(const Model *m, const char *called) {
    uint64_t s = 0;
    for (char c = 'A'; c <= 'Z'; c++)
        if (!strchr(called, c)) s |= tok_bit(m, c);
    return s;
}

static Pattern *add(Grammar *g, const Model *m, const char *literal) {
    Pattern *p = &g->alt[g->n++];
    p->len = 0;
    for (const char *c = literal; *c; c++) p->set[p->len++] = tok_bit(m, *c);
    return p;
}

// "SOLVE " and an answer that fits the board: shown squares copied, a blank
// takes any uncalled letter.
static void add_solve(Grammar *g, const Model *m, const char *board, const char *called) {
    Pattern *p = add(g, m, "SOLVE ");
    uint64_t blank = letters_not_in(m, called);
    for (const char *c = board; *c; c++) p->set[p->len++] = *c == '_' ? blank : tok_bit(m, *c);
}

static int can_buy(const char *board, const char *called) {
    int cons = 0;
    for (const char *c = called; *c; c++) {
        if (strchr("AEIOU", *c)) continue;
        cons++;
        if (strchr(board, *c)) return 1;
    }
    return cons == 21;
}

// Greedy decoding under g. lp[k] gets the log-softmax over every token
// before the grammar, at step k. Returns the length written to out, or -1.
static int decode(const Model *m, Seq *s, const char *prompt, const Grammar *g, char *out,
                  double lp[][MODEL_MAX_VOCAB]) {
    int ids[512], n = 0, maxlen = 0;
    for (const char *c = prompt; *c && n < 512; c++) ids[n++] = model_token(m, *c);
    for (int a = 0; a < g->n; a++)
        if (g->alt[a].len > maxlen) maxlen = g->alt[a].len;
    if (n + maxlen > m->block) return -1;
    seq_reset(s);
    const float *logits = model_feed(m, s, ids, n);
    int alive[MAX_ALTS], nalive = g->n;
    for (int a = 0; a < g->n; a++) alive[a] = a;
    int k = 0;
    for (;;) {
        // done once a pattern is complete
        for (int a = 0; a < nalive; a++)
            if (g->alt[alive[a]].len == k) {
                out[k] = 0;
                return k;
            }
        double mx = -INFINITY, sum = 0;
        for (int t = 0; t < m->vocab; t++) mx = fmax(mx, logits[t]);
        for (int t = 0; t < m->vocab; t++) sum += exp(logits[t] - mx);
        for (int t = 0; t < m->vocab; t++) lp[k][t] = logits[t] - mx - log(sum);
        uint64_t allowed = 0;
        for (int a = 0; a < nalive; a++) allowed |= g->alt[alive[a]].set[k];
        int best = -1;
        for (int t = 0; t < m->vocab; t++)
            if ((allowed >> t & 1) && (best < 0 || logits[t] > logits[best])) best = t;
        if (best < 0) return -1;
        out[k++] = m->alphabet[best];
        int na = 0;
        for (int a = 0; a < nalive; a++)
            if (g->alt[alive[a]].set[k - 1] >> best & 1) alive[na++] = alive[a];
        nalive = na;
        int more = 0;   // feed the token only if some pattern goes on
        for (int a = 0; a < nalive; a++) more |= g->alt[alive[a]].len > k;
        if (more) logits = model_feed(m, s, &best, 1);
    }
}

// Python's round(x, nd), as the Python service reported it.
static double py_round(double x, int nd) {
    char buf[64];
    snprintf(buf, sizeof buf, "%.*f", nd, x);
    return strtod(buf, NULL);
}

// Confidence of the answer after "SOLVE " in out, from the step log
// probabilities and the board's allowed sets.
static void score(const Model *m, const Grammar *g, int solve_alt, const char *out, double lp[][MODEL_MAX_VOCAB],
                  TfMove *mv) {
    const Pattern *p = &g->alt[solve_alt];
    double prod = 1.0;
    mv->nletters = 0;
    for (int k = 6; out[k]; k++) {
        double norm = 0;
        for (int t = 0; t < m->vocab; t++)
            if (p->set[k] >> t & 1) norm += exp(lp[k][t]);
        int t = model_token(m, out[k]);
        double q = norm > 0 ? exp(lp[k][t]) / norm : 0.0;
        mv->letters[mv->nletters++] = q;
        prod *= q;
    }
    mv->has_conf = 1;
    mv->confidence = prod;
}

static void prompt_of(char *buf, size_t n, const char *mode, const char *category, const char *board,
                      const char *called, int dash) {
    int k = snprintf(buf, n, "%s|", mode);
    for (const char *c = category; *c && k + 1 < (int)n; c++)
        buf[k++] = (char)(*c >= 'a' && *c <= 'z' ? *c - 32 : *c);
    buf[k] = 0;
    snprintf(buf + k, n - (size_t)k, "|%s|%s|", board, called[0] || !dash ? called : "-");
}

static double lp_buf[MAX_PAT][MODEL_MAX_VOCAB];

int tf_main(const Model *m, Seq *s, const char *category, const char *board, const char *called, double min_p,
            TfMove *mv) {
    memset(mv, 0, sizeof *mv);
    char prompt[512], out[MAX_PAT + 1];
    prompt_of(prompt, sizeof prompt, "MAIN", category, board, called, 1);
    Grammar g = {.n = 0};
    uint64_t left = letters_not_in(m, called);
    Pattern *p = add(&g, m, "CALL ");
    p->set[p->len++] = left;
    if (can_buy(board, called)) {
        p = add(&g, m, "BUY ");
        p->set[p->len++] = left;
    }
    int letter_moves = g.n;
    add_solve(&g, m, board, called);
    if (decode(m, s, prompt, &g, out, lp_buf) < 0) return -1;
    snprintf(mv->move, sizeof mv->move, "%s", out);
    if (!strncmp(out, "SOLVE ", 6) && strchr(board, '_')) {
        score(m, &g, letter_moves, out, lp_buf, mv);
        if (py_round(mv->confidence, 4) < min_p) {
            // not sure enough: the best letter move instead
            snprintf(mv->turned_down, sizeof mv->turned_down, "%s", out + 6);
            g.n = letter_moves;
            if (decode(m, s, prompt, &g, out, lp_buf) < 0) return -1;
            snprintf(mv->move, sizeof mv->move, "%s", out);
        }
    }
    return 0;
}

int tf_bonus_pick(const Model *m, Seq *s, const char *category, const char *board, const char *called,
                  TfMove *mv) {
    memset(mv, 0, sizeof *mv);
    char prompt[512], out[MAX_PAT + 1];
    prompt_of(prompt, sizeof prompt, "BONUS", category, board, called, 0);
    Grammar g = {.n = 0};
    Pattern *p = add(&g, m, "PICK ");
    uint64_t az = letters_not_in(m, "");
    for (int i = 0; i < 3; i++) p->set[p->len++] = az;
    p->set[p->len++] = tok_bit(m, ' ');
    p->set[p->len++] = az;
    if (decode(m, s, prompt, &g, out, lp_buf) < 0) return -1;
    snprintf(mv->move, sizeof mv->move, "%s", out);
    return 0;
}

int tf_guess(const Model *m, Seq *s, const char *category, const char *board, const char *called, TfMove *mv) {
    memset(mv, 0, sizeof *mv);
    char prompt[512], out[MAX_PAT + 1];
    prompt_of(prompt, sizeof prompt, "MAIN", category, board, called, 1);
    Grammar g = {.n = 0};
    add_solve(&g, m, board, called);
    if (decode(m, s, prompt, &g, out, lp_buf) < 0) return -1;
    snprintf(mv->move, sizeof mv->move, "%s", out);
    score(m, &g, 0, out, lp_buf, mv);
    return 0;
}

int tf_bonus_solve(const Model *m, Seq *s, const char *category, const char *board, const char *called,
                   TfMove *mv) {
    memset(mv, 0, sizeof *mv);
    char prompt[512], out[MAX_PAT + 1];
    prompt_of(prompt, sizeof prompt, "BONUS", category, board, called, 0);
    Grammar g = {.n = 0};
    add_solve(&g, m, board, called);
    if (decode(m, s, prompt, &g, out, lp_buf) < 0) return -1;
    snprintf(mv->move, sizeof mv->move, "%s", out);
    score(m, &g, 0, out, lp_buf, mv);
    return 0;
}
