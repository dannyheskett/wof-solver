// smart-solve, ported from scripts/wof_common.py and scripts/solvers.py.
//
// Words are grouped by shape (length and apostrophe positions), each group
// sorted A-Z, and each word kept as 26 position bitmasks: P[i][c] has bit k
// set when word i has letter c at position k. A board word's candidates are
// the words of its shape that match every called letter exactly. Each
// candidate weighs 1 plus its count in the category's puzzles.
//
// The scores are summed in numpy's order (pairwise, see pw_sum) and rounded
// as Python's round() does (py_round), so ties break the same way.
#include "smart.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORD 31        // bits in a position mask
#define MAX_TEXT 256
#define MAX_SPANS 64

static const char *VOWELS = "AEIOU";
static int is_vowel(char c) { return c && strchr(VOWELS, c) != NULL; }

typedef struct {
    char shape[MAX_WORD + 1];
    int n, cap;
    char **words;
    int32_t (*P)[26];
} Group;

typedef struct {               // a category's weights: (word index, count) per group
    int n, cap;
    int *idx, *count;
} Sparse;

typedef struct {
    char *name;
    Sparse *by_group;          // [ngroups]
} Category;

struct Smart {
    Group *g;
    int ng, capg;
    Category *cat;
    int ncat, capcat;
    char order[27];            // A-Z, most common in the puzzles first
    // scratch, reused per request
    double *wbuf;
    int wcap;
};

// ------------------------------------------------------------ helpers --

static void *grow(void *p, int *cap, int need, size_t size) {
    if (need <= *cap) return p;
    int c = *cap ? *cap : 16;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    *cap = c;
    return q;
}

static char *dup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (!d) exit(1);
    memcpy(d, s, n);
    return d;
}

static void shape_of(const char *key, char *out) {
    int i = 0;
    for (; key[i]; i++) out[i] = key[i] == '\'' ? '\'' : '_';
    out[i] = 0;
}

static void posmasks(const char *word, int32_t row[26]) {
    memset(row, 0, 26 * sizeof row[0]);
    for (int k = 0; word[k]; k++)
        if (word[k] >= 'A' && word[k] <= 'Z') row[word[k] - 'A'] |= (int32_t)1 << k;
}

// numpy's float64 sum: pairwise, in blocks of up to 128 with 8 accumulators.
static double pw_sum(const double *a, long n) {
    if (n < 8) {
        double r = 0.;
        for (long i = 0; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        long i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    long n2 = n / 2;
    n2 -= n2 % 8;
    return pw_sum(a, n2) + pw_sum(a + n2, n - n2);
}

// Python's round(x, nd): the exact binary value, rounded half to even.
static double py_round(double x, int nd) {
    char buf[64];
    snprintf(buf, sizeof buf, "%.*f", nd, x);
    return strtod(buf, NULL);
}

// Board words of a text, as in wof_common.word_keys: split on spaces and
// hyphens, keep letters, apostrophes and '_'. start[i] is where word i's
// run begins in the text; keys with only apostrophes are kept here and
// skipped by the callers that need to.
typedef struct {
    char key[MAX_TEXT];
    int start, len;            // the run in the text
} Run;

static int runs(const char *text, Run *out, int max) {
    int n = 0, i = 0, len = (int)strlen(text);
    while (i < len && n < max) {
        while (i < len && (text[i] == ' ' || text[i] == '-')) i++;
        if (i >= len) break;
        Run *r = &out[n];
        r->start = i;
        int k = 0;
        while (i < len && text[i] != ' ' && text[i] != '-') {
            char c = text[i++];
            if ((c >= 'A' && c <= 'Z') || c == '\'' || c == '_') r->key[k++] = c;
        }
        r->key[k] = 0;
        r->len = i - r->start;
        n++;
    }
    return n;
}

static int has_letter(const char *key) {
    for (; *key; key++)
        if (*key != '\'') return 1;
    return 0;
}

// --------------------------------------------------------------- load --

typedef struct {
    char **w;
    int n, cap;
} Words;

static void words_add(Words *ws, const char *w) {
    ws->w = grow(ws->w, &ws->cap, ws->n + 1, sizeof *ws->w);
    ws->w[ws->n++] = dup(w);
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int find_group(const Smart *s, const char *shape) {
    // groups are sorted by shape after loading
    int lo = 0, hi = s->ng - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(s->g[mid].shape, shape);
        if (!c) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int find_word(const Group *g, const char *w) {
    int lo = 0, hi = g->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(g->words[mid], w);
        if (!c) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static Category *category(Smart *s, const char *name, int create) {
    for (int i = 0; i < s->ncat; i++)
        if (!strcmp(s->cat[i].name, name)) return &s->cat[i];
    if (!create) return NULL;
    s->cat = grow(s->cat, &s->capcat, s->ncat + 1, sizeof *s->cat);
    Category *c = &s->cat[s->ncat++];
    c->name = dup(name);
    c->by_group = calloc((size_t)s->ng, sizeof *c->by_group);
    if (!c->by_group) exit(1);
    return c;
}

// One CSV record of two fields, answer and category, with "" quoting.
static int csv_row(FILE *f, char *a, char *b, size_t n) {
    char *field[2] = {a, b};
    int ch = fgetc(f);
    if (ch == EOF) return 0;
    for (int fi = 0; fi < 2; fi++) {
        size_t k = 0;
        int quoted = ch == '"';
        if (quoted) ch = fgetc(f);
        for (;;) {
            if (ch == EOF) break;
            if (quoted && ch == '"') {
                ch = fgetc(f);
                if (ch != '"') {
                    quoted = 0;
                    continue;
                }
            } else if (!quoted && (ch == ',' || ch == '\n' || ch == '\r')) {
                break;
            }
            if (k + 1 < n) field[fi][k++] = (char)ch;
            ch = fgetc(f);
        }
        field[fi][k] = 0;
        if (fi == 0 && ch == ',') ch = fgetc(f);
    }
    while (ch != EOF && ch != '\n') ch = fgetc(f);
    return 1;
}

typedef struct {
    char answer[MAX_TEXT], category[MAX_TEXT];
} Puzzle;

Smart *smart_load(const char *wordlist, const char *puzzles, char *err, size_t errlen) {
    FILE *f = fopen(wordlist, "r");
    if (!f) {
        snprintf(err, errlen, "%s: cannot open", wordlist);
        return NULL;
    }
    Words ws = {0};
    char line[512];
    while (fgets(line, sizeof line, f)) {
        // str.strip().upper(), then [A-Z']+ with at least one letter
        char *p = line, *e = line + strlen(line);
        while (p < e && isspace((unsigned char)*p)) p++;
        while (e > p && isspace((unsigned char)e[-1])) e--;
        *e = 0;
        int ok = *p != 0;
        for (char *c = p; *c && ok; c++) {
            *c = (char)toupper((unsigned char)*c);
            ok = (*c >= 'A' && *c <= 'Z') || *c == '\'';
        }
        if (ok && has_letter(p)) words_add(&ws, p);
    }
    fclose(f);

    // puzzles is one or more CSV paths, comma-separated, read in order
    Puzzle *pz = NULL;
    int npz = 0, cappz = 0;
    char a[MAX_TEXT], b[MAX_TEXT], path[512];
    for (const char *p = puzzles; *p;) {
        size_t n = strcspn(p, ",");
        snprintf(path, sizeof path, "%.*s", (int)n, p);
        p += n + (p[n] == ',');
        f = fopen(path, "r");
        if (!f) {
            snprintf(err, errlen, "%s: cannot open", path);
            free(pz);
            return NULL;
        }
        csv_row(f, a, b, sizeof a);   // header
        while (csv_row(f, a, b, sizeof a)) {
            if (!a[0]) continue;
            pz = grow(pz, &cappz, npz + 1, sizeof *pz);
            snprintf(pz[npz].answer, MAX_TEXT, "%s", a);
            snprintf(pz[npz].category, MAX_TEXT, "%s", b);
            npz++;
        }
        fclose(f);
    }

    long freq[26] = {0};
    Run r[MAX_SPANS];
    for (int i = 0; i < npz; i++) {
        for (const char *c = pz[i].answer; *c; c++)
            if (*c >= 'A' && *c <= 'Z') freq[*c - 'A']++;
        int nr = runs(pz[i].answer, r, MAX_SPANS);
        for (int k = 0; k < nr; k++)
            if (has_letter(r[k].key)) words_add(&ws, r[k].key);
    }

    Smart *s = calloc(1, sizeof *s);
    if (!s) exit(1);
    for (int i = 0; i < 26; i++) s->order[i] = (char)('A' + i);
    for (int i = 0; i < 26; i++)            // by count, then A-Z (a stable insertion sort)
        for (int j = i; j > 0 && freq[s->order[j] - 'A'] > freq[s->order[j - 1] - 'A']; j--) {
            char t = s->order[j];
            s->order[j] = s->order[j - 1];
            s->order[j - 1] = t;
        }

    // unique words, sorted A-Z, into groups by shape
    qsort(ws.w, (size_t)ws.n, sizeof *ws.w, cmp_str);
    int too_long = 0;
    for (int i = 0; i < ws.n; i++) {
        if (i && !strcmp(ws.w[i], ws.w[i - 1])) continue;
        if (strlen(ws.w[i]) > MAX_WORD) {
            too_long++;
            continue;
        }
        char sh[MAX_WORD + 1];
        shape_of(ws.w[i], sh);
        int gi = -1;
        for (int k = 0; k < s->ng; k++)
            if (!strcmp(s->g[k].shape, sh)) { gi = k; break; }
        if (gi < 0) {
            s->g = grow(s->g, &s->capg, s->ng + 1, sizeof *s->g);
            gi = s->ng++;
            memset(&s->g[gi], 0, sizeof s->g[gi]);
            snprintf(s->g[gi].shape, sizeof s->g[gi].shape, "%s", sh);
        }
        Group *g = &s->g[gi];
        int cap = g->cap;
        g->words = grow(g->words, &g->cap, g->n + 1, sizeof *g->words);
        if (g->cap != cap) {
            g->P = realloc(g->P, (size_t)g->cap * sizeof *g->P);
            if (!g->P) exit(1);
        }
        g->words[g->n] = ws.w[i];
        posmasks(ws.w[i], g->P[g->n]);
        g->n++;
    }
    if (too_long) fprintf(stderr, "smart-solve: %d words over %d letters left out\n", too_long, MAX_WORD);
    // sort groups by shape, for find_group
    for (int i = 1; i < s->ng; i++)
        for (int j = i; j > 0 && strcmp(s->g[j].shape, s->g[j - 1].shape) < 0; j--) {
            Group t = s->g[j];
            s->g[j] = s->g[j - 1];
            s->g[j - 1] = t;
        }

    // category weights
    for (int i = 0; i < npz; i++) {
        Category *c = category(s, pz[i].category, 1);
        int nr = runs(pz[i].answer, r, MAX_SPANS);
        for (int k = 0; k < nr; k++) {
            if (!has_letter(r[k].key) || strlen(r[k].key) > MAX_WORD) continue;
            char sh[MAX_WORD + 1];
            shape_of(r[k].key, sh);
            int gi = find_group(s, sh), wi = gi < 0 ? -1 : find_word(&s->g[gi], r[k].key);
            if (wi < 0) continue;
            Sparse *sp = &c->by_group[gi];
            int j = 0;
            while (j < sp->n && sp->idx[j] != wi) j++;
            if (j == sp->n) {
                int cap = sp->cap;
                sp->idx = grow(sp->idx, &sp->cap, sp->n + 1, sizeof *sp->idx);
                if (sp->cap != cap) {
                    sp->count = realloc(sp->count, (size_t)sp->cap * sizeof *sp->count);
                    if (!sp->count) exit(1);
                }
                sp->idx[sp->n] = wi;
                sp->count[sp->n++] = 0;
            }
            sp->count[j]++;
        }
    }
    free(pz);
    free(ws.w);   // the strings now belong to the groups (duplicates leak, once)
    return s;
}

void smart_free(Smart *s) {
    if (!s) return;
    for (int i = 0; i < s->ng; i++) {
        for (int k = 0; k < s->g[i].n; k++) free(s->g[i].words[k]);
        free(s->g[i].words);
        free(s->g[i].P);
    }
    for (int i = 0; i < s->ncat; i++) {
        for (int k = 0; k < s->ng; k++) free(s->cat[i].by_group[k].idx), free(s->cat[i].by_group[k].count);
        free(s->cat[i].by_group);
        free(s->cat[i].name);
    }
    free(s->g);
    free(s->cat);
    free(s->wbuf);
    free(s);
}

// Weights of every word in group gi for a category: 1 + its count there.
// Valid until the next call.
static const double *weights(Smart *s, const char *cat, int gi) {
    const Group *g = &s->g[gi];
    s->wbuf = grow(s->wbuf, &s->wcap, g->n, sizeof *s->wbuf);
    for (int i = 0; i < g->n; i++) s->wbuf[i] = 1.0;
    const Category *c = category(s, cat, 0);
    if (c) {
        const Sparse *sp = &c->by_group[gi];
        for (int j = 0; j < sp->n; j++) s->wbuf[sp->idx[j]] += sp->count[j];
    }
    return s->wbuf;
}

// --------------------------------------------------------------- view --

typedef struct {
    int blanks[MAX_TEXT], nblank;   // board positions of its '_' squares
    char key[MAX_TEXT];
    int group;                      // -1: no word of this shape
    int *idx, n;                    // candidates, ascending
    double *w;                      // their weights
    double total;
} Span;

typedef struct {
    Span sp[MAX_SPANS];
    int n;
} View;

static void view_free(View *v) {
    for (int i = 0; i < v->n; i++) free(v->sp[i].idx), free(v->sp[i].w);
}

static int view(Smart *s, const char *cat, const char *board, const char *called, View *v) {
    Run r[MAX_SPANS];
    int nr = runs(board, r, MAX_SPANS);
    v->n = 0;
    for (int k = 0; k < nr; k++) {
        Span *sp = &v->sp[v->n];
        sp->nblank = 0;
        for (int i = 0; i < r[k].len; i++)
            if (board[r[k].start + i] == '_') sp->blanks[sp->nblank++] = r[k].start + i;
        if (!sp->nblank) continue;
        memcpy(sp->key, r[k].key, sizeof sp->key);
        sp->idx = NULL, sp->w = NULL, sp->n = 0, sp->total = 0;
        char sh[MAX_TEXT];
        shape_of(sp->key, sh);
        sp->group = strlen(sh) > MAX_WORD ? -1 : find_group(s, sh);
        v->n++;
        if (sp->group < 0) continue;
        const Group *g = &s->g[sp->group];
        int32_t want[26];
        posmasks(sp->key, want);
        sp->idx = malloc((size_t)g->n * sizeof *sp->idx);
        if (!sp->idx) exit(1);
        for (int i = 0; i < g->n; i++) {
            int ok = 1;
            for (const char *c = called; *c && ok; c++) ok = g->P[i][*c - 'A'] == want[*c - 'A'];
            if (ok) sp->idx[sp->n++] = i;
        }
        const double *w = weights(s, cat, sp->group);
        sp->w = malloc((size_t)(sp->n ? sp->n : 1) * sizeof *sp->w);
        if (!sp->w) exit(1);
        for (int i = 0; i < sp->n; i++) sp->w[i] = w[sp->idx[i]];
        sp->total = pw_sum(sp->w, sp->n);
    }
    return 0;
}

// Fill each span with its heaviest candidate (the first, on a tie).
static void fill(const Smart *s, const char *board, const View *v, char *out) {
    strcpy(out, board);
    for (int k = 0; k < v->n; k++) {
        const Span *sp = &v->sp[k];
        if (!sp->n) continue;
        int best = 0;
        for (int i = 1; i < sp->n; i++)
            if (sp->w[i] > sp->w[best]) best = i;
        const char *word = s->g[sp->group].words[sp->idx[best]];
        int b = 0;
        for (int i = 0; sp->key[i] && b < sp->nblank; i++)
            if (sp->key[i] == '_') out[sp->blanks[b++]] = word[i];
    }
}

static int can_buy(const char *board, const char *called) {
    int cons = 0;
    for (const char *c = called; *c; c++) {
        if (is_vowel(*c)) continue;
        cons++;
        if (strchr(board, *c)) return 1;
    }
    return cons == 21;
}

// ---------------------------------------------------------- smart pick --

typedef struct {
    int32_t v;
    double w;
} VW;

static int cmp_vw(const void *a, const void *b) {
    int32_t x = ((const VW *)a)->v, y = ((const VW *)b)->v;
    return (x > y) - (x < y);
}

// Expected log2(candidates left) for one word after letter c.
static double word_score(const Smart *s, const Span *sp, int c, VW *buf, double *terms) {
    const Group *g = &s->g[sp->group];
    for (int i = 0; i < sp->n; i++) buf[i] = (VW){g->P[sp->idx[i]][c], sp->w[i]};
    qsort(buf, (size_t)sp->n, sizeof *buf, cmp_vw);
    int nt = 0;
    for (int i = 0; i < sp->n;) {
        int j = i;
        double W = 0;
        while (j < sp->n && buf[j].v == buf[i].v) W += buf[j++].w;
        terms[nt++] = W / sp->total * log2((double)(j - i));
        i = j;
    }
    return pw_sum(terms, nt);
}

static char smart_pick(const Smart *s, const View *v, const char *called, int vowels) {
    char letters[27];
    int nl = 0;
    for (int i = 0; i < 26; i++) {
        char c = s->order[i];
        if (!strchr(called, c) && (vowels || !is_vowel(c))) letters[nl++] = c;
    }
    if (!nl) return 0;
    double totals[26] = {0};
    int maxn = 1;
    for (int k = 0; k < v->n; k++)
        if (v->sp[k].n > maxn) maxn = v->sp[k].n;
    VW *buf = malloc((size_t)maxn * sizeof *buf);
    double *terms = malloc((size_t)maxn * sizeof *terms);
    if (!buf || !terms) exit(1);
    for (int k = 0; k < v->n; k++) {
        const Span *sp = &v->sp[k];
        if (sp->n <= 1) continue;
        for (int i = 0; i < nl; i++) totals[i] += word_score(s, sp, letters[i] - 'A', buf, terms);
    }
    free(buf);
    free(terms);
    int best = 0;   // letters are in order, so the first of a tie is the more common
    double bs = py_round(totals[0], 9);
    for (int i = 1; i < nl; i++) {
        double r = py_round(totals[i], 9);
        if (r < bs) best = i, bs = r;
    }
    return letters[best];
}

// ------------------------------------------------------------ best set --

typedef struct {
    int n, ncode[26];
    int *code[26];      // per letter, per candidate (heaviest first): rank of its mask
    double *w, total;
} Live;

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

// A set of 64-bit keys, cleared in O(1) by bumping the stamp.
typedef struct {
    int64_t *key;
    unsigned *stamp, now;
    size_t mask;
} KeySet;

static void keyset_init(KeySet *ks, int n) {
    size_t cap = 16;
    while (cap < (size_t)n * 2) cap *= 2;
    ks->key = malloc(cap * sizeof *ks->key);
    ks->stamp = calloc(cap, sizeof *ks->stamp);
    if (!ks->key || !ks->stamp) exit(1);
    ks->mask = cap - 1;
    ks->now = 0;
}

static int keyset_add(KeySet *ks, int64_t k) {   // 1 when k is new
    uint64_t h = (uint64_t)k * 0x9E3779B97F4A7C15ull;
    size_t i = (size_t)(h >> 32) & ks->mask;
    while (ks->stamp[i] == ks->now) {
        if (ks->key[i] == k) return 0;
        i = (i + 1) & ks->mask;
    }
    ks->stamp[i] = ks->now;
    ks->key[i] = k;
    return 1;
}

static int best_set(const Smart *s, const View *v, const char *cons, const char *vows, char *chosen_out) {
    int rank[26];
    for (int i = 0; i < 26; i++) rank[s->order[i] - 'A'] = i;
    Live live[MAX_SPANS];
    int nlive = 0, maxn = 1;
    char letters[27];
    snprintf(letters, sizeof letters, "%s%s", cons, vows);
    for (int k = 0; k < v->n; k++) {
        const Span *sp = &v->sp[k];
        if (sp->n < 2) continue;
        Live *L = &live[nlive++];
        L->n = sp->n;
        if (sp->n > maxn) maxn = sp->n;
        // heaviest first, stable
        int *o = malloc((size_t)sp->n * sizeof *o);
        if (!o) exit(1);
        for (int i = 0; i < sp->n; i++) o[i] = i;
        {   // stable merge sort by descending weight
            int *tmp = malloc((size_t)sp->n * sizeof *tmp);
            if (!tmp) exit(1);
            for (int width = 1; width < sp->n; width *= 2)
                for (int lo = 0; lo < sp->n; lo += 2 * width) {
                    int mid = lo + width < sp->n ? lo + width : sp->n;
                    int hi = lo + 2 * width < sp->n ? lo + 2 * width : sp->n;
                    int a = lo, b = mid, t = lo;
                    while (a < mid && b < hi) tmp[t++] = sp->w[o[b]] > sp->w[o[a]] ? o[b++] : o[a++];
                    while (a < mid) tmp[t++] = o[a++];
                    while (b < hi) tmp[t++] = o[b++];
                    memcpy(o + lo, tmp + lo, (size_t)(hi - lo) * sizeof *o);
                }
            free(tmp);
        }
        L->w = malloc((size_t)sp->n * sizeof *L->w);
        if (!L->w) exit(1);
        for (int i = 0; i < sp->n; i++) L->w[i] = sp->w[o[i]];
        L->total = sp->total;
        const Group *g = &s->g[sp->group];
        int32_t *vals = malloc((size_t)sp->n * sizeof *vals), *uniq = malloc((size_t)sp->n * sizeof *uniq);
        if (!vals || !uniq) exit(1);
        memset(L->code, 0, sizeof L->code);
        for (const char *c = letters; *c; c++) {
            int ci = *c - 'A';
            for (int i = 0; i < sp->n; i++) vals[i] = g->P[sp->idx[o[i]]][ci];
            memcpy(uniq, vals, (size_t)sp->n * sizeof *uniq);
            qsort(uniq, (size_t)sp->n, sizeof *uniq, cmp_i32);
            int nu = 0;
            for (int i = 0; i < sp->n; i++)
                if (!i || uniq[i] != uniq[nu - 1]) uniq[nu++] = uniq[i];
            L->code[ci] = malloc((size_t)sp->n * sizeof *L->code[ci]);
            if (!L->code[ci]) exit(1);
            for (int i = 0; i < sp->n; i++) {
                int lo = 0, hi = nu - 1;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (uniq[mid] < vals[i]) lo = mid + 1;
                    else hi = mid;
                }
                L->code[ci][i] = lo;
            }
            L->ncode[ci] = nu;
        }
        free(vals);
        free(uniq);
        free(o);
    }

    KeySet ks;
    keyset_init(&ks, maxn);
    int64_t *key = malloc((size_t)maxn * sizeof *key);
    if (!key) exit(1);
    int nc = (int)strlen(cons), nv = (int)strlen(vows);
    double best_p = 0, best_tie[4] = {0};
    int have = 0;
    char best[5] = {0};
    for (int a = 0; a < nc; a++)
        for (int b = a + 1; b < nc; b++)
            for (int c = b + 1; c < nc; c++)
                for (int d = 0; d < nv; d++) {
                    char ch[4] = {cons[a], cons[b], cons[c], vows[d]};
                    double p = 1.0;
                    for (int l = 0; l < nlive; l++) {
                        const Live *L = &live[l];
                        for (int i = 0; i < L->n; i++) key[i] = 0;
                        for (int q = 0; q < 4; q++) {
                            int ci = ch[q] - 'A', n = L->ncode[ci];
                            const int *code = L->code[ci];
                            for (int i = 0; i < L->n; i++) key[i] = key[i] * n + code[i];
                        }
                        ks.now++;
                        double S = 0;
                        for (int i = 0; i < L->n; i++)
                            if (keyset_add(&ks, key[i])) S += L->w[i];
                        p *= S / L->total;
                        if (have && p < best_p - 1e-12) break;
                    }
                    // key (round(p, 12), [-rank, ...] of the ranks sorted ascending), the largest wins
                    double rp = py_round(p, 12);
                    double tie[4];
                    int r4[4] = {rank[ch[0] - 'A'], rank[ch[1] - 'A'], rank[ch[2] - 'A'], rank[ch[3] - 'A']};
                    for (int i = 1; i < 4; i++)
                        for (int j = i; j > 0 && r4[j] < r4[j - 1]; j--) {
                            int t = r4[j];
                            r4[j] = r4[j - 1];
                            r4[j - 1] = t;
                        }
                    for (int i = 0; i < 4; i++) tie[i] = -r4[i];
                    int better = !have || rp > best_p;
                    if (have && rp == best_p)
                        for (int i = 0; i < 4; i++)
                            if (tie[i] != best_tie[i]) {
                                better = tie[i] > best_tie[i];
                                break;
                            }
                    if (better) {
                        have = 1, best_p = rp;
                        memcpy(best_tie, tie, sizeof tie);
                        memcpy(best, ch, 4);
                    }
                }
    free(key);
    free(ks.key);
    free(ks.stamp);
    for (int l = 0; l < nlive; l++) {
        for (int c = 0; c < 26; c++) free(live[l].code[c]);
        free(live[l].w);
    }
    if (!have) return -1;
    memcpy(chosen_out, best, 4);
    return 0;
}

// --------------------------------------------------------------- moves --

int smart_main(Smart *s, const char *cat, const char *board, const char *called, char *out, size_t n) {
    View v;
    view(s, cat, board, called, &v);
    int pinned = 1;
    for (int k = 0; k < v.n; k++) pinned &= v.sp[k].n == 1;
    int rc = 0;
    if (pinned) {
        char filled[MAX_TEXT];
        fill(s, board, &v, filled);
        snprintf(out, n, "SOLVE %s", filled);
    } else {
        char c = smart_pick(s, &v, called, can_buy(board, called));
        if (c) snprintf(out, n, "%s %c", is_vowel(c) ? "BUY" : "CALL", c);
        else rc = -1;
    }
    view_free(&v);
    return rc;
}

int smart_bonus_pick(Smart *s, const char *cat, const char *board, const char *called, char *out, size_t n) {
    char cons[27] = {0}, vows[6] = {0};
    int nc = 0, nv = 0;
    for (int i = 0; i < 26; i++) {
        char c = s->order[i];
        if (strchr(called, c)) continue;
        if (is_vowel(c)) vows[nv++] = c;
        else cons[nc++] = c;
    }
    View v;
    view(s, cat, board, called, &v);
    char chosen[4];
    int rc = best_set(s, &v, cons, vows, chosen);
    view_free(&v);
    if (rc) return -1;
    char c3[4] = {chosen[0], chosen[1], chosen[2], 0};
    for (int i = 1; i < 3; i++)
        for (int j = i; j > 0 && c3[j] < c3[j - 1]; j--) {
            char t = c3[j];
            c3[j] = c3[j - 1];
            c3[j - 1] = t;
        }
    snprintf(out, n, "PICK %s %c", c3, chosen[3]);
    return 0;
}

int smart_bonus_solve(Smart *s, const char *cat, const char *board, const char *called, char *out, size_t n) {
    View v;
    view(s, cat, board, called, &v);
    char filled[MAX_TEXT];
    fill(s, board, &v, filled);
    view_free(&v);
    snprintf(out, n, "SOLVE %s", filled);
    return 0;
}
