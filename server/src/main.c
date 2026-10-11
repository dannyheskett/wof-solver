// wof-server: the whole WOF Solver demo in one process. It runs
// transformer-solve (the 59M-parameter transformer) and smart-solve, serves
// the move API the WASM client calls, and serves the client itself.
//
//   POST /move
//   {"mode": "main" | "bonus_pick" | "bonus_solve",
//    "category": "Around The House",
//    "board": "A _A__L_ __ ___ ______",
//    "called": "AL",
//    "solvers": ["smart", "transformer"]}      optional, default both
//
//   -> {"smart": {"move": "CALL N", "ms": 3},
//       "transformer": {"move": "CALL N", "confidence": ..., "letters": [...],
//                       "turned_down": ..., "ms": 140}}
//
//   GET /categories          [{"name": "Thing", "auto": true}, ...]
//   GET /puzzle              {"category": ..., "answer": ...}, a random control
//                            puzzle, every puzzle equally likely
//   GET /puzzle?category=X   a random control puzzle in category X
//   GET /presets             the demo examples (scripts/13_build_demo.py)
//   GET /health
//   GET /, /<file>           the client, from WEB_ROOT, loaded at start
//
//   POST /decode[?solve=1[&mode=main|bonus_pick|bonus_solve]]
//   body: a JPEG or PNG screencap or photo of the puzzle board, up to
//   DECODE_MAX_BYTES, sent as is (Content-Type image/jpeg or image/png)
//   -> {"found": true, "present": 0.99,
//       "category": "Food And Drink", "category_read": "FOOD & DRINK",
//       "board": "A _A__L_ __ ___ ______", "called": "AL",
//       "rows": ["#A#_A__L_####", ...]        4 rows of 14: # unused, _ hidden,
//                                             * lit, ~ covered, a letter or
//                                             punctuation shown, a space off
//                                             the board's corners
//       "guessed": [[row, col], ...],        lit or covered squares in play,
//                                             counted as hidden
//       "unsure": [[row, col], ...],         squares read under 80% confidence
//       "ms": 640}
//   With no board in the image: {"found": false, "present": 0.02, "ms": ...}.
//   A board counts as found when present is at least 0.5 and at most
//   MAX_UNSURE squares are unsure; otherwise "found" is false and "unsure"
//   gives the count. Real board frames have 7 or fewer unsure squares 99%
//   of the time; title cards and close-ups that fool present have dozens.
//   "called" is the letters showing: a screencap does not show misses. With
//   solve=1 the reply also carries "moves": what /move answers for that board
//   (mode main unless given; the bonus modes add R S T L N E to called), or
//   "moves_error" when the board cannot be played, and "guess":
//   {"answer": ..., "confidence": 0.94}, transformer-solve's best SOLVE for
//   the board however unsure (its MAIN prompt with only a SOLVE allowed):
//   the single answer for snap and solve. A board with no hidden squares is
//   its own answer, confidence 1. The board decoder is
//   scripts/14_vision/decoder_net.py (server/src/decoder.c). /decode has its
//   own limit, DECODE_RATE a minute per client, and reads one image at a time.
//
// The caller holds the answer, shows letters and keeps score. /move only sees
// what a contestant sees. "called" is the letters called so far in any order
// ("" before any call). In bonus_pick the board shows R S T L N E and called
// includes them. "confidence" and "letters" come with every transformer-solve
// SOLVE, kept or turned down: each character's probability among those its
// square allows, and their product. A MAIN SOLVE under MIN_P while a square is
// blank is turned down (its answer in "turned_down") and a letter is called.
//
// Each connection gets a thread. Moves run one at a time per solver: there is
// one model state and one CPU. "ms" is a move's compute time, not counting
// time spent waiting for the solver.
//
// Settings, from the environment:
//   PORT (8080), MIN_P (0.99), RATE (600 requests a minute per client),
//   DECODE_RATE (10 a minute per client), DECODE_MAX_BYTES (4 MB), DECODER
//   (the decoder weights, models/decoder.bin; without it /decode answers 503),
//   CORS_ORIGINS (comma-separated, default https://danheskett.com),
//   WEIGHTS, WORDLIST, PUZZLES (comma-separated), DEMO, WEB_ROOT (paths,
//   defaults below).
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <malloc.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "cJSON.h"
#include "decoder.h"
#include "image.h"
#include "model.h"
#include "picohttpparser.h"
#include "smart.h"
#include "transformer.h"

#define MAX_HEADERS 32
#define MAX_REQUEST 8192
#define MAX_BODY 4096
#define MAX_PIXELS (16 * 1000 * 1000)   // the largest image /decode decodes
#define MAX_UNSURE 12                    // unsure squares a found board may have
#define MAX_CONNS 64
#define IDLE_SECONDS 30
#define MAX_NEW 80     // the longest move the Python service allowed for

static const char *FREE = "RSTLNE";

static double env_num(const char *name, double def) {
    const char *v = getenv(name);
    return v && *v ? atof(v) : def;
}

static const char *env_str(const char *name, const char *def) {
    const char *v = getenv(name);
    return v && *v ? v : def;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

// ------------------------------------------------------------- engines --

static Model model;
static Seq *seq;
static Smart *smart;
static pthread_mutex_t model_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t smart_lock = PTHREAD_MUTEX_INITIALIZER;
static double min_p;
static Decoder *decoder;   // NULL when no decoder weights were found
static pthread_mutex_t decode_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t decode_max_bytes;

// ---------------------------------------------------------------- demo --

typedef struct {
    char *category, *answer;
} Puzzle;

static Puzzle *puzzles;
static int npuzzles;
static char *categories_json, *presets_json;
static char **category_names;   // the categories' names, for /decode
static int ncategory_names;
static pthread_mutex_t rng_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t rng_state;

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = 0;
    if (len) *len = (size_t)n;
    return buf;
}

static char *copy_str(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (!d) exit(1);
    memcpy(d, s, n);
    return d;
}

static int load_demo(const char *path) {
    char *text = read_file(path, NULL);
    cJSON *root = text ? cJSON_Parse(text) : NULL;
    free(text);
    if (!root) return -1;
    cJSON *cats = cJSON_GetObjectItem(root, "categories"), *ps = cJSON_GetObjectItem(root, "puzzles");
    cJSON *pre = cJSON_GetObjectItem(root, "presets");
    if (!cJSON_IsArray(cats) || !cJSON_IsArray(ps) || !cJSON_IsArray(pre)) {
        cJSON_Delete(root);
        return -1;
    }
    categories_json = cJSON_PrintUnformatted(cats);
    cJSON *cat;
    cJSON_ArrayForEach(cat, cats) {
        cJSON *name = cJSON_GetObjectItem(cat, "name");
        if (!cJSON_IsString(name)) continue;
        category_names = realloc(category_names, (size_t)(ncategory_names + 1) * sizeof *category_names);
        category_names[ncategory_names++] = copy_str(name->valuestring);
    }
    presets_json = cJSON_PrintUnformatted(pre);
    npuzzles = cJSON_GetArraySize(ps);
    puzzles = calloc((size_t)npuzzles, sizeof *puzzles);
    int i = 0;
    cJSON *p;
    cJSON_ArrayForEach(p, ps) {
        cJSON *c = cJSON_GetArrayItem(p, 0), *a = cJSON_GetArrayItem(p, 1);
        if (!cJSON_IsString(c) || !cJSON_IsString(a)) continue;
        puzzles[i].category = copy_str(c->valuestring);
        puzzles[i].answer = copy_str(a->valuestring);
        i++;
    }
    npuzzles = i;
    cJSON_Delete(root);
    return 0;
}

// xorshift64*, seeded from the clock: the puzzle draw needs no more.
static uint64_t rnd(void) {
    pthread_mutex_lock(&rng_lock);
    uint64_t x = rng_state;
    x ^= x >> 12, x ^= x << 25, x ^= x >> 27;
    rng_state = x;
    pthread_mutex_unlock(&rng_lock);
    return x * 0x2545F4914F6CDD1Dull;
}

// ---------------------------------------------------------- web files --

typedef struct {
    char *name, *body;
    size_t len;
    const char *type;
} WebFile;

static WebFile *web;
static int nweb;

static const char *content_type(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return "application/octet-stream";
    if (!strcmp(dot, ".html")) return "text/html; charset=utf-8";
    if (!strcmp(dot, ".js")) return "text/javascript";
    if (!strcmp(dot, ".wasm")) return "application/wasm";
    if (!strcmp(dot, ".css")) return "text/css";
    if (!strcmp(dot, ".png")) return "image/png";
    if (!strcmp(dot, ".ico")) return "image/x-icon";
    return "application/octet-stream";
}

// Every regular file directly in dir, read into memory. Requests match a
// name exactly, so no path ever reaches the filesystem.
static void load_web(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "web root %s not found, serving the API only\n", dir);
        return;
    }
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) || !S_ISREG(st.st_mode)) continue;
        size_t len;
        char *body = read_file(path, &len);
        if (!body) continue;
        web = realloc(web, (size_t)(nweb + 1) * sizeof *web);
        web[nweb++] = (WebFile){copy_str(e->d_name), body, len, content_type(e->d_name)};
    }
    closedir(d);
}

static const WebFile *web_file(const char *name) {
    for (int i = 0; i < nweb; i++)
        if (!strcmp(web[i].name, name)) return &web[i];
    return NULL;
}

// --------------------------------------------------------- rate limit --

// A token bucket per client: RATE requests a minute, bursts up to RATE.
// Buckets live in a fixed table, keyed by a hash of the client address; a
// collision shares a bucket, which only makes the limit stricter.
// /decode has its own table and rate (DECODE_RATE), so decodes do not use up
// a client's moves.
#define BUCKETS 4096
typedef struct {
    struct {
        double tokens, last;
        int used;
    } bucket[BUCKETS];
    pthread_mutex_t lock;
    double rate;
} Limiter;

static Limiter moves = {.lock = PTHREAD_MUTEX_INITIALIZER}, decodes = {.lock = PTHREAD_MUTEX_INITIALIZER};

static int limit(Limiter *L, const char *client) {
    uint64_t h = 1469598103934665603ull;
    for (const char *c = client; *c; c++) h = (h ^ (unsigned char)*c) * 1099511628211ull;
    pthread_mutex_lock(&L->lock);
    double t = now();
    int i = (int)(h % BUCKETS);
    if (!L->bucket[i].used) L->bucket[i].used = 1, L->bucket[i].tokens = L->rate, L->bucket[i].last = t;
    L->bucket[i].tokens += (t - L->bucket[i].last) * L->rate / 60;
    if (L->bucket[i].tokens > L->rate) L->bucket[i].tokens = L->rate;
    L->bucket[i].last = t;
    int ok = L->bucket[i].tokens >= 1;
    if (ok) L->bucket[i].tokens -= 1;
    pthread_mutex_unlock(&L->lock);
    return ok;
}

static int allow(const char *client) { return limit(&moves, client); }

// ---------------------------------------------------------------- HTTP --

static char **origins;
static int norigins;

typedef struct {
    int fd;
    char peer[64];
} Conn;

typedef struct {
    const char *method, *path;
    size_t method_len, path_len;
    struct phr_header headers[MAX_HEADERS];
    size_t nheaders;
    int minor;
    const char *body;
    size_t body_len;
    char origin[256], client[128];
    int close;
} Request;

static int send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k, n -= (size_t)k;
    }
    return 0;
}

static const char *reason(int code) {
    switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

static int allowed_origin(const char *o) {
    for (int i = 0; i < norigins; i++)
        if (!strcmp(origins[i], o)) return 1;
    return 0;
}

static int respond(int fd, const Request *rq, int code, const char *type, const char *body, size_t len,
                   const char *cache) {
    char head[1024];
    int n = snprintf(head, sizeof head, "HTTP/1.1 %d %s\r\nContent-Length: %zu\r\n", code, reason(code), len);
    if (type) n += snprintf(head + n, sizeof head - (size_t)n, "Content-Type: %s\r\n", type);
    if (cache) n += snprintf(head + n, sizeof head - (size_t)n, "Cache-Control: %s\r\n", cache);
    if (rq->origin[0] && allowed_origin(rq->origin))
        n += snprintf(head + n, sizeof head - (size_t)n,
                      "Access-Control-Allow-Origin: %s\r\nVary: Origin\r\n"
                      "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: Content-Type\r\n",
                      rq->origin);
    n += snprintf(head + n, sizeof head - (size_t)n, "Connection: %s\r\n\r\n", rq->close ? "close" : "keep-alive");
    if (send_all(fd, head, (size_t)n)) return -1;
    return len ? send_all(fd, body, len) : 0;
}

static int json_reply(int fd, const Request *rq, int code, const char *json) {
    return respond(fd, rq, code, "application/json", json, strlen(json), "no-store");
}

static int error_reply(int fd, const Request *rq, int code, const char *msg) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "error", msg);
    char *s = cJSON_PrintUnformatted(o);
    int rc = json_reply(fd, rq, code, s);
    cJSON_free(s);
    cJSON_Delete(o);
    return rc;
}

static int url_decode(const char *in, size_t n, char *out, size_t cap);

// ---------------------------------------------------------------- /move --

static const char *MODES[] = {"main", "bonus_pick", "bonus_solve"};

static int in_alphabet(char c) { return c != '|' && c != '\n' && model_token(&model, c) >= 0; }

// The request as fields, or an error message.
static const char *check(cJSON *st, int *mode, char *category, char *board, char *called, int *want) {
    if (!cJSON_IsObject(st)) return "body must be a JSON object";
    cJSON *jm = cJSON_GetObjectItem(st, "mode");
    *mode = -1;
    for (int i = 0; i < 3; i++)
        if (cJSON_IsString(jm) && !strcmp(jm->valuestring, MODES[i])) *mode = i;
    if (*mode < 0) return "mode must be main, bonus_pick or bonus_solve";
    cJSON *jc = cJSON_GetObjectItem(st, "category"), *jb = cJSON_GetObjectItem(st, "board");
    cJSON *jk = cJSON_GetObjectItem(st, "called"), *js = cJSON_GetObjectItem(st, "solvers");
    want[0] = want[1] = 1;
    if (js) {
        want[0] = want[1] = 0;
        int n = 0;
        cJSON *x;
        if (!cJSON_IsArray(js)) return "solvers must be a list of smart and/or transformer";
        cJSON_ArrayForEach(x, js) {
            if (cJSON_IsString(x) && !strcmp(x->valuestring, "smart")) want[0] = 1;
            else if (cJSON_IsString(x) && !strcmp(x->valuestring, "transformer")) want[1] = 1;
            else return "solvers must be a list of smart and/or transformer";
            n++;
        }
        if (!n) return "solvers must be a list of smart and/or transformer";
    }
    if (!cJSON_IsString(jc) || !cJSON_IsString(jb) || !jc->valuestring[0] || !jb->valuestring[0])
        return "category and board are required";
    if (strlen(jc->valuestring) > 64 || strlen(jb->valuestring) > TF_MAX_TEXT) return "board too long";
    strcpy(category, jc->valuestring);
    strcpy(board, jb->valuestring);
    for (const char *c = category; *c; c++) {
        char u = *c >= 'a' && *c <= 'z' ? (char)(*c - 32) : *c;
        if (!in_alphabet(u)) return "category and board may only use A-Z, space and ' & - . , ! ?";
    }
    for (const char *c = board; *c; c++)
        if (!in_alphabet(*c)) return "category and board may only use A-Z, space and ' & - . , ! ?";
    // called: upper case, each letter once, sorted
    int seen[26] = {0};
    if (jk && !cJSON_IsString(jk)) return "called must be letters A-Z";
    for (const char *c = jk ? jk->valuestring : ""; *c; c++) {
        char u = *c >= 'a' && *c <= 'z' ? (char)(*c - 32) : *c;
        if (u < 'A' || u > 'Z') return "called must be letters A-Z";
        seen[u - 'A'] = 1;
    }
    int n = 0;
    for (int i = 0; i < 26; i++)
        if (seen[i]) called[n++] = (char)('A' + i);
    called[n] = 0;
    if ((int)(strlen(category) + strlen(board) + strlen(called)) + 10 + MAX_NEW > model.block)
        return "board too long";
    if (*mode != 0)
        for (const char *c = FREE; *c; c++)
            if (!seen[*c - 'A']) return "a BONUS board has R S T L N E called";
    return NULL;
}

static double py_round(double x, int nd) {
    char buf[64];
    snprintf(buf, sizeof buf, "%.*f", nd, x);
    return strtod(buf, NULL);
}

// Both solvers' moves for a checked board, added to out. Returns NULL, or
// the error /move answers 400 with.
static const char *solve(cJSON *out, int mode, const char *category, const char *board, const char *called,
                         const int *want) {
    if (want[0]) {
        char mv[256];
        pthread_mutex_lock(&smart_lock);
        double t0 = now();
        int rc = mode == 0   ? smart_main(smart, category, board, called, mv, sizeof mv)
                 : mode == 1 ? smart_bonus_pick(smart, category, board, called, mv, sizeof mv)
                             : smart_bonus_solve(smart, category, board, called, mv, sizeof mv);
        double ms = 1000 * (now() - t0);
        pthread_mutex_unlock(&smart_lock);
        if (rc) return "smart-solve has no move for this board";
        cJSON *o = cJSON_AddObjectToObject(out, "smart");
        cJSON_AddStringToObject(o, "move", mv);
        cJSON_AddNumberToObject(o, "ms", (double)(long)(ms + 0.5));
    }
    if (want[1]) {
        TfMove mv;
        pthread_mutex_lock(&model_lock);
        double t0 = now();
        int rc = mode == 0   ? tf_main(&model, seq, category, board, called, min_p, &mv)
                 : mode == 1 ? tf_bonus_pick(&model, seq, category, board, called, &mv)
                             : tf_bonus_solve(&model, seq, category, board, called, &mv);
        double ms = 1000 * (now() - t0);
        pthread_mutex_unlock(&model_lock);
        if (rc) return "board too long";
        cJSON *o = cJSON_AddObjectToObject(out, "transformer");
        cJSON_AddStringToObject(o, "move", mv.move);
        if (mv.has_conf) {
            cJSON_AddNumberToObject(o, "confidence", py_round(mv.confidence, 4));
            cJSON *l = cJSON_AddArrayToObject(o, "letters");
            for (int i = 0; i < mv.nletters; i++) cJSON_AddItemToArray(l, cJSON_CreateNumber(py_round(mv.letters[i], 3)));
        }
        if (mv.turned_down[0]) cJSON_AddStringToObject(o, "turned_down", mv.turned_down);
        cJSON_AddNumberToObject(o, "ms", (double)(long)(ms + 0.5));
    }
    return NULL;
}

static int do_move(int fd, const Request *rq) {
    if (rq->body_len > MAX_BODY) return error_reply(fd, rq, 413, "body too large");
    char body[MAX_BODY + 1];
    memcpy(body, rq->body, rq->body_len);
    body[rq->body_len] = 0;
    cJSON *st = cJSON_Parse(body);
    if (!st) return error_reply(fd, rq, 400, "body must be JSON");
    int mode, want[2];
    char category[80], board[TF_MAX_TEXT + 1], called[27];
    const char *err = check(st, &mode, category, board, called, want);
    cJSON_Delete(st);
    if (err) return error_reply(fd, rq, 400, err);

    cJSON *out = cJSON_CreateObject();
    err = solve(out, mode, category, board, called, want);
    if (err) {
        cJSON_Delete(out);
        return error_reply(fd, rq, 400, err);
    }
    char *s = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
    int rc = json_reply(fd, rq, 200, s);
    cJSON_free(s);
    return rc;
}

// -------------------------------------------------------------- /decode --

// The category as written on screen (FOOD & DRINK, LIVING THINGS) as one of
// the solvers' names (Food And Drink, Living Thing): upper case, & as AND,
// then an exact match, the same without a final S, or the closest name, it
// or its singular at most 25% different. Anything else comes back
// title-cased as written, and the solvers weigh its words equally. As
// board.category_from_strip.
static int edit_distance(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b), prev[129], cur[129];
    if (la > 128 || lb > 128) return 1 << 20;
    for (int j = 0; j <= lb; j++) prev[j] = j;
    for (int i = 1; i <= la; i++) {
        cur[0] = i;
        for (int j = 1; j <= lb; j++) {
            int c = prev[j - 1] + (a[i - 1] != b[j - 1]);
            if (prev[j] + 1 < c) c = prev[j] + 1;
            if (cur[j - 1] + 1 < c) c = cur[j - 1] + 1;
            cur[j] = c;
        }
        memcpy(prev, cur, sizeof(int) * (size_t)(lb + 1));
    }
    return prev[lb];
}

static void lower(const char *in, char *out, size_t cap) {
    size_t j = 0;
    for (; in[j] && j + 1 < cap; j++) out[j] = in[j] >= 'A' && in[j] <= 'Z' ? (char)(in[j] + 32) : in[j];
    out[j] = 0;
}

static void map_category(const char *read, char *out, size_t cap) {
    char t[128];
    size_t n = 0;
    for (const char *c = read; *c && n + 6 < sizeof t; c++) {
        char u = *c >= 'a' && *c <= 'z' ? (char)(*c - 32) : *c;
        if (u == '&') n += (size_t)snprintf(t + n, sizeof t - n, " AND ");
        else if ((u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') || u == '\'' || u == ' ') t[n++] = u;
    }
    t[n] = 0;
    // collapse spaces and title-case, as Python's " ".join(split()).title()
    char title[128];
    size_t k = 0;
    int start = 1;
    for (size_t i = 0; t[i] && k + 1 < sizeof title; i++) {
        if (t[i] == ' ') {
            if (k && title[k - 1] != ' ') title[k++] = ' ';
            start = 1;
            continue;
        }
        int alpha = t[i] >= 'A' && t[i] <= 'Z';
        title[k++] = alpha && !start ? (char)(t[i] + 32) : t[i];
        start = !alpha;
    }
    while (k && title[k - 1] == ' ') k--;
    title[k] = 0;
    snprintf(out, cap, "%s", title);
    if (!k || !ncategory_names) return;
    char lo[128], single[128], name_lo[128];
    lower(title, lo, sizeof lo);
    snprintf(single, sizeof single, "%s", lo);
    if (k > 1 && single[k - 1] == 's') single[k - 1] = 0;
    int best = 0, best_d = 1 << 20;
    for (int i = 0; i < ncategory_names; i++) {
        lower(category_names[i], name_lo, sizeof name_lo);
        if (!strcmp(name_lo, lo) || !strcmp(name_lo, single)) {
            snprintf(out, cap, "%s", category_names[i]);
            return;
        }
        int d = edit_distance(lo, name_lo), ds = edit_distance(single, name_lo);
        if (ds < d) d = ds;
        if (d < best_d) best_d = d, best = i;
    }
    size_t a = strlen(lo), b = strlen(category_names[best]);
    if (best_d * 100 <= 25 * (int)(a > b ? a : b)) snprintf(out, cap, "%s", category_names[best]);
}

// Square classes by index.
enum { CLS_OFF = 0, CLS_HIDDEN = 1, CLS_LIT = 2, CLS_COVERED = DEC_CLASSES - 1 };

static int cls_of(const DecodeResult *r, int s) {
    int best = 0;
    for (int k = 1; k < DEC_CLASSES; k++)
        if (r->probs[s][k] > r->probs[s][best]) best = k;
    return best;
}

static void add_square(cJSON *list, int row, int col) {
    cJSON *rc = cJSON_CreateArray();
    cJSON_AddItemToArray(rc, cJSON_CreateNumber(row));
    cJSON_AddItemToArray(rc, cJSON_CreateNumber(col));
    cJSON_AddItemToArray(list, rc);
}

// The solvers' board text from the read squares, as board.to_board: lit and
// covered squares in play count as hidden, a covered square is in play when
// squares in play lie on both sides of it in its row, or when it runs left
// from its row's first square in play no further than the column where the
// other rows start (the show starts every line in the same column). Unused
// squares split words, and rows join with a space unless a row ends in a
// hyphen.
static void board_text(const DecodeResult *r, char *board, size_t cap, cJSON *guessed) {
    size_t n = 0;
    board[0] = 0;
    int first[4], start = 14;
    for (int row = 0; row < 4; row++) {
        first[row] = -1;
        for (int c = 0; c < 14 && first[row] < 0; c++) {
            int s = decoder_square(row, c);
            if (s >= 0 && cls_of(r, s) != CLS_OFF && cls_of(r, s) != CLS_COVERED) first[row] = c;
        }
        if (first[row] >= 0 && first[row] < start) start = first[row];
    }
    for (int row = 0; row < 4; row++) {
        int cls[14], cols[14], m = 0;
        for (int c = 0; c < 14; c++) {
            int s = decoder_square(row, c);
            if (s >= 0) cls[m] = cls_of(r, s), cols[m++] = c;
        }
        int lead_from = 14;   // covered squares from this column to the first in play start the line
        if (first[row] >= 0) {
            int i = 0;
            while (cols[i] != first[row]) i++;
            for (i--; i >= 0 && cls[i] == CLS_COVERED && cols[i] >= start; i--) lead_from = cols[i];
        }
        char line[32];
        size_t k = 0;
        int gap = 0;
        for (int i = 0; i < m; i++) {
            int in_play = cls[i] != CLS_OFF && cls[i] != CLS_COVERED;
            if (cls[i] == CLS_COVERED) {
                int left = 0, right = 0;
                for (int j = 0; j < i; j++) left |= cls[j] != CLS_OFF && cls[j] != CLS_COVERED;
                for (int j = i + 1; j < m; j++) right |= cls[j] != CLS_OFF && cls[j] != CLS_COVERED;
                in_play = (left && right) || (cols[i] >= lead_from && cols[i] < first[row]);
            }
            if (!in_play) {
                gap = k > 0;
                continue;
            }
            if (cls[i] == CLS_LIT || cls[i] == CLS_COVERED) add_square(guessed, row, cols[i]);
            if (gap) line[k++] = ' ', gap = 0;
            line[k++] = cls[i] == CLS_LIT || cls[i] == CLS_COVERED ? '_' : DEC_CLASS_CHARS[cls[i]];
        }
        line[k] = 0;
        if (!k) continue;
        if (n && board[n - 1] != '-' && n + 1 < cap) board[n++] = ' ';
        n += (size_t)snprintf(board + n, cap - n, "%s", line);
        if (n >= cap) n = cap - 1;
    }
}

// The value of name= in the query: 1 when found, 0 when absent, -1 if bad.
static int query_param(const char *query, size_t qlen, const char *name, char *out, size_t cap) {
    size_t nl = strlen(name);
    for (size_t i = 0; i < qlen;) {
        size_t j = i;
        while (j < qlen && query[j] != '&') j++;
        if (j - i > nl && !strncmp(query + i, name, nl) && query[i + nl] == '=')
            return url_decode(query + i + nl + 1, j - i - nl - 1, out, cap) ? -1 : 1;
        i = j + 1;
    }
    return 0;
}

static int do_decode(int fd, const Request *rq, const char *query, size_t qlen) {
    if (!decoder) return error_reply(fd, rq, 503, "the board decoder is not loaded");
    if (!rq->body_len) return error_reply(fd, rq, 400, "send the image as the request body");
    char solve_s[8] = "", mode_s[16] = "main";
    if (query_param(query, qlen, "solve", solve_s, sizeof solve_s) < 0
        || query_param(query, qlen, "mode", mode_s, sizeof mode_s) < 0)
        return error_reply(fd, rq, 400, "bad query");
    int want_solve = !strcmp(solve_s, "1") || !strcmp(solve_s, "true");
    int w, h;
    if (image_info((const uint8_t *)rq->body, rq->body_len, &w, &h))
        return error_reply(fd, rq, 400, "the image must be a JPEG or PNG");
    if ((double)w * h > MAX_PIXELS) return error_reply(fd, rq, 413, "the image is over 16 megapixels");
    DecodeResult *r = malloc(sizeof *r);
    uint8_t *in = malloc((size_t)DEC_IN_W * DEC_IN_H * 3);
    if (!r || !in) {
        free(r), free(in);
        return error_reply(fd, rq, 503, "out of memory");
    }
    pthread_mutex_lock(&decode_lock);   // one image at a time bounds the memory a read takes
    double t0 = now();
    uint8_t *rgb = image_rgb((const uint8_t *)rq->body, rq->body_len, &w, &h);
    int rc = -1;
    if (rgb) {
        decoder_letterbox(rgb, w, h, in);
        image_free(rgb);
        rc = decoder_forward(decoder, in, r);
    }
    double ms = 1000 * (now() - t0);
    free(in);
    malloc_trim(0);   // hand back what the read freed
    pthread_mutex_unlock(&decode_lock);
    if (!rgb || rc) {
        free(r);
        return rgb ? error_reply(fd, rq, 503, "out of memory")
                   : error_reply(fd, rq, 400, "the image could not be decoded");
    }
    cJSON *out = cJSON_CreateObject();
    int n_unsure = 0;
    for (int s = 0; s < DEC_SQUARES; s++) n_unsure += r->conf[s] < 0.8f;
    int found = r->present >= 0.5f && n_unsure <= MAX_UNSURE;
    cJSON_AddBoolToObject(out, "found", found);
    cJSON_AddNumberToObject(out, "present", py_round(r->present, 4));
    if (!found && r->present >= 0.5f) cJSON_AddNumberToObject(out, "unsure", n_unsure);
    if (found) {
        char category[128], board[128], called[27];
        map_category(r->category, category, sizeof category);
        cJSON *guessed = cJSON_CreateArray(), *unsure = cJSON_CreateArray(), *rows = cJSON_CreateArray();
        board_text(r, board, sizeof board, guessed);
        int seen[26] = {0}, n = 0;
        for (const char *c = board; *c; c++)
            if (*c >= 'A' && *c <= 'Z') seen[*c - 'A'] = 1;
        for (int i = 0; i < 26; i++)
            if (seen[i]) called[n++] = (char)('A' + i);
        called[n] = 0;
        for (int row = 0; row < 4; row++) {
            char line[15];
            for (int c = 0; c < 14; c++) {
                int s = decoder_square(row, c);
                line[c] = s < 0 ? ' ' : DEC_CLASS_CHARS[cls_of(r, s)];
                if (s >= 0 && r->conf[s] < 0.8f) add_square(unsure, row, c);
            }
            line[14] = 0;
            cJSON_AddItemToArray(rows, cJSON_CreateString(line));
        }
        cJSON_AddStringToObject(out, "category", category);
        cJSON_AddStringToObject(out, "category_read", r->category);
        cJSON_AddStringToObject(out, "board", board);
        cJSON_AddStringToObject(out, "called", called);
        cJSON_AddItemToObject(out, "rows", rows);
        cJSON_AddItemToObject(out, "guessed", guessed);
        cJSON_AddItemToObject(out, "unsure", unsure);
        if (want_solve) {
            // through the same checks and solvers as POST /move
            cJSON *st = cJSON_CreateObject();
            cJSON_AddStringToObject(st, "mode", mode_s);
            cJSON_AddStringToObject(st, "category", category);
            cJSON_AddStringToObject(st, "board", board);
            char called_all[40];
            snprintf(called_all, sizeof called_all, "%s%s", called, strcmp(mode_s, "main") ? FREE : "");
            cJSON_AddStringToObject(st, "called", called_all);
            int mode, want[2];
            char c2[80], b2[TF_MAX_TEXT + 1], k2[27];
            const char *err = strchr(board, '_') ? check(st, &mode, c2, b2, k2, want)
                                                 : "the board has no hidden squares";
            cJSON_Delete(st);
            if (!err || strchr(board, '_') == NULL) {
                cJSON *gs = cJSON_AddObjectToObject(out, "guess");
                if (!strchr(board, '_')) {
                    cJSON_AddStringToObject(gs, "answer", board);
                    cJSON_AddNumberToObject(gs, "confidence", 1);
                } else {
                    TfMove tg;
                    pthread_mutex_lock(&model_lock);
                    int grc = tf_guess(&model, seq, c2, b2, called, &tg);
                    pthread_mutex_unlock(&model_lock);
                    if (!grc) {
                        cJSON_AddStringToObject(gs, "answer", tg.move + 6);
                        cJSON_AddNumberToObject(gs, "confidence", py_round(tg.confidence, 4));
                    }
                }
            }
            cJSON *mv = cJSON_CreateObject();
            if (!err) err = solve(mv, mode, c2, b2, k2, want);
            if (err) {
                cJSON_Delete(mv);
                cJSON_AddStringToObject(out, "moves_error", err);
            } else {
                cJSON_AddItemToObject(out, "moves", mv);
            }
        }
    }
    free(r);
    cJSON_AddNumberToObject(out, "ms", (double)(long)(ms + 0.5));
    char *s = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
    int rc2 = json_reply(fd, rq, 200, s);
    cJSON_free(s);
    return rc2;
}

// -------------------------------------------------------------- routing --

static int url_decode(const char *in, size_t n, char *out, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        char c = in[i];
        if (c == '+') c = ' ';
        else if (c == '%' && i + 2 < n) {
            char hex[3] = {in[i + 1], in[i + 2], 0};
            char *end;
            long v = strtol(hex, &end, 16);
            if (*end) return -1;
            c = (char)v;
            i += 2;
        } else if (c == '%') {
            return -1;
        }
        if (k + 1 >= cap) return -1;
        out[k++] = c;
    }
    out[k] = 0;
    return 0;
}

static int do_puzzle(int fd, const Request *rq, const char *query, size_t qlen) {
    const Puzzle *p = NULL;
    char want[128] = "";
    int has_want = 0;
    // the only parameter is category=
    for (size_t i = 0; i < qlen;) {
        size_t j = i;
        while (j < qlen && query[j] != '&') j++;
        if (j - i > 9 && !strncmp(query + i, "category=", 9)) {
            if (url_decode(query + i + 9, j - i - 9, want, sizeof want)) return error_reply(fd, rq, 400, "bad query");
            has_want = 1;
        }
        i = j + 1;
    }
    if (has_want) {
        int n = 0;
        for (int i = 0; i < npuzzles; i++) n += !strcmp(puzzles[i].category, want);
        if (!n) {
            char msg[200];
            snprintf(msg, sizeof msg, "no puzzles in category '%s'", want);
            return error_reply(fd, rq, 404, msg);
        }
        int k = (int)(rnd() % (uint64_t)n);
        for (int i = 0; i < npuzzles; i++)
            if (!strcmp(puzzles[i].category, want) && k-- == 0) p = &puzzles[i];
    } else {
        if (!npuzzles) return error_reply(fd, rq, 404, "no puzzles");
        p = &puzzles[rnd() % (uint64_t)npuzzles];
    }
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "category", p->category);
    cJSON_AddStringToObject(o, "answer", p->answer);
    char *s = cJSON_PrintUnformatted(o);
    cJSON_Delete(o);
    int rc = json_reply(fd, rq, 200, s);
    cJSON_free(s);
    return rc;
}

static int route(int fd, Request *rq) {
    const char *q = memchr(rq->path, '?', rq->path_len);
    size_t plen = q ? (size_t)(q - rq->path) : rq->path_len;
    const char *query = q ? q + 1 : "";
    size_t qlen = q ? rq->path_len - plen - 1 : 0;
    char path[256];
    if (plen >= sizeof path) return error_reply(fd, rq, 404, "not found");
    memcpy(path, rq->path, plen);
    path[plen] = 0;
#define METHOD(m) (rq->method_len == strlen(m) && !memcmp(rq->method, m, rq->method_len))

    if (METHOD("OPTIONS")) return respond(fd, rq, 204, NULL, NULL, 0, NULL);
    if (METHOD("GET") || METHOD("HEAD")) {
        if (!strcmp(path, "/health")) return json_reply(fd, rq, 200, "{\"status\":\"ok\"}");
        if (!strcmp(path, "/categories") || !strcmp(path, "/presets") || !strcmp(path, "/puzzle")) {
            if (!allow(rq->client)) return error_reply(fd, rq, 429, "too many requests, slow down");
            if (!strcmp(path, "/categories")) return json_reply(fd, rq, 200, categories_json);
            if (!strcmp(path, "/presets")) return json_reply(fd, rq, 200, presets_json);
            return do_puzzle(fd, rq, query, qlen);
        }
        const WebFile *f = web_file(!strcmp(path, "/") ? "wof-solver.html" : path + 1);
        if (f) return respond(fd, rq, 200, f->type, f->body, METHOD("HEAD") ? 0 : f->len, "no-cache");
        return error_reply(fd, rq, 404, "not found");
    }
    if (METHOD("POST")) {
        if (!strcmp(path, "/decode")) {
            if (!limit(&decodes, rq->client)) return error_reply(fd, rq, 429, "too many images, slow down");
            return do_decode(fd, rq, query, qlen);
        }
        if (strcmp(path, "/move")) return error_reply(fd, rq, 404, "not found");
        if (!allow(rq->client)) return error_reply(fd, rq, 429, "too many requests, slow down");
        return do_move(fd, rq);
    }
    rq->close = 1;
    return error_reply(fd, rq, 405, "method not allowed");
#undef METHOD
}

static void header(const Request *rq, const char *name, char *out, size_t cap) {
    out[0] = 0;
    for (size_t i = 0; i < rq->nheaders; i++)
        if (rq->headers[i].name_len == strlen(name) && !strncasecmp(rq->headers[i].name, name, strlen(name))) {
            size_t n = rq->headers[i].value_len < cap - 1 ? rq->headers[i].value_len : cap - 1;
            memcpy(out, rq->headers[i].value, n);
            out[n] = 0;
            return;
        }
}

static pthread_mutex_t conn_lock = PTHREAD_MUTEX_INITIALIZER;
static int nconns;

static void *serve_conn(void *arg) {
    Conn c = *(Conn *)arg;
    free(arg);
    struct timeval tv = {IDLE_SECONDS, 0};
    setsockopt(c.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(c.fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    char buf[MAX_REQUEST + MAX_BODY];
    size_t have = 0;
    for (;;) {
        Request rq;
        memset(&rq, 0, sizeof rq);
        int hlen;
        for (;;) {   // read until the headers are complete
            rq.nheaders = MAX_HEADERS;
            hlen = phr_parse_request(buf, have, &rq.method, &rq.method_len, &rq.path, &rq.path_len, &rq.minor,
                                     rq.headers, &rq.nheaders, 0);
            if (hlen > 0) break;
            if (hlen == -1 || have >= MAX_REQUEST) goto done;
            ssize_t k = recv(c.fd, buf + have, MAX_REQUEST - have, 0);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) goto done;
            have += (size_t)k;
        }
        char v[256];
        header(&rq, "Content-Length", v, sizeof v);
        long clen = v[0] ? strtol(v, NULL, 10) : 0;
        // only POST /decode takes a body over MAX_BODY: an image, read into
        // its own buffer rather than buf
        int big = rq.method_len == 4 && !memcmp(rq.method, "POST", 4) && rq.path_len >= 7
                  && !memcmp(rq.path, "/decode", 7) && (rq.path_len == 7 || rq.path[7] == '?');
        if (clen < 0 || clen > (big ? (long)decode_max_bytes : MAX_BODY)) {
            rq.close = 1;
            error_reply(c.fd, &rq, 413, "body too large");
            goto done;
        }
        header(&rq, "Transfer-Encoding", v, sizeof v);
        if (v[0]) {   // the client never sends chunked bodies
            rq.close = 1;
            error_reply(c.fd, &rq, 400, "chunked bodies are not supported");
            goto done;
        }
        char *image = NULL;
        size_t used;   // bytes of buf this request took
        if (big && clen > MAX_BODY) {
            image = malloc((size_t)clen);
            if (!image) {
                rq.close = 1;
                error_reply(c.fd, &rq, 503, "out of memory");
                goto done;
            }
            size_t got = have - (size_t)hlen < (size_t)clen ? have - (size_t)hlen : (size_t)clen;
            memcpy(image, buf + hlen, got);
            used = (size_t)hlen + got;
            while (got < (size_t)clen) {
                ssize_t k = recv(c.fd, image + got, (size_t)clen - got, 0);
                if (k < 0 && errno == EINTR) continue;
                if (k <= 0) {
                    free(image);
                    goto done;
                }
                got += (size_t)k;
            }
            rq.body = image;
        } else {
            while (have < (size_t)hlen + (size_t)clen) {
                ssize_t k = recv(c.fd, buf + have, sizeof buf - have, 0);
                if (k < 0 && errno == EINTR) continue;
                if (k <= 0) goto done;
                have += (size_t)k;
            }
            rq.body = buf + hlen;
            used = (size_t)hlen + (size_t)clen;
        }
        rq.body_len = (size_t)clen;
        header(&rq, "Origin", rq.origin, sizeof rq.origin);
        header(&rq, "Fly-Client-IP", rq.client, sizeof rq.client);
        if (!rq.client[0]) snprintf(rq.client, sizeof rq.client, "%s", c.peer);
        header(&rq, "Connection", v, sizeof v);
        rq.close = rq.minor == 0 ? strcasecmp(v, "keep-alive") != 0 : strcasecmp(v, "close") == 0;
        int stop = route(c.fd, &rq) || rq.close;
        free(image);
        if (stop) break;
        memmove(buf, buf + used, have - used);   // keep any pipelined bytes
        have -= used;
    }
done:
    close(c.fd);
    pthread_mutex_lock(&conn_lock);
    nconns--;
    pthread_mutex_unlock(&conn_lock);
    return NULL;
}

// ----------------------------------------------------------------- main --

static void split_origins(const char *list) {
    char *s = copy_str(list);
    for (char *tok = strtok(s, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok == ' ') tok++;
        origins = realloc(origins, (size_t)(norigins + 1) * sizeof *origins);
        origins[norigins++] = copy_str(tok);
    }
    free(s);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    // Large allocations (a /decode read's scratch, an uploaded image) come
    // from mmap and go back to the kernel when freed. glibc would otherwise
    // raise this threshold after the first one and keep later ones in the
    // heap, and the memory it kept would push the transformer's weights out
    // of the page cache on the 256 MB machine.
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int port = (int)env_num("PORT", 8080);
    min_p = env_num("MIN_P", 0.99);
    moves.rate = env_num("RATE", 600);
    decodes.rate = env_num("DECODE_RATE", 10);
    decode_max_bytes = (size_t)env_num("DECODE_MAX_BYTES", 4 << 20);
    split_origins(env_str("CORS_ORIGINS", "https://danheskett.com"));
    rng_state = (uint64_t)(now() * 1e9) | 1;

    double t0 = now();
    char err[512];
    const char *weights = env_str("WEIGHTS", "models/wof.bin");
    if (model_load(&model, weights, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    seq = seq_new(&model);
    smart = smart_load(env_str("WORDLIST", "data/wordlist/american-english.txt"), env_str("PUZZLES", "data/train.csv,data/dev.csv"),
                       err, sizeof err);
    if (!seq || !smart) {
        fprintf(stderr, "%s\n", smart ? "out of memory" : err);
        return 1;
    }
    const char *demo = env_str("DEMO", "data/demo.json");
    if (load_demo(demo)) {
        fprintf(stderr, "%s: cannot read the demo data\n", demo);
        return 1;
    }
    load_web(env_str("WEB_ROOT", "web"));
    const char *dec_path = env_str("DECODER", "models/decoder.bin");
    decoder = decoder_load(dec_path, err, sizeof err);
    if (!decoder) fprintf(stderr, "%s; POST /decode is off\n", err);
    printf("loaded %s, smart-solve, %d puzzles, %d web files in %.1f s\n", weights, npuzzles, nweb, now() - t0);

    int ls = socket(AF_INET6, SOCK_STREAM, 0);
    int one = 1, zero = 0;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    setsockopt(ls, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);   // IPv4 too
    struct sockaddr_in6 addr = {.sin6_family = AF_INET6, .sin6_port = htons((uint16_t)port), .sin6_addr = in6addr_any};
    if (ls < 0 || bind(ls, (struct sockaddr *)&addr, sizeof addr) || listen(ls, 128)) {
        perror("listen");
        return 1;
    }
    printf("wof-server on :%d, MIN_P %.2f\n", port, min_p);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 1 << 20);
    for (;;) {
        struct sockaddr_in6 peer;
        socklen_t plen = sizeof peer;
        int fd = accept(ls, (struct sockaddr *)&peer, &plen);
        if (fd < 0) continue;
        pthread_mutex_lock(&conn_lock);
        int full = nconns >= MAX_CONNS;
        if (!full) nconns++;
        pthread_mutex_unlock(&conn_lock);
        if (full) {   // shed load rather than queue unbounded threads
            static const char busy[] = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(fd, busy, sizeof busy - 1);
            close(fd);
            continue;
        }
        Conn *c = malloc(sizeof *c);
        c->fd = fd;
        inet_ntop(AF_INET6, &peer.sin6_addr, c->peer, sizeof c->peer);
        pthread_t th;
        if (pthread_create(&th, &attr, serve_conn, c)) {
            close(fd);
            free(c);
            pthread_mutex_lock(&conn_lock);
            nconns--;
            pthread_mutex_unlock(&conn_lock);
        }
    }
}
