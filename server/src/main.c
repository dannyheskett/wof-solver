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
//   CORS_ORIGINS (comma-separated, default https://danheskett.com),
//   WEIGHTS, WORDLIST, PUZZLES (comma-separated), DEMO, WEB_ROOT (paths,
//   defaults below).
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
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
#include "model.h"
#include "picohttpparser.h"
#include "smart.h"
#include "transformer.h"

#define MAX_HEADERS 32
#define MAX_REQUEST 8192
#define MAX_BODY 4096
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

// ---------------------------------------------------------------- demo --

typedef struct {
    char *category, *answer;
} Puzzle;

static Puzzle *puzzles;
static int npuzzles;
static char *categories_json, *presets_json;
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
#define BUCKETS 4096
static struct {
    double tokens, last;
    int used;
} bucket[BUCKETS];
static pthread_mutex_t bucket_lock = PTHREAD_MUTEX_INITIALIZER;
static double rate;

static int allow(const char *client) {
    uint64_t h = 1469598103934665603ull;
    for (const char *c = client; *c; c++) h = (h ^ (unsigned char)*c) * 1099511628211ull;
    pthread_mutex_lock(&bucket_lock);
    double t = now();
    int i = (int)(h % BUCKETS);
    if (!bucket[i].used) bucket[i].used = 1, bucket[i].tokens = rate, bucket[i].last = t;
    bucket[i].tokens += (t - bucket[i].last) * rate / 60;
    if (bucket[i].tokens > rate) bucket[i].tokens = rate;
    bucket[i].last = t;
    int ok = bucket[i].tokens >= 1;
    if (ok) bucket[i].tokens -= 1;
    pthread_mutex_unlock(&bucket_lock);
    return ok;
}

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
    if (want[0]) {
        char mv[256];
        pthread_mutex_lock(&smart_lock);
        double t0 = now();
        int rc = mode == 0   ? smart_main(smart, category, board, called, mv, sizeof mv)
                 : mode == 1 ? smart_bonus_pick(smart, category, board, called, mv, sizeof mv)
                             : smart_bonus_solve(smart, category, board, called, mv, sizeof mv);
        double ms = 1000 * (now() - t0);
        pthread_mutex_unlock(&smart_lock);
        if (rc) {
            cJSON_Delete(out);
            return error_reply(fd, rq, 400, "smart-solve has no move for this board");
        }
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
        if (rc) {
            cJSON_Delete(out);
            return error_reply(fd, rq, 400, "board too long");
        }
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
    char *s = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
    int rc = json_reply(fd, rq, 200, s);
    cJSON_free(s);
    return rc;
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
        if (clen < 0 || clen > MAX_BODY) {
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
        while (have < (size_t)hlen + (size_t)clen) {
            ssize_t k = recv(c.fd, buf + have, sizeof buf - have, 0);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) goto done;
            have += (size_t)k;
        }
        rq.body = buf + hlen;
        rq.body_len = (size_t)clen;
        header(&rq, "Origin", rq.origin, sizeof rq.origin);
        header(&rq, "Fly-Client-IP", rq.client, sizeof rq.client);
        if (!rq.client[0]) snprintf(rq.client, sizeof rq.client, "%s", c.peer);
        header(&rq, "Connection", v, sizeof v);
        rq.close = rq.minor == 0 ? strcasecmp(v, "keep-alive") != 0 : strcasecmp(v, "close") == 0;
        if (route(c.fd, &rq) || rq.close) break;
        size_t used = (size_t)hlen + (size_t)clen;   // keep any pipelined bytes
        memmove(buf, buf + used, have - used);
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
    setvbuf(stdout, NULL, _IOLBF, 0);
    int port = (int)env_num("PORT", 8080);
    min_p = env_num("MIN_P", 0.99);
    rate = env_num("RATE", 600);
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
