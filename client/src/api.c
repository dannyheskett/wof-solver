#include "api.h"

#include <emscripten.h>
#include <emscripten/fetch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char base[256] = "https://wof-solver.fly.dev";

// ?api=<url> overrides the API, for testing against a local server.
EM_JS(char *, js_api_param, (void), {
    var v = new URLSearchParams(window.location.search).get("api");
    return v ? stringToNewUTF8(v) : 0;
});

void api_init(void) {
    char *v = js_api_param();
    if (v) {
        snprintf(base, sizeof base, "%s", v);
        free(v);
    }
    size_t n = strlen(base);
    if (n && base[n - 1] == '/') base[n - 1] = 0;
}

const char *api_base(void) { return base; }

typedef struct {
    ApiCallback cb;
    void *user;
    char *body;   // POST data, kept alive until the request ends
} Pending;

static void done(emscripten_fetch_t *f) {
    Pending *p = f->userData;
    char *text = malloc((size_t)f->numBytes + 1);
    if (text) {
        if (f->numBytes) memcpy(text, f->data, (size_t)f->numBytes);
        text[f->numBytes] = 0;
    }
    p->cb(p->user, f->status, text ? text : "");
    free(text);
    free(p->body);
    free(p);
    emscripten_fetch_close(f);
}

static void request(const char *method, const char *path, const char *json, ApiCallback cb, void *user) {
    Pending *p = calloc(1, sizeof *p);
    p->cb = cb;
    p->user = user;
    emscripten_fetch_attr_t a;
    emscripten_fetch_attr_init(&a);
    strcpy(a.requestMethod, method);
    a.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    a.timeoutMSecs = 120000;   // a stalled request fails instead of hanging
    a.onsuccess = done;
    a.onerror = done;
    a.userData = p;
    static const char *headers[] = {"Content-Type", "application/json", NULL};
    if (json) {
        size_t n = strlen(json) + 1;
        p->body = malloc(n);
        memcpy(p->body, json, n);
        a.requestData = p->body;
        a.requestDataSize = strlen(p->body);
        a.requestHeaders = headers;
    }
    char url[512];
    snprintf(url, sizeof url, "%s%s", base, path);
    emscripten_fetch(&a, url);
}

void api_get(const char *path, ApiCallback cb, void *user) { request("GET", path, NULL, cb, user); }

void api_post(const char *path, const char *json, ApiCallback cb, void *user) {
    request("POST", path, json, cb, user);
}
