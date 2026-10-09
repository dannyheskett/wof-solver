// Calls to the move API (serve/app.py) from the browser, through Emscripten's
// fetch. All calls are asynchronous: the callback runs on a later frame.
#ifndef API_H
#define API_H

// body is NUL-terminated (empty on a network error); status is the HTTP
// status, or 0 when the server could not be reached.
typedef void (*ApiCallback)(void *user, int status, const char *body);

// The API base URL: ?api=<url> on the page, else https://wof-solver.fly.dev.
void api_init(void);
const char *api_base(void);
void api_get(const char *path, ApiCallback cb, void *user);
void api_post(const char *path, const char *json, ApiCallback cb, void *user);

#endif
