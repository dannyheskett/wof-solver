// WOF Solver demo: smart-solve and transformer-solve play the same Wheel of
// Fortune puzzle head to head, each on its own board, one move per turn.
//
// Modes
//   Random       a random control puzzle, played at a steady pace; the next
//                one starts when a match ends
//   Demo         hand-picked examples where transformer-solve wins, each with
//                a caption
//   Your puzzle  pick a category, type an answer, watch them play it
// Once started, each mode plays by itself until its work is done: Random plays
// game after game until the server stops responding, Demo plays every example
// in order, Your puzzle plays its one game. Play and Pause (or Space) freeze
// and resume it in every mode. The MAIN/BONUS switch picks the round for
// Random and Your puzzle.
//
// Each turn asks both solvers for their move at the same moment. Each side
// has a stopwatch, a disc that starts filling then, fills while that side
// thinks, and locks at the milliseconds its answer took to arrive, when its
// move shows. Both fill at the same rate, so the gap between them is the gap
// in thinking time as the viewer waited for it.
//
// transformer-solve's proposed answers are coloured by its confidence, letter
// by letter: red (a guess), orange, yellow, green (99% and up: it solves).
#include <emscripten.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"
#include "cJSON.h"
#include "font_nunito.h"
#include "match.h"
#include "raylib.h"

// The logical canvas, scaled to fit the window: 1280x800 when the window is
// wider than tall (landscape), 720 wide and at least 1280 tall when it is
// taller (portrait).
static int W = 1280, H = 800;
static bool portrait;
#define MARGIN 24
#define HEADER_H 64              // landscape: one header row; portrait: two
// Where the two panels, the result box and the scoreboard go, set by layout()
static Rectangle side_box[2], result_box, score_box;
#define BODY 18            // the one body text size
#define PACE 1.6f          // seconds per turn in Random
#define NEXT_MATCH 4.0f    // seconds between Random matches
#define MAX_CATS 64
#define MAX_PRESETS 16

typedef enum { MODE_RANDOM, MODE_DEMO, MODE_OWN } Mode;

static const char *SIDE_NAME[NSIDES] = {"smart-solve", "transformer-solve"};
static const char *SIDE_API[NSIDES] = {"smart", "transformer"};

static const Color BG = {16, 24, 40, 255};
static const Color PANEL = {26, 38, 62, 255};
static const Color TEXT = {235, 238, 245, 255};
static const Color DIM = {150, 160, 185, 255};
static const Color ACCENT = {80, 140, 255, 255};
static const Color FELT = {22, 96, 70, 255};       // unused board squares
static const Color TILE = {248, 248, 244, 255};
static const Color HIT = {60, 170, 100, 255};
static const Color MISS = {200, 70, 70, 255};
static const Color SMART_GUESS = {120, 170, 255, 255};
static const Color WIN = {250, 200, 40, 255};
static const Color TIE = {70, 150, 255, 255};
static const Color LOSE = {225, 60, 60, 255};
#define WIN_FLASHES 4      // the result frame flashes this many times, then stays

static Font font;
static Match match;
static Mode mode = MODE_RANDOM;
static Round round_switch = ROUND_MAIN;
static bool playing = true;
static double last_step, over_at;
static bool counted;              // this match is in the tally
static int tally[4];              // smart wins, transformer wins, ties, both lose
static char error_msg[160];       // why play stopped, shown under the category

static struct { char name[48]; bool autoplay; } cats[MAX_CATS];
static int ncats;
static struct { Round round; char category[48], answer[MAX_ANSWER], note[200]; } presets[MAX_PRESETS];
static int npresets, preset;

static bool have_match, fetching_puzzle;
// The server picks Random's puzzles. The page skips one it has already shown
// this visit, by hash, and asks again (up to MAX_SKIPS times in a row).
#define MAX_SEEN 4096
#define MAX_SKIPS 8
static unsigned seen[MAX_SEEN];
static int nseen, skips;
static int puzzle_fails;           // failed /puzzle requests in a row
static double puzzle_retry_at;
static bool need_puzzle;           // Random wants its next puzzle
// categories and Demo examples load at start, and again until they arrive
static bool cats_loading, presets_loading;
static double cats_retry_at, presets_retry_at;
static int retries[NSIDES];
// The turn in progress: each side's request, from when it went out
static bool turn_active;
static bool want[NSIDES];          // this side's move for the turn has not arrived
static bool moved[NSIDES];         // this side plays this turn
static double sent_at[NSIDES];
static double retry_at[NSIDES];

// Your puzzle form
static int own_cat;
static char own_text[MAX_ANSWER * 2];
static bool own_focus, dropdown, dropdown_just_opened;
static bool own_modal;        // Your puzzle's New game dialog is up
static Rectangle own_field;   // its answer field, for losing focus on a click elsewhere
static const char *own_error;

// ------------------------------------------------------------- drawing --

static void text(const char *s, float x, float y, float size, Color c) {
    DrawTextEx(font, s, (Vector2){x, y}, size, 0, c);
}

static float text_w(const char *s, float size) { return MeasureTextEx(font, s, size, 0).x; }

static void text_c(const char *s, float cx, float y, float size, Color c) {
    text(s, cx - text_w(s, size) / 2, y, size, c);
}

static Color lerp(Color a, Color b, float t) {
    return (Color){(unsigned char)(a.r + (b.r - a.r) * t), (unsigned char)(a.g + (b.g - a.g) * t),
                   (unsigned char)(a.b + (b.b - a.b) * t), 255};
}

// Confidence bands, blended at the edges: under 50% red, to 90% orange, to 99%
// yellow, 99% and up green (the cutoff: transformer-solve solves).
static Color conf_color(float p) {
    static const float at[] = {0.0f, 0.5f, 0.9f, 0.99f, 1.0f};
    static const Color c[] = {{215, 60, 55, 255}, {240, 140, 45, 255}, {240, 205, 60, 255},
                              {70, 185, 95, 255}, {70, 185, 95, 255}};
    if (p <= 0) return c[0];
    for (int i = 1; i < 5; i++)
        if (p <= at[i]) return lerp(c[i - 1], c[i], (p - at[i - 1]) / (at[i] - at[i - 1]));
    return c[4];
}

static Vector2 mouse(void) {
    float s = fminf(GetScreenWidth() / (float)W, GetScreenHeight() / (float)H);
    Vector2 m = GetMousePosition();
    return (Vector2){(m.x - (GetScreenWidth() - W * s) / 2) / s, (m.y - (GetScreenHeight() - H * s) / 2) / s};
}

// While a menu is open, everything under it ignores the mouse.
static bool input_blocked;
static bool hover(Rectangle r) { return !input_blocked && CheckCollisionPointRec(mouse(), r); }

// Pick the canvas by the window's shape and place the panels on it.
// Landscape: the panels side by side, the result and scoreboard under them.
// Portrait: header in two rows, the panels stacked, then the result, then the
// scoreboard.
static void layout(void) {
    portrait = GetScreenHeight() > GetScreenWidth();
    W = portrait ? 720 : 1280;
    // portrait: at least 1280 tall, taller on a narrow phone so it fills the screen
    H = portrait ? (int)fmaxf(1280, 720.0f * GetScreenHeight() / GetScreenWidth()) : 800;
    if (!portrait) {
        float pw = (W - 3 * MARGIN) / 2.0f, py = 86, ph = 614, by = py + ph + 12, bh = H - by - 16;
        side_box[0] = (Rectangle){MARGIN, py, pw, ph};
        side_box[1] = (Rectangle){2 * MARGIN + pw, py, pw, ph};
        result_box = (Rectangle){MARGIN, by, pw, bh};
        score_box = (Rectangle){2 * MARGIN + pw, by, pw, bh};
    } else {
        float pw = W - 2 * MARGIN, py = 124, sh = 72, rh = 64;
        float ph = (H - 16 - sh - 12 - rh - 12 - py - 12) / 2;
        side_box[0] = (Rectangle){MARGIN, py, pw, ph};
        side_box[1] = (Rectangle){MARGIN, py + ph + 12, pw, ph};
        result_box = (Rectangle){MARGIN, py + 2 * ph + 24, pw, rh};
        score_box = (Rectangle){MARGIN, H - 16 - sh, pw, sh};
    }
}

static bool button(Rectangle r, const char *label, bool enabled, bool active) {
    bool over = enabled && hover(r);
    Color bg = active ? ACCENT : over ? (Color){52, 70, 105, 255} : (Color){40, 55, 85, 255};
    if (!enabled) bg = (Color){32, 42, 62, 255};
    DrawRectangleRounded(r, 0.3f, 6, bg);
    text_c(label, r.x + r.width / 2, r.y + (r.height - 20) / 2, 20, enabled ? TEXT : DIM);
    return over && IsMouseButtonReleased(MOUSE_BUTTON_LEFT);
}

static bool toggle(Rectangle r, bool on, bool enabled) {
    float knob = r.height - 6;
    DrawRectangleRounded(r, 1, 12, enabled ? (on ? ACCENT : (Color){60, 75, 105, 255}) : (Color){32, 42, 62, 255});
    float x = on ? r.x + r.width - knob - 3 : r.x + 3;
    DrawRectangleRounded((Rectangle){x, r.y + 3, knob, knob}, 1, 12, enabled ? TEXT : DIM);
    return enabled && hover(r) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT);
}

// ---------------------------------------------------------------- moves --

typedef struct { int side; unsigned gen; } MoveReq;

static void got_move(void *user, int status, const char *body) {
    MoveReq *r = user;
    int s = r->side;
    bool current = r->gen == match.gen;
    free(r);
    if (!current) return;
    Side *d = &match.side[s];
    d->waiting = false;
    cJSON *root = status == 200 ? cJSON_Parse(body) : NULL;
    cJSON *o = root ? cJSON_GetObjectItem(root, SIDE_API[s]) : NULL;
    cJSON *mv = o ? cJSON_GetObjectItem(o, "move") : NULL;
    if (!cJSON_IsString(mv)) {
        // Without the server the game cannot go on: a refused request (400)
        // stops play at once, anything else is retried a few times first.
        // Play stops where it is and Play resumes it; nothing is scored.
        cJSON *e = status == 400 && root ? cJSON_GetObjectItem(root, "error") : NULL;
        if (status == 400 || ++retries[s] > 5) {
            snprintf(error_msg, sizeof error_msg, "%s", cJSON_IsString(e) ? e->valuestring
                                                        : "The server stopped responding.");
            playing = false;
            retries[s] = 0, retry_at[s] = 0;
        } else {
            retry_at[s] = GetTime() + (status == 429 ? 5.0 : 1.5 * retries[s]);
        }
        cJSON_Delete(root);
        return;
    }
    retries[s] = 0;
    Move m = {0};
    snprintf(m.move, sizeof m.move, "%s", mv->valuestring);
    cJSON *td = cJSON_GetObjectItem(o, "turned_down");
    cJSON *conf = cJSON_GetObjectItem(o, "confidence");
    cJSON *letters = cJSON_GetObjectItem(o, "letters");
    cJSON *ms = cJSON_GetObjectItem(o, "ms");
    if (cJSON_IsString(td)) {
        m.turned_down = true;
        snprintf(m.tried, sizeof m.tried, "%s", td->valuestring);
    } else if (strncmp(m.move, "SOLVE ", 6) == 0) {
        snprintf(m.tried, sizeof m.tried, "%s", m.move + 6);
    }
    if (cJSON_IsNumber(conf)) {
        m.has_conf = true;
        m.conf = (float)conf->valuedouble;
    }
    int i = 0;
    cJSON *x;
    cJSON_ArrayForEach(x, letters) if (i < MAX_ANSWER) m.letters[i++] = (float)x->valuedouble;
    if (cJSON_IsNumber(ms)) m.ms = ms->valueint;
    cJSON_Delete(root);
    m.wall = (int)((GetTime() - sent_at[s]) * 1000 + 0.5);
    match_apply(&match, s, &m);
    d->shown = d->nmoves;   // the move shows the moment it arrives
    want[s] = false;
}

// Send each request the turn wants, again after a failure.
static void fetch_moves(void) {
    if (!have_match) return;
    for (int s = 0; s < NSIDES; s++) {
        const char *md;
        char board[MAX_ANSWER], called[32];
        if (!playing || !want[s] || GetTime() < retry_at[s] || !match_next_request(&match, s, &md, board, called))
            continue;
        cJSON *req = cJSON_CreateObject();
        cJSON_AddStringToObject(req, "mode", md);
        cJSON_AddStringToObject(req, "category", match.category);
        cJSON_AddStringToObject(req, "board", board);
        cJSON_AddStringToObject(req, "called", called);
        cJSON *who = cJSON_AddArrayToObject(req, "solvers");
        cJSON_AddItemToArray(who, cJSON_CreateString(SIDE_API[s]));
        char *json = cJSON_PrintUnformatted(req);
        cJSON_Delete(req);
        MoveReq *r = malloc(sizeof *r);
        *r = (MoveReq){s, match.gen};
        match.side[s].waiting = true;
        api_post("/move", json, got_move, r);
        cJSON_free(json);
    }
}

static void start(Round round, const char *category, const char *answer) {
    // callers may pass match.category / match.answer, which match_start clears
    char cat[sizeof match.category], ans[sizeof match.answer];
    snprintf(cat, sizeof cat, "%s", category);
    snprintf(ans, sizeof ans, "%s", answer);
    match_start(&match, round, cat, ans);
    have_match = true;
    need_puzzle = false;
    last_step = GetTime();
    over_at = 0;
    counted = false;
    turn_active = false;
    for (int s = 0; s < NSIDES; s++) retries[s] = 0, retry_at[s] = 0, want[s] = moved[s] = false;
    error_msg[0] = 0;
}

static bool side_live(int s) { return side_status(&match, s, match.side[s].shown) == PLAYING; }

// Start a turn: both sides that are still playing ask for their move now.
static bool begin_turn(void) {
    if (!have_match || turn_active || match_over(&match)) return false;
    bool any = false;
    for (int s = 0; s < NSIDES; s++) {
        moved[s] = want[s] = side_live(s);
        if (moved[s]) sent_at[s] = GetTime(), any = true;
    }
    if (any) turn_active = true, match.turn++;
    return any;
}

// ------------------------------------------------------------- modes --

static void random_puzzle(void);

// FNV-1a of category and answer
static unsigned puzzle_hash(const char *category, const char *answer) {
    unsigned h = 2166136261u;
    for (const char *p = category; *p; p++) h = (h ^ (unsigned char)*p) * 16777619u;
    h = (h ^ '|') * 16777619u;
    for (const char *p = answer; *p; p++) h = (h ^ (unsigned char)*p) * 16777619u;
    return h;
}

static void got_puzzle(void *user, int status, const char *body) {
    (void)user;
    fetching_puzzle = false;
    if (mode != MODE_RANDOM) return;
    cJSON *root = status == 200 ? cJSON_Parse(body) : NULL;
    cJSON *c = cJSON_GetObjectItem(root, "category"), *a = cJSON_GetObjectItem(root, "answer");
    if (cJSON_IsString(c) && cJSON_IsString(a)) {
        unsigned h = puzzle_hash(c->valuestring, a->valuestring);
        bool dup = false;
        for (int i = 0; i < nseen && !dup; i++) dup = seen[i] == h;
        if (dup && skips++ < MAX_SKIPS) {
            cJSON_Delete(root);
            random_puzzle();   // shown already this visit: ask for another
            return;
        }
        skips = 0;
        puzzle_fails = 0;
        need_puzzle = false;
        if (!dup && nseen < MAX_SEEN) seen[nseen++] = h;
        start(round_switch, c->valuestring, a->valuestring);
    } else if (++puzzle_fails > 5) {   // the server stopped responding: Random stops
        snprintf(error_msg, sizeof error_msg, "The server stopped responding.");
        playing = false;
        puzzle_fails = 0;   // Play tries again
    } else {
        puzzle_retry_at = GetTime() + 1.5 * puzzle_fails;
    }
    cJSON_Delete(root);
}

static void random_puzzle(void) {
    if (fetching_puzzle) return;
    fetching_puzzle = true;
    api_get("/puzzle", got_puzzle, NULL);
}

static void show_preset(int i) {
    if (!npresets) return;
    preset = (i + npresets) % npresets;
    start(presets[preset].round, presets[preset].category, presets[preset].answer);
}

static void got_categories(void *user, int status, const char *body) {
    (void)user;
    cats_loading = false;
    cJSON *root = status == 200 ? cJSON_Parse(body) : NULL, *c;
    ncats = 0;
    cJSON_ArrayForEach(c, root) {
        cJSON *n = cJSON_GetObjectItem(c, "name"), *a = cJSON_GetObjectItem(c, "auto");
        if (ncats < MAX_CATS && cJSON_IsString(n)) {
            snprintf(cats[ncats].name, sizeof cats[ncats].name, "%s", n->valuestring);
            cats[ncats++].autoplay = cJSON_IsTrue(a);
        }
    }
    cJSON_Delete(root);
    if (!ncats) cats_retry_at = GetTime() + 3;
}

static void got_presets(void *user, int status, const char *body) {
    (void)user;
    presets_loading = false;
    cJSON *root = status == 200 ? cJSON_Parse(body) : NULL, *p;
    npresets = 0;
    cJSON_ArrayForEach(p, root) {
        if (npresets >= MAX_PRESETS) break;
        cJSON *m = cJSON_GetObjectItem(p, "mode"), *c = cJSON_GetObjectItem(p, "category"),
              *a = cJSON_GetObjectItem(p, "answer"), *n = cJSON_GetObjectItem(p, "note");
        if (!cJSON_IsString(c) || !cJSON_IsString(a)) continue;
        presets[npresets].round = cJSON_IsString(m) && !strcmp(m->valuestring, "bonus") ? ROUND_BONUS : ROUND_MAIN;
        snprintf(presets[npresets].category, sizeof presets[npresets].category, "%s", c->valuestring);
        snprintf(presets[npresets].answer, sizeof presets[npresets].answer, "%s", a->valuestring);
        snprintf(presets[npresets].note, sizeof presets[npresets].note, "%s", cJSON_IsString(n) ? n->valuestring : "");
        npresets++;
    }
    cJSON_Delete(root);
    if (!npresets) presets_retry_at = GetTime() + 3;
    else if (mode == MODE_DEMO) show_preset(0);
}

static void set_mode(Mode m) {
    mode = m;
    have_match = false;
    match.gen++;   // drop replies for the old match
    dropdown = own_focus = false;
    error_msg[0] = 0;
    playing = m != MODE_OWN;   // Your puzzle starts when its puzzle is set
    puzzle_fails = 0, puzzle_retry_at = 0;
    need_puzzle = m == MODE_RANDOM;
    if (m == MODE_DEMO) show_preset(0);
    else own_modal = own_focus = true;
}

// Typing on a phone: the canvas cannot raise the keyboard, so use a prompt.
EM_JS(int, js_touch, (void), { return ("ontouchstart" in window) ? 1 : 0; });
EM_JS(char *, js_prompt, (const char *q, const char *cur), {
    var v = window.prompt(UTF8ToString(q), UTF8ToString(cur));
    return v === null ? 0 : stringToNewUTF8(v);
});

static void play_own(void) {
    char buf[sizeof own_text];
    strcpy(buf, own_text);
    own_error = ncats ? check_answer(buf) : "Categories are still loading.";
    if (own_error) return;
    strcpy(own_text, buf);
    own_focus = false;
    own_modal = false;
    start(round_switch, cats[own_cat].name, own_text);
    playing = true;
}

// ---------------------------------------------------------------- board --

#define TILE_W 36
#define TILE_H 46
#define GAP 3

static void draw_board(int s, float x0, float y0) {
    Layout l;
    char board[MAX_ANSWER];
    const Side *d = &match.side[s];
    side_board(&match, s, d->shown, board);
    bool laid = layout_answer(match.answer, &l);
    // a SOLVE, shown in the blank squares (a turned-down guess is not)
    const Move *last = d->shown ? &d->moves[d->shown - 1] : NULL;
    const char *tried = last && last->tried[0] && !last->turned_down ? last->tried : NULL;

    for (int r = 0; r < BOARD_ROWS; r++)
        for (int c = 0; c < 14; c++) {
            if ((r == 0 || r == 3) && (c == 0 || c == 13)) continue;
            DrawRectangle((int)(x0 + c * (TILE_W + GAP)), (int)(y0 + r * (TILE_H + GAP)), TILE_W, TILE_H, FELT);
        }
    if (!laid) return;
    for (int i = 0; i < l.len; i++) {
        if (l.row[i] < 0 || match.answer[i] == ' ') continue;
        float x = x0 + l.col[i] * (TILE_W + GAP), y = y0 + l.row[i] * (TILE_H + GAP);
        char ch[2] = {board[i], 0};
        Color bg = TILE, fg = (Color){20, 20, 30, 255};
        if (board[i] == '_') {
            ch[0] = 0;
            if (tried && (int)strlen(tried) > i && tried[i] != ' ') {
                ch[0] = tried[i];
                bg = s == TRANSFORMER && last->has_conf ? conf_color(last->letters[i]) : SMART_GUESS;
            }
        }
        DrawRectangle((int)x, (int)y, TILE_W, TILE_H, bg);
        if (ch[0]) text_c(ch, x + TILE_W / 2.0f, y + 6, 32, fg);
    }
}

static const char *status_word(Status st) {
    switch (st) {
    case SOLVED: return "Solved";
    case WRONG: return "Wrong answer";
    case INVALID: return "Invalid move";
    case OUT_OF_MOVES: return "Out of moves";
    default: return "";
    }
}

// Milliseconds on this side's stopwatch: running while its move is on the
// way, else locked at the time its latest move took to arrive.
static int watch_ms(int s, bool *running) {
    const Side *d = &match.side[s];
    *running = turn_active && want[s];
    if (*running) return (int)((GetTime() - sent_at[s]) * 1000);
    return d->shown ? d->moves[d->shown - 1].wall : 0;
}

// The stopwatch: a small solid disc that fills clockwise from 12 o'clock,
// one full turn per second on both sides. Muted gold while the side thinks,
// its own muted colour once locked; each finished second leaves the disc
// full in a darker shade under the next turn. The time shows beside it once
// it stops.
static Color shade(Color c, float k) {
    return (Color){(unsigned char)(c.r * k), (unsigned char)(c.g * k), (unsigned char)(c.b * k), 255};
}

static const Color WATCH_RUN = {196, 162, 70, 255};
static const Color WATCH_SMART = {96, 128, 184, 255};
static const Color WATCH_TF = {72, 148, 98, 255};

static void draw_stopwatch(int s, float cx, float cy, float r) {
    bool running;
    int ms = watch_ms(s, &running);
    Color fill = running ? WATCH_RUN : s == TRANSFORMER ? WATCH_TF : WATCH_SMART;
    Vector2 c = {cx, cy};
    DrawCircleV(c, r, (Color){14, 20, 34, 255});
    int laps = ms / 1000;
    float frac = (ms % 1000) / 1000.0f;
    if (laps) DrawCircleV(c, r, shade(fill, laps > 1 ? 0.45f : 0.6f));
    if (frac > 0) DrawCircleSector(c, r, -90, -90 + 360 * frac, 48, fill);
    DrawRing(c, r, r + 1.5f, 0, 360, 48, (Color){60, 78, 115, 255});
}

// Beside the stopwatch, once it has stopped: the time, then the turn's gap to
// the other side (once both have stopped) and this side's running total.
// The stopwatch's time, and the gap to the other side once both have
// stopped, right-aligned to end at x.
static void draw_watch_notes(int s, float x, float y) {
    bool running, other;
    int ms = watch_ms(s, &running), oms = watch_ms(1 - s, &other);
    char t[32], g[32] = "";
    snprintf(t, sizeof t, "%.1f s", (ms / 100) / 10.0);
    if (!running && !other && moved[s] && moved[1 - s] && match.side[s].shown) {
        int gap = ms / 100 - oms / 100;
        snprintf(g, sizeof g, "%c%.1f s", gap < 0 ? '-' : '+', abs(gap) / 10.0);
    }
    float tw = text_w(t, BODY);
    text(t, x - tw, y, BODY, TEXT);
    if (g[0]) text(g, x - tw - 16 - text_w(g, BODY), y, BODY, DIM);
}

// How a finished match ends: one side wins (gold frame on it), both solve
// level (blue frames on both), or neither solves (red frames on both).
typedef enum { RESULT_WIN, RESULT_TIE, RESULT_LOSE } Result;

static Result match_result(void) {
    if (match_winner(&match) >= 0) return RESULT_WIN;
    return side_status(&match, SMART, match.side[SMART].shown) == SOLVED ? RESULT_TIE : RESULT_LOSE;
}




// A green check for a right guess, a red X for a wrong one: sz px square,
// left edge at x, centred vertically on cy.
static void guess_mark(bool right, float x, float cy, float sz) {
    float k = sz / 14, t = fmaxf(3, 3 * k);
    if (right) {
        DrawLineEx((Vector2){x, cy}, (Vector2){x + 5 * k, cy + 5 * k}, t, HIT);
        DrawLineEx((Vector2){x + 5 * k, cy + 5 * k}, (Vector2){x + 14 * k, cy - 6 * k}, t, HIT);
    } else {
        DrawLineEx((Vector2){x + k, cy - 6 * k}, (Vector2){x + 13 * k, cy + 6 * k}, t, MISS);
        DrawLineEx((Vector2){x + 13 * k, cy - 6 * k}, (Vector2){x + k, cy + 6 * k}, t, MISS);
    }
}

static void draw_side(int s, float x0, float y0, float w, float h) {
    const Side *d = &match.side[s];
    DrawRectangleRounded((Rectangle){x0, y0, w, h}, 0.03f, 8, PANEL);
    // everything lines up with the board's left and right edges
    const float BW = 14 * (TILE_W + GAP) - GAP, bx = x0 + (w - BW) / 2;
    if (have_match && match_over(&match) && over_at) {
        Result r = match_result();
        if (r != RESULT_WIN || match_winner(&match) == s) {
            Color c = r == RESULT_WIN ? WIN : r == RESULT_TIE ? TIE : LOSE;
            const char *tag = r == RESULT_WIN ? "WINNER" : r == RESULT_TIE ? "TIE" : "LOST";
            double t = GetTime() - over_at;
            bool on = t > WIN_FLASHES * 0.5 || fmod(t, 0.5) < 0.3;
            if (on) DrawRectangleRoundedLinesEx((Rectangle){x0 - 3, y0 - 3, w + 6, h + 6}, 0.03f, 8, 6, c);
            float tw = text_w(tag, BODY) + 24;
            DrawRectangleRounded((Rectangle){bx + BW - tw, y0 + 14, tw, 28}, 0.5f, 8, c);
            text_c(tag, bx + BW - tw / 2, y0 + 18, BODY, r == RESULT_WIN ? (Color){30, 25, 10, 255} : TEXT);
        }
    }
    text(SIDE_NAME[s], bx, y0 + 14, 24, TEXT);
    if (!have_match) return;

    float y = y0 + 52;
    draw_board(s, bx, y);
    y += 4 * (TILE_H + GAP) - GAP + 16;

    // called letters, exactly the board's width: green hit, red miss
    char called[40];
    side_called(&match, s, d->shown, called);
    const float CW = (BW - 25 * GAP) / 26;
    for (int i = 0; i < 26; i++) {
        char c = (char)('A' + i), str[2] = {c, 0};
        bool was = strchr(called, c) != NULL;
        bool given = match.round == ROUND_BONUS && strchr("RSTLNE", c);   // shown free in BONUS
        Color bg = !was ? (Color){40, 55, 85, 255} : given ? (Color){85, 100, 135, 255}
                 : strchr(match.answer, c) ? HIT : MISS;
        float cx = bx + i * (CW + GAP);
        DrawRectangle((int)cx, (int)y, (int)CW, 26, bg);
        text_c(str, cx + CW / 2, y + 4, 16, was ? TEXT : DIM);
    }
    y += 26 + 16;

    // two equal columns under the board: score and moves on the left; the
    // stopwatch and (transformer-solve) the last guess on the right
    // the latest guess sits in a strip along the bottom of the panel
    const float STRIP_H = 48;
    const Rectangle strip = {bx, y0 + h - 16 - STRIP_H, BW, STRIP_H};
    const float lx = bx, bottom = strip.y - 8;
    char stats[160];
    Status st = side_status(&match, s, d->shown);
    snprintf(stats, sizeof stats, "Letters %d   Misses %d   %s", side_letters(&match, s, d->shown),
             side_misses(&match, s, d->shown), status_word(st));
    // the stats row: score on the left, the stopwatch at the right edge
    float r = 11;
    text(stats, lx, y + r - BODY / 2.0f, BODY, st == SOLVED ? HIT : (st == PLAYING ? TEXT : MISS));
    draw_stopwatch(s, bx + BW - r, y + r, r);
    draw_watch_notes(s, bx + BW - 2 * r - 14, y + r - BODY / 2.0f);
    y += 2 * r + 14;

    // the moves, newest last, as many as fit: one full-width line each, four
    // columns (turn, move, guessed answer, confidence), one font and size;
    // the guess and its confidence in grey
    const float LINE = 24, C_MOVE = lx + 36, C_GUESS = lx + 160, C_PCT = bx + BW;
    const float GUESS_W = C_PCT - 70 - C_GUESS;
    int first = d->shown;
    while (first > 0 && y + (d->shown - first + 1) * LINE <= bottom) first--;
    for (int i = first; i < d->shown; i++) {
        const Move *m = &d->moves[i];
        char num[8];
        snprintf(num, sizeof num, "%d", i + 1);
        text(num, C_MOVE - 12 - text_w(num, BODY), y, BODY, TEXT);
        const char *mv = strncmp(m->move, "SOLVE ", 6) == 0 ? "SOLVE" : m->move;
        text(mv, C_MOVE, y, BODY, TEXT);
        if (m->tried[0]) {
            // cut to the column with "..."
            char guess[MAX_ANSWER + 4];
            snprintf(guess, sizeof guess, "%s", m->tried);
            if (text_w(guess, BODY) > GUESS_W) {
                int n = (int)strlen(guess);
                while (n > 0) {
                    guess[--n] = 0;
                    char t[MAX_ANSWER + 4];
                    snprintf(t, sizeof t, "%s...", guess);
                    if (text_w(t, BODY) <= GUESS_W) { strcpy(guess, t); break; }
                }
            }
            text(guess, C_GUESS, y, BODY, DIM);
            if (m->has_conf) {
                char pct[16];
                snprintf(pct, sizeof pct, "%.1f%%", m->conf * 100);
                text(pct, C_PCT - text_w(pct, BODY), y, BODY, DIM);
            }
        }
        y += LINE;
    }

    // the strip: the latest guessed answer, what became of it, and for a
    // submitted one a large check or X
    DrawRectangleRounded(strip, 0.2f, 8, (Color){14, 20, 34, 255});
    DrawRectangleRoundedLinesEx(strip, 0.2f, 8, 1.5f, (Color){60, 78, 115, 255});
    const Move *g = NULL;
    for (int i = d->shown - 1; i >= 0 && !g; i--)
        if (d->moves[i].tried[0]) g = &d->moves[i];
    if (g) {
        // Solved / Wrong for a submitted answer; Held back (grey, no mark)
        // for one transformer-solve did not submit
        bool right = strcmp(g->tried, match.answer) == 0;
        float cy = strip.y + STRIP_H / 2;
        char status[32];
        Color c = DIM;
        if (g->turned_down) {
            snprintf(status, sizeof status, "Held back %.1f%%", g->conf * 100);
        } else {
            guess_mark(right, strip.x + 16, cy, 32);
            snprintf(status, sizeof status, "%s", right ? "Solved" : "Wrong");
            c = right ? HIT : MISS;
        }
        float tx = strip.x + 16 + 32 + 16;
        text(status, tx, cy - BODY / 2.0f, BODY, c);
        text(g->tried, tx + text_w(status, BODY) + 16, cy - BODY / 2.0f, BODY, c);
    }
}

// ---------------------------------------------------------------- frame --

static void action(void);
static bool work_left(void);

static void play_icon(float x, float cy, float sz, Color c) {
    DrawTriangle((Vector2){x, cy - sz / 2}, (Vector2){x, cy + sz / 2}, (Vector2){x + sz * 0.85f, cy}, c);
}

// The top bar's controls share one look: the same height, corners, fill and
// edge, lighter under the mouse.
#define BAR_Y 16
#define BAR_H 40
#define BAR_GAP 12
static void control_frame(Rectangle r, bool over, bool enabled) {
    Color fill = !enabled ? (Color){32, 42, 62, 255} : over ? (Color){52, 70, 105, 255} : (Color){40, 55, 85, 255};
    DrawRectangleRounded(r, 0.25f, 6, fill);
    DrawRectangleRoundedLinesEx(r, 0.25f, 6, 1.5f, (Color){70, 90, 130, 255});
}

// VCR-style Play and Pause keys.
static bool vcr_key(Rectangle r, bool enabled, bool pause) {
    bool over = enabled && hover(r);
    control_frame(r, over, enabled);
    Color c = enabled ? (Color){225, 228, 235, 255} : (Color){110, 115, 128, 255};
    float cx = r.x + r.width / 2, cy = r.y + r.height / 2, sz = 16;
    if (pause) {
        DrawRectangle((int)(cx - 7), (int)(cy - sz / 2), 5, (int)sz, c);
        DrawRectangle((int)(cx + 2), (int)(cy - sz / 2), 5, (int)sz, c);
    } else {
        play_icon(cx - 6, cy, sz, c);
    }
    return over && IsMouseButtonReleased(MOUSE_BUTTON_LEFT);
}

static const char *MODE_NAME[] = {"Random", "Demo", "Your puzzle"};
#define VCR_W 56
#define KEY_W (2 * VCR_W + 6)
// The controls row: the header row in landscape, the second header row in
// portrait, narrower there so New game still fits.
#define CTRL_Y (portrait ? BAR_Y + BAR_H + 8 : BAR_Y)
#define ROUND_W (portrait ? 210 : 230)
#define MODE_W (portrait ? 160 : 190)
#define MODE_BUTTON ((Rectangle){W - 24 - KEY_W - BAR_GAP - ROUND_W - BAR_GAP - MODE_W, CTRL_Y, MODE_W, BAR_H})
static bool mode_open, mode_just_opened;

static void draw_mode_menu(void) {
    if (!mode_open) return;
    Rectangle mb = MODE_BUTTON, box = {mb.x, mb.y + mb.height + 6, mb.width, 3 * 40 + 8};
    DrawRectangleRounded(box, 0.08f, 6, (Color){34, 48, 76, 255});
    DrawRectangleRoundedLinesEx(box, 0.08f, 6, 1.5f, (Color){70, 90, 130, 255});
    for (int i = 0; i < 3; i++) {
        Rectangle r = {box.x + 4, box.y + 4 + i * 40.0f, box.width - 8, 38};
        bool over = hover(r);
        if (over || mode == (Mode)i) DrawRectangleRounded(r, 0.2f, 6, mode == (Mode)i ? ACCENT : (Color){52, 70, 105, 255});
        text(MODE_NAME[i], r.x + 14, r.y + 8, 20, TEXT);
        if (over && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            mode_open = false;
            if (mode != (Mode)i) set_mode((Mode)i);
            return;
        }
    }
    // a click anywhere else closes it, the selector included (but not the
    // click that opened it)
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && !hover(box) && !mode_just_opened) mode_open = false;
    mode_just_opened = false;
}

// The logo: a little prize wheel, eight coloured wedges around a gold hub.
#define LOGO_R 18
#define TITLE_X (24 + 2 * LOGO_R + 12)
static void draw_logo(float cx, float cy, float r) {
    static const Color wedge[8] = {{235, 80, 80, 255},  {250, 170, 50, 255}, {250, 215, 70, 255}, {90, 200, 110, 255},
                                   {70, 170, 235, 255}, {120, 110, 230, 255}, {210, 100, 200, 255}, {245, 130, 160, 255}};
    for (int i = 0; i < 8; i++) DrawCircleSector((Vector2){cx, cy}, r, -90 + i * 45.0f, -45 + i * 45.0f, 12, wedge[i]);
    DrawRing((Vector2){cx, cy}, r, r + 2, 0, 360, 48, (Color){248, 248, 244, 255});
    DrawCircleV((Vector2){cx, cy}, r * 0.28f, (Color){250, 200, 40, 255});
    DrawTriangle((Vector2){cx - 5, cy - r - 6}, (Vector2){cx, cy - r + 4}, (Vector2){cx + 5, cy - r - 6}, (Color){248, 248, 244, 255});
}

static void draw_top(void) {
    draw_logo(24 + LOGO_R, BAR_Y + BAR_H / 2.0f + 2, LOGO_R);
    text("WOF Solver", TITLE_X, BAR_Y + (BAR_H - 34) / 2.0f, 34, TEXT);
    // right-aligned: mode selector, round switch, game control
    Rectangle mb = MODE_BUTTON;
    bool mover = hover(mb);
    control_frame(mb, mover || mode_open, true);
    text(MODE_NAME[mode], mb.x + 16, mb.y + 10, 20, TEXT);
    float ax = mb.x + mb.width - 28, ay = mb.y + 17;
    DrawTriangle((Vector2){ax, ay}, (Vector2){ax + 6, ay + 7}, (Vector2){ax + 12, ay}, TEXT);
    if (mover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) mode_open = mode_just_opened = true;
    if (mode == MODE_OWN && !own_modal) {
        Rectangle ng = {mb.x - BAR_GAP - 140, CTRL_Y, 140, BAR_H};
        bool over = hover(ng);
        control_frame(ng, over, true);
        text_c("New game", ng.x + ng.width / 2, ng.y + 10, 20, TEXT);
        if (over && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) own_modal = own_focus = true;
    }
    bool fixed = mode == MODE_DEMO;
    Round shown = fixed && have_match ? match.round : round_switch;
    Rectangle box = {mb.x + MODE_W + BAR_GAP, CTRL_Y, ROUND_W, BAR_H};
    control_frame(box, false, !fixed);
    text("MAIN", box.x + 18, box.y + 10, 20, shown == ROUND_MAIN ? TEXT : DIM);
    if (toggle((Rectangle){box.x + (ROUND_W - 60) / 2.0f, box.y + 7, 60, 26}, shown == ROUND_BONUS, !fixed)) {
        round_switch = round_switch == ROUND_MAIN ? ROUND_BONUS : ROUND_MAIN;
        if (have_match && mode == MODE_OWN) start(round_switch, match.category, match.answer);
        else if (mode == MODE_RANDOM && have_match) start(round_switch, match.category, match.answer);
    }
    text("BONUS", box.x + ROUND_W - 18 - text_w("BONUS", 20), box.y + 10, 20, shown == ROUND_BONUS ? TEXT : DIM);
    Rectangle play = {W - 24 - KEY_W, CTRL_Y, VCR_W, BAR_H}, pause = {play.x + VCR_W + 6, CTRL_Y, VCR_W, BAR_H};
    if (vcr_key(play, !playing && work_left(), false)) action();
    if (vcr_key(pause, playing, true)) playing = false;
}

static void draw_info(void) {
    // why play stopped, under the header
    // the category in the header row, centred between the title and the controls
    if (have_match) {
        float l = TITLE_X + text_w("WOF Solver", 34);
        float r = portrait ? W - 24 : MODE_BUTTON.x - (mode == MODE_OWN && !own_modal ? 140 + BAR_GAP : 0);
        text_c(match.category, (l + r) / 2, BAR_Y + (BAR_H - 24) / 2.0f, 24, TEXT);
    }
    if (error_msg[0]) text_c(error_msg, W / 2.0f, portrait ? 106 : 63, 16, MISS);
}

// Your puzzle's New game dialog, over the dimmed game: category, answer,
// then Cancel and Play.
// Full-screen notices over the dimmed page: the overview at start, and the
// check-in every CHECK_IN games. Play waits while one is up.
#define CHECK_IN 50
typedef enum { NOTICE_NONE, NOTICE_INTRO, NOTICE_CHECK_IN } Notice;
static Notice notice = NOTICE_INTRO;
static int games_done;            // finished games this visit

static const char *INTRO =
    "This is a research project that tests whether a small transformer model can learn to play Wheel of "
    "Fortune. transformer-solve is a 59M-parameter transformer trained from scratch on about 64,000 past "
    "puzzles, and it plays head to head against smart-solve, a word-list solver, on puzzles neither one "
    "has seen. Each turn, both call a letter or guess the answer. transformer-solve solves only when it's "
    "at least 99% sure, and smart-solve solves once every word on the board matches exactly one word in "
    "its list.";

// Text wrapped at spaces to width w, lines `step` apart from y; draw=false
// only counts. Returns the number of lines.
#define NOTICE_TEXT 22
static int wrapped(const char *t, float x, float y, float w, float step, bool draw) {
    char line[512] = "";
    int n = 0;
    for (const char *p = t; *p;) {
        const char *e = strchr(p, ' ');
        int len = e ? (int)(e - p) : (int)strlen(p);
        char next[600];
        snprintf(next, sizeof next, "%s%s%.*s", line, line[0] ? " " : "", len, p);
        if (line[0] && text_w(next, NOTICE_TEXT) > w) {
            if (draw) text(line, x, y + n * step, NOTICE_TEXT, TEXT);
            n++;
            snprintf(line, sizeof line, "%.*s", len, p);
        } else {
            snprintf(line, sizeof line, "%s", next);
        }
        p += len;
        while (*p == ' ') p++;
    }
    if (line[0]) {
        if (draw) text(line, x, y + n * step, NOTICE_TEXT, TEXT);
        n++;
    }
    return n;
}

static void draw_notice(void) {
    if (notice == NOTICE_NONE) return;
    DrawRectangle(0, 0, W, H, (Color){8, 12, 20, 225});
    // both notices are the size the overview needs
    const float STEP = 32, PAD = 28;
    float pw = fminf(680, W - 48), tw = pw - 2 * PAD;
    float ph = PAD + 40 + 24 + wrapped(INTRO, 0, 0, tw, STEP, false) * STEP + 24 + BAR_H + PAD;
    Rectangle p = {(W - pw) / 2, (H - ph) / 2, pw, ph};
    DrawRectangleRounded(p, 0.05f, 8, PANEL);
    DrawRectangleRoundedLinesEx(p, 0.05f, 8, 2, (Color){70, 90, 130, 255});
    float x = p.x + PAD, y = p.y + PAD;
    const char *label;
    if (notice == NOTICE_INTRO) {
        draw_logo(x + LOGO_R, y + 18, LOGO_R);
        text("WOF Solver", x + 2 * LOGO_R + 12, y, 34, TEXT);
        wrapped(INTRO, x, y + 64, tw, STEP, true);
        label = "Start";
    } else {
        text("Still watching?", x, y, 34, TEXT);
        wrapped(TextFormat("That's %d games. Press Continue to keep playing.", games_done), x, y + 64, tw, STEP, true);
        label = "Continue";
    }
    Rectangle b = {p.x + p.width - PAD - 150, p.y + p.height - PAD - BAR_H, 150, BAR_H};
    if (button(b, label, true, true) || IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {
        playing = true;
        notice = NOTICE_NONE;
    }
}

static void draw_new_game(void) {
    if (mode != MODE_OWN || !own_modal) return;
    float top = CTRL_Y + BAR_H + 8, pw = fminf(760, W - 48);
    DrawRectangle(0, (int)top, W, H - (int)top, (Color){8, 12, 20, 210});
    Rectangle p = {(W - pw) / 2.0f, 200, pw, 230};
    DrawRectangleRounded(p, 0.06f, 8, PANEL);
    DrawRectangleRoundedLinesEx(p, 0.06f, 8, 2, (Color){70, 90, 130, 255});
    text("New game", p.x + 28, p.y + 22, 28, TEXT);
    float y = p.y + 80;
    const char *cat = ncats ? cats[own_cat].name : "...";
    Rectangle cb = {p.x + 28, y, 230, BAR_H};
    bool cover = hover(cb) || dropdown;
    control_frame(cb, cover, ncats > 0);
    text(cat, cb.x + 16, cb.y + 10, 20, TEXT);
    float ax = cb.x + cb.width - 28, ay = cb.y + 17;
    DrawTriangle((Vector2){ax, ay}, (Vector2){ax + 6, ay + 7}, (Vector2){ax + 12, ay}, TEXT);
    if (ncats && hover(cb) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dropdown = dropdown_just_opened = true;
    own_field = (Rectangle){cb.x + cb.width + BAR_GAP, y, p.width - 56 - cb.width - BAR_GAP, BAR_H};
    DrawRectangleRounded(own_field, 0.25f, 6, own_focus ? (Color){50, 66, 100, 255} : (Color){40, 55, 85, 255});
    DrawRectangleRoundedLinesEx(own_field, 0.25f, 6, 1.5f, (Color){70, 90, 130, 255});
    bool blink = own_focus && fmod(GetTime(), 1.0) < 0.5;
    char shown[sizeof own_text + 2];
    snprintf(shown, sizeof shown, "%s%s", own_text, blink ? "_" : "");
    text(own_text[0] || own_focus ? shown : "e.g. A PIECE OF CAKE", own_field.x + 14, own_field.y + 9, 22,
         own_text[0] || own_focus ? TEXT : DIM);
    if (hover(own_field) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        if (js_touch()) {
            char *v = js_prompt("Puzzle answer", own_text);
            if (v) { snprintf(own_text, sizeof own_text, "%s", v); free(v); play_own(); }
        } else {
            own_focus = true;
        }
    }
    if (own_error) text(own_error, p.x + 28, y + 50, 17, MISS);
    Rectangle play = {p.x + p.width - 28 - 130, p.y + p.height - 28 - BAR_H, 130, BAR_H};
    if (button(play, "Play", true, true)) play_own();
    Rectangle cancel = {play.x - BAR_GAP - 130, play.y, 130, BAR_H};
    if (button(cancel, "Cancel", true, false)) own_modal = own_focus = dropdown = false;
}

static void draw_dropdown(void) {
    if (!dropdown) return;
    // the click that opened the list lands on it too, so skip that frame
    bool click = IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && !dropdown_just_opened;
    dropdown_just_opened = false;
    int cols = portrait ? 2 : 3, rows = (ncats + cols - 1) / cols;
    float top = CTRL_Y + BAR_H + 10, cw = (W - 48 - 24 - (cols - 1) * 10.0f) / cols;
    Rectangle box = {24, top, W - 48, rows * 30.0f + 20};
    DrawRectangleRounded(box, 0.02f, 6, (Color){34, 48, 76, 255});
    for (int i = 0; i < ncats; i++) {
        Rectangle r = {36 + (i / rows) * (cw + 10), top + 10 + (i % rows) * 30.0f, cw, 28};
        bool over = hover(r);
        if (over || i == own_cat) DrawRectangleRec(r, i == own_cat ? ACCENT : (Color){52, 70, 105, 255});
        text(cats[i].name, r.x + 10, r.y + 3, 20, TEXT);
        if (over && click) { own_cat = i; dropdown = false; return; }
    }
    if (click && !hover(box)) dropdown = false;
}

// Whether the mode has anything left to play: Random always (until the server
// stops responding), Demo until its last example ends, Your puzzle until its
// game ends.
static bool work_left(void) {
    if (mode == MODE_RANDOM) return true;
    if (!have_match) return false;
    if (!match_over(&match)) return true;
    return mode == MODE_DEMO && preset < npresets - 1;
}

// Play/Pause: freeze the game where it is, or resume it.
static void action(void) {
    if (playing) playing = false;
    else if (work_left()) playing = true, error_msg[0] = 0;
}

// The scoreboard: running totals for this visit, a label over a number in
// each box.
static const char *SCORE_LABEL[4] = {"smart-solve", "transformer-solve", "ties", "both lose"};

static void draw_scoreboard(Rectangle r) {
    Color c[4] = {SMART_GUESS, conf_color(1), TIE, LOSE};
    float cw = (r.width - 3 * 8) / 4;
    for (int i = 0; i < 4; i++) {
        Rectangle cell = {r.x + i * (cw + 8), r.y, cw, r.height};
        DrawRectangleRounded(cell, 0.15f, 8, PANEL);
        DrawRectangleRoundedLinesEx(cell, 0.15f, 8, 1.5f, (Color){60, 78, 115, 255});
        text_c(SCORE_LABEL[i], cell.x + cell.width / 2, cell.y + 8, 16, c[i]);
        char n[16];
        snprintf(n, sizeof n, "%d", tally[i]);
        text_c(n, cell.x + cell.width / 2, cell.y + 30, 30, TEXT);
    }
}

// The result panel: the last finished match, who won and its answer.
typedef struct { Result r; int winner; char answer[MAX_ANSWER]; } LogEntry;
static LogEntry last_result;
static bool have_result;

static void log_result(void) {
    last_result.r = match_result();
    last_result.winner = match_winner(&match);
    snprintf(last_result.answer, sizeof last_result.answer, "%s", match.answer);
    have_result = true;
}

static float log_entry(const LogEntry *e, float x, float y, bool draw) {
    char head[48];
    Color c;
    if (e->r == RESULT_TIE) snprintf(head, sizeof head, "Tie:"), c = TIE;
    else if (e->r == RESULT_LOSE) snprintf(head, sizeof head, "Both lose:"), c = LOSE;
    else {
        snprintf(head, sizeof head, "%s wins:", SIDE_NAME[e->winner]);
        c = e->winner == TRANSFORMER ? conf_color(1) : SMART_GUESS;
    }
    float w = text_w(head, 24) + 10;
    if (draw) {
        text(head, x, y, 24, c);
        text(e->answer, x + w, y, 24, TEXT);
    }
    return w + text_w(e->answer, 24);
}

static Camera2D cam;

static void draw_result(Rectangle r) {
    DrawRectangleRounded(r, 0.15f, 8, PANEL);
    DrawRectangleRoundedLinesEx(r, 0.15f, 8, 1.5f, (Color){60, 78, 115, 255});
    float cx = r.x + r.width / 2, ly = r.y + (r.height - 24) / 2;
    if (have_result) {
        float w = log_entry(&last_result, 0, 0, false);
        log_entry(&last_result, cx - w / 2, ly, true);
    }
}

static void draw_bottom(void) {
    // the result under the left panel, the scoreboard under the right one
    draw_result(result_box);
    draw_scoreboard(score_box);
}


static void typing(void) {
    if (mode != MODE_OWN || !own_focus) return;
    int c;
    while ((c = GetCharPressed()) > 0) {
        size_t n = strlen(own_text);
        if (c >= 'a' && c <= 'z') c -= 32;
        if (n < MAX_ANSWER - 1 && c < 128 && ((c >= 'A' && c <= 'Z') || strchr(" '&-.,!?", c))) {
            own_text[n] = (char)c;
            own_text[n + 1] = 0;
            own_error = NULL;
        }
    }
    if ((IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE)) && own_text[0])
        own_text[strlen(own_text) - 1] = 0;
    if (IsKeyPressed(KEY_ENTER)) play_own();
}

static void frame(void) {
    layout();
    fetch_moves();
    typing();
    if (IsKeyPressed(KEY_ESCAPE)) {   // closes the category list, then the dialog
        if (dropdown) dropdown = false;
        else own_focus = own_modal = false;
    }
    double now = GetTime();
    if (!own_focus && !own_modal && notice == NOTICE_NONE && IsKeyPressed(KEY_SPACE)) action();
    if (turn_active) {   // the turn ends when every side that played has its move
        bool done = true;
        for (int s = 0; s < NSIDES; s++) done &= !want[s];
        if (done) turn_active = false, last_step = now;
    }
    if (playing && !turn_active && now - last_step >= PACE) begin_turn();
    if (have_match && match_over(&match) && over_at == 0) over_at = now;
    if (have_match && match_over(&match) && !counted) {
        Result r = match_result();
        tally[r == RESULT_WIN ? match_winner(&match) : r == RESULT_TIE ? 2 : 3]++;
        counted = true;
        if (++games_done % CHECK_IN == 0) playing = false, notice = NOTICE_CHECK_IN;
        log_result();
    }
    // after a match: Random takes the next puzzle, Demo the next example;
    // when there is nothing left, play stops
    if (playing && over_at && now - over_at > NEXT_MATCH && !need_puzzle) {
        if (mode == MODE_RANDOM) need_puzzle = true;
        else if (mode == MODE_DEMO && preset < npresets - 1) show_preset(preset + 1);
        else playing = false;
    }
    // Random's next puzzle; the last game stays on screen until it arrives
    if (mode == MODE_RANDOM && (playing || notice == NOTICE_INTRO) && need_puzzle && !fetching_puzzle &&
        now >= puzzle_retry_at) {
        puzzle_retry_at = 0;
        random_puzzle();
    }
    if (!ncats && !cats_loading && now >= cats_retry_at) {
        cats_loading = true;
        api_get("/categories", got_categories, NULL);
    }
    if (!npresets && !presets_loading && now >= presets_retry_at) {
        presets_loading = true;
        api_get("/presets", got_presets, NULL);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && own_focus && !CheckCollisionPointRec(mouse(), own_field)) own_focus = false;

    float sc = fminf(GetScreenWidth() / (float)W, GetScreenHeight() / (float)H);
    cam = (Camera2D){.offset = {(GetScreenWidth() - W * sc) / 2, (GetScreenHeight() - H * sc) / 2}, .zoom = sc};
    BeginDrawing();
    ClearBackground(BG);
    BeginMode2D(cam);
    bool modal = mode == MODE_OWN && own_modal, up = notice != NOTICE_NONE;
    input_blocked = mode_open || up;
    draw_top();
    input_blocked = mode_open || modal || up;
    draw_info();
    for (int s = 0; s < NSIDES; s++) draw_side(s, side_box[s].x, side_box[s].y, side_box[s].width, side_box[s].height);
    draw_bottom();
    input_blocked = mode_open || dropdown || up;
    draw_new_game();
    input_blocked = mode_open || up;
    draw_dropdown();
    input_blocked = up;
    draw_mode_menu();
    input_blocked = false;
    draw_notice();
    EndMode2D();
    EndDrawing();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "WOF Solver");
    font = LoadFontFromMemory(".ttf", nunito_ttf, (int)nunito_ttf_len, 64, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
    SetExitKey(KEY_NULL);
    api_init();

    set_mode(MODE_RANDOM);
    playing = false;   // the first game loads behind the overview and waits for Start
    emscripten_set_main_loop(frame, 0, 1);
    return 0;
}
