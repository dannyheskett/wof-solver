// Game logic for a head-to-head match. The rules match scripts/11_evaluate.py:
// a letter called twice, a consonant bought, a vowel called, or a vowel bought
// before a called consonant shows (the money rule) is invalid and ends that
// side's game, as does its first SOLVE, right or wrong.
#include "match.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static const int ROW_WIDTH[BOARD_ROWS] = {12, 14, 14, 12};
static const char *FREE = "RSTLNE";
static const char *ALLOWED = " '&-.,!?";

static bool is_letter(char c) { return c >= 'A' && c <= 'Z'; }
static bool is_vowel(char c) { return c && strchr("AEIOU", c) != NULL; }

// ---------------------------------------------------------------- layout --

// Rows break between words or right after a hyphen, as in
// wof_common.fits_board. Each piece is a run of the answer; `joined` means
// it follows the previous piece with no space (after a hyphen).
typedef struct { int start, len; bool joined; } Piece;

static int pieces(const char *a, Piece *out) {
    int n = 0, i = 0, len = (int)strlen(a);
    while (i < len) {
        while (i < len && a[i] == ' ') i++;
        if (i >= len) break;
        int start = i;
        bool joined = false;
        while (i < len && a[i] != ' ') {
            int ps = i;
            while (i < len && a[i] != ' ' && a[i] != '-') i++;
            if (i < len && a[i] == '-') i++;           // the hyphen stays on this piece
            out[n++] = (Piece){ps, i - ps, joined};
            joined = true;
        }
        (void)start;
    }
    return n;
}

static bool place(const char *a, const Piece *p, int np, int first, Layout *out) {
    int rows[BOARD_ROWS][2];   // first piece, piece count per row
    int i = 0, used = 0;
    for (int r = first; r < BOARD_ROWS && i < np; r++) {
        int w = 0, start = i;
        while (i < np) {
            int need = p[i].len + (w && !p[i].joined ? 1 : 0);
            if (w + need > ROW_WIDTH[r]) break;
            w += need;
            i++;
        }
        if (i == start) return false;                   // a piece longer than the row
        rows[used][0] = start;
        rows[used][1] = i - start;
        used++;
    }
    if (i < np) return false;
    int len = (int)strlen(a);
    out->len = len;
    for (int k = 0; k < len; k++) out->row[k] = out->col[k] = -1;
    for (int u = 0; u < used; u++) {
        int r = first + u, w = 0;
        for (int k = rows[u][0]; k < rows[u][0] + rows[u][1]; k++)
            w += p[k].len + (k > rows[u][0] && !p[k].joined ? 1 : 0);
        // centre in the row; rows 1 and 4 are inset one square on the 14 grid
        int col = (ROW_WIDTH[r] == 12 ? 1 : 0) + (ROW_WIDTH[r] - w) / 2;
        for (int k = rows[u][0]; k < rows[u][0] + rows[u][1]; k++) {
            if (k > rows[u][0] && !p[k].joined) {        // the space before it
                out->row[p[k].start - 1] = r;
                out->col[p[k].start - 1] = col++;
            }
            for (int c = 0; c < p[k].len; c++) {
                out->row[p[k].start + c] = r;
                out->col[p[k].start + c] = col++;
            }
        }
    }
    return true;
}

bool layout_answer(const char *answer, Layout *out) {
    Piece p[MAX_ANSWER];
    int np = pieces(answer, p);
    if (np == 0 || (int)strlen(answer) >= MAX_ANSWER) return false;
    // short answers sit on the two middle rows, as on the show
    return place(answer, p, np, 1, out) || place(answer, p, np, 0, out);
}

const char *check_answer(char *a) {
    // uppercase, trim, single spaces
    char buf[MAX_ANSWER * 2];
    int n = 0;
    for (const char *c = a; *c && n < (int)sizeof buf - 1; c++) {
        char ch = (char)toupper((unsigned char)*c);
        if (ch == ' ' && (n == 0 || buf[n - 1] == ' ')) continue;
        buf[n++] = ch;
    }
    while (n && buf[n - 1] == ' ') n--;
    buf[n] = 0;
    if (n == 0) return "Type a puzzle.";
    if (n >= MAX_ANSWER) return "Too long for the board.";
    bool letter = false;
    for (int i = 0; i < n; i++) {
        if (is_letter(buf[i])) letter = true;
        else if (!strchr(ALLOWED, buf[i])) return "Use letters, spaces and ' & - . , ! ? only.";
    }
    if (!letter) return "A puzzle needs at least one letter.";
    Layout l;
    if (!layout_answer(buf, &l)) return "Too long for the board (4 rows: 12, 14, 14, 12).";
    strcpy(a, buf);
    return NULL;
}

// ----------------------------------------------------------------- match --

static void board_of(const char *answer, const char *called, char *board) {
    int i = 0;
    for (; answer[i]; i++)
        board[i] = is_letter(answer[i]) && !strchr(called, answer[i]) ? '_' : answer[i];
    board[i] = 0;
}

static bool can_buy(const char *board, const char *called) {
    int cons = 0;
    for (const char *c = called; *c; c++) {
        if (is_vowel(*c)) continue;
        cons++;
        if (strchr(board, *c)) return true;
    }
    return cons == 21;
}

void match_start(Match *m, Round round, const char *category, const char *answer) {
    unsigned gen = m->gen + 1;
    memset(m, 0, sizeof *m);
    m->gen = gen;
    m->round = round;
    snprintf(m->category, sizeof m->category, "%s", category);
    snprintf(m->answer, sizeof m->answer, "%s", answer);
    for (int s = 0; s < NSIDES; s++)
        if (round == ROUND_BONUS) strcpy(m->side[s].called, FREE);
}

static void sorted(const char *in, char *out) {
    int n = 0;
    for (char c = 'A'; c <= 'Z'; c++)
        if (strchr(in, c)) out[n++] = c;
    out[n] = 0;
}

bool match_next_request(const Match *m, int s, const char **mode, char *board, char *called) {
    const Side *d = &m->side[s];
    if (d->waiting || d->failed || d->status != PLAYING) return false;
    board_of(m->answer, d->called, board);
    sorted(d->called, called);
    if (m->round == ROUND_MAIN) *mode = "main";
    else *mode = d->nmoves == 0 ? "bonus_pick" : "bonus_solve";
    return true;
}

static int count_in(const char *answer, char c) {
    int n = 0;
    for (const char *a = answer; *a; a++) n += *a == c;
    return n;
}

static void add_called(Side *d, char c) {
    size_t n = strlen(d->called);
    d->called[n] = c;
    d->called[n + 1] = 0;
}

void match_apply(Match *m, int s, const Move *in) {
    Side *d = &m->side[s];
    if (d->status != PLAYING || d->nmoves >= MAX_MOVES) return;
    Move *mv = &d->moves[d->nmoves++];
    *mv = *in;
    char board[MAX_ANSWER];
    board_of(m->answer, d->called, board);

    if (strncmp(mv->move, "SOLVE ", 6) == 0) {
        d->status = strcmp(mv->move + 6, m->answer) == 0 ? SOLVED : WRONG;
        return;
    }
    if (m->round == ROUND_BONUS) {
        // PICK BCD A: three new consonants, one new vowel
        char c[4] = {0}, v = 0;
        bool ok = sscanf(mv->move, "PICK %3[A-Z] %c", c, &v) == 2 && strlen(c) == 3
                  && is_vowel(v) && !strchr(FREE, v)
                  && c[0] != c[1] && c[0] != c[2] && c[1] != c[2];
        for (int i = 0; ok && i < 3; i++)
            ok = !is_vowel(c[i]) && !strchr(FREE, c[i]);
        if (!ok) { d->status = INVALID; return; }
        snprintf(mv->picked, sizeof mv->picked, "%s%c", c, v);
        for (int i = 0; i < 4; i++) {
            add_called(d, mv->picked[i]);
            mv->hits += count_in(m->answer, mv->picked[i]);
        }
        return;
    }
    char verb[8] = {0}, x = 0;
    bool ok = sscanf(mv->move, "%7s %c", verb, &x) == 2 && is_letter(x) && !strchr(d->called, x)
              && strlen(mv->move) == strlen(verb) + 2;
    if (ok && strcmp(verb, "CALL") == 0) ok = !is_vowel(x);
    else if (ok && strcmp(verb, "BUY") == 0) ok = is_vowel(x) && can_buy(board, d->called);
    else ok = false;
    if (!ok) { d->status = INVALID; return; }
    mv->letter = x;
    mv->hits = count_in(m->answer, x);
    add_called(d, x);
    if (d->nmoves >= MAX_MOVES) d->status = OUT_OF_MOVES;
}

// ---------------------------------------------------------------- viewer --

void side_called(const Match *m, int s, int shown, char *called) {
    const Side *d = &m->side[s];
    strcpy(called, m->round == ROUND_BONUS ? FREE : "");
    for (int i = 0; i < shown && i < d->nmoves; i++) {
        const Move *mv = &d->moves[i];
        if (mv->letter) strncat(called, &mv->letter, 1);
        if (mv->picked[0]) strcat(called, mv->picked);
    }
}

void side_board(const Match *m, int s, int shown, char *board) {
    char called[40];
    side_called(m, s, shown, called);
    board_of(m->answer, called, board);
}

Status side_status(const Match *m, int s, int shown) {
    const Side *d = &m->side[s];
    return shown >= d->nmoves ? d->status : PLAYING;
}

int side_misses(const Match *m, int s, int shown) {
    char called[40];
    side_called(m, s, shown, called);
    int n = 0;
    for (const char *c = called + (m->round == ROUND_BONUS ? 6 : 0); *c; c++)
        n += !strchr(m->answer, *c);
    return n;
}

int side_letters(const Match *m, int s, int shown) {
    char called[40];
    side_called(m, s, shown, called);
    return (int)strlen(called) - (m->round == ROUND_BONUS ? 6 : 0);
}

static bool side_done(const Match *m, int s) {
    const Side *d = &m->side[s];
    return d->failed || side_status(m, s, d->shown) != PLAYING;
}

bool match_step(Match *m) {
    for (int s = 0; s < NSIDES; s++)
        if (!side_done(m, s) && m->side[s].shown >= m->side[s].nmoves) return false;
    bool moved = false;
    for (int s = 0; s < NSIDES; s++)
        if (!side_done(m, s)) { m->side[s].shown++; moved = true; }
    if (moved) m->turn++;
    return moved;
}

bool match_over(const Match *m) {
    return side_done(m, SMART) && side_done(m, TRANSFORMER);
}

int match_winner(const Match *m) {
    int sh[NSIDES];
    bool ok[NSIDES];
    for (int s = 0; s < NSIDES; s++) {
        sh[s] = m->side[s].shown;
        ok[s] = side_status(m, s, sh[s]) == SOLVED;
    }
    if (ok[SMART] != ok[TRANSFORMER]) return ok[SMART] ? SMART : TRANSFORMER;
    if (!ok[SMART]) return -1;
    int a = side_misses(m, SMART, sh[SMART]), b = side_misses(m, TRANSFORMER, sh[TRANSFORMER]);
    if (a != b) return a < b ? SMART : TRANSFORMER;
    a = side_letters(m, SMART, sh[SMART]);
    b = side_letters(m, TRANSFORMER, sh[TRANSFORMER]);
    if (a != b) return a < b ? SMART : TRANSFORMER;
    return -1;
}
