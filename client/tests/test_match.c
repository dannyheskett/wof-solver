// Native tests of the game logic in src/match.c: make test
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../src/match.h"

static Move mv(const char *text) { Move m = {0}; snprintf(m.move, sizeof m.move, "%s", text); return m; }

int main(void) {
    Layout l;
    // short answers sit on the middle rows, centred
    assert(layout_answer("A PIECE OF CAKE", &l));
    assert(l.row[0] == 1 && l.row[11] == 2);
    // 4 rows of 12/14/14/12 at most
    assert(!layout_answer("SUPERCALIFRAGILISTICEXPIALIDOCIOUS", &l));

    char a[80] = "  a piece   of cake ";
    assert(check_answer(a) == NULL && strcmp(a, "A PIECE OF CAKE") == 0);
    char b[80] = "HELLO@WORLD";
    assert(check_answer(b) != NULL);

    Match m = {0};
    match_start(&m, ROUND_MAIN, "Thing", "A PIECE OF CAKE");
    Move x = mv("BUY E");                     // no vowel before a consonant hits
    match_apply(&m, SMART, &x);
    assert(m.side[SMART].status == INVALID);
    x = mv("CALL K");
    match_apply(&m, TRANSFORMER, &x);
    assert(m.side[TRANSFORMER].moves[0].hits == 1);
    x = mv("BUY E");
    match_apply(&m, TRANSFORMER, &x);
    assert(m.side[TRANSFORMER].moves[1].hits == 3);
    x = mv("CALL K");                         // called twice
    Match m2 = m;
    match_apply(&m2, TRANSFORMER, &x);
    assert(m2.side[TRANSFORMER].status == INVALID);
    x = mv("SOLVE A PIECE OF CAKE");
    match_apply(&m, TRANSFORMER, &x);
    assert(m.side[TRANSFORMER].status == SOLVED);
    while (match_step(&m)) {}
    assert(match_over(&m) && match_winner(&m) == TRANSFORMER);

    match_start(&m, ROUND_BONUS, "Phrase", "YOU CRACK ME UP");
    x = mv("PICK BDM O");
    match_apply(&m, SMART, &x);
    assert(m.side[SMART].moves[0].hits == 2);
    x = mv("PICK BBM O");                     // repeated consonant
    match_apply(&m, TRANSFORMER, &x);
    assert(m.side[TRANSFORMER].status == INVALID);
    puts("match tests passed");
    return 0;
}
