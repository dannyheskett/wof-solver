// A head-to-head match: smart-solve and transformer-solve each play the same
// puzzle on their own board, one move per turn. Moves come from the move API
// (api.c) and are fetched ahead of what the viewer has seen, so stepping is
// instant. This file is the game logic only: no drawing, no network.
#ifndef MATCH_H
#define MATCH_H

#include <stdbool.h>

#define MAX_ANSWER 64
#define MAX_MOVES 40
#define BOARD_ROWS 4

enum { SMART, TRANSFORMER, NSIDES };
typedef enum { ROUND_MAIN, ROUND_BONUS } Round;
typedef enum { PLAYING, SOLVED, WRONG, INVALID, OUT_OF_MOVES } Status;

typedef struct {
    char move[MAX_ANSWER + 8];   // CALL X, BUY X, SOLVE ..., PICK XYZ V
    char tried[MAX_ANSWER];      // the answer behind "confidence": the SOLVE
                                 // itself, or one that was turned down
    bool turned_down;
    bool has_conf;
    float conf;                  // transformer-solve only
    float letters[MAX_ANSWER];   // its confidence per character of tried
    int ms;                      // compute time on the server
    int wall;                    // from request to answer, as the viewer waited
    // filled in when the move is applied
    char letter;                 // CALL/BUY letter
    int hits;                    // squares it showed
    char picked[5];              // PICK letters
} Move;

typedef struct {
    Move moves[MAX_MOVES];
    int nmoves;                  // fetched so far
    int shown;                   // revealed to the viewer
    bool waiting;                // a request is out
    bool failed;                 // the API could not be reached
    // state after all fetched moves
    char called[27];             // in call order
    Status status;
} Side;

typedef struct {
    Round round;
    char category[48];
    char answer[MAX_ANSWER];
    Side side[NSIDES];
    int turn;                    // turns shown
    unsigned gen;                // bumped on every new match, so late replies are dropped
} Match;

// Board layout on the 12/14/14/12 TV board. row/col of each answer character
// (-1 for a space dropped at a row break). Returns false when it does not fit.
typedef struct {
    int row[MAX_ANSWER], col[MAX_ANSWER];
    int len;
} Layout;

bool layout_answer(const char *answer, Layout *out);
// An error message for a puzzle a user typed, or NULL when it is playable.
// Uppercases it in place.
const char *check_answer(char *answer);

void match_start(Match *m, Round round, const char *category, const char *answer);
// What the API should be asked next for this side: false when the side is done
// or waiting. Fills mode ("main", "bonus_pick", "bonus_solve"), board, called.
bool match_next_request(const Match *m, int s, const char **mode, char *board, char *called);
// Apply a fetched move to a side.
void match_apply(Match *m, int s, const Move *mv);

// What the viewer sees after `shown` moves of side s.
void side_board(const Match *m, int s, int shown, char *board);
void side_called(const Match *m, int s, int shown, char *called);
Status side_status(const Match *m, int s, int shown);
int side_misses(const Match *m, int s, int shown);
int side_letters(const Match *m, int s, int shown);

// One step forward for the viewer. Returns false if a side still needs its move.
bool match_step(Match *m);
bool match_over(const Match *m);
// SMART, TRANSFORMER, or -1 for no winner; only meaningful when match_over.
int match_winner(const Match *m);

#endif
