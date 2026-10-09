// transformer-solve: moves from the 59M-parameter transformer (model.h).
// Each move is the greedy completion of the training-format prompt, e.g.
//     MAIN|THING|A _A__L_ __ ___ ______|AL|   ->   BUY E
// held to the rules at every character: only the move shapes the round
// allows, only uncalled letters, no BUY before a vowel can be bought, and a
// SOLVE that fits the board square by square.
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"

#define TF_MAX_TEXT 128

typedef struct {
    char move[TF_MAX_TEXT + 16];
    // for a SOLVE, kept or turned down: the model's probability for each
    // character of the answer among those its square allows, and their product
    int has_conf;
    double confidence;
    double letters[TF_MAX_TEXT];
    int nletters;
    char turned_down[TF_MAX_TEXT + 16];   // a MAIN SOLVE under min_p, "" if none
} TfMove;

// board has '_' for hidden squares, called is sorted A-Z. A MAIN SOLVE with
// confidence under min_p while a square is blank is turned down, and the
// best letter move is made instead. Return 0, or -1 if the prompt is too
// long for the model.
int tf_main(const Model *m, Seq *s, const char *category, const char *board, const char *called, double min_p,
            TfMove *out);
int tf_bonus_pick(const Model *m, Seq *s, const char *category, const char *board, const char *called,
                  TfMove *out);
int tf_bonus_solve(const Model *m, Seq *s, const char *category, const char *board, const char *called,
                   TfMove *out);

#endif
