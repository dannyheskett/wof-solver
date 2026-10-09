// smart-solve: the word-list solver of scripts/solvers.py (SmartSolver)
// and scripts/wof_common.py (Vocab, BoardWords), in C. Its moves
// match the Python version's, tie-breaks included.
#ifndef SMART_H
#define SMART_H

#include <stddef.h>

typedef struct Smart Smart;

// The word list is the SCOWL file plus every word of the puzzle CSVs
// (answer,category; one path or several, comma-separated), which also give
// the per-category weights and the letter order. NULL on failure, with a message in err.
Smart *smart_load(const char *wordlist, const char *puzzles, char *err, size_t errlen);
void smart_free(Smart *s);

// One move for a board ('_' for hidden squares) and the called letters
// (sorted A-Z). The move goes to out as "CALL X", "BUY X", "SOLVE ..." or
// "PICK BCD A". Returns 0, or -1 when there is no move to make.
int smart_main(Smart *s, const char *category, const char *board, const char *called, char *out, size_t n);
int smart_bonus_pick(Smart *s, const char *category, const char *board, const char *called, char *out,
                     size_t n);
int smart_bonus_solve(Smart *s, const char *category, const char *board, const char *called, char *out,
                      size_t n);

#endif
