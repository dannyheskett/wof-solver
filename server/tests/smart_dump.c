// For the parity check (tests/smart_parity.py): reads game states, one per
// line as mode<TAB>category<TAB>board<TAB>called, and prints smart-solve's
// move for each. Timing goes to stderr.
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "smart.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: smart_dump wordlist.txt puzzles.csv[,more.csv] < states\n");
        return 2;
    }
    char err[256];
    double t0 = now();
    Smart *s = smart_load(argv[1], argv[2], err, sizeof err);
    if (!s) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    fprintf(stderr, "loaded in %.0f ms\n", 1000 * (now() - t0));
    char line[1024], out[512];
    double t[3] = {0};
    int n[3] = {0};
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        char *f[4] = {line};
        for (int i = 1; i < 4; i++) {
            f[i] = strchr(f[i - 1], '\t');
            if (!f[i]) return 3;
            *f[i]++ = 0;
        }
        int m = !strcmp(f[0], "main") ? 0 : !strcmp(f[0], "bonus_pick") ? 1 : 2;
        t0 = now();
        int rc = m == 0 ? smart_main(s, f[1], f[2], f[3], out, sizeof out)
               : m == 1 ? smart_bonus_pick(s, f[1], f[2], f[3], out, sizeof out)
                        : smart_bonus_solve(s, f[1], f[2], f[3], out, sizeof out);
        t[m] += now() - t0, n[m]++;
        printf("%s\n", rc ? "(none)" : out);
    }
    fprintf(stderr, "ms per move: main %.2f (%d), bonus_pick %.2f (%d), bonus_solve %.2f (%d)\n",
            n[0] ? 1000 * t[0] / n[0] : 0, n[0], n[1] ? 1000 * t[1] / n[1] : 0, n[1],
            n[2] ? 1000 * t[2] / n[2] : 0, n[2]);
    smart_free(s);
    return 0;
}
