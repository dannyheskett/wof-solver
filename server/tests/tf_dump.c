// For the parity check (tests/tf_parity.py): reads game states, one per line
// as mode<TAB>category<TAB>board<TAB>called, and prints transformer-solve's
// move for each, then its confidence (or -). Timing goes to stderr.
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "transformer.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: tf_dump weights.bin < states\n");
        return 2;
    }
    Model m;
    char err[256];
    if (model_load(&m, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    Seq *s = seq_new(&m);
    char line[1024];
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
        int k = !strcmp(f[0], "main") ? 0 : !strcmp(f[0], "bonus_pick") ? 1 : 2;
        TfMove mv;
        double t0 = now();
        int rc = k == 0 ? tf_main(&m, s, f[1], f[2], f[3], 0.99, &mv)
               : k == 1 ? tf_bonus_pick(&m, s, f[1], f[2], f[3], &mv)
                        : tf_bonus_solve(&m, s, f[1], f[2], f[3], &mv);
        t[k] += now() - t0, n[k]++;
        if (rc) printf("(none)\t-\n");
        else if (mv.has_conf) printf("%s\t%.6f\n", mv.move, mv.confidence);
        else printf("%s\t-\n", mv.move);
        fflush(stdout);
    }
    fprintf(stderr, "ms per move: main %.1f (%d), bonus_pick %.1f (%d), bonus_solve %.1f (%d)\n",
            n[0] ? 1000 * t[0] / n[0] : 0, n[0], n[1] ? 1000 * t[1] / n[1] : 0, n[1],
            n[2] ? 1000 * t[2] / n[2] : 0, n[2]);
    return 0;
}
