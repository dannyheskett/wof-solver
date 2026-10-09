// For the parity check (tests/model_parity.py): reads prompts, one per line,
// and prints the log-softmax of the next token after each, then its greedy
// completion up to a newline. Timing goes to stderr.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "model.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: model_dump weights.bin < prompts\n");
        return 2;
    }
    Model m;
    char err[256];
    if (model_load(&m, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    Seq *s = seq_new(&m);
    char line[512];
    int ids[512];
    double t_prompt = 0, t_gen = 0;
    int n_prompt = 0, n_gen = 0;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        int n = 0;
        for (char *c = line; *c; c++) ids[n++] = model_token(&m, *c);
        seq_reset(s);
        double t0 = now();
        const float *lg = model_feed(&m, s, ids, n);
        t_prompt += now() - t0, n_prompt += n;
        float mx = -INFINITY, sum = 0;
        for (int i = 0; i < m.vocab; i++) mx = fmaxf(mx, lg[i]);
        for (int i = 0; i < m.vocab; i++) sum += expf(lg[i] - mx);
        for (int i = 0; i < m.vocab; i++) printf("%s%.6f", i ? " " : "", lg[i] - mx - logf(sum));
        printf("\n");
        char out[128];
        int k = 0;
        t0 = now();
        while (k < 80 && s->n < m.block) {
            int best = 0;
            for (int i = 1; i < m.vocab; i++) if (lg[i] > lg[best]) best = i;
            if (m.alphabet[best] == '\n') break;
            out[k++] = m.alphabet[best];
            lg = model_feed(&m, s, &best, 1);
            n_gen++;
        }
        t_gen += now() - t0;
        out[k] = 0;
        printf("%s\n", out);
    }
    fprintf(stderr, "prompt %.2f ms/token (%d), generate %.2f ms/token (%d)\n",
            1000 * t_prompt / (n_prompt ? n_prompt : 1), n_prompt, 1000 * t_gen / (n_gen ? n_gen : 1), n_gen);
    seq_free(s);
    model_free(&m);
    return 0;
}
