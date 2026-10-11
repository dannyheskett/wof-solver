// For the parity check (tests/decoder_parity.py): reads raw RGB images, each
// as a line "w h path" on stdin (the file holds w*h*3 bytes), letterboxes
// each and runs the decoder. Prints one JSON line per image: the letterboxed
// input's byte sum, the present probability, the category and the 52 squares'
// class probabilities. With a third argument, also writes each letterboxed
// input to <dir>/<n>.rgb. Timing goes to stderr.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "decoder.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: decoder_dump decoder.bin [letterbox_dir] < list\n");
        return 2;
    }
    char err[256];
    Decoder *d = decoder_load(argv[1], err, sizeof err);
    if (!d) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    static uint8_t in[DEC_IN_H * DEC_IN_W * 3];
    static DecodeResult r;
    int w, h, n = 0;
    char path[1024];
    while (scanf("%d %d %1023s", &w, &h, path) == 3) {
        size_t len = (size_t)w * h * 3;
        uint8_t *rgb = malloc(len);
        FILE *f = fopen(path, "rb");
        if (!rgb || !f || fread(rgb, 1, len, f) != len) {
            fprintf(stderr, "%s: cannot read %zu bytes\n", path, len);
            return 1;
        }
        fclose(f);
        double t0 = now();
        decoder_letterbox(rgb, w, h, in);
        double t1 = now();
        if (decoder_forward(d, in, &r)) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        double t2 = now();
        fprintf(stderr, "%s: letterbox %.3f s, forward %.3f s\n", path, t1 - t0, t2 - t1);
        if (argc == 3) {
            char out[1100];
            snprintf(out, sizeof out, "%s/%d.rgb", argv[2], n);
            FILE *o = fopen(out, "wb");
            fwrite(in, 1, sizeof in, o);
            fclose(o);
        }
        unsigned long long sum = 0;
        for (size_t i = 0; i < sizeof in; i++) sum += in[i];
        printf("{\"sum\": %llu, \"present\": %.6f, \"category\": \"%s\", \"probs\": [", sum, r.present,
               r.category);
        for (int s = 0; s < DEC_SQUARES; s++) {
            printf("%s[", s ? ", " : "");
            for (int k = 0; k < DEC_CLASSES; k++) printf("%s%.6g", k ? ", " : "", r.probs[s][k]);
            printf("]");
        }
        printf("]}\n");
        fflush(stdout);
        free(rgb);
        n++;
    }
    decoder_free(d);
    return 0;
}
