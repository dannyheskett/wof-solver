// The board decoder (scripts/14_vision/decoder_net.py) in plain C: load the
// weights written by scripts/14_vision/export_decoder.py, read one image,
// return the 52 squares, the category text and whether a board is there.
// Single-threaded, float32 throughout. Weights are about 10 MB, mapped from
// the file, so they are page cache the kernel can drop rather than memory
// the process holds. A read allocates about 40 MB of scratch and frees it.
#ifndef DECODER_H
#define DECODER_H

#include <stddef.h>
#include <stdint.h>

#define DEC_IN_W 640
#define DEC_IN_H 360
#define DEC_SQUARES 52
#define DEC_CLASSES 37
#define DEC_CAT_LEN 24
#define DEC_CAT_CHARS 45

// The square classes, in the order of the model's outputs (board.CLASSES):
// '#' unused, '_' hidden, '*' lit, a letter or punctuation shown, '~' covered.
extern const char DEC_CLASS_CHARS[DEC_CLASSES + 1];

typedef struct Decoder Decoder;

typedef struct {
    float present;                       // probability a board is in the image
    char grid[4][15];                    // 14 class chars a row, ' ' off-grid
    float conf[DEC_SQUARES];             // probability of each square's class
    float probs[DEC_SQUARES][DEC_CLASSES];
    char category[DEC_CAT_LEN + 1];      // the category strip as read
} DecodeResult;

// Returns a decoder, or NULL with a message in err.
Decoder *decoder_load(const char *path, char *err, size_t errlen);
void decoder_free(Decoder *d);

// Letterbox an RGB image (w x h, 3 bytes a pixel) into the 640 x 360 input,
// resizing as Pillow's BILINEAR does, centred on black.
void decoder_letterbox(const uint8_t *rgb, int w, int h, uint8_t *out /* 360*640*3 */);

// Run the model on a letterboxed input. Returns 0, or -1 when out of memory.
int decoder_forward(const Decoder *d, const uint8_t *input, DecodeResult *out);

// The square index (0..51) of grid row r, column c, or -1 off-grid.
int decoder_square(int r, int c);

#endif
