// JPEG and PNG decoding from memory (stb_image), for POST /decode.
#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>
#include <stdint.h>

// The image's size without decoding it. Returns 0, or -1 if it is not a
// JPEG or PNG this decoder reads.
int image_info(const uint8_t *data, size_t len, int *w, int *h);
// The pixels as RGB, 3 bytes each, or NULL. Free with image_free.
uint8_t *image_rgb(const uint8_t *data, size_t len, int *w, int *h);
void image_free(uint8_t *rgb);

#endif
