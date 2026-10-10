// Image decoding for POST /decode: stb_image, JPEG and PNG only, from memory.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MAX_DIMENSIONS 16384
#include "stb_image.h"

#include "image.h"

int image_info(const uint8_t *data, size_t len, int *w, int *h) {
    int n;
    return stbi_info_from_memory(data, (int)len, w, h, &n) ? 0 : -1;
}

uint8_t *image_rgb(const uint8_t *data, size_t len, int *w, int *h) {
    int n;
    return stbi_load_from_memory(data, (int)len, w, h, &n, 3);
}

void image_free(uint8_t *rgb) { stbi_image_free(rgb); }
