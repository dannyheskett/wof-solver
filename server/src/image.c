// Image decoding for POST /decode, from memory: JPEG and PNG with stb_image,
// WebP with libwebp. Both decode into malloc'd memory, freed with free().
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MAX_DIMENSIONS 16384
#include "stb_image.h"

#include <stdlib.h>
#include <string.h>
#include <webp/decode.h>

#include "image.h"

static int is_webp(const uint8_t *data, size_t len) {
    return len >= 12 && !memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WEBP", 4);
}

int image_info(const uint8_t *data, size_t len, int *w, int *h) {
    if (is_webp(data, len)) return WebPGetInfo(data, len, w, h) ? 0 : -1;
    int n;
    return stbi_info_from_memory(data, (int)len, w, h, &n) ? 0 : -1;
}

uint8_t *image_rgb(const uint8_t *data, size_t len, int *w, int *h) {
    if (is_webp(data, len)) {
        if (!WebPGetInfo(data, len, w, h)) return NULL;
        size_t stride = (size_t)*w * 3, size = stride * (size_t)*h;
        uint8_t *rgb = malloc(size);
        if (rgb && !WebPDecodeRGBInto(data, len, rgb, size, (int)stride)) {
            free(rgb);
            rgb = NULL;
        }
        return rgb;
    }
    int n;
    return stbi_load_from_memory(data, (int)len, w, h, &n, 3);   // STBI_MALLOC is malloc
}

void image_free(uint8_t *rgb) { free(rgb); }
