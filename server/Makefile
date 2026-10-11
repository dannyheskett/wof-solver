# wof-server, and the tools behind its parity checks (tests/*.py).
CC ?= cc
CFLAGS ?= -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -O3 -march=x86-64-v3 -ffp-contract=fast
LDLIBS = -lm -pthread
SERVER_LIBS = -lwebp
CJSON = ../client/third_party/cjson
PICO = third_party/picohttpparser
STB = third_party/stb
SERVER_SRC = src/main.c src/model.c src/smart.c src/transformer.c src/decoder.c src/image.c \
             $(CJSON)/cJSON.c $(PICO)/picohttpparser.c

all: build/wof-server build/model_dump build/smart_dump build/tf_dump build/decoder_dump

build/wof-server: $(SERVER_SRC) src/*.h | build
	$(CC) $(CFLAGS) -Isrc -I$(CJSON) -I$(PICO) -I$(STB) $(SERVER_SRC) -o $@ $(SERVER_LIBS) $(LDLIBS)

build/model_dump: tests/model_dump.c src/model.c src/model.h | build
	$(CC) $(CFLAGS) -Isrc tests/model_dump.c src/model.c -o $@ $(LDLIBS)

build/smart_dump: tests/smart_dump.c src/smart.c src/smart.h | build
	$(CC) $(CFLAGS) -Isrc tests/smart_dump.c src/smart.c -o $@ $(LDLIBS)

build/tf_dump: tests/tf_dump.c src/transformer.c src/transformer.h src/model.c src/model.h | build
	$(CC) $(CFLAGS) -Isrc tests/tf_dump.c src/transformer.c src/model.c -o $@ $(LDLIBS)

build/decoder_dump: tests/decoder_dump.c src/decoder.c src/decoder.h | build
	$(CC) $(CFLAGS) -Isrc tests/decoder_dump.c src/decoder.c -o $@ $(LDLIBS)

build:
	mkdir -p build

.PHONY: all clean
clean:
	rm -rf build
