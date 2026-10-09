# WOF Solver demo client: C99 + raylib, built for the web with Emscripten.
#
#   bash scripts/build_raylib_web.sh   once: raylib 6.0 for the web
#   make web                           build/web/wof-solver.{html,js,wasm}
#   make web-serve                     http://localhost:8000/wof-solver.html
#   make test                          native tests of the game logic
#
# The page talks to https://wof-solver.fly.dev; add ?api=<url> to use
# another move API (for a local one, run it with
# CORS_ORIGINS=http://localhost:8000).
RAYLIB_WEB := third_party/raylib-install-web
SRC        := src/main.c src/match.c src/api.c third_party/cjson/cJSON.c
CFLAGS     := -std=c99 -Wall -Wextra -Isrc -Ithird_party/cjson -I$(RAYLIB_WEB)/include -DPLATFORM_WEB -Os
# Fixed memory, as in the games: a growable heap gives resizable
# ArrayBuffers, which browsers reject in WebGL texImage2D.
LDFLAGS    := -Os -sUSE_GLFW=3 -sFETCH=1 -sINITIAL_MEMORY=67108864 \
              -sEXPORTED_RUNTIME_METHODS=stringToNewUTF8,UTF8ToString \
              --shell-file web/shell.html $(RAYLIB_WEB)/lib/libraylib.a
OUT        := build/web/wof-solver.html

web: $(OUT)

$(OUT): $(SRC) $(wildcard src/*.h) web/shell.html
	@mkdir -p build/web
	emcc $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
	@echo "[web] built $@"

web-serve: $(OUT)
	@echo "http://localhost:8000/wof-solver.html"
	@cd build/web && python3 -m http.server 8000

test:
	@mkdir -p build
	cc -std=c99 -Wall -Wextra -O2 -o build/test_match tests/test_match.c src/match.c
	./build/test_match

clean:
	rm -rf build

.PHONY: web web-serve test clean
