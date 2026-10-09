# wof-server: one C binary serving transformer-solve, smart-solve, the move
# API and the WASM client. Build from the wof-solver directory:
#   fly deploy . --config server/fly.toml --dockerfile server/Dockerfile
# The client is built beforehand (make -C client web) and copied in as is.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY server/Makefile server/
COPY server/src server/src
COPY server/third_party server/third_party
COPY client/third_party/cjson client/third_party/cjson
RUN make -C server build/wof-server

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /src/server/build/wof-server /usr/local/bin/wof-server
COPY runs/ft2b/wof.bin models/wof.bin
COPY data/train.csv data/dev.csv data/demo.json data/
COPY data/wordlist/american-english.txt data/wordlist/
COPY client/build/web/ web/
ENV PORT=8080 MIN_P=0.99
EXPOSE 8080
CMD ["wof-server"]
