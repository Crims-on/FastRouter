FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/fastrouter /usr/local/bin/fastrouter
ENV DATA_DIR=/data PORT=20128 HOST=0.0.0.0
VOLUME /data
EXPOSE 20128
CMD ["fastrouter"]
