# Control-plane image (api, scheduler, migrate). Pure Python + ffprobe; multi-arch (arm64 ok).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV MIGRATIONS_DIR=/app/migrations
COPY pyproject.toml ./
COPY shared ./shared
COPY services ./services
COPY migrations ./migrations
RUN pip install --no-cache-dir .
RUN useradd --system --uid 10001 vi && mkdir -p /data/media && chown vi /data/media
USER vi
