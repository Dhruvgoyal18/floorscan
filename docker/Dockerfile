# Clean-machine image: CPU-only, weights baked in at build time so the walk-in needs no network.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY scan scan
COPY eval eval
COPY scripts scripts
RUN pip install --no-cache-dir -e ".[dev]" \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[models]" "mapanything @ git+https://github.com/facebookresearch/map-anything.git" \
 && python scripts/fetch_weights.py
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["scan"]
# usage: docker build -f docker/Dockerfile -t floorscan . && docker run --rm -v $PWD/caps:/caps -v $PWD/out:/app/out floorscan run /caps/<capture>
