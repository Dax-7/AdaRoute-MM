FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    OLLAMA_BASE_URL=http://ollama:11434

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY adaroute /app/adaroute
COPY scripts /app/scripts
COPY configs /app/configs
COPY docker /app/docker
COPY main.py README.md /app/

RUN chmod +x /app/docker/scripts/*.sh

CMD ["sh", "/app/docker/scripts/run_all.sh"]
