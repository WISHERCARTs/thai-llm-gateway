# Image for Cloud Run (and local docker compose). Cloud Run sets PORT; 8080 locally.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gateway ./gateway
# Warm PyThaiNLP's dictionary and stop words at build time so the first request is not slow.
RUN python -c "from gateway.guardrails.pii import find_pii; find_pii('นายสมชาย ใจดี')"

RUN useradd --create-home --uid 10001 gateway
USER gateway

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
