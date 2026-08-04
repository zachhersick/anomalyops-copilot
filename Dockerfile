FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ANOMALYOPS_RETRIEVAL_BACKEND=manifest
ENV ANOMALYOPS_MANIFEST_PATH=/app/outputs/chunks.json
ENV ANOMALYOPS_AI_PROVIDER=deterministic

COPY pyproject.toml ./
COPY copilot ./copilot
COPY scripts ./scripts
COPY data_sources/anomaly_detection_platform ./data_sources/anomaly_detection_platform

RUN pip install --no-cache-dir .

RUN mkdir -p outputs \
    && python scripts/ingest_sources.py \
        data_sources/anomaly_detection_platform \
        --output outputs/chunks.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "copilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]