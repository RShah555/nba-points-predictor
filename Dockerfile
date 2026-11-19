# API image. The trained model in models/ is baked in; league data is downloaded on first start.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY models ./models
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app

ENV NBA_DATA_DIR=/app/data NBA_ARTIFACTS_DIR=/app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "nba_predictor.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
