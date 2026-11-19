# The whole app in one image: FastAPI serves the API and the built dashboard on one port.
#
#   docker build -t nba-predictor . && docker run -p 7860:7860 nba-predictor
#
# Expects models/ (trained bundle) and data/player_logs_<last season>.csv in the build context.
# The season in progress is downloaded from cdn.nba.com at startup and kept up to date.

FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY --chown=app models ./models
COPY --chown=app data ./data
COPY --from=web --chown=app /web/out ./web/out
USER app

ENV NBA_DATA_DIR=/app/data NBA_ARTIFACTS_DIR=/app/models NBA_WEB_DIR=/app/web/out PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn nba_predictor.api:create_app --factory --host 0.0.0.0 --port ${PORT}"]
