FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY artifacts ./artifacts

ENV ARTIFACTS_DIR=/app/artifacts
# Render (and most PaaS Docker hosts) inject their own $PORT at runtime,
# which overrides this default -- it only matters for `docker run` locally.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
