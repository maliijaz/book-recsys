FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline

# Default command runs the full offline pipeline; override with e.g.
# `docker compose run --rm pipeline python -m pipeline.cli train --model ncf`
CMD ["python", "-m", "pipeline.cli", "run-all"]
