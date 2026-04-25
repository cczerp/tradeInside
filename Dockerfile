FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build deps for pandas/numpy wheels (slim image is missing them
# on some platforms). gcc is needed only at install time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && apt-get purge -y --auto-remove gcc

COPY . .

ENV TRADEINSIDE_DB=/data/tradeinsider.db \
    LOG_FILE=/logs/insider_trading.log

EXPOSE 8000

# Default to the API. Override in compose / k8s for the pipeline workers.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
