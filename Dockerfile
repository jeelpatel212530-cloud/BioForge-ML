# Inference image. Build AFTER `make train` so that artifacts/ contains the serving bundle
# (model.joblib, ad_reference.npz, model_meta.json). No MLflow database or training data needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY configs ./configs
COPY src ./src
COPY artifacts ./artifacts

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=4)); sys.exit(0 if d['model_loaded'] else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
