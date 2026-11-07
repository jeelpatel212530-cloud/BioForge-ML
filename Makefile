PY ?= python

.PHONY: help install ingest refresh split train all test serve serve-dev mlflow-ui docker-build docker-run clean

help:
	@echo "make install | ingest | split | train | all | test | serve | mlflow-ui | docker-build | docker-run"

install:
	$(PY) -m pip install -r requirements.txt

ingest:  ## fetch ChEMBL (cached after first run) + curate + 2D/3D features
	$(PY) -m src.data.ingest_chembl

refresh:  ## force re-download from ChEMBL
	$(PY) -m src.data.ingest_chembl --refresh

split:  ## write the primary scaffold-split table (for inspection)
	$(PY) -m src.data.scaffold_split

train:  ## ablation + AD calibration + MLflow + export serving bundle to artifacts/
	$(PY) -m src.models.train

all: ingest split train

test:
	$(PY) -m pytest -q

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

serve-dev:
	uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:  ## requires `make train` first (the image bundles artifacts/)
	docker build -t bioforge-ml .

docker-run:
	docker run --rm -p 8000:8000 bioforge-ml

clean:
	rm -rf .pytest_cache mlruns mlflow.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
