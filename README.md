# BioForge-ML

An end-to-end chemoinformatics and machine learning pipeline for molecular bioactivity prediction, featuring physics-informed descriptors, scaffold-based validation, applicability domain assessment, and production REST API serving.

[![Live API](https://img.shields.io/badge/Render-Live%20Demo-success?logo=render&logoColor=white)](https://bioforge-ml.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white)](https://bioforge-ml.onrender.com/docs)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live API & Documentation

The service is containerized with Docker and continuously deployed on Render.

* **Interactive Swagger UI:** [https://bioforge-ml.onrender.com/docs](https://bioforge-ml.onrender.com/docs)
* **ReDoc Interface:** [https://bioforge-ml.onrender.com/redoc](https://bioforge-ml.onrender.com/redoc)
* **OpenAPI Specification:** [https://bioforge-ml.onrender.com/openapi.json](https://bioforge-ml.onrender.com/openapi.json)

> **Note:** Free-tier instances spin down during periods of inactivity. Please allow 30–50 seconds for the initial warm-up request.

---

## 🧬 Key Features

* **Data Ingestion:** Automated bioactivity retrieval directly from the ChEMBL database with standardized SMILES parsing and target filtering.
* **Bemis-Murcko Scaffold Splitting:** Prevents optimistic cross-validation bias by partitioning train/validation/test sets by molecular core topologies rather than random selection.
* **Hybrid Feature Engineering:** Combines circular Morgan fingerprints (ECFP equivalent) with physicochemical properties (molecular weight, LogP, TPSA, hydrogen bond donors/acceptors, rotatable bonds).
* **Applicability Domain (AD) Estimation:** Identifies structurally out-of-distribution molecules to prevent uncalibrated predictions in production.
* **Artifact Tracking & Model Storage:** Compressed Joblib models packaged with configuration schemas for consistent low-latency inference.
* **REST API:** High-throughput prediction endpoints built with FastAPI and strictly validated using Pydantic schemas.

---

## 📂 Project Structure

```text
BioForge-ML/
├── artifacts/              # Serialized model weights, scalers, and artifacts
├── configs/
│   └── config.yaml         # Training parameters, feature flags, and thresholds
├── data/
│   ├── raw/                # Unprocessed ChEMBL downloads
│   └── processed/          # Cleaned, standardized molecular data
├── reports/                # Validation metrics, ROC/PR curves, and evaluation tables
├── src/
│   ├── api/                # FastAPI application and request/response schemas
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/               # Ingestion and Bemis-Murcko splitting routines
│   │   ├── ingest_chembl.py
│   │   └── scaffold_split.py
│   ├── features/           # Molecular physics and fingerprint feature builders
│   │   ├── builders.py
│   │   ├── fingerprints.py
│   │   └── physics.py
│   ├── models/             # Training engine, reporting, and applicability domain
│   │   ├── applicability.py
│   │   ├── reporting.py
│   │   └── train.py
│   └── utils/              # Configuration loaders and logging utilities
├── tests/                  # Pytest test suite covering split integrity and inference
├── Dockerfile              # Multi-stage production container configuration
├── Makefile                # Build, test, and training orchestration targets
├── requirements.txt        # Core ML and data engineering dependencies
└── requirements-serve.txt  # Lightweight API runtime dependencies
An end-to-end, reproducible QSAR pipeline: **ChEMBL IC50 ingestion → structure curation → 2D + 3D
conformer-ensemble features → scaffold-split evaluation with a built-in ablation → MLflow
tracking/registry → FastAPI service with an applicability-domain guardrail.**

The point of this project is not "a model that predicts IC50". It is a pipeline that **measures
whether its own ideas work** (do 3D features help? does the applicability domain actually predict
error?) and refuses to hide the answer.

---

## What makes it different from a tutorial QSAR notebook

| Concern | Typical tutorial | BioForge-ML |
|---|---|---|
| Data | whatever the API returns, first N rows | filtered server- *and* client-side (binding assays, curated `pchembl_value`, no validity flags, no mutants, `=` relation only); salts stripped, charges neutralised; duplicates aggregated by median; molecules with conflicting replicates (std > 1 log unit) dropped |
| Target | raw nM IC50 | `pIC50 = 9 − log10(IC50 nM)` |
| 3D features | none, or one random conformer | ETKDGv3 **conformer ensemble**, MMFF94-relaxed, **Boltzmann-averaged** shape descriptors |
| Validation | random split (leaks analogs) | seeded **scaffold split**, repeated over several seeds → mean ± std |
| Claims | "3D physics improves the model" | **ablation**: mean baseline vs 2D vs 2D+3D vs ECFP4 vs ECFP4+2D+3D, with a paired per-seed estimate of the 3D effect |
| Safety | returns a number for any SMILES | Tanimoto **applicability domain** + tier, nearest training neighbour, warning; threshold can be checked against real error (`reports/ad_calibration.csv`) |
| Serving | needs the training environment | self-contained bundle (`model.joblib`, `ad_reference.npz` – no pickle, `model_meta.json`); minimal Docker image |
| Honesty | — | pipeline warns when the chosen model **does not beat predicting the mean** |

## Quick start

Tested on Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make install            # or: pip install -r requirements.txt
make test               # 22 tests, ~15 s, no network needed

make all                # = ingest + split + train   (needs internet for the first ingest)
make serve              # API on http://localhost:8000/docs
make mlflow-ui          # experiment tracking on http://localhost:5000
```

No `make` (e.g. Windows)? Run the underlying commands:

```bash
python -m src.data.ingest_chembl      # add --refresh to force a re-download
python -m src.data.scaffold_split
python -m src.models.train
uvicorn src.api.main:app --port 8000
```

Change the protein target, record limit, conformer count, split type, thresholds, etc. in
[`configs/config.yaml`](configs/config.yaml) – every stage reads that one file.

**Run time (rough):** ~0.7 s per drug-sized molecule for a 10-conformer ensemble on one core;
featurization is parallelised (`physics.n_jobs`). A 5 000-molecule target takes minutes on a laptop.
The ablation then trains 5 models × 5 split seeds.

### Try the API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"}'      # gefitinib
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"smiles": "CCCCCCCCCCCC"}'                                        # dodecane -> LOW_OOD_WARNING
```

Response fields: `predicted_pIC50`, `estimated_ic50_nM`, `in_applicability_domain`,
`max_tanimoto_similarity`, `knn_mean_tanimoto`, `confidence_tier`
(`HIGH_IN_DOMAIN` / `MODERATE_IN_DOMAIN` / `LOW_OOD_WARNING`), `nearest_training_smiles`,
`computed_physics_features`, `warnings`. Other endpoints: `GET /health`, `GET /model-info`
(metrics, ablation summary, feature importances, library versions of the served model).

> **Note on gefitinib:** it is a famous EGFR drug and is almost certainly *in* the ChEMBL training
> data, so a high similarity for it only proves the guard works on training molecules. The real test
> of the guard is the held-out analysis in `reports/ad_error_vs_similarity.png`.

### Docker

```bash
make train && make docker-build && make docker-run      # image bundles artifacts/, needs no MLflow DB
```

## Results

> Run `make all`, then paste `reports/ablation_summary.md` here and commit the plots in `reports/`
> (`ablation_rmse.png`, `ad_error_vs_similarity.png`, `ad_calibration.csv`).
> **No numbers are published in this repo on purpose**: they depend on the target, the ChEMBL release
> you download, and your configuration. Report them yourself, together with the "Sanity check vs.
> predicting the mean" line and the "Effect of 3D features" line that the training run writes.

## How it works

```
ChEMBL REST ──► fetch (paginated, retries, filters) ──► curate ──► featurize (parallel)
                                                                       │
        data/processed/<target>_physics_features.parquet ◄─────────────┘
                              │
                 scaffold split (seeded; Murcko or generic)
                              │
        ablation over seeds ──┴──► reports/ (tables, plots)
                              │
   pick serving model ► AD calibration on held-out test ► refit on train+val
                              │
          artifacts/ (model.joblib, ad_reference.npz, model_meta.json)  +  MLflow registry (alias "champion")
                              │
                     FastAPI  /predict
```

**Features** (`src/features/physics.py`): 9 two-dimensional descriptors (MW, cLogP, TPSA, HBD, HBA,
rotatable bonds, heavy atoms, aromatic rings, Fsp3) and 9 three-dimensional ones computed on an
ETKDGv3/MMFF94 conformer ensemble: lowest MMFF94 energy, energy spread across conformers,
Boltzmann-averaged radius of gyration, its spread, asphericity, eccentricity, NPR1/NPR2 (size-independent
shape) and the smallest principal moment of inertia. Training and serving share one code path
(`src/features/builders.py`) so there is no train/serve skew.

**Splitting** (`src/data/scaffold_split.py`): whole scaffold groups go to one partition (leakage is
checked and raises an error). Groups too large to fit validation/test go to train first; the rest are
shuffled by seed, so each seed gives a different, reproducible split.

**Applicability domain** (`src/models/applicability.py`): ECFP4 bit vectors of the training set;
Tanimoto similarity via vectorised matrix products (unit-tested against RDKit). `Tmax ≥ 0.35` →
in domain (configurable).

## Limitations (read before citing any number)

- **IC50 is not binding free energy.** It depends on assay conditions (substrate/ATP concentration,
  enzyme construct). `pIC50` is a convenient log scale, not ΔG. The median-aggregation and replicate
  filters reduce, but do not remove, inter-assay noise.
- **MMFF94 energies are gas-phase and size-dependent**; they are not binding energies. The ablation
  exists precisely to test whether the 3D block adds anything beyond 2D descriptors and ECFP4. On many
  targets, fingerprints alone are hard to beat.
- **No protein structure is used** (ligand-based only). No docking or protonation-state modelling.
- **Scaffold splits are stricter than random splits, not leak-proof**: near-analogs with different
  Murcko scaffolds can straddle partitions. Try `split.scaffold_type: generic`.
- **The 0.35 Tanimoto threshold is a heuristic.** Use `reports/ad_calibration.csv` to check whether
  error really rises below it for your target.
- **Test sets are small** for single-target data; look at the std across seeds, not just the mean.
- Hyper-parameters are fixed (no tuning); the validation split is used for model selection only.
- Single target per trained model; the ChEMBL query parameters were verified against mocked API
  responses in the unit tests. If a future ChEMBL release changes a filter name, `fetch()` also
  re-validates every record client-side, so results stay clean (but could be fewer than expected).

## Repository layout

```
configs/config.yaml          single source of truth for every stage
src/data/ingest_chembl.py    fetch → curate → featurize
src/data/scaffold_split.py   seeded Murcko / generic scaffold splitting
src/features/                physics (2D+3D), fingerprints, shared matrix builder
src/models/train.py          ablation, AD calibration, MLflow, artifact export
src/models/applicability.py  Tanimoto AD guard (pickle-free persistence)
src/models/reporting.py      tables & plots
src/api/                     FastAPI app + Pydantic v2 schemas
tests/                       22 tests incl. an end-to-end train→serve smoke test
Dockerfile, requirements*.txt, Makefile, .github/workflows/ci.yml
```

## Roadmap ideas

Hyper-parameter search with nested scaffold CV · quantile/conformal prediction intervals ·
multi-target models · SHAP explanations · batch endpoint · drift monitoring on incoming SMILES.

## License

MIT – see [LICENSE](LICENSE).
