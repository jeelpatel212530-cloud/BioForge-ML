"""FastAPI service: physics featurization + prediction + applicability-domain guardrail.

The service loads a self-contained bundle (model.joblib, ad_reference.npz, model_meta.json) from
artifacts/, so it needs neither the MLflow database nor the training data at runtime.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchPredictionError,
    BatchPredictionRequest,
    BatchPredictionResponse,
    MoleculePredictionRequest,
    MoleculePredictionResponse,
)
from src.features.builders import build_matrix
from src.features.physics import DESCRIPTOR_COLUMNS, compute_features_with_reason
from src.models.applicability import ApplicabilityDomainGuard
from src.utils.config import load_config

logger = logging.getLogger("bioforge.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.clear()
    try:
        cfg = load_config()
        art = Path(cfg["paths"]["artifacts_dir"])
        STATE["model"] = joblib.load(art / "model.joblib")
        STATE["ad_guard"] = ApplicabilityDomainGuard.load(art / "ad_reference.npz")
        STATE["meta"] = json.loads((art / "model_meta.json").read_text())
        STATE["max_smiles_length"] = int(cfg["serving"].get("max_smiles_length", 400))
        logger.info("Loaded model '%s'.", STATE["meta"]["model_name"])
    except Exception as exc:  # start anyway so /health can report what is wrong
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Could not load serving bundle. Run `make train` first.")
    yield
    STATE.clear()


app = FastAPI(
    title="BioForge-ML Inference API",
    description="Conformer-ensemble QSAR with a Tanimoto applicability-domain guardrail.",
    version="2.1.0",
    lifespan=lifespan,
)


def _require_ready() -> None:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail=f"Model not loaded. {STATE.get('error', '')}".strip())


def _predict_with_uncertainty(pipeline, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (predictions [n], tree_std [n] or None). Computes ensemble std for RandomForest."""
    preds = np.asarray(pipeline.predict(X), dtype=float)
    try:
        imputer = pipeline.named_steps.get("imputer")
        reg = pipeline.named_steps.get("model")
        X_imp = imputer.transform(X) if imputer is not None else X
        if hasattr(reg, "estimators_") and not hasattr(reg, "learning_rate"):
            # RandomForestRegressor: each estimator in estimators_ is a full tree prediction
            tree_preds = np.column_stack([tree.predict(X_imp) for tree in reg.estimators_])
            return preds, np.std(tree_preds, axis=1, ddof=0)
    except Exception:
        pass
    return preds, None


def _build_response(
    query_smiles: str,
    feats: dict,
    pred: float,
    uncertainty: Optional[float],
    ad: dict,
    meta: dict,
) -> MoleculePredictionResponse:
    warnings = []
    if not ad["in_applicability_domain"]:
        warnings.append(
            "Query is structurally distant from the training set; treat this prediction as unreliable."
        )
    return MoleculePredictionResponse(
        query_smiles=query_smiles,
        canonical_smiles=str(feats["canonical_smiles"]),
        predicted_pIC50=round(float(pred), 4),
        estimated_ic50_nM=round(float(10 ** (9.0 - float(pred))), 4),
        prediction_uncertainty_pIC50=round(float(uncertainty), 4) if uncertainty is not None else None,
        in_applicability_domain=bool(ad["in_applicability_domain"]),
        max_tanimoto_similarity=float(ad["max_tanimoto_similarity"]),
        knn_mean_tanimoto=float(ad["knn_mean_tanimoto"]),
        confidence_tier=str(ad["confidence_tier"]),
        nearest_training_smiles=ad["nearest_training_smiles"],
        model_name=meta["model_name"],
        feature_set=meta["feature_set"],
        computed_physics_features={k: float(feats[k]) for k in DESCRIPTOR_COLUMNS},
        warnings=warnings,
    )


@app.get("/health")
def health_check():
    meta = STATE.get("meta", {})
    return {
        "status": "healthy" if "model" in STATE else "degraded",
        "model_loaded": "model" in STATE,
        "model_name": meta.get("model_name"),
        "target_chembl_id": meta.get("target_chembl_id"),
        "error": STATE.get("error"),
    }


@app.get("/model-info")
def model_info():
    _require_ready()
    meta = dict(STATE["meta"])
    meta["n_reference_molecules"] = STATE["ad_guard"].n_reference
    return meta


@app.post("/predict", response_model=MoleculePredictionResponse)
def predict_bioactivity(payload: MoleculePredictionRequest):
    _require_ready()
    max_len = STATE.get("max_smiles_length", 400)
    if len(payload.smiles) > max_len:
        raise HTTPException(
            status_code=422,
            detail=f"SMILES length ({len(payload.smiles)}) exceeds configured max_smiles_length ({max_len}).",
        )

    meta, guard = STATE["meta"], STATE["ad_guard"]
    feats, reason = compute_features_with_reason(payload.smiles, **meta["physics"])
    if feats is None:
        raise HTTPException(status_code=422, detail=reason or "Featurization failed.")

    row = pd.DataFrame([feats])
    fp = meta["fingerprint"]
    X = build_matrix(meta["feature_set"], row, fp["radius"], fp["nbits"])
    preds, stds = _predict_with_uncertainty(STATE["model"], X)
    ad = guard.evaluate_smiles(str(feats["canonical_smiles"]))

    return _build_response(
        query_smiles=payload.smiles,
        feats=feats,
        pred=float(preds[0]),
        uncertainty=float(stds[0]) if stds is not None else None,
        ad=ad,
        meta=meta,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_bioactivity_batch(payload: BatchPredictionRequest):
    _require_ready()
    max_len = STATE.get("max_smiles_length", 400)
    meta, guard = STATE["meta"], STATE["ad_guard"]

    valid_queries = []
    valid_feats = []
    errors = []

    for idx, raw_smi in enumerate(payload.smiles):
        smi = raw_smi.strip() if isinstance(raw_smi, str) else ""
        if not smi:
            errors.append(BatchPredictionError(index=idx, query_smiles=str(raw_smi), detail="Empty SMILES string."))
            continue
        if len(smi) > max_len:
            errors.append(
                BatchPredictionError(
                    index=idx,
                    query_smiles=smi,
                    detail=f"SMILES length ({len(smi)}) exceeds configured max_smiles_length ({max_len}).",
                )
            )
            continue
        feats, reason = compute_features_with_reason(smi, **meta["physics"])
        if feats is None:
            errors.append(
                BatchPredictionError(index=idx, query_smiles=smi, detail=reason or "Featurization failed.")
            )
            continue
        valid_queries.append((idx, smi))
        valid_feats.append(feats)

    predictions = []
    if valid_feats:
        df_batch = pd.DataFrame(valid_feats)
        fp = meta["fingerprint"]
        X_batch = build_matrix(meta["feature_set"], df_batch, fp["radius"], fp["nbits"])
        preds, stds = _predict_with_uncertainty(STATE["model"], X_batch)
        ad_list = guard.evaluate_batch(df_batch["canonical_smiles"].tolist())

        for i, ((_, q_smi), feats, pred, ad) in enumerate(zip(valid_queries, valid_feats, preds, ad_list)):
            unc = float(stds[i]) if stds is not None else None
            predictions.append(_build_response(q_smi, feats, float(pred), unc, ad, meta))

    n_in_domain = sum(1 for p in predictions if p.in_applicability_domain)
    return BatchPredictionResponse(
        n_total=len(payload.smiles),
        n_succeeded=len(predictions),
        n_failed=len(errors),
        n_in_domain=n_in_domain,
        predictions=predictions,
        errors=errors,
    )
