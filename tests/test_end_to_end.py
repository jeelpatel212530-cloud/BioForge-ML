"""Smoke test: split -> ablation -> MLflow -> artifact export -> live API calls (tiny synthetic data)."""
import copy
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.models.train import run_training
from src.utils.config import load_config


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory, synthetic_features):
    root = tmp_path_factory.mktemp("bioforge")
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["paths"] = {k: str(root / k) for k in ("raw_dir", "processed_dir", "artifacts_dir", "reports_dir")}
    cfg["mlflow"].update(tracking_uri=f"sqlite:///{root / 'mlflow.db'}", artifact_root=str(root / "mlruns"))
    cfg["evaluation"]["seeds"] = [0, 1]
    cfg["models"]["gbr"].update(n_estimators=25)
    cfg["models"]["rf"].update(n_estimators=25, n_jobs=1)
    (root / "processed_dir").mkdir()
    synthetic_features.to_parquet(root / "processed_dir" / f"{cfg['target_chembl_id']}_physics_features.parquet", index=False)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    meta = run_training(cfg)
    return cfg, cfg_path, meta


def test_training_exports_complete_bundle(trained_env):
    cfg, _, meta = trained_env
    art, rep = Path(cfg["paths"]["artifacts_dir"]), Path(cfg["paths"]["reports_dir"])
    for name in ("model.joblib", "ad_reference.npz", "model_meta.json"):
        assert (art / name).exists()
    for name in ("ablation_results.csv", "ablation_summary.md", "ablation_rmse.png", "ad_calibration.csv", "ad_error_vs_similarity.png"):
        assert (rep / name).exists()
    assert meta["model_name"] != "baseline_mean"
    assert json.loads((art / "model_meta.json").read_text())["feature_set"] == meta["feature_set"]


def test_api_in_domain_vs_ood_and_batch(trained_env, monkeypatch):
    cfg, cfg_path, _ = trained_env
    monkeypatch.setenv("BIOFORGE_CONFIG", str(cfg_path))
    from src.api.main import app
    from src.models.applicability import ApplicabilityDomainGuard

    # a molecule that is guaranteed to be in the AD reference set (test partition is held out)
    in_ref = str(ApplicabilityDomainGuard.load(Path(cfg["paths"]["artifacts_dir"]) / "ad_reference.npz").smiles[0])

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "healthy" and health["model_loaded"]
        assert client.get("/model-info").json()["n_reference_molecules"] > 0

        ok = client.post("/predict", json={"smiles": in_ref})
        assert ok.status_code == 200
        body = ok.json()
        assert body["in_applicability_domain"] is True and body["max_tanimoto_similarity"] > 0.99
        assert body["estimated_ic50_nM"] > 0 and len(body["computed_physics_features"]) == 18
        assert body["warnings"] == []

        ood = client.post("/predict", json={"smiles": "CCCCCCCCCCCC"}).json()
        assert ood["in_applicability_domain"] is False and ood["confidence_tier"] == "LOW_OOD_WARNING"
        assert ood["warnings"]

        # Batch endpoint check (1 in-domain, 1 OOD, 1 invalid SMILES)
        batch_resp = client.post(
            "/predict/batch",
            json={"smiles": [in_ref, "CCCCCCCCCCCC", "not_a_valid_smiles"]},
        )
        assert batch_resp.status_code == 200
        batch_data = batch_resp.json()
        assert batch_data["n_total"] == 3
        assert batch_data["n_succeeded"] == 2
        assert batch_data["n_failed"] == 1
        assert batch_data["n_in_domain"] == 1
        assert len(batch_data["predictions"]) == 2
        assert len(batch_data["errors"]) == 1
        assert batch_data["errors"][0]["index"] == 2

        assert client.post("/predict", json={"smiles": "not_a_smiles"}).status_code == 422
        assert client.post("/predict", json={"smiles": ""}).status_code == 422
        assert client.post("/predict", json={"smiles": "C" * 500}).status_code == 422


def test_api_degrades_gracefully_without_artifacts(tmp_path, monkeypatch):
    cfg = load_config("configs/config.yaml")
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "missing")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("BIOFORGE_CONFIG", str(p))
    from src.api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json={"smiles": "c1ccccc1C"}).status_code == 503
