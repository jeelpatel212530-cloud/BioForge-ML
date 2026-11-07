"""Training, ablation study, AD calibration, MLflow tracking and serving-bundle export.

What this script does
  1. Loads the featurized dataset.
  2. Runs a feature-set ablation over several seeded scaffold splits (mean ± std):
       baseline_mean | gbr_desc_2d | gbr_desc_2d_3d | rf_ecfp4 | rf_ecfp4_desc_2d_3d
     This answers "do the 3D physics features actually help?" with data instead of assertion.
  3. Selects the serving model (best mean validation RMSE, or serving.model from the config).
  4. Calibrates the applicability domain: does low Tmax actually predict high error?
  5. Refits the chosen model (train+val), exports artifacts/ (model.joblib, ad_reference.npz,
     model_meta.json) and logs/registers everything in MLflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import rdkit
import sklearn
from mlflow.tracking import MlflowClient
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.scaffold_split import add_scaffolds, assign_scaffold_splits
from src.features.builders import FEATURE_SETS, build_matrix
from src.features.physics import DESCRIPTOR_COLUMNS
from src.data.ingest_chembl import physics_kwargs_from_cfg
from src.models.applicability import ApplicabilityDomainGuard
from src.models.reporting import (
    ad_calibration_table,
    ad_error_spearman,
    plot_ablation,
    plot_ad_error,
    summarize_results,
    write_summary_markdown,
)
from src.utils.config import load_config

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# name -> (feature_set, regressor kind)
CANDIDATES: Dict[str, Tuple[str, str]] = {
    "baseline_mean": ("desc_2d", "dummy"),
    "gbr_desc_2d": ("desc_2d", "gbr"),
    "gbr_desc_2d_3d": ("desc_2d_3d", "gbr"),
    "rf_ecfp4": ("ecfp4", "rf"),
    "rf_ecfp4_desc_2d_3d": ("ecfp4_desc_2d_3d", "rf"),
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        rho = 0.0
    else:
        rho = float(spearmanr(y_true, y_pred)[0])
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": rho,
    }


def make_pipeline(kind: str, cfg: dict) -> Pipeline:
    if kind == "dummy":
        reg = DummyRegressor(strategy="mean")
    elif kind == "gbr":
        reg = GradientBoostingRegressor(**cfg["models"]["gbr"])
    elif kind == "rf":
        reg = RandomForestRegressor(**cfg["models"]["rf"])
    else:
        raise ValueError(f"Unknown regressor kind '{kind}'")
    # trees are scale-invariant, so no scaler; the imputer is a safety net for missing values
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])


def _indices(split_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    idx = {s: np.where(split_df["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    for s, v in idx.items():
        if len(v) == 0:
            raise ValueError(f"The '{s}' partition is empty; the dataset is too small or too clustered to split.")
    return idx


def _setup_mlflow(cfg: dict) -> None:
    mc = cfg["mlflow"]
    mlflow.set_tracking_uri(mc["tracking_uri"])
    if mlflow.get_experiment_by_name(mc["experiment_name"]) is None:
        root = Path(mc["artifact_root"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(mc["experiment_name"], artifact_location=root.as_uri())
    mlflow.set_experiment(mc["experiment_name"])


def _log_model(pipeline: Pipeline, name: str):
    """Log + register the model. cloudpickle format: this is our own, trusted artifact, and
    MLflow >= 3's default skops format rejects scikit-learn tree ensembles as 'untrusted types'."""
    fmt = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    try:  # MLflow >= 3
        return mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=name, serialization_format=fmt)
    except TypeError:  # MLflow 2.x
        return mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=name, serialization_format=fmt)


def run_training(cfg_or_path=None) -> Dict[str, object]:
    cfg = cfg_or_path if isinstance(cfg_or_path, dict) else load_config(cfg_or_path)
    target = cfg["target_chembl_id"]
    paths = cfg["paths"]
    reports_dir, art_dir = Path(paths["reports_dir"]), Path(paths["artifacts_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(Path(paths["processed_dir"]) / f"{target}_physics_features.parquet").reset_index(drop=True)
    sc, fp = cfg["split"], cfg["fingerprint"]
    df = add_scaffolds(df, sc["scaffold_type"])
    y = df["pIC50"].to_numpy(dtype=float)
    logger.info("Loaded %d molecules, %d unique scaffolds.", len(df), df["scaffold_smiles"].nunique())

    needed = sorted({fs for fs, _ in CANDIDATES.values()})
    matrices = {fs: build_matrix(fs, df, fp["radius"], fp["nbits"]) for fs in needed}

    _setup_mlflow(cfg)
    with mlflow.start_run(run_name="ablation_and_serving_model") as run:
        mlflow.log_params(
            {
                "target_chembl_id": target,
                "n_molecules": len(df),
                "n_scaffolds": int(df["scaffold_smiles"].nunique()),
                "scaffold_type": sc["scaffold_type"],
                "primary_split_seed": sc["seed"],
                "evaluation_seeds": str(cfg["evaluation"]["seeds"]),
                "n_conformers": cfg["physics"]["n_conformers"],
                **{f"gbr_{k}": v for k, v in cfg["models"]["gbr"].items()},
                **{f"rf_{k}": v for k, v in cfg["models"]["rf"].items()},
            }
        )

        # ---------------------------------------------------------------- 1. ablation
        rows = []
        for seed in cfg["evaluation"]["seeds"]:
            split_df = assign_scaffold_splits(df, sc["train_frac"], sc["val_frac"], seed=seed, scaffold_type=sc["scaffold_type"])
            idx = _indices(split_df)
            for name, (fs, kind) in CANDIDATES.items():
                pipe = make_pipeline(kind, cfg)
                X = matrices[fs]
                pipe.fit(X[idx["train"]], y[idx["train"]])
                for split in ("val", "test"):
                    m = compute_metrics(y[idx[split]], pipe.predict(X[idx[split]]))
                    rows.append({"seed": seed, "model": name, "split": split, **m})
            logger.info("Finished ablation for split seed %s", seed)
        results = pd.DataFrame(rows)
        summary = summarize_results(results)
        results.to_csv(reports_dir / "ablation_results.csv", index=False)
        summary.to_csv(reports_dir / "ablation_summary.csv", index=False)

        for _, r in summary.iterrows():
            for metric in ("rmse", "mae", "r2", "spearman"):
                mlflow.log_metric(f"{r['model']}.{r['split']}.{metric}_mean", float(r[f"{metric}_mean"]))
                if not np.isnan(r[f"{metric}_std"]):
                    mlflow.log_metric(f"{r['model']}.{r['split']}.{metric}_std", float(r[f"{metric}_std"]))

        # paired effect of the 3D features: same seeds, same model, only the feature set differs
        test = results[results["split"] == "test"].pivot(index="seed", columns="model", values="rmse")
        delta = test["gbr_desc_2d_3d"] - test["gbr_desc_2d"]
        notes = (
            f"**Effect of 3D features** (GBR, test RMSE, `desc_2d_3d` minus `desc_2d`, per seed): "
            f"mean {delta.mean():+.3f} ± {delta.std(ddof=0):.3f} pIC50 units "
            f"(negative = 3D helps; helps in {(delta < 0).sum()}/{len(delta)} splits)."
        )
        mlflow.log_metric("delta_test_rmse_3d_vs_2d_mean", float(delta.mean()))
        plot_ablation(summary, reports_dir / "ablation_rmse.png")

        # ---------------------------------------------------------------- 2. pick serving model
        choice = cfg["serving"]["model"]
        val = summary[(summary["split"] == "val") & (summary["model"] != "baseline_mean")].set_index("model")
        chosen = val["rmse_mean"].idxmin() if choice == "auto" else choice
        if chosen not in CANDIDATES or chosen == "baseline_mean":
            raise ValueError(f"serving.model must be 'auto' or one of {[c for c in CANDIDATES if c != 'baseline_mean']}")
        feature_set, kind = CANDIDATES[chosen]
        logger.info("Serving model: %s (feature_set=%s)", chosen, feature_set)

        test_rmse = summary[summary["split"] == "test"].set_index("model")["rmse_mean"]
        beats_baseline = bool(test_rmse[chosen] < test_rmse["baseline_mean"])
        mlflow.log_metric("selected_beats_mean_baseline", float(beats_baseline))
        notes += (
            f"\n\n**Sanity check vs. predicting the mean:** `{chosen}` test RMSE {test_rmse[chosen]:.3f} "
            f"vs baseline {test_rmse['baseline_mean']:.3f} -> "
            + ("beats the baseline." if beats_baseline else "DOES NOT beat the baseline; do not trust this model.")
        )
        if not beats_baseline:
            logger.warning("Selected model does NOT beat the mean baseline on scaffold-split test data.")
        logger.info(notes)
        write_summary_markdown(summary, reports_dir / "ablation_summary.md", notes)

        # ---------------------------------------------------------------- 3. AD calibration (train-only fit)
        ad_cfg = cfg["applicability_domain"]
        primary = assign_scaffold_splits(df, sc["train_frac"], sc["val_frac"], seed=sc["seed"], scaffold_type=sc["scaffold_type"])
        idx = _indices(primary)
        X = matrices[feature_set]
        eval_pipe = make_pipeline(kind, cfg).fit(X[idx["train"]], y[idx["train"]])
        test_pred = eval_pipe.predict(X[idx["test"]])
        abs_err = np.abs(test_pred - y[idx["test"]])
        guard_eval = ApplicabilityDomainGuard(
            fp["radius"], fp["nbits"], ad_cfg["tanimoto_threshold"], ad_cfg["high_confidence_threshold"], ad_cfg["top_k_neighbors"]
        ).fit(df.loc[idx["train"], "canonical_smiles"].tolist())
        tmax = guard_eval.max_similarity(df.loc[idx["test"], "canonical_smiles"].tolist())
        cal = ad_calibration_table(tmax, abs_err)
        cal.to_csv(reports_dir / "ad_calibration.csv", index=False)
        plot_ad_error(tmax, abs_err, ad_cfg["tanimoto_threshold"], reports_dir / "ad_error_vs_similarity.png")
        rho = ad_error_spearman(tmax, abs_err)
        if rho is not None:
            mlflow.log_metric("ad_spearman_tmax_vs_abs_error", rho)
        logger.info("AD check: Spearman(Tmax, |error|) = %s (negative = AD is informative)", rho)

        eval_metrics = {s: compute_metrics(y[idx[s]], eval_pipe.predict(X[idx[s]])) for s in ("val", "test")}
        for s, m in eval_metrics.items():
            for k, v in m.items():
                mlflow.log_metric(f"selected.{s}_{k}", v)

        # ---------------------------------------------------------------- 4. final fit + export
        if cfg["serving"]["refit_on_train_val"]:
            fit_idx = np.concatenate([idx["train"], idx["val"]])
        else:
            fit_idx = idx["train"]
        final_pipe = make_pipeline(kind, cfg).fit(X[fit_idx], y[fit_idx])
        guard = ApplicabilityDomainGuard(
            fp["radius"], fp["nbits"], ad_cfg["tanimoto_threshold"], ad_cfg["high_confidence_threshold"], ad_cfg["top_k_neighbors"]
        ).fit(df.loc[fit_idx, "canonical_smiles"].tolist())

        model_path, ad_path, meta_path = art_dir / "model.joblib", art_dir / "ad_reference.npz", art_dir / "model_meta.json"
        joblib.dump(final_pipe, model_path)
        guard.save(ad_path)

        importances = getattr(final_pipe.named_steps["model"], "feature_importances_", None)
        top_features = None
        if importances is not None and feature_set in ("desc_2d", "desc_2d_3d"):
            names = FEATURE_SETS[feature_set][0]
            top_features = dict(sorted(zip(names, map(float, importances)), key=lambda kv: -kv[1]))

        meta = {
            "model_name": chosen,
            "feature_set": feature_set,
            "target_chembl_id": target,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train_rows": int(len(fit_idx)),
            "n_test_rows_held_out": int(len(idx["test"])),
            "split_seed": sc["seed"],
            "scaffold_type": sc["scaffold_type"],
            "metrics_primary_split": eval_metrics,
            "beats_mean_baseline_on_test": beats_baseline,
            "ablation_test_rmse_mean": {
                r["model"]: round(float(r["rmse_mean"]), 4) for _, r in summary[summary["split"] == "test"].iterrows()
            },
            "feature_importances": top_features,
            "fingerprint": {"radius": fp["radius"], "nbits": fp["nbits"]},
            "physics": physics_kwargs_from_cfg(cfg),
            "applicability_domain": {
                "tanimoto_threshold": ad_cfg["tanimoto_threshold"],
                "high_confidence_threshold": ad_cfg["high_confidence_threshold"],
                "spearman_tmax_vs_abs_error": rho,
            },
            "descriptor_columns": DESCRIPTOR_COLUMNS,
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "rdkit": rdkit.__version__,
                "mlflow": mlflow.__version__,
            },
            "mlflow_run_id": run.info.run_id,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # ---------------------------------------------------------------- 5. MLflow registry
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        for p in (ad_path, meta_path):
            mlflow.log_artifact(str(p), artifact_path="serving_bundle")
        _log_model(final_pipe, cfg["mlflow"]["registered_model_name"])
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg['mlflow']['registered_model_name']}'")
        latest = max(int(v.version) for v in versions)
        client.set_registered_model_alias(cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["alias"], str(latest))
        logger.info("Registered %s v%s with alias '%s'.", cfg["mlflow"]["registered_model_name"], latest, cfg["mlflow"]["alias"])

    logger.info("Serving bundle written to %s", art_dir)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run ablation, train the serving model, export artifacts, log to MLflow.")
    ap.add_argument("--config", default=None)
    run_training(ap.parse_args().config)
