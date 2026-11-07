"""ChEMBL ingestion, curation and 3D physics featurization.

Stages: fetch (paginated REST, retries, server+client side filters) -> curate (standardize,
pIC50, median-aggregate duplicates, drop conflicting replicates) -> featurize (parallel).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from src.features.physics import (
    DESCRIPTOR_COLUMNS,
    compute_features_with_reason,
    standardize_smiles,
)
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk"
CHEMBL_API_URL = f"{CHEMBL_BASE}/chembl/api/data/activity.json"


# --------------------------------------------------------------------------- fetching
def parse_activity(act: dict, require_pchembl: bool = True, exclude_mutants: bool = True) -> Optional[dict]:
    """Client-side validation, so results stay clean even if a server-side filter is ignored."""
    smiles = act.get("canonical_smiles")
    value = act.get("standard_value")
    if not smiles or value is None:
        return None
    if act.get("standard_type") not in (None, "IC50"):
        return None
    if act.get("standard_units") not in (None, "nM"):
        return None
    if act.get("standard_relation") not in (None, "="):
        return None
    if act.get("data_validity_comment"):
        return None
    if act.get("potential_duplicate") in (1, True, "1", "true", "True"):
        return None
    if require_pchembl and act.get("pchembl_value") in (None, ""):
        return None
    if exclude_mutants and act.get("assay_variant_mutation"):
        return None
    try:
        ic50 = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ic50) or ic50 <= 0:
        return None
    return {
        "molecule_chembl_id": act.get("molecule_chembl_id"),
        "raw_smiles": smiles,
        "ic50_nM": ic50,
        "assay_chembl_id": act.get("assay_chembl_id"),
        "document_chembl_id": act.get("document_chembl_id"),
    }


def make_session(max_retries: int = 5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class ChEMBLDataIngestor:
    def __init__(self, target_chembl_id: str, chembl_cfg: dict, session: Optional[requests.Session] = None):
        self.target_id = target_chembl_id
        self.cfg = chembl_cfg
        self.session = session or make_session(chembl_cfg.get("max_retries", 5))

    def _params(self) -> dict:
        p = {
            "target_chembl_id": self.target_id,
            "standard_type": "IC50",
            "standard_units": "nM",
            "standard_relation": "=",
            "limit": int(self.cfg.get("page_size", 1000)),
        }
        if self.cfg.get("assay_type"):
            p["assay_type"] = self.cfg["assay_type"]
        if self.cfg.get("require_pchembl", True):
            p["pchembl_value__isnull"] = "false"
        p["data_validity_comment__isnull"] = "true"
        return p

    def fetch(self, max_records: Optional[int] = None) -> pd.DataFrame:
        """Follow ChEMBL's `page_meta.next` links until max_records valid rows are collected."""
        max_records = max_records if max_records is not None else self.cfg.get("max_records")
        logger.info("Fetching IC50 records for %s (max_records=%s)...", self.target_id, max_records)
        url, params = CHEMBL_API_URL, self._params()
        timeout = self.cfg.get("timeout_s", 60)
        records: List[dict] = []
        pages = scanned = 0
        while url:
            resp = self.session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            pages += 1
            for act in payload.get("activities", []):
                scanned += 1
                rec = parse_activity(
                    act,
                    require_pchembl=self.cfg.get("require_pchembl", True),
                    exclude_mutants=self.cfg.get("exclude_mutants", True),
                )
                if rec:
                    records.append(rec)
            if max_records and len(records) >= max_records:
                records = records[:max_records]
                break
            nxt = (payload.get("page_meta") or {}).get("next")
            url, params = (urljoin(CHEMBL_BASE, nxt) if nxt else None), None
        logger.info("Fetched %d pages, scanned %d activities, kept %d valid records.", pages, scanned, len(records))
        return pd.DataFrame(records)


# --------------------------------------------------------------------------- curation
def curate(raw: pd.DataFrame, curation_cfg: dict) -> pd.DataFrame:
    """Standardize structures, convert to pIC50, aggregate duplicates by median."""
    if raw.empty:
        raise ValueError("No raw records to curate.")
    min_ha, max_ha = curation_cfg["min_heavy_atoms"], curation_cfg["max_heavy_atoms"]
    cache: Dict[str, Optional[str]] = {
        s: standardize_smiles(s, min_ha, max_ha) for s in raw["raw_smiles"].unique()
    }
    df = raw.copy()
    df["canonical_smiles"] = df["raw_smiles"].map(cache)
    n0 = len(df)
    df = df.dropna(subset=["canonical_smiles"])
    logger.info("Structure standardization kept %d / %d records.", len(df), n0)

    df["pIC50"] = 9.0 - np.log10(df["ic50_nM"].astype(float))
    agg = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pIC50=("pIC50", "median"),
            pIC50_std=("pIC50", lambda x: float(np.std(x, ddof=0))),
            n_measurements=("pIC50", "size"),
            n_assays=("assay_chembl_id", "nunique"),
        )
    )
    max_std = curation_cfg.get("max_replicate_std")
    if max_std is not None:
        n_before = len(agg)
        agg = agg[agg["pIC50_std"] <= max_std]
        logger.info("Dropped %d molecules with conflicting replicates (std > %.2f).", n_before - len(agg), max_std)
    return agg.reset_index(drop=True)


# --------------------------------------------------------------------------- featurization
def _featurize_one(smiles: str, physics_kwargs: dict) -> Optional[dict]:
    feats, _ = compute_features_with_reason(smiles, **physics_kwargs)
    return feats


def physics_kwargs_from_cfg(cfg: dict) -> dict:
    p, c = cfg["physics"], cfg["curation"]
    return {
        "n_conformers": p["n_conformers"],
        "mmff_max_iters": p["mmff_max_iters"],
        "prune_rms": p["prune_rms"],
        "seed": p["random_seed"],
        "temp_k": p["boltzmann_temp_k"],
        "min_heavy_atoms": c["min_heavy_atoms"],
        "max_heavy_atoms": c["max_heavy_atoms"],
    }


def featurize(curated: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    kwargs = physics_kwargs_from_cfg(cfg)
    n_jobs = cfg["physics"].get("n_jobs", -1)
    n_jobs = (os.cpu_count() or 1) if n_jobs in (-1, None) else int(n_jobs)
    smiles = curated["canonical_smiles"].tolist()
    worker = partial(_featurize_one, physics_kwargs=kwargs)
    logger.info("Computing 2D + 3D conformer-ensemble features for %d molecules (%d workers)...", len(smiles), n_jobs)
    if n_jobs == 1:
        results = [worker(s) for s in tqdm(smiles)]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(tqdm(ex.map(worker, smiles, chunksize=8), total=len(smiles)))
    rows = []
    for (_, base), feats in zip(curated.iterrows(), results):
        if feats is None:
            continue
        row = base.to_dict()
        row.update({k: v for k, v in feats.items() if k != "canonical_smiles"})
        rows.append(row)
    out = pd.DataFrame(rows)
    logger.info("Featurization succeeded for %d / %d molecules.", len(out), len(curated))
    if out.empty:
        raise ValueError("Featurization failed for every molecule.")
    return out


# --------------------------------------------------------------------------- entry point
def run_ingestion(config_path=None, refresh: bool = False) -> Path:
    cfg = load_config(config_path)
    target = cfg["target_chembl_id"]
    raw_dir, processed_dir = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{target}_ic50_raw.parquet"
    if raw_path.exists() and not refresh:
        logger.info("Using cached raw data %s (pass --refresh to re-download).", raw_path)
        raw = pd.read_parquet(raw_path)
    else:
        raw = ChEMBLDataIngestor(target, cfg["chembl"]).fetch()
        raw.to_parquet(raw_path, index=False)

    curated = curate(raw, cfg["curation"])
    featurized = featurize(curated, cfg)
    featurized = featurized[
        ["molecule_chembl_id", "canonical_smiles", "pIC50", "pIC50_std", "n_measurements", "n_assays"]
        + DESCRIPTOR_COLUMNS
        + ["n_conformers_used"]
    ]
    out_path = processed_dir / f"{target}_physics_features.parquet"
    featurized.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d molecules).", out_path, len(featurized))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest ChEMBL IC50 data and compute 2D/3D features.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download even if a raw cache exists")
    args = ap.parse_args()
    run_ingestion(args.config, args.refresh)
