"""Seeded, balanced scaffold splitting (Bemis-Murcko or generic scaffolds).

Whole scaffold groups go to exactly one partition, so no scaffold is shared between
train/val/test. Groups too large to fit val/test are placed into train first (largest first); the remaining
groups are shuffled with the seed, which gives different-but-reproducible splits per seed.

Caveat: Murcko scaffolds still leave near-analogs on both sides (e.g. one ring atom different).
Use scaffold_type: "generic" in the config for a stricter split.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.utils.config import load_config

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_scaffold(smiles: str, scaffold_type: str = "murcko") -> str:
    """Return scaffold SMILES; 'ACYCLIC' for ring-free molecules, 'INVALID' if unparseable."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return "INVALID"
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_type == "generic":
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        smi = Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return "INVALID"
    return smi if smi else "ACYCLIC"


def add_scaffolds(df: pd.DataFrame, scaffold_type: str = "murcko") -> pd.DataFrame:
    if scaffold_type not in ("murcko", "generic"):
        raise ValueError("scaffold_type must be 'murcko' or 'generic'")
    df = df.copy()
    df["scaffold_smiles"] = df["canonical_smiles"].map(lambda s: extract_scaffold(s, scaffold_type))
    return df


def assign_scaffold_splits(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 0,
    scaffold_type: str = "murcko",
) -> pd.DataFrame:
    """Return a copy of df with 'scaffold_smiles' and 'split' (train/val/test) columns."""
    if "scaffold_smiles" not in df.columns:
        df = add_scaffolds(df, scaffold_type)
    else:
        df = df.copy()
    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, scaf in enumerate(df["scaffold_smiles"].tolist()):
        groups[scaf].append(i)
    clusters = list(groups.values())

    # Clusters too large to ever fit into val or test must live in train: place them first,
    # largest first. Everything else is shuffled with the seed -> different splits per seed.
    cap = max(n_val, n_test)
    oversized = sorted((c for c in clusters if len(c) > cap), key=len, reverse=True)
    rest = [c for c in clusters if len(c) <= cap]
    rng = np.random.RandomState(seed)
    rng.shuffle(rest)
    ordered = oversized + rest

    split = np.empty(n, dtype=object)
    n_tr = n_va = 0
    for cluster in ordered:
        if n_tr + len(cluster) <= n_train:
            split[cluster] = "train"
            n_tr += len(cluster)
        elif n_va + len(cluster) <= n_val:
            split[cluster] = "val"
            n_va += len(cluster)
        else:
            split[cluster] = "test"
    df["split"] = split
    check_no_scaffold_leakage(df)
    return df


def check_no_scaffold_leakage(df: pd.DataFrame) -> None:
    sets = {s: set(df.loc[df["split"] == s, "scaffold_smiles"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"Scaffold leakage between {a} and {b}: {len(overlap)} shared scaffolds")


def run_split(config_path=None) -> Path:
    cfg = load_config(config_path)
    target = cfg["target_chembl_id"]
    processed = Path(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(processed / f"{target}_physics_features.parquet")
    sc = cfg["split"]
    out = assign_scaffold_splits(
        df, sc["train_frac"], sc["val_frac"], seed=sc["seed"], scaffold_type=sc["scaffold_type"]
    )
    out_path = processed / f"{target}_scaffold_split.parquet"
    out.to_parquet(out_path, index=False)
    counts = out["split"].value_counts().to_dict()
    n_scaf = out.groupby("split")["scaffold_smiles"].nunique().to_dict()
    logger.info("Split sizes: %s | scaffolds per split: %s | wrote %s", counts, n_scaf, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create the primary scaffold split table (for inspection).")
    ap.add_argument("--config", default=None)
    run_split(ap.parse_args().config)
