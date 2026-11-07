"""Shared fixtures: a small synthetic dataset covering 12 scaffolds (NOT real bioactivity data)."""
import numpy as np
import pandas as pd
import pytest

from src.features.physics import compute_features_with_reason

CORES = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1", "c1ncc2ccccc2n1",
    "C1CCCCC1", "C1CCNCC1", "c1ccsc1", "c1ccoc1", "c1ccc(-c2ccccc2)cc1", "c1cncnc1",
]
SUBS = ["C", "Cl", "F", "O", "N", "CC", "OC", "C#N"]


@pytest.fixture(scope="session")
def synthetic_features() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for ci, core in enumerate(CORES):
        for sub in SUBS:
            feats, _ = compute_features_with_reason(core + sub, n_conformers=3, min_heavy_atoms=5)
            if feats is None:
                continue
            feats["pIC50"] = 5.0 + 0.2 * ci + 0.05 * feats["heavy_atoms"] + rng.normal(0, 0.2)
            feats["molecule_chembl_id"] = f"SYN{len(rows)}"
            feats["pIC50_std"] = 0.0
            feats["n_measurements"] = 1
            feats["n_assays"] = 1
            rows.append(feats)
    return pd.DataFrame(rows)
