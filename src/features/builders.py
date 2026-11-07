"""Single code path that turns a dataframe into a model matrix.

Training and serving both call build_matrix(), which removes train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fingerprints import smiles_to_counts
from src.features.physics import DESCRIPTOR_COLUMNS, FEATURES_2D

# feature_set -> (descriptor columns, include ECFP4 counts)
FEATURE_SETS = {
    "desc_2d": (FEATURES_2D, False),
    "desc_2d_3d": (DESCRIPTOR_COLUMNS, False),
    "ecfp4": ([], True),
    "ecfp4_desc_2d_3d": (DESCRIPTOR_COLUMNS, True),
}


def build_matrix(feature_set: str, df: pd.DataFrame, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Options: {sorted(FEATURE_SETS)}")
    cols, use_fp = FEATURE_SETS[feature_set]
    parts = []
    if use_fp:
        parts.append(smiles_to_counts(df["canonical_smiles"].tolist(), radius, nbits))
    if cols:
        parts.append(df[cols].to_numpy(dtype=np.float32))
    return np.hstack(parts)
