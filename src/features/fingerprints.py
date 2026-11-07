"""Morgan (ECFP) fingerprints as numpy matrices."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def _generator(radius: int, nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def smiles_to_bits(smiles: List[str], radius: int = 2, nbits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bit matrix float32 [n, nbits], valid mask [n]). Invalid SMILES -> zero row."""
    gen = _generator(radius, nbits)
    out = np.zeros((len(smiles), nbits), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        out[i] = gen.GetFingerprintAsNumPy(mol)
        valid[i] = True
    return out, valid


def smiles_to_counts(smiles: List[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Count-based ECFP (float32 [n, nbits]); counts keep more information than bits."""
    gen = _generator(radius, nbits)
    out = np.zeros((len(smiles), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            out[i] = gen.GetCountFingerprintAsNumPy(mol)
    return out
