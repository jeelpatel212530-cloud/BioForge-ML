"""Tanimoto applicability-domain (AD) guard.

Reference fingerprints are stored as packed bits in a .npz file (no pickle -> safe to load and
stable across library versions). Similarity is computed vectorised with matrix products.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.features.fingerprints import smiles_to_bits


class ApplicabilityDomainGuard:
    def __init__(
        self,
        radius: int = 2,
        nbits: int = 2048,
        threshold: float = 0.35,
        high_threshold: float = 0.50,
        top_k: int = 5,
    ):
        self.radius = radius
        self.nbits = nbits
        self.threshold = threshold
        self.high_threshold = high_threshold
        self.top_k = top_k
        self.smiles: np.ndarray = np.array([], dtype=str)
        self._bits: Optional[np.ndarray] = None   # [n_ref, nbits] float32
        self._counts: Optional[np.ndarray] = None  # [n_ref] bits set per reference

    @property
    def n_reference(self) -> int:
        return 0 if self._bits is None else int(self._bits.shape[0])

    def fit(self, train_smiles: List[str]) -> "ApplicabilityDomainGuard":
        bits, valid = smiles_to_bits(list(train_smiles), self.radius, self.nbits)
        self._bits = bits[valid]
        self._counts = self._bits.sum(axis=1)
        self.smiles = np.array([s for s, v in zip(train_smiles, valid) if v], dtype=str)
        return self

    # ------------------------------------------------------------------ similarity
    def similarity_matrix(self, query_smiles: List[str]):
        """Return (sim [m, n_ref], valid mask [m]). Rows for invalid SMILES are zero."""
        if self._bits is None or self.n_reference == 0:
            raise RuntimeError("ApplicabilityDomainGuard has no reference set; call fit() or load().")
        q, valid = smiles_to_bits(list(query_smiles), self.radius, self.nbits)
        inter = q @ self._bits.T
        union = q.sum(axis=1)[:, None] + self._counts[None, :] - inter
        sim = inter / np.maximum(union, 1.0)
        return sim, valid

    def max_similarity(self, query_smiles: List[str]) -> np.ndarray:
        """Tmax per query (NaN for unparseable SMILES)."""
        sim, valid = self.similarity_matrix(query_smiles)
        out = sim.max(axis=1)
        out[~valid] = np.nan
        return out

    # ------------------------------------------------------------------ single & batch evaluation
    def evaluate_batch(self, query_smiles: List[str]) -> List[Dict[str, object]]:
        """Vectorised applicability-domain evaluation for a batch of SMILES."""
        if not query_smiles:
            return []
        if self.n_reference == 0:
            return [self._invalid() for _ in query_smiles]
        sim, valid = self.similarity_matrix(query_smiles)
        k = min(self.top_k, sim.shape[1])
        out: List[Dict[str, object]] = []
        for i, is_valid in enumerate(valid):
            if not is_valid:
                out.append(self._invalid())
                continue
            row = sim[i]
            top = np.sort(np.partition(row, -k)[-k:])[::-1]
            max_sim = float(top[0])
            in_ad = bool(max_sim >= self.threshold)
            if not in_ad:
                tier = "LOW_OOD_WARNING"
            elif max_sim >= self.high_threshold:
                tier = "HIGH_IN_DOMAIN"
            else:
                tier = "MODERATE_IN_DOMAIN"
            out.append(
                {
                    "max_tanimoto_similarity": round(max_sim, 4),
                    "knn_mean_tanimoto": round(float(top.mean()), 4),
                    "in_applicability_domain": in_ad,
                    "confidence_tier": tier,
                    "nearest_training_smiles": str(self.smiles[int(row.argmax())]),
                }
            )
        return out

    def evaluate_smiles(self, query_smiles: str) -> Dict[str, object]:
        return self.evaluate_batch([query_smiles])[0]

    @staticmethod
    def _invalid() -> Dict[str, object]:
        return {
            "max_tanimoto_similarity": 0.0,
            "knn_mean_tanimoto": 0.0,
            "in_applicability_domain": False,
            "confidence_tier": "INVALID_OR_UNINITIALIZED",
            "nearest_training_smiles": None,
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(self._bits.astype(np.uint8), axis=1)
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                packed=packed,
                smiles=self.smiles,
                radius=self.radius,
                nbits=self.nbits,
                threshold=self.threshold,
                high_threshold=self.high_threshold,
                top_k=self.top_k,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ApplicabilityDomainGuard":
        data = np.load(Path(path), allow_pickle=False)
        guard = cls(
            radius=int(data["radius"]),
            nbits=int(data["nbits"]),
            threshold=float(data["threshold"]),
            high_threshold=float(data["high_threshold"]),
            top_k=int(data["top_k"]),
        )
        guard._bits = np.unpackbits(data["packed"], axis=1, count=guard.nbits).astype(np.float32)
        guard._counts = guard._bits.sum(axis=1)
        guard.smiles = data["smiles"]
        return guard
