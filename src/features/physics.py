"""2D descriptors + 3D conformer-ensemble physics features.

Pipeline per molecule:
  1. keep the largest fragment (strips counter-ions/solvents) and neutralise charges
  2. embed an ensemble of conformers with ETKDGv3
  3. relax every conformer with MMFF94
  4. compute shape descriptors per conformer and Boltzmann-average them

Why an ensemble: many kinase inhibitors are flexible, so a single conformer gives noisy shape
descriptors. Averaging over a Boltzmann-weighted ensemble is far more stable.

Honest caveat: MMFF94 energies are gas-phase, size-dependent numbers. They are *not* binding
energies; the model decides whether they carry any signal (see the ablation in train.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

FEATURES_2D = [
    "mol_wt",
    "logp",
    "tpsa",
    "num_hbd",
    "num_hba",
    "rotatable_bonds",
    "heavy_atoms",
    "aromatic_rings",
    "fraction_csp3",
]

FEATURES_3D = [
    "mmff94_energy_kcal_mol",          # lowest-energy conformer
    "mmff94_energy_spread_kcal_mol",   # energy range over the ensemble (flexibility proxy)
    "radius_of_gyration_A",            # Boltzmann-averaged
    "rg_std_A",                        # Boltzmann-weighted std of Rg across conformers (shape flexibility)
    "asphericity",
    "eccentricity",
    "npr1",                            # normalised principal moment ratios (size-independent shape)
    "npr2",
    "principal_moment_inertia_1",
]

DESCRIPTOR_COLUMNS = FEATURES_2D + FEATURES_3D

KB_KCAL_PER_MOL_K = 0.0019872041
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(
    smiles: str, min_heavy_atoms: int = 5, max_heavy_atoms: int = 80
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Parse, keep the largest fragment, neutralise. Returns (mol, None) or (None, reason)."""
    if not smiles or not isinstance(smiles, str):
        return None, "Empty or non-string SMILES."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Could not parse SMILES."
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None, "No atoms found."
    mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Descriptors.MolWt(m)))
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:  # keep the charged form rather than failing
        pass
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < min_heavy_atoms:
        return None, f"Molecule too small ({n_heavy} heavy atoms; minimum {min_heavy_atoms})."
    if n_heavy > max_heavy_atoms:
        return None, f"Molecule too large ({n_heavy} heavy atoms; maximum {max_heavy_atoms})."
    return mol, None


def standardize_smiles(
    smiles: str, min_heavy_atoms: int = 5, max_heavy_atoms: int = 80
) -> Optional[str]:
    mol, _ = standardize_mol(smiles, min_heavy_atoms, max_heavy_atoms)
    return None if mol is None else Chem.MolToSmiles(mol)


def compute_features_with_reason(
    smiles: str,
    n_conformers: int = 10,
    mmff_max_iters: int = 2000,
    prune_rms: float = 0.5,
    seed: int = 42,
    temp_k: float = 298.15,
    min_heavy_atoms: int = 5,
    max_heavy_atoms: int = 80,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Return (features, None) or (None, human-readable failure reason)."""
    mol, reason = standardize_mol(smiles, min_heavy_atoms, max_heavy_atoms)
    if mol is None:
        return None, reason
    try:
        canonical = Chem.MolToSmiles(mol)
        feats: Dict[str, object] = {
            "canonical_smiles": canonical,
            "mol_wt": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "num_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
            "num_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
            "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }

        molh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.pruneRmsThresh = float(prune_rms)
        params.numThreads = 1
        cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=int(n_conformers), params=params))
        if not cids:
            params.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=int(n_conformers), params=params))
        if not cids:
            return None, "3D embedding failed (no conformer could be generated)."
        if not AllChem.MMFFHasAllMoleculeParams(molh):
            return None, "MMFF94 parameters are not available for this molecule."

        results = AllChem.MMFFOptimizeMoleculeConfs(
            molh, maxIters=int(mmff_max_iters), numThreads=1, mmffVariant="MMFF94"
        )
        conf_ids = [c.GetId() for c in molh.GetConformers()]
        # status == 0: converged; status == 1: maxIters reached (valid energy); status == -1: force-field failed
        converged = [i for i, (status, _) in enumerate(results) if status == 0]
        valid_ff = [i for i, (status, _) in enumerate(results) if status != -1]
        if not valid_ff:
            return None, "MMFF94 force-field minimization failed for all conformers."
        use = converged if converged else valid_ff
        energies = np.array([results[i][1] for i in use], dtype=float)
        ids = [conf_ids[i] for i in use]

        kt = KB_KCAL_PER_MOL_K * temp_k
        weights = np.exp(-(energies - energies.min()) / kt)
        weights = weights / weights.sum()

        def per_conf(fn) -> np.ndarray:
            return np.array([fn(molh, confId=cid) for cid in ids], dtype=float)

        rg = per_conf(Descriptors3D.RadiusOfGyration)
        rg_mean = float((weights * rg).sum())
        rg_std = float(np.sqrt(np.maximum(0.0, (weights * (rg - rg_mean) ** 2).sum())))

        feats.update(
            {
                "mmff94_energy_kcal_mol": float(energies.min()),
                "mmff94_energy_spread_kcal_mol": float(energies.max() - energies.min()),
                "radius_of_gyration_A": rg_mean,
                "rg_std_A": rg_std,
                "asphericity": float((weights * per_conf(Descriptors3D.Asphericity)).sum()),
                "eccentricity": float((weights * per_conf(Descriptors3D.Eccentricity)).sum()),
                "npr1": float((weights * per_conf(Descriptors3D.NPR1)).sum()),
                "npr2": float((weights * per_conf(Descriptors3D.NPR2)).sum()),
                "principal_moment_inertia_1": float((weights * per_conf(Descriptors3D.PMI1)).sum()),
                "n_conformers_used": int(len(ids)),
            }
        )
        for col in DESCRIPTOR_COLUMNS:
            v = feats[col]
            if v is None or math.isnan(v) or math.isinf(v):
                return None, f"Descriptor '{col}' was not finite."
        return feats, None
    except Exception as exc:  # RDKit can raise on odd valence / geometry edge cases
        return None, f"Feature computation failed: {type(exc).__name__}."


def compute_features(smiles: str, **kwargs) -> Optional[Dict[str, object]]:
    feats, _ = compute_features_with_reason(smiles, **kwargs)
    return feats
