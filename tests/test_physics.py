import math

from src.features.physics import (
    DESCRIPTOR_COLUMNS,
    compute_features_with_reason,
    standardize_smiles,
)

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


def test_gefitinib_features_are_finite_and_in_range():
    feats, reason = compute_features_with_reason(GEFITINIB, n_conformers=4)
    assert reason is None
    assert all(math.isfinite(feats[c]) for c in DESCRIPTOR_COLUMNS)
    assert 440 < feats["mol_wt"] < 450
    assert 0.0 <= feats["asphericity"] <= 1.0
    assert 0.0 <= feats["eccentricity"] <= 1.0
    assert feats["radius_of_gyration_A"] > 0
    assert feats["mmff94_energy_spread_kcal_mol"] >= 0
    assert feats["n_conformers_used"] >= 1


def test_features_are_deterministic():
    a, _ = compute_features_with_reason(GEFITINIB, n_conformers=4, seed=7)
    b, _ = compute_features_with_reason(GEFITINIB, n_conformers=4, seed=7)
    assert a == b


def test_salt_is_stripped_and_charges_neutralised():
    assert standardize_smiles("Cl." + GEFITINIB) == standardize_smiles(GEFITINIB)
    assert standardize_smiles("CCCC(=O)[O-].[Na+]") == standardize_smiles("CCCC(=O)O")


def test_failure_reasons():
    assert compute_features_with_reason("not_a_smiles")[0] is None
    assert "small" in compute_features_with_reason("CCC")[1]
    assert "large" in compute_features_with_reason("C" * 120, max_heavy_atoms=80)[1]
    assert compute_features_with_reason("")[0] is None
