import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.models.applicability import ApplicabilityDomainGuard

TRAIN = ["c1ccccc1C", "c1ccccc1Cl", "c1ccncc1C", "c1ccc2ccccc2c1O", "c1ccccc1OC"]


def test_vectorised_tanimoto_matches_rdkit():
    guard = ApplicabilityDomainGuard().fit(TRAIN)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ref = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in TRAIN]
    for q in ["c1ccccc1F", "CCCCCCCCCCCC", "c1ccc2[nH]ccc2c1"]:
        expected = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(q)), ref)
        got, valid = guard.similarity_matrix([q])
        assert valid[0]
        np.testing.assert_allclose(got[0], expected, atol=1e-6)


def test_in_domain_vs_ood_and_tiers():
    guard = ApplicabilityDomainGuard(threshold=0.35, high_threshold=0.5).fit(TRAIN)
    same = guard.evaluate_smiles("c1ccccc1C")
    assert same["in_applicability_domain"] and same["confidence_tier"] == "HIGH_IN_DOMAIN"
    assert same["max_tanimoto_similarity"] == 1.0 and same["nearest_training_smiles"] == "c1ccccc1C"
    ood = guard.evaluate_smiles("CCCCCCCCCCCC")
    assert not ood["in_applicability_domain"] and ood["confidence_tier"] == "LOW_OOD_WARNING"


def test_evaluate_batch_matches_single():
    guard = ApplicabilityDomainGuard(threshold=0.35, high_threshold=0.5).fit(TRAIN)
    queries = ["c1ccccc1C", "CCCCCCCCCCCC", "not_a_smiles"]
    batch_res = guard.evaluate_batch(queries)
    assert len(batch_res) == 3
    for q, b in zip(queries, batch_res):
        assert b == guard.evaluate_smiles(q)


def test_invalid_and_uninitialised():
    guard = ApplicabilityDomainGuard().fit(TRAIN)
    assert guard.evaluate_smiles("not_smiles")["confidence_tier"] == "INVALID_OR_UNINITIALIZED"
    assert ApplicabilityDomainGuard().evaluate_smiles("CCO")["confidence_tier"] == "INVALID_OR_UNINITIALIZED"


def test_save_load_roundtrip_without_pickle(tmp_path):
    guard = ApplicabilityDomainGuard(threshold=0.4, top_k=3).fit(TRAIN)
    path = tmp_path / "ad.npz"
    guard.save(path)
    loaded = ApplicabilityDomainGuard.load(path)
    assert loaded.threshold == 0.4 and loaded.top_k == 3 and loaded.n_reference == len(TRAIN)
    assert loaded.evaluate_smiles("c1ccccc1F") == guard.evaluate_smiles("c1ccccc1F")
