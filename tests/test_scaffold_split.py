import pytest

from src.data.scaffold_split import assign_scaffold_splits, extract_scaffold


def test_extract_scaffold():
    assert extract_scaffold("c1ccccc1CCC") == "c1ccccc1"
    assert extract_scaffold("CCCCCC") == "ACYCLIC"
    assert extract_scaffold("garbage") == "INVALID"
    # generic scaffolds collapse benzene and cyclohexane to the same skeleton
    assert extract_scaffold("c1ccccc1C", "generic") == extract_scaffold("C1CCCCC1C", "generic")


@pytest.mark.parametrize("scaffold_type", ["murcko", "generic"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_scaffold_leakage_and_all_rows_assigned(synthetic_features, scaffold_type, seed):
    out = assign_scaffold_splits(synthetic_features, seed=seed, scaffold_type=scaffold_type)
    assert set(out["split"]) <= {"train", "val", "test"}
    assert len(out) == len(synthetic_features)
    sets = {s: set(out.loc[out["split"] == s, "scaffold_smiles"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert (out["split"] == "train").sum() > 0.6 * len(out)


def test_split_is_reproducible_per_seed_and_varies_across_seeds(synthetic_features):
    a = assign_scaffold_splits(synthetic_features, seed=3)["split"].tolist()
    b = assign_scaffold_splits(synthetic_features, seed=3)["split"].tolist()
    assert a == b
    others = [assign_scaffold_splits(synthetic_features, seed=s)["split"].tolist() for s in range(1, 8)]
    assert any(o != a for o in others)
