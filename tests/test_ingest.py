import pandas as pd

from src.data.ingest_chembl import ChEMBLDataIngestor, curate, parse_activity

CHEMBL_CFG = {"page_size": 2, "assay_type": "B", "require_pchembl": True, "exclude_mutants": True}


def _act(smiles, value, **kw):
    base = {
        "canonical_smiles": smiles, "standard_value": str(value), "standard_type": "IC50",
        "standard_units": "nM", "standard_relation": "=", "pchembl_value": "7.0",
        "molecule_chembl_id": "CHEMBL1", "assay_chembl_id": "CHEMBLA1", "document_chembl_id": "CHEMBLD1",
        "data_validity_comment": None, "potential_duplicate": 0, "assay_variant_mutation": None,
    }
    base.update(kw)
    return base


class FakeResponse:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.pages[len(self.calls) - 1])


def test_parse_activity_filters():
    assert parse_activity(_act("c1ccccc1C", 10)) is not None
    assert parse_activity(_act("c1ccccc1C", 10, standard_relation=">")) is None
    assert parse_activity(_act("c1ccccc1C", 10, data_validity_comment="Outside typical range")) is None
    assert parse_activity(_act("c1ccccc1C", 10, potential_duplicate=1)) is None
    assert parse_activity(_act("c1ccccc1C", 10, pchembl_value=None)) is None
    assert parse_activity(_act("c1ccccc1C", 10, assay_variant_mutation="L858R")) is None
    assert parse_activity(_act("c1ccccc1C", -5)) is None
    assert parse_activity(_act("c1ccccc1C", "abc")) is None
    assert parse_activity(_act(None, 10)) is None


def test_pagination_follows_next_links_and_respects_max_records():
    pages = [
        {"activities": [_act("c1ccccc1C", 10), _act("c1ccccc1Cl", 20, standard_relation="<")],
         "page_meta": {"next": "/chembl/api/data/activity.json?offset=2"}},
        {"activities": [_act("c1ccncc1C", 30), _act("c1ccccc1F", 40)], "page_meta": {"next": None}},
    ]
    session = FakeSession(pages)
    df = ChEMBLDataIngestor("CHEMBL203", CHEMBL_CFG, session=session).fetch(max_records=10)
    assert len(df) == 3 and len(session.calls) == 2
    assert session.calls[1][0].startswith("https://www.ebi.ac.uk/chembl/api/data/activity.json?offset=2")
    assert session.calls[0][1]["pchembl_value__isnull"] == "false"

    session2 = FakeSession(pages)
    assert len(ChEMBLDataIngestor("CHEMBL203", CHEMBL_CFG, session=session2).fetch(max_records=1)) == 1
    assert len(session2.calls) == 1


def test_curate_median_aggregation_and_conflict_removal():
    raw = pd.DataFrame(
        [
            # same molecule given as free base and as a salt -> one canonical entry, median pIC50
            {"molecule_chembl_id": "A", "raw_smiles": "c1ccccc1CCO", "ic50_nM": 10.0, "assay_chembl_id": "a1"},
            {"molecule_chembl_id": "A", "raw_smiles": "Cl.c1ccccc1CCO", "ic50_nM": 100.0, "assay_chembl_id": "a2"},
            {"molecule_chembl_id": "A", "raw_smiles": "c1ccccc1CCO", "ic50_nM": 1000.0, "assay_chembl_id": "a3"},
            # conflicting replicates (4 log units apart) -> removed
            {"molecule_chembl_id": "B", "raw_smiles": "c1ccncc1CCl", "ic50_nM": 1.0, "assay_chembl_id": "a1"},
            {"molecule_chembl_id": "B", "raw_smiles": "c1ccncc1CCl", "ic50_nM": 10000.0, "assay_chembl_id": "a2"},
            # too small -> removed
            {"molecule_chembl_id": "C", "raw_smiles": "CCC", "ic50_nM": 5.0, "assay_chembl_id": "a1"},
        ]
    )
    out = curate(raw, {"min_heavy_atoms": 5, "max_heavy_atoms": 80, "max_replicate_std": 1.0})
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_measurements"] == 3 and row["n_assays"] == 3
    assert abs(row["pIC50"] - 7.0) < 1e-9  # median of 8, 7, 6
