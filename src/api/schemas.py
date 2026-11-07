from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MoleculePredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Input chemical structure in SMILES notation.",
        examples=["COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"],
    )


class MoleculePredictionResponse(BaseModel):
    query_smiles: str
    canonical_smiles: str
    predicted_pIC50: float
    estimated_ic50_nM: float
    prediction_uncertainty_pIC50: Optional[float] = None
    in_applicability_domain: bool
    max_tanimoto_similarity: float
    knn_mean_tanimoto: float
    confidence_tier: str
    nearest_training_smiles: Optional[str] = None
    model_name: str
    feature_set: str
    computed_physics_features: Dict[str, float]
    warnings: List[str] = []


class BatchPredictionError(BaseModel):
    index: int
    query_smiles: str
    detail: str


class BatchPredictionRequest(BaseModel):
    smiles: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch of input SMILES strings (up to 100 molecules).",
        examples=[["COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "CCCCCCCCCCCC"]],
    )


class BatchPredictionResponse(BaseModel):
    n_total: int
    n_succeeded: int
    n_failed: int
    n_in_domain: int
    predictions: List[MoleculePredictionResponse]
    errors: List[BatchPredictionError] = []
