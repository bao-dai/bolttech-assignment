import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add genai module to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'source' / '3_genai'))

from multi_persona_explainer import (
    load_model_artifacts, get_claim_context, generate_explanation
)

# -- Global state --------------------------------------------
ARTIFACTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global ARTIFACTS
    print("Loading model artifacts...")
    ARTIFACTS = load_model_artifacts()
    print(f"Ready. {len(ARTIFACTS['claims_df'])} claims loaded.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Claim Approval Agent",
    description=(
        "GenAI-powered insurance claim approval prediction and explanation API. "
        "Predicts claim approval status and generates multi-persona explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# -- Request / Response Models -------------------------------

class PredictRequest(BaseModel):
    claim_idx: int = Field(..., ge=0, description="Index of the claim in the dataset")


class PredictResponse(BaseModel):
    claim_idx: int
    prediction: str
    approval_probability: float
    threshold: float
    damage_type: str
    damage_severity: str
    incident_type: str
    is_gradual_wear: str
    top_factors: list


class ExplainRequest(BaseModel):
    claim_idx: int = Field(..., ge=0, description="Index of the claim in the dataset")
    persona: str = Field(
        default="customer",
        description="Persona for explanation: 'customer', 'adjuster', or 'manager'"
    )


class ExplainResponse(BaseModel):
    claim_idx: int
    prediction: str
    approval_probability: float
    persona: str
    explanation: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_columns: list
    optimal_threshold: float
    llm_model: str
    total_claims: int


# -- Endpoints -----------------------------------------------

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model_loaded": ARTIFACTS is not None}


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    config = ARTIFACTS['config']
    return ModelInfoResponse(
        model_type="LightGBM (v4  -  structural + LLM features)",
        n_features=len(config['feature_columns']),
        feature_columns=config['feature_columns'],
        optimal_threshold=config['optimal_threshold'],
        llm_model=config.get('llm_model', 'claude-haiku-4-5-20251001'),
        total_claims=len(ARTIFACTS['claims_df']),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict claim approval status."""
    if req.claim_idx >= len(ARTIFACTS['claims_df']):
        raise HTTPException(status_code=404, detail=f"Claim index {req.claim_idx} not found")

    ctx = get_claim_context(ARTIFACTS, req.claim_idx)

    return PredictResponse(
        claim_idx=req.claim_idx,
        prediction=ctx['prediction'],
        approval_probability=ctx['approval_probability'],
        threshold=ctx['threshold'],
        damage_type=ctx['llm_damage_type'],
        damage_severity=ctx['llm_damage_severity'],
        incident_type=ctx['llm_incident_type'],
        is_gradual_wear=ctx['llm_is_gradual_wear'],
        top_factors=ctx['top_factors'][:5],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """Generate a persona-specific explanation for a claim."""
    if req.claim_idx >= len(ARTIFACTS['claims_df']):
        raise HTTPException(status_code=404, detail=f"Claim index {req.claim_idx} not found")

    if req.persona not in ['customer', 'adjuster', 'manager']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona '{req.persona}'. Use 'customer', 'adjuster', or 'manager'."
        )

    ctx = get_claim_context(ARTIFACTS, req.claim_idx)
    explanation = generate_explanation(ctx, req.persona)

    return ExplainResponse(
        claim_idx=req.claim_idx,
        prediction=ctx['prediction'],
        approval_probability=ctx['approval_probability'],
        persona=req.persona,
        explanation=explanation,
    )


# -- Main ----------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
