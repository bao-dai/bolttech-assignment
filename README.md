# GenAI-Powered Claim Approval Agent

This repo contains a solution for Take-Home Assignment from bolttech.

## Quick Start

```bash
# 1. Setup
conda activate bolt
pip install -r requirements.txt

# 2. Create .env file (required for GenAI features)
#    The Gradio demo, FastAPI /explain endpoint, LLM feature extraction,
#    and synthetic data generation all call the Claude API.
echo "CLAUDE_API_KEY=sk-ant-..." > .env

# 3. Run the Gradio Demo (interactive UI)
cd source/3_genai
python demo_app.py
# Opens http://127.0.0.1:7860

# 4. Run the FastAPI Server
cd source/4_api_deployment
uvicorn app:app --host 127.0.0.1 --port 8000
# API docs at http://127.0.0.1:8000/docs
```

## Main components

The project follows the assignment structure:

### 1: EDA (`source/1_eda/`)
> Data understanding

| File | What It Covers |
|------|---------------|
| `01_eda_summary.md` | Key findings: class imbalance, dead features, high-signal features, multilingual text |
| `02_ml_model_plan.md` | Plan for ML modelling based on EDA findings |
| `eda_feature_engineering.ipynb` | Full notebook: data loading --> cleaning --> 45 engineered features |

### 2: ML Modelling (`source/2_ml_modelling/`)
> 4 iterative attempts, each building on the last.

| File | Attempt | F1 Macro | Key Idea |
|------|---------|----------|----------|
| `model_training.ipynb` | v1 | 0.566 | Structural features only |
| `01_first_attempt_analysis.md` | | | Why v1 failed --> need text signal |
| `model_training_v2.ipynb` | v2 | 0.595 | + Sentence-transformer embeddings |
| `02_second_attempt_analysis.md` | | | Why v2 wasn't enough --> need deeper NLP |
| `model_training_v3.ipynb` | v3 | 0.608 | + BERT / LSTM / TF-IDF comparison |
| `03_third_attempt_analysis.md` | | | Why v3 plateaued --> need domain-aware features |
| `model_training_v4.ipynb` | v4 | **0.677** | + LLM feature extraction (we'll use Claude for this) |
| `04_fourth_attempt_analysis.md` | | | Final analysis, production model selection |
| `extract_llm_features.py` | | | Script: Claude extracts 10 structured features per claim |

### 3: GenAI (`source/3_genai/`)
> Multi-persona explanations + synthetic data generation.

| File | What It Covers |
|------|---------------|
| `task2_genai_report.md` | Prompt engineering strategy, 3 personas, grounding approach |
| `multi_persona_explainer.py` | Core module: `explain_claim()` --> customer/adjuster/manager explanations |
| `demo_app.py` | Gradio UI for interactive demo |
| `genai_explanations.ipynb` | Notebook with demo outputs (declined, approved, borderline claims) |
| `task2b_synthetic_data_report.md` | Synthetic data strategy, findings, model evaluation |
| `synthetic_claim_generator.py` | Generates declined/borderline/edge-case scenarios via Claude |
| `evaluate_synthetic.py` | Tests production model on synthetic claims |

### 4: API & Deployment (`source/4_api_deployment/`)
> RESTful API + AWS design.

| File | What It Covers |
|------|---------------|
| `app.py` | FastAPI: `/predict`, `/explain`, `/health`, `/model/info` |
| `task3_api_deployment_report.md` | AWS architecture (ECS Fargate), MLOps/LLMOps, cost estimates |

### 5: Evaluation & Monitoring (`source/5_evaluation_monitoring/`)
> Production monitoring plan.

| File | What It Covers |
|------|---------------|
| `task4_evaluation_monitoring.md` | ML metrics, GenAI quality, drift detection, fairness, responsible AI |

## Model Progression

| Version | Approach | F1 Macro | F1 Declined | Improvement |
|---------|----------|----------|-------------|-------------|
| v1 | Structural features + LightGBM | 0.566 | 0.280 | Baseline |
| v2 | + Sentence-transformer embeddings | 0.595 | 0.294 | +5.1% |
| v3 | + Fine-tuned BERT / LSTM / TF-IDF | 0.608 | 0.365 | +7.4% |
| **v4** | **+ LLM feature extraction (Claude)** | **0.677** | **0.456** | **+19.6%** |

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM over deep learning | 2,880 samples: tree models outperform neural nets at this scale |
| Claude Haiku for LLM features | Cost-efficient ($0.50/1K claims), fast, multilingual |
| SHAP-grounded explanations | Prevents hallucination: LLM explains what the model decided, not what it guesses |
| 4 iterative attempts | Shows thought process progression, not just final result |
| `is_unbalance=True` over Optuna | Default LightGBM consistently beat tuned versions on minority class |

## Assumptions

- `status = "Completed"` is interpreted as claim approved/paid out
- `status = "Declined"` is interpreted as claim denied
- All `issueDesc` text is processed in its original language (Swedish/Dutch/Finnish/English)
- Claude Haiku API is used for both LLM feature extraction and explanation generation
- Local deployment (FastAPI + Gradio) is the primary demo; AWS design is documented but not deployed

## Services Used

- **Claude Haiku** (Anthropic API): LLM feature extraction + explanation generation
- **sentence-transformers** (Hugging Face): multilingual text embeddings (v2)
- **distilbert-base-multilingual-cased** (Hugging Face): BERT fine-tuning (v3)
- **LightGBM**: production ML model
- **SHAP**: model explainability
- **FastAPI**: REST API
- **Gradio**: interactive demo UI
