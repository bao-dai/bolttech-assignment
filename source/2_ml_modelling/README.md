# ML Modelling

4 iterative model training attempts. Each one builds on what I learned from the previous.

## Notebooks (run in order)

1. `model_training.ipynb` - v1, structural features only. Logistic regression + LightGBM + XGBoost
2. `model_training_v2.ipynb` - v2, added sentence-transformer text embeddings
3. `model_training_v3.ipynb` - v3, tried TF-IDF, BiLSTM, fine-tuned BERT. Warning: this one takes a while on CPU because of BERT
4. `model_training_v4.ipynb` - v4, LLM-extracted features via Claude. Best results

## Scripts

- `extract_llm_features.py` - calls Claude Haiku API to extract structured features (damage type, severity, etc) from each claim description. Outputs to `../../data/llm_features.csv`. Needs `.env` with `CLAUDE_API_KEY`

## Quick run

```
conda activate bolt
cd source/2_ml_modelling

# run a notebook
jupyter notebook model_training_v4.ipynb

# or run LLM feature extraction standalone
python extract_llm_features.py
```

Models get saved to `../../models/`
