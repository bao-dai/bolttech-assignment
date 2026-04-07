# GenAI

Multi-persona explanations and synthetic data generation. All scripts here need the Claude API key in `.env`

## Explanations

- `multi_persona_explainer.py` - core module. Has `explain_claim()` which generates explanations for 3 personas (customer, adjuster, manager). Grounded in SHAP values so it doesnt hallucinate
- `demo_app.py` - Gradio UI, probably the best way to demo this. Pick a claim, pick a persona, get an explanation
- `generate_sample_explanations.py` - batch version, generates explanations for a mix of claims and saves to CSV
- `genai_explanations.ipynb` - notebook version with some examples and the prompt strategy writeup

## Synthetic data

- `synthetic_claim_generator.py` - generates fake but realistic claim scenarios. Focuses on declined patterns, borderline cases, and edge cases (rare device types etc)
- `evaluate_synthetic.py` - runs the production model on synthetic claims to see how it handles them

## How to run

```
conda activate bolt
cd source/3_genai

# interactive demo
python demo_app.py

# batch generate explanations
python generate_sample_explanations.py --n 10

# generate synthetic claims
python synthetic_claim_generator.py --n 50 --focus all

# evaluate model on synthetic data
python evaluate_synthetic.py
```

