import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

CLIENT = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
MODEL = "claude-haiku-4-5-20251001"


def load_model_artifacts():
    """Load the trained model, SHAP values, and configs."""
    models_dir = PROJECT_ROOT / 'models'

    model = joblib.load(models_dir / 'lgb_claim_model_v4.joblib')

    with open(models_dir / 'model_config_v4.json', 'r') as f:
        config = json.load(f)

    shap_df = pd.read_csv(models_dir / 'shap_values_v4.csv')
    llm_features = pd.read_csv(PROJECT_ROOT / 'data' / 'llm_features.csv')
    claims_df = pd.read_csv(PROJECT_ROOT / 'data' / 'claims_cleaned.csv')

    # Merge LLM encoded features into claims_df so model can predict
    llm_encoders = joblib.load(models_dir / 'llm_encoders_v4.joblib')
    llm_feature_cols = [c for c in llm_features.columns if c not in ['row_idx', 'error']]
    for col in llm_feature_cols:
        le = llm_encoders[col]
        claims_df[f'llm_{col}'] = le.transform(llm_features[col].fillna('unknown').astype(str))

    return {
        'model': model,
        'config': config,
        'shap_df': shap_df,
        'llm_features': llm_features,
        'claims_df': claims_df,
    }


def get_claim_context(artifacts, claim_idx):
    """Build a rich context object for a single claim."""
    claim = artifacts['claims_df'].iloc[claim_idx]
    shap_row = artifacts['shap_df'].iloc[claim_idx]
    llm_row = artifacts['llm_features'].iloc[claim_idx]
    config = artifacts['config']

    # Model prediction
    feature_cols = config['feature_columns']
    X_row = artifacts['claims_df'][feature_cols].fillna(0).iloc[claim_idx:claim_idx+1]
    proba = artifacts['model'].predict_proba(X_row)[0]
    threshold = config['optimal_threshold']
    prediction = 'Approved' if proba[1] >= threshold else 'Declined'

    # Top SHAP contributors (sorted by absolute impact)
    shap_impacts = []
    for col in feature_cols:
        val = shap_row[col]
        if abs(val) > 0.01:
            shap_impacts.append({
                'feature': col,
                'shap_value': round(float(val), 4),
                'direction': 'toward approval' if val > 0 else 'toward decline',
                'feature_value': str(claim.get(col, 'N/A')),
            })
    shap_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    top_factors = shap_impacts[:8]

    return {
        'prediction': prediction,
        'confidence': round(float(max(proba)), 3),
        'approval_probability': round(float(proba[1]), 3),
        'threshold': threshold,
        'claim_type': str(claim.get('claimType', '')),
        'coverage': str(claim.get('coverage', '')),
        'country': str(claim.get('country', '')),
        'device_type': str(claim.get('deviceType', '')),
        'make': str(claim.get('make', '')),
        'model_name': str(claim.get('model', '')),
        'rrp': float(claim.get('rrp', 0)),
        'excess_fee': float(claim.get('excessFee', 0)),
        'issue_description': str(claim.get('issueDesc', '')),
        'policy_status': str(claim.get('policyStatus', '')),
        'actual_status': str(claim.get('status', '')),
        'llm_damage_type': str(llm_row.get('damage_type', '')),
        'llm_damage_severity': str(llm_row.get('damage_severity', '')),
        'llm_incident_type': str(llm_row.get('incident_type', '')),
        'llm_incident_clarity': str(llm_row.get('incident_clarity', '')),
        'llm_is_gradual_wear': str(llm_row.get('is_gradual_wear', '')),
        'llm_user_at_fault': str(llm_row.get('user_at_fault', '')),
        'llm_emotional_tone': str(llm_row.get('emotional_tone', '')),
        'llm_device_functional': str(llm_row.get('device_functional', '')),
        'top_factors': top_factors,
    }


# -- Persona Prompts ------------------------------------------

CUSTOMER_PROMPT = """You are a friendly, empathetic customer service representative at an insurance company.
A customer has filed a claim and you need to explain the decision to them.

Claim details:
- Device: {make} {model_name} ({device_type})
- Claim type: {claim_type}
- Coverage: {coverage}
- Issue: {issue_description}
- Decision: **{prediction}**
- Confidence: {confidence}

Key factors in this decision:
{factors_text}

Damage assessment: {llm_damage_type} ({llm_damage_severity} severity), incident type: {llm_incident_type}
{gradual_wear_note}

Write a clear, empathetic explanation for the customer. Guidelines:
1. Start with the decision clearly
2. Explain the main reasons in plain language (no technical jargon, no SHAP values)
3. If declined: explain what specifically led to the decline and what the customer can do (appeal, provide more info)
4. If approved: confirm what happens next
5. Keep it under 200 words
6. Be respectful and professional  -  this is their device and their claim"""

ADJUSTER_PROMPT = """You are writing an internal claims adjuster note. Provide a technical, evidence-based analysis of this claim decision.

Claim details:
- Device: {make} {model_name} ({device_type}), RRP: {rrp}, Excess: {excess_fee}
- Claim type: {claim_type}, Coverage: {coverage}, Country: {country}
- Policy status: {policy_status}
- Issue description: {issue_description}
- ML Decision: **{prediction}** (approval probability: {approval_probability}, threshold: {threshold})

AI-assessed damage profile:
- Damage type: {llm_damage_type} | Severity: {llm_damage_severity}
- Incident type: {llm_incident_type} | Clarity: {llm_incident_clarity}
- Gradual wear: {llm_is_gradual_wear} | User at fault: {llm_user_at_fault}
- Device functional: {llm_device_functional} | Emotional tone: {llm_emotional_tone}

Top model contributing factors (SHAP analysis):
{factors_text_technical}

Write an adjuster note. Guidelines:
1. Summarize the prediction and confidence level
2. List the key evidence supporting or contradicting the decision
3. Flag any risk factors or inconsistencies
4. Note if this case warrants manual review
5. Reference specific data points (RRP, damage type, policy coverage)
6. Keep it under 250 words, structured with bullet points"""

MANAGER_PROMPT = """You are writing a brief executive summary of a claim decision for a claims operations manager who reviews flagged cases.

Claim: {claim_type} on {make} {model_name} ({device_type})  -  {country}
Decision: **{prediction}** (confidence: {confidence})
Device value: {rrp}, Excess: {excess_fee}
Damage: {llm_damage_type} ({llm_damage_severity}), caused by: {llm_incident_type}
Gradual wear detected: {llm_is_gradual_wear}

Key risk factors:
{factors_text}

Write a 3-4 sentence executive summary covering:
1. The decision and why
2. Any flags that warrant attention (borderline confidence, gradual wear, high value)
3. Recommended action (auto-approve, auto-decline, or escalate to manual review)"""


def format_factors_customer(factors):
    lines = []
    for f in factors[:5]:
        name = f['feature'].replace('_', ' ').replace('llm ', '')
        lines.append(f"- {name}: contributes {f['direction']}")
    return '\n'.join(lines)


def format_factors_technical(factors):
    lines = []
    for f in factors[:8]:
        lines.append(f"- {f['feature']}: value={f['feature_value']}, "
                     f"SHAP={f['shap_value']:+.4f} ({f['direction']})")
    return '\n'.join(lines)


def generate_explanation(claim_context, persona='customer'):
    """Generate a persona-specific explanation for a claim decision."""

    gradual_note = ""
    if claim_context['llm_is_gradual_wear'] == 'True':
        gradual_note = "NOTE: The damage appears to be gradual wear rather than a sudden incident, which is typically not covered under accidental damage policies."

    if persona == 'customer':
        prompt = CUSTOMER_PROMPT.format(
            **claim_context,
            factors_text=format_factors_customer(claim_context['top_factors']),
            gradual_wear_note=gradual_note,
        )
    elif persona == 'adjuster':
        prompt = ADJUSTER_PROMPT.format(
            **claim_context,
            factors_text_technical=format_factors_technical(claim_context['top_factors']),
        )
    elif persona == 'manager':
        prompt = MANAGER_PROMPT.format(
            **claim_context,
            factors_text=format_factors_customer(claim_context['top_factors']),
        )
    else:
        raise ValueError(f"Unknown persona: {persona}")

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def explain_claim(artifacts, claim_idx, personas=None):
    """Generate explanations for a claim across multiple personas."""
    if personas is None:
        personas = ['customer', 'adjuster', 'manager']

    context = get_claim_context(artifacts, claim_idx)
    explanations = {'claim_context': context}

    for persona in personas:
        explanations[persona] = generate_explanation(context, persona)

    return explanations


# -- CLI entrypoint ------------------------------------------

if __name__ == '__main__':
    import sys

    print('Loading model artifacts...')
    artifacts = load_model_artifacts()
    print('Done.\n')

    # Demo: explain a few claims
    claim_indices = [13, 0, 100]  # Mix of declined and completed
    if len(sys.argv) > 1:
        claim_indices = [int(x) for x in sys.argv[1:]]

    for idx in claim_indices:
        print(f'\n{"="*70}')
        print(f'CLAIM #{idx}')
        print(f'{"="*70}')

        result = explain_claim(artifacts, idx)
        ctx = result['claim_context']

        print(f'Prediction: {ctx["prediction"]} (P(approved)={ctx["approval_probability"]:.3f})')
        print(f'Actual: {ctx["actual_status"]}')
        print(f'Claim: {ctx["claim_type"]} | {ctx["llm_damage_type"]} | {ctx["llm_incident_type"]}')

        for persona in ['customer', 'adjuster', 'manager']:
            print(f'\n--- {persona.upper()} EXPLANATION ---')
            print(result[persona])
