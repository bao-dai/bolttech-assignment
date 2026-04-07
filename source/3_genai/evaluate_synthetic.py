import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'source' / '3_genai'))
sys.path.insert(0, str(PROJECT_ROOT / 'source' / '2_ml_modelling'))

from multi_persona_explainer import load_model_artifacts, generate_explanation
from extract_llm_features import extract_features


def main():
    print('Loading model artifacts...')
    artifacts = load_model_artifacts()
    config = artifacts['config']

    print('Loading synthetic claims...')
    syn_df = pd.read_csv(PROJECT_ROOT / 'data' / 'synthetic' / 'synthetic_claims.csv')
    print(f'Synthetic claims: {len(syn_df)}')
    print(f'By type: {syn_df["scenario_type"].value_counts().to_dict()}')
    print(f'By expected status: {syn_df["status"].value_counts().to_dict()}')

    # Step 1: Extract LLM features for synthetic claims
    print('\nExtracting LLM features for synthetic claims...')
    llm_results = []
    for i, row in syn_df.iterrows():
        result = extract_features(
            i,
            str(row.get('issueDesc', '')),
            str(row.get('claimType', '')),
            str(row.get('coverage', '')),
            str(row.get('deviceType', 'SMARTPHONES')),
            str(row.get('make', 'WUAWEI')),
            str(row.get('model', '')),
        )
        llm_results.append(result)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(syn_df)}')

    llm_syn_df = pd.DataFrame(llm_results)

    # Step 2: Build feature matrix for synthetic claims
    # We need the same features as the trained model
    structural_features = config['structural_features']
    llm_encoded_cols = config['llm_features']
    llm_encoders = joblib.load(PROJECT_ROOT / 'models' / 'llm_encoders_v4.joblib')

    # Map structural features  -  many won't exist in synthetic data, fill with 0
    X_struct = pd.DataFrame(0, index=range(len(syn_df)), columns=structural_features)

    # Fill what we can from synthetic data
    field_map = {
        'excessFee': 'excessFee', 'rrp': 'rrp',
        'claimType_encoded': None, 'country_encoded': None,
        'coverage_encoded': None, 'channel_encoded': None,
        'deviceType_encoded': None,
    }

    for col in structural_features:
        if col in syn_df.columns:
            X_struct[col] = syn_df[col].fillna(0).values
        elif col == 'rrp' and 'rrp' in syn_df.columns:
            X_struct[col] = pd.to_numeric(syn_df['rrp'], errors='coerce').fillna(0).values
        elif col == 'excessFee' and 'excessFee' in syn_df.columns:
            X_struct[col] = pd.to_numeric(syn_df['excessFee'], errors='coerce').fillna(0).values

    # Encode LLM features
    X_llm = pd.DataFrame()
    llm_feature_cols = [c for c in llm_syn_df.columns if c not in ['row_idx', 'error']]
    for col in llm_feature_cols:
        enc_col = f'llm_{col}'
        if enc_col in llm_encoded_cols and col in llm_encoders:
            le = llm_encoders[col]
            vals = llm_syn_df[col].fillna('unknown').astype(str)
            # Handle unseen labels
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else 'unknown' if 'unknown' in known else le.classes_[0])
            X_llm[enc_col] = le.transform(vals)

    # Fill missing LLM columns with 0
    for col in llm_encoded_cols:
        if col not in X_llm.columns:
            X_llm[col] = 0

    # Combine
    X = pd.concat([X_struct, X_llm[llm_encoded_cols]], axis=1).fillna(0).values

    # Step 3: Predict
    model = artifacts['model']
    probas = model.predict_proba(X)[:, 1]
    threshold = config['optimal_threshold']
    predictions = ['Approved' if p >= threshold else 'Declined' for p in probas]

    # Step 4: Evaluate
    syn_df['ml_prediction'] = predictions
    syn_df['ml_probability'] = probas

    # Map expected status
    expected_map = {'Declined': 'Declined', 'Completed': 'Approved', 'Borderline': 'Uncertain'}
    syn_df['expected'] = syn_df['status'].map(expected_map)

    print(f'\n{"="*60}')
    print('SYNTHETIC CLAIM EVALUATION RESULTS')
    print(f'{"="*60}')

    # Declined scenarios  -  model should predict Declined
    declined = syn_df[syn_df['scenario_type'] == 'declined']
    declined_correct = (declined['ml_prediction'] == 'Declined').sum()
    print(f'\nDeclined scenarios ({len(declined)} claims):')
    print(f'  Model predicted Declined: {declined_correct}/{len(declined)} ({declined_correct/len(declined):.0%})')
    print(f'  Mean P(approved): {declined["ml_probability"].mean():.3f}')

    # Borderline scenarios  -  should be near threshold
    borderline = syn_df[syn_df['scenario_type'] == 'borderline']
    near_threshold = ((borderline['ml_probability'] - threshold).abs() < 0.2).sum()
    print(f'\nBorderline scenarios ({len(borderline)} claims):')
    print(f'  Near threshold (within 0.2): {near_threshold}/{len(borderline)} ({near_threshold/len(borderline):.0%})')
    print(f'  Mean P(approved): {borderline["ml_probability"].mean():.3f}')
    print(f'  Predictions: {borderline["ml_prediction"].value_counts().to_dict()}')

    # Edge cases
    edge = syn_df[syn_df['scenario_type'] == 'edge_cases']
    print(f'\nEdge case scenarios ({len(edge)} claims):')
    print(f'  Predictions: {edge["ml_prediction"].value_counts().to_dict()}')
    print(f'  Mean P(approved): {edge["ml_probability"].mean():.3f}')

    # Overall
    print(f'\nOverall predictions on synthetic data:')
    print(syn_df['ml_prediction'].value_counts().to_string())

    # Save enriched results
    output_path = PROJECT_ROOT / 'data' / 'synthetic' / 'synthetic_claims_evaluated.csv'
    syn_df.to_csv(output_path, index=False)
    print(f'\nSaved evaluated results to {output_path}')

    # Show a few interesting cases
    print(f'\n{"="*60}')
    print('Sample: Declined scenario the model GOT RIGHT')
    print(f'{"="*60}')
    correct_declined = declined[declined['ml_prediction'] == 'Declined']
    if len(correct_declined) > 0:
        s = correct_declined.iloc[0]
        print(f'  Type: {s.get("claimType")} | P(approved)={s["ml_probability"]:.3f}')
        print(f'  Reason: {s.get("decline_reason", "N/A")}')
        print(f'  Desc: {str(s.get("issueDesc", ""))[:200]}...')

    wrong_declined = declined[declined['ml_prediction'] == 'Approved']
    if len(wrong_declined) > 0:
        print(f'\n{"="*60}')
        print(f'Sample: Declined scenario the model MISSED ({len(wrong_declined)} total)')
        print(f'{"="*60}')
        s = wrong_declined.iloc[0]
        print(f'  Type: {s.get("claimType")} | P(approved)={s["ml_probability"]:.3f}')
        print(f'  Reason: {s.get("decline_reason", "N/A")}')
        print(f'  Desc: {str(s.get("issueDesc", ""))[:200]}...')


if __name__ == '__main__':
    main()
