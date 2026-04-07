import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multi_persona_explainer import load_model_artifacts, explain_claim

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'sample_explanations.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8, help='Number of sample claims')
    args = parser.parse_args()

    print('Loading model artifacts...')
    artifacts = load_model_artifacts()
    claims_df = artifacts['claims_df']
    config = artifacts['config']

    np.random.seed(42)

    # Pick diverse claims: declined, approved, borderline
    declined_idx = claims_df[claims_df['target'] == 0].index.tolist()
    approved_idx = claims_df[claims_df['target'] == 1].index.tolist()

    # Borderline
    feature_cols = config['feature_columns']
    X_all = claims_df[feature_cols].fillna(0).values
    probas = artifacts['model'].predict_proba(X_all)[:, 1]
    distances = np.abs(probas - config['optimal_threshold'])
    borderline_idx = np.argsort(distances)[:10].tolist()

    n_each = max(1, args.n // 3)
    sample = (
        list(np.random.choice(declined_idx, min(n_each, len(declined_idx)), replace=False))
        + list(np.random.choice(approved_idx, min(n_each, len(approved_idx)), replace=False))
        + borderline_idx[:max(1, args.n - 2 * n_each)]
    )

    print(f'Generating explanations for {len(sample)} claims...\n')

    results = []
    for idx in sample:
        idx = int(idx)
        result = explain_claim(artifacts, idx, personas=['customer', 'adjuster', 'manager'])
        ctx = result['claim_context']

        print(f'  Claim #{idx}: {ctx["prediction"]} (P={ctx["approval_probability"]:.3f}) '
              f' -  {ctx["claim_type"]}, {ctx["llm_damage_type"]}')

        results.append({
            'claim_idx': idx,
            'prediction': ctx['prediction'],
            'actual': ctx['actual_status'],
            'probability': ctx['approval_probability'],
            'claim_type': ctx['claim_type'],
            'damage_type': ctx['llm_damage_type'],
            'customer_explanation': result['customer'],
            'adjuster_explanation': result['adjuster'],
            'manager_explanation': result['manager'],
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved {len(df)} explanations to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
