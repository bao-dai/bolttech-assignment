import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

CLIENT = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
MODEL = "claude-haiku-4-5-20251001"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'llm_features.csv'
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'llm_features_cache.json'

EXTRACTION_PROMPT = """You are an insurance claim analyst. Analyze this claim description and extract structured features.
The text may be in Swedish, Dutch, Finnish, or English. Understand the language and extract features in English.

Claim description:
<claim>
{text}
</claim>

Additional context:
- Claim type: {claim_type}
- Coverage: {coverage}
- Device: {device_type} ({make} {model})

Extract the following features as a JSON object. Use ONLY the allowed values listed:

{{
  "damage_type": one of ["screen_crack", "screen_shatter", "water_damage", "theft", "total_loss", "cosmetic_scratch", "camera_damage", "battery_issue", "software_issue", "multiple_damage", "other"],
  "damage_severity": one of ["minor", "moderate", "severe"],
  "incident_clarity": one of ["vague", "moderate", "detailed"],
  "incident_type": one of ["drop", "impact", "water", "theft", "crush", "unknown_cause", "gradual_wear", "other"],
  "user_at_fault": one of [true, false],
  "is_gradual_wear": one of [true, false],
  "has_police_report": one of [true, false, "not_applicable"],
  "emotional_tone": one of ["neutral", "frustrated", "apologetic", "distressed", "matter_of_fact"],
  "device_functional": one of [true, false, "partial"],
  "third_party_involved": one of [true, false]
}}

Return ONLY the JSON object, no other text."""


def extract_features(row_idx, text, claim_type, coverage, device_type, make, model_name):
    """Call Claude to extract features from a single claim."""
    prompt = EXTRACTION_PROMPT.format(
        text=text[:1500],  # Truncate very long texts
        claim_type=claim_type,
        coverage=coverage,
        device_type=device_type,
        make=make,
        model=model_name,
    )

    for attempt in range(3):
        try:
            response = CLIENT.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            # Parse JSON  -  handle potential markdown wrapping
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            features = json.loads(content)
            features['row_idx'] = row_idx
            return features

        except json.JSONDecodeError:
            # Try to extract JSON from response
            try:
                start = content.index('{')
                end = content.rindex('}') + 1
                features = json.loads(content[start:end])
                features['row_idx'] = row_idx
                return features
            except (ValueError, json.JSONDecodeError):
                if attempt < 2:
                    time.sleep(1)
                    continue
                return {'row_idx': row_idx, 'error': 'parse_failed'}

        except anthropic.RateLimitError:
            time.sleep(5 * (attempt + 1))
            continue

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return {'row_idx': row_idx, 'error': str(e)[:100]}

    return {'row_idx': row_idx, 'error': 'max_retries'}


def main():
    # Load data
    df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / 'data' / 'claims_cleaned.csv')
    df_raw = pd.read_excel(
        Path(__file__).resolve().parent.parent.parent / 'claim_use_case_dataset.xlsx',
        engine='openpyxl'
    )

    texts = (df['issueDesc'].fillna('') + ' ' + df_raw['other'].fillna('').astype(str)).str.strip().tolist()

    # Load cache if exists
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as f:
            cache = {item['row_idx']: item for item in json.load(f)}
        print(f'Loaded {len(cache)} cached results')

    # Determine which rows need processing
    to_process = [(i, texts[i]) for i in range(len(texts)) if i not in cache]
    print(f'Total: {len(texts)}, Cached: {len(cache)}, To process: {len(to_process)}')

    if not to_process:
        print('All rows cached, building output...')
    else:
        results = list(cache.values())
        batch_size = 50
        total_batches = (len(to_process) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch = to_process[batch_start:batch_start + batch_size]

            print(f'Batch {batch_num+1}/{total_batches} ({len(batch)} items)...')

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for idx, text in batch:
                    row = df.iloc[idx]
                    future = executor.submit(
                        extract_features,
                        idx, text,
                        row.get('claimType', ''),
                        row.get('coverage', ''),
                        row.get('deviceType', 'SMARTPHONES'),
                        row.get('make', 'WUAWEI'),
                        row.get('model', ''),
                    )
                    futures[future] = idx

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    cache[result['row_idx']] = result

            # Save cache after each batch
            with open(CACHE_PATH, 'w') as f:
                json.dump(list(cache.values()), f)

            errors = sum(1 for r in results if 'error' in r)
            print(f'  Processed: {len(results)}, Errors: {errors}')

            # Small delay between batches
            if batch_num < total_batches - 1:
                time.sleep(1)

    # Build output DataFrame
    all_results = list(cache.values())
    features_df = pd.DataFrame(all_results).sort_values('row_idx').reset_index(drop=True)

    # Report
    print(f'\nExtraction complete: {len(features_df)} rows')
    if 'error' in features_df.columns:
        error_count = features_df['error'].notna().sum()
        print(f'Errors: {error_count} ({error_count/len(features_df):.1%})')

    # Save
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved to {OUTPUT_PATH}')

    # Preview
    print('\nSample output:')
    for col in features_df.columns:
        if col not in ['row_idx', 'error']:
            print(f'  {col}: {features_df[col].value_counts().head(5).to_dict()}')


if __name__ == '__main__':
    main()
