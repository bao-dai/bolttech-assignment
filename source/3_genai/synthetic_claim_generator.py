import json
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

CLIENT = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
MODEL = "claude-haiku-4-5-20251001"
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'synthetic'


# -- Prompt Templates ----------------------------------------

DECLINED_SCENARIO_PROMPT = """You are an insurance data generation expert. Generate a realistic DECLINED insurance claim scenario for a device protection policy.

The claim should be declined for one of these common reasons:
- Gradual wear and tear (not a sudden accident)
- Pre-existing damage claimed as new
- Damage inconsistent with described incident
- Claim filed outside policy coverage period
- Vague or suspicious description lacking specific details
- Device issue is a manufacturing defect (warranty, not insurance)

Context from real data:
- Countries: SE (Sweden), NL (Netherlands), FI (Finland)
- Devices: WUAWEI smartphones, tablets, laptops, wearables
- Coverage types: ADLD (accidental damage + liquid damage), ADLD/THEFT
- Claim types: Accidental Damage, Theft, Liquid Damage
- Descriptions are typically in Swedish, Dutch, or English

Generate a JSON object with these fields:
{{
  "claimType": one of ["Accidental Damage", "Theft", "Liquid Damage"],
  "coverage": one of ["ADLD", "ADLD/THEFT"],
  "country": one of ["SE", "NL", "FI"],
  "deviceType": one of ["SMARTPHONES", "TABLET", "LAPTOP", "WEARABLES"],
  "make": "WUAWEI",
  "model": "WUAWEI-SYN-{number}",
  "rrp": a realistic price between 500 and 25000,
  "excessFee": a realistic excess between 50 and 2000,
  "channel": one of ["Online Portal", "Phone Call", "Email"],
  "policyStatus": "Active",
  "issueDesc": a realistic claim description (100-300 words) in {language} that would lead to DECLINE. Include subtle red flags that an adjuster would notice.,
  "decline_reason": brief explanation of why this should be declined,
  "status": "Declined",
  "turnOnOff": true or false,
  "touchScreen": true or false,
  "frontCamera": true or false,
  "backCamera": true or false,
  "audio": true or false,
  "mic": true or false
}}

Make the scenario realistic and nuanced  -  not obviously fraudulent, but containing the kind of subtle signals that distinguish legitimate from illegitimate claims.

Return ONLY the JSON object."""

BORDERLINE_SCENARIO_PROMPT = """You are an insurance data generation expert. Generate a realistic BORDERLINE insurance claim scenario  -  one where the approval decision is genuinely uncertain.

The claim should have:
- Some legitimate elements (real accident, policy is active)
- BUT also some concerning elements (vague details, delayed reporting, partial inconsistency)
- A reasonable adjuster could argue either way

Context from real data:
- Countries: SE (Sweden), NL (Netherlands), FI (Finland)
- Devices: WUAWEI smartphones, tablets, laptops, wearables
- Coverage types: ADLD, ADLD/THEFT
- Claim types: Accidental Damage, Theft, Liquid Damage

Generate a JSON object with these fields:
{{
  "claimType": one of ["Accidental Damage", "Theft", "Liquid Damage"],
  "coverage": one of ["ADLD", "ADLD/THEFT"],
  "country": one of ["SE", "NL", "FI"],
  "deviceType": one of ["SMARTPHONES", "TABLET", "LAPTOP", "WEARABLES"],
  "make": "WUAWEI",
  "model": "WUAWEI-SYN-{number}",
  "rrp": a realistic price between 500 and 25000,
  "excessFee": a realistic excess between 50 and 2000,
  "channel": one of ["Online Portal", "Phone Call", "Email"],
  "policyStatus": "Active",
  "issueDesc": a realistic claim description (100-300 words) in {language} with BOTH legitimate and concerning elements,
  "borderline_factors": list of factors making this borderline (both for and against approval),
  "status": "Borderline",
  "turnOnOff": true or false,
  "touchScreen": true or false,
  "frontCamera": true or false,
  "backCamera": true or false,
  "audio": true or false,
  "mic": true or false
}}

Return ONLY the JSON object."""

EDGE_CASE_PROMPT = """You are an insurance data generation expert. Generate a realistic EDGE CASE insurance claim scenario that would be unusual or challenging for an ML model.

Focus on one of these underrepresented patterns:
- Liquid Damage (only 2.3% of real data)
- Theft with unusual circumstances (only 3.4% of real data)
- Wearable/Earbuds/Laptop claims (rare device types)
- Multi-device incident
- Claim involving a third party
- Very high value device (RRP > 30000)
- Claim filed via unusual channel (Whatsapp, Facebook)

Context: Countries are SE/NL/FI, devices are WUAWEI brand.

Generate a JSON object with these fields:
{{
  "claimType": one of ["Accidental Damage", "Theft", "Liquid Damage"],
  "coverage": one of ["ADLD", "ADLD/THEFT"],
  "country": one of ["SE", "NL", "FI"],
  "deviceType": one of ["SMARTPHONES", "TABLET", "LAPTOP", "WEARABLES", "EARBUDS"],
  "make": "WUAWEI",
  "model": "WUAWEI-SYN-{number}",
  "rrp": a realistic price,
  "excessFee": a realistic excess,
  "channel": one of ["Online Portal", "Phone Call", "Email", "Whatsapp"],
  "policyStatus": "Active",
  "issueDesc": a realistic claim description (100-300 words) in {language},
  "edge_case_type": what makes this an edge case,
  "expected_outcome": "Completed" or "Declined" with brief justification,
  "status": "Completed" or "Declined",
  "turnOnOff": true or false or null,
  "touchScreen": true or false or null,
  "frontCamera": true or false or null,
  "backCamera": true or false or null,
  "audio": true or false or null,
  "mic": true or false or null
}}

Return ONLY the JSON object."""


def generate_scenario(scenario_type, index, language='English'):
    """Generate a single synthetic scenario."""
    if scenario_type == 'declined':
        prompt = DECLINED_SCENARIO_PROMPT.format(number=index, language=language)
    elif scenario_type == 'borderline':
        prompt = BORDERLINE_SCENARIO_PROMPT.format(number=index, language=language)
    elif scenario_type == 'edge_cases':
        prompt = EDGE_CASE_PROMPT.format(number=index, language=language)
    else:
        raise ValueError(f"Unknown type: {scenario_type}")

    for attempt in range(3):
        try:
            response = CLIENT.messages.create(
                model=MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            # Parse JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.index('{')
                end = content.rindex('}') + 1
                data = json.loads(content[start:end])

            data['synthetic_id'] = f'SYN-{scenario_type[:3].upper()}-{index:04d}'
            data['scenario_type'] = scenario_type
            return data

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return {
                'synthetic_id': f'SYN-{scenario_type[:3].upper()}-{index:04d}',
                'scenario_type': scenario_type,
                'error': str(e)[:200]
            }

    return {'synthetic_id': f'SYN-ERR-{index}', 'error': 'max_retries'}


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic claim scenarios')
    parser.add_argument('--n', type=int, default=50, help='Number of scenarios to generate')
    parser.add_argument('--focus', type=str, default='all',
                        choices=['declined', 'borderline', 'edge_cases', 'all'],
                        help='Type of scenarios to focus on')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Determine scenario mix
    if args.focus == 'all':
        n_declined = args.n // 2
        n_borderline = args.n // 4
        n_edge = args.n - n_declined - n_borderline
        tasks = (
            [('declined', i) for i in range(n_declined)]
            + [('borderline', i) for i in range(n_borderline)]
            + [('edge_cases', i) for i in range(n_edge)]
        )
    else:
        tasks = [(args.focus, i) for i in range(args.n)]

    # Vary languages
    languages = ['English', 'Swedish', 'Dutch']

    print(f'Generating {len(tasks)} synthetic scenarios...')
    print(f'  Types: {dict(pd.Series([t[0] for t in tasks]).value_counts())}')

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for scenario_type, idx in tasks:
            lang = languages[idx % len(languages)]
            future = executor.submit(generate_scenario, scenario_type, idx, lang)
            futures[future] = (scenario_type, idx)

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 10 == 0:
                errors = sum(1 for r in results if 'error' in r)
                print(f'  Progress: {i+1}/{len(tasks)} (errors: {errors})')

    # Build DataFrame
    df = pd.DataFrame(results)
    error_count = df['error'].notna().sum() if 'error' in df.columns else 0
    print(f'\nGenerated: {len(df)} scenarios, Errors: {error_count}')

    # Save
    output_file = OUTPUT_DIR / 'synthetic_claims.csv'
    df.to_csv(output_file, index=False)
    print(f'Saved to {output_file}')

    # Summary
    if 'scenario_type' in df.columns:
        print(f'\nBy type:')
        print(df['scenario_type'].value_counts().to_string())

    if 'claimType' in df.columns:
        print(f'\nBy claim type:')
        print(df['claimType'].value_counts().to_string())

    if 'status' in df.columns:
        print(f'\nBy status:')
        print(df['status'].value_counts().to_string())

    # Show a sample
    if 'issueDesc' in df.columns:
        print(f'\n{"="*60}')
        print('Sample synthetic claim:')
        print(f'{"="*60}')
        sample = df[df.get('error', pd.Series([None]*len(df))).isna()].iloc[0]
        for col in ['synthetic_id', 'scenario_type', 'claimType', 'coverage',
                     'country', 'deviceType', 'rrp', 'status']:
            if col in sample.index:
                print(f'  {col}: {sample[col]}')
        if 'issueDesc' in sample.index:
            print(f'  issueDesc: {str(sample["issueDesc"])[:300]}...')
        for col in ['decline_reason', 'borderline_factors', 'edge_case_type']:
            if col in sample.index and pd.notna(sample[col]):
                print(f'  {col}: {sample[col]}')


if __name__ == '__main__':
    main()
