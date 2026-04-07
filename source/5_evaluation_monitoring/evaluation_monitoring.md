# Evaluation & Monitoring

How we'd monitor this system if it went to production.

## ML model metrics

Track these weekly by running the model against claims that already got a human decision:

| Metric | Target | Alert if below |
|--------|--------|----------------|
| F1 Macro | > 0.65 | 0.60 |
| F1 Declined | > 0.40 | 0.35 |
| Declined Recall | > 35% | 30% |
| Declined Precision | > 45% | 35% |
| PR-AUC | > 0.90 | 0.85 |

Also track per country (NL/SE/FI) and per claim type separately. If one segment drops while others stay stable, that's useful signal.

### Drift detection

Three things to watch for:

- **Data drift** - input feature distributions shift. Use PSI (population stability index), alert if any feature PSI > 0.2
- **Concept drift** - the relationship between features and outcome changes. Catch this through weekly F1 evaluation
- **Label drift** - the approval/decline ratio changes. If suddenly 25% of claims are declined instead of 16%, something changed in the business rules or claim population

## GenAI output quality

### For explanations

Hardest part to automate. What I'd check:

- **Groundedness** - does the explanation actually reference things from the claim data and SHAP values? Should be 100%, anything else is hallucination
- **Persona fit** - customer explanations shouldn't have SHAP numbers or internal jargon. Adjuster notes should have specific data points. Manual review on maybe 5% sample
- **Actionability** - declined customer explanations should always say what they can do next (appeal, provide more info etc)
- **Length** - customer < 200 words, adjuster < 250, manager < 100. Check if the LLM is going over

### For LLM feature extraction

- Parse success rate should be > 99% (did the LLM return valid JSON)
- Feature accuracy - compare LLM's damage_type, is_gradual_wear against human labels. Target > 85% agreement
- Latency < 3s per claim with Haiku

### Hallucination checks

- Automated: make sure every fact in the explanation exists in the actual claim data
- Flag cases where SHAP says "toward approval" but the explanation says the opposite
- Flag if the explanation mentions dollar amounts or dates that don't match the claim record

## System health

Standard API monitoring stuff:

- Response time (P50/P95/P99) - alert if P99 > 5s
- Error rate - alert if > 1%
- Claude API availability - track our timeout rate, alert if > 0.5%
- Claude API cost - alert if it spikes above 2x the daily average
- Memory - alert at 80% container memory

## Fairness

Things that could go wrong:

- **Country bias** - model might treat NL claims differently from SE/FI. Monitor decline rates per country, alert if they diverge more than 10% from training distribution
- **Device value bias** - expensive phones might get approved more easily. Track correlation between RRP and approval
- **Language bias** - the LLM feature extraction might work better on English or Dutch than on Finnish. Check extraction accuracy per language
- **Text length** - longer descriptions probably give the model more to work with. Check if short descriptions get unfairly declined

## Transparency

Every prediction is traceable:
1. Raw claim data goes in
2. LLM extracts structured features (auditable - you can read the raw text and check)
3. Model makes prediction with SHAP values showing exactly why
4. Explanation is generated from those SHAP values (grounded, not made up)

No black box anywhere in the pipeline.

## Human oversight

| When | What happens |
|------|-------------|
| Prediction near the threshold (within 10%) | Goes to human reviewer |
| High value claim (RRP > 20k) | Always needs human sign-off |
| LLM extraction disagrees with device diagnostics | Flagged for adjuster |
| Customer appeals a decline | Goes to human with full model context attached |

## Improvement loop

Basically: every week, check if the model is still performing. If F1 drops, figure out whether the data changed (data drift) or the patterns changed (concept drift). Either way, retrain on fresh data, A/B test the new model against the old one, deploy if it's better. Also do a full retrain quarterly regardless.

## For production

- Weekly eval pipeline (Step Functions + Lambda)
- CloudWatch dashboards
- Anthropic API cost alerts
- Human review queue for borderline/high-value claims
- Prompt versioning + A/B testing
- Quarterly retrain schedule
- Fairness monitoring per country/device
- Runbook for drift incidents
