# Design Document - GenAI-Powered Claim Approval Agent

## Table of Contents

1. Problem & Data Understanding
2. ML Modelling - Iterative Approach
3. GenAI - Explanations & Synthetic Data
4. API & Deployment
5. Evaluation & Monitoring
6. Key Decisions & Tradeoffs

---

## 1. Problem & Data Understanding

### The Problem

Insurance companies process thousands of claims. Each one needs to be approved or declined, and the decision needs to be explainable. Right now this is manual and inconsistent. The goal is to automate the prediction and provide clear, audience-appropriate explanations.

### The Dataset

2,880 insurance claims from three countries (Netherlands, Sweden, Finland). All devices are WUAWEI brand (anonymized). 35 columns covering policy info, device diagnostics, financials, and a free-text claim description.

The target is `status`: Completed (2,427 claims, 84.3%) vs Declined (453 claims, 15.7%). This imbalance is the central challenge - a model that always predicts "approved" gets 84% accuracy but catches zero declines.

### What the EDA Revealed

**Dead features (10 of them):** `deviceCost` is literally all zeros. `balanceRRP` equals `oldBalanceRRP` for every row. `smashed` is always 0 when assessed. `buttons`, `connection`, `charging` have 99%+ zeros. `relationship` is "self" for everyone. Keeping these just adds noise so I dropped them all.

**The text is the richest signal:** The `issueDesc` column has multilingual claim narratives (Swedish, Dutch, Finnish, English) describing what happened. These are what a human adjuster actually reads to make a decision. Most are 100-300 words, 94% have PII redacted with ***.

**Missingness is informative:** The boolean device condition columns (touchScreen, frontCamera, audio, etc) are ~47% missing. But this isn't random - theft claims don't have a device to diagnose, so all diagnostics are null. The count of "not assessed" features turns out to be a useful predictor.

**Class imbalance by segment:** Decline rates vary a lot. SE + Theft has a 34.9% decline rate, NL + Liquid Damage is 23.8%, while SE + Accidental Damage is only 12.7%. Country and claim type interact.

**Correlation issues:** `rrp` and `balanceRRP` are nearly identical. Kept `rrp` as the more interpretable one.

After cleaning and feature engineering I ended up with 45 features across financial, device condition, text statistics, policy, and encoded categoricals. After dropping the dead ones, 33 went into modelling.

---

## 2. ML Modelling - Iterative Approach

I did 4 iterations. Each one taught me something that shaped the next. This section walks through the full journey.

### Attempt 1: Structural Features Only

**Approach:** Take the 33 cleaned features (no text content, just metadata and diagnostics) and train LightGBM, Logistic Regression, and XGBoost.

**Results:**
- Best: LightGBM with `is_unbalance=True` - F1 macro 0.566, F1 declined 0.280
- Logistic Regression actually had better declined recall (53%) but worse overall F1
- Optuna-tuned LightGBM performed WORSE on declined class (F1 declined 0.210 vs 0.280 default). The optimizer maximized macro F1 by sacrificing minority class recall. This pattern repeats in every attempt.

**Why it failed:** We're completely ignoring what the claim description says. The `issueDesc` column is probably the most informative feature - it's what adjusters read to decide - and we only extracted surface stats like word count and character length. Those don't tell you whether someone described a clear accident or vague gradual wear.

**What I learned:** Need to actually use the text content, not just metadata about it.

### Attempt 2: + Sentence-Transformer Embeddings

**Approach:** Used `paraphrase-multilingual-MiniLM-L12-v2` to embed each claim description into a 384-dim vector. This model handles Swedish, Dutch, Finnish, and English natively. Reduced to 20 dimensions with PCA (2,880 rows can't support 384 features) and added as features alongside the 33 structural ones.

Also recovered the `other` column which was a free-text damage description that got wrongly treated as boolean in the EDA. Concatenated it with `issueDesc` before embedding.

**Results:**
- Best: LightGBM + Embeddings (Default) - F1 macro 0.595, F1 declined 0.294
- SHAP showed embeddings contribute 63% of feature importance vs 37% structural. The text really does dominate.
- Tested SMOTE oversampling - it actually hurt (F1 declined 0.255 vs 0.294 without). With PCA-compressed embeddings, SMOTE creates unrealistic synthetic samples.
- LR + Embeddings had the best declined recall (58%) - a linear boundary in embedding space works surprisingly well for this

**Why it wasn't enough:** PCA compression threw away 35% of the text variance. And sentence-level embeddings average the entire description into one vector, losing specific details. "Phone fell on concrete" and "screen developed cracks over time" might end up with similar embeddings even though they have very different outcomes.

**What I learned:** Text signal is real and dominant. But need either more dimensions or a smarter way to extract the signal.

### Attempt 3: Deeper NLP (TF-IDF, LSTM, BERT)

Tried 4 approaches from simplest to most complex to see what works best for this data.

**TF-IDF + SVD + LightGBM (simplest):** Bag-of-words with word and character n-grams, reduced to 80 dimensions. F1 macro 0.591. Surprisingly competitive - specific vocabulary (damage words like "spricka"/crack, "stulen"/stolen) carries real signal that embeddings average away.

**BiLSTM (medium):** Tokenized text, random embeddings, bidirectional LSTM. F1 macro 0.551 - the worst. With 2,880 multilingual samples and no pretrained knowledge, the LSTM can't learn meaningful representations from scratch. Expected this but wanted to confirm.

**Fine-tuned BERT (complex):** Used `distilbert-base-multilingual-cased`, fine-tuned the classification head on our data. F1 macro 0.603, best F1 declined (0.365) and 46% declined recall. Transfer learning works - the model already understands these languages and just needs to learn the claim-specific patterns.

**BERT Hybrid (BERT embeddings + structural --> LightGBM):** Extract CLS embeddings from fine-tuned BERT, combine with structural features in LightGBM. F1 macro 0.608, most stable model (std = 0.009 across folds). Best of both worlds.

**Why I moved on:** BERT fine-tuning takes ~1 hour on CPU for 5-fold CV. The improvement over v2's sentence-transformer (0.595 --> 0.608) is real but diminishing. The text signal is clearly the bottleneck but embeddings seem to have a ceiling. I needed a fundamentally different way to extract text information.

### Attempt 4: LLM Feature Extraction (the breakthrough)

**Approach:** Instead of treating text as vectors, ask an LLM to actually UNDERSTAND each claim and extract structured features. Used Claude Haiku to process all 2,880 descriptions.

The LLM extracts 10 fields per claim:
- `incident_type` (drop, theft, gradual_wear, unknown_cause, etc)
- `damage_type` (screen_crack, water_damage, cosmetic_scratch, etc)
- `damage_severity` (minor, moderate, severe)
- `incident_clarity` (vague, moderate, detailed)
- `is_gradual_wear` (true/false - the #1 red flag for decline)
- `user_at_fault` (true/false)
- `has_police_report` (for theft claims)
- `emotional_tone` (matter_of_fact, frustrated, apologetic, etc)
- `device_functional` (true, false, partial)
- `third_party_involved` (true/false)

These get label-encoded and combined with the 33 structural features for 43 total.

**Results:**
- LightGBM + Structural + LLM: F1 macro **0.677**, F1 declined **0.456**
- LLM features ALONE (just 10 features): F1 macro 0.642 - better than every v1-v3 model
- Top SHAP: incident_type (0.63), damage_type (0.28), is_gradual_wear (0.24)
- With threshold tuned to 0.40: F1 macro 0.685, declined precision 53%, declined recall 40%

**Why this worked so well:** The LLM applies insurance domain knowledge. It knows that "the screen has been getting worse over time" is gradual wear (not covered), while "I dropped it on concrete" is a clear accident (covered). No embedding model has this domain understanding. And the output is 10 interpretable features instead of a 768-dim black box.

**Full progression:**
- v1: 0.566 --> v2: 0.595 --> v3: 0.608 --> v4: 0.677 (total +19.6%)

### Why LightGBM Throughout

I stuck with LightGBM across all 4 attempts because:
1. Best performer on tabular data at this scale (2,880 rows)
2. Handles mixed feature types without preprocessing
3. `is_unbalance=True` handles the 84/16 split cleanly
4. TreeSHAP gives exact per-feature explanations (needed for GenAI pipeline)
5. Fast enough to iterate quickly with Optuna

### A Consistent Finding About Tuning

Default LightGBM with `is_unbalance=True` beat Optuna-tuned versions in every single attempt on the declined class. The optimizer finds that the easiest way to improve macro F1 is to be more conservative (predict "approved" more), which sacrifices minority recall. For production I'd use the default parameters.

---

## 3. GenAI - Explanations & Synthetic Data

### Multi-Persona Explanations

For each claim, the system generates explanations tailored to 3 audiences. The core module is `multi_persona_explainer.py`.

**How it works:**
1. Get the ML model's prediction and probability for a claim
2. Pull SHAP values showing which features pushed toward approval or decline
3. Pull LLM-extracted features (damage type, severity, gradual wear flag, etc)
4. Build a persona-specific prompt with all this context injected
5. Send to Claude Haiku, get back a tailored explanation

**The 3 personas:**

Customer gets a friendly explanation in plain language. No SHAP numbers, no internal jargon. Always includes what they can do next - appeal the decision, provide additional information, etc. For declined claims this is the most important persona because the customer wants to know why.

Claims Adjuster gets a technical note with specific data points. References SHAP contributions, the LLM damage assessment, policy details, and RRP. Flags risk factors and recommends whether the case needs manual review. Uses bullet points.

Manager gets a 3-4 sentence executive summary. Covers the decision, key flags, and a recommended action (auto-approve, auto-decline, or escalate).

**Prompt engineering strategy:**

The most important principle is grounding. Every explanation is built from actual model outputs - SHAP values, LLM features, claim data. The LLM doesn't guess why a claim was declined, it explains what the model actually computed. This prevents hallucination.

Role assignment at the top of each prompt ("You are a friendly customer service representative" vs "You are writing an internal adjuster note") controls tone and vocabulary without needing long instructions. Word limits and format requirements (bullets for adjuster, paragraphs for customer) are specified explicitly.

Guardrails: customer prompts say "no SHAP values, no internal model details." Adjuster prompts say "reference specific data points." Manager prompts say "3-4 sentences max."

### Synthetic Data Generation

Used Claude to generate 50 targeted claim scenarios across 3 types:
- 25 declined patterns (gradual wear, pre-existing damage, vague descriptions, inconsistencies)
- 12 borderline cases (legitimate accident mixed with concerning signals)
- 13 edge cases (rare device types, liquid damage on wearables, unusual channels)

The prompts provide real data context (countries, device types, coverage plans, decline reasons from actual data) and request both structured fields and a narrative description in varying languages.

**Interesting finding:** The production model predicted "Approved" for ALL 50 synthetic claims (mean P(approved) = 0.996). This happened because synthetic claims lack realistic structural feature patterns - policy dates, device diagnostics, exact RRP values are all defaults. The model depends 69.4% on structural features, so when those are missing or default, it falls back to "approve."

This is actually a useful discovery: it reveals a structural feature dependency in the model that's a deployment risk. If the data pipeline ever sends incomplete structural data, the model will approve everything. Synthetic data should be generated with realistic structural features sampled from the training distribution to be useful for augmentation.

---

## 4. API & Deployment

### What Was Built

FastAPI app with 4 endpoints:
- `POST /predict` - takes a claim index, returns prediction + probability + SHAP factors + damage assessment
- `POST /explain` - takes a claim index and persona, returns a generated explanation
- `GET /health` - health check
- `GET /model/info` - model metadata (feature list, threshold, LLM model used)

Also built a Gradio UI (`demo_app.py`) for interactive exploration. Pick a claim (or random declined/approved/borderline), select a persona, get the full prediction + explanation.

### AWS Deployment Design

If deploying to AWS, the architecture would be:

**ECS Fargate** for compute. Serverless containers, auto-scaling, no server management. The model is a 2.4MB LightGBM - SageMaker would be overkill (minimum ~$50/month for the smallest endpoint). Fargate is pay-per-use.

**API Gateway** in front for rate limiting, authentication, HTTPS termination.

**S3** for model artifact storage with versioning. Rolling back a model means changing a version pointer.

**Secrets Manager** for the Claude API key. Never in code or environment files.

**CloudWatch** for monitoring - logs, custom metrics, alarms.

**CodePipeline + CodeBuild** for CI/CD. Push to main triggers lint, test, Docker build, and blue-green deployment.

Estimated cost at 1,000 claims/day: ~$33/month total (Fargate ~$10, API Gateway ~$3, Claude Haiku API ~$15, CloudWatch ~$5).

### MLOps / LLMOps

**Model versioning:** Each model version (v1 through v4) is stored with its config, SHAP values, and comparison metrics. In production, models go to S3 with version tags. Rollback means pointing to a previous version.

**Prompt versioning:** Prompt templates live in code (`multi_persona_explainer.py`) and are versioned via git like everything else. A/B testing different prompt versions is possible by routing a percentage of traffic to each.

**CI/CD pipeline:** Push --> lint + unit tests --> model sanity checks (prediction on known claims) --> Docker build --> push to ECR --> blue-green deploy to ECS.

**Retraining:** Triggered when weekly F1 drops below 0.60, or quarterly regardless. New data arrives in S3, Step Functions orchestrates retraining, new model is compared against production, deployed if better.

---

## 5. Evaluation & Monitoring

### ML Model Monitoring

Track weekly by running the model against claims that already got a human adjuster decision:
- F1 Macro (target > 0.65, alert below 0.60)
- F1 Declined (target > 0.40, alert below 0.35)
- Declined Recall (target > 35%)
- PR-AUC (target > 0.90)

Track per-segment (country, claim type, device type). If NL performance drops while SE stays stable, that's a targeted problem.

### Drift Detection

- **Data drift** - monitor feature distributions with PSI. Alert if any feature shifts significantly (PSI > 0.2)
- **Concept drift** - weekly F1 on new labeled data catches when the relationship between features and outcomes changes
- **Label drift** - if the approval/decline ratio shifts more than 10%, either business rules changed or the claim population shifted

### GenAI Quality

Hardest to automate. Key checks:
- Groundedness: does the explanation reference actual SHAP factors and claim data? Should be 100%
- Persona adherence: customer explanations shouldn't have jargon, adjuster notes should have data points. Manual review on ~5% sample
- Hallucination: flag if the explanation mentions facts not in the claim record, or if SHAP direction disagrees with explanation reasoning
- LLM feature extraction: parse success rate > 99%, feature accuracy > 85% vs human labels

### Fairness & Responsible AI

Things to watch:
- Country bias - decline rates per country shouldn't diverge more than 10% from training distribution
- Device value bias - expensive phones shouldn't get preferential treatment
- Language bias - LLM extraction might work better for some languages
- Text length bias - short descriptions shouldn't be unfairly declined

Every prediction has a full audit trail: raw data --> LLM features --> SHAP values --> explanation. No black box anywhere.

### Human Oversight

Borderline predictions (probability within 10% of threshold) get flagged for human review. High-value claims (RRP > 20k) always need human sign-off. Customer appeals go to a human adjuster with the full model context attached.

---

## 6. Key Decisions & Tradeoffs

**LightGBM over deep learning.** 2,880 samples. Tree models outperform neural nets at this scale, and SHAP integration is native and exact.

**Claude Haiku for LLM features.** ~$0.50 per 1,000 claims. Worth it because 10 LLM features outperform 768-dim BERT embeddings. For production, could distill into a small classifier to avoid API dependency.

**SHAP-grounded explanations.** The LLM explains what the model actually decided rather than guessing. Prevents hallucination and makes every explanation auditable.

**4 iterative attempts instead of jumping to the best approach.** Shows the thought process. Each attempt reveals something about the data that motivates the next. The progression (structural --> embeddings --> BERT --> LLM features) tells a coherent story about what matters for this problem.

**Default LightGBM over Optuna-tuned.** Tuning consistently sacrificed minority class performance for marginal macro gains. For an imbalanced problem where catching declines matters, the untuned model with `is_unbalance=True` is better.

**Single model over ensemble.** Dataset is too small for meaningful ensemble benefit. Complexity isn't justified.

### Services Used

- Claude Haiku (Anthropic API) - LLM feature extraction + explanation generation
- sentence-transformers (Hugging Face) - multilingual text embeddings (v2)
- distilbert-base-multilingual-cased (Hugging Face) - BERT fine-tuning (v3)
- LightGBM - production ML model
- SHAP - model explainability
- FastAPI - REST API
- Gradio - interactive demo UI

### Assumptions

- "Completed" = claim approved/paid out, "Declined" = claim denied
- All text processed in original language (Swedish/Dutch/Finnish/English)
- Local deployment for demo; AWS design is documented but not deployed
- Claude Haiku API is available for both feature extraction and explanation generation
