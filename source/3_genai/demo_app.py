import gradio as gr
import numpy as np
from multi_persona_explainer import (
    load_model_artifacts, get_claim_context, generate_explanation
)

print("Loading model artifacts...")
ARTIFACTS = load_model_artifacts()
CLAIMS_DF = ARTIFACTS['claims_df']
print(f"Ready. {len(CLAIMS_DF)} claims loaded.\n")


def predict_and_explain(claim_idx, persona):
    """Core function: predict + explain a claim for a given persona."""
    claim_idx = int(claim_idx)
    if claim_idx < 0 or claim_idx >= len(CLAIMS_DF):
        return "Invalid index", "", "", ""

    ctx = get_claim_context(ARTIFACTS, claim_idx)

    # Summary card
    summary = (
        f"**Prediction: {ctx['prediction']}** "
        f"(P(approved) = {ctx['approval_probability']:.3f}, threshold = {ctx['threshold']:.2f})\n\n"
        f"**Actual outcome:** {ctx['actual_status']}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Claim type | {ctx['claim_type']} |\n"
        f"| Coverage | {ctx['coverage']} |\n"
        f"| Country | {ctx['country']} |\n"
        f"| Device | {ctx['make']} {ctx['model_name']} ({ctx['device_type']}) |\n"
        f"| RRP | {ctx['rrp']:.0f} | Excess | {ctx['excess_fee']:.0f} |\n"
        f"| Policy | {ctx['policy_status']} |\n"
    )

    # LLM damage assessment
    damage_info = (
        f"| Feature | Value |\n|---|---|\n"
        f"| Damage type | {ctx['llm_damage_type']} |\n"
        f"| Severity | {ctx['llm_damage_severity']} |\n"
        f"| Incident type | {ctx['llm_incident_type']} |\n"
        f"| Incident clarity | {ctx['llm_incident_clarity']} |\n"
        f"| Gradual wear | {ctx['llm_is_gradual_wear']} |\n"
        f"| User at fault | {ctx['llm_user_at_fault']} |\n"
        f"| Emotional tone | {ctx['llm_emotional_tone']} |\n"
        f"| Device functional | {ctx['llm_device_functional']} |\n"
    )

    # SHAP factors
    factors = ""
    for f in ctx['top_factors'][:6]:
        emoji = "🔴" if "decline" in f['direction'] else "🟢"
        factors += f"{emoji} **{f['feature']}**: {f['shap_value']:+.4f} ({f['direction']})\n\n"

    # Generate explanation
    explanation = generate_explanation(ctx, persona)

    return summary, damage_info, factors, explanation


def get_random_claim(claim_type):
    """Get a random claim index of a specific type."""
    if claim_type == "Declined":
        indices = CLAIMS_DF[CLAIMS_DF['target'] == 0].index.tolist()
    elif claim_type == "Approved":
        indices = CLAIMS_DF[CLAIMS_DF['target'] == 1].index.tolist()
    else:  # Borderline
        config = ARTIFACTS['config']
        feature_cols = config['feature_columns']
        X_all = CLAIMS_DF[feature_cols].fillna(0).values
        probas = ARTIFACTS['model'].predict_proba(X_all)[:, 1]
        distances = np.abs(probas - config['optimal_threshold'])
        indices = np.argsort(distances)[:20].tolist()

    idx = int(np.random.choice(indices))
    return idx


# Build Gradio interface
with gr.Blocks(
    title="Claim Decision Explainer",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🏥 Claim Decision Explainer\n"
        "**GenAI-powered multi-persona explanations** for insurance claim approval decisions.\n\n"
        "Select a claim and persona to generate a tailored explanation grounded in "
        "ML model predictions, SHAP feature importance, and LLM-extracted damage assessment."
    )

    with gr.Row():
        with gr.Column(scale=1):
            claim_idx = gr.Number(label="Claim Index", value=13, precision=0)
            persona = gr.Radio(
                choices=["customer", "adjuster", "manager"],
                value="customer",
                label="Persona"
            )
            explain_btn = gr.Button("Generate Explanation", variant="primary")

            gr.Markdown("### Quick Pick")
            with gr.Row():
                btn_declined = gr.Button("Random Declined")
                btn_approved = gr.Button("Random Approved")
                btn_borderline = gr.Button("Random Borderline")

        with gr.Column(scale=2):
            summary_out = gr.Markdown(label="Claim Summary")
            with gr.Row():
                damage_out = gr.Markdown(label="LLM Damage Assessment")
                factors_out = gr.Markdown(label="Top SHAP Factors")

    gr.Markdown("### Explanation")
    explanation_out = gr.Markdown()

    # Wire up events
    explain_btn.click(
        fn=predict_and_explain,
        inputs=[claim_idx, persona],
        outputs=[summary_out, damage_out, factors_out, explanation_out],
    )

    def pick_and_explain(claim_type, persona_val):
        idx = get_random_claim(claim_type)
        results = predict_and_explain(idx, persona_val)
        return idx, *results

    btn_declined.click(
        fn=lambda p: pick_and_explain("Declined", p),
        inputs=[persona],
        outputs=[claim_idx, summary_out, damage_out, factors_out, explanation_out],
    )
    btn_approved.click(
        fn=lambda p: pick_and_explain("Approved", p),
        inputs=[persona],
        outputs=[claim_idx, summary_out, damage_out, factors_out, explanation_out],
    )
    btn_borderline.click(
        fn=lambda p: pick_and_explain("Borderline", p),
        inputs=[persona],
        outputs=[claim_idx, summary_out, damage_out, factors_out, explanation_out],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
