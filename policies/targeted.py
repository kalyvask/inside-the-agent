"""
Targeted policy: empirically validated steering combination for promo-trap
avoidance, applied at Step 0 only.

Features used (Llama 3.1-8B-Instruct + Goodfire SAE l19):
  - f26737  delta -6.0   labelled "f26737_invented_action_supp"
  - f23803  delta +6.0   labelled "f23803_goal_anchor"

These IDs were validated by verify/step0_calibration.py — they reliably flip
the agent's Step-0 decision from "click promotional button" to "type in search"
while preserving output coherence.

NOTE on labels: earlier drafts called these `hallucination` and `goal_tracking`
based on contrast prompts that weren't strongly verified. Renaming to neutral
functional IDs until each feature has been independently characterized
(activation example panel, per-feature ablation studies, decoder-vector
visualization). They are NOT yet "the hallucination feature" or "the goal-
tracking feature" — they are features whose suppression / amplification at
Step 0 causally alters the agent's first action.
"""

from sae.steering_controller import SteeringPlan


TARGETED_EDITS = [
    {"feature_id": 26737, "delta": -6.0, "label": "f26737_invented_action_supp"},
    {"feature_id": 23803, "delta": +6.0, "label": "f23803_goal_anchor"},
]


def targeted_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """
    Apply the validated combination at Step 0 only.

    After Step 0 we run with zero steering so real button clicks
    (add-to-cart) are not also suppressed.
    """
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    for edit in TARGETED_EDITS:
        plan.add(
            feature_id=edit["feature_id"],
            delta=edit["delta"],
            label=edit["label"],
            source="targeted",
        )
    return plan
