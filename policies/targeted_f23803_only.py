"""
Per-feature ablation: f23803 alone (distraction-avoidance-vocab amplification only).

Companion to policies/targeted_f26737_only.py — together with the combined
`targeted` policy they form the 3-way ablation. See that file's docstring
for the rationale.
"""

from sae.steering_controller import SteeringPlan


def targeted_f23803_only_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """Amplify f23803 alone at step 0. Distraction-avoidance-vocab amplification."""
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    plan.add(
        feature_id=23803,
        delta=6.0,
        label="f23803_distraction_avoidance_vocab",
        source="targeted_f23803_only",
    )
    return plan
