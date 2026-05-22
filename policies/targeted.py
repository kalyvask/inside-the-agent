"""
Targeted policy: empirically validated steering combination for promo-trap
avoidance.

Discovered via verify/step0_calibration.py — these are the features whose
intervention flips the agent's Step-0 decision from "click promo button" to
"type in search bar" while keeping output coherent.

Features used (Llama 3.1-8B-Instruct + Goodfire SAE l19):
  - hallucination f26737 at -6.0  (suppresses "click invented buttons")
  - goal_tracking f23803 at +6.0  (reinforces "stay on goal")

These two together are the strongest validated steering for the demo.
Use as `--policy targeted` in the runner.
"""

from sae.steering_controller import SteeringPlan


# Empirically validated step-0 winners.
TARGETED_EDITS = [
    {"feature_id": 26737, "delta": -6.0, "label": "hallucination"},
    {"feature_id": 23803, "delta": +6.0, "label": "goal_tracking"},
]


def targeted_policy(features_dict: dict, step_idx: int, catalog: dict | None = None) -> SteeringPlan:
    """
    Apply the calibrated combination ONLY at Step 0 where the promo trap lives.

    Suppressing "hallucination f26737" prevents the agent from clicking
    invented buttons (incl. the promo trap), but also prevents it from
    clicking REAL buttons later. So we only intervene at the trap moment;
    once the agent is past Step 0 the page is search-results / cart-only
    and there's no trap to defend against.
    """
    plan = SteeringPlan()
    if step_idx != 0:
        return plan  # baseline behavior from step 1 onward
    for edit in TARGETED_EDITS:
        plan.add(
            feature_id=edit["feature_id"],
            delta=edit["delta"],
            label=edit["label"],
            source="targeted",
        )
    return plan
