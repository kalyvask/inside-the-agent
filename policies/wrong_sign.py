"""
Wrong-sign steering: ablation condition for the TARGETED policy.

Takes the targeted policy's edits and FLIPS THE SIGN. If the targeted direction
is what matters (not "any intervention"), flipping should hurt performance.
The chart we'll report shows: targeted >> baseline ≈ wrong-sign ≈ random.

Mirrors the step-0-only behavior of the targeted policy.
"""

from policies.targeted import TARGETED_EDITS
from sae.steering_controller import SteeringPlan


def wrong_sign_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    plan = SteeringPlan()
    if step_idx != 0:
        return plan  # mirror targeted's step-0-only application
    for edit in TARGETED_EDITS:
        plan.add(
            feature_id=edit["feature_id"],
            delta=-edit["delta"],  # FLIP
            label=edit["label"],
            source="wrong-sign",
        )
    return plan
