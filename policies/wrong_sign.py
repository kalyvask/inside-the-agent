"""
Wrong-sign steering: ablation condition.

Takes the targeted policy's edits and FLIPS THE SIGN. If the targeted direction
matters, flipping it should make things worse (or at least not better). This
distinguishes "any intervention helps" from "this specific intervention helps."
"""

from policies.dynamic import dynamic_policy
from sae.steering_controller import SteeringPlan


def wrong_sign_policy(features_dict: dict, step_idx: int, catalog: dict | None = None) -> SteeringPlan:
    base = dynamic_policy(features_dict, step_idx, catalog)
    flipped = SteeringPlan(clamp_min=base.clamp_min, clamp_max=base.clamp_max)
    for e in base.edits:
        flipped.add(
            feature_id=e.feature_id,
            delta=-e.delta,  # FLIP
            label=e.label,
            source="wrong-sign",
        )
    return flipped
