"""
Random-feature steering: control condition for the 4-condition benchmark.

Applies the same magnitudes as the targeted policy, but on RANDOMLY selected
features. If targeted wins and random does not, the targeting is what matters.
"""

import random
from sae.steering_controller import SteeringPlan


def random_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    seed: int = 0,
    d_features: int = 65536,  # filled at runtime from SAE
    n_random: int = 5,
    magnitude_range: tuple[float, float] = (-5.0, 5.0),
) -> SteeringPlan:
    rng = random.Random(seed + step_idx)
    plan = SteeringPlan()
    for _ in range(n_random):
        fid = rng.randint(0, d_features - 1)
        delta = rng.uniform(*magnitude_range)
        plan.add(feature_id=fid, delta=delta, label="random", source="random")
    return plan
