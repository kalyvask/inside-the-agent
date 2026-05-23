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
    d_features: int = 65536,
    n_random: int = 2,  # mirror targeted's 2-feature count
    magnitude_range: tuple[float, float] = (-6.0, 6.0),  # mirror targeted's magnitudes
) -> SteeringPlan:
    """
    Random control: same surface as targeted (2 features, ±6 magnitudes,
    step-0 only) but features chosen uniformly at random across the SAE.

    If random helps comparably to targeted, the result isn't about the
    SPECIFIC features — it's about ANY perturbation. We expect random to
    perform near baseline.
    """
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    rng = random.Random(seed)
    for _ in range(n_random):
        fid = rng.randint(0, d_features - 1)
        delta = rng.uniform(*magnitude_range)
        plan.add(feature_id=fid, delta=delta, label="random", source="random")
    return plan
