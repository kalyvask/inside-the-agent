"""
Random-feature steering: control condition for the 4-condition benchmark.

Applies the same surface as TARGETED — 2 features, ±6 magnitudes, step 0 only —
but draws features uniformly at random across the SAE.

The seed is per-trial. Every (task_id, trial_seed) pair gets a different feature
set. If random helps comparably to targeted across many trials, the targeting
isn't what matters; if random is significantly lower, the specific features do.
"""

import random

from sae.steering_controller import SteeringPlan


def random_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    trial_seed: int = 0,
    d_features: int = 65536,
    n_random: int = 2,
    magnitude_range: tuple[float, float] = (-6.0, 6.0),
    **_,
) -> SteeringPlan:
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    # Mix trial seed with a per-policy salt so two policies that both ask for
    # randomness at the same trial don't reuse the same draw.
    rng = random.Random(f"random_policy:{int(trial_seed)}")
    for _i in range(n_random):
        fid = rng.randint(0, d_features - 1)
        delta = rng.uniform(*magnitude_range)
        plan.add(feature_id=fid, delta=delta, label=f"random_f{fid}", source="random")
    return plan
