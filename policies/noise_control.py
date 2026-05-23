"""
Noise control policy: matched-norm random residual perturbation.

This is the strongest mechanistic control. Instead of choosing random SAE
features (which still lives in the decoder subspace), we inject a random
GAUSSIAN VECTOR in raw residual space at the same position the targeted
policy modifies, scaled to the same L2 norm as the targeted residual delta.

If targeted beats this noise control on the benchmark, specific DIRECTIONS
matter, not just magnitude. If targeted ≈ noise control, then any
sufficiently-large perturbation breaks the promo-click bias.

Implementation note: the noise injection happens directly on the brain-server
via the steer_act_with_noise method, not through the SteeringPlan/feature-edit
machinery. The policy returns an empty SteeringPlan and signals the runner via
a special marker that the noise endpoint should be called instead.
"""

from sae.steering_controller import SteeringPlan

# The targeted residual delta is roughly:
#   -6 * W_dec[26737] + 6 * W_dec[23803]
# Both decoder rows have norm ~1, so |delta| is roughly 6*sqrt(2) ≈ 8.5,
# minus some cancellation if the vectors are correlated. Matching at 6.0 is
# a conservative (slightly under) noise norm; matching at 8.5 over-perturbs.
# Use 6.0 as the canonical control magnitude.
NOISE_NORM = 6.0


def noise_control_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    trial_seed: int = 0,
    **_,
) -> SteeringPlan:
    """Returns an empty plan. The noise injection is handled by the runner
    via a special path that calls brain_server.steer_act_with_noise."""
    plan = SteeringPlan()
    # Store the noise parameters on the plan as metadata so the runner picks them up.
    plan._noise_seed = int(trial_seed)
    plan._noise_norm = NOISE_NORM
    plan._noise_active = (step_idx == 0)
    return plan
