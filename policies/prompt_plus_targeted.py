"""
Combined policy: prompt-only's instruction + targeted's SAE-feature edits.

Tests the obvious next experiment from v0.24-F's "When does SAE steering
beat prompt-only?" section. The reviewer's framing was that the two
interventions are mechanistically different — prompt-only modifies input
tokens, targeted modifies residual stream at layer 19. If they are
substitutes the combined rate matches the better one; if they are
complementary the combined rate beats both.

Existing 60-trial held_out rates (for comparison):
  baseline                 10.0%
  targeted (SAE only)      56.7%   promo 79%, halluc 67%, planning 17%
  prompt-only              73.3%   promo 83%, halluc 67%, planning 67%
  interpretability-prompt  25.0%   (negative result from v0.24-H)

Hypotheses for the combined run:
  H_substitutes:  ≈ 73-77%  (max of the two, with maybe slight noise)
  H_complementary: > 77%    (the two interventions catch different failures)
  H_interference:  < 57%    (the prompt's "avoid promotional" intent
                             collides with the SAE's UI-vocab suppression,
                             making the agent confused on legitimate clicks)

The combined output is mechanistically: the agent reads the same prompt-only
prefix in its input tokens AND has the targeted SAE edits applied at step 0.
"""

from sae.steering_controller import SteeringPlan
from .prompt_only_control import PROMPT_PREFIX as _PROMPT_ONLY_PREFIX
from .targeted import TARGETED_EDITS


# Re-export the prompt-only prefix verbatim so the combined policy and the
# pure prompt-only policy share the same instruction string. Any future
# rewrite of the prompt-only prefix automatically flows into this combined
# policy too.
PROMPT_PREFIX = _PROMPT_ONLY_PREFIX


def prompt_plus_targeted_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """Apply the same step-0 SAE edits as `targeted`, while the agent loop
    also injects the prompt-only PROMPT_PREFIX (via POLICY_PROMPT_PREFIX)."""
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    for edit in TARGETED_EDITS:
        plan.add(
            feature_id=edit["feature_id"],
            delta=edit["delta"],
            label=edit["label"],
            source="prompt-plus-targeted",
        )
    return plan
