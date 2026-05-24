"""
Baseline policy with the strict-JSON system-prompt prefix.

Discovered in verify/calibration_70b_fix.py: Llama-3.3-70B at temperature 0.05
and with the default system prompt emits malformed JSON ("target": "Add to
cart" on "USB-C Cable") on the calibration prompt — the action parser
rejects it. The same model with a strict-JSON prompt prefix produces clean
parseable actions on every steering condition tested.

This policy applies that strict prefix with NO SAE steering. It is the
cross-scale baseline for the 70B (the 8B uses the regular baseline because
its default-prompt output is already parseable).

The same policy can be run on the 8B too if you want a 8B-baseline-strict
comparison point, but the main use is BRAIN_APP_NAME=inside-the-agent-70b.
"""

from sae.steering_controller import SteeringPlan


# Stricter JSON-format spec. Names the specific failure mode (prose-leaking
# into the target field) and shows a correct/wrong example pair. Format
# guidance only — no behavioral instruction about traps.
#
# v0.24-K bugfix: the original example used `button#add-usb-c-cable` (CSS
# selector style) which ShopGym's env rejects. Successful trajectories use
# the bare element_id `add-usb-c-cable`. The earlier 0/60 run with this
# policy was a result of prescribing the wrong selector format, not a real
# 70B limitation.
PROMPT_PREFIX = (
    "STRICT JSON OUTPUT (the action parser requires exact compliance):\n"
    "- Respond with EXACTLY ONE JSON object. Output starts with { and ends with }.\n"
    "- target MUST be a SINGLE element_id like \"add-usb-c-cable\" or "
    "\"search-input\". NEVER a phrase like '\"Add to cart\" on \"USB-C Cable\"' "
    "and NEVER a CSS-style selector like \"button#add-usb-c-cable\".\n"
    "- No prose before or after. No code fences. No markdown.\n"
    "\n"
    "CORRECT: {\"action\": \"click\", \"target\": \"add-usb-c-cable\"}\n"
    "CORRECT: {\"action\": \"type\", \"target\": \"search-input\", \"text\": \"usb-c cable\"}\n"
    "WRONG:   {\"action\": \"click\", \"target\": \"Add to cart\" on \"USB-C Cable\"}\n"
    "WRONG:   {\"action\": \"click\", \"target\": \"button#add-usb-c-cable\"}\n"
    "\n"
)


def baseline_strict_prompt_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """No SAE edits. The strict-JSON prompt prefix does the format-rescue work."""
    return SteeringPlan()
