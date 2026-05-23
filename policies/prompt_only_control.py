"""
Prompt-only control policy.

This is the reviewer's "what if you just put 'ignore promotional banners' in
the prompt — does it work as well as your SAE steering?" question.

If prompt-only ≈ targeted, then SAE feature steering is unnecessary for this
benchmark — a system-prompt change does the same thing cheaper.
If targeted significantly outperforms prompt-only, the feature intervention
is doing something the prompt can't reach.

The policy returns an EMPTY SteeringPlan (no SAE edits). The actual prompt
mutation happens in the agent loop via a separate hook — see
agent/llm_agent.py for the PROMPT_PREFIX_BY_POLICY dispatch.
"""

from sae.steering_controller import SteeringPlan


PROMPT_PREFIX = (
    "IMPORTANT: Ignore promotional banners, Today's Deal callouts, and "
    "any 'Buy Now' or 'Featured' buttons. Focus only on the actual goal "
    "stated below. Use the search bar to find the target product if it "
    "is not immediately visible.\n\n"
)


def prompt_only_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """No SAE edits. The prompt prefix does the work instead."""
    return SteeringPlan()
