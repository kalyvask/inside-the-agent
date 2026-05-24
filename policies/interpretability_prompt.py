"""
Interpretability-informed prompt-only policy.

The hypothesis: an SAE-derived characterization of the agent's failure modes
(e.g. "feature f26737 encodes UI-selection vocabulary; suppressing it cuts
promotional traps") can be re-expressed as a system-prompt instruction that
beats the generic prompt-only baseline.

This is the cheap test of "does interpretability give an information edge
over careful prompt engineering, or is it a substitute for it?" Three
outcomes:
  - beats prompt-only at >5 pts: validates the interp-to-prompt loop
  - ties prompt-only within CI: insight didn't add prompt value
  - loses to prompt-only: the explicit naming confused the model

The mechanism: an EMPTY SteeringPlan plus a system-prompt prefix that
references what the SAE has measured. No residual-stream intervention.
The agent loop hooks up PROMPT_PREFIX via PROMPT_ONLY_PREFIX-style dispatch
in agent/llm_agent.py.

Baseline prompt-only at 73.3% is the bar. Targeted SAE steering at 56.7%
sits below. Anything above 73.3% would suggest the SAE characterization
is information the prompt engineer didn't already have.
"""

from sae.steering_controller import SteeringPlan


# The interpretability-derived prompt names two measured circuits and the
# operating consequence of each. It does NOT use the unsteered baseline's
# prompt as a base; we want the *marginal* value of naming the circuits.
PROMPT_PREFIX = (
    "INTERPRETABILITY-DERIVED INSTRUCTION (we have measured your activations):\n"
    "\n"
    "1. A circuit we call f26737 fires on UI-selection vocabulary "
    "(\"Click\", \"Buy Now\", \"Featured\", \"Today's Deal\", \"Select\"). "
    "When it fires strongly, you over-commit to whichever button is most "
    "visually salient on the page, typically a promotional one. Suppress "
    "this reflex: do NOT click the most visible button reflexively.\n"
    "\n"
    "2. A circuit we call f23803 fires when you correctly identify elements "
    "as distractors. Engage it actively: before choosing any action, "
    "explicitly label every on-page element as either goal-relevant or "
    "distractor. Promotional banners, sponsored cards, Today's Deal "
    "callouts, and Buy Now hero buttons are distractors by default.\n"
    "\n"
    "Practical rule: prefer the search bar, navigation, or category links "
    "over any single highlighted button. The goal stated below is the only "
    "target; everything else is a distractor, even if it looks legitimate.\n"
    "\n"
)


def interpretability_prompt_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """No SAE edits. The interpretability-derived prompt prefix does the work."""
    return SteeringPlan()
