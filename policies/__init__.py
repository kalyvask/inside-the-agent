"""
Steering policies. Four conditions for the 4-condition benchmark:
  - baseline:    no steering (empty plan)
  - static:      fixed feature interventions
  - random:      same magnitudes as static but on random features (control)
  - wrong-sign:  same features as static but with opposite signs (ablation)
  - targeted:    static or dynamic, whichever performs best on calibration

If targeted wins and random + wrong-sign don't, the result is causally credible.
"""

from .static import static_policy
from .dynamic import dynamic_policy
from .random_policy import random_policy
from .wrong_sign import wrong_sign_policy
from .targeted import targeted_policy
from .prompt_only_control import prompt_only_policy, PROMPT_PREFIX as PROMPT_ONLY_PREFIX
from .noise_control import noise_control_policy
from .failure_mining import failure_mining_policy
from .targeted_f26737_only import targeted_f26737_only_policy
from .targeted_f23803_only import targeted_f23803_only_policy

POLICY_REGISTRY = {
    "baseline": None,
    "static": static_policy,
    "dynamic": dynamic_policy,          # v0.9: per-step adaptive on failure-mined features
    "random": random_policy,
    "wrong-sign": wrong_sign_policy,
    "targeted": targeted_policy,        # contrast-derived f26737 + f23803 at step 0
    "prompt-only": prompt_only_policy,  # v0.4-D: prompt-prefix-only control
    "noise": noise_control_policy,      # v0.5-B: matched-norm residual noise
    "failure-mining": failure_mining_policy,  # v0.9: data-derived 4 features at step 0
    # v0.22 per-feature ablation — same edits as targeted, but only one
    # of the two. Answers: synergistic or additive?
    "targeted-f26737-only": targeted_f26737_only_policy,
    "targeted-f23803-only": targeted_f23803_only_policy,
}

# Per-policy system prompt overrides. Empty string = use the default prompt.
POLICY_PROMPT_PREFIX = {
    "prompt-only": PROMPT_ONLY_PREFIX,
}

__all__ = ["POLICY_REGISTRY", "static_policy", "dynamic_policy", "random_policy", "wrong_sign_policy"]
