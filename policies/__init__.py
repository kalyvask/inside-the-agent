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

POLICY_REGISTRY = {
    "baseline": None,
    "static": static_policy,
    "dynamic": dynamic_policy,
    "random": random_policy,
    "wrong-sign": wrong_sign_policy,
    "targeted": targeted_policy,  # empirically validated by step0_calibration
}

__all__ = ["POLICY_REGISTRY", "static_policy", "dynamic_policy", "random_policy", "wrong_sign_policy"]
