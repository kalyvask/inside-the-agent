"""
Per-feature ablation: f26737 alone (UI-selection-vocab suppression only).

v0.22 — splits the targeted-combo policy into its two component edits so
we can answer the reviewer question: are the two edits synergistic or
additive? Side-by-side comparison:

  targeted          f26737 = -6  AND  f23803 = +6   (combined, headline)
  targeted_f26737   f26737 = -6  alone
  targeted_f23803   f23803 = +6  alone

If targeted ≈ f26737_only + f23803_only effects, the two are additive.
If targeted > f26737_only + f23803_only, they synergize (one prepares
the state in which the other lands more cleanly).

Same step-0-only application, same magnitudes as the combined policy.
"""

from sae.steering_controller import SteeringPlan


def targeted_f26737_only_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """Suppress f26737 alone at step 0. UI-selection-vocab suppression."""
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    plan.add(
        feature_id=26737,
        delta=-6.0,
        label="f26737_ui_selection_vocab",
        source="targeted_f26737_only",
    )
    return plan
