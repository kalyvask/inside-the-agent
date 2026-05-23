"""
Failure-mining policy: data-derived steering target.

Reviewer/README open question: 'Mining surfaced 4 features (50853, 19079,
39820, 44602) that fire in 100% of baseline failures. These are stronger
candidates for true "promo-trap" representation than what contrast discovery
surfaced. Worth testing as steering targets.'

Implementation rationale:
- The targeted policy uses 2 features (f26737, f23803) discovered via contrast
  prompts + logit-lens characterization. Effective but lexically narrow.
- These 4 features were discovered by FAILURE MINING: count which features
  fire above-threshold at the failure step across all 25 baseline failures.
  These 4 were present in 100%. Data-driven not contrast-driven.
- Suppressing all 4 at step 0 tests whether the failure signature is causal
  (suppress -> success rate climbs) or symptomatic (suppress -> no change,
  because they just CORRELATE with failure without driving it).

Step-0-only so it's directly comparable to the targeted policy in the
headline benchmark. Magnitude -4 (lighter than targeted's -6) so the
combined |Δ| (4 × 4 = 16) is comparable to targeted's two-feature |Δ|
of 6 + 6 = 12 — not exact, but close enough for a fair comparison.

If `failure_mining` beats `targeted`, the data-derived features are
stronger steering targets, and contrast discovery should be deprioritized.
If they're comparable, both approaches converge on similar circuits. If
`failure_mining` underperforms, the 100%-firing features are likely
*symptoms* of failure rather than its cause.
"""

from sae.steering_controller import SteeringPlan


FAILURE_MINING_EDITS = [
    {"feature_id": 50853, "delta": -4.0, "label": "f50853_fail_mode_a"},
    {"feature_id": 19079, "delta": -4.0, "label": "f19079_fail_mode_b"},
    {"feature_id": 39820, "delta": -4.0, "label": "f39820_fail_mode_c"},
    {"feature_id": 44602, "delta": -4.0, "label": "f44602_fail_mode_d"},
]


def failure_mining_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """Suppress the 4 features that mining showed fire in 100% of baseline
    failures. Step-0 only for comparability with the targeted policy."""
    plan = SteeringPlan()
    if step_idx != 0:
        return plan
    for edit in FAILURE_MINING_EDITS:
        plan.add(
            feature_id=edit["feature_id"],
            delta=edit["delta"],
            label=edit["label"],
            source="failure_mining",
        )
    return plan
