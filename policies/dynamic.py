"""
Dynamic (adaptive) steering policy.

Reviewer/README open question: 'Current targeted policy intervenes only at
Step 0. A policy that watches feature activations across the trajectory and
intervenes only at risky moments would be a stronger general claim.'

v0.9 design:
- Watch the 4 failure-mining features (50853, 19079, 39820, 44602 — see
  policies/failure_mining.py for the data-derived rationale) AND the
  promotional-bias feature 34048 (from sae/features.yaml contrast
  discovery).
- At EVERY step (not just step 0), check activations. If any watched
  feature exceeds its trigger threshold, suppress it.
- Optionally amplify a goal-tracking feature if its activation drops
  below a "losing focus" floor.
- Default thresholds are conservative — tuned to fire on roughly 25%
  of the steps in a typical trial. Too aggressive and it becomes
  always-on steering (i.e. step-0-equivalent); too lax and it never
  fires.

This is the contrast with `targeted` (fixed step-0 edit) and `failure_mining`
(fixed step-0 4-feature edit): dynamic should be at-least-as-good with
fewer interventions if the failure-mode features are causally implicated
at the exact moments they fire above threshold.

Earlier v0.4 dynamic_policy used catalog label substrings to find feature
IDs; most features in the catalog are unlabelled, so it never fired. v0.9
hard-codes the IDs that were identified via failure mining + contrast.
"""

from __future__ import annotations

import json
from pathlib import Path

from sae.steering_controller import SteeringPlan


# Failure-mining features: each fires in 100% of baseline failures at the
# failure step. Watched per-step; suppressed if their activation is above
# this threshold AT THIS STEP.
#
# v0.23: thresholds can be OVERRIDDEN from policies/dynamic_thresholds.json
# if that file exists (produced by verify/learn_dynamic_thresholds.py).
# Hand-set defaults remain the fallback.
WATCH_FEATURES = {
    50853: {"threshold": 0.40, "delta": -4.0, "label": "f50853_fail_mode_a"},
    19079: {"threshold": 0.40, "delta": -4.0, "label": "f19079_fail_mode_b"},
    39820: {"threshold": 0.40, "delta": -4.0, "label": "f39820_fail_mode_c"},
    44602: {"threshold": 0.40, "delta": -4.0, "label": "f44602_fail_mode_d"},
    # Contrast-discovered promo bias feature from sae/features.yaml.
    34048: {"threshold": 0.50, "delta": -3.0, "label": "f34048_promotional_bias"},
}


def _load_learned_thresholds() -> None:
    """v0.23: replace WATCH_FEATURES[*]['threshold'] with values learned
    by verify/learn_dynamic_thresholds.py if that JSON output exists. No-op
    on missing file; per-feature no-op if the file lacks a usable entry."""
    p = Path(__file__).parent / "dynamic_thresholds.json"
    if not p.exists():
        return
    try:
        learned = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for fid_str, info in learned.items():
        try:
            fid = int(fid_str)
        except ValueError:
            continue
        if fid not in WATCH_FEATURES:
            continue
        thr = info.get("threshold")
        if isinstance(thr, (int, float)) and 0 < thr < 5:
            WATCH_FEATURES[fid]["threshold"] = float(thr)


_load_learned_thresholds()

# If overall goal-tracking activation drops below this, amplify the
# distraction-avoidance feature (we don't have a labelled goal-tracking
# feature with reliable activation patterns, so we use f23803 as proxy).
GOAL_AMP_FEATURE_ID = 23803
GOAL_AMP_TRIGGER_MAX_TOP_ACTIVATION = 0.20  # if every top feature is < 0.2 the
                                            # agent is "drifting"; amplify
GOAL_AMP_DELTA = 4.0


def dynamic_policy(
    features_dict: dict,
    step_idx: int,
    catalog: dict | None = None,
    **_,
) -> SteeringPlan:
    """Per-step adaptive steering. Returns an empty plan if no rule fires."""
    plan = SteeringPlan()

    # Rule 1: suppress each watched feature individually whenever it's hot.
    for fid, cfg in WATCH_FEATURES.items():
        activation = features_dict.get(fid, 0.0)
        if activation >= cfg["threshold"]:
            plan.add(
                feature_id=fid,
                delta=cfg["delta"],
                label=cfg["label"] + f"_act{activation:.2f}",
                source="dynamic",
            )

    # Rule 2: if the model looks "drifty" (no feature in features_dict is
    # firing strongly), amplify the goal-anchor feature. This is the
    # 'losing focus' rescue branch.
    if features_dict:
        peak = max(features_dict.values()) if features_dict else 0.0
        if peak < GOAL_AMP_TRIGGER_MAX_TOP_ACTIVATION:
            plan.add(
                feature_id=GOAL_AMP_FEATURE_ID,
                delta=GOAL_AMP_DELTA,
                label=f"f{GOAL_AMP_FEATURE_ID}_goal_anchor_rescue",
                source="dynamic",
            )

    return plan
