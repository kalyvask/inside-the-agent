"""
Dynamic steering policy: rule-based on the live feature stream.

Watches for problematic feature patterns and intervenes only when they fire.
Day 4 work: tune thresholds on calibration tasks.
"""

from sae.steering_controller import SteeringPlan


def _by_label(catalog: dict, label_substring: str) -> int | None:
    """Find first feature id whose label contains the substring."""
    if not catalog:
        return None
    for fid, entry in catalog.items():
        if label_substring.lower() in entry.get("label", "").lower():
            return fid
    return None


def dynamic_policy(features_dict: dict, step_idx: int, catalog: dict | None = None) -> SteeringPlan:
    """
    Apply per-step rules based on what features are firing.

    Returns an empty plan if no rule triggers.
    """
    plan = SteeringPlan()
    if not catalog:
        return plan

    promo_id = _by_label(catalog, "promotional")
    goal_id = _by_label(catalog, "goal")
    halluc_id = _by_label(catalog, "halluc") or _by_label(catalog, "confab")
    uncert_id = _by_label(catalog, "uncertain")
    impulse_id = _by_label(catalog, "impuls")
    planning_id = _by_label(catalog, "planning")

    # Rule 1: distracted by promo, low goal tracking -> suppress promo, amplify goal.
    if promo_id and features_dict.get(promo_id, 0) > 0.6:
        plan.suppress(promo_id, 3.0, "promotional bias", source="dynamic")
        if goal_id:
            plan.amplify(goal_id, 4.0, "goal tracking", source="dynamic")

    # Rule 2: about to hallucinate with low uncertainty -> suppress + nudge uncertainty.
    if halluc_id and features_dict.get(halluc_id, 0) > 0.5:
        plan.suppress(halluc_id, 3.0, "hallucination", source="dynamic")
        if uncert_id:
            plan.amplify(uncert_id, 3.0, "uncertainty", source="dynamic")

    # Rule 3: impulsive on a complex step -> amplify planning.
    if impulse_id and features_dict.get(impulse_id, 0) > 0.5:
        plan.suppress(impulse_id, 2.0, "impulsive action", source="dynamic")
        if planning_id:
            plan.amplify(planning_id, 5.0, "planning", source="dynamic")

    return plan
