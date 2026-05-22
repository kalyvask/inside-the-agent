"""
Static steering policy: fixed feature interventions applied at every step.

Day 4 work: populate the feature IDs from the verified Day 3 catalog.
"""

from sae.steering_controller import SteeringPlan


def static_policy(features_dict: dict, step_idx: int, catalog: dict | None = None) -> SteeringPlan:
    """
    Apply the same intervention every step. Picks one feature per category.

    Selection rule per category:
      1. Prefer features with tuning_status == 'tuned' (calibrated by tune_deltas)
      2. Then by confidence (high > medium > low)
      3. Then by |contrast_score|
      4. Then by absolute delta strength

    Skips features without a usable delta (fragile/no-effect/zero).
    """
    plan = SteeringPlan()
    if not catalog:
        return plan

    by_category = {"behavioral": [], "epistemic": [], "task": [], "risk": []}
    for fid, entry in catalog.items():
        cat = entry.get("category")
        if cat in by_category:
            by_category[cat].append((fid, entry))

    def sort_key(item):
        _, e = item
        is_tuned = 1 if e.get("tuning_status") == "tuned" else 0
        conf = {"high": 3, "medium": 2, "low": 1}.get(e.get("confidence", "low"), 0)
        cs = abs(float(e.get("contrast_score", 0) or 0))
        return (is_tuned, conf, cs)

    # Compound steering scaling. Single-feature deltas tune cleanly at ±3-6,
    # but applying 4 simultaneously (one per category) destroys coherence.
    # Empirical scale of 0.35 keeps total |delta| around 4-5, preserving the
    # model's outputs while still applying multi-axis pressure.
    COMPOUND_SCALE = 0.35

    for cat, entries in by_category.items():
        if not entries:
            continue
        entries.sort(key=sort_key, reverse=True)
        # Pick first entry whose delta is usable.
        for fid, entry in entries:
            delta = entry.get("recommended_delta", 0.0)
            if delta is None or delta == 0:
                continue
            plan.add(
                feature_id=fid,
                delta=float(delta) * COMPOUND_SCALE,
                label=entry.get("label", ""),
                source="static",
            )
            break  # one per category
    return plan
