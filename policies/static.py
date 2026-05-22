"""
Static steering policy: fixed feature interventions applied at every step.

Day 4 work: populate the feature IDs from the verified Day 3 catalog.
"""

from sae.steering_controller import SteeringPlan


def static_policy(features_dict: dict, step_idx: int, catalog: dict | None = None) -> SteeringPlan:
    """
    Apply the same intervention every step. Catalog has the recommended deltas.

    Until catalog is populated, returns an empty plan (= baseline behavior).
    """
    plan = SteeringPlan()
    if not catalog:
        return plan

    # Pick the strongest catalog entries from each category.
    by_category = {"behavioral": [], "epistemic": [], "task": [], "risk": []}
    for fid, entry in catalog.items():
        cat = entry.get("category")
        if cat in by_category:
            by_category[cat].append((fid, entry))

    # Sort each category by confidence, take top entry.
    for cat, entries in by_category.items():
        if not entries:
            continue
        entries.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x[1].get("confidence", "low"), 0), reverse=True)
        fid, entry = entries[0]
        delta = entry.get("recommended_delta", 0.0)
        if delta == 0:
            continue
        plan.add(
            feature_id=fid,
            delta=delta,
            label=entry.get("label", ""),
            source="static",
        )
    return plan
