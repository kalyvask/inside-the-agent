"""Tests for the steering policies."""

from policies import POLICY_REGISTRY, POLICY_PROMPT_PREFIX
from policies.targeted import targeted_policy, TARGETED_EDITS
from policies.random_policy import random_policy
from policies.wrong_sign import wrong_sign_policy
from policies.prompt_only_control import prompt_only_policy, PROMPT_PREFIX


def test_registry_has_all_policies():
    expected = {"baseline", "static", "dynamic", "random", "wrong-sign", "targeted", "prompt-only"}
    assert expected.issubset(set(POLICY_REGISTRY))


def test_baseline_is_none():
    assert POLICY_REGISTRY["baseline"] is None


def test_targeted_returns_two_edits_at_step_0():
    plan = targeted_policy({}, step_idx=0, catalog={})
    assert len(plan.edits) == 2
    assert {e.feature_id for e in plan.edits} == {26737, 23803}


def test_targeted_empty_after_step_0():
    plan = targeted_policy({}, step_idx=1, catalog={})
    assert plan.edits == []


def test_wrong_sign_flips_targeted():
    """Wrong-sign should have same features as targeted with flipped deltas."""
    targeted = targeted_policy({}, step_idx=0, catalog={})
    wrong = wrong_sign_policy({}, step_idx=0, catalog={})
    t_map = {e.feature_id: e.delta for e in targeted.edits}
    w_map = {e.feature_id: e.delta for e in wrong.edits}
    assert set(t_map) == set(w_map)
    for fid in t_map:
        assert t_map[fid] == -w_map[fid], f"fid {fid} sign not flipped"


def test_random_uses_trial_seed():
    """Two trials should draw DIFFERENT feature IDs."""
    p1 = random_policy({}, step_idx=0, trial_seed=0)
    p2 = random_policy({}, step_idx=0, trial_seed=1)
    ids1 = sorted(e.feature_id for e in p1.edits)
    ids2 = sorted(e.feature_id for e in p2.edits)
    assert ids1 != ids2, "Same trial seed shouldn't produce same features"


def test_random_deterministic_per_seed():
    """Same seed → same features."""
    p1 = random_policy({}, step_idx=0, trial_seed=42)
    p2 = random_policy({}, step_idx=0, trial_seed=42)
    ids1 = sorted(e.feature_id for e in p1.edits)
    ids2 = sorted(e.feature_id for e in p2.edits)
    assert ids1 == ids2


def test_random_empty_after_step_0():
    plan = random_policy({}, step_idx=1, trial_seed=0)
    assert plan.edits == []


def test_prompt_only_no_sae_edits():
    """The prompt-only policy returns an empty SteeringPlan."""
    plan = prompt_only_policy({}, step_idx=0)
    assert plan.edits == []


def test_prompt_only_has_prefix_registered():
    assert "prompt-only" in POLICY_PROMPT_PREFIX
    assert POLICY_PROMPT_PREFIX["prompt-only"] == PROMPT_PREFIX
    assert "promotional banners" in PROMPT_PREFIX.lower()


def test_targeted_edits_have_clear_labels():
    """v0.4 rename: labels should reflect logit-lens findings, not assumptions."""
    labels = [e["label"] for e in TARGETED_EDITS]
    # The labels should mention what the features were renamed to
    assert any("ui_selection" in l or "selection_vocab" in l for l in labels), (
        f"Expected ui_selection mention; got {labels}"
    )
    assert any("distraction" in l for l in labels), (
        f"Expected distraction mention; got {labels}"
    )
