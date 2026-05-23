"""
Regression test for v0.7-B noise routing.

Reviewer caught (P0-4) that noise_control_policy set plan._noise_* but the
agent loop never read those — every "noise" run was effectively a baseline
with a noise label. This test wires a mock brain_call that records each call
and asserts that when the noise policy is active, the agent invokes
mode="noise" with the right seed and norm.
"""

from __future__ import annotations

from agent.llm_agent import AgentConfig, SAEAgent
from policies.noise_control import NOISE_NORM, noise_control_policy


class _MockBrain:
    """Records every brain_call invocation; returns canned responses so the
    agent loop completes without hitting Modal."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt: str, edits=None, mode: str = "act", **kw):
        # Preserve whether `edits` was passed (possibly empty) vs not at all —
        # bool({}) is False, so the previous `if edits else None` collapsed
        # an empty plan into None and lost that distinction.
        self.calls.append({
            "mode": mode,
            "edits": dict(edits) if edits is not None else None,
            **kw,
        })
        # Read returns top_features list
        if mode == "read":
            return {"top_features": []}
        # Act / noise both return a parseable JSON action.
        return {
            "response": '{"action": "type", "target": "search-input", "text": "x"}',
            "top_features": [],
        }


class _MockEnv:
    """Minimal env: returns one observation then terminates immediately."""

    def reset(self, task):
        return {"url": "test://noop", "page_summary": "PAGE", "screenshot_path": ""}

    def step(self, action):
        return ({"url": "test://noop", "page_summary": "PAGE", "screenshot_path": ""},
                0.0,
                True)  # done after one step


def test_noise_policy_routes_to_noise_endpoint():
    brain = _MockBrain()
    env = _MockEnv()
    agent = SAEAgent(
        brain_call=brain,
        env=env,
        policy=noise_control_policy,
        config=AgentConfig(max_steps=1),
    )
    agent.run({"id": "noise_smoke", "instruction": "do nothing"}, seed=42, policy_name="noise")

    # Find the act-step call (mode != read). At step 0 with the noise policy
    # active, the agent MUST call mode="noise" with our seed + canonical norm.
    act_calls = [c for c in brain.calls if c["mode"] in ("noise", "act")]
    assert len(act_calls) == 1, f"expected exactly one act-style call, got {len(act_calls)}"
    call = act_calls[0]
    assert call["mode"] == "noise", \
        f"noise_control_policy at step 0 must route to mode='noise', got {call['mode']!r}"
    assert call["noise_seed"] == 42, \
        f"noise_seed must thread from trial_seed; got {call.get('noise_seed')!r}"
    assert abs(call["noise_norm"] - NOISE_NORM) < 1e-6, \
        f"noise_norm must equal NOISE_NORM ({NOISE_NORM}); got {call.get('noise_norm')!r}"


def test_baseline_policy_still_routes_to_act():
    """Baseline (no policy) must still call mode='act' with empty edits.
    Guards against the noise-routing branch accidentally hijacking baseline."""
    brain = _MockBrain()
    env = _MockEnv()
    agent = SAEAgent(
        brain_call=brain,
        env=env,
        policy=None,  # baseline
        config=AgentConfig(max_steps=1),
    )
    agent.run({"id": "baseline_smoke", "instruction": "do nothing"}, seed=0, policy_name="baseline")

    act_calls = [c for c in brain.calls if c["mode"] in ("noise", "act")]
    assert len(act_calls) == 1
    assert act_calls[0]["mode"] == "act"
    assert act_calls[0]["edits"] == {}
