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
    """Minimal env: returns one observation then terminates immediately.

    `executed_flag` lets a test simulate Playwright dispatch failure (e.g.
    click target not in DOM) so we can assert the trajectory log captures
    that distinct from JSON-parse validity (v0.8 P1 fix)."""

    def __init__(self, executed_flag: bool = True):
        self.executed_flag = executed_flag

    def reset(self, task):
        return {"url": "test://noop", "page_summary": "PAGE", "screenshot_path": ""}

    def step(self, action):
        obs = {"url": "test://noop", "page_summary": "PAGE",
               "screenshot_path": "", "executed": self.executed_flag}
        return (obs, 0.0, True)  # done after one step


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

    # At step 0 with the noise policy active, the agent MUST call
    # mode="noise" with our seed + canonical norm.
    noise_calls = [c for c in brain.calls if c["mode"] == "noise"]
    assert len(noise_calls) == 1, \
        f"noise policy must call mode='noise' exactly once at step 0, got {len(noise_calls)}"
    noise_call = noise_calls[0]
    assert noise_call["noise_seed"] == 42, \
        f"noise_seed must thread from trial_seed; got {noise_call.get('noise_seed')!r}"
    assert abs(noise_call["noise_norm"] - NOISE_NORM) < 1e-6, \
        f"noise_norm must equal NOISE_NORM ({NOISE_NORM}); got {noise_call.get('noise_norm')!r}"

    # v0.18: when intervention happens (noise IS an intervention), the
    # agent ALSO makes a counterfactual call with edits={} on the same
    # prompt to render the "without edit" side-by-side. That's a SECOND
    # mode='act' call.
    act_calls = [c for c in brain.calls if c["mode"] == "act"]
    assert len(act_calls) == 1, \
        f"v0.18 counterfactual: noise-step should trigger exactly one " \
        f"mode='act' twin call (edits={{}}), got {len(act_calls)}"
    assert act_calls[0]["edits"] == {}, \
        "v0.18 counterfactual call must use empty edits"


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
