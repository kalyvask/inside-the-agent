"""
Regression test for v0.8 executed-vs-valid_action split.

Reviewer P1: valid_action used to mean 'JSON parsed' but was reported as if
it meant 'agent acted on the world'. Now they're separated: valid_action
is JSON-parse-success; executed is browser-dispatch-success.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.llm_agent import AgentConfig, SAEAgent

from tests.test_noise_routing import _MockBrain, _MockEnv  # reuse mocks


def _last_step(run_id_log_path: str) -> dict:
    """Load the final step record from a trajectory jsonl."""
    rows = [
        json.loads(l)
        for l in Path(run_id_log_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return rows[-1]


def test_executed_true_when_env_executes(tmp_path, monkeypatch):
    """Happy path: env reports executed=True; trajectory captures it."""
    monkeypatch.chdir(tmp_path)
    agent = SAEAgent(
        brain_call=_MockBrain(),
        env=_MockEnv(executed_flag=True),
        policy=None,
        config=AgentConfig(max_steps=1),
    )
    r = agent.run({"id": "exec_ok", "instruction": "x"}, seed=0, policy_name="baseline")
    step = _last_step(r["log_path"])
    assert step["result"]["valid_action"] is True
    assert step["result"]["executed"] is True


def test_executed_false_when_env_rejects(tmp_path, monkeypatch):
    """When the env signals Playwright dispatch failure (selector not found,
    etc.), trajectory records executed=False even though valid_action stays
    True — the action was well-formed JSON but didn't land on the DOM."""
    monkeypatch.chdir(tmp_path)
    agent = SAEAgent(
        brain_call=_MockBrain(),
        env=_MockEnv(executed_flag=False),
        policy=None,
        config=AgentConfig(max_steps=1),
    )
    r = agent.run({"id": "exec_fail", "instruction": "x"}, seed=0, policy_name="baseline")
    step = _last_step(r["log_path"])
    assert step["result"]["valid_action"] is True, \
        "JSON-parse success should not flip when execution fails"
    assert step["result"]["executed"] is False, \
        "executed must reflect env-level dispatch failure"
