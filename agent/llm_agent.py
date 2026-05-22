"""
Agent loop: observe -> think (via brain-server) -> act (via browser-worker).

Day 2 work: wire up the actual BrowserGym/Playwright integration in
shopgym/storefront_template.py and bench/runner.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.prompts import build_chat_prompt
from agent.trajectory import (
    FeatureLog,
    ModelCallLog,
    ObservationLog,
    ResultLog,
    StepLog,
    SteeringLog,
    TrajectoryLogger,
    make_run_id,
)
from sae.steering_controller import SteeringController


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def parse_action(raw: str) -> dict:
    """Extract the first JSON object from the model's response."""
    raw = raw.strip()
    # Strip possible code fence.
    raw = re.sub(r"^```(json)?", "", raw).strip("`").strip()
    # Find first {...} block.
    match = re.search(r"\{[^{}]*\}", raw)
    if not match:
        return {"action": "invalid", "raw": raw}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "invalid", "raw": match.group(0)}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    max_steps: int = 15
    max_new_tokens: int = 96
    temperature: float = 0.2


class SAEAgent:
    """
    Orchestrates one task run.

    brain_call: callable(prompt, edits, max_new_tokens) -> {"response", "top_features"}
                wraps the Modal brain-server's steer_act endpoint
    env:        BrowserGym-style env with .reset(task) -> obs, .step(action) -> (obs, reward, done)
    policy:     callable(features_dict, step_idx) -> SteeringPlan
                or None for baseline (no steering)
    """

    def __init__(
        self,
        brain_call: Callable[..., dict],
        env: Any,
        policy: Callable | None = None,
        feature_catalog: dict[int, dict] | None = None,
        config: AgentConfig | None = None,
    ):
        self.brain = brain_call
        self.env = env
        self.policy = policy
        self.catalog = feature_catalog or {}
        self.cfg = config or AgentConfig()
        self.controller = SteeringController()

    def _label_features(self, raw_features: list[dict]) -> list[FeatureLog]:
        out = []
        for f in raw_features:
            entry = self.catalog.get(f["id"], {})
            out.append(
                FeatureLog(
                    id=f["id"],
                    label=entry.get("label", ""),
                    activation=f["activation"],
                    confidence={"high": 0.9, "medium": 0.6, "low": 0.3}.get(
                        entry.get("confidence", ""), 0.0
                    ),
                )
            )
        return out

    def run(self, task: dict, seed: int = 0, policy_name: str = "baseline") -> dict:
        run_id = make_run_id(task["id"], seed, policy_name)
        logger = TrajectoryLogger(run_id=run_id)
        obs = self.env.reset(task)
        history = []
        total_reward = 0.0
        done = False

        try:
            for step_idx in range(self.cfg.max_steps):
                # Build prompt
                prompt = build_chat_prompt(
                    goal=task["instruction"],
                    page_summary=obs["page_summary"],
                    history=history,
                )

                # Read features first (always; required for policy)
                # Then decide steering, then steered generate.
                # For latency, we can do this in one call if brain-server supports it.
                feats_only = self.brain(prompt=prompt, edits={}, mode="read")
                feature_dict = {f["id"]: f["activation"] for f in feats_only["top_features"]}

                # Apply policy
                if self.policy is not None:
                    plan = self.policy(feature_dict, step_idx, catalog=self.catalog)
                else:
                    plan = self.controller.get_plan()  # empty plan = baseline
                self.controller.set_plan(plan)

                # Generate (with steering if policy populated the plan)
                result = self.brain(
                    prompt=prompt,
                    edits=plan.to_dict(),
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                )

                action = parse_action(result["response"])
                history.append(action)

                # Execute
                next_obs, reward, env_done = self.env.step(action)
                total_reward += reward
                done = env_done

                # Log
                step_log = StepLog(
                    run_id=run_id,
                    task_id=task["id"],
                    policy=policy_name,
                    step=step_idx,
                    observation=ObservationLog(
                        url=obs.get("url", ""),
                        dom_text=obs.get("page_summary", "")[:500],
                        screenshot_path=obs.get("screenshot_path", ""),
                    ),
                    model=ModelCallLog(
                        prompt_hash=StepLog.hash_prompt(prompt),
                        raw_response=result["response"],
                        parsed_action=action,
                    ),
                    features=self._label_features(result["top_features"]),
                    steering=[
                        SteeringLog(
                            feature_id=e.feature_id,
                            label=e.label,
                            delta=e.delta,
                            source=e.source,
                        )
                        for e in plan.edits
                    ],
                    result=ResultLog(
                        valid_action=(action.get("action") != "invalid"),
                        reward=reward,
                        done=env_done,
                    ),
                )
                logger.log(step_log)

                obs = next_obs
                if done:
                    break
        finally:
            logger.close()

        return {
            "run_id": run_id,
            "task_id": task["id"],
            "policy": policy_name,
            "steps": step_idx + 1,
            "total_reward": total_reward,
            "success": bool(total_reward > 0),
            "log_path": str(logger.path),
        }
