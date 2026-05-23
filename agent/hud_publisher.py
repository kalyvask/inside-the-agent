"""
Hook to publish agent events to the WebSocket server (agent.ws_server).

Wire into SAEAgent.run() to broadcast each step to the HUD. If the ws_server
is not running, publish is a no-op — the agent run is never blocked.

Usage:
  from agent.hud_publisher import HudPublisher
  publisher = HudPublisher()  # autodetect WS_SERVER_URL env or default localhost:8765
  publisher.step_started(run_id, task_id, step)
  publisher.features_read(features)
  publisher.action_chosen(action)
  publisher.steering_applied(edits)
  publisher.env_updated(screenshot_path)
  publisher.task_done(success)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


class HudPublisher:
    def __init__(self, url: str | None = None, enabled: bool | None = None):
        self.url = url or os.environ.get("WS_SERVER_URL", "http://localhost:8765")
        if enabled is None:
            enabled = os.environ.get("HUD_PUBLISH", "1") != "0"
        self.enabled = enabled and requests is not None
        self._task_id: str | None = None
        self._step: int = 0

    def _publish(self, event_type: str, **kwargs):
        if not self.enabled:
            return
        payload = {
            "type": event_type,
            "task_id": self._task_id,
            "step": self._step,
            "timestamp": time.time(),
            **kwargs,
        }
        try:
            requests.post(f"{self.url}/publish", json=payload, timeout=0.5)
        except Exception:
            # Silently ignore — HUD is optional.
            pass

    def step_started(self, run_id: str, task_id: str, step: int):
        self._task_id = task_id
        self._step = step
        # On the very first step, emit a demo_banner to clear any verdict from
        # a previous run and seed the HUD with task metadata.
        if step == 0:
            self._publish(
                "demo_banner",
                run_id=run_id,
                task_id=task_id,
            )
        self._publish("step_started", run_id=run_id)

    def policy_meta(
        self,
        policy: str,
        position_mode: str,
        seed: int | None = None,
        max_steps: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        steering_endpoint: str | None = None,
    ):
        """v0.7-D: emit the steering scope + decoding params at the start of a
        run so the HUD's debugging cockpit can show 'targeted @ all' or
        'noise @ last_prompt_only' instead of just the policy name.

        Reviewer note: the headline 83% targeted result depends on
        position_mode='all'; surgical 'last_prompt_only' gives 0% in our
        tests. The HUD should make the active scope unmissable so the
        audience can't conflate the two conditions."""
        self._publish(
            "policy_meta",
            policy=policy,
            position_mode=position_mode,
            seed=seed,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steering_endpoint=steering_endpoint,
        )

    def baseline_action(self, step: int, action: dict):
        """v0.7-D: emit a cached baseline action for the same task+step the
        agent is on, so the HUD can show a before/after diff. The runner
        looks up data/baselines/<task_id>.jsonl and emits this if a row
        exists for the current step."""
        self._publish("baseline_action", step=step, action=action)

    def counterfactual_action(self, step: int, action: dict, prompt_hash: str = ""):
        """v0.18: live counterfactual emitted at any step where the policy
        applied steering edits. We call brain TWICE on the IDENTICAL prompt
        — once with edits, once without — and publish the un-steered
        prediction here. The HUD renders it alongside the actual action so
        the audience sees the direct causal answer to 'is the steering
        doing anything?'. The prompts are bit-identical (same hash) so the
        comparison is clean — only the residual-stream edit differs."""
        self._publish(
            "counterfactual_action",
            step=step,
            action=action,
            prompt_hash=prompt_hash,
        )

    def features_read(self, features: list[dict]):
        self._publish("features_read", features=features)

    def action_chosen(self, action: dict):
        self._publish("action_chosen", action=action)

    def steering_applied(self, edits: list[dict]):
        self._publish("steering_applied", edits=edits)

    def env_updated(self, screenshot_path: str):
        # Convert local Windows path to a ws_server-served URL so the HUD
        # can load it via http. Mirrors the conversion in replay_trajectory.py.
        if screenshot_path:
            filename = Path(screenshot_path).name
            url = f"{self.url}/screenshots/{filename}"
        else:
            url = ""
        self._publish("env_updated", screenshot_path=url)

    def task_done(self, success: bool):
        self._publish("task_done", success=success)

    def drain_commands(self) -> list[dict]:
        """v0.3: fetch pending HUD-issued steering commands.

        Returns a list of dicts: [{feature_id, delta, label, source, one_shot}].
        The ws_server clears them after returning, so each command is
        delivered to exactly one drain call.
        """
        if not self.enabled:
            return []
        try:
            r = requests.get(f"{self.url}/control/pending", timeout=0.5)
            return r.json().get("commands", [])
        except Exception:
            return []
