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
        self._publish("step_started", run_id=run_id)

    def features_read(self, features: list[dict]):
        self._publish("features_read", features=features)

    def action_chosen(self, action: dict):
        self._publish("action_chosen", action=action)

    def steering_applied(self, edits: list[dict]):
        self._publish("steering_applied", edits=edits)

    def env_updated(self, screenshot_path: str):
        self._publish("env_updated", screenshot_path=screenshot_path)

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
