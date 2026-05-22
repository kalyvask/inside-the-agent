"""
SteeringController: composes feature-level interventions for the brain-server.

Design choices (from review feedback):
- DELTAS, not absolute values. new = clamp(old + delta, min, max).
- Clamps prevent steering from creating activations far outside training distribution.
- Apply steering only during action generation, not the entire prompt.
- Compose as a list of FeatureEdit, not a single dict, for replay + audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class FeatureEdit:
    feature_id: int
    delta: float  # additive change in feature space
    label: str = ""  # optional human-readable label for logging
    source: Literal["static", "dynamic", "manual", "random", "wrong-sign"] = "manual"


@dataclass
class SteeringPlan:
    edits: list[FeatureEdit] = field(default_factory=list)
    clamp_min: float = -10.0
    clamp_max: float = 10.0

    def add(self, feature_id: int, delta: float, label: str = "", source: str = "manual"):
        self.edits.append(
            FeatureEdit(feature_id=feature_id, delta=delta, label=label, source=source)
        )

    def amplify(self, feature_id: int, magnitude: float = 5.0, label: str = "", source: str = "manual"):
        self.add(feature_id, +abs(magnitude), label, source)

    def suppress(self, feature_id: int, magnitude: float = 2.0, label: str = "", source: str = "manual"):
        self.add(feature_id, -abs(magnitude), label, source)

    def reset(self):
        self.edits = []

    def to_dict(self) -> dict:
        """Format expected by the brain-server's steer_act endpoint."""
        return {str(e.feature_id): e.delta for e in self.edits}

    def to_log(self) -> list[dict]:
        """For per-step trajectory log."""
        return [asdict(e) for e in self.edits]


class SteeringController:
    """
    Holds the current steering plan. Policies (static/dynamic/random/wrong-sign)
    populate the plan; the agent loop reads it before each model call.
    """

    def __init__(
        self,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
    ):
        self.plan = SteeringPlan(clamp_min=clamp_min, clamp_max=clamp_max)

    def set_plan(self, plan: SteeringPlan):
        self.plan = plan

    def get_plan(self) -> SteeringPlan:
        return self.plan

    def reset(self):
        self.plan.reset()
