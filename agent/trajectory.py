"""
Standardized per-step trajectory log.

Schema follows the design doc: every step in every run produces one StepLog.
The HUD subscribes to this stream; the benchmark analysis reads from it; the
4-condition comparison reproduces results from the JSONL files.

JSONL output path:
  data/trajectories/{run_id}.jsonl   (one line per step)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class ObservationLog:
    url: str = ""
    dom_text: str = ""
    screenshot_path: str = ""


@dataclass
class ModelCallLog:
    prompt_hash: str = ""
    raw_response: str = ""
    parsed_action: dict = field(default_factory=dict)


@dataclass
class FeatureLog:
    id: int
    label: str = ""
    activation: float = 0.0
    confidence: float = 0.0  # filled from features.yaml if labeled


@dataclass
class SteeringLog:
    feature_id: int
    label: str = ""
    delta: float = 0.0
    source: str = "manual"  # static, dynamic, manual, random, wrong-sign


@dataclass
class ResultLog:
    valid_action: bool = True
    reward: float = 0.0
    done: bool = False
    error: str = ""


@dataclass
class StepLog:
    run_id: str
    task_id: str
    policy: Literal["baseline", "static", "dynamic", "random", "wrong-sign", "targeted", "manual"]
    step: int
    timestamp: float = field(default_factory=time.time)
    observation: ObservationLog = field(default_factory=ObservationLog)
    model: ModelCallLog = field(default_factory=ModelCallLog)
    features: list[FeatureLog] = field(default_factory=list)
    steering: list[SteeringLog] = field(default_factory=list)
    result: ResultLog = field(default_factory=ResultLog)

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TrajectoryLogger:
    def __init__(self, run_id: str, base_dir: str | Path = "data/trajectories"):
        self.run_id = run_id
        self.path = Path(base_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, step: StepLog):
        self._file.write(step.to_json() + "\n")
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def make_run_id(task_id: str, seed: int, policy: str) -> str:
    return f"{task_id}_seed_{seed}_{policy}"
