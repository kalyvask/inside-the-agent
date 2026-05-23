"""Tests for the trajectory schema and logger."""

import json
import tempfile
from pathlib import Path

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


def test_make_run_id_includes_components():
    rid = make_run_id("promo_held_001", seed=2, policy="targeted")
    assert "promo_held_001" in rid
    assert "seed_2" in rid
    assert "targeted" in rid


def test_step_log_roundtrip():
    step = StepLog(
        run_id="test_run",
        task_id="promo_held_001",
        policy="targeted",
        step=0,
        observation=ObservationLog(url="about:blank", dom_text="...", screenshot_path="x.png"),
        model=ModelCallLog(prompt_hash="abc123", raw_response="...", parsed_action={"action": "click"}),
        features=[FeatureLog(id=42, label="planning", activation=1.5, confidence=0.8)],
        steering=[SteeringLog(feature_id=42, label="planning", delta=3.0, source="static")],
        result=ResultLog(valid_action=True, reward=0.0, done=False),
    )
    json_str = step.to_json()
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "test_run"
    assert parsed["features"][0]["id"] == 42
    assert parsed["steering"][0]["delta"] == 3.0


def test_trajectory_logger_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        logger = TrajectoryLogger(run_id="test_run", base_dir=tmp)
        for i in range(3):
            logger.log(StepLog(
                run_id="test_run",
                task_id="t1",
                policy="baseline",
                step=i,
            ))
        logger.close()

        log_file = Path(tmp) / "test_run.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["step"] == 0


def test_hash_prompt_is_stable():
    h1 = StepLog.hash_prompt("hello world")
    h2 = StepLog.hash_prompt("hello world")
    h3 = StepLog.hash_prompt("hello world!")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
