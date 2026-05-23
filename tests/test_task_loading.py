"""Tests for the benchmark task loader."""

import json
import tempfile
from pathlib import Path

from bench.runner import _load_tasks


def test_load_single_file_list():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"id": "a"}, {"id": "b"}], f)
        path = f.name
    try:
        tasks = _load_tasks(path)
        assert len(tasks) == 2
        assert tasks[0]["id"] == "a"
    finally:
        Path(path).unlink()


def test_load_single_file_object():
    """Singleton dict should auto-wrap into a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"id": "solo"}, f)
        path = f.name
    try:
        tasks = _load_tasks(path)
        assert len(tasks) == 1
        assert tasks[0]["id"] == "solo"
    finally:
        Path(path).unlink()


def test_load_with_limit():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"id": f"t{i}"} for i in range(10)], f)
        path = f.name
    try:
        tasks = _load_tasks(path, limit=3)
        assert len(tasks) == 3
        assert tasks[2]["id"] == "t2"
    finally:
        Path(path).unlink()


def test_held_out_tasks_file_exists_and_valid():
    """The repo's actual held_out.json should load."""
    p = Path("shopgym/tasks/held_out.json")
    if not p.exists():
        return  # repo layout may differ in CI; smoke-only
    tasks = _load_tasks(str(p))
    assert len(tasks) > 0
    for t in tasks:
        assert "id" in t
        assert "verifier" in t
