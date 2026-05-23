"""
record_demo.py — single command to set the stage for the live demo recording.

What it does, in order:
  1. POST /clear on the ws_server to wipe stale events (so the HUD starts blank
     and previous-run Walmart/CAPTCHA screenshots don't bleed in).
  2. Warm the Modal brain (one quick read_features call) so the actual demo
     doesn't eat a 30s cold-start in the recording.
  3. Print a 3-second countdown so you can click "Start recording" in
     Windows Game Bar (Win+G), OBS, or whatever capture tool you use.
  4. Fire the targeted run on the chosen task with HUD_PUBLISH=1 — events
     stream to the running ws_server which the HUD picks up.
  5. Print clean exit info so you can stop the recording.

Usage:
    # Defaults: eBay /deals, targeted policy, position_mode=all, pause 4s
    python record_demo.py

    # Switch task / policy:
    python record_demo.py --task shopgym/tasks/real_aliexpress.json --policy targeted

    # Quieter (no countdown, no clear):
    python record_demo.py --no-clear --no-countdown

This does NOT screen-record by itself — Windows doesn't have a built-in
headless screen-capture API. Bring your own (OBS / Game Bar / ShareX).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import typer

try:
    import requests
except ImportError:
    requests = None  # type: ignore

app = typer.Typer(add_completion=False)


WS_SERVER_URL = os.environ.get("WS_SERVER_URL", "http://localhost:8765")


def _clear_hud() -> bool:
    if requests is None:
        return False
    try:
        r = requests.post(f"{WS_SERVER_URL}/clear", timeout=2)
        if r.ok:
            data = r.json().get("cleared", {})
            print(f"[record_demo] cleared HUD: {data}")
            return True
    except Exception as e:
        print(f"[record_demo] /clear failed: {e}")
    return False


def _warm_brain() -> None:
    print("[record_demo] warming Modal brain (read_features on a tiny prompt)…")
    try:
        # Use the existing verify.sae_smoke --quick which exits after the
        # first successful read_features call.
        subprocess.run(
            [sys.executable, "-u", "-m", "verify.sae_smoke", "--quick"],
            timeout=120,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[record_demo] brain warm.")
    except subprocess.TimeoutExpired:
        print("[record_demo] brain warm-up timed out (cold start in progress).")
    except FileNotFoundError:
        # verify.sae_smoke not available — skip
        pass


def _countdown(seconds: int = 3) -> None:
    print()
    print("[record_demo] start recording NOW (Win+G / OBS / ShareX).")
    for i in range(seconds, 0, -1):
        print(f"[record_demo]   ...firing in {i}")
        time.sleep(1)
    print()


@app.command()
def main(
    task: str = typer.Option(
        "shopgym/tasks/real_ebay.json",
        help="Task JSON file. Defaults to the validated eBay /deals task.",
    ),
    policy: str = typer.Option(
        "targeted",
        help="Policy name. 'targeted' is the demo headliner.",
    ),
    position_mode: str = typer.Option(
        "all",
        help="Steering scope. 'all' matches the headline result.",
    ),
    pause: float = typer.Option(
        4.0,
        help="Seconds between steps. 4.0 gives the audience time to follow.",
    ),
    limit: int = typer.Option(1, help="Number of tasks to run."),
    trials: int = typer.Option(1, help="Trials per task."),
    output_suffix: str = typer.Option(
        "demo",
        help="Output filename suffix to keep recording runs out of the rerun's "
             "canonical jsonl files.",
    ),
    clear: bool = typer.Option(True, help="POST /clear before recording."),
    warm: bool = typer.Option(True, help="Warm the Modal brain first."),
    countdown: bool = typer.Option(True, help="3-2-1 before the run."),
):
    """Set the stage + fire the live demo run."""
    print("=" * 60)
    print("[record_demo] inside-the-agent — live demo recorder")
    print("=" * 60)
    print(f"  task:          {task}")
    print(f"  policy:        {policy}")
    print(f"  position_mode: {position_mode}")
    print(f"  pause:         {pause}s")
    print(f"  HUD:           {WS_SERVER_URL}  (open in browser BEFORE you start)")
    print()

    if not Path(task).exists():
        print(f"[record_demo] ERROR: task file not found: {task}")
        raise typer.Exit(1)

    if clear:
        _clear_hud()
    if warm:
        _warm_brain()
    if countdown:
        _countdown(3)

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "HUD_PUBLISH": "1",
        "OUTPUT_SUFFIX": output_suffix,
    }
    cmd = [
        sys.executable, "-u", "-m", "bench.runner",
        "--policy", policy,
        "--tasks", task,
        "--trials", str(trials),
        "--limit", str(limit),
        "--pause", str(pause),
        "--position-mode", position_mode,
    ]
    print("[record_demo] running:", " ".join(cmd))
    start = time.time()
    rc = subprocess.call(cmd, env=env)
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"[record_demo] run done (exit={rc}, elapsed={elapsed:.1f}s)")
    print(f"[record_demo] STOP RECORDING NOW")
    print("=" * 60)
    print()
    print("Trajectory:")
    print(f"  data/trajectories/*_{policy}.jsonl")
    print(f"  data/results/{policy}_{output_suffix}.jsonl")
    print(f"  screenshots: data/screenshots/")


if __name__ == "__main__":
    app()
