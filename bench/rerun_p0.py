"""
bench/rerun_p0.py — sequential rerun of all 6 policies on the held-out task
suite at position_mode=all. Closes reviewer P0-1 (incomplete artifacts).

Designed to be run as a background subprocess. HUD_PUBLISH defaults to 0 so
this doesn't flood the live HUD with stale events. Progress is logged to
data/rerun_p0.log so the foreground process can tail it.

Order is intentional:
  1. baseline   — populates data/baselines/<task_id>.jsonl cache for the HUD
                  before/after-diff panel. Other policies need this cache.
  2. targeted   — the headline condition.
  3. wrong-sign — direction-flip ablation.
  4. random     — feature-identity control (24 trials at fixed per-trial seeds).
  5. noise      — matched-norm noise control (now routes to steer_act_with_noise
                  after v0.7-B). Will be EMPTY before that fix landed.
  6. prompt-only — system-prompt control. No SAE edits, just words.

Output:
  data/results/<policy>.jsonl     # 24 lines each by the end
  data/baselines/<task_id>.jsonl  # written once after baseline completes
  data/rerun_p0.log               # progress log
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


POLICIES = [
    "baseline",
    "targeted",
    "wrong-sign",
    "random",
    "noise",
    "prompt-only",
]

TASKS = "shopgym/tasks/held_out.json"
TRIALS = 3  # 8 tasks * 3 trials = 24 runs per policy
POSITION_MODE = "all"


def main():
    log_path = Path("data/rerun_p0.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "HUD_PUBLISH": "0"}  # don't flood the HUD with rerun events

    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"[rerun_p0] starting at {time.strftime('%H:%M:%S')}\n"
            f"[rerun_p0] tasks={TASKS} trials={TRIALS} position_mode={POSITION_MODE}\n"
            f"[rerun_p0] policies={POLICIES}\n\n"
        )
        log.flush()

        for i, policy in enumerate(POLICIES, 1):
            start = time.time()
            log.write(f"[rerun_p0] {i}/{len(POLICIES)} running '{policy}'...\n")
            log.flush()
            cmd = [
                sys.executable, "-u", "-m", "bench.runner",
                "--policy", policy,
                "--tasks", TASKS,
                "--trials", str(TRIALS),
                "--position-mode", POSITION_MODE,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=2400,  # 40 min per policy
                )
                elapsed = time.time() - start
                log.write(
                    f"[rerun_p0] '{policy}' exit={proc.returncode} "
                    f"elapsed={elapsed:.0f}s\n"
                )
                # Save the run's stdout in case we need to forensically check.
                Path(f"data/rerun_{policy}.out").write_text(
                    proc.stdout or "", encoding="utf-8"
                )
            except subprocess.TimeoutExpired:
                log.write(f"[rerun_p0] '{policy}' TIMED OUT after 40 min\n")
            except Exception as e:
                log.write(f"[rerun_p0] '{policy}' EXCEPTION: {e}\n")
            log.flush()

        log.write(f"\n[rerun_p0] done at {time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
