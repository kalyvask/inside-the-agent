"""
bench/rerun_p0_2_scope.py — closes reviewer P0-2 (steering scope comparison).

The current headline (83% targeted) is at position_mode='all'. Reviewer:
'publish a table by steering scope: all, all_prompt, last_prompt_only.'

This script reruns targeted at the two non-default scopes on the held-out
suite, writing to dedicated files via OUTPUT_SUFFIX:
  data/results/targeted_all_prompt.jsonl
  data/results/targeted_last_prompt_only.jsonl

bench/report.py auto-discovers those filenames and emits the comparison
table. Run AFTER bench/rerun_p0.py finishes — these reuse the same Modal
container, so back-to-back avoids cold-starts.

Run:
    python -m bench.rerun_p0_2_scope
    # writes ~24 trials per scope, ~24 min total on warm Modal
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


SCOPES = ["all_prompt", "last_prompt_only"]  # 'all' is already in targeted.jsonl
TASKS = "shopgym/tasks/held_out.json"
TRIALS = 3


def main():
    log_path = Path("data/rerun_p0_2_scope.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "HUD_PUBLISH": "0"}

    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"[rerun_p0_2] starting at {time.strftime('%H:%M:%S')}\n"
            f"[rerun_p0_2] scopes={SCOPES}\n\n"
        )
        log.flush()
        for scope in SCOPES:
            start = time.time()
            log.write(f"[rerun_p0_2] running targeted @ position_mode={scope}\n")
            log.flush()
            env = {**base_env, "OUTPUT_SUFFIX": scope}
            cmd = [
                sys.executable, "-u", "-m", "bench.runner",
                "--policy", "targeted",
                "--tasks", TASKS,
                "--trials", str(TRIALS),
                "--position-mode", scope,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=2400,
                )
                elapsed = time.time() - start
                log.write(
                    f"[rerun_p0_2] targeted@{scope} exit={proc.returncode} "
                    f"elapsed={elapsed:.0f}s\n"
                )
                Path(f"data/rerun_targeted_{scope}.out").write_text(
                    proc.stdout or "", encoding="utf-8"
                )
            except subprocess.TimeoutExpired:
                log.write(f"[rerun_p0_2] targeted@{scope} TIMED OUT\n")
            except Exception as e:
                log.write(f"[rerun_p0_2] targeted@{scope} EXCEPTION: {e}\n")
            log.flush()

        log.write(f"\n[rerun_p0_2] done at {time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
