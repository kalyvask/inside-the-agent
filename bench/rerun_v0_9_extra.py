"""
bench/rerun_v0_9_extra.py — measure the two new v0.9 policies on the
held-out suite.

Runs:
  - failure-mining (4 data-derived features suppressed at step 0)
  - dynamic        (per-step adaptive intervention on the same 4 + promo)

Sequential, HUD_PUBLISH=0 so doesn't flood the live HUD. Run after
bench/rerun_p0.py finishes:

    python -m bench.rerun_v0_9_extra

Output:
  data/results/failure-mining.jsonl  (60 trials)
  data/results/dynamic.jsonl         (60 trials)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


POLICIES = ["failure-mining", "dynamic"]
TASKS = "shopgym/tasks/held_out.json"
TRIALS = 3
POSITION_MODE = "all"


def main():
    log_path = Path("data/rerun_v0_9_extra.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONUTF8": "1", "HUD_PUBLISH": "0"}

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[v0_9_extra] starting at {time.strftime('%H:%M:%S')}\n\n")
        log.flush()
        for i, policy in enumerate(POLICIES, 1):
            start = time.time()
            log.write(f"[v0_9_extra] {i}/{len(POLICIES)} running '{policy}'\n")
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
                    cmd, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=2400,
                )
                elapsed = time.time() - start
                log.write(
                    f"[v0_9_extra] '{policy}' exit={proc.returncode} "
                    f"elapsed={elapsed:.0f}s\n"
                )
                Path(f"data/rerun_{policy}.out").write_text(
                    proc.stdout or "", encoding="utf-8"
                )
            except subprocess.TimeoutExpired:
                log.write(f"[v0_9_extra] '{policy}' TIMED OUT\n")
            except Exception as e:
                log.write(f"[v0_9_extra] '{policy}' EXCEPTION: {e}\n")
            log.flush()
        log.write(f"\n[v0_9_extra] done at {time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
