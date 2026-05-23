"""
bench/rerun_v0_22.py — P2 closeout pipeline.

Three reviewer items, run as one chained background process so
v0_8_finalize doesn't need to coordinate:

  Step 1  per-feature ablation
            targeted-f26737-only × 60 trials
            targeted-f23803-only × 60 trials
          (~20 min Modal each, ~40 min total)

  Step 2  strict-cart canonical pass
            Already wired in v0.22: ShopGymEnv._check_verifier now ALSO
            computes the strict verifier and stashes it on the env.
            agent.llm_agent reads strict_success() and writes it to
            the trial result. So we just rerun all 6 main policies
            ON HELD-OUT (60 trials each) and the strict_success column
            appears automatically.

  Step 3  larger corpus probe (1000 wikitext prompts)
            verify.corpus_probe_large
            (~10 min Modal)

Total: ~75 min Modal for steps 1-2, ~10 min Modal for step 3.

HUD_PUBLISH=0 throughout — these don't compete with the live HUD.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run(label: str, cmd: list[str], log_file, timeout: int = 3600) -> int:
    start = time.time()
    log_file.write(f"\n[v0_22] === {label} ===\n[v0_22] cmd: {' '.join(cmd)}\n")
    log_file.flush()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONUTF8": "1", "HUD_PUBLISH": "0"}
    try:
        proc = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        log_file.write(
            f"[v0_22] {label} exit={proc.returncode} elapsed={elapsed:.0f}s\n"
        )
        Path(f"data/rerun_v0_22_{label}.out").write_text(
            proc.stdout or "", encoding="utf-8"
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        log_file.write(f"[v0_22] {label} TIMED OUT\n")
        return -1
    except Exception as e:
        log_file.write(f"[v0_22] {label} EXCEPTION: {e}\n")
        return -2


def main():
    log_path = Path("data/rerun_v0_22.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[v0_22] starting at {time.strftime('%H:%M:%S')}\n")
        log.flush()

        # Step 1: per-feature ablation (both single-feature policies)
        for policy in ["targeted-f26737-only", "targeted-f23803-only"]:
            run(
                f"ablation_{policy}",
                [
                    sys.executable, "-u", "-m", "bench.runner",
                    "--policy", policy,
                    "--tasks", "shopgym/tasks/held_out.json",
                    "--trials", "3",
                    "--position-mode", "all",
                ],
                log,
                timeout=2400,
            )

        # Step 2: strict-cart pass — rerun all 6 main policies. The runner now
        # auto-captures strict_success via the v0.22 env change; the resulting
        # data/results/<policy>.jsonl rows will have both lenient AND strict.
        for policy in ["baseline", "targeted", "wrong-sign", "random",
                       "noise", "prompt-only"]:
            run(
                f"strict_{policy}",
                [
                    sys.executable, "-u", "-m", "bench.runner",
                    "--policy", policy,
                    "--tasks", "shopgym/tasks/held_out.json",
                    "--trials", "3",
                    "--position-mode", "all",
                ],
                log,
                timeout=2400,
            )

        # Step 3: larger corpus probe
        run(
            "corpus_probe_large",
            [sys.executable, "-u", "-m", "verify.corpus_probe_large",
             "--n-prompts", "1000"],
            log,
            timeout=2400,
        )

        log.write(f"\n[v0_22] done at {time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
