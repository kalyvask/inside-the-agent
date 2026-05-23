"""
bench/v0_8_finalize.py — single command to wrap up v0.8.

Run this AFTER bench/rerun_p0.py finishes (i.e. once data/rerun_p0.log
ends with the 'done at' line). It chains:

  1. bench.rerun_p0_2_scope — runs targeted at all_prompt + last_prompt_only
     so the P0-2 scope-comparison table is populated.
  2. Regenerates artifacts/benchmark_report.md from data/results/.
  3. Updates artifacts/seed_manifest.json's headline_results block with the
     computed rates so artifact_check passes.
  4. Runs bench.artifact_check as a final gate.

Usage:
    python -m bench.v0_8_finalize

Idempotent; safe to re-run.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path


RESULTS_DIR = Path("data/results")
MANIFEST = Path("artifacts/seed_manifest.json")
REPORT = Path("artifacts/benchmark_report.md")


def _wilson(s: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z = 1.96
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0, center - half), 3), round(min(1, center + half), 3)]


def _count(p: Path) -> tuple[int, int]:
    if not p.exists():
        return (0, 0)
    succ = total = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        if r.get("success"):
            succ += 1
    return (succ, total)


def step1_scope_rerun():
    print("[v0_8_finalize] step 1/4: P0-2 scope reruns")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "bench.rerun_p0_2_scope"],
        text=True,
    )
    print(f"[v0_8_finalize] scope reruns exit={proc.returncode}")


def step2_regen_report():
    print("[v0_8_finalize] step 2/4: regenerating benchmark report")
    proc = subprocess.run(
        [sys.executable, "-m", "bench.report"], text=True
    )
    print(f"[v0_8_finalize] report exit={proc.returncode}")


def step3_refresh_manifest():
    print("[v0_8_finalize] step 3/4: refreshing seed_manifest headline_results")
    if not MANIFEST.exists():
        print("[v0_8_finalize] WARNING: no seed_manifest.json to update")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    headline = manifest.setdefault("headline_results", {})
    headline["as_of"] = f"v0.8 — refreshed {time.strftime('%Y-%m-%d %H:%M UTC')}"
    rates = headline.setdefault("policy_success_rates", {})

    # Update each policy whose JSONL exists. Don't drop old keys — leave
    # legacy entries for traceability.
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        stem = path.stem
        # Skip scope-suffixed files (those go in a separate block).
        if any(stem.endswith(s) for s in ("_last_prompt_only", "_all_prompt", "_all")):
            continue
        succ, n = _count(path)
        if n == 0:
            continue
        rate = succ / n
        rates[stem] = {
            "successes": succ,
            "trials": n,
            "rate": round(rate, 3),
            "wilson_ci_95": _wilson(succ, n),
        }
        # Drop the v0.2 PENDING note from random if present.
        if stem == "random":
            rates[stem].pop("note", None)

    # Scope-comparison block (P0-2)
    scope_block = headline.setdefault("policy_success_rates_by_scope", {})
    for path in sorted(RESULTS_DIR.glob("targeted_*.jsonl")):
        scope = path.stem[len("targeted_"):]
        succ, n = _count(path)
        if n == 0:
            continue
        scope_block[scope] = {
            "successes": succ,
            "trials": n,
            "rate": round(succ / n, 3),
            "wilson_ci_95": _wilson(succ, n),
        }
    # Always include the canonical 'all' entry from targeted.jsonl
    succ_all, n_all = _count(RESULTS_DIR / "targeted.jsonl")
    if n_all > 0:
        scope_block["all"] = {
            "successes": succ_all,
            "trials": n_all,
            "rate": round(succ_all / n_all, 3),
            "wilson_ci_95": _wilson(succ_all, n_all),
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[v0_8_finalize] manifest updated: {len(rates)} policies, "
          f"{len(scope_block)} scope entries")


def step4_artifact_check():
    print("[v0_8_finalize] step 4/4: artifact_check (CI gate)")
    proc = subprocess.run(
        [sys.executable, "-m", "bench.artifact_check"], text=True
    )
    print(f"[v0_8_finalize] artifact_check exit={proc.returncode}")
    return proc.returncode


def main():
    step1_scope_rerun()
    step2_regen_report()
    step3_refresh_manifest()
    # Regen the report once more so the README's "source of truth" reflects
    # the just-updated manifest (the report itself doesn't depend on the
    # manifest, but keeping the order tidy helps).
    step2_regen_report()
    return step4_artifact_check()


if __name__ == "__main__":
    sys.exit(main())
