"""
bench/strict_audit.py — print lenient vs strict-cart rates for every
committed canonical policy snapshot. Used to promote strict-cart from
"captured field" to "headline-table column" per roadmap item 6.

Usage:
    python -m bench.strict_audit
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path

RESULTS = Path("artifacts/results")


def wilson(s: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


def main():
    policies = [
        "baseline", "wrong-sign", "random", "noise", "targeted",
        "failure-mining", "prompt-only", "interpretability-prompt",
        "prompt-plus-targeted", "baseline-strict-prompt", "dynamic",
    ]
    print(f"{'policy':<28s} {'n':>4s}  {'lenient':>14s}  {'strict':>14s}  diff")
    print("-" * 76)
    for p in policies:
        path = RESULTS / f"{p}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rows)
        lenient = sum(1 for r in rows if r.get("success"))
        strict = sum(1 for r in rows if r.get("strict_success"))
        lr = lenient / n if n else 0
        sr = strict / n if n else 0
        l_ci = wilson(lenient, n)
        s_ci = wilson(strict, n)
        diff = sr - lr
        print(
            f"{p:<28s} {n:>4d}  "
            f"{lenient:>3d} ({lr:>5.1%}) "
            f"{strict:>3d} ({sr:>5.1%})  "
            f"{diff:+.1%}"
        )
    print()
    print("Note: strict_success = cart contains target exactly once AND no other")
    print("product was added. lenient_success = cart contains target at all.")


if __name__ == "__main__":
    main()
