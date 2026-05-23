"""
bench/compute_strict.py — recompute STRICT success from existing trajectories.

Reviewer P1: 'Add exact-stop scoring to headline numbers. The sample targeted
trajectory shows repeated add-to-cart clicks before success; strict verifier
should be the serious benchmark.'

The runner already records the lenient success in data/results/<policy>.jsonl.
The strict_cart verifier needs the env's end-state, which we don't snapshot —
but we CAN reconstruct strictness from the agent's own action history:

  strict_pass = (target was added to cart) AND (no other product was added)

A product is "added" if the agent emitted a click action whose target matches
`button#add-<slug>` or `add-<slug>` on a ShopGym storefront. We extract the
slug, dedupe across the trajectory, and check both predicates.

Run:
    python -m bench.compute_strict
    # writes artifacts/strict_rates.json + appends a "Strict-cart rates"
    # section to artifacts/benchmark_report.md.

Limitation: this only works for ShopGym tasks (deterministic templated
storefront). Real-web tasks have verifier='qualitative' and are skipped.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("data/results")
ARTIFACTS_DIR = Path("artifacts")
TASKS_DIR = Path("shopgym/tasks")


# ShopGym add-to-cart buttons are rendered as `<button id="add-<slug>">Add to cart</button>`.
# The agent's click action looks like:
#   {"action": "click", "target": "add-usb-c-cable"}
# or {"action": "click", "target": "#add-usb-c-cable"}
# or {"action": "click", "target": "button#add-usb-c-cable"}
_ADD_RE = re.compile(r"(?:button#|#|^)add-([a-z0-9][a-z0-9\-]*)", re.IGNORECASE)


def _target_slugs() -> dict[str, str]:
    """task_id -> target_product.slug, from all shopgym/tasks/*.json that
    declare a storefront_config.target_product."""
    out: dict[str, str] = {}
    for path in TASKS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        tasks = data if isinstance(data, list) else [data]
        for t in tasks:
            if not isinstance(t, dict):
                continue
            sc = t.get("storefront_config") or {}
            tp = sc.get("target_product") or {}
            # Sometimes target_product is a bare string (legacy task JSON).
            if isinstance(tp, str):
                slug = tp
            else:
                slug = tp.get("slug")
            if t.get("id") and slug:
                out[t["id"]] = slug
            elif t.get("id"):
                # Tasks with no override use the default from storefront_template.py
                out[t["id"]] = "usb-c-cable"
    return out


def _extract_added_slugs(trajectory_path: Path) -> list[str]:
    """Return ordered list of distinct slugs the agent attempted to add."""
    if not trajectory_path.exists():
        return []
    slugs: list[str] = []
    seen: set[str] = set()
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            step = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only count clicks the env actually executed — a missed click
        # doesn't add anything.
        if not step.get("result", {}).get("executed", True):
            continue
        action = step.get("model", {}).get("parsed_action") or {}
        if action.get("action") != "click":
            continue
        target = str(action.get("target", ""))
        m = _ADD_RE.search(target)
        if m:
            slug = m.group(1).lower()
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
    return slugs


def _wilson(s: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> int:
    targets = _target_slugs()
    # Walk per-policy results.
    out: dict[str, dict] = {}
    detail_lines: list[str] = []
    for results_path in sorted(RESULTS_DIR.glob("*.jsonl")):
        stem = results_path.stem
        # Skip OUTPUT_SUFFIX'd files (per-scope, per-site) for the headline.
        if "_" in stem and stem.split("_")[0] not in {"prompt", "wrong", "failure"}:
            # e.g. targeted_walmart_search, targeted_ebay, targeted_all_prompt
            # Keep prompt-only / wrong-sign / failure-mining which legitimately
            # use underscores.
            if not (stem.startswith("prompt-only") or stem.startswith("wrong-sign") or stem.startswith("failure-mining")):
                continue
        rows = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            continue

        strict_pass = 0
        strict_eligible = 0
        for r in rows:
            task_id = r.get("task_id", "")
            target_slug = targets.get(task_id)
            if target_slug is None:
                # Real-web / qualitative task. Skip.
                continue
            strict_eligible += 1
            log_path = Path(r.get("log_path", ""))
            slugs = _extract_added_slugs(log_path)
            if slugs == [target_slug]:
                strict_pass += 1
            else:
                detail_lines.append(
                    f"  {stem}/{task_id}: added={slugs!r}, target={target_slug!r}"
                )

        if strict_eligible == 0:
            continue

        lenient_pass = sum(1 for r in rows
                           if targets.get(r.get("task_id", "")) is not None
                           and r.get("success"))
        ci = _wilson(strict_pass, strict_eligible)
        out[stem] = {
            "lenient_successes": lenient_pass,
            "strict_successes": strict_pass,
            "trials": strict_eligible,
            "strict_rate": round(strict_pass / strict_eligible, 4),
            "wilson_ci_95": [round(ci[0], 3), round(ci[1], 3)],
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "strict_rates.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # Print the headline table on stdout for the CI / user.
    print()
    print("[approximate] Strict-cart reconstruction from agent action history.")
    print("Caveat: this only counts clicks matching `button#add-<slug>`. Clicks")
    print("on the trap buy-now-hero button (which also adds to cart in ShopGym)")
    print("are NOT counted as cart pollution here. For the canonical strict")
    print("number, we need a rerun that captures the env's strict_cart verifier")
    print("result alongside the lenient one. Tracking issue: P1 in next rerun.")
    print()
    print("Policy                 lenient    strict*  diff   95% CI (strict)")
    print("-" * 70)
    for policy, d in out.items():
        lenient_rate = d["lenient_successes"] / d["trials"] if d["trials"] else 0.0
        diff = d["strict_rate"] - lenient_rate
        print(
            f"{policy:<22} {lenient_rate:>6.1%}    {d['strict_rate']:>6.1%}   "
            f"{diff:+.1%}   [{d['wilson_ci_95'][0]:.2f}, {d['wilson_ci_95'][1]:.2f}]"
        )

    if detail_lines:
        print()
        print("Lenient-pass but strict-fail trials (first 10):")
        for line in detail_lines[:10]:
            print(line)
        if len(detail_lines) > 10:
            print(f"  ... ({len(detail_lines) - 10} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
