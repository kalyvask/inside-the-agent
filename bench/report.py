"""
bench/report.py — generate artifacts/benchmark_report.md from data/results/.

Reviewer P1: 'Add a benchmark report generated from artifacts, not
hand-written README numbers.' Decouples the headline numbers from prose
so a stale jsonl row can't silently mislead a reviewer reading the README.

Output: a Markdown file with:
  - Per-policy headline table (n, successes, rate, Wilson 95% CI)
  - Per-task breakdown (which tasks does each policy nail?)
  - Action-quality breakdown (valid_action vs executed, post v0.8)
  - Provenance: timestamp, manifest version, total trials, position_mode
  - Steering-scope comparison if multiple position_modes are present
    in the file naming convention (e.g. targeted_last_prompt_only.jsonl)

Run:
    python -m bench.report
    # writes artifacts/benchmark_report.md

The README can then link to this file or embed it via a transclude.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("data/results")
TRAJECTORY_DIR = Path("data/trajectories")
MANIFEST_PATH = Path("artifacts/seed_manifest.json")
OUT_PATH = Path("artifacts/benchmark_report.md")


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, center - half), min(1.0, center + half))


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _trajectory_action_stats(trajectory_path: Path) -> tuple[int, int, int]:
    """Returns (n_steps, n_valid_action, n_executed) for one trajectory.

    valid_action: model emitted parseable JSON.
    executed:     env (ShopGym/WebEnv) actually dispatched the action.
                  v0.8+; older logs without the field count as True.
    """
    if not trajectory_path.exists():
        return (0, 0, 0)
    n_steps = n_valid = n_exec = 0
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            step = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_steps += 1
        res = step.get("result", {})
        if res.get("valid_action", True):
            n_valid += 1
        if res.get("executed", True):
            n_exec += 1
    return (n_steps, n_valid, n_exec)


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.1%}, {hi:.1%}]"


def _per_policy_section(policies: dict[str, list[dict]]) -> str:
    lines = ["## Per-policy headline (in-distribution, ShopGym held-out)\n"]
    lines.append("| Policy | n | Successes | Rate | Wilson 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for name, rows in policies.items():
        n = len(rows)
        succ = sum(1 for r in rows if r.get("success"))
        rate = succ / n if n else 0.0
        ci = _wilson_ci(succ, n)
        lines.append(f"| `{name}` | {n} | {succ} | {rate:.1%} | {_fmt_ci(*ci)} |")
    lines.append("")
    return "\n".join(lines)


def _per_category_section(
    policies: dict[str, list[dict]], task_categories: dict[str, str]
) -> str:
    """v0.9: cross-domain breakdown — split each policy's rate by task category
    (promotional / hallucination / planning). Answers reviewer's 'cross-domain'
    open question: does the targeted policy generalize beyond promo traps?

    For each policy, computes successes/n at each category. Empty cell if the
    policy didn't run any tasks of that category."""
    if not task_categories:
        return ""
    categories = sorted(set(task_categories.values()))
    if len(categories) <= 1:
        return ""
    lines = [f"## Cross-domain breakdown (by task category)\n"]
    lines.append(
        "Held-out suite covers three task categories. The targeted policy was "
        "calibrated on the promotional split. Lift on hallucination + planning "
        "is the cross-domain generalization claim.\n"
    )
    header = "| Policy | " + " | ".join(f"`{c}`" for c in categories) + " |"
    sep = "|---|" + "|".join("---:" for _ in categories) + "|"
    lines.append(header)
    lines.append(sep)
    for policy, rows in policies.items():
        row_cells = [f"`{policy}`"]
        for cat in categories:
            cat_rows = [r for r in rows
                        if task_categories.get(r["task_id"]) == cat]
            if not cat_rows:
                row_cells.append("—")
                continue
            n = len(cat_rows)
            s = sum(1 for r in cat_rows if r.get("success"))
            row_cells.append(f"{s}/{n} = {s/n:.0%}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _load_task_categories() -> dict[str, str]:
    """Map task_id -> category from shopgym/tasks/*.json."""
    out = {}
    task_dir = Path("shopgym/tasks")
    if not task_dir.exists():
        return out
    for path in task_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        tasks = data if isinstance(data, list) else [data]
        for t in tasks:
            if isinstance(t, dict) and t.get("id") and t.get("category"):
                out[t["id"]] = t["category"]
    return out


def _per_task_section(policies: dict[str, list[dict]]) -> str:
    """Cross-tab: rows are tasks, columns are policies, cells are success rate
    on that (task, policy) cell across trials."""
    # Build (task_id, policy) -> [success bool, ...]
    cell = defaultdict(lambda: defaultdict(list))
    task_ids = set()
    for policy, rows in policies.items():
        for r in rows:
            task_ids.add(r["task_id"])
            cell[r["task_id"]][policy].append(bool(r.get("success")))
    if not task_ids:
        return ""

    sorted_tasks = sorted(task_ids)
    sorted_policies = list(policies.keys())
    lines = ["## Per-task breakdown\n"]
    lines.append("Each cell shows `successes/trials` for that policy on that task.\n")
    header = "| task |" + "|".join(f" `{p}` " for p in sorted_policies) + "|"
    sep = "|---|" + "|".join("---:" for _ in sorted_policies) + "|"
    lines.append(header)
    lines.append(sep)
    for t in sorted_tasks:
        cells = [f"`{t}`"]
        for p in sorted_policies:
            outcomes = cell[t][p]
            if not outcomes:
                cells.append("—")
            else:
                s = sum(outcomes)
                n = len(outcomes)
                cells.append(f"{s}/{n}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _action_quality_section(policies: dict[str, list[dict]]) -> str:
    """v0.8: split valid_action (JSON parsed) from executed (DOM actually
    dispatched). Lower executed rate exposes selector-finding flakiness."""
    lines = ["## Action quality (v0.8: JSON-parse vs Playwright-dispatch)\n"]
    lines.append("| Policy | n steps | `valid_action` rate | `executed` rate | parse-but-no-exec |")
    lines.append("|---|---:|---:|---:|---:|")
    any_data = False
    for policy, rows in policies.items():
        tot_steps = tot_valid = tot_exec = 0
        for r in rows:
            log_path = Path(r.get("log_path", ""))
            n, v, e = _trajectory_action_stats(log_path)
            tot_steps += n
            tot_valid += v
            tot_exec += e
        if tot_steps == 0:
            continue
        any_data = True
        valid_rate = tot_valid / tot_steps
        exec_rate = tot_exec / tot_steps
        parse_no_exec = tot_valid - tot_exec
        lines.append(
            f"| `{policy}` | {tot_steps} | {valid_rate:.1%} | {exec_rate:.1%} | "
            f"{parse_no_exec} |"
        )
    if not any_data:
        lines.append("| (no trajectory data available) | | | | |")
    lines.append("")
    return "\n".join(lines)


def _scope_comparison_section() -> str:
    """If files like targeted_last_prompt_only.jsonl exist, build a scope table.

    Naming convention: `<policy>_<position_mode>.jsonl` for non-default scope.
    Default scope (`all`) lives at `<policy>.jsonl`.
    """
    scopes: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        stem = path.stem
        # Try splitting on _last_prompt_only / _all_prompt / _all suffix
        for suffix in ("_last_prompt_only", "_all_prompt", "_all"):
            if stem.endswith(suffix):
                policy = stem[: -len(suffix)]
                mode = suffix.lstrip("_")
                scopes[policy][mode] = _load_jsonl(path)
                break
        else:
            # No suffix: default is 'all'
            scopes[stem].setdefault("all", _load_jsonl(path))
    # Only emit a table if at least one policy has data at multiple scopes.
    if not any(len(v) > 1 for v in scopes.values()):
        return ""
    lines = ["## Steering scope comparison (P0-2)\n"]
    lines.append(
        "How robust is the headline to where in the residual stream the "
        "delta is applied? `all` = every position; `last_prompt_only` = only "
        "the last token of prefill (more surgical); `all_prompt` = prompt "
        "positions only.\n"
    )
    cols = ["all", "all_prompt", "last_prompt_only"]
    lines.append("| Policy | " + " | ".join(f"`{c}`" for c in cols) + " |")
    lines.append("|---|" + "|".join("---:" for _ in cols) + "|")
    for policy, by_mode in scopes.items():
        if len(by_mode) < 2:
            continue
        cells = [f"`{policy}`"]
        for mode in cols:
            rows = by_mode.get(mode, [])
            if not rows:
                cells.append("—")
                continue
            n = len(rows)
            s = sum(1 for r in rows if r.get("success"))
            cells.append(f"{s}/{n} = {s/n:.0%}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    # Discover policies present in data/results/. Skip per-scope variants
    # (handled by _scope_comparison_section).
    policies: dict[str, list[dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        stem = path.stem
        # Skip scope-variant files for the headline table (they go in their
        # own section).
        if any(stem.endswith(suf) for suf in ("_last_prompt_only", "_all_prompt", "_all")):
            continue
        rows = _load_jsonl(path)
        if rows:
            policies[stem] = rows

    parts = []
    parts.append(f"# Benchmark report — `inside-the-agent`\n")
    parts.append(
        f"_Generated by `python -m bench.report` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S UTC')}_\n"
        f"This file is regenerated from `data/results/*.jsonl`. **Do not edit by "
        f"hand** — your changes will be overwritten on the next run.\n"
    )
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        v = manifest.get("manifest_version", "?")
        b = manifest.get("benchmark", {})
        parts.append(
            f"**Manifest:** version `{v}`, task suite `{b.get('task_suite', '?')}`, "
            f"trials per policy `{b.get('total_runs_per_policy', '?')}`, "
            f"seeds `{b.get('trial_seeds', '?')}`.\n"
        )

    parts.append(_per_policy_section(policies))
    parts.append(_per_category_section(policies, _load_task_categories()))
    parts.append(_per_task_section(policies))
    parts.append(_action_quality_section(policies))
    scope = _scope_comparison_section()
    if scope:
        parts.append(scope)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
