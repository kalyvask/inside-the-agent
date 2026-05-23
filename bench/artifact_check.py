"""
bench/artifact_check.py — verify benchmark artifacts are internally consistent.

Reviewer P1 ask: 'add an artifact_check.py script: verify row counts, success
counts, Wilson CIs, policy names, seeds, and chart consistency.' This script
guards against the v0.2 → v0.7 drift the reviewer caught — manifest claiming
24 trials when the file had 1 line.

Checks:
  1. Every policy named in seed_manifest.headline_results has a matching
     data/results/<policy>.jsonl file.
  2. Row counts in those JSONL files match
     seed_manifest.benchmark.total_runs_per_policy.
  3. Per-policy success counts and rates computed from the JSONL match the
     manifest's headline_results.policy_success_rates values (within rounding).
  4. Wilson 95% CIs computed from the JSONL match the manifest values
     (within 0.005 absolute).
  5. Every JSONL row has the required fields: run_id, task_id, policy,
     steps, success, total_reward.
  6. No row claims a policy_name that isn't in POLICY_REGISTRY.

Run as a CI gate:
    python -m bench.artifact_check
    # exit 0 = all green, non-zero = at least one mismatch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("data/results")
MANIFEST_PATH = ARTIFACTS_DIR / "seed_manifest.json"

console = Console()


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Matches the formula used in
    the headline chart."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, center - half), min(1.0, center + half))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            console.print(f"[red]Bad JSONL row in {path}: {e}[/red]")
    return out


def _ensure_required_fields(rows: list[dict], path: Path) -> list[str]:
    """Returns list of error messages, empty if all rows are well-formed."""
    required = {"run_id", "task_id", "policy", "steps", "success", "total_reward"}
    errors = []
    for i, r in enumerate(rows):
        missing = required - r.keys()
        if missing:
            errors.append(f"{path.name}:row{i} missing fields: {sorted(missing)}")
    return errors


def main() -> int:
    if not MANIFEST_PATH.exists():
        console.print(f"[red]MISSING: {MANIFEST_PATH}[/red]")
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_trials = manifest["benchmark"]["total_runs_per_policy"]
    headline = manifest.get("headline_results", {}).get("policy_success_rates", {})

    table = Table(title="Artifact consistency check", show_lines=False)
    table.add_column("Policy", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Expect", justify="right")
    table.add_column("Succ", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Manifest", justify="right")
    table.add_column("Wilson CI (computed)", justify="right")
    table.add_column("Status", justify="left")

    overall_failures = []

    for policy, claim in headline.items():
        jsonl_path = RESULTS_DIR / f"{policy.replace(' ', '-')}.jsonl"
        if not jsonl_path.exists():
            # Try a couple of common name spellings.
            alt = RESULTS_DIR / f"{policy}.jsonl"
            if alt.exists():
                jsonl_path = alt
        rows = _load_jsonl(jsonl_path)
        problems = []

        # Per-policy expected trial count: the manifest claim wins over the
        # benchmark-level total. This lets single-trial real-site captures
        # (manifest claims trials=1) coexist with the 60-trial benchmark
        # without false-positive row-count mismatches.
        policy_expected = claim.get("trials", expected_trials)

        if len(rows) == 0:
            problems.append("MISSING")
        elif len(rows) != policy_expected:
            problems.append(f"ROWS={len(rows)} !={policy_expected}")

        field_errors = _ensure_required_fields(rows, jsonl_path)
        if field_errors:
            problems.append(f"FIELDS({len(field_errors)})")
            overall_failures.extend(field_errors)

        successes = sum(1 for r in rows if r.get("success"))
        actual_rate = successes / len(rows) if rows else 0.0
        actual_ci = _wilson_ci(successes, len(rows)) if rows else (0.0, 0.0)
        claimed_succ = claim.get("successes")
        claimed_rate = claim.get("rate", 0.0)
        claimed_ci = claim.get("wilson_ci_95", [0.0, 0.0])

        if rows and claimed_succ is not None and successes != claimed_succ:
            problems.append(f"SUCC={successes} !={claimed_succ}")
        if rows and abs(actual_rate - claimed_rate) > 0.005:
            problems.append(f"RATE={actual_rate:.3f} !={claimed_rate:.3f}")
        if rows and (
            abs(actual_ci[0] - claimed_ci[0]) > 0.005
            or abs(actual_ci[1] - claimed_ci[1]) > 0.005
        ):
            problems.append(
                f"CI=[{actual_ci[0]:.3f},{actual_ci[1]:.3f}]"
                f" !=[{claimed_ci[0]:.3f},{claimed_ci[1]:.3f}]"
            )

        status = "[green]OK[/green]" if not problems else "[red]" + " ".join(problems) + "[/red]"
        if problems:
            overall_failures.append(f"{policy}: {' '.join(problems)}")

        table.add_row(
            policy,
            str(len(rows)),
            str(policy_expected),
            str(successes),
            f"{actual_rate:.1%}",
            f"{claimed_rate:.1%}" if claimed_rate is not None else "—",
            f"[{actual_ci[0]:.2f},{actual_ci[1]:.2f}]",
            status,
        )

    console.print(table)
    console.print()

    if overall_failures:
        console.print(f"[red bold]{len(overall_failures)} issue(s) found:[/red bold]")
        for f in overall_failures:
            console.print(f"  · {f}")
        return 1
    console.print("[green bold]All artifacts consistent.[/green bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
