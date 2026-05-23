"""
verify/learn_dynamic_thresholds.py — v0.23 P2 prototype.

Learns per-feature activation thresholds for `policies/dynamic.py` from
saved trajectory data instead of using the hand-set 0.40 default.

Method:
  For each watched feature F in WATCH_FEATURES (from policies/dynamic.py),
  partition all logged step records into "success" and "failure" buckets
  by the trial's lenient outcome. At each candidate threshold τ in {0.1,
  0.15, ..., 1.5}, compute:

     precision(τ) = P(failure | feature F activation >= τ)
     recall(τ)    = P(feature F activation >= τ | failure)
     f1(τ)        = 2 · precision · recall / (precision + recall)

  Pick the τ that maximizes f1(τ). That's the new threshold for feature F.

Output:
  policies/dynamic_thresholds.json
    {
      "50853": {"threshold": 0.55, "f1": 0.71, "precision": 0.68, "recall": 0.74},
      "19079": {"threshold": 0.42, ...},
      ...
    }

The next dynamic_policy revision reads this file at import time and uses
the learned thresholds. Falls back to the hand-set defaults if the file
is missing.

Inputs:
  Default: all data/trajectories/*_baseline.jsonl (so we learn from runs
  where the policy DID NOT intervene). Override with --trajectories.
  Failure label: read from the corresponding data/results/baseline.jsonl
  row matched by (task_id, seed).

Usage:
    python -m verify.learn_dynamic_thresholds
    python -m verify.learn_dynamic_thresholds --policy baseline --trajectories "data/trajectories/promo_*_baseline.jsonl"
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)


# Watched features = same set policies/dynamic.py watches. Importing
# WATCH_FEATURES directly avoids the constant drift problem.
try:
    from policies.dynamic import WATCH_FEATURES
except ImportError:
    WATCH_FEATURES = {
        50853: {}, 19079: {}, 39820: {}, 44602: {}, 34048: {},
    }


CANDIDATE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 31)]  # 0.05 .. 1.50


def _load_trial_success(policy: str) -> dict[tuple[str, int], bool]:
    """task_id × seed → success bool, from data/results/<policy>.jsonl."""
    out: dict[tuple[str, int], bool] = {}
    path = Path(f"data/results/{policy}.jsonl")
    if not path.exists():
        console.print(f"[yellow]no data/results/{policy}.jsonl[/yellow]")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = r.get("run_id", "")
        # run_id is like "promo_held_001_seed_0_baseline" — last digit is seed.
        # Safer: parse from log_path or trial fields.
        task_id = r.get("task_id", "")
        # seed isn't a direct field but log_path includes _seed_N_
        log_path = r.get("log_path", "")
        seed = None
        if "_seed_" in log_path:
            try:
                seed = int(log_path.split("_seed_")[1].split("_")[0])
            except ValueError:
                pass
        if task_id and seed is not None:
            out[(task_id, seed)] = bool(r.get("success"))
    return out


def _collect_activations(
    trajectory_glob: str,
    success_map: dict[tuple[str, int], bool],
) -> dict[int, list[tuple[float, bool]]]:
    """For each step in every matched trajectory, record (activation, was_failure)
    for every watched feature. Returns {feature_id: [(act, fail_flag), ...]}."""
    samples: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    files = sorted(glob.glob(trajectory_glob))
    if not files:
        console.print(f"[red]No trajectory files matched: {trajectory_glob}[/red]")
        return samples
    n_files = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                steps = [json.loads(l) for l in f if l.strip()]
        except Exception as e:
            console.print(f"[yellow]skipping {path}: {e}[/yellow]")
            continue
        if not steps:
            continue
        first = steps[0]
        task_id = first.get("task_id", "")
        log_path_stem = Path(path).stem
        # filename pattern: <task_id>_seed_<N>_<policy>.jsonl
        try:
            seed = int(log_path_stem.split("_seed_")[1].split("_")[0])
        except (ValueError, IndexError):
            seed = 0
        is_failure = not success_map.get((task_id, seed), True)
        n_files += 1
        for step in steps:
            for feat in step.get("features", []):
                fid = feat.get("id")
                act = feat.get("activation", 0.0)
                if fid in WATCH_FEATURES:
                    samples[fid].append((act, is_failure))
    console.print(f"[green]Processed {n_files} trajectories[/green]")
    return samples


def _learn_threshold(samples: list[tuple[float, bool]]) -> dict:
    """For one feature, find the threshold τ that maximizes F1 score
    on (act >= τ → predict failure)."""
    if not samples:
        return {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
    n_failures = sum(1 for _, fail in samples if fail)
    if n_failures == 0:
        return {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": len(samples)}

    best = {"threshold": 0.40, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": len(samples)}
    for τ in CANDIDATE_THRESHOLDS:
        tp = sum(1 for act, fail in samples if act >= τ and fail)
        fp = sum(1 for act, fail in samples if act >= τ and not fail)
        fn = sum(1 for act, fail in samples if act < τ and fail)
        if tp == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {
                "threshold": τ,
                "f1": round(f1, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "n": len(samples),
            }
    return best


@app.command()
def main(
    policy: str = typer.Option(
        "baseline",
        help="Policy whose trajectories we learn from. baseline is the natural "
             "choice because it lets the watched features fire freely without "
             "the targeted suppression bias.",
    ),
    trajectory_glob: str = typer.Option(
        "data/trajectories/*_baseline.jsonl",
        help="Glob pattern for trajectory files. Default = all baseline runs.",
    ),
    out_path: str = typer.Option(
        "policies/dynamic_thresholds.json",
        help="Where to write the learned thresholds. dynamic_policy reads this.",
    ),
):
    """Learn per-feature thresholds for the dynamic policy."""
    console.print(f"[bold]Learning thresholds from {policy} trajectories[/bold]")
    success_map = _load_trial_success(policy)
    console.print(f"[green]Loaded {len(success_map)} trial outcomes[/green]")
    samples = _collect_activations(trajectory_glob, success_map)

    table = Table(title="Learned thresholds (failure-prediction F1)")
    table.add_column("feature")
    table.add_column("τ", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("n samples", justify="right")

    out: dict[str, dict] = {}
    for fid in WATCH_FEATURES:
        learned = _learn_threshold(samples.get(fid, []))
        out[str(fid)] = learned
        table.add_row(
            f"f{fid}",
            f"{learned['threshold']:.2f}" if learned['threshold'] is not None else "—",
            f"{learned['f1']:.2f}",
            f"{learned['precision']:.2f}",
            f"{learned['recall']:.2f}",
            str(learned["n"]),
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    console.print(table)
    console.print(f"\n[bold]Wrote {out_path}[/bold]")
    console.print(
        "[dim]To use these thresholds: update policies/dynamic.py to read this "
        "file at import time. Falls back to hand-set defaults if file is "
        "missing or a feature has no learned threshold.[/dim]"
    )


if __name__ == "__main__":
    app()
