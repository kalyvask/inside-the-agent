"""
Day 7 analysis: load benchmark results, compute headline numbers + CIs,
mine failure patterns, generate the chart for the demo.

Usage:
  python -m bench.analysis             # uses data/results/*.jsonl
  python -m bench.analysis --plot      # also generate matplotlib chart
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


def _load_results(results_dir: str = "data/results") -> dict[str, list[dict]]:
    """Load per-policy results from JSONL files. Returns {policy: [runs]}."""
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        policy = path.stem
        runs = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        out[policy] = runs
    return out


def _load_trajectories(trajectories_dir: str = "data/trajectories") -> dict[str, list[dict]]:
    """Load per-run trajectories. Returns {run_id: [step_logs]}."""
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(trajectories_dir).glob("*.jsonl")):
        run_id = path.stem
        steps = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        out[run_id] = steps
    return out


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Headline analysis
# ---------------------------------------------------------------------------


def headline_table(results: dict[str, list[dict]]) -> None:
    """Per-policy, per-category success with 95% CIs."""
    by_policy_category = defaultdict(lambda: defaultdict(list))
    for policy, runs in results.items():
        for r in runs:
            cat = r.get("task_id", "").split("_")[0]  # promo, halluc, plan
            by_policy_category[policy][cat].append(bool(r.get("success")))

    all_cats = sorted(
        {c for cats in by_policy_category.values() for c in cats.keys()}
    )

    table = Table(title="Headline: success rate per policy × category")
    table.add_column("Policy")
    for cat in all_cats:
        table.add_column(cat)
    table.add_column("overall")

    for policy in sorted(by_policy_category):
        row = [policy]
        all_successes = 0
        all_trials = 0
        for cat in all_cats:
            outs = by_policy_category[policy].get(cat, [])
            s = sum(outs)
            n = len(outs)
            all_successes += s
            all_trials += n
            if n == 0:
                row.append("—")
            else:
                lo, hi = _wilson_ci(s, n)
                row.append(f"{s/n:.1%} [{lo:.1%}, {hi:.1%}]")
        if all_trials == 0:
            row.append("—")
        else:
            lo, hi = _wilson_ci(all_successes, all_trials)
            row.append(f"{all_successes/all_trials:.1%} [{lo:.1%}, {hi:.1%}]")
        table.add_row(*row)

    console.print(table)


def control_verdict(results: dict[str, list[dict]]) -> None:
    """
    The key scientific gate.

    'targeted' (or 'dynamic'/'static') wins
    'random' and 'wrong-sign' do not beat baseline
    => causal effect of targeted steering is credible.
    """
    rates = {}
    for policy, runs in results.items():
        if not runs:
            continue
        rates[policy] = sum(r["success"] for r in runs) / len(runs)

    if not rates:
        console.print("[yellow]No results to analyze.[/yellow]")
        return

    table = Table(title="Control conditions verdict")
    table.add_column("Policy")
    table.add_column("Success")
    table.add_column("Δ vs baseline")
    baseline = rates.get("baseline", 0.0)
    for policy in ["baseline", "random", "wrong-sign", "static", "dynamic", "targeted"]:
        if policy not in rates:
            continue
        d = rates[policy] - baseline
        table.add_row(policy, f"{rates[policy]:.1%}", f"{d:+.1%}")
    console.print(table)

    targeted = rates.get("targeted") or rates.get("dynamic") or rates.get("static")
    random = rates.get("random", baseline)
    wrong = rates.get("wrong-sign", baseline)

    if targeted is None:
        console.print("[yellow]No targeted policy results to evaluate.[/yellow]")
        return

    targeted_beats = targeted > baseline + 0.05
    random_doesnt = random <= baseline + 0.02
    wrong_doesnt = wrong <= baseline + 0.02

    if targeted_beats and random_doesnt and wrong_doesnt:
        console.print(
            "[bold green]✓ CAUSAL VERDICT: targeted beats baseline, controls do not. "
            "The targeted feature edits are responsible for the improvement.[/]"
        )
    elif targeted_beats:
        console.print(
            "[yellow]⚠ Targeted beats baseline but controls also showed lift. "
            "Causal claim weakens — any intervention may help.[/]"
        )
    else:
        console.print(
            "[red]✗ Targeted did not beat baseline by ≥5 points. "
            "Result not credible as a steering effect.[/]"
        )


# ---------------------------------------------------------------------------
# Failure mode mining
# ---------------------------------------------------------------------------


def mine_failure_modes(trajectories: dict[str, list[dict]], top_n: int = 3) -> None:
    """
    For each failed baseline run, look at features firing at the failure step
    (last action before run ended). Cluster by majority-feature.
    """
    failure_signatures: list[tuple[str, frozenset]] = []
    for run_id, steps in trajectories.items():
        if not steps:
            continue
        if "baseline" not in run_id:
            continue  # Only mine baseline failures
        # Was the last step a success?
        last = steps[-1]
        if last.get("result", {}).get("reward", 0) > 0:
            continue
        # Take top-5 features at the last action.
        feats = last.get("features", [])[:5]
        sig = frozenset(f["id"] for f in feats)
        failure_signatures.append((run_id, sig))

    if not failure_signatures:
        console.print("[yellow]No baseline failures with feature logs to mine.[/yellow]")
        return

    # Count which features show up across failures.
    feature_counts = Counter()
    for _, sig in failure_signatures:
        for fid in sig:
            feature_counts[fid] += 1

    console.print(f"\n[bold]Top features at moment of baseline failure (across {len(failure_signatures)} failures):[/]")
    for fid, count in feature_counts.most_common(10):
        rate = count / len(failure_signatures)
        console.print(f"  feature {fid:>6d}: appears in {count}/{len(failure_signatures)} failures ({rate:.0%})")

    console.print("\n[dim]Day 7 work: cluster these into 3 named failure modes for the demo.[/dim]")


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------


def plot_headline(results: dict[str, list[dict]], out_path: str = "data/results/headline.png") -> None:
    """Generate the headline bar chart for the demo deck."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping plot.[/yellow]")
        return

    cats_order = ["promo", "halluc", "plan"]
    policies_order = ["baseline", "random", "wrong-sign", "targeted"]
    avail = [p for p in policies_order if p in results]

    rates = {}
    cis = {}
    for policy in avail:
        rates[policy] = {}
        cis[policy] = {}
        by_cat = defaultdict(list)
        for r in results[policy]:
            cat = r["task_id"].split("_")[0]
            by_cat[cat].append(bool(r["success"]))
        for cat in cats_order:
            outs = by_cat.get(cat, [])
            n = len(outs)
            s = sum(outs)
            rates[policy][cat] = (s / n) if n > 0 else 0
            cis[policy][cat] = _wilson_ci(s, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_pol = len(avail)
    width = 0.8 / n_pol
    x = list(range(len(cats_order)))

    for i, policy in enumerate(avail):
        vals = [rates[policy][cat] for cat in cats_order]
        lo = [vals[j] - cis[policy][cats_order[j]][0] for j in range(len(cats_order))]
        hi = [cis[policy][cats_order[j]][1] - vals[j] for j in range(len(cats_order))]
        ax.bar(
            [xi + i * width for xi in x],
            vals,
            width,
            label=policy,
            yerr=[lo, hi],
            capsize=4,
        )

    ax.set_xticks([xi + (n_pol - 1) * width / 2 for xi in x])
    ax.set_xticklabels(cats_order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title("ShopGym held-out — success by category and policy")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    console.print(f"Saved chart to {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.command()
def main(
    results_dir: str = typer.Option("data/results"),
    trajectories_dir: str = typer.Option("data/trajectories"),
    plot: bool = typer.Option(False, help="Generate headline chart PNG"),
):
    results = _load_results(results_dir)
    trajectories = _load_trajectories(trajectories_dir)

    if not results:
        console.print("[red]No results found. Run `make bench` for each policy first.[/red]")
        raise typer.Exit(1)

    headline_table(results)
    console.print()
    control_verdict(results)
    console.print()
    mine_failure_modes(trajectories)

    if plot:
        plot_headline(results)


if __name__ == "__main__":
    app()
