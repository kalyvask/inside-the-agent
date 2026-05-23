"""
verify/compositional_steering.py — v0.23 P2 prototype.

Tests the "feature cluster carries the meaning" hypothesis. For each of
f26737's top-K decoder-cosine neighbors, run an A/B comparison:

  base policy:        f26737 -6   (the targeted f26737-only ablation)
  composed policy:    f26737 -6  +  neighbor_i  -6 (same magnitude)

If compositional > base, the feature cluster carries the meaning (neighbors
in decoder-cosine space share a circuit). If compositional ≈ base, the
single feature carries it. If compositional << base, the neighbor cancels
out the effect (suggesting it encodes the opposite concept).

The hard part is the brain-call orchestration — we already have
feature_decoder_similarity on the Modal server. This script wires it
together with a small per-condition benchmark.

Cost: 1 decoder_similarity call (~1s) + K * trial_count brain calls
(default K=5, trials=2 = 10 trials). At ~5s/step * 12 steps avg = ~10
min Modal total.

Run AFTER v0_8_finalize is done so Modal containers aren't overloaded.

Usage:
    python -m verify.compositional_steering
    python -m verify.compositional_steering --top-k 5 --trials 2 --task promo_held_001
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)


SEED_FEATURE_ID = 26737
SEED_DELTA = -6.0
SEED_LABEL = "f26737_ui_selection_vocab"


def _make_brain_call():
    import modal
    app_name = os.environ.get("BRAIN_APP_NAME", "inside-the-agent")
    BrainServer = modal.Cls.from_name(app_name, "BrainServer")
    return BrainServer()


@app.command()
def main(
    top_k: int = typer.Option(5, help="Number of decoder-cosine neighbors to test"),
    trials: int = typer.Option(2, help="Trials per (neighbor) condition"),
    task: str = typer.Option(
        "promo_held_001",
        help="Single task id from held_out.json to run on. Default is the "
             "task with a clean baseline trajectory cached.",
    ),
    out_path: str = typer.Option("artifacts/compositional_steering.json"),
):
    """For each of f26737's top-K decoder neighbors, A/B vs f26737-alone."""

    # 1. Get the decoder neighbors of f26737.
    server = _make_brain_call()
    console.print(f"[cyan]Fetching top {top_k} decoder neighbors of f{SEED_FEATURE_ID}...[/cyan]")
    sims = server.feature_decoder_similarity.remote(
        feature_ids=[SEED_FEATURE_ID], top_k=top_k
    )
    neighbors = sims["similarities"][SEED_FEATURE_ID]
    console.print(f"[green]Neighbors:[/green]")
    for n in neighbors:
        console.print(f"  f{n['feature_id']}  cosine={n['cosine_sim']:.3f}")

    # 2. Build a temporary policies dict on-the-fly. We do this by calling
    #    bench.runner programmatically per condition. Each condition is
    #    "f26737 alone" or "f26737 + neighbor_i".
    #
    #    For brevity in this prototype, we use the run_one_trial path of
    #    bench.runner via subprocess + a custom policy file written on-
    #    the-fly. (A cleaner version would expose a Python entry-point.)

    import subprocess
    tmp_policy_dir = Path("policies/_compositional_tmp")
    tmp_policy_dir.mkdir(parents=True, exist_ok=True)
    init_path = tmp_policy_dir / "__init__.py"
    init_path.write_text("# tmp module for compositional_steering\n", encoding="utf-8")

    conditions = [
        {
            "name": "f26737_only_baseline",
            "edits": [(SEED_FEATURE_ID, SEED_DELTA, SEED_LABEL)],
        },
    ]
    for n in neighbors:
        nid = n["feature_id"]
        conditions.append({
            "name": f"f26737_plus_f{nid}",
            "edits": [
                (SEED_FEATURE_ID, SEED_DELTA, SEED_LABEL),
                (nid, SEED_DELTA, f"f{nid}_neighbor_cos{n['cosine_sim']:.2f}"),
            ],
        })

    results: list[dict] = []
    for cond in conditions:
        console.print(f"\n[bold yellow]Running condition: {cond['name']}[/bold yellow]")
        # Write the temp policy module
        policy_file = tmp_policy_dir / f"_p_{cond['name']}.py"
        edits_repr = ", ".join(
            f'(feature_id={fid}, delta={d}, label={repr(lbl)})'
            for fid, d, lbl in cond["edits"]
        )
        edits_lines = "\n".join(
            f'    plan.add(feature_id={fid}, delta={d}, label={repr(lbl)}, source={repr(cond["name"])})'
            for fid, d, lbl in cond["edits"]
        )
        policy_file.write_text(
            f'"""auto-generated compositional condition: {cond["name"]}"""\n'
            f"from sae.steering_controller import SteeringPlan\n\n"
            f"def policy(features_dict, step_idx, catalog=None, **_):\n"
            f"    plan = SteeringPlan()\n"
            f"    if step_idx != 0:\n"
            f"        return plan\n"
            f"{edits_lines}\n"
            f"    return plan\n",
            encoding="utf-8",
        )

        # Run the bench runner once per trial. Since we can't easily
        # register a temp policy into POLICY_REGISTRY at runtime via
        # subprocess, we use the `targeted-f26737-only` policy as a
        # stand-in only for the baseline condition and skip neighbors
        # in this prototype. Real usage: lift the runner into a Python
        # entrypoint that takes a policy_fn callable directly.

        if cond["name"] == "f26737_only_baseline":
            policy_name = "targeted-f26737-only"
        else:
            console.print(
                f"[yellow]  skipped — needs runtime policy registration "
                f"(documented limitation of this prototype). Run the named "
                f"condition manually after registering the policy.[/yellow]"
            )
            results.append({
                "condition": cond["name"],
                "status": "not_run",
                "reason": "prototype: needs Python-entry-point runner",
                "edits": cond["edits"],
            })
            continue

        # Run trials
        succs = 0
        for trial_seed in range(trials):
            cmd = [
                sys.executable, "-u", "-m", "bench.runner",
                "--policy", policy_name,
                "--tasks", "shopgym/tasks/held_out.json",
                "--trials", "1",
                "--limit", "1",
                "--position-mode", "all",
                "--output", f"data/results_compositional/{cond['name']}_seed{trial_seed}",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    env={**os.environ, "HUD_PUBLISH": "0", "OUTPUT_SUFFIX": cond["name"]},
                    capture_output=True, text=True, timeout=1200,
                )
                # Parse success from runner stdout
                for line in proc.stdout.splitlines():
                    if "Success rate" in line and "1/1" in line:
                        succs += 1
                        break
                    if "100.00%" in line:
                        succs += 1
                        break
            except subprocess.TimeoutExpired:
                console.print(f"[red]  trial {trial_seed} timed out[/red]")

        rate = succs / max(1, trials)
        console.print(f"  → {succs}/{trials} = {rate:.0%}")
        results.append({
            "condition": cond["name"],
            "edits": cond["edits"],
            "successes": succs,
            "trials": trials,
            "rate": rate,
        })

    # 3. Output
    out = {
        "seed_feature_id": SEED_FEATURE_ID,
        "seed_delta": SEED_DELTA,
        "neighbors": neighbors,
        "results": results,
        "task": task,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    console.print(f"\n[bold]Wrote {out_path}[/bold]")

    # Headline table
    table = Table(title="Compositional steering — f26737 + neighbor", show_lines=False)
    table.add_column("condition")
    table.add_column("edits")
    table.add_column("rate", justify="right")
    for r in results:
        edits_str = " + ".join(f"f{e[0]}({e[1]:+.0f})" for e in r["edits"])
        if r.get("status") == "not_run":
            rate_str = "[yellow]skipped[/yellow]"
        else:
            rate_str = f"{r['successes']}/{r['trials']} = {r['rate']*100:.0f}%"
        table.add_row(r["condition"], edits_str, rate_str)
    console.print(table)
    console.print(
        "\n[dim]Prototype limitation: only the baseline-f26737-only condition runs "
        "directly. For neighbor conditions, register each as a real policy in "
        "policies/__init__.py POLICY_REGISTRY first, then re-run with --policy <name>.[/dim]"
    )


if __name__ == "__main__":
    app()
