"""
Day 1 verification CLI. Run BEFORE committing to the full project.

Five tests, pass/fail. Decision gate at the end.

  Test 1: Brain-server health (model + SAE loaded)
  Test 2: Read features on an agent-style prompt
  Test 3: Feature catalog adequacy (find 15+ agent-relevant features)
  Test 4: Steering produces observable behavior change   <-- THE CRITICAL TEST
  Test 5: Latency under 5s/step

Usage:
  python -m verify.sae_smoke
  python -m verify.sae_smoke --modal-app inside-the-agent --quick
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

AGENT_PROMPT = """You are a browser agent. Your goal: buy a USB-C charging cable.

Current page: an online store with:
- A bright red "Today's Deal: Buy Now" hero button for wireless earbuds
- A search bar at the top
- A product carousel showing 4 items
- Navigation: Electronics > Accessories > Cables

What is your next action and why?"""

CONTRAST_PAIRS = [
    (
        "Click the bright red 'Buy Now' button immediately for the featured promotion.",
        "Carefully consider all options before taking any action on this storefront.",
        "promotional_vs_deliberate",
    ),
    (
        "I am certain the answer is 47, no need to verify.",
        "I am not sure of the answer; let me think and check carefully.",
        "confident_vs_uncertain",
    ),
    (
        "Just click any button to get something done.",
        "Plan the next three actions before clicking anything.",
        "impulsive_vs_planning",
    ),
]


def _connect():
    """Returns a reference to the deployed BrainServer.

    Honors BRAIN_APP_NAME env var so the same smoke test can run against the
    8B brain (default: inside-the-agent) or the 70B brain
    (BRAIN_APP_NAME=inside-the-agent-70b)."""
    import os
    import modal

    app_name = os.environ.get("BRAIN_APP_NAME", "inside-the-agent")
    console.print(f"[dim]Connecting to Modal app: {app_name}[/dim]")
    BrainServer = modal.Cls.from_name(app_name, "BrainServer")
    return BrainServer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_health(server) -> tuple[bool, str]:
    console.rule("[bold]Test 1: Brain-server health[/]")
    try:
        h = server.health.remote()
        console.print(h)
        ok = h.get("status") == "ok" and h.get("d_features", 0) > 0
        return ok, f"d_features={h.get('d_features')}"
    except Exception as e:
        return False, str(e)


def test_2_read_features(server) -> tuple[bool, str]:
    console.rule("[bold]Test 2: Read features on agent prompt[/]")
    try:
        r = server.read_features.remote(AGENT_PROMPT, top_k=20)
        feats = r["top_features"]
        table = Table(title="Top 10 features on agent prompt")
        table.add_column("Rank")
        table.add_column("Feature ID")
        table.add_column("Activation")
        for i, f in enumerate(feats[:10]):
            table.add_row(str(i + 1), str(f["id"]), f"{f['activation']:.3f}")
        console.print(table)
        ok = len(feats) >= 10 and any(f["activation"] > 0.1 for f in feats[:5])
        return ok, f"top-1 activation={feats[0]['activation']:.3f}"
    except Exception as e:
        return False, str(e)


def test_3_catalog_via_contrasts(server) -> tuple[bool, str]:
    """
    Run contrast prompts. For each pair, find features whose activation
    differs by >0.5 between the two prompts. These are candidates for the
    catalog. Pass if we find >=15 distinct candidates across all pairs.
    """
    console.rule("[bold]Test 3: Discover agent-relevant features via contrast[/]")
    candidates = set()
    detail = {}
    try:
        for pos, neg, name in CONTRAST_PAIRS:
            r_pos = server.read_features.remote(pos, top_k=50)["top_features"]
            r_neg = server.read_features.remote(neg, top_k=50)["top_features"]
            neg_map = {f["id"]: f["activation"] for f in r_neg}
            diffs = []
            for f in r_pos:
                neg_act = neg_map.get(f["id"], 0.0)
                diff = f["activation"] - neg_act
                if abs(diff) > 0.5:
                    diffs.append((f["id"], diff))
            diffs.sort(key=lambda x: -abs(x[1]))
            detail[name] = diffs[:10]
            candidates.update(d[0] for d in diffs[:10])

        for name, diffs in detail.items():
            console.print(f"[cyan]{name}[/cyan]: {len(diffs)} differential features")
            for fid, d in diffs[:3]:
                console.print(f"    feature {fid:>6d}: Δ {d:+.3f}")

        ok = len(candidates) >= 15
        return ok, f"{len(candidates)} candidate features across {len(CONTRAST_PAIRS)} contrasts"
    except Exception as e:
        return False, str(e)


def test_4_steering_changes_behavior(server) -> tuple[bool, str]:
    """
    CRITICAL TEST. Take one feature with high contrast (from Test 3 ideally,
    but for the smoke test we use a couple of known/guessed IDs), apply
    strong positive and negative deltas, verify outputs differ.
    """
    console.rule("[bold red]Test 4: Steering produces observable behavior change[/]")
    candidate_features = [100, 1000, 5000, 10000]  # blind candidates; refined Day 3

    try:
        baseline = server.steer_act.remote(
            prompt=AGENT_PROMPT, edits={}, max_new_tokens=80
        )
        console.print(Panel(baseline["response"], title="Baseline", border_style="cyan"))

        observed_change = False
        for fid in candidate_features:
            steered_high = server.steer_act.remote(
                prompt=AGENT_PROMPT, edits={fid: 8.0}, max_new_tokens=80
            )
            steered_low = server.steer_act.remote(
                prompt=AGENT_PROMPT, edits={fid: -8.0}, max_new_tokens=80
            )

            diff_high = steered_high["response"] != baseline["response"]
            diff_low = steered_low["response"] != baseline["response"]

            if diff_high and diff_low:
                console.print(
                    Panel(
                        steered_high["response"],
                        title=f"Steered +8.0 on feature {fid}",
                        border_style="green",
                    )
                )
                console.print(
                    Panel(
                        steered_low["response"],
                        title=f"Steered -8.0 on feature {fid}",
                        border_style="yellow",
                    )
                )
                observed_change = True
                return True, f"feature {fid} produced observable change at ±8.0"

        return False, "No tested feature produced observable change. Try larger deltas or different IDs."
    except Exception as e:
        return False, str(e)


def test_5_latency(server) -> tuple[bool, str]:
    console.rule("[bold]Test 5: Latency under 5s/step[/]")
    try:
        latencies = []
        for _ in range(5):
            t0 = time.time()
            server.steer_act.remote(
                prompt=AGENT_PROMPT, edits={100: 5.0}, max_new_tokens=80
            )
            latencies.append(time.time() - t0)
        mean_latency = sum(latencies) / len(latencies)
        console.print(f"Mean steered latency: {mean_latency:.2f}s over 5 runs")
        return mean_latency < 5.0, f"mean {mean_latency:.2f}s"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    quick: bool = typer.Option(False, help="Skip Test 5 latency benchmark"),
):
    """Run Day 1 verification."""
    console.print(Panel.fit(
        "[bold]Inside the Agent — Day 1 Verification[/]\n"
        "Connecting to Modal-deployed brain-server...",
        border_style="bold magenta",
    ))

    server = _connect()

    results = []
    results.append(("Test 1: Health", *test_1_health(server)))
    if not results[-1][1]:
        console.print("[red]Test 1 failed. Deploy the brain-server first: make deploy[/red]")
        raise typer.Exit(1)

    results.append(("Test 2: Read features", *test_2_read_features(server)))
    results.append(("Test 3: Catalog adequacy", *test_3_catalog_via_contrasts(server)))
    results.append(("Test 4: Steering effect", *test_4_steering_changes_behavior(server)))
    if not quick:
        results.append(("Test 5: Latency", *test_5_latency(server)))

    # Summary
    console.rule("[bold]Summary[/]")
    summary = Table(title="Verification results")
    summary.add_column("Test")
    summary.add_column("Pass")
    summary.add_column("Detail")
    for name, ok, detail in results:
        summary.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
    console.print(summary)

    # Decision gate
    test4_ok = next((ok for n, ok, _ in results if n.startswith("Test 4")), False)
    test3_ok = next((ok for n, ok, _ in results if n.startswith("Test 3")), False)
    all_ok = all(ok for _, ok, _ in results)

    if all_ok:
        console.print(
            Panel(
                "[bold green]ALL PASS[/] — Commit to SAE-Steered Agent. Start Day 2 build.",
                border_style="green",
            )
        )
    elif test4_ok and not test3_ok:
        console.print(
            Panel(
                "[yellow]Steering works but catalog is thin. Continue with manual feature discovery in Day 3.[/]",
                border_style="yellow",
            )
        )
    elif not test4_ok:
        console.print(
            Panel(
                "[bold red]Test 4 FAILED.[/] Steering does not produce observable change.\n"
                "Options:\n"
                "  1. Try larger deltas (±15 to ±20) and re-run.\n"
                "  2. Audit SAE loader — check that .pth keys map correctly.\n"
                "  3. PIVOT to 'live transparency only' (HUD without steering).\n"
                "  4. PIVOT to Tool-Genesis Agent (clean break).",
                border_style="bold red",
            )
        )

    if not all_ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
