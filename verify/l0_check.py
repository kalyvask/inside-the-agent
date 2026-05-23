"""
L0 sparsity verification of the loaded SAE encoder.

Goodfire's released SAE for Llama 3.1-8B layer 19 reports L0 ≈ 91 active
features per token. If our ReLU-encoder produces L0 wildly different from
that (e.g. 10x lower or 100x higher), the encoder formulation is wrong and
all feature IDs/activations downstream are suspect (likely needs TopK
wrapping or different activation conventions).

Usage:
    python -m verify.l0_check                # default 30 prompts
    python -m verify.l0_check --n 100

Outputs mean, median, std, and per-prompt L0; writes artifacts/l0_report.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import modal
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)

EXPECTED_L0 = 91   # from Goodfire model card
PROMPTS = [
    "You are a browser agent. The page shows a USB-C cable and a promotional banner.",
    "Plan the next three actions before clicking anything.",
    "Add the cheapest cable to the cart.",
    "Carefully consider all options before adding to cart.",
    "Click the buy now button to purchase the featured item.",
    "Search for USB-C cables in the catalog.",
    "I am uncertain which product matches the goal.",
    "The promotional banner says Today's Deal — buy now!",
    "Decline the upsell modal and proceed to checkout.",
    "Apply discount code SAVE10 to the order.",
    "Find the cheapest item and add it to cart.",
    "Goal: buy a phone case. Stay focused on the goal.",
    "Hello, how are you today?",
    "Generate a Python script that sorts a list.",
    "Translate the following sentence into French.",
    "The capital of France is Paris.",
    "What is the boiling point of water?",
    "Compose a haiku about autumn leaves.",
    "Explain quantum entanglement in simple terms.",
    "Schedule a meeting for next Tuesday at 3pm.",
    "Please write a polite refusal email.",
    "Add the product to cart and proceed to checkout.",
    "Recommend three books about decision-making.",
    "Quick. Click whatever stands out.",
    "Reset the cart and start over from scratch.",
    "Compare the prices of cable A and cable B.",
    "Read the product description before buying.",
    "Type the search query into the search bar.",
    "Confirm the order summary before completing the purchase.",
    "Sort search results by lowest price first.",
]


def _connect():
    BrainServer = modal.Cls.from_name("inside-the-agent", "BrainServer")
    return BrainServer()


@app.command()
def main(
    n: int = typer.Option(30, help="Number of prompts to sample"),
    out: str = typer.Option("artifacts/l0_report.json", help="Output path"),
):
    server = _connect()
    prompts = PROMPTS[:n]

    console.print(f"[cyan]Querying brain-server for top-k=200 features on {len(prompts)} prompts...[/]")
    results = []
    for p in prompts:
        # Get many features so we can count how many are above near-zero.
        r = server.read_features.remote(p, top_k=200)
        feats = r["top_features"]
        # L0 = number of features with non-trivial activation
        # Goodfire's L0 counts active features, not just top-k. We approximate by
        # counting features with activation > 0.01 (anything noticeably non-zero).
        l0_strict = sum(1 for f in feats if f["activation"] > 0.01)
        l0_loose = sum(1 for f in feats if f["activation"] > 0.001)
        results.append({
            "prompt": p[:60],
            "l0_strict_001": l0_strict,
            "l0_loose_0001": l0_loose,
            "top1_act": feats[0]["activation"] if feats else 0,
        })

    strict = [r["l0_strict_001"] for r in results]
    loose = [r["l0_loose_0001"] for r in results]

    table = Table(title="L0 sparsity by prompt")
    table.add_column("Prompt (truncated)")
    table.add_column("L0 strict (>0.01)")
    table.add_column("L0 loose (>0.001)")
    for r in results[:10]:
        table.add_row(r["prompt"], str(r["l0_strict_001"]), str(r["l0_loose_0001"]))
    if len(results) > 10:
        table.add_row("...", "...", "...")
    console.print(table)

    console.rule()
    console.print(f"[bold]Mean L0 strict (>0.01):[/]   {statistics.mean(strict):.1f}")
    console.print(f"[bold]Median L0 strict:[/]          {statistics.median(strict):.1f}")
    console.print(f"[bold]Stdev L0 strict:[/]           {statistics.stdev(strict):.1f}")
    console.print(f"[bold]Mean L0 loose (>0.001):[/]   {statistics.mean(loose):.1f}")
    console.print()
    console.print(f"[bold]Expected (Goodfire card):[/] {EXPECTED_L0}")

    ratio = statistics.mean(strict) / EXPECTED_L0
    if 0.5 <= ratio <= 2.0:
        console.print(f"[green]Within 2x of expected. Encoder looks plausible.[/]")
    elif 0.1 <= ratio <= 10.0:
        console.print(
            f"[yellow]Off by {ratio:.1f}x. Possibly noise threshold issue; "
            f"investigate but probably ok.[/]"
        )
    else:
        console.print(
            f"[bold red]Off by {ratio:.1f}x. Encoder likely WRONG — "
            f"needs TopK wrapping or different activation convention.[/]"
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_prompts": len(prompts),
        "expected_l0": EXPECTED_L0,
        "mean_l0_strict_001": statistics.mean(strict),
        "median_l0_strict_001": statistics.median(strict),
        "stdev_l0_strict_001": statistics.stdev(strict),
        "mean_l0_loose_0001": statistics.mean(loose),
        "per_prompt": results,
    }, indent=2))
    console.print(f"\nWrote {out_path}")


if __name__ == "__main__":
    app()
