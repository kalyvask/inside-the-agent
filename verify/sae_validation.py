"""
Full SAE validation suite. Run this before trusting any feature claim.

Checks:
  1. Mean L0 per token vs Goodfire's reported 91
  2. Mean reconstruction relative error ||x - x_hat|| / ||x||
  3. Decoder and encoder norm distributions
  4. Wrong-layer sanity (L19 SAE applied to L0 activations should be garbage)
  5. Deterministic top-feature on a fixed prompt

Outputs artifacts/sae_validation.json with full numbers.

Usage:
    python -m verify.sae_validation
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
from rich.console import Console
from rich.table import Table

console = Console()


def _connect():
    BrainServer = modal.Cls.from_name("inside-the-agent", "BrainServer")
    return BrainServer()


def main():
    server = _connect()
    console.print("[cyan]Running full SAE validation suite...[/cyan]")
    report = server.sae_validation.remote()

    # Pretty-print
    console.rule("[bold]L0 sparsity[/]")
    console.print(f"Mean L0 per token: {report['mean_l0_per_token']:.1f}")
    console.print(f"Expected from card: {report['expected_l0_from_card']}")
    ratio = report['mean_l0_per_token'] / report['expected_l0_from_card']
    if 0.5 <= ratio <= 2.0:
        console.print(f"[green]Within 2x of expected ({ratio:.2f}x). PASS.[/green]")
    else:
        console.print(f"[red]Off by {ratio:.1f}x. ENCODER LIKELY WRONG.[/red]")

    console.rule("[bold]Reconstruction error[/]")
    err = report["mean_reconstruction_relative_error"]
    console.print(f"Mean ||x - x_hat|| / ||x||: {err:.4f}")
    if err < 0.3:
        console.print(f"[green]Reconstruction is tight. PASS.[/green]")
    elif err < 0.6:
        console.print(f"[yellow]Reconstruction is loose ({err:.2f}). Common for sparse SAEs but worth noting.[/yellow]")
    else:
        console.print(f"[red]Reconstruction is too loose ({err:.2f}). Encoder/decoder may be miscalibrated.[/red]")

    console.rule("[bold]Decoder + encoder norms[/]")
    table = Table()
    table.add_column("Matrix")
    table.add_column("Min")
    table.add_column("Median")
    table.add_column("Max")
    table.add_column("Mean")
    table.add_column("Std")
    for name, key in [("W_dec rows", "decoder_norms"), ("W_enc cols", "encoder_norms")]:
        s = report[key]
        table.add_row(
            name,
            f"{s['min']:.3f}",
            f"{s['median']:.3f}",
            f"{s['max']:.3f}",
            f"{s['mean']:.3f}",
            f"{s['std']:.3f}",
        )
    console.print(table)

    console.rule("[bold]Wrong-layer sanity (apply L19 SAE to L0 activations)[/]")
    wrong_l0 = report["wrong_layer_l0_layer_0"]
    console.print(f"L0 when SAE is mis-applied to layer 0: {wrong_l0:.1f}")
    correct_l0 = report["mean_l0_per_token"]
    if wrong_l0 is None:
        console.print("[yellow]Could not capture layer 0 activations.[/yellow]")
    elif wrong_l0 > correct_l0 * 1.5 or wrong_l0 < correct_l0 * 0.5:
        console.print(
            f"[green]Wrong-layer L0 ({wrong_l0:.1f}) differs from correct-layer L0 "
            f"({correct_l0:.1f}). Hook position is correct. PASS.[/green]"
        )
    else:
        console.print(
            f"[yellow]Wrong-layer L0 ({wrong_l0:.1f}) is suspiciously similar to "
            f"correct ({correct_l0:.1f}). Layer hook may not be doing what we think.[/yellow]"
        )

    console.rule("[bold]Deterministic top-features on 'The cat sat on the mat.'[/]")
    sanity = report["sanity_top_features_cat_prompt"]
    table2 = Table()
    table2.add_column("Rank")
    table2.add_column("Feature ID")
    table2.add_column("Activation")
    for i, f in enumerate(sanity, 1):
        table2.add_row(str(i), str(f["feature_id"]), f"{f['activation']:.3f}")
    console.print(table2)
    console.print(
        "[dim]These IDs should be stable across reruns. If you see different IDs on a fresh "
        "deploy, something about the SAE loading changed.[/dim]"
    )

    # Save
    out = Path("artifacts/sae_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
