"""
Day 3 — Pass 2: calibrate steering magnitudes.

The auto-discovered deltas in features.yaml (±3 / ±4) are too aggressive and
produce degenerate output. This script tests each feature at progressively
smaller magnitudes and picks the LARGEST magnitude where the agent's output
stays coherent while still differing from baseline.

Coherence checks:
  - No single token dominates >30% of output
  - No 3-gram repeats >20% of trigrams
  - At least 5 distinct word tokens
  - Output has at least one alphabetic word

Outputs an updated features.yaml with calibrated `recommended_delta` per feature.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import modal
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


AGENT_TEST_PROMPT = """You are a browser agent. Goal: buy a USB-C charging cable.

Current page: an online store with:
- A bright red "Today's Deal: Buy Now" hero button for wireless earbuds
- A search bar at the top
- A product carousel showing 4 items

What is your next action and why? Respond in one short sentence."""


# Wider range now that the residual-stream clamp bug is fixed.
# Goodfire's typical demo magnitudes are 2-8 on their SDK.
CANDIDATE_MAGNITUDES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def is_coherent(text: str) -> tuple[bool, str]:
    """Return (coherent, reason_if_not)."""
    text = text.strip()
    if len(text) < 30:
        return False, "too_short"

    tokens = text.split()
    if len(tokens) < 8:
        return False, "too_few_tokens"

    # Need actual English words.
    words = _WORD_RE.findall(text)
    if len(words) < 5:
        return False, "no_words"

    counter = Counter(tokens)
    top_freq = counter.most_common(1)[0][1] / len(tokens)
    if top_freq > 0.3:
        return False, f"token_dominance_{top_freq:.0%}"

    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if trigrams:
        trigram_count = Counter(trigrams)
        top_trigram = trigram_count.most_common(1)[0][1] / len(trigrams)
        if top_trigram > 0.2:
            return False, f"trigram_repeat_{top_trigram:.0%}"

    return True, "ok"


def differs(baseline: str, steered: str) -> bool:
    """Are the two outputs meaningfully different?"""
    if baseline.strip() == steered.strip():
        return False
    # Token-level Jaccard - if very similar, they don't differ meaningfully.
    b_tokens = set(baseline.lower().split())
    s_tokens = set(steered.lower().split())
    if not b_tokens or not s_tokens:
        return True
    overlap = len(b_tokens & s_tokens) / len(b_tokens | s_tokens)
    return overlap < 0.9


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------


def _connect():
    import os
    app_name = os.environ.get("BRAIN_APP_NAME", "inside-the-agent")
    BrainServer = modal.Cls.from_name(app_name, "BrainServer")
    return BrainServer()


def find_best_magnitude(
    server, feature_id: int, sign: int, baseline_response: str
) -> tuple[float | None, str, str]:
    """
    Sign: +1 for amplify, -1 for suppress.

    Returns (best_magnitude, status, evidence) where status is one of:
      - 'tuned'      : found a working magnitude
      - 'no_effect'  : even at 2.5 no coherent change observable
      - 'too_fragile': only breaks coherence, no coherent change found
    """
    best = None
    last_steered = ""
    # Iterate from SMALLEST to LARGEST. Stop when we hit incoherence.
    for mag in CANDIDATE_MAGNITUDES:
        polarity = sign * mag
        result = server.steer_act.remote(
            prompt=AGENT_TEST_PROMPT,
            edits={feature_id: polarity},
            max_new_tokens=80,
            temperature=0.2,
        )
        text = result["response"].strip()
        coherent, reason = is_coherent(text)
        diff = differs(baseline_response, text)
        last_steered = text

        if coherent and diff:
            best = polarity  # keep updating to find the LARGEST working magnitude
        elif not coherent:
            # Output broke at this magnitude; stop scaling up.
            return (best, "tuned" if best is not None else "too_fragile", last_steered)

    return (best, "tuned" if best is not None else "no_effect", last_steered)


def main():
    server = _connect()

    # FEATURES_OUT_PATH env var so the 70B run can target a separate yaml.
    import os
    yaml_path = Path(os.environ.get("FEATURES_OUT_PATH", "sae/features.yaml"))
    catalog = yaml.safe_load(yaml_path.read_text()) or {}

    # Get one baseline response we can compare against. Same prompt is used
    # across all feature tests for consistency.
    console.print("[cyan]Getting baseline response...[/cyan]")
    baseline = server.steer_act.remote(
        prompt=AGENT_TEST_PROMPT, edits={}, max_new_tokens=80, temperature=0.2
    )["response"].strip()
    console.print(f"Baseline: [dim]{baseline[:140]}...[/dim]\n")

    results = []
    for category, entries in catalog.items():
        if not entries:
            continue
        console.rule(f"[bold]{category}[/]")
        for entry in entries:
            fid = entry["id"]
            old_delta = entry.get("recommended_delta", 0.0)
            sign = 1 if old_delta > 0 else -1
            console.print(f"Tuning feature {fid} ({entry.get('label')}, sign={sign:+d})...")
            best, status, evidence = find_best_magnitude(server, fid, sign, baseline)

            if status == "tuned":
                entry["recommended_delta"] = float(best)
                entry["calibrated_delta"] = float(best)
                entry["tuning_status"] = "tuned"
            elif status == "no_effect":
                entry["calibrated_delta"] = None
                entry["tuning_status"] = "no_observable_effect_within_2.5"
                # downgrade confidence
                entry["confidence"] = "low"
            else:
                entry["calibrated_delta"] = None
                entry["tuning_status"] = "fragile_breaks_immediately"
                entry["confidence"] = "low"

            # Update the causal_effect excerpt with the latest steered output.
            entry.setdefault("causal_effect", {})["tuned_excerpt"] = evidence[:160]

            results.append({
                "feature_id": fid,
                "label": entry.get("label"),
                "old_delta": old_delta,
                "new_delta": best,
                "status": status,
            })
            console.print(
                f"  → best magnitude: {best if best is not None else 'NONE':>6}, status: {status}"
            )

    # Write back
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated by verify/feature_drill.py and tuned by verify/tune_deltas.py\n")
        f.write("# Each `recommended_delta` is the largest magnitude that produced\n")
        f.write("# a coherent + behaviorally-distinct output on a USB-C cable agent prompt.\n\n")
        yaml.safe_dump(catalog, f, sort_keys=False, default_flow_style=False)

    # Summary table
    console.rule("[bold]Tuning summary[/]")
    table = Table(title="Magnitude calibration")
    table.add_column("Feature")
    table.add_column("Label")
    table.add_column("Old δ")
    table.add_column("Tuned δ")
    table.add_column("Status")
    for r in results:
        table.add_row(
            str(r["feature_id"]),
            r["label"] or "",
            f"{r['old_delta']:+.1f}",
            f"{r['new_delta']:+.1f}" if r["new_delta"] is not None else "—",
            r["status"],
        )
    console.print(table)

    tuned = sum(1 for r in results if r["status"] == "tuned")
    console.print(f"\n[bold]{tuned} of {len(results)} features tuned successfully.[/]")
    console.print(f"YAML updated at: [cyan]{yaml_path}[/cyan]")


if __name__ == "__main__":
    main()
