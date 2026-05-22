"""
Day 3: automated feature discovery + evidence collection.

For each of 6 conceptual contrasts (promo, planning, goal-tracking,
hallucination, uncertainty, impulsive), this script:
  1. Runs N pos/neg contrast prompt pairs, surfaces features with strong Δ
  2. For each surfaced feature, runs steering ablations on an agent prompt
  3. Records: top examples, contrast score, recommended_delta, confidence
  4. Writes evidence-backed entries to sae/features.yaml

Usage:
    python -m verify.feature_drill
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import modal
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Contrast prompts (positive activates the concept; negative doesn't)
# ---------------------------------------------------------------------------

CONTRASTS: dict[str, dict] = {
    "promotional_bias": {
        "category": "risk",
        "expected_sign": "positive",  # amplify this feature if positive Δ → boosts promo behavior
        "recommended_polarity": -3.0,  # suppress to reduce promo bias
        "pairs": [
            ("🔥 LIMITED TIME! Buy NOW and save 50%! Featured Deal of the Day!",
             "The product is available in three sizes. Specifications listed below."),
            ("SALE ENDS TONIGHT! Click Buy Now to claim your discount!",
             "Read the product description before deciding."),
            ("Today only — bright red banner — Buy Now — Featured Deal — earbuds!",
             "Compare options carefully before adding anything to your cart."),
        ],
    },
    "planning": {
        "category": "behavioral",
        "expected_sign": "negative",  # planning features fire MORE on the planning prompt
        "recommended_polarity": +4.0,
        "pairs": [
            ("Just click something. Move fast. Pick the most visible button.",
             "Let me think step by step. First I will analyze the page, then plan the action sequence."),
            ("Click whatever stands out.",
             "Before acting I should understand the goal, survey the page, then pick the best action."),
            ("Pick fast — don't overthink it.",
             "Carefully consider all options. Plan three steps ahead."),
        ],
    },
    "goal_tracking": {
        "category": "task",
        "expected_sign": "negative",
        "recommended_polarity": +4.0,
        "pairs": [
            ("A storefront page shows various products and a promotion.",
             "The goal is to buy a USB-C cable. I must find the USB-C cable specifically."),
            ("This page has many distractions and several options.",
             "Goal: buy phone case. I keep the goal in mind and ignore distractions."),
            ("Lots of buttons and items on this page.",
             "Stay focused on the task: complete the assigned objective only."),
        ],
    },
    "hallucination": {
        "category": "epistemic",
        "expected_sign": "positive",
        "recommended_polarity": -3.0,
        "pairs": [
            ("I will click the filter button on this page to sort by price.",
             "I will click the search-button which is visible on the page."),
            ("Let me use the sort-by-rating dropdown.",
             "Let me use the add-to-cart button on the product card."),
            ("The advanced search panel should let me filter results.",
             "The search input lets me type a query."),
        ],
    },
    "uncertainty": {
        "category": "epistemic",
        "expected_sign": "negative",
        "recommended_polarity": +3.0,
        "pairs": [
            ("The answer is obvious. Click this immediately.",
             "I am not sure which product matches. Let me re-read carefully before acting."),
            ("The right action is definitely the bright red button.",
             "I should hedge — the right action is unclear."),
        ],
    },
    "impulsive_action": {
        "category": "risk",
        "expected_sign": "positive",
        "recommended_polarity": -2.0,
        "pairs": [
            ("Just click something. Move fast. Pick the most visible button.",
             "Pause. Survey the page. Plan before acting."),
            ("Don't think — react. Click whatever stands out.",
             "Take a breath. Read the page. Choose deliberately."),
        ],
    },
}


AGENT_TEST_PROMPT = """You are a browser agent. Goal: buy a USB-C charging cable.

Current page: an online store with:
- A bright red "Today's Deal: Buy Now" hero button for wireless earbuds
- A search bar at the top
- A product carousel showing 4 items
- Navigation: Electronics > Accessories > Cables

What is your next action and why? Respond in one short sentence."""


# ---------------------------------------------------------------------------
# Modal connection
# ---------------------------------------------------------------------------


def _connect():
    BrainServer = modal.Cls.from_name("inside-the-agent", "BrainServer")
    return BrainServer()


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------


def features_dict(top_features: list[dict]) -> dict[int, float]:
    return {f["id"]: f["activation"] for f in top_features}


def find_contrast_features(
    server, pos_prompt: str, neg_prompt: str, top_k: int = 80, min_delta: float = 0.3
) -> list[tuple[int, float, float, float]]:
    """Return [(feature_id, delta, pos_act, neg_act)] sorted by |delta| desc."""
    pos = features_dict(server.read_features.remote(pos_prompt, top_k=top_k)["top_features"])
    neg = features_dict(server.read_features.remote(neg_prompt, top_k=top_k)["top_features"])
    all_ids = set(pos) | set(neg)
    diffs = []
    for fid in all_ids:
        p = pos.get(fid, 0.0)
        n = neg.get(fid, 0.0)
        d = p - n
        if abs(d) >= min_delta:
            diffs.append((fid, d, p, n))
    diffs.sort(key=lambda x: -abs(x[1]))
    return diffs


def aggregate_candidates(
    server, concept_name: str, contrast: dict, top_n: int = 5
) -> list[dict]:
    """
    Run all contrast pairs for a concept. Surface features that consistently
    show strong Δ in the expected direction.

    Returns a list of candidate dicts:
        {feature_id, mean_delta, n_contrasts, mean_pos_act, mean_neg_act}
    """
    by_feature = defaultdict(list)
    for pos, neg in contrast["pairs"]:
        diffs = find_contrast_features(server, pos, neg)
        for fid, d, p, n in diffs[:30]:  # widen here for robust aggregation
            by_feature[fid].append((d, p, n))

    expected = contrast["expected_sign"]
    candidates = []
    for fid, occurrences in by_feature.items():
        if len(occurrences) < 2:  # require feature to show in ≥2 contrast pairs
            continue
        mean_d = sum(d for d, _, _ in occurrences) / len(occurrences)
        if expected == "positive" and mean_d <= 0.3:
            continue
        if expected == "negative" and mean_d >= -0.3:
            continue
        mean_p = sum(p for _, p, _ in occurrences) / len(occurrences)
        mean_n = sum(n for _, _, n in occurrences) / len(occurrences)
        candidates.append({
            "feature_id": fid,
            "mean_delta": mean_d,
            "n_contrasts": len(occurrences),
            "mean_pos_act": mean_p,
            "mean_neg_act": mean_n,
        })

    candidates.sort(key=lambda c: -abs(c["mean_delta"]))
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Steering ablation
# ---------------------------------------------------------------------------


def steering_ablation(server, feature_id: int, polarity: float) -> dict:
    """Compare baseline vs steered output on the agent test prompt."""
    baseline = server.steer_act.remote(
        prompt=AGENT_TEST_PROMPT, edits={}, max_new_tokens=80, temperature=0.2
    )
    steered = server.steer_act.remote(
        prompt=AGENT_TEST_PROMPT,
        edits={feature_id: polarity},
        max_new_tokens=80,
        temperature=0.2,
    )
    return {
        "baseline_response": baseline["response"].strip()[:200],
        "steered_response": steered["response"].strip()[:200],
        "differs": baseline["response"].strip() != steered["response"].strip(),
        "polarity": polarity,
    }


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------


def build_yaml_entry(concept_name: str, contrast: dict, candidate: dict, ablation: dict) -> dict:
    return {
        "id": candidate["feature_id"],
        "label": concept_name,
        "category": contrast["category"],
        "top_activation_examples": [pos for pos, _ in contrast["pairs"]],
        "contrast_score": round(candidate["mean_delta"], 3),
        "n_contrasts_supporting": candidate["n_contrasts"],
        "mean_activation_on_positive": round(candidate["mean_pos_act"], 3),
        "mean_activation_on_negative": round(candidate["mean_neg_act"], 3),
        "causal_effect": {
            f"polarity_{ablation['polarity']:+.1f}": "baseline_differs" if ablation["differs"] else "no_observable_change",
            "baseline_excerpt": ablation["baseline_response"][:140],
            "steered_excerpt": ablation["steered_response"][:140],
        },
        "recommended_delta": contrast["recommended_polarity"],
        "confidence": _confidence(candidate, ablation),
    }


def _confidence(candidate: dict, ablation: dict) -> str:
    """Heuristic confidence based on contrast strength + ablation observable."""
    score = abs(candidate["mean_delta"])
    if score > 1.5 and ablation["differs"]:
        return "high"
    if score > 0.8 and ablation["differs"]:
        return "medium"
    return "low"


def write_features_yaml(entries_by_category: dict[str, list[dict]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by verify/feature_drill.py\n")
        f.write("# Review each entry against the activation examples + causal_effect.\n")
        f.write("# Refine `recommended_delta` per feature based on demo experience.\n\n")
        yaml.safe_dump(entries_by_category, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    console.print("[bold cyan]Day 3 — Feature Discovery (automated)[/bold cyan]")
    server = _connect()

    entries_by_category: dict[str, list[dict]] = defaultdict(list)
    all_results = []

    for concept_name, contrast in CONTRASTS.items():
        console.rule(f"[bold]{concept_name}[/]")
        candidates = aggregate_candidates(server, concept_name, contrast, top_n=3)
        if not candidates:
            console.print(f"[yellow]No robust candidates for {concept_name}[/yellow]")
            continue

        for cand in candidates:
            ablation = steering_ablation(
                server, cand["feature_id"], contrast["recommended_polarity"]
            )
            entry = build_yaml_entry(concept_name, contrast, cand, ablation)
            entries_by_category[contrast["category"]].append(entry)
            all_results.append((concept_name, cand, ablation, entry["confidence"]))
            console.print(
                f"  feature {cand['feature_id']:>6d}: "
                f"Δ {cand['mean_delta']:+.3f}, "
                f"{cand['n_contrasts']} contrasts, "
                f"ablation_differs={ablation['differs']}, "
                f"confidence={entry['confidence']}"
            )

    out_path = "sae/features.yaml"
    write_features_yaml(dict(entries_by_category), out_path)

    # Summary table
    console.rule("[bold]Summary[/]")
    table = Table(title=f"Discovered features (written to {out_path})")
    table.add_column("Concept")
    table.add_column("Feature ID")
    table.add_column("Δ")
    table.add_column("Steering observable?")
    table.add_column("Confidence")
    for concept, cand, ab, conf in all_results:
        table.add_row(
            concept,
            str(cand["feature_id"]),
            f"{cand['mean_delta']:+.3f}",
            "✓" if ab["differs"] else "✗",
            conf,
        )
    console.print(table)

    high = sum(1 for *_, c in all_results if c == "high")
    med = sum(1 for *_, c in all_results if c == "medium")
    low = sum(1 for *_, c in all_results if c == "low")
    console.print(
        f"\n[bold]Total: {len(all_results)} features[/bold]  "
        f"(high={high}, medium={med}, low={low})"
    )
    console.print(f"YAML written to: [cyan]{out_path}[/cyan]")
    console.print(
        "[dim]Next: review entries, refine recommended_delta if needed, "
        "then run `make bench POLICY=baseline` and `POLICY=targeted`.[/dim]"
    )


if __name__ == "__main__":
    main()
