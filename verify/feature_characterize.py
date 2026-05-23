"""
v0.4: Combined feature characterization.

For each candidate feature, query the brain-server for:
  1. Logit lens — top tokens the feature promotes and suppresses
  2. Decoder neighbors — features with similar W_dec direction
  3. Top-activating prompts from a curated corpus

Combine evidence into a per-feature "characterization card" with an honest
confidence rating.

Outputs:
  artifacts/feature_characterization.json   — full data
  docs/feature_cards/f{id}.md               — one card per feature

Usage:
    python -m verify.feature_characterize
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import modal
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)


CANDIDATE_FEATURES = {
    26737: "validated_targeted_invented_action_supp",
    23803: "validated_targeted_goal_anchor",
    50853: "100pct_failure_1",
    19079: "100pct_failure_2",
    39820: "100pct_failure_3",
    44602: "100pct_failure_4",
    38249: "80pct_failure",
}


CORPUS = {
    "shopping_with_promo": [
        "🔥 LIMITED TIME! Buy NOW and save 50%!",
        "SALE ENDS TONIGHT! Click Buy Now to claim your discount!",
        "Today's Deal: Wireless Earbuds for $39.99 — Buy Now!",
        "Featured deal of the day. Premium product on offer.",
        "Bright red banner: Buy Now. Limited time discount.",
        "The hero promotional button is highlighted in red.",
    ],
    "shopping_neutral": [
        "Add a USB-C cable to the cart.",
        "Compare three cables by price.",
        "The product description lists size options.",
        "Items are organized by category.",
        "Read the product specs carefully.",
        "Shipping options are available at checkout.",
    ],
    "planning": [
        "Let me think step by step about which action to take.",
        "I should plan three actions before clicking anything.",
        "First survey the page, then choose the best action.",
        "Carefully consider all options before deciding.",
        "Deliberation requires reading the page thoroughly.",
        "I should pause and weigh the alternatives.",
    ],
    "impulsive": [
        "Just click something. Move fast.",
        "Pick the most visible button immediately.",
        "Don't think — react. Click whatever stands out.",
        "Move quickly without deliberation.",
        "First impulse: click the bright button.",
    ],
    "search_ui": [
        "Type the search query into the search bar.",
        "Click the search button to filter results.",
        "Use the search input to find a product.",
        "Search results page filtered by query.",
        "The search bar is at the top of the page.",
    ],
    "hallucinated_ui": [
        "I'll click the filter button — wait, it doesn't exist on this page.",
        "Use the sort-by-rating dropdown to reorder results.",
        "The advanced search panel should let me set filters.",
        "Click the recommendation widget on the side.",
        "The price-range slider lets me narrow results.",
    ],
    "uncertain": [
        "I'm not sure which product matches the goal.",
        "The answer is unclear. Let me re-read the description.",
        "I should hedge — the right action is ambiguous.",
        "I don't know which option is best.",
        "Maybe I should ask for clarification first.",
    ],
    "goal_tracking": [
        "Goal: buy a USB-C cable. Stay focused.",
        "The objective is the USB-C cable specifically.",
        "Keep the task in mind and ignore distractions.",
        "Don't lose sight of what I was asked to buy.",
        "Goal-anchored: only the cable matters here.",
    ],
    "neutral_text": [
        "Hello, how are you today?",
        "The capital of France is Paris.",
        "Translate the sentence into French.",
        "Compose a haiku about autumn.",
        "Explain quantum entanglement in simple terms.",
    ],
}


def _connect():
    BrainServer = modal.Cls.from_name("inside-the-agent", "BrainServer")
    return BrainServer()


def features_dict(top_features: list[dict]) -> dict[int, float]:
    return {f["id"]: f["activation"] for f in top_features}


@app.command()
def main(
    out_dir: str = typer.Option("artifacts", help="Output directory"),
    cards_dir: str = typer.Option("docs/feature_cards", help="Per-feature markdown cards"),
):
    server = _connect()

    feature_ids = list(CANDIDATE_FEATURES.keys())
    console.rule(f"[bold]Characterizing {len(feature_ids)} candidate features[/]")

    # 1. Logit lens for each feature
    console.print("\n[bold cyan]Step 1: Logit lens per feature[/]")
    logit_lens_results = {}
    for fid in feature_ids:
        r = server.feature_logit_lens.remote(fid, top_k=20)
        logit_lens_results[fid] = r
        promoted_tokens = [p["token"] for p in r["promoted"][:8]]
        console.print(f"  f{fid:>6d} → promotes: {' | '.join(repr(t)[:15] for t in promoted_tokens)}")

    # 2. Decoder neighbors (one batch call)
    console.print("\n[bold cyan]Step 2: Decoder similarity neighbors[/]")
    sim_response = server.feature_decoder_similarity.remote(feature_ids, top_k=10)
    similarities = sim_response["similarities"]
    for fid in feature_ids:
        if fid not in similarities:
            continue
        neighbors = similarities[fid]
        top = [f"f{n['feature_id']}({n['cosine_sim']:.2f})" for n in neighbors[:5]]
        console.print(f"  f{fid:>6d} → neighbors: {' | '.join(top)}")

    # 3. Top-activating prompts from corpus
    console.print("\n[bold cyan]Step 3: Corpus probe (top-activating prompts)[/]")
    # Accumulate per-feature activations across all corpus prompts
    feature_activations_by_prompt: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    for category, prompts in CORPUS.items():
        for prompt in prompts:
            r = server.read_features.remote(prompt, top_k=300)
            fd = features_dict(r["top_features"])
            for fid in feature_ids:
                act = fd.get(fid, 0.0)
                if act > 0.0:
                    feature_activations_by_prompt[fid].append((act, category, prompt))

    # For each feature, sort by activation, take top-10
    top_activating: dict[int, list[dict]] = {}
    for fid in feature_ids:
        rows = sorted(feature_activations_by_prompt[fid], reverse=True)[:10]
        top_activating[fid] = [
            {"activation": a, "category": c, "prompt": p} for a, c, p in rows
        ]
        # Print summary
        if rows:
            top_cats = [r["category"] for r in top_activating[fid][:5]]
            console.print(f"  f{fid:>6d} → top categories: {', '.join(top_cats)}")

    # 4. Write everything to artifacts/feature_characterization.json
    out_path = Path(out_dir) / "feature_characterization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "feature_ids": feature_ids,
        "candidates": CANDIDATE_FEATURES,
        "logit_lens": logit_lens_results,
        "decoder_neighbors": similarities,
        "top_activating_prompts": top_activating,
    }
    out_path.write_text(json.dumps(combined, indent=2))
    console.print(f"\nWrote {out_path}")

    # 5. Per-feature markdown cards
    cards_path = Path(cards_dir)
    cards_path.mkdir(parents=True, exist_ok=True)

    for fid in feature_ids:
        ll = logit_lens_results[fid]
        promoted = [f'`{p["token"]}` (+{p["score"]:.2f})' for p in ll["promoted"][:15]]
        suppressed = [f'`{p["token"]}` ({p["score"]:.2f})' for p in ll["suppressed"][:10]]
        neighbors = similarities.get(fid, [])
        top_acts = top_activating[fid]

        # Synthesize a hypothesis from the evidence
        category_counts = defaultdict(int)
        for a in top_acts:
            category_counts[a["category"]] += 1
        top_cat = max(category_counts, key=category_counts.get) if category_counts else "unknown"

        confidence = "low"
        if top_acts and top_acts[0]["activation"] > 1.5 and len(set(a["category"] for a in top_acts[:5])) <= 2:
            confidence = "medium"
        if top_acts and top_acts[0]["activation"] > 2.5 and len(set(a["category"] for a in top_acts[:3])) == 1:
            confidence = "high"

        md = [
            f"# Feature {fid}",
            "",
            f"**Provisional label:** `{CANDIDATE_FEATURES[fid]}`",
            f"**Confidence:** {confidence}",
            f"**Modal candidate concept (from top corpus category):** {top_cat}",
            "",
            "## Logit lens — top promoted tokens",
            "",
            "These tokens the feature most pushes the model toward (when fired):",
            "",
            ", ".join(promoted),
            "",
            "## Logit lens — top suppressed tokens",
            "",
            "These tokens the feature pushes the model AWAY from:",
            "",
            ", ".join(suppressed),
            "",
            "## Decoder-direction neighbors",
            "",
            "Features with the most similar decoder vectors (likely encoding related concepts):",
            "",
        ]
        for n in neighbors[:10]:
            md.append(f"- f{n['feature_id']} (cosine = {n['cosine_sim']:.3f})")
        md.append("")
        md.append("## Top-activating corpus prompts")
        md.append("")
        md.append("Prompts from a curated corpus where this feature fires strongest:")
        md.append("")
        md.append("| Rank | Category | Activation | Prompt |")
        md.append("|---|---|---|---|")
        for i, a in enumerate(top_acts[:10], 1):
            md.append(f"| {i} | {a['category']} | {a['activation']:.2f} | {a['prompt'][:80]} |")
        md.append("")
        md.append("## Notes")
        md.append("")
        md.append(
            "Hypothesis: this feature appears to encode something related to "
            f"**{top_cat}**, based on the corpus categories where it activates strongest. "
            "The logit lens tokens above provide additional clues. The confidence rating "
            "reflects how consistent the evidence is across methods."
        )
        md.append("")
        md.append(
            "**Caveat:** This is provisional. The activations come from a small (~40-prompt) "
            "curated corpus. A larger, naturalistic probe would tighten the claim."
        )

        card_path = cards_path / f"f{fid}.md"
        card_path.write_text("\n".join(md))

    console.print(f"\nWrote {len(feature_ids)} feature cards to {cards_path}/")
    console.print("[dim]Read them to assemble the v0.4 feature characterization story.[/dim]")


if __name__ == "__main__":
    app()
