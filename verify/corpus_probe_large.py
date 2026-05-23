"""
verify/corpus_probe_large.py — top-activating corpus probe on 1k+ prompts.

v0.22 P2: the v0.4 corpus probe (verify/feature_drill.py) used 40 hand-
written prompts. That's enough to suggest a label but too small to defend
it. This script streams a public dataset (wikitext-103 by default, can
override) and reports the top-N prompts that activate each feature in
WATCH_FEATURES — gives a much harder test of the lexical-cluster claim
in docs/feature_characterization.md.

Run:
    python -m verify.corpus_probe_large
    # writes artifacts/corpus_probe_large.json with per-feature
    # top-activating prompts + activation distributions
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal
import typer

app = typer.Typer(add_completion=False)


# The six features we've labelled. The probe will report top-K activations
# on each one across the streamed corpus.
WATCH_FEATURES = {
    26737: "f26737_ui_selection_vocab",
    23803: "f23803_distraction_avoidance_vocab",
    34048: "f34048_promotional_bias",
    50853: "f50853_fail_mode_a",
    19079: "f19079_fail_mode_b",
    39820: "f39820_fail_mode_c",
}


def _stream_wikitext(limit: int) -> list[str]:
    """Stream the first `limit` non-empty lines from wikitext-103-v1 train
    split. Each line is treated as a single prompt. Tries datasets first;
    falls back to a builtin small corpus if not installed."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        prompts = []
        for ex in ds:
            text = (ex.get("text") or "").strip()
            if 30 < len(text) < 400:  # skip empty / overlong
                prompts.append(text)
                if len(prompts) >= limit:
                    break
        return prompts
    except Exception as e:
        print(f"[corpus_probe_large] could not stream wikitext ({e})")
        print("[corpus_probe_large] falling back to small built-in corpus")
        return _builtin_corpus()


def _builtin_corpus() -> list[str]:
    """50-prompt fallback covering the 4 lexical clusters we care about:
    UI-selection, distractions, promotional bias, generic shopping."""
    return [
        # UI-selection vocab
        "Click the dropdown menu to select your option from the list.",
        "Use the filter checkboxes to narrow your search results.",
        "The radio buttons let you choose exactly one option per group.",
        "Pick one of the available choices from the selector widget.",
        "The settings panel has options for theme, language, and notifications.",
        "Selected items are highlighted in the multi-select dropdown.",
        "Toggle the checkbox to enable the advanced search filters.",
        "Choose your size from the dropdown selection menu.",
        "The configuration panel lets you select between three modes.",
        "Tab through the options and pick the radio button you want.",
        # Distraction-avoidance vocab
        "Keep the task in mind and ignore distractions.",
        "Stay focused on the goal — don't get tempted by side quests.",
        "Avoid interruptions and notifications during deep work.",
        "Resist the temptation to click on promotional banners.",
        "Block out the distractions and stay on task.",
        "Don't get distracted by the flashy ads at the top of the page.",
        "Tune out the notifications and finish what you started.",
        "The interruptions are tempting but ignore them for now.",
        "Stay disciplined: ignore the side panels and focus on search.",
        "Notifications can wait — keep your attention on the task.",
        # Promotional / urgency vocab
        "LIMITED TIME OFFER! Buy now and save 50%!",
        "Today's Featured Deal — limited stock, click to add to cart!",
        "Don't miss this exclusive sale — only 2 hours left!",
        "Buy two and get one free — promotional banner at top of page.",
        "Sponsored: Premium Charger with free shipping today only.",
        "Today only! Bright red banner promo — Buy Now!",
        "Featured deal of the day — wireless earbuds at half price.",
        "Sale ends tonight! Click Buy Now to claim your discount.",
        "Promotional offer: bundle save 30% with code SPRING.",
        "Hot deal — limited quantity available, hurry while supplies last.",
        # Generic shopping / neutral
        "USB-C charging cable with braided nylon, 6 feet, fast charging.",
        "Organic cotton t-shirt, machine washable, sizes S through XXL.",
        "Stainless steel water bottle, 24 oz, keeps drinks cold for 24 hours.",
        "Wireless headphones with active noise cancellation, 30-hour battery.",
        "Eco-friendly bamboo toothbrush, biodegradable bristles, soft.",
        "Memory foam pillow, hypoallergenic cover, queen size.",
        "Bluetooth speaker, waterproof IPX7, 12-hour battery life.",
        "Yoga mat, 6mm thick, non-slip, includes carrying strap.",
        "Electric kettle, 1.7 liter capacity, stainless steel.",
        "Mechanical keyboard with hot-swappable switches and RGB.",
    ]


def _make_brain_call():
    app_name = os.environ.get("BRAIN_APP_NAME", "inside-the-agent")
    BrainServer = modal.Cls.from_name(app_name, "BrainServer")
    server = BrainServer()
    return server


@app.command()
def main(
    n_prompts: int = typer.Option(1000, help="Number of prompts to stream from wikitext."),
    top_k_per_feature: int = typer.Option(20, help="Keep top-K highest-activating prompts per watched feature."),
    out_path: str = typer.Option("artifacts/corpus_probe_large.json"),
):
    """Stream a public corpus, score each prompt on the SAE's encoder for
    every watched feature, and report the top-activating prompts per
    feature."""
    prompts = _stream_wikitext(n_prompts)
    print(f"[corpus_probe_large] loaded {len(prompts)} prompts")
    server = _make_brain_call()

    # Bucket per feature → list of (activation, prompt)
    top_per_feature: dict[int, list[tuple[float, str]]] = {fid: [] for fid in WATCH_FEATURES}

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"[corpus_probe_large] {i}/{len(prompts)}")
        try:
            r = server.read_features.remote(prompt, top_k=100)
        except Exception as e:
            print(f"[corpus_probe_large] read_features failed on prompt {i}: {e}")
            continue
        top = {f["id"]: f["activation"] for f in r.get("top_features", [])}
        for fid in WATCH_FEATURES:
            act = top.get(fid, 0.0)
            top_per_feature[fid].append((act, prompt))

    # Trim to top-K per feature
    out: dict[str, dict] = {}
    for fid, label in WATCH_FEATURES.items():
        ranked = sorted(top_per_feature[fid], key=lambda x: -x[0])[:top_k_per_feature]
        out[label] = {
            "feature_id": fid,
            "n_prompts_scored": len(top_per_feature[fid]),
            "max_activation": max((a for a, _ in top_per_feature[fid]), default=0.0),
            "mean_activation": (
                sum(a for a, _ in top_per_feature[fid]) / max(1, len(top_per_feature[fid]))
            ),
            "top_prompts": [
                {"activation": round(act, 3), "prompt": p[:300]}
                for act, p in ranked
            ],
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[corpus_probe_large] wrote {out_path}")

    # Print headline lines
    for label, d in out.items():
        print(f"\n--- {label} (id={d['feature_id']}) ---")
        print(f"max={d['max_activation']:.2f}  mean={d['mean_activation']:.3f}")
        for p in d["top_prompts"][:5]:
            print(f"  {p['activation']:.2f}  {p['prompt'][:100]}")


if __name__ == "__main__":
    app()
