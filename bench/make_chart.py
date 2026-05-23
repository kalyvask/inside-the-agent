"""
bench/make_chart.py — v0.20 headline chart from current artifacts.

Generates artifacts/headline.png from data/results/<policy>.jsonl with two
panels:
  1. Overall success rate per policy (6 bars, Wilson 95% CI errorbars).
  2. Per-category cross-domain breakdown (3 grouped bars: promo / hallucination
     / planning), showing where the targeted policy lifts and where it
     hurts.

Run:
    python -m bench.make_chart
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("data/results")
TASKS_DIR = Path("shopgym/tasks")
OUT_PATH = Path("artifacts/headline.png")

# Order + colors picked for narrative clarity. Targeted is emerald (the
# headline). Prompt-only is blue (alternative method, beats targeted overall
# but for a mechanistic-different reason). Controls are muted.
POLICY_ORDER = ["baseline", "wrong-sign", "random", "noise", "prompt-only", "targeted"]
COLORS = {
    "baseline":     "#737373",  # zinc-500
    "wrong-sign":   "#a78bfa",  # violet-400
    "random":       "#fb923c",  # orange-400
    "noise":        "#38bdf8",  # sky-400
    "prompt-only":  "#60a5fa",  # blue-400
    "targeted":     "#34d399",  # emerald-400
}


def _wilson(s: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _load_results(policy: str) -> list[dict]:
    path = RESULTS_DIR / f"{policy}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _task_categories() -> dict[str, str]:
    """Map task_id -> category (collapses 'promotional' to 'promo')."""
    out: dict[str, str] = {}
    for path in TASKS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        tasks = data if isinstance(data, list) else [data]
        for t in tasks:
            if isinstance(t, dict) and t.get("id") and t.get("category"):
                cat = t["category"]
                # Normalize: 'promotional' -> 'promo' so the chart only has
                # 3 buckets, not 4.
                if cat in ("promotional", "promo"):
                    cat = "promo"
                out[t["id"]] = cat
    return out


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )

    # ============ Left panel: overall headline rate per policy ============
    overall_rates = []
    overall_cis = []
    overall_labels = []
    overall_colors = []
    for policy in POLICY_ORDER:
        rows = _load_results(policy)
        if not rows:
            continue
        n = len(rows)
        s = sum(1 for r in rows if r.get("success"))
        rate = s / n
        ci = _wilson(s, n)
        overall_rates.append(rate)
        overall_cis.append((rate - ci[0], ci[1] - rate))
        overall_labels.append(f"{policy}\n(n={n})")
        overall_colors.append(COLORS.get(policy, "#888"))

    x = list(range(len(overall_rates)))
    yerr_lo = [c[0] for c in overall_cis]
    yerr_hi = [c[1] for c in overall_cis]
    bars = ax_left.bar(
        x, overall_rates, color=overall_colors, edgecolor="#444",
        yerr=[yerr_lo, yerr_hi], capsize=5, error_kw={"ecolor": "#444"},
    )
    for i, (rate, bar) in enumerate(zip(overall_rates, bars)):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate*100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(overall_labels, fontsize=9)
    ax_left.set_ylabel("Success rate")
    ax_left.set_ylim(0, 1.0)
    ax_left.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_left.set_title(
        "Overall success rate — held-out 20 tasks × 3 trials",
        fontsize=12, pad=10,
    )
    ax_left.grid(axis="y", alpha=0.2)

    # ============ Right panel: per-category breakdown ============
    cats = _task_categories()
    cat_order = ["promo", "hallucination", "planning"]
    cat_labels = {
        "promo": "promo\n(calibrated)",
        "hallucination": "hallucination\n(cross-domain)",
        "planning": "planning\n(out-of-distribution)",
    }

    per_cat_rates: dict[str, dict[str, float]] = {}
    for policy in POLICY_ORDER:
        rows = _load_results(policy)
        if not rows:
            continue
        by_cat = defaultdict(list)
        for r in rows:
            tid = r.get("task_id", "")
            cat = cats.get(tid)
            if cat in cat_order:
                by_cat[cat].append(bool(r.get("success")))
        per_cat_rates[policy] = {
            cat: (sum(by_cat[cat]) / len(by_cat[cat]) if by_cat[cat] else 0.0)
            for cat in cat_order
        }

    n_policies = len(per_cat_rates)
    n_cats = len(cat_order)
    width = 0.85 / n_policies
    x = list(range(n_cats))
    for i, policy in enumerate(POLICY_ORDER):
        if policy not in per_cat_rates:
            continue
        vals = [per_cat_rates[policy][cat] for cat in cat_order]
        offset = (list(per_cat_rates.keys()).index(policy)) * width
        bars = ax_right.bar(
            [xi + offset for xi in x], vals, width,
            color=COLORS.get(policy, "#888"),
            edgecolor="#444",
            label=policy,
        )
        # Label only targeted bars to keep the chart legible
        if policy == "targeted":
            for j, (val, bar) in enumerate(zip(vals, bars)):
                ax_right.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.02,
                    f"{val*100:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=COLORS["targeted"],
                )
    ax_right.set_xticks([xi + (n_policies - 1) * width / 2 for xi in x])
    ax_right.set_xticklabels([cat_labels[c] for c in cat_order], fontsize=9)
    ax_right.set_ylabel("Success rate")
    ax_right.set_ylim(0, 1.0)
    ax_right.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_right.set_title(
        "Cross-domain breakdown — the mechanism is category-specific",
        fontsize=12, pad=10,
    )
    ax_right.legend(loc="upper right", fontsize=8, ncol=2)
    ax_right.grid(axis="y", alpha=0.2)

    # ============ Caption + finalize ============
    fig.suptitle(
        "inside-the-agent · v0.8 held-out (20 tasks: 8 promo + 6 hallucination + 6 planning)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.01,
        "Targeted = two SAE feature edits at step 0 (f26737 −6 + f23803 +6, position_mode=all). "
        "Wilson 95% CIs on overall rates.",
        ha="center", fontsize=8, color="#666",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
