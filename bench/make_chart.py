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
# headline SAE result). Controls (wrong-sign / random / noise) are muted —
# they isolate the causal direction of the edit. prompt-only (a non-SAE
# system-prompt control, 73.3%) is intentionally omitted from this chart:
# it beats targeted on average for a mechanistically-different reason, so
# putting it on the SAE-causal chart muddies the comparison. It remains fully
# documented in the README cross-method table + "when does SAE beat prompt"
# section.
POLICY_ORDER = ["baseline", "wrong-sign", "random", "noise", "targeted", "prompt-plus-targeted"]
COLORS = {
    "baseline":             "#737373",  # zinc-500
    "wrong-sign":           "#a78bfa",  # violet-400
    "random":               "#fb923c",  # orange-400
    "noise":                "#38bdf8",  # sky-400
    "prompt-only":          "#60a5fa",  # blue-400
    "targeted":             "#34d399",  # emerald-400
    "prompt-plus-targeted": "#0d9488",  # teal-600 (SAE edits + prompt, stacked best)
}
# Short display names for x-axis ticks + legend (policy keys are verbose).
DISPLAY = {"prompt-plus-targeted": "prompt+targeted"}


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


def make_cross_scale() -> None:
    """v0.25: cross-scale figure — small 8B + SAE steering vs the 70B ceiling.

    Data-driven from artifacts/seed_manifest.json (the same artifact
    bench/artifact_check.py validates), so the bars cannot silently drift
    from the published numbers. Bars: 8B baseline, 8B + SAE (the SAE-alone
    causal lift), 8B + SAE + prompt (stacked best), 70B baseline. The caption
    + callout disclose that the +prompt bar is mostly the prompt (prompt-only
    alone is 73.3%) and that the 70B is a different model with a format
    prompt, so the framing stays honest."""
    import json
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    manifest = json.loads(Path("artifacts/seed_manifest.json").read_text(encoding="utf-8"))
    hr = manifest["headline_results"]["policy_success_rates"]
    cs = manifest["cross_scale"]["70b_baseline_strict_results"]

    series = [
        ("Llama-3.1-8B\nbaseline", hr["baseline"]["rate"], hr["baseline"]["wilson_ci_95"], "#737373"),
        ("Llama-3.1-8B\n+ SAE steering", hr["targeted"]["rate"], hr["targeted"]["wilson_ci_95"], "#34d399"),
        ("Llama-3.1-8B\n+ SAE + prompt", hr["prompt-plus-targeted"]["rate"], hr["prompt-plus-targeted"]["wilson_ci_95"], "#0d9488"),
        ("Llama-3.3-70B\nbaseline (format prompt)", cs["rate"], cs["wilson_ci_95"], "#475569"),
    ]
    rates = [s[1] for s in series]
    gap_sae = (rates[1] - rates[0]) / (rates[-1] - rates[0])
    gap_stack = (rates[2] - rates[0]) / (rates[-1] - rates[0])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    x = list(range(len(series)))
    yerr_lo = [s[1] - s[2][0] for s in series]
    yerr_hi = [s[2][1] - s[1] for s in series]
    bars = ax.bar(
        x, rates, color=[s[3] for s in series], edgecolor="#444",
        yerr=[yerr_lo, yerr_hi], capsize=6, error_kw={"ecolor": "#444"}, width=0.62,
    )
    for s, bar in zip(series, bars):
        ax.text(bar.get_x() + bar.get_width() / 2, s[1] + 0.025,
                f"{s[1]*100:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    # Gap guides: dashed lines at the 8B floor and the 70B ceiling so the
    # reader can see what fraction of that span the SAE bar fills.
    ax.axhline(rates[0], color="#737373", ls=":", lw=1, alpha=0.55)
    ax.axhline(rates[-1], color="#475569", ls=":", lw=1, alpha=0.55)
    ax.text(
        0.02, 0.97,
        f"SAE alone closes {gap_sae*100:.0f}% of the gap to the 70B;\n"
        f"+ prompt reaches {gap_stack*100:.0f}%. But prompt-only (73%) only\n"
        f"works if you already know the trap; the SAE\n"
        f"edits are found by reading the model's signals.",
        transform=ax.transAxes, fontsize=10, fontweight="bold", color="#0f766e",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#34d399", lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in series], fontsize=10)
    ax.set_ylabel("Success rate — held-out 20 tasks × 3 trials (lenient)")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(
        "Cross-scale — SAE steering closes half the gap to a model 8× larger",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5, 0.005,
        "Llama-3.1-8B + two SAE edits at step 0 (f26737 −6, f23803 +6). Wilson 95% CIs, n=60 each. "
        "Stacked, SAE adds only ~+1.7pt over prompt-only (73.3%) on average. 70B is a different "
        "model with a 1-line strict-JSON format prompt — a ceiling reference, not a controlled comparison.",
        ha="center", fontsize=7.5, color="#666", wrap=True,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path("artifacts/cross_scale.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


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
        overall_labels.append(f"{DISPLAY.get(policy, policy)}\n(n={n})")
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
        "Success rate by policy — held-out 20 tasks × 3 trials",
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
            label=DISPLAY.get(policy, policy),
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
        "Wilson 95% CIs. prompt+targeted (75%) is mostly the prompt: prompt-only alone is 73.3%. But "
        "prompt-only requires an instruction that already names the trap; the SAE edits were instead found "
        "by reading the model's own activations, not by knowing the failure in advance. See README.",
        ha="center", fontsize=8, color="#666",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")
    make_cross_scale()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
