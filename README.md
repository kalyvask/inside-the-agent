# Inside the Agent

> *A reproducible harness for SAE feature interventions on browser agents.*

A fully open reference implementation of **SAE-feature-level steering on a browser agent**, with a deterministic benchmark and a live interpretability telemetry surface.

Two empirically-validated SAE feature edits at one decision step shift success rate by **+83 percentage points** on a held-out promotional-trap benchmark, with a sign-flipped control dropping back into baseline's confidence interval. The features themselves are *under-characterized* — we name them by feature ID, not by semantic label, until they are independently validated.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/runs--on-Modal-purple.svg)](https://modal.com)
[![Built for Stanford CS153](https://img.shields.io/badge/built--for-Stanford%20CS153-red.svg)](https://web.stanford.edu/class/cs153/)

---

## Headline result

8 held-out promotional-trap shopping tasks, 3 trials per policy (**24 trials each**), Wilson 95% CIs.

> 📊 **Source of truth: [`artifacts/benchmark_report.md`](artifacts/benchmark_report.md)** — auto-regenerated from `data/results/*.jsonl` via `python -m bench.report`. The table below is a snapshot from `v0.2`; it is verified against the auto-generated report on every CI run by [`python -m bench.artifact_check`](bench/artifact_check.py). Discrepancies fail the build.
>
> _v0.7 reviewer feedback closed: noise control + prompt-only control now route through their proper endpoints (P0-3 / P0-4 fixed); real-web tasks no longer report a fake 0% success rate (P0-5); hard_held_out task knobs are wired through (P0-3). v0.8 reruns at all six policies are in flight — table below updates when complete._

| Policy | Success | 95% CI | Δ vs baseline | Notes |
|---|---|---|---|---|
| baseline (no steering) | **0.0%** | [0.0%, 13.8%] | — | Falls for the trap every time |
| wrong-sign | 4.2% | [0.7%, 20.2%] | +4 pts | Sign-flipped targeted edits |
| random | 0.0% _(was 45.8% pre-v0.2 seed fix)_ | [0.0%, 13.8%] | — | Random feature edits w/ proper per-trial seeds (v0.2-A) |
| prompt-only (system-prompt control) | 75.0% | [55.1%, 88.0%] | +75 pts | "Avoid promotional banners; use search" in the prompt |
| noise (matched-norm random perturbation) | _pending v0.8 rerun_ | — | — | Reviewer P0-4: was dead code pre-v0.7-B |
| **targeted — 2 SAE feature edits at Step 0** | **83.3%** | **[64.1%, 93.3%]** | **+83 pts** | f26737 (-6) + f23803 (+6), `position_mode=all` |

![Headline chart](artifacts/headline.png)

### How to read these numbers honestly

- **Wrong-sign at 4.2%** sits inside baseline's CI. Flipping the targeted edits' signs erases the effect — direction matters causally, not just "any intervention."
- **Random at 45.8%** is a real but noisy lift. Random feature perturbations sometimes break the promo-click bias by accident. The **targeted-vs-random gap (+37 pts)** is the quantitative claim about *which* features matter.
- **Targeted at 83.3%** is the validated effect of two specific feature edits.

This is *not* a claim that we found "the promotional bias feature." It is a claim that **two specific SAE features, when intervened at the first decision step, causally shift the agent's success rate on this benchmark**. The features' semantic content is unverified — they may encode something narrower or broader than "promotional bias."

See `docs/methodology.md` for the full writeup, limitations, and known caveats.

---

## What this is

LLM agents are black boxes. When Claude / GPT-5 / Llama get tricked by a promotional banner, click an invented button, or wander away from the goal, the failure is observable but the cause isn't.

Mechanistic interpretability has produced **Sparse Autoencoder (SAE) features** — concept-level decompositions of the model's residual stream where each feature ideally encodes one human-interpretable concept. Until now those features have been used almost exclusively for *post-hoc analysis*.

This project wires them into a working agent as a **runtime intervention surface**:

- **Read** which features fire at every decision step (live telemetry)
- **Intervene** by adding feature-level deltas to the residual stream during inference
- **See** it all in a HUD: feature activations, intervention timeline, success/failure verdict

### What this is *not* (yet)

- Not a verified mapping from features to semantic concepts. We use feature IDs in policy code; we do not claim feature 26737 "is" the hallucination feature.
- Not a fully position-aware steering layer. The hook applies to the entire layer-19 residual stream during the steered forward pass, not just the action-generation token.
- Not yet an interactive control surface. The HUD shows the model's internals; the sidebar dials are placeholders until a HUD-to-runner command channel is wired in v0.2.

## Demo (3-minute video)

The recipe to produce the demo video lives in [`docs/recording_guide.md`](docs/recording_guide.md). It uses the trajectory replayer (no Modal calls during recording) and the polished HUD components built specifically for the video.

## Architecture

Three loosely-coupled processes:

```
hud (local Next.js)
  Verdict overlay + Steering flash + Feature bars colored by category
        ▲
        │ WebSocket events
        │
browser-worker (local Python)
  ShopGym deterministic storefronts + Playwright + verifiers
        │ HTTP: /act, /features, /steer_act
        ▼
brain-server (Modal L40S)
  Llama 3.1-8B-Instruct (BF16) + Goodfire SAE on layer 19
```

## Quickstart

### Prerequisites

- Python 3.11+ with pip
- Node 20+ with npm
- A Modal account (free; `pip install modal && modal token new`)
- A HuggingFace account with the Llama 3.1-8B-Instruct license accepted (gated repo)

### Install

```bash
git clone https://github.com/kalyvask/inside-the-agent
cd inside-the-agent

pip install -e ".[dev]"
playwright install chromium
cd hud && npm install && cd ..

cp .env.example .env
# Fill in HF_TOKEN, ANTHROPIC_API_KEY

modal token new
modal secret create hf-token HF_TOKEN=hf_xxx...
modal deploy modal_deploy/app.py
```

### Day 1 — verify (5-test gate)

```bash
make verify
```

Runs five tests against the deployed brain-server:
1. Model + SAE load
2. Feature catalog has agent-relevant features
3. Feature reading on agent-style prompts
4. Steering produces observable behavior change
5. Latency under 5s/step

### Reproduce the headline result

```bash
# Feature discovery + magnitude tuning (~10 min)
python -m verify.feature_drill
python -m verify.tune_deltas

# Step-0 calibration to find features that flip the first decision
python -m verify.step0_calibration

# Full 4-policy benchmark on 8 held-out tasks × 3 trials (~90 min)
for p in baseline random wrong-sign targeted; do
  python -m bench.runner --policy $p --tasks shopgym/tasks/held_out.json \\
    --trials 3 --limit 8
done

# Chart + Wilson CIs + failure-mode mining
python -m bench.analysis --plot
```

### Watch the HUD live (for recording)

In separate terminals:

```bash
# Terminal 1
python -m agent.ws_server

# Terminal 2
cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev
# Open http://localhost:3000

# Terminal 3 — replay a saved trajectory
python -m verify.replay_trajectory \\
  data/trajectories/promo_held_001_seed_0_targeted.jsonl --slow
```

## Repository layout

```
inside-the-agent/
├── modal_deploy/         brain-server (Modal app, Llama + Goodfire SAE)
│   ├── app.py            primary: Llama 3.1-8B + Goodfire SAE l19
│   └── app_gemma.py      fallback: Gemma 2-9B + Gemma Scope (not gated)
├── sae/                  loader, feature reader, steering controller
├── agent/                trajectory schema, prompts, agent loop, HUD publisher
├── policies/             baseline / random / wrong-sign / static / dynamic / targeted
├── shopgym/              deterministic storefronts + 30 tasks
├── bench/                runner, verifiers, analysis with Wilson CIs
├── hud/                  Next.js HUD with feature bars, verdict overlay, steering flash
├── verify/               5-test verification, feature discovery, tuning, calibration, replayer
├── docs/                 methodology paper, demo script, recording guide
└── data/                 trajectories, results, charts (gitignored)
```

## Open questions / future work

1. **Failure-mode features.** Mining surfaced 4 features (50853, 19079, 39820, 44602) that fire in 100% of baseline failures. These are stronger candidates for true "promo-trap" representation than what contrast discovery surfaced. Worth testing as steering targets.
2. **Cross-domain.** Held-out evaluation focused on promotional traps. Hallucination-prone and multi-step planning tasks are part of ShopGym but not yet in the reported benchmark.
3. **Cross-model.** Results are specific to Llama 3.1-8B + Goodfire's layer-19 SAE. Gemma 2-9B + Gemma Scope is scaffolded in `modal_deploy/app_gemma.py` but not measured.
4. **Dynamic steering.** Current targeted policy intervenes only at Step 0. A policy that watches feature activations across the trajectory and intervenes only at risky moments would be a stronger general claim.

## Built on

- **[Anthropic — Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)** (the SAE → frontier-model story)
- **[Goodfire AI — Llama-3.1-8B-Instruct SAE on layer 19](https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19)** (the open-weight SAE this project uses)
- **[Cho et al. — Control RL with SAE Features](https://arxiv.org/abs/2602.10437)** (the architecture this paper proposes; this project ships an open implementation)
- **[Modal](https://modal.com)** for the brain-server compute
- **[Playwright](https://playwright.dev)** for ShopGym browser automation

## License

MIT (code), CC-BY-4.0 (writeup in `docs/`).

## Citation

If this is useful in your own work:

```bibtex
@misc{kalyvas2026insidetheagent,
  title  = {Inside the Agent: A Live Interpretability HUD for Open-Source AI},
  author = {Kalyvas, Alexandros},
  year   = {2026},
  howpublished = {Stanford CS153 Frontier Systems},
  url    = {https://github.com/kalyvask/inside-the-agent}
}
```

## Acknowledgements

CS153 Frontier Systems (Stanford GSB / SOE, Spring 2026). Thanks to the Goodfire AI team for releasing the open SAE that made this possible.
