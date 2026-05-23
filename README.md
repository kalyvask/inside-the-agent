# Inside the Agent

> *Today's agents fail mysteriously. Ours fails legibly.*

A fully open, reproducible reference implementation of **SAE-steered language agents on browser tasks**, with a live interpretability HUD and a deterministic benchmark.

Two SAE feature edits at one decision step take success from **0% → 83%** on a held-out promotional-trap benchmark.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/runs--on-Modal-purple.svg)](https://modal.com)
[![Built for Stanford CS153](https://img.shields.io/badge/built--for-Stanford%20CS153-red.svg)](https://web.stanford.edu/class/cs153/)

---

## Headline result

8 held-out promotional-trap shopping tasks, 3 trials per policy (**24 trials each**), Wilson 95% CIs:

| Policy | Success | 95% CI | Δ vs baseline |
|---|---|---|---|
| baseline (no steering) | **0.0%** | [0.0%, 13.8%] | — |
| wrong-sign | 4.2% | [0.7%, 20.2%] | +4 pts |
| random | 45.8% | [27.9%, 64.9%] | +46 pts |
| **targeted — 2 SAE edits at Step 0** | **83.3%** | **[64.1%, 93.3%]** | **+83 pts** |

![Headline chart](data/results/headline.png)

The **wrong-sign control at 4.2%** is the smoking gun: flipping the targeted edits' signs drops performance back into baseline's CI. **Direction matters causally** — not just "any intervention helps."

See `docs/methodology.md` for the 4-page writeup.

---

## What this is

LLM agents are black boxes. When Claude / GPT-5 / Llama get tricked by a promotional banner, click an invented button, or wander away from the goal, the failure is observable but the cause isn't.

Mechanistic interpretability has produced **Sparse Autoencoder (SAE) features** — concept-level decompositions of the model's residual stream where each feature ideally encodes one human-interpretable concept (planning, hallucination, promotional language). Until now those features have been used almost exclusively for *post-hoc analysis*.

This project wires them into a working agent as a **runtime control surface**:

- **Read** which features fire at every decision step
- **Steer** behavior by applying feature-level deltas at inference time
- **See** it all live in a HUD

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
