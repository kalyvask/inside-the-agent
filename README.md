# Inside the Agent

A fully open, reproducible reference implementation of SAE-steered language agents on browser tasks, with live human-in-the-loop steering and a deterministic benchmark.

**Tagline:** *We made an AI agent transparent. You can see which concepts it's thinking about, and turn them down in real time.*

**Stanford CS153 Frontier Systems final project.** Demo: May 29, 2026.

## Headline result

8 held-out promotional-trap tasks, 3 trials per policy (24 trials each), Wilson 95% CIs:

| Policy | Success | 95% CI | Δ vs baseline |
|---|---|---|---|
| baseline (no steering) | **0.0%** | [0.0%, 13.8%] | — |
| wrong-sign | 4.2% | [0.7%, 20.2%] | +4 pts |
| random | 45.8% | [27.9%, 64.9%] | +46 pts |
| **targeted (2 SAE edits, Step 0 only)** | **83.3%** | **[64.1%, 93.3%]** | **+83 pts** |

The wrong-sign control at 4% (in baseline's CI) is the smoking gun: flipping the targeted edits' signs drops performance back to baseline. **Direction matters causally**, not "any intervention helps." See `data/results/headline.png` and `docs/methodology.md` for details.

## What this is

The agent shops on a deterministic mini-storefront (ShopGym). A live HUD shows which Sparse Autoencoder (SAE) features fire as the agent reasons. A human can apply targeted feature deltas to change behavior in real time, without retraining.

Stack:
- **Base model:** Llama 3.1-8B-Instruct (BF16)
- **SAE:** Goodfire/Llama-3.1-8B-Instruct-SAE-l19 (open weights)
- **Compute:** Modal serverless GPU (L40S 48GB)
- **Browser env:** custom ShopGym (deterministic storefronts) + optional BrowserGym tasks
- **HUD:** local Next.js + WebSocket
- **Storage:** local JSONL + optional Cloudflare R2/D1

No proprietary inference API required for the core demo.

## Quickstart

### Prerequisites
- Python 3.11+, Node 20+
- Modal account (`pip install modal && modal token new`)
- HuggingFace token with Llama 3.1-8B-Instruct license accepted (`huggingface-cli login`)

### Install
```bash
pip install -e .
cd hud && npm install && cd ..
```

### Day 1 verification (THE most important hour)
```bash
make verify
# OR
python -m verify.sae_smoke
```

This runs 5 tests against the Modal-hosted brain-server:
1. Model + SAE load
2. Feature catalog has agent-relevant features
3. Feature reading on agent-style prompts
4. Steering produces observable behavior change
5. Latency under 5s/step

If all 5 pass → commit to project. If Test 4 fails → pivot to "live transparency only" demo.

### Benchmark (Day 6)
```bash
make bench POLICY=baseline
make bench POLICY=random
make bench POLICY=wrong-sign
make bench POLICY=targeted
```

### Live HUD (Day 4 onward)
```bash
make demo
# Visit http://localhost:3000
```

## Architecture (3 processes)

```
hud (local Next.js) ──WebSocket──> browser-worker (local) ──HTTP──> brain-server (Modal L40S)
                                          │
                                          ▼
                                    ShopGym storefronts
                                    (Playwright)
```

See `docs/methodology.md` for design rationale.

## Repository layout

```
inside-the-agent/
├── modal_deploy/       # brain-server (Modal app loading Llama + SAE)
├── agent/              # agent loop, prompts, trajectory schema
├── sae/                # SAE loader, feature reader, steering controller
├── policies/           # static, dynamic, random, wrong-sign control policies
├── shopgym/            # deterministic storefronts + tasks
├── bench/              # task definitions, verifiers, runner, analysis
├── hud/                # Next.js local HUD
├── verify/             # Day 1 verification CLI
├── notebooks/          # exploratory notebooks
├── docs/               # methodology paper, slides
└── data/               # trajectories, benchmark results (gitignored)
```

## Decision gates

| Day | Date | Gate |
|---|---|---|
| 1 | May 22 | Verification passes → commit. Else → "transparency only" fallback. |
| 3 | May 24 | Steered agent visibly differs from baseline → continue. Else → audit catalog. |
| 6 | May 27 | Targeted ≥ +10pt over baseline AND random+wrong-sign ≤ baseline → results credible. |

## Acknowledgements

Built on top of:
- [Goodfire AI's open SAE checkpoints](https://huggingface.co/Goodfire)
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [Anthropic's "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Cho et al. "Control RL with SAE Features"](https://arxiv.org/abs/2602.10437)

## License

MIT (code), CC-BY-4.0 (writeup).
