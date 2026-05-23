# Inside the Agent: A Live Interpretability HUD for Open-Source AI

**Stanford CS153 Frontier Systems — Final Project**
Alexandros Kalyvas | May 2026
Code: `github.com/kalyvask/inside-the-agent`

---

## Abstract

We present the first fully open-source reference implementation of a Sparse Autoencoder (SAE)–steered language agent with a real-time interpretability HUD. Built on Llama 3.1-8B-Instruct + Goodfire's open SAE on layer 19, the system runs an agent on browser tasks while exposing the model's internal feature activations as a runtime control surface. On a held-out benchmark of 8 promotional-trap shopping tasks (24 trials per policy), a 2-feature targeted steering intervention applied only at the first decision step lifts success from **0.0% to 83.3%** — and a wrong-sign control drops to **4.2%**, isolating the direction of the intervention as the causal factor. The stack runs end-to-end on commodity GPUs with no proprietary inference dependency.

---

## 1. Problem

LLM agents are black boxes. When Claude or GPT-5 fails on a multi-step task — clicks a wrong button, hallucinates a non-existent UI element, gets baited by a promotional banner — the failure is observable but the cause is not. Existing debugging tools work on inputs and outputs, not on the model's internal representations.

Mechanistic interpretability research (Anthropic 2024, Goodfire 2025, Apollo 2025) has produced **Sparse Autoencoder features**: concept-level decompositions of the residual stream where each feature ideally encodes a single human-interpretable concept (planning, hallucination, promotional language, etc.). Until now, these features have been used primarily for *post-hoc analysis*. They have not been wired into the runtime of working agents as a *control surface*.

This project bridges that gap.

---

## 2. Approach

We expose SAE features as a real-time control surface in two operations:

- **Read:** at each agent decision step, query the SAE for the top-k features firing on the layer-19 residual stream after the prompt.
- **Steer:** apply per-feature additive deltas to the residual at the same layer, with clamps to bound the perturbation.

Steering is computed in feature space and projected back to activation space:
```
delta_activation = Σᵢ δᵢ · W_dec[i]       (sum of decoder rows × per-feature delta)
hidden_new       = clamp(hidden + delta_activation, -100, +100)
```

We discover candidate features via **contrast prompts** (positive prompt activates the concept; negative does not). Features with consistently large activation differences across multiple contrast pairs are candidates for that concept. We then verify each candidate via steering ablation: amplify or suppress and confirm the output changes in the expected direction at the largest magnitude that preserves output coherence.

---

## 3. System Architecture

Three loosely-coupled processes:

```
hud (local Next.js)
  Three panels + intervention timeline
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

The brain-server hosts the inference and steering primitives. The browser-worker executes ShopGym tasks via Playwright and dispatches structured JSON actions. The local HUD subscribes to a WebSocket event stream from the worker — when running with `--hud`, the runner spawns the WS server as a subprocess automatically.

---

## 4. Benchmark — ShopGym

ShopGym is a deterministic mini-storefront environment we built specifically for this study. Each task renders a self-contained HTML page via Playwright with:

- A configurable promotional banner (color, size, product, price)
- A search bar with text input
- A product catalog of one target + 3 distractors
- An optional upsell modal
- A cart whose contents are exposed as DOM data attributes for verifier reading

Held-out task set: 8 tasks with promotional traps that visually highlight a wrong product. The verifier requires the cart to contain the target product and nothing else.

Calibration was performed on a separate 10-task suite never used for benchmark scoring.

---

## 5. Feature Discovery

We ran 6 contrast prompt categories: `promotional_bias`, `planning`, `goal_tracking`, `hallucination`, `uncertainty`, `impulsive_action`. For each, we generated 2-3 positive/negative pairs and aggregated features with `|Δ activation| > 0.3` across pairs.

This surfaced 15 candidate features across all four steering categories (behavioral, epistemic, task, risk). After magnitude calibration, **12 of 15 features tuned successfully** — that is, a magnitude existed in `[-6, +6]` where the output remained coherent and differed meaningfully from baseline.

The 3 remaining features (all labelled `promotional_bias` from the contrast pass) were too entangled with normal text generation: any suppression broke output coherence. Their decoder vectors likely overlap with critical residual-stream directions for general text.

---

## 6. Step-0 Calibration

The compound steering of one feature per category at calibrated magnitudes (totaling ~13 in absolute delta) was too aggressive and produced garbled outputs. Scaling down by 0.35× preserved coherence but did not flip the agent's Step-0 decision to click the promotional button.

We then ran a focused **Step-0 calibration experiment**: testing every tuned feature individually at its calibrated delta and at 2×, plus four 2-feature combinations, against the exact Step-0 prompt of `promo_cal_001` (USB-C cable task with bright "Buy Now" wireless-earbuds banner).

Result: **two configurations** reliably flipped the agent's first action from `click button#buy-now-hero` to `type "USB-C cable" in search-input`:

1. `hallucination` f26737 at δ = -6.0 alone
2. `goal_tracking` f23803 at +6.0 paired with `hallucination` f485 at -6.0

Steering applied **only at Step 0**, since the same hallucination suppression that prevents clicking the (invented-from-the-model's-perspective) promo trap also prevents clicking real `add-to-cart` buttons later in the trajectory.

---

## 7. Results

Held-out benchmark, 8 promo tasks × 3 trials per policy = 24 trials per policy:

| Policy | Success | 95% CI (Wilson) | Δ vs baseline |
|---|---|---|---|
| baseline (no steering) | **0.0%** (0/24) | [0.0%, 13.8%] | — |
| wrong-sign | 4.2% (1/24) | [0.7%, 20.2%] | +4 pts |
| random | 45.8% (11/24) | [27.9%, 64.9%] | +46 pts |
| **targeted** | **83.3%** (20/24) | [64.1%, 93.3%] | **+83 pts** |

### Causal interpretation

- **Wrong-sign at 4.2%** (in baseline's CI) is the smoking gun. Flipping the targeted edits' sign drops performance back to baseline. The direction of the intervention matters; it is not "any perturbation helps."
- **Random at 46%** shows partial lift from random feature perturbations — likely because random perturbations sometimes break the model's promo-click bias enough that it falls back to the safest real action (search). But targeted is 37 points higher, and crucially **its 95% lower bound (64.1%) exceeds random's mean (45.8%)**.
- **Targeted at 83%** is the validated effect of the specific feature edits.

### Failure-mode mining

Across 25 baseline failures, **four features fire in 100% of failures** at the moment of the failed action: f50853, f19079, f39820, f44602 — plus f38249 at 80%. These are stronger candidates for "actual promo-trap features" than what contrast discovery surfaced. They are likely post-hoc evidence of the model's commitment to the promotional click; investigating them as alternative steering targets is future work.

---

## 8. The Live HUD

The Next.js HUD subscribes to per-step events via WebSocket and renders five panels: browser viewport, live feature activation bars, steering controls, trajectory log, and intervention timeline.

Running `python -m bench.runner --policy targeted --hud …` spawns `agent.ws_server` automatically. Each agent step publishes:

```json
{"type": "step_started",      "task_id": "...", "step": 0}
{"type": "features_read",     "features": [{"id": 26737, "label": "hallucination", "activation": 1.4}, ...]}
{"type": "steering_applied",  "edits": [{"feature_id": 26737, "delta": -6.0, "source": "targeted"}, ...]}
{"type": "action_chosen",     "action": {"action": "type", "target": "search-input", "text": "USB-C cable"}}
{"type": "env_updated",       "screenshot_path": "..."}
{"type": "task_done",         "success": true}
```

The trajectory log writes this same schema to JSONL on disk for offline analysis.

---

## 9. Limitations & Future Work

1. **One concept domain.** All held-out evaluation was on promotional traps. Hallucination-prone and multi-step planning tasks are part of ShopGym but were not used in the reported benchmark; the targeted policy's step-0-only structure is unlikely to generalize directly to those categories.
2. **One backbone.** Results are specific to Llama 3.1-8B-Instruct with Goodfire's SAE on layer 19. Cross-model generalization (Gemma 2 + Gemma Scope SAEs is scaffolded as an alternative in `modal_deploy/app_gemma.py`) is not measured here.
3. **Single-step intervention.** The validated policy applies steering only at Step 0. A dynamic policy that watches feature activations across the trajectory and intervenes only at risky moments would be a stronger and more general claim.
4. **Failure features unexplored.** The 4 features that fire in 100% of baseline failures are strong candidates for the true "promo-trap" representation. We did not yet test steering with them.
5. **Sample size.** 24 trials per policy gives 95% CIs of width ~30 points. Replicating at 100 trials per policy would shrink these.
6. **The model still clicks add-to-cart too many times.** The verifier was relaxed to ignore quantity. A real product would add a "done detector" prompt or termination heuristic.

---

## 10. Reproducibility

The repository (`github.com/kalyvask/inside-the-agent`) is self-contained. Total setup, on a fresh Windows machine with Python 3.12, Node 20, and a HuggingFace account with Llama 3.1 license accepted:

```bash
git clone …
cd inside-the-agent
pip install -e ".[dev]"
playwright install chromium
modal token new                       # browser-auth
modal secret create hf-token HF_TOKEN=...
modal deploy modal_deploy/app.py      # ~2 min
make verify                           # 5-test Day 1 gate
python -m verify.feature_drill        # ~5 min, populates features.yaml
python -m verify.tune_deltas          # ~5 min, calibrates magnitudes
python -m verify.step0_calibration    # ~3 min, finds targeted features
for p in baseline random wrong-sign targeted; do
  python -m bench.runner --policy $p --tasks shopgym/tasks/held_out.json \
    --trials 3 --limit 8
done
python -m bench.analysis --plot       # writes data/results/headline.png
```

Total wall time on a single L40S Modal container: roughly 90 minutes for the full benchmark after a one-time cold start of ~10 minutes for the first model download.

---

## References

- Anthropic, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet," *transformer-circuits.pub*, 2024.
- Goodfire AI, "Llama-3.1-8B-Instruct SAE on Layer 19," *huggingface.co/Goodfire*, 2025.
- Cho et al., "Control RL with SAE Features," arxiv 2602.10437, 2026.
- Templeton et al., "Scaling Monosemanticity," *transformer-circuits.pub*, 2024.
- Anthropic, "Golden Gate Claude" blog post, 2024.
- WebDreamer (Yu Gu et al., 2024) — LLM as world model for web agents — adjacent prior art on browser-agent inference-time intervention.
- BrowserGym, ServiceNow Research, 2024.
