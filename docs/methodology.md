# Inside the Agent — Methodology

**Version:** 0.1 (Day 0 — May 21, 2026)
**Status:** scaffolding; to be expanded throughout the build.

## Problem

Modern LLM agents are opaque. When an agent fails on a multi-step task — clicks the wrong button, hallucinates a UI element, gets baited by a promotion — the failure is observable but the cause is not. Existing debugging tools work on inputs/outputs, not on the model's internal representations.

## Approach

We expose the model's internal **Sparse Autoencoder (SAE) features** as a runtime control surface. SAE features are concept-level decompositions of the residual stream, produced by training a sparse linear autoencoder on the model's intermediate activations.

Each feature ideally corresponds to a single interpretable concept (e.g., "planning behavior," "promotional bias"). With them:
- **Read:** observe which concepts fire as the agent reasons.
- **Steer:** apply additive deltas to feature activations during generation, with clamps to prevent out-of-distribution behavior.

## Architecture (three processes)

- **brain-server** (Modal L40S): hosts Llama 3.1-8B-Instruct (BF16) and Goodfire's open SAE on layer 19. Exposes `/health`, `/read_features`, `/steer_act`.
- **browser-worker** (local): runs ShopGym storefronts via Playwright, dispatches agent actions, writes per-step trajectories as JSONL.
- **hud** (local Next.js): subscribes to a WebSocket event stream from the browser-worker. Displays browser viewport, live feature bars, steering controls, intervention timeline.

## Benchmark

**ShopGym** — a custom deterministic suite of mini-storefronts, configurable in their distractor patterns (promo banners, upsell modals, look-alike products, required-field traps). 30 tasks split:
- **10 calibration tasks** for feature discovery + steering tuning
- **20 held-out tasks** for the reported benchmark

Four agent conditions on held-out:
1. **baseline** — no steering
2. **random-feature steering** — same magnitudes on randomly chosen features (control)
3. **wrong-sign steering** — same features as targeted but flipped (ablation)
4. **targeted steering** — the dynamic policy

If targeted wins and random + wrong-sign do not, the result is causally credible.

## Feature discovery

Contrast prompts identify features that differentiate concept-relevant from concept-irrelevant text. Each feature in `sae/features.yaml` has:
- `top_activation_examples` (3+ verified prompts)
- `contrast_score` (positive vs. neutral)
- `causal_effect` (measured by steering ablations)
- `recommended_delta`
- `confidence` (low/medium/high)

No dependency on Goodfire's hosted Ember label API.

## Steering primitives

```
new_activation = clamp(old_activation + sum(delta_i * W_dec[i]), min, max)
```

Applied only during action generation, not the full prompt. Clamp bounds default to [-10, 10] (verified empirically on Day 1).

## Metrics

Per-task: success (binary), wrong-click rate, invalid-action rate, hallucinated-element rate, steps-to-success, latency-per-step.

Per-condition: mean success rate per category (promotional / hallucination / planning), with 95% confidence intervals over 3 trials.

## Reproducibility

`git clone && make install && make deploy && make verify && make bench` reproduces all results. Modal cold-start: ~2 min. Full benchmark: ~6 hours. Total compute cost: under $80 (Modal L40S).

## References

- Anthropic, "Scaling Monosemanticity," 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity
- Goodfire AI, Llama-3.1-8B-Instruct-SAE-l19 release. https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19
- Cho et al., "Control RL with SAE Features," arxiv 2602.10437.
- WebDreamer (Yu Gu et al.), arxiv 2411.06559.
