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
- **Targeted at 83.3%** is the validated effect of two specific feature edits at step 0 only — steps 1 onward run with zero steering.
- **Position-mode caveat.** The 83% uses `position_mode=all` (the residual delta hits every position during the steered forward pass). The more surgical `position_mode=last_prompt_only` — which only modifies the last prefill token — gives **0%** in our tests on this benchmark. The effect is real and causal; it is not yet localized to a single token. Scope-comparison table in `artifacts/benchmark_report.md`.
- **Verifier caveat.** Headline rate uses the lenient verifier (cart contains target). A strict-cart pass — "cart contains target exactly once, no other product polluted" — is wired (`bench/compute_strict.py`) and on the roadmap as the canonical headline.

This is *not* a claim that we found "the promotional bias feature." It is a claim that **two specific SAE features, when intervened at the first decision step, causally shift the agent's success rate on this benchmark**. The features are characterized via three independent methods (logit lens, corpus probe, ablation) and labelled by what the methods agree on — `f26737_ui_selection_vocab` and `f23803_distraction_avoidance_vocab`. Full evidence in [`docs/feature_characterization.md`](docs/feature_characterization.md).

See `docs/methodology.md` for the full writeup and method details.

---

## What this is

LLM agents are black boxes. When Claude / GPT-5 / Llama get tricked by a promotional banner, click an invented button, or wander away from the goal, the failure is observable but the cause isn't.

Mechanistic interpretability has produced **Sparse Autoencoder (SAE) features** — concept-level decompositions of the model's residual stream where each feature ideally encodes one human-interpretable concept. Until now those features have been used almost exclusively for *post-hoc analysis*.

This project wires them into a working agent as a **runtime intervention surface**:

- **Read** which features fire at every decision step (live telemetry)
- **Intervene** by adding feature-level deltas to the residual stream during inference
- **See** it all in a HUD: feature activations, intervention timeline, before/after action diff, success/failure verdict

### What ships in the box (as of v0.13)

- An **interactive cockpit** for browser-agent SAE interventions. The HUD shows live SAE feature activations, an effect-size strip per active edit with source-coded colors, a command queue for HUD-issued edits that drain at the next agent step, a baseline-vs-current action diff per step, and a 3-second viewport-ring pulse + source badge whenever a steering edit lands. The HUD-to-runner channel is wired (`POST /control` → `ws_server` queue → `HudPublisher.drain_commands()` → merged into the next `SteeringPlan`).
- A **reproducible testbed** for runtime feature interventions. `bench/artifact_check.py` verifies that every published number in `seed_manifest.json` matches the rows in `data/results/*.jsonl` and fails CI on drift. `bench/report.py` regenerates `artifacts/benchmark_report.md` (per-policy, per-task, per-category, action-quality) from raw artifacts. `bench/compute_strict.py` approximates strict-cart from trajectory action histories.
- A working bridge between **Goodfire's open SAE** and a Playwright-driven shopping agent, with **9 policies**: baseline, static, adaptive-dynamic (watches failure-mining features per step), random (per-trial-seeded), wrong-sign (direction-flip ablation), targeted (2 contrast-derived features at step 0), prompt-only (system-prompt control), noise (matched-norm random residual perturbation), and failure-mining (4 data-derived features that fire in 100% of baseline failures).
- A **live segment on real public sites**. `shopgym/web_env.py` is a generic Playwright env that the agent uses the same way it uses ShopGym. Validated on eBay /deals (works headlessly) and AliExpress; Walmart documented as bot-walled (PerimeterX fingerprints beyond cookies). The v0.8 `executed: bool` field on every step surfaces the honest gap between "model emitted valid JSON" and "Playwright actually clicked something."

## Demo (live cockpit on real eBay)

One terminal command launches the whole live demo:

```bash
# In one terminal (start once, leave running):
python -m agent.ws_server                # localhost:8765
cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev   # localhost:3000

# Open the HUD in your browser, then:
python record_demo.py
# That's the whole demo. It clears HUD state, warms the Modal brain,
# does a 3-second countdown so you can hit "record" in OBS / Win+G,
# then fires the targeted policy on shopgym/tasks/real_ebay.json.
```

The HUD viewport will show eBay's /deals page → search results → results filters. At step 0, the targeted policy's `f26737=-6` and `f23803=+6` edits fire, the viewport gets a 3-second emerald pulse + ring + "⚡ INTERVENTION · targeted · 2 edits" badge, and the Effect Size strip shows both bipolar bars. The agent then submits the search and starts drilling into results.

Replay an offline trajectory instead (no Modal calls during recording):

```bash
python -m verify.replay_trajectory \
  data/trajectories/promo_held_001_seed_0_targeted.jsonl --slow
```

Full runbook + 60-second talk track: [`docs/live_demo.md`](docs/live_demo.md). Recording recipe: [`docs/recording_guide.md`](docs/recording_guide.md).

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

# Full 9-policy benchmark on the 20-task held-out suite × 3 trials
python -m bench.rerun_p0           # baseline / targeted / wrong-sign / random / noise / prompt-only
python -m bench.rerun_v0_9_extra   # failure-mining / dynamic (v0.9 additions)
python -m bench.rerun_p0_2_scope   # targeted at last_prompt_only + all_prompt (scope comparison)

# One-shot orchestrator that runs everything above + regenerates artifacts:
python -m bench.v0_8_finalize

# Inspect / verify the artifacts
python -m bench.artifact_check     # CI gate: verifies manifest matches data/results
python -m bench.report             # regenerates artifacts/benchmark_report.md
python -m bench.compute_strict     # approximate strict-cart from action history
```

### Watch the HUD live (for the demo)

Three terminals:

```bash
# Terminal 1 — WebSocket bridge (long-lived)
python -m agent.ws_server

# Terminal 2 — Next.js HUD frontend (long-lived)
cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev
# Open http://localhost:3000

# Terminal 3 — one-command live demo
python record_demo.py
# Or: python record_demo.py --task shopgym/tasks/held_out.json --pause 6.0

# Or replay an offline trajectory (no Modal cost):
python -m verify.replay_trajectory \\
  data/trajectories/promo_held_001_seed_0_targeted.jsonl --slow
```

### Warm a real-website session (only if you hit bot detection)

```bash
# Opens a real Chrome window. Click through any CAPTCHA / cookies,
# then ask your AI assistant to "go save" — it creates the sentinel
# file and the script writes data/<site>_storage_state.json.
python warm_session.py --url https://www.walmart.com/ \
    --out data/walmart_storage_state.json --channel chrome
```

## Repository layout

```
inside-the-agent/
├── modal_deploy/         brain-server (Modal app, Llama + Goodfire SAE)
│   ├── app.py            primary: Llama 3.1-8B + Goodfire SAE l19, with
│   │                     steer_act / steer_act_with_noise / read_features /
│   │                     feature_logit_lens / feature_decoder_similarity /
│   │                     sae_validation endpoints
│   └── app_gemma.py      fallback: Gemma 2-9B + Gemma Scope (not gated).
│                         Runbook: docs/cross_model_path.md
├── sae/                  loader, steering controller, feature catalog
│   └── features.yaml     v0.4 logit-lens + v0.9 failure-mining labels
├── agent/                trajectory schema, prompts, agent loop, HUD publisher
│   ├── llm_agent.py      core loop: read features → policy → steer → act
│   ├── hud_publisher.py  events to ws_server (policy_meta, baseline_action,
│   │                     step_started, features_read, steering_applied,
│   │                     action_chosen, env_updated, task_done)
│   └── ws_server.py      FastAPI bridge — /feed (WS) /publish /control
│                         /control/pending /clear /screenshots /health
├── policies/             9 policies in POLICY_REGISTRY:
│                         baseline · static · dynamic (adaptive) ·
│                         random · wrong-sign · targeted · prompt-only ·
│                         noise · failure-mining
├── shopgym/              deterministic storefronts (templated) + WebEnv
│   ├── storefront_template.py  ShopGym env + verifier hookup
│   ├── web_env.py              generic Playwright env for real sites
│   └── tasks/                  held_out.json (20 tasks: 8 promo + 6 halluc
│                               + 6 planning), real_ebay.json, real_walmart.json,
│                               real_aliexpress.json
├── bench/
│   ├── runner.py               main CLI: --policy --tasks --hud --pause --position-mode
│   ├── rerun_p0.py             sequential rerun of all 6 main policies
│   ├── rerun_v0_9_extra.py     failure-mining + dynamic
│   ├── rerun_p0_2_scope.py     targeted at last_prompt_only + all_prompt
│   ├── v0_8_finalize.py        chains all reruns + report regen + manifest refresh
│   ├── artifact_check.py       CI gate: verifies manifest ↔ jsonl consistency
│   ├── report.py               regenerates artifacts/benchmark_report.md
│   ├── compute_strict.py       approximate strict-cart from action histories
│   └── verifiers.py            lenient + strict + upsell verifiers
├── hud/                  Next.js cockpit on localhost:3000
│   ├── app/page.tsx            layout + event handlers
│   └── components/             DemoBanner (policy + scope + seed badges),
│                               BrowserViewport, FeatureBars,
│                               SteeringControls, CommandQueue, EffectSizeStrip,
│                               InterventionTimeline, BeforeAfterDiff,
│                               Verdict, SteeringFlash
├── verify/               feature discovery + verification tooling:
│                         sae_smoke, sae_validation, feature_drill,
│                         feature_characterize (logit lens),
│                         tune_deltas, step0_calibration, feature_ablations,
│                         replay_trajectory
├── docs/                 methodology, feature_characterization, demo_script,
│                         live_demo, real_world_generalization,
│                         cross_model_path, recording_guide, data_splits
├── tests/                46 unit tests (action parser, trajectory schema,
│                         verifiers, task config, noise routing, executed
│                         tracking, ...)
├── notebooks/            explore_demo_pages.py (12-site survey)
├── artifacts/            committed subset of data/: seed_manifest.json,
│                         headline.png, benchmark_report.md, strict_rates.json,
│                         sample_trajectory_*.jsonl
├── record_demo.py        one-command live demo launcher (clear + warm + countdown + fire)
├── warm_session.py       headed-Chrome cookie warm-up for bot-walled sites
└── data/                 trajectories, results, baselines, screenshots (gitignored)
```

## Roadmap

### Immediate (this week — demo polish)

1. **Main rerun + auto-finalize** _(running now, ~2h)_. `bench/rerun_p0.py` is replacing the stale v0.2 artifact rows. `bench/v0_8_finalize.py` auto-chains scope reruns + report regen + manifest refresh + artifact_check.
2. **Regenerate `artifacts/headline.png`** from the new numbers — current chart is v0.2.
3. **Refresh README headline table** with v0.7+ rates (random=0% after seed fix, noise + prompt-only rows added).
4. **Flip `artifact_check` from soft-fail to hard-fail in CI** once the artifact rows are consistent.
5. **Record the live cockpit clip** via `python record_demo.py` + screen capture.

### Short-term (1-2 weeks — close P1 reviewer items)

6. **Strict-cart as canonical headline.** Reviewer P1: lenient verifier hides repeated add-to-cart pollution. Run a strict pass that captures `cart_contains_target_exactly_once` alongside lenient.
7. **Per-feature ablation studies.** `f26737` alone vs `f23803` alone vs combined — closes the "is the effect synergistic or additive?" question.
8. **Sponsored-vs-organic decision** on a search-results page. Needs the real-site selector flake addressed first (LLM emits `search-result-N` patterns that don't exist in real DOMs).
9. **HUD: latency badge per step** — credibility marker, ~30 min of plumbing existing timestamps.
10. **HUD: counterfactual baseline diff.** Currently uses a cache from a prior baseline run; live counterfactual = call brain twice/step (with + without edits), shows true per-step divergence. Doubles brain cost.

### Medium-term (next month — strengthen the science)

11. **Cross-model Gemma replication.** Scaffolded in `modal_deploy/app_gemma.py`; runbook in [`docs/cross_model_path.md`](docs/cross_model_path.md). ~$15 Modal + 3 hours attended. Closes the biggest reviewer ask: *"is the result Llama-specific or general?"*
12. **Larger corpus probe.** v0.4 corpus is 40 prompts; streaming 1k+ prompts from a public dataset would tighten the labels for `f26737` and `f23803`.
13. **Failure-mining feature semantic characterization.** `f50853 / f19079 / f39820 / f44602` are still tagged `fail_mode_a/b/c/d` — their logit lens returned code symbols, not English clusters. Try attention-pattern analysis + a bigger corpus to see whether they're causal or symptomatic.
14. **Cross-reference with Neuronpedia.** Other public SAE explorers may have richer data on our features; haven't checked.
15. **HUD trajectory replay mode.** Browse `data/trajectories/*.jsonl` offline, scrub through past runs in the cockpit without rerunning the agent.

### Long-term (months — research direction)

16. **Multi-domain expansion.** Beyond promo / halluc / planning — add forms, comparison shopping, multi-step planning suites. Test whether targeted generalizes across task types.
17. **Dynamic policy v2.** Current adaptive thresholds (0.40 for failure-mining features) are hand-set. Learn thresholds from a validation split.
18. **Compositional steering.** Pair `f26737` with each of its decoder-neighbors (cosine sim > 0.5) — does the steering effect amplify? Tests whether feature clusters or single features carry the meaning.
19. **Reusable testbed.** Package the runner + HUD + brain-server contract so others can plug in their SAE + their model. The wedge per reviewer P2: *"reproducible testbed for runtime feature interventions in browser agents, with live telemetry and controllable steering."*
20. **Failure causality vs correlation.** The 4 failure-mining features fire in 100% of failures — but a heartbeat fires in 100% of car accidents. The `failure-mining` policy (v0.9) tests whether suppressing them actually rescues behavior, separating the causal from correlational story.

### Status of the 4 original "Open questions"

The four open questions from earlier reviewer feedback are now wired and measurable in the codebase:

| Original ask | Status | Where |
|---|---|---|
| Failure-mode features as steering targets | ✅ built (`failure-mining` policy + catalog labels) | `policies/failure_mining.py` |
| Cross-domain (hallucination + planning) | ✅ wired | per-category section in `artifacts/benchmark_report.md` |
| Cross-model (Gemma 2-9B + Gemma Scope) | 📘 runbook ready | `docs/cross_model_path.md` |
| Dynamic steering (not just step 0) | ✅ rewritten | `policies/dynamic.py` watches failure features per step |

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
