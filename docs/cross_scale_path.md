# Cross-scale path: Llama-3.3-70B + Goodfire l50 SAE

## Why this is the next test

At the Goodfire fireside at GSB (Feb 12 2026), **Myra Deng** stated that *bigger models are easier to interpret* — that internal representations become cleaner and more disentangled with scale. Our v0.8 finding on Llama-8B is mechanically consistent with the opposite: the layer-19 features look **lexically narrow** ("click this option" vocab), which suppresses promotional traps (79% success) AND hallucination tasks (67%) — but also suppresses legitimate clicks on planning tasks (17%, below baseline).

If Myra's claim holds, the 70B model's SAE features should be more *semantic* (encoding goal-tracking / action-correctness) and less *lexical* (encoding which verb to emit). That would predict the planning failure mode goes away on 70B.

It's a clean experiment — Goodfire publishes both SAEs openly, so we can run this without asking for anything.

## What you can do without further help

`modal_deploy/app_70b.py` is scaffolded. It mirrors `modal_deploy/app.py` with three changes:

| What | 8B (current) | 70B (this path) |
|---|---|---|
| `BASE_MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.3-70B-Instruct` |
| `SAE_REPO_ID` | `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` |
| `SAE_LAYER_INDEX` | 19 (of 32) | 50 (of 80) |
| Modal app name | `inside-the-agent` | `inside-the-agent-70b` |
| GPU | `L40S` (48 GB) | `H200` (141 GB) — single-GPU fit |

The rest of the project — `bench/runner.py`, `policies/`, `verify/feature_drill.py`, the HUD — works unchanged. We deliberately built around a `BRAIN_APP_NAME` env var so the same agent can target either brain.

## Realistic timeline

| Step | Wallclock | Modal cost | Notes |
|---|---|---|---|
| **0. Finish `app_70b.py` method bodies** | 30 min local | $0 | Copy the read_features / steer_act / etc. method bodies from `app.py` into `app_70b.py` (TODO marker in the file). They're SAE-size-generic — no logic changes. |
| **1. Deploy + cold start** | 5-10 min | $0.50 | `modal deploy modal_deploy/app_70b.py` — first run downloads 140 GB of 70B weights into the hf-cache volume. Future cold starts are 2-3 min. |
| **2. SAE smoke check** | 5 min | $0.50 | `BRAIN_APP_NAME=inside-the-agent-70b python -m verify.sae_smoke --quick`. Confirms L0 ≈ expected (Goodfire's l50 should be ~70-100), reconstruction error is sane, layer-0 sanity fails as designed. |
| **3. Feature discovery** | ~30 min | $3-5 | `verify.feature_drill` runs the contrast prompts. Surfaces candidate features for "ui-selection" and "distraction-avoidance" in the 70B SAE's feature space (which is unrelated to the 8B's f26737 / f23803 — feature indices don't transfer across SAEs). |
| **4. Magnitude tuning** | ~30 min | $3-5 | `verify.tune_deltas` picks the deltas that produce coherent + behaviorally-distinct outputs without breaking generation. Expect different magnitudes than the 8B's ±6. |
| **5. Step-0 calibration** | ~30 min | $3-5 | `verify.step0_calibration` finds the feature pair that most reliably flips the first decision. Output: the new "targeted_70b" feature IDs + deltas. |
| **6. Baseline benchmark** | ~60 min | $6-10 | `python -m bench.runner --policy baseline --tasks held_out.json` on the 60-trial suite. |
| **7. Targeted benchmark** | ~60 min | $6-10 | Same but with the new step-0 edits from step 5. |
| **Total** | **~4 hours attended** | **~$25-40** | Most of it on Modal GPU time |

## What actually happened (v0.24-I — 2026-05-23)

Three days into the cross-scale run, the picture is more nuanced than the original three outcomes suggested. Documented here to keep the runbook honest.

**1. Deploy + verification worked cleanly.** App `inside-the-agent-70b` is live (`Llama-3.3-70B-Instruct` + `Goodfire/Llama-3.3-70B-Instruct-SAE-l50`). All four `sae_smoke` tests pass: health, read_features, contrast pairs, steering effect. cuDNN SDPA hit "No execution plans support the graph" on the H200; bypassed via `attn_implementation="eager"` (v0.24-C).

**2. Feature discovery surfaced 14 candidates across 6 contrasts** (v0.24-E, see `sae/features_70b.yaml`): 7 high-confidence, 5 medium, 2 low. Promo bias (f25161, f25021, f19733), planning (f4346), goal-tracking (f19808, f16718, f51865) cluster as expected. The IDs are completely different from the 8B's (f26737, f23803) because SAE feature indices do not transfer across SAEs.

**3. `tune_deltas` converged on ±6.0 for every single feature.** The CANDIDATE_MAGNITUDES range tops out at 6.0; the 70B coherence guard passed at that magnitude for every feature tested. Suggests either the 70B is robust to that magnitude across feature directions, or the coherence threshold is too permissive.

**4. `step0_calibration` initially looked like a clean negative.** Baseline emitted malformed JSON (`{"action": "click", "target": "Add to cart" on "USB-C Cable"}`). The action parser rejected it. Most single-feature interventions at ±6 also produced INVALID; all 4 compositions tested produced INVALID; promo-bias features specifically never rescued at any magnitude.

**5. A three-fix smoke (`verify/calibration_70b_fix.py`) found the root cause was prompt format, not steering.** Tried (a) temperature 0.05, (b) a stricter JSON-only system prompt that names the prose-leak failure mode and shows correct/wrong examples, (c) gentler composition magnitudes (±2, ±3, ±4). With the strict prompt + temp 0.05, **every condition produced valid + correct actions** including the previously-failing compositions:

| Condition | Baseline prompt | Strict prompt |
|---|---|---|
| baseline (no steering) | INVALID | `type "usb-c cable"` (search) |
| promo+goal ±6 | INVALID | ✓ click USB-C cable |
| promo+goal ±3 | INVALID | ✓ click USB-C cable |
| promo+goal ±2 | INVALID | ✓ click USB-C cable |
| impulsive+goal ±2 | INVALID | ✓ click USB-C cable |

Raw outputs in `artifacts/calibration_70b_fix.json`.

**6. The implication for the cross-scale story.** With the strict prompt the 70B baseline already chooses search/correct-action on the calibration prompt, while the 8B baseline (with its default prompt) falls for the promo trap. This is a *different* finding than the original three outcomes anticipated — not "scale fixes interpretability" or "scale doesn't fix it", but **"scale + proper prompting reduces the failure mode the SAE intervention was designed to fix"**. The interpretability lift is most valuable where the base model needs it most. The new headline-candidate story: smaller model + SAE intervention approaches larger model's unaided baseline at a fraction of the inference cost.

The remaining empirical question: what is the 70B's baseline rate on the full 60-trial held_out suite with the strict prompt? That number quantifies "approaches" in the previous sentence. Captured next via `BRAIN_APP_NAME=inside-the-agent-70b python -m bench.runner --policy baseline-strict-prompt ...`.

### v0.24-K outcome

Two iterations were needed:
- **First run (v0.24-K-v1) scored 0/60**, but the failure was an operator bug not a model finding: the strict prompt taught the model to emit CSS-style selectors (`button#add-usb-c-cable`), which ShopGym's env rejects (every action `executed=False`). The 70B's JSON was perfectly formed and semantically correct; the env couldn't act on it.
- **Fix**: the strict prompt now teaches bare element_ids (`add-usb-c-cable`), matching the env's parser convention. Validated with a 1-trial smoke (succeeded in 3 steps).
- **Second run (v0.24-K final)**: **60/60 = 100%** across all three categories (promo 24/24, halluc 18/18, planning 18/18). Wilson 95% CI [0.94, 1.0]. Avg 2.9 steps per trial (vs the 8B's ~12).

The 70B-baseline-strict is **saturated** on the held-out suite. SAE intervention on the 70B was therefore not benchmarked in full; there is no room for it to add value at this scale on this task class.

**The cross-scale claim, locked in v0.24-K:**

> Llama-3.1-8B-Instruct + SAE intervention + a one-line prompt lifts overall success from **10% baseline to 75% (prompt-plus-targeted)**, closing **72% of the (10 → 100) cross-scale gap** to Llama-3.3-70B-Instruct's unaided baseline at approximately **one-eighth the inference cost**. The interpretability lift is most valuable where the base model needs it most. On a saturated 70B, SAE intervention adds nothing; on a struggling 8B, two feature edits and a sentence close 65 percentage points.

Honest caveat: the 70B uses a 1-line format-rescue prompt prefix (strict-JSON guidance, no behavioral instruction about traps). This is a deployment-implementation detail to make the 70B's verbose default JSON output parseable by ShopGym, NOT a behavioral intervention. The 8B does not require this fix. The cross-scale comparison therefore controls for prompt-format compliance, not decision quality.

## How to read the result

Three plausible outcomes:

| Outcome | What it means |
|---|---|
| **Targeted 70B beats targeted 8B AND fixes planning** (e.g. targeted=70%+, planning>=baseline) | Myra's claim validated — scale brings interpretable semantic features. Strong cross-scale generalization claim. Worth a paper. |
| **Targeted 70B matches 8B (~57%) with the same planning failure** | Scale alone doesn't fix the lexical limit. The intervention is genuinely lexical regardless of model size. Honest finding, less paper-worthy but still publishable. |
| **Targeted 70B underperforms** (e.g. 70B-with-edits < 70B-baseline) | The 70B SAE captures different concepts at layer 50; our intervention pattern doesn't transfer. Important null result. |

Each is a publishable finding. There's no "wasted run" here.

## What to ask Myra in the LinkedIn message

Now that we know what we're doing:

- **NOT** "do you have a 262k variant" (likely no — Goodfire's open releases are what they publish)
- **NOT** "do you have a different layer" (same)
- **YES** "planning to try your 70B SAE — any layer / sparsity / discovery recommendations you'd share?"

That's framed as you-as-the-doer asking for guidance, not requesting handouts.

## When to actually run this

**Not before May 29 demo.** It's 4 hours of attended work + $25-40 Modal. The current v0.8 numbers + the captured Google Shopping trajectories carry the demo.

Best timing: **week after demo day**. Run it as a follow-up data point that you can:

1. Add to the README as a v0.24 update.
2. Cite in the LinkedIn follow-up to Myra (turns the cold message into an iterating conversation).
3. Submit to an interpretability workshop alongside the original v0.8 results.

## Honest non-goals

- **NOT training your own 70B SAE.** Goodfire spent meaningful compute training this one; replicating that infrastructure is a separate one-day project that competes with their team without distinguishing your work. Use theirs.
- **NOT comparing across model FAMILIES** (e.g., Llama vs DeepSeek vs GPT-OSS). Cross-family comparisons confound model + SAE + training-data differences. Stay within Llama for a clean scale test.
- **NOT chasing the layer choice.** Layer 50 is what Goodfire published for the 70B. Don't re-invent the SAE-layer-selection problem.

## Connection back to the project narrative

If the 70B run lands well, the README headline becomes:

> *Two SAE feature edits at step 0 lift a Llama-8B agent from 10% to 57% on a 20-task held-out suite (79% on promo, 67% cross-domain to hallucination, 17% on planning). The same intervention pattern on Llama-70B lifts from X% to Y%, AND planning rises to Z% — supporting Goodfire's "bigger models are easier to interpret" thesis at the level of agentic intervention robustness, not just SAE-feature monosemanticity.*

That paragraph is publishable as a workshop short.

If the 70B run shows the planning failure persists, the README still gets:

> *...The same intervention pattern on Llama-70B replicates the promo + hallucination effect (P% / H%) but planning stays at Z%. The lexical-feature limit appears to be intrinsic to the SAE training objective, not the base model's representational capacity.*

That's also publishable, just as a different framing.
