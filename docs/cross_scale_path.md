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
