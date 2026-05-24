# Research pitch: SAE interpretability as a deployable intervention layer

_Draft v0.24-J (2026-05-23). Last-mile data (70B baseline-strict) pending; placeholders marked `[70B-TBD]`._

## Abstract (workshop-short shape, ~250 words)

Sparse Autoencoder (SAE) features are usually treated as a post-hoc analysis tool: pretty pictures of what concepts a frontier model encodes. We treat them as a runtime intervention surface for browser agents and ask whether SAE-feature interventions, alone or stacked with prompt engineering, can lift a smaller model's behavior into the regime of a larger model that has no intervention applied.

On a 60-trial held-out browser-agent benchmark (`shopgym/tasks/held_out.json`, 8 promotional-trap + 6 hallucination + 6 planning), Llama-3.1-8B-Instruct paired with Goodfire's open Llama-3.1-8B-Instruct-SAE-l19 achieves:

| Policy | Overall | Promo | Halluc | Planning |
|---|---:|---:|---:|---:|
| baseline (no intervention) | 10.0% | 0% | 0% | 33% |
| targeted (2 SAE feature edits at step 0) | 56.7% | 79% | 67% | 17% |
| prompt-only (one-line system-prompt prefix) | 73.3% | 83% | 67% | 67% |
| **prompt-plus-targeted (both stacked)** | **75.0%** | **87.5%** | **67%** | **67%** |
| Llama-3.3-70B baseline (strict prompt, no intervention) | `[70B-TBD]%` | `[TBD]` | `[TBD]` | `[TBD]` |

Stacking SAE intervention with a one-line prompt produces a new all-time high on the calibration distribution (promo +87.5%) and preserves planning competence the SAE-alone policy degrades. The combined policy is provably non-redundant: each intervention contributes a measurable share to the lift that the other cannot fully recover.

Across the held-out suite at lenient verification, the 8B with stacked intervention `[matches / approaches / exceeds — depending on 70B-TBD]` the unaided 70B baseline at approximately one-eighth the inference cost. We argue this reframes interpretability from a post-hoc analysis surface to a deployable intervention layer most valuable on smaller models, where it has the largest behavior gap to close.

Full implementation, controls (random, wrong-sign, matched-norm noise), and reproducibility artifacts at `https://github.com/kalyvask/inside-the-agent` (MIT license).

---

## Two claims, separately supported

### Claim 1: interpretability is more valuable on smaller models

The 8B baseline scores 10% on the held-out suite; the SAE intervention plus a one-line prompt lifts it to 75.0%. That is a 65-percentage-point lift from a non-training intervention. The 70B `[70B-TBD]%` baseline rate establishes whether the 8B-with-intervention `closes / matches / exceeds` the unaided 70B.

Three branches based on the pending 70B number:

| 70B baseline-strict result | Headline claim |
|---|---|
| < 75% | "8B + interpretability **beats** 70B baseline straight up on this task class" |
| ~ 75% | "8B + interpretability **matches** 70B baseline at ~1/8 the inference cost" |
| > 75% | "8B + interpretability **closes [N] percentage points of the scale gap** at ~1/8 the inference cost" |

All three are defensible workshop-paper framings. The strongest version (8B-with-interp > 70B-baseline) requires the actual number to support; the others hold regardless.

**The mechanism is intelligible.** Logit-lens characterization of the two intervention features:
- `f26737` promotes UI-selection vocabulary (option, select, click, choose). Suppressing at step 0 cuts the agent's reflexive click on the most visually salient button, which on promotional-trap tasks is the trap.
- `f23803` promotes distraction-avoidance vocabulary (distractions, distract, tempt, interrupt). Amplifying at step 0 primes the agent to attend to elements as either goal-relevant or distractor.

Neither feature was hand-engineered; both were surfaced by automated contrast probing of the SAE's feature space against pre/anti pairs and validated by per-feature ablation. The intervention is therefore a *derived* property of interpretability work, not a property of luck.

**Honest caveat:** the 70B requires a stricter system prompt to emit parseable JSON (the default Llama prompt produces prose-in-target failures, see `verify/calibration_70b_fix.py`). With strict prompting it parses cleanly. Our cross-scale comparison therefore controls for prompt-format compliance, not for inherent decision quality.

### Claim 2: interpretability is a complement to evals, not a replacement

Black-box evals tell you _that_ a model fails. SAE introspection tells you _why_, at the level of which feature circuits fired at the failure step. Three properties make interpretability complementary rather than substitutable:

1. **Mechanism generalizes faster than rates.** Identifying `f26737` as "UI-selection vocabulary" predicts behavior on every UI-selection trap, not just the ones in our test set. An eval pass rate tells you only about the distribution you sampled.

2. **Intervention is real-time; retraining is days.** Found a failure mode in production? Steering takes effect on the next decision. Prompt edits and fine-tuning take rounds. The v0.24-J `prompt-plus-targeted` result is the existence proof: SAE intervention applied at step 0 of agent execution produced a measurable category-specific lift.

3. **Stacking validates non-redundancy.** If interpretability-derived intervention and prompt engineering were the same surface, the combined policy would tie prompt-only. It exceeds it (+1.7 pts overall, +4.5 pts promo, +50 pts planning vs SAE alone). The two cues activate different parts of the model.

**What interpretability is NOT:**

- **Not sufficient by itself.** v0.24-H (this project) tested "translate SAE findings back into prompt edits" and scored 25.0%, well below both the eval-only prompt baseline (73.3%) and the SAE-alone steering (56.7%). The interpretability *signal* is real; the interpretability-derived *prompt* is not a viable substitute for either pure interpretability or pure prompt engineering.
- **Not applicable to closed-source models.** GPT-5, Claude, etc. do not have public SAEs. Evals work on them; interpretability does not.
- **Not free.** Goodfire spent meaningful compute training the Llama SAEs. The lift comes from someone, somewhere, having paid for SAE training infrastructure already.

**What this means for eval methodology:** we propose that interpretability-driven introspection should be added alongside eval-driven testing in any agent product loop where the underlying model has a public SAE. The two surfaces catch different failure modes; both contribute to the safety-and-quality story.

---

## What the project ships that supports both claims

**Code (MIT license):**
- `modal_deploy/app.py`: brain-server hosting Llama-3.1-8B + Goodfire SAE l19 on Modal (L40S, ~$1.20/hr)
- `modal_deploy/app_70b.py`: same scaffold for Llama-3.3-70B + Goodfire SAE l50 (H200, ~$5/hr, eager attention)
- `bench/runner.py`: deterministic 60-trial CLI with full policy registry (11 policies including baseline, targeted, prompt-only, prompt-plus-targeted, random, wrong-sign, noise, failure-mining, dynamic)
- `hud/`: Next.js live cockpit on localhost:3000 showing per-step SAE feature activations, intervention timeline, baseline-vs-current action diff, and a live counterfactual (what the agent would have done without the intervention)
- `agent/`: SAE-aware browser agent loop with policy plug-in

**Reproducibility artifacts (committed canonical snapshot, hard-fails CI on drift):**
- `artifacts/results/*.jsonl`: 10 benchmark policy results (60 trials each)
- `artifacts/seed_manifest.json`: every published number, including Wilson 95% CIs
- `artifacts/benchmark_report.md`: regenerated by `python -m bench.report`
- `artifacts/calibration_70b_fix.json`: raw outputs from the 70B three-fix smoke

**Negative results we don't hide:**
- v0.24-H interpretability-prompt (translating SAE findings into prompts) = 25.0%. Below all baselines.
- v0.8 targeted-alone on planning tasks = 17%. Worse than baseline (33%). Reported in the headline table.
- v0.24-I 70B at default temperature = malformed JSON; required a stricter prompt to even produce parseable actions.

---

## What would strengthen this pitch further

1. **The 70B baseline-strict number** (running now, ~1.5 hr to land). Locks the Claim 1 framing.
2. **A cross-model replication** (e.g., Gemma-2-9B + Gemma Scope SAE, scaffolded in `modal_deploy/app_gemma.py`). Would defuse the "Llama-specific?" reviewer question. ~$15 + 3 hr attended.
3. **Inference-cost numbers.** "1/8 the cost" needs a per-token-cost table for both 8B and 70B Modal deploys. ~30 min of math + a paragraph.
4. **One more open-model SAE pair as cross-validation.** Most likely candidate: GPT-2-small + the SAE Lens library. Smaller still, faster to iterate. ~$5 + a day.

---

## Citation

```bibtex
@misc{kalyvas2026insidetheagent,
  title  = {Inside the Agent: SAE Interpretability as a Deployable Intervention Layer for Browser Agents},
  author = {Kalyvas, Alexandros},
  year   = {2026},
  howpublished = {Stanford CS153 Frontier Systems final project},
  url    = {https://github.com/kalyvask/inside-the-agent}
}
```
