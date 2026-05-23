# Cross-model replication: Gemma 2 9B + Gemma Scope SAE

## Why this matters

The headline 83% targeted result is specific to **Llama 3.1-8B-Instruct +
Goodfire's layer-19 SAE**. A reasonable reviewer asks: is this a property of
that specific pair, or a property of "SAE-mediated steering in browser
agents" more broadly? Replication on a different backbone + different SAE
provider answers that.

The Gemma path is scaffolded but not run. This doc captures the runbook
so the work is a one-day project, not a one-week project.

## What's already in place

- `modal_deploy/app_gemma.py` — full Modal app definition mirroring `app.py`
  but with `google/gemma-2-9b-it` and `google/gemma-scope-9b-it-res`
  (layer 20, width 16k, L0 ~71).
- `BRAIN_APP_NAME` env var — `bench/runner.py:_make_brain_call` reads this
  and points the same agent loop at either Llama or Gemma without code
  changes.
- All policies, the HUD, the benchmark suite, and the verifiers are
  backbone-agnostic. The only Llama-specific concept is the catalog of
  feature IDs (f26737, f23803, etc.).

## What needs to happen (one-day plan)

1. **Deploy Gemma server** (~15 min)
   ```bash
   modal deploy modal_deploy/app_gemma.py
   # First cold-start downloads ~18 GB to hf-cache volume; subsequent
   # invocations reuse.
   ```

2. **Validate the Gemma SAE** (~30 min, reuses existing verify tooling)
   ```bash
   BRAIN_APP_NAME=inside-the-agent-gemma python -m verify.sae_smoke
   BRAIN_APP_NAME=inside-the-agent-gemma python -m verify.sae_validation
   # Confirm measured L0 within 2x of expected (71), reconstruction
   # error sane, layer-0 sanity check fails as designed.
   ```

3. **Discover Gemma feature candidates** (~1 hour)
   ```bash
   BRAIN_APP_NAME=inside-the-agent-gemma python -m verify.feature_drill \
       --tasks shopgym/tasks/calibration.json
   # Reuse the same contrast prompts that surfaced f26737 / f23803 on
   # Llama; record which Gemma features have the highest contrast scores.
   ```

4. **Tune deltas on calibration** (~30 min)
   ```bash
   BRAIN_APP_NAME=inside-the-agent-gemma python -m verify.tune_deltas
   # Same tuning loop as Llama: -6, -3, +3, +6 magnitudes, pick the pair
   # that flips step-0 behavior most reliably.
   ```

5. **Run the benchmark** (~30 min on L40S, more on smaller GPU)
   ```bash
   BRAIN_APP_NAME=inside-the-agent-gemma \
   python -m bench.runner --policy baseline --tasks shopgym/tasks/held_out.json --trials 3
   # Then targeted, then random, etc.
   ```

6. **Compare to the Llama headline**
   - Report side-by-side rates per policy in `artifacts/benchmark_report.md`
     under a new "Cross-model replication" section.
   - Flag any condition where the relative ordering differs (e.g. if
     wrong-sign WORKS on Gemma it means our control isn't as clean as
     we thought).

## What success looks like

| Outcome | What it means |
|---|---|
| Gemma targeted >> Gemma baseline, similar to Llama | Strong cross-model replication. The technique generalizes. |
| Gemma targeted ≈ Gemma random | Llama result was lucky / overfit to the specific SAE. Recalibrate from scratch. |
| Gemma SAE doesn't even read coherent features on agent prompts | Gemma Scope layer-20 isn't the right intervention point; try layer 12 or 8. |

## What we deliberately punt on

- **Strict policy transfer**: don't try to apply Llama's feature IDs to Gemma
  directly. The SAE feature spaces are unrelated. Each backbone needs its
  own discovery + tuning pass.

- **Joint plots**: don't overclaim "the agent has X feature" by combining
  Llama and Gemma results. They share a benchmark, not a representation.

## Cost estimate

- One full Gemma calibration + benchmark pass: ~$10-15 Modal compute on
  L40S, assuming hot containers.
- Cold container warm-up: ~$2 (Gemma 2 9B + SAE weights load in ~3 min
  from the volume).
