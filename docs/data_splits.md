# Data splits: calibration / validation / held-out

The reviewer asked us to be explicit about which tasks were used for what. Here is the breakdown.

## The three splits

### Calibration (`shopgym/tasks/calibration.json`)
10 tasks used during development to:
- Discover candidate features via `verify/feature_drill.py`
- Tune steering magnitudes via `verify/tune_deltas.py`
- Find the step-0 winning configuration via `verify/step0_calibration.py`

**These tasks are seen by the system designer.** No benchmark numbers reported on this set count.

### Held-out (`shopgym/tasks/held_out.json`)
8 tasks used as the validation set for the headline benchmark. The targeted policy's feature IDs and magnitudes were locked before running this set. Results on this set are what the README reports.

**Caveat:** the targeted policy was validated on the *same kind* of trap (promo banners) as the calibration set, just different specific tasks. So this is more like a "validation set" than a held-out test set in the strictest sense.

### Hard held-out (`shopgym/tasks/hard_held_out.json`)
5 tasks introduced in v0.3-D using the harder ShopGym variants:
- `hide_products_until_search` — agent must search before seeing the catalog
- Strict verifier: `cart_contains_target_exactly_once`

Currently the targeted policy is NOT validated on this set — these tasks test whether the intervention generalizes beyond the original trap structure. Treat as a true held-out test set.

### Future: Final held-out (`shopgym/tasks/final_test.json`)
Not yet built. Plan for v0.6:
- New trap types (visually subtle buttons, modals during checkout, fake discount codes)
- DOM noise from v0.5 (`dom_noise_buttons` config)
- Cross-domain tasks (e.g. an information-finding task with no shopping element)
- Reserved for one-shot evaluation only — never used during development

## Honest interpretation of the current numbers

When you read "targeted achieves 83.3% on 8 held-out tasks":

✅ **What's true:** the same 8 tasks were not used to choose the targeted feature IDs or magnitudes.

⚠️ **What's also true:** the *kind* of task (USB-C-cable-with-promo-banner) is similar to the calibration task that produced the feature IDs. We're testing trap-avoidance on similar traps.

❌ **What's NOT validated:** that the targeted policy generalizes to traps with structurally different distractors, or to non-shopping tasks.

The `hard_held_out` set partially addresses this — those tasks require search before catalog visibility — but they still use the same promo-banner mechanism. A truly cross-distribution test (e.g., a hallucination-prone form task) is still on the v0.6 roadmap.

## How to run each split

```bash
# Calibration (development only — don't quote these numbers externally)
python -m bench.runner --policy targeted --tasks shopgym/tasks/calibration.json --trials 3 --limit 10

# Held-out (the reported headline benchmark)
python -m bench.runner --policy targeted --tasks shopgym/tasks/held_out.json --trials 3 --limit 8

# Hard held-out (true generalization test)
python -m bench.runner --policy targeted --tasks shopgym/tasks/hard_held_out.json --trials 3 --limit 5
```

For the 4-policy comparison, swap `targeted` for `baseline`, `random`, `wrong-sign`, `prompt-only`, or `noise`.

## What this means for the demo pitch

When asked "Where exactly is the targeted policy validated?", the honest answer is:

> "On 8 held-out promo-trap tasks, with 3 trials per task = 24 trial outcomes per policy. The trap pattern is similar to the 10 calibration tasks we used during development. We have 5 harder tasks (`hard_held_out`) that we have *not* yet run targeted against — these will be the v0.6 test."

Not "we made the model transparent on browser agents in general." The work is more constrained than that, and the repo now says so.
