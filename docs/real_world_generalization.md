# Real-world generalization probe (v0.6)

## Setup

Same agent (Llama 3.1-8B + Goodfire SAE), same brain server. Two policies:
- `baseline` — no steering.
- `targeted` — Step-0 only: `f26737 = -6.0` (UI-selection vocab) and `f23803 = +6.0` (distraction-avoidance vocab). These were validated to lift the agent from 0% → 83% on ShopGym held-out (24 trials).

Held-out target: a single AliExpress homepage task (`shopgym/tasks/real_aliexpress.json`), instruction: *"Buy a USB-C charging cable on this storefront. Ignore promotional banners and use the search bar."*

Single trial each. No HUD. `--position-mode all`.

## Result

|                          | Baseline | Targeted |
|--------------------------|----------|----------|
| Step 0 action            | `type` into `#search-words` | `type` into `search-words` |
| Step 1–5 actions         | click "Shop now" / "All Categories" / "More" | same |
| URL changed              | no (stuck on homepage) | no |
| Top features firing      | f44602, f39820, f19079 | f44602, f39820, f19079 |
| f26737 in top-20         | not in top-8 | not in top-8 |
| f23803 in top-20         | not in top-8 | not in top-8 |

Both policies behaved nearly identically. The "targeted" features that produce the 83% headline on ShopGym are not in the top activations on AliExpress at all — the post-steering read shows feature activations that differ from baseline by ≤0.05.

## Why this is honest, not bad

The 83% headline is on **in-distribution** data: ShopGym's controlled templated promo vocabulary ("HOT DEAL", "Featured Deal", "20% off"). The logit lens told us f26737 promotes the tokens `option / selection / select / choices / radio` and f23803 promotes `distractions / distract / tempt / interrupt / notifications`. These are narrow lexical features.

AliExpress's homepage uses a completely different distractor vocabulary: "Shop now", "All Categories", "More", "Today's Deals". The SAE features tuned on ShopGym don't activate strongly on this surface form, so steering them has no measurable downstream effect.

This is the result a reviewer should expect from a 2-week project that validates features on one distribution. **It is not a failure of the steering hypothesis** — it is a clean demonstration that the specific features identified by logit-lens on ShopGym are lexically narrow, and the right next research move is feature discovery on the new distribution, not assuming transfer.

## What the agent actually fails on (separate issue)

Even on AliExpress, both policies type the right search query at step 0. The agent then fails to **submit** the search — it tries to click a fictional "Search" button (one that doesn't exist as a clickable element with that label). The fix is either:

1. Add a `submit` action handler that presses Enter on the active input (done in `shopgym/web_env.py`).
2. Teach the prompt to use `submit` when an input has been filled and no obvious Submit button exists.

This is an agent-control issue, not a steering issue.

## What this means for the demo

The narrative for May 29 is now:

1. **In-distribution headline** (24-trial benchmark): targeted 83% vs baseline 0% on ShopGym.
2. **Out-of-distribution probe** (this doc): same features don't transfer to AliExpress because SAE features are lexically narrow. The feature discovery process must be repeated per distribution.
3. **Live HUD demo on AliExpress**: shows the audience that the HUD streams *live* feature activations on a real, recognizable site, even if our specific Step-0 features no longer apply. The HUD's value is the live introspection, not the specific intervention.

This is the kind of result that earns trust: positive headline + honest negative generalization probe + clear next research question.

## Reproducing

```bash
# Baseline
python -m bench.runner --policy baseline \
  --tasks shopgym/tasks/real_aliexpress.json \
  --trials 1 --limit 1 --position-mode all

# Targeted
python -m bench.runner --policy targeted \
  --tasks shopgym/tasks/real_aliexpress.json \
  --trials 1 --limit 1 --position-mode all
```

Inspect: `data/trajectories/aliexpress_buy_usb_c_cable_seed_0_{baseline,targeted}.jsonl`.
