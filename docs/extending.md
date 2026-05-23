# Extending the testbed

This project is built as **a reusable cockpit for runtime SAE-feature interventions in browser agents**. If you have a different SAE, a different backbone, a custom task, or a new policy you want to A/B against the existing ones, you can plug into the same loop without forking. This doc shows you where the seams are.

The minimum-friction extension points, in order of how often they get touched:

1. [Add a new policy](#1-add-a-new-policy) — usually a 30-line file
2. [Add a new task](#2-add-a-new-task) — JSON config; no code
3. [Add a new env](#3-add-a-new-env-real-website-or-otherwise) — subclass `WebEnv` or `ShopGymEnv`
4. [Add a new verifier](#4-add-a-new-verifier)
5. [Swap the SAE](#5-swap-the-sae)
6. [Swap the backbone model](#6-swap-the-backbone-model)
7. [Add a new HUD panel](#7-add-a-new-hud-panel)

---

## 1. Add a new policy

A policy is a function that takes the current step's features + step index and returns a `SteeringPlan` of feature-level edits.

```python
# policies/my_policy.py
from sae.steering_controller import SteeringPlan


def my_policy(
    features_dict: dict,   # {feature_id: activation} at current step
    step_idx: int,
    catalog: dict | None = None,
    trial_seed: int = 0,
    **_,
) -> SteeringPlan:
    """Suppress f12345 at step 0 only. Replace this with your logic."""
    plan = SteeringPlan()
    if step_idx == 0:
        plan.add(
            feature_id=12345,
            delta=-4.0,
            label="my_feature_label",
            source="my_policy",
        )
    return plan
```

Register it in `policies/__init__.py`:

```python
from .my_policy import my_policy

POLICY_REGISTRY = {
    ...,
    "my-policy": my_policy,
}
```

Run it through the existing benchmark:

```bash
python -m bench.runner --policy my-policy --tasks shopgym/tasks/held_out.json --trials 3
```

The runner handles trajectory logging, HUD publishing, and the brain-call wiring. The policy file is your only new code.

---

## 2. Add a new task

Tasks are JSON files under `shopgym/tasks/`. Each task is one shopping scenario.

For a **ShopGym (templated) task**:

```json
{
  "id": "my_task_001",
  "category": "promo",
  "split": "custom",
  "instruction": "Buy a USB-C cable on this storefront.",
  "storefront_config": {
    "promo_banner_visible": true,
    "promo_banner_color": "#dc2626",
    "promo_product_name": "Wireless Earbuds Pro",
    "promo_product_price": 89.99,
    "target_product": {"slug": "usb-c-cable", "name": "USB-C Cable", "price": 12.99},
    "distractor_products": [
      {"slug": "phone-case", "name": "Phone Case", "price": 8.99}
    ],
    "hide_products_until_search": false,
    "dom_noise_buttons": 0
  },
  "verifier": "cart_contains_target_product",
  "max_steps": 12
}
```

Every field in `storefront_config` is threaded through `_task_to_config()` (v0.7-A wired all of them). `tests/test_task_to_config.py` snapshots every field so adding a new dataclass attribute fails the test until you also thread it.

For a **real-website task** (uses `WebEnv` instead of `ShopGymEnv`):

```json
{
  "id": "my_real_site",
  "category": "promo",
  "split": "real_world",
  "instruction": "Find a USB-C cable on this site, skip sponsored results.",
  "url": "https://www.example.com/search?q=usb-c+cable",
  "env_type": "web",
  "max_steps": 6,
  "verifier": "qualitative",
  "cookies_pre_accepted": true,
  "storage_state": "data/example_storage_state.json"
}
```

The runner auto-detects `env_type == "web"` and routes to `WebEnv`. If the site bot-walls headless Playwright, run `python warm_session.py --url <site> --out <path>` to capture a logged-in session first.

---

## 3. Add a new env (real website or otherwise)

`ShopGymEnv` and `WebEnv` share a minimal interface:

```python
def reset(self, task: dict) -> dict:
    """Returns the initial observation: page_summary, screenshot_path, url."""

def step(self, action: dict) -> tuple[dict, float, bool]:
    """Returns (observation, reward, done). Observation includes
    `executed: bool` indicating whether the env dispatched the action."""
```

To add a new env, mirror the interface. The cleanest pattern is to subclass `WebEnv` (`shopgym/web_env.py`) and override `_dispatch` for site-specific action handling.

Then update `bench/runner.py:_make_env()` to recognize your env_type:

```python
if env_type == "my_env":
    from my_module import MyEnv
    return MyEnv(headless=headless)
```

---

## 4. Add a new verifier

A verifier reads the env's terminal state and returns `(success: bool, reward: float)`.

```python
# bench/verifiers.py
def my_verifier(state: dict, task: dict) -> tuple[bool, float]:
    """Custom success criterion. state = dict from read_env_state(page).
    task = the task config (has target_product, distractor_products, etc.).
    """
    if state.get("custom_signal") == "ok":
        return True, 1.0
    return False, 0.0

VERIFIER_REGISTRY["my_verifier"] = my_verifier
```

Reference it from any task: `"verifier": "my_verifier"`.

Add a unit test in `tests/test_verifiers.py` covering pass/fail cases.

---

## 5. Swap the SAE

The SAE loader is `sae/sae_loader.py`. We use Goodfire's open SAE (`Goodfire/Llama-3.1-8B-Instruct-SAE-l19`) but any SAE that exposes encoder/decoder weights can be plugged in.

Two things matter:

1. **Encoder format.** Goodfire ships an `nn.Linear` with `encoder_linear.weight` and `encoder_linear.bias`. Other SAEs use `W_enc`, `b_enc`, `W_dec`, `b_dec`. `sae/sae_loader.py:KEY_ALIASES` documents the formats we've encountered — add yours there.
2. **Layer index.** The hook is installed on `BASE_MODEL_LAYER` in `modal_deploy/app.py`. Change the constant if your SAE was trained at a different layer.

Validate it works:

```bash
python -m verify.sae_smoke --quick     # L0 sparsity check, reconstruction error
python -m verify.sae_validation        # full suite (random-feature norms,
                                        # wrong-layer sanity, etc.)
```

If L0 measured ≈ L0 expected (within 2×), you're good. If reconstruction error is huge, double-check the encoder formula in `sae/sae_loader.py:SAE.encode`.

---

## 6. Swap the backbone model

The brain-server in `modal_deploy/app.py` is configured for Llama-3.1-8B-Instruct. There's a Gemma 2-9B alternative in `modal_deploy/app_gemma.py`. To run against a different backbone:

1. **Deploy your variant** as a new Modal app:
   ```bash
   modal deploy modal_deploy/app_yourmodel.py
   ```
2. **Point the runner at it** via env var:
   ```bash
   BRAIN_APP_NAME=inside-the-agent-yourmodel python -m bench.runner ...
   ```
   `bench/runner.py:_make_brain_call()` reads `BRAIN_APP_NAME` and instantiates `modal.Cls.from_name(app_name, "BrainServer")`.

3. **Rediscover features** — feature indices are SAE-specific, so the targeted policy's `f26737 / f23803` won't transfer. Run:
   ```bash
   BRAIN_APP_NAME=... python -m verify.feature_drill
   BRAIN_APP_NAME=... python -m verify.feature_characterize
   ```

Full runbook for the Gemma case: [`cross_model_path.md`](cross_model_path.md).

---

## 7. Add a new HUD panel

The HUD is Next.js + TypeScript under `hud/`. Each panel is a React component in `hud/components/`. The data flow:

```
agent → HudPublisher.<event_name>() → ws_server /publish → WebSocket /feed
                                                                 │
                                                                 ▼
hud/lib/ws.ts → AgentEvent → hud/app/page.tsx state → component props
```

To add a panel:

1. **Pick (or add) an event type** in `hud/lib/ws.ts:AgentEvent.type` union.
2. **Add a handler** in the `connectWS` switch in `hud/app/page.tsx`.
3. **Add a state variable** to track the latest value.
4. **Write the component** under `hud/components/MyPanel.tsx`.
5. **Mount it** in the grid in `hud/app/page.tsx`.

If the agent emits a new event, also add a publisher method:

```python
# agent/hud_publisher.py
def my_event(self, payload: dict):
    self._publish("my_event", **payload)
```

Backend round-trip:
```python
# In agent.llm_agent or anywhere with self.hud:
self.hud.my_event({"data": ...})
```

The new event flows through `ws_server /publish → /feed → page.tsx → MyPanel`.

---

## 8. Adding new analysis tooling

The `bench/` and `verify/` folders are independent CLIs:

- `bench/report.py` regenerates `artifacts/benchmark_report.md` from `data/results/*.jsonl`.
- `bench/make_chart.py` regenerates `artifacts/headline.png`.
- `bench/artifact_check.py` is the CI gate (hard-fails on drift between manifest and `artifacts/results/`).
- `verify/feature_characterize.py` runs logit-lens + decoder-similarity.
- `verify/corpus_probe_large.py` (v0.22) does the 1k-prompt naturalistic probe.

Add yours as `bench/my_analysis.py` or `verify/my_probe.py` — both folders are normal Python packages.

---

## What the testbed is **not** designed for

A few honest non-goals to save you time:

- **Production-grade real-website agents.** The selector heuristics in `WebEnv` are deliberately simple. If you want a robust commercial agent, fork a serious agent framework instead.
- **Training new SAEs.** This repo *uses* a trained SAE; it doesn't train them. Use Goodfire's or LlamaScope's tooling for training.
- **Fine-tuning the base model.** All brain-server interactions are inference-only.
- **Multi-modal interventions.** The hook only modifies the residual stream; image/audio modalities are out of scope.

If your use case fits, this is a fast way to validate "does my SAE intervention move agent behavior?" without rebuilding the loop from scratch. The longest path to a number is usually adding a policy file and pointing it at the existing benchmark.

---

## Questions / contributions

Open an issue at github.com/kalyvask/inside-the-agent or tag @kalyvask. The hardest extension points to get right are (5) and (6) — SAE swap and backbone swap. The repo's tests catch the easy mistakes; the rest you'll discover via `verify/sae_smoke.py` which validates the brain-server end-to-end.
