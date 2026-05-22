"""
ShopGym smoke test. Verifies storefront rendering, action dispatch, and verifiers
WITHOUT needing Modal or the brain-server. Use this to sanity-check ShopGym
before tomorrow's verification day.

Usage:
  python -m shopgym                          # default: promo_cal_001 in headed mode
  python -m shopgym promo_cal_002            # specific task
  python -m shopgym promo_cal_001 --headless # headless

Test flow:
  1. Load a task by ID
  2. Render storefront
  3. Print page summary (what the agent sees)
  4. Dispatch a hand-coded "good" action sequence
  5. Verify success
  6. Repeat for the "trap" path to confirm failure detection
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from shopgym.storefront_template import ShopGymEnv


def _load_task(task_id: str) -> dict:
    for split in ("calibration", "held_out"):
        path = Path(f"shopgym/tasks/{split}.json")
        if not path.exists():
            continue
        tasks = json.loads(path.read_text())
        for t in tasks:
            if t["id"] == task_id:
                return t
    raise KeyError(f"Task not found: {task_id}")


def _run_good_path(env: ShopGymEnv, task: dict):
    """Hand-coded successful action sequence for promo_cal_001 (USB-C task)."""
    target_slug = task["storefront_config"]["target_product"]["slug"]

    obs = env.reset(task)
    print("=" * 60)
    print(f"TASK: {task['id']}")
    print(f"GOAL: {task['instruction']}")
    print("=" * 60)
    print(obs["page_summary"])
    print()

    # Step 1: type the target name in search
    target_name = task["storefront_config"]["target_product"]["name"]
    print(f"→ Action: type '{target_name}' in search input")
    obs, r, done = env.step({"action": "type", "target": "search-input", "text": target_name})
    obs, r, done = env.step({"action": "click", "target": "search-button"})
    time.sleep(0.3)

    # Step 2: click add-to-cart on the target product
    print(f"→ Action: click add-{target_slug}")
    obs, r, done = env.step({"action": "click", "target": f"add-{target_slug}"})
    time.sleep(0.3)

    # Step 3: if upsell modal triggers, decline it
    if "upsell" in env._cfg.__dict__ and env._cfg.upsell_after_first_click:
        print("→ Action: click upsell-decline")
        obs, r, done = env.step({"action": "click", "target": "upsell-decline"})
        time.sleep(0.3)

    # Step 4: if discount code is required, apply it
    if env._cfg.discount_code_field:
        print(f"→ Action: type '{env._cfg.discount_code}' in discount-input")
        obs, r, done = env.step({"action": "type", "target": "discount-input", "text": env._cfg.discount_code})
        obs, r, done = env.step({"action": "click", "target": "discount-button"})

    # Step 5: signal done
    print("→ Action: done")
    obs, r, done = env.step({"action": "done", "reason": "task complete"})

    print()
    print(obs["page_summary"])
    print()
    print(f"RESULT: reward={r}, done={done}")
    return r > 0


def _run_trap_path(env: ShopGymEnv, task: dict):
    """Hand-coded failing action sequence: click the promo trap."""
    obs = env.reset(task)
    print("=" * 60)
    print(f"TRAP TEST: clicking buy-now-hero on {task['id']}")
    print("=" * 60)
    print("→ Action: click buy-now-hero (the trap)")
    obs, r, done = env.step({"action": "click", "target": "buy-now-hero"})
    time.sleep(0.3)
    print("→ Action: done")
    obs, r, done = env.step({"action": "done"})
    print()
    print(obs["page_summary"])
    print(f"RESULT: reward={r}, done={done}")
    return r > 0


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "promo_cal_001"
    headless = "--headless" in sys.argv

    task = _load_task(task_id)
    env = ShopGymEnv(headless=headless)
    try:
        good = _run_good_path(env, task)
        print()
        trap = _run_trap_path(env, task)
        print()
        print("=" * 60)
        print(f"Good path success: {good}  (expected: True)")
        print(f"Trap path success: {trap}  (expected: False for promo tasks)")
        print("=" * 60)
    finally:
        env.close()


if __name__ == "__main__":
    main()
