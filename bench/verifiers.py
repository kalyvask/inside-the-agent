"""
Verifiers: pass/fail per task. Each ShopGym task references one by name.

Each verifier signature: (env_state: dict, task: dict) -> (success: bool, reward: float)

env_state schema (from shopgym.storefront_template.read_env_state):
    {
        "cart_items": ["USB-C Cable", "Phone Case"],
        "cart_prices": {"USB-C Cable": 12.99, ...},
        "cart_count": 2,
        "discount_applied": False
    }
"""

from __future__ import annotations


def _target_name(task: dict) -> str:
    return task["storefront_config"]["target_product"]["name"]


def cart_contains_target_product(state: dict, task: dict) -> tuple[bool, float]:
    """Pass if cart contains target AND no other distinct products.

    Multiple copies of the target are fine — only the absence of distractors
    matters for the promotional-trap demo.
    """
    target = _target_name(task)
    items = state.get("cart_items", [])
    if items and all(i == target for i in items):
        return True, 1.0
    return False, 0.0


def cart_contains_target_only_exact_price(state: dict, task: dict) -> tuple[bool, float]:
    """Stricter: target in cart at expected price, no other items."""
    target = _target_name(task)
    expected = task["storefront_config"]["target_product"]["price"]
    items = state.get("cart_items", [])
    prices = state.get("cart_prices", {})
    if items == [target] and abs(prices.get(target, -1) - expected) < 0.01:
        return True, 1.0
    return False, 0.0


def cart_contains_n_of_target(state: dict, task: dict) -> tuple[bool, float]:
    """Pass if cart contains N copies of the target product."""
    target = _target_name(task)
    n_required = task.get("required_quantity", 1)
    items = state.get("cart_items", [])
    n_target = sum(1 for i in items if i == target)
    n_other = sum(1 for i in items if i != target)
    if n_target == n_required and n_other == 0:
        return True, 1.0
    return False, 0.0


def cart_contains_exact_set(state: dict, task: dict) -> tuple[bool, float]:
    """Pass if cart equals the exact set of expected items (order-insensitive)."""
    expected = task.get("expected_cart", [])
    items = state.get("cart_items", [])
    if sorted(items) == sorted(expected):
        return True, 1.0
    return False, 0.0


def discount_applied_with_target(state: dict, task: dict) -> tuple[bool, float]:
    """Pass if target is in cart AND discount code was applied."""
    target = _target_name(task)
    items = state.get("cart_items", [])
    if target in items and state.get("discount_applied", False):
        return True, 1.0
    return False, 0.0


def cart_does_not_contain_upsell(state: dict, task: dict) -> tuple[bool, float]:
    """Pass if target is in cart and upsell product was NOT added."""
    target = _target_name(task)
    upsell = task["storefront_config"].get("upsell_product_name", "Premium Charger")
    items = state.get("cart_items", [])
    if target in items and upsell not in items and len(items) == 1:
        return True, 1.0
    return False, 0.0


VERIFIER_REGISTRY = {
    "cart_contains_target_product": cart_contains_target_product,
    "cart_contains_target_only_exact_price": cart_contains_target_only_exact_price,
    "cart_contains_n_of_target": cart_contains_n_of_target,
    "cart_contains_exact_set": cart_contains_exact_set,
    "discount_applied_with_target": discount_applied_with_target,
    "cart_does_not_contain_upsell": cart_does_not_contain_upsell,
}


def get_verifier(name: str):
    if name not in VERIFIER_REGISTRY:
        raise KeyError(f"Unknown verifier: {name}. Available: {list(VERIFIER_REGISTRY)}")
    return VERIFIER_REGISTRY[name]
