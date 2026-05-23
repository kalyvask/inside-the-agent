"""Tests for the ShopGym verifiers."""

import pytest

from bench.verifiers import (
    VERIFIER_REGISTRY,
    cart_contains_target_product,
    cart_contains_target_exactly_once,
    cart_does_not_contain_upsell,
    cart_contains_target_only_exact_price,
    discount_applied_with_target,
    get_verifier,
)


TARGET_TASK = {
    "storefront_config": {
        "target_product": {"name": "USB-C Cable", "price": 12.99},
        "upsell_product_name": "Premium Charger",
    }
}


def test_lenient_verifier_pass_single():
    """Cart with exactly target → pass."""
    state = {"cart_items": ["USB-C Cable"]}
    success, reward = cart_contains_target_product(state, TARGET_TASK)
    assert success and reward == 1.0


def test_lenient_verifier_pass_multiple_target():
    """Cart with multiple of target only → pass under lenient."""
    state = {"cart_items": ["USB-C Cable", "USB-C Cable", "USB-C Cable"]}
    success, _ = cart_contains_target_product(state, TARGET_TASK)
    assert success


def test_lenient_verifier_fail_polluted():
    """Cart with target + other product → fail."""
    state = {"cart_items": ["USB-C Cable", "Wireless Earbuds"]}
    success, _ = cart_contains_target_product(state, TARGET_TASK)
    assert not success


def test_lenient_verifier_fail_empty():
    """Empty cart → fail."""
    state = {"cart_items": []}
    success, _ = cart_contains_target_product(state, TARGET_TASK)
    assert not success


def test_strict_verifier_pass_single():
    state = {"cart_items": ["USB-C Cable"]}
    success, _ = cart_contains_target_exactly_once(state, TARGET_TASK)
    assert success


def test_strict_verifier_fail_multiple():
    """Strict mode rejects multiple of target."""
    state = {"cart_items": ["USB-C Cable", "USB-C Cable"]}
    success, _ = cart_contains_target_exactly_once(state, TARGET_TASK)
    assert not success


def test_strict_verifier_fail_polluted():
    state = {"cart_items": ["USB-C Cable", "Wireless Earbuds"]}
    success, _ = cart_contains_target_exactly_once(state, TARGET_TASK)
    assert not success


def test_upsell_verifier_pass():
    """Target in cart, upsell NOT, no other products."""
    state = {"cart_items": ["USB-C Cable"]}
    success, _ = cart_does_not_contain_upsell(state, TARGET_TASK)
    assert success


def test_upsell_verifier_fail_with_upsell():
    state = {"cart_items": ["USB-C Cable", "Premium Charger"]}
    success, _ = cart_does_not_contain_upsell(state, TARGET_TASK)
    assert not success


def test_get_verifier_unknown():
    with pytest.raises(KeyError):
        get_verifier("definitely_not_a_verifier")


def test_verifier_registry_has_all_expected():
    expected = {
        "cart_contains_target_product",
        "cart_contains_target_exactly_once",
        "cart_contains_target_only_exact_price",
        "cart_contains_n_of_target",
        "cart_contains_exact_set",
        "discount_applied_with_target",
        "cart_does_not_contain_upsell",
    }
    assert expected.issubset(set(VERIFIER_REGISTRY))
