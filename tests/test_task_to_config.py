"""
Regression tests for shopgym._task_to_config.

v0.7-A motivation: a reviewer caught that _task_to_config silently dropped
hide_products_until_search and dom_noise_buttons from task JSON, meaning the
'hard_held_out' benchmark rendered identical to easy tasks. These tests
snapshot every public StorefrontConfig field so that a future addition that
forgets to thread a field through fails loudly.
"""

from __future__ import annotations

import dataclasses

import pytest

from shopgym.storefront_template import StorefrontConfig, _task_to_config


def test_every_storefrontconfig_field_is_threaded_through():
    """Build a task JSON that overrides EVERY field on StorefrontConfig, then
    verify _task_to_config carries each override into the returned dataclass.

    If you add a new field to StorefrontConfig, this test will fail with a
    clear message until you also update _task_to_config to read it.
    """
    sc_overrides = {
        "promo_banner_visible": False,
        "promo_banner_color": "#00ff00",
        "promo_banner_font_size": 99,
        "promo_product_name": "TEST_PROMO",
        "promo_product_price": 1234.56,
        "target_product": {
            "slug": "test-target-slug",
            "name": "TEST_TARGET",
            "price": 7.77,
        },
        "distractor_products": [
            {"slug": "d1", "name": "Distractor 1", "price": 1.11},
            {"slug": "d2", "name": "Distractor 2", "price": 2.22},
        ],
        "upsell_after_first_click": True,
        "upsell_product_name": "TEST_UPSELL",
        "upsell_product_price": 9.99,
        "discount_code_field": True,
        "discount_code": "TEST10",
        "hide_products_until_search": True,
        "visually_hide_target_button": True,
        "dom_noise_buttons": 4,
        "dom_noise_button_labels": ["A", "B", "C", "D"],
    }
    cfg = _task_to_config({"storefront_config": sc_overrides})

    # Spot-check the previously-broken P0 fields by name.
    assert cfg.hide_products_until_search is True, \
        "hide_products_until_search must flow from task JSON into StorefrontConfig"
    assert cfg.dom_noise_buttons == 4, \
        "dom_noise_buttons must flow from task JSON into StorefrontConfig"
    assert cfg.dom_noise_button_labels == ["A", "B", "C", "D"]

    # Snapshot pass: every overridable field on the dataclass must reflect
    # the override OR be in this allow-list of fields the test JSON doesn't
    # try to set (they have safe defaults).
    fields_not_set_by_test = set()  # the test sets everything currently
    for f in dataclasses.fields(cfg):
        if f.name in fields_not_set_by_test:
            continue
        if f.name == "target_product":
            assert cfg.target_product.slug == "test-target-slug"
            assert cfg.target_product.name == "TEST_TARGET"
            assert cfg.target_product.price == 7.77
            continue
        if f.name == "distractor_products":
            assert len(cfg.distractor_products) == 2
            assert cfg.distractor_products[0].slug == "d1"
            continue
        # All other fields: scalar comparison against the override.
        expected = sc_overrides.get(f.name)
        if expected is None:
            # The override didn't include this field — either it's missing
            # from sc_overrides (test bug) or it isn't task-configurable.
            pytest.fail(
                f"StorefrontConfig field {f.name!r} is not exercised by "
                f"this test. Add it to sc_overrides above so we catch "
                f"silent-drop regressions like the P0-3 bug."
            )
        actual = getattr(cfg, f.name)
        assert actual == expected, \
            f"_task_to_config dropped field {f.name!r}: got {actual!r}, " \
            f"expected {expected!r}"


def test_empty_task_falls_back_to_safe_defaults():
    """A task with no storefront_config block must still produce a valid
    StorefrontConfig with the canonical ShopGym defaults."""
    cfg = _task_to_config({})
    assert cfg.promo_banner_visible is True
    assert cfg.hide_products_until_search is False
    assert cfg.dom_noise_buttons == 0
    assert cfg.target_product.slug == "usb-c-cable"
    assert len(cfg.distractor_products) == 3


def test_hard_held_out_task_actually_enables_hard_mode():
    """The regression that motivated v0.7-A: hard_held_out.json sets
    hide_products_until_search=true; _task_to_config must propagate it."""
    hard_task = {
        "id": "hard_promo_001",
        "storefront_config": {
            "hide_products_until_search": True,
        },
    }
    cfg = _task_to_config(hard_task)
    assert cfg.hide_products_until_search is True
