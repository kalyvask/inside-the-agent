"""
ShopGym: deterministic mini-storefronts for controlled benchmark.

Day 2 work: build storefront_template.py + 10 calibration tasks + 20 held-out.
"""

from .storefront_template import StorefrontTemplate, ShopGymEnv

__all__ = ["StorefrontTemplate", "ShopGymEnv"]
