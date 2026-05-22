"""
ShopGym: deterministic mini-storefronts for controlled benchmark.
"""

from .storefront_template import (
    ShopGymEnv,
    StorefrontConfig,
    Product,
    render_storefront_html,
    extract_page_summary,
    read_env_state,
)

__all__ = [
    "ShopGymEnv",
    "StorefrontConfig",
    "Product",
    "render_storefront_html",
    "extract_page_summary",
    "read_env_state",
]
