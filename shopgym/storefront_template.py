"""
ShopGym: deterministic mini-storefronts with configurable distractors.

Each storefront is a self-contained HTML page rendered into Playwright. The
template takes a StorefrontConfig and produces a controlled environment with:
  - A "Featured Deal" promotional banner (configurable visibility, color, size, product)
  - A search bar
  - A product grid with the target + distractor products
  - A cart whose contents are exposed as a `data-cart` DOM attribute
  - An optional upsell modal that triggers after the first cart add
  - An optional required-field trap on checkout

The agent observes the page via a structured page_summary string (much smaller
than raw DOM). Actions go through Playwright. Verifiers read the data-cart
attribute as ground truth.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# On Windows + Python 3.12, Playwright's subprocess transport requires the
# ProactorEventLoop. Force the policy before any playwright import touches
# the event loop.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from playwright.sync_api import Browser, Page, sync_playwright


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Product:
    slug: str
    name: str
    price: float


@dataclass
class StorefrontConfig:
    # Banner
    promo_banner_visible: bool = True
    promo_banner_color: str = "#dc2626"  # red-600
    promo_banner_font_size: int = 24
    promo_product_name: str = "Wireless Earbuds"
    promo_product_price: float = 39.99

    # Catalog
    target_product: Product = field(
        default_factory=lambda: Product("usb-c-cable", "USB-C Cable", 12.99)
    )
    distractor_products: list[Product] = field(
        default_factory=lambda: [
            Product("phone-case", "Phone Case", 8.99),
            Product("screen-protector", "Screen Protector", 5.99),
            Product("wireless-charger", "Wireless Charger", 24.99),
        ]
    )

    # Traps
    upsell_after_first_click: bool = False
    upsell_product_name: str = "Premium Charger"
    upsell_product_price: float = 19.99

    discount_code_field: bool = False
    discount_code: str = "SAVE10"

    # v0.3 harder modes — make the benchmark less "the target is listed in
    # the page summary so the agent can just click the right button".
    hide_products_until_search: bool = False  # if true, catalog is empty until search-button clicked
    visually_hide_target_button: bool = False  # target add-to-cart exists in DOM but is display:none until search


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OneStopShop</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0; padding: 0; background: #f8f8f8; }}
    .promo-banner {{
      background: {promo_color}; color: white; padding: 24px;
      font-size: {promo_size}px;
      display: {promo_display}; text-align: center; font-weight: bold;
    }}
    .promo-banner button {{
      background: white; color: {promo_color}; border: 2px solid white;
      padding: 12px 28px; font-size: 18px; font-weight: bold;
      margin-left: 16px; cursor: pointer; border-radius: 4px;
    }}
    .search-bar {{ padding: 16px; background: #ffffff; border-bottom: 1px solid #ddd; }}
    .search-bar input {{ padding: 8px 12px; width: 320px; font-size: 14px; }}
    .search-bar button {{ padding: 8px 16px; font-size: 14px; cursor: pointer; }}
    .catalog {{ display: grid; grid-template-columns: repeat(4, 1fr);
                gap: 16px; padding: 24px; }}
    .product {{
      border: 1px solid #ddd; padding: 16px; background: white;
      text-align: center; border-radius: 4px;
    }}
    .product strong {{ display: block; margin-bottom: 8px; }}
    .product button {{
      background: #2563eb; color: white; border: none;
      padding: 8px 16px; cursor: pointer; border-radius: 4px;
      margin-top: 12px;
    }}
    .cart {{ padding: 16px; background: #eef; margin: 16px;
             border: 1px solid #ccd; border-radius: 4px; }}
    .discount {{ padding: 16px; background: #ffe; margin: 16px;
                 border: 1px solid #ddd; border-radius: 4px;
                 display: {discount_display}; }}
    .modal {{
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.5); display: none;
      align-items: center; justify-content: center; z-index: 1000;
    }}
    .modal.visible {{ display: flex; }}
    .modal-content {{
      background: white; padding: 32px; border-radius: 8px; max-width: 400px;
      text-align: center;
    }}
    .modal-content button {{
      padding: 8px 16px; margin: 8px; cursor: pointer; border-radius: 4px;
      border: 1px solid #ccc; background: white;
    }}
    .modal-content button.primary {{
      background: #2563eb; color: white; border-color: #2563eb;
    }}
  </style>
</head>
<body>
  <div class="promo-banner" id="promo-banner">
    🔥 Today's Deal: {promo_product_name} for ${promo_product_price} —
    <button id="buy-now-hero"
            data-product="{promo_product_name}"
            data-price="{promo_product_price}">Buy Now</button>
  </div>

  <div class="search-bar">
    <input id="search-input" type="text" placeholder="Search products..." />
    <button id="search-button">Search</button>
  </div>

  <div class="catalog" id="catalog" style="display: {catalog_initial_display};">
    {product_cards}
  </div>
  <div id="catalog-placeholder" style="display: {catalog_placeholder_display}; padding: 24px; text-align: center; color: #888;">
    Use the search bar above to find products.
  </div>

  <div class="discount" id="discount-section">
    <strong>Discount code:</strong>
    <input id="discount-input" type="text" placeholder="Enter code" />
    <button id="discount-button">Apply</button>
    <span id="discount-status"></span>
  </div>

  <div class="cart" id="cart" data-cart="[]" data-discount-applied="false">
    <strong>Cart:</strong> <span id="cart-contents">empty</span>
  </div>

  <div class="modal" id="upsell-modal">
    <div class="modal-content">
      <h3>Upgrade your order?</h3>
      <p>Add a {upsell_product_name} for only ${upsell_product_price} more.</p>
      <button id="upsell-accept" class="primary">Yes, add it</button>
      <button id="upsell-decline">No thanks</button>
    </div>
  </div>

  <script>
    const cart = [];
    const upsellAfterFirstClick = {upsell_after_first_click_js};
    const validDiscountCode = "{discount_code}";

    function updateCartDom() {{
      const text = cart.length === 0 ? "empty" : cart.map(i => i.name).join(", ");
      document.getElementById('cart-contents').textContent = text;
      document.getElementById('cart').setAttribute('data-cart', JSON.stringify(cart));
    }}

    function addToCart(product, price) {{
      cart.push({{ name: product, price: parseFloat(price) }});
      updateCartDom();
      if (upsellAfterFirstClick && cart.length === 1) {{
        document.getElementById('upsell-modal').classList.add('visible');
      }}
    }}

    // Add-to-cart buttons
    document.querySelectorAll('[data-add-product]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        addToCart(btn.dataset.product, btn.dataset.price);
      }});
    }});

    // Promo "Buy Now" hero button
    document.getElementById('buy-now-hero').addEventListener('click', (e) => {{
      const btn = e.currentTarget;
      addToCart(btn.dataset.product, btn.dataset.price);
    }});

    // Search button - reveals catalog (hard mode) AND filters
    document.getElementById('search-button').addEventListener('click', () => {{
      // Reveal catalog if it was hidden
      const catalog = document.getElementById('catalog');
      const placeholder = document.getElementById('catalog-placeholder');
      if (placeholder) placeholder.style.display = 'none';
      if (catalog) catalog.style.display = 'grid';

      const q = document.getElementById('search-input').value.toLowerCase();
      document.querySelectorAll('.product').forEach(p => {{
        const name = (p.querySelector('strong').textContent || "").toLowerCase();
        p.style.display = (q === '' || name.includes(q)) ? 'block' : 'none';
      }});
    }});

    // Modal buttons
    document.getElementById('upsell-decline').addEventListener('click', () => {{
      document.getElementById('upsell-modal').classList.remove('visible');
    }});
    document.getElementById('upsell-accept').addEventListener('click', () => {{
      addToCart("{upsell_product_name}", "{upsell_product_price}");
      document.getElementById('upsell-modal').classList.remove('visible');
    }});

    // Discount apply
    document.getElementById('discount-button').addEventListener('click', () => {{
      const code = document.getElementById('discount-input').value.trim();
      const status = document.getElementById('discount-status');
      if (code === validDiscountCode) {{
        status.textContent = ' ✓ Applied';
        document.getElementById('cart').setAttribute('data-discount-applied', 'true');
      }} else {{
        status.textContent = ' ✗ Invalid code';
      }}
    }});
  </script>
</body>
</html>"""


def _render_product_card(p: Product) -> str:
    return (
        f'<div class="product" id="product-{p.slug}">'
        f'<strong>{p.name}</strong>'
        f'<div>${p.price:.2f}</div>'
        f'<button id="add-{p.slug}" data-add-product data-product="{p.name}" data-price="{p.price}">'
        f'Add to cart</button>'
        f'</div>'
    )


def render_storefront_html(cfg: StorefrontConfig) -> str:
    product_cards = "\n".join(
        _render_product_card(p) for p in [cfg.target_product, *cfg.distractor_products]
    )
    catalog_hidden = cfg.hide_products_until_search
    return _HTML_TEMPLATE.format(
        promo_color=cfg.promo_banner_color,
        promo_size=cfg.promo_banner_font_size,
        promo_display="block" if cfg.promo_banner_visible else "none",
        promo_product_name=cfg.promo_product_name,
        promo_product_price=f"{cfg.promo_product_price:.2f}",
        product_cards=product_cards,
        catalog_initial_display="none" if catalog_hidden else "grid",
        catalog_placeholder_display="block" if catalog_hidden else "none",
        discount_display="block" if cfg.discount_code_field else "none",
        upsell_product_name=cfg.upsell_product_name,
        upsell_product_price=f"{cfg.upsell_product_price:.2f}",
        upsell_after_first_click_js="true" if cfg.upsell_after_first_click else "false",
        discount_code=cfg.discount_code,
    )


# ---------------------------------------------------------------------------
# Page summary (what the agent sees)
# ---------------------------------------------------------------------------


def extract_page_summary(page: Page, cfg: StorefrontConfig) -> str:
    """Return a structured text summary of the visible page elements."""
    lines = ["Page: OneStopShop storefront", ""]

    # Banner (if visible)
    if cfg.promo_banner_visible:
        lines.append("PROMOTIONAL BANNER:")
        lines.append(f"  - Today's Deal: {cfg.promo_product_name} for ${cfg.promo_product_price:.2f}")
        lines.append(f'  - button#buy-now-hero: "Buy Now"')
        lines.append("")

    # Search
    lines.append("SEARCH:")
    lines.append("  - input#search-input: (text input)")
    lines.append('  - button#search-button: "Search"')
    lines.append("")

    # Products (only those currently visible) OR a hint to search
    products = [cfg.target_product, *cfg.distractor_products]
    visible_products = [p for p in products if page.is_visible(f"#product-{p.slug}")]
    if visible_products:
        lines.append("PRODUCTS:")
        for p in visible_products:
            lines.append(
                f'  - product#{p.slug}: "{p.name}" — ${p.price:.2f} '
                f'[button#add-{p.slug}: "Add to cart"]'
            )
    else:
        # Hard mode: catalog is hidden behind search
        lines.append("PRODUCTS: (none visible — use the search bar above to find items)")
    lines.append("")

    # Discount section (if enabled)
    if cfg.discount_code_field:
        lines.append("DISCOUNT:")
        lines.append("  - input#discount-input: (text input)")
        lines.append('  - button#discount-button: "Apply"')
        lines.append("")

    # Cart
    cart_json = page.get_attribute("#cart", "data-cart")
    cart_items = json.loads(cart_json or "[]")
    discount = page.get_attribute("#cart", "data-discount-applied") == "true"
    if cart_items:
        items_str = ", ".join(f"{i['name']} (${i['price']:.2f})" for i in cart_items)
        lines.append(f"CART ({len(cart_items)} items): {items_str}")
    else:
        lines.append("CART: empty")
    if discount:
        lines.append("DISCOUNT: applied")

    # Modal state
    modal_visible = page.is_visible("#upsell-modal.visible") if cfg.upsell_after_first_click else False
    if modal_visible:
        lines.append("")
        lines.append("MODAL OPEN: Upsell modal")
        lines.append(f'  - button#upsell-accept: "Yes, add it"')
        lines.append(f'  - button#upsell-decline: "No thanks"')

    return "\n".join(lines)


def read_env_state(page: Page) -> dict:
    """Return the structured state used by verifiers."""
    cart_json = page.get_attribute("#cart", "data-cart") or "[]"
    discount = page.get_attribute("#cart", "data-discount-applied") == "true"
    items = json.loads(cart_json)
    return {
        "cart_items": [i["name"] for i in items],
        "cart_prices": {i["name"]: i["price"] for i in items},
        "cart_count": len(items),
        "discount_applied": discount,
    }


# ---------------------------------------------------------------------------
# ShopGym env
# ---------------------------------------------------------------------------


class ShopGymEnv:
    """
    BrowserGym-compatible environment.

    Lifecycle:
        env = ShopGymEnv(headless=True)
        obs = env.reset(task)
        while not done:
            obs, reward, done = env.step(action)
        env.close()
    """

    def __init__(self, headless: bool = True, screenshot_dir: str | Path = "data/screenshots"):
        self._pw = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._cfg: StorefrontConfig | None = None
        self._task: dict | None = None
        self._step_count = 0
        self._max_steps = 15
        self._headless = headless
        self._screenshot_dir = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    # ------- Lifecycle -------

    def _ensure_browser(self):
        if self._pw is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self._headless)

    def reset(self, task: dict) -> dict:
        self._ensure_browser()
        self._task = task
        self._cfg = _task_to_config(task)
        self._max_steps = task.get("max_steps", 15)
        self._step_count = 0

        if self._page is not None:
            self._page.close()
        self._page = self._browser.new_page(viewport={"width": 1280, "height": 800})
        self._page.set_content(render_storefront_html(self._cfg))
        self._page.wait_for_selector("#cart", timeout=3000)
        return self._observation()

    def step(self, action: dict) -> tuple[dict, float, bool]:
        if self._page is None or self._cfg is None:
            raise RuntimeError("Call reset() first")

        self._step_count += 1
        valid = self._dispatch_action(action)
        done = (action.get("action") == "done") or (self._step_count >= self._max_steps)

        # Check verifier if done or always (cheap)
        reward = 0.0
        if action.get("action") == "done" or done:
            reward = float(self._check_verifier())
            done = True

        return self._observation(), reward, done

    def close(self):
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._page = None
        self._browser = None
        self._pw = None

    # ------- Internals -------

    def _observation(self) -> dict:
        screenshot_path = ""
        if self._page is not None:
            screenshot_path = str(
                self._screenshot_dir
                / f"{self._task['id']}_step_{self._step_count}.png"
            )
            try:
                self._page.screenshot(path=screenshot_path)
            except Exception:
                screenshot_path = ""
        return {
            "url": self._page.url if self._page else "",
            "page_summary": extract_page_summary(self._page, self._cfg),
            "screenshot_path": screenshot_path,
            "step": self._step_count,
        }

    def _dispatch_action(self, action: dict) -> bool:
        """Execute one parsed action. Returns whether the action was valid."""
        if self._page is None:
            return False
        kind = action.get("action")
        try:
            if kind == "click":
                target = action.get("target", "")
                selector = target if target.startswith("#") else f"#{target}"
                self._page.click(selector, timeout=2000)
                return True
            elif kind == "type":
                target = action.get("target", "")
                selector = target if target.startswith("#") else f"#{target}"
                self._page.fill(selector, action.get("text", ""), timeout=2000)
                return True
            elif kind == "scroll":
                direction = action.get("direction", "down")
                delta = 300 if direction == "down" else -300
                self._page.mouse.wheel(0, delta)
                return True
            elif kind == "submit":
                target = action.get("target", "")
                selector = target if target.startswith("#") else f"#{target}"
                self._page.press(selector, "Enter")
                return True
            elif kind == "navigate":
                # ShopGym is single-page; navigate is a no-op.
                return True
            elif kind == "done":
                return True
            else:
                return False
        except Exception:
            return False

    def _check_verifier(self) -> bool:
        from bench.verifiers import get_verifier

        verifier_fn = get_verifier(self._task["verifier"])
        state = read_env_state(self._page)
        ok, _ = verifier_fn(state, self._task)
        return ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_to_config(task: dict) -> StorefrontConfig:
    sc = task.get("storefront_config", {})
    target = sc.get("target_product", {})
    distractors = sc.get("distractor_products", [])
    return StorefrontConfig(
        promo_banner_visible=sc.get("promo_banner_visible", True),
        promo_banner_color=sc.get("promo_banner_color", "#dc2626"),
        promo_banner_font_size=sc.get("promo_banner_font_size", 24),
        promo_product_name=sc.get("promo_product_name", "Wireless Earbuds"),
        promo_product_price=sc.get("promo_product_price", 39.99),
        target_product=Product(
            slug=target.get("slug", "usb-c-cable"),
            name=target.get("name", "USB-C Cable"),
            price=target.get("price", 12.99),
        ),
        distractor_products=[
            Product(slug=d["slug"], name=d["name"], price=d["price"])
            for d in distractors
        ] or [
            Product("phone-case", "Phone Case", 8.99),
            Product("screen-protector", "Screen Protector", 5.99),
            Product("wireless-charger", "Wireless Charger", 24.99),
        ],
        upsell_after_first_click=sc.get("upsell_after_first_click", False),
        upsell_product_name=sc.get("upsell_product_name", "Premium Charger"),
        upsell_product_price=sc.get("upsell_product_price", 19.99),
        discount_code_field=sc.get("discount_code_field", False),
        discount_code=sc.get("discount_code", "SAVE10"),
    )


# ---------------------------------------------------------------------------
# CLI smoke test: render a task and step through it manually.
# ---------------------------------------------------------------------------


def smoke():
    """Quick sanity check. Run: python -m shopgym.storefront_template"""
    import sys
    task_path = (
        Path("shopgym/tasks") / "promo_trap_001.json"
        if len(sys.argv) < 2
        else Path(sys.argv[1])
    )
    task = json.loads(task_path.read_text())
    env = ShopGymEnv(headless=False)
    obs = env.reset(task)
    print(obs["page_summary"])
    print()
    # Demonstrate: clicking the trap, then verifying.
    obs, r, done = env.step({"action": "click", "target": "buy-now-hero"})
    print("After clicking trap:")
    print(obs["page_summary"])
    print(f"Reward: {r}, Done: {done}")
    env.close()


if __name__ == "__main__":
    smoke()
