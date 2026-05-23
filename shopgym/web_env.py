"""
WebEnv: a generic Playwright-based environment for real public websites.

Same interface as ShopGymEnv (reset, step) so the agent loop doesn't change.
Differences:
  - Navigates to an arbitrary URL specified in the task config
  - Auto-dismisses common cookie banners on first load
  - Extracts a structured page summary from the real DOM (not a controlled template)
  - Heuristic action dispatch: maps the agent's JSON action to real CSS selectors

Designed to be pointed at AliExpress, Walmart, eBay, Best Buy, Etsy, etc.
Task config schema:
    {
      "id": "aliexpress_buy_phone_case",
      "instruction": "Buy a USB-C charging cable on this storefront.",
      "url": "https://www.aliexpress.com/",
      "verifier": "manual",          (or skip — the demo is about the agent's reasoning, not cart state)
      "cookies_pre_accepted": false,
      "max_steps": 8
    }
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from playwright.sync_api import Browser, Page, sync_playwright


# Common cookie-banner-accept patterns. Try these in order on first page load.
COOKIE_ACCEPT_PATTERNS = [
    'button:has-text("Accept all")',
    'button:has-text("Accept All")',
    'button:has-text("Accept Cookies")',
    'button:has-text("Allow all")',
    'button:has-text("Allow All")',
    'button:has-text("Accept")',
    'button:has-text("OK")',
    'button:has-text("I accept")',
    'button:has-text("Agree")',
    'button:has-text("Got it")',
    'button:has-text("Continue")',
    '[id*="cookie"] button:has-text("Accept")',
    '[class*="cookie"] button:has-text("Accept")',
    '[aria-label="Accept all"]',
    '[aria-label="Accept cookies"]',
]


@dataclass
class WebEnv:
    """Generic real-website environment."""

    headless: bool = True
    screenshot_dir: str | Path = "data/screenshots"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    viewport_width: int = 1280
    viewport_height: int = 800
    nav_timeout_ms: int = 30000
    page_max_chars: int = 6000  # truncate page summary to fit LLM context

    # Set on reset
    _pw: Any = None
    _browser: Browser | None = None
    _ctx: Any = None
    _page: Page | None = None
    _task: dict | None = None
    _step_count: int = 0
    _cookies_dismissed: bool = False

    def __post_init__(self):
        self._screenshot_dir = Path(self.screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_browser(self):
        if self._pw is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._ctx = self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                user_agent=self.user_agent,
                locale="en-US",
            )
            # Stealth touchups: hide webdriver flag from common detectors.
            self._ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )

    def reset(self, task: dict) -> dict:
        self._ensure_browser()
        self._task = task
        self._step_count = 0
        self._cookies_dismissed = bool(task.get("cookies_pre_accepted", False))

        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
        self._page = self._ctx.new_page()

        url = task["url"]
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=self.nav_timeout_ms)
        except Exception as e:
            return self._observation(extra_text=f"NAV ERROR: {e}")

        # Give the page a beat to settle (lazy-loaded content).
        time.sleep(1.0)

        # Try to dismiss the cookie banner once.
        if not self._cookies_dismissed:
            self._dismiss_cookies()

        return self._observation()

    def _dismiss_cookies(self):
        for selector in COOKIE_ACCEPT_PATTERNS:
            try:
                el = self._page.locator(selector).first
                if el.is_visible(timeout=500):
                    el.click(timeout=1500)
                    self._cookies_dismissed = True
                    time.sleep(0.5)
                    return
            except Exception:
                continue

    def step(self, action: dict) -> tuple[dict, float, bool]:
        self._step_count += 1
        if not self._page:
            return self._observation(extra_text="NO PAGE"), 0.0, True

        kind = action.get("action", "")
        ok = self._dispatch(kind, action)

        # If max_steps reached, end the episode. Verifier here is intentionally
        # weak — for real sites we use "did the agent do something coherent" as
        # the demo signal, not a strict cart check.
        max_steps = (self._task or {}).get("max_steps", 8)
        done = self._step_count >= max_steps or kind == "done"
        reward = 0.0
        return self._observation(extra_text="" if ok else f"INVALID ACTION: {kind}"), reward, done

    def _dispatch(self, kind: str, action: dict) -> bool:
        try:
            if kind == "click":
                target = action.get("target", "")
                return self._click(target)
            if kind == "type":
                target = action.get("target", "")
                text = action.get("text", "")
                return self._type(target, text)
            if kind == "navigate":
                url = action.get("url", "")
                self._page.goto(url, wait_until="domcontentloaded", timeout=self.nav_timeout_ms)
                return True
            if kind == "scroll":
                direction = action.get("direction", "down")
                delta = 600 if direction == "down" else -600
                self._page.evaluate(f"window.scrollBy(0, {delta})")
                return True
            if kind == "done":
                return True
        except Exception:
            return False
        return False

    def _click(self, target: str) -> bool:
        """Try several selector strategies. Real-site selectors are flaky."""
        candidates = [
            target,                                # raw
            f"#{target}" if not target.startswith("#") else target,
            f'[data-test="{target}"]',
            f'[data-testid="{target}"]',
            f'[aria-label="{target}"]',
            f'text="{target}"',
            f'button:has-text("{target}")',
            f'a:has-text("{target}")',
            f'[role="button"]:has-text("{target}")',
        ]
        for sel in candidates:
            try:
                el = self._page.locator(sel).first
                if el.is_visible(timeout=500):
                    el.click(timeout=2000)
                    time.sleep(0.6)
                    return True
            except Exception:
                continue
        return False

    def _type(self, target: str, text: str) -> bool:
        candidates = [
            target,
            f"#{target}" if not target.startswith("#") else target,
            f'[data-test="{target}"]',
            f'[name="{target}"]',
            f'[placeholder*="{target}"]',
            f'input[aria-label*="{target}"]',
            f'input[type="search"]',
            f'input[type="text"]',
        ]
        for sel in candidates:
            try:
                el = self._page.locator(sel).first
                if el.is_visible(timeout=500):
                    el.fill(text, timeout=2000)
                    time.sleep(0.4)
                    return True
            except Exception:
                continue
        return False

    def _observation(self, extra_text: str = "") -> dict:
        page_summary = self._build_page_summary()
        if extra_text:
            page_summary = page_summary + "\n\n" + extra_text

        screenshot_path = None
        if self._page:
            try:
                screenshot_path = str(
                    self._screenshot_dir
                    / f"{self._task['id']}_step_{self._step_count}.png"
                )
                self._page.screenshot(path=screenshot_path, full_page=False)
            except Exception:
                screenshot_path = None

        return {
            "url": self._page.url if self._page else "",
            "page_summary": page_summary,
            "screenshot_path": screenshot_path or "",
        }

    def _build_page_summary(self) -> str:
        """Walk the real DOM and produce a compact text summary the LLM can use."""
        if not self._page:
            return "(no page)"
        try:
            title = self._page.title()
            url = self._page.url
        except Exception:
            return "(page error)"

        lines = [f"PAGE: {title}", f"URL: {url}", ""]

        # Visible H1/H2/H3 headings
        try:
            headings = self._page.eval_on_selector_all(
                "h1, h2, h3",
                """els => els
                    .filter(e => e.offsetParent !== null)
                    .slice(0, 8)
                    .map(e => e.tagName + ': ' + (e.innerText || '').trim().slice(0, 80))""",
            )
            if headings:
                lines.append("HEADINGS:")
                for h in headings:
                    if h:
                        lines.append(f"  - {h}")
                lines.append("")
        except Exception:
            pass

        # Visible buttons
        try:
            buttons = self._page.eval_on_selector_all(
                "button, [role=button], a.button, input[type=submit]",
                """els => els
                    .filter(e => e.offsetParent !== null && (e.innerText || e.value || e.getAttribute('aria-label') || '').trim())
                    .slice(0, 20)
                    .map(e => {
                        const id = e.id || e.getAttribute('data-test') || e.getAttribute('data-testid') || '';
                        const txt = (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 60);
                        return JSON.stringify({id: id, text: txt});
                    })""",
            )
            if buttons:
                lines.append("BUTTONS (visible):")
                for raw in buttons:
                    try:
                        import json
                        b = json.loads(raw)
                        sel = f"#{b['id']}" if b['id'] else f'"{b["text"]}"'
                        lines.append(f'  - {sel}: "{b["text"]}"')
                    except Exception:
                        pass
                lines.append("")
        except Exception:
            pass

        # Visible text inputs
        try:
            inputs = self._page.eval_on_selector_all(
                "input[type=text], input[type=search], input:not([type]), textarea",
                """els => els
                    .filter(e => e.offsetParent !== null)
                    .slice(0, 8)
                    .map(e => JSON.stringify({
                        id: e.id || e.name || e.getAttribute('data-test') || '',
                        placeholder: (e.placeholder || e.getAttribute('aria-label') || '').slice(0, 60),
                        type: e.type || 'text'
                    }))""",
            )
            if inputs:
                lines.append("INPUTS (visible):")
                for raw in inputs:
                    try:
                        import json
                        i = json.loads(raw)
                        sel = f"#{i['id']}" if i['id'] else "(unnamed input)"
                        lines.append(f'  - {sel}: placeholder="{i["placeholder"]}", type={i["type"]}')
                    except Exception:
                        pass
                lines.append("")
        except Exception:
            pass

        # Top promotional / featured items — heuristic by class names
        try:
            promos = self._page.eval_on_selector_all(
                "[class*='deal' i], [class*='promo' i], [class*='banner' i], [class*='featured' i]",
                """els => els
                    .filter(e => e.offsetParent !== null && (e.innerText || '').trim().length > 5 && (e.innerText || '').length < 200)
                    .slice(0, 5)
                    .map(e => (e.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 80))""",
            )
            if promos:
                lines.append("PROMOTIONAL CONTENT (heuristic — may overlap):")
                for p in promos:
                    if p:
                        lines.append(f"  - {p}")
                lines.append("")
        except Exception:
            pass

        out = "\n".join(lines)
        if len(out) > self.page_max_chars:
            out = out[: self.page_max_chars] + "\n[truncated]"
        return out

    def close(self):
        try:
            if self._page:
                self._page.close()
            if self._ctx:
                self._ctx.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
