"""
warm_session.py — one-shot pre-demo utility to save a Playwright storage_state
for a real public website.

Use case: Walmart (and many other large retailers) serve a PRESS-AND-HOLD bot
challenge on first headless-Chromium visit. The agent can't solve it and we
won't try to bypass it. Instead: open a headed browser as the human, click
through whatever challenge appears, then save the resulting cookies +
localStorage to a JSON file. The agent loads that storage_state and Walmart
trusts the session.

Usage:
    python warm_session.py --url https://www.walmart.com/ \\
        --out data/walmart_storage_state.json

    # Then point the task at it:
    #   "url": "https://www.walmart.com/",
    #   "storage_state": "data/walmart_storage_state.json"

This is the same pattern Playwright's docs recommend for auth state. The
audience never sees the warm-up — it happens before the demo starts, like
plugging in a laptop.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from playwright.sync_api import sync_playwright


app = typer.Typer(add_completion=False)


@app.command()
def main(
    url: str = typer.Option(..., help="URL to warm — e.g. https://www.walmart.com/"),
    out: str = typer.Option(..., help="Output storage_state JSON path"),
    viewport_width: int = typer.Option(1280),
    viewport_height: int = typer.Option(800),
):
    """Open a headed browser, let the user solve any bot challenge and accept
    cookies, then save storage_state for later headless runs."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[warm] Opening headed Chromium at {url}")
    print("[warm] Solve any CAPTCHA / cookie banner, browse around if needed.")
    print("[warm] When ready, return to THIS terminal and press Enter to save.")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded")

        try:
            input(">>> Press Enter here once you've cleared CAPTCHAs / cookies... ")
        except KeyboardInterrupt:
            print("[warm] aborted — no state saved.")
            ctx.close()
            browser.close()
            return

        ctx.storage_state(path=str(out_path))
        print(f"[warm] Saved storage state to {out_path}")
        print(f"[warm] Headless agent runs will now load these cookies.")
        ctx.close()
        browser.close()


if __name__ == "__main__":
    app()
