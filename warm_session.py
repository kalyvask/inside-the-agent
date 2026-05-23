"""
warm_session.py — one-shot pre-demo utility to save a Playwright storage_state
for a real public website.

Use case: Walmart (and many other large retailers) serve a PRESS-AND-HOLD bot
challenge on first headless-Chromium visit. The agent can't solve it and we
won't try to bypass it. Instead: open a headed browser as the human, click
through whatever challenge appears, then save the resulting cookies +
localStorage to a JSON file. The agent loads that storage_state and the site
trusts the session.

Windows-specific note: Playwright's bundled Chromium hits a `spawn UNKNOWN`
launch bug on some Windows + AppData layouts. We work around it by passing
`channel="chrome"` so Playwright launches the user's installed Chrome instead.
If that also fails on your machine, see "CDP attach mode" at the bottom of
this docstring.

Usage:
    python warm_session.py --url https://www.walmart.com/search?q=usb+c+cable \\
        --out data/walmart_storage_state.json

A real Chrome window opens. Solve the CAPTCHA / cookie banner. When done,
create a sentinel file at the path printed in the terminal — or, in this
project's setup, ask the AI assistant to run `type nul > <sentinel>` from
its terminal. The script polls for the sentinel, then saves the storage_state
and exits.

CDP attach mode (manual fallback if `channel="chrome"` still fails):
    1. Close all running Chrome windows.
    2. Run:
         "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ^
            --remote-debugging-port=9222 ^
            --user-data-dir="%TEMP%\\chrome-warm-profile"
    3. In that Chrome, navigate to the demo URL and clear the CAPTCHA.
    4. Run `python warm_session.py --cdp-url http://localhost:9222 --out ...`
       — the script attaches, grabs cookies, saves, and exits.
"""

from __future__ import annotations

import asyncio
import functools
import sys
import tempfile
import time
from pathlib import Path

import typer

# Force unbuffered output so background-task viewers see progress immediately.
print = functools.partial(print, flush=True)  # type: ignore[assignment]

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from playwright.sync_api import sync_playwright


app = typer.Typer(add_completion=False)


def _poll_sentinel(sentinel: Path, timeout_seconds: int) -> bool:
    """Block until the sentinel file appears or timeout elapses.

    Returns True if the sentinel showed up, False on timeout. The sentinel
    file is deleted on success so a stale one from a previous run can't
    auto-trigger this one.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if sentinel.exists():
            try:
                sentinel.unlink()
            except Exception:
                pass
            return True
        time.sleep(0.5)
    return False


@app.command()
def main(
    url: str = typer.Option(None, help="URL to warm (launch mode). Skip if using --cdp-url."),
    out: str = typer.Option(..., help="Output storage_state JSON path"),
    cdp_url: str = typer.Option(
        None,
        help="If set, attach to an already-running Chrome via CDP at this URL "
             "(e.g. http://localhost:9222). Use this only if --channel chrome "
             "still fails to launch.",
    ),
    channel: str = typer.Option(
        "chrome",
        help="Playwright channel. 'chrome' uses your installed Chrome (recommended "
             "on Windows). 'chromium' uses the Playwright-bundled build (hits "
             "spawn UNKNOWN on some Windows machines).",
    ),
    viewport_width: int = typer.Option(1280),
    viewport_height: int = typer.Option(800),
    timeout_seconds: int = typer.Option(
        600,
        help="How long to wait for the sentinel file before giving up (10 min default).",
    ),
):
    """Warm cookies for a real website — solve the CAPTCHA once, save state for
    headless demo runs."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = out_path.with_suffix(".proceed")
    # Clear any stale sentinel from a previous abandoned run.
    if sentinel.exists():
        try:
            sentinel.unlink()
        except Exception:
            pass

    print(f"[warm] storage_state target: {out_path}")
    print(f"[warm] sentinel path:        {sentinel}")
    print()

    with sync_playwright() as pw:
        if cdp_url:
            # CDP attach mode: connect to a Chrome the user already launched
            # with --remote-debugging-port. The user navigates and solves the
            # CAPTCHA in that Chrome themselves.
            print(f"[warm] Attaching via CDP at {cdp_url} ...")
            browser = pw.chromium.connect_over_cdp(cdp_url)
            if not browser.contexts:
                print("[warm] ERROR: no contexts on the attached Chrome.")
                return
            ctx = browser.contexts[0]
            print("[warm] Attached. Make sure the right page is open in Chrome,")
            print("[warm] solve any CAPTCHA, then signal ready by creating:")
            print(f"[warm]    {sentinel}")
        else:
            # Launch mode: open a dedicated Chrome instance using your
            # installed Chrome (channel="chrome") with an isolated temp
            # user_data_dir so we don't fight your normal Chrome session
            # for profile-lock ownership.
            if not url:
                print("[warm] ERROR: --url required unless --cdp-url given.")
                return
            user_data_dir = Path(tempfile.gettempdir()) / "playwright-warm-chrome"
            user_data_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[warm] Launching Chrome (channel={channel}) at {url}\n"
                f"[warm] using profile dir: {user_data_dir}"
            )
            try:
                ctx = pw.chromium.launch_persistent_context(
                    user_data_dir=str(user_data_dir),
                    headless=False,
                    channel=channel,
                    viewport={"width": viewport_width, "height": viewport_height},
                    locale="en-US",
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as e:
                print(f"[warm] launch failed: {e}")
                print("[warm] Fall back to CDP attach mode — see warm_session.py docstring.")
                return
            ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                print(f"[warm] page.goto warning: {e}")
            print("[warm] Chrome window should now be open in front of you.")
            print("[warm] Solve any CAPTCHA + cookie banner.")
            print("[warm] When done, signal ready by creating:")
            print(f"[warm]    {sentinel}")
            print(f"[warm] (will time out after {timeout_seconds} seconds)")
            # In persistent-context mode there is no separate `browser` object
            # we need to close at the end — ctx.close() shuts everything down.
            browser = None

        if not _poll_sentinel(sentinel, timeout_seconds):
            print("[warm] Timed out waiting for sentinel — no state saved.")
            if not cdp_url:
                try:
                    ctx.close()
                except Exception:
                    pass
            return

        ctx.storage_state(path=str(out_path))
        print(f"[warm] Saved storage_state to {out_path}")
        if not cdp_url:
            try:
                ctx.close()
            except Exception:
                pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
        print("[warm] Done — headless agent runs will now load these cookies.")


if __name__ == "__main__":
    app()
