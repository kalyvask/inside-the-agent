"""
Quick survey across candidate demo pages.

Goal: find the page that best matches the ShopGym "single-page-with-clear-trap"
structure on a real site, without burning Modal compute. For each URL:
  - load via WebEnv
  - print page summary (HEADINGS, BUTTONS, PROMOS)
  - snapshot screenshot to data/screenshots/explore_<slug>.png
  - check for CAPTCHA-overlay text in the summary

Pages worth comparing:
  1. AliExpress homepage — works, dense with promo banners
  2. AliExpress search results for usb-c cable — has Sponsored badge tradition
  3. AliExpress deals page — promo-heavy
  4. Walmart homepage — only useful if storage_state is warmed
  5. Walmart search results for usb-c cable
  6. Best Buy deals page — alternative US-audience site
"""

from __future__ import annotations

from pathlib import Path

from shopgym import WebEnv


CANDIDATES = [
    # Known-working from v0.6
    {"slug": "aliexpress_home", "url": "https://www.aliexpress.com/"},

    # Walmart (only useful if storage_state warmed)
    {"slug": "walmart_home", "url": "https://www.walmart.com/",
     "storage_state": "data/walmart_storage_state.json"},
    {"slug": "walmart_search_usbc",
     "url": "https://www.walmart.com/search?q=usb+c+cable",
     "storage_state": "data/walmart_storage_state.json"},

    # New US-recognized retailers to survey (no storage state — pure
    # headless reachability test, looking for "loads cleanly + has clear
    # promo-banner distractor + has search bar near top of DOM").
    {"slug": "target_home", "url": "https://www.target.com/"},
    {"slug": "target_deals", "url": "https://www.target.com/c/top-deals/-/N-4xrm3"},
    {"slug": "bestbuy_home", "url": "https://www.bestbuy.com/"},
    {"slug": "ebay_home", "url": "https://www.ebay.com/"},
    {"slug": "ebay_deals", "url": "https://www.ebay.com/deals"},
    {"slug": "macys_home", "url": "https://www.macys.com/"},
    {"slug": "costco_home", "url": "https://www.costco.com/"},
    {"slug": "kohls_home", "url": "https://www.kohls.com/"},
    {"slug": "etsy_home", "url": "https://www.etsy.com/"},
]


def main():
    env = WebEnv(headless=True)
    Path("data/screenshots").mkdir(parents=True, exist_ok=True)

    print(f"{'slug':<28}  {'len(summary)':>12}  {'captcha?':>10}  url")
    print("-" * 100)

    for cand in CANDIDATES:
        task = {
            "id": f"explore_{cand['slug']}",
            "url": cand["url"],
            "max_steps": 1,
            "cookies_pre_accepted": False,
        }
        if "storage_state" in cand:
            task["storage_state"] = cand["storage_state"]

        try:
            obs = env.reset(task)
        except Exception as e:
            print(f"{cand['slug']:<28}  ERROR: {e}")
            continue

        summary = obs.get("page_summary", "")
        # Heuristic CAPTCHA / bot-wall detection
        captcha = any(
            tok in summary.lower()
            for tok in [
                "press & hold",
                "press and hold",
                "robot or human",
                "verify you are human",
                "are you a robot",
                "captcha",
                "blocked",
            ]
        )
        print(
            f"{cand['slug']:<28}  {len(summary):>12}  "
            f"{'YES' if captcha else 'no':>10}  {cand['url'][:50]}"
        )

        # Save the full summary alongside the screenshot
        Path(f"data/screenshots/explore_{cand['slug']}_summary.txt").write_text(
            summary, encoding="utf-8"
        )

    env.close()


if __name__ == "__main__":
    main()
