# Live real-website demo — runbook (v0.9-B)

This is the "live on a real popular website" segment for May 29. The ShopGym
83% vs 0% headline (see `demo_script.md`) is the canonical result; this segment
is about showing the audience the HUD streaming live SAE features on a site
they recognize.

## Demo target: **eBay** (US-recognized, headless-friendly, promo-rich)

**Why eBay won out over Walmart**:

- Walmart's PerimeterX bot detection fires on Playwright even with warmed
  storage_state cookies. The fingerprint is on Chrome headers + JS-execution
  patterns + mouse-movement-absence, not just session cookies. Result: HUD
  viewport showed Walmart's "Robot or human?" PRESS-AND-HOLD modal instead
  of the actual shopping page during the v0.9-A test run.
- eBay's homepage loads cleanly headlessly (1912-char page summary, 4 of 6
  agent actions actually execute on real DOM elements vs. Walmart's 1 of 6).
- US audience recognizes eBay instantly.
- The page has **8+ H2 promotional banners** — Memorial Day -20%, Carhartt
  -45%, Ralph Lauren -70%, "20% off for Memorial Day" — richer promo
  distraction signal than Walmart's homepage carousel.

`shopgym/tasks/real_ebay.json` is the canonical real-web task.

## Backup targets (in priority order)

1. **AliExpress** (`shopgym/tasks/real_aliexpress.json`) — proven to work
   headlessly, less US-recognized but solid demo.
2. **Walmart with warmed storage_state** (`shopgym/tasks/real_walmart.json`)
   — works ~50% of the time. Re-warm via
   `python warm_session.py --url https://www.walmart.com/search?q=usb+c+cable --out data/walmart_storage_state.json`
   no more than 1 hour before the demo. If you see the PRESS-AND-HOLD modal
   in the HUD viewport mid-demo, you've been bot-detected.
3. **Target / Best Buy** — both load headlessly per
   `notebooks/explore_demo_pages.py`, untested with the agent.

## Pre-demo setup (~5 min)

```bash
# Terminal A — long-lived ws_server (already running if you've been working
# in this repo this session)
python -m agent.ws_server

# Terminal B — long-lived HUD frontend
cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev

# Browser
open http://localhost:3000

# Terminal C — warm the brain
python -m verify.sae_smoke --quick
```

## Demo (audience-facing)

```bash
# Terminal C — fire the live targeted run on eBay
HUD_PUBLISH=1 OUTPUT_SUFFIX=ebay python -u -m bench.runner \
    --policy targeted \
    --tasks shopgym/tasks/real_ebay.json \
    --trials 1 --limit 1 \
    --pause 4.0 \
    --position-mode all
```

`HUD_PUBLISH=1` directs events to the running ws_server (instead of
spawning a new one). `OUTPUT_SUFFIX=ebay` keeps the result file out of the
main rerun's `targeted.jsonl`. `--pause 4.0` makes each step take 5-7s so
the audience can follow.

## Expected trajectory (verified May 22, 6 steps, ~35s end-to-end)

| Step | Action | Exec | Live HUD behaviour |
|---|---|---|---|
| 0 | `type #gh-ac "USB-C cable"` | ✅ | Effect-size strip shows 2 emerald bars (`f26737 -6`, `f23803 +6`); intervention log records both with `TARGETED` badges |
| 1 | `click #gh-search-btn` | ✅ | Viewport flips to eBay search results |
| 2 | `click rso` | ❌ | `executed=False` — invented selector. The new v0.8 `executed` field surfaces this honestly. |
| 3 | `click "Cable Length"` | ✅ | Filter sidebar interaction visible in viewport |
| 4 | `click "see all"` | ❌ | Another invented selector |
| 5 | `click "Brand"` | ✅ | Another real filter |

## Talk track (60 seconds during the run)

> "Same agent. Same brain server. Real eBay.
>
> [step 0] Step zero — the agent reads the page, the HUD's Effect Size strip lights up with our two targeted edits: suppress feature 26737 by six, amplify 23803 by six. Targeted policy, scope `all`, seed zero — that's the badge at the top.
>
> [step 1] Step one — the agent submits the search. You can see in the viewport: real eBay, real product results.
>
> [throughout] In the bottom-right, the intervention log keeps the audit. The features panel on the right streams live SAE activations on every step. Notice the features are DIFFERENT from what fires on ShopGym — we're outside the calibration distribution, and the HUD shows that honestly.
>
> Look at the `executed` flags in the trajectory log: two steps didn't land — the model emitted parseable JSON but our selector heuristics didn't find a match in eBay's DOM. That's exactly the kind of failure mode the v0.8 instrumentation surfaces — separating 'JSON parsed' from 'browser actually acted.'
>
> The eighty-three percent headline at the bottom of the slide is twenty-four trials of step-zero divergence on ShopGym. This live segment is the cockpit working on a real site you've all shopped on."

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| eBay layout changes day-of | We use heuristic selectors, robust to most changes. Worst case: pivot to AliExpress. |
| Modal cold start | Warm 5 min before via `python -m verify.sae_smoke --quick`. |
| HUD doesn't reconnect | Hard-refresh localhost:3000. The auto-reconnect logic (v0.7-E) retries every 1.5s. |
| Re-run mid-demo wants the same Modal | Stop the rerun, do the live demo, resume after. Or just let them coexist — Modal scales containers. |

## Why this matters for the demo narrative

The reviewer's framing: *"reproducible testbed for runtime feature
interventions in browser agents, with live telemetry and controllable
steering."* The live eBay segment shows:

1. The HUD cockpit working on a real site (not a templated storefront)
2. Live SAE feature streaming on actual e-commerce text
3. Step-0 steering visible in the Effect Size strip
4. The v0.8 `executed` field exposing the honest gap between
   "model parses valid JSON" and "browser actually executed it"

The 83% on ShopGym remains the headline. This is the "and here's it running
on the open web" beat.
