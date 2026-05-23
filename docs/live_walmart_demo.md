# Live Walmart demo — runbook

This is the "live on a real popular website" bonus segment for May 29. The ShopGym
83% vs 0% headline (see `demo_script.md`) is the canonical result; this segment
is about showing the audience the HUD streaming live SAE features on a site
they recognize.

## Why Walmart

- US audience recognizes the homepage instantly.
- "Today's Deals" carousel is the natural-habitat distractor — same shape as
  the ShopGym promo trap, just with different vocabulary.
- The honest twist: our ShopGym-tuned features (f26737, f23803) don't appear
  in the top activations on Walmart's homepage at all (see
  `real_world_generalization.md`). The HUD lets the audience watch this
  generalization gap in real time.

## Pre-demo setup (do BEFORE the audience walks in)

### 1. Warm Walmart cookies (one-time per machine, expires after ~24h)

```bash
python warm_session.py --url https://www.walmart.com/ \
    --out data/walmart_storage_state.json
```

A headed Chrome window opens. Click through the PRESS-AND-HOLD bot challenge
and any cookie banner. Browse around for ~10 seconds so a session cookie sets.
Return to the terminal and press Enter. State is saved.

### 2. Warm the Modal brain-server

```bash
python -m verify.sae_smoke --quick
```

The first call cold-starts the container (~30s on L40S). Re-run once to confirm
warm-state latency is <3s per call. Container stays alive ~10 min after last call.

### 3. Start the HUD frontend (terminal A — leave running)

```bash
cd hud
NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev
```

Open http://localhost:3000 in the browser you'll project. Confirm the panels
render (they'll show empty until the agent runs).

## Demo (audience-facing)

### 4. Run the agent with HUD on Walmart (terminal B)

```bash
python -m bench.runner \
    --policy targeted \
    --tasks shopgym/tasks/real_walmart.json \
    --trials 1 --limit 1 \
    --hud --pause 4.0 \
    --position-mode all
```

- `--hud` starts the ws_server subprocess and emits events to the Next.js HUD.
- `--pause 4.0` makes each step take 5–7s total so the audience can follow.
- The agent loads the storage_state, skips the CAPTCHA, lands on the actual
  homepage.

### 5. What the audience sees

| Panel | Content |
|---|---|
| Viewport | Live screenshots of `walmart.com` — homepage, then search results |
| Top features | f44602, f39820, f19079 firing (Walmart's lexical features, not ours) |
| Steering applied | At step 0: f26737 −6, f23803 +6 (the targeted edits — visible but ineffective on this distribution, which IS the honest narrative) |
| Intervention timeline | Time-stamped log of every steering event |
| Verdict | Updates as the agent navigates |

## Talk track (60 seconds during the run)

> "This is the same agent, same brain-server, same SAE. On our controlled
> ShopGym storefront, suppressing feature 26737 and boosting 23803 takes the
> agent from 0% to 83% on the held-out benchmark. Watch what happens on real
> Walmart.
>
> [step 0 fires] Steering applied — you can see it in the timeline. But look
> at the top features panel. Our targeted features aren't even in Walmart's
> top eight. The agent types the right query, then gets pulled into the
> Departments carousel.
>
> This is exactly what you'd expect: SAE features encode lexical patterns,
> not abstract concepts. The features tuned on 'HOT DEAL' don't fire on
> 'Today's Deals' or 'Departments'. The HUD lets you watch that mismatch
> happen in real time. The fix isn't more steering force — it's repeating
> the feature-discovery process on this distribution."

## Risks

| Risk | Mitigation |
|---|---|
| Walmart bot-detects mid-demo | Storage state expires after ~24h. Re-warm same morning. Have a pre-recorded backup clip in `data/demo_recordings/walmart_*.mp4`. |
| Walmart layout changes the day before | Run a full smoke pass the morning of. Walmart redesigns happen ~quarterly so day-of breakage is unlikely. |
| Modal cold-start | Warm in step 2. Container stays alive for ~10 min after last call. |
| Search input selector changes | The agent uses heuristic selectors (8 strategies) — robust to one or two changing. |
| Captcha re-appears despite storage_state | Switch the task config to `shopgym/tasks/real_aliexpress.json` — that site loaded cleanly in v0.6-A smoke tests. |

## Fallback site order if Walmart breaks the morning of

1. AliExpress (`shopgym/tasks/real_aliexpress.json`) — confirmed working v0.6-A.
2. Backup: re-record the Walmart run to MP4 in advance, embed in slide 5.

## Files involved

- `warm_session.py` — one-shot cookie warm-up utility.
- `shopgym/web_env.py` — generic real-website env, accepts `storage_state`.
- `shopgym/tasks/real_walmart.json` — task config pointing at the storage_state.
- `bench/runner.py` — `--hud` flag auto-starts ws_server.
- `agent/ws_server.py` — WebSocket bridge + `/screenshots` static mount.
- `hud/` — Next.js frontend.
