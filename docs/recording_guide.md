# Recording the Demo Video

Recipe for a polished ~2-3 minute demo. The on-screen segment is a **deterministic
replay** of a saved real-eBay trajectory — no Modal calls, no bot-detection
roulette, no network surprises mid-take.

## What you are recording

The canonical demo artifact is the protected replay file:

```
data/trajectories/ebay_demo_targeted.jsonl   (6 steps, real eBay Deals page)
```

It carries the two targeted SAE edits at step 0 (`f26737 -6`, `f23803 +6`), real
screenshots per step (`data/screenshots/ebay_demo_targeted_step_*.png`), and ends
with a neutral "task ended" verdict (real-web tasks are qualitative — no
automated verifier). The "demo" name is deliberate: live runs write
`{task_id}_seed_{n}_{policy}.jsonl`, so they can never overwrite this file.

## Prerequisites

- **OBS Studio** (or any screen recorder), 1080p/30fps, MP4
- Repo cloned, `npm install` done in `hud/`
- A microphone

## Phase 1 — bring up the stack (~3 min)

```bash
# Terminal A — WebSocket bridge (leave running)
python -m agent.ws_server

# Terminal B — HUD, PRODUCTION build (leave running)
# Note (Windows/OneDrive): use build+start, not `npm run dev` — the dev server's
# file watcher wedges under OneDrive. NEXT_PUBLIC_WS_URL is baked at BUILD time.
cd hud
NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run build
npm run start

# Browser: open http://localhost:3000 and hard-refresh (Ctrl+Shift+R)
```

## Phase 2 — fire the replay (one command per take)

```bash
# Clear any stale events, then replay the demo trajectory to the HUD.
curl -X POST http://localhost:8765/clear
curl -X POST http://localhost:8765/replay \
     -H "Content-Type: application/json" \
     -d '{"trajectory_path": "data/trajectories/ebay_demo_targeted.jsonl", "step_delay": 1.5}'
```

Or point-and-click: **▶ REPLAY SAVED** (top right of the HUD) → pick
`ebay_demo_targeted`. `step_delay` 1.5 reads well on video; 1.2 is snappier; the
full replay runs ~12s and the header ends on **DONE** with a "task ended" card.

**Do NOT click the green ▶ TARGETED / ▷ BASELINE buttons during a take.** Those
fire LIVE runs (Modal + real eBay): slow, nondeterministic, and pointless for a
recording. The HUD disables them while a run is live and ws_server rejects
concurrent runs (409) — but the clean habit is: replay only.

## What to capture per take

- **Step 0**: the two emerald steering bars in EFFECT SIZE + both `TARGETED`
  entries in the intervention log — this is the money shot.
- **Steps 1-5**: real eBay viewport updating, live SAE features streaming on the
  right, the trajectory panel filling one clean line per step.
- **End**: header flips to DONE; neutral "task ended" card (no fake ✓/✗ on
  real-web tasks — say so in the voiceover, it's a credibility point).

Then cut to the charts, full-screen:

- `artifacts/headline.png` — baseline 10% → controls flat → targeted 56.7% →
  prompt+targeted 75%
- `artifacts/cross_scale.png` — 8B → 8B+SAE → 8B+SAE+prompt → 70B ladder

## Talk track

The 60-second eBay talk track lives in [`docs/live_demo.md`](live_demo.md). The
beat structure that works for the video:

1. **Hook** (~10s): agents are black boxes; this one is legible and steerable.
2. **The bars** (~20s): SAE features = the model's own concepts, read live.
3. **The replay** (~45s): step-0 edits flash, agent searches "usb-c cable" past
   the deal traps; point at the executed/invalid flags as honest telemetry.
4. **The numbers** (~30s): headline chart; controls prove direction is causal.
5. **Close** (~10s): open weights, repo link.

## Optional: a baseline-contrast clip

There is no protected baseline eBay file yet. If you want the side-by-side:
run ONE live baseline capture (`--policy baseline`, one run at a time — never
two concurrently), then copy it to a protected name the same way:
`ebay_demo_baseline.jsonl` + renamed screenshots, and replay that.

## Common gotchas

| Symptom | Fix |
|---|---|
| HUD tab shows stale/jumbled steps from past runs | `POST /clear`, then hard-refresh the tab — the trajectory panel accumulates until cleared |
| Replay shows "Skipping unparseable line" warnings | The file was damaged by concurrent runs; re-capture, or use the protected `ebay_demo_*` file |
| Viewport black / "waiting for first action" | Nothing loaded yet — fire the replay; the panel only fills when events stream |
| `npm run dev` hangs at "Starting…" | OneDrive file watcher — use the production build+start flow above |
| Verdict shows FAILURE on a real-web replay | Old build — rebuild the HUD (fixed: real-web tasks render neutral "task ended") |

## Export

1080p H.264 ~8 Mbps. Also export a 60-90s "money shot" cut (replay + headline
chart only) as backup, plus stills of the step-0 steering flash and both charts.
