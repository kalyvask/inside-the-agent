# Recording the Demo Video

Step-by-step recipe for producing a polished 3-minute demo video. Uses pre-recorded trajectories — no live Modal calls during recording.

## Prerequisites

- **OBS Studio** (free, https://obsproject.com/) or any screen recorder
- **DaVinci Resolve** (free) or iMovie for editing
- Your repo cloned and `npm install` done in `hud/`
- A microphone (laptop built-in is fine for class)

## Phase 1 — Prepare the trajectories (~5 min)

You already have trajectories from the Day 6 benchmark in `data/trajectories/`. Pick the most dramatic baseline failure and the cleanest targeted success:

```bash
# Inspect available trajectories
ls data/trajectories/

# Recommended picks:
#   data/trajectories/promo_held_001_seed_0_baseline.jsonl  (baseline fail)
#   data/trajectories/promo_held_001_seed_0_targeted.jsonl  (targeted win)
```

If you want fresh ones with different products, rerun the runner on a single task.

## Phase 2 — Set up the recording scene (~10 min)

### Window layout

Open these in this order for clean alt-tab:

1. **Slide 1 of your deck** (title slide) — Keynote/Google Slides full-screen
2. **OneStopShop screenshot or live view** — open one of the storefronts in Chrome
   - Easiest: `python -m shopgym promo_cal_001` runs Playwright in headed mode and shows the storefront
3. **HUD in a Chrome tab** — `http://localhost:3000`
4. **Terminal for kicking off replays** — split into 2 (HUD server + replayer)
5. **Headline chart** — open `data/results/headline.png` full-screen

### OBS scenes

- **Scene "Storefront-only"**: Window source = the OneStopShop browser tab
- **Scene "Split HUD"**: Window source = HUD tab (3/4 width) + storefront screenshot or browser (1/4)
- **Scene "Chart"**: Image source = `headline.png`
- **Scene "Slide"**: Window source = your slide presentation

Recommended record settings: **1080p, 30fps, MP4 container** (universal).

## Phase 3 — Record each segment (~15 min)

Work in **6 short clips**, then stitch in post. Each is ~30 seconds.

### Clip 1: Title (0:00–0:20)

- Scene: Slide
- Voice: *"Today's AI agents are black boxes. We made one transparent."*
- Cut to next when title fades.

### Clip 2: The trap (0:20–0:50)

- Scene: Storefront-only
- Use OBS zoom/cursor effects to highlight the red "Buy Now" hero button
- Voice: *"Goal — buy a USB-C cable. But this page has a bright red 'Today's Deal' button for wireless earbuds. That's the trap. Watch what happens."*

### Clip 3: Baseline plays out (0:50–1:30)

In one terminal:
```bash
python -m agent.ws_server
```
In another:
```bash
cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev
```
Open the HUD at `http://localhost:3000`. Then in a third terminal:
```bash
python -m verify.replay_trajectory \\
    data/trajectories/promo_held_001_seed_0_baseline.jsonl --slow
```

- Scene: Split HUD
- Voice: *"Baseline policy. Step 0 — the agent clicks the trap. Cart shows wireless earbuds. The verifier needs just the cable. Polluted. Failure."*
- The Verdict overlay will pop the red ✗ FAILURE card at the end. Capture that, then cut.

### Clip 4: Targeted succeeds (1:30–2:15)

Same terminals. Restart the HUD (refresh the browser tab to clear state) and replay the targeted trajectory:
```bash
python -m verify.replay_trajectory \\
    data/trajectories/promo_held_001_seed_0_targeted.jsonl --slow
```

- Scene: Split HUD
- Voice: *"Same task. Two SAE feature edits at Step 0. Watch the yellow steering flash — hallucination feature suppressed by 6, goal_tracking amplified by 6. The agent's first action: type 'USB-C cable' in search. Step 1 — clicks the right add-to-cart. Steps 2 onward — zero steering. Success."*
- The Verdict overlay will pop the green ✓ SUCCESS card. Capture that, then cut.

### Clip 5: Headline chart (2:15–2:45)

- Scene: Chart
- Voice: *"24 trials per policy. Baseline 0. Wrong-sign 4 — flipping the steering direction drops it back to baseline's CI. Random 46 — random perturbations help sometimes. Targeted 83. The wrong-sign result is the smoking gun: direction matters causally."*

### Clip 6: Close (2:45–3:00)

- Scene: Slide (final slide with repo link)
- Voice: *"Today's agents fail mysteriously. Ours fails legibly. Two SAE feature edits at one decision step take success from 0 to 83 on 24 held-out trials. Llama 3.1-8B, Goodfire SAE — both open weights. Repo at github dot com slash kalyvask slash inside-the-agent."*

## Phase 4 — Edit (~30 min)

In your video editor:

1. Drop the 6 clips on the timeline in order
2. Trim dead air (use the audio waveform to find pauses)
3. Add a 0.2s crossfade between clips
4. Add lower-third captions for the headline numbers (`-83 pts`, `+46 pts`, etc.)
5. Add a soft music bed (royalty-free; Epidemic Sound or Pixabay)
6. Export at **1080p, H.264, 8 Mbps** — about 200MB

## Phase 5 — Backup

After your master cut, also export:
- A 90-second "money shot" version (just clips 2-4 + verdict)
- A still image of the HUD with the steering flash visible
- A still image of the headline chart

These are insurance in case the live demo fragments and you need to dive to a backup.

## Common gotchas

| Symptom | Fix |
|---|---|
| HUD shows mock data instead of real | Refresh the browser tab; `NEXT_PUBLIC_WS_URL` must be set when `npm run dev` is invoked, not after |
| Verdict overlay doesn't appear | Check the trajectory's last step has `result.reward` set; the replayer derives `success` from that |
| Bars don't animate smoothly | OBS may be capturing at <30fps; bump to 60fps if you have the CPU |
| Audio is too quiet | OBS → Audio Mixer → click the gear → Filter → add Gain → +6 dB |
| File is too big | Re-encode at H.264 6 Mbps; quality difference is invisible on a projector |

## What "good" looks like

- **Bars visibly move** — feature activations changing step-to-step
- **Steering flash visible** at the Step 0 of the targeted clip
- **Verdict card appears clearly** at the end of each policy clip (red ✗ for baseline, green ✓ for targeted)
- **Audio is clean** — no room echo, no laptop fan hum
- **Total runtime 2:30 – 3:30** — class won't sit through more than that

Good luck. Hit "go" on the recording when you've got the storefront warm and the HUD refreshed.
