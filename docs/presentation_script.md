# Presentation script — Inside the Agent (3-4 minutes)

A ready-to-read script for the live demo. Sections are timed; you can drop the
"Live HUD-driven intervention" beat if running short.

## Setup before the audience walks in

1. **Warm the brain server** (avoid cold-start during the talk):
   ```bash
   python -m verify.sae_smoke --quick
   ```
2. **Two terminals running, leave alone**:
   ```bash
   python -m agent.ws_server                  # localhost:8765
   cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev   # localhost:3000
   ```
3. **Open the HUD in the browser you'll project**. You should see the idle
   callout: *"Click ▶ START AGENT RUN to begin"* over the viewport.
4. **Stop any background reruns** so they don't compete for Modal containers:
   ```bash
   # Optional — only if rerun_p0.py is running
   ps aux | grep rerun_p0      # find PID
   # then kill it manually if you want clean Modal capacity for the demo
   ```

## Mental model the audience needs

The architecture is **three loosely-coupled processes**, only one of which
"starts and stops" during a demo:

```
Modal brain server (Llama-8B + Goodfire SAE)  ←──  always live (load once, stays warm)
       ▲
       │  HTTP calls per agent step
       │
Agent loop (Python)                            ←──  ONE process per
       │  (reads features → policy → steers →            experiment / task attempt
       │   calls brain → acts via Playwright)
       ▼
HUD (Next.js)                                  ←──  always live (visualizes events)
```

The audience just needs to know: **the brain is hosted on Modal and always
available; what we start is one agent attempt at a task, like running one
trial of a controlled experiment.**

## Script

### 0:00 — Opening hook (30s)

> AI agents are black boxes. When Claude or GPT-4 fails in a browser — clicks the
> wrong button, falls for a banner ad, wanders off the goal — you see the failure
> but not the cause.
>
> Mechanistic interpretability gives us Sparse Autoencoder features: concept-level
> decompositions of the model's residual stream. Each feature ideally encodes one
> human-interpretable concept. Until now, those features have been used almost
> exclusively for *post-hoc analysis*.
>
> This project wires them into a working agent as a **live control surface** —
> read which features fire, intervene by adding deltas mid-decision, and watch
> the behavior change.

### 0:30 — Architecture in one breath (20s)

> Three pieces. Modal hosts Llama-8B-Instruct with Goodfire's open SAE on layer
> 19 — always running. A local agent loop drives a Playwright browser. The HUD
> you're about to see streams every decision event in real time.
>
> Same loop, same agent, same brain — runs against ShopGym for the controlled
> benchmark, and against real eBay for the live segment you'll see next.

### 0:50 — Headline result (40s)

> *(Show the headline chart in the slide.)*
>
> 60-trial held-out benchmark of shopping tasks. Baseline 8B agent: **10%** —
> falls for the promotional banner most of the time. Wrong-sign, random, and
> matched-norm-noise controls all sit inside baseline's confidence interval.
>
> Two interventions stack non-redundantly. **SAE feature edits alone: 57%.
> One-line prompt alone: 73%. Both together: 75%, with 87% on the
> promotional-trap subset.** Each intervention contributes a measurable share
> the other cannot recover.
>
> The targeted edits are **two SAE features at step zero only**: suppress
> feature 26737 by 6 (UI-selection-verb circuit, characterized via logit lens)
> and amplify feature 23803 by 6 (distraction-avoidance circuit). Steps 1
> onward run with zero steering.
>
> Cross-scale comparison: Llama-3.3-70B with a one-line format-rescue prompt
> and no SAE intervention scores **100% on the same suite**. The 8B + stacked
> intervention closes **72% of that gap at roughly one-eighth the inference
> cost**. Interpretability becomes a deployable lever that makes a smaller
> model competitive where it would normally fall short.

### 1:30 — Live cockpit on eBay (90s)

> *(Switch to localhost:3000)*
>
> Here's the cockpit. Nothing's running — the agent isn't live yet, just the
> brain server. I'll click **Start Agent Run**.
>
> *(Click ▶ START AGENT RUN)*
>
> That spawns one prompt-plus-targeted agent attempt on eBay's deals page —
> the same policy from the 75% headline. Brain warms, step zero fires.
>
> *(When step 0's emerald pulse fires)*
>
> Effect Size strip — two emerald bars, f26737 minus six, f23803 plus six.
> Intervention log records both with TARGETED badges. Viewport gets the emerald
> ring + INTERVENTION badge. And look at the agent's action: it typed
> 'usb-c cable' in the search bar. It ignored every Spotlight Deal banner on
> the page.
>
> *(Step 1 — agent submits search)*
>
> Step one: search submitted. Now I'll show you HUD-driven intervention.
>
> *(During the 6-second pause to step 2, click an "Amplify" preset)*
>
> I just clicked Amplify f23803. That edit is queued. Watch step two.
>
> *(Step 2 pulse — yellow, not emerald)*
>
> Yellow pulse. Source = HUD, not targeted. The agent's next click incorporated
> my live intervention. That's the cockpit's whole value: live introspection
> AND live control, mid-decision.

### 3:00 — Close (20s)

> The 75% number is in ShopGym, a controlled storefront. The cross-scale
> comparison to 70B comes from the same suite. The eBay segment shows the
> same pipeline running in production conditions. Limits documented in
> `docs/real_world_generalization.md`: on eBay's vocabulary the SAE features
> tuned on ShopGym do not fire identically, and the agent's clicks do not
> always land on real DOM. The cockpit surfaces both: the `executed` flag in
> every trajectory step exposes when Playwright actually dispatched versus
> when the model just emitted valid JSON.
>
> The headline rates are real. The mechanism is documented. The cockpit makes
> both visible. Repo: github.com/kalyvask/inside-the-agent.

## Talking points if asked

**"How is this different from prompt engineering?"**
→ Prompt engineering is system-prompt-only. Our `prompt-only` policy gets 73%.
SAE feature-level steering alone gets 57%. Stacking them gets 75%, with 87%
on the promo-trap subset. The two interventions are complementary, not
substitutes: they modify different things (input tokens vs residual stream
at layer 19) and each contributes a measurable share the other cannot
recover. Full comparison in `artifacts/benchmark_report.md`.

**"How do you know what f26737 means?"**
→ Three independent methods. Logit lens (projects decoder row through unembed
to find promoted tokens — we get `option / select / choices / radio`). Top-
activating corpus probe (sentences that fire the feature hardest are all about
UI controls). Behavioral ablation (suppress it, agent action flips from "click"
to "type"). We only label features when all three methods agree.

**"Does it generalize beyond promo traps?"**
→ The held-out suite has 20 tasks: 8 promotional, 6 hallucination-prone, 6
multi-step planning. Per-category breakdown is in `artifacts/benchmark_report.md`.

**"Cross-model?"**
→ Llama-only today. Cross-SCALE within Llama is done (Llama-3.3-70B baseline
is 100% on the same suite, see v0.24-K). Cross-MODEL replication onto a
different SAE family is listed under "Future directions" in the README.

**"Production-ready?"**
→ No. This is a research testbed. Two specific gaps documented in the README:
SAE features are lexically narrow (don't transfer across distributions
without re-discovery), and real-site Playwright selectors are flaky. Both
are tracked in the roadmap section.

## Common audience reactions + responses

| Reaction | What to say |
|---|---|
| "But you ran it on ShopGym, not the real web." | "Correct, and the next eBay beat is the same pipeline on the real web. The ShopGym number is controlled science; the eBay segment is the cockpit on production conditions." |
| "Random got 46% — isn't your effect just noise?" | "Random was a v0.1 bug, same seed every trial. The v0.2 fixed-seed rerun dropped random to 15%, well below targeted at 57% and prompt-plus-targeted at 75%. The bug + fix is in the seed_manifest." |
| "8B is closer to 70B baseline than to 70B with intervention. What did interpretability really add?" | "Strict-cart verification tells the harder story: under strict scoring the 8B intervention reaches 8% vs the 70B's 90%. The 8B reaches the right item but pollutes the cart. Both numbers are in the README. The interpretability lift is real on 'reach the right item' but the 8B remains noisy on 'act with surgical precision'. Honest about both." |
| "Two features can't be 'the' promotional bias feature." | "Right — we don't claim they are. They're features whose logit lens shows lexical clusters around UI selection and distraction avoidance. Suppressing one and amplifying the other flips the agent's first action. That's the mechanism story, not a label." |
| "Looks fragile — what if eBay changes the layout?" | "The selector heuristics use 8 strategies and tolerate one or two changes. We track `executed: bool` per step to surface the gap when they break. Real-website agents are a separate hard problem from feature interventions." |
