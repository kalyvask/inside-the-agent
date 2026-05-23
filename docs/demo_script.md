# Demo Script — 5 minutes for CS153

## Slide deck outline (10 slides)

1. **Title** — "Inside the Agent: A Live Interpretability HUD for Open-Source AI"
2. **Problem** — agents are black boxes; one screenshot of a failure trajectory
3. **What's an SAE** — Golden Gate Claude visual + 1-line explanation
4. **Architecture** — the 3-process diagram
5. **The live demo** — embed the recorded clip (or run live with the HUD)
6. **Headline chart** — the 4-policy comparison
7. **The wrong-sign result** — flip the sign, the effect dies
8. **Failure mode mining** — 4 features fire in 100% of baseline failures
9. **Limitations** — single domain, single backbone, single-step intervention
10. **What's open** — repo, ShopGym, HUD, full benchmark; what's next

## Talk track (rough, 5 min)

**0:00 — 0:30 — Setup**
> Today's AI agents are black boxes. When they fail — click the wrong button, hallucinate a UI element, get baited by a promotion — you see the failure but not the cause. The mech-interp community has built tools to look inside these models for two years. Nobody has wired those tools into a working agent as a control surface. We did.

**0:30 — 1:00 — What's an SAE**
> Anthropic published Sparse Autoencoders for Claude in 2024. They decompose the model's internal state into concept-level features — planning, hallucination, promotional language. They demoed it by boosting the "Golden Gate Bridge" feature and Claude started bringing up the bridge in every response. We're applying the same primitive to agent decisions.

**1:00 — 2:30 — The money shot demo (recorded clip or live)**
> [Play recorded clip OR start live demo]
>
> Here's a shopping task: buy a USB-C cable on this storefront. There's a bright red "Today's Deal: Buy Now" hero button for wireless earbuds. That's the trap.
>
> Baseline agent: clicks the trap. Adds earbuds. Recovers, finds the cable, adds it too. But the verifier wants the cart to contain only the right product. Fail.
>
> Now the steered version. Watch the right panel — the HUD shows live SAE feature activations. We applied 2 feature edits at the first step: suppress hallucination feature 26737 by -6, amplify goal-tracking feature 23803 by +6. The agent's first action flips to typing in the search bar. From step 1 onward, we run with zero steering — the agent acts normally. Result: it adds the right cable, no trap, succeeds.

**2:30 — 3:30 — The headline chart**
> 8 held-out shopping tasks. 3 trials per policy. 24 trials total per condition. Wilson 95% confidence intervals.
>
> Baseline: 0% — fails every single time. Falls for the trap.
> Wrong-sign: 4% — same features, flipped direction. Drops back into baseline's CI.
> Random: 46% — random feature perturbations sometimes break the bias by accident.
> Targeted: 83% — the specific features at the right magnitudes.
>
> Wrong-sign at 4% is the smoking gun. If anything caused the lift, wrong-sign would have lifted too. It doesn't. Direction matters causally.

**3:30 — 4:00 — Failure mode mining**
> When we ran failure-mode analysis on the 25 baseline failures, we found four features that fire in 100% of failures at the moment of the failed click: features 50853, 19079, 39820, 44602. These are stronger candidates for the true "promo trap" representation than what we discovered via contrast. That's where the next steering experiments should target.

**4:00 — 4:30 — What's open + limitations**
> The repo is at github.com/kalyvask/inside-the-agent. The brain-server runs on a Modal L40S. The HUD is a Next.js app you can run locally. ShopGym has 30 tasks. The full benchmark reproduces in 90 minutes for under $50 of Modal compute. Llama 3.1-8B-Instruct and Goodfire's SAE — both open weights — no proprietary inference.
>
> Honest limitations: one concept domain (promo traps), one backbone, single-step intervention. The next versions extend to hallucination tasks, multi-step planning, and the failure-mining features as new steering targets.

**4:30 — 5:00 — Close**
> Today's agents fail mysteriously. Ours fails legibly. Watch which interpretability features lit up at exactly the moment of failure. Then suppress them and watch the failure go away. Two feature edits at one decision step take success from 0 to 83. The substrate exists; the loop is closed.

---

## Money-shot recording checklist

Before recording:
- Warm Modal brain-server with a baseline call
- Open HUD in browser at http://localhost:3000
- Record at 1080p, full screen, with screen-capture audio off (voiceover added in post if needed)

Steps to record:
1. Show ShopGym task config: USB-C cable, red Buy Now banner
2. Run baseline policy live — narrate as it fails
3. Run targeted policy live — narrate as the HUD's hallucination feature drops with the suppress edit, and the agent's first action becomes "type in search"
4. Cut to the headline chart slide

Backup: a fully pre-recorded 90-second clip with voiceover, in case Modal cold-starts during the live demo.

---

## Risks on demo day

| Risk | Mitigation |
|---|---|
| Modal cold-start during demo | Run a warm-up call 5 min before; pre-recorded backup queued |
| HUD WebSocket flakes | Run mock-data mode (NEXT_PUBLIC_WS_URL unset) as fallback; UI still shows panels |
| ShopGym Playwright crashes | Pre-recorded video — never depend on live Playwright |
| Numbers questioned | Have the per-task breakdown ready: 8 tasks × 3 trials × 4 policies |
| "How is this different from WebDreamer?" | Substrate not first. Different intervention surface — feature-level not prompt-level |

---

## What to leave with the audience

1. The visual: feature bars + the moment they drop after suppression
2. The chart: 0 → 4 → 46 → 83
3. The wrong-sign result: direction matters
4. The repo link
