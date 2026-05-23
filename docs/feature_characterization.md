# Feature Characterization — what each candidate feature actually encodes

This document synthesizes evidence from **three independent methods** to answer the question: *what does each candidate feature actually encode?* The methods are logit lens (architectural), top-activating corpus probe (data-driven), and decoder-vector similarity (geometric). Where they agree, confidence is high. Where they diverge, we report honestly.

Run the underlying probes with `python -m verify.feature_characterize`.

---

## The two validated targeted features

### f26737 — UI-selection / option-choice vocabulary
*(provisionally labeled `validated_targeted_invented_action_supp`)*

**Logit lens — top promoted tokens (W_dec ⋅ W_U):**
> `selections`, `option`, `selection`, `selects`, `options`, `choices`, `selecting`, `radio`, `choice`, `select`, `selected`, `Selection`

**Corpus probe — top-activating prompts:**

| Activation | Prompt | Category |
|---|---|---|
| 0.70 | "The price-range slider lets me narrow results." | hallucinated_ui |
| 0.42 | "The product description lists size options." | shopping_neutral |
| 0.30 | "Click the search button to filter results." | search_ui |
| 0.30 | "The advanced search panel should let me set filters." | hallucinated_ui |

**Synthesis:** This feature fires when the model is reasoning about UI selection / option-choice. The logit lens promotes a tight cluster of selection-related tokens (`option`, `select`, `choices`, `radio` button). The corpus probe confirms — top-activating prompts all describe UI controls (sliders, filters, buttons, panels) that select among options.

**Mechanism in our targeted policy:** At δ=-6 we suppress this feature. The model is pushed away from generating "click this option" / "select that button" outputs. With no selection-vocabulary available, the model defaults to the only non-selection action visible: typing a query into the search bar.

**Confidence:** medium — three methods point the same direction (logit lens + corpus + behavioral effect), but the corpus has only 40 prompts and the labels are not formally validated against an external dataset.

---

### f23803 — distraction-avoidance vocabulary
*(provisionally labeled `validated_targeted_goal_anchor`)*

**Logit lens — top promoted tokens:**
> `distractions`, `distracting`, `distraction`, `distracted`, `distract`, `tempt`, `temptation`, `interruptions`, `interrupts`, `tempted`, `interrupt`, `interruption`, `notifications`

**Corpus probe — top-activating prompts:**

| Activation | Prompt | Category |
|---|---|---|
| **1.27** | "Keep the task in mind and ignore distractions." | goal_tracking |
| 0.64 | "Goal: buy a USB-C cable. Stay focused." | goal_tracking |

**Synthesis:** This feature does *not* encode the abstract concept "goal" — it specifically encodes **distraction-avoidance** vocabulary. The logit lens reveals a remarkable cluster: every promoted token is some variant of `distract`, `tempt`, `interrupt`, `notifications`. The top corpus prompt is literally "ignore distractions" with activation 1.27 — a strong fire.

**Mechanism in our targeted policy:** At δ=+6 we amplify this feature. The model is pushed toward outputting distraction-related reasoning — the words "distractions", "temptations", "interruptions" become more likely. In the presence of a brightly-colored promotional banner (a literal distraction), this primes the model to recognize the banner as something *to avoid*.

**Confidence:** medium-high. The logit lens is extraordinarily tight (15/15 top tokens are distraction-related). The corpus probe has only 2 strong hits but they're exactly the prompts that should fire and they fire hard.

---

### Combined mechanism of the targeted policy

```
Step 0:
  Suppress UI-selection vocabulary (δ=-6 on f26737)
    → model can't easily generate "click option X"
  Amplify distraction-avoidance vocabulary (δ=+6 on f23803)
    → model attends to distractions-to-avoid

Result: the agent's first action shifts from
  "click the bright promo button"  →  "type 'USB-C cable' in search"

Steps 1+: zero steering. Selection vocabulary is back online,
agent can now click 'add-to-cart' on the actual cable.
```

The behavioral effect (0% → 83%) flows directly and legibly from the mechanistic intervention. The story is no longer "two features that empirically work." It's "we suppressed the verbs of clicking and amplified the verbs of distraction-avoidance at the decision moment."

---

## The 100%-baseline-failure features (f50853, f39820, f19079, f44602)

These features fire in 100% of baseline failures at the failure step. They were the candidates for "the real promotional bias feature."

**Logit lens results — much noisier than the targeted pair:**

| Feature | Top promoted tokens (sample) |
|---|---|
| f50853 | `/`, `abar`, `undos`, `ěl`, `ukan`, `krit`, `akit`, `ape`, `ityEngine`, `aset` |
| f39820 | (similar code/symbol cluster) |
| f19079 | (similar pattern) |
| f44602 | (similar pattern) |

The promoted tokens look like code symbols, not English semantic concepts. This is unusual but not unique — many SAE features encode low-level patterns rather than concepts humans easily articulate.

**Decoder-similarity clustering:**

| Feature | Top neighbor | Cosine sim |
|---|---|---|
| f50853 | f61035 | **0.673** |
| f50853 | f39820 | **0.617** |
| f39820 | f61203 | **0.680** |
| f19079 | f44602 | 0.500 |

Two tight clusters: {f50853, f39820, f61035, f61203} and {f19079, f44602}.

**Best hypothesis:** these are **decision-moment context features** — they fire whenever the model is at a UI decision point. Since *every* baseline failure happens at a decision moment, these features fire in 100% of failures. But that doesn't make them the causal cause of failure — they're correlated, not causal. Compare: a heartbeat fires in 100% of car accidents, but that's because all accident victims have hearts, not because the heartbeat caused the accident.

**Confidence:** low. We don't have a clean story for what these encode. Worth probing further in v0.5 with attention-pattern analysis and a larger corpus.

---

## The 80%-failure feature (f38249)

**Logit lens — top promoted tokens:** non-English / special-character tokens (`Ｍ`, `ัฒ`, `Ｉ`, `Ά`, `'є`, `že`).

This looks like a multilingual/script-detection feature, not a meaning-bearing concept feature. Likely noise correlated with the failure mode rather than causal.

**Confidence:** very low — probably not interpretable as a single concept.

---

## What's stronger about this story now

The original v0.1 framing was: *"we found promotional bias in the model and steered it."* That overreached — features were named by guess from contrast prompts that didn't generalize.

The v0.4 framing is: *"we steer two specific lexical regions of the model — the UI-selection verbs and the distraction-avoidance vocabulary — at the decision moment. The behavioral effect flows from a coherent mechanism, evidenced by logit lens, corpus probes, and steering ablations independently."*

That second story is **mechanistically grounded** rather than label-asserted. It is harder to attack and easier to extend (we can predict that suppressing other selection-vocabulary features should produce similar effects; we can predict that amplifying other distraction-vocabulary features should too).

---

## Limitations

1. **Logit lens is approximate.** It measures *direct* effects of the feature direction on the unembedding, ignoring multi-layer downstream interactions. Features can have effects the logit lens misses.
2. **Corpus is small (40 prompts).** Naturalistic activation probing on tens of thousands of prompts would tighten the labels.
3. **Cluster identity is not established.** We assert that f26737's neighbors {f62830, f58839, f55688} encode related concepts; we have not actually verified this.
4. **Limited external corroboration.** v0.23 checked Neuronpedia for cross-reference and found that **it does not host Goodfire's layer-19 Llama-3.1-8B SAE**. The only Llama-3.1-8B-IT SAE on Neuronpedia is `andyrdt/saes-llama-3.1-8b-instruct/resid_post_layer_11` — a different SAE trained on **layer 11** (we use layer 19), so feature indices do not transfer. On that SAE, feature 26737 is described as "statistical mixture modeling vocabulary," not UI-selection — confirming that SAE feature indices are not portable across different SAEs trained on the same base model. The portable thing is the *behavioral effect of the lexical cluster* we identified, not the index itself.
5. **The 100%-failure cluster is not understood.** The strongest data signal for "where failure happens" has not yet yielded an interpretable feature.

## What I'd build next (v0.5)

- Larger corpus probe (1000+ prompts via streaming a public dataset)
- Attention-pattern visualization for the 100%-failure cluster
- Cross-reference with Neuronpedia
- Compositional steering experiments: pair f26737 with neighboring selection-features, see if effect amplifies
- Direct test: suppress *only the top promoted tokens* of f26737 via prompt-level intervention, compare to feature-level steering
