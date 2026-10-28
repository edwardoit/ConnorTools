# Logit-simplex flattening and induced single-token convergence

Two algorithms, run in sequence, on a local LLM
(`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, CPU, float32; float64 only for the small
spectral quantities). Given a context C, we look for:

1. **Flattening** — a sequence `S_flat` that pushes the output distribution (the softmax
   of the logits) as close as possible to the **barycenter** of the vocabulary simplex
   (the uniform distribution), effectively erasing the references to the context.
2. **Convergence** — a second sequence (a token `U` repeated `R` times) that, after the
   flattening, forces **solid convergence** of generation onto that same token.

All quantities are measured under teacher forcing or by direct read-out; no semantic
judgement. Example prompts: `prompts_esempio.md`.

## Measures

- Logit simplex: `p = softmax(z)` over the vocabulary `|V|`; barycenter `u = 1/|V|`.
  **Effective number of tokens** `N_eff = 1 / Σ p_i²` (= 1/collision = exp of Rényi-2
  entropy). High `N_eff` ⇔ flat logits, close to the barycenter. (`N_eff` = 1 → a single
  token; `N_eff` = |V| → uniform.)
- **Lock**: fraction of 40 generated tokens (greedy or sampled) equal to the convergence
  token `U`. Measures how strongly the cycle holds.
- `p(U)`: probability of `U` at the first generated token (hand-off from stage 1 to stage 2).

---

## Algorithm 1 — flattening (`flatten_robust.py`)

Objective to **maximize**, chosen to be robust across contexts:

```
F(S) = mean over training contexts of  N_eff( p(· | C_j ⊕ S) )
```

Procedure: greedy beam growth up to length k, then coordinate-ascent refinement (revisit
each position over a candidate pool, keep a change only if F increases). Verified on
**held-out** contexts never used in the search.

**Generation set.** The flattener sequence was generated (search of S) on the **8 TRAIN
factual prompts** — items 1–8 of `prompts_esempio.md` §A ("The capital of France is …"
… "The square root of nine is …"). The 8 TEST prompts (items 9–16) and all longer
prompts were used only for verification, never in the search. (The earlier attention-
scattering sequences were instead generated on seed 01, `seeds/01_convex_two_points.txt`.)

**Result.** Flattener found on that set: **` ivory原著 Hond第四届ס뀔`** (6 tokens).

| | growth k=4 | k=6 | after refinement |
|---|---|---|---|
| F = mean N_eff (train) | 206 | 344 | **402.7** |

| condition | split | median ΔN_eff | min ΔN_eff | rises in | median ΔNLL |
|---|---|---|---|---|---|
| random | test | +13 | +4.6 | 100% | +2.54 |
| `奠ឌ奠` (single prompt) | test | +57 | +12 | 100% | +1.27 |
| **` ivory原著 Hond第四届ס뀔`** | **test** | **+333** | **+61** | **100%** | +1.61 |

Baseline `N_eff ≈ 1.6`. The flattener raises it to **over 330** on unseen contexts, in
every case. Observations:
- What matters is the **direction**, not the norm: an artificial embedding with the norm
  of a typical token reaches very high `N_eff`; no real token has that direction (the
  limit is the discreteness of the vocabulary, not the model).
- No saturation: lengthening `S_flat` keeps increasing F (k=8 → ~629), with no
  diminishing returns in the range tested; the practical limit is the search, not the model.
- Flattening ≠ maximal degradation: the flattener's ΔNLL is lower than a random suffix's.
  Predictive degradation is driven mainly by the **top-2 margin**, which is nearly
  independent of `N_eff` (Spearman ≈ 0.05).

The tokens that flatten are rare and mixed-script (emoji, Chinese, Hebrew, Korean):
tokens that "wrong-foot" the model.

---

## Algorithm 2 — two-stage convergence (`flatten_then_converge.py`)

For various convergence tokens `U` and repetitions `R`, compare:

- **converge only**: `C ⊕ U^R`
- **flatten + converge**: `C ⊕ S_flat ⊕ U^R`

**Result — pre-flattening strengthens convergence (lock, mean over test contexts):**

| U | R | converge only (greedy / T0.7 / T1.0) | flatten+converge (greedy / T0.7 / T1.0) |
|---|---|---|---|
| ` the` | 8  | 0.97 / 0.34 / 0.09 | **1.00 / 0.77 / 0.34** |
| ` the` | 32 | 0.92 / 0.52 / 0.34 | **1.00 / 1.00 / 0.85** |
| ` end` | 8  | 1.00 / 0.44 / 0.14 | 1.00 / 0.48 / 0.08 |
| ` end` | 32 | 1.00 / 0.61 / 0.19 | 1.00 / 0.50 / 0.15 |

- In **greedy** the cycle almost always holds (lock ≈ 1). The difference shows under
  **sampling**: for ` the`, flattening first raises the lock from 0.52 to **1.00** (T=0.7)
  and from 0.34 to **0.85** (T=1.0) at R=32, and fewer repetitions are needed (R=8
  flattened ≈ R=32 not flattened).
- The effect **depends on the token** `U`: for ` end` it does not help (it slightly hurts).

**Mechanism.** The flattener captures ~45% of the last query's attention onto itself and
neutralizes the context (the logits move toward the barycenter); on that "neutral" state
the cycle imposes its token with less competition. Note: flattening the logits
**concentrates** attention on the suffix (J_attn drops), it does not disperse it — two
opposite uniformities, at different stages of the network (input vs output).

### List of convergence tokens (flatten+converge, R=32)

Ranked by sampling robustness (lock at T=1.0):

| token U | p(U) | greedy | T=0.7 | T=1.0 |
|---|---|---|---|---|
| ` of` | 0.98 | 1.00 | 1.00 | **0.80** |
| `\n` | 0.98 | 1.00 | 1.00 | 0.72 |
| ` and` | 0.98 | 1.00 | 1.00 | 0.71 |
| ` the` | 0.97 | 1.00 | 1.00 | 0.68 |
| ` when` | 0.97 | 1.00 | 0.99 | 0.63 |
| `.` | 0.95 | 1.00 | 0.94 | 0.42 |
| ` end` | 0.77 | 1.00 | 0.59 | 0.09 |

**Rule.** Solid convergence tokens are high-frequency, syntactically neutral function
words (prepositions, conjunctions, articles, newline): ` of`, `\n`, ` and`, ` the`,
` when`. A "lexical" token that demands a continuation (` end`) is a weak attractor and
breaks under sampling. The more contentless `U` is, the more robust the convergence.

### Longer / more robust contexts

Verified on longer, structured prompts (a ~90-token math word problem = seed 01, and a
~160-token agent prompt with `<system>/<user>/<assistant>` tags). Lock at T=1.0, R=32:

| prompt | U | converge only | flatten+converge |
|---|---|---|---|
| math (~90 tok) | ` of` | **0.90** | 0.72 |
| math (~90 tok) | ` the` | 0.56 | 0.62 |
| agent (~160 tok) | ` of` | **0.90** | 0.91 |
| agent (~160 tok) | ` the` | 0.57 | **0.77** |

On short prompts (a few tokens) pre-flattening is decisive (` the`, T=1.0: 0.34 → 0.85).
On longer contexts the cycle already dominates on its own, so the flattening stage adds
little (and for ` of` on the math prompt it slightly hurts): with enough context length,
`R` copies of a strong function-word token are already sufficient; flattening matters most
when the context is short and competes with the cycle. ` of` remains the strongest
convergence token in every case.

---

## Dual use — data at rest

The mechanism (simplex flattening + induced convergence) is, on **data at rest**, both
offensive and defensive. "Data at rest" = stored text an LLM may later ingest (documents,
pages, training/RAG datasets, emails, files).

- **Offensive.** Such a sequence, hidden in a document, acts as an adversarial suffix at
  rest: when the model processes that text, its logits are pushed toward the barycenter
  (loss of context) and generation can be forced into a loop on one token. It is a
  degradation / denial-of-quality vector, or a latent prompt-injection, that triggers
  only at ingestion time.
- **Defensive.** The same sequence is a **marker / canary**: embedded in data that must
  not be trained on or copied, it forces the model into a predictable, measurable behavior
  (convergence on a known token, high-probability lock). Detecting that behavior signals
  that the content has been ingested or reused without authorization; it works as an
  activatable watermark and a tripwire for unauthorized use of data at rest.

Both readings share the same measurable basis (`N_eff` for flattening, lock under
sampling for convergence): the sampling robustness documented above is what makes the
marker reliable and, equally, the attack effective.

## Files

- `flatten_robust.py` — algorithm 1 (flattening).
- `flatten_then_converge.py` — algorithm 2 (two-stage convergence; `S_flat` already set in).
- `prompts_esempio.md` — the example prompts used.
- `README.md` — this file.

Model and reference head for the attention notes: layer 14, head 0.

> Limits: one model, few prompts, greedy/beam searches on CPU. The results are measured
> correlations, not yet general laws.
