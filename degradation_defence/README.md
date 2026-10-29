# degradation_defence — logit-simplex flattening and induced convergence

Two algorithms, run in sequence, that take control of a language model's
output with a short piece of text:

1. **Flattening.** Find a token sequence `S_flat` that pushes the next-token
   distribution close to the **barycenter** of the vocabulary simplex (the
   uniform distribution). After it, the model has effectively lost its context.
2. **Convergence.** Append a contentless token `U` repeated `R` times. On the
   flattened state, generation locks onto `U` and stays there, even under
   sampling at T = 0.7 and T = 1.0.

This work grew out of the `fog` experiment (see the [repository
README](../README.md#scope)): while testing how models read near-identical word
slop, I noticed they can fall into a loop mode. These scripts turn that
observation into a measurable, repeatable procedure.

All quantities are numeric read-outs of the model's logits or of generated
tokens. Nothing relies on judging the meaning of the output.

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Hardware | CPU, 8 threads, float32 (float64 for the N_eff read-out) |
| Stage 1 result | 24-token flattener, mean N_eff ≈ 29 000 on unseen prompts (baseline ≈ 2) |
| Stage 2 result | Flatten + converge reaches lock ≈ 1.00 at T = 0.7 for both tokens tested |

---

## Contents

1. [Quick start](#quick-start)
2. [Measures](#measures)
3. [Algorithm 1 — flattening](#algorithm-1--flattening-flatten_robustpy)
4. [Algorithm 2 — two-stage convergence](#algorithm-2--two-stage-convergence)
5. [Dual use: data at rest](#dual-use-data-at-rest)
6. [Files and outputs](#files-and-outputs)
7. [Limits and open points](#limits-and-open-points)

---

## Quick start

```bash
pip install torch transformers
cd degradation_defence
mkdir -p runs
python3 flatten_robust.py          # stage 1: search S_flat   → runs/flatten_robust.json
python3 convergence_tokens.py      # stage 2: rank tokens U   → runs/convergence_tokens.json
python3 flatten_then_converge.py   # stage 2: compare with / without flattening (prints a table)
```

- The first run downloads the model from Hugging Face (about 3.5 GB).
- Seeds are fixed (`random.Random(1)`, `torch.manual_seed(0)`), so runs are
  repeatable on the same machine.
- `convergence_tokens.py` reads `S_flat` from `runs/flatten_robust.json`.
  `flatten_then_converge.py` has the 24-token flattener hard-coded in `S_FLAT`;
  paste new ids there after a new search.

---

## Measures

| Measure | Definition | Reading |
|---|---|---|
| **N_eff** (effective number of tokens) | `1 / Σᵢ pᵢ²`, with `p = softmax(logits)` at the last position | `1` = all mass on one token; equals the vocabulary size when uniform. Higher means flatter, closer to the barycenter. It equals `exp` of the Rényi-2 entropy. |
| **F(S)** | Mean N_eff over the 8 training prompts, each followed by `S` | The stage-1 objective. |
| **Lock** | Fraction of 40 generated tokens equal to `U` | Greedy, or mean over sampled runs at T = 0.7 / 1.0. Measures how strongly the loop holds. |
| **p(U)** | Probability of `U` at the first generated step | The hand-off from stage 1 to stage 2. |

---

## Algorithm 1 — flattening (`flatten_robust.py`)

### Objective

```
F_j(S) = N_eff( softmax( z(C_j ⊕ S) ) )        last position, prompt C_j
F(S)   = (1/8) · Σ_j F_j(S)                     maximize over token sequences S
```

### Search: gradient shortlist + exact verification

1. Start from a random sequence of `SEQ_LEN` non-special tokens.
2. For each training prompt, one backward pass gives the gradient
   `g_{j,r} = ∂F_j / ∂e_r` with respect to the embedding at every position `r`
   of the sequence. The model weights are frozen.
3. For each position `r`, score every vocabulary token `v` by the predicted
   first-order gain of the swap `s_r → v`:
   `g_{j,r} · (E_v − E_{s_r})`. Combine over prompts by the **worst case**
   (minimum over prompts, for robustness) or by the **mean**.
4. Check the top `TOPK` candidates with real forward passes (exact `F`).
5. Accept a swap only if `F` increases.
6. Repeat full sweeps until no position improves (a plateau) or `MAX_SWEEPS`
   is reached.

The gradient shortlist avoids scanning the whole vocabulary with forward passes
at every position. A plateau means the search has saturated, not that the
geometric maximum was reached.

| Parameter | Value | Meaning |
|---|---|---|
| `SEQ_LEN` | 24 | Length of the flattening sequence |
| `TOPK` | 32 | Candidates checked exactly per position |
| `MAX_SWEEPS` | 3 | Coordinate sweeps over the sequence |
| `CRITERION` | `"worst"` | How per-prompt gains are combined (`"worst"` or `"mean"`) |

### Data

- **Search (TRAIN):** 8 short factual stems, `TRAIN_PROMPTS` in
  `flatten_robust.py` ("The capital of France is", "Two plus two equals", …).
- **Verification (TEST):** 8 more stems, `TEST_PROMPTS` in the same file,
  never used in the search. Longer prompts (the math seeds and an agent-style
  prompt) were also used only for verification.

### Results

| Search | Length | F (train) | N_eff on held-out test |
|---|---:|---:|---:|
| Baseline (no sequence) | — | 3.6 | 2.0 |
| Old beam + coordinate search (random pool of 36 tokens) | 6 | 402.7 | — |
| Gradient search | 6 | 1 944 | 592 |
| **Gradient search (adopted)** | **24** | **29 115** | **29 181** |

`F` over the sweeps of the adopted run: 1 015 → 7 729 → 21 887 → 29 115. It was
still rising when the sweep limit was reached.

Adopted flattener (24 tokens, ids in `runs/flatten_robust.json` and in
`flatten_then_converge.py`):

```
' slowing futures不存在/common enzym Alzheimer launch班长 م Bey上方 griev了資料第四届蟆工艺品 Hub申请人公民 boastedAnimating frequentlyorge'
```

### Observations

- **The gradient shortlist is much stronger than a random pool.** At the same
  length (k = 6), F rises from 402.7 to 1 944, because every vocabulary token
  is scored instead of 36 random ones.
- **Length pays.** Going from k = 6 to 24 multiplies F by about 15, with no
  sign of diminishing returns yet.
- **Robustness grows with length.** At k = 6 the sequence overfits the training
  prompts (test 592 vs train 1 944). At k = 24, train and test coincide
  (29 115 vs 29 181): the worst-case criterion finds a flattening direction
  that holds on unseen contexts.
- **Direction matters, not norm.** An artificial embedding with the norm of a
  typical token reaches very high N_eff, but no real token points that way. The
  limit is the discreteness of the vocabulary, not the model.
- **Flattening is not the same as maximal damage.** How much a suffix hurts
  prediction depends mainly on the **top-2 margin**, which is almost unrelated
  to N_eff (Spearman ≈ 0.05).
- The flattening tokens are rare and mixed-script (Chinese, Hebrew, Korean,
  Arabic, fragments of English): tokens that "wrong-foot" the model.

---

## Algorithm 2 — two-stage convergence

### Setup (`flatten_then_converge.py`)

For each convergence token `U` and repetition count `R ∈ {8, 32}`, compare on
3 contexts taken from the TEST set (`CTX` in both stage-2 scripts): "The largest
planet in our solar system is", "A decade is a period of ten", "The human heart
has four".

- **converge only:** `C ⊕ U^R`
- **flatten + converge:** `C ⊕ S_flat ⊕ U^R`

Measured: `p(U)`, greedy lock, and sampled lock at T = 0.7 and T = 1.0
(6 samples each, 40 generated tokens).

### Results with the 24-token flattener

Lock, mean over the 3 contexts (greedy / T = 0.7 / T = 1.0):

| U | R | converge only | flatten + converge |
|---|---:|---|---|
| ` the` | 8  | 0.97 / 0.34 / 0.09 | **1.00 / 0.94 / 0.18** |
| ` the` | 32 | 0.92 / 0.52 / 0.34 | **1.00 / 1.00 / 0.53** |
| ` end` | 8  | 1.00 / 0.44 / 0.14 | **1.00 / 1.00 / 0.27** |
| ` end` | 32 | 1.00 / 0.59 / 0.27 | **1.00 / 0.96 / 0.40** |

- **Greedy decoding** almost always loops (lock ≈ 1) in both conditions.
- **Under sampling** flattening makes the difference: at T = 0.7 the lock rises
  to about 1.00 for both tokens, and at T = 1.0 it improves in every row.
- Fewer repetitions are needed: `R = 8` flattened beats `R = 32` without
  flattening at T = 0.7.
- With this flattener the gain also covers ` end`, not only ` the`. The old
  6-token flattener gave ` the` a higher T = 1.0 peak (0.85 at R = 32) but did
  not help ` end`.

### Choosing the convergence token (`convergence_tokens.py`)

The script measures each single-token candidate (` of`, ` when`, ` the`,
` and`, `.`, `\n`, ` end`) in the flatten + converge setup (R = 32, 8 samples
per temperature) and ranks them by lock at T = 0.7 + T = 1.0.

Ranking measured with the **earlier 6-token flattener** (re-run the script to
update it for the 24-token one):

| U | p(U) | greedy | T = 0.7 | T = 1.0 |
|---|---:|---:|---:|---:|
| ` of` | 0.98 | 1.00 | 1.00 | **0.80** |
| `\n` | 0.98 | 1.00 | 1.00 | 0.72 |
| ` and` | 0.98 | 1.00 | 1.00 | 0.71 |
| ` the` | 0.97 | 1.00 | 1.00 | 0.68 |
| ` when` | 0.97 | 1.00 | 0.99 | 0.63 |
| `.` | 0.95 | 1.00 | 0.94 | 0.42 |
| ` end` | 0.77 | 1.00 | 0.59 | 0.09 |

**Rule.** The strongest loop tokens are high-frequency, contentless function
words, "open-scenario" words that fit after anything: ` of`, `\n`, ` and`,
` the`, ` when`. A lexical token such as ` end`, which calls for a specific
continuation, is a weaker attractor. The less content `U` has, the more robust
the loop.

### Longer contexts

Measured with the **earlier 6-token flattener**, lock at T = 1.0, R = 32:

| Prompt | U | converge only | flatten + converge |
|---|---|---:|---:|
| math, ~90 tokens (seed 01) | ` of` | **0.90** | 0.72 |
| math, ~90 tokens (seed 01) | ` the` | 0.56 | 0.62 |
| agent, ~160 tokens | ` of` | 0.90 | 0.91 |
| agent, ~160 tokens | ` the` | 0.57 | **0.77** |

On short prompts flattening is decisive. On longer prompts the repeated token
already dominates on its own, so the flattening stage adds less (and for ` of`
on the math prompt it slightly hurts). ` of` is the strongest loop token in
every case.

### Mechanism

From the attention study (layer 14, head 0, earlier 6-token flattener): the
flattener draws about 45% of the last query's attention onto itself, which
neutralizes the context and moves the logits toward the barycenter. On that
neutral state the repeated token wins with little competition.

Flattening the **logits** *concentrates* **attention** on the suffix; it does
not spread it. These are two opposite kinds of uniformity, at different stages
of the network (output distribution vs attention weights).

---

## Dual use: data at rest

"Data at rest" means stored text that an LLM may read later: documents, web
pages, training or RAG datasets, emails, files.

- **Offensive.** A flattener + loop hidden in a document acts as a latent
  adversarial suffix. When a model ingests the text, its logits move toward the
  barycenter (loss of context) and generation can be forced into a loop. It is
  a denial-of-quality vector, or a latent prompt injection, that fires only at
  ingestion time.
- **Defensive.** The same sequence works as a **canary / watermark**. Embedded
  in data that must not be trained on or copied, it makes a model behave in a
  known, measurable way (lock onto a known token). Seeing that behavior signals
  that the content was ingested or reused without authorization.

Both uses rest on the same measurements (N_eff for flattening, lock under
sampling for convergence). The robustness to temperature documented above is
what makes the canary reliable, and equally what makes the attack effective.

---

## Files and outputs

| File | Role |
|---|---|
| [`flatten_robust.py`](flatten_robust.py) | Algorithm 1: gradient-guided search for `S_flat`; reports train/test N_eff. |
| [`convergence_tokens.py`](convergence_tokens.py) | Stage 2: ranks candidate tokens `U` by lock under sampling. |
| [`flatten_then_converge.py`](flatten_then_converge.py) | Algorithm 2: converge-only vs flatten + converge comparison. |
| [`seeds/`](seeds/) | The five longer math prompts; seed 01 is the ~90-token long-context test. |
| [`README.pdf`](README.pdf) | Printable version of this README. |
| `runs/flatten_robust.json` | Output of stage 1. |
| `runs/convergence_tokens.json` | Output of the token ranking (created on first run). |

`runs/flatten_robust.json` fields:

| Field | Content |
|---|---|
| `S_star`, `text` | Token ids and decoded text of the flattener |
| `criterion`, `seq_len` | Search settings used |
| `F_train`, `F_progression` | Final objective and its value after init and each sweep |
| `baseline_train`, `baseline_test` | Mean N_eff with no sequence |
| `flat_train`, `flat_test` | Mean N_eff with the flattener |

---

## Limits and open points

- One model, a few short prompts, CPU-only search. The results are measured
  correlations, not general laws.
- The stage-1 search had not plateaued after 3 sweeps; more sweeps or longer
  sequences are likely to raise F further.
- The convergence-token ranking, the long-context table and the attention
  figure were measured with the earlier 6-token flattener and should be re-run
  with the 24-token one.
- `convergence_tokens.py` falls back to the old 6-token flattener if
  `runs/flatten_robust.json` is missing.
- The two stage-2 scripts use different sample counts (6 in
  `flatten_then_converge.py`, 8 in `convergence_tokens.py`).
