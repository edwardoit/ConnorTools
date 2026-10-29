# ConnorTools

## Scope

This repository collects two experiments on one weakness of large language
models: **repetitive, near-identical input ("word slop") is not really read,
and it can be used to steer the model.**

**1. fog: evidence that models don't see what changes in the slop.**
`fog` builds a wall of near-identical words (`hello`, `helloo`, `hellooo`, …)
and hides a message only in the *order* of those words. It started as a test of
how big models and their MCP tools handle this kind of input. My observation:
the model collapses the wall into a summary, much like `array.length` reduces
an array to one integer. The tokens are still read one by one and fill the
KV cache, but the model does not track *which* variant sits *where*. The
small suffix variations (`o`, `oo`, `ooo`, …) leave no lasting trace in the
keys that later queries attend to (the future q·k alignment), so the positional
structure goes unnoticed. The message survives intact: same words, same counts,
only the order changes. `fog` is the working proof that this channel exists and
is invisible to a reader that treats slop as noise.

**2. Logit flattening + convergence: the analysis that follows from it.**
While running the first test I noticed that big models can fall into a **loop
mode**, repeating one token. Analysing the logits, I looked for what makes that
happen on purpose, and found a two-stage method:

1. **Flatten.** A short, necessary token sequence pushes the output
   distribution (the logit simplex) close to its **barycenter**, the uniform
   distribution. The model "forgets" its context.
2. **Converge.** On that neutral state, a repeated open-scenario word, i.e. a
   contentless function word such as ` of`, ` the`, ` and` or `\n`, induces a
   loop on that token.

The two-stage approach is **robust to temperature**: under sampling, where
repetition alone often breaks, flattening first keeps the loop at lock ≈ 1.00
at T = 0.7 and raises it in every case at T = 1.0.
Both stages are measured numerically (effective number of tokens, lock rate),
with no human judgement of the output.

## The two tools

Two small, self-contained research tools. The first shows that structure
can hide inside word slop; the second uses what the model does with such
input to steer its output:

| Tool | Folder | What it does | Language / deps |
|---|---|---|---|
| **fog** | [`fog/`](fog/) | Hides a byte payload in the *order* of a wall of `hello`, `helloo`, `hellooo`, … tokens, without changing how often each token appears. | Python 3.10+ (stdlib only) · browser decoder in plain JS |
| **Logit flattening + convergence** | [`degradation_defence`](degradation_defence/) | Searches for a short token sequence that pushes an LLM's next-token distribution toward uniform ("flattening"), then shows that a repeated token after it locks generation onto that token ("convergence"). | Python, PyTorch, Hugging Face `transformers` |

Each folder has its own detailed README with the full math and results:
[`fog/README.md`](fog/README.md) (printable version with rendered formulas:
[`fog/README.pdf`](fog/README.pdf)) and
[`degradation_defence/README.md`](degradation_defence/README.md) (printable:
[`degradation_defence/README.pdf`](degradation_defence/README.pdf)). This file is the
overview: what each algorithm does, how it works, how to run it, and its limits.

---

## 1. fog — a frequency-preserving word wall

### Idea

The only "alphabet" is 30 near-identical words:

```
Σ = { "hell" + "o"*k  |  k = 1 … 30 }   →   hello, helloo, hellooo, …
```

A random wall of these words is generated, then a message is written into it
**only by swapping pairs of words**. Swaps never change how many times each
variant appears, so the word histogram before and after embedding is identical.
That invariant is the core trick: the receiver can rebuild everything needed
for decoding from the final wall plus a password.

> **fog is steganography, not encryption.** The password is the only secret.
> Anyone who has it (or guesses it) can read the payload. Encrypt sensitive data
> before embedding it.

### How it works

1. **Carrier.** Sample `N` words independently and uniformly from the 30
   variants (`secrets.SystemRandom` by default; `wall_seed` makes it
   reproducible).
2. **Order-invariant salt.** Count each variant, write the 30 counts as
   big-endian 32-bit integers, and hash them:
   `S = SHA-256(BE32(f₁) ‖ … ‖ BE32(f₃₀))`.
   Because swaps don't change counts, `S` is the same before and after
   embedding.
3. **Password → pairs.** Build a deterministic byte stream
   `R_c = SHA-256(S ‖ UTF8(password) ‖ BE64(c))`, use it to drive an unbiased
   Fisher–Yates shuffle of `0 … N-1` (rejection sampling for random integers),
   and take consecutive elements as pairs `(i₀,i₁), (i₂,i₃), …`. Pairs holding
   the same variant are skipped (their order can't carry a bit); an odd last
   index is ignored.
4. **Encode.** The payload is `BE32(length) ‖ message bytes`, MSB first. For
   each usable pair `(a, b)`:
   - bit `0` ⇔ `len(w[a]) > len(w[b])`
   - bit `1` ⇔ `len(w[a]) < len(w[b])`

   If the pair already shows the target bit, leave it; otherwise swap the two
   words.
5. **Decode.** From the final wall: recompute `S`, rebuild the same pairs from
   the password, read bits from usable pairs, take the first 32 bits as the
   length, then read that many bytes.

### Capacity

Two random words collide with probability 1/30, so about 29/30 of the `N/2`
pairs are usable:

```
expected capacity ≈ (N / 2) · (29 / 30) = 29N / 60 bits
```

A message of `m` bytes needs `32 + 8m` usable pairs. `generate_wall` sizes the
wall from this estimate with a 10% margin, measures the real capacity, and grows
the wall until the message fits.

### Usage

Python (run from the repository root):

```python
from fog.generator import generate_wall, read_wall

wall = generate_wall(b"secret message", key="THE_PASSPHRASE")   # n is optional
text = " ".join(wall)

assert read_wall(text.split(), "THE_PASSPHRASE") == b"secret message"
```

Built-in self-test (checks that frequencies and salt are unchanged and that the
round trip works):

```bash
python3 fog/generator.py
```

Browser decoder — serve the folder over HTTP (Web Crypto and ES modules don't
work from `file://`):

```bash
python3 -m http.server 8000 --directory fog
```

Then open <http://localhost:8000/decoder.html>, paste a wall and the password.

### Main API (`fog/generator.py`)

| Function | Purpose |
|---|---|
| `generate_carrier(n, wall_seed=None)` | Random wall of `n` tokens. |
| `frequencies(wall)` / `salt_from_wall(wall)` | Variant counts and the SHA-256 salt derived from them. |
| `pairing(wall, key)` | Password- and wall-specific list of index pairs. |
| `capacity(wall, key)` | Number of usable pairs (bits) for that password. |
| `insert(wall, key, message)` | Embed `message` (bytes) by swapping; returns a new wall. |
| `read_wall(wall, key)` | Extract the payload bytes. |
| `generate_wall(message, key, n=None, wall_seed=None, margin=1.1)` | Carrier + sizing + embedding in one call. |

The JavaScript decoder ([`fog/decoder.js`](fog/decoder.js)) mirrors the Python
salt, permutation, and pairing exactly and exports `decodeWall`, `extractBits`,
`pairing`, and `saltFromWall`.

### Security model and limits

- The wall, counts, salt, and algorithm are all public; only the password is
  secret. The salt binds the pairing to this wall but adds no password entropy.
- No integrity check: adding, removing, or reordering words corrupts decoding.
- Not a password verifier: a wrong password can, rarely, yield a plausible
  length and valid UTF-8 by chance.
- Wall size barely helps against guessing (each guess costs `O(N)`); use a
  strong passphrase.

---

## 2. Logit-simplex flattening and induced single-token convergence

Folder: [`degradation_defence`](degradation_defence/) · full write-up:
[`degradation_defence/README.md`](degradation_defence/README.md)
([PDF](degradation_defence/README.pdf)). Model under
study: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, on CPU, float32.

Two algorithms run in sequence on a context `C`:

1. **Flattening** — find a token sequence `S_flat` that pushes the next-token
   distribution as close as possible to the uniform distribution (the
   barycenter of the vocabulary simplex), washing out the context.
2. **Convergence** — append one contentless token `U` repeated `R` times and
   measure how strongly generation then locks onto `U`.

### Measures

- **Effective number of tokens:** `N_eff = 1 / Σ pᵢ²` where `p = softmax(logits)`
  at the last position. `1` means one token takes all the mass, `|V|` means
  fully uniform. Higher = flatter.
- **Lock:** fraction of 40 generated tokens equal to `U` (greedy, or sampled at
  T = 0.7 / 1.0).
- **p(U):** probability of `U` at the first generated step.

### Algorithm 1 — flattening (`flatten_robust.py`)

Maximize `F(S) = mean over 8 TRAIN prompts of N_eff(p(· | Cⱼ ⊕ S))` with a
**gradient-guided search**: one backward pass per prompt gives `∂N_eff/∂e_r` at
every position; each vocabulary token is scored by its predicted first-order
gain `g · (E_v − E_{s_r})` (worst case over prompts); the top 32 per position
are checked with exact forward passes; a swap is kept only if `F` rises.

Result: a 24-token sequence reaches mean N_eff **29 115** on TRAIN and
**29 181** on 8 held-out TEST prompts (baseline ≈ 2–4). Longer sequences
generalize better: at 6 tokens the search overfits (train 1 944, test 592).

### Algorithm 2 — two-stage convergence

`flatten_then_converge.py` compares `C ⊕ U^R` with `C ⊕ S_flat ⊕ U^R`;
`convergence_tokens.py` ranks candidate tokens `U`.

- Greedy decoding loops in both cases (lock ≈ 1).
- **Under sampling, flattening first makes the loop hold:** at T = 0.7 the lock
  goes from 0.34–0.59 to 0.94–1.00 for both ` the` and ` end`; at T = 1.0 it
  improves in every case.
- The best loop tokens are contentless function words (` of`, `\n`, ` and`,
  ` the`, ` when`), per a ranking measured with the earlier 6-token flattener.
- On longer prompts the repeated token already dominates on its own, so
  flattening adds less.

### Usage

```bash
pip install torch transformers
cd degradation_defence
mkdir -p runs
python3 flatten_robust.py          # stage 1 → runs/flatten_robust.json
python3 convergence_tokens.py      # stage 2 token ranking → runs/convergence_tokens.json
python3 flatten_then_converge.py   # stage 2 comparison table
```

The first run downloads the model (about 3.5 GB). All scripts use 8 CPU threads
and fixed seeds.

### Dual use

The same mechanism applies to **data at rest** (documents, web pages, datasets,
emails an LLM may later read):

- **Offensive:** a hidden flattener + loop acts as a latent adversarial suffix
  that degrades output or forces a loop when the text is ingested.
- **Defensive:** the same sequence can serve as a canary / watermark: if a
  model shows the known, measurable lock behavior on it, the content was likely
  ingested or trained on without authorization.

### Limits

One model, a few prompts, CPU-only search. The results are measured
correlations, not general laws. Some stage-2 tables were measured with an
earlier 6-token flattener; the folder README marks which ones.

---

## Repository layout

```
ConnorTools/
├── README.md                  ← this overview
├── fog/
│   ├── generator.py           encoder + reader (Python)
│   ├── decoder.html/.js       browser decoder
│   ├── img/                   decoder screenshots
│   ├── README.md              full write-up
│   └── README.pdf             printable write-up with rendered formulas
└── degradation_defence/
    ├── flatten_robust.py         stage 1: gradient search for S_flat
    ├── convergence_tokens.py     stage 2: rank loop tokens
    ├── flatten_then_converge.py  stage 2: with / without flattening
    ├── seeds/                    longer math prompts
    ├── runs/                     saved results (JSON)
    ├── README.md                 full write-up and result tables
    └── README.pdf                printable write-up
```
