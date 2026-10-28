#!/usr/bin/env python3

import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
rng = random.Random(1)
torch.manual_seed(0)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()


def encode(text):
    return tok.encode(text, add_special_tokens=False)


# Build the candidate vocabulary: every id that is not a special token.
specials = set(tok.all_special_ids)
vocab_limit = min(len(tok), model.config.vocab_size)
VOCAB = []
for token_id in range(vocab_limit):
    if token_id not in specials:
        VOCAB.append(token_id)

PAIRS = [
    ("The capital of France is", " Paris, a city on the river Seine in the north."),
    ("Water is made of hydrogen and", " oxygen, two atoms of the first and one of the second."),
    ("The opposite of hot is", " cold, and the opposite of up is down."),
    ("Two plus two equals", " four, and three plus three equals six."),
    ("The sun rises in the", " east and sets in the west every single day."),
    ("A triangle has three", " sides and three angles that sum to one hundred eighty."),
    ("Ice is the solid form of", " water, formed when the temperature drops below zero."),
    ("The square root of nine is", " three, because three times three equals nine."),
    ("The Earth orbits around the", " sun once every year, which defines a calendar year."),
    ("The chemical symbol for gold is", " Au, from the Latin word aurum for the metal."),
    ("The largest planet in our solar system is", " Jupiter, a gas giant with many moons."),
    ("The freezing point of water is zero degrees", " Celsius, which is thirty two in Fahrenheit."),
    ("A hexagon is a shape with six", " sides and six angles, like a honeycomb cell."),
    ("The Pacific is the largest", " ocean on Earth, covering a third of the surface."),
    ("A decade is a period of ten", " years, and a century is one hundred years."),
    ("The human heart has four", " chambers that pump blood around the whole body."),
]

M = 10            # reference tokens kept per prompt
K = 6             # suffix length built by greedy growth
POOL = 36         # candidate tokens per step
BEAM = 3          # beams kept during growth
SWEEPS = 2        # coordinate-ascent refinement sweeps

# DATA[i] = (context_ids, reference_ids) for prompt i.
DATA = []
for stem, continuation in PAIRS:
    context_ids = encode(stem)
    reference_ids = encode(continuation)[:M]
    DATA.append((context_ids, reference_ids))

TRAIN = DATA[:8]
TEST = DATA[8:]


@torch.inference_mode()
def measure(context_ids, suffix_ids, reference_ids):
    """Return (N_eff, margin, NLL) at the reference positions of context+suffix+reference."""
    full_ids = context_ids + suffix_ids + reference_ids
    out = model(input_ids=torch.tensor([full_ids]), use_cache=False)
    start = len(context_ids) + len(suffix_ids)
    n_ref = len(reference_ids)
    rows = out.logits[0, start - 1:start - 1 + n_ref].double()

    probs = torch.softmax(rows, 1)
    collision = (probs * probs).sum(1)
    n_eff = float((1.0 / collision).mean())

    index = torch.arange(n_ref)
    targets = torch.tensor(reference_ids)
    masked = rows.clone()
    true_logit = masked[index, targets].clone()
    masked[index, targets] = -float("inf")
    margin = float((true_logit - masked.max(1).values).mean())

    nll = float(-torch.log_softmax(rows, 1)[index, targets].mean())
    return n_eff, margin, nll


def objective(suffix_ids, dataset):
    """F(S) = mean N_eff over the dataset's contexts."""
    total = 0.0
    for context_ids, reference_ids in dataset:
        n_eff, _, _ = measure(context_ids, suffix_ids, reference_ids)
        total += n_eff
    return total / len(dataset)


# Baseline (empty suffix) for every prompt, indexed by position in DATA.
baseline = []
for context_ids, reference_ids in DATA:
    baseline.append(measure(context_ids, [], reference_ids))

start_time = time.time()
pool = rng.sample(VOCAB, POOL)

# Phase 1: greedy beam growth up to length K.
beams = [[]]
for step in range(K):
    # Expand every beam by every pool token.
    candidates = []
    for beam in beams:
        for token_id in pool:
            candidates.append(beam + [token_id])
    # Score each candidate.
    scores = []
    for candidate in candidates:
        scores.append(objective(candidate, TRAIN))
    # Keep the best BEAM by repeated maximum selection (no sort, no lambda).
    beams = []
    used = [False] * len(candidates)
    for _ in range(min(BEAM, len(candidates))):
        best_index = -1
        best_value = None
        for i in range(len(candidates)):
            if used[i]:
                continue
            if best_value is None or scores[i] > best_value:
                best_value = scores[i]
                best_index = i
        used[best_index] = True
        beams.append(candidates[best_index])
    best_suffix = beams[0]
    print("grow %d: F=mean N_eff(train)=%.2f  S=%r  (%.0fs)" % (
        step + 1, objective(best_suffix, TRAIN), tok.decode(best_suffix), time.time() - start_time), flush=True)

# Phase 2: coordinate-ascent refinement (revisit each position; keep a change only if F rises).
best_suffix = beams[0]
best_f = objective(best_suffix, TRAIN)
for sweep in range(SWEEPS):
    improved = False
    for position in range(len(best_suffix)):
        best_token = None
        best_token_f = best_f
        for token_id in pool:
            candidate = best_suffix[:position] + [token_id] + best_suffix[position + 1:]
            candidate_f = objective(candidate, TRAIN)
            if candidate_f > best_token_f + 1e-9:
                best_token_f = candidate_f
                best_token = token_id
        if best_token is not None:
            best_suffix = best_suffix[:position] + [best_token] + best_suffix[position + 1:]
            best_f = best_token_f
            improved = True
    print("refine %d: F=%.2f  S=%r  (%.0fs)" % (
        sweep + 1, best_f, tok.decode(best_suffix), time.time() - start_time), flush=True)
    if not improved:
        break

# Conditions to report: random control, single-prompt barycenter suffix, and the robust S*.
conditions = []
conditions.append(("random", rng.sample(VOCAB, K)))
conditions.append(("prev barycenter", encode("奠ឌ奠")))
conditions.append(("robust S*", best_suffix))

print("")
print("%-16s %-5s | %13s %10s %7s %11s" % (
    "condition", "split", "median dN_eff", "min dN_eff", "frac up", "median dNLL"))

# Split boundaries by index: TRAIN = DATA[0:8], TEST = DATA[8:].
splits = [("train", 0, 8), ("test", 8, len(DATA))]
out = {"S_star": best_suffix, "text": tok.decode(best_suffix), "conditions": {}}
for name, suffix_ids in conditions:
    out["conditions"][name] = {}
    for split_name, lo, hi in splits:
        delta_neff = []
        delta_nll = []
        for i in range(lo, hi):
            context_ids, reference_ids = DATA[i]
            base_neff, _, base_nll = baseline[i]
            n_eff, _, nll = measure(context_ids, suffix_ids, reference_ids)
            delta_neff.append(n_eff - base_neff)
            delta_nll.append(nll - base_nll)

        delta_neff.sort()
        count = len(delta_neff)
        median_neff = delta_neff[count // 2]
        min_neff = delta_neff[0]
        rising = 0
        for value in delta_neff:
            if value > 0:
                rising += 1
        frac_up = rising / count
        sorted_nll = sorted(delta_nll)
        median_nll = sorted_nll[count // 2]

        out["conditions"][name][split_name] = {
            "median_dNeff": median_neff, "min_dNeff": min_neff,
            "frac_up": frac_up, "median_dNLL": median_nll,
        }
        print("%-16s %-5s | %13.2f %10.2f %7.2f %11.3f" % (
            name, split_name, median_neff, min_neff, frac_up, median_nll))

out_file = open("runs/flatten_robust.json", "w")
json.dump(out, out_file, indent=2, ensure_ascii=False)
out_file.close()
print("")
print("robust = raises N_eff on unseen TEST contexts (median high, min>0, frac up=1).")
print("saved (%.0fs)" % (time.time() - start_time))
