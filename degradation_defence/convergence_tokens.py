#!/usr/bin/env python3
"""Rank candidate convergence tokens U for the stage-2 cycle (flatten + converge).

For each single-token U: C + S_flat + U^R, measure p(U) at hand-off, and the lock
(fraction of G generated tokens == U) greedy / T=0.7 / T=1.0, averaged over contexts.
A good convergence token is a strong self-reinforcing attractor: high lock under
sampling. Output a ranked list. S_flat is loaded from flatten_robust.json.

Style note: plain, explicit control flow (classic for-loops, explicit accumulators,
no comprehensions or lambdas), Java-like. Only the tensor ops are vectorized.
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
torch.manual_seed(0)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()


def encode(text):
    return tok.encode(text, add_special_tokens=False)


# Stage-1 flattener. Prefer the saved search result; fall back to the known sequence.
S_FLAT = [69816, 116755, 58115, 113447, 123796, 144928]   # ' ivory原著 Hond第四届ס뀔'
try:
    saved = json.load(open("runs/flatten_robust.json"))
    S_FLAT = saved["S_star"]
except (OSError, KeyError, ValueError):
    pass

CTX = [
    "The largest planet in our solar system is",
    "A decade is a period of ten",
    "The human heart has four",
]

R = 32            # cycle repetitions
G = 40            # generated tokens
NUM_SAMPLES = 8   # samples per sampled temperature

# Candidate convergence tokens; keep only those that map to a single token.
CANDIDATES = [" of", " when", " the", " and", ".", "\n", " end"]
U_LIST = []
for text in CANDIDATES:
    ids = encode(text)
    if len(ids) == 1:
        U_LIST.append((text, ids[0]))


@torch.inference_mode()
def prob_handoff(input_ids, target_id):
    """Probability of the target token at the first generated step."""
    out = model(input_ids=torch.tensor([input_ids]), logits_to_keep=1)
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    return float(probs[target_id])


@torch.inference_mode()
def lock(input_ids, target_id, temperature):
    """Fraction of G generated tokens equal to the target.
    temperature == 0 is greedy; otherwise mean over NUM_SAMPLES sampled runs."""
    if temperature == 0:
        out = model(input_ids=torch.tensor([input_ids]), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
        hits = 0
        for _ in range(G):
            if next_id == target_id:
                hits += 1
            out = model(input_ids=torch.tensor([[next_id]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
        return hits / G

    batch = []
    for _ in range(NUM_SAMPLES):
        batch.append(list(input_ids))
    out = model(input_ids=torch.tensor(batch), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    hits = torch.zeros(NUM_SAMPLES)
    probs = torch.softmax(out.logits[:, -1].float() / temperature, -1)
    next_ids = torch.multinomial(probs, 1)[:, 0]
    for _ in range(G):
        hits += (next_ids == target_id).float()
        out = model(input_ids=next_ids[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        probs = torch.softmax(out.logits[:, -1].float() / temperature, -1)
        next_ids = torch.multinomial(probs, 1)[:, 0]
    return float((hits / G).mean())


# Measure every candidate, averaged over contexts.
rows = []
for text, target_id in U_LIST:
    sum_p = 0.0
    sum_greedy = 0.0
    sum_t07 = 0.0
    sum_t10 = 0.0
    for context in CTX:
        input_ids = encode(context) + S_FLAT
        for _ in range(R):
            input_ids = input_ids + [target_id]
        sum_p += prob_handoff(input_ids, target_id)
        sum_greedy += lock(input_ids, target_id, 0)
        sum_t07 += lock(input_ids, target_id, 0.7)
        sum_t10 += lock(input_ids, target_id, 1.0)
    count = len(CTX)
    row = {
        "token": text, "id": target_id,
        "p_handoff": sum_p / count, "greedy": sum_greedy / count,
        "T0.7": sum_t07 / count, "T1.0": sum_t10 / count,
    }
    rows.append(row)
    print("  %-10r p=%.3f greedy=%.2f T0.7=%.2f T1.0=%.2f" % (
        text, row["p_handoff"], row["greedy"], row["T0.7"], row["T1.0"]), flush=True)

# Rank by sampling robustness (T=0.7 + T=1.0), using repeated maximum selection (no lambda).
ranked = []
used = [False] * len(rows)
for _ in range(len(rows)):
    best_index = -1
    best_score = None
    for i in range(len(rows)):
        if used[i]:
            continue
        score = rows[i]["T0.7"] + rows[i]["T1.0"]
        if best_score is None or score > best_score:
            best_score = score
            best_index = i
    used[best_index] = True
    ranked.append(rows[best_index])

print("")
print("=== ranked by sampling robustness (T0.7 + T1.0) ===")
for row in ranked:
    print("  %-10r T0.7=%.2f T1.0=%.2f greedy=%.2f p=%.3f" % (
        row["token"], row["T0.7"], row["T1.0"], row["greedy"], row["p_handoff"]))

out_file = open("runs/convergence_tokens.json", "w")
json.dump(ranked, out_file, indent=2, ensure_ascii=False)
out_file.close()
print("")
print("saved runs/convergence_tokens.json")
