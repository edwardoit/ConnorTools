#!/usr/bin/env python3
"""Two-stage: S_flat pushes logits to the barycenter, then S_conv forces solid
convergence onto ONE repeated token. Does flattening first make convergence more solid?

For each context C and target token U (repeated R times as the converging cycle):
  cond A: C + U^R            (converge only)
  cond B: C + S_flat + U^R   (flatten first, then converge)
Measure: p(U) at the first generated step (handoff), greedy lock (frac of G generated
tokens == U), and sampling robustness at T=0.7 and T=1.0 (mean frac == U over samples).

Style note: written with plain, explicit control flow (classic for-loops, explicit
accumulators, no comprehensions or lambdas), Java-like.
"""
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
torch.manual_seed(0)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()


def encode(text):
    return tok.encode(text, add_special_tokens=False)


# Stage-1 flattener (output of flatten_robust.py, refined; mean N_eff over train F=402.7).
# Re-run flatten_robust.py to regenerate it, or override the token ids below.
S_FLAT = [69816, 116755, 58115, 113447, 123796, 144928]   # ' ivory原著 Hond第四届ס뀔'

CTX = [
    "The largest planet in our solar system is",
    "A decade is a period of ten",
    "The human heart has four",
]

# Convergence tokens to test (name, single token id).
TARGETS = [
    ("' the'", encode(" the")[0]),
    ("' end'", encode(" end")[0]),
]
REPETITIONS = [8, 32]
G = 40            # generated tokens
NUM_SAMPLES = 6   # samples per sampled temperature


@torch.inference_mode()
def prob_handoff(input_ids, target_id):
    """Probability of the target token at the first generated step."""
    out = model(input_ids=torch.tensor([input_ids]), use_cache=False, logits_to_keep=1)
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    return float(probs[target_id])


@torch.inference_mode()
def greedy_lock(input_ids, target_id):
    """Fraction of G greedily generated tokens equal to the target."""
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


@torch.inference_mode()
def sampled_lock(input_ids, target_id, temperature):
    """Mean fraction of G sampled tokens equal to the target, over NUM_SAMPLES runs."""
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


def average(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


print("%-7s %3s %-18s | %12s %11s %10s %10s" % (
    "target", "R", "cond", "p(U) handoff", "greedy lock", "T=0.7 lock", "T=1.0 lock"))

conditions = [("converge only", []), ("flatten+converge", S_FLAT)]
for target_name, target_id in TARGETS:
    for repetitions in REPETITIONS:
        for cond_name, prefix in conditions:
            handoff_values = []
            greedy_values = []
            t07_values = []
            t10_values = []
            for context in CTX:
                input_ids = encode(context) + prefix
                for _ in range(repetitions):
                    input_ids = input_ids + [target_id]
                handoff_values.append(prob_handoff(input_ids, target_id))
                greedy_values.append(greedy_lock(input_ids, target_id))
                t07_values.append(sampled_lock(input_ids, target_id, 0.7))
                t10_values.append(sampled_lock(input_ids, target_id, 1.0))
            print("%-7s %3d %-18s | %12.3f %11.2f %10.2f %10.2f" % (
                target_name, repetitions, cond_name,
                average(handoff_values), average(greedy_values),
                average(t07_values), average(t10_values)), flush=True)
