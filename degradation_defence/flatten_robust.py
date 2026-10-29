#!/usr/bin/env python3
"""Gradient-guided search for a flattening sequence S (push softmax toward the barycenter).

Objective (maximize), over 8 training prompts C_j:

    F_j(S) = N_eff( softmax( z(C_j + S) ) ) = 1 / sum_i p_{j,i}^2        (last position)
    F(S)   = (1/8) * sum_j F_j(S)

Search (gradient shortlist + exact verification):
  1. initialize a sequence of SEQ_LEN tokens;
  2. compute, per training prompt, the gradient g_{j,r} = d F_j / d e_r at every suffix
     position r (one backward per prompt gives all positions);
  3. for each position r, score every vocabulary token v by the predicted first-order
     gain of swapping s_r -> v,  g_{j,r} . (E_v - E_{s_r}), combined over prompts by the
     MEAN (average predicted gain) or the WORST case (min over prompts, for robustness);
  4. verify the top candidates with real forward passes (exact F);
  5. accept a substitution only if F(S) increases;
  6. repeat sweeps until no position improves (plateau).

The gradient shortlist avoids exploring the whole vocabulary at each position. A plateau
signals saturation of the search, not necessarily the geometric maximum.

Style note: plain, explicit control flow (classic for-loops, explicit accumulators,
no comprehensions or lambdas), Java-like. Only the tensor ops are vectorized.
"""
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

# Freeze the model; gradients are taken only with respect to the input embeddings.
for parameter in model.parameters():
    parameter.requires_grad_(False)
EMB = model.get_input_embeddings().weight          # |V| x d


def encode(text):
    return tok.encode(text, add_special_tokens=False)


# Candidate vocabulary: every id that is not a special token.
specials = set(tok.all_special_ids)
vocab_limit = min(len(tok), model.config.vocab_size)
VOCAB = []
for token_id in range(vocab_limit):
    if token_id not in specials:
        VOCAB.append(token_id)
VOCAB_TENSOR = torch.tensor(VOCAB)
EMB_VOCAB = EMB[VOCAB_TENSOR].detach()             # |V'| x d, candidate embeddings

TRAIN_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
    "A triangle has three",
    "Ice is the solid form of",
    "The square root of nine is",
]
TEST_PROMPTS = [
    "The Earth orbits around the",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "The freezing point of water is zero degrees",
    "A hexagon is a shape with six",
    "The Pacific is the largest",
    "A decade is a period of ten",
    "The human heart has four",
]

SEQ_LEN = 24          # sequence length to search (the specification uses 24)
TOPK = 32             # candidates verified with real forwards, per position
MAX_SWEEPS = 3        # coordinate sweeps over the sequence
CRITERION = "worst"   # "mean" = average predicted gain, "worst" = min over prompts

TRAIN = []
for prompt in TRAIN_PROMPTS:
    TRAIN.append(encode(prompt))
TEST = []
for prompt in TEST_PROMPTS:
    TEST.append(encode(prompt))


@torch.inference_mode()
def neff_many(context_ids, suffix_list):
    """N_eff of the final-position distribution for each suffix (all equal length), batched."""
    batch = []
    for suffix in suffix_list:
        batch.append(context_ids + suffix)
    out = model(input_ids=torch.tensor(batch), use_cache=False)
    last = out.logits[:, -1].double()
    probs = torch.softmax(last, -1)
    collision = (probs * probs).sum(-1)
    values = (1.0 / collision)
    result = []
    for i in range(len(suffix_list)):
        result.append(float(values[i]))
    return result


def objective_many(suffix_list, contexts):
    """F = mean over the given contexts of N_eff, for each suffix. Returns a list."""
    totals = [0.0] * len(suffix_list)
    for context_ids in contexts:
        per_suffix = neff_many(context_ids, suffix_list)
        for i in range(len(suffix_list)):
            totals[i] += per_suffix[i]
    means = []
    for i in range(len(suffix_list)):
        means.append(totals[i] / len(contexts))
    return means


def objective(suffix, contexts):
    return objective_many([suffix], contexts)[0]


def gradients_per_prompt(suffix):
    """For each training prompt, one backward gives d N_eff / d e_r at every suffix
    position. Returns a list (one entry per prompt) of tensors shaped [SEQ_LEN, d]."""
    per_prompt = []
    for context_ids in TRAIN:
        full_ids = context_ids + suffix
        embedded = EMB[torch.tensor([full_ids])].detach().clone()
        embedded.requires_grad_(True)
        out = model(inputs_embeds=embedded, use_cache=False)
        last = out.logits[0, -1].float()
        probs = torch.softmax(last, 0)
        n_eff = 1.0 / (probs * probs).sum()
        n_eff.backward()
        suffix_start = len(context_ids)
        grad_rows = embedded.grad[0, suffix_start:suffix_start + len(suffix)].detach()
        per_prompt.append(grad_rows)
    return per_prompt


def shortlist(suffix, position, grads, k):
    """Top-k vocabulary ids by predicted first-order gain of swapping suffix[position]."""
    base_vec = EMB[suffix[position]].detach()           # d
    combined = None
    for grad_rows in grads:
        direction = grad_rows[position].detach()        # d
        # predicted gain for every candidate v: (E_v - E_base) . direction
        gain = torch.matmul(EMB_VOCAB, direction) - torch.dot(base_vec, direction)
        if combined is None:
            if CRITERION == "worst":
                combined = gain.clone()
            else:
                combined = gain.clone()
        else:
            if CRITERION == "worst":
                combined = torch.minimum(combined, gain)
            else:
                combined = combined + gain
    if CRITERION != "worst":
        combined = combined / len(grads)
    top = torch.topk(combined, k).indices
    ids = []
    for i in range(top.shape[0]):
        ids.append(VOCAB[int(top[i])])
    return ids


# 1. Initialize a random sequence of SEQ_LEN tokens.
suffix = rng.sample(VOCAB, SEQ_LEN)
best_f = objective(suffix, TRAIN)
start_time = time.time()
progression = [best_f]
print("init: F=%.2f  (%.0fs)" % (best_f, time.time() - start_time), flush=True)

# 5-6. Coordinate sweeps: gradient shortlist, exact verification, accept gains, until plateau.
for sweep in range(MAX_SWEEPS):
    improved = False
    grads = gradients_per_prompt(suffix)          # 2. gradients on the 8 prompts
    for position in range(SEQ_LEN):
        candidate_ids = shortlist(suffix, position, grads, TOPK)   # 3. candidates
        trial_suffixes = []
        for token_id in candidate_ids:
            trial = suffix[:position] + [token_id] + suffix[position + 1:]
            trial_suffixes.append(trial)
        trial_f = objective_many(trial_suffixes, TRAIN)            # 4. exact forwards
        best_index = -1
        best_value = best_f
        for i in range(len(trial_f)):
            if trial_f[i] > best_value + 1e-9:
                best_value = trial_f[i]
                best_index = i
        if best_index >= 0:                                        # 5. accept if F rises
            suffix = trial_suffixes[best_index]
            best_f = best_value
            improved = True
    progression.append(best_f)
    print("sweep %d: F=%.2f  S=%r  (%.0fs)" % (
        sweep + 1, best_f, tok.decode(suffix), time.time() - start_time), flush=True)
    if not improved:                                               # 6. plateau -> stop
        print("plateau reached (no position improved)", flush=True)
        break

# Report: does the sequence raise N_eff on held-out TEST prompts too?
print("")
print("final F (train) = %.2f" % best_f)
print("F progression  = %s" % ", ".join("%.1f" % value for value in progression))

base_train = objective([], TRAIN)
base_test = objective([], TEST)
flat_train = objective(suffix, TRAIN)
flat_test = objective(suffix, TEST)
print("")
print("%-6s | %12s %12s" % ("split", "baseline N_eff", "with S N_eff"))
print("%-6s | %12.2f %12.2f" % ("train", base_train, flat_train))
print("%-6s | %12.2f %12.2f" % ("test", base_test, flat_test))

out = {
    "S_star": suffix,
    "text": tok.decode(suffix),
    "criterion": CRITERION,
    "seq_len": SEQ_LEN,
    "F_train": best_f,
    "F_progression": progression,
    "baseline_train": base_train, "baseline_test": base_test,
    "flat_train": flat_train, "flat_test": flat_test,
}
out_file = open("runs/flatten_robust.json", "w")
json.dump(out, out_file, indent=2, ensure_ascii=False)
out_file.close()
print("")
print("saved runs/flatten_robust.json  (%.0fs)" % (time.time() - start_time))
