#!/usr/bin/env python3
"""Why does a batched forward disagree with a sequential one?

Speculative decoding scores k+1 tokens in one pass; plain decoding scores them
one at a time. In exact arithmetic these are the same computation. In bf16 they
are not, and the output diverges. This measures the two places that can happen
and tests a specific claim about when it matters.

  logit path    same starting cache, same tokens, one batched pass versus N
                sequential passes. Any difference here is the forward itself:
                different matmul shapes take different reduction orders, and
                floating-point addition is not associative.

  cache path    the keys and values each path writes. Once these differ, every
                later token is conditioned on a different state, which is why a
                divergence can surface long after the pass that caused it.

The claim under test: an argmax flip requires the top-two logit gap to be
smaller than the numerical disagreement between the paths. If flips only occur
below that threshold, the mechanism is pinned down rather than asserted.

    python scripts/mechanism.py --dtype bfloat16 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROMPTS = [
    "List three prime numbers.", "The capital of France is", "def add(a, b):",
    "Explain why the sky is blue in one sentence.", "Write a haiku about winter.",
    "What is 17 times 23?", "Name the largest ocean on Earth.",
    "Summarise the plot of Hamlet in two sentences.",
    "Describe the water cycle.", "Who wrote Pride and Prejudice?",
]


@torch.inference_mode()
def prime(model, ids: list[int]):
    cache = DynamicCache()
    x = torch.tensor([ids], device=model.device)
    out = model(x, past_key_values=cache, use_cache=True)
    return cache, out.logits[0, -1]


@torch.inference_mode()
def sequential_logits(model, cache, ids: list[int]) -> torch.Tensor:
    """Feed one token at a time; return the logit row produced at each."""
    rows = []
    for t in ids:
        x = torch.tensor([[t]], device=model.device)
        rows.append(model(x, past_key_values=cache, use_cache=True).logits[0, -1])
    return torch.stack(rows)


@torch.inference_mode()
def batched_logits(model, cache, ids: list[int]) -> torch.Tensor:
    """Feed all tokens in one pass; return the logit row at each position."""
    x = torch.tensor([ids], device=model.device)
    return model(x, past_key_values=cache, use_cache=True).logits[0]


def cache_max_diff(a: DynamicCache, b: DynamicCache) -> float:
    """Largest disagreement between two caches holding the same tokens."""
    worst = 0.0
    ka = a.key_cache if hasattr(a, "key_cache") else [l.keys for l in a.layers]
    kb = b.key_cache if hasattr(b, "key_cache") else [l.keys for l in b.layers]
    va = a.value_cache if hasattr(a, "value_cache") else [l.values for l in a.layers]
    vb = b.value_cache if hasattr(b, "value_cache") else [l.values for l in b.layers]
    for x, y in list(zip(ka, kb)) + list(zip(va, vb)):
        n = min(x.shape[-2], y.shape[-2])
        worst = max(worst, float((x[..., :n, :].float() - y[..., :n, :].float()).abs().max()))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--window", type=int, default=8, help="tokens scored per comparison")
    ap.add_argument("--steps", type=int, default=6, help="windows per prompt")
    ap.add_argument("--out", type=Path, default=Path("results/mechanism.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    print(f"{args.model}  {args.device}  {args.dtype}  window={args.window}\n")

    logit_diffs, margins, flips, cache_diffs = [], [], [], []

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        # a fixed continuation, so both paths score exactly the same tokens
        gen = model.generate(torch.tensor([ids], device=model.device),
                             max_new_tokens=args.window * args.steps,
                             do_sample=False, pad_token_id=tok.eos_token_id)[0].tolist()
        cont = gen[len(ids):]

        for s in range(args.steps):
            chunk = cont[s * args.window:(s + 1) * args.window]
            if len(chunk) < args.window:
                break
            ctx = ids + cont[: s * args.window]
            seq_cache, _ = prime(model, ctx)
            bat_cache, _ = prime(model, ctx)

            seq = sequential_logits(model, seq_cache, chunk).float()
            bat = batched_logits(model, bat_cache, chunk).float()

            d = (seq - bat).abs().max(dim=-1).values          # per position
            top2 = seq.topk(2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1])                # how close the race is
            flip = seq.argmax(-1) != bat.argmax(-1)

            logit_diffs += d.tolist()
            margins += margin.tolist()
            flips += flip.tolist()
            cache_diffs.append(cache_max_diff(seq_cache, bat_cache))

    d = np.array(logit_diffs); m = np.array(margins); f = np.array(flips, dtype=bool)
    print(f"{len(d)} scored positions, {f.sum()} argmax flips ({f.mean():.1%})\n")
    print("numerical disagreement between the two paths (max |logit difference|)")
    print(f"  median {np.median(d):.5f}   p95 {np.percentile(d,95):.5f}   max {d.max():.5f}")
    print("\nKV cache disagreement after scoring the same tokens both ways")
    print(f"  median {np.median(cache_diffs):.6f}   max {np.max(cache_diffs):.6f}")

    print("\nflip rate by how close the top-two race is")
    print(f"{'top1 - top2 margin':>22}{'positions':>11}{'flips':>8}{'flip rate':>11}")
    edges = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m >= lo) & (m < hi)
        if not sel.any():
            continue
        label = f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}"
        print(f"{label:>22}{int(sel.sum()):>11}{int(f[sel].sum()):>8}{f[sel].mean():>10.1%}")

    if f.any():
        print(f"\nlargest margin that still flipped: {m[f].max():.5f}")
        print(f"p99 of the numerical disagreement:  {np.percentile(d, 99):.5f}")
        print(f"flips with margin above p99 disagreement: "
              f"{int((f & (m > np.percentile(d, 99))).sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"logit_diff": d.tolist(), "margin": m.tolist(), "flip": f.tolist(),
         "cache_diff": cache_diffs, "dtype": args.dtype, "model": args.model}, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
