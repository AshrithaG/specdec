#!/usr/bin/env python3
"""Correctness tests that need no GPU and no large model.

Two properties pin the implementation down:

  self-speculation   when draft and target are the same model, every proposal
                     must be accepted, so acceptance is 1.0. Anything less means
                     the caches have drifted out of sync.
  output identity    greedy verification must reproduce plain decoding exactly,
                     for any k.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from specdec.engine import generate_baseline, generate_speculative  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("model", nargs="?", default="Qwen/Qwen3-0.6B")
_ap.add_argument("--device", default="cpu")
_ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
_args = _ap.parse_args()
MODEL = _args.model
PROMPTS = [
    "List three prime numbers.",
    "The capital of France is",
    "def add(a, b):",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=getattr(torch, _args.dtype)).to(_args.device).eval()
print(f"{MODEL} on {model.device} in {_args.dtype}\n")
print("Self-speculation: draft and target are the same weights, so in exact")
print("arithmetic every proposal must be accepted. Anything below 1.000 is the")
print("model disagreeing with itself, which can only come from the numerics of")
print("scoring k+1 tokens in one pass versus one token at a time.\n")

ok = True
for prompt in PROMPTS:
    base = generate_baseline(model, tok, prompt, max_new_tokens=24)
    print(f"prompt: {prompt!r}")
    for k in (1, 2, 4, 5):
        spec = generate_speculative(model, model, tok, prompt, k=k, max_new_tokens=24)
        identical = spec.token_ids[:len(base.token_ids)] == base.token_ids[:len(spec.token_ids)]
        # self-speculation: the draft is the target, so nothing should be rejected
        full_accept = spec.acceptance_rate > 0.999
        flag = "" if (identical and full_accept) else "   <-- FAIL"
        print(f"  k={k}  acceptance {spec.acceptance_rate:.3f}  identical {identical}{flag}")
        ok = ok and identical and full_accept
    print()

print("PASS" if ok else "FAIL: implementation is not equivalent to plain decoding")
sys.exit(0 if ok else 1)
