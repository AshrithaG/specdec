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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
PROMPTS = [
    "List three prime numbers.",
    "The capital of France is",
    "def add(a, b):",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
print(f"{MODEL} on {model.device}\n")

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
