#!/usr/bin/env python3
"""Where does speculative decoding diverge from plain decoding, and why?

Greedy speculative decoding is advertised as lossless. That proof assumes exact
arithmetic. This locates the first token where the two disagree and reports
whether it was a *verified* token (a draft proposal the target confirmed) or the
*free* token (the target's own prediction at the end of the verification batch,
which nothing checks).

If divergences land on free tokens, the losslessness guarantee is leaking
through the one position the algorithm cannot verify, and the cause is the
batched forward giving a different argmax than the sequential one.

    python scripts/diagnose_divergence.py --dtype bfloat16 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from specdec.engine import generate_baseline, generate_speculative  # noqa: E402

PROMPTS = [
    "List three prime numbers.", "The capital of France is", "def add(a, b):",
    "Explain why the sky is blue in one sentence.", "Write a haiku about winter.",
    "What is 17 times 23?", "Name the largest ocean on Earth.",
    "Summarise the plot of Hamlet in two sentences.",
]


def emitted_kinds(result) -> list[str]:
    """Label every emitted token: 'verified' if a draft proposal the target
    confirmed, 'free' if the target's unchecked prediction."""
    kinds = []
    for s in result.steps:
        kinds.extend(["verified"] * s.accepted)
        kinds.append("free")
    return kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    print(f"{args.model}  {args.device}  {args.dtype}")
    print("draft and target are the same weights, so any divergence is arithmetic\n")

    tally = {"verified": 0, "free": 0}
    runs = diverged = 0
    for k in args.k:
        for prompt in PROMPTS:
            base = generate_baseline(model, tok, prompt, args.max_new)
            spec = generate_speculative(model, model, tok, prompt, k=k,
                                        max_new_tokens=args.max_new)
            runs += 1
            n = min(len(base.token_ids), len(spec.token_ids))
            first = next((i for i in range(n)
                          if base.token_ids[i] != spec.token_ids[i]), None)
            if first is None:
                continue
            diverged += 1
            kinds = emitted_kinds(spec)
            kind = kinds[first] if first < len(kinds) else "unknown"
            tally[kind] = tally.get(kind, 0) + 1
            print(f"  k={k} {prompt[:34]!r:38} first divergence at token {first:>3} "
                  f"({kind}), acceptance {spec.acceptance_rate:.3f}")

    print(f"\n{diverged} of {runs} runs diverged from plain decoding")
    print(f"  first divergence was a FREE (unverified) token : {tally.get('free', 0)}")
    print(f"  first divergence was a VERIFIED token          : {tally.get('verified', 0)}")
    if diverged and tally.get("verified", 0) == 0:
        print("\nEvery divergence landed on the one token the algorithm does not check.")


if __name__ == "__main__":
    main()
