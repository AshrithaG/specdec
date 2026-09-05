#!/usr/bin/env python3
"""Speculative decoding: does it pay, by how much, and where does it stop paying.

    python scripts/benchmark.py --target Qwen/Qwen3-4B-Instruct-2507 \
        --draft Qwen/Qwen3-0.6B --k 1 2 4 6 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from specdec.engine import (break_even_acceptance, generate_baseline,  # noqa: E402
                            generate_speculative)

# Deliberately mixed. Acceptance rate is not a property of a model pair alone;
# it is a property of the pair and the text. Code and structured output are
# predictable and should draft well; open-ended prose should draft worst.
PROMPTS = {
    "code": [
        "Write a Python function that reverses a linked list. Return only code.",
        "Write a SQL query that finds the second highest salary per department.",
        "Implement binary search over a sorted list in Python.",
    ],
    "factual": [
        "What is the capital of Australia, and when did it become the capital?",
        "Explain what a B-tree index is and why databases use it.",
        "What causes the seasons on Earth?",
    ],
    "reasoning": [
        "A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph on "
        "the same track. When does the second catch the first? Show your work.",
        "If all bloops are razzies and all razzies are lazzies, are all bloops "
        "lazzies? Explain.",
    ],
    "creative": [
        "Write the opening paragraph of a story about a lighthouse keeper.",
        "Describe a city at dawn, in three sentences, without using the word light.",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"loading target {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loading draft  {args.draft}")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=torch.bfloat16, device_map="cuda").eval()

    flat = [(d, p) for d, ps in PROMPTS.items() for p in ps]
    report: dict = {"target": args.target, "draft": args.draft, "by_k": {}, "baseline": {}}

    # ---- baseline, and the correctness reference
    print("\nbaseline (target only)")
    base_out, base_tps = {}, []
    for domain, prompt in flat:
        r = generate_baseline(target, tok, prompt, args.max_new)
        base_out[prompt] = r.text
        base_tps.append(r.tokens_per_second)
        print(f"  {domain:<10} {r.tokens_per_second:6.1f} tok/s  ({r.n_tokens} tokens)")
    report["baseline"] = {"tokens_per_second": float(np.mean(base_tps))}
    print(f"  mean {np.mean(base_tps):.1f} tok/s")

    # ---- cost ratio, needed for the break-even threshold
    warm = generate_speculative(target, draft, tok, flat[0][1], k=4, max_new_tokens=32)
    per_draft = warm.draft_seconds / max(sum(s.proposed for s in warm.steps), 1)
    per_verify = warm.verify_seconds / max(len(warm.steps), 1)
    ratio = per_draft / per_verify
    report["cost_ratio"] = {"draft_per_token_s": per_draft,
                            "target_per_pass_s": per_verify, "ratio": ratio}
    print(f"\ndraft pass costs {ratio:.3f} of a target pass")

    for k in args.k:
        print(f"\nk = {k}")
        rows, mismatches = [], 0
        by_domain: dict[str, list[float]] = {}
        pos_hits = np.zeros(k); pos_total = np.zeros(k)
        for domain, prompt in flat:
            r = generate_speculative(target, draft, tok, prompt, k=k,
                                     max_new_tokens=args.max_new)
            if r.text.strip() != base_out[prompt].strip():
                mismatches += 1
            for s in r.steps:
                for i, ok in enumerate(s.position_accepted):
                    pos_total[i] += 1
                    pos_hits[i] += bool(ok)
            by_domain.setdefault(domain, []).append(r.acceptance_rate)
            rows.append({"domain": domain, "acceptance": r.acceptance_rate,
                         "tok_s": r.tokens_per_second,
                         "draft_frac": r.draft_seconds / max(r.seconds, 1e-9)})
        acc = float(np.mean([x["acceptance"] for x in rows]))
        tps = float(np.mean([x["tok_s"] for x in rows]))
        speedup = tps / report["baseline"]["tokens_per_second"]
        thresh = break_even_acceptance(per_draft, per_verify, k)
        print(f"  acceptance {acc:.3f}   {tps:6.1f} tok/s   {speedup:.2f}x baseline"
              f"   break-even acceptance {thresh:.3f}"
              f"   output identical: {len(flat)-mismatches}/{len(flat)}")
        for d, v in sorted(by_domain.items()):
            print(f"    {d:<10} acceptance {np.mean(v):.3f}")
        print("    acceptance by position in the draft window: "
              + " ".join(f"{h/t:.2f}" if t else "-" for h, t in zip(pos_hits, pos_total)))
        report["by_k"][str(k)] = {
            "acceptance": acc, "tokens_per_second": tps, "speedup": speedup,
            "break_even_acceptance": thresh, "identical_outputs": len(flat) - mismatches,
            "by_domain": {d: float(np.mean(v)) for d, v in by_domain.items()},
            "acceptance_by_position": [float(h / t) if t else None
                                       for h, t in zip(pos_hits, pos_total)],
            "rows": rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
