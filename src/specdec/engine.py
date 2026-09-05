"""Speculative decoding, written out rather than called.

The idea in one paragraph: a small draft model proposes k tokens
autoregressively, which is cheap; the large target model then verifies all k in
a single forward pass, which costs about the same as generating one token
because decoding is memory-bound rather than compute-bound. Every accepted token
is a token the target never had to generate serially. Rejections cost the draft
work and nothing else, because the target's own prediction at the first mismatch
is used, so the output distribution is unchanged.

Two things this implementation is careful about, since they are where
from-scratch versions usually go wrong:

  KV cache trimming   after a rejection the cache for both models holds tokens
                      that are no longer part of the sequence, and they have to
                      be dropped or the next step conditions on tokens the
                      sequence does not contain
  the free token      the target's forward pass over k draft tokens also yields
                      a prediction for the position after the last accepted one,
                      so a step that accepts all k tokens actually yields k+1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


@dataclass
class StepRecord:
    """One draft-and-verify round, kept so the analysis can look inside."""
    proposed: int
    accepted: int
    draft_seconds: float
    verify_seconds: float
    position_accepted: list[bool] = field(default_factory=list)


@dataclass
class Result:
    text: str
    token_ids: list[int]
    steps: list[StepRecord]
    seconds: float

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def acceptance_rate(self) -> float:
        p = sum(s.proposed for s in self.steps)
        return sum(s.accepted for s in self.steps) / p if p else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.n_tokens / self.seconds if self.seconds else 0.0

    @property
    def draft_seconds(self) -> float:
        return sum(s.draft_seconds for s in self.steps)

    @property
    def verify_seconds(self) -> float:
        return sum(s.verify_seconds for s in self.steps)


def _trim(cache: DynamicCache, length: int) -> None:
    """Drop cached keys and values past `length`.

    Verification runs the target over tokens that may be rejected, so its cache
    ends up holding entries for tokens that leave the sequence. Leaving them
    there is the classic silent bug: generation continues, output looks
    plausible, and the model is attending to tokens that were never emitted.
    """
    if hasattr(cache, "crop"):
        # transformers is moving crop() to a negative "remove this many" form;
        # a positive "truncate to" argument is deprecated and goes away in 5.18
        current = cache.get_seq_length() if hasattr(cache, "get_seq_length") else None
        if current is None:
            cache.crop(length)
        elif current > length:
            cache.crop(length - current)   # negative: drop the excess
        return
    for layer in range(len(cache.key_cache)):           # older transformers
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :length]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :length]


@torch.inference_mode()
def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> Result:
    """Ordinary autoregressive decoding from the target model.

    The reference both for speed and for output: speculative decoding with
    greedy verification must produce exactly this text, and the benchmark
    asserts that rather than assuming it.
    """
    device = model.device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()
    out_ids: list[int] = []

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    cur = ids
    for _ in range(max_new_tokens):
        logits = model(cur, past_key_values=cache, use_cache=True).logits
        nxt = int(logits[0, -1].argmax())
        out_ids.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[nxt]], device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None

    return Result(tokenizer.decode(out_ids, skip_special_tokens=True), out_ids, [],
                  time.perf_counter() - t0)


@torch.inference_mode()
def generate_speculative(target, draft, tokenizer, prompt: str, k: int = 4,
                         max_new_tokens: int = 128) -> Result:
    """Speculative decoding with greedy verification.

    Greedy rather than the stochastic rejection-sampling rule, deliberately: at
    temperature 0 the acceptance criterion is "did the draft pick the same token
    the target would have", which makes the output provably identical to
    `generate_baseline` and turns correctness into an assertion instead of a
    distributional argument.

    The bookkeeping to be careful about is how many tokens each cache actually
    holds. A rejection leaves both caches containing keys for tokens that are no
    longer in the sequence, and the two models end up at different lengths
    because the draft ran ahead. Rather than infer those lengths, they are
    tracked explicitly (`t_len`, `d_len`) and each model is fed exactly the
    tokens it has not seen.
    """
    device = target.device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    tokens: list[int] = list(prompt_ids)   # the sequence that actually exists
    t_cache, d_cache = DynamicCache(), DynamicCache()
    t_len = d_len = 0                      # tokens whose KV is cached
    steps: list[StepRecord] = []

    def run(model, cache, ids: list[int]):
        x = torch.tensor([ids], device=device)
        return model(x, past_key_values=cache, use_cache=True).logits[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    while len(tokens) - prompt_len < max_new_tokens:
        base_len = len(tokens)

        # ---- draft k tokens, feeding only what the draft has not seen
        td = time.perf_counter()
        logits = run(draft, d_cache, tokens[d_len:])
        d_len = base_len
        proposals: list[int] = []
        for i in range(k):
            nxt = int(logits[-1].argmax())
            proposals.append(nxt)
            if i < k - 1:                  # the last proposal needs no follow-up pass
                logits = run(draft, d_cache, [nxt])
                d_len += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        draft_s = time.perf_counter() - td

        # ---- verify all k in a single target pass
        tv = time.perf_counter()
        v_in = tokens[t_len:] + proposals
        v_logits = run(target, t_cache, v_in)
        t_len = base_len + k
        # position of the first proposal inside v_in; the target's prediction
        # for proposal i is the logit at the position immediately before it
        first = len(v_in) - k
        target_choice = [int(v_logits[first - 1 + i].argmax()) for i in range(k + 1)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        verify_s = time.perf_counter() - tv

        # ---- accept the longest prefix the target agrees with
        accepted, per_position = 0, []
        for i, prop in enumerate(proposals):
            ok = target_choice[i] == prop
            per_position.append(ok)
            if not ok:
                break
            accepted += 1

        # the target's own prediction at the first disagreement is already
        # computed and correct, so a fully accepted window yields k+1 tokens
        emitted = proposals[:accepted] + [target_choice[accepted]]
        tokens.extend(emitted)
        steps.append(StepRecord(len(proposals), accepted, draft_s, verify_s, per_position))

        # ---- drop cached keys for tokens that did not survive verification.
        # The free token is in neither cache, so both stop one short of the
        # sequence and are fed it on the next pass.
        t_len = base_len + accepted
        d_len = min(d_len, base_len + accepted)
        _trim(t_cache, t_len)
        _trim(d_cache, d_len)

        if tokenizer.eos_token_id in emitted:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    out_ids = tokens[prompt_len:][:max_new_tokens]
    return Result(tokenizer.decode(out_ids, skip_special_tokens=True), out_ids, steps, seconds)


def break_even_acceptance(draft_cost: float, target_cost: float, k: int) -> float:
    """The acceptance rate below which speculation is a net loss.

    A speculative step costs k draft passes plus one target pass and yields
    1 + k*a tokens, where a is the acceptance rate. Plain decoding costs one
    target pass per token. Setting the two equal and solving for a gives the
    threshold, which is why a fast draft model matters more than an accurate
    one up to a point.
    """
    r = draft_cost / target_cost
    denom = k * (1.0 - k * r) if k * r < 1 else 0.0
    return float("inf") if denom <= 0 else max(0.0, (k * r) / denom)
