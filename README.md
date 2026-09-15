# Speculative decoding, written out and measured

A small draft model proposes k tokens; the large target model verifies all k in
one forward pass and accepts the longest prefix it agrees with. Every accepted
token is one the target never generated serially. Implemented here rather than
imported, and measured on an RTX 4090 with Qwen3-0.6B drafting for
Qwen3-4B-Instruct.

Two results, and neither is the one the technique is usually sold on.

```bash
python scripts/test_correctness.py                       # no GPU needed
python scripts/benchmark.py --k 1 2 4 6 8
python scripts/mechanism.py --dtype bfloat16 --device cuda
```

## It lost, and the reason is a single number

| k | acceptance | tok/s | speedup | predicted by the cost model |
|---|---|---|---|---|
| baseline | | 38.4 | 1.00x | |
| 1 | 0.709 | 33.5 | 0.87x | 1.04x |
| 2 | 0.623 | 31.8 | 0.83x | 0.98x |
| 4 | 0.482 | 26.8 | 0.70x | 0.81x |
| 6 | 0.397 | 22.6 | 0.59x | 0.69x |
| 8 | 0.338 | 19.5 | 0.51x | 0.60x |

Speculation is a net loss at every window size, and gets worse as k grows.

The number that decides it: **a Qwen3-0.6B forward pass costs 0.649 of a
Qwen3-4B pass**, 19.9 ms against 30.7 ms. A model with roughly a seventh of the
parameters costs two-thirds as much per call, because single-token decoding here
is dominated by Python dispatch and kernel launch rather than by arithmetic.

That ratio is the entire decision. A speculative step costs `T(1 + k·r)` and
yields `1 + k·a` tokens, so it breaks even when

```
a = r
```

The window size cancels. Speculation pays exactly when the draft model accepts
more often than it costs, and nothing else matters. Here r is 0.649, so only
k=1 clears the bar at all, at 0.709, and only barely.

The cost model predicts measured throughput to within 0.09 to 0.17x at every k.
The residual is the algorithm's own bookkeeping, cache trimming and tensor
construction and the Python loop, which is enough to turn k=1's predicted 1.04x
into a measured 0.87x. **Speculative decoding needs an inference stack where
small models are genuinely cheap.** In eager PyTorch they are not, which is why
production implementations live inside serving engines rather than in loops like
this one.

## Acceptance is a property of the text, not just the model pair

| k | code | reasoning | factual | creative |
|---|---|---|---|---|
| 1 | 0.80 | 0.83 | 0.69 | 0.49 |
| 4 | 0.60 | 0.60 | 0.45 | 0.23 |
| 8 | 0.42 | 0.46 | 0.31 | 0.12 |

Code and step-by-step reasoning draft well because the next token is often
forced: a closing bracket, a keyword, the rest of an identifier. Open-ended
prose is where a small model and a large one genuinely disagree, and acceptance
collapses to 0.12 at k=8. A deployment serving mostly code completion and one
serving mostly chat are not the same decision.

Acceptance also *rises* with position inside the window (0.68, 0.66, 0.72, 0.78,
0.84, 0.83, 0.79, 0.83 at k=8), which looks backwards until you notice it is
conditional: position i is only reached when every earlier position was
accepted, so the deeper measurements are taken only on stretches that were
already going well.

## The lossless guarantee does not survive bf16

Greedy speculative decoding is provably lossless: it must reproduce the target
model's own output exactly. It does not.

With draft and target set to the **same weights**, so the model cannot disagree
with itself for any logical reason:

| precision | argmax flips | median logit disagreement | KV cache disagreement |
|---|---|---|---|
| float32 | 0 / 480 | 0.00002 | 0.000092 |
| bfloat16 | 5 / 480 | 0.20312 | 1.00 (max 4.06) |

Identical code, identical GPU, identical positions. Only the dtype changes.

The cause is that verification scores k+1 tokens in one pass while plain
decoding scores them one at a time. Different matmul shapes take different
reduction orders, floating-point addition is not associative, and the two paths
end up 0.2 apart in logit space. That only changes the output when two
candidates are closer together than that:

| top1 minus top2 margin | positions | flip rate |
|---|---|---|
| < 0.01 | 6 | **50.0%** |
| 0.1 to 0.25 | 33 | 6.1% |
| 0.25 to 0.5 | 38 | 0% |
| > 0.5 | 403 | 0% |

The largest margin that ever flipped was 0.125, against a 99th-percentile
numerical disagreement of 0.484. **No flip occurred above that threshold**,
which was the prediction set before running it.

I first guessed the leak was confined to the "free" token, the target's own
prediction closing a fully accepted window, since that is the one position
nothing verifies. That was wrong: divergences land on verified tokens too. The
KV cache row above is why. The two paths write caches differing by a median of
1.0 in bf16 against 0.000092 in float32, so once they disagree, every later
token is conditioned on a different state and the divergence surfaces wherever
the next close race happens to be.

None of this makes speculative decoding unsound. It makes "lossless" a statement
about exact arithmetic, and reduced precision is not exact arithmetic.

## Correctness

`scripts/test_correctness.py` runs the model as its own draft and target. In
exact arithmetic every proposal must then be accepted, so acceptance below 1.000
means the KV caches have drifted out of sync. It passes at k=1, 2, 4 and 5 with
byte-identical output in float32.

That test earned its place. The first implementation primed both caches with the
full prompt and then fed the last prompt token again, so every model saw it
twice and all subsequent cache arithmetic was off by one. It produced fluent,
plausible text and matched plain decoding on 1 prompt in 10. The rewrite tracks
cached lengths explicitly and feeds each model exactly the tokens it has not
seen.

## What this is not

One draft-target pair, one model family, one GPU, ten prompts, greedy decoding
only. The stochastic rejection-sampling variant that preserves the sampling
distribution at temperature above zero is not implemented; greedy was chosen
because it makes correctness checkable by assertion.

The headline result is about eager PyTorch, not about speculative decoding as a
technique. A serving engine with batched draft execution, CUDA graphs, and a
much smaller draft model would produce a different cost ratio and therefore a
different conclusion. The contribution here is the framing: the decision reduces
to one measurable number, and that number is easy to measure before building
anything.

## License

Code released under the MIT License (see `LICENSE`).
