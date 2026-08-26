# Methodology: Assumptions and Simplifications

This project builds a transformer language model small enough to compute
by hand, then shows one gradient-descent update on it, worked out
numerically in Mathematica, reimplemented (unchanged) in JavaScript for
the browser demo, and reproducible on paper. This document is the
honest accounting of what was kept, what was left out, and why — so
that nobody mistakes the toy model for a scaled-down production system.

## The one design principle

Every choice below optimizes for **mathematical transparency**, not
realism, performance, or software elegance. If a component could be
removed without losing the ability to trace every number back to an
explicit calculation, it was removed.

## Fixed architecture

| Choice | Value | Why |
|---|---|---|
| Vocabulary | 6 tokens (`<BOS>`, `run`, `casing`, `shoe`, `cement`, `<EOS>`) | Small enough to list every logit and probability in one table; large enough to have a real "wrong answer" competing with the right one. |
| Sequence length | 2 tokens | The minimum length at which a causal mask does anything at all (position 1 vs. position 2). |
| Embedding dimension | 2 | Small enough to write every vector as an ordered pair; large enough that attention (a 2×2 dot product) is a genuine matrix operation, not a scalar. |
| Attention heads | 1 | One head shows the complete mechanism. A second head duplicates arithmetic without adding a new idea to explain. |
| Transformer blocks | 1 | With one block, there is no long path for a gradient to travel, and nothing to stack. |
| Parameters | Tens (`W_Q`, `W_K`, `W_V` are 2×2 each; `W_Out` is 2×6; embeddings are 6×2) | Small enough to write out every parameter in a single notebook cell. |
| Training examples | 1 (`"run casing"` → `"shoe"`) | One example is enough to compute a real loss and a real gradient; more would need a running total, not a new mechanic. |
| Randomness | None. Every embedding and every weight matrix is a fixed, chosen constant. | Reproducibility. A reader with a calculator must get the exact same numbers we did. |

## What's omitted, and why it's acceptable at this scale

| Component | Role in a real transformer | Why it's safe to omit here |
|---|---|---|
| LayerNorm | Rescales activations between steps so training stays stable across many stacked layers. | With one block and tens of parameters, there is nothing to destabilize. |
| Residual (skip) connections | Give gradients a shortcut path through many stacked layers. | One layer means there is no long path for a gradient to vanish along. |
| Dropout | Regularizes against overfitting on large datasets. | There is exactly one training example; overfitting isn't the risk here, transparency is the goal. |
| Multi-head attention | Runs several attention computations in parallel and combines them, letting the model attend to different things at once. | One head already demonstrates the full query/key/value/softmax mechanism. A second head would repeat the same arithmetic. |
| A large feed-forward network inside the block | Adds a per-position nonlinearity (ReLU/GELU) between attention and the output, letting stacked blocks build genuinely nonlinear features rather than larger linear combinations. | Keeps the path from attention output to logits short enough that the whole thing fits on one page: attention output → one output projection → logits. The notebook's Section 9 includes a small optional illustration of what this sublayer's nonlinearity would add, without folding it into the trained example. |
| Positional encodings | Tell the model where each token sits in the sequence. | With only 2 positions, the causal mask happens to fully distinguish "first token" from "second token" — but see the caveat below; this is a narrower excuse than it first looks. |

### The positional-encoding excuse is narrower than it looks

Self-attention on its own has no built-in sense of order: an (uncausal)
attention layer fed the same set of token vectors in a different order
produces the same set of outputs, merely permuted the same way —
attention is a function of content-based similarity between vectors,
not of where those vectors sit in a list. Causal masking breaks that
symmetry here only because, with exactly 2 positions, there are only 2
possible masks — "see only yourself" and "see everyone so far" — and
that happens to be enough to tell position 1 from position 2. A mask
tells a position what it may look at, not where it sits or how far away
another token is, so it is not a general substitute for positional
encoding. Real transformers add an explicit positional signal (learned
position embeddings, sinusoidal encodings, or rotary embeddings/RoPE)
precisely because causal masking alone stops being enough once
sequences are longer than 2 tokens.

## Q, K, and V are token-dependent, not fixed matrices

Because "run" and "casing"'s embeddings happen to be the two standard
unit vectors, the input matrix `X` is the identity, so `Q = X·W_Q`
reduces to exactly `W_Q` (and likewise for `K`, `V`) — a convenient
coincidence used throughout the worked example so the arithmetic is
easy to check by hand. It is not a general property: `Q`, `K`, and `V`
are computed from the input, and a different input produces different
values, even with `W_Q`, `W_K`, `W_V` held fixed. The notebook (Section
4) makes this concrete with a counterexample — rescaling "casing"'s
embedding from `(0, 1)` to `(0, 2)` changes `Q`'s second row from
`(1, 1)` to `(2, 2)`, no longer matching `W_Q`. The interactive section
and the web demo's embedding slider let you verify this directly.

## What the one training step does and does not update

The worked example computes and applies a gradient for **`W_Out`
only** — the output projection from the attention representation to
vocabulary logits. `W_Q`, `W_K`, `W_V`, and the embeddings are left
unchanged.

This is a scope decision, not an oversight. The gradient of a
softmax-plus-cross-entropy loss with respect to the logits has a
famously simple closed form (`probabilities − one-hot(target)`), and
propagating it one more step to `W_Out` is a single outer product.
Propagating it further back — through the attention softmax and into
`Q`, `K`, `V`, and the embeddings — is ordinary calculus, but requires
differentiating through the attention-weight softmax itself (a full
Jacobian, not a simple subtraction). The Mathematica notebook
(`Mathematica/TinyTransformerByHand.nb`, Section 12) states this limit
explicitly and sketches the further chain-rule steps for anyone who
wants to continue by hand, rather than presenting a fully
backpropagated model that wasn't actually derived end-to-end.

## Numerical verification, not just narrative

Every claim in this project is checked programmatically before being
reported as a result — see `Mathematica/TinyTransformerByHand.wl`,
`` TinyTransformerByHand`RunAllChecks[] ``, 27 checks covering matrix
dimensions, softmax rows summing to 1, correct causal-mask behavior,
correct target-probability extraction, correct cross-entropy, correct
gradient sign, and — the two claims the whole demonstration depends on
— that P("shoe") increases and the loss decreases after the one
training update. The browser demo (`Web/index.html`) reimplements the
same equations directly in JavaScript from those already-verified
values; it does not re-derive or independently estimate anything.

## What this project is not claiming

- This is **not** a large language model (LLM), and it is **not** a
  scaled-down version of ChatGPT, Claude, or any production system.
- One hand-worked update on one example does not demonstrate everything
  about how real models are trained — only that the arithmetic at the
  core of it (matrix multiplication, softmax, cross-entropy, gradients,
  gradient descent) is exactly what's shown here, repeated over far
  more parameters, data, and training steps, and combined with the
  additional architectural components listed above that this toy model
  omits.
- The scale comparison in the notebook's Section 16 and the project
  README is illustrative, not a claim that production models are
  "this architecture, just bigger." They also differ in the specific
  components listed above.
