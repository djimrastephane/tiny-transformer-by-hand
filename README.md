# Tiny Transformer, By Hand

[![Pages deploy status](https://github.com/djimrastephane/tiny-transformer-by-hand/actions/workflows/pages.yml/badge.svg)](https://github.com/djimrastephane/tiny-transformer-by-hand/actions/workflows/pages.yml)
[![Live demo](https://img.shields.io/badge/demo-live-45C7B8)](https://djimrastephane.github.io/tiny-transformer-by-hand/)
[![Mathematica checks](https://img.shields.io/badge/Mathematica_checks-34%2F34_passing-E8A33D)](Mathematica/TinyTransformerByHand.wl)
[![LoRA checks](https://img.shields.io/badge/LoRA_checks-21%2F21_passing-E8A33D)](Mathematica/TinyLoRAByHand.wl)
[![Calculator-precision checks](https://img.shields.io/badge/calculator_precision-80%2F80_passing-8592AC)](calculations/verify_calculator_precision.py)
[![No LLM required](https://img.shields.io/badge/requires-a_calculator-8592AC)](calculations/HAND_CALCULATION.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An intentionally reduced causal self-attention language model for
tracing the mathematics of next-token training — not a complete
transformer block. See ["What's simplified, and why it's acceptable
here"](#whats-simplified-and-why-its-acceptable-here) below for the
precise scope.*

**Can we build a transformer language model small enough that a human can
reproduce its important calculations by hand? Yes.** This project is that
model — a 6-word vocabulary, 2-token sequences, embedding dimension 2, one
attention head, one block, tens of parameters — worked out explicitly, one
matrix multiplication at a time, then trained once, live, in front of you.

**Try it now — no install, no account, no code:**
[**Open the interactive demo →**](https://djimrastephane.github.io/tiny-transformer-by-hand/)
*(source is the static, self-contained `Web/index.html`, auto-deployed to
GitHub Pages by `.github/workflows/pages.yml` on every push — see
"Running the web demo" below.)*

**Prefer a visual story?**
[**View the LinkedIn carousel →**](https://claude.ai/code/artifact/b2a63bf4-2ded-4087-90e1-4b0dbf627c8a)
*(8 slides, editable, and exportable to PNG/PDF per slide — source in
`figures/carousel/`.)* Plus its own companion:
[**the Tiny LoRA carousel →**](https://claude.ai/code/artifact/56763acd-6fc9-4939-ad5c-66d57657d7b2)
*(6 slides — source in `figures/lora-carousel/`.)*

---

## The result, up front

Given the input **"run casing"**, the model should predict **"shoe"** (as
in a casing shoe). Before any training, it doesn't — "run" outranks it:

| Before training | | After 1 gradient-descent update | |
|---|---|---|---|
| run | 28.0% | **shoe** | **43.2%** |
| shoe | 20.1% | run | 18.0% |
| Loss | 1.6035 | Loss | **0.8404** |

The ranking flips after a single update: "shoe" overtakes "run" for the
top spot, and does so decisively (a ~25-point margin).

One hand-computable update — computed here with a fixed learning rate of
2, updating only the output projection `W_Out` — is enough to flip this
one example's top prediction from wrong to right, roughly double
P("shoe"), and roughly halve the loss. Every number above is checked
programmatically (34 automated tests — see `Mathematica/TinyTransformerByHand.wl`)
and reproducible with paper and a calculator (see `calculations/HAND_CALCULATION.md`).

## What this is, and is not

This is a transformer: the same attention-then-project computational
pattern used in real language models, at a scale a person can compute by
hand. It is **not** a large language model (LLM), and it is **not** a
scaled-down version of ChatGPT, Claude, or any production system. Every
number here belongs to a teaching-sized toy built to expose mechanics, not
approximate a real model's behavior.

## Ways into the project

| Layer | What it's for | Where |
|---|---|---|
| **Interactive web demo** | The main thing to share and click through. Zero-install, runs entirely in the browser. Sliders for learning rate and one embedding value, a training-step toggle, a live probability readout, a "Show the maths" panel exposing every matrix, and a "Paper Mode" 9-step worksheet walkthrough. | `Web/index.html` ([live](https://djimrastephane.github.io/tiny-transformer-by-hand/)) |
| **Full notebook, statically rendered** | Every derivation Mathematica computed, laid out as one linear, no-install page — for a reader who wants to see the whole worked argument (not just play with sliders) without needing Mathematica or Wolfram Player installed. Not interactive; the web demo above covers that. | `Web/notebook.html` ([live](https://djimrastephane.github.io/tiny-transformer-by-hand/notebook.html)) |
| **LoRA companion, statically rendered** | Same treatment as the row above, for the LoRA companion notebook: freeze `W_Out`, train `ΔW = B·A`, and see the full step-by-step comparison against full fine-tuning. | `Web/lora-notebook.html` ([live](https://djimrastephane.github.io/tiny-transformer-by-hand/lora-notebook.html)) |
| **Mathematica notebook** | The source of mathematical truth, and the only way to actually re-evaluate every cell yourself. Every operation — embedding lookup, Q/K/V, scaled dot-product attention, causal masking, softmax, cross-entropy, the gradient, gradient descent — derived and displayed explicitly, plus an automated 34-check verification suite. Requires Mathematica or (free) Wolfram Player. | `Mathematica/TinyTransformerByHand.nb`, `Mathematica/TinyTransformerByHand.wl` |
| **Hand-calculation worksheet** | Paper and a calculator. Given values, fill-in-the-blank steps, and an answer key — for both the main notebook and the LoRA companion. The claim that a calculator (not just Mathematica) can reproduce these numbers is itself checked by a script, not just asserted. | `calculations/HAND_CALCULATION.md`, `LORA_HAND_CALCULATION.md`, `verify_calculator_precision.py` |

The web demo and the static notebook page are not independent
reimplementations guessing at the same answer — both are generated
directly from the equations and numbers already derived and verified in
the Mathematica notebook. Mathematica is the development and
verification environment; the browser pages are the delivery mechanism.

There's also an 8-slide LinkedIn carousel (`figures/carousel/`) covering
the same story for social posts —
[view it here](https://claude.ai/code/artifact/b2a63bf4-2ded-4087-90e1-4b0dbf627c8a) —
built from these same verified numbers, not a separate retelling.

## Companion project: Tiny LoRA, By Hand

Same frozen model, a different question: instead of updating all 12
entries of `W_Out` directly (full fine-tuning, what the main notebook
does), freeze `W_Out` and train a rank-1 correction `ΔW = B·A` (2×1 times
1×6, 8 numbers total) on top of it instead — the same idea real LoRA
fine-tuning uses on much larger models, small enough here to compute by
hand.

| Stage | P("shoe") | Loss | Top prediction | Numbers moved |
|---|---|---|---|---|
| Before training | 20.1% | 1.6035 | run (wrong) | — |
| Full fine-tune, 1 step | **43.2%** | **0.8404** | **shoe** (correct) | 12 (all of `W_Out`) |
| LoRA, step 1 | 31.0% | 1.1714 | run (wrong) | 2 (just `B`) |
| LoRA, step 2 (bonus) | **79.7%** | **0.2263** | **shoe** (correct) | up to 8 (`B` and `A`) |

A genuine, checked LoRA fact falls out of this worked example for free:
on step 1, `B` starts at zero (standard practice), so the gradient with
respect to `A` is *exactly* zero — only `B` can move on the very first
step. `A` only starts learning once `B` has moved. By step 2, LoRA has
not just caught up to one step of full fine-tuning — it has overtaken it,
while never touching more than 8 of `W_Out`'s effective parameters.

21/21 automated checks pass — see `Mathematica/TinyLoRAByHand.wl`.
**[Read it as a static, no-install web page →](https://djimrastephane.github.io/tiny-transformer-by-hand/lora-notebook.html)**
Source: `Mathematica/TinyLoRAByHand.nb`,
`Mathematica/TinyLoRAByHand.wl`,
`calculations/LORA_HAND_CALCULATION.md`.

There's also its own 6-slide LinkedIn carousel (`figures/lora-carousel/`)
— [view it here](https://claude.ai/code/artifact/56763acd-6fc9-4939-ad5c-66d57657d7b2) —
telling this same before/full-fine-tune/LoRA-step-1/LoRA-step-2 story.

## Repository structure

```
tiny-transformer-by-hand/
    README.md                     - this file
    Mathematica/
        TinyTransformerByHand.nb  - the full worked notebook (16 sections + interactive + worksheet)
        TinyTransformerByHand.wl  - standalone reference implementation + RunAllChecks[] verification suite
        TinyLoRAByHand.nb         - companion notebook: freeze W_Out, train a rank-1 correction instead
        TinyLoRAByHand.wl         - LoRA reference implementation + RunLoRAChecks[] verification suite
        ExportNotebookValues.wls  - exports notebook_values.json for Web/build_notebook.py
        ExportLoRAValues.wls      - exports lora_values.json for Web/build_lora_notebook.py
    Web/
        index.html                - the interactive browser demo (self-contained, no build step)
        notebook.html             - the full notebook, statically rendered (no install, not interactive)
        lora-notebook.html        - the LoRA companion notebook, statically rendered (no install, not interactive)
    calculations/
        HAND_CALCULATION.md       - paper-and-calculator worksheet with an answer key
        LORA_HAND_CALCULATION.md  - paper-and-calculator worksheet for the LoRA companion, answer key included
        verify_calculator_precision.py - redoes both worksheets at 4-decimal precision, diffs against Mathematica's ground truth
    figures/
        README.md                 - the LinkedIn carousels: links, slide-by-slide contents, source files
        carousel/                 - the 8 slide sources (.dc.html), canvas layout, and the seeded canvas
        lora-carousel/            - the LoRA companion's own 6-slide carousel (.dc.html), canvas layout, and seeded canvas
    methodology/
        ASSUMPTIONS.md            - what was simplified, why, and what the one training step does and doesn't update
```

## Architecture

| | This project | A modern pretrained language model |
|---|---|---|
| Vocabulary | 6 tokens | tens of thousands to 1,000,000+ |
| Sequence length | 2 tokens | thousands to millions of tokens (context window) |
| Embedding dimension | 2 | hundreds to tens of thousands |
| Attention heads | 1 | dozens, run in parallel |
| Transformer blocks | 1 | dozens to 100+, stacked |
| Parameters | tens | billions, sometimes hundreds of billions |
| Training corpus | 1 next-token example | up to trillions of training tokens |
| Training updates shown | 1, worked by hand | millions to billions |
| Randomness | none — every value is a fixed, deterministic constant | random initialization, stochastic training |

The underlying operations in the right-hand column remain ordinary
numerical operations. Production models combine them at vastly greater
scale, with additional architectural, training, and systems complexity
— several of those pieces are listed below, and this toy model omits
them for clarity.

### What's simplified, and why it's acceptable here

| Omitted | Why it's fine for this demonstration |
|---|---|
| LayerNorm | Stabilizes training across many stacked layers; with one block there's nothing to destabilize. |
| Residual (skip) connections | Help gradients flow through many layers; there is only one layer here. |
| Dropout | A regularizer against overfitting on large datasets; there is exactly one training example. |
| Multi-head attention | Several attention computations run in parallel and are combined; one head already shows the full mechanism. |
| A large feed-forward network inside the block | Keeps the path from attention output to logits short and traceable to a single output projection. |
| Positional encodings | With 2 positions and a causal mask already distinguishing "first" from "second", an explicit position signal adds notation without adding insight. |

See `methodology/ASSUMPTIONS.md` for the full reasoning, including exactly
what the one worked training step does and does not update (only `W_Out`
— the embeddings and `W_Q`/`W_K`/`W_V` are left unchanged, deliberately).

## Running the web demo

`Web/index.html`, `Web/notebook.html`, and `Web/lora-notebook.html` are
all single, self-contained files — no build step, no dependencies
beyond one Google Fonts stylesheet link, no server required:

- **Locally:** just open any of the three files in a modern browser.
- **Live on GitHub Pages:** [djimrastephane.github.io/tiny-transformer-by-hand](https://djimrastephane.github.io/tiny-transformer-by-hand/)
  (interactive demo),
  [.../notebook.html](https://djimrastephane.github.io/tiny-transformer-by-hand/notebook.html)
  (static notebook), and
  [.../lora-notebook.html](https://djimrastephane.github.io/tiny-transformer-by-hand/lora-notebook.html)
  (static LoRA companion) — no server-side compute, everything runs
  client-side. Deployment is automatic: `.github/workflows/pages.yml`
  redeploys the whole `Web/` folder to Pages on every push that touches
  it, so editing any of these files and pushing to `main` is the entire
  release process.

`notebook.html` and `lora-notebook.html` are generated from
Mathematica's own computed values (exported as JSON, never hand-retyped)
by `Web/build_notebook.py` and `Web/build_lora_notebook.py` respectively
— see those scripts (and `Mathematica/ExportNotebookValues.wls` /
`ExportLoRAValues.wls`) if you change either notebook and need to
regenerate its page.

## Running the Mathematica notebook

1. Open `Mathematica/TinyTransformerByHand.nb` from the `Mathematica/`
   folder (so it can find `TinyTransformerByHand.wl` alongside it).
2. **Evaluate ▸ Evaluate Notebook** to run every calculation top to
   bottom and render the tables, matrices, bar charts, and the
   interactive control panel.
3. To check the underlying implementation independently, run from a
   terminal inside `Mathematica/`:
   `wolframscript -code 'Get["TinyTransformerByHand.wl"]; TinyTransformerByHand`RunAllChecks[]'`
   — it prints a pass/fail line for every mathematical property this
   project depends on (27/27 passing). The notebook's Appendix A runs
   this suite plus `` TinyTransformerByHand`RunAlternateTargetChecks["cement"] ``
   (7 more checks, for the "Bonus: does this generalize?" section) —
   27 + 7 = the 34/34 total in the badge at the top of this README.

Built and verified against **Wolfram Language 15.0.1**; it uses only
long-stable language features (`MatrixForm`, `Grid`, `BarChart`,
`Manipulate`), so it should run unmodified on **Mathematica / Wolfram
Desktop 13.0 or later**.

The LoRA companion notebook (`Mathematica/TinyLoRAByHand.nb`) works the
same way: open it from the `Mathematica/` folder, evaluate top to bottom,
and its own Appendix runs
`wolframscript -code 'Get["TinyTransformerByHand.wl"]; Get["TinyLoRAByHand.wl"]; TinyLoRAByHand`RunLoRAChecks[]'`
for the same kind of independent check (currently 21/21 passing).

## What you should understand after going through this

- What a token, a token ID, and an embedding are, and how "run casing"
  becomes two 2-dimensional vectors.
- What query, key, and value vectors are, mechanically: three ordinary
  matrix products.
- How scaled dot-product attention scores are computed, why they're
  divided by √d, and why next-token prediction needs a causal mask (so a
  position can't "see" the very token it's predicting).
- What softmax does, shown as explicit exponentiate-sum-divide
  arithmetic, not a black-box call.
- The difference between attention weights, model weights, logits,
  probabilities, and gradients — five different kinds of numbers this
  project is careful never to conflate.
- What cross-entropy loss measures, and where the softmax-cross-entropy
  gradient (`probabilities − one-hot target`) comes from — checked
  numerically against a finite-difference estimate, not taken on faith.
- How one gradient-descent update on the output projection changes a
  real, computed probability, and why P("shoe") rises and the loss falls
  in this worked example.
- Why this toy model leaves out LayerNorm, residual connections,
  dropout, multi-head attention, a large feed-forward network, and
  positional encodings — an acceptable trade for transparency, not an
  oversight.
- That the arithmetic inside a modern LLM is the same handful of
  operations shown here, repeated over vastly more parameters and data,
  combined with the additional architectural pieces this project omits —
  not a fundamentally different kind of math.

## License

[MIT](LICENSE) — use, modify, and share freely, including for teaching
your own version of this example.
