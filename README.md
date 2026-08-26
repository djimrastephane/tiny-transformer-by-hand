# Tiny Transformer, By Hand

**Can we build a transformer language model small enough that a human can
reproduce its important calculations by hand? Yes.** This project is that
model — a 6-word vocabulary, 2-token sequences, embedding dimension 2, one
attention head, one block, tens of parameters — worked out explicitly, one
matrix multiplication at a time, then trained once, live, in front of you.

**Try it now — no install, no account, no code:**
[**Open the interactive demo →**](https://claude.ai/code/artifact/686ca801-70dc-49aa-9526-50c9d2143bb8)
*(also shipped as a static, self-contained page at `Web/index.html` — see
"Running the web demo" below to host it yourself, e.g. on GitHub Pages.)*

**Prefer a 6-slide visual story?**
[**View the LinkedIn carousel →**](https://claude.ai/code/artifact/b2a63bf4-2ded-4087-90e1-4b0dbf627c8a)
*(editable, and exportable to PNG/PDF per slide — source in `figures/carousel/`.)*

---

## The result, up front

Given the input **"run casing"**, the model should predict **"shoe"** (as
in a casing shoe). Before any training, it doesn't:

| | Before training | After 1 gradient-descent update |
|---|---|---|
| P("shoe") | 20.1% | **43.2%** |
| Loss (−ln P) | 1.6035 | **0.8404** |
| Model's top guess | "run" (wrong) | **"shoe" (correct)** |

One hand-computable update — computed here with a fixed learning rate of
2, updating only the output projection `W_Out` — is enough to flip this
one example's top prediction from wrong to right, roughly double
P("shoe"), and roughly halve the loss. Every number above is checked
programmatically (27 automated tests — see `Mathematica/TinyTransformerByHand.wl`)
and reproducible with paper and a calculator (see `calculations/HAND_CALCULATION.md`).

## What this is, and is not

This is a transformer: the same attention-then-project computational
pattern used in real language models, at a scale a person can compute by
hand. It is **not** a large language model (LLM), and it is **not** a
scaled-down version of ChatGPT, Claude, or any production system. Every
number here belongs to a teaching-sized toy built to expose mechanics, not
approximate a real model's behavior.

## Three ways into the project

| Layer | What it's for | Where |
|---|---|---|
| **Interactive web demo** | The main thing to share and click through. Zero-install, runs entirely in the browser. Sliders for learning rate and one embedding value, a training-step toggle, a live probability readout, a "Show the maths" panel exposing every matrix, and a "Paper Mode" 9-step worksheet walkthrough. | `Web/index.html` |
| **Mathematica notebook** | The source of mathematical truth. Every operation — embedding lookup, Q/K/V, scaled dot-product attention, causal masking, softmax, cross-entropy, the gradient, gradient descent — derived and displayed explicitly, plus an automated 27-check verification suite. | `Mathematica/TinyTransformerByHand.nb`, `Mathematica/TinyTransformerByHand.wl` |
| **Hand-calculation worksheet** | Paper and a calculator. Given values, fill-in-the-blank steps, and an answer key. | `calculations/HAND_CALCULATION.md` |

The web demo is not an independent reimplementation guessing at the same
answer — its JavaScript is a direct, unchanged port of the equations
already derived and verified in the Mathematica notebook. Mathematica is
the development and verification environment; the browser page is the
delivery mechanism.

There's also a 7-slide LinkedIn carousel (`figures/carousel/`) covering
the same story for social posts —
[view it here](https://claude.ai/code/artifact/b2a63bf4-2ded-4087-90e1-4b0dbf627c8a) —
built from these same verified numbers, not a separate retelling.

## Repository structure

```
tiny-transformer-by-hand/
    README.md                     - this file
    Mathematica/
        TinyTransformerByHand.nb  - the full worked notebook (16 sections + interactive + worksheet)
        TinyTransformerByHand.wl  - standalone reference implementation + RunAllChecks[] verification suite
    Web/
        index.html                - the interactive browser demo (self-contained, no build step)
    calculations/
        HAND_CALCULATION.md       - paper-and-calculator worksheet with an answer key
    figures/
        README.md                 - the LinkedIn carousel: links, slide-by-slide contents, source files
        carousel/                 - the 7 slide sources (.dc.html), canvas layout, and the seeded canvas
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
| Training examples used | 1 | trillions of tokens |
| Training updates shown | 1, worked by hand | millions to billions |
| Randomness | none — every value is a fixed, deterministic constant | random initialization, stochastic training |

The arithmetic did not become magic in the right-hand column — the scale
became enormous, and several architectural pieces were added back in (see
below) that this toy model omits for clarity.

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

`Web/index.html` is a single, self-contained file — no build step, no
dependencies beyond one Google Fonts stylesheet link, no server required:

- **Locally:** just open `Web/index.html` in any modern browser.
- **On GitHub Pages:** push this repository to GitHub, enable Pages for
  the repo (Settings → Pages), and point it at the branch/folder
  containing `Web/index.html` (or copy it to the repo root / a `docs/`
  folder, per your Pages configuration) — no server-side compute needed,
  the entire model runs client-side in JavaScript.

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
   project depends on (currently: 27/27 passing).

Built and verified against **Wolfram Language 15.0.1**; it uses only
long-stable language features (`MatrixForm`, `Grid`, `BarChart`,
`Manipulate`), so it should run unmodified on **Mathematica / Wolfram
Desktop 13.0 or later**.

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
