# Figures

This folder holds the static visual assets used for the LinkedIn
carousel and other social/promotional posts about this project.

## LinkedIn carousel

**[View / edit / export the carousel →](https://claude.ai/code/artifact/b2a63bf4-2ded-4087-90e1-4b0dbf627c8a)**

`carousel/` contains the 8-slide story (1080×1350 each), built with
Claude Design's canvas editor:

1. Hook — "Can you train a transformer with a calculator?"
2. The deliberately tiny model — vocabulary, embedding dimension, heads, blocks, parameters
3. "run casing" → Q, K, V — the actual 2×2 matrices (with a note that Q/K/V are token-dependent, not fixed matrices — see below)
4. Attention → logits → probabilities, with the model's (wrong) top guess before training
5. Teaching it the answer — cross-entropy loss, the gradient, the update rule
6. Before/after the one training step, and the payoff line
7. Bonus: does this generalize? — retargeting the same example toward "cement" instead of "shoe"; P(cement) rises from the lowest starting probability of all 6 tokens (7.3%) to 21.8%, but only barely overtakes "run" (a ~0.13-point margin, vs. shoe's decisive ~25-point win)
8. What's simplified, and why — the omissions list from `methodology/ASSUMPTIONS.md`, including the sharper positional-encoding/permutation-invariance caveat and a nod to the notebook's optional feed-forward/ReLU illustration

Every number on every slide is copied from the same verified values used
throughout this project (see `Mathematica/TinyTransformerByHand.wl` and
`calculations/HAND_CALCULATION.md`) — nothing here was estimated or
re-derived independently.

- `carousel/Main.dc.html`, `Slide2.dc.html` … `Slide8.dc.html` — the
  source for each slide (editable design-component files).
- `carousel/canvas.json` — how the eight slides are laid out on the canvas.
- `carousel/tiny-transformer-linkedin-carousel.html` — the seeded,
  published canvas; open it directly, or use the published link, to
  view, edit, and export each slide as PNG/PDF for posting.

To change a slide: edit the corresponding `.dc.html` (or use the
canvas's own editor and re-extract), then re-seed and republish — see
the project's design-canvas tooling for the exact commands.

Slide 6's closing line points readers to a live browser demo — that's
[**Web/index.html**](https://djimrastephane.github.io/tiny-transformer-by-hand/),
served from GitHub Pages, the interactive, zero-install version of this
same model (see the project's top-level `README.md`).

## LinkedIn carousel: Tiny LoRA, By Hand

**[View / edit / export the LoRA carousel →](https://claude.ai/code/artifact/56763acd-6fc9-4939-ad5c-66d57657d7b2)**

`lora-carousel/` is a separate, 6-slide carousel (same 1080×1350 format
and visual identity as above) telling the companion project's story:
freeze `W_Out`, train a rank-1 correction `ΔW = B·A` instead, and watch
it catch up to (then overtake) one step of full fine-tuning:

1. Hook — "What if you froze the model and taught it anyway?"
2. The idea — `ΔW = B·A`, a rank-1 correction sized 2×1 times 1×6, laid out against the frozen 2×6 `W_Out`
3. Step zero — `B0` is zero, so `ΔW0` is exactly zero and the model matches the frozen baseline
4. Step 1's surprising fact — only `B` (2 numbers) moves, since `A`'s gradient is exactly zero while `B` is zero; real progress, but the top prediction is still wrong
5. Step 2 (bonus) — the full before/full-fine-tune/LoRA-step-1/LoRA-step-2 comparison table; LoRA overtakes full fine-tuning having touched at most 8 numbers
6. What this does and doesn't prove — the same honest caveats as `methodology/ASSUMPTIONS.md`, plus the one fact that isn't a simplification: B always moves first

Every number is copied from `Mathematica/TinyLoRAByHand.wl` and
`calculations/LORA_HAND_CALCULATION.md` — nothing here was estimated or
re-derived independently.

- `lora-carousel/Main.dc.html`, `Slide2.dc.html` … `Slide6.dc.html` —
  the source for each slide.
- `lora-carousel/canvas.json` — how the six slides are laid out on the canvas.
- `lora-carousel/tiny-lora-linkedin-carousel.html` — the seeded,
  published canvas; open it directly, or use the published link, to
  view, edit, and export each slide as PNG/PDF for posting.
