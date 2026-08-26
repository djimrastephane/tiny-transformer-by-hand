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
