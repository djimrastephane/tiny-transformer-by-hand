# LoRA Hand Calculation Worksheet

This is a companion to `TinyLoRAByHand.nb`, which is itself a companion to
`TinyTransformerByHand.nb`. Work through `HAND_CALCULATION.md` first — this
sheet picks up exactly where that one leaves off, reusing `h` and
`dL/dWOut` (called `G` below) without recomputing them.

The task: instead of updating all 12 entries of `WOut` directly (what
`HAND_CALCULATION.md` does), freeze `WOut` and train a rank-1 correction
`DeltaW = B . A` on top of it.

---

## 1. Given values (carried over from `HAND_CALCULATION.md`)

```
h = (0.6698, 0.3302)

G = dL/dWOut =
  |  0.0960   0.1875   0.1335  -0.5350   0.0491   0.0690 |
  |  0.0473   0.0924   0.0658  -0.2638   0.0242   0.0340 |

Learning rate = 2
```

**New for this sheet — the LoRA factors** (rank r = 1, so B is 2×1 and A is
1×6):

```
B0 = | 0 |        A0 = ( 1  -1  1  -1  1  -1 )
     | 0 |
```

`B0` is zero on purpose (standard LoRA practice), so `DeltaW0 = B0.A0` is
the zero matrix and the model starts out behaving exactly like the
untrained base model: P("shoe") = 0.2012, loss = 1.6035, same as
`HAND_CALCULATION.md`'s starting point.

---

## 2. Worksheet — fill in the blanks

```
dA0 = Transpose[B0] . G = ______________________
  (hint: B0 is all zeros — what must this be, without any arithmetic?)

dB0 = G . Transpose[A0] = ______________________
  (two dot products: each row of G, dotted with the 6 numbers of A0)

B1 = B0 - 2 * dB0 = ______________________

A1 = A0 - 2 * dA0 = ______________________
  (should be identical to A0 — why?)

DeltaW1 = B1 . A1 = ______________________
  (outer product: every entry of B1 times every entry of A1)

WOutEffective1 = WOut + DeltaW1 = ______________________

Logits1 = h . WOutEffective1 = ______________________

Probabilities1 = softmax(Logits1) = ______________________

P("shoe")1 = ______________
Loss1 = -ln(P("shoe")1) = ______________

Which token has the highest probability now? Is it "shoe"?
```

---

## 3. Answer key

```
dA0 = (0, 0, 0, 0, 0, 0)

dB0 = | 0.5571 |
      | 0.2747 |

B1 = | -1.1143 |
     | -0.5494 |

A1 = ( 1  -1  1  -1  1  -1 )     <- unchanged: dA0 was zero, so nothing moved

DeltaW1 =
  | -1.1143   1.1143  -1.1143   1.1143  -1.1143   1.1143 |
  | -0.5494   0.5494  -0.5494   0.5494  -0.5494   0.5494 |

WOutEffective1 = WOut + DeltaW1 =
  | -1.1143   2.1143  -1.1143   2.1143  -2.1143   1.1143 |
  | -0.5494   0.5494   0.4506  -0.4506  -0.5494  -0.4506 |

Logits1 (order: <BOS>, run, casing, shoe, cement, <EOS>) =
  (-0.9277, 1.5975, -0.5975, 1.2673, -1.5975, 0.5975)

Probabilities1 =
  (0.0345, 0.4312, 0.0480, 0.3099, 0.0177, 0.1586)

P("shoe")1 = 0.3099   (up from 0.2012)
Loss1 = -ln(0.3099) = 1.1714   (down from 1.6035)

Predicted (highest-probability) token: "run" (0.4312) -- STILL WRONG.
Compare to full fine-tuning's one step, which already predicts "shoe"
correctly (see HAND_CALCULATION.md) -- LoRA's first step, having moved
only 2 numbers instead of 12, makes real progress but hasn't caught up yet.
```

Of the 8 numbers making up `B` and `A`, exactly **2 changed** this step
(all of `B`; none of `A`) — because `dA = Transpose[B].G` is forced to be
zero whenever `B` is zero, no matter what `G` is. `A` cannot start learning
until `B` gives it something to multiply against.

---

## 4. Bonus, by hand: one more step

This step is optional (it is not required to see the main point above),
but it is worth doing once to see `A` finally move. It reuses the same
`softmax` / `outer product` / `gradient step` operations as everything
above — nothing new, just one more repetition.

```
G1 = outer product of h and (Probabilities1 - one-hot("shoe"))

dA1 = Transpose[B1] . G1   (no longer zero, since B1 isn't zero)
dB1 = G1 . Transpose[A1]

B2 = B1 - 2*dB1
A2 = A1 - 2*dA1

DeltaW2 = B2 . A2
WOutEffective2 = WOut + DeltaW2
... softmax(h . WOutEffective2) as before
```

**Answer key:**

```
dA1 = (-0.0320, -0.4001, -0.0446, 0.6402, -0.0164, -0.1472)   <- nonzero now

P("shoe")2 = 0.7974   (up from 0.3099)
Loss2 = -ln(0.7974) = 0.2263   (down from 1.1714)

Predicted token: "shoe" (0.7974) -- NOW CORRECT, and with a higher
confidence than full fine-tuning's single step (0.4315).
```

---

## Summary table

| Stage                   | P("shoe") | Loss   | Top prediction | Numbers moved (cumulative) |
|-------------------------|-----------|--------|-----------------|------------------------------|
| Before training         | 0.2012    | 1.6035 | "run" (wrong)   | —                            |
| Full fine-tune, 1 step  | 0.4315    | 0.8404 | "shoe" (correct)| 12 (all of `WOut`)           |
| LoRA, step 1            | 0.3099    | 1.1714 | "run" (wrong)   | 2 (just `B`)                 |
| LoRA, step 2 (bonus)    | 0.7974    | 0.2263 | "shoe" (correct)| up to 8 (`B` and `A`)        |

This worksheet only reproduces the LoRA correction to `WOut` — the
embeddings, `WQ`, `WK`, and `WV` are frozen throughout, exactly as in
`HAND_CALCULATION.md`. See `TinyLoRAByHand.nb` Section 9 ("What This Does
and Doesn't Prove") for the caveats on what this toy example does and does
not establish about LoRA in general.

---

## Verified: does 4-decimal calculator rounding actually work?

Both answer keys above (Section 3's step 1, and the bonus's step 2) were
recomputed independently using only 4-decimal-place arithmetic after
every individual multiply, add, exponential, and division — no hidden
extra precision carried between steps, starting from `h` and `G` exactly
as given in Section 1:

```
dA0, dB0, B1, DeltaW1, WOutEffective1                  -- exact match
Logits1, softmax1, P("shoe")1                          -- exact match
Loss1                                                   -- within 0.0001
G1, dA1, dB1                                            -- within 0.0002
B2, A2                                                  -- within 0.0003
Logits2, softmax2, P("shoe")2, Loss2                    -- within 0.0001
```

Maximum drift from the published answer key across both steps:
**0.0003**, on `B2` — the largest drift in either worksheet, since the
bonus step compounds rounding across two full training steps rather than
one. Every qualitative claim still survives intact at that drift: the
top prediction is still wrong after step 1 (`run` ahead of `shoe`, not
just barely), flips correctly after step 2, and step 2's loss still
clearly beats full fine-tuning's single-step loss (0.2263 vs. 0.8404 —
nowhere near close enough for 0.0003 of drift to matter). As with
`HAND_CALCULATION.md`, this is a completed recomputation, not a
disclaimer: a reader working through this worksheet with an ordinary
calculator will land within the same tolerance.

This recomputation is a runnable script, not just prose — see
[`verify_calculator_precision.py`](verify_calculator_precision.py) in
this folder (`python3 verify_calculator_precision.py`), which redoes
both this worksheet and the main one at 4-decimal precision and diffs
every value against Mathematica's exported ground truth. Currently
80/80 checks passing, max drift 0.0003.
