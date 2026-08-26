# Hand Calculation Worksheet

This is a companion to `TinyTransformerByHand.nb`. It gives you everything
needed to reproduce the notebook's central forward pass — and one training
update — with paper and a calculator (one that has `e^x` and `ln` buttons).
Every value below is introduced and computed inside the notebook; nothing
here is new.

The task: given the two-word input **"run casing"**, compute the model's
probability for the next token, compare it to the true next token
**"shoe"**, and see how one gradient-descent update changes that
probability.

---

## 1. Given values

**Vocabulary** (token : ID):

| Token   | ID |
|---------|----|
| `<BOS>` | 1  |
| `run`   | 2  |
| `casing`| 3  |
| `shoe`  | 4  |
| `cement`| 5  |
| `<EOS>` | 6  |

**Input tokens:** `"run"`, `"casing"` → token IDs `{2, 3}`
**Target token:** `"shoe"` → token ID `4`

**Embeddings** (dimension 2), for the two tokens in our input:

```
run    -> (1, 0)
casing -> (0, 1)
```

**Input matrix X** (row 1 = "run", row 2 = "casing"):

```
X = | 1  0 |
    | 0  1 |
```

**Weight matrices** (all fixed, all 2×2):

```
WQ = | 1  0 |      WK = | 1  1 |      WV = | 1  0 |
     | 1  1 |           | 0  1 |           | 0  1 |
```

**Output projection** WOut (2 rows × 6 columns, one column per vocabulary
token, in vocabulary order `<BOS>, run, casing, shoe, cement, <EOS>`):

```
WOut = |  0   1   0   1  -1   0 |
       |  0   0   1  -1   0  -1 |
```

**Scaling constant:** d = 2 (the embedding dimension), so scores are
divided by √2 ≈ 1.41421.

**Learning rate** for the one training step: 2.

---

## 2. Worksheet — fill in the blanks

Work through these in order. Each line only needs the matrices/values
above and simple arithmetic (matrix multiplication is just row·column dot
products; softmax needs `e^x`).

```
Q = X . WQ = ______________________

K = X . WK = ______________________

V = X . WV = ______________________

Q . Transpose[K] = ______________________

Scaled scores = (Q . Transpose[K]) / sqrt(2) = ______________________

Masked scores (row 1 = "run": position 2 is in the future, replace that
entry with -infinity; row 2 = "casing": nothing is masked) =
______________________

Attention probabilities (softmax each row of the masked scores) =
______________________

Attention output = Attention probabilities . V = ______________________
  -> take row 2 of this; call it h. This is the representation used to
     predict the token after "casing".

Logits = h . WOut  (6 numbers, one per vocabulary token) =
______________________

Probabilities = softmax(Logits) = ______________________

P("shoe") = ______________  (the probability entry for token 4)

Loss = -ln( P("shoe") ) = ______________
```

### Softmax reminder

For a row of numbers z_1, ..., z_n:

```
softmax(z_i) = e^(z_i) / (e^(z_1) + e^(z_2) + ... + e^(z_n))
```

Exponentiate every entry, add them up, then divide each exponential by
that sum. The results are always positive and always sum to 1.

---

## 3. Training step — fill in the blanks

```
One-hot target (1 at position 4, "shoe", 0 elsewhere) =
  (0, 0, 0, 1, 0, 0)

dL/dLogits = Probabilities - one-hot target = ______________________

dL/dWOut = outer product of h and dL/dLogits
         (a 2x6 matrix: row i, column j = h[i] * dLogits[j]) =
______________________

WOut_new = WOut - (learning rate) * dL/dWOut
         = WOut - 2 * dL/dWOut = ______________________

Logits_new = h . WOut_new = ______________________

Probabilities_new = softmax(Logits_new) = ______________________

P("shoe")_new = ______________

Loss_new = -ln( P("shoe")_new ) = ______________
```

Check: is `P("shoe")_new` bigger than `P("shoe")`? Is `Loss_new` smaller
than `Loss`? (They should be — that's the whole point of the update.)

---

## 4. Answer key

Values below are rounded to 4-6 significant figures. Your hand
calculation should match within rounding error.

```
Q = | 1  0 |          K = | 1  1 |          V = | 1  0 |
    | 1  1 |              | 0  1 |              | 0  1 |

Q . Transpose[K] = | 1  0 |
                    | 2  1 |

Scaled scores = | 0.70711   0      |
                 | 1.41421   0.70711 |

Masked scores = | 0.70711   -infinity |
                 | 1.41421   0.70711  |

Attention probabilities = | 1.0000   0.0000 |
                           | 0.6698   0.3302 |

Attention output = | 1.0000   0.0000 |
                    | 0.6698   0.3302 |

h = (0.6698, 0.3302)

Logits (order: <BOS>, run, casing, shoe, cement, <EOS>) =
  (0, 0.6698, 0.3302, 0.3395, -0.6698, -0.3302)

Probabilities =
  (0.1433, 0.2799, 0.1993, 0.2012, 0.0733, 0.1030)

P("shoe") = 0.2012
Loss = -ln(0.2012) = 1.6035

Predicted (highest-probability) token before training: "run" (0.2799)
-- note this is WRONG; the target is "shoe".

dL/dLogits =
  (0.1433, 0.2799, 0.1993, -0.7988, 0.0733, 0.1030)

dL/dWOut =
  |  0.0960   0.1875   0.1335  -0.5350   0.0491   0.0690 |
  |  0.0473   0.0924   0.0658  -0.2638   0.0242   0.0340 |

WOut_new =
  | -0.1919   0.6251  -0.2670   2.0700  -1.0982  -0.1379 |
  | -0.0946  -0.1849   0.8683  -0.4724  -0.0484  -1.0680 |

Logits_new =
  (-0.1598, 0.3576, 0.1079, 1.2304, -0.7515, -0.4451)

Probabilities_new =
  (0.1075, 0.1803, 0.1405, 0.4315, 0.0595, 0.0808)

P("shoe")_new = 0.4315   (up from 0.2012)
Loss_new = -ln(0.4315) = 0.8404   (down from 1.6035)

Predicted token after training: "shoe" (0.4315) -- now CORRECT.
```

One gradient-descent step, computed entirely by hand from the numbers
above, flips the model's top prediction from "run" to "shoe" for this
one example, while more than doubling P("shoe") and roughly halving the
loss.

This worksheet only reproduces the *output-layer* update (the gradient
with respect to `WOut`). See Section 12 of `TinyTransformerByHand.nb`
("What has — and has not — been updated") for why the embeddings, `WQ`,
`WK`, and `WV` are left unchanged in this particular worked example, and
for an optional, more advanced discussion of continuing the gradient
further back through attention.
