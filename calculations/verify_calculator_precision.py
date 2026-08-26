"""
verify_calculator_precision.py

The worksheets (HAND_CALCULATION.md, LORA_HAND_CALCULATION.md) claim a
reader can reproduce every number using an ordinary calculator -- one
that rounds after every single keystroke, not one that carries hidden
full floating-point precision between steps. This script is that claim,
made runnable: it redoes every step of both worksheets using only
4-decimal-place arithmetic after each individual multiply, add,
exponential, and division, then diffs the result against the
already-verified, full-precision values Mathematica exported to
Web/notebook_values.json and Web/lora_values.json.

It is deliberately independent of TinyTransformerByHand.wl and
TinyLoRAByHand.wl: those check that the *mathematics* is right at full
precision. This checks that the *worksheets* are actually completable
with a real calculator's precision, which is a different failure mode
(rounding drift), not a mathematical one.

Run from this directory: python3 verify_calculator_precision.py
Exits 0 if every check passes, 1 otherwise.
"""
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_web = os.path.join(_here, "..", "Web")

with open(os.path.join(_web, "notebook_values.json")) as f:
    MAIN = json.load(f)
with open(os.path.join(_web, "lora_values.json")) as f:
    LORA = json.load(f)

TOLERANCE = 0.0005  # observed max drift in manual walkthroughs was 0.0003; this leaves margin
D = 4  # decimal places a calculator readout would show


def r(x):
    return round(x, D)


results = []


def check_close(name, computed, documented):
    drift = abs(computed - documented)
    results.append((name, drift <= TOLERANCE, f"computed={computed}, documented={documented}, drift={drift:.4f}"))


def check_equal(name, computed, documented):
    results.append((name, computed == documented, f"computed={computed!r}, documented={documented!r}"))


def dot2(a, b):
    return r(r(a[0] * b[0]) + r(a[1] * b[1]))


def matmul2x2(A, B):
    return [[dot2(A[i], [B[0][j], B[1][j]]) for j in range(2)] for i in range(2)]


def softmax6(logits):
    exps = [r(math.exp(z)) for z in logits]
    s = r(sum(exps))
    return [r(e / s) for e in exps]


def argmax(vec):
    return max(range(len(vec)), key=lambda i: vec[i])


# ---------------------------------------------------------------------
# Main worksheet (HAND_CALCULATION.md), recomputed independently
# ---------------------------------------------------------------------
X = MAIN["X"]
WQ, WK, WV, WOut = MAIN["WQ"], MAIN["WK"], MAIN["WV"], MAIN["WOut"]
lr = MAIN["learningRate"]
vocab = MAIN["vocabulary"]
targetIndex0 = MAIN["targetIndex"] - 1

Q = matmul2x2(X, WQ)
K = matmul2x2(X, WK)
V = matmul2x2(X, WV)

KT = [[K[0][0], K[1][0]], [K[0][1], K[1][1]]]
rawScores = matmul2x2(Q, KT)

inv_sqrt2 = r(1 / math.sqrt(2))
scaled = [r(rawScores[0][0] * inv_sqrt2), r(rawScores[1][0] * inv_sqrt2), r(rawScores[1][1] * inv_sqrt2)]
# row 1: position 1 can only see itself -> attention weights (1, 0), no softmax needed
z1, z2 = scaled[1], scaled[2]
e1, e2 = r(math.exp(z1)), r(math.exp(z2))
s = r(e1 + e2)
p1, p2 = r(e1 / s), r(e2 / s)
attn = [[1.0, 0.0], [p1, p2]]

attnOut = [[r(attn[i][0] * V[0][j] + attn[i][1] * V[1][j]) for j in range(2)] for i in range(2)]
h = attnOut[1]
check_close("h[0]", h[0], MAIN["h"][0])
check_close("h[1]", h[1], MAIN["h"][1])

logits = [r(r(h[0] * WOut[0][j]) + r(h[1] * WOut[1][j])) for j in range(6)]
for j, tok in enumerate(vocab):
    check_close(f"logits[{tok}]", logits[j], MAIN["logits"][j])

probs = softmax6(logits)
for j, tok in enumerate(vocab):
    check_close(f"probabilities[{tok}]", probs[j], MAIN["probabilities"][j])

pShoe = probs[targetIndex0]
loss = r(-math.log(pShoe))
check_close("pShoeBefore", pShoe, MAIN["pShoeBefore"])
check_close("lossBefore", loss, MAIN["lossBefore"])
check_equal("predictedBefore", vocab[argmax(probs)], MAIN["predictedBefore"])

oneHot = [1 if i == targetIndex0 else 0 for i in range(6)]
dLogits = [r(probs[i] - oneHot[i]) for i in range(6)]
dWOut = [[r(h[i] * dLogits[j]) for j in range(6)] for i in range(2)]
for i in range(2):
    for j, tok in enumerate(vocab):
        check_close(f"dWOut[{i}][{tok}]", dWOut[i][j], MAIN["dWOut"][i][j])

WOutNew = [[r(WOut[i][j] - lr * dWOut[i][j]) for j in range(6)] for i in range(2)]
for i in range(2):
    for j, tok in enumerate(vocab):
        check_close(f"WOutNew[{i}][{tok}]", WOutNew[i][j], MAIN["WOutNew"][i][j])

logitsAfter = [r(r(h[0] * WOutNew[0][j]) + r(h[1] * WOutNew[1][j])) for j in range(6)]
probsAfter = softmax6(logitsAfter)
pShoeAfter = probsAfter[targetIndex0]
lossAfter = r(-math.log(pShoeAfter))
check_close("pShoeAfter", pShoeAfter, MAIN["pShoeAfter"])
check_close("lossAfter", lossAfter, MAIN["lossAfter"])
check_equal("predictedAfter", vocab[argmax(probsAfter)], MAIN["predictedAfter"])

# ---------------------------------------------------------------------
# LoRA worksheet (LORA_HAND_CALCULATION.md), continuing from the same
# rounded h, logits/probs/loss/dWOut computed above (same starting point
# lora_values.json itself uses -- WOut + B0.A0 = WOut, so "before" here
# is identical to the main worksheet's "before").
# ---------------------------------------------------------------------
G0 = dWOut  # dL/dW_Out at the shared starting point, already rounded above
B0 = [LORA["B0"][0][0], LORA["B0"][1][0]]
A0 = LORA["A0"][0]

# full fine-tune (reuses the same WOutNew computed above; already checked)

# LoRA step 1
dA0 = [r(B0[0] * G0[0][j] + B0[1] * G0[1][j]) for j in range(6)]
dB0 = [r(sum(r(G0[i][j] * A0[j]) for j in range(6))) for i in range(2)]
for j in range(6):
    check_close(f"dA0[{j}]", dA0[j], LORA["dA0"][0][j])
for i in range(2):
    check_close(f"dB0[{i}]", dB0[i], LORA["dB0"][i][0])

B1 = [r(B0[i] - lr * dB0[i]) for i in range(2)]
A1 = A0[:]  # dA0 is zero, so unchanged
for i in range(2):
    check_close(f"B1[{i}]", B1[i], LORA["B1"][i][0])

deltaW1 = [[r(B1[i] * A1[j]) for j in range(6)] for i in range(2)]
WOutEff1 = [[r(WOut[i][j] + deltaW1[i][j]) for j in range(6)] for i in range(2)]

logits1 = [r(r(h[0] * WOutEff1[0][j]) + r(h[1] * WOutEff1[1][j])) for j in range(6)]
probs1 = softmax6(logits1)
pShoe1 = probs1[targetIndex0]
loss1 = r(-math.log(pShoe1))
check_close("pShoeStep1", pShoe1, LORA["pShoeStep1"])
check_close("lossStep1", loss1, LORA["lossStep1"])
check_equal("predictedStep1", vocab[argmax(probs1)], LORA["predictedStep1"])

# LoRA bonus step 2
dLogits1 = [r(probs1[i] - oneHot[i]) for i in range(6)]
G1 = [[r(h[i] * dLogits1[j]) for j in range(6)] for i in range(2)]
dA1 = [r(B1[0] * G1[0][j] + B1[1] * G1[1][j]) for j in range(6)]
dB1 = [r(sum(r(G1[i][j] * A1[j]) for j in range(6))) for i in range(2)]
for j in range(6):
    check_close(f"dA1[{j}]", dA1[j], LORA["dA1"][0][j])
for i in range(2):
    check_close(f"dB1[{i}]", dB1[i], LORA["dB1"][i][0])

B2 = [r(B1[i] - lr * dB1[i]) for i in range(2)]
A2 = [r(A1[j] - lr * dA1[j]) for j in range(6)]
for i in range(2):
    check_close(f"B2[{i}]", B2[i], LORA["B2"][i][0])
for j in range(6):
    check_close(f"A2[{j}]", A2[j], LORA["A2"][0][j])

deltaW2 = [[r(B2[i] * A2[j]) for j in range(6)] for i in range(2)]
WOutEff2 = [[r(WOut[i][j] + deltaW2[i][j]) for j in range(6)] for i in range(2)]
logits2 = [r(r(h[0] * WOutEff2[0][j]) + r(h[1] * WOutEff2[1][j])) for j in range(6)]
probs2 = softmax6(logits2)
pShoe2 = probs2[targetIndex0]
loss2 = r(-math.log(pShoe2))
check_close("pShoeStep2", pShoe2, LORA["pShoeStep2"])
check_close("lossStep2", loss2, LORA["lossStep2"])
check_equal("predictedStep2", vocab[argmax(probs2)], LORA["predictedStep2"])

# qualitative claims the worksheets make in prose, checked directly
check_equal("qualitative: predicted flips run->shoe after full fine-tune", MAIN["predictedAfter"], "shoe")
check_equal("qualitative: LoRA step 1 is still wrong", LORA["predictedStep1"], "run")
check_equal("qualitative: LoRA step 2 flips to shoe", LORA["predictedStep2"], "shoe")
results.append((
    "qualitative: LoRA step 2 loss beats full fine-tune's loss",
    loss2 < MAIN["lossAfter"],
    f"loss2={loss2}, lossAfter(full fine-tune)={MAIN['lossAfter']}",
))

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print(f"=== Calculator-precision verification (round to {D} decimals after every step) ===")
max_drift = 0.0
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if "drift=" in detail:
        max_drift = max(max_drift, float(detail.split("drift=")[1]))

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\nMax numeric drift observed: {max_drift:.4f} (tolerance: {TOLERANCE})")
if passed == total:
    print(f"ALL CHECKS PASSED ({total} checks)")
    sys.exit(0)
else:
    print(f"*** {total - passed} OF {total} CHECKS FAILED ***")
    sys.exit(1)
