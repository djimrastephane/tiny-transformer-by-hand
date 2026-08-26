(* ::Package:: *)

(* =====================================================================
   TinyTransformerByHand.wl

   Reference implementation of a deliberately tiny transformer language
   model, small enough that every number in it can be reproduced with
   paper and a calculator.

   This file exists for two reasons:

     1. It is the single source of truth for every number that appears
        in TinyTransformerByHand.nb and HAND_CALCULATION.md. If you
        change a weight here, re-run this file to see how every
        downstream quantity changes.

     2. It runs a battery of sanity checks (dimensions, softmax sums,
        causal masking, sign of the gradient update, etc.) so that the
        notebook's claims are checked programmatically, not just typed
        in and hoped for.

   This is NOT a general-purpose transformer library. It intentionally
   avoids NetChain / NetGraph / NetTrain / Predict / Classify. Every
   operation is ordinary matrix arithmetic so nothing is hidden behind
   Mathematica's neural-network framework.
   ===================================================================== *)

BeginPackage["TinyTransformerByHand`"];

Vocabulary::usage = "List of the 6 vocabulary tokens, in ID order (ID = position in the list).";
EmbeddingMatrix::usage = "6x2 matrix. Row i is the embedding of Vocabulary[[i]].";
WQ::usage = "2x2 query weight matrix.";
WK::usage = "2x2 key weight matrix.";
WV::usage = "2x2 value weight matrix.";
WOut::usage = "2x6 output projection matrix (embedding dim -> vocabulary logits).";
ModelLearningRate::usage = "Scalar learning rate used for the single worked training step.";

TokenIDs::usage = "TokenIDs[tokens_List] gives the vocabulary ID (1-indexed) for each token string.";
EmbedTokens::usage = "EmbedTokens[ids_List] returns the input matrix X (rows = positions, columns = embedding dims).";
ComputeQKV::usage = "ComputeQKV[X_,wq_,wk_,wv_] returns {Q,K,V}.";
AttentionScoresRaw::usage = "AttentionScoresRaw[Q_,K_] returns Q.Transpose[K].";
ScaleScores::usage = "ScaleScores[scores_,d_] returns N[scores/Sqrt[d]] (numericized immediately, matching the notebook's own convention, so downstream operations like Ordering never have to compare deeply nested exact expressions).";
CausalMask::usage = "CausalMask[scores_] replaces disallowed (future) positions with -Infinity.";
SoftmaxRow::usage = "SoftmaxRow[row_] applies the softmax formula to a single row, ignoring -Infinity entries.";
AttentionProbabilities::usage = "AttentionProbabilities[maskedScores_] applies SoftmaxRow to every row.";
AttentionOutputMatrix::usage = "AttentionOutputMatrix[attn_,V_] returns attn.V.";
Logits::usage = "Logits[h_,wOut_] returns h.wOut, a length-6 vector.";
Softmax::usage = "Softmax[z_] returns the softmax probability vector for logit vector z.";
CrossEntropyLoss::usage = "CrossEntropyLoss[p_,targetIndex_] returns -Log[p[[targetIndex]]].";
OneHot::usage = "OneHot[targetIndex_,n_] returns a length-n one-hot vector.";
LogitGradient::usage = "LogitGradient[p_,targetIndex_] returns p - oneHot(targetIndex), i.e. dL/dlogits.";
OutputWeightGradient::usage = "OutputWeightGradient[h_,dLogits_] returns the gradient of the loss with respect to WOut, i.e. Outer[Times,h,dLogits].";
GradientDescentStep::usage = "GradientDescentStep[w_,grad_,lr_] returns w - lr*grad.";
ForwardPass::usage = "ForwardPass[wOutMatrix_, wqMatrix_, wkMatrix_, wvMatrix_, embMatrix_] runs the full forward pass for the fixed input \"run casing\" and returns an Association of every intermediate quantity.";
ForwardPassGeneral::usage = "ForwardPassGeneral[wOutMatrix_, wqMatrix_, wkMatrix_, wvMatrix_, embMatrix_, targetToken_] is ForwardPass generalized to an arbitrary target token, for demonstrating that the same mechanism works for any training target, not just \"shoe\".";
RunAllChecks::usage = "RunAllChecks[] runs the full worked example end to end and prints a pass/fail report for every mathematical property the notebook relies on.";
RunAlternateTargetChecks::usage = "RunAlternateTargetChecks[targetToken_] runs the same class of checks as RunAllChecks[], but training toward targetToken instead of \"shoe\", to confirm the mechanism generalizes.";

Begin["`Private`"];

(* ---------------------------------------------------------------------
   1. Vocabulary and embeddings
   --------------------------------------------------------------------- *)

Vocabulary = {"<BOS>", "run", "casing", "shoe", "cement", "<EOS>"};

(* Row i = embedding of Vocabulary[[i]]. Dimension 2, deliberately simple:
   "run" and "casing" are the two standard basis vectors, so the input
   matrix X for our example sentence is literally the identity matrix. *)
EmbeddingMatrix = {
  {0, 0},   (* <BOS>   *)
  {1, 0},   (* run     *)
  {0, 1},   (* casing  *)
  {1, 1},   (* shoe    *)
  {-1, 1},  (* cement  *)
  {0, -1}   (* <EOS>   *)
};

(* ---------------------------------------------------------------------
   2. Model parameters (fixed, deterministic - no randomness anywhere)
   --------------------------------------------------------------------- *)

WQ = {{1, 0}, {1, 1}};
WK = {{1, 1}, {0, 1}};
WV = {{1, 0}, {0, 1}};

(* 2 (embedding dim) x 6 (vocabulary size). Column j is the projection
   direction for Vocabulary[[j]]. *)
WOut = {
  {0, 1, 0,  1, -1,  0},
  {0, 0, 1, -1,  0, -1}
};

ModelLearningRate = 2;

(* ---------------------------------------------------------------------
   3. Forward-pass building blocks
   --------------------------------------------------------------------- *)

TokenIDs[tokens_List] := Flatten[Position[Vocabulary, #] & /@ tokens];

EmbedTokens[ids_List] := EmbeddingMatrix[[ids]];

ComputeQKV[X_, wq_, wk_, wv_] := {X.wq, X.wk, X.wv};

AttentionScoresRaw[Q_, K_] := Q.Transpose[K];

ScaleScores[scores_, d_] := N[scores/Sqrt[d]];

(* Position i may only attend to position j <= i (no looking at the future). *)
CausalMask[scores_] := Module[{n = Length[scores]},
  Table[
    If[j <= i, scores[[i, j]], -Infinity],
    {i, n}, {j, n}
  ]
];

SoftmaxRow[row_] := Module[{finite, m, ex},
  finite = Select[row, # > -Infinity &];
  m = Max[finite];
  ex = If[# > -Infinity, Exp[# - m], 0] & /@ row;
  ex/Total[ex]
];

AttentionProbabilities[maskedScores_] := SoftmaxRow /@ maskedScores;

AttentionOutputMatrix[attn_, V_] := attn.V;

Logits[h_, wOut_] := h.wOut;

Softmax[z_] := Module[{m = Max[z], ex},
  ex = Exp[z - m];
  ex/Total[ex]
];

OneHot[targetIndex_, n_] := ReplacePart[ConstantArray[0, n], targetIndex -> 1];

CrossEntropyLoss[p_, targetIndex_] := -Log[p[[targetIndex]]];

LogitGradient[p_, targetIndex_] := p - OneHot[targetIndex, Length[p]];

OutputWeightGradient[h_, dLogits_] := Outer[Times, h, dLogits];

GradientDescentStep[w_, grad_, lr_] := w - lr*grad;

(* ---------------------------------------------------------------------
   4. Full forward pass, parameterized so it can be re-run after the
      training update (or from the interactive Manipulate) without
      duplicating logic.
   --------------------------------------------------------------------- *)

ForwardPass[wOutMatrix_, wqMatrix_, wkMatrix_, wvMatrix_, embMatrix_] := Module[
  {inputTokens, ids, X, Q, K, V, rawScores, scaled, masked, attn, attnOut, h,
   logits, probs, targetIndex, loss, dLogits, dWOut},

  inputTokens = {"run", "casing"};
  ids = TokenIDs[inputTokens];
  X = embMatrix[[ids]];

  {Q, K, V} = ComputeQKV[X, wqMatrix, wkMatrix, wvMatrix];

  rawScores = AttentionScoresRaw[Q, K];
  scaled = ScaleScores[rawScores, 2];
  masked = CausalMask[scaled];
  attn = AttentionProbabilities[masked];
  attnOut = AttentionOutputMatrix[attn, V];

  h = attnOut[[2]]; (* representation at the final position, used to predict the next token *)

  logits = Logits[h, wOutMatrix];
  probs = Softmax[logits];

  targetIndex = First[TokenIDs[{"shoe"}]];
  loss = CrossEntropyLoss[probs, targetIndex];
  dLogits = LogitGradient[probs, targetIndex];
  dWOut = OutputWeightGradient[h, dLogits];

  <|
    "InputTokens" -> inputTokens,
    "TokenIDs" -> ids,
    "X" -> X,
    "Q" -> Q, "K" -> K, "V" -> V,
    "RawScores" -> rawScores,
    "ScaledScores" -> scaled,
    "MaskedScores" -> masked,
    "AttentionWeights" -> attn,
    "AttentionOutput" -> attnOut,
    "h" -> h,
    "Logits" -> logits,
    "Probabilities" -> probs,
    "TargetIndex" -> targetIndex,
    "TargetToken" -> "shoe",
    "Loss" -> loss,
    "dLogits" -> dLogits,
    "dWOut" -> dWOut
  |>
];

(* ---------------------------------------------------------------------
   4b. Same forward pass, generalized to an arbitrary target token.
       Everything through h is identical to ForwardPass -- attention
       never looks at the target. Only the loss/gradient section
       changes. Kept as a separate function (rather than adding a
       parameter to ForwardPass) so nothing that already depends on
       ForwardPass's exact behavior is at risk of changing.
   --------------------------------------------------------------------- *)

ForwardPassGeneral[wOutMatrix_, wqMatrix_, wkMatrix_, wvMatrix_, embMatrix_, targetToken_String] := Module[
  {inputTokens, ids, X, Q, K, V, rawScores, scaled, masked, attn, attnOut, h,
   logits, probs, targetIndex, loss, dLogits, dWOut},

  inputTokens = {"run", "casing"};
  ids = TokenIDs[inputTokens];
  X = embMatrix[[ids]];

  {Q, K, V} = ComputeQKV[X, wqMatrix, wkMatrix, wvMatrix];

  rawScores = AttentionScoresRaw[Q, K];
  scaled = ScaleScores[rawScores, 2];
  masked = CausalMask[scaled];
  attn = AttentionProbabilities[masked];
  attnOut = AttentionOutputMatrix[attn, V];

  h = attnOut[[2]];

  logits = Logits[h, wOutMatrix];
  probs = Softmax[logits];

  targetIndex = First[TokenIDs[{targetToken}]];
  loss = CrossEntropyLoss[probs, targetIndex];
  dLogits = LogitGradient[probs, targetIndex];
  dWOut = OutputWeightGradient[h, dLogits];

  <|
    "InputTokens" -> inputTokens,
    "TokenIDs" -> ids,
    "X" -> X,
    "Q" -> Q, "K" -> K, "V" -> V,
    "RawScores" -> rawScores,
    "ScaledScores" -> scaled,
    "MaskedScores" -> masked,
    "AttentionWeights" -> attn,
    "AttentionOutput" -> attnOut,
    "h" -> h,
    "Logits" -> logits,
    "Probabilities" -> probs,
    "TargetIndex" -> targetIndex,
    "TargetToken" -> targetToken,
    "Loss" -> loss,
    "dLogits" -> dLogits,
    "dWOut" -> dWOut
  |>
];

(* ---------------------------------------------------------------------
   5. Verification suite
   --------------------------------------------------------------------- *)

RunAllChecks[] := Module[
  {before, wOutNew, after, results = {}, addCheck},

  addCheck[name_, cond_] := AppendTo[results, name -> TrueQ[cond]];

  before = ForwardPass[WOut, WQ, WK, WV, EmbeddingMatrix];

  (* dimensions *)
  addCheck["X is 2x2", Dimensions[before["X"]] === {2, 2}];
  addCheck["Q is 2x2", Dimensions[before["Q"]] === {2, 2}];
  addCheck["K is 2x2", Dimensions[before["K"]] === {2, 2}];
  addCheck["V is 2x2", Dimensions[before["V"]] === {2, 2}];
  addCheck["Attention weights are 2x2", Dimensions[before["AttentionWeights"]] === {2, 2}];
  addCheck["Attention output is 2x2", Dimensions[before["AttentionOutput"]] === {2, 2}];
  addCheck["h is length 2", Length[before["h"]] === 2];
  addCheck["Logits is length 6", Length[before["Logits"]] === 6];
  addCheck["Probabilities is length 6", Length[before["Probabilities"]] === 6];
  addCheck["dLogits is length 6", Length[before["dLogits"]] === 6];
  addCheck["dWOut is 2x6", Dimensions[before["dWOut"]] === {2, 6}];

  (* softmax rows sum to 1 *)
  addCheck["Attention row 1 sums to 1", Abs[Total[before["AttentionWeights"][[1]]] - 1] < 10^-9];
  addCheck["Attention row 2 sums to 1", Abs[Total[before["AttentionWeights"][[2]]] - 1] < 10^-9];
  addCheck["Output probabilities sum to 1", Abs[Total[before["Probabilities"]] - 1] < 10^-9];

  (* causal mask behaves correctly: position 1 must have zero weight on position 2 *)
  addCheck["Causal mask: position 1 cannot see position 2", before["AttentionWeights"][[1, 2]] == 0];
  addCheck["Causal mask: position 2 CAN see position 1", before["AttentionWeights"][[2, 1]] > 0];

  (* target extraction *)
  addCheck["Target index points to \"shoe\"", Vocabulary[[before["TargetIndex"]]] === "shoe"];
  addCheck["P(shoe) equals Probabilities[[TargetIndex]]",
    before["Probabilities"][[before["TargetIndex"]]] == before["Probabilities"][[4]]];

  (* cross-entropy sanity: loss = -Log[p_target], strictly positive since p_target < 1 *)
  addCheck["Loss equals -Log[P(shoe)]",
    Abs[before["Loss"] - (-Log[before["Probabilities"][[before["TargetIndex"]]]])] < 10^-12];
  addCheck["Loss is positive", before["Loss"] > 0];

  (* gradient sanity: dLogits sums to 0 (probabilities sum to 1, one-hot sums to 1) *)
  addCheck["dLogits sums to ~0", Abs[Total[before["dLogits"]]] < 10^-9];
  (* the only negative entry of dLogits should be at the target index, since p_j - 0 >= 0 for j != target
     and p_target - 1 < 0 *)
  addCheck["dLogits is negative only at the target index",
    Count[before["dLogits"], x_ /; x < 0] == 1 && before["dLogits"][[before["TargetIndex"]]] < 0];

  (* gradient descent update uses correct sign: W_new = W_old - lr*grad *)
  wOutNew = GradientDescentStep[WOut, before["dWOut"], ModelLearningRate];
  addCheck["Update formula matches W - lr*grad",
    wOutNew == WOut - ModelLearningRate*before["dWOut"]];
  addCheck["Updated weights actually differ from old weights",
    wOutNew != WOut];

  after = ForwardPass[wOutNew, WQ, WK, WV, EmbeddingMatrix];

  addCheck["Updated model actually uses updated WOut (logits changed)",
    after["Logits"] != before["Logits"]];
  addCheck["P(shoe) increases after the update",
    after["Probabilities"][[after["TargetIndex"]]] > before["Probabilities"][[before["TargetIndex"]]]];
  addCheck["Loss decreases after the update", after["Loss"] < before["Loss"]];

  Print["=== TinyTransformerByHand verification ==="];
  Scan[
    Print[If[Last[#], "  [PASS] ", "  [FAIL] "], First[#]] &,
    results
  ];
  If[AllTrue[results, Last],
    Print["ALL CHECKS PASSED (", Length[results], " checks)"],
    Print["*** SOME CHECKS FAILED ***"]
  ];

  <|"Before" -> before, "WOutNew" -> wOutNew, "After" -> after, "Checks" -> results|>
];

RunAlternateTargetChecks[targetToken_String] := Module[
  {before, wOutNew, after, results = {}, addCheck},

  addCheck[name_, cond_] := AppendTo[results, name -> TrueQ[cond]];

  before = ForwardPassGeneral[WOut, WQ, WK, WV, EmbeddingMatrix, targetToken];

  addCheck["Target index points to \"" <> targetToken <> "\"",
    Vocabulary[[before["TargetIndex"]]] === targetToken];
  addCheck["Output probabilities sum to 1",
    Abs[Total[before["Probabilities"]] - 1] < 10^-9];
  addCheck["Loss equals -Log[P(target)]",
    Abs[before["Loss"] - (-Log[before["Probabilities"][[before["TargetIndex"]]]])] < 10^-12];
  addCheck["dLogits is negative only at the target index",
    Count[before["dLogits"], x_ /; x < 0] == 1 && before["dLogits"][[before["TargetIndex"]]] < 0];

  wOutNew = GradientDescentStep[WOut, before["dWOut"], ModelLearningRate];
  addCheck["Updated weights actually differ from old weights", wOutNew != WOut];

  after = ForwardPassGeneral[wOutNew, WQ, WK, WV, EmbeddingMatrix, targetToken];

  addCheck["P(" <> targetToken <> ") increases after the update",
    after["Probabilities"][[after["TargetIndex"]]] > before["Probabilities"][[before["TargetIndex"]]]];
  addCheck["Loss decreases after the update", after["Loss"] < before["Loss"]];

  Print["=== Alternate-target verification: \"" <> targetToken <> "\" ==="];
  Scan[
    Print[If[Last[#], "  [PASS] ", "  [FAIL] "], First[#]] &,
    results
  ];
  If[AllTrue[results, Last],
    Print["ALL CHECKS PASSED (", Length[results], " checks)"],
    Print["*** SOME CHECKS FAILED ***"]
  ];

  <|"Before" -> before, "WOutNew" -> wOutNew, "After" -> after, "Checks" -> results|>
];

End[];
EndPackage[];
