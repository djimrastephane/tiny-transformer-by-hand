(* Exports every computed value needed to build Web/notebook.html (the
   static, no-Mathematica-required rendering of this notebook) as JSON.
   Run from this directory: wolframscript -file ExportNotebookValues.wls

   This is the only source for Web/build_notebook.py's numbers - nothing
   in that page is hand-retyped from Mathematica output. *)

SetDirectory[DirectoryName[$InputFileName]];
Get["TinyTransformerByHand.wl"];

Vocabulary = TinyTransformerByHand`Vocabulary;
EmbeddingMatrix = TinyTransformerByHand`EmbeddingMatrix;
WQ = TinyTransformerByHand`WQ; WK = TinyTransformerByHand`WK;
WV = TinyTransformerByHand`WV; WOut = TinyTransformerByHand`WOut;
lr = TinyTransformerByHand`ModelLearningRate;

before = TinyTransformerByHand`ForwardPass[WOut, WQ, WK, WV, EmbeddingMatrix];
wOutNew = TinyTransformerByHand`GradientDescentStep[WOut, before["dWOut"], lr];
after = TinyTransformerByHand`ForwardPass[wOutNew, WQ, WK, WV, EmbeddingMatrix];

cementBefore = TinyTransformerByHand`ForwardPassGeneral[WOut, WQ, WK, WV, EmbeddingMatrix, "cement"];
cementWOutNew = TinyTransformerByHand`GradientDescentStep[WOut, cementBefore["dWOut"], lr];
cementAfter = TinyTransformerByHand`ForwardPassGeneral[cementWOutNew, WQ, WK, WV, EmbeddingMatrix, "cement"];

(* finite-difference gradient check, matching notebook Section 12 *)
eps = 10^-6;
wOutPerturbed = WOut; wOutPerturbed[[1, 4]] += eps;
beforePerturbed = TinyTransformerByHand`ForwardPass[wOutPerturbed, WQ, WK, WV, EmbeddingMatrix];
numericSlope = (beforePerturbed["Loss"] - before["Loss"])/eps;

data = <|
  "vocabulary" -> Vocabulary,
  "embeddingMatrix" -> EmbeddingMatrix,
  "WQ" -> WQ, "WK" -> WK, "WV" -> WV, "WOut" -> WOut,
  "learningRate" -> lr,
  "inputTokens" -> {"run", "casing"},
  "tokenIDs" -> before["TokenIDs"],
  "X" -> before["X"],
  "Q" -> before["Q"], "K" -> before["K"], "V" -> before["V"],
  "rawScores" -> before["RawScores"],
  "scaledScores" -> N[before["ScaledScores"]],
  "maskedScores" -> (N[before["MaskedScores"]] /. -Infinity -> "-Infinity"),
  "attentionWeights" -> N[before["AttentionWeights"]],
  "attentionOutput" -> N[before["AttentionOutput"]],
  "h" -> N[before["h"]],
  "logits" -> N[before["Logits"]],
  "probabilities" -> N[before["Probabilities"]],
  "targetIndex" -> before["TargetIndex"],
  "pShoeBefore" -> N[before["Probabilities"][[before["TargetIndex"]]]],
  "lossBefore" -> N[before["Loss"]],
  "dLogits" -> N[before["dLogits"]],
  "dWOut" -> N[before["dWOut"]],
  "numericSlope" -> N[numericSlope],
  "analyticSlope" -> N[before["dWOut"][[1, 4]]],
  "WOutNew" -> N[wOutNew],
  "WOutDelta" -> N[wOutNew - WOut],
  "logitsAfter" -> N[after["Logits"]],
  "probabilitiesAfter" -> N[after["Probabilities"]],
  "pShoeAfter" -> N[after["Probabilities"][[after["TargetIndex"]]]],
  "lossAfter" -> N[after["Loss"]],
  "predictedBefore" -> Vocabulary[[First[Ordering[before["Probabilities"], -1]]]],
  "predictedAfter" -> Vocabulary[[First[Ordering[after["Probabilities"], -1]]]],
  "cementTargetIndex" -> cementBefore["TargetIndex"],
  "pCementBefore" -> N[cementBefore["Probabilities"][[cementBefore["TargetIndex"]]]],
  "lossCementBefore" -> N[cementBefore["Loss"]],
  "pCementAfter" -> N[cementAfter["Probabilities"][[cementAfter["TargetIndex"]]]],
  "lossCementAfter" -> N[cementAfter["Loss"]],
  "probabilitiesCementAfter" -> N[cementAfter["Probabilities"]],
  "predictedCementBefore" -> Vocabulary[[First[Ordering[cementBefore["Probabilities"], -1]]]],
  "predictedCementAfter" -> Vocabulary[[First[Ordering[cementAfter["Probabilities"], -1]]]],
  "builtInSoftmaxAgrees" -> (Max[Abs[before["Probabilities"] - TinyTransformerByHand`Softmax[before["Logits"]]]] < 10^-10)
|>;

outputPath = "../Web/notebook_values.json";
Export[outputPath, data, "JSON"];
Print["Exported ", outputPath, " (", FileByteCount[outputPath], " bytes)"];
Print["Sanity: pShoeAfter > pShoeBefore: ", data["pShoeAfter"] > data["pShoeBefore"]];
Print["Sanity: pCementAfter > pCementBefore: ", data["pCementAfter"] > data["pCementBefore"]];
