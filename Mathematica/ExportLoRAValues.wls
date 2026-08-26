(* Exports every computed value needed to build Web/lora-notebook.html
   (the static, no-Mathematica-required rendering of TinyLoRAByHand.nb)
   as JSON. Run from this directory:
     wolframscript -file ExportLoRAValues.wls

   This is the only source for Web/build_lora_notebook.py's numbers -
   nothing in that page is hand-retyped from Mathematica output. *)

SetDirectory[DirectoryName[$InputFileName]];
Get["TinyTransformerByHand.wl"];
Get["TinyLoRAByHand.wl"];

Vocabulary = TinyTransformerByHand`Vocabulary;
EmbeddingMatrix = TinyTransformerByHand`EmbeddingMatrix;
WQ = TinyTransformerByHand`WQ; WK = TinyTransformerByHand`WK;
WV = TinyTransformerByHand`WV; WOut = TinyTransformerByHand`WOut;
lr = TinyTransformerByHand`ModelLearningRate;
B0 = TinyLoRAByHand`LoRAB0; A0 = TinyLoRAByHand`LoRAA0;

before = TinyTransformerByHand`ForwardPass[WOut, WQ, WK, WV, EmbeddingMatrix];

(* full fine-tune, for the comparison table *)
wOutFull = TinyTransformerByHand`GradientDescentStep[WOut, before["dWOut"], lr];
afterFull = TinyTransformerByHand`ForwardPass[wOutFull, WQ, WK, WV, EmbeddingMatrix];

(* LoRA step 1 *)
step1 = TinyLoRAByHand`RunLoRAStep[WOut, B0, A0, WQ, WK, WV, EmbeddingMatrix, lr];
B1 = step1["BNew"]; A1 = step1["ANew"];
deltaW1 = TinyLoRAByHand`LoRADelta[B1, A1];
after1 = TinyTransformerByHand`ForwardPass[WOut + deltaW1, WQ, WK, WV, EmbeddingMatrix];

(* LoRA step 2 (bonus) *)
step2 = TinyLoRAByHand`RunLoRAStep[WOut, B1, A1, WQ, WK, WV, EmbeddingMatrix, lr];
B2 = step2["BNew"]; A2 = step2["ANew"];
deltaW2 = TinyLoRAByHand`LoRADelta[B2, A2];
after2 = TinyTransformerByHand`ForwardPass[WOut + deltaW2, WQ, WK, WV, EmbeddingMatrix];
G1 = step2["dWOut"]; (* the gradient RunLoRAStep computed at the WOut+deltaW1 forward pass, i.e. step 2's starting gradient *)

data = <|
  "vocabulary" -> Vocabulary,
  "WOut" -> WOut,
  "learningRate" -> lr,
  "targetIndex" -> before["TargetIndex"],
  "h" -> N[before["h"]],

  "pBefore" -> N[before["Probabilities"]],
  "pShoeBefore" -> N[before["Probabilities"][[before["TargetIndex"]]]],
  "lossBefore" -> N[before["Loss"]],
  "predictedBefore" -> Vocabulary[[First[Ordering[before["Probabilities"], -1]]]],

  "G0" -> N[before["dWOut"]],

  "WOutFull" -> N[wOutFull],
  "pFull" -> N[afterFull["Probabilities"]],
  "pShoeFull" -> N[afterFull["Probabilities"][[afterFull["TargetIndex"]]]],
  "lossFull" -> N[afterFull["Loss"]],
  "predictedFull" -> Vocabulary[[First[Ordering[afterFull["Probabilities"], -1]]]],

  "B0" -> N[B0], "A0" -> N[A0],
  "dA0" -> N[step1["dA"]], "dB0" -> N[step1["dB"]],
  "B1" -> N[B1], "A1" -> N[A1],
  "deltaW1" -> N[deltaW1],
  "WOutEff1" -> N[WOut + deltaW1],
  "pStep1" -> N[after1["Probabilities"]],
  "pShoeStep1" -> N[after1["Probabilities"][[after1["TargetIndex"]]]],
  "lossStep1" -> N[after1["Loss"]],
  "predictedStep1" -> Vocabulary[[First[Ordering[after1["Probabilities"], -1]]]],

  "G1" -> N[G1],

  "dA1" -> N[step2["dA"]], "dB1" -> N[step2["dB"]],
  "B2" -> N[B2], "A2" -> N[A2],
  "deltaW2" -> N[deltaW2],
  "WOutEff2" -> N[WOut + deltaW2],
  "pStep2" -> N[after2["Probabilities"]],
  "pShoeStep2" -> N[after2["Probabilities"][[after2["TargetIndex"]]]],
  "lossStep2" -> N[after2["Loss"]],
  "predictedStep2" -> Vocabulary[[First[Ordering[after2["Probabilities"], -1]]]]
|>;

outputPath = "../Web/lora_values.json";
Export[outputPath, data, "JSON"];
Print["Exported ", outputPath, " (", FileByteCount[outputPath], " bytes)"];
Print["Sanity: pShoeStep1 > pShoeBefore: ", data["pShoeStep1"] > data["pShoeBefore"]];
Print["Sanity: pShoeStep2 > pShoeStep1: ", data["pShoeStep2"] > data["pShoeStep1"]];
Print["Sanity: pShoeStep2's loss beats full fine-tune's: ", data["lossStep2"] < data["lossFull"]];
