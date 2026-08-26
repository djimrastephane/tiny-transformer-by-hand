(* ::Package:: *)

(* =====================================================================
   TinyLoRAByHand.wl

   Companion to TinyTransformerByHand.wl. Same frozen tiny transformer
   (same vocabulary, same embeddings, same WQ/WK/WV, same starting
   WOut, same single training example "run casing" -> "shoe"), but
   instead of updating every entry of WOut directly (full fine-tuning),
   this file freezes WOut and learns a rank-1 correction

       DeltaW = B . A

   where B is 2x1 and A is 1x6, so DeltaW is 2x6 -- the same shape as
   WOut -- but built from only 8 numbers instead of WOut's 12, and (as
   the checks below demonstrate) only 2 of those 8 numbers actually
   move on the very first training step.

   This is NOT a general-purpose LoRA library. It is the same kind of
   object as TinyTransformerByHand.wl: small enough to run entirely by
   hand, with every claim checked programmatically rather than typed
   in and hoped for.
   ===================================================================== *)

BeginPackage["TinyLoRAByHand`", {"TinyTransformerByHand`"}];

LoRARank::usage = "The rank r of the LoRA update used throughout this file. Fixed at 1.";
LoRAB0::usage = "Initial B factor, 2x1. Zero, per standard LoRA practice, so DeltaW = B.A starts as a no-op.";
LoRAA0::usage = "Initial A factor, 1x6. A fixed, deterministic 'random-like' vector (no randomness anywhere in this file, same convention as TinyTransformerByHand.wl).";

LoRADelta::usage = "LoRADelta[B_,A_] returns B.A, the rank-1 correction to WOut.";
LoRABackward::usage = "LoRABackward[G_,B_,A_] returns {dA,dB}, the gradients of the loss with respect to the LoRA factors, given G = dL/d(WOut+DeltaW) (i.e. OutputWeightGradient[h,dLogits] at the current step).";
RunLoRAStep::usage = "RunLoRAStep[wOutBase_,B_,A_,wq_,wk_,wv_,emb_,lr_] runs one forward pass at WOut=wOutBase+B.A, then one gradient step on B and A (WOut itself is never touched). Returns an Association with the forward-pass result plus the updated B, A, and gradients.";
RunLoRAChecks::usage = "RunLoRAChecks[] runs the two-step LoRA training example alongside one step of ordinary full fine-tuning on the same starting point, and prints a pass/fail report comparing them.";

Begin["`Private`"];

LoRARank = 1;

LoRAB0 = {{0}, {0}};
LoRAA0 = {{1, -1, 1, -1, 1, -1}};

LoRADelta[B_, A_] := B.A;

LoRABackward[G_, B_, A_] := {Transpose[B].G, G.Transpose[A]};

RunLoRAStep[wOutBase_, B_, A_, wq_, wk_, wv_, emb_, lr_] := Module[
  {wOutEffective, fp, dA, dB, bNew, aNew},

  wOutEffective = wOutBase + LoRADelta[B, A];
  fp = ForwardPass[wOutEffective, wq, wk, wv, emb];

  {dA, dB} = LoRABackward[fp["dWOut"], B, A];
  bNew = GradientDescentStep[B, dB, lr];
  aNew = GradientDescentStep[A, dA, lr];

  Append[fp, <|"B" -> B, "A" -> A, "DeltaW" -> LoRADelta[B, A], "dA" -> dA, "dB" -> dB, "BNew" -> bNew, "ANew" -> aNew|>]
];

RunLoRAChecks[] := Module[
  {results = {}, addCheck, lr, before, wOutFull, afterFull,
   step1, step2, B1, A1, B2, A2},

  addCheck[name_, cond_] := AppendTo[results, name -> TrueQ[cond]];
  lr = ModelLearningRate;

  (* --- shared starting point --- *)
  before = ForwardPass[WOut, WQ, WK, WV, EmbeddingMatrix];

  addCheck["B0 is 2x1", Dimensions[LoRAB0] === {2, 1}];
  addCheck["A0 is 1x6", Dimensions[LoRAA0] === {1, 6}];
  addCheck["DeltaW0 = B0.A0 is 2x6", Dimensions[LoRADelta[LoRAB0, LoRAA0]] === {2, 6}];
  addCheck["DeltaW0 is the zero matrix (LoRA starts as a no-op)",
    LoRADelta[LoRAB0, LoRAA0] == ConstantArray[0, {2, 6}]];

  (* --- step 1 --- *)
  step1 = RunLoRAStep[WOut, LoRAB0, LoRAA0, WQ, WK, WV, EmbeddingMatrix, lr];

  addCheck["LoRA step-1 forward pass starts from the same WOut+DeltaW0 = WOut, so matches the shared baseline",
    step1["Logits"] == before["Logits"]];
  addCheck["dA is exactly zero at step 1 (DeltaW/dA depends on B, and B0=0)",
    step1["dA"] == ConstantArray[0, {1, 6}]];
  addCheck["dB is NOT zero at step 1 (something has to move)",
    step1["dB"] != ConstantArray[0, {2, 1}]];

  B1 = step1["BNew"]; A1 = step1["ANew"];
  addCheck["A is unchanged after step 1 (gradient was exactly zero)", A1 == LoRAA0];
  addCheck["B has changed after step 1", B1 != LoRAB0];

  (* --- full fine-tuning, one step, same starting point --- *)
  wOutFull = GradientDescentStep[WOut, before["dWOut"], lr];
  afterFull = ForwardPass[wOutFull, WQ, WK, WV, EmbeddingMatrix];

  addCheck["Full fine-tune: P(shoe) increases after its one step",
    afterFull["Probabilities"][[afterFull["TargetIndex"]]] > before["Probabilities"][[before["TargetIndex"]]]];
  addCheck["Full fine-tune: loss decreases after its one step",
    afterFull["Loss"] < before["Loss"]];

  (* --- LoRA after its own step 1 (WOut + B1.A1) --- *)
  step1 = Append[step1, <|"After" -> ForwardPass[WOut + LoRADelta[B1, A1], WQ, WK, WV, EmbeddingMatrix]|>];
  addCheck["LoRA step 1: P(shoe) increases even though only B (2 numbers) moved",
    step1["After"]["Probabilities"][[step1["After"]["TargetIndex"]]] > before["Probabilities"][[before["TargetIndex"]]]];
  addCheck["LoRA step 1: loss decreases even though only B (2 numbers) moved",
    step1["After"]["Loss"] < before["Loss"]];
  addCheck["Full fine-tune's one step (12 numbers move) beats LoRA's one step (2 numbers move)",
    afterFull["Loss"] < step1["After"]["Loss"]];
  addCheck["Full fine-tune's one step flips the top prediction to \"shoe\"",
    Vocabulary[[First[Ordering[afterFull["Probabilities"], -1]]]] === "shoe"];
  addCheck["LoRA step 1's top prediction has NOT yet flipped to \"shoe\" (still \"run\")",
    Vocabulary[[First[Ordering[step1["After"]["Probabilities"], -1]]]] =!= "shoe"];

  (* --- step 2 (bonus): now B1 is nonzero, so A should start moving too --- *)
  step2 = RunLoRAStep[WOut, B1, A1, WQ, WK, WV, EmbeddingMatrix, lr];
  addCheck["dA is NOT zero at step 2 (B is nonzero now, so A finally gets a gradient)",
    step2["dA"] != ConstantArray[0, {1, 6}]];

  B2 = step2["BNew"]; A2 = step2["ANew"];
  step2 = Append[step2, <|"After" -> ForwardPass[WOut + LoRADelta[B2, A2], WQ, WK, WV, EmbeddingMatrix]|>];

  addCheck["LoRA step 2: loss keeps decreasing",
    step2["After"]["Loss"] < step1["After"]["Loss"]];
  addCheck["LoRA step 2: P(shoe) keeps increasing",
    step2["After"]["Probabilities"][[step2["After"]["TargetIndex"]]] >
      step1["After"]["Probabilities"][[step1["After"]["TargetIndex"]]]];
  addCheck["LoRA, given one more (cheap) step, overtakes full fine-tuning's single step",
    step2["After"]["Loss"] < afterFull["Loss"]];
  addCheck["LoRA step 2's top prediction has now flipped to \"shoe\"",
    Vocabulary[[First[Ordering[step2["After"]["Probabilities"], -1]]]] === "shoe"];

  Print["=== TinyLoRAByHand verification ==="];
  Scan[
    Print[If[Last[#], "  [PASS] ", "  [FAIL] "], First[#]] &,
    results
  ];
  If[AllTrue[results, Last],
    Print["ALL CHECKS PASSED (", Length[results], " checks)"],
    Print["*** SOME CHECKS FAILED ***"]
  ];

  <|
    "Before" -> before,
    "FullFTWOutNew" -> wOutFull, "AfterFullFT" -> afterFull,
    "Step1" -> step1, "Step2" -> step2,
    "Checks" -> results
  |>
];

End[];
EndPackage[];
