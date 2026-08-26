"""
Builds Web/notebook.html: a static, no-Mathematica-required rendering of
TinyTransformerByHand.nb, from values Mathematica already computed and
verified (never hand-retyped here).

Regenerate after any change to the notebook's numbers:
  1. wolframscript -file ../Mathematica/ExportNotebookValues.wls
  2. python3 build_notebook.py
"""
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_here, 'notebook_values.json')) as f:
    D = json.load(f)

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

VOCAB_RAW = D['vocabulary']
VOCAB = [esc(t) for t in VOCAB_RAW]
for k in ['predictedBefore', 'predictedAfter', 'predictedCementBefore', 'predictedCementAfter']:
    D[k] = esc(D[k])

def fnum(x, d=4):
    if x == "-Infinity":
        return "&minus;&infin;"
    return f"{x:.{d}f}"

def pct(x, d=1):
    return f"{x*100:.{d}f}%"

def matbox(name, M, d=4, highlight=None):
    """M is either a flat list (vector) or list of lists (matrix)."""
    is_matrix = isinstance(M[0], list)
    rows = M if is_matrix else [M]
    cols = len(rows[0])
    cells = []
    idx = 0
    for row in rows:
        for v in row:
            cls = "cell"
            if v != "-Infinity" and isinstance(v, (int, float)) and v < 0:
                cls += " neg"
            if highlight and idx in highlight:
                cls += " hl"
            cells.append(f'<div class="{cls}">{fnum(v, d) if isinstance(v, (int,float)) or v=="-Infinity" else v}</div>')
            idx += 1
    return (f'<div class="matbox" style="grid-template-columns:repeat({cols},1fr);">'
            f'<div class="mname">{name}</div>' + "".join(cells) + '</div>')

def vocab_table(rows, headers):
    thead = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for r in rows:
        trs += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table class="datatable"><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>'

def prob_bars(probs, target_idx_0based, predicted_token=None):
    rows = ""
    for i, tok in enumerate(VOCAB):
        p = probs[i]
        is_target = (i == target_idx_0based)
        cls = "barrow" + (" is-target" if is_target else "")
        rows += (f'<div class="{cls}"><span class="bname">{tok}</span>'
                  f'<span class="btrack"><span class="bfill" style="width:{p*100:.4f}%"></span></span>'
                  f'<span class="bpct">{pct(p)}</span></div>')
    return f'<div class="barlist">{rows}</div>'

CEMENT_IDX0 = D['cementTargetIndex'] - 1  # convert 1-indexed Mathematica -> 0-indexed
SHOE_IDX0 = D['targetIndex'] - 1

html = f"""<meta charset="utf-8">
<title>Tiny Transformer Notebook</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  :root{{
    --bg:#0B0F1A; --panel:#131A2B; --panel-raised:#182238; --line:#263353; --line-soft:#1B2540;
    --text:#E7ECF5; --text-dim:#8592AC; --text-faint:#5B6784;
    --amber:#E8A33D; --teal:#45C7B8; --bad:#E2604F; --good:#5FD08A;
    --focus:#7FC9FF; --shadow:0 8px 30px rgba(0,0,0,0.35);
  }}
  :root[data-theme="light"], :root:not([data-theme="dark"]) {{
    --bg:#EDF0F5; --panel:#FFFFFF; --panel-raised:#F6F8FC; --line:#C9D2E2; --line-soft:#DCE3EE;
    --text:#121826; --text-dim:#51617D; --text-faint:#8A97B2;
    --amber:#9A5F00; --teal:#0E7A70; --bad:#B23A2C; --good:#177A47;
    --focus:#0B63C5; --shadow:0 8px 24px rgba(30,40,70,0.12);
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --bg:#0B0F1A; --panel:#131A2B; --panel-raised:#182238; --line:#263353; --line-soft:#1B2540;
      --text:#E7ECF5; --text-dim:#8592AC; --text-faint:#5B6784;
      --amber:#E8A33D; --teal:#45C7B8; --bad:#E2604F; --good:#5FD08A;
      --focus:#7FC9FF; --shadow:0 8px 30px rgba(0,0,0,0.35);
    }}
  }}
  :root[data-theme="dark"]{{
    --bg:#0B0F1A; --panel:#131A2B; --panel-raised:#182238; --line:#263353; --line-soft:#1B2540;
    --text:#E7ECF5; --text-dim:#8592AC; --text-faint:#5B6784;
    --amber:#E8A33D; --teal:#45C7B8; --bad:#E2604F; --good:#5FD08A;
    --focus:#7FC9FF; --shadow:0 8px 30px rgba(0,0,0,0.35);
  }}

  *{{ box-sizing:border-box; }}
  html,body{{ margin:0; padding:0; }}
  body{{
    background:var(--bg); color:var(--text);
    font-family:"IBM Plex Sans", system-ui, sans-serif;
    -webkit-font-smoothing:antialiased; line-height:1.6;
  }}
  a{{ color:var(--teal); }}
  a:focus-visible{{ outline:2px solid var(--focus); outline-offset:2px; }}
  h1,h2,h3{{ font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif; text-wrap:balance; margin:0; }}
  .mono{{ font-family:"IBM Plex Mono", monospace; font-variant-numeric:tabular-nums; }}

  .wrap{{ max-width:820px; margin:0 auto; padding:32px 20px 100px; }}

  header.top{{
    display:flex; align-items:baseline; justify-content:space-between; gap:16px;
    padding-bottom:18px; margin-bottom:8px; border-bottom:1px solid var(--line); flex-wrap:wrap;
  }}
  .kicker{{ font-family:"IBM Plex Mono",monospace; font-size:0.72rem; letter-spacing:0.16em; color:var(--text-faint); text-transform:uppercase; }}
  header.top h1{{ font-size:1.7rem; margin-top:4px; }}
  .badge{{
    font-family:"IBM Plex Mono",monospace; font-size:0.68rem; letter-spacing:0.05em; color:var(--text-dim);
    border:1px solid var(--line); border-radius:3px; padding:5px 9px; white-space:nowrap;
  }}
  .toplinks{{ font-size:0.85rem; color:var(--text-dim); margin:14px 0 40px; }}

  section{{ margin-bottom:40px; }}
  .sec-label{{
    font-family:"IBM Plex Mono",monospace; font-size:0.72rem; letter-spacing:0.12em; color:var(--teal);
    text-transform:uppercase; margin-bottom:10px;
  }}
  h2.sec-title{{ font-size:1.35rem; margin-bottom:14px; }}
  h3.sub-title{{ font-size:1.05rem; color:var(--text); margin:22px 0 10px; }}
  p{{ color:var(--text-dim); max-width:70ch; }}
  p b, li b {{ color: var(--text); }}
  .panel{{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:20px; box-shadow:var(--shadow); }}

  table.datatable{{ border-collapse:collapse; width:100%; margin:14px 0; font-size:0.92rem; }}
  table.datatable th, table.datatable td{{
    border-bottom:1px solid var(--line-soft); padding:8px 10px; text-align:left; font-family:"IBM Plex Mono",monospace;
  }}
  table.datatable th{{ color:var(--text-faint); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }}
  table.datatable td{{ color:var(--text); }}
  table.datatable tr.hl td{{ color:var(--amber); font-weight:600; }}

  .flow{{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:14px 0; }}
  .op{{ color:var(--text-faint); font-size:1.1rem; }}
  .matbox{{
    font-family:"IBM Plex Mono",monospace; font-size:0.85rem;
    background:var(--panel-raised); border:1px solid var(--line); border-radius:4px;
    padding:10px 12px; display:inline-grid; gap:3px 14px;
  }}
  .matbox .mname{{ grid-column:1/-1; color:var(--text-faint); font-size:0.65rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px; }}
  .matbox .cell{{ text-align:right; color:var(--text); min-width:2.6em; }}
  .matbox .cell.neg{{ color:var(--bad); }}
  .matbox .cell.hl{{ color:var(--amber); font-weight:600; }}

  .barlist{{ display:flex; flex-direction:column; gap:8px; margin:14px 0; }}
  .barrow{{ display:grid; grid-template-columns:70px 1fr 60px; align-items:center; gap:10px; font-size:0.85rem; }}
  .bname{{ font-family:"IBM Plex Mono",monospace; color:var(--text-dim); text-align:right; }}
  .barrow.is-target .bname{{ color:var(--amber); font-weight:600; }}
  .btrack{{ background:var(--line-soft); border-radius:3px; height:14px; overflow:hidden; }}
  .bfill{{ height:100%; background:var(--text-faint); border-radius:3px 0 0 3px; }}
  .barrow.is-target .bfill{{ background:var(--amber); }}
  .bpct{{ font-family:"IBM Plex Mono",monospace; text-align:right; }}

  .note{{ font-size:0.88rem; color:var(--text-dim); background:var(--panel-raised); border:1px solid var(--line);
    border-radius:6px; padding:14px 16px; margin:14px 0; }}
  .note b{{ color:var(--text); }}

  .disclose{{ background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
  .disclose > summary{{
    list-style:none; cursor:pointer; display:flex; align-items:center; justify-content:space-between;
    padding:14px 18px; font-family:"IBM Plex Sans Condensed",sans-serif; text-transform:uppercase;
    letter-spacing:0.04em; font-weight:600; font-size:0.85rem; color:var(--text);
  }}
  .disclose > summary::-webkit-details-marker{{ display:none; }}
  .disclose > summary .chev{{ color:var(--teal); transition:transform 0.2s ease; }}
  .disclose[open] > summary .chev{{ transform:rotate(90deg); }}
  .disclose-body{{ padding:0 18px 18px; border-top:1px solid var(--line-soft); }}

  .compare{{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line);
    border-radius:8px; overflow:hidden; margin:16px 0; }}
  .compare > div{{ background:var(--panel); padding:18px; }}
  .compare .cap{{ font-family:"IBM Plex Mono",monospace; font-size:0.68rem; letter-spacing:0.1em; color:var(--text-faint);
    text-transform:uppercase; margin-bottom:10px; }}
  .compare .after .cap{{ color:var(--good); }}
  .metric{{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--line-soft); }}
  .metric:last-child{{ border-bottom:none; }}
  .metric .k{{ color:var(--text-dim); font-size:0.85rem; }}
  .metric .v{{ font-family:"IBM Plex Mono",monospace; }}
  .compare .after .v{{ color:var(--good); }}

  ul.checklist{{ list-style:none; padding:0; margin:14px 0; columns:2; column-gap:24px; }}
  ul.checklist li{{ font-size:0.82rem; color:var(--text-dim); padding:4px 0; break-inside:avoid; }}
  ul.checklist li::before{{ content:"\\2713  "; color:var(--good); font-weight:700; }}

  .cta-box{{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; padding:20px; }}
  .cta-box p{{ margin:4px 0 0; max-width:52ch; }}
  a.cta{{
    font-family:"IBM Plex Sans Condensed",sans-serif; font-weight:600; letter-spacing:0.04em; text-transform:uppercase;
    background:var(--amber); color:#1a1204; border-radius:4px; padding:10px 18px; font-size:0.82rem;
    text-decoration:none; white-space:nowrap;
  }}

  footer{{ border-top:1px solid var(--line); padding-top:20px; margin-top:60px; color:var(--text-faint); font-size:0.8rem; }}
  footer a {{ color: var(--teal); }}

  @media (max-width:600px){{
    .compare{{ grid-template-columns:1fr; }}
    ul.checklist{{ columns:1; }}
  }}
</style>

<div class="wrap">

  <header class="top">
    <div>
      <div class="kicker">Tiny Transformer &mdash; By Hand</div>
      <h1>The Full Worked Notebook</h1>
    </div>
    <span class="badge">static export &middot; no Mathematica needed</span>
  </header>

  <p class="toplinks">
    This page is a static, read-only rendering of <code>Mathematica/TinyTransformerByHand.nb</code> &mdash;
    every number below was computed and verified in Wolfram Mathematica, then exported directly (no hand-retyping).
    It has no interactive sliders; for that, see the
    <a href="./index.html">live browser demo</a>. Prefer paper and a calculator? See
    <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/calculations/HAND_CALCULATION.md">the hand-calculation worksheet</a>.
  </p>

  <section>
    <p>Modern language models contain billions of parameters and perform trillions of arithmetic operations per response. Nobody reproduces that by hand, and nobody needs to &mdash; but the arithmetic itself is not exotic. It is addition, multiplication, exponentials, and logarithms, arranged in a particular pattern called a transformer.</p>
    <p>This notebook shrinks that pattern down until every number fits on a sheet of paper: a vocabulary of 6 words, sequences of 2 tokens, an embedding dimension of 2, tens of parameters total. Every matrix multiplication, every softmax, every gradient below can be checked with a calculator.</p>
    <p>The example is a drilling/completions phrase: given the two-word input <b>&ldquo;run casing&rdquo;</b>, the model should predict that the next word is <b>&ldquo;shoe&rdquo;</b> (as in a casing shoe). This is a toy transformer <b>language model</b> &mdash; not an LLM, and not any kind of reproduction of ChatGPT or similar production systems.</p>
    <p>Stated with full technical scope: this is an intentionally reduced causal self-attention language model for tracing the mathematics of next-token training &mdash; not a complete transformer block. Its main forward path has no feed-forward network, no residual connections, no LayerNorm, and no positional encoding; the next section explains why each omission is fine at this scale.</p>
  </section>

  <section>
    <h3 class="sub-title">What we left out, and why</h3>
    <p>LayerNorm, residual connections, dropout, multi-head attention, a large feed-forward network, and positional encodings are all omitted &mdash; each one would add notation without adding insight at this scale (1 head, 1 block, tens of parameters). The positional-encoding omission gets a sharper caveat: self-attention itself has no built-in sense of order. A causal mask only tells position 1 from position 2 here because, with exactly 2 tokens, there are only 2 possible masks; it says what a position may look at, not where it sits. Real transformers add an explicit positional signal for exactly this reason. Full reasoning: <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/methodology/ASSUMPTIONS.md">methodology/ASSUMPTIONS.md</a>.</p>
  </section>

  <section>
    <div class="sec-label">Section 1</div>
    <h2 class="sec-title">The Prediction Problem</h2>
    <p>Given &ldquo;run casing&rdquo;, we want the model to predict the single most likely next word. The model does not know English or drilling engineering &mdash; it manipulates numbers derived from the input tokens, arranged so that, after training, the right answer gets a high probability. In plain language, a language model computes P(next token | previous tokens): the probability of the next token, given the tokens that came before it.</p>
  </section>

  <section>
    <div class="sec-label">Section 2</div>
    <h2 class="sec-title">Vocabulary and Token IDs</h2>
    <p>The vocabulary has 6 tokens, each with an ID (its position in the list):</p>
    {vocab_table([[t, i+1] for i, t in enumerate(VOCAB)], ["Token", "Token ID"])}
    <p>&ldquo;run casing&rdquo; becomes token IDs <span class="mono">{D['tokenIDs']}</span>.</p>
  </section>

  <section>
    <div class="sec-label">Section 3</div>
    <h2 class="sec-title">Embedding Lookup</h2>
    <p>Each token ID is replaced with a length-2 vector. &ldquo;run&rdquo; is set to (1, 0) and &ldquo;casing&rdquo; to (0, 1) &mdash; the two simplest possible distinct vectors in 2 dimensions.</p>
    {vocab_table([[t, str(D['embeddingMatrix'][i])] for i, t in enumerate(VOCAB)], ["Token", "Embedding"])}
    <p>Selecting rows 2 and 3 builds the input matrix X (2 rows for 2 tokens, 2 columns for 2 embedding dimensions):</p>
    <div class="flow">{matbox("X", D['X'], 0)}</div>
    <p>Because &ldquo;run&rdquo; and &ldquo;casing&rdquo; happen to be the two standard unit vectors, X is exactly the identity matrix &mdash; a convenient coincidence that makes the next multiplications easy to check by hand.</p>
  </section>

  <section>
    <div class="sec-label">Section 4</div>
    <h2 class="sec-title">Query, Key, and Value Matrices</h2>
    <p>Three fixed 2&times;2 weight matrices, W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>:</p>
    <div class="flow">{matbox("W_Q", D['WQ'], 0)}{matbox("W_K", D['WK'], 0)}{matbox("W_V", D['WV'], 0)}</div>
    <p>Query, key, and value are ordinary matrix products:</p>
    <div class="flow">{matbox("X", D['X'], 0)}<span class="op">&middot;</span>{matbox("W_Q", D['WQ'], 0)}<span class="op">=</span>{matbox("Q", D['Q'], 0)}</div>
    <div class="flow">{matbox("X", D['X'], 0)}<span class="op">&middot;</span>{matbox("W_K", D['WK'], 0)}<span class="op">=</span>{matbox("K", D['K'], 0)}</div>
    <div class="flow">{matbox("X", D['X'], 0)}<span class="op">&middot;</span>{matbox("W_V", D['WV'], 0)}<span class="op">=</span>{matbox("V", D['V'], 0)}</div>
    <div class="note"><b>Q, K, and V are token-dependent, not fixed matrices.</b> They equal W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> here only because X happens to be the identity matrix. Feed different tokens in and Q, K, V change while the weight matrices don't &mdash; see the notebook's Section 4 for a worked counterexample, or drag the embedding slider in the <a href="./index.html">live demo</a>.</div>
  </section>

  <section>
    <div class="sec-label">Section 5</div>
    <h2 class="sec-title">Attention Scores</h2>
    <p>Q &middot; K<sup>T</sup> measures how well each query aligns with each key:</p>
    <div class="flow">{matbox("Q", D['Q'], 0)}<span class="op">&middot;</span>{matbox("K^T", [[D['K'][0][0],D['K'][1][0]],[D['K'][0][1],D['K'][1][1]]], 0)}<span class="op">=</span>{matbox("raw", D['rawScores'], 0)}</div>
    <p>Dividing by &radic;d (d = 2) keeps scores in a stable range regardless of embedding dimension, preventing the softmax below from becoming extremely peaked:</p>
    <div class="flow">{matbox("scaled = raw / &radic;2", D['scaledScores'])}</div>
  </section>

  <section>
    <div class="sec-label">Section 6</div>
    <h2 class="sec-title">Causal Mask</h2>
    <p>Position 1 (&ldquo;run&rdquo;) cannot look at position 2 (&ldquo;casing&rdquo;) &mdash; that would let it see part of the very continuation it's predicting. Position 2 may look at everything up to and including itself. The disallowed entry is set to &minus;&infin; so the softmax below gives it exactly zero weight:</p>
    <div class="flow">{matbox("masked scores", D['maskedScores'])}</div>
  </section>

  <section>
    <div class="sec-label">Section 7</div>
    <h2 class="sec-title">Softmax</h2>
    <p>Softmax[z<sub>i</sub>] = Exp[z<sub>i</sub>] / Sum[Exp[z<sub>j</sub>]]. Every score is exponentiated, then divided by the row's total, so each row becomes a valid probability distribution. Here it is worked by hand for row 2 (the &ldquo;casing&rdquo; row, the one with two real numbers to compare) &mdash; exactly the kind of calculation you could type into a phone calculator:</p>
    <div class="flow">{matbox("row 2 scores", D['maskedScores'][1])}<span class="op">Exp &rarr;</span>{matbox("exp", D['row2Exps'])}</div>
    <p>Sum of exponentials: <span class="mono">{fnum(D['row2Exps'][0])} + {fnum(D['row2Exps'][1])} = {fnum(D['row2SumExps'])}</span>. Divide each exponential by that sum:</p>
    <div class="flow">{matbox("row 2 probabilities", [D['row2Exps'][0]/D['row2SumExps'], D['row2Exps'][1]/D['row2SumExps']])}</div>
    <p>Now the same formula, row by row, for the whole matrix:</p>
    <div class="flow">{matbox("attention weights", D['attentionWeights'])}</div>
    <p>Row 1 is (1, 0): position 1 puts all its attention on itself, since the mask left it no choice. Row 2 matches the hand calculation above: about (0.670, 0.330) &mdash; position 2 splits its attention mostly onto position 1, with some weight on itself. Both rows sum to 1.</p>
  </section>

  <section>
    <div class="sec-label">Section 8</div>
    <h2 class="sec-title">Weighted Values</h2>
    <p>AttentionOutput = AttentionWeights &middot; V &mdash; blending each position's value vectors by its attention weights:</p>
    <div class="flow">{matbox("attn", D['attentionWeights'])}<span class="op">&middot;</span>{matbox("V", D['V'], 0)}<span class="op">=</span>{matbox("attnOut", D['attentionOutput'])}</div>
    <p>Row 2 &mdash; the representation built at position 2 after mixing in position 1 &mdash; is what predicts the token after &ldquo;casing&rdquo;. Call it <b>h</b>:</p>
    <div class="flow">{matbox("h", D['h'], highlight=[0,1])}</div>
  </section>

  <section>
    <div class="sec-label">Section 9</div>
    <h2 class="sec-title">Output Projection and Logits</h2>
    <p>W<sub>Out</sub> (2 rows &times; 6 columns) converts h into a score for every vocabulary token:</p>
    <div class="flow">{matbox("W_Out", D['WOut'], 0)}</div>
    {vocab_table([[t, fnum(D['logits'][i], 4)] for i, t in enumerate(VOCAB)], ["Token", "Logit"])}
    <p>These are logits, not probabilities: they can be negative, don't sum to 1, and only rank tokens relative to each other. &ldquo;run&rdquo; and &ldquo;shoe&rdquo; have the two largest logits, with run's roughly twice shoe's.</p>
    <h3 class="sub-title">Optional: what a feed-forward layer would add</h3>
    <p>Logits = h &middot; W<sub>Out</sub> is one linear map; the only nonlinearity anywhere in this notebook is the softmax. Real transformer blocks add one more explicit nonlinearity &mdash; ReLU(h &middot; W1 + b1) &middot; W2 + b2 &mdash; per position. Applying just the ReLU half to h (a standalone illustration, not part of the trained example):</p>
    <div class="flow">{matbox("h", D['h'])}<span class="op">&middot;</span>{matbox("W1", [[1,-1],[-1,1]], 0)}<span class="op">=</span>{matbox("pre-act", [D['h'][0]-D['h'][1], D['h'][1]-D['h'][0]])}<span class="op">ReLU &rarr;</span>{matbox("post-act", [max(0,D['h'][0]-D['h'][1]), max(0,D['h'][1]-D['h'][0])])}</div>
    <p>The negative coordinate becomes exactly zero &mdash; ReLU has discarded that direction entirely.</p>
  </section>

  <section>
    <div class="sec-label">Section 10</div>
    <h2 class="sec-title">Next-Token Probabilities</h2>
    {vocab_table([[t, fnum(D['logits'][i],4), pct(D['probabilities'][i])] for i, t in enumerate(VOCAB)], ["Token", "Logit", "Probability"])}
    {prob_bars(D['probabilities'], SHOE_IDX0)}
    <p>Before any training, the model's highest-probability guess is <b>&ldquo;{D['predictedBefore']}&rdquo;</b> ({pct(D['probabilities'][1])}), not the correct answer &ldquo;shoe&rdquo; ({pct(D['pShoeBefore'])}, second-highest &mdash; just ahead of &ldquo;casing&rdquo;).</p>
  </section>

  <section>
    <div class="sec-label">Section 11</div>
    <h2 class="sec-title">Cross-Entropy Loss</h2>
    <p>Loss = &minus;Log[P(correct token)]. A perfect prediction (P=1) gives loss 0; as P shrinks toward 0, the loss grows without bound &mdash; a heavy penalty for confidently ruling out the right answer.</p>
    <div class="compare" style="grid-template-columns:1fr;">
      <div><div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeBefore'])}</span></div>
      <div class="metric"><span class="k">Loss = &minus;Log[P(&ldquo;shoe&rdquo;)]</span><span class="v mono">{fnum(D['lossBefore'])}</span></div></div>
    </div>
  </section>

  <section>
    <div class="sec-label">Section 12</div>
    <h2 class="sec-title">One Training Step</h2>
    <p>For softmax + cross-entropy, the gradient of the loss with respect to the logits has a simple closed form: dL/dLogits = probabilities &minus; oneHot(target).</p>
    <div class="flow">{matbox("dLogits = P - oneHot", D['dLogits'], highlight=[SHOE_IDX0])}</div>
    <p>Every entry is positive except &ldquo;shoe&rdquo;'s ({fnum(D['dLogits'][SHOE_IDX0])}) &mdash; increasing any wrong token's logit would raise the loss; increasing shoe's would lower it.</p>
    <p>The gradient with respect to W<sub>Out</sub> is the outer product of h and dLogits:</p>
    <div class="flow">{matbox("dW_Out = Outer[Times, h, dLogits]", D['dWOut'])}</div>
    <div class="note">
      <b>Checked numerically, not just asserted.</b> Nudging W<sub>Out</sub>[[1,4]] by &epsilon;=10<sup>-6</sup> and recomputing the loss gives a finite-difference slope of <span class="mono">{fnum(D['numericSlope'],6)}</span>, matching the analytic gradient's <span class="mono">{fnum(D['analyticSlope'],6)}</span> to about 6 significant figures.
    </div>
    <p>This training step computes and applies a gradient for <b>W<sub>Out</sub> only</b>. W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, and the embeddings are left unchanged &mdash; a deliberate scope limit, since propagating further requires the softmax attention Jacobian, not a simple subtraction. See the notebook's Section 12 for the further chain-rule sketch.</p>
    <div class="note">Put plainly: in this worked update, the model learns only at the output projection. The attention mechanism participates fully in the forward pass &mdash; it produced the h this update trains on &mdash; but its parameters (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, the embeddings) are frozen for this step. This lets us demonstrate real gradient-based learning without introducing the much larger attention-softmax Jacobian.</div>
  </section>

  <section>
    <div class="sec-label">Section 13</div>
    <h2 class="sec-title">Gradient Descent Update</h2>
    <p>W<sub>new</sub> = W<sub>old</sub> &minus; learningRate &times; gradient, with learning rate {D['learningRate']}:</p>
    <div class="flow">{matbox("W_Out (new)", D['WOutNew'])}</div>
    <p style="font-size:0.85rem;">A deliberately large learning rate of {D['learningRate']} is used so the effect of a single update is visually obvious in one step &mdash; not a recommended learning rate for training real neural networks, which use many small steps rather than one large one.</p>
    <p>The change (new &minus; old) is not uniform &mdash; column 4 (&ldquo;shoe&rdquo;) moved the most, in the direction that increases its logit:</p>
    <div class="flow">{matbox("&Delta; W_Out", D['WOutDelta'], highlight=[3,9])}</div>
  </section>

  <section>
    <div class="sec-label">Section 14</div>
    <h2 class="sec-title">Run the Model Again</h2>
    <p>Q, K, V, and h don't change (only W<sub>Out</sub> was updated), so we reuse h and recompute just the last two steps:</p>
    <div class="compare">
      <div class="before"><div class="cap">Before training</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeBefore'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossBefore'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedBefore']}</span></div>
      </div>
      <div class="after"><div class="cap">After 1 update</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeAfter'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossAfter'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedAfter']}</span></div>
      </div>
    </div>
    <p>Both checked computationally, not assumed: P(&ldquo;shoe&rdquo;) rose from {pct(D['pShoeBefore'])} to {pct(D['pShoeAfter'])}, and the loss fell from {fnum(D['lossBefore'])} to {fnum(D['lossAfter'])}. The predicted token flips from &ldquo;{D['predictedBefore']}&rdquo; to &ldquo;{D['predictedAfter']}&rdquo; &mdash; one gradient step was enough to change the model's top guess.</p>
    {prob_bars(D['probabilitiesAfter'], SHOE_IDX0)}
  </section>

  <section>
    <div class="sec-label">Section 15</div>
    <h2 class="sec-title">What Just Happened?</h2>
    <ul style="color:var(--text-dim);">
      <li>The forward pass turned &ldquo;run casing&rdquo; into a full probability distribution over the vocabulary.</li>
      <li>Cross-entropy loss measured exactly how wrong that distribution was, given the true answer &ldquo;shoe&rdquo;.</li>
      <li>The gradient measured how each W<sub>Out</sub> parameter would affect that loss if nudged &mdash; checked numerically, not just derived.</li>
      <li>Gradient descent moved every weight a small step opposite its gradient.</li>
      <li>Running the identical input again through the updated weights produced a better distribution: higher P(&ldquo;shoe&rdquo;), lower loss.</li>
    </ul>
    <p>That loop &mdash; predict, measure error, compute gradient, update weights, predict again &mdash; is the numerical core of how transformer language models learn. Real pretraining repeats this basic learning cycle over enormous datasets and very large parameter sets, while different training methods may update all or only selected parameters (this notebook's own update, on just W<sub>Out</sub>, is itself an example of the latter). This notebook doesn't claim one hand-worked update captures everything about training a real model &mdash; only that the arithmetic at the heart of it is exactly this.</p>
  </section>

  <section>
    <div class="sec-label">Section 16</div>
    <h2 class="sec-title">Scale Comparison</h2>
    {vocab_table([
        ["Vocabulary size", "~6 tokens", "tens of thousands to ~1,000,000+"],
        ["Embedding dimension", "2", "hundreds to tens of thousands"],
        ["Attention heads", "1", "dozens, run in parallel per layer"],
        ["Transformer blocks", "1", "dozens to over 100, stacked"],
        ["Parameters", "tens", "billions, sometimes hundreds of billions"],
        ["Training corpus", "1 next-token example", "up to trillions of training tokens"],
        ["Training updates", "1 (shown by hand)", "millions to billions"],
    ], ["", "Toy model (this notebook)", "A modern pretrained language model"])}
    <p>The underlying operations remain ordinary numerical operations. Production models combine them at vastly greater scale, with additional architectural, training, and systems complexity: (per the omissions above) LayerNorm, residual connections, multi-head attention, larger feed-forward networks, positional encodings, and more.</p>
  </section>

  <section>
    <p style="font-size:0.85rem; color:var(--text-faint);">The main story is complete as of Section 16 above. What follows is supplementary evidence, not the headline result.</p>
    <details class="disclose">
      <summary>Bonus: does this generalize? (training toward a different target) <span class="chev">&#9656;</span></summary>
      <div class="disclose-body">
        <p>Sections 2&ndash;8 never look at the target token &mdash; X, Q, K, V, attention, and h depend only on the input. So does the exact same mechanism work retargeted at <b>&ldquo;cement&rdquo;</b> instead of &ldquo;shoe&rdquo; (cementing a casing string is an equally real next step)? h is unchanged; only the loss, gradient, and update differ.</p>
        <div class="compare">
          <div class="before"><div class="cap">Before training</div>
            <div class="metric"><span class="k">P(&ldquo;cement&rdquo;)</span><span class="v mono">{fnum(D['pCementBefore'])}</span></div>
            <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossCementBefore'])}</span></div>
            <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedCementBefore']}</span></div>
          </div>
          <div class="after"><div class="cap">After 1 update</div>
            <div class="metric"><span class="k">P(&ldquo;cement&rdquo;)</span><span class="v mono">{fnum(D['pCementAfter'])}</span></div>
            <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossCementAfter'])}</span></div>
            <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedCementAfter']}</span></div>
          </div>
        </div>
        <p>&ldquo;cement&rdquo; starts as the single lowest-probability token of all 6 ({pct(D['pCementBefore'])}) &mdash; a much worse starting guess than &ldquo;shoe&rdquo; had ({pct(D['pShoeBefore'])}). The same mechanism, same learning rate, still raises P(&ldquo;cement&rdquo;) to {pct(D['pCementAfter'])} and lowers the loss to {fnum(D['lossCementAfter'])} &mdash; but &ldquo;cement&rdquo; only barely overtakes &ldquo;run&rdquo; for the top spot ({pct(D['probabilitiesCementAfter'][4])} vs. {pct(D['probabilitiesCementAfter'][1])}, a margin of about {(D['probabilitiesCementAfter'][4]-D['probabilitiesCementAfter'][1])*100:.2f} points), nowhere near shoe's decisive win. <b>Generalizing correctly does not mean generalizing identically</b> &mdash; the same size of step covers proportionally less ground when there's further to go.</p>
      </div>
    </details>
  </section>

  <section>
    <div class="panel cta-box">
      <div>
        <h3 class="sub-title" style="margin-top:0;">Want to explore this interactively?</h3>
        <p>This static page can't run the notebook's Manipulate sliders. The live browser demo covers exactly that: drag the learning rate, an embedding value, and a training-step switch, and watch every number above recompute in real time.</p>
      </div>
      <a class="cta" href="./index.html">Open the live demo</a>
    </div>
  </section>

  <section>
    <div class="sec-label">Verification</div>
    <h2 class="sec-title">Every Claim Above, Checked Programmatically</h2>
    <p>Before any number here was trusted, it was checked by <code>TinyTransformerByHand.wl</code>'s automated suite &mdash; 34 checks in total, all passing:</p>
    <ul class="checklist">
      <li>Every matrix has the dimensions it should</li>
      <li>Every softmax row sums to 1</li>
      <li>The causal mask zeroes the disallowed weight</li>
      <li>Position 2 can still see position 1</li>
      <li>The target probability is pulled from the right index</li>
      <li>Cross-entropy equals &minus;Log[P(target)]</li>
      <li>The loss is strictly positive</li>
      <li>dLogits sums to ~0 and is negative only at the target</li>
      <li>The update formula matches W &minus; lr&middot;gradient</li>
      <li>Updated weights actually differ from the old ones</li>
      <li>The updated model actually uses the updated weights</li>
      <li>P(&ldquo;shoe&rdquo;) increases after the update</li>
      <li>Loss decreases after the update</li>
      <li>All 7 of the above repeat correctly for the &ldquo;cement&rdquo; alternate target</li>
      <li>Built-in Softmax[] agrees with the hand-written version: {str(D['builtInSoftmaxAgrees']).lower()}</li>
    </ul>
  </section>

  <footer>
    <p>Full project: <a href="https://github.com/djimrastephane/tiny-transformer-by-hand">github.com/djimrastephane/tiny-transformer-by-hand</a>
    &middot; <a href="./index.html">Live demo</a>
    &middot; <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/Mathematica/TinyTransformerByHand.nb">Source notebook</a>
    &middot; <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/methodology/ASSUMPTIONS.md">Methodology</a></p>
    <p>Companion page: <a href="./lora-notebook.html">Tiny LoRA, By Hand</a> &mdash; freezes W<sub>Out</sub> and trains a rank-1 correction instead of updating it directly.</p>
    <p>This is a toy transformer language model built for transparency, not a large language model and not a reproduction of ChatGPT or any production system.</p>
  </footer>

</div>
"""

out_path = os.path.join(_here, 'notebook.html')
with open(out_path, 'w') as f:
    f.write(html)

print("Wrote", out_path, "(", len(html), "bytes )")
