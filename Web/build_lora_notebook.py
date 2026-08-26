"""
Builds Web/lora-notebook.html: a static, no-Mathematica-required
rendering of TinyLoRAByHand.nb, from values Mathematica already computed
and verified (never hand-retyped here).

Regenerate after any change to the LoRA notebook's numbers:
  1. wolframscript -file ../Mathematica/ExportLoRAValues.wls
  2. python3 build_lora_notebook.py
"""
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_here, 'lora_values.json')) as f:
    D = json.load(f)

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

VOCAB_RAW = D['vocabulary']
VOCAB = [esc(t) for t in VOCAB_RAW]
for k in ['predictedBefore', 'predictedFull', 'predictedStep1', 'predictedStep2']:
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
            if isinstance(v, (int, float)) and v < 0:
                cls += " neg"
            if highlight and idx in highlight:
                cls += " hl"
            cells.append(f'<div class="{cls}">{fnum(v, d)}</div>')
            idx += 1
    return (f'<div class="matbox" style="grid-template-columns:repeat({cols},1fr);">'
            f'<div class="mname">{name}</div>' + "".join(cells) + '</div>')

def vocab_table(rows, headers):
    thead = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for r in rows:
        trs += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table class="datatable"><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>'

def prob_bars(probs, target_idx_0based):
    rows = ""
    for i, tok in enumerate(VOCAB):
        p = probs[i]
        is_target = (i == target_idx_0based)
        cls = "barrow" + (" is-target" if is_target else "")
        rows += (f'<div class="{cls}"><span class="bname">{tok}</span>'
                  f'<span class="btrack"><span class="bfill" style="width:{p*100:.4f}%"></span></span>'
                  f'<span class="bpct">{pct(p)}</span></div>')
    return f'<div class="barlist">{rows}</div>'

SHOE_IDX0 = D['targetIndex'] - 1

html = f"""<meta charset="utf-8">
<title>Tiny LoRA Notebook</title>
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

  .compare{{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line);
    border-radius:8px; overflow:hidden; margin:16px 0; }}
  .compare > div{{ background:var(--panel); padding:18px; }}
  .compare .cap{{ font-family:"IBM Plex Mono",monospace; font-size:0.68rem; letter-spacing:0.1em; color:var(--text-faint);
    text-transform:uppercase; margin-bottom:10px; }}
  .compare .after .cap{{ color:var(--good); }}
  .compare .wrong .cap{{ color:var(--bad); }}
  .metric{{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--line-soft); }}
  .metric:last-child{{ border-bottom:none; }}
  .metric .k{{ color:var(--text-dim); font-size:0.85rem; }}
  .metric .v{{ font-family:"IBM Plex Mono",monospace; }}
  .compare .after .v{{ color:var(--good); }}
  .compare .wrong .v.pred{{ color:var(--bad); }}

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
      <div class="kicker">Tiny LoRA &mdash; By Hand</div>
      <h1>Freeze the Model. Train a Correction.</h1>
    </div>
    <span class="badge">static export &middot; no Mathematica needed</span>
  </header>

  <p class="toplinks">
    This page is a static, read-only rendering of <code>Mathematica/TinyLoRAByHand.nb</code>, a companion to
    <a href="./notebook.html">the main Tiny Transformer notebook</a> and its
    <a href="./index.html">live browser demo</a> &mdash; read those first if you haven't. Prefer paper and a
    calculator? See
    <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/calculations/LORA_HAND_CALCULATION.md">the LoRA hand-calculation worksheet</a>.
  </p>

  <section>
    <p>The main notebook trains this model by updating every one of <b>W_Out</b>'s 12 numbers directly &mdash; &ldquo;full fine-tuning.&rdquo; This page freezes W_Out completely and instead trains a small add-on: a rank-1 correction <b>&Delta;W = B &middot; A</b>, where B is 2&times;1 and A is 1&times;6, together 8 numbers. This is LoRA (Low-Rank Adaptation), a real technique used to adapt large language models cheaply &mdash; small enough here to compute by hand.</p>
    <p>Everything about the frozen part of the model is identical to the main notebook: same vocabulary, same embeddings, same W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, same starting W<sub>Out</sub>.</p>
  </section>

  <section>
    <div class="sec-label">Section 1</div>
    <h2 class="sec-title">Recap: The Frozen Model</h2>
    <p>W<sub>Out</sub> (2&times;6, never updated in this page) and h (the attention output at the final position, computed once from the frozen embeddings and W<sub>Q</sub>/W<sub>K</sub>/W<sub>V</sub>, never touched by anything below):</p>
    <div class="flow">{matbox("W_Out (frozen)", D['WOut'], 0)}</div>
    <div class="flow">{matbox("h", D['h'], highlight=[0,1])}</div>
    <p>Because h never depends on W<sub>Out</sub> or its LoRA correction, it is exactly the same number in every section below &mdash; only the final projection step changes.</p>
  </section>

  <section>
    <div class="sec-label">Section 2</div>
    <h2 class="sec-title">Full Fine-Tuning, Recapped</h2>
    <p>The one training step from the main notebook &mdash; the baseline LoRA is compared against:</p>
    <div class="compare">
      <div class="before"><div class="cap">Before training</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeBefore'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossBefore'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedBefore']}</span></div>
      </div>
      <div class="after"><div class="cap">Full fine-tune, 1 step</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeFull'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossFull'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedFull']}</span></div>
      </div>
    </div>
    <p>Every one of W<sub>Out</sub>'s 12 numbers moved to produce that result &mdash; the gradient G below has no zero entries:</p>
    <div class="flow">{matbox("G = dL/dW_Out", D['G0'])}</div>
  </section>

  <section>
    <div class="sec-label">Section 3</div>
    <h2 class="sec-title">The LoRA Idea</h2>
    <p>Instead of replacing W<sub>Out</sub>, LoRA leaves it untouched forever and adds a correction: <b>W<sub>Out,effective</sub> = W<sub>Out</sub> + &Delta;W</b>, where &Delta;W = B &middot; A. With r = 1 (the smallest possible rank), B is 2&times;1 and A is 1&times;6 &mdash; 8 numbers total, versus W<sub>Out</sub>'s 12.</p>
    <div class="note">In a real language model, W<sub>Out</sub>-sized matrices have millions of entries, so r&middot;(m+n) can be thousands of times smaller than m&middot;n &mdash; that's where LoRA's real saving comes from. Here, with m=2 and n=6, the saving is modest (a third fewer numbers). Rank 1 is kept anyway because it's the only rank small enough to multiply out (an outer product) by hand.</div>
  </section>

  <section>
    <div class="sec-label">Section 4</div>
    <h2 class="sec-title">Setting Up B and A</h2>
    <p>Standard LoRA practice: initialize B at zero (so &Delta;W starts as a complete no-op) and A at some fixed nonzero pattern &mdash; ordinarily random; this project has no randomness anywhere, so A0 is a fixed, deterministic pattern instead.</p>
    <div class="flow">{matbox("B0", D['B0'])}<span class="op">&middot;</span>{matbox("A0", D['A0'], 0)}<span class="op">=</span>{matbox("&Delta;W0", [[0]*6,[0]*6])}</div>
    <p>&Delta;W0 is exactly the zero matrix, so the model behaves identically to the frozen base: same P(&ldquo;shoe&rdquo;) &asymp; {pct(D['pShoeBefore'])}, same loss &asymp; {fnum(D['lossBefore'])} as Section 2's starting point.</p>
  </section>

  <section>
    <div class="sec-label">Section 5</div>
    <h2 class="sec-title">Step 1: The Gradient Through B and A</h2>
    <p>Because &Delta;W = B&middot;A enters the logits the same way W<sub>Out</sub> does, the gradient with respect to &Delta;W is the same G already computed in Section 2. The new part is propagating G into separate gradients for B and A: <b>dL/dB = G &middot; Transpose[A]</b> and <b>dL/dA = Transpose[B] &middot; G</b>.</p>
    <div class="flow">{matbox("dA0 = Transpose[B0].G", D['dA0'])}<span class="op">&nbsp;</span>{matbox("dB0 = G.Transpose[A0]", D['dB0'])}</div>
    <div class="note"><b>dA0 is exactly zero.</b> Not a coincidence: dA = Transpose[B]&middot;G, and B0 is zero, so no matter what G is, dA0 must be zero too. A's only effect on the model is through B&middot;A &mdash; if B is zero, scaling A up or down doesn't change B&middot;A at all, so the loss can't &ldquo;feel&rdquo; A yet. dB0, by contrast, doesn't depend on B at all, so it's free to be nonzero.</div>
  </section>

  <section>
    <div class="sec-label">Section 6</div>
    <h2 class="sec-title">Step 1: Updating B and A</h2>
    <div class="flow">{matbox("B1 = B0 - lr&middot;dB0", D['B1'])}<span class="op">&nbsp;</span>{matbox("A1 = A0 - lr&middot;dA0", D['A1'], 0)}</div>
    <p>B1 is a new pair of numbers; A1 is bit-for-bit identical to A0. Of the 8 numbers making up B and A, exactly <b>2 moved</b> this step.</p>
  </section>

  <section>
    <div class="sec-label">Section 7</div>
    <h2 class="sec-title">Step 1: Does It Work?</h2>
    <div class="compare">
      <div class="after"><div class="cap">Full fine-tune, 1 step</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeFull'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossFull'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono">{D['predictedFull']} &mdash; correct</span></div>
      </div>
      <div class="wrong"><div class="cap">LoRA, step 1 (only B moved)</div>
        <div class="metric"><span class="k">P(&ldquo;shoe&rdquo;)</span><span class="v mono">{fnum(D['pShoeStep1'])}</span></div>
        <div class="metric"><span class="k">Loss</span><span class="v mono">{fnum(D['lossStep1'])}</span></div>
        <div class="metric"><span class="k">Predicted</span><span class="v mono pred">{D['predictedStep1']} &mdash; still wrong</span></div>
      </div>
    </div>
    <p>Real progress &mdash; P(&ldquo;shoe&rdquo;) rose and the loss fell &mdash; from moving only 2 of W<sub>Out</sub>'s effective 8 numbers. But the top prediction hasn't flipped yet: full fine-tuning, with its full 12-number capacity, already got there.</p>
  </section>

  <section>
    <div class="sec-label">Section 8</div>
    <h2 class="sec-title">Bonus: Give LoRA One More Step</h2>
    <p>Now that B1 is nonzero, A should no longer be stuck &mdash; dA = Transpose[B]&middot;G depends on B, which is no longer zero. Recomputing the gradient at the new effective weights:</p>
    <div class="flow">{matbox("dA1 (no longer zero)", D['dA1'])}</div>
    <p>As soon as B carries any signal, A starts receiving a real gradient too &mdash; a genuine property of LoRA training in practice, not a simplification specific to this toy. One more gradient step on both factors:</p>
    <table class="datatable">
      <thead><tr><th>Stage</th><th>P(&ldquo;shoe&rdquo;)</th><th>Loss</th><th>Top guess</th><th>Numbers moved</th></tr></thead>
      <tbody>
        <tr><td>Before training</td><td>{pct(D['pShoeBefore'])}</td><td>{fnum(D['lossBefore'])}</td><td>{D['predictedBefore']}</td><td>&mdash;</td></tr>
        <tr><td>Full fine-tune, 1 step</td><td>{pct(D['pShoeFull'])}</td><td>{fnum(D['lossFull'])}</td><td>{D['predictedFull']}</td><td>12</td></tr>
        <tr><td>LoRA, step 1</td><td>{pct(D['pShoeStep1'])}</td><td>{fnum(D['lossStep1'])}</td><td>{D['predictedStep1']}</td><td>2</td></tr>
        <tr class="hl"><td>LoRA, step 2</td><td>{pct(D['pShoeStep2'])}</td><td>{fnum(D['lossStep2'])}</td><td>{D['predictedStep2']}</td><td>up to 8</td></tr>
      </tbody>
    </table>
    <p>By its own second step &mdash; never touching more than 8 of W<sub>Out</sub>'s effective parameters &mdash; LoRA doesn't just catch up to one step of full fine-tuning. It overtakes it on every metric measured. This isn't a general law (it depends on the learning rate, A's initialization, and this loss surface being simple at this scale) but it's a real, checked result for this exact example.</p>
    {prob_bars(D['pStep2'], SHOE_IDX0)}
  </section>

  <section>
    <div class="sec-label">Section 9</div>
    <h2 class="sec-title">What This Does and Doesn't Prove</h2>
    <ul style="color:var(--text-dim);">
      <li>The parameter saving (8 vs. 12) isn't the point at this scale. LoRA's real value shows up when W has millions or billions of parameters.</li>
      <li>Rank 1 was chosen only because it's the smallest rank that multiplies out by hand &mdash; real deployments commonly use ranks from 4 to 64.</li>
      <li>A0 was fixed by hand rather than drawn at random, to keep this project free of randomness &mdash; real implementations initialize A randomly (with B still fixed at zero); the &ldquo;A can't move until B does&rdquo; behavior above is real, not an artifact of this specific A0.</li>
      <li>Only W<sub>Out</sub> gets a LoRA correction here, matching the layer the main notebook already trains. In practice LoRA more often targets the attention projections (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>).</li>
      <li>Two steps were used deliberately, to expose the &ldquo;only B moves at step 1&rdquo; fact &mdash; not a claim about how many steps LoRA needs in general.</li>
    </ul>
  </section>

  <section>
    <div class="panel cta-box">
      <div>
        <h3 class="sub-title" style="margin-top:0;">Want the full worked notebook?</h3>
        <p>This static page mirrors <code>TinyLoRAByHand.nb</code> but can't run its cells. Open the notebook in Mathematica or Wolfram Player to re-evaluate every step yourself.</p>
      </div>
      <a class="cta" href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/Mathematica/TinyLoRAByHand.nb">View the notebook source</a>
    </div>
  </section>

  <section>
    <div class="sec-label">Verification</div>
    <h2 class="sec-title">Every Claim Above, Checked Programmatically</h2>
    <p>Before any number here was trusted, it was checked by <code>TinyLoRAByHand.wl</code>'s automated suite &mdash; 21 checks in total, all passing:</p>
    <ul class="checklist">
      <li>B0 is 2&times;1, A0 is 1&times;6, &Delta;W0 is 2&times;6</li>
      <li>&Delta;W0 is exactly the zero matrix</li>
      <li>LoRA's step-1 forward pass matches the shared baseline</li>
      <li>dA is exactly zero at step 1; dB is not</li>
      <li>A is unchanged after step 1; B has changed</li>
      <li>Full fine-tune's P(shoe) rises and loss falls</li>
      <li>LoRA step 1's P(shoe) rises and loss falls, despite moving only B</li>
      <li>Full fine-tune beats LoRA step 1 on loss</li>
      <li>Full fine-tune's step flips the top prediction to &ldquo;shoe&rdquo;</li>
      <li>LoRA step 1's top prediction has NOT yet flipped</li>
      <li>dA is nonzero at step 2</li>
      <li>LoRA step 2's loss and P(shoe) keep improving</li>
      <li>LoRA step 2 overtakes full fine-tuning's loss</li>
      <li>LoRA step 2's top prediction has now flipped to &ldquo;shoe&rdquo;</li>
    </ul>
  </section>

  <footer>
    <p>Full project: <a href="https://github.com/djimrastephane/tiny-transformer-by-hand">github.com/djimrastephane/tiny-transformer-by-hand</a>
    &middot; <a href="./notebook.html">Main notebook</a>
    &middot; <a href="./index.html">Live demo</a>
    &middot; <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/Mathematica/TinyLoRAByHand.nb">LoRA notebook source</a>
    &middot; <a href="https://github.com/djimrastephane/tiny-transformer-by-hand/blob/main/methodology/ASSUMPTIONS.md">Methodology</a></p>
    <p>This is a toy transformer language model built for transparency, not a large language model and not a reproduction of ChatGPT or any production system.</p>
  </footer>

</div>
"""

out_path = os.path.join(_here, 'lora-notebook.html')
with open(out_path, 'w') as f:
    f.write(html)

print("Wrote", out_path, "(", len(html), "bytes )")
