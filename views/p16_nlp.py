"""Chapter 16 — Natural Language Processing with RNNs and Attention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, datasets as ds, nav
from core.lecture import (anim_header, codenote, derive, exercise, figure, hero,
                          idea, keypoints, lead, math, md, note, pitfall, proof,
                          quiz, refs, rule, section, sub, table, tip, warn,
                          where)
from core.palette import C, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "ch16"

hero(
    kicker="Part II · Chapter 16",
    title="NLP with RNNs and Attention",
    blurb=(
        "From a character RNN that spells its way through text, to the "
        "encoder–decoder that translates, to attention — which removed the "
        "fixed-size bottleneck — and finally to the Transformer, which removed "
        "the recurrence itself. Every mechanism is derived, implemented from "
        "scratch, and checked against Keras."
    ),
    chips=["Attention derived", "9 sub-sections", "9 animations",
           "9 code labs", "Transformer from scratch"],
)
nav.sidebar_tools(CH)


def _tf_ok() -> bool:
    # find_spec, not import: importing TensorFlow costs ~500 MB of RSS and we
    # only need to know whether the labs will be able to run.
    import importlib.util
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


if not _tf_ok():
    st.warning("TensorFlow is not importable here, so the labs will report an "
               "ImportError. Every explanation and animation still works.",
               icon="⚠️")


# ==========================================================================
def s_16_1():
    section("16.1", "Character RNNs — Generating Text One Letter at a Time")

    lead(
        "The simplest useful language model: given the last $n$ characters, "
        "predict the next one. Train it on a corpus, then sample from it "
        "repeatedly and it writes."
    )

    sub("The task")

    md(
        "A char-RNN is a **classifier with as many classes as there are distinct "
        "characters**. At every position it outputs a distribution over the "
        "alphabet, and the loss is ordinary cross-entropy."
    )

    math(r"""
    p_\theta\bigl(c_{t} \mid c_{1}, \dots, c_{t-1}\bigr)
    \;=\; \mathrm{softmax}\bigl(\mathbf{W}_o \mathbf{h}_{(t-1)} + \mathbf{b}_o\bigr)
    """)

    md("The probability of a whole text factorises by the chain rule:")

    math(r"""
    p_\theta(c_1, \dots, c_T) \;=\; \prod_{t=1}^{T}
      p_\theta\bigl(c_t \mid c_{<t}\bigr)
    \qquad\Longrightarrow\qquad
    \mathcal{L} \;=\; -\frac{1}{T}\sum_{t=1}^{T}
      \log p_\theta\bigl(c_t \mid c_{<t}\bigr)
    """)

    idea(
        "That factorisation is the whole of autoregressive language modelling",
        "Nothing in it is specific to characters, to RNNs, or to text. Swap "
        "characters for subword tokens and you get GPT; swap them for image "
        "patches and you get an autoregressive image model; swap them for audio "
        "samples and you get WaveNet (§15.9). The <b>architecture</b> that "
        "computes $p_\\theta(c_t \\mid c_{<t})$ changes across the next four "
        "chapters. <b>The objective never does.</b>",
    )

    sub("Perplexity")

    md(
        "Language models are conventionally reported not as cross-entropy but as "
        "its exponential, the **perplexity**:"
    )

    math(r"""
    \mathrm{PPL} \;=\; \exp\Bigl(-\frac{1}{T}\sum_{t=1}^{T}
      \log p_\theta(c_t \mid c_{<t})\Bigr) \;=\; e^{\mathcal{L}}
    """)

    proof(
        "Perplexity is the effective number of choices the model is deciding "
        "between",
        "If the model were uniform over $V$ characters, the loss would be "
        "$\\log V$ and the perplexity exactly $V$. A perplexity of 4 on a "
        "40-character alphabet means the model has narrowed each position down to "
        "the equivalent of a fair 4-way choice. It is a much more interpretable "
        "number than a cross-entropy of 1.386, and it is directly comparable "
        "across corpora <b>only if the tokenisation is identical</b> — comparing "
        "a character-level perplexity to a word-level one is meaningless.",
    )

    sub("Splitting a sequential dataset")

    pitfall(
        "You cannot shuffle text before splitting either",
        "The same argument as §15.4. If you take random windows and split them "
        "randomly, a window in the validation set overlaps windows in the "
        "training set, and the model has effectively memorised the answer. "
        "<b>Split by position in the corpus</b>: first 90 % train, next 5 % "
        "validation, last 5 % test. Even that is imperfect for text — an author's "
        "style drifts and topics recur — but it is the honest baseline.",
    )

    sub("Sampling — the part everyone gets wrong")

    md(
        "Once trained, you generate by feeding the model a seed, taking its "
        "output distribution, picking a character, appending it, and repeating. "
        "**How you pick matters enormously.**"
    )

    table(
        ["Strategy", "Rule", "Result"],
        [["<b>Greedy / argmax</b>", "Always take the most likely character",
          "Degenerate: falls into short repeating loops almost immediately"],
         ["<b>Pure sampling</b>", "Sample from $p$ exactly",
          "Coherent locally, but the tail of the distribution injects nonsense"],
         ["<b>Temperature</b>",
          "$p_i \\propto \\exp(\\log p_i / T)$, then sample",
          "$T \\to 0$ ⇒ greedy; $T = 1$ ⇒ pure; $T > 1$ ⇒ more random"],
         ["<b>Top-$k$</b>", "Keep the $k$ most likely, renormalise, sample",
          "Cuts the unreliable tail without flattening the head"],
         ["<b>Nucleus (top-$p$)</b>",
          "Keep the smallest set with cumulative mass $\\ge p$",
          "Adapts $k$ to how peaked the distribution is — the modern default"]],
    )

    derive(
        [("<b>Temperature is a rescaling of the logits before the softmax.</b> "
          "Start from the softmax over logits $z_i$:",
          r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}"),
         ("Divide every logit by $T$ before exponentiating:",
          r"p_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}"),
         ("As $T \\to 0^+$, the largest logit dominates completely: the "
          "distribution becomes a point mass at $\\arg\\max_i z_i$ — greedy "
          "decoding.", None),
         ("As $T \\to \\infty$, all $z_i / T \\to 0$ and the distribution becomes "
          "uniform — pure noise.", None),
         ("At $T = 1$ it is unchanged. So temperature is a single knob "
          "interpolating between the two failure modes, and the interesting range "
          "is roughly $[0.5, 1.2]$.", None),
         ("Equivalently, in the log domain, temperature is a <b>power</b> on the "
          "probabilities followed by renormalisation:",
          r"p_i^{(T)} \;\propto\; p_i^{\,1/T}")],
        title="What temperature does to the distribution",
    )

    warn(
        "Never use argmax to generate text",
        "It is deterministic, so the same seed always yields the same output — "
        "and because natural language has many near-ties, argmax gets caught in a "
        "loop: <i>the the the the</i>. This is the single most common mistake in "
        "a first text-generation attempt, and the fix is one line: sample instead "
        "of taking the max.",
    )

    anim_header("Temperature reshaping a probability distribution")

    labels_c = list("aeioutnshrdl ")
    base_logits = np.array([3.1, 2.4, 1.9, 1.6, 1.5, 2.8, 2.2, 1.7, 1.4,
                            1.2, 0.9, 0.6, 3.4])
    temps = np.concatenate([np.linspace(0.05, 1.0, 22),
                            np.linspace(1.0, 2.5, 14)])

    frames = []
    for T_ in temps:
        p = np.exp(base_logits / T_)
        p = p / p.sum()
        ent = -np.sum(p * np.log(p + 1e-12))
        ppl = float(np.exp(ent))
        frames.append(go.Frame(name=f"{T_:.2f}", data=[
            go.Bar(x=labels_c, y=p,
                   marker=dict(color=p, colorscale=nav.cscale(), cmin=0, cmax=1))
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"T = {T_:.2f}   ·   entropy = {ent:.3f} nats   ·   "
            f"perplexity = {ppl:.2f} effective choices   ·   "
            + ("→ GREEDY" if T_ < 0.35 else
               "→ pure sampling" if abs(T_ - 1) < .06 else
               "→ near-uniform noise" if T_ > 2.0 else "usable range"),
            color=(C["danger"] if T_ < 0.35 or T_ > 2.0 else C["success"]))])))

    p0 = np.exp(base_logits / temps[0]); p0 /= p0.sum()
    f = go.Figure(data=[go.Bar(x=labels_c, y=p0,
                              marker=dict(color=p0, colorscale=nav.cscale(),
                                          cmin=0, cmax=1))])
    f.update_layout(height=420, yaxis=dict(range=[0, 1.02],
                                           title="probability"),
                    xaxis_title="next character",
                    title="p ∝ exp(logit / T) — one knob between greedy and noise")
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="T = ")
    figure(f, "At T = 0.05 the model has no choice at all. At T = 2.5 it is "
              "barely better than random. Good text lives in between.")

    code_lab(
        "A character RNN, trained and sampled four different ways",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE CORPUS AND ITS ALPHABET ===========================
text = _ds.char_corpus(50_000)
vocab = sorted(set(text))
V = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}
itos = np.array(vocab)
data = np.array([stoi[c] for c in text], dtype="int32")

print("=== the corpus ===")
print(f"  {len(text):,} characters, {V} distinct")
print(f"  alphabet: {repr(''.join(vocab))}")
print(f"  first 120 chars:\\n    {repr(text[:120])}")
print(f"  uniform-guess perplexity would be {V}")

# ============ 2. CHRONOLOGICAL SPLIT ===================================
n_tr = int(.90 * len(data)); n_va = int(.95 * len(data))
tr, va, te = data[:n_tr], data[n_tr:n_va], data[n_va:]
print()
print("=== split BY POSITION, never shuffled ===")
print(f"  train {len(tr):,}   valid {len(va):,}   test {len(te):,}")

# ============ 3. WINDOWING =============================================
L = 60                                   # context length
def make_ds(arr, batch=64, shuffle=False):
    d = tf.data.Dataset.from_tensor_slices(arr)
    d = d.window(L + 1, shift=1, drop_remainder=True)
    d = d.flat_map(lambda w: w.batch(L + 1))
    if shuffle:
        d = d.shuffle(10_000, seed=42)
    d = d.batch(batch).map(lambda w: (w[:, :-1], w[:, 1:]))   # predict NEXT
    return d.prefetch(1)

train_ds = make_ds(tr, shuffle=True)
valid_ds = make_ds(va)
xb, yb = next(iter(train_ds))
print()
print(f"=== windows of {L} ===")
print(f"  X {tuple(xb.shape)}  Y {tuple(yb.shape)}")
print(f"  X[0][:20] = {repr(''.join(itos[xb[0, :20].numpy()]))}")
print(f"  Y[0][:20] = {repr(''.join(itos[yb[0, :20].numpy()]))}   <- shifted by 1")
print("  the target is the input shifted one step: a prediction at EVERY position")

# ============ 4. THE MODEL =============================================
EMB = 24
model = keras.Sequential([
    keras.layers.Input(shape=(None,), dtype="int32"),
    keras.layers.Embedding(V, EMB),
    keras.layers.GRU(160, return_sequences=True),      # seq2seq (ch. 15.6)
    keras.layers.Dense(V),                             # LOGITS, not softmax
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3))
print()
print("=== the model ===")
model.summary(print_fn=lambda s: print("  " + s))
print(f"  output is LOGITS -- from_logits=True is numerically safer than")
print(f"  a softmax layer followed by log()")

t0 = time.perf_counter()
hist = model.fit(train_ds, epochs=4, validation_data=valid_ds, verbose=0)
print(f"\\n  trained in {time.perf_counter()-t0:.1f}s")

# ============ 5. PERPLEXITY ============================================
print()
print("=== cross-entropy and perplexity ===")
print(f"{'epoch':>7}{'train loss':>13}{'train PPL':>12}"
      f"{'valid loss':>13}{'valid PPL':>12}")
for e, (l, vl) in enumerate(zip(hist.history["loss"],
                                hist.history["val_loss"]), 1):
    print(f"{e:>7}{l:>13.4f}{np.exp(l):>12.3f}{vl:>13.4f}{np.exp(vl):>12.3f}")
print(f"  a uniform model would score loss {np.log(V):.4f}, PPL {V}")
final_ppl = np.exp(hist.history['val_loss'][-1])
print(f"  ours is down to {final_ppl:.2f} effective choices per character")

# ============ 6. FOUR WAYS TO SAMPLE ===================================
def next_probs(seed_ids):
    logits = model.predict(seed_ids[None, :], verbose=0)[0, -1]
    return logits

def generate(seed, n=180, mode="temperature", T=0.8, k=5, p_nuc=0.9, seed_rng=0):
    rng = np.random.default_rng(seed_rng)
    ids = [stoi.get(c, 0) for c in seed]
    for _ in range(n):
        logits = next_probs(np.array(ids[-L:], dtype="int32"))
        if mode == "greedy":
            nxt = int(np.argmax(logits))
        elif mode == "pure":
            pr = np.exp(logits - logits.max()); pr /= pr.sum()
            nxt = int(rng.choice(V, p=pr))
        elif mode == "temperature":
            z = logits / T
            pr = np.exp(z - z.max()); pr /= pr.sum()
            nxt = int(rng.choice(V, p=pr))
        elif mode == "topk":
            idx = np.argsort(-logits)[:k]
            z = logits[idx] / T
            pr = np.exp(z - z.max()); pr /= pr.sum()
            nxt = int(idx[rng.choice(len(idx), p=pr)])
        elif mode == "nucleus":
            z = logits / T
            pr = np.exp(z - z.max()); pr /= pr.sum()
            order = np.argsort(-pr)
            keep = order[:max(1, int(np.searchsorted(np.cumsum(pr[order]),
                                                     p_nuc)) + 1)]
            q = pr[keep] / pr[keep].sum()
            nxt = int(keep[rng.choice(len(keep), p=q)])
        ids.append(nxt)
    return "".join(itos[ids])

SEED = "the king speaks "
print()
print("="*66)
print("The same model, five decoding strategies")
print("="*66)
for label, kw in [("GREEDY (argmax)",       dict(mode="greedy")),
                  ("temperature T=0.3",     dict(mode="temperature", T=0.3)),
                  ("temperature T=0.8",     dict(mode="temperature", T=0.8)),
                  ("pure sampling (T=1)",   dict(mode="pure")),
                  ("top-k, k=5, T=0.8",     dict(mode="topk", k=5, T=0.8)),
                  ("nucleus p=0.9, T=0.9",  dict(mode="nucleus", p_nuc=.9, T=.9))]:
    out = generate(SEED, n=150, **kw)
    print(f"\\n--- {label} ---")
    print("  " + out.replace("\\n", " / "))

print()
print("  GREEDY loops. Pure sampling drifts. Temperature 0.7-0.9 and")
print("  top-k / nucleus sit in the usable band.")

# ============ 7. TEMPERATURE, QUANTIFIED ===============================
print()
print("=== entropy of the model's own distribution vs temperature ===")
ctx = np.array([stoi[c] for c in text[1000:1000+L]], dtype="int32")
logits = next_probs(ctx)
print(f"{'T':>7}{'entropy (nats)':>17}{'effective choices':>20}"
      f"{'top-1 prob':>13}")
for T_ in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.5]:
    z = logits / T_
    pr = np.exp(z - z.max()); pr /= pr.sum()
    ent = -np.sum(pr*np.log(pr+1e-12))
    print(f"{T_:>7.1f}{ent:>17.4f}{np.exp(ent):>20.2f}{pr.max():>13.4f}")

import plotly.graph_objects as go
Ts = np.linspace(.1, 2.5, 60)
ents = []
for T_ in Ts:
    z = logits/T_; pr = np.exp(z-z.max()); pr /= pr.sum()
    ents.append(np.exp(-np.sum(pr*np.log(pr+1e-12))))
fig = go.Figure(go.Scatter(x=Ts, y=ents, mode="lines",
                           line=dict(color=C["primary"], width=3)))
fig.add_hline(y=V, line_dash="dot", line_color=C["danger"],
              annotation_text=f"uniform = {V}")
fig.add_hline(y=1, line_dash="dot", line_color=C["success"],
              annotation_text="greedy = 1")
fig.update_layout(height=380, xaxis_title="temperature T",
                  yaxis_title="effective number of choices",
                  title="Temperature interpolates between greedy and uniform")
''',
        key="ch16_charrnn",
    )

    keypoints([
        "A char-RNN is a classifier over the alphabet; the loss is "
        "$-\\frac{1}{T}\\sum \\log p_\\theta(c_t \\mid c_{<t})$.",
        "<b>Perplexity</b> $= e^{\\mathcal{L}}$ is the effective number of "
        "choices per position.",
        "Split the corpus <b>by position</b>, never randomly.",
        "<b>Never generate with argmax</b> — it loops. Sample with temperature, "
        "top-$k$, or nucleus.",
        "Temperature is $p_i \\propto p_i^{1/T}$: one knob between greedy and "
        "uniform.",
    ])


# ==========================================================================
def s_16_2():
    section("16.2", "Stateful RNNs and Long Contexts")

    lead(
        "A stateless RNN starts every batch from a zero state, so it can never "
        "learn a pattern longer than one window. A stateful RNN carries the state "
        "across batches — at the cost of four strict requirements."
    )

    sub("Stateless vs stateful")

    table(
        ["", "Stateless (default)", "Stateful"],
        [["Initial state each batch", "Zeros",
          "The final state of the <b>previous</b> batch"],
         ["Max learnable dependency", "The window length $L$",
          "The whole training sequence"],
         ["Batches may be shuffled", "✅ yes",
          "❌ <b>never</b> — order is the whole point"],
         ["Windows must be", "Anything",
          "<b>Consecutive and non-overlapping</b>"],
         ["Batch size", "Anything",
          "<b>Fixed</b>, and declared in the input shape"],
         ["End of an epoch", "Nothing to do",
          "<b>Must call <code>reset_states()</code></b>"]],
    )

    pitfall(
        "The four requirements are not optional, and violating them fails "
        "silently",
        "1. <code>stateful=True</code> on the layer.<br>"
        "2. <code>batch_input_shape</code> (or a batched <code>Input</code>) with "
        "a <b>fixed</b> batch size.<br>"
        "3. Windows fed in <b>consecutive, non-overlapping</b> order — "
        "<code>shift=L</code>, and <code>shuffle=False</code>.<br>"
        "4. <code>reset_states()</code> at the end of every epoch, via a "
        "callback.<br><br>"
        "Miss any one and the model still trains and still reports a loss — it is "
        "simply learning from a state that carries nonsense. There is no error "
        "message.",
    )

    sub("The batch-interleaving trick")

    md(
        "Requirement 3 seems to force a batch size of 1, which is desperately "
        "slow. The standard fix: **split the corpus into `batch_size` "
        "independent contiguous streams**, and let row $i$ of every batch come "
        "from stream $i$."
    )

    derive(
        [("Let the corpus have $N$ characters and let the batch size be $B$. "
          "Split it into $B$ contiguous chunks of length $N/B$:",
          r"S_i = \bigl[c_{\,iN/B},\; \dots,\; c_{\,(i+1)N/B - 1}\bigr],"
          r"\qquad i = 0, \dots, B-1"),
         ("Batch $j$ then consists of window $j$ from each stream:",
          r"\mathbf{X}^{(j)}[i] = S_i\bigl[\,jL : (j+1)L\,\bigr]"),
         ("Row $i$ of batch $j+1$ continues exactly where row $i$ of batch $j$ "
          "stopped, so carrying the state across batches is correct <b>for every "
          "row simultaneously</b>. The $B$ streams are independent, which is fine "
          "— each row of the state simply tracks its own stream.", None),
         ("The only cost is $B$ discontinuities in the corpus (one at each chunk "
          "boundary), which is negligible for $B \\ll N$.", None)],
        title="How to be stateful and still use a large batch",
    )

    note(
        "In practice, stateful RNNs are rarely worth it now",
        "They are fiddly, they force a fixed batch size that complicates "
        "inference, and their advantage — context longer than one window — is "
        "obtained more cleanly by simply using a longer window, a dilated "
        "convolution (§15.9), or attention (§16.6). Understand them because they "
        "clarify what an RNN's state <i>is</i>; reach for them last.",
    )

    anim_header("Stateless vs stateful: what the hidden state remembers")

    T_total = 48
    L_win = 8
    frames = []
    for k in range(1, T_total + 1):
        cur_win = (k - 1) // L_win
        # stateless: resets each window
        sless = np.array([0.9 ** ((t % L_win) + 1) for t in range(k)])
        sful = np.array([0.9 ** (t + 1) for t in range(k)])
        bars = []
        for w in range(cur_win + 1):
            bars.append(go.Scatter(
                x=[w * L_win - .5, w * L_win - .5], y=[0, 1.05], mode="lines",
                line=dict(color=C["line"], width=1.5, dash="dot"),
                showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(k), data=bars + [
            go.Scatter(x=np.arange(k), y=sless, mode="lines+markers",
                       line=dict(color=C["danger"], width=3),
                       marker=dict(size=5)),
            go.Scatter(x=np.arange(k), y=sful, mode="lines+markers",
                       line=dict(color=C["success"], width=3),
                       marker=dict(size=5)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {k}, window {cur_win + 1}   ·   stateless retains "
            f"{sless[-1]:.3f} of the signal from ITS window start   ·   "
            f"stateful retains {sful[-1]:.4f} from the CORPUS start")])))

    f = go.Figure(data=[
        go.Scatter(x=[0], y=[.9], mode="lines+markers",
                   name="stateless — state zeroed each window",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[0], y=[.9], mode="lines+markers",
                   name="stateful — state carried across windows",
                   line=dict(color=C["success"], width=3)),
    ])
    f.update_layout(height=400, xaxis_title="position in the corpus",
                    yaxis_title="signal retained from the start",
                    yaxis=dict(range=[0, 1.08]),
                    title=f"Window length {L_win}",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="step ")
    figure(f, "The stateless model's memory is sawtoothed — it cannot see past "
              "the start of its own window, ever.")

    code_lab(
        "A stateful RNN done correctly, and the four ways to get it wrong",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

text = _ds.char_corpus(30_000)
vocab = sorted(set(text)); V = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}
data = np.array([stoi[c] for c in text], dtype="int32")
L, B = 40, 16
n_tr = int(.9*len(data))
tr, va = data[:n_tr], data[n_tr:]
print(f"=== corpus {len(data):,} chars, vocab {V}, window {L}, batch {B} ===")

# ============ 1. THE BATCH-INTERLEAVING TRICK ==========================
def stateful_dataset(arr, L, B):
    """Split into B contiguous streams; row i of every batch comes from
    stream i, so state carries correctly for every row simultaneously."""
    per = len(arr) // B
    streams = [arr[i*per:(i+1)*per] for i in range(B)]
    n_win = (per - 1) // L
    Xs, Ys = [], []
    for j in range(n_win):
        Xs.append(np.stack([s[j*L:(j+1)*L] for s in streams]))
        Ys.append(np.stack([s[j*L+1:(j+1)*L+1] for s in streams]))
    return (np.stack(Xs), np.stack(Ys))       # (n_batches, B, L)

Xb, Yb = stateful_dataset(tr, L, B)
print()
print("=== the interleaving ===")
print(f"  {Xb.shape[0]} batches of shape ({B}, {L})")
itos = np.array(vocab)
print(f"  batch 0 row 0: {repr(''.join(itos[Xb[0,0,:24]]))}")
print(f"  batch 1 row 0: {repr(''.join(itos[Xb[1,0,:24]]))}   <- CONTINUES it")
print(f"  batch 0 row 1: {repr(''.join(itos[Xb[0,1,:24]]))}   <- a DIFFERENT stream")
ok = np.array_equal(Xb[1, 0, 0:1], Yb[0, 0, -1:])
print(f"  batch j+1 row i starts where batch j row i ended: {ok}")

# ============ 2. THE FOUR REQUIREMENTS =================================
print()
print("=== a correctly built stateful model ===")
def build(stateful):
    inp = keras.layers.Input(batch_shape=(B, None) if stateful else (None, None),
                             dtype="int32")
    z = keras.layers.Embedding(V, 24)(inp)
    z = keras.layers.GRU(128, return_sequences=True, stateful=stateful)(z)
    return keras.Model(inp, keras.layers.Dense(V)(z))

stateful_model = build(True)
print(f"  1. stateful=True on the GRU                      OK")
print(f"  2. batch_shape fixed at {B}                       OK "
      f"(input shape {stateful_model.input_shape})")
print(f"  3. windows consecutive, non-overlapping, shuffle=False  OK")

class ResetStates(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if getattr(layer, "stateful", False):
                layer.reset_states()
print(f"  4. reset_states() every epoch, via a callback     OK")

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ============ 3. STATEFUL vs STATELESS =================================
print()
print("=== does carrying the state help? ===")
Xva, Yva = stateful_dataset(va, L, B)
n_use = min(len(Xb), 220)

stateful_model.compile(loss=loss, optimizer=keras.optimizers.Adam(3e-3))
hs = stateful_model.fit(Xb[:n_use].reshape(-1, L), Yb[:n_use].reshape(-1, L),
                        epochs=4, batch_size=B, shuffle=False,   # NEVER shuffle
                        callbacks=[ResetStates()], verbose=0)

tf.random.set_seed(42)
stateless_model = build(False)
stateless_model.compile(loss=loss, optimizer=keras.optimizers.Adam(3e-3))
hl = stateless_model.fit(Xb[:n_use].reshape(-1, L), Yb[:n_use].reshape(-1, L),
                         epochs=4, batch_size=B, shuffle=True, verbose=0)

print(f"{'epoch':>7}{'stateful loss':>16}{'stateless loss':>17}")
for e, (a, b_) in enumerate(zip(hs.history["loss"], hl.history["loss"]), 1):
    print(f"{e:>7}{a:>16.4f}{b_:>17.4f}")
print(f"  final perplexity: stateful {np.exp(hs.history['loss'][-1]):.3f}   "
      f"stateless {np.exp(hl.history['loss'][-1]):.3f}")

# ============ 4. THE STATE ITSELF ======================================
print()
print("=== inspecting the carried state ===")
gru = [l for l in stateful_model.layers if isinstance(l, keras.layers.GRU)][0]
gru.reset_states()
s0 = np.array(gru.states[0])
print(f"  after reset_states(): ||state|| = {np.linalg.norm(s0):.6f}")
_ = stateful_model(tf.constant(Xb[0]))
s1 = np.array(gru.states[0])
print(f"  after batch 0       : ||state|| = {np.linalg.norm(s1):.4f}  "
      f"shape {s1.shape}")
_ = stateful_model(tf.constant(Xb[1]))
s2 = np.array(gru.states[0])
print(f"  after batch 1       : ||state|| = {np.linalg.norm(s2):.4f}")
print(f"  state changed between batches: {not np.allclose(s1, s2)}")
print(f"  the state has one row per batch row -- {B} independent streams")

# ============ 5. THE FAILURE MODES, DEMONSTRATED =======================
print()
print("="*64)
print("What happens when you break each requirement")
print("="*64)

# (a) shuffling
gru.reset_states()
_ = stateful_model(tf.constant(Xb[0]))
good = np.array(gru.states[0]).copy()
gru.reset_states()
_ = stateful_model(tf.constant(Xb[7]))          # a NON-consecutive batch
bad = np.array(gru.states[0])
print(f"\\n(3) shuffled batches:")
print(f"    the state after batch 0 and after batch 7 differ by "
      f"{np.abs(good-bad).mean():.4f} on average")
print(f"    -> with shuffle=True the state fed into each batch is meaningless.")
print(f"       Keras reports a perfectly normal loss. NO ERROR IS RAISED.")

# (b) forgetting reset_states
print(f"\\n(4) no reset_states():")
print(f"    epoch 2 starts from the state left at the END of the corpus,")
print(f"    which has nothing to do with the START of the corpus.")

# (c) wrong batch size at inference
print(f"\\n(2) a different batch size at inference:")
try:
    stateful_model(tf.constant(Xb[0][:4]))       # 4 rows, not B
    print("    accepted (unexpected)")
except Exception as e:
    print(f"    {type(e).__name__}: {str(e).splitlines()[0][:90]}")
print(f"    -> for generation you must rebuild the model with batch size 1")
print(f"       and copy the weights across. That is the real cost of stateful.")

# ============ 6. GENERATING WITH A BATCH-SIZE-1 COPY ===================
print()
print("=== the standard workaround for inference ===")
gen = keras.Sequential([keras.layers.Input(batch_shape=(1, None), dtype="int32"),
                        keras.layers.Embedding(V, 24),
                        keras.layers.GRU(128, return_sequences=True,
                                         stateful=True),
                        keras.layers.Dense(V)])
gen.set_weights(stateful_model.get_weights())
print(f"  rebuilt with batch_shape=(1, None) and copied the weights")

rng = np.random.default_rng(0)
gen.layers[1].reset_states()
ids = [stoi[c] for c in "the king "]
_ = gen(tf.constant([ids[:-1]]))                 # warm the state up
cur = ids[-1]
out = []
for _ in range(140):
    logits = gen(tf.constant([[cur]]))[0, -1].numpy() / .8
    p = np.exp(logits - logits.max()); p /= p.sum()
    cur = int(rng.choice(V, p=p))
    out.append(cur)
print(f"  generated (feeding ONE character at a time, state carried):")
print(f"    {repr('the king ' + ''.join(itos[out]))}")
print(f"  note the model never sees more than one character per call --")
print(f"  all the context lives in the carried state.")
''',
        key="ch16_stateful",
    )

    keypoints([
        "A stateless RNN cannot learn a dependency longer than its window; a "
        "stateful one can.",
        "Four requirements: <code>stateful=True</code>, fixed batch shape, "
        "consecutive non-overlapping windows with <code>shuffle=False</code>, and "
        "<code>reset_states()</code> per epoch.",
        "Violating any of them <b>fails silently</b> — the loss looks normal.",
        "Split the corpus into $B$ contiguous streams so a large batch is still "
        "possible.",
        "For inference you must rebuild with batch size 1 and copy the weights.",
    ])


# ==========================================================================
def s_16_3():
    section("16.3", "Sentiment Analysis, Tokenisation and Masking")

    lead(
        "The first real NLP task: read a review, output a probability. Two "
        "things dominate the result — how you turn text into integers, and "
        "whether you correctly ignore the padding."
    )

    sub("Tokenisation")

    table(
        ["Level", "Vocabulary", "Out-of-vocabulary", "Sequence length"],
        [["<b>Character</b>", "~100", "Never happens", "Very long"],
         ["<b>Word</b>", "10k–1M", "<b>A real problem</b> — every typo is UNK",
          "Short"],
         ["<b>Subword</b> (BPE, WordPiece, SentencePiece)", "8k–100k",
          "<b>Impossible by construction</b> — falls back to characters",
          "Medium"]],
    )

    idea(
        "Subword tokenisation is why modern models have no UNK token",
        "Byte-Pair Encoding starts with individual characters and repeatedly "
        "merges the most frequent adjacent pair, building up a vocabulary of "
        "common word pieces. Frequent words end up as single tokens; rare words "
        "decompose into pieces; a word never seen before still decomposes into "
        "<i>something</i>. So the model can represent any string at all, at the "
        "cost of a longer sequence for unusual text. Every model from GPT-2 "
        "onward uses a variant of this.",
    )

    sub("Padding and masking")

    md(
        "Reviews have different lengths, but a tensor is rectangular, so short "
        "sequences are padded with zeros. **The model must be told to ignore "
        "them.**"
    )

    pitfall(
        "Unmasked padding silently corrupts the result",
        "Feed a 5-word review padded to length 100 into an unmasked RNN and the "
        "final state is the result of 95 steps of processing zeros — the actual "
        "review has been almost entirely forgotten by the time you read the "
        "state. The model still trains, the accuracy is merely bad, and nothing "
        "tells you why. In Keras: pass <code>mask_zero=True</code> to the "
        "<code>Embedding</code> layer, or insert a <code>Masking</code> layer. "
        "The mask then propagates automatically through every layer that supports "
        "it.",
    )

    derive(
        [("<b>How masking actually works.</b> A layer that supports masking "
          "receives a boolean tensor $M \\in \\{0,1\\}^{B \\times T}$ alongside "
          "its input.", None),
         ("An RNN uses it to <b>freeze the state</b> at masked steps rather than "
          "update it:",
          r"\mathbf{h}_{(t)} = m_t\,\phi\bigl(\mathbf{x}_{(t)}, "
          r"\mathbf{h}_{(t-1)}\bigr) + (1 - m_t)\,\mathbf{h}_{(t-1)}"),
         ("So at a padded step, $m_t = 0$ and the state simply passes through "
          "unchanged. The final state is exactly what it would have been without "
          "the padding.", None),
         ("A pooling layer uses the mask to <b>exclude</b> masked positions from "
          "the average:",
          r"\bar{\mathbf{h}} = \frac{\sum_t m_t \mathbf{h}_{(t)}}{\sum_t m_t}"),
         ("And an attention layer (§16.6) uses it to set masked logits to "
          "$-\\infty$ before the softmax, so they receive exactly zero weight:",
          r"\alpha_t = \frac{\exp\bigl(e_t + (m_t - 1)\cdot\infty\bigr)}"
          r"{\sum_{t'} \exp\bigl(e_{t'} + (m_{t'} - 1)\cdot\infty\bigr)}"),
         ("In practice $-\\infty$ is implemented as a large negative constant "
          "such as $-10^{9}$, because $\\exp(-\\infty)$ produces NaN gradients.",
          None)],
        title="What a mask does inside each layer type",
    )

    sub("Reusing pretrained embeddings")

    md(
        "Training word vectors from scratch on a small corpus wastes most of the "
        "capacity. Pretrained embeddings (word2vec, GloVe, fastText) or whole "
        "pretrained encoders (BERT, §16.8) bring in knowledge from billions of "
        "words."
    )

    tip(
        "The transfer-learning ladder for NLP",
        "1. <b>Random embeddings</b> — fine when you have millions of labelled "
        "examples.<br>"
        "2. <b>Pretrained word embeddings, frozen</b> — the cheapest big win on "
        "small datasets.<br>"
        "3. <b>Pretrained embeddings, fine-tuned</b> — better, but overfits "
        "quickly on small data.<br>"
        "4. <b>A pretrained encoder (BERT etc.), fine-tuned</b> — the modern "
        "default; contextual, not one vector per word type.<br><br>"
        "The jump from 3 to 4 is the important one: a static embedding gives "
        "<i>bank</i> one vector; a contextual encoder gives it a different vector "
        "in <i>river bank</i> and <i>savings bank</i>.",
    )

    anim_header("What an unmasked RNN does to a short, heavily padded review")

    T_pad = 40
    real_len = 7
    decay = .88
    frames = []
    for k in range(1, T_pad + 1):
        sig_unmasked = np.array([decay ** max(0, t - real_len + 1)
                                 if t >= real_len - 1 else 1.0
                                 for t in range(k)])
        sig_masked = np.ones(k)          # the state is frozen: nothing decays
        cols = [C["accent"] if t < real_len else alpha(C["line"], .8)
                for t in range(k)]
        frames.append(go.Frame(name=str(k), data=[
            go.Bar(x=np.arange(k), y=np.ones(k) * .06,
                   marker=dict(color=cols), width=.85),
            go.Scatter(x=np.arange(k), y=sig_unmasked, mode="lines+markers",
                       line=dict(color=C["danger"], width=3),
                       marker=dict(size=5)),
            go.Scatter(x=np.arange(k), y=sig_masked, mode="lines+markers",
                       line=dict(color=C["success"], width=3),
                       marker=dict(size=5)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {k}   ·   {'REAL TOKEN' if k <= real_len else 'PADDING'}"
            f"   ·   unmasked retains {sig_unmasked[-1]:.4f} of the review"
            f"   ·   masked retains {sig_masked[-1]:.2f}",
            color=C["accent"] if k <= real_len else C["danger"])])))

    f = go.Figure(data=[
        go.Bar(x=[0], y=[.06], marker=dict(color=[C["accent"]]),
               showlegend=False),
        go.Scatter(x=[0], y=[1], mode="lines+markers",
                   name="unmasked — the review is forgotten",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[0], y=[1], mode="lines+markers",
                   name="masked — state frozen at the padding",
                   line=dict(color=C["success"], width=3)),
    ])
    f.add_vrect(x0=real_len - .5, x1=T_pad, fillcolor=alpha(C["line"], .25),
                line_width=0, annotation_text="padding")
    f.update_layout(height=410, xaxis_title="time step",
                    yaxis_title="signal from the real review retained",
                    yaxis=dict(range=[0, 1.12]),
                    title="A 7-word review padded to 40",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="step ")
    figure(f, "By step 40, the unmasked model's state is 98 % processed zeros. "
              "It still trains. It is simply worse, for no visible reason.")

    code_lab(
        "Tokenisation, BPE by hand, masking, and what masking is worth",
        '''import numpy as np, collections
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

texts, labels = _ds.sentiment_corpus(5000)
n_tr = 4000
Xtr_txt, ytr = texts[:n_tr], labels[:n_tr]
Xva_txt, yva = texts[n_tr:], labels[n_tr:]
print("=== the corpus ===")
for t, y in list(zip(texts, labels))[:5]:
    print(f"  [{'pos' if y else 'neg'}] {t}")
print(f"  {len(texts)} reviews, {labels.mean():.1%} positive")
lens = [len(t.split()) for t in texts]
print(f"  length: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}")

# ============ 1. BYTE-PAIR ENCODING, BY HAND ===========================
print()
print("=== BPE: merge the most frequent adjacent pair, repeatedly ===")
def bpe_train(corpus, n_merges=40):
    words = collections.Counter(w for t in corpus for w in t.split())
    # each word is a tuple of characters plus an end marker
    splits = {w: tuple(w) + ("</w>",) for w in words}
    merges = []
    for step in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            s = splits[w]
            for i in range(len(s)-1):
                pairs[(s[i], s[i+1])] += c
        if not pairs:
            break
        best = pairs.most_common(1)[0]
        merges.append(best[0])
        a, b = best[0]
        for w in splits:
            s, out = splits[w], []
            i = 0
            while i < len(s):
                if i < len(s)-1 and s[i] == a and s[i+1] == b:
                    out.append(a+b); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = tuple(out)
        if step < 8:
            print(f"  merge {step+1:>2}: {a!r} + {b!r} -> {a+b!r} "
                  f"({best[1]} occurrences)")
    return merges, splits

merges, splits = bpe_train(Xtr_txt[:800], n_merges=60)
print(f"  ... {len(merges)} merges total")
print()
print("  resulting tokenisation of some words:")
for w in ["the", "gorgeous", "interminable", "photography"]:
    if w in splits:
        print(f"    {w:<15} -> {list(splits[w])}")

def bpe_apply(word, merges):
    s = tuple(word) + ("</w>",)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i < len(s)-1 and s[i] == a and s[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(s[i]); i += 1
        s = tuple(out)
    return list(s)
print()
print("  a word the tokeniser has NEVER SEEN:")
for w in ["zqxwv", "unbelievableness"]:
    print(f"    {w:<20} -> {bpe_apply(w, merges)}")
print("  it decomposes into pieces. THERE IS NO 'UNKNOWN' TOKEN.")

# ============ 2. KERAS TEXTVECTORIZATION ===============================
print()
print("=== word-level vectorisation ===")
MAXLEN, VOCAB = 24, 400
vec = keras.layers.TextVectorization(max_tokens=VOCAB,
                                     output_sequence_length=MAXLEN)
vec.adapt(tf.constant(Xtr_txt))
voc = vec.get_vocabulary()
print(f"  vocabulary {len(voc)}: {voc[:12]} ...")
print(f"  index 0 = {voc[0]!r} (PAD), index 1 = {voc[1]!r} (OOV)")

demo = vec(tf.constant(["a gorgeous script", "zzz unseen words here"]))
print(f"  'a gorgeous script'      -> {demo[0].numpy()[:8]} ...")
print(f"  'zzz unseen words here'  -> {demo[1].numpy()[:8]} ...   "
      f"(1 = OOV, 0 = PAD)")

Xtr = vec(tf.constant(Xtr_txt)).numpy()
Xva = vec(tf.constant(Xva_txt)).numpy()
pad_frac = (Xtr == 0).mean()
print(f"  {pad_frac:.1%} of every batch is PADDING")

# ============ 3. DOES MASKING MATTER? ==================================
print()
print("=== masked vs unmasked, same architecture ===")
def build(mask, pool="last"):
    inp = keras.layers.Input(shape=(MAXLEN,), dtype="int64")
    z = keras.layers.Embedding(len(voc), 32, mask_zero=mask)(inp)
    if pool == "last":
        z = keras.layers.GRU(48)(z)
    else:
        z = keras.layers.GRU(48, return_sequences=True)(z)
        z = keras.layers.GlobalAveragePooling1D()(z)
    return keras.Model(inp, keras.layers.Dense(1, activation="sigmoid")(z))

print(f"{'model':<38}{'valid accuracy':>17}")
res = {}
for nm, kw in [("GRU, mask_zero=FALSE", dict(mask=False)),
               ("GRU, mask_zero=True", dict(mask=True)),
               ("GRU + avg-pool, mask_zero=FALSE", dict(mask=False, pool="avg")),
               ("GRU + avg-pool, mask_zero=True", dict(mask=True, pool="avg"))]:
    tf.random.set_seed(0)
    m = build(**kw)
    m.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(3e-3),
              metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=8, batch_size=64, verbose=0)
    acc = m.evaluate(Xva, yva, verbose=0, return_dict=True)["accuracy"]
    res[nm] = acc
    print(f"{nm:<38}{acc:>17.4f}")

# ============ 4. WHERE THE MASK GOES ===================================
print()
print("=== the mask is computed and propagated automatically ===")
emb = keras.layers.Embedding(len(voc), 8, mask_zero=True)
sample = tf.constant(Xtr[:2])
out = emb(sample)
mask = emb.compute_mask(sample).numpy()
print(f"  input  : {sample[0].numpy()[:14]}")
print(f"  mask   : {mask[0][:14].astype(int)}   (0 where the input was 0)")
print(f"  real tokens in row 0: {mask[0].sum()} of {MAXLEN}")

gru = keras.layers.GRU(6, return_sequences=True)
seq = gru(out, mask=emb.compute_mask(sample)).numpy()
last_real = int(mask[0].sum()) - 1
print()
print(f"  GRU output at the last REAL step  (t={last_real}): "
      f"{seq[0, last_real].round(3)}")
print(f"  GRU output at the last PAD  step  (t={MAXLEN-1}): "
      f"{seq[0, -1].round(3)}")
print(f"  identical: {np.allclose(seq[0, last_real], seq[0, -1], atol=1e-6)}")
print("  -> the state was FROZEN through the padding, exactly as derived")

# --- the same thing without a mask -----------------------------------
seq_nm = keras.layers.GRU(6, return_sequences=True)(out).numpy()
drift = np.abs(seq_nm[0, last_real] - seq_nm[0, -1]).mean()
print(f"\\n  WITHOUT a mask, the state drifts by {drift:.4f} through the padding")

# ============ 5. PRETRAINED EMBEDDINGS =================================
print()
print("=== simulating the transfer-learning ladder ===")
# stand-in for GloVe: train embeddings on a much larger unlabelled corpus
big_txt, _ = _ds.sentiment_corpus(20000, seed=7)
tf.random.set_seed(0)
pre = keras.Sequential([keras.layers.Input(shape=(MAXLEN,), dtype="int64"),
                        keras.layers.Embedding(len(voc), 32, mask_zero=True),
                        keras.layers.GRU(48, return_sequences=True),
                        keras.layers.Dense(len(voc))])
Xbig = vec(tf.constant(big_txt)).numpy()
pre.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(3e-3))
pre.fit(Xbig[:, :-1], Xbig[:, 1:], epochs=3, batch_size=128, verbose=0)
pretrained_W = pre.layers[0].get_weights()[0]
print(f"  pretrained {pretrained_W.shape} embeddings on {len(big_txt)} "
      f"unlabelled texts (a language-modelling objective)")

print()
print(f"{'setup':<34}{'trainable params':>18}{'valid accuracy':>17}")
for nm, init, trainable, n_lab in [
        ("random, 200 labels",   None,          True,  200),
        ("pretrained frozen, 200 labels", pretrained_W, False, 200),
        ("pretrained fine-tuned, 200 lab", pretrained_W, True, 200),
        ("random, 4000 labels",  None,          True,  4000)]:
    tf.random.set_seed(0)
    e = keras.layers.Embedding(len(voc), 32, mask_zero=True,
                               trainable=trainable)
    inp = keras.layers.Input(shape=(MAXLEN,), dtype="int64")
    z = e(inp)
    z = keras.layers.GRU(48)(z)
    m = keras.Model(inp, keras.layers.Dense(1, activation="sigmoid")(z))
    if init is not None:
        e.set_weights([init])
    m.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(3e-3),
              metrics=["accuracy"])
    m.fit(Xtr[:n_lab], ytr[:n_lab], epochs=14, batch_size=32, verbose=0)
    acc = m.evaluate(Xva, yva, verbose=0, return_dict=True)["accuracy"]
    print(f"{nm:<34}{sum(int(np.prod(w.shape)) for w in m.trainable_weights):>18,}"
          f"{acc:>17.4f}")
print()
print("  frozen pretrained embeddings win when labels are scarce.")
print("  with plenty of labels the advantage shrinks -- exactly the")
print("  transfer-learning pattern of chapter 11.")

import plotly.graph_objects as go
fig = go.Figure(go.Bar(x=list(res.keys()), y=list(res.values()),
                       marker=dict(color=[C["danger"], C["success"],
                                          C["danger"], C["success"]])))
fig.update_layout(height=400, yaxis_title="validation accuracy",
                  yaxis=dict(range=[.5, 1.0]),
                  title="Masking is one keyword argument and it is not optional")
''',
        key="ch16_sentiment",
    )

    quiz(
        "You pad a batch of 5-word reviews to length 200 and forget "
        "<code>mask_zero=True</code>. What happens?",
        ["Keras raises a ValueError",
         "The model trains normally and is simply worse — 195 steps of the RNN "
         "process zeros and wash out the review",
         "The padding is ignored automatically",
         "The loss becomes NaN"],
        1,
        "Nothing errors. The RNN dutifully updates its state 195 more times on "
        "zero inputs, and by the final step almost nothing of the review "
        "survives. The accuracy is bad and there is no diagnostic pointing at "
        "the cause — which is what makes this the most expensive one-word bug in "
        "NLP.",
        key="ch16q1",
    )

    keypoints([
        "<b>Subword tokenisation</b> (BPE) eliminates out-of-vocabulary words by "
        "construction.",
        "Padding must be <b>masked</b>: <code>mask_zero=True</code> on the "
        "Embedding, and the mask propagates.",
        "A mask freezes the RNN state, excludes positions from pooling, and sets "
        "attention logits to $-\\infty$.",
        "Unmasked padding <b>fails silently</b> — the model trains and is merely "
        "worse.",
        "Pretrained embeddings help most when labels are scarce; contextual "
        "encoders help more still.",
    ])

# ==========================================================================
def s_16_4():
    section("16.4", "An Encoder–Decoder Network for Machine Translation")

    lead(
        "Translation is the archetypal sequence-to-sequence problem: the input "
        "and output have <b>different lengths</b> and <b>different alphabets</b>. "
        "The encoder–decoder was the first neural architecture that handled it "
        "end to end."
    )

    sub("The architecture")

    md(
        "An **encoder** RNN reads the source sequence and compresses it into its "
        "final state. That state initialises a **decoder** RNN, which generates "
        "the target sequence one token at a time."
    )

    math(r"""
    \mathbf{c} \;=\; \mathrm{Encoder}\bigl(x_1, \dots, x_{T_x}\bigr)
    \;=\; \mathbf{h}^{\text{enc}}_{(T_x)}
    """)
    math(r"""
    p\bigl(y_1, \dots, y_{T_y} \mid \mathbf{x}\bigr)
    \;=\; \prod_{t=1}^{T_y}
      p\bigl(y_t \mid y_{<t},\, \mathbf{c}\bigr)
    """)

    where({
        r"\mathbf{c}": "the <b>context vector</b> — everything the decoder will "
                       "ever know about the source",
        r"T_x, T_y": "the source and target lengths, which need not be equal",
        r"y_{<t}": "the tokens generated so far — this is what makes it "
                   "<b>autoregressive</b>",
    })

    sub("Teacher forcing")

    md(
        "At training time the decoder is fed the **true** previous token, not its "
        "own prediction. This is called *teacher forcing*, and it is what makes "
        "training parallel over the time axis."
    )

    derive(
        [("<b>Without</b> teacher forcing, the decoder input at step $t$ is its "
          "own sample from step $t-1$:",
          r"\hat y_t \sim p_\theta\bigl(\cdot \mid \hat y_{<t}, \mathbf{c}\bigr)"),
         ("This is sequential — you cannot compute step $t$ until step $t-1$ has "
          "been sampled — and early in training the model's own outputs are "
          "garbage, so it learns from garbage context. Convergence is slow or "
          "absent.", None),
         ("<b>With</b> teacher forcing, the decoder input at step $t$ is the "
          "ground truth $y_{t-1}$:",
          r"\mathcal{L} = -\sum_{t=1}^{T_y} \log p_\theta\bigl(y_t \mid "
          r"y_{<t}^{\;\text{true}}, \mathbf{c}\bigr)"),
         ("Every step's input is known in advance, so the whole target sequence "
          "can be pushed through the decoder in <b>one parallel pass</b> — the "
          "same trick as the seq2seq forecasting of §15.6.", None),
         ("<b>The cost is exposure bias.</b> At training time the decoder only "
          "ever sees correct prefixes; at inference it sees its own, possibly "
          "wrong, prefixes. One mistake moves it into a state distribution it was "
          "never trained on, and errors compound — the same structural problem as "
          "recursive forecasting (§15.5).", None),
         ("Mitigations: <b>scheduled sampling</b> (mix in the model's own "
          "predictions with a probability that rises during training), "
          "<b>beam search</b> at inference (§16.5), and — most effectively — "
          "simply training on far more data.", None)],
        title="Teacher forcing, and the exposure bias it buys",
    )

    sub("Special tokens")

    table(
        ["Token", "Purpose", "Where"],
        [["<code>&lt;sos&gt;</code> / <code>&lt;bos&gt;</code>",
          "Start of sequence — the decoder's first input",
          "Prepended to the decoder input"],
         ["<code>&lt;eos&gt;</code>",
          "End of sequence — <b>tells generation when to stop</b>",
          "Appended to the decoder target"],
         ["<code>&lt;pad&gt;</code>", "Padding, always index 0", "Masked (§16.3)"],
         ["<code>&lt;unk&gt;</code>", "Out-of-vocabulary",
          "Unnecessary with subword tokenisation"]],
    )

    pitfall(
        "Without an <code>&lt;eos&gt;</code> token, generation never terminates",
        "The decoder is a loop, and something has to break it. If you forget to "
        "append <code>&lt;eos&gt;</code> to the training targets, the model never "
        "learns to emit it, and at inference you either run to an arbitrary "
        "maximum length or loop forever. Equally: the decoder <b>input</b> is the "
        "target shifted right with <code>&lt;sos&gt;</code> prepended, while the "
        "decoder <b>target</b> has <code>&lt;eos&gt;</code> appended. Getting "
        "that off-by-one wrong makes the model predict the token it was just "
        "given — a suspiciously low loss and useless output.",
    )

    anim_header("Teacher forcing vs free running")

    src_toks = ["March", "3", ",", "2019"]
    tgt_toks = ["2", "0", "1", "9", "-", "0", "3", "-", "0", "3"]
    wrong_at = 4
    frames = []
    for k in range(1, len(tgt_toks) + 1):
        ann = []
        shapes = []
        # encoder row
        for i, t in enumerate(src_toks):
            shapes.append(go.Scatter(
                x=[i * 1.4, i * 1.4 + 1.1, i * 1.4 + 1.1, i * 1.4, i * 1.4],
                y=[2.6, 2.6, 3.2, 3.2, 2.6], fill="toself",
                fillcolor=alpha(C["accent"], .85),
                line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=i * 1.4 + .55, y=2.9, text=t, showarrow=False,
                            font=dict(size=10, color="#fff")))
        ann.append(dict(x=len(src_toks) * 1.4 + .6, y=2.9, text="→ c",
                        showarrow=False,
                        font=dict(size=12, color=C["ink"])))
        # teacher-forced row and free-running row
        for row, (y0, label, col) in enumerate(
                [(1.3, "teacher forcing (training)", C["success"]),
                 (0.0, "free running (inference)", C["danger"])]):
            for i in range(k):
                bad = (row == 1 and i >= wrong_at)
                tok = tgt_toks[i] if not bad else ("?" if i > wrong_at else "1")
                shapes.append(go.Scatter(
                    x=[i * 1.4, i * 1.4 + 1.1, i * 1.4 + 1.1, i * 1.4, i * 1.4],
                    y=[y0, y0, y0 + .6, y0 + .6, y0], fill="toself",
                    fillcolor=alpha(C["danger"] if bad else col, .85),
                    line=dict(color="#fff", width=2),
                    showlegend=False, hoverinfo="skip"))
                ann.append(dict(x=i * 1.4 + .55, y=y0 + .3, text=tok,
                                showarrow=False,
                                font=dict(size=10, color="#fff")))
            ann.append(dict(x=-.3, y=y0 + .3, text=label, showarrow=False,
                            xanchor="right",
                            font=dict(size=10, color=C["ink_soft"])))
        msg = ("teacher forcing: input is always the TRUE previous token"
               if k <= wrong_at else
               f"free running: step {wrong_at} was wrong, and every step after "
               f"it is conditioned on that mistake")
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       msg,
                                       color=C["success"] if k <= wrong_at
                                       else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=430, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-3.2, len(tgt_toks) * 1.4]),
                    yaxis=dict(visible=False, range=[-1.0, 3.6]),
                    annotations=list(frames[0].layout.annotations),
                    title="'March 3, 2019' → '2019-03-03'")
    anim.animate(f, frames, duration=nav.anim_ms(650), slider_prefix="step ")
    figure(f, "Exposure bias: at training every prefix is correct; at inference "
              "one mistake changes the distribution of everything downstream.")

    code_lab(
        "An encoder–decoder that learns to normalise dates",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE TASK ==============================================
src, tgt = _ds.date_pairs(9000)
print("=== date normalisation: a real seq2seq task ===")
for a, b in list(zip(src, tgt))[:6]:
    print(f"  {a:<22} -> {b}")
print("  four input formats, one output format.")
print("  the alignment is NON-MONOTONIC: the year comes LAST in the input")
print("  and FIRST in the output. That is what makes attention visible.")

# ============ 2. VOCABULARIES AND SPECIAL TOKENS =======================
PAD, SOS, EOS = 0, 1, 2
in_chars = sorted(set("".join(src)))
out_chars = sorted(set("".join(tgt)))
in_stoi = {c: i + 1 for i, c in enumerate(in_chars)}          # 0 = PAD
out_stoi = {c: i + 3 for i, c in enumerate(out_chars)}        # 0,1,2 reserved
out_itos = {i: c for c, i in out_stoi.items()}
out_itos.update({PAD: "_", SOS: "<", EOS: ">"})
V_IN, V_OUT = len(in_stoi) + 1, len(out_stoi) + 3
T_IN = max(len(s) for s in src)
T_OUT = len(tgt[0]) + 1                                       # room for <eos>
print()
print(f"  source vocab {V_IN} chars, max length {T_IN}")
print(f"  target vocab {V_OUT} (incl. PAD={PAD}, SOS={SOS}, EOS={EOS}), "
      f"length {T_OUT}")

def encode_src(s):
    ids = [in_stoi[c] for c in s]
    return ids + [PAD]*(T_IN - len(ids))

X = np.array([encode_src(s) for s in src], dtype="int32")
Yfull = np.array([[out_stoi[c] for c in t] + [EOS] for t in tgt], dtype="int32")
# decoder INPUT  = <sos> + target[:-1]      (shifted right)
# decoder TARGET = target + <eos>
Ydec_in = np.concatenate([np.full((len(Yfull), 1), SOS, dtype="int32"),
                          Yfull[:, :-1]], axis=1)
Ydec_out = Yfull
print()
print("=== the off-by-one that everyone gets wrong ===")
print(f"  target text     : {tgt[0]}")
print(f"  decoder INPUT   : {''.join(out_itos[i] for i in Ydec_in[0])}"
      f"   <- <sos> prepended, shifted right")
print(f"  decoder TARGET  : {''.join(out_itos[i] for i in Ydec_out[0])}"
      f"   <- <eos> appended")
print("  at every step the decoder sees the PREVIOUS true token and")
print("  must predict the NEXT one.")

n_tr = 8000
Xtr, Xva = X[:n_tr], X[n_tr:]
Dtr_in, Dva_in = Ydec_in[:n_tr], Ydec_in[n_tr:]
Dtr_out, Dva_out = Ydec_out[:n_tr], Ydec_out[n_tr:]

# ============ 3. THE MODEL =============================================
EMB, UNITS = 32, 128
enc_in = keras.layers.Input(shape=(T_IN,), dtype="int32", name="enc_in")
dec_in = keras.layers.Input(shape=(T_OUT,), dtype="int32", name="dec_in")

enc_emb = keras.layers.Embedding(V_IN, EMB, mask_zero=True)(enc_in)
_, enc_h = keras.layers.GRU(UNITS, return_state=True)(enc_emb)   # THE CONTEXT

dec_emb = keras.layers.Embedding(V_OUT, EMB)(dec_in)
dec_seq = keras.layers.GRU(UNITS, return_sequences=True)(
    dec_emb, initial_state=enc_h)                                # SEEDED BY IT
logits = keras.layers.Dense(V_OUT)(dec_seq)
model = keras.Model([enc_in, dec_in], logits)

print()
print("=== the model ===")
print(f"  encoder GRU final state -> the context vector c, {UNITS} numbers")
print(f"  the decoder is initialised with c and then never sees the source again")
print(f"  parameters: {model.count_params():,}")

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3),
              metrics=["accuracy"])
hist = model.fit([Xtr, Dtr_in], Dtr_out, epochs=12, batch_size=128, verbose=0,
                 validation_data=([Xva, Dva_in], Dva_out))
print(f"  final per-character accuracy: train "
      f"{hist.history['accuracy'][-1]:.4f}  valid "
      f"{hist.history['val_accuracy'][-1]:.4f}")

# ============ 4. INFERENCE: ONE TOKEN AT A TIME ========================
print()
print("=== free-running inference (no teacher any more) ===")
def translate(strings):
    xs = np.array([encode_src(s) for s in strings], dtype="int32")
    dec = np.full((len(xs), T_OUT), PAD, dtype="int32")
    dec[:, 0] = SOS
    for t in range(T_OUT - 1):
        lg = model.predict([xs, dec], verbose=0)[:, t]
        dec[:, t + 1] = lg.argmax(-1)          # greedy is fine for a
    out = []                                   # deterministic task like this
    for row in dec[:, 1:]:
        s = ""
        for i in row:
            if i == EOS:
                break
            s += out_itos.get(int(i), "?")
        out.append(s)
    return out

tests = ["March 3, 2019", "7 July 1985", "Dec 25 2021", "01/02/1999",
         "September 30, 2044"]
preds = translate(tests)
print(f"{'input':<24}{'predicted':>14}{'expected':>14}{'ok':>5}")
import datetime as _dt
for s, p in zip(tests, preds):
    print(f"  {s:<22}{p:>14}", end="")
    print(f"{'':>14}{'':>5}")

# --- accuracy on the held-out set ------------------------------------
sample_idx = np.arange(len(Xva))[:400]
pred = translate([src[n_tr + i] for i in sample_idx])
true = [tgt[n_tr + i] for i in sample_idx]
exact = np.mean([p == t for p, t in zip(pred, true)])
charwise = np.mean([np.mean([a == b for a, b in zip(p.ljust(10), t)])
                    for p, t in zip(pred, true)])
print()
print(f"  held-out EXACT-match accuracy    : {exact:.4f}")
print(f"  held-out per-character accuracy  : {charwise:.4f}")
print("  exact match is the honest metric -- one wrong character ruins a date")

print()
print("  a few failures:")
shown = 0
for p, t, s in zip(pred, true, [src[n_tr+i] for i in sample_idx]):
    if p != t and shown < 5:
        print(f"    {s:<22} -> {p:<12} (expected {t})")
        shown += 1
if shown == 0:
    print("    none in the first 400 -- the task is fully learned")

# ============ 5. TEACHER FORCING vs FREE RUNNING, MEASURED =============
print()
print("=== the exposure-bias gap ===")
tf_acc = model.evaluate([Xva, Dva_in], Dva_out, verbose=0,
                        return_dict=True)["accuracy"]
print(f"  per-character accuracy WITH teacher forcing : {tf_acc:.4f}")
print(f"  per-character accuracy free running         : {charwise:.4f}")
print(f"  gap = {tf_acc - charwise:+.4f}")
print("  the model is always better when it is handed correct prefixes.")
print("  that gap IS exposure bias.")

# ============ 6. THE CONTEXT VECTOR IS THE BOTTLENECK ==================
print()
print("=== how much can one vector hold? ===")
print(f"{'context size':>14}{'params':>10}{'exact match':>14}")
for units in [8, 16, 32, 64, 128]:
    tf.random.set_seed(0)
    ei = keras.layers.Input(shape=(T_IN,), dtype="int32")
    di = keras.layers.Input(shape=(T_OUT,), dtype="int32")
    ee = keras.layers.Embedding(V_IN, EMB, mask_zero=True)(ei)
    _, eh = keras.layers.GRU(units, return_state=True)(ee)
    de = keras.layers.Embedding(V_OUT, EMB)(di)
    ds_ = keras.layers.GRU(units, return_sequences=True)(de, initial_state=eh)
    m = keras.Model([ei, di], keras.layers.Dense(V_OUT)(ds_))
    m.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3))
    m.fit([Xtr, Dtr_in], Dtr_out, epochs=10, batch_size=128, verbose=0)
    dec = np.full((400, T_OUT), PAD, dtype="int32"); dec[:, 0] = SOS
    xs = Xva[:400]
    for t in range(T_OUT - 1):
        dec[:, t+1] = m.predict([xs, dec], verbose=0)[:, t].argmax(-1)
    got = []
    for row in dec[:, 1:]:
        s = ""
        for i in row:
            if i == EOS: break
            s += out_itos.get(int(i), "?")
        got.append(s)
    ex = np.mean([g == tgt[n_tr+i] for i, g in enumerate(got)])
    print(f"{units:>14}{m.count_params():>10,}{ex:>14.4f}")
print()
print("  a date is short, so even 16 numbers nearly suffice.")
print("  a 40-word sentence does NOT fit. That is what attention fixes.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hist.history["accuracy"], mode="lines", name="train",
                line=dict(color=C["train"], width=3))
fig.add_scatter(y=hist.history["val_accuracy"], mode="lines", name="valid",
                line=dict(color=C["valid"], width=3))
fig.update_layout(height=380, xaxis_title="epoch",
                  yaxis_title="per-character accuracy",
                  title="Encoder–decoder learning to normalise dates")
''',
        key="ch16_encdec",
    )

    keypoints([
        "The encoder compresses the source into a <b>context vector</b>; the "
        "decoder generates from it autoregressively.",
        "<b>Teacher forcing</b> feeds the true previous token, making training "
        "parallel over time.",
        "The price is <b>exposure bias</b>: at inference the decoder conditions "
        "on its own mistakes.",
        "Decoder <b>input</b> = target shifted right with "
        "<code>&lt;sos&gt;</code>; decoder <b>target</b> = target with "
        "<code>&lt;eos&gt;</code>.",
        "The single context vector is a hard bottleneck — the motivation for "
        "attention.",
    ])


# ==========================================================================
def s_16_5():
    section("16.5", "Bidirectional RNNs and Beam Search")

    lead(
        "Two independent improvements: let the encoder read the sentence "
        "backwards as well as forwards, and stop making greedy decisions at "
        "decode time."
    )

    sub("Bidirectional RNNs")

    md(
        "A causal RNN at position $t$ has seen $x_1 \\dots x_t$ only. For "
        "*encoding* — where the whole input is available up front — that is a "
        "pointless restriction. Run a second RNN right-to-left and concatenate."
    )

    math(r"""
    \overrightarrow{\mathbf{h}}_{(t)} = \phi\bigl(\mathbf{x}_{(t)}, \overrightarrow{\mathbf{h}}_{(t-1)}\bigr),
    \qquad
    \overleftarrow{\mathbf{h}}_{(t)} = \phi'\bigl(\mathbf{x}_{(t)}, \overleftarrow{\mathbf{h}}_{(t+1)}\bigr)
    """)
    math(r"""
    \mathbf{h}_{(t)} = \bigl[\,\overrightarrow{\mathbf{h}}_{(t)} \;;\; \overleftarrow{\mathbf{h}}_{(t)}\,\bigr]
    \;\in\; \mathbb{R}^{2u}
    """)

    warn(
        "Bidirectional layers can encode, but they can never generate",
        "The backward pass at position $t$ depends on $x_{t+1}, \\dots, x_T$ — "
        "the future. That is fine for an encoder, a tagger, or a classifier, "
        "where the whole input is given. It is <b>impossible</b> for an "
        "autoregressive decoder, where position $t+1$ does not exist yet. This is "
        "exactly the causal-masking distinction that reappears in the Transformer "
        "(§16.7): BERT is bidirectional and cannot generate; GPT is causal and "
        "can.",
    )

    codenote(
        "The output size doubles, and so does the state",
        "<code>Bidirectional(GRU(64))</code> outputs 128 features, not 64, and "
        "returns <b>four</b> states for an LSTM (forward $h$, forward $c$, "
        "backward $h$, backward $c$). If you are seeding a decoder from a "
        "bidirectional encoder, you must concatenate or project them — a "
        "<code>Dense</code> layer down to the decoder's size is the usual fix, "
        "and forgetting it is a common shape error.",
    )

    sub("Beam search")

    md(
        "Greedy decoding takes the most likely token at every step. That is not "
        "the same as producing the most likely *sequence*."
    )

    derive(
        [("We want the sequence that maximises the joint probability:",
          r"\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \prod_{t=1}^{T}"
          r" p_\theta\bigl(y_t \mid y_{<t}, \mathbf{x}\bigr)"),
         ("Exhaustive search is $\\mathcal{O}(V^{T})$ — for a 30 000-token "
          "vocabulary and a 20-token output that is $30\\,000^{20}$, which is not "
          "a number anyone will ever evaluate.", None),
         ("<b>Greedy</b> search takes $\\arg\\max$ at every step. It is "
          "$\\mathcal{O}(TV)$ and it is <b>not optimal</b>: a locally suboptimal "
          "token can open up a far better continuation.", None),
         ("<b>Beam search</b> keeps the $k$ best partial sequences at every step. "
          "At each step it expands all $k$ beams by all $V$ tokens, scores the "
          "$kV$ candidates, and keeps the best $k$:",
          r"\mathcal{B}_t = \operatorname*{top-}k_{\;\mathbf{y}\in "
          r"\mathcal{B}_{t-1}\times\mathcal{V}}\;\;"
          r"\sum_{i=1}^{t} \log p_\theta\bigl(y_i \mid y_{<i}, \mathbf{x}\bigr)"),
         ("Cost is $\\mathcal{O}(kTV)$ — a factor $k$ over greedy, still "
          "polynomial. $k = 1$ recovers greedy; $k \\to V^T$ recovers exhaustive "
          "search.", None),
         ("<b>Work in log space.</b> A product of 20 probabilities each around "
          "$10^{-2}$ underflows float32. Summing log-probabilities is exact and "
          "monotone-equivalent.", None)],
        title="Beam search as bounded-width best-first decoding",
    )

    pitfall(
        "Raw log-probability scoring is biased toward short sequences",
        "Every additional token adds a negative log-probability, so a shorter "
        "sequence always scores higher, all else equal. Un-normalised beam search "
        "produces translations that are systematically too short. The standard "
        "fix is <b>length normalisation</b>: divide the score by $T^{\\alpha}$ "
        "with $\\alpha \\approx 0.6$–$0.7$, tuned on a validation set. Wu et al. "
        "(2016) also add a coverage penalty to discourage ignoring parts of the "
        "source.",
    )

    note(
        "A larger beam is not always better",
        "Beam search improves BLEU up to roughly $k = 4$–$10$ and then "
        "<b>degrades</b>. The reason is that the model's most-likely sequence is "
        "often a short, generic, high-probability one — 'I don't know' is a very "
        "probable reply to almost anything. Searching harder finds it more "
        "reliably. This is the <i>beam search curse</i>, and it is why "
        "open-ended generation uses sampling (§16.1) while translation uses a "
        "small beam.",
    )

    anim_header("Beam search exploring the tree, step by step")

    rng = np.random.default_rng(7)
    Vsm, Tsm, K = 4, 5, 3
    tok = ["a", "b", "c", "d"]
    # a fixed random model: logprob of each token given a beam id
    logp = {}

    def token_logp(prefix):
        h = abs(hash(prefix)) % (2 ** 31)
        r = np.random.default_rng(h)
        z = r.normal(0, 1.4, Vsm)
        z = z - np.log(np.exp(z).sum())
        return z

    beams = [("", 0.0)]
    history = [list(beams)]
    greedy = ("", 0.0)
    for t in range(Tsm):
        cand = []
        for pre, sc in beams:
            lp = token_logp(pre)
            for v in range(Vsm):
                cand.append((pre + tok[v], sc + lp[v]))
        cand.sort(key=lambda x: -x[1])
        beams = cand[:K]
        history.append(list(beams))
        g_lp = token_logp(greedy[0])
        gv = int(np.argmax(g_lp))
        greedy = (greedy[0] + tok[gv], greedy[1] + g_lp[gv])

    frames = []
    for t, bs in enumerate(history):
        ys = [b[1] for b in bs]
        names = [b[0] if b[0] else "∅" for b in bs]
        g_prefix = greedy[0][:t]
        g_score = 0.0
        pre = ""
        for c in g_prefix:
            g_score += token_logp(pre)[tok.index(c)]
            pre += c
        frames.append(go.Frame(name=str(t), data=[
            go.Bar(x=names, y=ys, marker=dict(color=SEQ[:len(ys)]),
                   text=[f"{v:.2f}" for v in ys], textposition="outside"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {t}   ·   beam width k={K}   ·   best beam "
            f"{names[0]!r} = {ys[0]:.3f}   ·   greedy path "
            f"{(g_prefix or '∅')!r} = {g_score:.3f}"
            + ("   ·   BEAM WINS" if ys[0] > g_score + 1e-9 else ""),
            color=C["success"] if ys[0] > g_score + 1e-9 else C["ink_soft"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=420, yaxis_title="cumulative log-probability",
                    xaxis_title="partial sequence",
                    title=f"Beam search, k = {K}")
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="step ")
    figure(f, "The beam keeps candidates that were not the single best at their "
              "step — and one of them usually ends up ahead.")

    code_lab(
        "Bidirectional encoders, and beam search implemented from scratch",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. BIDIRECTIONAL SHAPES ==================================
print("=== Bidirectional doubles everything ===")
x = tf.zeros((2, 12, 5))
uni = keras.layers.GRU(64, return_sequences=True)
bi  = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))
print(f"  GRU(64)                -> {tuple(uni(x).shape)}   "
      f"{uni.count_params():>7,} params")
print(f"  Bidirectional(GRU(64)) -> {tuple(bi(x).shape)}   "
      f"{bi.count_params():>7,} params   <- 2x")

lstm_bi = keras.layers.Bidirectional(keras.layers.LSTM(32, return_state=True))
outs = lstm_bi(tf.zeros((2, 12, 5)))
print(f"  Bidirectional(LSTM, return_state=True) returns {len(outs)} tensors:")
print(f"    output {tuple(outs[0].shape)}, then fwd h, fwd c, bwd h, bwd c")
print("  to seed a decoder you must concatenate or project them")

# --- merge modes -----------------------------------------------------
print()
print("=== merge_mode ===")
for mode in ["concat", "sum", "ave", "mul"]:
    l = keras.layers.Bidirectional(keras.layers.GRU(64), merge_mode=mode)
    print(f"  merge_mode={mode:<8} -> {tuple(l(x).shape)}")

# ============ 2. BIDIRECTIONAL WHERE IT IS LEGAL =======================
print()
print("=== a tagging task: bidirectional helps a lot ===")
# task: label each character of a date string with its ROLE (day/month/year/sep)
src, tgt = _ds.date_pairs(6000)
chars = sorted(set("".join(src)))
stoi = {c: i+1 for i, c in enumerate(chars)}
T_IN = max(len(s) for s in src)

def roles(s, iso):
    """Label each char by which ISO field it contributes to."""
    y, yr, mo, dy = [], iso[:4], iso[5:7], iso[8:10]
    for c in s:
        if c.isdigit():
            y.append(1)                      # a digit -- ambiguous ALONE
        elif c.isalpha():
            y.append(2)                      # a month name
        else:
            y.append(3)                      # a separator
    return y

X = np.zeros((len(src), T_IN), dtype="int32")
Y = np.zeros((len(src), T_IN), dtype="int32")
for i, (s, t) in enumerate(zip(src, tgt)):
    ids = [stoi[c] for c in s]
    X[i, :len(ids)] = ids
    Y[i, :len(ids)] = roles(s, t)
n_tr = 5000

print(f"{'encoder':<34}{'params':>9}{'valid accuracy':>17}")
for nm, make in [
        ("GRU(48) — causal",
         lambda: keras.layers.GRU(48, return_sequences=True)),
        ("Bidirectional(GRU(48))",
         lambda: keras.layers.Bidirectional(
             keras.layers.GRU(48, return_sequences=True)))]:
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(T_IN,), dtype="int32"),
                          keras.layers.Embedding(len(stoi)+1, 24, mask_zero=True),
                          make(),
                          keras.layers.Dense(4)])
    m.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3), metrics=["accuracy"])
    m.fit(X[:n_tr], Y[:n_tr], epochs=6, batch_size=128, verbose=0)
    acc = m.evaluate(X[n_tr:], Y[n_tr:], verbose=0, return_dict=True)["accuracy"]
    print(f"{nm:<34}{m.count_params():>9,}{acc:>17.4f}")
print("  a causal encoder cannot see '/1999' when it is reading the leading '01'")

# ============ 3. WHY BIDIRECTIONAL CANNOT DECODE =======================
print()
print("=== the reason a decoder can never be bidirectional ===")
print("  forward  h(t) depends on x(1..t)      -- available at generation time")
print("  backward h(t) depends on x(t..T)      -- x(t+1) DOES NOT EXIST YET")
print("  so a bidirectional layer can encode, tag or classify,")
print("  but it can never generate. BERT vs GPT is exactly this distinction.")

# ============ 4. BEAM SEARCH FROM SCRATCH ==============================
print()
print("="*66)
print("Beam search")
print("="*66)

# train a small encoder-decoder to decode with
PAD, SOS, EOS = 0, 1, 2
out_chars = sorted(set("".join(tgt)))
o_stoi = {c: i+3 for i, c in enumerate(out_chars)}
o_itos = {i: c for c, i in o_stoi.items()}
V_IN, V_OUT = len(stoi)+1, len(o_stoi)+3
T_OUT = len(tgt[0]) + 1
Yf = np.array([[o_stoi[c] for c in t] + [EOS] for t in tgt], dtype="int32")
Din = np.concatenate([np.full((len(Yf),1), SOS, dtype="int32"), Yf[:,:-1]], 1)

ei = keras.layers.Input(shape=(T_IN,), dtype="int32")
di = keras.layers.Input(shape=(T_OUT,), dtype="int32")
ee = keras.layers.Embedding(V_IN, 32, mask_zero=True)(ei)
_, eh = keras.layers.GRU(96, return_state=True)(ee)
de = keras.layers.Embedding(V_OUT, 32)(di)
ds_ = keras.layers.GRU(96, return_sequences=True)(de, initial_state=eh)
model = keras.Model([ei, di], keras.layers.Dense(V_OUT)(ds_))
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3))
model.fit([X[:n_tr], Din[:n_tr]], Yf[:n_tr], epochs=6, batch_size=128, verbose=0)
print("  trained a small encoder-decoder to decode with")

def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

def decode_batch(rows, k=1, alpha=0.0, max_len=T_OUT-1):
    """Beam search over a whole batch at once.

    All (batch x k) beams go through the model in ONE call per step.
    Calling model.predict() inside a per-beam loop is ~100x slower and is
    the single most common reason a beam-search implementation is unusable.
    """
    B = len(rows)
    xs = np.repeat(rows, k, axis=0).astype("int32")          # (B*k, T_IN)
    toks = np.full((B*k, T_OUT), PAD, dtype="int32"); toks[:, 0] = SOS
    # only beam 0 is live at step 0, or all k beams would pick the same tokens
    scores = np.full((B, k), -1e9); scores[:, 0] = 0.0
    scores = scores.reshape(-1)
    done = np.zeros(B*k, dtype=bool)

    for t in range(max_len):
        lp = log_softmax(np.asarray(model([xs, toks], training=False))[:, t])
        lp[done] = -1e9
        lp[done, PAD] = 0.0            # a finished beam repeats PAD for free
        cand = (scores[:, None] + lp).reshape(B, k*V_OUT)
        best = np.argsort(-cand, axis=1)[:, :k]
        new_scores = np.take_along_axis(cand, best, 1)
        beam_idx, tok_idx = best // V_OUT, best % V_OUT
        nt = np.empty_like(toks); nd = np.empty_like(done)
        for b in range(B):
            src_rows = b*k + beam_idx[b]
            nt[b*k:(b+1)*k] = toks[src_rows]
            nt[b*k:(b+1)*k, t+1] = tok_idx[b]
            nd[b*k:(b+1)*k] = done[src_rows] | (tok_idx[b] == EOS)
        toks, done, scores = nt, nd, new_scores.reshape(-1)
        if done.all():
            break

    out = []
    for b in range(B):
        cands = []
        for j in range(k):
            txt = ""
            for i in toks[b*k+j, 1:]:
                if i == EOS:
                    break
                txt += o_itos.get(int(i), "?")
            n = max(1, len(txt))
            sc = scores[b*k+j]
            # LENGTH NORMALISATION happens in the RANKING, not in the score
            cands.append((txt, float(sc), sc/(n**alpha) if alpha else sc))
        cands.sort(key=lambda c: -c[2])
        out.append((cands[0][0], cands[0][1]))
    return out

print()
print("=== greedy vs beam, on held-out dates ===")
N_TEST = 200
idx = np.arange(n_tr, n_tr + N_TEST)
rows = X[idx]
truth = [tgt[i] for i in idx]
print(f"{'decoder':<28}{'exact match':>14}{'mean logprob':>16}{'time':>9}")
results, decoded = {}, {}
for nm, k, alpha in [("greedy (k=1)", 1, 0.0),
                     ("beam k=3", 3, 0.0),
                     ("beam k=3, length-norm 0.7", 3, 0.7),
                     ("beam k=5, length-norm 0.7", 5, 0.7)]:
    t0 = time.perf_counter()
    got = decode_batch(rows, k=k, alpha=alpha)
    dt = time.perf_counter() - t0
    hits = sum(g == t for (g, _), t in zip(got, truth))
    lp = float(np.mean([s for _, s in got]))
    results[nm] = hits/N_TEST
    decoded[nm] = [g for g, _ in got]
    print(f"{nm:<28}{hits/N_TEST:>14.4f}{lp:>16.4f}{dt:>8.1f}s")

# ============ 5. WHERE BEAM SEARCH BEATS GREEDY ========================
print()
print("=== a case where greedy takes the locally-best wrong turn ===")
g_all = decoded["greedy (k=1)"]
b_all = decoded["beam k=5, length-norm 0.7"]
shown = 0
for j, i in enumerate(idx):
    if g_all[j] != b_all[j] and shown < 4:
        print(f"  input  {src[i]:<22}")
        print(f"    greedy {g_all[j]:<12} {'OK' if g_all[j]==tgt[i] else 'WRONG'}")
        print(f"    beam   {b_all[j]:<12} {'OK' if b_all[j]==tgt[i] else 'WRONG'}")
        print(f"    truth  {tgt[i]}")
        shown += 1
if shown == 0:
    print("  none in this sample -- the task is easy enough that greedy suffices.")
    print("  on translation, beam search is worth 1-2 BLEU points.")

# ============ 6. THE SHORT-SEQUENCE BIAS ===============================
print()
print("=== why length normalisation is needed ===")
print("  every extra token adds a NEGATIVE log-probability, so:")
for n, p in [(4, .8), (8, .8), (16, .8), (32, .8)]:
    print(f"    a {n:>2}-token sequence at p=0.8 per token scores "
          f"{n*np.log(p):>8.3f}  (normalised {np.log(p):>7.3f})")
print("  un-normalised beam search therefore prefers SHORT outputs,")
print("  which is a systematic bias, not a modelling failure.")

import plotly.graph_objects as go
fig = go.Figure(go.Bar(x=list(results.keys()), y=list(results.values()),
                       marker=dict(color=SEQ[:len(results)]),
                       text=[f"{v:.3f}" for v in results.values()],
                       textposition="outside"))
fig.update_layout(height=400, yaxis_title="exact-match accuracy",
                  title="Decoding strategy")
''',
        key="ch16_beam",
    )

    keypoints([
        "<b>Bidirectional</b> concatenates a forward and a backward RNN — output "
        "size doubles.",
        "It can encode, tag or classify, but <b>never generate</b>: the backward "
        "pass needs the future.",
        "<b>Greedy decoding maximises each step, not the sequence.</b> Beam "
        "search keeps $k$ candidates.",
        "Always score in <b>log space</b>, and apply <b>length normalisation</b> "
        "($T^{\\alpha}$, $\\alpha \\approx 0.7$).",
        "Beams beyond ~10 make output <i>worse</i> — the beam-search curse.",
    ])


# ==========================================================================
def s_16_6():
    section("16.6", "Attention Mechanisms")

    lead(
        "The decoder should not be limited to one context vector. Let it compute "
        "a <b>different</b> weighted mixture of the encoder states at every "
        "output step, and choose those weights itself."
    )

    sub("The idea")

    md(
        "Instead of a single $\\mathbf{c}$, the decoder at step $t$ gets its own "
        "context $\\mathbf{c}_{(t)}$: a weighted average of **all** encoder "
        "outputs, with weights that depend on what the decoder currently needs."
    )

    math(r"""
    \mathbf{c}_{(t)} \;=\; \sum_{i=1}^{T_x} \alpha_{(t,i)}\, \mathbf{h}^{\text{enc}}_{(i)}
    \qquad\text{with}\qquad
    \sum_{i=1}^{T_x} \alpha_{(t,i)} = 1,\;\; \alpha_{(t,i)} \ge 0
    """)

    md("The weights come from a softmax over **alignment scores**:")

    math(r"""
    \alpha_{(t,i)} \;=\;
      \frac{\exp\bigl(e_{(t,i)}\bigr)}{\sum_{i'=1}^{T_x}\exp\bigl(e_{(t,i')}\bigr)},
    \qquad
    e_{(t,i)} \;=\; a\bigl(\mathbf{h}^{\text{dec}}_{(t-1)},\, \mathbf{h}^{\text{enc}}_{(i)}\bigr)
    """)

    sub("The two scoring functions")

    table(
        ["Name", "Score $e_{(t,i)}$", "Parameters", "Note"],
        [["<b>Bahdanau</b> (additive, 2014)",
          "$\\mathbf{v}^\\top \\tanh\\bigl(\\mathbf{W}[\\mathbf{h}^{\\text{dec}}_{(t-1)}; \\mathbf{h}^{\\text{enc}}_{(i)}]\\bigr)$",
          "$\\mathbf{W}, \\mathbf{v}$",
          "A tiny MLP; works when the two spaces differ in size"],
         ["<b>Luong</b> (multiplicative, 2015)",
          "$\\mathbf{h}^{\\text{dec}\\top}_{(t)} \\mathbf{h}^{\\text{enc}}_{(i)}$",
          "none (or one $\\mathbf{W}$ for the 'general' form)",
          "<b>One matrix multiply</b> — far faster, and what won"],
         ["<b>Scaled dot-product</b> (2017)",
          "$\\dfrac{\\mathbf{q}^\\top \\mathbf{k}}{\\sqrt{d_k}}$", "none",
          "Luong plus a scaling factor — the Transformer's (§16.7)"]],
    )

    derive(
        [("<b>Why the $1/\\sqrt{d_k}$ scaling.</b> Suppose the components of "
          "$\\mathbf{q}$ and $\\mathbf{k}$ are independent with mean 0 and "
          "variance 1.", None),
         ("Their dot product is a sum of $d_k$ independent products:",
          r"\mathbf{q}^\top\mathbf{k} = \sum_{j=1}^{d_k} q_j k_j"),
         ("Each term has mean $\\mathbb{E}[q_j k_j] = 0$ and variance "
          "$\\mathrm{Var}(q_j k_j) = 1$, so by independence:",
          r"\mathbb{E}\bigl[\mathbf{q}^\top\mathbf{k}\bigr] = 0,"
          r"\qquad \mathrm{Var}\bigl(\mathbf{q}^\top\mathbf{k}\bigr) = d_k"),
         ("The standard deviation therefore grows as $\\sqrt{d_k}$. With "
          "$d_k = 512$, logits routinely reach $\\pm 20$ — and a softmax over "
          "logits that far apart is a <b>one-hot vector</b>.", None),
         ("A saturated softmax has vanishing gradient: "
          "$\\partial \\alpha_i / \\partial e_j = \\alpha_i(\\delta_{ij} - "
          "\\alpha_j) \\to 0$ when any $\\alpha_i \\to 1$. Attention would stop "
          "learning.", None),
         ("Dividing by $\\sqrt{d_k}$ restores unit variance regardless of "
          "dimension, keeping the softmax in its responsive range:",
          r"\mathrm{Var}\Bigl(\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_k}}\Bigr) = 1"),
         ("<b>That single $\\sqrt{d_k}$ is the difference between a Transformer "
          "that trains and one that does not.</b> It is not a cosmetic detail.",
          None)],
        title="Why scaled dot-product attention divides by √dₖ",
    )

    sub("Queries, keys and values")

    md(
        "The modern framing generalises all of the above. Attention is a "
        "**differentiable dictionary lookup**:"
    )

    math(r"""
    \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
    \;=\; \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
    """)

    where({
        r"\mathbf{Q} \in \mathbb{R}^{n_q \times d_k}":
            "the <b>queries</b> — what each output position is looking for",
        r"\mathbf{K} \in \mathbb{R}^{n_v \times d_k}":
            "the <b>keys</b> — what each input position offers as an index",
        r"\mathbf{V} \in \mathbb{R}^{n_v \times d_v}":
            "the <b>values</b> — what is actually retrieved",
    })

    idea(
        "A hard dictionary lookup, made differentiable",
        "In a Python dict, you compare a key for exact equality and retrieve one "
        "value. Attention compares <i>every</i> key by dot product, turns the "
        "similarities into a probability distribution, and returns a "
        "<b>weighted average</b> of all values. Sharpen the softmax toward "
        "one-hot and you recover the ordinary lookup exactly. Because it is a "
        "soft average, it is differentiable — so the network can learn <i>what "
        "to look up</i> by gradient descent. That is the whole trick.",
    )

    proof(
        "Attention's gradient path between any two positions has length 1",
        "In an RNN, output $t$ depends on input $i$ through $t - i$ recurrent "
        "steps, so the gradient is a product of $t-i$ Jacobians — the vanishing "
        "problem of §15.2. In attention, output $t$ is a <b>direct weighted sum</b> "
        "of all inputs: $\\partial \\mathbf{c}_t / \\partial \\mathbf{v}_i = "
        "\\alpha_{ti} \\mathbf{I}$, a single term. Distance in the sequence "
        "costs <b>nothing</b> in gradient path length. This is why attention "
        "handles long-range dependencies that no RNN can — and it is a stronger "
        "statement than 'attention has a larger receptive field'.",
    )

    anim_header("Attention weights across a decoding run")

    src_t = ["March", "3", ",", "2019", "<pad>"]
    out_chars = ["2", "0", "1", "9", "-", "0", "3", "-", "0", "3"]
    # the axis labels must be UNIQUE and non-numeric, or Plotly coerces them
    # to numbers and the heatmap cells land off-scale
    out_t = [f"{i+1}: {c}" for i, c in enumerate(out_chars)]
    src_lab = [f"{i+1}: {c}" for i, c in enumerate(src_t)]
    focus = [3, 3, 3, 3, 2, 0, 0, 2, 1, 1]
    rng2 = np.random.default_rng(3)
    A = np.zeros((len(out_t), len(src_t)))
    for i, fo in enumerate(focus):
        w = np.exp(-1.6 * np.abs(np.arange(len(src_t)) - fo))
        w[-1] = 1e-6                                   # padding is masked
        w = w * (1 + .25 * rng2.random(len(src_t)))
        A[i] = w / w.sum()

    frames = []
    for k in range(1, len(out_t) + 1):
        Z = np.full((len(out_t), len(src_t)), np.nan)
        Z[:k] = A[:k]
        frames.append(go.Frame(name=str(k), data=[
            go.Heatmap(z=Z, x=src_lab, y=out_t, colorscale=nav.cscale(),
                       zmin=0, zmax=A.max(), xgap=2, ygap=2,
                       colorbar=dict(title="α")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"output step {k}: emitting {out_chars[k-1]!r}   ·   attending mostly "
            f"to {src_t[int(np.argmax(A[k-1]))]!r} "
            f"(α = {A[k-1].max():.2f})   ·   padding weight "
            f"{A[k-1, -1]:.4f}")])))

    _Z0 = np.full_like(A, np.nan)
    _Z0[0] = A[0]          # seed one real row so the colour scale is set
    f = go.Figure(data=[go.Heatmap(z=_Z0, x=src_lab, y=out_t,
                                   colorscale=nav.cscale(), zmin=0,
                                   zmax=A.max(), xgap=2, ygap=2,
                                   colorbar=dict(title="α"))])
    f.update_layout(height=480, xaxis_title="source token",
                    yaxis_title="generated token",
                    xaxis=dict(type="category"),
                    yaxis=dict(type="category", autorange="reversed"),
                    title="Alignment learned by attention — note it is NOT diagonal")
    anim.animate(f, frames, duration=nav.anim_ms(520), slider_prefix="step ")
    figure(f, "The year is written first in the output but comes last in the "
              "input. A single context vector cannot express that; attention "
              "learns it as an alignment.")

    code_lab(
        "Attention from scratch — three scoring functions, and the √dₖ proof",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

# ============ 1. THE THREE SCORING FUNCTIONS ===========================
rng = np.random.default_rng(0)
Tx, Dh = 6, 8
H = rng.normal(0, 1, (Tx, Dh))        # encoder outputs
s = rng.normal(0, 1, Dh)              # decoder state

def luong(s, H):            return H @ s
def scaled_dot(s, H):       return (H @ s) / np.sqrt(H.shape[-1])
def bahdanau(s, H, W, v):   return np.tanh(np.concatenate(
                                [np.tile(s, (len(H), 1)), H], 1) @ W) @ v

W = rng.normal(0, .3, (2*Dh, 12)); v = rng.normal(0, .3, 12)
print("=== three ways to score alignment ===")
print(f"{'scorer':<22}{'scores (first 4)':>34}{'max alpha':>12}")
for nm, e in [("Luong (dot)", luong(s, H)),
              ("scaled dot-product", scaled_dot(s, H)),
              ("Bahdanau (additive)", bahdanau(s, H, W, v))]:
    a = softmax(e)
    print(f"{nm:<22}{str(np.round(e[:4], 3)):>34}{a.max():>12.4f}")

c = softmax(scaled_dot(s, H)) @ H
print(f"\\n  context vector = sum_i alpha_i h_i, shape {c.shape}")
print(f"  it is a CONVEX COMBINATION of the encoder outputs:")
print(f"    weights sum to {softmax(scaled_dot(s, H)).sum():.6f}, all >= 0")

# ============ 2. WHY sqrt(d_k) -- THE VARIANCE ARGUMENT ================
print()
print("=== the variance of a dot product grows with the dimension ===")
print(f"{'d_k':>7}{'Var(q.k)':>12}{'predicted':>12}{'max softmax':>14}"
      f"{'max gradient':>15}")
for dk in [4, 16, 64, 256, 1024]:
    q = rng.normal(0, 1, (4000, dk)); k = rng.normal(0, 1, (4000, dk))
    dots = (q*k).sum(1)
    # what a softmax over 8 such logits looks like
    logits = rng.normal(0, np.sqrt(dk), 8)
    a = softmax(logits)
    grad = float((a*(1-a)).max())         # d alpha_i / d e_i
    print(f"{dk:>7}{dots.var():>12.1f}{dk:>12}{a.max():>14.4f}{grad:>15.6f}")
print("  at d_k=1024 the softmax is essentially one-hot and its GRADIENT IS ~0")

print()
print("=== the same, WITH the 1/sqrt(d_k) scaling ===")
print(f"{'d_k':>7}{'Var(q.k/sqrt(dk))':>21}{'max softmax':>14}{'max gradient':>15}")
for dk in [4, 16, 64, 256, 1024]:
    q = rng.normal(0, 1, (4000, dk)); k = rng.normal(0, 1, (4000, dk))
    dots = (q*k).sum(1) / np.sqrt(dk)
    logits = rng.normal(0, 1, 8)
    a = softmax(logits)
    print(f"{dk:>7}{dots.var():>21.4f}{a.max():>14.4f}"
          f"{float((a*(1-a)).max()):>15.6f}")
print("  variance is 1 at EVERY dimension -- the softmax stays responsive.")
print("  this single constant is the difference between a Transformer that")
print("  trains and one that does not.")

# ============ 3. Q, K, V AS A DIFFERENTIABLE DICTIONARY ================
print()
print("=== attention IS a soft dictionary lookup ===")
keys   = np.eye(4)                       # 4 orthogonal keys
values = np.array([[10., 0.], [0., 10.], [-10., 0.], [0., -10.]])
query  = np.array([1., 0., 0., 0.])      # exactly key 0

def attend(Q, K, V, scale=1.0, mask=None):
    e = (Q @ K.T) / scale
    if mask is not None:
        e = np.where(mask, e, -1e9)
    a = softmax(e)
    return a @ V, a

for temp, label in [(0.05, "sharp   (temperature 0.05)"),
                    (1.0,  "default (temperature 1)"),
                    (10.0, "flat    (temperature 10)")]:
    out, a = attend(query[None], keys, values, scale=temp)
    print(f"  {label:<28} weights {np.round(a[0], 3)}  -> {np.round(out[0], 2)}")
print(f"  a hard dict lookup would return {values[0]} exactly.")
print("  sharpen the softmax and attention CONVERGES to that lookup --")
print("  but stays differentiable everywhere, which a dict is not.")

# ============ 4. MASKING ===============================================
print()
print("=== masking padding, and masking the future ===")
Tq = 5
scores = rng.normal(0, 1, (Tq, Tq))
pad_mask = np.array([True, True, True, False, False])       # last 2 are PAD
_, a_pad = attend(np.eye(Tq), np.eye(Tq)*0 + scores, np.eye(Tq),
                  mask=np.tile(pad_mask, (Tq, 1)))
print(f"  padding mask     : {pad_mask.astype(int)}")
print(f"  weight on padding: {a_pad[:, 3:].sum():.2e}   (exactly 0 in practice)")

causal = np.tril(np.ones((Tq, Tq), dtype=bool))
_, a_c = attend(np.eye(Tq), scores, np.eye(Tq), mask=causal)
print(f"\\n  causal mask (lower triangular):")
for i in range(Tq):
    print(f"    step {i}: {np.round(a_c[i], 3)}   "
          f"(zero from column {i+1} on)")
print("  -1e9, not -inf: exp(-inf) gives NaN gradients")

# ============ 5. ATTENTION ON THE DATE TASK ============================
print()
print("="*66)
print("Encoder-decoder WITH and WITHOUT attention")
print("="*66)
src, tgt = _ds.date_pairs(9000)
PAD, SOS, EOS = 0, 1, 2
i_chars = sorted(set("".join(src))); o_chars = sorted(set("".join(tgt)))
i_stoi = {c: i+1 for i, c in enumerate(i_chars)}
o_stoi = {c: i+3 for i, c in enumerate(o_chars)}
o_itos = {i: c for c, i in o_stoi.items()}
T_IN = max(len(s) for s in src); T_OUT = len(tgt[0]) + 1
V_IN, V_OUT = len(i_stoi)+1, len(o_stoi)+3
X = np.zeros((len(src), T_IN), dtype="int32")
for i, s_ in enumerate(src):
    X[i, :len(s_)] = [i_stoi[c] for c in s_]
Yf = np.array([[o_stoi[c] for c in t] + [EOS] for t in tgt], dtype="int32")
Din = np.concatenate([np.full((len(Yf),1), SOS, dtype="int32"), Yf[:,:-1]], 1)
n_tr = 8000

def build(use_attention):
    """Returns (model, probe). The probe shares weights and outputs the
    alignment matrix -- built in the SAME functional graph, which is the
    only reliable way to expose an intermediate tensor in Keras 3."""
    ei = keras.layers.Input(shape=(T_IN,), dtype="int32")
    di = keras.layers.Input(shape=(T_OUT,), dtype="int32")
    ee = keras.layers.Embedding(V_IN, 32, mask_zero=True)(ei)
    enc_seq, enc_h = keras.layers.GRU(96, return_sequences=True,
                                      return_state=True)(ee)
    de = keras.layers.Embedding(V_OUT, 32)(di)
    dec_seq = keras.layers.GRU(96, return_sequences=True)(de,
                                                          initial_state=enc_h)
    probe = None
    if use_attention:
        ctx, scores = keras.layers.Attention(use_scale=True)(
            [dec_seq, enc_seq], return_attention_scores=True)
        dec_seq = keras.layers.Concatenate()([dec_seq, ctx])
    out = keras.layers.Dense(V_OUT)(dec_seq)
    model = keras.Model([ei, di], out)
    if use_attention:
        probe = keras.Model([ei, di], scores)
    return model, probe

print(f"{'model':<24}{'params':>10}{'valid char acc':>17}")
models = {}
for nm, flag in [("no attention", False), ("WITH attention", True)]:
    tf.random.set_seed(0)
    m, probe = build(flag)
    m.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-3), metrics=["accuracy"])
    m.fit([X[:n_tr], Din[:n_tr]], Yf[:n_tr], epochs=10, batch_size=128,
          verbose=0)
    acc = m.evaluate([X[n_tr:], Din[n_tr:]], Yf[n_tr:], verbose=0,
                     return_dict=True)["accuracy"]
    models[nm] = (m, probe)
    print(f"{nm:<24}{m.count_params():>10,}{acc:>17.4f}")

# ============ 6. WHAT THE ATTENTION LEARNED ============================
print()
print("=== extracting the alignment matrix ===")
att_model, probe = models["WITH attention"]
sample = 3
scores_np = probe.predict([X[sample:sample+1], Din[sample:sample+1]],
                          verbose=0)[0]
print(f"  source : {src[sample]}")
print(f"  target : {tgt[sample]}")
print(f"  alignment matrix shape {scores_np.shape} "
      f"(output steps x source positions)")
print()
print(f"  {'out':>5} {'attends to':>28}")
for t in range(min(T_OUT-1, len(tgt[sample]))):
    j = int(np.argmax(scores_np[t]))
    ch = src[sample][j] if j < len(src[sample]) else "<pad>"
    print(f"  {tgt[sample][t]:>5} {f'position {j} ({ch!r})':>28}  "
          f"alpha={scores_np[t, j]:.3f}")
print()
print("  the year digits attend to the YEAR in the source, wherever it is.")
print("  the alignment is learned, not hard-coded, and it is not diagonal.")

import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(
    z=scores_np[:len(tgt[sample])],
    x=[f"{c}" for c in src[sample]] + ["·"]*(T_IN-len(src[sample])),
    y=list(tgt[sample]), colorscale="Viridis", xgap=1, ygap=1))
fig.update_layout(height=440, yaxis=dict(autorange="reversed"),
                  xaxis_title="source character", yaxis_title="output character",
                  title=f"Learned alignment: {src[sample]} -> {tgt[sample]}")
''',
        key="ch16_attention",
    )

    quiz(
        "Why is the dot product divided by $\\sqrt{d_k}$?",
        ["To make the weights sum to 1",
         "Because $\\mathrm{Var}(\\mathbf{q}^\\top\\mathbf{k}) = d_k$, so without "
         "it the softmax saturates and its gradient vanishes",
         "To make the computation faster",
         "It is an arbitrary constant chosen empirically"],
        1,
        "With unit-variance components, the dot product has variance $d_k$, so "
        "logits scale as $\\sqrt{d_k}$. At $d_k = 512$ that gives a near-one-hot "
        "softmax whose gradient $\\alpha_i(1 - \\alpha_i)$ is ~0 — attention "
        "would stop learning. Dividing by $\\sqrt{d_k}$ restores unit variance at "
        "every model size.",
        key="ch16q2",
    )

    keypoints([
        "Attention gives the decoder a <b>different context vector at every "
        "step</b>: $\\mathbf{c}_t = \\sum_i \\alpha_{ti}\\mathbf{h}_i$.",
        "Weights come from a softmax over alignment scores — additive "
        "(Bahdanau) or multiplicative (Luong).",
        "$\\mathrm{softmax}(\\mathbf{Q}\\mathbf{K}^\\top/\\sqrt{d_k})\\mathbf{V}$ "
        "is a <b>differentiable dictionary lookup</b>.",
        "The $\\sqrt{d_k}$ exists because $\\mathrm{Var}(\\mathbf{q}^\\top"
        "\\mathbf{k}) = d_k$ — without it the softmax saturates.",
        "<b>The gradient path between any two positions has length 1</b> — no "
        "product over the sequence.",
    ])

# ==========================================================================
def s_16_7():
    section("16.7", "Attention Is All You Need — The Transformer")

    lead(
        "Vaswani et al. (2017) removed the recurrence entirely and kept only the "
        "attention. The result trains in parallel over the whole sequence, has a "
        "gradient path of length 1 between any two positions, and is the "
        "architecture behind essentially every large model since."
    )

    sub("Self-attention")

    md(
        "Cross-attention (§16.6) let the decoder look at the encoder. "
        "**Self-attention** lets a sequence look at *itself*: $\\mathbf{Q}$, "
        "$\\mathbf{K}$ and $\\mathbf{V}$ are all linear projections of the same "
        "input."
    )

    math(r"""
    \mathbf{Q} = \mathbf{X}\mathbf{W}^{Q},\quad
    \mathbf{K} = \mathbf{X}\mathbf{W}^{K},\quad
    \mathbf{V} = \mathbf{X}\mathbf{W}^{V}
    """)
    math(r"""
    \mathrm{SelfAttention}(\mathbf{X}) = \mathrm{softmax}\!\left(
      \frac{\mathbf{X}\mathbf{W}^{Q}(\mathbf{X}\mathbf{W}^{K})^{\top}}{\sqrt{d_k}}
    \right)\mathbf{X}\mathbf{W}^{V}
    """)

    sub("Multi-head attention")

    md(
        "One attention pattern per layer is limiting: a word may need to attend "
        "to its syntactic head *and* its coreferent *and* the topic. Run $h$ "
        "attention operations in parallel on lower-dimensional projections and "
        "concatenate."
    )

    math(r"""
    \mathrm{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V}) =
      \bigl[\mathrm{head}_1;\dots;\mathrm{head}_h\bigr]\mathbf{W}^{O},
    \qquad
    \mathrm{head}_i = \mathrm{Attention}\bigl(\mathbf{Q}\mathbf{W}^{Q}_i,
      \mathbf{K}\mathbf{W}^{K}_i, \mathbf{V}\mathbf{W}^{V}_i\bigr)
    """)

    note(
        "Multi-head attention costs nothing extra",
        "Each head works in dimension $d_k = d_{\\text{model}}/h$, so the total "
        "parameter count and FLOP count are essentially identical to one "
        "full-dimension head. With $d_{\\text{model}} = 512$ and $h = 8$, each "
        "head is 64-dimensional. You get $h$ different attention patterns "
        "<b>for free</b> — which is why it is always used.",
    )

    sub("Positional encoding")

    pitfall(
        "Self-attention is permutation-equivariant — it cannot see word order",
        "Shuffle the rows of $\\mathbf{X}$ and every output row is the same, just "
        "shuffled with it. Without positional information, "
        "<i>dog bites man</i> and <i>man bites dog</i> are literally the same "
        "input. Something must inject position, and that something is the "
        "<b>positional encoding</b>, added to the embeddings before the first "
        "layer.",
    )

    math(r"""
    P_{p,\,2i} = \sin\!\left(\frac{p}{10000^{\,2i/d}}\right),
    \qquad
    P_{p,\,2i+1} = \cos\!\left(\frac{p}{10000^{\,2i/d}}\right)
    """)

    derive(
        [("<b>Why sinusoids of geometrically spaced frequencies.</b> Dimension "
          "pair $i$ oscillates with wavelength "
          "$\\lambda_i = 2\\pi \\cdot 10000^{2i/d}$ — from $2\\pi$ at $i=0$ to "
          "roughly $10^4 \\cdot 2\\pi$ at $i = d/2$. Together they form a "
          "positional 'binary code' at many scales.", None),
         ("<b>The key property is that relative position is a linear map.</b> "
          "Write the pair at frequency $\\omega_i = 10000^{-2i/d}$ as a 2-vector "
          "$\\mathbf{p}^{(i)}_p = (\\sin \\omega_i p,\\; \\cos \\omega_i p)$. "
          "Then for any fixed offset $k$:",
          r"\mathbf{p}^{(i)}_{p+k} = \begin{pmatrix}\cos\omega_i k & \sin\omega_i k\\ "
          r"-\sin\omega_i k & \cos\omega_i k\end{pmatrix}\mathbf{p}^{(i)}_{p}"),
         ("That is a <b>rotation matrix depending only on $k$, not on $p$</b>. So "
          "a linear layer can learn to attend 'three tokens back' uniformly, at "
          "any absolute position — which is exactly what a language model needs.",
          None),
         ("It also <b>extrapolates</b>: sinusoids are defined at every real "
          "position, so a model can in principle process sequences longer than it "
          "was trained on. (In practice learned or rotary encodings are now more "
          "common, but the relative-position argument is the same.)", None),
         ("<b>Why added, not concatenated.</b> Concatenation would spend "
          "dimensions on position that could carry meaning, and would force every "
          "downstream weight matrix to be wider. Addition costs nothing, and "
          "because embeddings are learned, the network can allocate subspaces to "
          "content and position on its own.", None)],
        title="Why sinusoidal positional encodings work",
    )

    sub("The block")

    table(
        ["Component", "Purpose", "Note"],
        [["Multi-head <b>self-attention</b>", "Mix information across positions",
          "The only component that moves information sideways"],
         ["<b>Skip connection</b> + <b>layer norm</b>",
          "Keep the gradient path short",
          "Exactly the §14.5 and §15.7 arguments"],
         ["<b>Feed-forward</b> (two Dense layers, $4\\times$ wider inside)",
          "Per-position non-linear processing",
          "Applied identically at every position — <b>two thirds of the "
          "parameters live here</b>"],
         ["Skip + layer norm again", "Same", ""],
         ["<b>Causal mask</b> (decoder only)",
          "Prevent position $t$ from seeing $t+1$",
          "A lower-triangular mask of $-10^9$"]],
    )

    idea(
        "Pre-norm vs post-norm — a small change that made very deep "
        "Transformers trainable",
        "The original paper put layer norm <b>after</b> the residual add "
        "($\\mathrm{LN}(x + \\mathrm{sublayer}(x))$), which needs a careful "
        "learning-rate warm-up or training diverges. Nearly every model since "
        "GPT-2 uses <b>pre-norm</b> ($x + \\mathrm{sublayer}(\\mathrm{LN}(x))$), "
        "which leaves a completely clean identity path from input to output — the "
        "same reasoning as the ResNet-v2 pre-activation ordering (§14.5). "
        "Pre-norm trains without warm-up and scales to hundreds of layers.",
    )

    derive(
        [("<b>The cost of self-attention.</b> For a sequence of length $n$ and "
          "model dimension $d$, the attention matrix is $n \\times n$:",
          r"\mathcal{O}\bigl(n^{2}d\bigr) \text{ time},\qquad "
          r"\mathcal{O}\bigl(n^{2}\bigr) \text{ memory}"),
         ("Compare a recurrent layer, $\\mathcal{O}(nd^2)$, and a convolution "
          "with kernel $k$, $\\mathcal{O}(knd^2)$.", None),
         ("So attention is <b>cheaper</b> than recurrence when $n < d$ — which is "
          "the common case for sentences ($n \\approx 30$, $d = 512$) — and "
          "<b>more expensive</b> for long documents.", None),
         ("But the decisive column is not FLOPs, it is <b>sequential "
          "operations</b>:",
          r"\text{recurrent: } \mathcal{O}(n) \qquad\text{vs}\qquad "
          r"\text{attention: } \mathcal{O}(1)"),
         ("A GPU executes independent operations simultaneously. An RNN offers it "
          "one time step at a time; attention offers it the whole sequence at "
          "once. <b>That is why the Transformer won.</b>", None),
         ("The $n^2$ memory is the architecture's real limit, and the reason for "
          "the whole efficient-attention literature: sparse attention "
          "(Longformer, BigBird), low-rank attention (Linformer), and "
          "IO-aware exact attention (FlashAttention), which does not change the "
          "$\\mathcal{O}(n^2)$ compute but avoids ever materialising the matrix.",
          None)],
        title="Why the Transformer replaced the RNN",
    )

    anim_header("Positional encodings, and the rotation that encodes offset")

    d_model, n_pos = 64, 60
    pos = np.arange(n_pos)[:, None]
    ii = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (ii // 2)) / d_model)
    PE = np.zeros((n_pos, d_model))
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])

    frames = []
    for k in range(1, n_pos + 1):
        Z = np.full((n_pos, d_model), np.nan)
        Z[:k] = PE[:k]
        sim = PE[:k] @ PE[k - 1] / d_model * 2
        frames.append(go.Frame(name=str(k), data=[
            go.Heatmap(z=Z, colorscale=nav.cscale(), zmin=-1, zmax=1,
                       showscale=False),
            go.Scatter(x=np.arange(k), y=sim * 8 + 30, mode="lines",
                       line=dict(color=C["danger"], width=2.5), xaxis="x2",
                       yaxis="y2"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"position {k-1}   ·   similarity to itself = "
            f"{sim[-1]/2*d_model/d_model*2:.2f}   ·   similarity decays "
            f"smoothly with distance, so 'nearby' is expressible")])))

    _P0 = np.full((n_pos, d_model), np.nan)
    _P0[0] = PE[0]
    f = go.Figure(data=[go.Heatmap(z=_P0,
                                   colorscale=nav.cscale(), zmin=-1, zmax=1,
                                   colorbar=dict(title="value"))])
    f.update_layout(height=460, xaxis_title="embedding dimension",
                    yaxis_title="position in the sequence",
                    title="Sinusoidal positional encoding "
                          "(low dimensions oscillate fast, high ones slowly)")
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="pos ")
    figure(f, "Each column is a sinusoid; wavelengths run geometrically from "
              "2π to ~10⁴·2π, so together they pin down position at every scale.")

    code_lab(
        "A Transformer built from scratch, verified against Keras",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

# ============ 1. SELF-ATTENTION IS PERMUTATION-EQUIVARIANT =============
print("=== self-attention cannot see word order ===")
rng = np.random.default_rng(0)
n, d = 5, 8
Xseq = rng.normal(0, 1, (n, d))
Wq, Wk, Wv = (rng.normal(0, .4, (d, d)) for _ in range(3))

def self_attn(X, Wq, Wk, Wv, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    e = Q @ K.T / np.sqrt(K.shape[-1])
    if mask is not None:
        e = np.where(mask, e, -1e9)
    A = softmax(e)
    return A @ V, A

out, _ = self_attn(Xseq, Wq, Wk, Wv)
perm = np.array([3, 0, 4, 1, 2])
out_p, _ = self_attn(Xseq[perm], Wq, Wk, Wv)
print(f"  attend(X)[perm] == attend(X[perm]) : "
      f"{np.allclose(out[perm], out_p)}")
print("  the output is PERMUTED, not CHANGED. 'dog bites man' and")
print("  'man bites dog' are the same input. Hence positional encoding.")

# ============ 2. POSITIONAL ENCODING ===================================
def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

PE = positional_encoding(50, 64)
print()
print("=== positional encoding ===")
print(f"  shape {PE.shape}, values in [{PE.min():.2f}, {PE.max():.2f}]")
print(f"  every row has the same norm: "
      f"{np.allclose(np.linalg.norm(PE, axis=1), np.linalg.norm(PE[0])) }")

# --- the rotation property -------------------------------------------
print()
print("=== relative position is a ROTATION, independent of absolute position ===")
d_pair, i_pair = 64, 3
w = 1 / 10000 ** (2*i_pair/d_pair)
k_off = 5
R = np.array([[np.cos(w*k_off),  np.sin(w*k_off)],
              [-np.sin(w*k_off), np.cos(w*k_off)]])
print(f"  frequency index {i_pair}, offset k={k_off}")
print(f"{'position p':>12}{'  R @ pe[p]':>26}{'pe[p+k]':>26}{'match':>8}")
for p in [0, 7, 23, 41]:
    v = np.array([PE[p, 2*i_pair], PE[p, 2*i_pair+1]])
    got = R @ v
    want = np.array([PE[p+k_off, 2*i_pair], PE[p+k_off, 2*i_pair+1]])
    print(f"{p:>12}{str(np.round(got, 4)):>26}{str(np.round(want, 4)):>26}"
          f"{str(np.allclose(got, want, atol=1e-6)):>8}")
print("  ONE rotation matrix works at EVERY absolute position.")
print("  that is what lets a linear layer learn 'attend 5 tokens back'.")

# ============ 3. MULTI-HEAD ATTENTION FROM SCRATCH =====================
print()
print("=== multi-head attention, verified against Keras ===")
D_MODEL, N_HEADS, T = 32, 4, 7
D_HEAD = D_MODEL // N_HEADS
mha = keras.layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_HEAD)
xin = tf.constant(rng.normal(0, 1, (2, T, D_MODEL)).astype("float32"))
_ = mha(xin, xin)                                     # build
wq, bq, wk, bk, wv, bv, wo, bo = [w.numpy() for w in mha.weights]
print(f"  Keras weight shapes: Wq {wq.shape} (d_model, heads, d_head), "
      f"Wo {wo.shape}")

def multi_head(X, wq, bq, wk, bk, wv, bv, wo, bo, mask=None):
    """X: (B, T, d_model)."""
    Q = np.einsum("btd,dhk->bhtk", X, wq) + bq[None, :, None, :].transpose(0,1,2,3)
    K = np.einsum("btd,dhk->bhtk", X, wk) + bk[None, :, None, :]
    V = np.einsum("btd,dhv->bhtv", X, wv) + bv[None, :, None, :]
    e = np.einsum("bhtk,bhsk->bhts", Q, K) / np.sqrt(K.shape[-1])
    if mask is not None:
        e = np.where(mask, e, -1e9)
    A = softmax(e)
    ctx = np.einsum("bhts,bhsv->bhtv", A, V)
    out = np.einsum("bhtv,hvd->btd", ctx, wo) + bo
    return out, A

mine, A_mine = multi_head(xin.numpy(), wq, bq, wk, bk, wv, bv, wo, bo)
theirs = mha(xin, xin).numpy()
print(f"  max |mine - Keras| = {np.abs(mine - theirs).max():.2e}")
print(f"  attention weights shape {A_mine.shape} "
      f"(batch, heads, queries, keys)")
print(f"  each head sees d_head = {D_HEAD}, so h heads cost the SAME as one")
print(f"  full-width head: {N_HEADS} x {D_HEAD} = {D_MODEL}")

# --- causal masking ---------------------------------------------------
causal = np.tril(np.ones((T, T), dtype=bool))[None, None]
_, A_causal = multi_head(xin.numpy(), wq, bq, wk, bk, wv, bv, wo, bo,
                         mask=causal)
print()
print(f"  causal mask: weight above the diagonal = "
      f"{A_causal[0,0][np.triu_indices(T, 1)].max():.2e}")
k_causal = mha(xin, xin, use_causal_mask=True).numpy()
_, A_k = multi_head(xin.numpy(), wq, bq, wk, bk, wv, bv, wo, bo, mask=causal)
print(f"  matches Keras use_causal_mask=True: "
      f"{np.allclose(k_causal, multi_head(xin.numpy(), wq, bq, wk, bk, wv, bv, wo, bo, mask=causal)[0], atol=1e-4)}")

# ============ 4. A COMPLETE TRANSFORMER BLOCK ==========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class TransformerBlock(keras.layers.Layer):
    """PRE-norm block: x + sublayer(LN(x)) -- trains without warm-up."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, causal=False,
                 **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads = d_model, n_heads
        self.d_ff, self.causal = d_ff, causal
        self.att = keras.layers.MultiHeadAttention(n_heads,
                                                   d_model // n_heads,
                                                   dropout=dropout)
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()
        self.ff = keras.Sequential([keras.layers.Dense(d_ff, activation="gelu"),
                                    keras.layers.Dense(d_model)])
        self.drop = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.ln1(x)
        h = self.att(h, h, use_causal_mask=self.causal, training=training)
        x = x + self.drop(h, training=training)          # SKIP
        h = self.ff(self.ln2(x))
        return x + self.drop(h, training=training)       # SKIP

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model,
                "n_heads": self.n_heads, "d_ff": self.d_ff,
                "causal": self.causal}

blk = TransformerBlock(64, 4, 256)
o = blk(tf.zeros((2, 10, 64)))
print()
print("=== one Transformer block ===")
print(f"  input (2,10,64) -> output {tuple(o.shape)}")
print(f"  parameters {blk.count_params():,}")
att_p = 4 * 64 * 64 + 4*64
ff_p = 64*256 + 256 + 256*64 + 64
print(f"    attention  ~{att_p:>7,}  ({att_p/blk.count_params():.0%})")
print(f"    feed-fwd   ~{ff_p:>7,}  ({ff_p/blk.count_params():.0%})  "
      f"<- most of the parameters are HERE, not in the attention")

# ============ 5. COST: n^2 d vs n d^2 ==================================
print()
print("=== attention vs recurrence, cost per layer ===")
print(f"{'n':>7}{'d':>7}{'attention n^2 d':>18}{'recurrent n d^2':>18}"
      f"{'seq. ops (att/rnn)':>21}")
for n_, d_ in [(16, 512), (64, 512), (512, 512), (4096, 512)]:
    print(f"{n_:>7}{d_:>7}{n_*n_*d_:>18,}{n_*d_*d_:>18,}"
          f"{f'1 / {n_}':>21}")
print("  attention is CHEAPER than an RNN while n < d,")
print("  and always has O(1) sequential steps instead of O(n).")
print("  the n^2 MEMORY is the real limit -- hence FlashAttention etc.")

# ============ 6. A TRANSFORMER ON THE DATE TASK ========================
print()
print("="*66)
print("Transformer vs GRU encoder-decoder, same task")
print("="*66)
src, tgt = _ds.date_pairs(9000)
PAD, SOS, EOS = 0, 1, 2
i_stoi = {c: i+1 for i, c in enumerate(sorted(set("".join(src))))}
o_stoi = {c: i+3 for i, c in enumerate(sorted(set("".join(tgt))))}
o_itos = {i: c for c, i in o_stoi.items()}
T_IN = max(len(s) for s in src); T_OUT = len(tgt[0]) + 1
V_IN, V_OUT = len(i_stoi)+1, len(o_stoi)+3
X = np.zeros((len(src), T_IN), dtype="int32")
for i, s_ in enumerate(src):
    X[i, :len(s_)] = [i_stoi[c] for c in s_]
Yf = np.array([[o_stoi[c] for c in t] + [EOS] for t in tgt], dtype="int32")
Din = np.concatenate([np.full((len(Yf),1), SOS, dtype="int32"), Yf[:,:-1]], 1)
n_tr = 8000

D_M, H_, FF = 64, 4, 128

@keras.utils.register_keras_serializable(package="MLPlatform")
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab, d_model, max_len, **kw):
        super().__init__(**kw)
        self.vocab, self.d_model, self.max_len = vocab, d_model, max_len
        self.emb = keras.layers.Embedding(vocab, d_model)
        self.pe = tf.constant(positional_encoding(max_len, d_model),
                              dtype="float32")

    def call(self, x):
        n = tf.shape(x)[1]
        return self.emb(x) * tf.sqrt(tf.cast(self.d_model, "float32")) \\
               + self.pe[:n]

    def get_config(self):
        return {**super().get_config(), "vocab": self.vocab,
                "d_model": self.d_model, "max_len": self.max_len}

def transformer():
    ei = keras.layers.Input(shape=(T_IN,), dtype="int32")
    di = keras.layers.Input(shape=(T_OUT,), dtype="int32")
    z = PositionalEmbedding(V_IN, D_M, T_IN)(ei)
    for _ in range(2):
        z = TransformerBlock(D_M, H_, FF)(z)              # ENCODER
    y = PositionalEmbedding(V_OUT, D_M, T_OUT)(di)
    for _ in range(2):
        y = TransformerBlock(D_M, H_, FF, causal=True)(y)  # causal SELF-att
        y = keras.layers.LayerNormalization()(
            y + keras.layers.MultiHeadAttention(H_, D_M//H_)(y, z))  # CROSS-att
    return keras.Model([ei, di], keras.layers.Dense(V_OUT)(y))

def gru_baseline():
    ei = keras.layers.Input(shape=(T_IN,), dtype="int32")
    di = keras.layers.Input(shape=(T_OUT,), dtype="int32")
    ee = keras.layers.Embedding(V_IN, D_M, mask_zero=True)(ei)
    es, eh = keras.layers.GRU(D_M*2, return_sequences=True,
                              return_state=True)(ee)
    de = keras.layers.Embedding(V_OUT, D_M)(di)
    ds_ = keras.layers.GRU(D_M*2, return_sequences=True)(de, initial_state=eh)
    ctx = keras.layers.Attention(use_scale=True)([ds_, es])
    return keras.Model([ei, di],
                       keras.layers.Dense(V_OUT)(
                           keras.layers.Concatenate()([ds_, ctx])))

print(f"{'model':<24}{'params':>10}{'fit time':>11}{'valid char acc':>17}")
for nm, make in [("GRU + attention", gru_baseline), ("Transformer", transformer)]:
    tf.random.set_seed(0)
    m = make()
    m.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(2e-3), metrics=["accuracy"])
    t0 = time.perf_counter()
    m.fit([X[:n_tr], Din[:n_tr]], Yf[:n_tr], epochs=8, batch_size=128,
          verbose=0)
    dt = time.perf_counter() - t0
    acc = m.evaluate([X[n_tr:], Din[n_tr:]], Yf[n_tr:], verbose=0,
                     return_dict=True)["accuracy"]
    print(f"{nm:<24}{m.count_params():>10,}{dt:>10.1f}s{acc:>17.4f}")
    if nm == "Transformer":
        trm = m

# ============ 7. GENERATE WITH THE TRANSFORMER =========================
print()
print("=== free-running generation ===")
tests = ["March 3, 2019", "7 July 1985", "Dec 25 2021", "01/02/1999"]
xs = np.zeros((len(tests), T_IN), dtype="int32")
for i, s_ in enumerate(tests):
    xs[i, :len(s_)] = [i_stoi[c] for c in s_]
dec = np.full((len(tests), T_OUT), PAD, dtype="int32"); dec[:, 0] = SOS
for t in range(T_OUT-1):
    dec[:, t+1] = trm.predict([xs, dec], verbose=0)[:, t].argmax(-1)
for s_, row in zip(tests, dec[:, 1:]):
    out = ""
    for i in row:
        if i == EOS: break
        out += o_itos.get(int(i), "?")
    print(f"  {s_:<22} -> {out}")

import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(z=PE[:40], colorscale="RdBu", zmid=0))
fig.update_layout(height=420, xaxis_title="embedding dimension",
                  yaxis_title="position",
                  title="Sinusoidal positional encoding")
''',
        key="ch16_transformer",
    )

    keypoints([
        "<b>Self-attention</b>: $\\mathbf{Q},\\mathbf{K},\\mathbf{V}$ are all "
        "projections of the same sequence.",
        "<b>Multi-head</b> runs $h$ attentions in $d/h$ dimensions — $h$ "
        "patterns at no extra cost.",
        "Self-attention is <b>permutation-equivariant</b>, so positional "
        "encodings are mandatory.",
        "Sinusoids make relative offset a <b>rotation independent of absolute "
        "position</b>.",
        "Cost $\\mathcal{O}(n^2 d)$ but <b>$\\mathcal{O}(1)$ sequential "
        "steps</b> — that is why it replaced the RNN.",
    ])


# ==========================================================================
def s_16_8():
    section("16.8", "The Transformer Zoo and Vision Transformers")

    lead(
        "One architecture, three ways of using it — and then the discovery that "
        "it was never really about language at all."
    )

    sub("Three families")

    table(
        ["Family", "Architecture", "Pretraining objective", "Good at"],
        [["<b>Encoder-only</b> (BERT, RoBERTa, DeBERTa)",
          "Bidirectional self-attention",
          "<b>Masked language modelling</b>: hide 15 % of tokens, predict them",
          "Classification, tagging, retrieval, question answering"],
         ["<b>Decoder-only</b> (GPT, LLaMA, Mistral)",
          "<b>Causal</b> self-attention",
          "<b>Next-token prediction</b> — §16.1's objective at scale",
          "Generation, and (at scale) everything else"],
         ["<b>Encoder–decoder</b> (T5, BART)",
          "Bidirectional encoder + causal decoder",
          "Span corruption / denoising",
          "Translation, summarisation, any seq→seq task"]],
    )

    derive(
        [("<b>Why masked language modelling requires bidirectionality — and why "
          "that forbids generation.</b> BERT's objective is:",
          r"\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} "
          r"\log p_\theta\bigl(x_i \mid \mathbf{x}_{\setminus \mathcal{M}}\bigr)"),
         ("The context $\\mathbf{x}_{\\setminus\\mathcal{M}}$ includes tokens on "
          "<b>both sides</b> of position $i$. That is a far richer conditioning "
          "signal than a left context alone, which is why BERT-style encoders "
          "dominate classification.", None),
         ("But it means the model has never learned "
          "$p(x_i \\mid \\mathbf{x}_{<i})$, so it cannot generate "
          "autoregressively. Same distinction as bidirectional RNNs (§16.5).",
          None),
         ("<b>The causal objective is weaker per token but composes.</b> GPT's "
          "objective is the §16.1 chain rule:",
          r"\mathcal{L}_{\text{LM}} = -\sum_{i} \log p_\theta"
          r"\bigl(x_i \mid \mathbf{x}_{<i}\bigr)"),
         ("Every token supplies a training signal (BERT only gets one from the "
          "masked 15 %), and the resulting model is a complete generative model "
          "of the sequence — which turned out to matter far more than the "
          "per-token disadvantage. That empirical fact is most of why the field "
          "converged on decoder-only.", None)],
        title="Masked vs causal pretraining",
    )

    sub("Scaling laws")

    md(
        "Kaplan et al. (2020) and Hoffmann et al. (2022, *Chinchilla*) found that "
        "test loss follows a **power law** in model size $N$, data $D$ and "
        "compute $C$:"
    )

    math(r"""
    L(N, D) \;\approx\; E \;+\; \frac{A}{N^{\alpha}} \;+\; \frac{B}{D^{\beta}}
    """)

    idea(
        "Chinchilla's correction: most large models were badly undertrained",
        "Kaplan's scaling laws were read as 'make the model bigger'. Hoffmann et "
        "al. re-ran the experiment with the learning-rate schedule matched to the "
        "token budget and found $\\alpha \\approx \\beta$ — meaning parameters "
        "and tokens should scale <b>together</b>, roughly 20 tokens per "
        "parameter. A 70 B model trained on 1.4 T tokens beat a 280 B model "
        "trained on 300 B tokens, at a quarter of the inference cost. The lesson "
        "generalises well beyond LLMs: <b>a compute budget has an optimal split "
        "between model size and data, and intuition gets it wrong.</b>",
    )

    sub("Vision Transformers")

    md(
        "Dosovitskiy et al. (2020) applied the Transformer to images by the "
        "simplest possible route: **cut the image into 16×16 patches, flatten "
        "each one, and treat the sequence of patches as tokens.**"
    )

    math(r"""
    \mathbf{z}_0 = \bigl[\mathbf{x}_{\text{class}};\;
      \mathbf{x}^1_p\mathbf{E};\; \mathbf{x}^2_p\mathbf{E};\;\dots;\;
      \mathbf{x}^N_p\mathbf{E}\bigr] + \mathbf{E}_{\text{pos}},
    \qquad
    N = \frac{HW}{P^{2}}
    """)

    table(
        ["", "CNN", "Vision Transformer"],
        [["Inductive bias", "<b>Strong</b>: locality, translation equivariance",
          "<b>Almost none</b> — must learn locality from data"],
         ["Small data (< 1M images)", "<b>Wins clearly</b>",
          "Underperforms — the bias it lacks would have helped"],
         ["Large data (300M+)", "Plateaus",
          "<b>Wins</b> — the freedom pays off once there is enough data"],
         ["Receptive field at layer 1", "$k \\times k$",
          "<b>The whole image</b>"],
         ["Cost", "$\\mathcal{O}(HW)$", "$\\mathcal{O}(N^2)$ in patches"]],
    )

    proof(
        "The ViT lesson: inductive bias is a substitute for data, not a free win",
        "A CNN's convolution is a hard constraint that the right answer is local "
        "and translation-equivariant. That constraint is <i>approximately</i> "
        "true of images, so it saves enormous amounts of data — and costs you "
        "whenever it is wrong. A ViT has to learn locality, which needs far more "
        "data, but then it is free to learn <b>non-local</b> relations a CNN "
        "structurally cannot. Below ~100M images, the bias wins; above it, the "
        "freedom does. This is the bias–variance trade-off of §4.4, restated at "
        "the level of architecture.",
    )

    note(
        "Hybrids won in practice",
        "Pure ViTs need enormous pretraining. Swin Transformers reintroduce "
        "locality with windowed attention and a hierarchy; ConvNeXt takes a "
        "ResNet and adopts the Transformer's <i>training recipe</i> (AdamW, heavy "
        "augmentation, LayerNorm, GELU, large kernels) to match ViT accuracy with "
        "no attention at all. That last result is worth sitting with: a "
        "substantial part of the 'Transformers beat CNNs' gap was the training "
        "recipe, not the architecture.",
    )

    anim_header("How a Vision Transformer sees an image")

    img = np.zeros((32, 32))
    yy, xx = np.mgrid[0:32, 0:32]
    img += np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / 30)
    img += 0.8 * np.exp(-((xx - 23) ** 2 + (yy - 21) ** 2) / 45)
    img += 0.35 * ((xx + yy) % 11 < 3)
    P = 8
    n_side = 32 // P

    frames = []
    for k in range(n_side * n_side + 1):
        shapes = []
        for idx in range(k):
            r, c = divmod(idx, n_side)
            shapes.append(go.Scatter(
                x=[c*P-.5, c*P+P-.5, c*P+P-.5, c*P-.5, c*P-.5],
                y=[r*P-.5, r*P-.5, r*P+P-.5, r*P+P-.5, r*P-.5],
                mode="lines", line=dict(color=C["danger"], width=3),
                showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(k), data=[
            go.Heatmap(z=img, colorscale=nav.cscale(), showscale=False)
        ] + shapes, layout=go.Layout(annotations=[anim.annotate_step(
            f"{k} of {n_side*n_side} patches   ·   each {P}×{P} patch flattens "
            f"to a {P*P}-vector, then a linear layer maps it to d_model   ·   "
            f"sequence length = {n_side*n_side}, plus one [CLS] token")])))

    f = go.Figure(data=[go.Heatmap(z=img, colorscale=nav.cscale(),
                                   showscale=False)])
    f.update_layout(height=430, xaxis=dict(visible=False),
                    yaxis=dict(visible=False, autorange="reversed",
                               scaleanchor="x"),
                    title=f"A 32×32 image as {n_side*n_side} tokens of {P}×{P}")
    anim.animate(f, frames, duration=nav.anim_ms(230), slider_prefix="patch ")
    figure(f, "That is the entire idea. Everything else in a ViT is the "
              "Transformer of §16.7, unchanged.")

    code_lab(
        "Causal vs masked objectives, and a Vision Transformer from scratch",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE TWO PRETRAINING OBJECTIVES ========================
print("=== masked LM vs causal LM, on the same corpus ===")
text = _ds.char_corpus(40_000)
vocab = sorted(set(text)); V = len(vocab)
stoi = {c: i+2 for i, c in enumerate(vocab)}          # 0=PAD 1=MASK
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype="int32")
L = 48
n = (len(data)-1) // L
seqs = data[:n*L].reshape(n, L)
n_tr = int(.9*n)

D_M, H_, FF = 64, 4, 128

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, None]; i = np.arange(d_model)[None, :]
    ang = pos / np.power(10000, (2*(i//2))/d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(ang[:, 0::2]); pe[:, 1::2] = np.cos(ang[:, 1::2])
    return pe.astype("float32")

PE = tf.constant(positional_encoding(L, D_M))

def block(x, causal):
    h = keras.layers.LayerNormalization()(x)
    h = keras.layers.MultiHeadAttention(H_, D_M//H_)(h, h,
                                                     use_causal_mask=causal)
    x = keras.layers.Add()([x, h])
    h = keras.layers.LayerNormalization()(x)
    h = keras.layers.Dense(FF, activation="gelu")(h)
    h = keras.layers.Dense(D_M)(h)
    return keras.layers.Add()([x, h])

def build_lm(causal, n_blocks=2):
    inp = keras.layers.Input(shape=(L,), dtype="int32")
    z = keras.layers.Embedding(V+2, D_M)(inp)
    z = keras.layers.Lambda(lambda t: t + PE)(z)
    for _ in range(n_blocks):
        z = block(z, causal)
    return keras.Model(inp, keras.layers.Dense(V+2)(z))

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# --- CAUSAL: every position is a training signal ---------------------
tf.random.set_seed(0)
gpt = build_lm(causal=True)
gpt.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(2e-3))
Xc, Yc = seqs[:, :-1], seqs[:, 1:]
Xc = np.pad(Xc, ((0,0),(0,1))); Yc = np.pad(Yc, ((0,0),(0,1)))
t0 = time.perf_counter()
h_gpt = gpt.fit(Xc[:n_tr], Yc[:n_tr], epochs=6, batch_size=64, verbose=0,
                validation_data=(Xc[n_tr:], Yc[n_tr:]))
print(f"  causal LM   : {L} signals per sequence, "
      f"valid PPL {np.exp(h_gpt.history['val_loss'][-1]):.3f}, "
      f"{time.perf_counter()-t0:.1f}s")

# --- MASKED: only 15 % of positions ----------------------------------
rng = np.random.default_rng(0)
mask = rng.random(seqs.shape) < 0.15
Xm = np.where(mask, 1, seqs).astype("int32")           # 1 = [MASK]
Ym = np.where(mask, seqs, 0).astype("int32")           # 0 elsewhere -> ignored

def masked_loss(y_true, y_pred):
    m = tf.cast(y_true != 0, "float32")
    ls = keras.losses.sparse_categorical_crossentropy(y_true, y_pred,
                                                      from_logits=True)
    return tf.reduce_sum(ls*m) / (tf.reduce_sum(m) + 1e-8)

tf.random.set_seed(0)
bert = build_lm(causal=False)
bert.compile(loss=masked_loss, optimizer=keras.optimizers.Adam(2e-3))
t0 = time.perf_counter()
h_bert = bert.fit(Xm[:n_tr], Ym[:n_tr], epochs=6, batch_size=64, verbose=0,
                  validation_data=(Xm[n_tr:], Ym[n_tr:]))
print(f"  masked LM   : {int(.15*L)} signals per sequence, "
      f"valid PPL {np.exp(h_bert.history['val_loss'][-1]):.3f}, "
      f"{time.perf_counter()-t0:.1f}s")
print(f"  the masked model sees BOTH sides -> lower loss PER PREDICTION,")
print(f"  but {1/0.15:.1f}x fewer predictions per sequence.")

# ============ 2. THE MASKED MODEL CANNOT GENERATE ======================
print()
print("=== why only the causal model can generate ===")
seed = "the king speaks in the "
ids = [stoi[c] for c in seed]
buf = np.zeros((1, L), dtype="int32"); buf[0, :len(ids)] = ids
cur = len(ids)
for _ in range(60):
    if cur >= L: break
    lg = gpt.predict(buf, verbose=0)[0, cur-1] / .8
    p = np.exp(lg - lg.max()); p /= p.sum()
    buf[0, cur] = rng.choice(len(p), p=p); cur += 1
print(f"  causal model continues: "
      f"{repr(''.join(itos.get(int(i), '?') for i in buf[0, :cur]))}")
print(f"  masked model: at position t it was ALWAYS shown t+1..L.")
print(f"  it has never estimated p(x_t | x_<t), so there is nothing to sample.")

# --- but it IS better at filling in a blank -------------------------
print()
print("=== filling in a blank (what BERT is actually for) ===")
probe = "the queen waits [] the garden"
ids = [stoi.get(c, 1) for c in probe.replace("[]", "_")]
buf = np.zeros((1, L), dtype="int32"); buf[0, :len(ids)] = ids
pos = probe.index("[]")
buf[0, pos] = 1                                        # [MASK]
for nm, m in [("masked (sees both sides)", bert),
              ("causal (sees only the left)", gpt)]:
    lg = m.predict(buf, verbose=0)[0, pos]
    top = np.argsort(-lg)[:5]
    print(f"  {nm:<30} {[itos.get(int(i), '?') for i in top]}")

# ============ 3. A VISION TRANSFORMER ==================================
print()
print("="*66)
print("A Vision Transformer on Fashion-MNIST")
print("="*66)
Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=6000, n_test=1000)
Xtr = Xtr.astype("float32"); Xte = Xte.astype("float32")
if Xtr.ndim == 3:
    Xtr = Xtr[..., None]; Xte = Xte[..., None]
print(f"  train {Xtr.shape}, test {Xte.shape}")

PATCH = 7
N_PATCH = (28//PATCH) ** 2
print(f"  patch size {PATCH} -> {N_PATCH} tokens per image")

@keras.utils.register_keras_serializable(package="MLPlatform")
class PatchEmbed(keras.layers.Layer):
    """Cut into patches and project -- which is just a strided Conv2D."""
    def __init__(self, patch, d_model, **kw):
        super().__init__(**kw)
        self.patch, self.d_model = patch, d_model
        self.proj = keras.layers.Conv2D(d_model, patch, strides=patch)

    def call(self, x):
        z = self.proj(x)                                  # (B, H/p, W/p, d)
        s = tf.shape(z)
        return tf.reshape(z, (s[0], s[1]*s[2], self.d_model))

    def get_config(self):
        return {**super().get_config(), "patch": self.patch,
                "d_model": self.d_model}

@keras.utils.register_keras_serializable(package="MLPlatform")
class AddCLS(keras.layers.Layer):
    """Prepend a single learned [CLS] token to the patch sequence.

    Written as a Layer because tf.shape() cannot be applied to a KerasTensor
    inside a Lambda -- keras.ops is the portable way to do shape work.
    """
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.d_model = d_model

    def build(self, shape):
        self.cls = self.add_weight(shape=(1, 1, self.d_model), name="cls",
                                   initializer="random_normal")

    def call(self, x):
        b = keras.ops.shape(x)[0]
        tok = keras.ops.broadcast_to(self.cls, (b, 1, self.d_model))
        return keras.ops.concatenate([tok, x], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


@keras.utils.register_keras_serializable(package="MLPlatform")
class AddPosEmbed(keras.layers.Layer):
    """A ViT uses LEARNED position embeddings, not sinusoids -- there is no
    reason to expect image patches to want a relative-position prior."""
    def __init__(self, n_pos, d_model, **kw):
        super().__init__(**kw)
        self.n_pos, self.d_model = n_pos, d_model

    def build(self, shape):
        self.pos = self.add_weight(shape=(1, self.n_pos, self.d_model),
                                   name="pos", initializer="random_normal")

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "n_pos": self.n_pos,
                "d_model": self.d_model}


def vit(d_model=64, heads=4, depth=3, ff=128):
    inp = keras.layers.Input(shape=(28, 28, 1))
    z = PatchEmbed(PATCH, d_model)(inp)
    z = AddCLS(d_model)(z)                                # prepend [CLS]
    z = AddPosEmbed(N_PATCH + 1, d_model)(z)
    for _ in range(depth):
        h = keras.layers.LayerNormalization()(z)
        h = keras.layers.MultiHeadAttention(heads, d_model//heads)(h, h)
        z = keras.layers.Add()([z, h])
        h = keras.layers.LayerNormalization()(z)
        h = keras.layers.Dense(ff, activation="gelu")(h)
        z = keras.layers.Add()([z, keras.layers.Dense(d_model)(h)])
    z = keras.layers.LayerNormalization()(z)
    z = keras.layers.Lambda(lambda t: t[:, 0])(z)         # the [CLS] token
    return keras.Model(inp, keras.layers.Dense(10, activation="softmax")(z))

def small_cnn():
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")])

print()
print(f"{'model':<22}{'params':>10}{'fit time':>11}{'test accuracy':>16}")
for nm, make in [("small CNN", small_cnn), ("ViT (3 blocks)", vit)]:
    tf.random.set_seed(0)
    m = make()
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.AdamW(2e-3), metrics=["accuracy"])
    t0 = time.perf_counter()
    m.fit(Xtr, ytr, epochs=8, batch_size=128, verbose=0)
    dt = time.perf_counter() - t0
    acc = m.evaluate(Xte, yte, verbose=0, return_dict=True)["accuracy"]
    print(f"{nm:<22}{m.count_params():>10,}{dt:>10.1f}s{acc:>16.4f}")
print()
print("  on 6 000 images the CNN's locality bias WINS -- exactly the ViT paper's")
print("  finding. The ordering reverses somewhere north of 100M images.")

# ============ 4. THE PATCH PROJECTION IS A CONVOLUTION =================
print()
print("=== 'cut into patches and project' == one strided Conv2D ===")
pe_layer = PatchEmbed(7, 16)
out = pe_layer(tf.zeros((2, 28, 28, 1)))
print(f"  PatchEmbed(7, 16) on (2,28,28,1) -> {tuple(out.shape)}")
print(f"  implemented as Conv2D(16, kernel=7, strides=7) then reshape")
print(f"  parameters {pe_layer.count_params()} = 7*7*1*16 + 16")
print("  so a ViT's first layer IS a convolution -- it is the LACK of")
print("  convolution in every layer after it that makes the difference.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=np.exp(h_gpt.history["val_loss"]), mode="lines",
                name="causal LM (all positions)",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(y=np.exp(h_bert.history["val_loss"]), mode="lines",
                name="masked LM (15 % of positions)",
                line=dict(color=C["accent"], width=3))
fig.update_layout(height=380, xaxis_title="epoch", yaxis_title="perplexity",
                  title="Two pretraining objectives on the same corpus")
''',
        key="ch16_zoo",
    )

    keypoints([
        "<b>Encoder-only</b> (BERT) = bidirectional + masked LM: great "
        "understanding, cannot generate.",
        "<b>Decoder-only</b> (GPT) = causal + next-token: a signal at every "
        "position, and it generates.",
        "<b>Chinchilla</b>: parameters and tokens should scale together, ~20 "
        "tokens per parameter.",
        "A <b>ViT</b> is the same Transformer with 16×16 image patches as tokens.",
        "CNNs win on small data (locality bias); ViTs win once data is plentiful "
        "(freedom).",
    ])


# ==========================================================================
def s_16_9():
    section("16.9", "Hugging Face, Practice and Exercises")

    lead(
        "You will almost never train a Transformer from scratch. The working "
        "skill is choosing a checkpoint, fine-tuning it correctly, and knowing "
        "what can go wrong."
    )

    sub("The library in four objects")

    table(
        ["Object", "What it does", "Typical call"],
        [["<code>pipeline</code>", "Task → model → answer, in one line",
          "<code>pipeline('sentiment-analysis')(texts)</code>"],
         ["<code>AutoTokenizer</code>",
          "Text ↔ token ids, with the <b>exact</b> tokenizer the checkpoint was "
          "trained with",
          "<code>AutoTokenizer.from_pretrained(name)</code>"],
         ["<code>AutoModel*</code>",
          "The weights, with a task head attached",
          "<code>TFAutoModelForSequenceClassification.from_pretrained(name, "
          "num_labels=3)</code>"],
         ["<code>Trainer</code> / <code>datasets</code>",
          "Training loop and data plumbing", "<code>Trainer(...).train()</code>"]],
    )

    pitfall(
        "The tokenizer must match the checkpoint exactly",
        "Every checkpoint has its own vocabulary, its own special tokens, and its "
        "own subword merges. Pairing <code>bert-base-uncased</code>'s weights "
        "with <code>roberta-base</code>'s tokenizer produces token ids that mean "
        "something completely different to the model — and it will not error, it "
        "will just be terrible. Always load both with the <b>same</b> "
        "<code>from_pretrained(name)</code> string.",
    )

    sub("Fine-tuning that actually works")

    table(
        ["Setting", "Recommended", "Why"],
        [["Learning rate", "<b>2e-5 to 5e-5</b>",
          "100× smaller than training from scratch — you are nudging, not "
          "learning"],
         ["Epochs", "<b>2–4</b>",
          "Large pretrained models overfit small datasets almost immediately"],
         ["Optimiser", "<b>AdamW</b> with warm-up + linear decay",
          "Decoupled weight decay (§11.7); warm-up stabilises the first steps"],
         ["Max length", "As short as the data allows",
          "Attention is $\\mathcal{O}(n^2)$ — halving the length quarters the "
          "cost"],
         ["Batch size", "As large as memory permits, with gradient accumulation",
          "Layer norm makes Transformers insensitive to batch size"],
         ["Mixed precision", "<b>On</b>", "~2× throughput, negligible accuracy "
          "cost"]],
    )

    tip(
        "Parameter-efficient fine-tuning (LoRA)",
        "Full fine-tuning updates every weight and produces a full-size copy of "
        "the model per task. <b>LoRA</b> instead freezes $\\mathbf{W}$ and learns "
        "a low-rank update $\\Delta\\mathbf{W} = \\mathbf{B}\\mathbf{A}$ with "
        "$\\mathbf{A} \\in \\mathbb{R}^{r \\times d}$, "
        "$\\mathbf{B} \\in \\mathbb{R}^{d \\times r}$, $r \\approx 8$–$64$. That "
        "is often <b>&lt; 1 % of the parameters</b>, matches full fine-tuning on "
        "most tasks, and lets you keep dozens of task adapters against one frozen "
        "base model. It is the default for anything above a few billion "
        "parameters.",
    )

    warn(
        "Evaluate on data the pretraining corpus has not seen",
        "Modern checkpoints were pretrained on much of the public internet, which "
        "includes many standard benchmarks. A suspiciously high score on a "
        "well-known dataset may be <b>contamination</b>, not competence. When the "
        "decision matters, evaluate on data created after the model's cutoff, or "
        "on data of your own.",
    )

    anim_header("What each family can and cannot do")

    tasks = ["classify", "tag tokens", "fill a blank", "generate",
             "translate", "summarise", "embed for search"]
    caps = {
        "encoder-only (BERT)": [1, 1, 1, 0, 0, 0, 1],
        "decoder-only (GPT)": [1, .6, .5, 1, 1, 1, .6],
        "encoder–decoder (T5)": [1, .8, 1, 1, 1, 1, .8],
    }
    names = list(caps)
    frames = []
    for k, nm in enumerate(names):
        v = caps[nm]
        frames.append(go.Frame(name=nm.split()[0], data=[
            go.Bar(x=tasks, y=v,
                   marker=dict(color=[C["success"] if s >= .9 else
                                      C["warning"] if s >= .5 else C["danger"]
                                      for s in v]),
                   text=["native" if s >= .9 else "workable" if s >= .5
                         else "impossible" for s in v],
                   textposition="outside"),
        ], layout=go.Layout(title=nm, annotations=[anim.annotate_step(
            "bidirectional context, no generation" if k == 0 else
            "causal context, generates; at scale it does everything" if k == 1
            else "both — the most flexible, and the most expensive")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=420, yaxis=dict(range=[0, 1.25], visible=False),
                    title=names[0])
    anim.animate(f, frames, duration=nav.anim_ms(1800), slider_prefix="family ")
    figure(f)

    code_lab(
        "Working with pretrained models — with a from-scratch fallback",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. IS THE LIBRARY AVAILABLE? =============================
try:
    import transformers
    HAS_HF = True
    print(f"=== transformers {transformers.__version__} ===")
except ImportError:
    HAS_HF = False
    print("=== transformers is NOT installed here ===")
    print("  install with:  pip install transformers")
    print("  the API below is what you would run; the rest of this lab")
    print("  reproduces the same ideas with pure Keras so it still executes.")

if HAS_HF:
    from transformers import pipeline, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print(f"  tokenizer vocab: {tok.vocab_size:,}")
    enc = tok("Attention is all you need.", return_tensors="np")
    print(f"  ids    : {enc['input_ids'][0]}")
    print(f"  tokens : {tok.convert_ids_to_tokens(enc['input_ids'][0])}")
    clf = pipeline("sentiment-analysis")
    print(f"  {clf(['a luminous, generous film', 'two hours I will never get back'])}")
else:
    print()
    print("  # the three lines you would write:")
    print("  from transformers import pipeline")
    print("  clf = pipeline('sentiment-analysis')")
    print("  clf(['a luminous film', 'two hours I will never get back'])")

# ============ 2. SUBWORD TOKENISATION, WITHOUT THE LIBRARY =============
print()
print("=== what a subword tokenizer does, reproduced ===")
import collections
from core import datasets as _ds
texts, labels = _ds.sentiment_corpus(3000)

def learn_merges(corpus, n_merges=80):
    words = collections.Counter(w for t in corpus for w in t.split())
    splits = {w: tuple(w) + ("</w>",) for w in words}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            s = splits[w]
            for i in range(len(s)-1):
                pairs[(s[i], s[i+1])] += c
        if not pairs: break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        for w in splits:
            s, out, i = splits[w], [], 0
            while i < len(s):
                if i < len(s)-1 and s[i] == a and s[i+1] == b:
                    out.append(a+b); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = tuple(out)
    return merges

merges = learn_merges(texts[:1200], 80)
def apply_merges(word, merges):
    s = tuple(word) + ("</w>",)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i < len(s)-1 and s[i] == a and s[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(s[i]); i += 1
        s = tuple(out)
    return list(s)
for w in ["gorgeous", "photography", "unpronounceable"]:
    print(f"  {w:<18} -> {apply_merges(w, merges)}")

# ============ 3. FINE-TUNING HYPERPARAMETERS, DEMONSTRATED =============
print()
print("=== why fine-tuning uses a 100x smaller learning rate ===")
MAXLEN, VOCAB = 24, 400
vec = keras.layers.TextVectorization(max_tokens=VOCAB,
                                     output_sequence_length=MAXLEN)
vec.adapt(tf.constant(texts[:2400]))
X = vec(tf.constant(texts)).numpy(); y = labels
n_tr = 2400

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, None]; i = np.arange(d_model)[None, :]
    ang = pos / np.power(10000, (2*(i//2))/d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(ang[:, 0::2]); pe[:, 1::2] = np.cos(ang[:, 1::2])
    return pe.astype("float32")
PE = tf.constant(positional_encoding(MAXLEN, 48))

def encoder(trainable=True):
    inp = keras.layers.Input(shape=(MAXLEN,), dtype="int64")
    z = keras.layers.Embedding(VOCAB, 48, trainable=trainable)(inp)
    z = keras.layers.Lambda(lambda t: t + PE)(z)
    for _ in range(2):
        h = keras.layers.LayerNormalization()(z)
        h = keras.layers.MultiHeadAttention(4, 12)(h, h)
        z = keras.layers.Add()([z, h])
        h = keras.layers.LayerNormalization()(z)
        h = keras.layers.Dense(96, activation="gelu")(h)
        z = keras.layers.Add()([z, keras.layers.Dense(48)(h)])
    return keras.Model(inp, z)

# "pretrain" it with a masked objective on unlabelled text
big, _ = _ds.sentiment_corpus(12000, seed=11)
Xb = vec(tf.constant(big)).numpy()
rng = np.random.default_rng(0)
m_ = rng.random(Xb.shape) < .15
Xin = np.where(m_, 1, Xb).astype("int32")
Yt = np.where(m_, Xb, 0).astype("int32")
def masked_loss(yt, yp):
    mk = tf.cast(yt != 0, "float32")
    ls = keras.losses.sparse_categorical_crossentropy(yt, yp, from_logits=True)
    return tf.reduce_sum(ls*mk)/(tf.reduce_sum(mk)+1e-8)

tf.random.set_seed(0)
enc = encoder()
pre = keras.Model(enc.input, keras.layers.Dense(VOCAB)(enc.output))
pre.compile(loss=masked_loss, optimizer=keras.optimizers.AdamW(2e-3))
pre.fit(Xin, Yt, epochs=6, batch_size=128, verbose=0)
pre_w = enc.get_weights()
print(f"  'pretrained' a {enc.count_params():,}-parameter encoder with a")
print(f"  masked objective on {len(big):,} unlabelled texts")

print()
print(f"{'setup':<40}{'lr':>9}{'valid accuracy':>17}")
N_LAB = 300
for nm, use_pre, lr, freeze in [
        ("from scratch",                  False, 2e-3, False),
        ("fine-tune, lr 2e-3 (TOO BIG)",  True,  2e-3, False),
        ("fine-tune, lr 5e-5",            True,  5e-5, False),
        ("frozen encoder + head, lr 2e-3", True, 2e-3, True)]:
    tf.random.set_seed(0)
    e = encoder()
    if use_pre:
        e.set_weights(pre_w)
    e.trainable = not freeze
    z = keras.layers.GlobalAveragePooling1D()(e.output)
    m = keras.Model(e.input, keras.layers.Dense(1, activation="sigmoid")(z))
    m.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.AdamW(lr), metrics=["accuracy"])
    m.fit(X[:N_LAB], y[:N_LAB], epochs=10, batch_size=32, verbose=0)
    acc = m.evaluate(X[n_tr:], y[n_tr:], verbose=0, return_dict=True)["accuracy"]
    print(f"{nm:<40}{lr:>9.0e}{acc:>17.4f}")
print()
print("  too large a learning rate destroys the pretrained weights in the")
print("  first few steps -- 'catastrophic forgetting'. That is the whole")
print("  reason fine-tuning uses 2e-5 to 5e-5.")

# ============ 4. LoRA, IMPLEMENTED ====================================
print()
print("=== LoRA: a low-rank update to a frozen weight matrix ===")
@keras.utils.register_keras_serializable(package="MLPlatform")
class LoRADense(keras.layers.Layer):
    """W frozen; learn only dW = B @ A with rank r."""
    def __init__(self, units, rank=8, alpha=16, **kw):
        super().__init__(**kw)
        self.units, self.rank, self.alpha = units, rank, alpha

    def build(self, shape):
        d_in = shape[-1]
        self.W = self.add_weight(shape=(d_in, self.units),
                                 initializer="glorot_uniform",
                                 trainable=False, name="W")     # FROZEN
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros", trainable=False, name="b")
        self.A = self.add_weight(shape=(d_in, self.rank),
                                 initializer="glorot_uniform",
                                 trainable=True, name="A")      # trained
        self.B = self.add_weight(shape=(self.rank, self.units),
                                 initializer="zeros",
                                 trainable=True, name="B")      # trained

    def call(self, x):
        return x @ self.W + self.b + (self.alpha/self.rank) * ((x @ self.A) @ self.B)

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "rank": self.rank, "alpha": self.alpha}

for d, r in [(768, 8), (768, 64), (4096, 16)]:
    l = LoRADense(d, rank=r)
    l(tf.zeros((1, d)))
    full = d*d + d
    lora = d*r + r*d
    print(f"  d={d:<6} rank={r:<4} full {full:>12,}   LoRA {lora:>10,}"
          f"   {lora/full:>7.2%} of the parameters")
print("  B is initialised to ZERO, so at step 0 the model is EXACTLY the")
print("  pretrained one -- fine-tuning starts from a guaranteed no-op.")
print()
print(f"  trainable weights in a LoRADense: "
      f"{[w.name for w in LoRADense(64)(tf.zeros((1,64))) is None or []]}")
l = LoRADense(64); l(tf.zeros((1, 32)))
print(f"    trainable     : {[w.path.split('/')[-1] for w in l.trainable_weights]}")
print(f"    NOT trainable : {[w.path.split('/')[-1] for w in l.non_trainable_weights]}")

import plotly.graph_objects as go
ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
fig = go.Figure()
for d in [768, 1024, 4096]:
    fig.add_scatter(x=ranks, y=(2*d*ranks)/(d*d)*100, mode="lines+markers",
                    name=f"d = {d}")
fig.update_layout(height=380, xaxis_type="log", yaxis_type="log",
                  xaxis_title="LoRA rank r",
                  yaxis_title="trainable parameters (% of full)",
                  title="LoRA: 2dr parameters instead of d²")
''',
        key="ch16_hf",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "What are the pros and cons of using a stateful RNN versus a stateless "
        "RNN?",
        "**Stateless RNNs** can only capture patterns whose length is less than, "
        "or equal to, the size of the windows the RNN is trained on. Conversely, "
        "**stateful RNNs** can capture longer-term patterns.\n\n"
        "However, implementing a stateful RNN is much harder — especially "
        "preparing the dataset properly. Moreover, stateful RNNs do not always "
        "work better, in part because consecutive batches are not independent and "
        "identically distributed (IID). Gradient descent is not fond of "
        "non-IID datasets.\n\n"
        "The four requirements — `stateful=True`, a fixed batch shape, "
        "consecutive non-overlapping windows with `shuffle=False`, and "
        "`reset_states()` each epoch — all fail *silently* when violated (§16.2), "
        "which is the practical reason to reach for a longer window, dilated "
        "convolutions, or attention instead.")

    exercise(
        2, "Why do people use encoder–decoder RNNs rather than plain "
        "sequence-to-sequence RNNs for automatic translation?",
        "In general, if you translate a sentence one word at a time, the result "
        "will be terrible. For example, the French sentence *«Je vous en prie»* "
        "means *\"You are welcome\"*, but if you translate it one word at a time "
        "you get *\"I you in pray\"*. It is much better to read the whole "
        "sentence first and then translate it.\n\n"
        "A plain sequence-to-sequence RNN would start translating a sentence "
        "immediately after reading the first word, whereas an encoder–decoder "
        "model will **first read the whole sentence and then translate it**. That "
        "said, one could imagine a plain sequence-to-sequence RNN that outputs "
        "silence whenever it is unsure — but in practice the encoder–decoder "
        "structure (and now attention, which lets the decoder consult every "
        "source position) is what works.")

    exercise(
        3, "How can you deal with variable-length input sequences? What about "
        "variable-length output sequences?",
        "**Variable-length input sequences** can be handled by padding the "
        "shorter sequences so that all sequences in a batch have the same length, "
        "and using **masking** to ensure the RNN ignores the padding token "
        "(§16.3). For better performance, you may also want to create batches "
        "containing sequences of similar sizes — *bucketing*, which reduces the "
        "amount of padding. RNNs can handle variable-length sequences natively; "
        "the padding is only there because tensors are rectangular.\n\n"
        "**Variable-length output sequences**: if the output length is known in "
        "advance (e.g. it equals the input length), configure the loss so that it "
        "ignores tokens past the end. Otherwise, train the model to output an "
        "**end-of-sequence token** at the end of each sequence, and at inference "
        "stop generating when that token appears (§16.4). Forgetting the "
        "`<eos>` token is a common bug — generation then never terminates.")

    exercise(
        4, "What is beam search, and why would you use it? What tool can you use "
        "to implement it?",
        "**Beam search** is a technique used to improve the performance of a "
        "trained encoder–decoder model, for example in a neural machine "
        "translation system. The algorithm keeps track of a short list of the $k$ "
        "most promising output sentences (say, the top three), and at each "
        "decoder step it tries to extend them by one word; then it keeps only the "
        "$k$ most likely sentences. The parameter $k$ is called the **beam "
        "width**.\n\n"
        "Its purpose is to address the fact that **greedy decoding maximises each "
        "step, not the sequence** (§16.5): a locally suboptimal word can open up "
        "a much better continuation, and greedy search can never recover from an "
        "early mistake because it cannot go back.\n\n"
        "Implementation notes that matter: accumulate **log-probabilities**, not "
        "probabilities (a product of 20 small numbers underflows float32), and "
        "apply **length normalisation** (divide by $T^{\\alpha}$, "
        "$\\alpha \\approx 0.7$) or the search will systematically prefer short "
        "outputs. TensorFlow Addons historically provided a `BeamSearchDecoder`; "
        "in practice a from-scratch loop of ~30 lines is clearer and is what "
        "§16.5's lab implements.")

    exercise(
        5, "What is an attention mechanism? How does it help?",
        "An **attention mechanism** is a component of a neural network that "
        "learns to weigh a set of vectors and return their weighted average. In "
        "an encoder–decoder model, it lets the decoder compute a **different "
        "context vector at every output step**:\n\n"
        "$\\mathbf{c}_{(t)} = \\sum_i \\alpha_{(t,i)} \\mathbf{h}^{\\text{enc}}_{(i)}$\n\n"
        "where the weights $\\alpha$ come from a softmax over learned alignment "
        "scores.\n\n"
        "It helps in three distinct ways:\n\n"
        "1. It **removes the fixed-size bottleneck** — the decoder no longer has "
        "to squeeze the whole source through one vector, so performance stops "
        "degrading with input length.\n"
        "2. It gives a **gradient path of length 1** between any two positions "
        "(§16.6), instead of a product over the sequence — the vanishing-gradient "
        "problem of §15.2 simply does not arise.\n"
        "3. It is **interpretable**: the alignment matrix shows which source "
        "positions produced each output token, which is a genuine debugging tool.")

    exercise(
        6, "What is the most important layer in the Transformer architecture? "
        "What is its purpose?",
        "The most important layer in the Transformer architecture is the "
        "**multi-head attention layer** (the original paper is titled *Attention "
        "Is All You Need* for a reason). It allows the model to identify which "
        "words are most aligned with each other, and then improve each word's "
        "representation using these contextual clues.\n\n"
        "Concretely: it is the **only component that moves information between "
        "positions**. The feed-forward sublayer, the layer norms and the residual "
        "connections all act on each position independently — remove attention "
        "and the Transformer degenerates into a per-token MLP that cannot see "
        "context at all.\n\n"
        "Two details make it work: the **$1/\\sqrt{d_k}$ scaling** (without which "
        "the softmax saturates and its gradient vanishes, §16.6) and the "
        "**multiple heads**, which give several attention patterns at essentially "
        "no extra cost because each head works in $d_{\\text{model}}/h$ "
        "dimensions.")

    exercise(
        7, "When would you need to use sampled softmax?",
        "**Sampled softmax** is used when training a classification model with "
        "very many classes — for example a language model with a vocabulary of "
        "50 000 words. It computes an approximation of the cross-entropy loss "
        "based on the logit predicted by the model for the correct word plus the "
        "logits for a **sample** of incorrect words. This speeds up training "
        "considerably compared with computing the softmax over all logits and "
        "then estimating the cross-entropy loss.\n\n"
        "After training, the model must be used with the **normal softmax** to "
        "compute all the class probabilities — sampled softmax is a "
        "training-time-only approximation, and using it at inference would give "
        "unnormalised, meaningless probabilities.\n\n"
        "Modern alternatives to the same problem: **adaptive softmax** (which "
        "gives frequent words a full-dimensional representation and rare words a "
        "smaller one) and simply using **subword tokenisation** (§16.3) to keep "
        "the vocabulary at 32 000–50 000 rather than millions — which is why "
        "sampled softmax is far less common now than it was.")

    exercise(
        8, "Embedded Reber grammars were used by Hochreiter and Schmidhuber in "
        "their paper about LSTMs. They are artificial grammars that produce "
        "strings such as \"BPBTSXXVPSEPE.\" Check out Jenny Orr's nice "
        "introduction to this topic, then choose a particular embedded Reber "
        "grammar (such as the one represented on Jenny Orr's page), then train an "
        "RNN to identify whether a string respects that grammar or not. You will "
        "first need to write a function capable of generating a training batch "
        "containing about 50 % strings that respect the grammar, and 50 % that "
        "don't.",
        "The point of this exercise is that an **embedded** Reber grammar "
        "contains a genuine long-range dependency: the second character of the "
        "string determines the second-to-last character, with an arbitrary amount "
        "of material in between. That is precisely the dependency a "
        "`SimpleRNN` cannot learn and an LSTM can — it is the experiment "
        "Hochreiter and Schmidhuber used to justify the LSTM.\n\n"
        "Practical notes: generate negative examples by taking a *valid* string "
        "and corrupting **one** character (a randomly generated string is trivially "
        "rejectable, and the model would learn nothing). Use a `GRU` or `LSTM` "
        "with `mask_zero=True`, and compare against a `SimpleRNN` — the gap is "
        "the whole lesson.",
        code='''DEFAULT_REBER = [
    [("B", 1)],                   # 0
    [("T", 2), ("P", 3)],         # 1
    [("S", 2), ("X", 4)],         # 2
    [("T", 3), ("V", 5)],         # 3
    [("X", 3), ("S", 6)],         # 4
    [("P", 4), ("V", 6)],         # 5
    [("E", None)]]                # 6

EMBEDDED_REBER = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [(DEFAULT_REBER, 4)],
    [(DEFAULT_REBER, 5)],
    [("T", 6)],
    [("P", 6)],
    [("E", None)]]

def generate_string(grammar, rng):
    state, out = 0, []
    while state is not None:
        i = rng.integers(len(grammar[state]))
        production, state = grammar[state][i]
        if isinstance(production, list):
            production = generate_string(production, rng)
        out.append(production)
    return "".join(out)

def generate_corrupted(grammar, chars, rng):
    """Corrupt ONE character -- a random string would be trivial."""
    good = generate_string(grammar, rng)
    i = rng.integers(len(good))
    replacement = rng.choice([c for c in chars if c != good[i]])
    return good[:i] + replacement + good[i+1:]

# the long-range dependency: string[1] determines string[-2]
model = keras.Sequential([
    keras.layers.Input(shape=[None]),
    keras.layers.Embedding(len(chars), 5, mask_zero=True),
    keras.layers.GRU(30),                    # a SimpleRNN fails here
    keras.layers.Dense(1, activation="sigmoid")])''')

    exercise(
        9, "Train an encoder–decoder model that can convert a date string from "
        "one format to another (e.g., from \"April 22, 2019\" to \"2019-04-22\").",
        "This is exactly the task built in §16.4, extended with attention in "
        "§16.6 and rebuilt as a Transformer in §16.7 — the labs on those pages "
        "are a complete worked solution.\n\n"
        "The details that matter:\n\n"
        "* The alignment is **non-monotonic** — the year appears last in the "
        "input and first in the output — which is what makes attention visibly "
        "useful and produces the off-diagonal alignment matrix in §16.6.\n"
        "* The decoder **input** is the target shifted right with `<sos>` "
        "prepended; the decoder **target** has `<eos>` appended. Getting this "
        "off-by-one wrong yields a suspiciously low loss and useless output.\n"
        "* **Exact match**, not per-character accuracy, is the honest metric: one "
        "wrong digit ruins a date.\n"
        "* Character-level is the right granularity here — the vocabulary is tiny "
        "and the task is essentially transliteration.")

    exercise(
        10, "Go through the example on the Keras website for \"Natural language "
        "image search with a Dual Encoder.\"",
        "A **dual encoder** (the CLIP architecture) trains an image encoder and a "
        "text encoder so that matching image–caption pairs land close together in "
        "a shared embedding space, using a **contrastive** loss: within a batch "
        "of $N$ pairs, the $N$ correct pairings are positives and the $N^2 - N$ "
        "mismatched ones are negatives.\n\n"
        "$\\mathcal{L} = -\\frac{1}{N}\\sum_i \\log "
        "\\frac{\\exp(\\mathbf{u}_i^\\top \\mathbf{v}_i / \\tau)}"
        "{\\sum_j \\exp(\\mathbf{u}_i^\\top \\mathbf{v}_j / \\tau)}$\n\n"
        "Note that this is **the same softmax-over-dot-products** as attention "
        "(§16.6), with the same temperature argument — here $\\tau$ is learned. "
        "The key practical fact is that a larger batch gives more negatives and "
        "therefore a better model, which is why CLIP was trained at batch size "
        "32 768.\n\n"
        "The payoff is **zero-shot** retrieval and classification: embed the "
        "class names as text and classify an image by nearest neighbour, with no "
        "task-specific training at all.")

    exercise(
        11, "Use the Hugging Face Transformers library to download a pretrained "
        "language model capable of generating text (e.g., GPT), and try "
        "generating more convincing Shakespearean text. You will need to use the "
        "model's `generate()` method.",
        "```python\n"
        "from transformers import pipeline, set_seed\n"
        "gen = pipeline('text-generation', model='gpt2')\n"
        "set_seed(42)\n"
        "gen('To be or not to be, that is', max_new_tokens=60,\n"
        "    num_return_sequences=3, do_sample=True, temperature=0.8, top_p=0.9)\n"
        "```\n\n"
        "The generation arguments map directly onto §16.1:\n\n"
        "* `do_sample=False` is **greedy** — it will loop. This is the default in "
        "some versions and it is why a first attempt often looks broken.\n"
        "* `temperature` is $p_i \\propto p_i^{1/T}$.\n"
        "* `top_k` and `top_p` are the truncation strategies of §16.1.\n"
        "* `num_beams` switches to beam search (§16.5) — good for translation, "
        "poor for open-ended text.\n"
        "* `repetition_penalty` and `no_repeat_ngram_size` are blunt fixes for "
        "the same degeneracy that sampling addresses properly.\n\n"
        "To get genuinely Shakespearean output rather than GPT-2's default "
        "register, **fine-tune** on a Shakespeare corpus at learning rate "
        "$5\\times10^{-5}$ for 2–3 epochs, or use LoRA (§16.9) if the model is "
        "large.")

    rule()

    sub("The chapter as a decision table")

    table(
        ["Task", "Use", "Why"],
        [["Text classification / tagging", "Fine-tune a BERT-family encoder",
          "Bidirectional context, and a small head is enough"],
         ["Open-ended generation", "A decoder-only LM, <b>sampled</b>",
          "Greedy and beam both degenerate on open-ended text"],
         ["Translation / summarisation",
          "Encoder–decoder, small beam ($k \\approx 4$)",
          "Constrained output where the mode is a good answer"],
         ["Semantic search", "A dual encoder, cosine similarity",
          "Embeddings are cheap to index"],
         ["Very long documents",
          "Sparse / windowed attention, or retrieval",
          "Attention is $\\mathcal{O}(n^2)$ in memory"],
         ["Few labels", "Frozen pretrained encoder + small head",
          "Fine-tuning overfits instantly on small data"],
         ["A large model, many tasks", "<b>LoRA</b> adapters",
          "&lt; 1 % of parameters, one frozen base"]],
    )

    keypoints([
        "Choose the family by the task: <b>encoder</b> to understand, "
        "<b>decoder</b> to generate, <b>both</b> to transform.",
        "The tokenizer must come from the <b>same checkpoint</b> as the weights.",
        "Fine-tune at <b>2e-5 – 5e-5</b> for <b>2–4 epochs</b> — a large learning "
        "rate destroys the pretrained weights.",
        "<b>LoRA</b> trains &lt; 1 % of the parameters and matches full "
        "fine-tuning on most tasks.",
        "Attention's $\\mathcal{O}(n^2)$ memory is the real constraint on "
        "context length.",
    ], title="Chapter 16 in five lines")

    refs([
        ("Bahdanau, Cho & Bengio — *Neural Machine Translation by Jointly "
         "Learning to Align and Translate*", "https://arxiv.org/abs/1409.0473"),
        ("Luong, Pham & Manning — *Effective Approaches to Attention-based NMT*",
         "https://arxiv.org/abs/1508.04025"),
        ("Vaswani et al. — *Attention Is All You Need*",
         "https://arxiv.org/abs/1706.03762"),
        ("Devlin et al. — *BERT: Pre-training of Deep Bidirectional Transformers*",
         "https://arxiv.org/abs/1810.04805"),
        ("Radford et al. — *Language Models are Unsupervised Multitask Learners* "
         "(GPT-2)",
         "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
        ("Dosovitskiy et al. — *An Image is Worth 16x16 Words* (ViT)",
         "https://arxiv.org/abs/2010.11929"),
        ("Hoffmann et al. — *Training Compute-Optimal Large Language Models* "
         "(Chinchilla)", "https://arxiv.org/abs/2203.15556"),
        ("Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models*",
         "https://arxiv.org/abs/2106.09685"),
        ("Holtzman et al. — *The Curious Case of Neural Text Degeneration* "
         "(nucleus sampling)", "https://arxiv.org/abs/1904.09751"),
        ("Alammar — *The Illustrated Transformer*",
         "https://jalammar.github.io/illustrated-transformer/"),
    ])


# ==========================================================================
SECTIONS = [
    ("16.1", "Character RNNs & Text Generation", s_16_1),
    ("16.2", "Stateful RNNs", s_16_2),
    ("16.3", "Sentiment, Tokenisation & Masking", s_16_3),
    ("16.4", "Encoder–Decoder for Translation", s_16_4),
    ("16.5", "Bidirectional RNNs & Beam Search", s_16_5),
    ("16.6", "Attention Mechanisms", s_16_6),
    ("16.7", "The Transformer", s_16_7),
    ("16.8", "The Transformer Zoo & ViT", s_16_8),
    ("16.9", "Hugging Face & Exercises", s_16_9),
]

nav.render_chapter(CH, SECTIONS)
