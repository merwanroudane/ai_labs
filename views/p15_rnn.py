"""Chapter 15 — Processing Sequences Using RNNs and CNNs."""

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
CH = "ch15"

hero(
    kicker="Part II · Chapter 15",
    title="Processing Sequences Using RNNs and CNNs",
    blurb=(
        "A recurrent neuron feeds its own output back as input, giving the network "
        "a memory. This chapter derives backpropagation through time and its "
        "$\\gamma^T$ gradient problem, builds forecasting models from a naive "
        "baseline up through ARMA, deep RNNs and seq2seq, then shows how LSTM, "
        "GRU and dilated 1-D convolutions each attack the long-sequence problem "
        "differently."
    ),
    chips=["BPTT derived", "9 sub-sections", "9 animations",
           "9 code labs", "LSTM · GRU · WaveNet"],
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
    st.warning(
        "**TensorFlow is not installed in this environment**, so the code "
        "labs on this page cannot run — the lecture, the mathematics and "
        "the animation all work normally. This is expected on the hosted "
        "demo, where TensorFlow needs more memory than the free tier "
        "allows. To run the labs, clone "
        "[the repository](https://github.com/merwanroudane/ai_labs) and "
        "install with `pip install -r requirements-local.txt`.",
        icon="⚠️")


# ==========================================================================
def s_15_1():
    section("15.1", "Recurrent Neurons and Layers")

    lead(
        "Take a neuron and add one connection: its own output, delayed by one "
        "step, fed back as an extra input. That single loop is the whole "
        "difference between a feedforward network and a recurrent one."
    )

    sub("A recurrent neuron")

    math(r"""
    \hat{\mathbf{y}}_{(t)} \;=\;
    \phi\Bigl(
      \mathbf{W}_x^\top \mathbf{x}_{(t)}
      \;+\; \mathbf{W}_{\hat y}^\top \hat{\mathbf{y}}_{(t-1)}
      \;+\; \mathbf{b}
    \Bigr)
    """)
    where({
        r"\mathbf{x}_{(t)}": "the input at time step $t$",
        r"\hat{\mathbf{y}}_{(t-1)}": "the layer's own output at the previous step",
        r"\mathbf{W}_x": "weights for the current input — shape "
                         "$(n_{\\text{inputs}}, n_{\\text{neurons}})$",
        r"\mathbf{W}_{\hat y}": "weights for the previous output — shape "
                                "$(n_{\\text{neurons}}, n_{\\text{neurons}})$",
        r"\phi": "the activation, usually $\\tanh$",
    })

    md("Over a whole mini-batch, both weight matrices can be stacked:")

    math(r"""
    \hat{\mathbf{Y}}_{(t)} \;=\;
    \phi\Bigl(
      \bigl[\, \mathbf{X}_{(t)} \;\; \hat{\mathbf{Y}}_{(t-1)} \,\bigr]\;
      \mathbf{W} \;+\; \mathbf{b}
    \Bigr),
    \qquad
    \mathbf{W} = \begin{bmatrix}\mathbf{W}_x \\ \mathbf{W}_{\hat y}\end{bmatrix}
    """)

    idea(
        "The weights are shared across time — that is the whole trick",
        "$\\mathbf{W}_x$ and $\\mathbf{W}_{\\hat y}$ are the <b>same matrices at "
        "every time step</b>. This is exactly the weight sharing of a "
        "convolutional layer (§14.1), applied along the time axis instead of the "
        "spatial axes. It is what lets one RNN handle sequences of <b>any "
        "length</b> with a fixed parameter count — and it is also what makes the "
        "gradient a product of $T$ identical factors, which is §15.2's problem.",
    )

    sub("Memory cells and state")

    md(
        "The part of a recurrent network that preserves information across steps "
        "is a **memory cell**. Its **state** $\\mathbf{h}_{(t)}$ is a function of "
        "the current input and the previous state:"
    )

    math(r"""
    \mathbf{h}_{(t)} \;=\; f\bigl(\mathbf{h}_{(t-1)},\, \mathbf{x}_{(t)}\bigr)
    \qquad\qquad
    \hat{\mathbf{y}}_{(t)} \;=\; g\bigl(\mathbf{h}_{(t)}\bigr)
    """)

    md(
        "For a basic RNN cell, $\\mathbf{h}_{(t)} = \\hat{\\mathbf{y}}_{(t)}$ — "
        "the state *is* the output. For LSTM and GRU (§15.8) they differ, and "
        "that separation is precisely what makes them work."
    )

    sub("The four sequence shapes")

    table(
        ["Type", "Input", "Output", "Example"],
        [["<b>Sequence-to-sequence</b>", "a sequence", "a sequence",
          "Forecasting: feed the last $N$ days, predict the next $N$ shifted by "
          "one"],
         ["<b>Sequence-to-vector</b>", "a sequence", "a single value",
          "Sentiment analysis: read a review, output one score"],
         ["<b>Vector-to-sequence</b>", "a single value (repeated)", "a sequence",
          "Image captioning: one CNN embedding → a sentence"],
         ["<b>Encoder–decoder</b>", "a sequence → vector → a sequence",
          "a sequence", "Translation — §15.6"]],
    )

    codenote(
        "return_sequences decides which one you get",
        "<code>SimpleRNN(20)</code> returns only the <b>last</b> output: shape "
        "$(batch, 20)$ — that is sequence-to-vector. "
        "<code>SimpleRNN(20, return_sequences=True)</code> returns <b>every</b> "
        "step: $(batch, T, 20)$ — sequence-to-sequence. When stacking RNN layers, "
        "<b>every layer except possibly the last must have "
        "<code>return_sequences=True</code></b>, or the next layer receives a "
        "vector instead of a sequence. This is the single most common RNN bug.",
    )

    anim_header("An RNN unrolled through time")
    md(
        "The same cell, drawn once per time step. The blue arrows are "
        "$\\mathbf{W}_x$ and the red ones $\\mathbf{W}_{\\hat y}$ — and they are "
        "**the same weights** at every step."
    )

    T = 6
    frames = []
    for k in range(1, T + 1):
        shapes, ann = [], []
        for t in range(T):
            on = t < k
            x0 = t * 1.9
            # the cell
            shapes.append(go.Scatter(
                x=[x0, x0 + 1.2, x0 + 1.2, x0, x0],
                y=[-.35, -.35, .35, .35, -.35], fill="toself",
                fillcolor=(alpha(C["primary"], .9) if t == k - 1
                           else alpha(C["primary"], .45) if on
                           else alpha(C["line"], .3)),
                line=dict(color="#fff" if on else C["line"], width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x0 + .6, y=0, text=f"cell<br>t={t}",
                            showarrow=False,
                            font=dict(size=9,
                                      color="#fff" if on else C["muted"])))
            if on:
                # input arrow
                shapes.append(go.Scatter(x=[x0 + .6, x0 + .6], y=[-1.15, -.35],
                                         mode="lines",
                                         line=dict(color=C["train"], width=3),
                                         showlegend=False, hoverinfo="skip"))
                ann.append(dict(x=x0 + .6, y=-1.4, text=f"x<sub>{t}</sub>",
                                showarrow=False,
                                font=dict(size=10, color=C["train"])))
                # output arrow
                shapes.append(go.Scatter(x=[x0 + .6, x0 + .6], y=[.35, 1.15],
                                         mode="lines",
                                         line=dict(color=C["success"], width=3),
                                         showlegend=False, hoverinfo="skip"))
                ann.append(dict(x=x0 + .6, y=1.4, text=f"ŷ<sub>{t}</sub>",
                                showarrow=False,
                                font=dict(size=10, color=C["success"])))
                # recurrent arrow
                if t > 0:
                    shapes.append(go.Scatter(
                        x=[x0 - .7, x0], y=[0, 0], mode="lines",
                        line=dict(color=C["danger"], width=3.5),
                        showlegend=False, hoverinfo="skip"))
                    ann.append(dict(x=x0 - .35, y=.32, text="h", showarrow=False,
                                    font=dict(size=9, color=C["danger"])))
        ann.append(dict(x=T * 1.9 / 2, y=-2.1,
                        text="W<sub>x</sub> (blue) and W<sub>ŷ</sub> (red) are "
                             "the SAME matrices at every step",
                        showarrow=False,
                        font=dict(size=11, color=C["ink_soft"])))
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=f"time step {k-1} of {T-1}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=400, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-1.3, T * 1.9 + .4]),
                    yaxis=dict(visible=False, range=[-2.5, 1.9]),
                    annotations=list(frames[0].layout.annotations),
                    title="time step 0")
    anim.animate(f, frames, duration=nav.anim_ms(800), slider_prefix="step ")
    figure(f)

    code_lab(
        "A recurrent cell from scratch, and the return_sequences trap",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. A RECURRENT CELL FROM SCRATCH =========================
def simple_rnn_forward(X, Wx, Wh, b, h0=None):
    """X: (batch, T, n_in). Returns all outputs and the final state."""
    batch, T, n_in = X.shape
    n_units = Wx.shape[1]
    h = np.zeros((batch, n_units)) if h0 is None else h0
    outputs = []
    for t in range(T):
        h = np.tanh(X[:, t] @ Wx + h @ Wh + b)      # THE SAME W AT EVERY STEP
        outputs.append(h)
    return np.stack(outputs, axis=1), h

rng = np.random.default_rng(0)
BATCH, T, N_IN, N_UNITS = 4, 5, 3, 6
X = rng.normal(0, 1, (BATCH, T, N_IN)).astype("float32")
Wx = rng.normal(0, .3, (N_IN, N_UNITS)).astype("float32")
Wh = rng.normal(0, .3, (N_UNITS, N_UNITS)).astype("float32")
b  = np.zeros(N_UNITS, dtype="float32")

seq, final = simple_rnn_forward(X, Wx, Wh, b)
print("=== a recurrent cell from scratch ===")
print(f"  input  {X.shape}  ->  all outputs {seq.shape}, final state {final.shape}")
print(f"  the final output IS the last element of the sequence: "
      f"{np.allclose(seq[:, -1], final)}")

# --- verify against Keras --------------------------------------------
layer = keras.layers.SimpleRNN(N_UNITS, return_sequences=True,
                               return_state=True, use_bias=True)
_ = layer(tf.zeros((1, T, N_IN)))          # build it
layer.set_weights([Wx, Wh, b])
k_seq, k_state = layer(tf.constant(X))
print(f"  max |mine - Keras| = {np.abs(seq - k_seq.numpy()).max():.2e}")

# ============ 2. THE PARAMETER COUNT ===================================
print()
print("=== a SimpleRNN's parameters ===")
for n_in, n_units in [(1, 20), (3, 20), (5, 32), (10, 100)]:
    l = keras.layers.SimpleRNN(n_units)
    l(tf.zeros((1, 7, n_in)))
    formula = n_in*n_units + n_units*n_units + n_units
    print(f"  n_in={n_in:>3} units={n_units:>4}:  "
          f"Wx {n_in}x{n_units} + Wh {n_units}x{n_units} + b {n_units} "
          f"= {formula:>7,}   (Keras: {l.count_params():>7,})")
print("  note the count does NOT depend on the sequence length T --")
print("  the SAME weights are reused at every step")

# ============ 3. THE SEQUENCE LENGTH IS FREE ===========================
print()
print("=== one layer, any sequence length ===")
rnn = keras.layers.SimpleRNN(16)
for T_ in [3, 10, 100, 1000]:
    out = rnn(tf.zeros((2, T_, 4)))
    print(f"  T={T_:>5}  ->  output {tuple(out.shape)}  "
          f"parameters {rnn.count_params()}")

# ============ 4. return_sequences: THE CLASSIC BUG =====================
print()
print("=== return_sequences ===")
x = tf.zeros((8, 20, 3))
for flag in (False, True):
    l = keras.layers.SimpleRNN(16, return_sequences=flag)
    print(f"  return_sequences={str(flag):<6} -> {str(tuple(l(x).shape)):<16} "
          f"{'sequence-to-VECTOR' if not flag else 'sequence-to-SEQUENCE'}")

print()
print("  stacking RNN layers -- the middle ones MUST return sequences:")
good = keras.Sequential([keras.layers.Input(shape=(20, 3)),
                         keras.layers.SimpleRNN(16, return_sequences=True),
                         keras.layers.SimpleRNN(16, return_sequences=True),
                         keras.layers.SimpleRNN(16),
                         keras.layers.Dense(1)])
print(f"    correct stack: output {tuple(good(tf.zeros((2, 20, 3))).shape)}")
try:
    bad = keras.Sequential([keras.layers.Input(shape=(20, 3)),
                           keras.layers.SimpleRNN(16),          # NOT sequences
                           keras.layers.SimpleRNN(16),
                           keras.layers.Dense(1)])
except Exception as e:
    print(f"    forgetting it: {type(e).__name__}")
    print(f"      {str(e).splitlines()[0][:80]}")

# ============ 5. THE FOUR SEQUENCE SHAPES ==============================
print()
print("=== the four architectures ===")
T_IN, T_OUT, N_FEAT = 20, 5, 3

seq2seq = keras.Sequential([keras.layers.Input(shape=(T_IN, N_FEAT)),
                            keras.layers.SimpleRNN(32, return_sequences=True),
                            keras.layers.Dense(1)])
seq2vec = keras.Sequential([keras.layers.Input(shape=(T_IN, N_FEAT)),
                            keras.layers.SimpleRNN(32),
                            keras.layers.Dense(1)])
vec2seq = keras.Sequential([keras.layers.Input(shape=(N_FEAT,)),
                            keras.layers.RepeatVector(T_OUT),
                            keras.layers.SimpleRNN(32, return_sequences=True),
                            keras.layers.Dense(1)])
encdec  = keras.Sequential([keras.layers.Input(shape=(T_IN, N_FEAT)),
                            keras.layers.SimpleRNN(32),          # encoder
                            keras.layers.RepeatVector(T_OUT),
                            keras.layers.SimpleRNN(32, return_sequences=True),
                            keras.layers.Dense(1)])              # decoder

for nm, m, in_shape in [("sequence-to-sequence", seq2seq, (T_IN, N_FEAT)),
                        ("sequence-to-vector", seq2vec, (T_IN, N_FEAT)),
                        ("vector-to-sequence", vec2seq, (N_FEAT,)),
                        ("encoder-decoder", encdec, (T_IN, N_FEAT))]:
    out_shape = tuple(m(tf.zeros((1,) + in_shape)).shape[1:])
    print(f"  {nm:<24} {str(in_shape):>12} -> "
          f"{str(out_shape):<10} {m.count_params():>7,} params")

# ============ 6. STATE vs OUTPUT =======================================
print()
print("=== for a SimpleRNN, state == output ===")
l = keras.layers.SimpleRNN(8, return_sequences=True, return_state=True)
seq_out, state = l(tf.random.normal((2, 6, 3)))
print(f"  sequence {tuple(seq_out.shape)}, state {tuple(state.shape)}")
print(f"  state == last output? {np.allclose(seq_out[:, -1].numpy(), state.numpy())}")

lstm = keras.layers.LSTM(8, return_sequences=True, return_state=True)
o, h, c = lstm(tf.random.normal((2, 6, 3)))
print()
print(f"  an LSTM returns TWO states:")
print(f"    output {tuple(o.shape)}, h {tuple(h.shape)}, c (cell) {tuple(c.shape)}")
print(f"    h == last output? {np.allclose(o[:, -1].numpy(), h.numpy())}")
print(f"    c is a SEPARATE long-term state -- that separation is why LSTM works")
''',
        key="ch15_rnn",
    )

    keypoints([
        "$\\hat{\\mathbf{y}}_{(t)} = \\phi(\\mathbf{W}_x^\\top\\mathbf{x}_{(t)} + "
        "\\mathbf{W}_{\\hat y}^\\top\\hat{\\mathbf{y}}_{(t-1)} + \\mathbf{b})$ — "
        "one loop, that is all.",
        "The weights are <b>shared across time</b>, so the parameter count is "
        "independent of the sequence length.",
        "A memory cell's <b>state</b> $\\mathbf{h}_{(t)}$ carries information "
        "forward; for a SimpleRNN it equals the output.",
        "Four shapes: seq-to-seq, seq-to-vector, vector-to-seq, encoder–decoder.",
        "<b>Every stacked RNN layer but the last needs "
        "<code>return_sequences=True</code></b>.",
    ])


# ==========================================================================
def s_15_2():
    section("15.2", "Training RNNs — Backpropagation Through Time")

    lead(
        "Unroll the network through time, then apply ordinary backpropagation to "
        "the unrolled graph. The mathematics is standard; the consequence is not."
    )

    sub("BPTT")

    derive(
        [("Unrolling an RNN over $T$ steps produces a feedforward network $T$ "
          "layers deep — but with the <b>same weights</b> in every layer. The "
          "forward pass is:",
          r"\mathbf{h}_{(t)} = \tanh\bigl(\mathbf{W}_x^\top\mathbf{x}_{(t)} "
          r"+ \mathbf{W}_h^\top\mathbf{h}_{(t-1)} + \mathbf{b}\bigr)"),
         ("The loss typically sums over the output steps:",
          r"\mathcal{L} = \sum_{t} \ell\bigl(\mathbf{y}_{(t)},\, \hat{\mathbf{y}}_{(t)}\bigr)"),
         ("Because $\\mathbf{W}_h$ appears at <b>every</b> step, its gradient is a "
          "<b>sum over all steps</b> of the contributions:",
          r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}_h} \;=\; "
          r"\sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_{(t)}}"
          r"\;\frac{\partial \mathbf{h}_{(t)}}{\partial \mathbf{W}_h}"),
         ("And $\\partial\\mathbf{h}_{(t)}/\\partial\\mathbf{W}_h$ itself depends "
          "on every earlier step, giving the crucial factor:",
          r"\frac{\partial \mathbf{h}_{(t)}}{\partial \mathbf{h}_{(k)}} \;=\; "
          r"\prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_{(i)}}{\partial \mathbf{h}_{(i-1)}} "
          r"\;=\; \prod_{i=k+1}^{t} \mathbf{W}_h^\top \,\mathrm{diag}\bigl(1 - \mathbf{h}_{(i)}^2\bigr)"),
         ("<b>That is a product of $t - k$ terms, all involving the same "
          "$\\mathbf{W}_h$.</b> Bounding its norm:",
          r"\left\lVert \frac{\partial \mathbf{h}_{(t)}}{\partial \mathbf{h}_{(k)}} \right\rVert "
          r"\;\le\; \bigl(\lVert\mathbf{W}_h\rVert \cdot \gamma\bigr)^{\,t-k},"
          r"\qquad \gamma = \max_i \lVert \mathrm{diag}(1 - \mathbf{h}_i^2)\rVert \le 1"),
         ("So if $\\lVert\\mathbf{W}_h\\rVert\\gamma < 1$ the gradient <b>vanishes "
          "exponentially in the time lag</b>; if $> 1$ it <b>explodes</b>. This "
          "is §11.1's problem again, but now the exponent is the <b>sequence "
          "length</b>, not the layer count — and sequences are routinely hundreds "
          "of steps long.", None),
         ("<b>The practical consequence:</b> a plain RNN cannot learn dependencies "
          "beyond roughly 10–20 steps. Everything in §15.7 and §15.8 exists to "
          "fix this.", None)],
        title="Deriving BPTT, and where the exponential comes from",
    )

    sub("Truncated BPTT")

    md(
        "Unrolling 10 000 steps is impossible — the memory alone would be "
        "prohibitive (§14.2's activation argument, times the sequence length). "
        "**Truncated BPTT** unrolls only the last $k$ steps, propagating the "
        "gradient back $k$ steps and no further."
    )

    table(
        ["", "Full BPTT", "Truncated BPTT"],
        [["Unroll length", "The whole sequence", "$k$ steps (e.g. 20–100)"],
         ["Memory", "$\\mathcal{O}(T)$ activations", "$\\mathcal{O}(k)$"],
         ["Longest learnable dependency", "$T$ (in principle)", "$k$"],
         ["Used by", "Short sequences", "Almost everything in practice"]],
    )

    sub("Stateful RNNs")

    md(
        "A **stateful** RNN preserves its hidden state from one batch to the next "
        "instead of resetting it. That lets the model carry information across "
        "batch boundaries — learning patterns longer than the window — at the cost "
        "of several strict requirements:"
    )

    pitfall(
        "Stateful RNNs have four requirements, and all of them matter",
        "<b>(1)</b> The batches must be <b>consecutive</b> — batch $n+1$ must "
        "continue exactly where batch $n$ stopped, so you <b>cannot shuffle</b>. "
        "<b>(2)</b> A fixed <code>batch_input_shape</code> is required, because "
        "the state tensor must persist. <b>(3)</b> Sequence $i$ of every batch "
        "must continue sequence $i$ of the previous batch — so preparing the data "
        "means interleaving, not simply windowing. <b>(4)</b> You must call "
        "<code>model.reset_states()</code> at the end of every epoch, or the state "
        "from the end of the data leaks into the beginning.",
    )

    anim_header("Gradient magnitude versus time lag")
    md(
        "The gradient reaching step $t-k$ from step $t$, for three values of "
        "$\\lVert\\mathbf{W}_h\\rVert$. Note the log scale: the decay is "
        "exponential, and even a modest $\\lVert\\mathbf{W}_h\\rVert = 0.9$ makes "
        "step 100 invisible."
    )

    lags = np.arange(0, 101)
    curves = {
        "‖W‖γ = 0.7  (vanishes fast)": .7 ** lags,
        "‖W‖γ = 0.9  (vanishes)": .9 ** lags,
        "‖W‖γ = 1.0  (the knife edge)": np.ones_like(lags, dtype=float),
        "‖W‖γ = 1.05 (explodes)": 1.05 ** lags,
    }
    cols = [C["danger"], C["warning"], C["success"], C["proof"]]

    frames = []
    for k in range(2, 102, 2):
        data = []
        info = []
        for i, (nm, v) in enumerate(curves.items()):
            data.append(go.Scatter(x=lags[:k], y=np.clip(v[:k], 1e-40, 1e40),
                                   mode="lines", line=dict(color=cols[i], width=3)))
            info.append(f"{nm.split()[2]}: {v[k-1]:.2e}")
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"lag {k} steps   |   " + "   ".join(info))])))

    f = go.Figure(data=[go.Scatter(x=lags[:2], y=v[:2], mode="lines", name=nm,
                                   line=dict(color=cols[i], width=3))
                        for i, (nm, v) in enumerate(curves.items())])
    f.add_hline(y=1e-7, line_dash="dot", line_color=C["muted"],
                annotation_text="float32 noise floor")
    f.update_layout(height=440, yaxis_type="log", xaxis_title="time lag (steps)",
                    yaxis_title="relative gradient magnitude",
                    yaxis=dict(range=[-20, 6]),
                    title="‖∂h(t)/∂h(t−k)‖ ≈ (‖W‖γ)^k",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(60), slider_prefix="lag ")
    figure(f, "At ‖W‖γ = 0.9 the gradient from 100 steps back is 2.7×10⁻⁵ of "
              "the local one — the network simply cannot learn that dependency.")

    code_lab(
        "BPTT from scratch, the gradient decay, and truncation",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. BPTT FROM SCRATCH =====================================
def rnn_forward(X, Wx, Wh, b):
    """Returns every hidden state (needed for the backward pass)."""
    B, T, _ = X.shape
    U = Wx.shape[1]
    H = [np.zeros((B, U))]
    for t in range(T):
        H.append(np.tanh(X[:, t] @ Wx + H[-1] @ Wh + b))
    return H                              # H[0] is the initial zero state

def rnn_backward(X, H, dY_last, Wx, Wh):
    """BPTT: propagate the gradient from the LAST output backwards."""
    B, T, _ = X.shape
    dWx = np.zeros_like(Wx); dWh = np.zeros_like(Wh)
    dh = dY_last                                   # gradient w.r.t. h_T
    lag_norms = []
    for t in reversed(range(T)):
        dz = dh * (1 - H[t+1]**2)                  # through tanh
        dWx += X[:, t].T @ dz
        dWh += H[t].T @ dz
        dh = dz @ Wh.T                             # THE PRODUCT ACCUMULATES
        lag_norms.append(np.linalg.norm(dh))
    return dWx, dWh, lag_norms[::-1]

rng = np.random.default_rng(0)
B, T, N_IN, U = 2, 60, 3, 8
X = rng.normal(0, 1, (B, T, N_IN))
b = np.zeros(U)

print("=== how far back does the gradient survive? ===")
print(f"{'||Wh|| scale':>14}{'grad at lag 1':>16}{'lag 20':>12}{'lag 40':>12}"
      f"{'lag 59':>12}")
for scale in [0.3, 0.7, 1.0, 1.4]:
    Wx = rng.normal(0, .3, (N_IN, U))
    Wh = rng.normal(0, 1, (U, U))
    Wh = Wh / np.linalg.norm(Wh, 2) * scale        # set the spectral norm
    H = rnn_forward(X, Wx, Wh, b)
    _, _, norms = rnn_backward(X, H, np.ones((B, U)), Wx, Wh)
    norms = np.array(norms)
    rel = norms / norms[-1]
    print(f"{scale:>14.1f}{rel[-1]:>16.4e}{rel[-21]:>12.4e}"
          f"{rel[-41]:>12.4e}{rel[0]:>12.4e}")
print("  the decay is EXPONENTIAL in the lag -- that is the whole problem")

# ============ 2. VERIFY AGAINST TENSORFLOW =============================
print()
print("=== gradient check against tf.GradientTape ===")
Wx = rng.normal(0, .3, (N_IN, U)); Wh = rng.normal(0, .3, (U, U))
H = rnn_forward(X, Wx, Wh, b)
dWx_mine, dWh_mine, _ = rnn_backward(X, H, np.ones((B, U)), Wx, Wh)

tWx = tf.Variable(Wx, dtype=tf.float64)
tWh = tf.Variable(Wh, dtype=tf.float64)
tb  = tf.Variable(b,  dtype=tf.float64)
tX  = tf.constant(X, dtype=tf.float64)
with tf.GradientTape() as tape:
    h = tf.zeros((B, U), dtype=tf.float64)
    for t in range(T):
        h = tf.tanh(tX[:, t] @ tWx + h @ tWh + tb)
    loss = tf.reduce_sum(h)                        # sum of the LAST output
g = tape.gradient(loss, [tWx, tWh])
print(f"  max |dWx mine - TF| = {np.abs(dWx_mine - g[0].numpy()).max():.3e}")
print(f"  max |dWh mine - TF| = {np.abs(dWh_mine - g[1].numpy()).max():.3e}")

# ============ 3. CAN AN RNN LEARN A LONG DEPENDENCY? ===================
print()
print("=== the copy task: output = the FIRST input, T steps later ===")
def copy_task(n, T, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, T, 1)).astype("float32")
    y = X[:, 0, 0].copy()                # the answer is the FIRST element
    X[:, 1:, 0] = r.normal(0, 1, (n, T-1))   # everything else is noise
    return X, y

print(f"{'sequence length':>17}{'SimpleRNN R^2':>16}{'LSTM R^2':>12}{'GRU R^2':>11}")
for T_ in [5, 10, 25, 50]:
    Xa, ya = copy_task(3000, T_, seed=1)
    Xb, yb = copy_task(800, T_, seed=2)
    scores = []
    for cell in [keras.layers.SimpleRNN, keras.layers.LSTM, keras.layers.GRU]:
        tf.random.set_seed(0)
        m = keras.Sequential([keras.layers.Input(shape=(T_, 1)),
                              cell(32), keras.layers.Dense(1)])
        m.compile(loss="mse", optimizer=keras.optimizers.Adam(5e-3))
        m.fit(Xa, ya, epochs=25, batch_size=64, verbose=0)
        p = m.predict(Xb, verbose=0).ravel()
        r2 = 1 - np.mean((p-yb)**2)/np.var(yb)
        scores.append(r2)
    print(f"{T_:>17}{scores[0]:>16.4f}{scores[1]:>12.4f}{scores[2]:>11.4f}")
print("  the SimpleRNN collapses as T grows; LSTM and GRU hold on (section 15.8)")

# ============ 4. TRUNCATED BPTT ========================================
print()
print("=== truncated BPTT: memory vs learnable lag ===")
print(f"{'unroll k':>10}{'activations kept':>20}{'longest dependency':>22}")
for k in [10, 50, 200, 1000]:
    print(f"{k:>10}{k*32*64:>20,}{k:>22}")
print("  memory is linear in k; so is the longest dependency you can learn")

# ============ 5. STATEFUL RNNs =========================================
print()
print("=== stateless vs stateful ===")
BATCH, WINDOW = 4, 10
stateless = keras.Sequential([keras.layers.Input(shape=(WINDOW, 1)),
                              keras.layers.SimpleRNN(8),
                              keras.layers.Dense(1)])
stateful = keras.Sequential([
    keras.layers.Input(batch_shape=(BATCH, WINDOW, 1)),   # FIXED batch size
    keras.layers.SimpleRNN(8, stateful=True),
    keras.layers.Dense(1)])
print(f"  stateless input shape : (None, {WINDOW}, 1)   <- any batch size")
print(f"  stateful  input shape : ({BATCH}, {WINDOW}, 1)   <- batch size is FIXED")

x = tf.random.normal((BATCH, WINDOW, 1))
print()
print(f"  stateless: same input twice -> identical output? "
      f"{np.allclose(stateless(x), stateless(x))}")
print(f"  stateful : same input twice -> identical output? "
      f"{np.allclose(stateful(x).numpy(), stateful(x).numpy())}"
      f"   <- the state CARRIED OVER")
# in Keras 3 reset_states() lives on the LAYER, not on the model
rnn_layer = [l for l in stateful.layers if getattr(l, "stateful", False)][0]
rnn_layer.reset_states()
first = stateful(x).numpy()
rnn_layer.reset_states()
print(f"  after reset_states(), the first call reproduces: "
      f"{np.allclose(first, stateful(x).numpy())}")
print()
print("  requirements: consecutive batches, no shuffling, fixed batch size,")
print("  and layer.reset_states() at the end of every epoch")
''',
        key="ch15_bptt",
    )

    quiz(
        "Why can a plain RNN not learn a dependency 100 steps back?",
        ["It runs out of memory",
         "The gradient is a product of 100 factors, so it vanishes or explodes "
         "exponentially",
         "It has too few parameters",
         "The activation function saturates"],
        1,
        "$\\partial\\mathbf{h}_{(t)}/\\partial\\mathbf{h}_{(t-100)}$ is a product "
        "of 100 copies of $\\mathbf{W}_h^\\top\\mathrm{diag}(1-\\mathbf{h}^2)$. At "
        "$\\lVert\\cdot\\rVert = 0.9$ that is $2.7\\times10^{-5}$; the signal is "
        "below the noise floor.",
        key="ch15q1",
    )

    keypoints([
        "BPTT = unroll through time, then ordinary backprop; $\\mathbf{W}_h$'s "
        "gradient is a <b>sum over all steps</b>.",
        "$\\partial\\mathbf{h}_{(t)}/\\partial\\mathbf{h}_{(k)}$ is a product of "
        "$t-k$ factors — exponential in the <b>time lag</b>.",
        "A plain RNN therefore cannot learn dependencies beyond ~10–20 steps.",
        "<b>Truncated BPTT</b> bounds memory <i>and</i> the longest learnable "
        "dependency at $k$.",
        "<b>Stateful</b> RNNs carry state across batches, but require consecutive, "
        "unshuffled, fixed-size batches and explicit resets.",
    ])


# ==========================================================================
def s_15_3():
    section("15.3", "Forecasting a Time Series — Baselines and ARMA")

    lead(
        "Before any neural network, establish what a trivially simple model "
        "achieves. On time series the trivial models are unusually strong, and "
        "beating them is the actual task."
    )

    sub("The baselines you must beat")

    table(
        ["Baseline", "Prediction", "When it is hard to beat"],
        [["<b>Naive / persistence</b>", "$\\hat y_{t+1} = y_t$",
          "Random walks — financial prices, essentially always"],
         ["<b>Seasonal naive</b>", "$\\hat y_{t+1} = y_{t+1-s}$",
          "<b>Strongly seasonal data</b> — this is the one that embarrasses "
          "people"],
         ["<b>Drift</b>", "$\\hat y_{t+h} = y_t + h\\frac{y_t - y_1}{t-1}$",
          "Trending series"],
         ["<b>Mean</b>", "$\\hat y_{t+1} = \\bar y$", "Stationary noise"]],
    )

    pitfall(
        "Publishing a model that loses to seasonal naive",
        "It happens constantly. Daily ridership, retail demand, electricity load "
        "— all have a strong weekly cycle, so \"same day last week\" is a very "
        "good forecast. A neural network that achieves 4 % MAPE sounds impressive "
        "until you discover the seasonal-naive baseline achieves 3.7 %. "
        "<b>Compute the baselines first, and report them alongside your model.</b>",
    )

    sub("Differencing and stationarity")

    md(
        "Most forecasting theory assumes **stationarity** — constant mean, "
        "variance and autocovariance over time. Real series rarely are. "
        "**Differencing** removes trend and seasonality:"
    )

    math(r"""
    \nabla y_t \;=\; y_t - y_{t-1}
    \qquad\qquad
    \nabla_s y_t \;=\; y_t - y_{t-s}
    \qquad\qquad
    \nabla^2 y_t \;=\; \nabla(\nabla y_t) = y_t - 2y_{t-1} + y_{t-2}
    """)

    idea(
        "Differencing is exactly what makes the naive baseline the null model",
        "If you difference once and the result is white noise, then "
        "$y_t = y_{t-1} + \\varepsilon_t$ — a random walk — and <b>no model can "
        "beat the naive forecast</b>, because the best prediction of "
        "$\\varepsilon_t$ is zero. Differencing and then checking for remaining "
        "structure is therefore the honest first question: is there anything here "
        "to model at all?",
    )

    sub("The ARMA family")

    math(r"""
    \hat y_{(t)} \;=\;
    \underbrace{\sum_{i=1}^{p} \alpha_i \, y_{(t-i)}}_{\text{AR}(p)}
    \;+\;
    \underbrace{\sum_{i=1}^{q} \theta_i \, \varepsilon_{(t-i)}}_{\text{MA}(q)},
    \qquad
    \varepsilon_{(t)} = y_{(t)} - \hat y_{(t)}
    """)

    table(
        ["Model", "Parameters", "What it captures"],
        [["<b>AR($p$)</b>", "$p$", "Dependence on the last $p$ values"],
         ["<b>MA($q$)</b>", "$q$", "Dependence on the last $q$ forecast "
          "<i>errors</i>"],
         ["<b>ARMA($p, q$)</b>", "$p + q$", "Both"],
         ["<b>ARIMA($p, d, q$)</b>", "$p + q$", "ARMA on the $d$-times "
          "differenced series — handles trend"],
         ["<b>SARIMA($p,d,q$)($P,D,Q$)$_s$</b>", "$p+q+P+Q$",
          "ARIMA plus a seasonal ARIMA at lag $s$"]],
    )

    md(
        "Choosing $p$ and $q$: use the **ACF** (autocorrelation) and **PACF** "
        "(partial autocorrelation) plots, or minimise AIC/BIC over a grid — the "
        "same model-selection argument as §9.7."
    )

    anim_header("Differencing a seasonal series until it is stationary")

    dfr = ds.ridership(n_days=365)
    y = dfr["rail"].to_numpy() / 1e6
    t = np.arange(len(y))

    stages = [
        ("original series", y, "trend + weekly seasonality + noise"),
        ("first difference ∇y", np.r_[np.nan, np.diff(y)],
         "trend removed, weekly cycle remains"),
        ("seasonal difference ∇₇y", np.r_[[np.nan]*7, y[7:] - y[:-7]],
         "weekly cycle removed, trend remains"),
        ("both: ∇∇₇y", np.r_[[np.nan]*8, np.diff(y[7:] - y[:-7])],
         "close to stationary — this is what ARIMA models"),
    ]

    frames = []
    for i, (nm, v, note_) in enumerate(stages):
        finite = v[np.isfinite(v)]
        # sample autocorrelation at lags 1 and 7
        def acf(a, lag):
            a = a - a.mean()
            return float(np.sum(a[lag:] * a[:-lag]) / np.sum(a * a))
        frames.append(go.Frame(name=str(i + 1), data=[
            go.Scatter(x=t, y=v, mode="lines",
                       line=dict(color=SEQ[i], width=1.8)),
            go.Bar(x=list(range(1, 22)),
                   y=[acf(finite, L) for L in range(1, 22)],
                   marker=dict(color=[C["danger"] if L in (7, 14, 21)
                                      else SEQ[i] for L in range(1, 22)])),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{nm}   ·   {note_}   ·   sd = {finite.std():.4f}   ·   "
            f"ACF(1) = {acf(finite,1):+.3f}   ACF(7) = {acf(finite,7):+.3f}")])))

    f = make_subplots(rows=2, cols=1, row_heights=[.6, .4],
                      subplot_titles=("the series",
                                      "autocorrelation (red = weekly lags)"))
    f.add_trace(go.Scatter(x=t, y=stages[0][1], mode="lines", showlegend=False,
                           line=dict(color=SEQ[0], width=1.8)), 1, 1)
    f.add_trace(go.Bar(x=list(range(1, 22)), y=[0]*21, showlegend=False,
                       marker=dict(color=SEQ[0])), 2, 1)
    f.update_xaxes(title_text="day", row=1, col=1)
    f.update_xaxes(title_text="lag", row=2, col=1)
    f.update_yaxes(range=[-.6, 1.05], row=2, col=1)
    f.update_layout(height=520, title="Differencing toward stationarity")
    anim.animate(f, frames, duration=nav.anim_ms(1600), slider_prefix="stage ")
    figure(f, "The spikes at lags 7, 14, 21 are the weekly cycle. Seasonal "
              "differencing removes them.")

    code_lab(
        "Baselines, differencing, ACF/PACF and SARIMA",
        '''import numpy as np, pandas as pd
from core import datasets as _ds

df = _ds.ridership(n_days=730)
y = df["rail"].to_numpy() / 1e6
print(f"=== {len(y)} daily observations ===")
print(f"  mean {y.mean():.3f}M  sd {y.std():.3f}M  "
      f"min {y.min():.3f}M  max {y.max():.3f}M")

SPLIT = 600
train, test = y[:SPLIT], y[SPLIT:]
H = len(test)

# ============ 1. THE BASELINES YOU MUST BEAT ===========================
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def mape(a, b): return float(np.mean(np.abs((a - b) / a)) * 100)

print()
print("=== one-step-ahead baselines on the test set ===")
print(f"{'baseline':<26}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}")

naive        = y[SPLIT-1:-1]                    # yesterday
seasonal     = y[SPLIT-7:-7]                    # same day last week
mean_pred    = np.full(H, train.mean())
drift_slope  = (train[-1] - train[0]) / (len(train) - 1)
drift        = train[-1] + drift_slope * np.arange(1, H+1)

for nm, pred in [("mean", mean_pred), ("naive (yesterday)", naive),
                 ("drift", drift), ("SEASONAL naive (t-7)", seasonal)]:
    print(f"{nm:<26}{mae(test, pred):>10.4f}{rmse(test, pred):>10.4f}"
          f"{mape(test, pred):>10.2f}")
print()
print("  Seasonal naive is the number to beat. It uses ZERO parameters.")

# ============ 2. IS THERE ANYTHING TO MODEL? ===========================
print()
print("=== differencing toward stationarity ===")
def acf(a, lag):
    a = np.asarray(a, dtype=float); a = a - a.mean()
    return float(np.sum(a[lag:] * a[:-lag]) / np.sum(a * a))

series = {
    "original y":        y,
    "first diff  ∇y":    np.diff(y),
    "seasonal    ∇7y":   y[7:] - y[:-7],
    "both        ∇∇7y":  np.diff(y[7:] - y[:-7]),
}
print(f"{'series':<20}{'sd':>9}{'ACF(1)':>10}{'ACF(7)':>10}{'ACF(14)':>10}"
      f"{'ACF(30)':>10}")
for nm, s in series.items():
    print(f"{nm:<20}{s.std():>9.4f}" +
          "".join(f"{acf(s, L):>10.3f}" for L in (1, 7, 14, 30)))
print()
print("  ∇7y kills the weekly spikes; ∇∇7y is close to white noise.")
print("  If a differenced series IS white noise, no model beats the naive one.")

# --- an Augmented Dickey-Fuller test, if statsmodels is available -----
try:
    from statsmodels.tsa.stattools import adfuller
    print()
    print(f"  {'series':<20}{'ADF statistic':>16}{'p-value':>11}   verdict")
    for nm, s in series.items():
        stat, pval = adfuller(s)[:2]
        verdict = "stationary" if pval < .05 else "NOT stationary"
        print(f"  {nm:<20}{stat:>16.4f}{pval:>11.4f}   {verdict}")
except ImportError:
    print("  (install statsmodels for the Augmented Dickey-Fuller test)")

# ============ 3. AR, MA, ARMA FROM SCRATCH =============================
print()
print("=== an AR(p) model by least squares ===")
def fit_ar(series, p):
    """Build the lag matrix and solve for the coefficients."""
    Xm = np.column_stack([series[p-i-1:-i-1] for i in range(p)])
    yv = series[p:]
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xm)), Xm], yv, rcond=None)
    return coef

def forecast_ar(series, coef, h):
    p = len(coef) - 1
    hist = list(series[-p:])
    out = []
    for _ in range(h):
        nxt = coef[0] + np.dot(coef[1:], hist[::-1][:p])
        out.append(nxt); hist.append(nxt)
    return np.array(out)

print(f"{'model':<12}{'coefficients':>44}{'test MAE':>11}")
for p in [1, 2, 7, 14]:
    coef = fit_ar(train, p)
    pred = forecast_ar(train, coef, H)
    shown = np.round(coef[:min(4, len(coef))], 3)
    print(f"{f'AR({p})':<12}{str(shown) + ('...' if p > 3 else ''):>44}"
          f"{mae(test, pred):>11.4f}")
print("  a long multi-step AR forecast decays to the mean -- that is expected")

# --- one-step-ahead AR is a much fairer comparison -------------------
print()
print("=== ONE-STEP-ahead (refit-free, rolling) ===")
print(f"{'model':<22}{'test MAE':>11}")
for p in [1, 7, 14, 28]:
    coef = fit_ar(train, p)
    preds = [coef[0] + np.dot(coef[1:], y[SPLIT+i-1:SPLIT+i-1-p:-1])
             for i in range(H)]
    print(f"{f'AR({p}) one-step':<22}{mae(test, np.array(preds)):>11.4f}")
print(f"{'seasonal naive':<22}{mae(test, seasonal):>11.4f}")

# ============ 4. SARIMA ================================================
try:
    from statsmodels.tsa.arima.model import ARIMA
    print()
    print("=== SARIMA ===")
    print(f"{'order':<34}{'AIC':>11}{'test MAE':>11}")
    for order, seasonal_order in [((1,0,0), (0,0,0,0)),
                                  ((2,0,1), (0,0,0,0)),
                                  ((1,1,1), (0,0,0,0)),
                                  ((1,0,0), (1,0,0,7)),
                                  ((2,0,1), (1,1,1,7))]:
        try:
            m = ARIMA(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit()
            pred = m.forecast(steps=H)
            label = f"SARIMA{order}x{seasonal_order}"
            print(f"{label:<34}{m.aic:>11.1f}{mae(test, pred):>11.4f}")
        except Exception as e:
            print(f"{str(order):<34}{'failed':>11}")
    print("  the seasonal terms are what actually matter here")
except ImportError:
    print()
    print("  (install statsmodels for SARIMA: pip install statsmodels)")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=y, mode="lines", name="actual",
                line=dict(color=C["truth"], width=1.5))
fig.add_scatter(x=np.arange(SPLIT, len(y)), y=seasonal, mode="lines",
                name="seasonal naive", line=dict(color=C["success"], width=2))
fig.add_vline(x=SPLIT, line_dash="dash", line_color=C["danger"],
              annotation_text="train / test")
fig.update_layout(height=400, xaxis_title="day", yaxis_title="ridership (M)",
                  title="The seasonal-naive baseline")
''',
        key="ch15_baselines",
    )

    keypoints([
        "<b>Compute the baselines first</b> — seasonal naive is often very hard "
        "to beat and costs nothing.",
        "Differencing removes trend ($\\nabla$) and seasonality ($\\nabla_s$); if "
        "the result is white noise, stop.",
        "<b>AR($p$)</b> uses past values, <b>MA($q$)</b> past errors, "
        "<b>ARIMA</b> adds differencing, <b>SARIMA</b> adds a seasonal copy.",
        "Choose $(p,q)$ from ACF/PACF or by minimising AIC/BIC.",
        "Always report your model <b>next to</b> the baselines, not alone.",
    ])

# ==========================================================================
def s_15_4():
    section("15.4", "Preparing the Data and the First Neural Forecasters")

    lead(
        "Windowing turns a time series into a supervised learning problem. Getting "
        "the windowing right — and the split right — matters more than the model."
    )

    sub("Windowing")

    md(
        "`tf.keras.utils.timeseries_dataset_from_array` does the work: it slides "
        "a window of length $L$ across the series and pairs each window with the "
        "target that follows it."
    )

    math(r"""
    \mathbf{x}^{(i)} = \bigl(y_i, y_{i+1}, \dots, y_{i+L-1}\bigr),
    \qquad
    y^{(i)} = y_{i+L}
    """)

    pitfall(
        "Never shuffle before splitting a time series",
        "A random split puts windows from <i>after</i> the test period into the "
        "training set. Since consecutive windows overlap by $L-1$ values, the "
        "model effectively memorises the test targets — and your validation score "
        "becomes fiction. <b>Always split by time</b>: train on the first period, "
        "validate on the next, test on the last. Shuffling <i>within</i> the "
        "training period is fine and helpful; shuffling <i>across</i> the split is "
        "leakage.",
    )

    warn(
        "The gap between splits matters too",
        "If your window is 56 days long and your validation set starts the day "
        "after training ends, the first 56 validation windows contain training "
        "days. For strict evaluation, leave a gap of at least $L$ steps between "
        "the splits — or accept that the first few validation points are "
        "contaminated.",
    )

    sub("Scaling")

    tip(
        "Fit the scaler on the training period only, and consider differencing "
        "instead",
        "The mean and standard deviation are statistics learned from data (§2.4). "
        "Computing them over the whole series leaks the future. For trending "
        "series, scaling is not enough anyway — a model trained on 2019 values "
        "will see 2024 values outside its training range, and neural networks "
        "extrapolate badly (§6.6). <b>Model the differences</b>, or divide by a "
        "rolling baseline, so the target stays in a stable range.",
    )

    sub("The simplest neural forecasters")

    table(
        ["Model", "Parameters (window 56)", "Note"],
        [["Linear (Dense(1) on the flattened window)", "57",
          "Often beats a naive RNN — it is an AR(56) model"],
         ["<code>SimpleRNN(1)</code>", "3",
          "One recurrent unit; too small to be useful"],
         ["<code>SimpleRNN(32)</code> + <code>Dense(1)</code>", "1 121",
          "The reasonable minimum"],
         ["Deep RNN (3 × 32) + Dense", "~7 400", "§15.5"]],
    )

    idea(
        "A linear model on a window IS an AR model",
        "<code>Dense(1)</code> applied to a flattened window of the last $L$ "
        "values computes $\\hat y = \\sum_i w_i y_{t-i} + b$ — which is exactly "
        "AR($L$) from §15.3, fitted by gradient descent instead of least squares. "
        "So the 'trivial' neural baseline is a classical model in disguise, and it "
        "is genuinely strong. If your RNN cannot beat it, the RNN is not earning "
        "its complexity.",
    )

    anim_header("Windowing a series into training examples")

    dfr = ds.ridership(n_days=120)
    yv = dfr["rail"].to_numpy() / 1e6
    L = 14
    n_show = 10

    frames = []
    for k in range(n_show):
        lo, hi = k * 4, k * 4 + L
        frames.append(go.Frame(name=str(k + 1), data=[
            go.Scatter(x=np.arange(len(yv)), y=yv, mode="lines",
                       line=dict(color=C["muted"], width=1.5)),
            go.Scatter(x=np.arange(lo, hi), y=yv[lo:hi], mode="lines+markers",
                       line=dict(color=C["train"], width=3),
                       marker=dict(size=6)),
            go.Scatter(x=[hi], y=[yv[hi]], mode="markers",
                       marker=dict(color=C["danger"], size=15, symbol="star",
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"window {k+1}: inputs are days {lo}–{hi-1} (blue), "
            f"target is day {hi} (red)   ·   x shape ({L}, 1), y scalar")])))

    f = go.Figure(data=[
        go.Scatter(x=np.arange(len(yv)), y=yv, mode="lines", name="series",
                   line=dict(color=C["muted"], width=1.5)),
        go.Scatter(x=np.arange(0, L), y=yv[:L], mode="lines+markers",
                   name="input window", line=dict(color=C["train"], width=3),
                   marker=dict(size=6)),
        go.Scatter(x=[L], y=[yv[L]], mode="markers", name="target",
                   marker=dict(color=C["danger"], size=15, symbol="star",
                               line=dict(color="#fff", width=2))),
    ])
    f.update_layout(height=400, xaxis_title="day", yaxis_title="ridership (M)",
                    title=f"Sliding a window of {L} days",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="window ")
    figure(f, "Consecutive windows overlap by L−1 values — which is exactly why "
              "shuffling before splitting leaks the future.")

    code_lab(
        "Windowing, the leakage demonstration, and the first models",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

df = _ds.ridership(n_days=730)
y = (df["rail"].to_numpy() / 1e6).astype("float32")
SEQ_LEN = 56

# ============ 1. WINDOWING =============================================
print("=== timeseries_dataset_from_array ===")
demo = keras.utils.timeseries_dataset_from_array(
    np.arange(12, dtype="float32"),           # 0..11
    targets=np.arange(12, dtype="float32")[3:],   # target = 3 steps ahead
    sequence_length=3, batch_size=2)
for xb, yb in demo:
    for i in range(len(xb)):
        print(f"  window {xb[i].numpy().astype(int)} -> target {int(yb[i])}")

# --- and by hand, so you can see the overlap -------------------------
def make_windows(series, L):
    X = np.lib.stride_tricks.sliding_window_view(series[:-1], L)
    return X[..., None].astype("float32"), series[L:].astype("float32")

Xw, yw = make_windows(y, SEQ_LEN)
print()
print(f"  {len(y)} observations -> {len(Xw)} windows of shape {Xw.shape[1:]}")
print(f"  consecutive windows share {SEQ_LEN-1} of {SEQ_LEN} values")

# ============ 2. THE LEAKAGE DEMONSTRATION =============================
print()
print("=== random split vs time split ===")
n = len(Xw)
rng = np.random.default_rng(0)

# WRONG: shuffle then split
perm = rng.permutation(n)
tr_i, te_i = perm[:int(.8*n)], perm[int(.8*n):]
# RIGHT: split by time
cut = int(.8*n)
tr_t, te_t = np.arange(cut), np.arange(cut, n)

def build_linear():
    return keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, 1)),
                             keras.layers.Flatten(),
                             keras.layers.Dense(1)])

print(f"{'split':<22}{'test MAE':>11}{'comment':>34}")
for nm, (a, b) in [("RANDOM (leaky)", (tr_i, te_i)),
                   ("chronological", (tr_t, te_t))]:
    tf.random.set_seed(0)
    m = build_linear()
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(1e-2))
    m.fit(Xw[a], yw[a], epochs=25, batch_size=64, verbose=0)
    err = np.ravel(m.evaluate(Xw[b], yw[b], verbose=0))[0]
    note = "optimistic -- future leaked in" if "RANDOM" in nm else "honest"
    print(f"{nm:<22}{err:>11.4f}{note:>34}")

# ============ 3. A PROPER THREE-WAY TIME SPLIT =========================
print()
print("=== chronological train / valid / test ===")
n_tr, n_va = 500, 100
y_tr, y_va, y_te = y[:n_tr], y[n_tr:n_tr+n_va], y[n_tr+n_va:]
print(f"  train days 0-{n_tr-1}, valid {n_tr}-{n_tr+n_va-1}, "
      f"test {n_tr+n_va}-{len(y)-1}")

# scale using the TRAINING period only
mu, sd = y_tr.mean(), y_tr.std()
print(f"  scaler fitted on TRAIN only: mean {mu:.4f} sd {sd:.4f}")
print(f"  (the full-series mean would be {y.mean():.4f} -- that is leakage)")

def dataset(series, L=SEQ_LEN, batch=32, shuffle=False):
    s = ((series - mu) / sd).astype("float32")
    return keras.utils.timeseries_dataset_from_array(
        s[:-1], targets=s[L:], sequence_length=L,
        batch_size=batch, shuffle=shuffle, seed=42)

train_ds = dataset(y_tr, shuffle=True)     # shuffling WITHIN train is fine
valid_ds = dataset(y_va)
test_ds  = dataset(y_te)
print(f"  train batches {len(list(train_ds))}, valid {len(list(valid_ds))}, "
      f"test {len(list(test_ds))}")

# ============ 4. THE MODELS ============================================
print()
print("=== the first neural forecasters ===")
def evaluate(model, ds_):
    err = np.ravel(model.evaluate(ds_, verbose=0))[0]
    return err * sd            # back to millions of riders

models = {
    "linear (= AR(56))":
        keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, 1)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(1)]),
    "SimpleRNN(1)":
        keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, 1)),
                          keras.layers.SimpleRNN(1)]),
    "SimpleRNN(32) + Dense":
        keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, 1)),
                          keras.layers.SimpleRNN(32),
                          keras.layers.Dense(1)]),
    "Dense(32) + Dense(1)":
        keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, 1)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)]),
}
print(f"{'model':<26}{'params':>9}{'valid MAE (M)':>16}")
for nm, m in models.items():
    tf.random.set_seed(0)
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
    m.fit(train_ds, epochs=25, validation_data=valid_ds, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                   restore_best_weights=True)])
    print(f"{nm:<26}{m.count_params():>9,}{evaluate(m, valid_ds):>16.4f}")

# --- the baselines, on the SAME validation windows -------------------
va = ((y_va - mu) / sd).astype("float32")
Xv, yv_ = make_windows(va, SEQ_LEN)
naive = Xv[:, -1, 0]
seasonal = Xv[:, -7, 0]
print(f"{'naive (t-1)':<26}{0:>9}{np.mean(np.abs(yv_-naive))*sd:>16.4f}")
print(f"{'SEASONAL naive (t-7)':<26}{0:>9}{np.mean(np.abs(yv_-seasonal))*sd:>16.4f}")
print()
print("  the LINEAR model is the one to beat -- it is an AR(56) fitted by SGD")

# ============ 5. WHAT THE LINEAR MODEL LEARNED =========================
lin = models["linear (= AR(56))"]
w = lin.layers[-1].get_weights()[0].ravel()
top = np.argsort(-np.abs(w))[:8]
print()
print("=== the 8 largest AR coefficients ===")
print(f"  {'lag':>6}{'weight':>10}")
for i in sorted(top):
    print(f"  {SEQ_LEN-i:>6}{w[i]:>10.4f}")
print("  large weights at lags 1, 7, 14 -- it rediscovered the weekly cycle")

import plotly.graph_objects as go
fig = go.Figure(go.Bar(x=[SEQ_LEN-i for i in range(SEQ_LEN)], y=w,
                       marker=dict(color=[C["danger"] if (SEQ_LEN-i) % 7 == 0
                                          else C["primary"]
                                          for i in range(SEQ_LEN)])))
fig.update_layout(height=380, xaxis_title="lag (days back)",
                  yaxis_title="learned weight",
                  title="The linear model's coefficients (red = multiples of 7)")
''',
        key="ch15_windowing",
    )

    keypoints([
        "Windowing turns a series into $(\\mathbf{x}, y)$ pairs; consecutive "
        "windows overlap by $L-1$.",
        "<b>Split chronologically, never randomly</b> — overlapping windows make "
        "a random split leak the future.",
        "Fit the scaler on the training period only; for trending series, "
        "difference instead of scaling.",
        "A <code>Dense(1)</code> on a flattened window <b>is</b> an AR($L$) model "
        "— and it is a strong baseline.",
        "If the RNN cannot beat the linear model, it is not earning its "
        "complexity.",
    ])


# ==========================================================================
def s_15_5():
    section("15.5", "Deep RNNs, Multivariate and Multi-Step Forecasting")

    lead(
        "Three independent extensions: more layers, more input series, and more "
        "output steps. Each is a small code change with a distinct consequence."
    )

    sub("Deep RNNs")

    md(
        "Stack recurrent layers exactly as you stack dense ones — remembering "
        "that every layer but the last needs `return_sequences=True` (§15.1)."
    )

    note(
        "Do not put an activation on the final RNN layer's output",
        "A <code>SimpleRNN</code> uses $\\tanh$ by default, so its output is "
        "bounded in $(-1, 1)$ — which makes it a poor final layer for regression "
        "unless your target happens to be in that range. Standard practice is to "
        "end with a <code>Dense(1)</code> (no activation) on top of the last RNN "
        "layer. That also decouples the number of recurrent units from the number "
        "of outputs.",
    )

    sub("Multivariate input")

    md(
        "Nothing changes structurally — the last axis of the input tensor simply "
        "becomes wider. The value is in what those extra series carry: **exogenous "
        "drivers** (weather, price, holidays) that the target series cannot "
        "predict from its own history."
    )

    math(r"""
    \mathbf{X} \in \mathbb{R}^{\,B \times T \times F}
    \qquad
    F = 1 \text{ (univariate)} \;\to\; F > 1 \text{ (multivariate)}
    """)

    pitfall(
        "Exogenous features must be KNOWN at forecast time",
        "Adding tomorrow's temperature as a feature will improve your validation "
        "score enormously — and be useless in production, where tomorrow's "
        "temperature is itself a forecast. The rule: a feature is legitimate only "
        "if its value is <b>available at the moment you make the prediction</b>. "
        "Calendar features (day of week, holiday flags) qualify; measured "
        "quantities generally do not, unless you lag them.",
    )

    sub("Forecasting several steps ahead")

    table(
        ["Strategy", "How", "Pro", "Con"],
        [["<b>Recursive</b>",
          "Predict one step, append it to the input, predict again",
          "One model, any horizon",
          "<b>Errors compound</b> — the model is fed its own mistakes"],
         ["<b>Direct</b>",
          "One model per horizon: $h$ separate models",
          "No error accumulation", "$h$ models to train and maintain"],
         ["<b>Multi-output</b>",
          "One model, <code>Dense(h)</code> output — all horizons at once",
          "One model, no accumulation, shares representation",
          "Fixed horizon; ignores the ordering of outputs"],
         ["<b>Seq2seq</b>",
          "Predict a whole sequence at every time step — §15.6",
          "Far more gradient signal per example", "More complex to set up"]],
    )

    derive(
        [("Why recursive forecasting degrades. Let the one-step model have error "
          "variance $\\sigma^2$, and suppose the true process is "
          "$y_{t+1} = a y_t + \\varepsilon$.", None),
         ("At horizon 1, the error is just the model's own:",
          r"\mathrm{Var}\bigl(e_1\bigr) = \sigma^2"),
         ("At horizon 2, the model is fed its own prediction, so the horizon-1 "
          "error propagates through the dynamics and adds to the fresh error:",
          r"\mathrm{Var}\bigl(e_2\bigr) = \sigma^2 + a^2\sigma^2"),
         ("Continuing, the variance is a geometric sum:",
          r"\mathrm{Var}\bigl(e_h\bigr) = \sigma^2 \sum_{i=0}^{h-1} a^{2i} "
          r"= \sigma^2\,\frac{1 - a^{2h}}{1 - a^{2}}"),
         ("For $|a| < 1$ this converges to $\\sigma^2/(1-a^2)$ — bounded, because "
          "the process is mean-reverting. For $|a| \\ge 1$ it <b>grows without "
          "bound</b>, which is why recursive forecasting of a trending or "
          "random-walk series diverges quickly.", None),
         ("A direct or multi-output model has error variance $\\sigma_h^2$ that "
          "depends on the difficulty of horizon $h$ but does <b>not compound</b>. "
          "That is the whole argument for predicting all horizons at once.",
          None)],
        title="Why recursive forecast errors compound",
    )

    anim_header("Recursive vs multi-output forecasting")

    dfr = ds.ridership(n_days=200)
    yv = (dfr["rail"].to_numpy() / 1e6)
    hist_end = 140
    H = 30
    rng = np.random.default_rng(2)
    truth = yv[hist_end:hist_end + H]
    # stylised: recursive error grows, multi-output stays flat
    rec_err = np.cumsum(rng.normal(0, .012, H)) + rng.normal(0, .01, H)
    dir_err = rng.normal(0, .022, H)
    recursive = truth + rec_err
    multi = truth + dir_err

    frames = []
    for k in range(1, H + 1):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=np.arange(hist_end - 40, hist_end),
                       y=yv[hist_end - 40:hist_end], mode="lines",
                       line=dict(color=C["muted"], width=2)),
            go.Scatter(x=np.arange(hist_end, hist_end + k), y=truth[:k],
                       mode="lines", line=dict(color=C["truth"], width=2.5,
                                               dash="dot")),
            go.Scatter(x=np.arange(hist_end, hist_end + k), y=recursive[:k],
                       mode="lines", line=dict(color=C["danger"], width=3)),
            go.Scatter(x=np.arange(hist_end, hist_end + k), y=multi[:k],
                       mode="lines", line=dict(color=C["success"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"horizon {k}   ·   recursive |error| = "
            f"{abs(rec_err[k-1]):.4f}   ·   multi-output |error| = "
            f"{abs(dir_err[k-1]):.4f}",
            color=C["danger"] if abs(rec_err[k-1]) > abs(dir_err[k-1])
            else C["success"])])))

    f = go.Figure(data=[
        go.Scatter(x=np.arange(hist_end - 40, hist_end),
                   y=yv[hist_end - 40:hist_end], mode="lines", name="history",
                   line=dict(color=C["muted"], width=2)),
        go.Scatter(x=[hist_end], y=truth[:1], mode="lines", name="truth",
                   line=dict(color=C["truth"], width=2.5, dash="dot")),
        go.Scatter(x=[hist_end], y=recursive[:1], mode="lines",
                   name="recursive (errors compound)",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[hist_end], y=multi[:1], mode="lines",
                   name="multi-output (no accumulation)",
                   line=dict(color=C["success"], width=3)),
    ])
    f.add_vline(x=hist_end, line_dash="dash", line_color=C["muted"],
                annotation_text="forecast origin")
    f.update_layout(height=430, xaxis_title="day", yaxis_title="ridership (M)",
                    title="Error accumulation over the forecast horizon",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(160), slider_prefix="horizon ")
    figure(f)

    code_lab(
        "Deep RNNs, exogenous features, and all three multi-step strategies",
        '''import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

df = _ds.ridership(n_days=730)
y = (df["rail"].to_numpy() / 1e6).astype("float32")
SEQ_LEN, HORIZON = 56, 14
n_tr, n_va = 480, 110

# ============ 1. MULTIVARIATE INPUT ====================================
print("=== building exogenous features ===")
idx = df.index
feats = pd.DataFrame(index=idx)
feats["rail"]     = y
feats["dow_sin"]  = np.sin(2*np.pi*idx.dayofweek/7)      # KNOWN in advance
feats["dow_cos"]  = np.cos(2*np.pi*idx.dayofweek/7)
feats["is_weekend"] = (idx.dayofweek >= 5).astype(float)
feats["doy_sin"]  = np.sin(2*np.pi*idx.dayofyear/365.25)
feats["doy_cos"]  = np.cos(2*np.pi*idx.dayofyear/365.25)
print(f"  {feats.shape[1]} features: {list(feats.columns)}")
print("  every one is a CALENDAR feature -- known at forecast time")
print("  a weather feature would NOT be, unless lagged")

F = feats.to_numpy().astype("float32")
mu, sd = F[:n_tr].mean(0), F[:n_tr].std(0) + 1e-7        # TRAIN only
Fs = (F - mu) / sd
y_s = Fs[:, 0]

def windows(arr, target, L, h=1):
    """arr: (T, F). Returns X (n, L, F) and Y (n,) or (n, h)."""
    n = len(arr) - L - h + 1
    X = np.stack([arr[i:i+L] for i in range(n)])
    if h == 1:
        Y = target[L:L+n]
    else:
        Y = np.stack([target[L+i:L+i+h] for i in range(n)])
    return X.astype("float32"), Y.astype("float32")

def split(X, Y):
    a = n_tr - SEQ_LEN
    b = a + n_va
    return (X[:a], Y[:a]), (X[a:b], Y[a:b]), (X[b:], Y[b:])

# ---- univariate vs multivariate --------------------------------------
print()
print("=== does the extra information help? ===")
print(f"{'input':<34}{'features':>10}{'valid MAE (M)':>16}")
for nm, cols in [("rail only", [0]),
                 ("rail + day-of-week", [0, 1, 2]),
                 ("rail + weekend flag", [0, 3]),
                 ("all calendar features", list(range(F.shape[1])))]:
    Xa, Ya = windows(Fs[:, cols], y_s, SEQ_LEN)
    (Xtr, Ytr), (Xva, Yva), _ = split(Xa, Ya)
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, len(cols))),
                          keras.layers.SimpleRNN(32),
                          keras.layers.Dense(1)])
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
    m.fit(Xtr, Ytr, epochs=30, batch_size=32, verbose=0,
          validation_data=(Xva, Yva),
          callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                   restore_best_weights=True)])
    err = np.ravel(m.evaluate(Xva, Yva, verbose=0))[0] * sd[0]
    print(f"{nm:<34}{len(cols):>10}{err:>16.4f}")

# ============ 2. DEEP RNNs =============================================
print()
print("=== stacking recurrent layers ===")
Xa, Ya = windows(Fs, y_s, SEQ_LEN)
(Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split(Xa, Ya)
NF = Xa.shape[-1]

def deep_rnn(n_layers, units=32, cell=keras.layers.SimpleRNN):
    layers = [keras.layers.Input(shape=(SEQ_LEN, NF))]
    for i in range(n_layers):
        last = (i == n_layers - 1)
        layers.append(cell(units, return_sequences=not last))
    layers.append(keras.layers.Dense(1))          # NO tanh on the output
    return keras.Sequential(layers)

print(f"{'depth':>7}{'params':>10}{'valid MAE (M)':>16}")
for depth in [1, 2, 3]:
    tf.random.set_seed(0)
    m = deep_rnn(depth)
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
    m.fit(Xtr, Ytr, epochs=30, batch_size=32, verbose=0,
          validation_data=(Xva, Yva),
          callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                   restore_best_weights=True)])
    print(f"{depth:>7}{m.count_params():>10,}"
          f"{np.ravel(m.evaluate(Xva, Yva, verbose=0))[0]*sd[0]:>16.4f}")

# --- the tanh-output trap --------------------------------------------
print()
print("=== why you end with Dense(1), not the RNN itself ===")
raw_rnn = keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, NF)),
                            keras.layers.SimpleRNN(1)])       # tanh output!
print(f"  SimpleRNN(1) output range: tanh -> (-1, 1)")
print(f"  the scaled target range  : [{y_s.min():.2f}, {y_s.max():.2f}]")
print(f"  values outside (-1,1) are UNREACHABLE: "
      f"{np.mean((y_s < -1) | (y_s > 1)):.1%} of the targets")

# ============ 3. MULTI-STEP: THREE STRATEGIES ==========================
print()
print("="*62)
print(f"Forecasting {HORIZON} steps ahead, three ways")
print("="*62)

Xh, Yh = windows(Fs, y_s, SEQ_LEN, h=HORIZON)
(Xtr_h, Ytr_h), (Xva_h, Yva_h), _ = split(Xh, Yh)
print(f"  X {Xtr_h.shape}  Y {Ytr_h.shape}")

# --- (a) RECURSIVE: one-step model, fed its own predictions ----------
tf.random.set_seed(0)
one_step = keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, NF)),
                             keras.layers.SimpleRNN(32),
                             keras.layers.Dense(1)])
one_step.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
one_step.fit(Xtr, Ytr, epochs=30, batch_size=32, verbose=0)

def recursive_forecast(model, X, h):
    """Feed the model its own predictions, h times."""
    Xc = X.copy()
    preds = []
    for step in range(h):
        p = model.predict(Xc, verbose=0).ravel()
        preds.append(p)
        nxt = Xc[:, -1:, :].copy()
        nxt[:, 0, 0] = p                       # the predicted rail value
        Xc = np.concatenate([Xc[:, 1:, :], nxt], axis=1)
    return np.stack(preds, axis=1)

rec = recursive_forecast(one_step, Xva_h[:200], HORIZON)

# --- (b) MULTI-OUTPUT: Dense(HORIZON) -------------------------------
tf.random.set_seed(0)
multi = keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, NF)),
                          keras.layers.SimpleRNN(32),
                          keras.layers.Dense(HORIZON)])       # ALL at once
multi.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
multi.fit(Xtr_h, Ytr_h, epochs=30, batch_size=32, verbose=0,
          validation_data=(Xva_h, Yva_h),
          callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                   restore_best_weights=True)])
mo = multi.predict(Xva_h[:200], verbose=0)

# --- (c) SEASONAL NAIVE baseline -------------------------------------
naive_h = np.stack([Xva_h[:200, -7 + (i % 7), 0] for i in range(HORIZON)], 1)

print()
print(f"{'horizon':>9}{'recursive':>12}{'multi-output':>15}{'seasonal naive':>17}")
truth = Yva_h[:200]
for h in [1, 3, 7, 14]:
    print(f"{h:>9}"
          f"{np.mean(np.abs(rec[:, h-1]-truth[:, h-1]))*sd[0]:>12.4f}"
          f"{np.mean(np.abs(mo[:, h-1]-truth[:, h-1]))*sd[0]:>15.4f}"
          f"{np.mean(np.abs(naive_h[:, h-1]-truth[:, h-1]))*sd[0]:>17.4f}")
print(f"{'MEAN':>9}"
      f"{np.mean(np.abs(rec-truth))*sd[0]:>12.4f}"
      f"{np.mean(np.abs(mo-truth))*sd[0]:>15.4f}"
      f"{np.mean(np.abs(naive_h-truth))*sd[0]:>17.4f}")
print()
print("  the recursive error GROWS with the horizon; the multi-output one does not")

# ============ 4. ERROR GROWTH, QUANTIFIED ==============================
print()
print("=== error variance vs horizon ===")
rec_e = np.abs(rec - truth).mean(0) * sd[0]
mo_e  = np.abs(mo - truth).mean(0) * sd[0]
print(f"{'h':>4}{'recursive':>12}{'ratio to h=1':>15}{'multi-output':>15}"
      f"{'ratio to h=1':>15}")
for h in range(1, HORIZON+1, 3):
    print(f"{h:>4}{rec_e[h-1]:>12.4f}{rec_e[h-1]/rec_e[0]:>15.2f}"
          f"{mo_e[h-1]:>15.4f}{mo_e[h-1]/mo_e[0]:>15.2f}")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=np.arange(1, HORIZON+1), y=rec_e, mode="lines+markers",
                name="recursive", line=dict(color=C["danger"], width=3))
fig.add_scatter(x=np.arange(1, HORIZON+1), y=mo_e, mode="lines+markers",
                name="multi-output", line=dict(color=C["success"], width=3))
fig.add_scatter(x=np.arange(1, HORIZON+1),
                y=np.abs(naive_h-truth).mean(0)*sd[0], mode="lines",
                name="seasonal naive", line=dict(color=C["muted"], width=2,
                                                 dash="dot"))
fig.update_layout(height=400, xaxis_title="forecast horizon (days)",
                  yaxis_title="MAE (millions)",
                  title="Error accumulation")
''',
        key="ch15_multistep",
    )

    keypoints([
        "Stack RNN layers with <code>return_sequences=True</code> on all but the "
        "last; end with <code>Dense(1)</code>, not a $\\tanh$ output.",
        "Multivariate input just widens the last axis — but features must be "
        "<b>known at forecast time</b>.",
        "<b>Recursive</b> forecasting compounds error: "
        "$\\mathrm{Var}(e_h) = \\sigma^2(1-a^{2h})/(1-a^2)$.",
        "<b>Multi-output</b> (<code>Dense(h)</code>) predicts every horizon at "
        "once with no accumulation.",
        "Compare every horizon against the seasonal-naive baseline separately.",
    ])


# ==========================================================================
def s_15_6():
    section("15.6", "Sequence-to-Sequence Forecasting")

    lead(
        "Instead of one prediction per window, predict a whole forecast at "
        "<b>every</b> time step. It looks like a small change and it multiplies "
        "the gradient signal by the sequence length."
    )

    sub("The idea")

    md(
        "Set `return_sequences=True` on the last RNN layer and wrap the output "
        "layer so it applies at every step. The target becomes a **sequence of "
        "sequences**: at step $t$, predict $(y_{t+1}, \\dots, y_{t+h})$."
    )

    math(r"""
    \mathbf{Y} \in \mathbb{R}^{\,B \times T \times h}
    \qquad\text{instead of}\qquad
    \mathbf{Y} \in \mathbb{R}^{\,B \times h}
    """)

    derive(
        [("<b>Why this helps so much.</b> A vector-output model produces one "
          "prediction per training window, so the loss provides one error signal "
          "per window.", None),
         ("A sequence-output model produces $T$ predictions per window, so the "
          "gradient is a sum of $T$ terms:",
          r"\frac{\partial\mathcal{L}}{\partial\boldsymbol\theta} = "
          r"\sum_{t=1}^{T}\frac{\partial \ell_t}{\partial \boldsymbol\theta}"),
         ("With $T = 56$, each training example contributes 56 error signals "
          "instead of one. Training is faster, more stable, and less prone to "
          "overfitting on small datasets.", None),
         ("<b>The gradient also reaches deeper.</b> In a vector-output model, only "
          "the last time step's hidden state is directly connected to the loss, so "
          "early steps receive gradient only through the long recurrent chain — "
          "which is exactly what vanishes (§15.2). With an output at every step, "
          "step $t$ has a <b>direct</b> path to a loss term, bypassing the "
          "product entirely. This is the same argument as GoogLeNet's auxiliary "
          "classifiers (§10.6) and deep supervision in general.", None),
         ("<b>At inference, use only the last step's output</b> — the earlier "
          "ones were forecasts made from partial windows, which you already know "
          "the answers to.", None)],
        title="Why sequence-to-sequence trains better",
    )

    codenote(
        "TimeDistributed, and why you rarely need it now",
        "Historically you wrapped the output layer: "
        "<code>TimeDistributed(Dense(h))</code>, which applies the same Dense "
        "layer independently at every time step. In current Keras, "
        "<code>Dense</code> applied to a 3-D tensor <b>already</b> operates on the "
        "last axis independently per step — so a bare <code>Dense(h)</code> does "
        "the same thing. <code>TimeDistributed</code> is still needed for layers "
        "that are not last-axis-wise, such as wrapping a whole "
        "<code>Conv2D</code> over a sequence of images.",
    )

    sub("Encoder–decoder")

    md(
        "For genuinely different input and output sequences — translation, "
        "summarisation — the **encoder–decoder** is the right structure: an "
        "encoder RNN compresses the input to a fixed vector, a decoder RNN expands "
        "it into the output sequence."
    )

    warn(
        "The fixed-size bottleneck is the encoder–decoder's fatal flaw",
        "Everything the decoder knows about a 200-word sentence must fit through "
        "one vector of, say, 512 numbers. Performance degrades sharply with input "
        "length, and the degradation is measurable and consistent. The fix is "
        "<b>attention</b> — letting the decoder look back at every encoder state "
        "rather than just the final one — which is Chapter 16, and which "
        "eventually made the recurrence itself unnecessary.",
    )

    anim_header("Vector output vs sequence output: where the gradient flows")

    T = 8
    modes = [
        ("vector output — one loss term",
         [T - 1], "only the last step touches the loss"),
        ("sequence output — T loss terms",
         list(range(T)), "every step has a DIRECT path to the loss"),
    ]
    frames = []
    for title, loss_steps, note_ in modes:
        shapes, ann = [], []
        for t in range(T):
            x0 = t * 1.5
            shapes.append(go.Scatter(
                x=[x0, x0 + 1.0, x0 + 1.0, x0, x0],
                y=[-.3, -.3, .3, .3, -.3], fill="toself",
                fillcolor=alpha(C["primary"], .8),
                line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x0 + .5, y=0, text=f"h<sub>{t}</sub>",
                            showarrow=False,
                            font=dict(size=10, color="#fff")))
            if t < T - 1:
                shapes.append(go.Scatter(x=[x0 + 1.0, x0 + 1.5], y=[0, 0],
                                         mode="lines",
                                         line=dict(color=C["muted"], width=2),
                                         showlegend=False, hoverinfo="skip"))
            if t in loss_steps:
                shapes.append(go.Scatter(x=[x0 + .5, x0 + .5], y=[.3, 1.15],
                                         mode="lines",
                                         line=dict(color=C["danger"], width=3.5),
                                         showlegend=False, hoverinfo="skip"))
                shapes.append(go.Scatter(
                    x=[x0 + .15, x0 + .85, x0 + .85, x0 + .15, x0 + .15],
                    y=[1.15, 1.15, 1.65, 1.65, 1.15], fill="toself",
                    fillcolor=alpha(C["danger"], .85),
                    line=dict(color="#fff", width=2),
                    showlegend=False, hoverinfo="skip"))
                ann.append(dict(x=x0 + .5, y=1.4, text="ℓ", showarrow=False,
                                font=dict(size=11, color="#fff")))
        ann.append(dict(x=T * 1.5 / 2, y=-1.1,
                        text=f"{len(loss_steps)} loss term(s) — {note_}",
                        showarrow=False,
                        font=dict(size=11, color=C["ink_soft"])))
        frames.append(go.Frame(name=title.split()[0], data=shapes,
                               layout=go.Layout(annotations=ann, title=title)))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=330, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, T * 1.5 + .3]),
                    yaxis=dict(visible=False, range=[-1.6, 2.1]),
                    annotations=list(frames[0].layout.annotations),
                    title=modes[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(2000), slider_prefix="mode ")
    figure(f, "With one loss term the gradient must traverse the whole recurrent "
              "chain to reach h₀. With T terms, every state has a short path.")

    code_lab(
        "Sequence-to-sequence forecasting, and the encoder–decoder bottleneck",
        '''import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

df = _ds.ridership(n_days=730)
idx = df.index
F = np.column_stack([
    df["rail"].to_numpy()/1e6,
    np.sin(2*np.pi*idx.dayofweek/7), np.cos(2*np.pi*idx.dayofweek/7),
    (idx.dayofweek >= 5).astype(float),
]).astype("float32")
SEQ_LEN, HORIZON = 56, 14
n_tr, n_va = 480, 110
mu, sd = F[:n_tr].mean(0), F[:n_tr].std(0) + 1e-7
Fs = (F - mu) / sd
target = Fs[:, 0]
NF = F.shape[1]

# ============ 1. THE TWO TARGET SHAPES =================================
def vector_targets(arr, tgt, L, h):
    """Y shape (n, h) -- one forecast per window."""
    n = len(arr) - L - h + 1
    X = np.stack([arr[i:i+L] for i in range(n)])
    Y = np.stack([tgt[i+L:i+L+h] for i in range(n)])
    return X.astype("float32"), Y.astype("float32")

def sequence_targets(arr, tgt, L, h):
    """Y shape (n, L, h) -- a forecast at EVERY step of the window."""
    n = len(arr) - L - h + 1
    X = np.stack([arr[i:i+L] for i in range(n)])
    Y = np.stack([[tgt[i+t+1:i+t+1+h] for t in range(L)] for i in range(n)])
    return X.astype("float32"), Y.astype("float32")

Xv, Yv = vector_targets(Fs, target, SEQ_LEN, HORIZON)
Xs, Ys = sequence_targets(Fs[:-HORIZON], target, SEQ_LEN, HORIZON)
print("=== target shapes ===")
print(f"  vector output  : X {Xv.shape}  Y {Yv.shape}")
print(f"  sequence output: X {Xs.shape}  Y {Ys.shape}")
print(f"  the seq2seq target is {SEQ_LEN}x larger -- {SEQ_LEN} loss terms "
      f"per window instead of 1")

a = n_tr - SEQ_LEN; b = a + n_va
def sp(X, Y): return (X[:a], Y[:a]), (X[a:b], Y[a:b])
(Xv_tr, Yv_tr), (Xv_va, Yv_va) = sp(Xv, Yv)
(Xs_tr, Ys_tr), (Xs_va, Ys_va) = sp(Xs, Ys)

# ============ 2. THE TWO MODELS ========================================
def vector_model(units=32):
    return keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, NF)),
                             keras.layers.SimpleRNN(units,
                                                    return_sequences=True),
                             keras.layers.SimpleRNN(units),      # LAST state
                             keras.layers.Dense(HORIZON)])

def seq2seq_model(units=32):
    return keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, NF)),
                             keras.layers.SimpleRNN(units,
                                                    return_sequences=True),
                             keras.layers.SimpleRNN(units,
                                                    return_sequences=True),
                             keras.layers.Dense(HORIZON)])   # applied per step

print()
print("=== Dense on a 3-D tensor already acts per time step ===")
z = tf.zeros((2, SEQ_LEN, 32))
print(f"  Dense(14) on {tuple(z.shape)} -> "
      f"{tuple(keras.layers.Dense(HORIZON)(z).shape)}")
print(f"  TimeDistributed(Dense(14))  -> "
      f"{tuple(keras.layers.TimeDistributed(keras.layers.Dense(HORIZON))(z).shape)}")
print("  identical -- TimeDistributed is only needed for non-last-axis layers")

# ============ 3. TRAIN BOTH ============================================
print()
print("=== vector output vs sequence output ===")

def last_step_mae(y_true, y_pred):
    """Only the LAST time step matters at inference."""
    return tf.reduce_mean(tf.abs(y_true[:, -1] - y_pred[:, -1]))

results = {}
tf.random.set_seed(0)
mv = vector_model()
mv.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
hv = mv.fit(Xv_tr, Yv_tr, epochs=25, batch_size=32, verbose=0,
            validation_data=(Xv_va, Yv_va))
pv = mv.predict(Xv_va, verbose=0)
results["vector output"] = (mv, np.mean(np.abs(pv - Yv_va))*sd[0],
                            len(hv.history["loss"]))

tf.random.set_seed(0)
ms = seq2seq_model()
ms.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3),
           metrics=[last_step_mae])
hs = ms.fit(Xs_tr, Ys_tr, epochs=25, batch_size=32, verbose=0,
            validation_data=(Xs_va, Ys_va))
ps = ms.predict(Xs_va, verbose=0)[:, -1]        # ONLY the last step
results["sequence output"] = (ms, np.mean(np.abs(ps - Ys_va[:, -1]))*sd[0],
                              len(hs.history["loss"]))

print(f"{'model':<22}{'params':>9}{'valid MAE (M)':>16}{'epochs':>9}")
for nm, (m, err, ep) in results.items():
    print(f"{nm:<22}{m.count_params():>9,}{err:>16.4f}{ep:>9}")
print()
print(f"  the seq2seq model has the SAME parameters but "
      f"{SEQ_LEN}x more loss terms")
print(f"  training loss after 5 epochs: vector {hv.history['loss'][4]:.4f}  "
      f"seq2seq {hs.history['loss'][4]:.4f}")

# ============ 4. PER-HORIZON BREAKDOWN =================================
print()
print("=== MAE by forecast horizon ===")
naive_h = np.stack([Xv_va[:, -7 + (i % 7), 0] for i in range(HORIZON)], 1)
print(f"{'horizon':>9}{'vector':>11}{'seq2seq':>11}{'seasonal naive':>17}")
for h in [1, 3, 7, 14]:
    print(f"{h:>9}"
          f"{np.mean(np.abs(pv[:, h-1]-Yv_va[:, h-1]))*sd[0]:>11.4f}"
          f"{np.mean(np.abs(ps[:, h-1]-Ys_va[:, -1, h-1]))*sd[0]:>11.4f}"
          f"{np.mean(np.abs(naive_h[:, h-1]-Yv_va[:, h-1]))*sd[0]:>17.4f}")

# ============ 5. THE ENCODER-DECODER BOTTLENECK ========================
print()
print("="*62)
print("The encoder-decoder fixed-size bottleneck")
print("="*62)

def encoder_decoder(units=32, latent=None):
    latent = latent or units
    inp = keras.layers.Input(shape=(SEQ_LEN, NF))
    # ENCODER: compress the whole input to ONE vector
    enc = keras.layers.SimpleRNN(latent)(inp)
    # DECODER: expand that vector into a sequence
    dec = keras.layers.RepeatVector(HORIZON)(enc)
    dec = keras.layers.SimpleRNN(units, return_sequences=True)(dec)
    out = keras.layers.Dense(1)(dec)
    return keras.Model(inp, keras.layers.Reshape((HORIZON,))(out))

print()
print("=== the bottleneck width matters ===")
print(f"{'latent size':>13}{'params':>10}{'valid MAE (M)':>16}")
for latent in [2, 4, 8, 16, 32, 64]:
    tf.random.set_seed(0)
    m = encoder_decoder(units=32, latent=latent)
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
    m.fit(Xv_tr, Yv_tr, epochs=25, batch_size=32, verbose=0)
    print(f"{latent:>13}{m.count_params():>10,}"
          f"{np.ravel(m.evaluate(Xv_va, Yv_va, verbose=0))[0]*sd[0]:>16.4f}")
print("  everything the decoder knows must fit through that vector")

# --- and it gets worse with longer inputs ----------------------------
print()
print("=== the bottleneck degrades with input length ===")
print(f"{'input length':>14}{'latent 8 MAE':>15}{'latent 64 MAE':>16}{'ratio':>9}")
for L in [14, 28, 56, 112]:
    row = []
    for latent in [8, 64]:
        Xl, Yl = vector_targets(Fs, target, L, HORIZON)
        aa = n_tr - L; bb = aa + n_va
        tf.random.set_seed(0)
        inp = keras.layers.Input(shape=(L, NF))
        enc = keras.layers.SimpleRNN(latent)(inp)
        dec = keras.layers.RepeatVector(HORIZON)(enc)
        dec = keras.layers.SimpleRNN(32, return_sequences=True)(dec)
        m = keras.Model(inp, keras.layers.Reshape((HORIZON,))(
            keras.layers.Dense(1)(dec)))
        m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3))
        m.fit(Xl[:aa], Yl[:aa], epochs=20, batch_size=32, verbose=0)
        row.append(np.ravel(m.evaluate(Xl[aa:bb], Yl[aa:bb], verbose=0))[0]*sd[0])
    print(f"{L:>14}{row[0]:>15.4f}{row[1]:>16.4f}{row[0]/row[1]:>9.2f}")
print()
print("  the narrow bottleneck falls further behind as the input grows.")
print("  ATTENTION (chapter 16) removes the bottleneck entirely by letting the")
print("  decoder look back at EVERY encoder state, not just the last one.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hv.history["loss"], mode="lines", name="vector: train",
                line=dict(color=C["danger"], width=2.5))
fig.add_scatter(y=hs.history["loss"], mode="lines", name="seq2seq: train",
                line=dict(color=C["success"], width=2.5))
fig.update_layout(height=380, xaxis_title="epoch", yaxis_title="MAE (scaled)",
                  yaxis_type="log",
                  title="Sequence output converges faster (more gradient signal)")
''',
        key="ch15_seq2seq",
    )

    keypoints([
        "Set <code>return_sequences=True</code> on the last RNN layer and predict "
        "a forecast at <b>every</b> step.",
        "$T$ loss terms per window instead of 1 — faster, more stable training.",
        "Every hidden state gets a <b>direct</b> path to a loss term, bypassing "
        "the vanishing product.",
        "<code>Dense</code> on a 3-D tensor is already per-step; "
        "<code>TimeDistributed</code> is for other layer types.",
        "The encoder–decoder's <b>fixed-size bottleneck</b> degrades with input "
        "length — attention (Ch. 16) removes it.",
    ])


# ==========================================================================
def s_15_7():
    section("15.7", "Handling Long Sequences — Unstable Gradients")

    lead(
        "Two distinct problems appear on long sequences: the gradient becomes "
        "unstable, and the memory of early inputs fades. This section fixes the "
        "first; §15.8 fixes the second."
    )

    sub("Why the Chapter 11 toolkit does not transfer directly")

    table(
        ["Technique", "Works in an RNN?", "Why"],
        [["<b>Good initialisation</b>", "✅", "Same argument as §11.1"],
         ["<b>Faster optimisers</b>", "✅", "Unchanged"],
         ["<b>Dropout</b>", "⚠️ with care",
          "Must use the <b>same mask at every time step</b> — "
          "<code>recurrent_dropout</code>, not a plain Dropout layer"],
         ["<b>Gradient clipping</b>", "✅ <b>essential</b>",
          "The primary defence against explosion (§11.4)"],
         ["<b>Saturating activations</b>", "✅ <b>and required</b>",
          "$\\tanh$ is the default precisely because it is bounded — see below"],
         ["<b>Batch normalisation</b>", "❌ <b>does not work</b>",
          "See below"]],
    )

    pitfall(
        "Batch normalisation does not work inside a recurrent layer",
        "You cannot apply it between time steps, because it would need separate "
        "statistics for every step — and the number of steps varies. Applying it "
        "<i>between</i> recurrent layers works slightly but helps little: "
        "Laurent et al. (2015) found BN helped only when applied to the inputs, "
        "not the hidden states. The correct replacement is "
        "<b>layer normalisation</b>, which normalises <i>across the features of "
        "one instance</i> rather than across the batch — so it is completely "
        "independent of both batch size and time step.",
    )

    sub("Layer normalisation")

    math(r"""
    \mathrm{LN}(\mathbf{x}) \;=\;
    \boldsymbol\alpha \otimes
    \frac{\mathbf{x} - \mu}{\sqrt{\sigma^{2} + \varepsilon}}
    \;+\; \boldsymbol\beta,
    \qquad
    \mu = \frac{1}{n}\sum_{i=1}^{n} x_i,
    \quad
    \sigma^{2} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^{2}
    """)

    proof(
        "Layer norm computes its statistics across the wrong axis — deliberately",
        "Batch norm averages over the <b>batch</b> axis: "
        "$\\mu_j = \\frac{1}{B}\\sum_b x_{bj}$, one statistic per feature. Layer "
        "norm averages over the <b>feature</b> axis: "
        "$\\mu_b = \\frac{1}{n}\\sum_j x_{bj}$, one statistic per instance. That "
        "single change means layer norm needs no batch, no moving averages, and "
        "behaves <b>identically at training and inference</b> — which is exactly "
        "what a recurrent (or, later, a Transformer) architecture needs.",
    )

    sub("Why tanh and not ReLU inside an RNN")

    warn(
        "ReLU in a recurrent layer is dangerous",
        "In a feedforward net, ReLU's unbounded output is fine — each layer has "
        "its own weights. In an RNN the <b>same</b> $\\mathbf{W}_h$ is applied "
        "repeatedly, so an unbounded activation can amplify without limit: if the "
        "state grows at each step, it keeps growing, and the network diverges "
        "within a few dozen steps. $\\tanh$ bounds the state to $(-1,1)$, which "
        "makes the recurrence contractive. If you do want ReLU, you <b>must</b> "
        "use gradient clipping and a carefully scaled (often orthogonal) "
        "recurrent initialisation.",
    )

    sub("Recurrent dropout")

    md(
        "Applying a *different* dropout mask at every time step injects noise that "
        "compounds along the sequence, destroying the memory. **Variational "
        "dropout** (Gal & Ghahramani, 2016) uses the **same mask at every step**, "
        "which regularises without disrupting the recurrence."
    )

    table(
        ["Argument", "Applies to", "Mask"],
        [["<code>dropout=</code>", "the <b>inputs</b> $\\mathbf{x}_{(t)}$",
          "Same mask at every step"],
         ["<code>recurrent_dropout=</code>",
          "the <b>hidden state</b> $\\mathbf{h}_{(t-1)}$",
          "Same mask at every step"],
         ["A separate <code>Dropout</code> layer", "between RNN layers",
          "Fine — it is not inside the recurrence"]],
    )

    anim_header("Batch norm vs layer norm: which axis is normalised")

    rng = np.random.default_rng(0)
    B, Ffeat = 6, 8
    Xm = rng.normal(3, 2, (B, Ffeat)) * rng.uniform(.4, 2.5, Ffeat)

    bn = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-7)
    ln = (Xm - Xm.mean(1, keepdims=True)) / (Xm.std(1, keepdims=True) + 1e-7)

    views = [
        ("raw activations (batch × features)", Xm, None, None),
        ("BATCH norm — statistics down each COLUMN", Xm, "col", bn),
        ("after batch norm", bn, None, None),
        ("LAYER norm — statistics across each ROW", Xm, "row", ln),
        ("after layer norm", ln, None, None),
    ]
    frames = []
    for i, (nm, Z, highlight, result) in enumerate(views):
        data = [go.Heatmap(z=Z[::-1], colorscale=nav.cscale(), showscale=False,
                           xgap=2, ygap=2, text=np.round(Z, 1)[::-1],
                           texttemplate="%{text}", textfont=dict(size=9))]
        if highlight == "col":
            data.append(go.Scatter(x=[2.5, 3.5, 3.5, 2.5, 2.5],
                                   y=[-.5, -.5, B - .5, B - .5, -.5],
                                   mode="lines",
                                   line=dict(color=C["danger"], width=4),
                                   showlegend=False, hoverinfo="skip"))
        elif highlight == "row":
            data.append(go.Scatter(x=[-.5, Ffeat - .5, Ffeat - .5, -.5, -.5],
                                   y=[2.5, 2.5, 3.5, 3.5, 2.5], mode="lines",
                                   line=dict(color=C["danger"], width=4),
                                   showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(i + 1), data=data,
                               layout=go.Layout(title=nm)))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, title=views[0][0],
                    xaxis=dict(title="feature", showgrid=False),
                    yaxis=dict(title="batch instance", showgrid=False))
    anim.animate(f, frames, duration=nav.anim_ms(1700), slider_prefix="step ")
    figure(f, "Batch norm needs the whole column — the whole batch. Layer norm "
              "needs only the row, so it works with a batch of one.")

    code_lab(
        "Layer norm from scratch, a custom LN cell, and recurrent dropout",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. BATCH NORM vs LAYER NORM ==============================
rng = np.random.default_rng(0)
X = rng.normal(3, 2, (6, 8)).astype("float32") * rng.uniform(.4, 2.5, 8)

bn_manual = (X - X.mean(0)) / (X.std(0) + 1e-7)
ln_manual = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-7)

print("=== the two normalisations ===")
print(f"  raw       : per-feature mean {X.mean(0).round(2)}")
print(f"  BATCH norm: per-feature mean {bn_manual.mean(0).round(4)}  "
      f"per-instance mean {bn_manual.mean(1).round(2)}")
print(f"  LAYER norm: per-feature mean {ln_manual.mean(0).round(2)}  "
      f"per-instance mean {ln_manual.mean(1).round(4)}")
print("  batch norm zeroes the COLUMN means; layer norm zeroes the ROW means")

ln_keras = keras.layers.LayerNormalization(epsilon=1e-7)
print(f"\\n  max |mine - keras.LayerNormalization| = "
      f"{np.abs(ln_manual - ln_keras(tf.constant(X)).numpy()).max():.5f}")

# --- layer norm works with a batch of ONE ----------------------------
print()
print("=== a batch of one ===")
one = tf.constant(X[:1])
print(f"  LayerNormalization : {ln_keras(one).numpy()[0][:4].round(3)}  fine")
bn_layer = keras.layers.BatchNormalization()
bn_layer(tf.constant(X), training=True)      # let it see a real batch first
print(f"  BatchNormalization on a batch of 1, training=True:")
out1 = bn_layer(one, training=True).numpy()[0][:4]
print(f"    {out1.round(3)}   <- variance of one sample is 0; meaningless")

# ============ 2. A CUSTOM LAYER-NORM RNN CELL ==========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class LNSimpleRNNCell(keras.layers.Layer):
    """A SimpleRNN cell with layer normalisation inside the recurrence."""
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                          activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]      # NORMALISE, THEN activate

    def get_config(self):
        return {**super().get_config(), "units": self.units}

print()
print("=== a layer-normalised RNN cell ===")
cell = LNSimpleRNNCell(16)
rnn = keras.layers.RNN(cell, return_sequences=True)
out = rnn(tf.random.normal((4, 20, 3)))
print(f"  RNN(LNSimpleRNNCell(16)) on (4,20,3) -> {tuple(out.shape)}")
print(f"  parameters: {rnn.count_params():,} "
      f"(SimpleRNN(16) has {keras.layers.SimpleRNN(16).__class__.__name__})")

# ============ 3. DOES IT HELP ON A LONG SEQUENCE? ======================
df = _ds.ridership(n_days=730)
idx = df.index
F = np.column_stack([df["rail"].to_numpy()/1e6,
                     np.sin(2*np.pi*idx.dayofweek/7),
                     np.cos(2*np.pi*idx.dayofweek/7)]).astype("float32")
L = 112                                   # a LONG window
n_tr, n_va = 480, 110
mu, sd = F[:n_tr].mean(0), F[:n_tr].std(0) + 1e-7
Fs = (F - mu)/sd
tgt = Fs[:, 0]
n = len(Fs) - L - 1
X = np.stack([Fs[i:i+L] for i in range(n)]).astype("float32")
Y = tgt[L:L+n].astype("float32")
a = n_tr - L; b = a + n_va
Xtr, Ytr, Xva, Yva = X[:a], Y[:a], X[a:b], Y[a:b]
print()
print(f"=== window length {L}, {len(Xtr)} training windows ===")

print(f"{'cell':<34}{'params':>9}{'valid MAE (M)':>16}")
configs = [
    ("SimpleRNN (tanh)",
     lambda: keras.layers.SimpleRNN(32)),
    ("SimpleRNN (relu) -- dangerous",
     lambda: keras.layers.SimpleRNN(32, activation="relu")),
    ("SimpleRNN + recurrent_dropout",
     lambda: keras.layers.SimpleRNN(32, dropout=.1, recurrent_dropout=.1)),
    ("RNN(LNSimpleRNNCell)",
     lambda: keras.layers.RNN(LNSimpleRNNCell(32))),
    ("LSTM",
     lambda: keras.layers.LSTM(32)),
]
for nm, make_cell in configs:
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(L, F.shape[1])),
                          make_cell(), keras.layers.Dense(1)])
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3, clipnorm=1.0))
    m.fit(Xtr, Ytr, epochs=25, batch_size=32, verbose=0)
    err = np.ravel(m.evaluate(Xva, Yva, verbose=0))[0]*sd[0]
    err_s = "diverged" if not np.isfinite(err) else f"{err:.4f}"
    print(f"{nm:<34}{m.count_params():>9,}{err_s:>16}")

# ============ 4. RELU IN AN RNN DIVERGES ===============================
print()
print("=== why tanh, not ReLU, inside the recurrence ===")
def state_growth(activation, T=60, scale=1.2, seed=0):
    r = np.random.default_rng(seed)
    U = 32
    Wh = r.normal(0, 1, (U, U)); Wh = Wh / np.linalg.norm(Wh, 2) * scale
    h = r.normal(0, .1, (1, U))
    norms = []
    for _ in range(T):
        z = h @ Wh
        h = np.tanh(z) if activation == "tanh" else np.maximum(0, z)
        norms.append(np.linalg.norm(h))
    return np.array(norms)

print(f"  ||W_h|| = 1.2, no input, 60 steps of pure recurrence:")
print(f"{'step':>7}{'tanh':>14}{'relu':>18}")
t_n = state_growth("tanh"); r_n = state_growth("relu")
for s in [1, 10, 20, 40, 60]:
    print(f"{s:>7}{t_n[s-1]:>14.4f}{r_n[s-1]:>18.4e}")
print("  tanh saturates and the state stays bounded; ReLU compounds without limit")

# ============ 5. GRADIENT CLIPPING IS ESSENTIAL ========================
print()
print("=== with and without clipping ===")
print(f"{'setup':<34}{'final train loss':>19}")
for nm, kw in [("no clipping", {}),
               ("clipnorm=1.0", dict(clipnorm=1.0)),
               ("clipvalue=0.5", dict(clipvalue=0.5))]:
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(L, F.shape[1])),
                          keras.layers.SimpleRNN(32, activation="relu"),
                          keras.layers.Dense(1)])
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(5e-3, **kw))
    h = m.fit(Xtr, Ytr, epochs=15, batch_size=32, verbose=0)
    v = h.history["loss"][-1]
    print(f"{nm:<34}{('NaN' if not np.isfinite(v) else f'{v:.4f}'):>19}")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=t_n, mode="lines", name="tanh",
                line=dict(color=C["success"], width=3))
fig.add_scatter(y=r_n, mode="lines", name="ReLU",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=380, yaxis_type="log", xaxis_title="recurrent step",
                  yaxis_title="||h||",
                  title="State magnitude under repeated application of W_h")
''',
        key="ch15_longseq",
    )

    keypoints([
        "<b>Batch norm does not work</b> inside a recurrent layer — use "
        "<b>layer normalisation</b>.",
        "Layer norm computes statistics across <b>features of one instance</b>, so "
        "it is batch- and time-step-independent.",
        "<b>$\\tanh$, not ReLU</b>, inside the recurrence: the same "
        "$\\mathbf{W}_h$ is applied repeatedly and must be contractive.",
        "<b>Gradient clipping is essential</b> for RNNs — the gradient product "
        "runs over the sequence length.",
        "Recurrent dropout must use the <b>same mask at every step</b> "
        "(<code>recurrent_dropout=</code>), never a plain Dropout layer.",
    ])

# ==========================================================================
def s_15_8():
    section("15.8", "LSTM and GRU — Solving Short-Term Memory")

    lead(
        "A plain RNN's state is overwritten at every step, so information decays "
        "geometrically. LSTM adds a second state that is <b>added to</b> rather "
        "than <b>replaced</b> — and that one change is the difference between "
        "20 steps of memory and 1 000."
    )

    sub("The LSTM cell")

    md(
        "An LSTM splits the state in two: $\\mathbf{h}_{(t)}$ is the **short-term** "
        "state (and the output), $\\mathbf{c}_{(t)}$ is the **long-term** state. "
        "Three gates control what happens to $\\mathbf{c}$:"
    )

    math(r"""
    \mathbf{i}_{(t)} = \sigma\bigl(\mathbf{W}_{xi}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hi}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_i\bigr)
    \qquad\text{(input gate)}
    """)
    math(r"""
    \mathbf{f}_{(t)} = \sigma\bigl(\mathbf{W}_{xf}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hf}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_f\bigr)
    \qquad\text{(forget gate)}
    """)
    math(r"""
    \mathbf{o}_{(t)} = \sigma\bigl(\mathbf{W}_{xo}^\top\mathbf{x}_{(t)} + \mathbf{W}_{ho}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_o\bigr)
    \qquad\text{(output gate)}
    """)
    math(r"""
    \mathbf{g}_{(t)} = \tanh\bigl(\mathbf{W}_{xg}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hg}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_g\bigr)
    \qquad\text{(candidate)}
    """)

    md("And then the two state updates — this is where the magic is:")

    math(r"""
    \boxed{\;\mathbf{c}_{(t)} \;=\; \mathbf{f}_{(t)} \otimes \mathbf{c}_{(t-1)}
      \;+\; \mathbf{i}_{(t)} \otimes \mathbf{g}_{(t)}\;}
    \qquad\qquad
    \mathbf{h}_{(t)} \;=\; \mathbf{o}_{(t)} \otimes \tanh\bigl(\mathbf{c}_{(t)}\bigr)
    """)

    where({
        r"\otimes": "element-wise multiplication",
        r"\sigma": "the logistic sigmoid, so every gate is in $(0,1)$ — a soft "
                   "switch",
        r"\mathbf{f}_{(t)}": "<b>what to keep</b> from the old long-term state",
        r"\mathbf{i}_{(t)}": "<b>how much of the candidate</b> to write in",
        r"\mathbf{o}_{(t)}": "<b>what part of $\\mathbf{c}$ to expose</b> as the "
                             "output",
    })

    derive(
        [("<b>The whole reason LSTM works</b> is the derivative of the cell-state "
          "update. Differentiate $\\mathbf{c}_{(t)}$ with respect to "
          "$\\mathbf{c}_{(t-1)}$:",
          r"\frac{\partial \mathbf{c}_{(t)}}{\partial \mathbf{c}_{(t-1)}} "
          r"\;=\; \mathrm{diag}\bigl(\mathbf{f}_{(t)}\bigr)"),
         ("Compare with a plain RNN, where the same derivative was "
          "$\\mathbf{W}_h^\\top\\mathrm{diag}(1 - \\mathbf{h}^2)$ — a matrix "
          "product with a bounded activation derivative.", None),
         ("Over $k$ steps the LSTM's cell-state Jacobian is simply:",
          r"\frac{\partial \mathbf{c}_{(t)}}{\partial \mathbf{c}_{(t-k)}} "
          r"\;=\; \prod_{i=0}^{k-1}\mathrm{diag}\bigl(\mathbf{f}_{(t-i)}\bigr)"),
         ("<b>If the forget gate stays near 1, this product stays near 1</b> — the "
          "gradient flows back unattenuated for hundreds of steps. This is the "
          "'constant error carousel' of Hochreiter & Schmidhuber's original "
          "paper, and it is exactly the same mechanism as a residual connection "
          "(§14.5): an additive path with a Jacobian near the identity.", None),
         ("Crucially, $\\mathbf{f}_{(t)}$ is <b>learned per dimension and per "
          "step</b>. The network can keep some dimensions of $\\mathbf{c}$ intact "
          "for a thousand steps while overwriting others every step. A plain "
          "RNN has one global decay rate imposed by $\\mathbf{W}_h$.", None),
         ("<b>Why the forget-gate bias is initialised to 1.</b> "
          "$\\sigma(1) \\approx 0.73$, so at initialisation the cell keeps ~73 % "
          "of its state each step rather than $\\sigma(0) = 0.5$. That biases the "
          "network toward remembering, which makes early training much easier — "
          "Keras does this by default via <code>unit_forget_bias=True</code>.",
          None)],
        title="Why LSTM's gradient does not vanish",
    )

    sub("The GRU cell")

    md(
        "Cho et al. (2014) simplified the LSTM: merge the two states into one, "
        "and merge the input and forget gates into a single **update gate**."
    )

    math(r"""
    \mathbf{z}_{(t)} = \sigma\bigl(\mathbf{W}_{xz}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hz}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_z\bigr)
    \qquad\text{(update gate)}
    """)
    math(r"""
    \mathbf{r}_{(t)} = \sigma\bigl(\mathbf{W}_{xr}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hr}^\top\mathbf{h}_{(t-1)} + \mathbf{b}_r\bigr)
    \qquad\text{(reset gate)}
    """)
    math(r"""
    \mathbf{g}_{(t)} = \tanh\bigl(\mathbf{W}_{xg}^\top\mathbf{x}_{(t)} + \mathbf{W}_{hg}^\top(\mathbf{r}_{(t)}\otimes\mathbf{h}_{(t-1)}) + \mathbf{b}_g\bigr)
    """)
    math(r"""
    \mathbf{h}_{(t)} \;=\; \mathbf{z}_{(t)} \otimes \mathbf{h}_{(t-1)}
      \;+\; \bigl(1 - \mathbf{z}_{(t)}\bigr) \otimes \mathbf{g}_{(t)}
    """)

    idea(
        "The GRU's single gate does two jobs at once",
        "In an LSTM, forgetting and writing are independent: you can keep the old "
        "value <i>and</i> add a new one. In a GRU they are <b>tied</b> — "
        "$\\mathbf{z}$ and $1-\\mathbf{z}$ — so whenever a dimension is written, "
        "the old value is erased by exactly the same amount. That is a real loss "
        "of expressiveness, and in practice it almost never matters: GRUs match "
        "LSTMs on most tasks with ~25 % fewer parameters and correspondingly "
        "faster training.",
    )

    table(
        ["", "SimpleRNN", "GRU", "LSTM"],
        [["States", "1 ($\\mathbf{h}$)", "1 ($\\mathbf{h}$)",
          "<b>2</b> ($\\mathbf{h}$, $\\mathbf{c}$)"],
         ["Gates", "0", "2 (update, reset)",
          "3 (input, forget, output)"],
         ["Parameters (units $u$, inputs $n$)",
          "$un + u^2 + u$", "$3(un + u^2 + u)$", "$4(un + u^2 + u)$"],
         ["Long-range memory", "~10–20 steps", "hundreds", "hundreds to 1 000+"],
         ["Speed", "fastest", "~25 % faster than LSTM", "slowest"],
         ["When to use", "Almost never",
          "<b>Default choice</b> — try this first",
          "Very long dependencies, or when GRU underperforms"]],
    )

    sub("Peephole connections")

    md(
        "A variant (Gers & Schmidhuber, 2000) lets the gates also see the "
        "long-term state $\\mathbf{c}_{(t-1)}$, giving them more context. It "
        "sometimes helps a little. Keras does not include it by default; it is "
        "largely of historical interest now that attention has displaced "
        "recurrence for most long-range problems."
    )

    anim_header("The forget gate keeping information alive")
    md(
        "The magnitude of a single cell-state dimension over 200 steps, for four "
        "forget-gate values. Compare with the plain RNN's fixed decay: **the LSTM "
        "chooses its own decay rate, per dimension, per step**."
    )

    steps = np.arange(0, 201)
    gates = {
        "forget gate f = 0.5": .5 ** steps,
        "forget gate f = 0.9": .9 ** steps,
        "forget gate f = 0.99": .99 ** steps,
        "forget gate f = 0.999 (near-perfect memory)": .999 ** steps,
    }
    plain = .85 ** steps

    frames = []
    for k in range(4, 202, 4):
        data = [go.Scatter(x=steps[:k], y=plain[:k], mode="lines",
                           line=dict(color=C["muted"], width=2.5, dash="dot"))]
        info = []
        for i, (nm, v) in enumerate(gates.items()):
            data.append(go.Scatter(x=steps[:k], y=v[:k], mode="lines",
                                   line=dict(color=SEQ[i], width=3)))
            info.append(f"f={nm.split('=')[1].split()[0]}: {v[k-1]:.2e}")
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"step {k}   |   " + "   ".join(info))])))

    f = go.Figure(data=[
        go.Scatter(x=steps[:4], y=plain[:4], mode="lines",
                   name="plain RNN (fixed decay ≈ 0.85)",
                   line=dict(color=C["muted"], width=2.5, dash="dot"))]
        + [go.Scatter(x=steps[:4], y=v[:4], mode="lines", name=nm,
                      line=dict(color=SEQ[i], width=3))
           for i, (nm, v) in enumerate(gates.items())])
    f.add_hline(y=1e-7, line_dash="dot", line_color=C["danger"],
                annotation_text="float32 noise floor")
    f.update_layout(height=450, yaxis_type="log", xaxis_title="time step",
                    yaxis_title="retained signal", yaxis=dict(range=[-16, .5]),
                    title="∂c(t)/∂c(t−k) = ∏ f — the LSTM chooses the decay",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(50), slider_prefix="step ")
    figure(f, "With f = 0.999 the signal survives 200 steps at 82 % strength. "
              "The plain RNN's fixed 0.85 decay is below the noise floor by "
              "step 100.")

    code_lab(
        "LSTM and GRU cells from scratch, and the memory they actually have",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

# ============ 1. AN LSTM CELL FROM SCRATCH =============================
def lstm_step(x, h, c, W, U, b, units):
    """One LSTM step. Keras packs the four gates into one matrix."""
    z = x @ W + h @ U + b                     # (batch, 4*units)
    i = sigmoid(z[:, :units])                 # input gate
    f = sigmoid(z[:, units:2*units])          # forget gate
    g = np.tanh(z[:, 2*units:3*units])        # candidate
    o = sigmoid(z[:, 3*units:])               # output gate
    c_new = f * c + i * g                     # THE ADDITIVE UPDATE
    h_new = o * np.tanh(c_new)
    return h_new, c_new, dict(i=i, f=f, g=g, o=o)

BATCH, T, N_IN, U = 3, 12, 4, 5
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (BATCH, T, N_IN)).astype("float32")

layer = keras.layers.LSTM(U, return_sequences=True, return_state=True)
_ = layer(tf.zeros((1, T, N_IN)))
W, Uu, b = layer.get_weights()
print("=== an LSTM cell from scratch ===")
print(f"  Keras packs the gates: W {W.shape}, U {Uu.shape}, b {b.shape}")
print(f"  4 gates x {U} units = {4*U} columns")

h = np.zeros((BATCH, U)); c = np.zeros((BATCH, U))
mine = []
for t in range(T):
    h, c, gates = lstm_step(X[:, t], h, c, W, Uu, b, U)
    mine.append(h.copy())
mine = np.stack(mine, 1)

k_seq, k_h, k_c = layer(tf.constant(X))
print(f"  max |mine - Keras| = {np.abs(mine - k_seq.numpy()).max():.2e}")
print(f"  final h matches: {np.allclose(h, k_h.numpy(), atol=1e-5)}")
print(f"  final c matches: {np.allclose(c, k_c.numpy(), atol=1e-5)}")

# ============ 2. THE GATES ARE SOFT SWITCHES ===========================
print()
print("=== the gate values at the last step ===")
for nm, key in [("input  i", "i"), ("forget f", "f"),
                ("output o", "o"), ("candidate g", "g")]:
    v = gates[key][0]
    print(f"  {nm:<12} {np.round(v, 3)}   range "
          f"[{v.min():.3f}, {v.max():.3f}]")
print("  i, f, o are sigmoids in (0,1) -- soft switches, differentiable")
print("  g is a tanh in (-1,1) -- the value to (maybe) write")

# ============ 3. THE FORGET-GATE BIAS ==================================
print()
print("=== unit_forget_bias ===")
for flag in (True, False):
    l = keras.layers.LSTM(8, unit_forget_bias=flag)
    l(tf.zeros((1, 5, 3)))
    bias = l.get_weights()[2]
    f_bias = bias[8:16]
    print(f"  unit_forget_bias={str(flag):<6} forget bias {np.round(f_bias, 2)}"
          f"  -> sigma = {sigmoid(f_bias).mean():.3f}")
print("  initialising the forget bias to 1 makes the cell KEEP ~73 % by default")
print("  which biases the network toward remembering early in training")

# ============ 4. A GRU CELL FROM SCRATCH ===============================
print()
print("=== a GRU cell from scratch ===")
def gru_step(x, h, W, U, b, units):
    """Keras uses 'reset after' by default, with a separate recurrent bias."""
    b_x, b_h = b[0], b[1]
    xz = x @ W[:, :units]            + b_x[:units]
    xr = x @ W[:, units:2*units]     + b_x[units:2*units]
    xg = x @ W[:, 2*units:]          + b_x[2*units:]
    hz = h @ U[:, :units]            + b_h[:units]
    hr = h @ U[:, units:2*units]     + b_h[units:2*units]
    hg = h @ U[:, 2*units:]          + b_h[2*units:]
    z = sigmoid(xz + hz)                       # update gate
    r = sigmoid(xr + hr)                       # reset gate
    g = np.tanh(xg + r * hg)                   # candidate (reset AFTER matmul)
    return z * h + (1 - z) * g                 # TIED: z and 1-z

gl = keras.layers.GRU(U, return_sequences=True)
_ = gl(tf.zeros((1, T, N_IN)))
Wg, Ug, bg = gl.get_weights()
h = np.zeros((BATCH, U))
mine_g = []
for t in range(T):
    h = gru_step(X[:, t], h, Wg, Ug, bg, U)
    mine_g.append(h.copy())
mine_g = np.stack(mine_g, 1)
print(f"  max |mine - Keras| = {np.abs(mine_g - gl(tf.constant(X)).numpy()).max():.2e}")
print(f"  the GRU ties forgetting and writing: z and (1-z)")
print(f"  the LSTM keeps them independent: f and i")

# ============ 5. PARAMETER COUNTS ======================================
print()
print("=== parameters: 1x, 3x, 4x ===")
print(f"{'cell':<14}{'formula':>26}{'n=10,u=64':>13}{'ratio':>8}")
n_in, u = 10, 64
base = n_in*u + u*u + u
for nm, mult, cell in [("SimpleRNN", 1, keras.layers.SimpleRNN),
                       ("GRU", 3, keras.layers.GRU),
                       ("LSTM", 4, keras.layers.LSTM)]:
    l = cell(u); l(tf.zeros((1, 5, n_in)))
    print(f"{nm:<14}{f'{mult} x (n*u + u^2 + u)':>26}{l.count_params():>13,}"
          f"{l.count_params()/base:>8.2f}")
print("  (the GRU's extra recurrent bias makes it slightly above exactly 3x)")

# ============ 6. HOW FAR BACK CAN EACH ONE REMEMBER? ===================
print()
print("="*62)
print("The copy task: output the FIRST input, T steps later")
print("="*62)
def copy_task(n, T, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, T, 1)).astype("float32")
    y = X[:, 0, 0].copy()
    X[:, 1:, 0] = r.normal(0, 1, (n, T-1))
    return X, y

print(f"{'T':>6}{'SimpleRNN':>12}{'GRU':>10}{'LSTM':>10}   (R^2, higher is better)")
for T_ in [10, 30, 60, 120]:
    Xa, ya = copy_task(2500, T_, seed=1)
    Xb, yb = copy_task(700, T_, seed=2)
    row = []
    for cell in [keras.layers.SimpleRNN, keras.layers.GRU, keras.layers.LSTM]:
        tf.random.set_seed(0)
        m = keras.Sequential([keras.layers.Input(shape=(T_, 1)),
                              cell(32), keras.layers.Dense(1)])
        m.compile(loss="mse", optimizer=keras.optimizers.Adam(5e-3, clipnorm=1.))
        m.fit(Xa, ya, epochs=20, batch_size=64, verbose=0)
        p = m.predict(Xb, verbose=0).ravel()
        row.append(1 - np.mean((p-yb)**2)/np.var(yb))
    print(f"{T_:>6}{row[0]:>12.4f}{row[1]:>10.4f}{row[2]:>10.4f}")
print()
print("  the SimpleRNN falls apart past ~30 steps; the gated cells do not")

# ============ 7. WHAT THE FORGET GATE ACTUALLY LEARNS ==================
print()
print("=== the learned forget gate on the copy task ===")
T_ = 60
Xa, ya = copy_task(2500, T_, seed=1)
tf.random.set_seed(0)
probe = keras.Sequential([keras.layers.Input(shape=(T_, 1)),
                          keras.layers.LSTM(16, return_sequences=False,
                                            name="lstm"),
                          keras.layers.Dense(1)])
probe.compile(loss="mse", optimizer=keras.optimizers.Adam(5e-3, clipnorm=1.))
probe.fit(Xa, ya, epochs=25, batch_size=64, verbose=0)

lw, lu, lb = probe.get_layer("lstm").get_weights()
UU = 16
h = np.zeros((1, UU)); c = np.zeros((1, UU))
f_history = []
for t in range(T_):
    h, c, g = lstm_step(Xa[:1, t], h, c, lw, lu, lb, UU)
    f_history.append(g["f"][0])
f_history = np.array(f_history)
print(f"  mean forget-gate value per unit (over 60 steps):")
print(f"    {np.round(f_history.mean(0), 3)}")
print(f"  the highest-retention unit keeps {f_history.mean(0).max():.3f} per step")
print(f"    -> after 60 steps it retains "
      f"{f_history.mean(0).max()**60:.4f} of the original signal")
print(f"  the lowest keeps {f_history.mean(0).min():.3f} "
      f"-> {f_history.mean(0).min()**60:.2e}")
print()
print("  DIFFERENT UNITS LEARNED DIFFERENT TIMESCALES. That is the whole point:")
print("  a plain RNN has one global decay imposed by W_h.")

import plotly.graph_objects as go
fig = go.Figure()
for u_ in range(UU):
    fig.add_scatter(y=f_history[:, u_], mode="lines",
                    line=dict(color=SEQ[u_ % len(SEQ)], width=1.6),
                    showlegend=False)
fig.update_layout(height=380, xaxis_title="time step", yaxis_title="forget gate f",
                  yaxis=dict(range=[0, 1]),
                  title="Each LSTM unit learns its own retention rate")
''',
        key="ch15_lstm",
    )

    quiz(
        "Why does an LSTM's gradient not vanish the way a plain RNN's does?",
        ["It uses ReLU instead of tanh",
         "$\\partial\\mathbf{c}_{(t)}/\\partial\\mathbf{c}_{(t-1)} = "
         "\\mathrm{diag}(\\mathbf{f}_{(t)})$, and the network can learn "
         "$\\mathbf{f} \\approx 1$",
         "It has more parameters",
         "It uses gradient clipping internally"],
        1,
        "The cell state is <i>added to</i>, not replaced, so its Jacobian is just "
        "the forget gate. If the network learns $\\mathbf{f} \\approx 1$ for some "
        "dimensions, the gradient flows through them unattenuated — the same "
        "additive-path mechanism as a residual connection.",
        key="ch15q2",
    )

    keypoints([
        "LSTM splits the state: $\\mathbf{h}$ (short-term, the output) and "
        "$\\mathbf{c}$ (long-term).",
        "$\\mathbf{c}_{(t)} = \\mathbf{f}\\otimes\\mathbf{c}_{(t-1)} + "
        "\\mathbf{i}\\otimes\\mathbf{g}$ — an <b>additive</b> update, so "
        "$\\partial\\mathbf{c}_{(t)}/\\partial\\mathbf{c}_{(t-1)} = "
        "\\mathrm{diag}(\\mathbf{f})$.",
        "Each unit learns <b>its own timescale</b>; a plain RNN has one global "
        "decay.",
        "<b>GRU</b> merges the states and ties the input/forget gates: ~25 % fewer "
        "parameters, usually the same accuracy.",
        "The forget bias is initialised to 1 so the cell starts out inclined to "
        "remember.",
    ])


# ==========================================================================
def s_15_9():
    section("15.9", "1-D Convolutions, WaveNet & Exercises")

    lead(
        "Recurrence is not the only way to process a sequence. A convolution "
        "along the time axis is fully parallel, has no vanishing-gradient product "
        "at all, and — with dilation — can see thousands of steps back."
    )

    sub("Conv1D")

    md(
        "Exactly the convolution of Chapter 14, applied along time instead of "
        "space. A kernel of width $f$ slides over the sequence:"
    )

    math(r"""
    z_{t,k} \;=\; b_k \;+\; \sum_{u=0}^{f-1}\sum_{c=0}^{F-1}
      x_{\,t s + u,\; c}\; w_{u,c,k}
    """)

    table(
        ["", "RNN", "Conv1D"],
        [["Parallelism", "❌ sequential by nature",
          "<b>✅ fully parallel</b> across time"],
         ["Gradient path", "Product of $T$ terms — vanishes",
          "<b>Depth</b> of the stack only — short"],
         ["Receptive field", "Unbounded in principle",
          "Fixed by kernel size × depth (but see dilation)"],
         ["Cost", "$\\mathcal{O}(T)$ sequential steps",
          "$\\mathcal{O}(1)$ sequential steps, $\\mathcal{O}(T)$ work"],
         ["Variable lengths", "Natural", "Natural (with 'same' padding)"]],
    )

    warn(
        "Use <code>padding='causal'</code>, not <code>'same'</code>, for "
        "forecasting",
        "<code>'same'</code> padding pads <b>both</b> ends, so output $t$ depends "
        "on inputs after $t$ — the model can see the future, your validation "
        "score is fiction, and the failure is completely silent. "
        "<code>'causal'</code> pads only the left, guaranteeing output $t$ depends "
        "on inputs $\\le t$. Use it for any autoregressive task.",
    )

    sub("WaveNet — dilated causal convolutions")

    md(
        "Van den Oord et al. (2016) stacked causal convolutions with "
        "**exponentially increasing dilation rates**: 1, 2, 4, 8, 16, … Each layer "
        "doubles the receptive field."
    )

    derive(
        [("For a stack of $L$ dilated convolutions with kernel size $f$ and "
          "dilation $r_l = 2^{l}$, the receptive field is:",
          r"R \;=\; 1 + (f-1)\sum_{l=0}^{L-1} 2^{\,l} "
          r"\;=\; 1 + (f-1)\bigl(2^{L} - 1\bigr)"),
         ("With $f = 2$ this is exactly $2^L$. So <b>10 layers see 1 024 steps</b>, "
          "and 20 layers see over a million.", None),
         ("Compare a stack of ordinary $f=2$ convolutions, where the receptive "
          "field is only $L + 1$ — linear rather than exponential.", None),
         ("<b>And the parameter count is identical.</b> Dilation inserts gaps "
          "between the kernel taps; it does not add taps. So you buy an "
          "exponentially larger context for free.", None),
         ("<b>The gradient path is $L$ layers deep, not $T$ steps long.</b> For "
          "$T = 1024$ and $L = 10$, that is a product of 10 terms instead of "
          "1 024 — which is why WaveNet trains stably on sequences where an RNN "
          "cannot.", None)],
        title="Why dilation gives exponential context for free",
    )

    idea(
        "This is the argument that eventually killed recurrence",
        "WaveNet showed that a purely convolutional model could beat RNNs on raw "
        "audio — 16 000 samples per second — precisely because it is parallel and "
        "has a short gradient path. The Transformer (Chapter 16) took the same "
        "two properties further: full parallelism, and a gradient path of length "
        "<b>1</b> between any two positions via attention. Recurrence lost on "
        "hardware efficiency as much as on accuracy.",
    )

    anim_header("Dilated causal convolutions: receptive field doubling per layer")

    T = 32
    layers_d = [1, 2, 4, 8, 16]
    frames = []
    for L in range(1, len(layers_d) + 1):
        shapes = []
        rf = 1 + 1 * (2 ** L - 1)
        for lvl in range(L + 1):
            y = lvl
            for t in range(T):
                active = (t >= T - 1 - min(rf - 1, T - 1)) if lvl <= L else True
                col = (C["accent"] if lvl == 0 else SEQ[(lvl - 1) % len(SEQ)])
                shapes.append(go.Scatter(
                    x=[t], y=[y], mode="markers",
                    marker=dict(size=9,
                                color=alpha(col, .9) if active
                                else alpha(C["line"], .35),
                                line=dict(color="#fff", width=.8)),
                    showlegend=False, hoverinfo="skip"))
        # draw the connections into the final output
        xs, ys = [], []
        pos = {L: [T - 1]}
        for lvl in range(L, 0, -1):
            d = layers_d[lvl - 1]
            nxt = []
            for p in pos[lvl]:
                for src in (p, p - d):
                    if src >= 0:
                        xs += [p, src, None]
                        ys += [lvl, lvl - 1, None]
                        nxt.append(src)
            pos[lvl - 1] = sorted(set(nxt))
        shapes.append(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=alpha(C["danger"], .55),
                                           width=1.4),
                                 showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(L), data=shapes,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"{L} layer(s), dilations "
                                   f"{layers_d[:L]}   ·   receptive field = "
                                   f"{min(rf, T)} steps   ·   "
                                   f"parameters unchanged")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=420, plot_bgcolor="#FFFFFF",
                    xaxis=dict(title="time step", range=[-1, T]),
                    yaxis=dict(title="layer", range=[-.6, len(layers_d) + .6],
                               dtick=1),
                    title="Dilated causal convolution stack")
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="layers ")
    figure(f, "Red lines trace what the last output actually depends on. Each "
              "layer doubles the reach; the parameter count never changes.")

    code_lab(
        "Conv1D, the causal-padding trap, and a WaveNet",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE CAUSAL PADDING TRAP ===============================
print("=== 'same' padding LEAKS THE FUTURE ===")
x = np.zeros((1, 10, 1), dtype="float32")
x[0, 5, 0] = 1.0                          # a single spike at t=5

for pad in ["same", "causal"]:
    conv = keras.layers.Conv1D(1, 3, padding=pad, use_bias=False,
                               kernel_initializer="ones")
    out = conv(tf.constant(x)).numpy()[0, :, 0]
    influenced = np.where(out != 0)[0]
    print(f"  padding='{pad}': a spike at t=5 influences outputs "
          f"{list(influenced)}")
print("  'same' lets t=4 see the spike at t=5 -- the FUTURE.")
print("  'causal' does not. Use it for any autoregressive task.")

# ============ 2. RECEPTIVE FIELD ARITHMETIC ============================
print()
print("=== receptive field: plain vs dilated ===")
def rf_plain(L, f=2):   return 1 + (f-1)*L
def rf_dilated(L, f=2): return 1 + (f-1)*(2**L - 1)

print(f"{'layers':>8}{'plain (f=2)':>14}{'dilated (f=2)':>16}{'ratio':>10}")
for L in [1, 2, 5, 10, 15, 20]:
    p, d = rf_plain(L), rf_dilated(L)
    print(f"{L:>8}{p:>14,}{d:>16,}{d/p:>10.1f}x")
print("  and the PARAMETER COUNT is identical -- dilation adds gaps, not taps")

# --- verify with real layers -----------------------------------------
print()
print(f"{'dilation_rate':>15}{'params':>9}{'output shape':>16}")
z = tf.zeros((1, 64, 8))
for r in [1, 2, 4, 8, 16]:
    l = keras.layers.Conv1D(8, 2, padding="causal", dilation_rate=r)
    out = l(z)
    print(f"{r:>15}{l.count_params():>9}{str(tuple(out.shape)):>16}")

# ============ 3. A WAVENET =============================================
print()
print("=== building a WaveNet ===")
def wavenet(input_shape, n_blocks=2, n_layers=5, filters=32, horizon=1):
    inp = keras.layers.Input(shape=input_shape)
    z = keras.layers.Conv1D(filters, 1, padding="causal")(inp)
    skips = []
    for _ in range(n_blocks):
        for rate in (2**i for i in range(n_layers)):
            # gated activation: tanh(conv) * sigmoid(conv)
            filt = keras.layers.Conv1D(filters, 2, padding="causal",
                                       dilation_rate=rate,
                                       activation="tanh")(z)
            gate = keras.layers.Conv1D(filters, 2, padding="causal",
                                       dilation_rate=rate,
                                       activation="sigmoid")(z)
            h = keras.layers.Multiply()([filt, gate])
            h = keras.layers.Conv1D(filters, 1)(h)
            z = keras.layers.Add()([z, h])          # RESIDUAL
            skips.append(h)                         # SKIP
    out = keras.layers.Activation("relu")(keras.layers.Add()(skips))
    out = keras.layers.Conv1D(filters, 1, activation="relu")(out)
    out = keras.layers.Conv1D(horizon, 1)(out)
    return keras.Model(inp, out)

L_IN, NF = 112, 3
wn = wavenet((L_IN, NF), n_blocks=2, n_layers=5)
rf = 1 + 2 * sum(2**i for i in range(5))
print(f"  2 blocks x 5 layers, dilations 1,2,4,8,16 (twice)")
print(f"  receptive field = {rf} steps")
print(f"  parameters = {wn.count_params():,}")
print(f"  output shape = {tuple(wn.output.shape[1:])}  (a prediction per step)")

# ============ 4. RACE THEM ON A REAL FORECAST ==========================
df = _ds.ridership(n_days=730)
idx = df.index
F = np.column_stack([df["rail"].to_numpy()/1e6,
                     np.sin(2*np.pi*idx.dayofweek/7),
                     np.cos(2*np.pi*idx.dayofweek/7)]).astype("float32")
n_tr, n_va = 480, 110
mu, sd = F[:n_tr].mean(0), F[:n_tr].std(0) + 1e-7
Fs = (F - mu)/sd
tgt = Fs[:, 0]
n = len(Fs) - L_IN - 1
X = np.stack([Fs[i:i+L_IN] for i in range(n)]).astype("float32")
Y = tgt[L_IN:L_IN+n].astype("float32")
a = n_tr - L_IN; b = a + n_va
Xtr, Ytr, Xva, Yva = X[:a], Y[:a], X[a:b], Y[a:b]

print()
print(f"=== forecasting: window {L_IN}, {len(Xtr)} training examples ===")
print(f"{'model':<28}{'params':>9}{'fit time':>11}{'valid MAE (M)':>16}")

def last_only(model):
    """Take just the final time step of a sequence-output model."""
    return keras.Sequential([model, keras.layers.Lambda(lambda t: t[:, -1, 0])])

models = {
    "SimpleRNN(32)":
        keras.Sequential([keras.layers.Input(shape=(L_IN, NF)),
                          keras.layers.SimpleRNN(32), keras.layers.Dense(1)]),
    "GRU(32)":
        keras.Sequential([keras.layers.Input(shape=(L_IN, NF)),
                          keras.layers.GRU(32), keras.layers.Dense(1)]),
    "LSTM(32)":
        keras.Sequential([keras.layers.Input(shape=(L_IN, NF)),
                          keras.layers.LSTM(32), keras.layers.Dense(1)]),
    "Conv1D stack (causal)":
        keras.Sequential([keras.layers.Input(shape=(L_IN, NF)),
                          keras.layers.Conv1D(32, 4, padding="causal",
                                              activation="relu"),
                          keras.layers.Conv1D(32, 4, padding="causal",
                                              strides=2, activation="relu"),
                          keras.layers.GlobalAveragePooling1D(),
                          keras.layers.Dense(1)]),
    "WaveNet":
        last_only(wavenet((L_IN, NF), n_blocks=2, n_layers=5, filters=24)),
}
for nm, m in models.items():
    tf.random.set_seed(0)
    m.compile(loss="mae", optimizer=keras.optimizers.Adam(2e-3, clipnorm=1.))
    t0 = time.perf_counter()
    m.fit(Xtr, Ytr, epochs=25, batch_size=32, verbose=0)
    dt = time.perf_counter()-t0
    err = np.ravel(m.evaluate(Xva, Yva, verbose=0))[0]*sd[0]
    print(f"{nm:<28}{m.count_params():>9,}{dt:>10.1f}s{err:>16.4f}")

# --- the baseline, as always -----------------------------------------
naive7 = Xva[:, -7, 0]
print(f"{'seasonal naive (t-7)':<28}{0:>9}{0:>10.1f}s"
      f"{np.mean(np.abs(Yva - naive7))*sd[0]:>16.4f}")

# ============ 5. PARALLELISM: THE REAL ADVANTAGE =======================
print()
print("=== sequential steps required ===")
print(f"{'model':<26}{'sequential ops':>18}{'note':>34}")
for nm, ops, note in [
        ("SimpleRNN / GRU / LSTM", f"O(T) = {L_IN}", "cannot be parallelised"),
        ("Conv1D stack (5 layers)", "O(1) = 5", "all time steps at once"),
        ("WaveNet (10 layers)", "O(1) = 10", "all time steps at once"),
        ("Transformer (ch. 16)", "O(1)", "any two positions, distance 1")]:
    print(f"{nm:<26}{ops:>18}{note:>34}")
print()
print("  This is why recurrence lost: not accuracy, but HARDWARE EFFICIENCY.")
print("  A GPU wants many independent operations; an RNN gives it one at a time.")

import plotly.graph_objects as go
fig = go.Figure()
L_axis = np.arange(1, 21)
fig.add_scatter(x=L_axis, y=[rf_plain(int(l)) for l in L_axis], mode="lines+markers",
                name="plain Conv1D (f=2)", line=dict(color=C["danger"], width=3))
fig.add_scatter(x=L_axis, y=[rf_dilated(int(l)) for l in L_axis],
                mode="lines+markers", name="dilated (rate 2^l)",
                line=dict(color=C["success"], width=3))
fig.update_layout(height=400, yaxis_type="log", xaxis_title="number of layers",
                  yaxis_title="receptive field (steps)",
                  title="Dilation buys exponential context at constant cost")
''',
        key="ch15_wavenet",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "Can you think of a few applications for a sequence-to-sequence RNN? "
        "What about a sequence-to-vector RNN, and a vector-to-sequence RNN?",
        "**Sequence-to-sequence:** predicting the weather (or any other time "
        "series), machine translation (with an encoder–decoder), video "
        "captioning, speech-to-text, music generation, identifying the chords of "
        "a song.\n\n"
        "**Sequence-to-vector:** classifying music samples by genre, analysing "
        "the sentiment of a book review, predicting what word an aphasic patient "
        "is thinking of based on readings from brain implants, predicting the "
        "probability that a user will want to watch a film based on their watch "
        "history (one of many possible implementations of collaborative "
        "filtering for a recommender system).\n\n"
        "**Vector-to-sequence:** image captioning, creating a music playlist "
        "based on an embedding of the current artist, generating a melody based "
        "on a set of parameters, locating pedestrians in a picture (e.g. a video "
        "frame from a self-driving car's camera).")

    exercise(
        2, "How many dimensions must the inputs of an RNN layer have? What does "
        "each dimension represent? What about its outputs?",
        "An RNN layer's inputs must be **3-D**. The first dimension is the "
        "**batch** dimension (its size is the batch size), the second represents "
        "**time** (its size is the number of time steps), and the third holds the "
        "inputs at each time step (its size is the number of **input features** "
        "per time step). For example, if you want to process a batch containing 5 "
        "time series of 10 time steps each, with 2 values per time step, the "
        "shape will be $(5, 10, 2)$.\n\n"
        "The outputs are also 3-D, with the same first two dimensions, but the "
        "last dimension is equal to the **number of neurons**. So if an RNN layer "
        "with 32 neurons processes the batch just described, the output will have "
        "shape $(5, 10, 32)$.")

    exercise(
        3, "If you want to build a deep sequence-to-sequence RNN, which RNN layers "
        "should have `return_sequences=True`? What about a sequence-to-vector RNN?",
        "To build a deep **sequence-to-sequence** RNN using Keras, you must set "
        "`return_sequences=True` for **all** RNN layers.\n\n"
        "To build a **sequence-to-vector** RNN, you must set "
        "`return_sequences=True` for all RNN layers **except for the top RNN "
        "layer**, which must have `return_sequences=False` (the default).\n\n"
        "Forgetting this on a middle layer is the single most common RNN bug: the "
        "next layer receives a 2-D tensor where it expected 3-D, and the error "
        "message points at the wrong place.")

    exercise(
        4, "Suppose you have a daily univariate time series, and you want to "
        "forecast the next seven days. Which RNN architecture should you use?",
        "The simplest RNN architecture is a stack of RNN layers where all layers "
        "use `return_sequences=True` except for the top layer, followed by a "
        "**dense output layer with 7 units** (one per forecast day). You can then "
        "train this model using random windows from the time series, where the "
        "target is a vector of the next 7 values.\n\n"
        "That is the **multi-output** strategy of §15.5, and it avoids the error "
        "accumulation that a recursive one-step model suffers.\n\n"
        "Better still is the **sequence-to-sequence** variant of §15.6: make "
        "every RNN layer return sequences, and let the `Dense(7)` output layer "
        "predict the next 7 values **at every time step**. That gives $T$ error "
        "signals per training window instead of one, which speeds up and "
        "stabilises training considerably. At inference you use only the last "
        "time step's output.")

    exercise(
        5, "What are the main difficulties when training RNNs? How can you handle "
        "them?",
        "The two main difficulties are **unstable gradients** (exploding or "
        "vanishing) and a **very limited short-term memory**. Both get worse as "
        "sequences get longer.\n\n"
        "**Unstable gradients:** use a smaller learning rate; use a saturating "
        "activation function such as $\\tanh$ (which is the default) rather than "
        "ReLU; and possibly use **gradient clipping**, **layer normalisation**, or "
        "dropout at each time step. Note that **batch normalisation does not work "
        "well inside a recurrent layer** (§15.7).\n\n"
        "**Limited short-term memory:** use **LSTM** or **GRU** layers (§15.8), "
        "whose gated additive state update gives the network a learnable "
        "retention rate per dimension. Alternatively, use **1-D convolutional "
        "layers** with dilation, or drop recurrence entirely in favour of "
        "attention (Chapter 16).")

    exercise(
        6, "Can you sketch the LSTM cell's architecture?",
        "The LSTM cell's architecture, in words:\n\n"
        "* **Two states** flow through the cell: the long-term state "
        "$\\mathbf{c}_{(t)}$ travels along the top from left to right, and the "
        "short-term state $\\mathbf{h}_{(t)}$ along the bottom.\n\n"
        "* The current input $\\mathbf{x}_{(t)}$ and previous short-term state "
        "$\\mathbf{h}_{(t-1)}$ feed **four** fully connected layers: the "
        "**forget gate** $\\mathbf{f}$ (sigmoid), the **input gate** "
        "$\\mathbf{i}$ (sigmoid), the **output gate** $\\mathbf{o}$ (sigmoid), and "
        "the **candidate** $\\mathbf{g}$ (tanh).\n\n"
        "* The long-term state is **first multiplied element-wise by the forget "
        "gate** (dropping some memories), **then added to** the input gate times "
        "the candidate (adding some memories):\n"
        "$\\mathbf{c}_{(t)} = \\mathbf{f}_{(t)} \\otimes \\mathbf{c}_{(t-1)} + "
        "\\mathbf{i}_{(t)} \\otimes \\mathbf{g}_{(t)}$\n\n"
        "* The result is passed through $\\tanh$ and filtered by the output gate "
        "to produce both the short-term state and the cell's output:\n"
        "$\\mathbf{h}_{(t)} = \\hat{\\mathbf{y}}_{(t)} = \\mathbf{o}_{(t)} "
        "\\otimes \\tanh(\\mathbf{c}_{(t)})$\n\n"
        "The critical structural feature is that $\\mathbf{c}$ is modified only by "
        "an element-wise multiply and an add — never by a matrix multiplication — "
        "which is exactly why its gradient does not vanish.")

    exercise(
        7, "Why would you want to use 1D convolutional layers in an RNN?",
        "An RNN layer is fundamentally **sequential**: to compute the outputs at "
        "time step $t$, it has to first compute the outputs at all earlier time "
        "steps. This makes it impossible to parallelise across time.\n\n"
        "A 1-D convolutional layer, in contrast, **lends itself well to "
        "parallelisation** because it does not hold a state between time steps — "
        "every output position can be computed simultaneously.\n\n"
        "It can also act as a **downsampler**: a Conv1D with stride 2 halves the "
        "sequence length before it reaches the RNN, which both speeds up the RNN "
        "and shortens the gradient path it must traverse. This hybrid — a few "
        "convolutional layers feeding a recurrent layer — was a standard "
        "architecture for audio and long time series.\n\n"
        "Taken further (WaveNet), dilated causal convolutions can replace the RNN "
        "entirely.")

    exercise(
        8, "Which neural network architecture could you use to classify videos?",
        "To classify videos based on their **visual content**, one possible "
        "architecture is to take (say) one frame per second, run every frame "
        "through the **same** convolutional neural network (e.g. a pretrained "
        "Xception, possibly frozen), feed the sequence of CNN outputs to a "
        "**sequence-to-vector RNN** (LSTM or GRU), and finally run its output "
        "through a softmax layer giving the class probabilities.\n\n"
        "For training, use cross-entropy as the cost function. If you also want "
        "to use the **audio** for classification, convert each second of audio to "
        "a spectrogram, pass it through a CNN, and feed the output to the RNN "
        "alongside the visual features (concatenating the two).\n\n"
        "Modern alternatives: **3-D convolutions** (treating time as a third "
        "spatial axis), or a **video Transformer** that attends over spatio-"
        "temporal patches (Chapter 16). In practice `keras.applications` + a GRU "
        "is still a strong and cheap baseline.")

    exercise(
        9, "Train a classification model for the SketchRNN dataset, available in "
        "TensorFlow Datasets.",
        "SketchRNN (`tfds.load('quickdraw_bitmap')` or the stroke-sequence "
        "version) is a good exercise because it can be attacked two completely "
        "different ways, and comparing them is the point:\n\n"
        "**As images.** The `quickdraw_bitmap` variant gives 28×28 bitmaps — a "
        "CNN (Chapter 14) reaches roughly 90 % on the full 345 classes.\n\n"
        "**As sequences.** The stroke variant gives a sequence of "
        "$(\\Delta x, \\Delta y, \\text{pen state})$ triples — a genuinely "
        "sequential representation. A stack of `Conv1D` layers followed by a "
        "bidirectional LSTM handles this well, and the sequence representation is "
        "far more compact than the bitmap.\n\n"
        "Practical notes: the sequences have very different lengths, so use "
        "`padded_batch` and **masking**; normalise the deltas; and start with a "
        "subset of ~10 classes to iterate quickly before scaling up.",
        code='''import tensorflow_datasets as tfds

# the stroke version: sequences of (dx, dy, pen_state)
train_set, valid_set = tfds.load(
    "quickdraw_sketch_rnn", split=["train[:90%]", "train[90%:]"],
    as_supervised=True)

model = keras.Sequential([
    keras.layers.Input(shape=[None, 3]),
    keras.layers.Masking(mask_value=0.0),
    keras.layers.Conv1D(48, 5, strides=2, activation="relu", padding="valid"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, 5, strides=2, activation="relu", padding="valid"),
    keras.layers.BatchNormalization(),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),
    keras.layers.Dense(n_classes, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
              metrics=["accuracy", "sparse_top_k_categorical_accuracy"])''')

    exercise(
        10, "Download the Bach chorales dataset and unzip it. It is composed of "
        "382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to "
        "640 time steps long, and each time step contains 4 integers, where each "
        "integer corresponds to a note's index on a piano (except for the value 0, "
        "which means that no note is played). Train a model — recurrent, "
        "convolutional, or both — that can predict the next time step (four "
        "notes), given a sequence of time steps from a chorale. Then use this "
        "model to generate Bach-like music, one note at a time: you can do this by "
        "giving the model the start of a chorale and asking it to predict the next "
        "time step, then appending these time steps to the input sequence and "
        "asking the model for the next note, and so on.",
        "The key design decisions:\n\n"
        "**Representation.** Treat the four voices as four separate tokens per "
        "time step and **flatten** them into one long sequence of length $4T$. "
        "That way the model learns both the harmony (within a time step) and the "
        "melody (across time steps) with one mechanism. Shift the note values so "
        "they form a compact vocabulary (~47 tokens including the rest).\n\n"
        "**Architecture.** An `Embedding` layer, then a stack of **dilated causal "
        "`Conv1D`** layers (a WaveNet, §15.9), then a `Dense(vocab, softmax)`. "
        "Dilations 1, 2, 4, 8 repeated a few times give a receptive field of "
        "several bars, and it trains far faster than an RNN.\n\n"
        "**Generation.** Feed a seed, predict the next token, append, repeat. "
        "**Do not take the argmax** — that produces repetitive, degenerate music. "
        "Sample from the softmax with a **temperature**: "
        "$p_i \\propto \\exp(\\log p_i / T)$. Around $T = 1.0$ gives musical "
        "results; $T \\to 0$ collapses to argmax; $T$ large gives noise.\n\n"
        "This is exactly the character-RNN generation procedure of Chapter 16, "
        "applied to notes instead of letters.",
        code='''# flatten the 4 voices into one sequence: harmony AND melody in one axis
def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:]           # predict the NEXT token
    return X, Y

def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1)
    return tf.reshape(window, [-1])          # (T, 4) -> (4T,)

model = keras.Sequential([
    keras.layers.Input(shape=[None]),
    keras.layers.Embedding(input_dim=n_notes, output_dim=5),
    keras.layers.Conv1D(32, 2, padding="causal", activation="relu",
                        dilation_rate=1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(48, 2, padding="causal", activation="relu",
                        dilation_rate=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, 2, padding="causal", activation="relu",
                        dilation_rate=4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(96, 2, padding="causal", activation="relu",
                        dilation_rate=8),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.Dense(n_notes, activation="softmax"),
])

def generate_chorale(model, seed_chords, length, temperature=1):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio, verbose=0)[0, -1:]
            rescaled_logits = tf.math.log(next_note_probas) / temperature
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)
            arpegio = tf.concat([arpegio, next_note], axis=1)   # SAMPLE, not argmax
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])''')

    rule()

    sub("The chapter as a decision table")

    table(
        ["Situation", "Use", "Why"],
        [["Short sequences (< 30 steps)", "GRU or even SimpleRNN",
          "The gradient product is short enough"],
         ["Medium sequences (30–500)", "<b>GRU</b>, or LSTM",
          "Gated cells hold the memory"],
         ["Long sequences (500+)", "<b>WaveNet</b> or a Transformer",
          "Gradient path is depth, not length"],
         ["Very long, needs parallelism", "<b>Transformer</b> (Ch. 16)",
          "Path length 1 between any two positions"],
         ["Forecasting a time series", "Compare against seasonal naive first",
          "The baseline is often very hard to beat"],
         ["Multi-step forecast", "Multi-output or seq2seq, never recursive",
          "Recursive errors compound"],
         ["Autoregressive generation", "<code>padding='causal'</code>, always",
          "'same' padding leaks the future silently"]],
    )

    keypoints([
        "An RNN's gradient is a product over the <b>sequence length</b> — the "
        "vanishing problem, made worse.",
        "<b>LSTM/GRU</b> replace the multiplicative state update with an additive, "
        "gated one, so each unit learns its own timescale.",
        "<b>Compute the seasonal-naive baseline</b> before claiming a forecasting "
        "result.",
        "<b>Dilated causal convolutions</b> give exponential context at constant "
        "parameter cost, and are fully parallel.",
        "Recurrence lost to attention on <b>hardware efficiency</b> as much as on "
        "accuracy — which is Chapter 16.",
    ], title="Chapter 15 in five lines")

    refs([
        ("Hochreiter & Schmidhuber — *Long Short-Term Memory*",
         "https://doi.org/10.1162/neco.1997.9.8.1735"),
        ("Cho et al. — *Learning Phrase Representations using RNN "
         "Encoder–Decoder* (GRU)", "https://arxiv.org/abs/1406.1078"),
        ("Pascanu, Mikolov & Bengio — *On the Difficulty of Training Recurrent "
         "Neural Networks*", "https://arxiv.org/abs/1211.5063"),
        ("Ba, Kiros & Hinton — *Layer Normalization*",
         "https://arxiv.org/abs/1607.06450"),
        ("Gal & Ghahramani — *A Theoretically Grounded Application of Dropout in "
         "Recurrent Neural Networks*", "https://arxiv.org/abs/1512.05287"),
        ("van den Oord et al. — *WaveNet: A Generative Model for Raw Audio*",
         "https://arxiv.org/abs/1609.03499"),
        ("Bai, Kolter & Koltun — *An Empirical Evaluation of Generic Convolutional "
         "and Recurrent Networks for Sequence Modeling*",
         "https://arxiv.org/abs/1803.01271"),
        ("Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*",
         "https://otexts.com/fpp3/"),
    ])


# ==========================================================================
SECTIONS = [
    ("15.1", "Recurrent Neurons and Layers", s_15_1),
    ("15.2", "Training RNNs — BPTT", s_15_2),
    ("15.3", "Baselines and ARMA", s_15_3),
    ("15.4", "Preparing the Data", s_15_4),
    ("15.5", "Deep, Multivariate, Multi-Step", s_15_5),
    ("15.6", "Sequence-to-Sequence", s_15_6),
    ("15.7", "Handling Long Sequences", s_15_7),
    ("15.8", "LSTM and GRU", s_15_8),
    ("15.9", "1-D Convolutions & Exercises", s_15_9),
]

nav.render_chapter(CH, SECTIONS)
