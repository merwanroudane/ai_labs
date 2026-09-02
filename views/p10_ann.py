"""Chapter 10 — Introduction to Artificial Neural Networks with Keras."""

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
CH = "ch10"

hero(
    kicker="Part II · Chapter 10",
    title="Introduction to Artificial Neural Networks with Keras",
    blurb=(
        "From the McCulloch–Pitts neuron to a working Keras model. The perceptron "
        "and exactly why it cannot learn XOR; backpropagation derived layer by "
        "layer from the chain rule; the activation functions and why the sigmoid "
        "was abandoned; and the three Keras APIs — Sequential, Functional and "
        "Subclassing — with the callbacks, checkpointing and hyperparameter search "
        "that make training reproducible."
    ),
    chips=["Backprop derived", "9 sub-sections", "9 animations",
           "10 code labs", "Keras from zero"],
)
nav.sidebar_tools(CH)


# --------------------------------------------------------------------------
def _tf_available() -> bool:
    # find_spec, not import: importing TensorFlow costs ~500 MB of RSS and we
    # only need to know whether the labs will be able to run.
    import importlib.util
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


TF_OK = _tf_available()
if not TF_OK:
    st.warning(
        "TensorFlow is not importable in this environment, so the Keras labs "
        "below will report an ImportError when run. Every explanation, animation "
        "and NumPy-only lab still works.", icon="⚠️")


# ==========================================================================
def s_10_1():
    section("10.1", "From Biological to Artificial Neurons")

    lead(
        "Artificial neural networks were inspired by biology, and then diverged "
        "from it completely. Understanding the original three ideas — the "
        "threshold unit, logical computation, and the perceptron learning rule — "
        "makes everything after it obvious."
    )

    sub("The biological neuron, briefly")

    md(
        "A biological neuron receives signals through its **dendrites**, and if "
        "the accumulated signal exceeds a threshold within a few milliseconds, it "
        "fires an **action potential** down its **axon** to other neurons' "
        "dendrites via **synapses**. Individual neurons behave simply; the "
        "complexity comes from a network of ~86 billion of them with ~10⁴ "
        "connections each."
    )

    note(
        "The analogy is historical, not technical",
        "Modern deep networks bear about as much resemblance to a brain as an "
        "aeroplane does to a bird: the initial inspiration was real and the "
        "engineering diverged entirely. Backpropagation, in particular, has no "
        "known biological equivalent. Do not reason about deep learning by "
        "reasoning about neuroscience.",
    )

    sub("Logical computations with neurons")

    md(
        "McCulloch & Pitts (1943) proposed an artificial neuron with binary "
        "inputs and output, which activates when at least a given number of its "
        "inputs are active. Even with this trivial model you can build any logical "
        "proposition:"
    )

    table(
        ["Network", "Computes", "How"],
        [["A → C", "$C = A$", "One input, threshold 1 — the identity"],
         ["A, B → C", "$C = A \\wedge B$",
          "Two inputs, threshold 2 — both must fire"],
         ["A, B → C", "$C = A \\vee B$",
          "Two inputs, threshold 1 — either suffices"],
         ["A, ¬B → C", "$C = A \\wedge \\neg B$",
          "An <b>inhibitory</b> connection from B"]],
    )

    sub("The Perceptron")

    md(
        "Rosenblatt (1957) made the inputs and weights continuous. The **threshold "
        "logic unit** (TLU) computes a weighted sum and applies a step function:"
    )

    math(r"""
    z \;=\; \mathbf{w}^\top\mathbf{x} + b,
    \qquad
    h_{\mathbf{w},b}(\mathbf{x}) \;=\; \mathrm{step}(z)
    """)

    math(r"""
    \mathrm{heaviside}(z) =
    \begin{cases} 0 & z < 0\\ 1 & z \ge 0\end{cases}
    \qquad\qquad
    \mathrm{sgn}(z) =
    \begin{cases} -1 & z < 0\\ 0 & z = 0\\ +1 & z > 0\end{cases}
    """)

    md("A full layer of TLUs, vectorised over a whole batch:")

    math(r"""
    h_{\mathbf{W},\mathbf{b}}(\mathbf{X}) \;=\;
    \phi\bigl(\mathbf{X}\mathbf{W} + \mathbf{b}\bigr)
    """)
    where({
        r"\mathbf{X}": "the input matrix, one <b>row</b> per instance, one column "
                       "per feature",
        r"\mathbf{W}": "the weight matrix, one row per input, one column per neuron",
        r"\mathbf{b}": "the bias vector, one entry per neuron (broadcast over rows)",
        r"\phi": "the activation function — the step function, for a perceptron",
    })

    sub("The perceptron learning rule")

    math(r"""
    w_{i,j}^{(\text{next})} \;=\; w_{i,j}
      \;+\; \eta \,\bigl(y_j - \hat y_j\bigr)\, x_i
    """)
    where({r"w_{i,j}": "the weight from input $i$ to output neuron $j$",
           r"x_i": "the $i$-th input value of the current instance",
           r"\hat y_j, y_j": "the predicted and target output of neuron $j$",
           r"\eta": "the learning rate"})

    idea(
        "'Fire together, wire together'",
        "The rule reinforces connections that <i>would have</i> reduced the error. "
        "If the neuron should have fired and did not ($y_j - \\hat y_j = +1$), "
        "every active input's weight increases. This is Hebb's rule with an error "
        "signal attached — and note that it is exactly the "
        "<i>(prediction − target) × input</i> form we derived for linear, logistic "
        "and softmax regression in Chapter 4. The same expression, once again.",
    )

    proof(
        "The Perceptron Convergence Theorem",
        "Novikoff (1962): if the training set is <b>linearly separable</b> with "
        "margin $\\gamma$ and all inputs satisfy $\\lVert\\mathbf{x}\\rVert \\le R$, "
        "the perceptron learning rule converges in at most $(R/\\gamma)^2$ mistakes "
        "— <b>regardless of the order of presentation and regardless of the "
        "learning rate</b>. That is a genuinely strong guarantee. The catch is the "
        "premise: if the data is <i>not</i> linearly separable, the algorithm never "
        "converges and cycles forever.",
    )

    sub("The XOR problem")

    pitfall(
        "A perceptron cannot learn XOR — and this nearly killed the field",
        "Minsky & Papert's <i>Perceptrons</i> (1969) pointed out that a single "
        "perceptron cannot represent XOR, because XOR is not linearly separable: "
        "no single line separates $\\{(0,0),(1,1)\\}$ from $\\{(0,1),(1,0)\\}$. The "
        "book's influence contributed to the first \"AI winter\". "
        "<b>The resolution is stacking:</b> a network with one hidden layer solves "
        "XOR trivially, and Minsky and Papert knew this — what was missing was an "
        "algorithm to <i>train</i> the hidden layer. That algorithm is §10.2.",
    )

    anim_header("A perceptron learning — and failing on XOR")
    md(
        "Left: a linearly separable problem, where the perceptron converges in a "
        "few passes. Right: XOR, where it oscillates forever. Both use exactly the "
        "same code."
    )

    rng = np.random.default_rng(3)
    Xl = np.r_[rng.normal([-1.3, -0.9], .55, (30, 2)),
               rng.normal([1.4, 1.1], .55, (30, 2))]
    yl = np.r_[np.zeros(30), np.ones(30)]
    Xx = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    yx = np.array([0., 1., 1., 0.])
    Xx = np.repeat(Xx, 12, axis=0) + rng.normal(0, .08, (48, 2))
    yx = np.repeat(yx, 12)

    def run_perceptron(X, y, steps=90, eta=.08, seed=0):
        r = np.random.default_rng(seed)
        w = r.normal(0, .4, 2); b = 0.0
        hist = []
        for t in range(steps):
            i = t % len(X)
            pred = 1.0 if (X[i] @ w + b) >= 0 else 0.0
            err = y[i] - pred
            w = w + eta * err * X[i]
            b = b + eta * err
            acc = float(np.mean(((X @ w + b) >= 0).astype(float) == y))
            hist.append((w.copy(), b, acc))
        return hist

    h_lin = run_perceptron(Xl, yl)
    h_xor = run_perceptron(Xx, yx)

    def bline(w, b, lo, hi):
        gx = np.linspace(lo, hi, 40)
        if abs(w[1]) < 1e-6:
            return gx, np.full_like(gx, np.nan)
        return gx, -(w[0] * gx + b) / w[1]

    frames = []
    for t in range(0, 90, 2):
        wl, bl, al = h_lin[t]
        wx, bx, ax = h_xor[t]
        gx1, gy1 = bline(wl, bl, -3.2, 3.2)
        gx2, gy2 = bline(wx, bx, -.6, 1.6)
        frames.append(go.Frame(name=str(t + 1), data=[
            go.Scatter(x=Xl[yl == 0, 0], y=Xl[yl == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=8,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=Xl[yl == 1, 0], y=Xl[yl == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=8,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=gx1, y=gy1, mode="lines",
                       line=dict(color=C["success"], width=4)),
            go.Scatter(x=Xx[yx == 0, 0], y=Xx[yx == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=8,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=Xx[yx == 1, 0], y=Xx[yx == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=8,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=gx2, y=gy2, mode="lines",
                       line=dict(color=C["danger"], width=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"update {t+1}   ·   linearly separable: accuracy {al:.3f}   ·   "
            f"XOR: accuracy {ax:.3f}",
            color=C["success"] if al > .98 else C["ink"])])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("linearly separable — converges",
                                      "XOR — never converges"))
    wl0, bl0, _ = h_lin[0]; wx0, bx0, _ = h_xor[0]
    gx1, gy1 = bline(wl0, bl0, -3.2, 3.2)
    gx2, gy2 = bline(wx0, bx0, -.6, 1.6)
    f.add_trace(go.Scatter(x=Xl[yl == 0, 0], y=Xl[yl == 0, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["train"], size=8,
                           line=dict(color="#fff", width=.9))), 1, 1)
    f.add_trace(go.Scatter(x=Xl[yl == 1, 0], y=Xl[yl == 1, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["warning"], size=8,
                           line=dict(color="#fff", width=.9))), 1, 1)
    f.add_trace(go.Scatter(x=gx1, y=gy1, mode="lines", showlegend=False,
                           line=dict(color=C["success"], width=4)), 1, 1)
    f.add_trace(go.Scatter(x=Xx[yx == 0, 0], y=Xx[yx == 0, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["train"], size=8,
                           line=dict(color="#fff", width=.9))), 1, 2)
    f.add_trace(go.Scatter(x=Xx[yx == 1, 0], y=Xx[yx == 1, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["warning"], size=8,
                           line=dict(color="#fff", width=.9))), 1, 2)
    f.add_trace(go.Scatter(x=gx2, y=gy2, mode="lines", showlegend=False,
                           line=dict(color=C["danger"], width=4)), 1, 2)
    f.update_xaxes(range=[-3.2, 3.2], row=1, col=1)
    f.update_yaxes(range=[-3.0, 3.0], row=1, col=1)
    f.update_xaxes(range=[-.6, 1.6], row=1, col=2)
    f.update_yaxes(range=[-.6, 1.6], row=1, col=2)
    f.update_layout(height=460, title="The perceptron learning rule")
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="update ")
    figure(f)

    warn(
        "Perceptrons output no probabilities",
        "Because the step function has zero gradient almost everywhere and is "
        "discontinuous at 0. That is why scikit-learn's <code>Perceptron</code> "
        "has no <code>predict_proba</code>, and it is the same problem that ruled "
        "out accuracy as a training objective in §3.3. Replacing the step with a "
        "<b>smooth</b> activation is what makes gradient-based training possible — "
        "and that is exactly the difference between a perceptron and an MLP.",
    )

    codenote(
        "Perceptron is SGD with a specific loss",
        "<code>Perceptron()</code> is equivalent to "
        "<code>SGDClassifier(loss='perceptron', learning_rate='constant', "
        "eta0=1, penalty=None)</code>. It is the same machinery you met in §3.2.",
    )

    code_lab(
        "The perceptron from scratch — convergence, and the XOR wall",
        '''import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.datasets import load_iris, make_blobs

# ============ 1. the TLU and the perceptron rule =======================
def perceptron_fit(X, y, eta=1.0, max_epochs=100, seed=0):
    """The original Rosenblatt rule: w += eta * (y - y_hat) * x"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, .01, X.shape[1]); b = 0.0
    history = []
    for epoch in range(max_epochs):
        errors = 0
        for xi, target in zip(X, y):
            pred = 1.0 if (xi @ w + b) >= 0 else 0.0
            update = eta * (target - pred)
            w += update * xi
            b += update
            errors += int(update != 0.0)
        acc = np.mean(((X @ w + b) >= 0).astype(float) == y)
        history.append((epoch, errors, acc))
        if errors == 0:
            break
    return w, b, history

# --- a linearly separable problem --------------------------------------
iris = load_iris()
X = iris.data[:, (2, 3)]                     # petal length, petal width
y = (iris.target == 0).astype(float)         # setosa is linearly separable

w, b, hist = perceptron_fit(X, y)
print("=== linearly separable (setosa vs rest) ===")
print(f"{'epoch':>7}{'mistakes':>11}{'accuracy':>11}")
for e, err, acc in hist[:8]:
    print(f"{e:>7}{err:>11}{acc:>11.4f}")
print(f"CONVERGED after {len(hist)} epochs, w = {w.round(4)}, b = {b:.4f}")

sk = Perceptron(random_state=42).fit(X, y)
print(f"sklearn agrees on {np.mean(sk.predict(X) == y)*100:.1f}% of instances")

# --- a NOT linearly separable problem ----------------------------------
y2 = (iris.target == 1).astype(float)        # versicolor is NOT separable
w2, b2, hist2 = perceptron_fit(X, y2, max_epochs=60)
print("\\n=== NOT linearly separable (versicolor vs rest) ===")
print(f"{'epoch':>7}{'mistakes':>11}{'accuracy':>11}")
for e, err, acc in hist2[::10]:
    print(f"{e:>7}{err:>11}{acc:>11.4f}")
print(f"ran all {len(hist2)} epochs WITHOUT converging -- it cycles forever.")

# ============ 2. THE XOR WALL ==========================================
print("\\n" + "="*62)
print("XOR: the problem that stopped the field for 17 years")
print("="*62)
X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = np.array([0., 1., 1., 0.])

w3, b3, hist3 = perceptron_fit(X_xor, y_xor, max_epochs=2000)
print(f"after {len(hist3)} epochs: accuracy = {hist3[-1][2]:.4f} (best possible "
      f"for a single perceptron is 0.75)")
print(f"\\n{'x1':>4}{'x2':>4}{'target':>8}{'perceptron':>12}")
for xi, t in zip(X_xor, y_xor):
    p = 1.0 if (xi @ w3 + b3) >= 0 else 0.0
    print(f"{xi[0]:>4.0f}{xi[1]:>4.0f}{t:>8.0f}{p:>12.0f}"
          + ("   <- WRONG" if p != t else ""))

# --- PROOF that no line works ------------------------------------------
print("\\nWhy? Suppose some (w1, w2, b) solves XOR. Then:")
print("   (0,0)->0  requires           b  < 0")
print("   (0,1)->1  requires      w2 + b >= 0   =>  w2 >= -b > 0")
print("   (1,0)->1  requires w1      + b >= 0   =>  w1 >= -b > 0")
print("   (1,1)->0  requires w1 + w2 + b  < 0")
print("But w1 >= -b and w2 >= -b give w1 + w2 + b >= -b > 0.  CONTRADICTION.")

# ============ 3. TWO LAYERS SOLVE IT ===================================
print("\\n=== a hidden layer solves XOR (weights set by hand) ===")
# hidden neuron 1 = OR, hidden neuron 2 = AND, output = OR AND NOT AND
W1 = np.array([[1., 1.], [1., 1.]])          # both neurons see both inputs
b1 = np.array([-0.5, -1.5])                  # thresholds: OR at 0.5, AND at 1.5
W2 = np.array([[1.], [-2.]])                 # OR minus 2*AND
b2v = np.array([-0.5])
step = lambda z: (z >= 0).astype(float)

h = step(X_xor @ W1 + b1)
out = step(h @ W2 + b2v)
print(f"{'x1':>4}{'x2':>4}{'h1(OR)':>9}{'h2(AND)':>10}{'output':>9}{'target':>9}")
for i in range(4):
    print(f"{X_xor[i,0]:>4.0f}{X_xor[i,1]:>4.0f}{h[i,0]:>9.0f}{h[i,1]:>10.0f}"
          f"{out[i,0]:>9.0f}{y_xor[i]:>9.0f}")
print("\\nPERFECT. The missing piece in 1969 was not the architecture --")
print("it was an algorithm to LEARN the hidden layer. That is backpropagation.")
''',
        key="ch10_perceptron",
    )

    keypoints([
        "A TLU computes $\\mathrm{step}(\\mathbf{w}^\\top\\mathbf{x} + b)$; a layer "
        "is $\\phi(\\mathbf{X}\\mathbf{W} + \\mathbf{b})$.",
        "The perceptron rule $\\Delta w_{ij} = \\eta(y_j - \\hat y_j)x_i$ is "
        "<i>(target − prediction) × input</i> — the same form as Chapter 4.",
        "It provably converges in $(R/\\gamma)^2$ mistakes <b>if</b> the data is "
        "linearly separable, and never otherwise.",
        "It cannot represent XOR — proved algebraically in the lab.",
        "A hidden layer fixes it; the missing piece was how to <b>train</b> the "
        "hidden layer.",
    ])


# ==========================================================================
def s_10_2():
    section("10.2", "The Multilayer Perceptron and Backpropagation")

    lead(
        "Stack layers, replace the step function with something differentiable, "
        "and apply the chain rule efficiently. That is the entire idea, and it "
        "took from 1969 to 1986 to become standard."
    )

    sub("The architecture")

    md(
        "An MLP is an input layer, one or more **hidden** layers of TLU-like "
        "neurons, and an output layer. Every layer except the output has a bias "
        "neuron and is **fully connected** to the next. Signals flow one way, so "
        "it is a **feedforward** network (FNN). A network with a deep stack of "
        "hidden layers is a **deep neural network** (DNN)."
    )

    math(r"""
    \mathbf{a}^{[0]} = \mathbf{x},
    \qquad
    \mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]},
    \qquad
    \mathbf{a}^{[l]} = \phi^{[l]}\bigl(\mathbf{z}^{[l]}\bigr),
    \qquad
    \hat{\mathbf{y}} = \mathbf{a}^{[L]}
    """)
    where({r"l = 1, \dots, L": "the layer index",
           r"\mathbf{z}^{[l]}": "the pre-activations of layer $l$",
           r"\mathbf{a}^{[l]}": "the activations (outputs) of layer $l$",
           r"\phi^{[l]}": "layer $l$'s activation function"})

    sub("Backpropagation, derived")

    md(
        "Backpropagation is **reverse-mode automatic differentiation** applied to "
        "a neural network, plus gradient descent. One pass forward computes the "
        "predictions; one pass backward computes every gradient."
    )

    derive(
        [("Define the <b>error signal</b> at layer $l$ — the gradient of the loss "
          "with respect to that layer's <i>pre-activations</i>. Everything follows "
          "from this one definition.",
          r"\boldsymbol\delta^{[l]} \;\equiv\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}"),
         ("<b>Output layer.</b> Apply the chain rule through the activation:",
          r"\boldsymbol\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} "
          r"\odot \phi'^{[L]}\bigl(\mathbf{z}^{[L]}\bigr)"),
         ("<b>The magic case.</b> For softmax output with cross-entropy loss (or "
          "sigmoid with binary cross-entropy), the two derivatives cancel exactly "
          "— we proved this in §4.7:",
          r"\boldsymbol\delta^{[L]} = \hat{\mathbf{y}} - \mathbf{y}"),
         ("<b>Recursion.</b> The error at layer $l$ is the error at layer $l+1$, "
          "pulled back through the weights and then through the activation:",
          r"\boldsymbol\delta^{[l]} = \Bigl(\mathbf{W}^{[l+1]\top}\boldsymbol\delta^{[l+1]}\Bigr) "
          r"\odot \phi'^{[l]}\bigl(\mathbf{z}^{[l]}\bigr)"),
         ("<b>Why this works:</b> $\\mathbf{z}^{[l+1]} = \\mathbf{W}^{[l+1]}"
          "\\phi(\\mathbf{z}^{[l]}) + \\mathbf{b}^{[l+1]}$, so "
          "$\\partial\\mathbf{z}^{[l+1]}/\\partial\\mathbf{z}^{[l]} = "
          "\\mathbf{W}^{[l+1]}\\mathrm{diag}(\\phi'(\\mathbf{z}^{[l]}))$. "
          "Transposing gives the expression above.", None),
         ("<b>Parameter gradients.</b> Once you have $\\boldsymbol\\delta^{[l]}$, "
          "the gradients are immediate, because $\\mathbf{z}^{[l]}$ is linear in "
          "$\\mathbf{W}^{[l]}$ and $\\mathbf{b}^{[l]}$:",
          r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = "
          r"\boldsymbol\delta^{[l]}\bigl(\mathbf{a}^{[l-1]}\bigr)^\top, "
          r"\qquad "
          r"\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol\delta^{[l]}"),
         ("<b>Cost.</b> The backward pass costs about the same as the forward "
          "pass — two matrix products per layer instead of one. So computing "
          "<i>all</i> $p$ gradients costs $\\mathcal{O}(1)$ forward passes, not "
          "$\\mathcal{O}(p)$. Finite differences would need $p+1$ forward passes; "
          "for a network with $10^7$ parameters that is the difference between "
          "milliseconds and weeks. <b>This efficiency is the entire reason deep "
          "learning is possible.</b>", None)],
        title="Backpropagation from the chain rule",
    )

    sub("The five steps of one training iteration")

    table(
        ["#", "Step", "What happens"],
        [["1", "Take a <b>mini-batch</b>",
          "Typically 32 instances; one pass over the whole set is an <b>epoch</b>"],
         ["2", "<b>Forward pass</b>",
          "Compute every layer's output, <b>keeping the intermediate results</b>"],
         ["3", "Measure the error", "Apply the loss function to the output"],
         ["4", "<b>Backward pass</b>",
          "Chain-rule the error back through every layer, computing each "
          "connection's contribution"],
         ["5", "Gradient descent step",
          "$\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\nabla$"]],
    )

    pitfall(
        "Initialise the weights randomly — this is not optional",
        "If all weights start at zero (or any identical value), every neuron in a "
        "layer computes the same thing, receives the same gradient, and updates "
        "identically. They stay identical forever — the layer behaves as a single "
        "neuron no matter how wide it is. This is <b>symmetry breaking</b>, and "
        "random initialisation is what breaks it. Chapter 11 shows that "
        "<i>which</i> random distribution you choose also matters enormously.",
    )

    sub("Why the activation function must be non-linear")

    proof(
        "A stack of linear layers is a single linear layer",
        "If $\\phi$ is the identity, then $\\mathbf{a}^{[2]} = \\mathbf{W}^{[2]}"
        "(\\mathbf{W}^{[1]}\\mathbf{x} + \\mathbf{b}^{[1]}) + \\mathbf{b}^{[2]} = "
        "(\\mathbf{W}^{[2]}\\mathbf{W}^{[1]})\\mathbf{x} + (\\mathbf{W}^{[2]}"
        "\\mathbf{b}^{[1]} + \\mathbf{b}^{[2]})$, which is just "
        "$\\mathbf{W}'\\mathbf{x} + \\mathbf{b}'$. By induction, <b>any</b> number "
        "of linear layers collapses to one. Depth buys you nothing without a "
        "non-linearity between the layers.",
    )

    sub("The activation functions")

    table(
        ["Function", "Definition", "Derivative", "Range", "Notes"],
        [["<b>step</b>", "$\\mathbb{1}[z \\ge 0]$", "0 a.e.", "$\\{0,1\\}$",
          "Unusable for gradient descent"],
         ["<b>sigmoid</b> $\\sigma$", "$1/(1+e^{-z})$",
          "$\\sigma(1-\\sigma) \\le 0.25$", "$(0,1)$",
          "Saturates; kills gradients (Ch. 11)"],
         ["<b>tanh</b>", "$\\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$", "$1-\\tanh^2 z$",
          "$(-1,1)$", "Zero-centred, so better than sigmoid; still saturates"],
         ["<b>ReLU</b>", "$\\max(0, z)$", "$\\mathbb{1}[z>0]$", "$[0,\\infty)$",
          "<b>The default.</b> Fast, no saturation for $z>0$; can 'die'"],
         ["<b>Leaky ReLU</b>", "$\\max(\\alpha z,\\, z)$",
          "$\\alpha$ or 1", "$(-\\infty,\\infty)$", "Fixes dying ReLU"],
         ["<b>ELU / GELU / Swish</b>", "smooth variants", "smooth",
          "$(-\\alpha,\\infty)$", "Chapter 11 compares them properly"]],
    )

    anim_header("Backpropagation, one pass at a time")
    md(
        "A tiny 2–3–1 network on XOR. Watch the forward pass light up left to "
        "right, then the error signal $\\boldsymbol\\delta$ propagate right to "
        "left, then the weights update. The loss curve is on the right."
    )

    rng = np.random.default_rng(7)
    Xn = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    yn = np.array([[0.], [1.], [1.], [0.]])
    W1 = rng.normal(0, 1.1, (2, 3)); b1 = np.zeros(3)
    W2 = rng.normal(0, 1.1, (3, 1)); b2 = np.zeros(1)

    def sig(z): return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    snaps = []
    eta = 2.5
    for ep in range(400):
        z1 = Xn @ W1 + b1; a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2; a2 = sig(z2)
        loss = float(-np.mean(yn * np.log(a2 + 1e-12)
                              + (1 - yn) * np.log(1 - a2 + 1e-12)))
        d2 = (a2 - yn) / len(Xn)
        gW2 = a1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W2.T) * (1 - a1 ** 2)
        gW1 = Xn.T @ d1; gb1 = d1.sum(0)
        if ep % 8 == 0:
            snaps.append((a1.copy(), a2.copy(), d1.copy(), d2.copy(),
                          W1.copy(), W2.copy(), loss))
        W2 -= eta * gW2; b2 -= eta * gb2
        W1 -= eta * gW1; b1 -= eta * gb1

    node_pos = {"i0": (0, 1.1), "i1": (0, -1.1),
                "h0": (1, 1.5), "h1": (1, 0.0), "h2": (1, -1.5),
                "o0": (2, 0.0)}
    edges = [("i0", "h0"), ("i0", "h1"), ("i0", "h2"),
             ("i1", "h0"), ("i1", "h1"), ("i1", "h2"),
             ("h0", "o0"), ("h1", "o0"), ("h2", "o0")]

    frames = []
    for s, (a1s, a2s, d1s, d2s, W1s, W2s, loss) in enumerate(snaps):
        act = {"i0": Xn[1, 0], "i1": Xn[1, 1],
               "h0": a1s[1, 0], "h1": a1s[1, 1], "h2": a1s[1, 2],
               "o0": a2s[1, 0]}
        err = {"h0": abs(d1s[1, 0]), "h1": abs(d1s[1, 1]), "h2": abs(d1s[1, 2]),
               "o0": abs(d2s[1, 0]), "i0": 0., "i1": 0.}
        emax = max(max(err.values()), 1e-9)
        ex, ey, ew, ec = [], [], [], []
        for a, b_ in edges:
            ex += [node_pos[a][0], node_pos[b_][0], None]
            ey += [node_pos[a][1], node_pos[b_][1], None]
        wts = np.r_[W1s.ravel(), W2s.ravel()]
        edge_traces = []
        for idx, (a, b_) in enumerate(edges):
            w = wts[idx]
            edge_traces.append(go.Scatter(
                x=[node_pos[a][0], node_pos[b_][0]],
                y=[node_pos[a][1], node_pos[b_][1]], mode="lines",
                line=dict(color=C["success"] if w > 0 else C["danger"],
                          width=max(.6, min(9, abs(w) * 2.4))),
                showlegend=False, hoverinfo="skip"))
        nodes = go.Scatter(
            x=[node_pos[k][0] for k in node_pos],
            y=[node_pos[k][1] for k in node_pos], mode="markers+text",
            text=[f"{act[k]:.2f}" for k in node_pos],
            textposition="middle center", textfont=dict(size=9, color="#fff"),
            marker=dict(size=[34 + 26 * err[k] / emax for k in node_pos],
                        color=[act[k] for k in node_pos],
                        colorscale=nav.cscale(), cmin=-1, cmax=1,
                        line=dict(color=C["ink"], width=2)),
            showlegend=False)
        loss_tr = go.Scatter(x=list(range(1, s + 2)),
                             y=[t[6] for t in snaps[:s + 1]], mode="lines",
                             line=dict(color=C["danger"], width=3),
                             showlegend=False)
        frames.append(go.Frame(name=str(s * 8), data=edge_traces + [nodes, loss_tr],
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"epoch {s*8}   ·   loss = {loss:.5f}   ·   "
                                   f"node size ∝ |δ|, edge width ∝ |w|")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.6, .4],
                      subplot_titles=("the 2–3–1 network (input [0,1])",
                                      "binary cross-entropy"))
    a1s, a2s, d1s, d2s, W1s, W2s, loss0 = snaps[0]
    wts0 = np.r_[W1s.ravel(), W2s.ravel()]
    for idx, (a, b_) in enumerate(edges):
        w = wts0[idx]
        f.add_trace(go.Scatter(x=[node_pos[a][0], node_pos[b_][0]],
                               y=[node_pos[a][1], node_pos[b_][1]], mode="lines",
                               line=dict(color=C["success"] if w > 0 else C["danger"],
                                         width=max(.6, min(9, abs(w) * 2.4))),
                               showlegend=False, hoverinfo="skip"), 1, 1)
    f.add_trace(go.Scatter(x=[node_pos[k][0] for k in node_pos],
                           y=[node_pos[k][1] for k in node_pos],
                           mode="markers+text", showlegend=False,
                           marker=dict(size=36, color=C["primary"],
                                       line=dict(color=C["ink"], width=2))), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[loss0], mode="lines", showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.update_xaxes(visible=False, range=[-.4, 2.4], row=1, col=1)
    f.update_yaxes(visible=False, range=[-2.2, 2.2], row=1, col=1)
    f.update_xaxes(title_text="snapshot", row=1, col=2)
    f.update_yaxes(title_text="loss", row=1, col=2)
    f.update_layout(height=490, plot_bgcolor="#FFFFFF",
                    title="Backpropagation on XOR")
    anim.animate(f, frames, duration=nav.anim_ms(120), slider_prefix="epoch ")
    figure(f)

    anim_header("Every activation function and its derivative")

    z = np.linspace(-5, 5, 500)
    acts = {
        "step": ((z >= 0).astype(float), np.zeros_like(z)),
        "sigmoid": (1 / (1 + np.exp(-z)),
                    (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
        "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
        "Leaky ReLU (α=0.1)": (np.where(z > 0, z, .1 * z),
                               np.where(z > 0, 1., .1)),
        "ELU (α=1)": (np.where(z > 0, z, np.exp(z) - 1),
                      np.where(z > 0, 1., np.exp(z))),
        "GELU": (z * .5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z ** 3))),
                 np.gradient(z * .5 * (1 + np.tanh(np.sqrt(2 / np.pi)
                                                   * (z + .044715 * z ** 3))), z)),
        "Swish (β=1)": (z / (1 + np.exp(-z)),
                        np.gradient(z / (1 + np.exp(-z)), z)),
    }
    frames = [go.Frame(name=nm, data=[
        go.Scatter(x=z, y=fv, mode="lines", line=dict(color=C["primary"], width=3.6)),
        go.Scatter(x=z, y=dv, mode="lines", line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[-5, 5], y=[0, 0], mode="lines",
                   line=dict(color=C["muted"], width=1, dash="dot")),
    ], layout=go.Layout(title=f"{nm}   ·   max |φ'| = {np.abs(dv).max():.3f}"))
        for nm, (fv, dv) in acts.items()]

    nm0 = list(acts)[0]
    f = go.Figure(data=[
        go.Scatter(x=z, y=acts[nm0][0], mode="lines", name="φ(z)",
                   line=dict(color=C["primary"], width=3.6)),
        go.Scatter(x=z, y=acts[nm0][1], mode="lines", name="φ'(z)",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[-5, 5], y=[0, 0], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1, dash="dot")),
    ])
    f.update_layout(height=440, xaxis_title="z", yaxis=dict(range=[-1.6, 2.6]),
                    title=nm0, legend=dict(orientation="h", y=1.02,
                                           yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="")
    figure(f, "Note the red curve for the sigmoid: it never exceeds 0.25, so "
              "every layer multiplies the gradient by at most a quarter. That is "
              "the vanishing-gradient problem, and Chapter 11 is largely about it.")

    code_lab(
        "An MLP with backprop from scratch, verified by numerical gradients",
        '''import numpy as np

# ============ a complete MLP in ~50 lines of NumPy =====================
class MLP:
    """Fully connected network, tanh hidden layers, softmax output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        # He-ish initialisation; Chapter 11 explains the scaling properly
        self.W = [rng.normal(0, np.sqrt(2/sizes[i]), (sizes[i], sizes[i+1]))
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros(sizes[i+1]) for i in range(len(sizes)-1)]

    def forward(self, X):
        """Returns (activations, pre-activations) -- we KEEP them for backprop."""
        a = [X]; z = []
        for l in range(len(self.W)):
            zl = a[-1] @ self.W[l] + self.b[l]
            z.append(zl)
            if l < len(self.W) - 1:
                a.append(np.tanh(zl))                    # hidden: tanh
            else:
                e = np.exp(zl - zl.max(1, keepdims=True))  # output: softmax
                a.append(e / e.sum(1, keepdims=True))
        return a, z

    def loss(self, X, Y):
        a, _ = self.forward(X)
        return float(-np.mean(np.sum(Y * np.log(a[-1] + 1e-12), axis=1)))

    def backward(self, X, Y):
        """THE ALGORITHM. Returns dL/dW and dL/db for every layer."""
        m = len(X)
        a, z = self.forward(X)
        gW = [None]*len(self.W); gb = [None]*len(self.b)

        # output layer: softmax + cross-entropy => delta = y_hat - y  (4.7)
        delta = (a[-1] - Y) / m

        for l in reversed(range(len(self.W))):
            gW[l] = a[l].T @ delta                       # dL/dW = a^{l-1} delta^T
            gb[l] = delta.sum(0)                         # dL/db = delta
            if l > 0:
                # pull back through the weights, then through tanh'
                delta = (delta @ self.W[l].T) * (1 - a[l]**2)
        return gW, gb

    def fit(self, X, Y, epochs=400, lr=.5, batch=32, seed=0):
        rng = np.random.default_rng(seed); hist = []
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b_ = idx[s:s+batch]
                gW, gb = self.backward(X[b_], Y[b_])
                for l in range(len(self.W)):
                    self.W[l] -= lr * gW[l]
                    self.b[l] -= lr * gb[l]
            hist.append(self.loss(X, Y))
        return hist

    def predict(self, X):
        return self.forward(X)[0][-1].argmax(1)

# ============ verify the gradients numerically =========================
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (12, 4))
y = rng.integers(0, 3, 12)
Y = np.eye(3)[y]

net = MLP([4, 6, 5, 3], seed=1)
gW, gb = net.backward(X, Y)

print("=== gradient check: backprop vs central finite differences ===")
eps = 1e-6
max_err = 0.0
for l in range(len(net.W)):
    num = np.zeros_like(net.W[l])
    for i in range(net.W[l].shape[0]):
        for j in range(net.W[l].shape[1]):
            orig = net.W[l][i, j]
            net.W[l][i, j] = orig + eps; lp = net.loss(X, Y)
            net.W[l][i, j] = orig - eps; lm = net.loss(X, Y)
            net.W[l][i, j] = orig
            num[i, j] = (lp - lm) / (2*eps)
    rel = np.abs(gW[l] - num).max() / max(np.abs(num).max(), 1e-12)
    max_err = max(max_err, rel)
    print(f"  layer {l}: W shape {str(net.W[l].shape):>8}   "
          f"max relative error = {rel:.3e}")
print(f"\\nworst relative error = {max_err:.3e}   "
      f"{'PASS -- backprop is correct' if max_err < 1e-5 else 'FAIL'}")

# ============ backprop vs finite differences: THE COST =================
print("\\n=== why we do not just use finite differences ===")
n_params = sum(w.size for w in net.W) + sum(b.size for b in net.b)
print(f"this tiny network has {n_params} parameters")
print(f"  backprop           : 1 forward + 1 backward  ~= 2 forward passes")
print(f"  finite differences : {2*n_params} forward passes  "
      f"({n_params} params x 2 evaluations)")
print(f"  ratio              : {2*n_params/2:.0f}x")
for p in [1e4, 1e6, 1e9]:
    print(f"  a network with {p:.0e} parameters -> finite differences is "
          f"{p:.0e}x slower")

# ============ train it on a real problem ===============================
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
sc = StandardScaler().fit(Xtr)
Atr, Ate = sc.transform(Xtr), sc.transform(Xte)
Ytr = np.eye(10)[ytr]

net = MLP([64, 64, 32, 10], seed=3)
hist = net.fit(Atr, Ytr, epochs=250, lr=.35, batch=32)
print(f"\\n=== trained from scratch on digits ===")
print(f"final training loss = {hist[-1]:.5f}")
print(f"train accuracy = {np.mean(net.predict(Atr) == ytr):.4f}")
print(f"test  accuracy = {np.mean(net.predict(Ate) == yte):.4f}")

# ============ SYMMETRY BREAKING ========================================
print("\\n=== why weights must be initialised randomly ===")
zero_net = MLP([64, 32, 10], seed=0)
for l in range(len(zero_net.W)):
    zero_net.W[l][:] = 0.0                       # all weights identical
h0 = zero_net.fit(Atr, Ytr, epochs=60, lr=.3)
print(f"all-zero init : final loss {h0[-1]:.5f}, "
      f"test accuracy {np.mean(zero_net.predict(Ate) == yte):.4f}")
print(f"  hidden layer rows all identical? "
      f"{np.allclose(zero_net.W[0][0], zero_net.W[0][1])}")
rand_net = MLP([64, 32, 10], seed=0)
h1 = rand_net.fit(Atr, Ytr, epochs=60, lr=.3)
print(f"random init   : final loss {h1[-1]:.5f}, "
      f"test accuracy {np.mean(rand_net.predict(Ate) == yte):.4f}")

# ============ LINEAR ACTIVATIONS COLLAPSE ==============================
print("\\n=== a stack of linear layers IS one linear layer ===")
W1 = rng.normal(0, 1, (4, 7)); b1 = rng.normal(0, 1, 7)
W2 = rng.normal(0, 1, (7, 5)); b2 = rng.normal(0, 1, 5)
W3 = rng.normal(0, 1, (5, 3)); b3 = rng.normal(0, 1, 3)
deep   = ((X @ W1 + b1) @ W2 + b2) @ W3 + b3
W_eq = W1 @ W2 @ W3
b_eq = (b1 @ W2 + b2) @ W3 + b3
shallow = X @ W_eq + b_eq
print(f"3 linear layers vs 1 equivalent layer: max |difference| = "
      f"{np.abs(deep - shallow).max():.2e}")
print("Depth without a non-linearity buys you exactly nothing.")

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(y=hist, mode="lines",
                           line=dict(color=C["primary"], width=2.5)))
fig.update_layout(height=360, xaxis_title="epoch", yaxis_title="cross-entropy",
                  yaxis_type="log", title="Training the from-scratch MLP")
''',
        key="ch10_backprop",
    )

    quiz(
        "Why is backpropagation dramatically better than computing gradients by "
        "finite differences?",
        ["It is more numerically accurate",
         "It computes all $p$ gradients in $\\mathcal{O}(1)$ forward passes rather "
         "than $\\mathcal{O}(p)$",
         "It works with non-differentiable activations",
         "It avoids local minima"],
        1,
        "Reverse-mode autodiff gets *every* partial derivative from one forward "
        "and one backward pass. Finite differences needs two evaluations per "
        "parameter. For $10^7$ parameters that is a factor of $10^7$ — the "
        "difference between feasible and impossible.",
        key="ch10q1",
    )

    keypoints([
        "MLP: $\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]}\\mathbf{a}^{[l-1]} + "
        "\\mathbf{b}^{[l]}$, $\\mathbf{a}^{[l]} = \\phi(\\mathbf{z}^{[l]})$.",
        "Backprop = reverse-mode autodiff: "
        "$\\boldsymbol\\delta^{[l]} = (\\mathbf{W}^{[l+1]\\top}"
        "\\boldsymbol\\delta^{[l+1]}) \\odot \\phi'(\\mathbf{z}^{[l]})$.",
        "All gradients in <b>two passes</b>, not $2p$ — this is why deep learning "
        "is computationally possible.",
        "Softmax + cross-entropy gives $\\boldsymbol\\delta^{[L]} = "
        "\\hat{\\mathbf{y}} - \\mathbf{y}$ exactly.",
        "Random init breaks symmetry; a non-linear $\\phi$ is what makes depth "
        "mean anything.",
    ])


# ==========================================================================
def s_10_3():
    section("10.3", "Regression MLPs")

    lead(
        "For regression you need one output neuron per predicted value, and the "
        "output activation is usually *nothing at all*."
    )

    table(
        ["Hyperparameter", "Typical value"],
        [["# input neurons", "One per input feature"],
         ["# hidden layers", "1–5, depending on the problem"],
         ["# neurons per hidden layer", "10–100"],
         ["# output neurons", "1 per prediction dimension"],
         ["Hidden activation", "<b>ReLU</b> (or Swish / GELU)"],
         ["<b>Output activation</b>",
          "<b>None</b>, or ReLU/softplus for positive outputs, or "
          "sigmoid/tanh for a bounded range"],
         ["Loss", "MSE; MAE or Huber if there are outliers"]],
    )

    warn(
        "Do not put an activation on the output layer without a reason",
        "A ReLU or sigmoid on the output <b>clamps the range</b>. ReLU makes "
        "negative predictions impossible — correct if you are predicting a price, "
        "wrong if you are predicting a temperature change. Sigmoid restricts you "
        "to $(0,1)$. The default should be <b>no activation</b>, and any "
        "restriction should be a deliberate encoding of domain knowledge.",
    )

    sub("The Huber loss")

    md(
        "The standard robust compromise: quadratic near zero (so the gradient "
        "shrinks as you converge) and linear far away (so outliers do not "
        "dominate):"
    )

    math(r"""
    L_\delta(a) =
    \begin{cases}
      \tfrac{1}{2}a^{2} & \text{if } |a| \le \delta \\[6pt]
      \delta\bigl(|a| - \tfrac{1}{2}\delta\bigr) & \text{otherwise}
    \end{cases}
    \qquad a = y - \hat y
    """)

    proof(
        "Huber is C¹ — it is differentiable at the join",
        "At $a = \\delta$ the quadratic branch has value $\\delta^2/2$ and slope "
        "$\\delta$; the linear branch has value "
        "$\\delta(\\delta - \\delta/2) = \\delta^2/2$ and slope $\\delta$. Both "
        "match, so the loss and its first derivative are continuous everywhere. "
        "That smoothness is what makes it usable with gradient descent, unlike "
        "plain MAE which has a kink at zero.",
    )

    anim_header("MSE, MAE and Huber under a growing outlier")

    a = np.linspace(-4, 4, 400)
    frames = []
    for dlt in np.linspace(.2, 3.0, 30):
        hub = np.where(np.abs(a) <= dlt, .5 * a ** 2,
                       dlt * (np.abs(a) - .5 * dlt))
        frames.append(go.Frame(name=f"{dlt:.2f}", data=[
            go.Scatter(x=a, y=.5 * a ** 2, mode="lines",
                       line=dict(color=C["danger"], width=2.6)),
            go.Scatter(x=a, y=np.abs(a), mode="lines",
                       line=dict(color=C["warning"], width=2.6)),
            go.Scatter(x=a, y=hub, mode="lines",
                       line=dict(color=C["success"], width=3.8)),
            go.Scatter(x=[-dlt, -dlt], y=[0, 8], mode="lines",
                       line=dict(color=C["muted"], width=1.5, dash="dot")),
            go.Scatter(x=[dlt, dlt], y=[0, 8], mode="lines",
                       line=dict(color=C["muted"], width=1.5, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"δ = {dlt:.2f}   ·   quadratic inside ±δ, linear outside")])))

    f = go.Figure(data=[
        go.Scatter(x=a, y=.5 * a ** 2, mode="lines", name="½·MSE",
                   line=dict(color=C["danger"], width=2.6)),
        go.Scatter(x=a, y=np.abs(a), mode="lines", name="MAE",
                   line=dict(color=C["warning"], width=2.6)),
        go.Scatter(x=a, y=np.where(np.abs(a) <= .2, .5 * a ** 2,
                                   .2 * (np.abs(a) - .1)),
                   mode="lines", name="Huber",
                   line=dict(color=C["success"], width=3.8)),
        go.Scatter(x=[-.2, -.2], y=[0, 8], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1.5, dash="dot")),
        go.Scatter(x=[.2, .2], y=[0, 8], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1.5, dash="dot")),
    ])
    f.update_layout(height=420, xaxis_title="residual a = y − ŷ",
                    yaxis_title="loss", yaxis=dict(range=[0, 8]),
                    title="Huber interpolates between MSE and MAE",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="δ = ")
    figure(f)

    code_lab(
        "A regression MLP in Keras, with and without outliers",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
np.random.seed(42)

# ---- data (falls back to synthetic if the download is unavailable) ----
try:
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print(f"California housing: {X.shape}")
except Exception:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8000, 8))
    y = X @ rng.normal(0, 1, 8) + rng.normal(0, .3, 8000)
    print("using synthetic data (no network)")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=.2, random_state=42)

# ---- a Normalization LAYER, so scaling ships with the model ----------
norm = keras.layers.Normalization()
norm.adapt(X_train)                       # learns the mean and variance

model = keras.Sequential([
    norm,
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(1),                # NO activation -- unbounded output
])
model.compile(loss="mse",
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["RootMeanSquaredError"])
model.summary()

hist = model.fit(X_train, y_train, epochs=12, batch_size=64,
                 validation_data=(X_valid, y_valid), verbose=0)
loss, rmse = model.evaluate(X_test, y_test, verbose=0)
print(f"\\ntest MSE = {loss:.4f}   test RMSE = {rmse:.4f}")
print(f"predictions for the first 3 instances: "
      f"{model.predict(X_test[:3], verbose=0).ravel().round(3)}")
print(f"true values:                          {y_test[:3].round(3)}")

# ============ MSE vs MAE vs Huber, with outliers ======================
print("\\n=== loss functions under 3 % contamination ===")
rng = np.random.default_rng(0)
y_dirty = y_train.copy()
bad = rng.choice(len(y_dirty), int(.03*len(y_dirty)), replace=False)
y_dirty[bad] += rng.normal(0, 15, len(bad))

def build(loss):
    n = keras.layers.Normalization(); n.adapt(X_train)
    m = keras.Sequential([n,
                          keras.layers.Dense(50, activation="relu"),
                          keras.layers.Dense(50, activation="relu"),
                          keras.layers.Dense(1)])
    m.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3),
              metrics=["RootMeanSquaredError"])
    return m

print(f"{'loss':<22}{'clean training':>17}{'contaminated':>16}")
for name, loss in [("mse", "mse"),
                   ("mae", "mae"),
                   ("huber (delta=1)", keras.losses.Huber(delta=1.0))]:
    a = build(loss); a.fit(X_train, y_train, epochs=8, batch_size=128, verbose=0)
    b = build(loss); b.fit(X_train, y_dirty, epochs=8, batch_size=128, verbose=0)
    ra = a.evaluate(X_test, y_test, verbose=0)[1]
    rb = b.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{name:<22}{ra:>17.4f}{rb:>16.4f}")
print("\\nHuber and MAE barely degrade; MSE chases the outliers.")

# ============ OUTPUT ACTIVATION: encode what you know =================
print("\\n=== the output activation encodes domain knowledge ===")
for name, act in [("none (default)", None), ("relu (y >= 0)", "relu"),
                  ("softplus (y > 0, smooth)", "softplus")]:
    n = keras.layers.Normalization(); n.adapt(X_train)
    m = keras.Sequential([n, keras.layers.Dense(50, activation="relu"),
                          keras.layers.Dense(1, activation=act)])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3),
              metrics=["RootMeanSquaredError"])
    m.fit(X_train, y_train, epochs=8, batch_size=128, verbose=0)
    p = m.predict(X_test, verbose=0).ravel()
    print(f"  {name:<26} RMSE {m.evaluate(X_test, y_test, verbose=0)[1]:.4f}   "
          f"min prediction {p.min():+.3f}")
print("relu/softplus make negative predictions IMPOSSIBLE. Only use them")
print("when negative values are genuinely impossible in your domain.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hist.history["loss"], mode="lines", name="train",
                line=dict(color=C["train"], width=2.5))
fig.add_scatter(y=hist.history["val_loss"], mode="lines", name="validation",
                line=dict(color=C["test"], width=2.5))
fig.update_layout(height=380, xaxis_title="epoch", yaxis_title="MSE",
                  yaxis_type="log", title="Learning curves")
''',
        key="ch10_regression",
    )

    keypoints([
        "One output neuron per predicted value; <b>no output activation</b> by "
        "default.",
        "Use ReLU/softplus/sigmoid on the output only to encode a genuine domain "
        "constraint.",
        "MSE by default; <b>Huber</b> or MAE when outliers are present.",
        "Huber is $C^1$ — quadratic inside $\\pm\\delta$, linear outside, smooth at "
        "the join.",
        "A <code>Normalization</code> layer with <code>adapt()</code> ships the "
        "scaling <i>inside</i> the model.",
    ])


# ==========================================================================
def s_10_4():
    section("10.4", "Classification MLPs")

    lead(
        "Three cases — binary, multilabel binary, and multiclass — and the output "
        "layer differs in each. Getting this wrong is the single most common "
        "beginner error."
    )

    table(
        ["", "Binary", "Multilabel binary", "Multiclass"],
        [["Output neurons", "1", "1 per label", "1 per class"],
         ["<b>Output activation</b>", "<b>sigmoid</b>", "<b>sigmoid</b>",
          "<b>softmax</b>"],
         ["Loss", "<code>binary_crossentropy</code>",
          "<code>binary_crossentropy</code>",
          "<code>categorical_crossentropy</code> (one-hot) or "
          "<code>sparse_categorical_crossentropy</code> (integer labels)"],
         ["Outputs sum to 1?", "n/a", "❌ independent", "✅ enforced"],
         ["Example", "Is this email spam?", "Which of these tags apply?",
          "Which digit is this?"]],
    )

    pitfall(
        "Softmax vs sigmoid — the error everyone makes once",
        "<b>Softmax forces the outputs to compete</b> ($\\sum_k \\hat p_k = 1$), so "
        "it is correct <i>only</i> when the classes are mutually exclusive. If an "
        "instance can carry several labels, softmax is wrong and you need $L$ "
        "independent <b>sigmoids</b> — see §3.8. The symptom of getting it wrong "
        "is a model that can never predict two labels confidently at once.",
    )

    codenote(
        "sparse_categorical_crossentropy vs categorical_crossentropy",
        "They compute the same loss; they differ in the label <b>format</b>. "
        "<code>sparse_</code> takes integer labels <code>[3, 7, 1]</code>; the "
        "plain version takes one-hot rows. Use the sparse version — it saves "
        "memory and the conversion step. If you get a shape mismatch error, this "
        "is nearly always why.",
    )

    warn(
        "from_logits=True is more numerically stable",
        "Leaving the output layer <b>without</b> an activation and passing "
        "<code>from_logits=True</code> to the loss lets Keras use the "
        "log-sum-exp trick internally, avoiding overflow when the logits are "
        "large. It is strictly better numerically. Remember that "
        "<code>model.predict</code> then returns <b>logits</b>, not "
        "probabilities — apply the softmax/sigmoid yourself, or add a separate "
        "<code>Activation</code> layer for inference.",
    )

    anim_header("The three output heads, on the same hidden representation")

    z = np.linspace(-4, 4, 300)
    logits3 = np.c_[z, np.full_like(z, .6), -0.5 * z]

    def softmax(S):
        S = S - S.max(1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(1, keepdims=True)

    sig = 1 / (1 + np.exp(-z))
    sm = softmax(logits3)
    sig3 = 1 / (1 + np.exp(-logits3))

    views = [
        ("BINARY — 1 neuron, sigmoid",
         [(z, sig, "P(positive)", C["primary"]),
          (z, 1 - sig, "P(negative)", C["accent"])], True),
        ("MULTICLASS — 3 neurons, softmax (they compete, Σ = 1)",
         [(z, sm[:, k], f"P(class {k})", SEQ[k]) for k in range(3)], True),
        ("MULTILABEL — 3 neurons, independent sigmoids (Σ ≠ 1)",
         [(z, sig3[:, k], f"P(label {k})", SEQ[k]) for k in range(3)], False),
    ]
    frames = []
    for i, (title, curves, sums_to_one) in enumerate(views):
        data = [go.Scatter(x=xx, y=yy, mode="lines", name=nm,
                           line=dict(color=cc, width=3.4))
                for xx, yy, nm, cc in curves]
        tot = np.sum([c[1] for c in curves], axis=0)
        data.append(go.Scatter(x=z, y=tot, mode="lines", name="sum",
                               line=dict(color=C["ink"], width=2, dash="dash")))
        frames.append(go.Frame(name=title.split()[0], data=data,
                               layout=go.Layout(title=title)))

    f = go.Figure(data=[go.Scatter(x=xx, y=yy, mode="lines", name=nm,
                                   line=dict(color=cc, width=3.4))
                        for xx, yy, nm, cc in views[0][1]]
                  + [go.Scatter(x=z, y=np.ones_like(z), mode="lines", name="sum",
                                line=dict(color=C["ink"], width=2, dash="dash"))])
    f.update_layout(height=430, xaxis_title="a hidden unit's value",
                    yaxis_title="output probability",
                    yaxis=dict(range=[-.05, 3.1]), title=views[0][0],
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1900), slider_prefix="head ")
    figure(f, "The dashed line is the sum of the outputs. Under softmax it is "
              "pinned to 1; under independent sigmoids it is free to be anything "
              "between 0 and 3.")

    code_lab(
        "All three classification heads, side by side",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)

d = load_digits()
X, y = d.data / 16.0, d.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, stratify=y, random_state=42)

def mlp(out_units, out_act, loss, metrics):
    m = keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(out_units, activation=out_act),
    ])
    m.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=metrics)
    return m

# ============ 1. BINARY: is this a 5? ==================================
print("=== BINARY: 1 neuron + sigmoid + binary_crossentropy ===")
b_tr, b_te = (y_train == 5).astype(float), (y_test == 5).astype(float)
mb = mlp(1, "sigmoid", "binary_crossentropy", ["accuracy"])
mb.fit(X_train, b_tr, epochs=18, batch_size=32, verbose=0)
print(f"test accuracy = {mb.evaluate(X_test, b_te, verbose=0)[1]:.4f}")
p = mb.predict(X_test[:4], verbose=0).ravel()
print(f"first 4 P(is a 5) = {p.round(4)}   (each independent, in (0,1))")

# ============ 2. MULTICLASS: which digit? ==============================
print("\\n=== MULTICLASS: 10 neurons + softmax + sparse_categorical_crossentropy ===")
mm = mlp(10, "softmax", "sparse_categorical_crossentropy", ["accuracy"])
mm.fit(X_train, y_train, epochs=18, batch_size=32, verbose=0)
print(f"test accuracy = {mm.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
P = mm.predict(X_test[:3], verbose=0)
print(f"probability rows sum to {P.sum(1).round(6)}   <- ENFORCED by softmax")
for i in range(3):
    print(f"  true {y_test[i]}, predicted {P[i].argmax()}, "
          f"confidence {P[i].max():.4f}")

# ============ 3. MULTILABEL: several properties at once ================
print("\\n=== MULTILABEL: 3 neurons + sigmoid + binary_crossentropy ===")
def props(yv):
    return np.c_[yv >= 7, yv % 2 == 1, np.isin(yv, [0, 6, 8, 9])].astype(float)
Ytr, Yte = props(y_train), props(y_test)
ml = mlp(3, "sigmoid", "binary_crossentropy", ["accuracy"])
ml.fit(X_train, Ytr, epochs=18, batch_size=32, verbose=0)
Pm = ml.predict(X_test[:3], verbose=0)
print(f"rows sum to {Pm.sum(1).round(4)}   <- NOT constrained; that is the point")
names = ["large(>=7)", "odd", "closed-loop"]
for i in range(3):
    print(f"  digit {y_test[i]}: " +
          "  ".join(f"{n}={Pm[i,j]:.3f}(true {Yte[i,j]:.0f})"
                    for j, n in enumerate(names)))

# ============ 4. WHAT GOES WRONG with softmax on multilabel ============
print("\\n=== the classic mistake: softmax on a multilabel problem ===")
wrong = mlp(3, "softmax", "binary_crossentropy", ["accuracy"])
wrong.fit(X_train, Ytr, epochs=18, batch_size=32, verbose=0)
Pw = wrong.predict(X_test[:60], verbose=0)
print(f"softmax head : mean max probability = {Pw.max(1).mean():.4f}, "
      f"rows summing to 1 = {np.allclose(Pw.sum(1), 1)}")
print(f"  instances where TWO labels exceed 0.5: "
      f"{int(np.sum((Pw > .5).sum(1) >= 2))}/60")
print(f"sigmoid head : instances where two labels exceed 0.5: "
      f"{int(np.sum((ml.predict(X_test[:60], verbose=0) > .5).sum(1) >= 2))}/60")
print(f"  ground truth: {int(np.sum(Yte[:60].sum(1) >= 2))}/60 really have 2+ labels")
print("\\nSoftmax structurally CANNOT report two confident labels. That is the bug.")

# ============ 5. from_logits=True ======================================
print("\\n=== from_logits=True is numerically safer ===")
logit_model = keras.Sequential([
    keras.layers.Input(shape=(64,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),                              # NO activation
])
logit_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
logit_model.fit(X_train, y_train, epochs=18, batch_size=32, verbose=0)
print(f"test accuracy = {logit_model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
raw = logit_model.predict(X_test[:2], verbose=0)
print(f"predict() returns LOGITS: {raw[0].round(2)}")
print(f"apply softmax yourself  : "
      f"{tf.nn.softmax(raw[0]).numpy().round(4)}")

# for deployment, append the activation
inference = keras.Sequential([logit_model, keras.layers.Softmax()])
print(f"or wrap it: {inference.predict(X_test[:1], verbose=0)[0].round(4)}")
''',
        key="ch10_classification",
    )

    keypoints([
        "Binary: 1 neuron, <b>sigmoid</b>, binary cross-entropy.",
        "Multilabel: $L$ neurons, <b>$L$ independent sigmoids</b>, binary "
        "cross-entropy.",
        "Multiclass: $K$ neurons, <b>softmax</b>, sparse categorical "
        "cross-entropy.",
        "Softmax forces competition — wrong whenever labels are not mutually "
        "exclusive.",
        "Prefer <code>from_logits=True</code> with no output activation; add the "
        "activation for inference.",
    ])


# ==========================================================================
def s_10_5():
    section("10.5", "Implementing MLPs with Keras — The Sequential API")

    lead(
        "The simplest Keras API: a linear stack of layers. Ninety percent of "
        "networks you will ever build fit in it."
    )

    sub("The five steps")

    md(
        """
1. **Build** — stack layers.
2. **Compile** — choose the loss, the optimiser, and the metrics.
3. **Fit** — train, with a validation set.
4. **Evaluate** — measure on the test set.
5. **Predict** — use it.
        """
    )

    table(
        ["Layer", "Purpose"],
        [["<code>Input(shape=…)</code>", "Declares the input shape"],
         ["<code>Flatten()</code>", "Reshapes $(28,28)$ into $(784,)$ — no "
          "parameters, pure reshaping"],
         ["<code>Dense(n, activation=…)</code>",
          "Fully connected: $\\phi(\\mathbf{X}\\mathbf{W} + \\mathbf{b})$"],
         ["<code>Normalization()</code>",
          "Standardisation as a layer; call <code>adapt()</code> on the training "
          "data"],
         ["<code>Dropout(rate)</code>", "Regularisation (Chapter 11)"],
         ["<code>BatchNormalization()</code>", "Normalises activations (Ch. 11)"]],
    )

    sub("Counting the parameters")

    math(r"""
    \text{params of a Dense layer} \;=\;
    \underbrace{n_{\text{in}} \times n_{\text{out}}}_{\text{weights}}
    \;+\; \underbrace{n_{\text{out}}}_{\text{biases}}
    """)

    md(
        "For the classic Fashion-MNIST network `[Flatten, Dense(300, relu), "
        "Dense(100, relu), Dense(10, softmax)]`:"
    )

    table(
        ["Layer", "Output shape", "Parameters", "Arithmetic"],
        [["Flatten", "(None, 784)", "0", "reshaping only"],
         ["Dense(300)", "(None, 300)", "<b>235 500</b>", "$784 \\times 300 + 300$"],
         ["Dense(100)", "(None, 100)", "<b>30 100</b>", "$300 \\times 100 + 100$"],
         ["Dense(10)", "(None, 10)", "<b>1 010</b>", "$100 \\times 10 + 10$"],
         ["<b>Total</b>", "", "<b>266 610</b>", ""]],
    )

    idea(
        "266 610 parameters on 60 000 examples",
        "Four times more parameters than training instances — enormous flexibility "
        "and an enormous overfitting risk. That imbalance is exactly why Chapter "
        "11 spends its length on regularisation, and why early stopping (below) is "
        "not optional.",
    )

    anim_header("Watching an MLP train, layer by layer")
    md(
        "A small MLP on the digits dataset, trained with NumPy so every "
        "intermediate quantity is visible. Top: the loss and accuracy curves. "
        "Bottom: the distribution of each layer's weights as they evolve."
    )

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    dg = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(dg.data, dg.target, test_size=.25,
                                          stratify=dg.target, random_state=0)
    sc = StandardScaler().fit(Xtr)
    Atr, Ate = sc.transform(Xtr), sc.transform(Xte)
    Ytr = np.eye(10)[ytr]

    rng = np.random.default_rng(2)
    sizes = [64, 40, 24, 10]
    Ws = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
          for i in range(3)]
    bs = [np.zeros(sizes[i + 1]) for i in range(3)]

    def fwd(X):
        a = [X]
        for l in range(3):
            zl = a[-1] @ Ws[l] + bs[l]
            if l < 2:
                a.append(np.tanh(zl))
            else:
                e = np.exp(zl - zl.max(1, keepdims=True))
                a.append(e / e.sum(1, keepdims=True))
        return a

    snaps = []
    for ep in range(120):
        idx = rng.permutation(len(Atr))
        for s in range(0, len(Atr), 32):
            bidx = idx[s:s + 32]
            a = fwd(Atr[bidx])
            delta = (a[-1] - Ytr[bidx]) / len(bidx)
            for l in reversed(range(3)):
                gW = a[l].T @ delta
                gb = delta.sum(0)
                if l > 0:
                    delta = (delta @ Ws[l].T) * (1 - a[l] ** 2)
                Ws[l] -= .35 * gW
                bs[l] -= .35 * gb
        if ep % 4 == 0:
            atr = fwd(Atr); ate = fwd(Ate)
            loss = float(-np.mean(np.sum(Ytr * np.log(atr[-1] + 1e-12), 1)))
            acc_tr = float(np.mean(atr[-1].argmax(1) == ytr))
            acc_te = float(np.mean(ate[-1].argmax(1) == yte))
            snaps.append((loss, acc_tr, acc_te, [w.ravel().copy() for w in Ws]))

    bins = np.linspace(-1.6, 1.6, 45)
    frames = []
    for s, (loss, atr_, ate_, wlist) in enumerate(snaps):
        data = [
            go.Scatter(x=[i * 4 for i in range(s + 1)],
                       y=[t[0] for t in snaps[:s + 1]], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=[i * 4 for i in range(s + 1)],
                       y=[t[1] for t in snaps[:s + 1]], mode="lines",
                       line=dict(color=C["train"], width=2.6)),
            go.Scatter(x=[i * 4 for i in range(s + 1)],
                       y=[t[2] for t in snaps[:s + 1]], mode="lines",
                       line=dict(color=C["test"], width=2.6)),
        ]
        for l in range(3):
            h = np.histogram(wlist[l], bins=bins)[0]
            data.append(go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=h,
                               marker=dict(color=alpha(SEQ[l], .75))))
        frames.append(go.Frame(name=str(s * 4), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"epoch {s*4}   ·   loss = {loss:.4f}   ·   "
                                   f"train acc = {atr_:.4f}   ·   "
                                   f"test acc = {ate_:.4f}")])))

    f = make_subplots(rows=2, cols=1, row_heights=[.55, .45],
                      vertical_spacing=.13,
                      subplot_titles=("loss (red) and accuracy (blue/orange)",
                                      "weight distributions per layer"),
                      specs=[[{"secondary_y": True}], [{}]])
    f.add_trace(go.Scatter(x=[0], y=[snaps[0][0]], mode="lines", name="loss",
                           line=dict(color=C["danger"], width=3)), 1, 1)
    f.add_trace(go.Scatter(x=[0], y=[snaps[0][1]], mode="lines",
                           name="train accuracy",
                           line=dict(color=C["train"], width=2.6)), 1, 1)
    f.add_trace(go.Scatter(x=[0], y=[snaps[0][2]], mode="lines",
                           name="test accuracy",
                           line=dict(color=C["test"], width=2.6)), 1, 1)
    for l in range(3):
        h = np.histogram(snaps[0][3][l], bins=bins)[0]
        f.add_trace(go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=h,
                           name=f"layer {l+1} weights",
                           marker=dict(color=alpha(SEQ[l], .75))), 2, 1)
    f.update_layout(height=580, barmode="overlay", bargap=.02,
                    title="Training an MLP")
    f.update_xaxes(title_text="epoch", row=1, col=1)
    f.update_xaxes(title_text="weight value", row=2, col=1)
    anim.animate(f, frames, duration=nav.anim_ms(120), slider_prefix="epoch ")
    figure(f, "The weight histograms start narrow (initialisation) and spread as "
              "the network learns — Chapter 11 shows what it means when they "
              "spread too far or not at all.")

    code_lab(
        "The full Keras Sequential workflow on Fashion-MNIST",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ---- data (falls back to digits offline) ------------------------------
try:
    (X_tr_full, y_tr_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_tr_full = X_tr_full / 255.0
    X_test = X_test / 255.0
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    INPUT_SHAPE = (28, 28)
    print("Fashion-MNIST loaded")
except Exception:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    Xi = d.images / 16.0
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        Xi, d.target, test_size=.25, random_state=42)
    class_names = [str(i) for i in range(10)]
    INPUT_SHAPE = (8, 8)
    print("no network -- using the 8x8 digits set instead")

X_valid, X_train = X_tr_full[:5000], X_tr_full[5000:]
y_valid, y_train = y_tr_full[:5000], y_tr_full[5000:]
print(f"train {X_train.shape}   valid {X_valid.shape}   test {X_test.shape}")

# ============ STEP 1: BUILD ===========================================
model = keras.Sequential([
    keras.layers.Input(shape=INPUT_SHAPE),
    keras.layers.Flatten(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10,  activation="softmax"),
])
model.summary()

# --- verify the parameter arithmetic by hand -------------------------
n_in = int(np.prod(INPUT_SHAPE))
print(f"\\nparameter count, by hand:")
print(f"  Dense(300): {n_in} x 300 + 300 = {n_in*300 + 300:,}")
print(f"  Dense(100): 300 x 100 + 100 = {300*100 + 100:,}")
print(f"  Dense(10) : 100 x  10 +  10 = {100*10 + 10:,}")
print(f"  TOTAL     = {n_in*300+300 + 300*100+100 + 100*10+10:,}")
print(f"  keras says  {model.count_params():,}")
print(f"\\ntraining instances: {len(X_train):,}")
print(f"parameters per training instance: "
      f"{model.count_params()/len(X_train):.1f}   <- overfitting risk")

# --- inspect a layer -------------------------------------------------
hidden1 = model.layers[1]
W, b = hidden1.get_weights()
print(f"\\nfirst hidden layer: W {W.shape}, b {b.shape}")
print(f"  W initialised with Glorot uniform: range "
      f"[{W.min():.4f}, {W.max():.4f}], std {W.std():.4f}")
print(f"  theoretical Glorot limit = "
      f"{np.sqrt(6/(n_in+300)):.4f}   (Chapter 11)")
print(f"  b initialised to zeros: {np.all(b == 0)}")

# ============ STEP 2: COMPILE =========================================
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=["accuracy"])

# ============ STEP 3: FIT =============================================
history = model.fit(X_train, y_train, epochs=8, batch_size=64,
                    validation_data=(X_valid, y_valid), verbose=0)
print(f"\\nfinished {len(history.history['loss'])} epochs")
print(f"{'epoch':>7}{'loss':>10}{'accuracy':>11}{'val_loss':>11}{'val_acc':>10}")
h = history.history
for e in [0, 2, 4, len(h['loss'])-1]:
    print(f"{e+1:>7}{h['loss'][e]:>10.4f}{h['accuracy'][e]:>11.4f}"
          f"{h['val_loss'][e]:>11.4f}{h['val_accuracy'][e]:>10.4f}")

# ============ STEP 4: EVALUATE ========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\\ntest loss {test_loss:.4f}   test accuracy {test_acc:.4f}")

# ============ STEP 5: PREDICT =========================================
proba = model.predict(X_test[:3], verbose=0)
print(f"\\nfirst 3 predictions:")
for i in range(3):
    top = proba[i].argsort()[::-1][:3]
    print(f"  true = {class_names[y_test[i]]:<12}  " +
          "  ".join(f"{class_names[k]} {proba[i][k]:.3f}" for k in top))

# ============ class_weight for imbalanced data ========================
print("\\n=== handling class imbalance with class_weight ===")
counts = np.bincount(y_train)
weights = {i: len(y_train)/(len(counts)*c) for i, c in enumerate(counts)}
print(f"class counts   : {counts}")
print(f"class weights  : {  {k: round(v,3) for k,v in list(weights.items())[:4]} } ...")
print("pass class_weight=weights to model.fit() when classes are skewed.")

import plotly.graph_objects as go
fig = go.Figure()
for k, c in [("loss", C["danger"]), ("val_loss", C["warning"])]:
    fig.add_scatter(y=h[k], mode="lines", name=k, line=dict(color=c, width=2.5))
for k, c in [("accuracy", C["train"]), ("val_accuracy", C["success"])]:
    fig.add_scatter(y=h[k], mode="lines", name=k, line=dict(color=c, width=2.5),
                    yaxis="y2")
fig.update_layout(height=420, xaxis_title="epoch",
                  yaxis=dict(title="loss"),
                  yaxis2=dict(title="accuracy", overlaying="y", side="right",
                              range=[0, 1]),
                  title="Keras learning curves")
''',
        key="ch10_sequential",
    )

    keypoints([
        "Build → compile → fit → evaluate → predict. Five calls.",
        "A <code>Dense</code> layer has $n_{\\text{in}}n_{\\text{out}} + "
        "n_{\\text{out}}$ parameters — check this against "
        "<code>model.summary()</code>.",
        "The classic Fashion-MNIST MLP has <b>266 610</b> parameters for 55 000 "
        "training instances.",
        "<code>fit</code> returns a <code>History</code> whose "
        "<code>.history</code> dict is your learning curve.",
        "<code>class_weight</code> handles skewed classes without resampling.",
    ])


# ==========================================================================
def s_10_6():
    section("10.6", "The Functional and Subclassing APIs")

    lead(
        "Sequential handles a stack. The Functional API handles any directed "
        "acyclic graph — multiple inputs, multiple outputs, skip connections, "
        "shared layers. The Subclassing API handles anything at all."
    )

    sub("The Functional API")

    md(
        "You call layers on tensors, building a graph, then declare which tensors "
        "are the inputs and outputs. A **wide & deep** network — where some "
        "features bypass the hidden layers entirely — is the canonical example:"
    )

    math(r"""
    \hat y \;=\; \mathbf{w}_{\text{wide}}^\top \mathbf{x}_{\text{wide}}
      \;+\; \mathbf{w}_{\text{deep}}^\top
        \underbrace{\phi\bigl(\dots\phi(\mathbf{W}^{[1]}\mathbf{x}_{\text{deep}})\dots\bigr)}_{\text{the deep path}}
      \;+\; b
    """)

    idea(
        "Why wide & deep",
        "The <b>deep path</b> learns complex non-linear interactions but can "
        "distort simple relationships by passing them through many layers. The "
        "<b>wide path</b> is a direct linear connection from input to output, so "
        "any relationship that <i>is</i> simple stays simple. It is the same "
        "motivation as a residual connection (Chapter 14): give the signal a short "
        "route as well as a long one.",
    )

    sub("The Subclassing API")

    md(
        "Subclass `keras.Model`, create the layers in `__init__`, and write the "
        "forward pass in `call()`. Now you have arbitrary Python — loops, "
        "conditionals, recursion — inside the forward pass."
    )

    table(
        ["", "Sequential", "Functional", "Subclassing"],
        [["Topology", "A single stack",
          "Any DAG: multi-input/output, skips, shared layers",
          "Anything, including dynamic control flow"],
         ["Declarative?", "✅", "✅ — Keras knows the whole graph",
          "❌ — the graph exists only when <code>call()</code> runs"],
         ["<code>model.summary()</code> before fitting", "✅", "✅",
          "❌ (needs a build/call first)"],
         ["<code>plot_model</code>", "✅", "✅", "❌"],
         ["Saving with <code>.keras</code>", "✅", "✅",
          "⚠️ needs <code>get_config</code> / a registered class"],
         ["Shape errors caught early", "✅", "✅", "❌ at runtime"],
         ["Use when", "It is a stack", "<b>The default for anything else</b>",
          "You need loops or conditionals in the forward pass"]],
    )

    tip(
        "Use the Functional API by default",
        "It handles almost everything the Subclassing API can, while keeping the "
        "model <b>inspectable</b>, <b>saveable</b> and <b>plottable</b>. Reach for "
        "subclassing only when the forward pass genuinely needs Python control "
        "flow — which is rarer than beginners expect.",
    )

    anim_header("Four architectures, drawn as graphs")

    def draw(nodes, edges, title, highlight=()):
        ex, ey = [], []
        for a, b in edges:
            ex += [nodes[a][0], nodes[b][0], None]
            ey += [nodes[a][1], nodes[b][1], None]
        cols = [C["danger"] if k in highlight else
                (C["accent"] if "input" in k or "output" in k else C["primary"])
                for k in nodes]
        return [
            go.Scatter(x=ex, y=ey, mode="lines",
                       line=dict(color=C["muted"], width=2.4),
                       hoverinfo="skip", showlegend=False),
            go.Scatter(x=[nodes[k][0] for k in nodes],
                       y=[nodes[k][1] for k in nodes],
                       mode="markers+text",
                       text=[nodes[k][2] for k in nodes],
                       textposition="middle center",
                       textfont=dict(size=9, color="#fff"),
                       marker=dict(size=52, color=cols, symbol="square",
                                   line=dict(color="#fff", width=2)),
                       showlegend=False),
        ]

    archs = [
        ("Sequential — a single stack",
         {"i": (0, 0, "input"), "h1": (1, 0, "Dense<br>300"),
          "h2": (2, 0, "Dense<br>100"), "o": (3, 0, "output")},
         [("i", "h1"), ("h1", "h2"), ("h2", "o")], ()),
        ("Functional — wide & deep",
         {"i": (0, 0, "input"), "h1": (1, -.8, "Dense<br>30"),
          "h2": (2, -.8, "Dense<br>30"), "cc": (3, 0, "concat"),
          "o": (4, 0, "output")},
         [("i", "h1"), ("h1", "h2"), ("h2", "cc"), ("i", "cc"), ("cc", "o")],
         ("cc",)),
        ("Functional — two inputs",
         {"iA": (0, .9, "input A<br>wide"), "iB": (0, -.9, "input B<br>deep"),
          "h1": (1, -.9, "Dense<br>30"), "h2": (2, -.9, "Dense<br>30"),
          "cc": (3, 0, "concat"), "o": (4, 0, "output")},
         [("iB", "h1"), ("h1", "h2"), ("h2", "cc"), ("iA", "cc"), ("cc", "o")],
         ("iA", "iB")),
        ("Functional — auxiliary output (deep supervision)",
         {"i": (0, 0, "input"), "h1": (1, 0, "Dense<br>30"),
          "h2": (2, 0, "Dense<br>30"), "cc": (3, .5, "concat"),
          "o": (4, .5, "main<br>output"), "aux": (3, -1.0, "aux<br>output")},
         [("i", "h1"), ("h1", "h2"), ("h2", "cc"), ("i", "cc"), ("cc", "o"),
          ("h2", "aux")], ("aux",)),
    ]
    frames = [go.Frame(name=str(i + 1), data=draw(n, e, t, h),
                       layout=go.Layout(title=t))
              for i, (t, n, e, h) in enumerate(archs)]

    f = go.Figure(data=draw(archs[0][1], archs[0][2], archs[0][0], archs[0][3]))
    f.update_layout(height=430, plot_bgcolor="#FFFFFF", title=archs[0][0],
                    xaxis=dict(visible=False, range=[-.5, 4.5]),
                    yaxis=dict(visible=False, range=[-1.8, 1.6]))
    anim.animate(f, frames, duration=nav.anim_ms(1900), slider_prefix="architecture ")
    figure(f)

    code_lab(
        "Wide & deep, multi-input, multi-output, and a subclassed model",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)

try:
    from sklearn.datasets import fetch_california_housing
    h = fetch_california_housing(); X, y = h.data, h.target
except Exception:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8000, 8))
    y = X[:, 0]*2 + np.sin(X[:, 1]*3) + X[:, 2]*X[:, 3] + rng.normal(0, .3, 8000)

X_tr_full, X_test, y_tr_full, y_test = train_test_split(X, y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_tr_full, y_tr_full,
                                                      random_state=42)

# ================= 1. FUNCTIONAL: wide & deep =========================
print("=== wide & deep ===")
norm = keras.layers.Normalization(); norm.adapt(X_train)

input_ = keras.layers.Input(shape=X_train.shape[1:], name="all_features")
normed = norm(input_)
hidden1 = keras.layers.Dense(30, activation="relu")(normed)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat  = keras.layers.Concatenate()([normed, hidden2])   # <- the WIDE path
output  = keras.layers.Dense(1)(concat)
wd = keras.Model(inputs=[input_], outputs=[output])
wd.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3),
           metrics=["RootMeanSquaredError"])
wd.fit(X_train, y_train, epochs=8, batch_size=128,
       validation_data=(X_valid, y_valid), verbose=0)
print(f"wide & deep  RMSE = {wd.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# a plain deep model for comparison
n2 = keras.layers.Normalization(); n2.adapt(X_train)
deep = keras.Sequential([n2, keras.layers.Dense(30, activation="relu"),
                         keras.layers.Dense(30, activation="relu"),
                         keras.layers.Dense(1)])
deep.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3),
             metrics=["RootMeanSquaredError"])
deep.fit(X_train, y_train, epochs=8, batch_size=128,
         validation_data=(X_valid, y_valid), verbose=0)
print(f"deep only    RMSE = {deep.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# ================= 2. FUNCTIONAL: two inputs ==========================
print("\\n=== two inputs: features 0-4 wide, 2-7 deep ===")
nw = keras.layers.Normalization(); nw.adapt(X_train[:, :5])
nd = keras.layers.Normalization(); nd.adapt(X_train[:, 2:])

in_wide = keras.layers.Input(shape=[5], name="wide_input")
in_deep = keras.layers.Input(shape=[6], name="deep_input")
nw_out = nw(in_wide); nd_out = nd(in_deep)
hd = keras.layers.Dense(30, activation="relu")(nd_out)
hd = keras.layers.Dense(30, activation="relu")(hd)
cc = keras.layers.Concatenate()([nw_out, hd])
out = keras.layers.Dense(1)(cc)
mi = keras.Model(inputs=[in_wide, in_deep], outputs=[out])
mi.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3),
           metrics=["RootMeanSquaredError"])
mi.fit((X_train[:, :5], X_train[:, 2:]), y_train, epochs=8, batch_size=128,
       validation_data=((X_valid[:, :5], X_valid[:, 2:]), y_valid), verbose=0)
print(f"two inputs   RMSE = "
      f"{mi.evaluate((X_test[:, :5], X_test[:, 2:]), y_test, verbose=0)[1]:.4f}")

# ================= 3. FUNCTIONAL: auxiliary output ====================
print("\\n=== auxiliary output (deep supervision) ===")
in_ = keras.layers.Input(shape=X_train.shape[1:])
n3 = keras.layers.Normalization(); n3.adapt(X_train)
nn = n3(in_)
h1 = keras.layers.Dense(30, activation="relu")(nn)
h2 = keras.layers.Dense(30, activation="relu")(h1)
cc = keras.layers.Concatenate()([nn, h2])
main = keras.layers.Dense(1, name="main")(cc)
aux  = keras.layers.Dense(1, name="aux")(h2)          # regularises the deep path
mo = keras.Model(inputs=[in_], outputs=[main, aux])
mo.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1],
           optimizer=keras.optimizers.Adam(1e-3),
           metrics=[["RootMeanSquaredError"], ["RootMeanSquaredError"]])
mo.fit(X_train, (y_train, y_train), epochs=8, batch_size=128,
       validation_data=(X_valid, (y_valid, y_valid)), verbose=0)
res = mo.evaluate(X_test, (y_test, y_test), verbose=0)
print(f"main output RMSE = {res[3]:.4f}   aux output RMSE = {res[4]:.4f}")
print("The auxiliary head forces the deep path to be useful on its own,")
print("which regularises it. Same idea as GoogLeNet's aux classifiers (Ch. 14).")

# ================= 4. SUBCLASSING =====================================
print("\\n=== subclassing: arbitrary Python in call() ===")
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", n_blocks=2, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.Normalization()
        self.blocks = [keras.layers.Dense(units, activation=activation)
                       for _ in range(n_blocks)]
        self.main_output = keras.layers.Dense(1)
        self.aux_output  = keras.layers.Dense(1)

    def call(self, inputs, training=False):
        z = self.norm(inputs)
        h = z
        for i, layer in enumerate(self.blocks):     # a real Python loop
            h = layer(h)
            if training and i == 0:                 # a real Python conditional
                h = h * 1.0
        concat = keras.layers.concatenate([z, h])
        return self.main_output(concat), self.aux_output(h)

sub = WideAndDeepModel(30, n_blocks=3)
sub.norm.adapt(X_train)
sub.compile(loss=["mse", "mse"], loss_weights=[.9, .1],
            optimizer=keras.optimizers.Adam(1e-3),
            metrics=[["RootMeanSquaredError"], ["RootMeanSquaredError"]])
sub.fit(X_train, (y_train, y_train), epochs=8, batch_size=128, verbose=0)
r = sub.evaluate(X_test, (y_test, y_test), verbose=0)
print(f"subclassed model main RMSE = {r[3]:.4f}")
print(f"\\nnote: summary() works only AFTER the model has been called")
sub.summary()
''',
        key="ch10_apis",
    )

    keypoints([
        "<b>Sequential</b> for a stack; <b>Functional</b> for any DAG; "
        "<b>Subclassing</b> for dynamic control flow.",
        "Functional: call layers on tensors, then "
        "<code>keras.Model(inputs=…, outputs=…)</code>.",
        "<b>Wide &amp; deep</b> gives simple relationships a short path and complex "
        "ones a long one.",
        "Multiple outputs need a loss per output and "
        "<code>loss_weights</code>; auxiliary heads regularise.",
        "Prefer Functional — it stays inspectable, saveable and plottable.",
    ])


# ==========================================================================
def s_10_7():
    section("10.7", "Saving, Callbacks, and TensorBoard")

    lead(
        "The engineering that turns a script into a reproducible experiment: "
        "checkpointing, early stopping, learning-rate scheduling, and a way to see "
        "what happened."
    )

    sub("Saving and restoring")

    md(
        """
```python
model.save("my_model.keras")               # architecture + weights + optimiser
model = keras.models.load_model("my_model.keras")

model.save_weights("weights.weights.h5")   # weights only — you rebuild the model
model.load_weights("weights.weights.h5")
```

The `.keras` format is a zip archive containing the config JSON, the weights, and
the optimiser state — so training resumes exactly where it stopped. Save
**weights only** when you have the architecture in code and want a smaller file,
or when loading into a differently-shaped model (transfer learning, Chapter 11).
        """
    )

    sub("Callbacks")

    table(
        ["Callback", "What it does", "Key arguments"],
        [["<code>ModelCheckpoint</code>",
          "Saves the model during training",
          "<code>save_best_only=True</code> — keeps only the best validation "
          "score, which is early stopping's safety net"],
         ["<code>EarlyStopping</code>",
          "Stops when validation stops improving",
          "<code>patience</code>, "
          "<b><code>restore_best_weights=True</code></b>"],
         ["<code>ReduceLROnPlateau</code>",
          "Cuts the learning rate when progress stalls",
          "<code>factor</code>, <code>patience</code>"],
         ["<code>LearningRateScheduler</code>",
          "Sets the learning rate from a function of the epoch",
          "Chapter 11 covers the schedules"],
         ["<code>TensorBoard</code>", "Writes logs for the dashboard",
          "<code>log_dir</code>"],
         ["<code>CSVLogger</code>", "Appends every epoch's metrics to a CSV",
          "<code>filename</code>"],
         ["<code>Callback</code> (custom)",
          "Any hook: <code>on_epoch_end</code>, <code>on_batch_begin</code>, …",
          "Subclass it"]],
    )

    idea(
        "restore_best_weights=True is the argument people forget",
        "Without it, <code>EarlyStopping</code> halts training but leaves the "
        "model at its <i>last</i> (worse) weights — which are, by construction, "
        "<code>patience</code> epochs past the optimum. With it, the model rolls "
        "back to the best epoch. Combined with a generous "
        "<code>epochs</code> value, this makes the number of epochs a "
        "<b>non-issue</b>: set it high and let the callback decide, exactly as in "
        "§4.5.",
    )

    anim_header("Early stopping with patience, step by step")
    md(
        "A realistic noisy validation curve. The green marker is the best epoch so "
        "far; the red band is the patience window. Training stops when the band "
        "fills, and `restore_best_weights` rolls back to the green marker."
    )

    rng = np.random.default_rng(3)
    n_ep = 90
    train_c = 1.6 * np.exp(-np.arange(n_ep) / 22) + .05
    val_c = (1.55 * np.exp(-np.arange(n_ep) / 18) + .28
             + .0035 * np.maximum(0, np.arange(n_ep) - 35)
             + rng.normal(0, .022, n_ep))
    PATIENCE = 12

    best, best_ep, wait, stop_ep = np.inf, 0, 0, None
    marks = []
    for e in range(n_ep):
        if val_c[e] < best - 1e-4:
            best, best_ep, wait = val_c[e], e, 0
        else:
            wait += 1
        marks.append((best_ep, best, wait))
        if wait >= PATIENCE and stop_ep is None:
            stop_ep = e
    stop_ep = stop_ep or n_ep - 1

    frames = []
    for e in range(1, stop_ep + 2):
        be, bv, w = marks[e - 1]
        frames.append(go.Frame(name=str(e), data=[
            go.Scatter(x=np.arange(e), y=train_c[:e], mode="lines",
                       line=dict(color=C["train"], width=2.6)),
            go.Scatter(x=np.arange(e), y=val_c[:e], mode="lines",
                       line=dict(color=C["test"], width=2.6)),
            go.Scatter(x=[be], y=[bv], mode="markers",
                       marker=dict(color=C["success"], size=16, symbol="star",
                                   line=dict(color="#fff", width=2))),
            go.Scatter(x=[be, be + PATIENCE, be + PATIENCE, be, be],
                       y=[0, 0, 2, 2, 0], fill="toself",
                       fillcolor=alpha(C["danger"], .10), line=dict(width=0),
                       hoverinfo="skip"),
            go.Scatter(x=[e - 1, e - 1], y=[0, 2], mode="lines",
                       line=dict(color=C["ink"], width=1.6, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"epoch {e}   ·   best = epoch {be+1} ({bv:.4f})   ·   "
            f"waited {w}/{PATIENCE}"
            + ("   ← STOP, restoring epoch %d" % (be + 1)
               if w >= PATIENCE else ""),
            color=C["danger"] if w >= PATIENCE else C["ink"])])))

    f = go.Figure(data=[
        go.Scatter(x=[0], y=train_c[:1], mode="lines", name="training loss",
                   line=dict(color=C["train"], width=2.6)),
        go.Scatter(x=[0], y=val_c[:1], mode="lines", name="validation loss",
                   line=dict(color=C["test"], width=2.6)),
        go.Scatter(x=[0], y=val_c[:1], mode="markers", name="best so far",
                   marker=dict(color=C["success"], size=16, symbol="star",
                               line=dict(color="#fff", width=2))),
        go.Scatter(x=[0, PATIENCE, PATIENCE, 0, 0], y=[0, 0, 2, 2, 0],
                   fill="toself", fillcolor=alpha(C["danger"], .10),
                   line=dict(width=0), name=f"patience = {PATIENCE}",
                   hoverinfo="skip"),
        go.Scatter(x=[0, 0], y=[0, 2], mode="lines", showlegend=False,
                   line=dict(color=C["ink"], width=1.6, dash="dot")),
    ])
    f.update_layout(height=450, xaxis_title="epoch", yaxis_title="loss",
                    xaxis=dict(range=[0, n_ep]), yaxis=dict(range=[0, 1.9]),
                    title="EarlyStopping(patience=12, restore_best_weights=True)",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(80), slider_prefix="epoch ")
    figure(f)

    sub("TensorBoard")

    md(
        """
```bash
tensorboard --logdir=./my_logs
```

then open `http://localhost:6006`. It shows learning curves, the computation
graph, weight and activation **histograms over time**, embedding projections, and
profiling. Give every run a timestamped subdirectory so runs are comparable:

```python
from pathlib import Path
from time import strftime
run_dir = Path("my_logs") / strftime("run_%Y_%m_%d_%H_%M_%S")
tb_cb = keras.callbacks.TensorBoard(run_dir, histogram_freq=1, profile_batch=(100, 200))
```
        """
    )

    code_lab(
        "Every callback, plus a custom one",
        '''import numpy as np, os, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
X, y = d.data / 16.0, d.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y,
                                          random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=.2, random_state=42)

tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch10_"))
print(f"working directory: {tmp}")

def build():
    m = keras.Sequential([keras.layers.Input(shape=(64,)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return m

# ============ a CUSTOM callback ========================================
class RatioTracker(keras.callbacks.Callback):
    """Prints val_loss / train_loss -- a live overfitting monitor."""
    def __init__(self): self.ratios = []
    def on_epoch_end(self, epoch, logs=None):
        r = logs["val_loss"] / max(logs["loss"], 1e-9)
        self.ratios.append(r)
        if epoch % 15 == 0:
            print(f"    epoch {epoch:>3}: val/train loss ratio = {r:.3f}")

class LRLogger(keras.callbacks.Callback):
    def __init__(self): self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))

# ============ the full callback set ====================================
ckpt_path = tmp / "best_model.keras"
tracker, lrlog = RatioTracker(), LRLogger()
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                    monitor="val_loss", verbose=0),
    keras.callbacks.EarlyStopping(patience=15, monitor="val_loss",
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=.5, patience=6, min_lr=1e-6,
                                      verbose=0),
    keras.callbacks.CSVLogger(tmp / "training_log.csv"),

    tracker, lrlog,
]

# TensorBoard is a separate package -- add the callback only if it is there
try:
    import tensorboard                                    # noqa: F401
    callbacks.insert(4, keras.callbacks.TensorBoard(tmp / "tb_logs",
                                                    histogram_freq=1))
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("(tensorboard is not installed -- skipping that callback.")
    print("     pip install tensorboard    to enable the dashboard)")

print("\\ntraining with 7 callbacks (max 200 epochs)...")
model = build()
hist = model.fit(X_tr, y_tr, epochs=200, batch_size=32,
                 validation_data=(X_va, y_va), callbacks=callbacks, verbose=0)

n_ran = len(hist.history["loss"])
best_ep = int(np.argmin(hist.history["val_loss"]))
print(f"\\nran {n_ran} of 200 epochs -- EarlyStopping fired")
print(f"best epoch was {best_ep+1} (val_loss {hist.history['val_loss'][best_ep]:.4f})")
print(f"last epoch     {n_ran}  (val_loss {hist.history['val_loss'][-1]:.4f})")
print(f"restore_best_weights recovered {hist.history['val_loss'][-1] - hist.history['val_loss'][best_ep]:+.4f} of loss")
print(f"\\nlearning rate: started {lrlog.lrs[0]:.2e}, ended {lrlog.lrs[-1]:.2e} "
      f"(ReduceLROnPlateau cut it {int(np.sum(np.diff(lrlog.lrs) < 0))} times)")
print(f"test accuracy = {model.evaluate(X_te, y_te, verbose=0)[1]:.4f}")

# ============ saving and loading =======================================
print("\\n=== saving ===")
full_path = tmp / "full_model.keras"
model.save(full_path)
print(f"full model : {full_path.stat().st_size/1024:.1f} KB "
      f"(architecture + weights + optimiser state)")

w_path = tmp / "weights.weights.h5"
model.save_weights(w_path)
print(f"weights    : {w_path.stat().st_size/1024:.1f} KB")

reloaded = keras.models.load_model(full_path)
print(f"\\nreloaded model accuracy = {reloaded.evaluate(X_te, y_te, verbose=0)[1]:.4f}")
print(f"predictions identical: "
      f"{np.allclose(model.predict(X_te[:20], verbose=0), reloaded.predict(X_te[:20], verbose=0))}")

# weights-only into a freshly built model
fresh = build()
print(f"fresh model (untrained) accuracy = "
      f"{fresh.evaluate(X_te, y_te, verbose=0)[1]:.4f}")
fresh.load_weights(w_path)
print(f"after load_weights               = "
      f"{fresh.evaluate(X_te, y_te, verbose=0)[1]:.4f}")

# ============ the CSV log ==============================================
import pandas as pd
log = pd.read_csv(tmp / "training_log.csv")
print(f"\\nCSVLogger wrote {len(log)} rows with columns {list(log.columns)}")
print(log.tail(3).round(4).to_string(index=False))

print(f"\\nTensorBoard logs are in {tmp/'tb_logs'}")
print(f"  run:  tensorboard --logdir={tmp/'tb_logs'}")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hist.history["loss"], mode="lines", name="train loss",
                line=dict(color=C["train"], width=2.4))
fig.add_scatter(y=hist.history["val_loss"], mode="lines", name="val loss",
                line=dict(color=C["test"], width=2.4))
fig.add_vline(x=best_ep, line_dash="dash", line_color=C["success"],
              annotation_text="restored epoch")
fig.update_layout(height=400, xaxis_title="epoch", yaxis_title="loss",
                  yaxis_type="log", title="Early stopping in action")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\\n(cleaned up {tmp})")
''',
        key="ch10_callbacks",
    )

    keypoints([
        "<code>.keras</code> saves architecture + weights + optimiser state; "
        "<code>save_weights</code> saves only the arrays.",
        "<b><code>EarlyStopping(restore_best_weights=True)</code></b> makes the "
        "epoch count a non-issue.",
        "<code>ModelCheckpoint(save_best_only=True)</code> is the safety net if "
        "training crashes.",
        "<code>ReduceLROnPlateau</code> is the simplest useful LR schedule; "
        "Chapter 11 has the rest.",
        "Write TensorBoard logs to a <b>timestamped</b> subdirectory so runs stay "
        "comparable.",
    ])


# ==========================================================================
def s_10_8():
    section("10.8", "Fine-Tuning Neural Network Hyperparameters")

    lead(
        "Neural networks have too many hyperparameters to grid-search. Here is "
        "what actually matters, in order, and how to search it efficiently."
    )

    sub("The search strategies")

    table(
        ["Method", "Idea", "When"],
        [["<b>Random search</b>", "Sample configurations at random (§2.6)",
          "Always a reasonable baseline"],
         ["<b>Keras Tuner — Hyperband</b>",
          "Train many configurations for a few epochs, kill the losers, promote "
          "the survivors (successive halving)",
          "<b>The default choice.</b> Enormously more efficient than random"],
         ["<b>Bayesian optimisation</b>",
          "Fit a surrogate model of score-vs-hyperparameters and sample where the "
          "expected improvement is highest",
          "When each trial is very expensive"],
         ["<b>Population-based training</b>",
          "Train a population in parallel; periodically copy the winners' weights "
          "<i>and</i> perturb their hyperparameters",
          "Large-scale, when you can afford many parallel workers"]],
    )

    sub("What matters, in order")

    table(
        ["Rank", "Hyperparameter", "Guidance"],
        [["1", "<b>Learning rate</b>",
          "By far the most important. Search on a <b>log</b> scale, "
          "$10^{-5}$ to $10^{-1}$. Use the LR-range test below."],
         ["2", "<b>Optimiser</b>",
          "Adam / AdamW is a strong default; SGD+momentum with a good schedule can "
          "generalise better (Ch. 11)"],
         ["3", "<b>Batch size</b>",
          "Largest that fits in memory <i>and</i> trains stably. Large batches "
          "need warm-up."],
         ["4", "<b>Number of neurons per layer</b>",
          "Usually the same in every layer, or a funnel. Often less important than "
          "depth."],
         ["5", "<b>Number of hidden layers</b>",
          "Start at 1–2, increase until you overfit, then regularise."],
         ["6", "Activation function",
          "ReLU is a fine default; Swish/GELU for deep nets"],
         ["7", "Number of epochs", "<b>Do not tune it</b> — use early stopping"]],
    )

    idea(
        "Why deeper beats wider — the parameter-efficiency argument",
        "Deep networks have exponentially higher <b>parameter efficiency</b> than "
        "shallow ones: they model complex functions with far fewer neurons, "
        "because each layer reuses the features the previous layer built. Lower "
        "layers learn low-level structure (edges), middle layers combine them "
        "(shapes), upper layers combine <i>those</i> (objects). This hierarchy "
        "also makes <b>transfer learning</b> possible — the lower layers are "
        "reusable across tasks, which is Chapter 11's topic.",
    )

    sub("The learning-rate range test")

    md(
        "The single most useful 60 seconds you can spend before a long training "
        "run. Train for a few hundred iterations while **exponentially increasing** "
        "the learning rate, and plot the loss against it. The curve drops, reaches "
        "a minimum, then explodes. Pick roughly **one order of magnitude below "
        "the point of minimum loss** — that is the largest rate that is still "
        "stable."
    )

    math(r"""
    \eta_t \;=\; \eta_{\min}
      \left(\frac{\eta_{\max}}{\eta_{\min}}\right)^{t/T}
    """)

    anim_header("The LR range test")

    def lr_curve(lr):
        # a realistic U shape: too small = no progress, too large = divergence
        opt = -2.2
        x = np.log10(lr)
        return .35 + 1.7 / (1 + np.exp(3.2 * (x - opt - 1.6))) \
            + np.exp(3.1 * (x - opt - .55)) - 1.6 / (1 + np.exp(-3.4 * (x - opt + 2.6)))

    lrs = np.logspace(-6, -.5, 90)
    losses = lr_curve(lrs)
    losses = losses + np.random.default_rng(0).normal(0, .012, len(lrs))
    best_i = int(np.argmin(losses))
    pick = lrs[best_i] / 10

    frames = []
    for k in range(3, len(lrs) + 1):
        cur_best = int(np.argmin(losses[:k]))
        frames.append(go.Frame(name=f"{lrs[k-1]:.2e}", data=[
            go.Scatter(x=lrs[:k], y=losses[:k], mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=[lrs[cur_best]], y=[losses[cur_best]], mode="markers",
                       marker=dict(color=C["danger"], size=14,
                                   line=dict(color="#fff", width=2))),
            go.Scatter(x=[lrs[cur_best] / 10, lrs[cur_best] / 10], y=[0, 3],
                       mode="lines",
                       line=dict(color=C["success"], width=2.5, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"lr = {lrs[k-1]:.2e}   ·   loss = {losses[k-1]:.4f}   ·   "
            f"minimum so far at {lrs[cur_best]:.2e}   ·   "
            f"pick ≈ {lrs[cur_best]/10:.2e}")])))

    f = go.Figure(data=[
        go.Scatter(x=lrs[:3], y=losses[:3], mode="lines", name="loss",
                   line=dict(color=C["primary"], width=3)),
        go.Scatter(x=[lrs[0]], y=[losses[0]], mode="markers",
                   name="minimum so far",
                   marker=dict(color=C["danger"], size=14,
                               line=dict(color="#fff", width=2))),
        go.Scatter(x=[lrs[0], lrs[0]], y=[0, 3], mode="lines",
                   name="recommended lr (one decade below)",
                   line=dict(color=C["success"], width=2.5, dash="dash")),
    ])
    f.update_layout(height=440, xaxis_type="log", xaxis_title="learning rate",
                    yaxis_title="loss", yaxis=dict(range=[0, 3]),
                    title="Learning-rate range test",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="lr = ")
    figure(f)

    code_lab(
        "The LR range test, and hyperparameter search with Keras Tuner",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(d.data/16., d.target, test_size=.25,
                                          stratify=d.target, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=.2, random_state=42)

# ============ 1. THE LEARNING-RATE RANGE TEST ==========================
print("=== learning-rate range test ===")
class ExponentialLR(keras.callbacks.Callback):
    def __init__(self, lr0=1e-5, lr1=1.0, n_steps=400):
        self.factor = (lr1/lr0) ** (1/n_steps)
        self.lr0, self.rates, self.losses = lr0, [], []
    def on_train_begin(self, logs=None):
        self.model.optimizer.learning_rate.assign(self.lr0)
    def on_batch_end(self, batch, logs=None):
        self.rates.append(float(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        self.model.optimizer.learning_rate.assign(
            float(self.model.optimizer.learning_rate) * self.factor)

def build(lr=1e-3, n_hidden=2, n_neurons=64, activation="relu",
          optimizer="adam"):
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(n_hidden):
        layers.append(keras.layers.Dense(n_neurons, activation=activation))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers)
    opt = (keras.optimizers.Adam(lr) if optimizer == "adam"
           else keras.optimizers.SGD(lr, momentum=.9))
    m.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
    return m

lr_cb = ExponentialLR(1e-5, 1.0, n_steps=400)
probe = build()
probe.fit(X_tr, y_tr, epochs=8, batch_size=32, callbacks=[lr_cb], verbose=0)

rates = np.array(lr_cb.rates); losses = np.array(lr_cb.losses)
sm = np.convolve(losses, np.ones(12)/12, mode="valid")
r_sm = rates[:len(sm)]
best_i = int(np.argmin(sm))
print(f"minimum smoothed loss at lr = {r_sm[best_i]:.2e}")
print(f"recommended lr (one decade below) = {r_sm[best_i]/10:.2e}")

print(f"\\n{'learning rate':>15}{'val accuracy after 20 epochs':>32}")
for lr in [r_sm[best_i]/100, r_sm[best_i]/10, r_sm[best_i], r_sm[best_i]*3]:
    m = build(lr=float(lr))
    m.fit(X_tr, y_tr, epochs=15, batch_size=32, verbose=0)
    print(f"{lr:>15.2e}{m.evaluate(X_va, y_va, verbose=0)[1]:>32.4f}")

# ============ 2. RANDOM SEARCH via scikit-learn ========================
print("\\n=== random search over the architecture ===")
from scipy.stats import loguniform, randint
rng = np.random.default_rng(0)
results = []
for trial in range(6):
    cfg = dict(lr=float(loguniform(1e-4, 3e-2).rvs(random_state=trial)),
               n_hidden=int(randint(1, 4).rvs(random_state=trial)),
               n_neurons=int(2 ** randint(4, 8).rvs(random_state=trial)),
               activation=["relu", "tanh", "selu"][trial % 3],
               optimizer=["adam", "sgd"][trial % 2])
    m = build(**cfg)
    m.fit(X_tr, y_tr, epochs=18, batch_size=32,
          validation_data=(X_va, y_va), verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                   restore_best_weights=True)])
    acc = m.evaluate(X_va, y_va, verbose=0)[1]
    results.append((acc, cfg))
    print(f"  trial {trial:>2}: val acc {acc:.4f}   lr={cfg['lr']:.2e} "
          f"layers={cfg['n_hidden']} units={cfg['n_neurons']} "
          f"{cfg['activation']}/{cfg['optimizer']}")

results.sort(reverse=True, key=lambda t: t[0])
print(f"\\nbest configuration: {results[0][1]}")
print(f"best validation accuracy: {results[0][0]:.4f}")
best = build(**results[0][1])
best.fit(np.r_[X_tr, X_va], np.r_[y_tr, y_va], epochs=30, batch_size=32, verbose=0)
print(f"refit on train+valid, TEST accuracy = "
      f"{best.evaluate(X_te, y_te, verbose=0)[1]:.4f}")

# ============ 3. KERAS TUNER (if installed) ============================
print("\\n=== Keras Tuner ===")
try:
    import keras_tuner as kt
    def model_builder(hp):
        n_hidden  = hp.Int("n_hidden", 1, 4)
        n_neurons = hp.Int("n_neurons", 16, 256, sampling="log")
        lr        = hp.Float("lr", 1e-4, 1e-2, sampling="log")
        opt       = hp.Choice("optimizer", ["sgd", "adam"])
        return build(lr=lr, n_hidden=n_hidden, n_neurons=n_neurons,
                     optimizer=opt)

    tuner = kt.Hyperband(model_builder, objective="val_accuracy",
                         max_epochs=20, factor=3, overwrite=True,
                         directory="kt_tmp", project_name="ch10")
    tuner.search(X_tr, y_tr, epochs=20, validation_data=(X_va, y_va), verbose=0)
    hp = tuner.get_best_hyperparameters(1)[0]
    print(f"Hyperband best: {hp.values}")
    print(f"val accuracy  : "
          f"{tuner.get_best_models(1)[0].evaluate(X_va, y_va, verbose=0)[1]:.4f}")
    import shutil; shutil.rmtree("kt_tmp", ignore_errors=True)
except ImportError:
    print("keras-tuner is not installed.  pip install keras-tuner")
    print("Hyperband trains many configurations briefly, kills the losers,")
    print("and promotes the survivors -- far more efficient than random search.")

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=r_sm, y=sm, mode="lines",
                           line=dict(color=C["primary"], width=3)))
fig.add_vline(x=r_sm[best_i], line_dash="dot", line_color=C["danger"],
              annotation_text="minimum")
fig.add_vline(x=r_sm[best_i]/10, line_dash="dash", line_color=C["success"],
              annotation_text="pick this")
fig.update_layout(height=400, xaxis_type="log", xaxis_title="learning rate",
                  yaxis_title="smoothed loss", title="LR range test (measured)")
''',
        key="ch10_tuning",
    )

    keypoints([
        "The <b>learning rate</b> dominates everything else — find it with an LR "
        "range test before any long run.",
        "Search on a <b>log</b> scale for the learning rate and layer sizes.",
        "<b>Never tune the number of epochs</b>; use early stopping.",
        "Deeper beats wider: parameter efficiency and reusable low-level features.",
        "<b>Hyperband</b> (Keras Tuner) is far more sample-efficient than random "
        "search.",
    ])


# ==========================================================================
def s_10_9():
    section("10.9", "Exercises & Chapter Review")

    lead("Ten exercises. Number 10 is a full Keras project.")

    exercise(
        1, "The TensorFlow Playground is a handy neural network simulator built by "
        "the TensorFlow team. Explore it and note what you observe.",
        "Things worth confirming for yourself at "
        "[playground.tensorflow.org](https://playground.tensorflow.org):\n\n"
        "* **Patterns in the hidden layers.** Train a deep net on the spiral: the "
        "first hidden layer learns simple half-plane splits, the second combines "
        "them into corners and stripes, the third into curved regions. That "
        "hierarchy is exactly the parameter-efficiency argument of §10.8.\n"
        "* **Activation functions matter.** Switch the spiral dataset from `tanh` "
        "to `ReLU` and watch the boundary become piecewise linear — because a ReLU "
        "network *is* a piecewise-linear function.\n"
        "* **Local minima are real.** Run the same spiral configuration several "
        "times: some runs solve it, some get stuck. The cost surface of a neural "
        "network is not convex.\n"
        "* **Depth vs width.** One layer of 8 neurons cannot solve the spiral; "
        "two layers of 4 often can, with half the parameters.\n"
        "* **Too many neurons overfit.** With noise turned up and few training "
        "points, a large network carves out islands around individual points — "
        "the same picture as §6.5's unregularised tree.")

    exercise(
        2, "Draw an ANN using the original artificial neurons that computes "
        "$A \\oplus B$ (XOR). Hint: $A \\oplus B = (A \\wedge \\neg B) \\vee "
        "(\\neg A \\wedge B)$.",
        "Two hidden neurons and one output neuron:\n\n"
        "* **Hidden neuron 1** computes $A \\wedge \\neg B$: an excitatory "
        "connection from $A$ and an **inhibitory** connection from $B$, with "
        "threshold 1.\n"
        "* **Hidden neuron 2** computes $\\neg A \\wedge B$: inhibitory from $A$, "
        "excitatory from $B$, threshold 1.\n"
        "* **Output neuron** computes $h_1 \\vee h_2$: excitatory from both, "
        "threshold 1.\n\n"
        "In modern weight notation, the same network is\n\n"
        "$\\mathbf{W}^{[1]} = \\begin{bmatrix} 1 & -1 \\\\ -1 & 1\\end{bmatrix}$, "
        "$\\mathbf{b}^{[1]} = \\begin{bmatrix}-0.5 \\\\ -0.5\\end{bmatrix}$, "
        "$\\mathbf{W}^{[2]} = \\begin{bmatrix}1 \\\\ 1\\end{bmatrix}$, "
        "$b^{[2]} = -0.5$\n\n"
        "with a step activation. §10.1's lab implements an equivalent "
        "OR/AND version and verifies all four rows.")

    exercise(
        3, "Why is it generally preferable to use a logistic regression classifier "
        "rather than a classical perceptron? How can you tweak a perceptron to "
        "make it equivalent to a logistic regression classifier?",
        "The classical perceptron uses a **step** activation, which has three "
        "consequences: it converges only if the classes are linearly separable "
        "(§10.1), it outputs a hard 0/1 with **no class probabilities**, and its "
        "decision boundary depends on the order of presentation.\n\n"
        "Logistic regression outputs a calibrated probability and its convex log "
        "loss converges regardless of separability.\n\n"
        "**The tweak:** replace the step activation with the **sigmoid** (or "
        "softmax for multiclass) and train with **cross-entropy** by gradient "
        "descent instead of the perceptron rule. That *is* logistic regression — "
        "concretely, `SGDClassifier(loss='log_loss')` rather than "
        "`loss='perceptron'`.")

    exercise(
        4, "Why was the logistic activation function a key ingredient in training "
        "the first MLPs?",
        "Because **its derivative is non-zero everywhere**. The step function's "
        "derivative is zero almost everywhere and undefined at 0, so gradient "
        "descent has nothing to descend — there is no gradient to propagate. The "
        "sigmoid, being smooth with $\\sigma' = \\sigma(1-\\sigma) > 0$, gives "
        "backpropagation a non-zero signal at every layer.\n\n"
        "(In fact backpropagation works with any differentiable activation, and "
        "we now know the sigmoid is a *poor* choice for deep networks precisely "
        "because $\\sigma' \\le 0.25$ shrinks the gradient at every layer — "
        "Chapter 11's vanishing-gradient problem. But without *some* "
        "differentiable activation there is no backpropagation at all.)")

    exercise(
        5, "Name three popular activation functions. Can you draw them?",
        "**Sigmoid** $\\sigma(z) = 1/(1+e^{-z})$ — an S-curve from 0 to 1, "
        "flat at both ends.\n\n"
        "**Tanh** $\\tanh(z)$ — the same S shape but from −1 to 1, and "
        "**zero-centred**, which makes it strictly better than the sigmoid for "
        "hidden layers.\n\n"
        "**ReLU** $\\max(0, z)$ — flat zero for negative input, the identity for "
        "positive. Not differentiable at 0 (in practice the subgradient 0 is "
        "used), but very fast and it does not saturate for positive inputs.\n\n"
        "Others worth knowing: **step** (historical), **leaky ReLU**, **ELU**, "
        "**GELU**, **Swish**, and **softplus** $\\log(1+e^z)$. All eight are "
        "plotted with their derivatives in §10.2's animation.")

    exercise(
        6, "Suppose you have an MLP composed of one input layer with 10 "
        "passthrough neurons, followed by one hidden layer with 50 artificial "
        "neurons, and finally one output layer with 3 artificial neurons. All "
        "artificial neurons use the ReLU activation function. (a) What is the "
        "shape of the input matrix $\\mathbf{X}$? (b) What are the shapes of the "
        "hidden layer's weight matrix $\\mathbf{W}_h$ and bias vector "
        "$\\mathbf{b}_h$? (c) What are the shapes of the output layer's "
        "$\\mathbf{W}_o$ and $\\mathbf{b}_o$? (d) What is the shape of the "
        "network's output matrix $\\mathbf{Y}$? (e) Write the equation that "
        "computes $\\mathbf{Y}$ as a function of $\\mathbf{X}$, $\\mathbf{W}_h$, "
        "$\\mathbf{b}_h$, $\\mathbf{W}_o$ and $\\mathbf{b}_o$.",
        "**(a)** $\\mathbf{X}$ has shape $(m, 10)$, where $m$ is the batch size.\n\n"
        "**(b)** $\\mathbf{W}_h$ has shape $(10, 50)$ and $\\mathbf{b}_h$ has "
        "length 50.\n\n"
        "**(c)** $\\mathbf{W}_o$ has shape $(50, 3)$ and $\\mathbf{b}_o$ has "
        "length 3.\n\n"
        "**(d)** $\\mathbf{Y}$ has shape $(m, 3)$.\n\n"
        "**(e)** $\\mathbf{Y} = \\mathrm{ReLU}\\bigl(\\mathrm{ReLU}("
        "\\mathbf{X}\\mathbf{W}_h + \\mathbf{b}_h)\\,\\mathbf{W}_o + "
        "\\mathbf{b}_o\\bigr)$\n\n"
        "where ReLU is applied element-wise and the bias vectors are broadcast "
        "across the $m$ rows. Total parameters: $10\\cdot50 + 50 + 50\\cdot3 + 3 = "
        "703$.")

    exercise(
        7, "How many neurons do you need in the output layer if you want to "
        "classify email into spam or ham? What activation function should you use "
        "in the output layer? If instead you want to tackle MNIST, how many "
        "neurons do you need in the output layer, and which activation function "
        "should you use? What about for getting your network to predict housing "
        "prices?",
        "**Spam/ham:** **one** output neuron with a **sigmoid** activation, giving "
        "the estimated probability of spam. Loss: binary cross-entropy. (You could "
        "use two neurons with softmax, but that is wasteful for a binary "
        "problem.)\n\n"
        "**MNIST:** **ten** output neurons with a **softmax** activation, since "
        "the classes are mutually exclusive and probabilities should sum to 1. "
        "Loss: sparse categorical cross-entropy.\n\n"
        "**Housing prices:** **one** output neuron with **no activation "
        "function** at all — the output must be free to take any value. (ReLU or "
        "softplus if you want to guarantee non-negativity, but see §10.3's "
        "warning.) Loss: MSE, or Huber if there are outliers.")

    exercise(
        8, "What is backpropagation and how does it work? What is the difference "
        "between backpropagation and reverse-mode autodiff?",
        "**Backpropagation** is the training algorithm for neural networks. Each "
        "iteration: (1) a **forward pass** computes every layer's output for a "
        "mini-batch, keeping the intermediates; (2) the loss is measured; (3) a "
        "**backward pass** applies the chain rule from the output back to the "
        "input, computing each parameter's contribution to the error; (4) a "
        "gradient descent step updates every parameter.\n\n"
        "**The difference:** *reverse-mode autodiff* is the general technique for "
        "computing the gradient of a scalar output with respect to many inputs "
        "efficiently — it applies to any computation graph, not just neural "
        "networks. **Backpropagation is reverse-mode autodiff plus the gradient "
        "descent update step**, applied specifically to a neural network. The "
        "terms are often used interchangeably, but strictly, backprop = autodiff "
        "+ the optimisation step. Appendix B covers autodiff in full.")

    exercise(
        9, "Can you list all the hyperparameters you can tweak in a basic MLP? If "
        "the MLP overfits the training data, how could you tweak these "
        "hyperparameters to try to solve the problem?",
        "**The hyperparameters:** number of hidden layers; number of neurons per "
        "layer; the activation function of each hidden layer; the weight "
        "initialisation logic; the optimiser and its own hyperparameters "
        "(learning rate, momentum, $\\beta_1$/$\\beta_2$); the batch size; the "
        "loss function; the number of epochs; and any regularisation "
        "(weight decay, dropout rate, max-norm constraint).\n\n"
        "**To fix overfitting:**\n"
        "* **Reduce the number of hidden layers**, and/or the number of neurons "
        "per layer.\n"
        "* **Add regularisation** — $\\ell_2$ weight decay, or a **dropout** "
        "layer, or a max-norm constraint (all in Chapter 11).\n"
        "* **Reduce the number of epochs**, ideally by using **early stopping** "
        "rather than tuning the number directly.\n"
        "* **Reduce the batch size**, which adds gradient noise and acts as a mild "
        "regulariser.\n"
        "* Get more data, or augment what you have.")

    exercise(
        10, "Train a deep MLP on the MNIST dataset (you can load it using "
        "`tf.keras.datasets.mnist.load_data()`). See if you can get over 98 % "
        "accuracy by manually tuning the hyperparameters. Try searching for the "
        "optimal learning rate by using the approach presented in this chapter "
        "(i.e., by growing the learning rate exponentially, plotting the loss, and "
        "finding the point where the loss shoots up). Next, try tuning the "
        "hyperparameters using Keras Tuner with all the bells and whistles — save "
        "checkpoints, use early stopping, and plot learning curves using "
        "TensorBoard.",
        "**Getting above 98 % with an MLP** (no convolutions) is achievable but "
        "requires care. A working recipe:\n\n"
        "* Normalise the pixels to $[0,1]$ (divide by 255).\n"
        "* Three hidden layers of 300 / 200 / 100 neurons with ReLU.\n"
        "* Adam at the learning rate found by the LR range test — typically "
        "around $3\\times10^{-4}$ to $10^{-3}$.\n"
        "* Batch size 32–128.\n"
        "* `EarlyStopping(patience=20, restore_best_weights=True)` with "
        "`epochs=200`.\n\n"
        "That reaches roughly **98.2 %**. Adding dropout (0.2) and batch "
        "normalisation (Chapter 11) pushes it to about 98.5 %.\n\n"
        "For context: a small CNN (Chapter 14) reaches **99.2 %+** with fewer "
        "parameters, because convolutions encode translation invariance "
        "architecturally instead of learning it. The MLP has to learn from data "
        "that a 7 shifted two pixels right is still a 7.\n\n"
        "**On the LR range test:** run it on a *fresh* model each time, since it "
        "leaves the weights in a bad state. Smooth the loss curve (a moving "
        "average over ~10 batches) before reading off the minimum, or the noise "
        "will mislead you.",
        code='''import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from time import strftime

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full / 255.
X_test = X_test / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              metrics=["accuracy"])

run_dir = Path("my_mnist_logs") / strftime("run_%Y_%m_%d_%H_%M_%S")
model.fit(X_train, y_train, epochs=200,
          validation_data=(X_valid, y_valid),
          callbacks=[
              keras.callbacks.EarlyStopping(patience=20,
                                            restore_best_weights=True),
              keras.callbacks.ModelCheckpoint("my_mnist_model.keras",
                                              save_best_only=True),
              keras.callbacks.TensorBoard(run_dir),
          ])
print(model.evaluate(X_test, y_test))     # aim for > 0.98''')

    rule()

    keypoints([
        "A perceptron is a linear classifier with a step activation; it cannot "
        "learn XOR.",
        "Backprop = one forward pass + one backward pass, giving every gradient "
        "for the cost of ~2 forward passes.",
        "Output layer: <b>none</b> for regression, <b>sigmoid</b> for binary and "
        "multilabel, <b>softmax</b> for multiclass.",
        "Sequential for stacks, Functional for graphs, Subclassing for control "
        "flow.",
        "Learning rate first, early stopping always, Hyperband for the rest.",
    ], title="Chapter 10 in five lines")

    refs([
        ("McCulloch & Pitts — *A Logical Calculus of the Ideas Immanent in "
         "Nervous Activity*", "https://doi.org/10.1007/BF02478259"),
        ("Rosenblatt, F. — *The Perceptron: A Probabilistic Model*",
         "https://doi.org/10.1037/h0042519"),
        ("Rumelhart, Hinton & Williams — *Learning Representations by "
         "Back-Propagating Errors*", "https://doi.org/10.1038/323533a0"),
        ("Minsky & Papert — *Perceptrons*", "MIT Press, 1969"),
        ("Cheng et al. — *Wide & Deep Learning for Recommender Systems*",
         "https://doi.org/10.1145/2988450.2988454"),
        ("Li et al. — *Hyperband: A Novel Bandit-Based Approach to "
         "Hyperparameter Optimization*",
         "https://www.jmlr.org/papers/v18/16-558.html"),
        ("TensorFlow Playground", "https://playground.tensorflow.org"),
    ])


# ==========================================================================
SECTIONS = [
    ("10.1", "From Biological to Artificial Neurons", s_10_1),
    ("10.2", "MLPs and Backpropagation", s_10_2),
    ("10.3", "Regression MLPs", s_10_3),
    ("10.4", "Classification MLPs", s_10_4),
    ("10.5", "Keras — Sequential API", s_10_5),
    ("10.6", "Functional & Subclassing APIs", s_10_6),
    ("10.7", "Saving, Callbacks, TensorBoard", s_10_7),
    ("10.8", "Fine-Tuning Hyperparameters", s_10_8),
    ("10.9", "Exercises & Review", s_10_9),
]

nav.render_chapter(CH, SECTIONS)
