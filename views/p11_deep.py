"""Chapter 11 — Training Deep Neural Networks."""

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
CH = "ch11"

hero(
    kicker="Part II · Chapter 11",
    title="Training Deep Neural Networks",
    blurb=(
        "Depth breaks naive training, and this chapter is the accumulated set of "
        "fixes. The vanishing-gradient problem quantified and solved by Glorot/He "
        "initialisation; the activation zoo; batch normalisation and why it works; "
        "transfer learning; every optimiser from momentum to AdamW derived; "
        "learning-rate schedules; and dropout with its Bayesian interpretation."
    ),
    chips=["All optimisers derived", "9 sub-sections", "10 animations",
           "9 code labs", "The practical playbook"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_11_1():
    section("11.1", "The Vanishing and Exploding Gradients Problem")

    lead(
        "Backpropagation multiplies a factor at every layer. Multiply 50 numbers "
        "that are each a bit less than 1 and you get zero; a bit more than 1 and "
        "you get infinity. That is the entire problem, and its solution follows "
        "from writing the product down."
    )

    sub("The mechanism")

    derive(
        [("From §10.2, the error signal propagates by",
          r"\boldsymbol\delta^{[l]} = \Bigl(\mathbf{W}^{[l+1]\top}"
          r"\boldsymbol\delta^{[l+1]}\Bigr) \odot \phi'\bigl(\mathbf{z}^{[l]}\bigr)"),
         ("Unrolling from the output layer $L$ down to layer $l$ gives a "
          "<b>product</b> of $L - l$ terms:",
          r"\boldsymbol\delta^{[l]} = \left(\prod_{k=l+1}^{L} "
          r"\mathbf{W}^{[k]\top}\mathrm{diag}\bigl(\phi'(\mathbf{z}^{[k-1]})\bigr)\right)"
          r"\boldsymbol\delta^{[L]}"),
         ("Take norms. If each factor has typical magnitude $\\gamma$, then",
          r"\bigl\lVert \boldsymbol\delta^{[l]} \bigr\rVert \;\sim\; "
          r"\gamma^{\,L-l}\,\bigl\lVert \boldsymbol\delta^{[L]} \bigr\rVert"),
         ("This is <b>exponential</b> in depth. Three regimes:", None),
         ("• $\\gamma < 1$ ⇒ <b>vanishing gradients</b>. Lower layers receive "
          "essentially nothing and never train. With $\\gamma = 0.5$ and 30 "
          "layers, the factor is $10^{-9}$.<br>"
          "• $\\gamma > 1$ ⇒ <b>exploding gradients</b>. Updates are enormous, the "
          "loss becomes <code>NaN</code>. With $\\gamma = 1.5$ and 30 layers, the "
          "factor is $10^{5}$.<br>"
          "• $\\gamma \\approx 1$ ⇒ trainable. This is the <b>only</b> workable "
          "regime, and it is what initialisation schemes engineer.", None),
         ("<b>Why the sigmoid is disastrous.</b> $\\sigma'(z) = "
          "\\sigma(z)(1-\\sigma(z)) \\le 1/4$, with equality only at $z = 0$. So "
          "the activation alone contributes $\\gamma \\le 1/4$ per layer, "
          "<i>before</i> the weights are even considered:",
          r"\bigl\lVert \boldsymbol\delta^{[l]}\bigr\rVert \;\le\; "
          r"\left(\tfrac14\right)^{L-l}\prod_k \bigl\lVert\mathbf{W}^{[k]}\bigr\rVert"),
         ("Ten sigmoid layers multiply the gradient by at most $4^{-10} \\approx "
          "10^{-6}$. That is why deep sigmoid networks were untrainable for two "
          "decades, and it is why ReLU — whose derivative is exactly 1 on the "
          "positive side — was such a decisive change.", None)],
        title="Why gradients vanish or explode exponentially in depth",
    )

    sub("Glorot and He initialisation")

    md(
        "Glorot & Bengio (2010) asked: what weight variance keeps the **signal "
        "variance constant** through the layers, both forward and backward?"
    )

    derive(
        [("<b>Forward pass.</b> For $z_j = \\sum_{i=1}^{n_{\\text{in}}} w_{ij}x_i$ "
          "with independent zero-mean $w$ and $x$:",
          r"\mathrm{Var}(z) = n_{\text{in}}\,\mathrm{Var}(w)\,\mathrm{Var}(x)"),
         ("To keep $\\mathrm{Var}(z) = \\mathrm{Var}(x)$ we need",
          r"\mathrm{Var}(w) = \frac{1}{n_{\text{in}}}"),
         ("<b>Backward pass.</b> The same argument applied to "
          "$\\boldsymbol\\delta$ flowing backwards through $\\mathbf{W}^\\top$ "
          "gives",
          r"\mathrm{Var}(w) = \frac{1}{n_{\text{out}}}"),
         ("These conflict unless $n_{\\text{in}} = n_{\\text{out}}$. Glorot's "
          "compromise is the <b>harmonic-style average</b> — hence "
          "$n_{\\text{avg}} = (n_{\\text{in}} + n_{\\text{out}})/2$:",
          r"\boxed{\;\mathrm{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}\;}"),
         ("<b>He initialisation</b> corrects for ReLU. ReLU zeroes half the "
          "inputs, so it halves the variance of what passes through — compensate "
          "by doubling the weight variance:",
          r"\boxed{\;\mathrm{Var}(w) = \frac{2}{n_{\text{in}}}\;}"),
         ("<b>LeCun initialisation</b> ($1/n_{\\text{in}}$) is the right choice for "
          "SELU, which is designed around it.", None)],
        title="Deriving Glorot, He and LeCun initialisation",
    )

    table(
        ["Initialisation", "$\\mathrm{Var}(w)$", "Normal", "Uniform limit $r$",
         "Use with"],
        [["<b>Glorot</b> (Xavier)", "$2/(n_{\\text{in}} + n_{\\text{out}})$",
          "$\\sigma = \\sqrt{2/(n_{\\text{in}}+n_{\\text{out}})}$",
          "$\\sqrt{6/(n_{\\text{in}}+n_{\\text{out}})}$",
          "None, tanh, sigmoid, softmax"],
         ["<b>He</b> (Kaiming)", "$2/n_{\\text{in}}$",
          "$\\sigma = \\sqrt{2/n_{\\text{in}}}$", "$\\sqrt{6/n_{\\text{in}}}$",
          "<b>ReLU</b> and variants, Swish"],
         ["<b>LeCun</b>", "$1/n_{\\text{in}}$", "$\\sigma = \\sqrt{1/n_{\\text{in}}}$",
          "$\\sqrt{3/n_{\\text{in}}}$", "SELU"]],
        "For the uniform variants, $w \\sim \\mathcal{U}(-r, r)$ so that "
        "$\\mathrm{Var}(w) = r^2/3$ matches the target.",
    )

    codenote(
        "Keras defaults",
        "<code>Dense</code> defaults to <code>kernel_initializer='glorot_uniform'</code>. "
        "For a ReLU network you should say "
        "<code>kernel_initializer='he_normal'</code> explicitly. It is a one-word "
        "change and on a 20-layer network it is the difference between training "
        "and not training.",
    )

    anim_header("Signal and gradient variance through 40 layers")
    md(
        "Forward activation variance (top) and backward gradient variance (bottom) "
        "measured layer by layer, for four initialisation schemes. The y-axes are "
        "logarithmic: watch three of the four curves fall off a cliff."
    )

    rng = np.random.default_rng(0)
    n_layers, width, batch = 40, 128, 512
    schemes = {
        "too small: σ = 0.01": ("fixed", .01, "relu"),
        "too large: σ = 0.5": ("fixed", .5, "relu"),
        "Glorot + sigmoid": ("glorot", None, "sigmoid"),
        "He + ReLU": ("he", None, "relu"),
    }

    @st.cache_data(show_spinner=False)
    def propagate(seed=0):
        out = {}
        r = np.random.default_rng(seed)
        for nm, (kind, sval, act) in schemes.items():
            x = r.normal(0, 1, (batch, width))
            fvar, acts, Ws = [], [x], []
            for l in range(n_layers):
                if kind == "fixed":
                    W = r.normal(0, sval, (width, width))
                elif kind == "glorot":
                    W = r.normal(0, np.sqrt(2 / (width + width)), (width, width))
                else:
                    W = r.normal(0, np.sqrt(2 / width), (width, width))
                Ws.append(W)
                z = acts[-1] @ W
                a = (np.maximum(0, z) if act == "relu"
                     else 1 / (1 + np.exp(-np.clip(z, -60, 60))))
                acts.append(a)
                fvar.append(float(a.var()))
            # backward
            d = r.normal(0, 1, (batch, width))
            bvar = []
            for l in reversed(range(n_layers)):
                z_pre = acts[l] @ Ws[l]
                dphi = ((z_pre > 0).astype(float) if act == "relu"
                        else (lambda s: s * (1 - s))(1 / (1 + np.exp(-np.clip(z_pre, -60, 60)))))
                d = (d * dphi) @ Ws[l].T
                d = np.clip(d, -1e30, 1e30)
                bvar.append(float(d.var()))
            out[nm] = (fvar, bvar[::-1])
        return out

    prop = propagate()
    xs = np.arange(1, n_layers + 1)

    frames = []
    for k in range(2, n_layers + 1):
        data = []
        info = []
        for i, nm in enumerate(schemes):
            fv, bv = prop[nm]
            data.append(go.Scatter(x=xs[:k], y=np.maximum(fv[:k], 1e-40),
                                   mode="lines", line=dict(color=SEQ[i], width=3)))
        for i, nm in enumerate(schemes):
            fv, bv = prop[nm]
            data.append(go.Scatter(x=xs[:k], y=np.maximum(bv[-k:], 1e-40),
                                   mode="lines", line=dict(color=SEQ[i], width=3)))
            info.append(f"{nm.split(':')[0]}: {bv[-k]:.1e}")
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"depth {k}   ·   gradient variance —   "
                                   + "   ".join(info))])))

    f = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      subplot_titles=("forward: activation variance per layer",
                                      "backward: gradient variance per layer"))
    for i, nm in enumerate(schemes):
        fv, bv = prop[nm]
        f.add_trace(go.Scatter(x=xs[:2], y=fv[:2], mode="lines", name=nm,
                               line=dict(color=SEQ[i], width=3)), 1, 1)
    for i, nm in enumerate(schemes):
        fv, bv = prop[nm]
        f.add_trace(go.Scatter(x=xs[:2], y=bv[-2:], mode="lines", showlegend=False,
                               line=dict(color=SEQ[i], width=3)), 2, 1)
    f.update_yaxes(type="log", title_text="Var(activation)", row=1, col=1)
    f.update_yaxes(type="log", title_text="Var(gradient)", row=2, col=1)
    f.update_xaxes(title_text="layer", row=2, col=1)
    f.update_layout(height=560, title="Initialisation decides whether depth works",
                    legend=dict(orientation="h", y=1.06, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(80), slider_prefix="depth ")
    figure(f, "Only He + ReLU keeps both variances near constant across 40 "
              "layers. σ = 0.01 vanishes, σ = 0.5 explodes, and Glorot + sigmoid "
              "vanishes because of the ≤¼ derivative.")

    code_lab(
        "Measure the vanishing gradient, then fix it",
        '''import numpy as np

rng = np.random.default_rng(0)
DEPTH, WIDTH, BATCH = 50, 100, 256

def forward_backward(init, activation, depth=DEPTH, width=WIDTH, seed=0):
    """Propagate signal forward and gradients backward; report the variances."""
    r = np.random.default_rng(seed)
    def make_W():
        if init == "he":      s = np.sqrt(2/width)
        elif init == "glorot": s = np.sqrt(2/(width+width))
        elif init == "lecun":  s = np.sqrt(1/width)
        else:                  s = float(init)          # a raw sigma
        return r.normal(0, s, (width, width))

    def act(z):
        if activation == "relu":    return np.maximum(0, z)
        if activation == "tanh":    return np.tanh(z)
        if activation == "sigmoid": return 1/(1+np.exp(-np.clip(z, -60, 60)))
        return z
    def dact(z):
        if activation == "relu":    return (z > 0).astype(float)
        if activation == "tanh":    return 1 - np.tanh(z)**2
        if activation == "sigmoid":
            s = 1/(1+np.exp(-np.clip(z, -60, 60))); return s*(1-s)
        return np.ones_like(z)

    Ws = [make_W() for _ in range(depth)]
    a = r.normal(0, 1, (BATCH, width))
    acts, zs = [a], []
    for W in Ws:
        z = acts[-1] @ W; zs.append(z); acts.append(act(z))
    fwd = [float(x.var()) for x in acts]

    d = r.normal(0, 1, (BATCH, width))
    bwd = []
    for l in reversed(range(depth)):
        d = (d * dact(zs[l])) @ Ws[l].T
        d = np.clip(d, -1e35, 1e35)
        bwd.append(float(d.var()))
    return fwd, bwd[::-1]

print("=== gradient variance at the FIRST layer, after 50 layers ===")
print(f"{'init':<14}{'activation':<12}{'fwd var (last)':>17}{'bwd var (first)':>18}"
      f"   verdict")
for init, actv in [(0.01, "relu"), (0.5, "relu"),
                   ("glorot", "sigmoid"), ("glorot", "tanh"),
                   ("he", "relu"), ("lecun", "tanh")]:
    fwd, bwd = forward_backward(init, actv)
    v = bwd[0]
    verdict = ("VANISHED" if v < 1e-8 else
               "EXPLODED" if v > 1e8 or not np.isfinite(v) else "healthy")
    print(f"{str(init):<14}{actv:<12}{fwd[-1]:>17.3e}{v:>18.3e}   {verdict}")

# ============ the exponential law, verified =============================
print("\\n=== gradient magnitude ~ gamma^depth ===")
print(f"{'depth':>7}{'He+ReLU':>14}{'Glorot+sigmoid':>18}{'ratio':>14}")
for depth in [5, 10, 20, 40, 80]:
    _, b_he = forward_backward("he", "relu", depth=depth)
    _, b_sg = forward_backward("glorot", "sigmoid", depth=depth)
    print(f"{depth:>7}{b_he[0]:>14.3e}{b_sg[0]:>18.3e}"
          f"{b_he[0]/max(b_sg[0],1e-300):>14.2e}")
print("\\nThe sigmoid column falls off a cliff; sigma'(z) <= 1/4 at every layer.")
print(f"theory: 50 sigmoid layers multiply the gradient by <= "
      f"{0.25**50:.2e} from the activation ALONE")

# ============ the initialisation formulas ===============================
print("\\n=== initialisation scales ===")
print(f"{'n_in':>7}{'n_out':>7}{'Glorot sigma':>15}{'He sigma':>12}"
      f"{'LeCun sigma':>14}{'Glorot limit r':>17}")
for n_in, n_out in [(784, 300), (300, 100), (100, 10), (512, 512)]:
    print(f"{n_in:>7}{n_out:>7}{np.sqrt(2/(n_in+n_out)):>15.5f}"
          f"{np.sqrt(2/n_in):>12.5f}{np.sqrt(1/n_in):>14.5f}"
          f"{np.sqrt(6/(n_in+n_out)):>17.5f}")

# ============ and in Keras ==============================================
import tensorflow as tf
from tensorflow import keras
print("\\n=== what Keras actually initialises ===")
m = keras.Sequential([keras.layers.Input(shape=(784,)),
                      keras.layers.Dense(300, activation="relu"),
                      keras.layers.Dense(300, activation="relu",
                                         kernel_initializer="he_normal"),
                      keras.layers.Dense(300, activation="selu",
                                         kernel_initializer="lecun_normal")])
for i, layer in enumerate(m.layers):
    W = layer.get_weights()[0]
    n_in, n_out = W.shape
    print(f"  layer {i}: {layer.kernel_initializer.__class__.__name__:<22}"
          f"measured std {W.std():.5f}   "
          f"Glorot {np.sqrt(2/(n_in+n_out)):.5f}   He {np.sqrt(2/n_in):.5f}   "
          f"LeCun {np.sqrt(1/n_in):.5f}")
print("\\nKeras defaults to glorot_uniform. For ReLU, say he_normal explicitly.")

# ============ does it matter in practice? ===============================
print("\\n=== training a 20-layer network, three initialisations ===")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
tf.random.set_seed(42)
for nm, init, actv in [("sigmoid + glorot", "glorot_uniform", "sigmoid"),
                       ("relu + glorot",    "glorot_uniform", "relu"),
                       ("relu + he_normal", "he_normal",      "relu")]:
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(20):
        layers.append(keras.layers.Dense(50, activation=actv,
                                         kernel_initializer=init))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    mm = keras.Sequential(layers)
    mm.compile(loss="sparse_categorical_crossentropy",
               optimizer=keras.optimizers.SGD(0.05), metrics=["accuracy"])
    h = mm.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
    print(f"  {nm:<20} final train acc {h.history['accuracy'][-1]:.4f}   "
          f"test acc {mm.evaluate(Xte, yte, verbose=0)[1]:.4f}")
''',
        key="ch11_vanishing",
    )

    quiz(
        "You build a 30-layer network with sigmoid activations and Glorot "
        "initialisation. Training loss barely moves. What is the most likely "
        "cause?",
        ["The learning rate is too large",
         "Vanishing gradients — $\\sigma' \\le 1/4$ compounds over 30 layers",
         "Not enough training data",
         "The batch size is too small"],
        1,
        "$(1/4)^{30} \\approx 10^{-18}$. The lower layers receive essentially no "
        "gradient signal at all. Swap to ReLU + He initialisation, and add batch "
        "normalisation (§11.3).",
        key="ch11q1",
    )

    keypoints([
        "Backprop multiplies a factor per layer, so gradients scale like "
        "$\\gamma^{L}$ — exponentially in depth.",
        "$\\sigma'(z) \\le 1/4$ makes deep sigmoid networks untrainable.",
        "<b>Glorot</b>: $\\mathrm{Var}(w) = 2/(n_{\\text{in}}+n_{\\text{out}})$ for "
        "tanh/sigmoid/linear.",
        "<b>He</b>: $\\mathrm{Var}(w) = 2/n_{\\text{in}}$ for ReLU — the factor 2 "
        "compensates for ReLU zeroing half the units.",
        "Keras defaults to Glorot; say <code>he_normal</code> explicitly for ReLU "
        "networks.",
    ])


# ==========================================================================
def s_11_2():
    section("11.2", "Better Activation Functions")

    lead(
        "ReLU made deep learning work. Then it turned out to have its own failure "
        "mode, and a decade of variants followed. Here is what each one fixes and "
        "when it is worth the cost."
    )

    sub("The dying ReLU problem")

    pitfall(
        "A ReLU unit can die permanently",
        "If a large gradient update drives a unit's weights so that "
        "$\\mathbf{w}^\\top\\mathbf{x} + b < 0$ for <b>every</b> training instance, "
        "the unit outputs 0 always. Its gradient is then $\\phi'(z) = 0$ always, "
        "so it receives no update, so it can never recover. It is dead. During "
        "training, up to <b>40 %</b> of a network's ReLU units can die, "
        "particularly with a large learning rate.",
    )

    sub("The variants")

    table(
        ["Function", "Definition", "Fixes", "Cost"],
        [["<b>Leaky ReLU</b>",
          "$\\max(\\alpha z,\\, z)$, $\\alpha \\approx 0.01$–$0.2$",
          "Dying ReLU — the small slope keeps a gradient alive", "Negligible"],
         ["<b>RReLU</b> (randomised)",
          "$\\alpha$ sampled randomly during training, averaged at test",
          "Dying ReLU + acts as a regulariser", "Negligible"],
         ["<b>PReLU</b> (parametric)", "$\\alpha$ is <b>learned</b> per neuron",
          "Dying ReLU; better on large datasets, overfits small ones",
          "One parameter per neuron"],
         ["<b>ELU</b>",
          "$z$ if $z>0$, else $\\alpha(e^{z}-1)$",
          "Dying ReLU + <b>negative mean</b> pushes activations toward zero mean",
          "Slower — an exponential"],
         ["<b>SELU</b>",
          "$\\lambda \\cdot \\mathrm{ELU}_\\alpha(z)$ with fixed $\\lambda, \\alpha$",
          "<b>Self-normalising</b>: preserves mean 0, variance 1 across layers",
          "Slower; strict preconditions"],
         ["<b>GELU</b>", "$z \\cdot \\Phi(z)$",
          "Smooth, non-monotonic; the standard in Transformers", "Slower"],
         ["<b>Swish / SiLU</b>", "$z \\cdot \\sigma(\\beta z)$",
          "Smooth, non-monotonic; found by neural architecture search", "Slower"],
         ["<b>Mish</b>", "$z \\tanh(\\mathrm{softplus}(z))$",
          "Smoother still; marginal gains", "Slowest"]],
    )

    sub("SELU and self-normalisation")

    math(r"""
    \mathrm{SELU}(z) \;=\; \lambda
    \begin{cases}
      z & \text{if } z > 0\\
      \alpha\bigl(e^{z} - 1\bigr) & \text{if } z \le 0
    \end{cases}
    """)
    where({r"\lambda": "$\\approx 1.0507$ — not a hyperparameter, a derived constant",
           r"\alpha": "$\\approx 1.6733$ — likewise derived"})

    proof(
        "Where λ and α come from",
        "Klambauer et al. (2017) solved for the constants that make the map "
        "$(\\mu, \\sigma^2) \\mapsto (\\mu', \\sigma'^2)$ have a <b>fixed point</b> "
        "at $(0, 1)$ — so that if a layer's inputs have mean 0 and variance 1, its "
        "outputs do too, and this is preserved through arbitrary depth. The "
        "network normalises itself, with no batch normalisation needed. The "
        "constants are the solution of a two-equation system involving Gaussian "
        "integrals; they are exact, not tuned.",
    )

    warn(
        "SELU's preconditions are strict, and all of them are required",
        "<b>(1)</b> Inputs must be standardised (mean 0, sd 1). "
        "<b>(2)</b> Every hidden layer's weights must use <b>LeCun normal</b> "
        "initialisation. <b>(3)</b> The architecture must be a plain stack — no "
        "skip connections. <b>(4)</b> Every layer must be <code>Dense</code> "
        "(the guarantee does not hold for convolutional or recurrent layers). "
        "<b>(5)</b> Regular dropout breaks the normalisation — you must use "
        "<code>AlphaDropout</code>. Violate any one and the self-normalising "
        "property is gone.",
    )

    sub("The practical ranking")

    md(
        """
Roughly, in order of preference for a hidden layer today:

**ReLU** → **Leaky ReLU** → **GELU / Swish** → **ELU** → **SELU** → tanh → sigmoid

* **ReLU** remains the right default: it is by far the fastest and it is well
  supported by every accelerator.
* **Leaky ReLU** if you observe dying units, at essentially no cost.
* **GELU / Swish** for large models where you can afford the compute — they
  consistently give a small accuracy gain, and GELU is standard in Transformers
  (Chapter 16).
* **SELU** only when you can satisfy all five preconditions and want a deep
  plain MLP with no batch normalisation.
        """
    )

    anim_header("Every activation and its derivative — the gradient's fate")

    z = np.linspace(-4, 4, 500)
    lam, alp = 1.0507009873554805, 1.6732632423543772
    Phi = .5 * (1 + np.vectorize(lambda t: np.math.erf(t / np.sqrt(2)))(z)) \
        if False else .5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z ** 3)))

    fns = {
        "sigmoid": (1 / (1 + np.exp(-z)),
                    (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
        "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
        "Leaky ReLU (α=0.2)": (np.where(z > 0, z, .2 * z),
                               np.where(z > 0, 1., .2)),
        "ELU (α=1)": (np.where(z > 0, z, np.exp(z) - 1),
                      np.where(z > 0, 1., np.exp(z))),
        "SELU": (lam * np.where(z > 0, z, alp * (np.exp(z) - 1)),
                 lam * np.where(z > 0, 1., alp * np.exp(z))),
        "GELU": (z * Phi, np.gradient(z * Phi, z)),
        "Swish (β=1)": (z / (1 + np.exp(-z)), np.gradient(z / (1 + np.exp(-z)), z)),
        "Mish": (z * np.tanh(np.log1p(np.exp(np.clip(z, -60, 60)))),
                 np.gradient(z * np.tanh(np.log1p(np.exp(np.clip(z, -60, 60)))), z)),
    }

    frames = []
    for nm, (fv, dv) in fns.items():
        neg_mean = float(np.mean(fv[z < 0]))
        frames.append(go.Frame(name=nm.split()[0], data=[
            go.Scatter(x=z, y=fv, mode="lines",
                       line=dict(color=C["primary"], width=3.8)),
            go.Scatter(x=z, y=dv, mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=[-4, 4], y=[1, 1], mode="lines",
                       line=dict(color=C["success"], width=1.5, dash="dot")),
            go.Scatter(x=[-4, 4], y=[0, 0], mode="lines",
                       line=dict(color=C["muted"], width=1, dash="dot")),
        ], layout=go.Layout(title=f"{nm}   ·   max φ' = {dv.max():.3f}   ·   "
                                  f"mean output for z<0 = {neg_mean:+.3f}   ·   "
                                  f"φ'(−2) = {dv[np.argmin(np.abs(z + 2))]:.4f}")))

    nm0 = list(fns)[0]
    f = go.Figure(data=[
        go.Scatter(x=z, y=fns[nm0][0], mode="lines", name="φ(z)",
                   line=dict(color=C["primary"], width=3.8)),
        go.Scatter(x=z, y=fns[nm0][1], mode="lines", name="φ'(z)",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[-4, 4], y=[1, 1], mode="lines", name="φ' = 1 (ideal)",
                   line=dict(color=C["success"], width=1.5, dash="dot")),
        go.Scatter(x=[-4, 4], y=[0, 0], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1, dash="dot")),
    ])
    f.update_layout(height=460, xaxis_title="z", yaxis=dict(range=[-2, 4.2]),
                    title=nm0, legend=dict(orientation="h", y=1.02,
                                           yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="")
    figure(f, "The green dotted line is φ' = 1, the ideal for gradient flow. "
              "ReLU sits exactly on it for z > 0 and at 0 below — that zero is "
              "the dying-ReLU problem, and every variant is a way of lifting it.")

    code_lab(
        "Measure dying ReLUs, and race the activations",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
sc = StandardScaler().fit(Xtr)
Atr, Ate = sc.transform(Xtr), sc.transform(Xte)

# ============ 1. COUNT THE DEAD RELUs ==================================
print("=== how many ReLU units die? ===")
def dead_fraction(lr, activation="relu", epochs=30, depth=6, width=64):
    tf.random.set_seed(0)
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(width, activation=activation,
                                         kernel_initializer="he_normal"))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr), metrics=["accuracy"])
    m.fit(Atr, ytr, epochs=epochs, batch_size=64, verbose=0)
    # a unit is DEAD if it outputs 0 for EVERY training instance
    dead_total = alive_total = 0
    probe = keras.Model(m.inputs, [l.output for l in m.layers[:-1]])
    for out in probe.predict(Atr, verbose=0):
        dead = np.all(out <= 1e-9, axis=0)
        dead_total += int(dead.sum()); alive_total += len(dead)
    return dead_total/alive_total, m.evaluate(Ate, yte, verbose=0)[1]

print(f"{'learning rate':>15}{'dead ReLU units':>18}{'test accuracy':>16}")
for lr in [0.01, 0.1, 0.5, 1.5]:
    frac, acc = dead_fraction(lr)
    print(f"{lr:>15}{frac:>17.1%}{acc:>16.4f}")
print("\\nA large learning rate kills units permanently: once w.x + b < 0 for")
print("every instance, the gradient is 0 forever and the unit never recovers.")

print(f"\\n{'activation':>16}{'dead units at lr=0.5':>23}{'test accuracy':>16}")
for actv in ["relu", "elu", "selu"]:
    frac, acc = dead_fraction(0.5, activation=actv)
    print(f"{actv:>16}{frac:>22.1%}{acc:>16.4f}")
print("ELU and SELU cannot die -- their derivative is never exactly zero.")

# ============ 2. RACE THE ACTIVATIONS ==================================
print("\\n=== the activation zoo, on the same architecture ===")
def build(act_layer, init="he_normal", depth=5, width=100):
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(width, kernel_initializer=init))
        layers.append(act_layer())
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(.05), metrics=["accuracy"])
    return m

zoo = {
    "ReLU":       (lambda: keras.layers.Activation("relu"), "he_normal"),
    "LeakyReLU":  (lambda: keras.layers.LeakyReLU(negative_slope=.2), "he_normal"),
    "PReLU":      (lambda: keras.layers.PReLU(), "he_normal"),
    "ELU":        (lambda: keras.layers.ELU(), "he_normal"),
    "SELU":       (lambda: keras.layers.Activation("selu"), "lecun_normal"),
    "GELU":       (lambda: keras.layers.Activation("gelu"), "he_normal"),
    "Swish":      (lambda: keras.layers.Activation("swish"), "he_normal"),
    "tanh":       (lambda: keras.layers.Activation("tanh"), "glorot_normal"),
    "sigmoid":    (lambda: keras.layers.Activation("sigmoid"), "glorot_normal"),
}
print(f"{'activation':<12}{'params':>9}{'fit time':>11}{'train acc':>12}"
      f"{'test acc':>11}")
for nm, (layer_fn, init) in zoo.items():
    tf.random.set_seed(0)
    m = build(layer_fn, init)
    t0 = time.perf_counter()
    h = m.fit(Atr, ytr, epochs=30, batch_size=64, verbose=0)
    dt = time.perf_counter()-t0
    print(f"{nm:<12}{m.count_params():>9,}{dt:>10.2f}s"
          f"{h.history['accuracy'][-1]:>12.4f}"
          f"{m.evaluate(Ate, yte, verbose=0)[1]:>11.4f}")

# ============ 3. SELU'S SELF-NORMALISATION =============================
print("\\n=== SELU keeps mean 0 and variance 1 through depth ===")
rng = np.random.default_rng(0)
lam, alp = 1.0507009873554805, 1.6732632423543772
def selu(z): return lam * np.where(z > 0, z, alp*(np.exp(np.clip(z,-60,60))-1))

print(f"{'layer':>7}" + "".join(f"{n:>20}" for n in
      ["SELU+LeCun mean/var", "ReLU+He mean/var", "tanh+Glorot mean/var"]))
configs = [(selu, lambda n: np.sqrt(1/n)),
           (lambda z: np.maximum(0, z), lambda n: np.sqrt(2/n)),
           (np.tanh, lambda n: np.sqrt(2/(n+n)))]
W_ = 200
states = [rng.normal(0, 1, (400, W_)) for _ in configs]
for l in range(1, 41):
    row = f"{l:>7}"
    for i, (act, sc_) in enumerate(configs):
        W = rng.normal(0, sc_(W_), (W_, W_))
        states[i] = act(states[i] @ W)
        row += f"{f'{states[i].mean():+.3f} / {states[i].var():.3f}':>20}"
    if l in (1, 5, 10, 20, 40):
        print(row)
print("\\nSELU holds (0, 1) at depth 40. ReLU's variance drifts; tanh's collapses.")
''',
        key="ch11_activations",
    )

    keypoints([
        "ReLU units <b>die</b> when $\\mathbf{w}^\\top\\mathbf{x}+b < 0$ for every "
        "instance — the gradient is then permanently zero.",
        "Leaky ReLU / PReLU / ELU / SELU all keep a non-zero derivative on the "
        "negative side.",
        "<b>SELU</b> is self-normalising, but only under five strict preconditions.",
        "<b>GELU</b> and <b>Swish</b> are the modern choices for large models; "
        "GELU is standard in Transformers.",
        "Default order: ReLU → Leaky ReLU → GELU/Swish → ELU → SELU.",
    ])


# ==========================================================================
def s_11_3():
    section("11.3", "Batch Normalization")

    lead(
        "Normalise each layer's inputs, using statistics computed over the "
        "mini-batch, and let the network learn the scale and offset it actually "
        "wants. It is the single most effective architectural trick in this "
        "chapter."
    )

    sub("The algorithm")

    math(r"""
    \boldsymbol\mu_B = \frac{1}{m_B}\sum_{i=1}^{m_B}\mathbf{x}^{(i)}
    \qquad
    \boldsymbol\sigma_B^{2} = \frac{1}{m_B}\sum_{i=1}^{m_B}
      \bigl(\mathbf{x}^{(i)} - \boldsymbol\mu_B\bigr)^{2}
    """)
    math(r"""
    \hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol\mu_B}
                                  {\sqrt{\boldsymbol\sigma_B^{2} + \varepsilon}}
    \qquad\qquad
    \mathbf{z}^{(i)} = \boldsymbol\gamma \otimes \hat{\mathbf{x}}^{(i)}
      + \boldsymbol\beta
    """)
    where({
        r"m_B": "the mini-batch size",
        r"\varepsilon": "a smoothing term, typically $10^{-5}$, preventing division by zero",
        r"\boldsymbol\gamma": "the <b>learned</b> output scale, one per feature",
        r"\boldsymbol\beta": "the <b>learned</b> output offset, one per feature",
        r"\otimes": "element-wise multiplication",
    })

    idea(
        "γ and β are what make it work, not the normalisation",
        "Normalising alone would <i>constrain</i> the network — it would force "
        "every layer's inputs to mean 0 and variance 1, which is not necessarily "
        "what the layer wants. Adding the learned $\\boldsymbol\\gamma$ and "
        "$\\boldsymbol\\beta$ restores full expressiveness: the layer can undo the "
        "normalisation entirely if that is optimal ($\\gamma = \\sigma_B$, "
        "$\\beta = \\mu_B$). What batch norm actually changes is the "
        "<b>parameterisation</b> — the scale of each layer's output becomes an "
        "explicit, directly-learned parameter instead of an emergent consequence "
        "of all the weights below it.",
    )

    sub("Why it helps — the honest answer")

    md(
        """
The original paper (Ioffe & Szegedy, 2015) attributed the effect to reducing
**internal covariate shift** — the change in each layer's input distribution as
the layers below it update. That explanation is now considered **incorrect**:
Santurkar et al. (2018) showed empirically that batch norm does not reduce
covariate shift, and that you can *inject* covariate shift after a BN layer
without hurting training.

The current understanding is that batch norm **smooths the optimisation
landscape** — it provably reduces the Lipschitz constant of the loss and of its
gradient, which means larger learning rates are stable and the loss surface has
fewer sharp cliffs.
        """
    )

    table(
        ["Benefit", "Mechanism"],
        [["Much larger learning rates work",
          "A smoother loss surface tolerates bigger steps"],
         ["Vanishing/exploding gradients largely disappear",
          "Each layer's input scale is fixed, so the $\\gamma^L$ product is "
          "controlled"],
         ["Acts as a regulariser",
          "The batch statistics are noisy, injecting noise into every activation "
          "— so you often need less dropout"],
         ["Less sensitive to initialisation",
          "The layer normalises whatever comes in"],
         ["Sometimes removes the need for input scaling",
          "A BN layer first in the network normalises the inputs"]],
    )

    sub("Training vs inference")

    md(
        "At training time BN uses the **current mini-batch's** statistics. At "
        "inference there may be no batch — you might be predicting one instance. "
        "So BN keeps an **exponential moving average** of the statistics seen "
        "during training and uses those:"
    )

    math(r"""
    \hat{\boldsymbol\mu} \leftarrow \rho\,\hat{\boldsymbol\mu}
      + (1-\rho)\,\boldsymbol\mu_B,
    \qquad
    \hat{\boldsymbol\sigma}^2 \leftarrow \rho\,\hat{\boldsymbol\sigma}^2
      + (1-\rho)\,\boldsymbol\sigma_B^2
    """)
    where({r"\rho": "the <code>momentum</code> argument, typically 0.99 or 0.999 — "
                    "closer to 1 for larger datasets"})

    md(
        "So a BN layer has **four** parameter vectors per feature: "
        "$\\boldsymbol\\gamma$ and $\\boldsymbol\\beta$ (learned by "
        "backpropagation) and $\\hat{\\boldsymbol\\mu}$, "
        "$\\hat{\\boldsymbol\\sigma}$ (**non-trainable**, estimated by the moving "
        "average)."
    )

    pitfall(
        "Three ways batch norm goes wrong",
        "<b>(1) Small batches.</b> With $m_B < 8$ the batch statistics are so "
        "noisy that BN hurts. Use <code>LayerNormalization</code> (Chapter 16) or "
        "<code>GroupNormalization</code> instead.<br>"
        "<b>(2) Forgetting <code>training=False</code>.</b> If you call the model "
        "manually rather than through <code>predict</code>, BN will use batch "
        "statistics at inference and your predictions will depend on what else is "
        "in the batch.<br>"
        "<b>(3) The bias term becomes redundant.</b> BN subtracts the mean, so any "
        "bias in the preceding layer is cancelled. Set "
        "<code>use_bias=False</code> on the layer before a BN layer — it saves "
        "parameters and changes nothing.",
    )

    sub("Where to put it")

    md(
        "Before or after the activation? The original paper says **before**; "
        "practice is divided and the difference is usually small. Placing it "
        "**before** the activation means the BN layer's $\\boldsymbol\\beta$ "
        "replaces the bias, so use `use_bias=False`:"
    )

    md(
        """
```python
# BN before the activation (the original formulation)
keras.layers.Dense(300, use_bias=False),
keras.layers.BatchNormalization(),
keras.layers.Activation("relu"),

# BN after the activation (also common, sometimes better)
keras.layers.Dense(300, activation="relu"),
keras.layers.BatchNormalization(),
```
        """
    )

    anim_header("Activation distributions with and without batch norm")
    md(
        "A 15-layer network's activation histograms as training proceeds. Without "
        "BN the distributions drift and spread; with BN they stay pinned near "
        "mean 0, variance 1 — which is exactly what makes the deeper layers "
        "trainable."
    )

    rng = np.random.default_rng(1)
    depth, width, batch = 15, 64, 400

    @st.cache_data(show_spinner=False)
    def bn_demo(seed=1):
        r = np.random.default_rng(seed)
        Ws = [r.normal(0, np.sqrt(2 / width), (width, width)) for _ in range(depth)]
        gain = np.linspace(.75, 1.35, depth)          # imperfect scaling
        out = {}
        for use_bn in (False, True):
            x = r.normal(0, 1, (batch, width))
            stats = []
            for l in range(depth):
                z = x @ (Ws[l] * gain[l])
                if use_bn:
                    z = (z - z.mean(0)) / np.sqrt(z.var(0) + 1e-5)
                x = np.maximum(0, z)
                stats.append((float(x.mean()), float(x.std()), x[:, :24].ravel()))
            out[use_bn] = stats
        return out

    dat = bn_demo()
    bins = np.linspace(-1, 6, 50)
    ctr = (bins[:-1] + bins[1:]) / 2

    frames = []
    for l in range(depth):
        m0, s0, v0 = dat[False][l]
        m1, s1, v1 = dat[True][l]
        frames.append(go.Frame(name=str(l + 1), data=[
            go.Bar(x=ctr, y=np.histogram(v0, bins=bins, density=True)[0],
                   marker=dict(color=alpha(C["danger"], .7))),
            go.Bar(x=ctr, y=np.histogram(v1, bins=bins, density=True)[0],
                   marker=dict(color=alpha(C["success"], .7))),
            go.Scatter(x=list(range(1, l + 2)),
                       y=[d[1] for d in dat[False][:l + 1]], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=list(range(1, l + 2)),
                       y=[d[1] for d in dat[True][:l + 1]], mode="lines",
                       line=dict(color=C["success"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"layer {l+1}/{depth}   ·   without BN: mean {m0:+.3f} sd {s0:.3f}"
            f"   ·   with BN: mean {m1:+.3f} sd {s1:.3f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("activation distribution at this layer",
                                      "activation std across layers"))
    m0, s0, v0 = dat[False][0]
    m1, s1, v1 = dat[True][0]
    f.add_trace(go.Bar(x=ctr, y=np.histogram(v0, bins=bins, density=True)[0],
                       name="without BN",
                       marker=dict(color=alpha(C["danger"], .7))), 1, 1)
    f.add_trace(go.Bar(x=ctr, y=np.histogram(v1, bins=bins, density=True)[0],
                       name="with BN",
                       marker=dict(color=alpha(C["success"], .7))), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[s0], mode="lines", showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=[1], y=[s1], mode="lines", showlegend=False,
                           line=dict(color=C["success"], width=3)), 1, 2)
    f.update_layout(height=460, barmode="overlay", bargap=.03,
                    title="Batch normalisation pins the activation scale")
    f.update_xaxes(title_text="activation value", row=1, col=1)
    f.update_xaxes(title_text="layer", row=1, col=2)
    f.update_yaxes(title_text="std", type="log", row=1, col=2)
    anim.animate(f, frames, duration=nav.anim_ms(500), slider_prefix="layer ")
    figure(f)

    code_lab(
        "Batch norm from scratch, and what it buys you",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.25,
                                      stratify=d.target, random_state=42)

# ============ 1. BATCH NORM FROM SCRATCH ===============================
def batch_norm_forward(X, gamma, beta, eps=1e-5):
    mu  = X.mean(0)
    var = X.var(0)
    Xhat = (X - mu) / np.sqrt(var + eps)
    return gamma * Xhat + beta, mu, var

rng = np.random.default_rng(0)
X = rng.normal(5, 3, (64, 8))                   # badly scaled input
gamma, beta = np.ones(8), np.zeros(8)
Z, mu, var = batch_norm_forward(X, gamma, beta)
print("=== batch norm from scratch ===")
print(f"input : mean {X.mean():+.4f}  std {X.std():.4f}")
print(f"output: mean {Z.mean():+.4f}  std {Z.std():.4f}   (gamma=1, beta=0)")

# gamma and beta let the layer UNDO the normalisation if it wants to
Z2, _, _ = batch_norm_forward(X, np.sqrt(var), mu)
print(f"with gamma=sigma_B, beta=mu_B: max |output - input| = "
      f"{np.abs(Z2 - X).max():.2e}   <- the identity is representable")

# verify against Keras
bn = keras.layers.BatchNormalization()
out = bn(tf.constant(X, dtype=tf.float32), training=True).numpy()
print(f"max |mine - keras| = {np.abs(Z - out).max():.5f}")

# ============ 2. WHAT BN BUYS: larger learning rates ===================
print("\\n=== BN lets you use much larger learning rates ===")
def build(use_bn, depth=10, width=100):
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(depth):
        if use_bn:
            layers += [keras.layers.Dense(width, use_bias=False,
                                          kernel_initializer="he_normal"),
                       keras.layers.BatchNormalization(),
                       keras.layers.Activation("relu")]
        else:
            layers.append(keras.layers.Dense(width, activation="relu",
                                             kernel_initializer="he_normal"))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    return keras.Sequential(layers)

print(f"{'learning rate':>15}{'without BN':>14}{'with BN':>12}")
for lr in [0.01, 0.1, 0.5, 2.0]:
    row = f"{lr:>15}"
    for use_bn in (False, True):
        tf.random.set_seed(0)
        m = build(use_bn)
        m.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr), metrics=["accuracy"])
        m.fit(Xtr, ytr, epochs=20, batch_size=64, verbose=0)
        acc = m.evaluate(Xte, yte, verbose=0)[1]
        row += f"{acc:>14.4f}" if not use_bn else f"{acc:>12.4f}"
    print(row)
print("\\nWithout BN, lr=2.0 diverges. With BN it still trains.")

# ============ 3. THE FOUR PARAMETER VECTORS ============================
print("\\n=== a BatchNormalization layer's parameters ===")
m = build(True, depth=2, width=50)
m.compile(loss="sparse_categorical_crossentropy",
          optimizer="adam", metrics=["accuracy"])
m.fit(Xtr, ytr, epochs=3, batch_size=64, verbose=0)
bn_layer = [l for l in m.layers if isinstance(l, keras.layers.BatchNormalization)][0]
print(f"{'name':<28}{'shape':>10}{'trainable':>12}")
for v in bn_layer.weights:
    print(f"{v.name:<28}{str(tuple(v.shape)):>10}{str(v.trainable):>12}")
print(f"\\ntrainable params in this BN layer     : "
      f"{sum(int(np.prod(v.shape)) for v in bn_layer.trainable_weights)}")
print(f"non-trainable (moving statistics)     : "
      f"{sum(int(np.prod(v.shape)) for v in bn_layer.non_trainable_weights)}")
print(f"\\nmoving_mean     : {bn_layer.moving_mean.numpy()[:4].round(4)}")
print(f"moving_variance : {bn_layer.moving_variance.numpy()[:4].round(4)}")

# ============ 4. training=True vs training=False =======================
print("\\n=== the inference-mode trap ===")
x1 = Xtr[:1]
print(f"single instance, training=False : "
      f"{m(x1, training=False).numpy()[0].argmax()} "
      f"(conf {m(x1, training=False).numpy()[0].max():.4f})")
big = np.r_[x1, Xtr[100:140]]
p_train = m(big, training=True).numpy()[0]
print(f"same instance in a batch, training=True: {p_train.argmax()} "
      f"(conf {p_train.max():.4f})   <- depends on the OTHER 40 instances!")
print("model.predict() always uses training=False. If you call the model")
print("directly, pass training=False explicitly.")

# ============ 5. use_bias=False before BN ==============================
print("\\n=== the bias before a BN layer is redundant ===")
with_bias = keras.Sequential([keras.layers.Input(shape=(64,)),
                              keras.layers.Dense(100, use_bias=True),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation("relu"),
                              keras.layers.Dense(10, activation="softmax")])
without   = keras.Sequential([keras.layers.Input(shape=(64,)),
                              keras.layers.Dense(100, use_bias=False),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation("relu"),
                              keras.layers.Dense(10, activation="softmax")])
print(f"use_bias=True  : {with_bias.count_params():,} parameters")
print(f"use_bias=False : {without.count_params():,} parameters "
      f"({with_bias.count_params()-without.count_params()} fewer, and identical maths)")

# ============ 6. small batches break BN ================================
print("\\n=== BN needs a reasonable batch size ===")
print(f"{'batch size':>12}{'test accuracy':>16}")
for bs in [2, 4, 8, 32, 128]:
    tf.random.set_seed(0)
    mm = build(True, depth=5, width=64)
    mm.compile(loss="sparse_categorical_crossentropy",
               optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    mm.fit(Xtr, ytr, epochs=8, batch_size=bs, verbose=0)
    print(f"{bs:>12}{mm.evaluate(Xte, yte, verbose=0)[1]:>16.4f}")
print("With batch_size <= 4 the batch statistics are too noisy.")
print("Use LayerNormalization (Ch. 16) or GroupNormalization instead.")
''',
        key="ch11_batchnorm",
    )

    keypoints([
        "BN standardises each layer's inputs over the mini-batch, then applies "
        "<b>learned</b> $\\boldsymbol\\gamma$ and $\\boldsymbol\\beta$.",
        "$\\boldsymbol\\gamma$/$\\boldsymbol\\beta$ preserve expressiveness — the "
        "identity is representable.",
        "It works by <b>smoothing the loss landscape</b>, not by reducing "
        "covariate shift (that explanation was disproved).",
        "Four vectors per feature: 2 trainable, 2 non-trainable moving statistics "
        "used at inference.",
        "Breaks with tiny batches; makes the preceding bias redundant "
        "(<code>use_bias=False</code>).",
    ])


# ==========================================================================
def s_11_4():
    section("11.4", "Gradient Clipping")

    lead(
        "The blunt instrument for exploding gradients: if the gradient is too "
        "big, make it smaller. Crude, and indispensable for recurrent networks."
    )

    sub("Two kinds of clipping")

    math(r"""
    \textbf{clipvalue:}\qquad
    g_i \;\leftarrow\; \mathrm{clip}\bigl(g_i,\; -c,\; +c\bigr)
    \qquad\text{applied element-wise}
    """)
    math(r"""
    \textbf{clipnorm:}\qquad
    \mathbf{g} \;\leftarrow\;
    \begin{cases}
      \mathbf{g} & \text{if } \lVert\mathbf{g}\rVert_2 \le c\\[6pt]
      c\,\dfrac{\mathbf{g}}{\lVert\mathbf{g}\rVert_2} & \text{otherwise}
    \end{cases}
    """)

    idea(
        "clipvalue changes the direction; clipnorm does not",
        "This is the whole difference and it matters. Take the gradient "
        "$\\mathbf{g} = (0.9,\\, 100)$ with <code>clipvalue=1.0</code>: you get "
        "$(0.9,\\, 1.0)$, which points almost diagonally — <b>a completely "
        "different direction</b> from the original, which pointed almost straight "
        "along the second axis. With <code>clipnorm=1.0</code> you get "
        "$(0.00899,\\, 0.99996)$ — the same direction, just shorter.<br><br>"
        "If preserving the descent direction matters (it usually does), use "
        "<code>clipnorm</code>. <code>clipvalue</code> is occasionally useful "
        "precisely <i>because</i> it rebalances, but that is a deliberate choice, "
        "not a default.",
    )

    anim_header("clipvalue vs clipnorm on the same gradient")

    ang = np.linspace(0, 2 * np.pi, 400)
    thetas = np.linspace(0.02, np.pi / 2 - .02, 40)
    cval = 1.0

    frames = []
    for th in thetas:
        g = np.array([np.cos(th), np.sin(th)]) * 3.6
        gv = np.clip(g, -cval, cval)
        n = np.linalg.norm(g)
        gn = g if n <= cval else cval * g / n
        ang_orig = np.degrees(np.arctan2(g[1], g[0]))
        ang_v = np.degrees(np.arctan2(gv[1], gv[0]))
        ang_n = np.degrees(np.arctan2(gn[1], gn[0]))
        frames.append(go.Frame(name=f"{np.degrees(th):.0f}", data=[
            go.Scatter(x=[-cval, cval, cval, -cval, -cval],
                       y=[-cval, -cval, cval, cval, -cval], mode="lines",
                       line=dict(color=C["warning"], width=2, dash="dot")),
            go.Scatter(x=cval * np.cos(ang), y=cval * np.sin(ang), mode="lines",
                       line=dict(color=C["success"], width=2, dash="dot")),
            go.Scatter(x=[0, g[0]], y=[0, g[1]], mode="lines+markers",
                       line=dict(color=C["ink"], width=3)),
            go.Scatter(x=[0, gv[0]], y=[0, gv[1]], mode="lines+markers",
                       line=dict(color=C["danger"], width=4)),
            go.Scatter(x=[0, gn[0]], y=[0, gn[1]], mode="lines+markers",
                       line=dict(color=C["success"], width=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"original ‖g‖ = {n:.2f} at {ang_orig:.1f}°   ·   "
            f"clipvalue → {ang_v:.1f}° (Δ {ang_v-ang_orig:+.1f}°)   ·   "
            f"clipnorm → {ang_n:.1f}° (Δ {ang_n-ang_orig:+.1f}°)",
            color=C["danger"])])))

    g0 = np.array([np.cos(thetas[0]), np.sin(thetas[0])]) * 3.6
    f = go.Figure(data=[
        go.Scatter(x=[-cval, cval, cval, -cval, -cval],
                   y=[-cval, -cval, cval, cval, -cval], mode="lines",
                   name="clipvalue box", line=dict(color=C["warning"], width=2,
                                                   dash="dot")),
        go.Scatter(x=cval * np.cos(ang), y=cval * np.sin(ang), mode="lines",
                   name="clipnorm ball", line=dict(color=C["success"], width=2,
                                                   dash="dot")),
        go.Scatter(x=[0, g0[0]], y=[0, g0[1]], mode="lines+markers",
                   name="original gradient", line=dict(color=C["ink"], width=3)),
        go.Scatter(x=[0, cval], y=[0, cval], mode="lines+markers",
                   name="after clipvalue", line=dict(color=C["danger"], width=4)),
        go.Scatter(x=[0, cval], y=[0, cval], mode="lines+markers",
                   name="after clipnorm", line=dict(color=C["success"], width=4)),
    ])
    f.update_layout(height=490, xaxis=dict(range=[-.4, 4], title="g₁",
                                           scaleanchor="y"),
                    yaxis=dict(range=[-.4, 4], title="g₂"),
                    title="clipvalue rebalances the components; clipnorm rescales "
                          "the vector",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="angle ")
    figure(f)

    tip(
        "How to choose the clipping threshold",
        "Train without clipping for a few hundred iterations and record the "
        "gradient norm. Set <code>clipnorm</code> to roughly the <b>90th "
        "percentile</b> of what you observe: normal steps pass through untouched, "
        "and only the genuine spikes are cut. The lab below does exactly this.",
    )

    note(
        "Clipping matters most for RNNs",
        "In a recurrent network the same weight matrix is applied at every time "
        "step, so the gradient product of §11.1 runs over the <i>sequence length</i> "
        "rather than the layer count — 100+ factors for a long sequence. Batch "
        "normalisation does not fit naturally into an RNN, so clipping is the "
        "standard defence. Chapter 15 returns to this.",
    )

    code_lab(
        "Gradient norms, clipping thresholds, and the direction change",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE DIRECTION CHANGE ===================================
print("=== clipvalue changes the direction, clipnorm does not ===")
def clip_value(g, c): return np.clip(g, -c, c)
def clip_norm(g, c):
    n = np.linalg.norm(g)
    return g if n <= c else c * g / n

def angle(u, v):
    return np.degrees(np.arccos(np.clip(u @ v /
                      (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)))

print(f"{'gradient':>26}{'clipvalue result':>26}{'Δangle':>9}"
      f"{'clipnorm result':>26}{'Δangle':>9}")
for g in [np.array([0.9, 100.]), np.array([50., 50.]),
          np.array([0.1, 0.1]), np.array([-30., 2., 0.5])]:
    gv, gn = clip_value(g, 1.0), clip_norm(g, 1.0)
    print(f"{str(g.round(2)):>26}{str(gv.round(4)):>26}{angle(g,gv):>9.2f}"
          f"{str(gn.round(4)):>26}{angle(g,gn):>9.2f}")
print("\\nclipnorm's angle change is always exactly 0. clipvalue's is not.")

# ============ 2. MEASURE REAL GRADIENT NORMS ===========================
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.25,
                                      stratify=d.target, random_state=42)

class GradNormLogger(keras.callbacks.Callback):
    def __init__(self): self.norms = []
    def on_train_batch_end(self, batch, logs=None): pass

def measure_grad_norms(model, X, y, n_batches=120, batch_size=32):
    """Record the global gradient norm for a series of mini-batches."""
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    norms = []
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        idx = rng.choice(len(X), batch_size, replace=False)
        with tf.GradientTape() as tape:
            loss = loss_fn(y[idx], model(X[idx], training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        norms.append(float(tf.linalg.global_norm(grads)))
    return np.array(norms)

def build(depth=8, width=100, init="he_normal"):
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(width, activation="relu",
                                         kernel_initializer=init))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    return keras.Sequential(layers)

tf.random.set_seed(0)
m = build()
norms = measure_grad_norms(m, Xtr, ytr)
print(f"\\n=== gradient norms over 120 mini-batches (untrained network) ===")
for p in [10, 50, 75, 90, 95, 99, 100]:
    print(f"  {p:>3}th percentile: {np.percentile(norms, p):>9.4f}")
print(f"\\n  mean {norms.mean():.4f}   max {norms.max():.4f}   "
      f"max/median = {norms.max()/np.median(norms):.1f}x")
suggested = float(np.percentile(norms, 90))
print(f"\\nsuggested clipnorm = 90th percentile = {suggested:.4f}")

# ============ 3. CLIPPING RESCUES A DIVERGING RUN ======================
print("\\n=== a badly initialised deep network ===")
print(f"{'clipping':<26}{'final loss':>14}{'test accuracy':>16}")
for nm, kw in [("none",                      {}),
               ("clipvalue=1.0",             dict(clipvalue=1.0)),
               (f"clipnorm={suggested:.2f}", dict(clipnorm=suggested)),
               ("clipnorm=1.0",              dict(clipnorm=1.0))]:
    tf.random.set_seed(0)
    # a deliberately bad initialisation to provoke exploding gradients
    mm = build(depth=12, init=keras.initializers.RandomNormal(stddev=0.45))
    mm.compile(loss="sparse_categorical_crossentropy",
               optimizer=keras.optimizers.SGD(learning_rate=0.15, **kw),
               metrics=["accuracy"])
    h = mm.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
    fl = h.history["loss"][-1]
    fl_s = "NaN/inf" if not np.isfinite(fl) else f"{fl:.4f}"
    print(f"{nm:<26}{fl_s:>14}{mm.evaluate(Xte, yte, verbose=0)[1]:>16.4f}")

# ============ 4. how often does clipping actually fire? ================
print(f"\\n=== how often would clipnorm={suggested:.2f} fire? ===")
for thr in [suggested, 1.0, 5.0, 20.0]:
    print(f"  threshold {thr:>7.3f}: clips {np.mean(norms > thr):>6.1%} of batches, "
          f"mean shrink factor {np.mean(np.minimum(1, thr/norms)):.3f}")
print("\\nA good threshold clips ~10 % of batches. Clipping 90 % of them means")
print("you have effectively just lowered the learning rate.")

import plotly.graph_objects as go
fig = go.Figure(go.Histogram(x=norms, nbinsx=45,
                             marker=dict(color=C["primary"])))
fig.add_vline(x=suggested, line_dash="dash", line_color=C["success"],
              annotation_text=f"90th pct = {suggested:.2f}")
fig.add_vline(x=norms.max(), line_dash="dot", line_color=C["danger"],
              annotation_text="max")
fig.update_layout(height=380, xaxis_title="global gradient norm",
                  yaxis_title="count", title="The gradient-norm distribution")
''',
        key="ch11_clipping",
    )

    keypoints([
        "<code>clipvalue</code> clips each component — it <b>changes the "
        "direction</b>.",
        "<code>clipnorm</code> rescales the whole vector — it <b>preserves the "
        "direction</b>. Prefer it.",
        "Set the threshold to about the 90th percentile of the observed gradient "
        "norms.",
        "Clipping most batches means you have just lowered the learning rate "
        "awkwardly.",
        "Essential for RNNs (Chapter 15), where BN does not fit and the gradient "
        "product runs over the sequence length.",
    ])


# ==========================================================================
def s_11_5():
    section("11.5", "Reusing Pretrained Layers — Transfer Learning")

    lead(
        "Do not train from scratch. Find a network trained on a similar task, keep "
        "its lower layers, and retrain only the top. This is the single largest "
        "practical lever in applied deep learning."
    )

    sub("Why it works")

    idea(
        "Low-level features are universal",
        "The lowest layers of a vision network learn edges, colour blobs and "
        "gradients; the next learn corners, textures and simple shapes; the next "
        "learn object parts. <b>None of that is specific to the original task</b> "
        "— every natural image has edges. Only the top layers encode "
        "task-specific concepts. So the lower you go, the more reusable the "
        "layer, and the more layers you can freeze.",
    )

    sub("The procedure")

    md(
        """
1. **Load** the pretrained model, excluding its output layer.
2. **Freeze** the reused layers: `layer.trainable = False`.
3. **Add** your own output layer (and possibly one or two new hidden layers).
4. **Compile** — this step is mandatory after changing `trainable`.
5. **Train** for a few epochs. The new layers get sensible weights.
6. **Unfreeze** the top few reused layers, **lower the learning rate by 10–100×**,
   recompile, and train again. This is **fine-tuning**.
        """
    )

    pitfall(
        "Three transfer-learning mistakes",
        "<b>(1) Forgetting to recompile.</b> Changing <code>trainable</code> has "
        "no effect until you call <code>compile()</code> again — Keras caches the "
        "trainable variable list at compile time. Your 'frozen' layers will keep "
        "training.<br>"
        "<b>(2) Unfreezing too early.</b> The randomly initialised output layer "
        "produces enormous gradients in the first epochs, which will destroy the "
        "carefully learned pretrained weights. Always freeze first, train the "
        "head, <i>then</i> unfreeze.<br>"
        "<b>(3) Not lowering the learning rate when fine-tuning.</b> The "
        "pretrained weights are nearly right; a large learning rate throws that "
        "away. Divide by 10 to 100.",
    )

    sub("How much to reuse")

    table(
        ["Similarity of tasks", "Size of your dataset", "Strategy"],
        [["Very similar", "Small", "Freeze <b>everything</b>, train only a new "
          "output layer (feature extraction)"],
         ["Very similar", "Large", "Freeze the bottom, fine-tune the top few "
          "layers"],
         ["Somewhat different", "Small", "Freeze the bottom layers only; you may "
          "need to <i>drop</i> the top pretrained layers rather than reuse them"],
         ["Somewhat different", "Large", "Fine-tune everything with a low "
          "learning rate"],
         ["Very different", "Large", "Transfer helps little; consider training "
          "from scratch — but pretrained weights are still often a better "
          "<i>initialisation</i> than random"]],
    )

    note(
        "Transfer learning does not work well for small dense networks",
        "It works spectacularly for deep convolutional networks (Chapter 14) and "
        "Transformers (Chapter 16), because those learn genuinely general "
        "hierarchical features. A small MLP learns few, highly task-specific "
        "patterns, and its layers are not organised into a reusable hierarchy — so "
        "there is little to transfer.",
    )

    sub("When you have no labelled data")

    table(
        ["Approach", "Idea", "Chapter"],
        [["<b>Unsupervised pretraining</b>",
          "Train an autoencoder or GAN on unlabelled data, then reuse its lower "
          "layers", "17"],
         ["<b>Self-supervised pretraining</b>",
          "Manufacture labels from the data itself — mask tokens and predict them, "
          "predict the next frame, contrastive learning", "16"],
         ["<b>Auxiliary-task pretraining</b>",
          "Train on a related task where labels <i>are</i> cheap, then transfer. "
          "E.g. train a face detector on abundant web faces, then fine-tune for "
          "your 10 specific people", "14"]],
    )

    anim_header("Fine-tuning: freeze, train the head, then unfreeze")
    md(
        "The three phases of a transfer-learning run. Phase 1 (blue) trains only "
        "the new head with everything else frozen. Phase 2 (green) unfreezes the "
        "top layers at 1/100th of the learning rate. The counterfactual — "
        "unfreezing immediately — is the red curve."
    )

    rng = np.random.default_rng(2)
    ep = np.arange(1, 61)
    phase1 = np.where(ep <= 20,
                      .92 - .55 * (1 - np.exp(-(ep) / 5)), np.nan)
    p1_end = .92 - .55 * (1 - np.exp(-20 / 5))
    phase2 = np.where(ep > 20,
                      p1_end - .22 * (1 - np.exp(-(ep - 20) / 9)), np.nan)
    naive = .92 - .30 * (1 - np.exp(-ep / 14)) + .18 * np.exp(-ep / 3)
    phase1 += rng.normal(0, .006, 60)
    phase2 += rng.normal(0, .005, 60)
    naive += rng.normal(0, .012, 60)

    frames = []
    for k in range(2, 61):
        ph = ("phase 1 — head only, base FROZEN" if k <= 20
              else "phase 2 — top unfrozen, lr / 100")
        col = C["train"] if k <= 20 else C["success"]
        val = phase1[k - 1] if k <= 20 else phase2[k - 1]
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=ep[:k], y=phase1[:k], mode="lines",
                       line=dict(color=C["train"], width=3.4)),
            go.Scatter(x=ep[:k], y=phase2[:k], mode="lines",
                       line=dict(color=C["success"], width=3.4)),
            go.Scatter(x=ep[:k], y=naive[:k], mode="lines",
                       line=dict(color=C["danger"], width=2.6, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"epoch {k}   ·   {ph}   ·   validation loss = {val:.4f}   ·   "
            f"naive (no freezing) = {naive[k-1]:.4f}", color=col)])))

    f = go.Figure(data=[
        go.Scatter(x=ep[:2], y=phase1[:2], mode="lines",
                   name="phase 1: base frozen",
                   line=dict(color=C["train"], width=3.4)),
        go.Scatter(x=ep[:2], y=phase2[:2], mode="lines",
                   name="phase 2: fine-tuning at lr/100",
                   line=dict(color=C["success"], width=3.4)),
        go.Scatter(x=ep[:2], y=naive[:2], mode="lines",
                   name="unfreezing immediately (destroys the weights)",
                   line=dict(color=C["danger"], width=2.6, dash="dash")),
    ])
    f.add_vline(x=20, line_dash="dot", line_color=C["muted"],
                annotation_text="unfreeze + recompile at lr/100")
    f.update_layout(height=440, xaxis_title="epoch",
                    yaxis_title="validation loss",
                    title="The two-phase transfer-learning schedule",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="epoch ")
    figure(f, "The red curve spikes at the start: the random head's large "
              "gradients wreck the pretrained features before they can help.")

    code_lab(
        "Transfer learning end to end, with the recompile trap demonstrated",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)

# ---- task A: digits 0-4 (plenty of data) ------------------------------
# ---- task B: digits 5-9 (only 200 labelled examples) ------------------
d = load_digits()
X, y = d.data/16., d.target
mask_A = y < 5
X_A, y_A = X[mask_A], y[mask_A]
X_B, y_B = X[~mask_A], y[~mask_A] - 5

XA_tr, XA_te, yA_tr, yA_te = train_test_split(X_A, y_A, test_size=.25,
                                              stratify=y_A, random_state=42)
XB_tr, XB_te, yB_tr, yB_te = train_test_split(X_B, y_B, test_size=.4,
                                              stratify=y_B, random_state=42)
XB_small, yB_small = XB_tr[:200], yB_tr[:200]        # a SMALL task-B set
print(f"task A: {len(XA_tr)} training instances (digits 0-4)")
print(f"task B: {len(XB_small)} training instances (digits 5-9)  <- scarce")

def base_model(n_out):
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal",
                           name="hidden1"),
        keras.layers.Dense(96,  activation="relu", kernel_initializer="he_normal",
                           name="hidden2"),
        keras.layers.Dense(64,  activation="relu", kernel_initializer="he_normal",
                           name="hidden3"),
        keras.layers.Dense(n_out, activation="softmax", name="output"),
    ])

# ============ 1. train on task A =======================================
model_A = base_model(5)
model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
model_A.fit(XA_tr, yA_tr, epochs=30, batch_size=32, verbose=0)
print(f"\\ntask A test accuracy = {model_A.evaluate(XA_te, yA_te, verbose=0)[1]:.4f}")

# ============ 2. BASELINE: train task B from scratch ===================
tf.random.set_seed(42)
scratch = base_model(5)
scratch.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
scratch.fit(XB_small, yB_small, epochs=40, batch_size=16, verbose=0)
acc_scratch = scratch.evaluate(XB_te, yB_te, verbose=0)[1]
print(f"\\ntask B FROM SCRATCH  ({len(XB_small)} examples) = {acc_scratch:.4f}")

# ============ 3. TRANSFER: clone, freeze, train head, then fine-tune ===
# CLONE so we do not corrupt model_A
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

model_B = keras.Sequential([keras.layers.Input(shape=(64,))]
                           + model_A_clone.layers[:-1])     # drop the old head
model_B.add(keras.layers.Dense(5, activation="softmax", name="new_output"))

# --- PHASE 1: freeze the reused layers -------------------------------
for layer in model_B.layers[:-1]:
    layer.trainable = False
model_B.compile(loss="sparse_categorical_crossentropy",          # RECOMPILE
                optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
print(f"\\nphase 1: {sum(int(np.prod(v.shape)) for v in model_B.trainable_variables):,}"
      f" trainable of {model_B.count_params():,} total")
model_B.fit(XB_small, yB_small, epochs=15, batch_size=16, verbose=0)
acc_frozen = model_B.evaluate(XB_te, yB_te, verbose=0)[1]
print(f"phase 1 (head only) test accuracy = {acc_frozen:.4f}")

# --- PHASE 2: unfreeze, LOWER the learning rate, recompile -----------
for layer in model_B.layers[:-1]:
    layer.trainable = True
model_B.compile(loss="sparse_categorical_crossentropy",          # RECOMPILE
                optimizer=keras.optimizers.Adam(1e-5),           # lr / 100
                metrics=["accuracy"])
print(f"phase 2: {sum(int(np.prod(v.shape)) for v in model_B.trainable_variables):,}"
      f" trainable, learning rate 1e-5")
model_B.fit(XB_small, yB_small, epochs=25, batch_size=16, verbose=0)
acc_ft = model_B.evaluate(XB_te, yB_te, verbose=0)[1]
print(f"phase 2 (fine-tuned)  test accuracy = {acc_ft:.4f}")

print(f"\\n{'approach':<34}{'test accuracy':>15}")
print(f"{'from scratch':<34}{acc_scratch:>15.4f}")
print(f"{'transfer, frozen base':<34}{acc_frozen:>15.4f}")
print(f"{'transfer, fine-tuned':<34}{acc_ft:>15.4f}")
print(f"{'improvement over scratch':<34}{acc_ft-acc_scratch:>+15.4f}")

# ============ 4. THE RECOMPILE TRAP ====================================
print("\\n=== the recompile trap ===")
m = keras.models.clone_model(model_A); m.set_weights(model_A.get_weights())
m.compile(loss="sparse_categorical_crossentropy",
          optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
before = [w.copy() for w in m.layers[0].get_weights()]
m.layers[0].trainable = False                 # frozen... but NOT recompiled
m.fit(XA_tr[:200], yA_tr[:200], epochs=5, verbose=0)
after = m.layers[0].get_weights()
print(f"set trainable=False WITHOUT recompiling: weights changed by "
      f"{np.abs(before[0]-after[0]).max():.2e}   <- IT STILL TRAINED")

m.compile(loss="sparse_categorical_crossentropy",
          optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
before = [w.copy() for w in m.layers[0].get_weights()]
m.fit(XA_tr[:200], yA_tr[:200], epochs=5, verbose=0)
after = m.layers[0].get_weights()
print(f"after recompiling:                       weights changed by "
      f"{np.abs(before[0]-after[0]).max():.2e}   <- properly frozen")

# ============ 5. HOW MANY LAYERS TO FREEZE? ============================
print("\\n=== freezing depth vs accuracy ===")
print(f"{'frozen layers':>15}{'trainable params':>19}{'test accuracy':>16}")
for n_freeze in range(0, 4):
    tf.random.set_seed(0)
    clone = keras.models.clone_model(model_A); clone.set_weights(model_A.get_weights())
    mb = keras.Sequential([keras.layers.Input(shape=(64,))] + clone.layers[:-1])
    mb.add(keras.layers.Dense(5, activation="softmax"))
    for l in mb.layers[:n_freeze]:
        l.trainable = False
    mb.compile(loss="sparse_categorical_crossentropy",
               optimizer=keras.optimizers.Adam(3e-4), metrics=["accuracy"])
    mb.fit(XB_small, yB_small, epochs=30, batch_size=16, verbose=0)
    n_tr = sum(int(np.prod(v.shape)) for v in mb.trainable_variables)
    print(f"{n_freeze:>15}{n_tr:>19,}{mb.evaluate(XB_te, yB_te, verbose=0)[1]:>16.4f}")
print("\\nWith only 200 labelled examples, freezing more layers usually wins --")
print("fewer trainable parameters means less overfitting.")
''',
        key="ch11_transfer",
    )

    keypoints([
        "Reuse the lower layers of a network trained on a similar task; they "
        "encode <b>universal</b> low-level features.",
        "Freeze → train the new head → unfreeze the top → <b>lower the learning "
        "rate 10–100×</b> → fine-tune.",
        "<b>You must recompile</b> after changing <code>trainable</code>, or "
        "nothing is actually frozen.",
        "Freeze more layers when your dataset is small; fewer when it is large.",
        "It works spectacularly for CNNs and Transformers, poorly for small MLPs.",
    ])


# ==========================================================================
def s_11_6():
    section("11.6", "Faster Optimizers")

    lead(
        "Plain gradient descent takes the same-sized step in every direction, "
        "regardless of the curvature. Every optimiser here is a different answer "
        "to \"how should the step size adapt?\""
    )

    sub("Momentum")

    math(r"""
    \mathbf{m} \;\leftarrow\; \beta\,\mathbf{m}
      \;-\; \eta\,\nabla_{\boldsymbol\theta} J(\boldsymbol\theta)
    \qquad\qquad
    \boldsymbol\theta \;\leftarrow\; \boldsymbol\theta + \mathbf{m}
    """)
    where({r"\beta": "the momentum, typically 0.9 — a friction term; $\\beta = 0$ "
                     "recovers plain gradient descent"})

    derive(
        [("At a constant gradient $\\mathbf{g}$, the momentum vector converges to "
          "a geometric series:",
          r"\mathbf{m}_\infty = -\eta\mathbf{g}\bigl(1 + \beta + \beta^2 + \dots\bigr) "
          r"= \frac{-\eta\mathbf{g}}{1 - \beta}"),
         ("So the <b>terminal velocity</b> is $\\frac{1}{1-\\beta}$ times the plain "
          "gradient descent step. With $\\beta = 0.9$ that is <b>10×</b> faster; "
          "with $\\beta = 0.99$, 100× faster.", None),
         ("This is exactly why momentum escapes plateaus: on a long flat valley "
          "floor the gradient is small but consistent, so the velocity builds up "
          "and the optimiser accelerates. Plain GD crawls.", None),
         ("It also damps oscillation. Across a narrow valley the gradient flips "
          "sign every step, so successive contributions cancel in $\\mathbf{m}$ — "
          "the oscillating component is suppressed while the consistent component "
          "accumulates.", None)],
        title="Terminal velocity: why momentum is 1/(1−β) times faster",
    )

    sub("Nesterov Accelerated Gradient")

    math(r"""
    \mathbf{m} \;\leftarrow\; \beta\,\mathbf{m}
      \;-\; \eta\,\nabla_{\boldsymbol\theta} J\bigl(\boldsymbol\theta
        + \beta\,\mathbf{m}\bigr)
    \qquad\qquad
    \boldsymbol\theta \;\leftarrow\; \boldsymbol\theta + \mathbf{m}
    """)

    md(
        "The only change: measure the gradient **slightly ahead**, at "
        "$\\boldsymbol\\theta + \\beta\\mathbf{m}$, where momentum is about to take "
        "you. Since the momentum vector generally points in the right direction, "
        "the look-ahead gradient is a more accurate estimate — and it applies a "
        "correction *before* overshooting rather than after. Nearly always "
        "slightly better than plain momentum, at no cost: `nesterov=True`."
    )

    sub("AdaGrad")

    math(r"""
    \mathbf{s} \;\leftarrow\; \mathbf{s} + \nabla J \otimes \nabla J
    \qquad\qquad
    \boldsymbol\theta \;\leftarrow\; \boldsymbol\theta
      - \eta\,\nabla J \oslash \sqrt{\mathbf{s} + \varepsilon}
    """)
    where({r"\otimes, \oslash": "element-wise multiplication and division",
           r"\mathbf{s}": "the running sum of squared gradients, per parameter"})

    md(
        "Each parameter gets its own effective learning rate, inversely "
        "proportional to the accumulated gradient magnitude in *that* direction. "
        "Steep directions are damped; flat directions keep a large step. This is "
        "**adaptive** learning and it points the update more directly at the "
        "global optimum."
    )

    warn(
        "AdaGrad stops too early",
        "$\\mathbf{s}$ only ever grows, so the effective learning rate "
        "$\\eta/\\sqrt{s}$ decays monotonically to zero. On a convex problem or a "
        "simple linear model this is fine — it is a principled schedule. On a "
        "neural network it typically stops before reaching a good solution. "
        "<b>Do not use AdaGrad for deep networks.</b> RMSProp is the fix.",
    )

    sub("RMSProp")

    math(r"""
    \mathbf{s} \;\leftarrow\; \rho\,\mathbf{s}
      + (1-\rho)\,\nabla J \otimes \nabla J
    \qquad\qquad
    \boldsymbol\theta \;\leftarrow\; \boldsymbol\theta
      - \eta\,\nabla J \oslash \sqrt{\mathbf{s} + \varepsilon}
    """)
    where({r"\rho": "the decay rate, default 0.9"})

    md(
        "One change from AdaGrad: an **exponentially decaying average** instead of "
        "a sum. Old gradients are forgotten, so $\\mathbf{s}$ reflects only the "
        "*recent* curvature and the learning rate does not collapse."
    )

    sub("Adam — adaptive moment estimation")

    math(r"""
    \mathbf{m} \leftarrow \beta_1 \mathbf{m}
      - (1 - \beta_1)\,\nabla J
    \qquad\qquad
    \mathbf{s} \leftarrow \beta_2 \mathbf{s}
      + (1 - \beta_2)\,\nabla J \otimes \nabla J
    """)
    math(r"""
    \hat{\mathbf{m}} \leftarrow \frac{\mathbf{m}}{1 - \beta_1^{\,t}}
    \qquad\qquad
    \hat{\mathbf{s}} \leftarrow \frac{\mathbf{s}}{1 - \beta_2^{\,t}}
    \qquad\qquad
    \boldsymbol\theta \leftarrow \boldsymbol\theta
      + \eta\,\hat{\mathbf{m}} \oslash \sqrt{\hat{\mathbf{s}} + \varepsilon}
    """)
    where({r"\beta_1": "momentum decay, default 0.9",
           r"\beta_2": "scaling decay, default 0.999",
           r"t": "the iteration counter, starting at 1",
           r"\varepsilon": "default $10^{-7}$"})

    proof(
        "Why the bias correction is needed",
        "$\\mathbf{m}$ and $\\mathbf{s}$ are initialised to zero, so early in "
        "training they are biased toward zero. Taking expectations with a "
        "stationary gradient: $\\mathbb{E}[\\mathbf{m}_t] = (1-\\beta_1^t)"
        "\\mathbb{E}[\\mathbf{g}]$. At $t = 1$ with $\\beta_2 = 0.999$, "
        "$\\mathbf{s}$ is only $0.1\\%$ of its true value, so "
        "$1/\\sqrt{\\mathbf{s}}$ would be ~32× too large. Dividing by "
        "$1 - \\beta_2^t$ removes the bias exactly. The correction matters for "
        "roughly the first $1/(1-\\beta_2) = 1000$ steps, then becomes "
        "negligible.",
    )

    md("**Adam = momentum + RMSProp + bias correction.** It is the default "
       "optimiser for most work.")

    sub("The variants")

    table(
        ["Optimiser", "Change from Adam", "When"],
        [["<b>AdaMax</b>",
          "Uses the $\\ell_\\infty$ norm: "
          "$\\mathbf{s} \\leftarrow \\max(\\beta_2\\mathbf{s}, |\\nabla J|)$ "
          "instead of the $\\ell_2$ norm",
          "More stable on some problems; try it if Adam misbehaves"],
         ["<b>Nadam</b>", "Adam + the Nesterov look-ahead",
          "Often converges slightly faster than Adam"],
         ["<b>AdamW</b>",
          "<b>Decoupled weight decay</b>: subtracts $\\eta\\lambda\\boldsymbol\\theta$ "
          "directly instead of adding $\\lambda\\boldsymbol\\theta$ to the gradient",
          "<b>Use this instead of Adam whenever you want $\\ell_2$ "
          "regularisation</b>"]],
    )

    idea(
        "Why AdamW exists — ℓ₂ regularisation and Adam interact badly",
        "Classic $\\ell_2$ adds $\\lambda\\boldsymbol\\theta$ to the gradient. Adam "
        "then <i>divides that by $\\sqrt{\\mathbf{s}}$ along with everything "
        "else</i> — so parameters with large historical gradients get "
        "<b>less</b> weight decay, which is exactly backwards. Loshchilov & Hutter "
        "(2019) showed that <b>decoupling</b> the decay from the adaptive scaling "
        "recovers the intended behaviour and consistently generalises better. If "
        "you are using Adam with weight decay, use "
        "<code>AdamW</code>.",
    )

    warn(
        "Adaptive optimisers can generalise worse",
        "Adam and friends converge fast but sometimes land in a sharper minimum "
        "that generalises less well than the one SGD+Nesterov finds. If your "
        "validation accuracy disappoints while training loss looks great, "
        "<b>try SGD with Nesterov momentum and a good learning-rate schedule</b>. "
        "This is a real, reproducible effect, not folklore.",
    )

    anim_header("Every optimiser racing on the same surface")
    md(
        "A narrow curved valley — the classic hard case. Plain SGD zig-zags "
        "across it; momentum accelerates along it; the adaptive methods rescale "
        "per-dimension and go almost straight."
    )

    def loss_fn(p):
        x, y = p[0], p[1]
        return 0.06 * x ** 2 + 3.2 * y ** 2

    def grad_fn(p):
        return np.array([0.12 * p[0], 6.4 * p[1]])

    def optimise(kind, steps=90, lr=.08, start=(-4.2, 1.6)):
        th = np.array(start, float)
        m = np.zeros(2); s = np.zeros(2); path = [th.copy()]
        for t in range(1, steps + 1):
            g = grad_fn(th)
            if kind == "SGD":
                th = th - lr * g
            elif kind == "Momentum":
                m = .9 * m - lr * g
                th = th + m
            elif kind == "Nesterov":
                g2 = grad_fn(th + .9 * m)
                m = .9 * m - lr * g2
                th = th + m
            elif kind == "AdaGrad":
                s = s + g * g
                th = th - (lr * 6) * g / (np.sqrt(s) + 1e-8)
            elif kind == "RMSProp":
                s = .9 * s + .1 * g * g
                th = th - (lr * 2) * g / (np.sqrt(s) + 1e-8)
            elif kind == "Adam":
                m = .9 * m - .1 * g
                s = .999 * s + .001 * g * g
                mh = m / (1 - .9 ** t); sh = s / (1 - .999 ** t)
                th = th + (lr * 3) * mh / (np.sqrt(sh) + 1e-8)
            th = np.clip(th, -8, 8)
            path.append(th.copy())
        return np.array(path)

    opts = ["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam"]
    paths = {o: optimise(o) for o in opts}

    gx = np.linspace(-5.5, 5.5, 130); gy = np.linspace(-2.2, 2.2, 130)
    GX, GY = np.meshgrid(gx, gy)
    Z = 0.06 * GX ** 2 + 3.2 * GY ** 2

    frames = []
    for k in range(1, 91):
        data = [go.Contour(x=gx, y=gy, z=Z, colorscale=nav.cscale(),
                           showscale=False, opacity=.8, ncontours=26)]
        info = []
        for i, o in enumerate(opts):
            p = paths[o]
            data.append(go.Scatter(x=p[:k + 1, 0], y=p[:k + 1, 1],
                                   mode="lines", line=dict(color=SEQ[i], width=2.8)))
            info.append(f"{o[:4]} {loss_fn(p[k]):.3f}")
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"step {k}   |   loss:  " + "  ".join(info))])))

    f = go.Figure(data=[go.Contour(x=gx, y=gy, z=Z, colorscale=nav.cscale(),
                                   showscale=False, opacity=.8, ncontours=26)]
                  + [go.Scatter(x=paths[o][:1, 0], y=paths[o][:1, 1], mode="lines",
                                name=o, line=dict(color=SEQ[i], width=2.8))
                     for i, o in enumerate(opts)])
    f.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="optimum",
                           marker=dict(color="#fff", size=15, symbol="star",
                                       line=dict(color=C["ink"], width=2))))
    f.update_layout(height=500, xaxis=dict(range=[-5.5, 5.5], title="θ₁"),
                    yaxis=dict(range=[-2.2, 2.2], title="θ₂"),
                    title="Six optimisers on an ill-conditioned valley "
                          "(κ ≈ 53)",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(80), slider_prefix="step ")
    figure(f)

    code_lab(
        "Every optimiser implemented from scratch, then raced in Keras",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

# ============ ALL SEVEN, FROM SCRATCH ==================================
def rosen(p):     return (1-p[0])**2 + 100*(p[1]-p[0]**2)**2
def rosen_grad(p):
    return np.array([-2*(1-p[0]) - 400*p[0]*(p[1]-p[0]**2),
                     200*(p[1]-p[0]**2)])

def run(kind, lr, steps=4000, start=(-1.5, 2.0), **kw):
    th = np.array(start, float)
    m = np.zeros(2); s = np.zeros(2); u = np.zeros(2)
    b1, b2, eps = kw.get("b1", .9), kw.get("b2", .999), 1e-8
    for t in range(1, steps+1):
        g = rosen_grad(th)
        if kind == "SGD":
            th -= lr*g
        elif kind == "Momentum":
            m = b1*m - lr*g;  th += m
        elif kind == "Nesterov":
            g = rosen_grad(th + b1*m); m = b1*m - lr*g; th += m
        elif kind == "AdaGrad":
            s += g*g;  th -= lr*g/(np.sqrt(s)+eps)
        elif kind == "RMSProp":
            s = .9*s + .1*g*g;  th -= lr*g/(np.sqrt(s)+eps)
        elif kind == "Adam":
            m = b1*m + (1-b1)*g
            s = b2*s + (1-b2)*g*g
            th -= lr*(m/(1-b1**t))/(np.sqrt(s/(1-b2**t))+eps)
        elif kind == "AdaMax":
            m = b1*m + (1-b1)*g
            u = np.maximum(b2*u, np.abs(g))          # l_inf norm
            th -= lr*(m/(1-b1**t))/(u+eps)
        elif kind == "Nadam":
            m = b1*m + (1-b1)*g
            s = b2*s + (1-b2)*g*g
            mh = (b1*m/(1-b1**(t+1))) + ((1-b1)*g/(1-b1**t))
            th -= lr*mh/(np.sqrt(s/(1-b2**t))+eps)
        elif kind == "AdamW":
            m = b1*m + (1-b1)*g
            s = b2*s + (1-b2)*g*g
            th -= lr*(m/(1-b1**t))/(np.sqrt(s/(1-b2**t))+eps)
            th -= lr*kw.get("wd", .01)*th            # DECOUPLED decay
        if not np.all(np.isfinite(th)): return th, np.inf
    return th, rosen(th)

print("=== Rosenbrock (optimum at (1,1), f=0), 4000 steps ===")
print(f"{'optimiser':<12}{'lr':>9}{'final theta':>26}{'final loss':>14}")
for kind, lr in [("SGD", 1e-3), ("Momentum", 1e-3), ("Nesterov", 1e-3),
                 ("AdaGrad", .3), ("RMSProp", .01), ("Adam", .02),
                 ("AdaMax", .02), ("Nadam", .02), ("AdamW", .02)]:
    th, L = run(kind, lr)
    print(f"{kind:<12}{lr:>9}{str(th.round(4)):>26}{L:>14.6f}")

# ============ MOMENTUM'S TERMINAL VELOCITY =============================
print("\\n=== terminal velocity = 1/(1-beta) times the plain GD step ===")
print(f"{'beta':>7}{'1/(1-beta)':>13}{'measured speed-up':>21}")
for beta in [0., .5, .9, .95, .99]:
    m_, g_, lr_ = 0., 1., .01
    for _ in range(3000):
        m_ = beta*m_ - lr_*g_
    print(f"{beta:>7}{1/(1-beta) if beta < 1 else np.inf:>13.1f}"
          f"{abs(m_)/lr_:>21.2f}")

# ============ ADAM'S BIAS CORRECTION ===================================
print("\\n=== why Adam needs bias correction ===")
b1, b2 = .9, .999
m_, s_ = 0., 0.
g_ = 1.0                                   # a constant gradient
print(f"{'t':>5}{'m (raw)':>12}{'m (corrected)':>16}{'s (raw)':>12}"
      f"{'s (corrected)':>16}{'step size ratio':>18}")
for t in range(1, 2001):
    m_ = b1*m_ + (1-b1)*g_
    s_ = b2*s_ + (1-b2)*g_*g_
    if t in (1, 2, 10, 100, 1000, 2000):
        mh, sh = m_/(1-b1**t), s_/(1-b2**t)
        raw = m_/(np.sqrt(s_)+1e-8)
        cor = mh/(np.sqrt(sh)+1e-8)
        print(f"{t:>5}{m_:>12.6f}{mh:>16.6f}{s_:>12.8f}{sh:>16.6f}"
              f"{raw/cor:>18.4f}")
print("At t=1 the uncorrected step would be ~3x too large. Correction fixes it.")

# ============ ADAMW vs ADAM + L2 =======================================
print("\\n=== decoupled weight decay ===")
print("classic L2 adds lambda*theta to the GRADIENT, so Adam then divides it")
print("by sqrt(s) -- parameters with big historical gradients get LESS decay.")
print("AdamW subtracts lambda*theta from the WEIGHTS directly.\\n")
print(f"{'|theta| after 2000 steps':<32}{'Adam + L2 in grad':>20}{'AdamW':>10}")
for g_big in [0.1, 10.0]:
    th_a, th_w = np.array([1.0]), np.array([1.0])
    m_a = s_a = m_w = s_w = np.zeros(1)
    for t in range(1, 2001):
        g = np.array([g_big])
        # Adam with L2 folded into the gradient
        gl = g + .01*th_a
        m_a = .9*m_a + .1*gl; s_a = .999*s_a + .001*gl*gl
        th_a = th_a - .001*(m_a/(1-.9**t))/(np.sqrt(s_a/(1-.999**t))+1e-8)
        # AdamW
        m_w = .9*m_w + .1*g; s_w = .999*s_w + .001*g*g
        th_w = th_w - .001*(m_w/(1-.9**t))/(np.sqrt(s_w/(1-.999**t))+1e-8)
        th_w = th_w - .001*.01*th_w
    print(f"{f'gradient magnitude {g_big}':<32}{abs(th_a[0]):>20.6f}"
          f"{abs(th_w[0]):>10.6f}")
print("Adam's decay depends on the gradient magnitude. AdamW's does not.")

# ============ RACE THEM ON A REAL NETWORK ==============================
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
print("\\n=== on a real network ===")
print(f"{'optimiser':<26}{'time':>9}{'train acc':>12}{'test acc':>11}")
optimisers = {
    "SGD":                 keras.optimizers.SGD(.05),
    "SGD + momentum":      keras.optimizers.SGD(.05, momentum=.9),
    "SGD + Nesterov":      keras.optimizers.SGD(.05, momentum=.9, nesterov=True),
    "Adagrad":             keras.optimizers.Adagrad(.05),
    "RMSprop":             keras.optimizers.RMSprop(1e-3),
    "Adam":                keras.optimizers.Adam(1e-3),
    "Adamax":              keras.optimizers.Adamax(1e-3),
    "Nadam":               keras.optimizers.Nadam(1e-3),
    "AdamW (wd=1e-4)":     keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
}
for nm, opt in optimisers.items():
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(64,)),
                          keras.layers.Dense(128, activation="relu",
                                             kernel_initializer="he_normal"),
                          keras.layers.Dense(64, activation="relu",
                                             kernel_initializer="he_normal"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
    t0 = time.perf_counter()
    h = m.fit(Xtr, ytr, epochs=25, batch_size=32, verbose=0)
    print(f"{nm:<26}{time.perf_counter()-t0:>8.2f}s"
          f"{h.history['accuracy'][-1]:>12.4f}"
          f"{m.evaluate(Xte, yte, verbose=0)[1]:>11.4f}")
''',
        key="ch11_optimizers",
    )

    quiz(
        "You are training with Adam and want $\\ell_2$ regularisation. Which "
        "should you use?",
        ["<code>Adam</code> with <code>kernel_regularizer=l2(0.01)</code>",
         "<code>AdamW</code> with <code>weight_decay=0.01</code>",
         "<code>SGD</code> — Adam does not support regularisation",
         "Either; they are mathematically identical"],
        1,
        "They are *not* identical. With Adam, the $\\ell_2$ term is added to the "
        "gradient and then divided by $\\sqrt{\\mathbf{s}}$, so parameters with "
        "large historical gradients get *less* decay — the opposite of what you "
        "want. AdamW decouples the decay from the adaptive scaling.",
        key="ch11q2",
    )

    keypoints([
        "Momentum reaches a <b>terminal velocity</b> of $1/(1-\\beta)$ times the "
        "plain GD step — 10× at $\\beta = 0.9$.",
        "Nesterov measures the gradient <b>ahead</b> of where momentum is going; "
        "free improvement.",
        "AdaGrad decays the learning rate to zero and <b>stops too early</b> on "
        "deep nets; RMSProp fixes it with a decaying average.",
        "<b>Adam = momentum + RMSProp + bias correction</b>. The bias correction "
        "matters for the first ~$1/(1-\\beta_2)$ steps.",
        "Use <b>AdamW</b> whenever you want weight decay; consider SGD+Nesterov "
        "when generalisation matters more than speed.",
    ])


# ==========================================================================
def s_11_7():
    section("11.7", "Learning Rate Scheduling")

    lead(
        "A constant learning rate is almost never optimal. Start large to make "
        "progress, end small to settle into the minimum — the only question is "
        "the shape of the curve in between."
    )

    table(
        ["Schedule", "Formula", "Notes"],
        [["<b>Power</b>",
          "$\\eta(t) = \\dfrac{\\eta_0}{(1 + t/s)^{c}}$",
          "Drops fast then slowly; <code>decay</code> in Keras optimisers"],
         ["<b>Exponential</b>",
          "$\\eta(t) = \\eta_0\\, 0.1^{\\,t/s}$",
          "Divides by 10 every $s$ steps; simple and effective"],
         ["<b>Piecewise constant</b>",
          "$\\eta_0$ for $n_1$ epochs, then $\\eta_1$, …",
          "Requires manual tuning but is easy to reason about"],
         ["<b>Performance</b>",
          "Reduce by a factor when validation stops improving",
          "<code>ReduceLROnPlateau</code>; needs no schedule design"],
         ["<b>1cycle</b>",
          "Rise linearly $\\eta_0 \\to \\eta_{\\max}$, fall back, then far below "
          "$\\eta_0$",
          "<b>Often dramatically faster</b>; Smith (2018)"],
         ["<b>Cosine annealing</b>",
          "$\\eta(t) = \\eta_{\\min} + \\tfrac12(\\eta_0 - \\eta_{\\min})"
          "\\bigl(1 + \\cos(\\pi t/T)\\bigr)$",
          "Smooth; the standard in modern vision and NLP training"],
         ["<b>Warm-up</b>",
          "Linear rise from ~0 over the first few hundred steps",
          "<b>Essential</b> for Transformers and large batches"]],
    )

    sub("1cycle — the fast one")

    md(
        """
Leslie Smith's 1cycle policy, in three phases:

1. **Warm-up:** increase the learning rate linearly from $\\eta_0$ to
   $\\eta_{\\max}$ over the first half of training. $\\eta_{\\max}$ is found with
   the LR range test (§10.8); $\\eta_0 \\approx \\eta_{\\max}/10$.
2. **Anneal:** decrease it linearly back to $\\eta_0$ over the second half.
3. **Finish:** drop it by several more orders of magnitude over the last few
   epochs.

Simultaneously, **momentum moves the opposite way** — from 0.95 down to 0.85
during the rise, back up during the fall. High learning rate with low momentum,
low learning rate with high momentum.
        """
    )

    idea(
        "Why the high-LR phase helps rather than hurts",
        "The large learning rate in the middle of training acts as a "
        "<b>regulariser</b>: it prevents the optimiser from settling into a sharp "
        "narrow minimum, keeping it in wide flat regions of the loss surface. Wide "
        "minima generalise better (they are robust to the small parameter "
        "perturbations that distinguish training from test distributions). The "
        "final low-LR phase then descends carefully within whichever wide basin "
        "the model ended up in.",
    )

    sub("Warm-up")

    warn(
        "Large batches and Transformers require warm-up",
        "At the very start of training the adaptive optimiser's second-moment "
        "estimate $\\mathbf{s}$ is based on almost no data, so the effective step "
        "size is enormous and erratic — the bias correction fixes the "
        "<i>expectation</i> but not the <i>variance</i>. Warm-up gives it a few "
        "hundred steps to stabilise. Without it, Transformer training frequently "
        "diverges in the first few hundred steps, which is why every Transformer "
        "recipe has a warm-up (Chapter 16).",
    )

    anim_header("Every schedule, drawn")

    T = 1000
    t = np.arange(T)
    scheds = {
        "constant": np.full(T, .01),
        "power (c=1, s=200)": .01 / (1 + t / 200) ** 1,
        "exponential (s=300)": .01 * .1 ** (t / 300),
        "piecewise": np.where(t < 300, .01, np.where(t < 700, .003, .0005)),
        "cosine annealing": .0005 + .5 * (.01 - .0005) * (1 + np.cos(np.pi * t / T)),
        "1cycle": np.concatenate([
            np.linspace(.001, .01, T // 2),
            np.linspace(.01, .001, int(T * .4)),
            np.linspace(.001, .00001, T - T // 2 - int(T * .4))]),
        "warm-up + cosine": np.concatenate([
            np.linspace(1e-5, .01, 100),
            .0005 + .5 * (.01 - .0005) * (1 + np.cos(np.pi * np.arange(T - 100)
                                                     / (T - 100)))]),
    }
    frames = []
    for nm, sch in scheds.items():
        frames.append(go.Frame(name=nm.split()[0], data=[
            go.Scatter(x=t, y=sch, mode="lines",
                       line=dict(color=C["primary"], width=3.6))]
            + [go.Scatter(x=t, y=v, mode="lines",
                          line=dict(color=alpha(C["muted"], .35), width=1.4))
               for k, v in scheds.items() if k != nm],
            layout=go.Layout(title=f"{nm}   ·   final η = {sch[-1]:.2e}   ·   "
                                   f"max η = {sch.max():.2e}")))

    nm0 = list(scheds)[0]
    f = go.Figure(data=[go.Scatter(x=t, y=scheds[nm0], mode="lines", name=nm0,
                                   line=dict(color=C["primary"], width=3.6))]
                  + [go.Scatter(x=t, y=v, mode="lines", name=k, showlegend=False,
                                line=dict(color=alpha(C["muted"], .35), width=1.4))
                     for k, v in scheds.items() if k != nm0])
    f.update_layout(height=440, xaxis_title="training step",
                    yaxis_title="learning rate", yaxis_type="log",
                    title=nm0)
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="")
    figure(f)

    code_lab(
        "Implement every schedule and measure which wins",
        '''import numpy as np, math
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=.2, random_state=42)
EPOCHS, BATCH = 40, 32
steps_per_epoch = math.ceil(len(Xtr)/BATCH)

def build():
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(64,  activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(10,  activation="softmax")])

# ============ EVERY SCHEDULE ===========================================
def power_decay(lr0=.01, s=10, c=1):
    return lambda epoch: lr0 / (1 + epoch/s)**c

def exponential_decay(lr0=.01, s=15):
    return lambda epoch: lr0 * 0.1**(epoch/s)

def piecewise(epoch):
    if epoch < 12:  return .01
    if epoch < 28:  return .003
    return .0005

def cosine(lr0=.01, lr_min=1e-5, T=EPOCHS):
    return lambda epoch: lr_min + .5*(lr0-lr_min)*(1+math.cos(math.pi*epoch/T))

def warmup_cosine(lr0=.01, lr_min=1e-5, warm=5, T=EPOCHS):
    def f(epoch):
        if epoch < warm:
            return lr0 * (epoch+1)/warm
        p = (epoch-warm)/(T-warm)
        return lr_min + .5*(lr0-lr_min)*(1+math.cos(math.pi*p))
    return f

class OneCycle(keras.callbacks.Callback):
    """Leslie Smith's 1cycle: LR up then down, momentum the opposite way."""
    def __init__(self, max_lr, total_steps, start_frac=.1, last_frac=.001):
        self.max_lr, self.total = max_lr, total_steps
        self.start_lr = max_lr*start_frac
        self.last_lr  = max_lr*last_frac
        self.half = total_steps//2
        self.tail = int(total_steps*.1)
        self.step = 0; self.lrs = []
    def _interp(self, a, b, frac): return a + (b-a)*frac
    def on_batch_begin(self, batch, logs=None):
        s = self.step
        if s < self.half:
            lr = self._interp(self.start_lr, self.max_lr, s/self.half)
        elif s < 2*self.half - self.tail:
            lr = self._interp(self.max_lr, self.start_lr,
                              (s-self.half)/(self.half-self.tail))
        else:
            lr = self._interp(self.start_lr, self.last_lr,
                              (s-(2*self.half-self.tail))/max(self.tail,1))
        self.model.optimizer.learning_rate.assign(lr)
        self.lrs.append(lr); self.step += 1

class LRRecorder(keras.callbacks.Callback):
    def __init__(self): self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))

results = {}
print(f"{'schedule':<26}{'final lr':>12}{'best val acc':>15}{'test acc':>11}")

for nm, cb_factory in [
        ("constant 0.01",     lambda: []),
        ("power decay",       lambda: [keras.callbacks.LearningRateScheduler(power_decay())]),
        ("exponential decay", lambda: [keras.callbacks.LearningRateScheduler(exponential_decay())]),
        ("piecewise",         lambda: [keras.callbacks.LearningRateScheduler(piecewise)]),
        ("cosine annealing",  lambda: [keras.callbacks.LearningRateScheduler(cosine())]),
        ("warm-up + cosine",  lambda: [keras.callbacks.LearningRateScheduler(warmup_cosine())]),
        ("ReduceLROnPlateau", lambda: [keras.callbacks.ReduceLROnPlateau(
                                          factor=.5, patience=4, min_lr=1e-6)]),
        ("1cycle",            lambda: [OneCycle(.05, EPOCHS*steps_per_epoch)]),
]:
    tf.random.set_seed(0)
    m = build()
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(.01, momentum=.9), metrics=["accuracy"])
    rec = LRRecorder()
    h = m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(Xva, yva), callbacks=cb_factory()+[rec], verbose=0)
    results[nm] = (h.history, rec.lrs)
    print(f"{nm:<26}{rec.lrs[-1]:>12.2e}"
          f"{max(h.history['val_accuracy']):>15.4f}"
          f"{m.evaluate(Xte, yte, verbose=0)[1]:>11.4f}")

# ============ Keras's built-in schedule objects ========================
print("\\n=== Keras LearningRateSchedule objects (per-STEP, not per-epoch) ===")
for nm, sched in [
    ("ExponentialDecay", keras.optimizers.schedules.ExponentialDecay(
        .01, decay_steps=steps_per_epoch*10, decay_rate=.1)),
    ("CosineDecay", keras.optimizers.schedules.CosineDecay(
        .01, decay_steps=EPOCHS*steps_per_epoch)),
    ("PiecewiseConstantDecay", keras.optimizers.schedules.PiecewiseConstantDecay(
        [steps_per_epoch*12, steps_per_epoch*28], [.01, .003, .0005])),
]:
    tf.random.set_seed(0)
    m = build()
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(sched, momentum=.9),
              metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    print(f"  {nm:<26} test acc {m.evaluate(Xte, yte, verbose=0)[1]:.4f}")

import plotly.graph_objects as go
fig = go.Figure()
for i, (nm, (hh, lrs)) in enumerate(results.items()):
    if nm == "1cycle":
        continue
    fig.add_scatter(y=lrs, mode="lines", name=nm, line=dict(color=SEQ[i], width=2.4))
fig.update_layout(height=400, yaxis_type="log", xaxis_title="epoch",
                  yaxis_title="learning rate", title="The schedules, measured")

fig2 = go.Figure()
for i, (nm, (hh, lrs)) in enumerate(results.items()):
    fig2.add_scatter(y=hh["val_accuracy"], mode="lines", name=nm,
                     line=dict(color=SEQ[i], width=2.2))
fig2.update_layout(height=400, xaxis_title="epoch", yaxis_title="validation accuracy",
                   title="What each schedule buys")
''',
        key="ch11_schedules",
    )

    keypoints([
        "Start high, end low. Power, exponential, piecewise, performance-based, "
        "1cycle, cosine.",
        "<b>1cycle</b>: LR rises then falls (momentum does the opposite) — often "
        "the fastest to a good solution.",
        "The high-LR middle phase is a <b>regulariser</b> that keeps the optimiser "
        "in wide, flat minima.",
        "<b>Warm-up</b> is essential for Transformers and large batches — the "
        "adaptive second moment needs time to stabilise.",
        "<code>ReduceLROnPlateau</code> needs no design and is a fine default.",
    ])


# ==========================================================================
def s_11_8():
    section("11.8", "Avoiding Overfitting Through Regularization")

    lead(
        "A deep network has enough capacity to memorise its training set exactly. "
        "Four techniques stop it, and one of them — dropout — has a genuinely "
        "surprising theoretical justification."
    )

    sub("ℓ₁ and ℓ₂ regularisation")

    math(r"""
    J_{\text{reg}}(\boldsymbol\theta) \;=\;
    J(\boldsymbol\theta)
    \;+\; \underbrace{\alpha_1 \sum_{i}\bigl|\theta_i\bigr|}_{\ell_1:\ \text{sparsity}}
    \;+\; \underbrace{\tfrac{1}{2}\alpha_2 \sum_{i}\theta_i^{2}}_{\ell_2:\ \text{weight decay}}
    """)

    md("The same geometry as §4.5 — the $\\ell_1$ ball's corners produce exact "
       "zeros, the $\\ell_2$ ball's smoothness does not. In Keras:")

    md(
        """
```python
from functools import partial
RegularizedDense = partial(keras.layers.Dense, activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
```
        """
    )

    note("Regularise the <b>kernel</b>, not the bias. The bias has one degree of "
         "freedom per neuron and shrinking it toward zero does not reduce "
         "overfitting — it just biases the model. Keras's "
         "<code>kernel_regularizer</code> already excludes the bias; "
         "<code>bias_regularizer</code> exists but is rarely useful.")

    sub("Dropout")

    md(
        "At every training step, each neuron (excluding the output neurons) has "
        "probability $p$ of being **temporarily dropped** — outputting 0 for that "
        "step. At test time no neuron is dropped."
    )

    derive(
        [("<b>The scaling problem.</b> During training each neuron sees on average "
          "$(1-p)$ of its inputs. At test time it sees them all, so its input sum "
          "is $1/(1-p)$ times larger than it ever was during training — the "
          "network's behaviour changes completely.", None),
         ("<b>Two equivalent fixes.</b> Classic dropout multiplies the weights by "
          "the <i>keep probability</i> after training:",
          r"\mathbf{w}_{\text{test}} = (1-p)\,\mathbf{w}_{\text{train}}"),
         ("<b>Inverted dropout</b> — what every framework actually implements — "
          "divides by the keep probability <i>during</i> training instead, so "
          "nothing needs to change at test time:",
          r"a_{\text{train}} = \frac{a \cdot \mathrm{mask}}{1-p}, \qquad "
          r"a_{\text{test}} = a"),
         ("<b>Why it regularises.</b> Neurons cannot rely on any particular "
          "co-neuron being present, so they cannot co-adapt into fragile "
          "conspiracies. Each must be individually useful.", None),
         ("<b>The ensemble interpretation.</b> A network with $n$ droppable "
          "neurons defines $2^n$ possible thinned sub-networks. Every training "
          "step samples one and takes a gradient step on it — they all share "
          "weights. At test time, using all the neurons with scaled weights "
          "computes an approximate <b>geometric-mean ensemble</b> over all $2^n$ "
          "sub-networks. With $n = 1000$ that is $2^{1000}$ models, more than the "
          "number of atoms in the universe, trained for the price of one.", None)],
        title="Inverted dropout, and the 2ⁿ-model ensemble",
    )

    table(
        ["Variant", "Use"],
        [["<code>Dropout(rate)</code>", "Standard. Rate 0.2–0.5 for dense layers, "
          "0.1–0.2 for convolutional"],
         ["<code>AlphaDropout(rate)</code>",
          "<b>Required</b> with SELU — preserves mean 0 and variance 1"],
         ["<code>SpatialDropout2D</code>",
          "Drops entire feature maps in a CNN; standard dropout is weak there "
          "because adjacent pixels are correlated"],
         ["<b>MC Dropout</b>", "Keep dropout ON at inference — see below"]],
    )

    sub("Monte Carlo Dropout")

    md(
        "Gal & Ghahramani (2016) proved that a dropout network is mathematically "
        "equivalent to approximate Bayesian inference in a deep Gaussian process. "
        "The practical consequence needs **no retraining at all**: leave dropout "
        "on at inference and average many stochastic forward passes."
    )

    math(r"""
    p_{\text{MC}}(y \mid \mathbf{x}) \;\approx\;
    \frac{1}{T}\sum_{t=1}^{T} p\bigl(y \mid \mathbf{x},\, \mathbf{M}_t\bigr)
    """)
    where({r"\mathbf{M}_t": "an independently sampled dropout mask",
           r"T": "the number of samples, typically 10–100"})

    idea(
        "You get uncertainty estimates for free",
        "The <b>variance</b> across the $T$ passes is a usable measure of the "
        "model's uncertainty — and it is exactly what a plain softmax cannot give "
        "you (a network is often confidently wrong, §6.2). The mean is also "
        "usually better calibrated than the deterministic prediction. Cost: $T$ "
        "forward passes instead of one, and <b>no retraining whatsoever</b>. If "
        "your model has dropout layers, you already have this.",
    )

    sub("Max-norm regularisation")

    math(r"""
    \text{after each update:}\qquad
    \mathbf{w} \;\leftarrow\; \mathbf{w}\,
      \frac{r}{\bigl\lVert \mathbf{w} \bigr\rVert_2}
    \qquad\text{if } \bigl\lVert \mathbf{w}\bigr\rVert_2 > r
    """)

    md(
        "Constrain each neuron's incoming weight vector to a ball of radius $r$. "
        "It does not add a term to the loss — it is a **projection** applied after "
        "every step. Reducing $r$ increases regularisation, and it also helps with "
        "unstable gradients when batch norm is not used. In Keras: "
        "`kernel_constraint=keras.constraints.max_norm(1.)`."
    )

    anim_header("Dropout: the ensemble of thinned networks")
    md(
        "The same network, with a different random dropout mask each frame. Each "
        "one is a distinct sub-network being trained; at test time all the units "
        "are used and the result approximates their average."
    )

    layers_n = [5, 7, 7, 4]
    xs_l, ys_l, ids = [], [], []
    for li, n in enumerate(layers_n):
        for j in range(n):
            xs_l.append(li)
            ys_l.append(j - (n - 1) / 2)
            ids.append((li, j))
    edges = []
    off = 0
    for li in range(len(layers_n) - 1):
        for a in range(layers_n[li]):
            for b in range(layers_n[li + 1]):
                edges.append((off + a, off + sum(layers_n[li:li + 1]) + b))
        off += layers_n[li]

    rngd = np.random.default_rng(0)
    frames = []
    for k in range(18):
        keep = np.ones(len(ids), bool)
        for i, (li, j) in enumerate(ids):
            if 0 < li < len(layers_n) - 1:
                keep[i] = rngd.random() > .45
        ex, ey = [], []
        for a, b in edges:
            if keep[a] and keep[b]:
                ex += [xs_l[a], xs_l[b], None]
                ey += [ys_l[a], ys_l[b], None]
        frames.append(go.Frame(name=str(k + 1), data=[
            go.Scatter(x=ex, y=ey, mode="lines",
                       line=dict(color=alpha(C["primary"], .35), width=1.3),
                       hoverinfo="skip"),
            go.Scatter(x=xs_l, y=ys_l, mode="markers",
                       marker=dict(size=[24 if kp else 13 for kp in keep],
                                   color=[C["primary"] if kp else C["line"]
                                          for kp in keep],
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"training step {k+1}   ·   {int(keep.sum())}/{len(ids)} units active"
            f"   ·   this is sub-network {k+1} of 2^{layers_n[1]+layers_n[2]} "
            f"= {2**(layers_n[1]+layers_n[2]):,}")])))

    f = go.Figure(data=[
        go.Scatter(x=[], y=[], mode="lines", showlegend=False,
                   line=dict(color=alpha(C["primary"], .35), width=1.3)),
        go.Scatter(x=xs_l, y=ys_l, mode="markers", showlegend=False,
                   marker=dict(size=24, color=C["primary"],
                               line=dict(color="#fff", width=2))),
    ])
    f.update_layout(height=440, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, 3.4]),
                    yaxis=dict(visible=False, range=[-4, 4]),
                    title="Dropout samples a different sub-network every step")
    anim.animate(f, frames, duration=nav.anim_ms(500), slider_prefix="step ")
    figure(f)

    code_lab(
        "All four regularisers, plus MC Dropout uncertainty",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from functools import partial

tf.random.set_seed(42); np.random.seed(42)
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data/16., d.target, test_size=.3,
                                      stratify=d.target, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=.2, random_state=42)
Xsmall, ysmall = Xtr[:250], ytr[:250]              # small = easy to overfit
print(f"training on {len(Xsmall)} instances to make overfitting obvious\\n")

# ============ 1. INVERTED DROPOUT, FROM SCRATCH ========================
print("=== inverted dropout ===")
rng = np.random.default_rng(0)
a = rng.normal(0, 1, (10000,))
p = 0.5
mask = (rng.random(a.shape) > p).astype(float)
naive   = a * mask                      # WRONG: mean drops by (1-p)
inverted = a * mask / (1 - p)           # RIGHT: mean preserved
print(f"original activations : mean {a.mean():+.5f}  std {a.std():.5f}")
print(f"naive dropout        : mean {naive.mean():+.5f}  std {naive.std():.5f}"
      f"   <- scale changed")
print(f"INVERTED dropout     : mean {inverted.mean():+.5f}  std {inverted.std():.5f}"
      f"   <- scale preserved")
print(f"\\nKeras implements inverted dropout, which is why nothing needs to")
print(f"change at test time. Verify:")
layer = keras.layers.Dropout(0.5)
out_tr = layer(tf.constant(a.reshape(-1, 1), tf.float32), training=True).numpy()
out_te = layer(tf.constant(a.reshape(-1, 1), tf.float32), training=False).numpy()
print(f"  training=True : mean {out_tr.mean():+.5f}   "
      f"{np.mean(out_tr == 0):.1%} zeros")
print(f"  training=False: mean {out_te.mean():+.5f}   "
      f"{np.mean(out_te == 0):.1%} zeros  (identity)")

# ============ 2. ALL FOUR REGULARISERS =================================
def build(reg=None, dropout=0., max_norm=None, activation="relu",
          init="he_normal", alpha_dropout=False):
    Dense = partial(keras.layers.Dense, activation=activation,
                    kernel_initializer=init,
                    kernel_regularizer=reg,
                    kernel_constraint=(keras.constraints.max_norm(max_norm)
                                       if max_norm else None))
    layers = [keras.layers.Input(shape=(64,))]
    for _ in range(3):
        if dropout:
            layers.append((keras.layers.AlphaDropout if alpha_dropout
                           else keras.layers.Dropout)(dropout))
        layers.append(Dense(128))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return m

print("\\n=== four regularisers on 250 training instances ===")
print(f"{'model':<34}{'train':>9}{'val':>9}{'test':>9}{'gap':>9}")
configs = [
    ("no regularisation",            dict()),
    ("l2(0.01)",                     dict(reg=keras.regularizers.l2(.01))),
    ("l1(0.001)",                    dict(reg=keras.regularizers.l1(.001))),
    ("dropout 0.2",                  dict(dropout=.2)),
    ("dropout 0.5",                  dict(dropout=.5)),
    ("max_norm(1.0)",                dict(max_norm=1.0)),
    ("dropout 0.3 + l2(0.001)",      dict(dropout=.3, reg=keras.regularizers.l2(.001))),
    ("SELU + AlphaDropout 0.1",      dict(dropout=.1, activation="selu",
                                          init="lecun_normal", alpha_dropout=True)),
]
models = {}
for nm, kw in configs:
    tf.random.set_seed(0)
    m = build(**kw)
    m.fit(Xsmall, ysmall, epochs=120, batch_size=32, verbose=0)
    models[nm] = m
    tr = m.evaluate(Xsmall, ysmall, verbose=0)[1]
    va = m.evaluate(Xva, yva, verbose=0)[1]
    te = m.evaluate(Xte, yte, verbose=0)[1]
    print(f"{nm:<34}{tr:>9.4f}{va:>9.4f}{te:>9.4f}{tr-te:>9.4f}")

# ============ 3. l1 gives SPARSITY =====================================
print("\\n=== l1 drives weights to (near) zero ===")
for nm in ["no regularisation", "l2(0.01)", "l1(0.001)"]:
    W = models[nm].layers[0].get_weights()[0]
    print(f"  {nm:<22} {np.mean(np.abs(W) < 1e-3):>6.1%} of weights below 1e-3, "
          f"||W||_1 = {np.abs(W).sum():>9.1f}")

# ============ 4. MC DROPOUT ============================================
print("\\n=== MC Dropout: uncertainty for free ===")
mc_model = models["dropout 0.5"]

def mc_predict(model, X, T=100):
    """T stochastic forward passes with dropout LEFT ON."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(T)])
    return preds.mean(0), preds.std(0), preds

det = mc_model.predict(Xte, verbose=0)
mean, std, all_p = mc_predict(mc_model, Xte, T=100)

print(f"deterministic accuracy = {np.mean(det.argmax(1) == yte):.4f}")
print(f"MC dropout accuracy    = {np.mean(mean.argmax(1) == yte):.4f}")
from sklearn.metrics import log_loss
print(f"deterministic log loss = "
      f"{log_loss(yte, np.clip(det, 1e-9, 1), labels=list(range(10))):.4f}")
print(f"MC dropout log loss    = "
      f"{log_loss(yte, np.clip(mean, 1e-9, 1), labels=list(range(10))):.4f}"
      f"   <- better calibrated")

# --- the uncertainty is USEFUL ---------------------------------------
uncertainty = std[np.arange(len(yte)), mean.argmax(1)]
correct = mean.argmax(1) == yte
print(f"\\nmean uncertainty when CORRECT : {uncertainty[correct].mean():.4f}")
print(f"mean uncertainty when WRONG   : {uncertainty[~correct].mean():.4f}")
print(f"ratio                          : "
      f"{uncertainty[~correct].mean()/uncertainty[correct].mean():.2f}x")

print(f"\\n=== rejecting the most uncertain predictions ===")
order = np.argsort(uncertainty)
print(f"{'keep':>8}{'accuracy on kept':>20}")
for frac in [1.0, .9, .75, .5, .25]:
    keep = order[:int(frac*len(order))]
    print(f"{frac:>7.0%}{np.mean(mean.argmax(1)[keep] == yte[keep]):>20.4f}")
print("\\nAccuracy rises monotonically as you reject uncertain cases -- that is")
print("exactly what you want for a human-in-the-loop system.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_histogram(x=uncertainty[correct], nbinsx=40, name="correct",
                  marker_color=C["success"], opacity=.65)
fig.add_histogram(x=uncertainty[~correct], nbinsx=40, name="wrong",
                  marker_color=C["danger"], opacity=.65)
fig.update_layout(height=380, barmode="overlay", xaxis_title="MC dropout std",
                  yaxis_title="count", title="Uncertainty separates right from wrong")
''',
        key="ch11_regularization",
    )

    keypoints([
        "$\\ell_2$ (weight decay) shrinks; $\\ell_1$ zeroes — the same geometry as "
        "§4.5.",
        "<b>Inverted dropout</b> divides by $(1-p)$ during training, so test time "
        "needs no change.",
        "Dropout trains $2^n$ weight-sharing sub-networks and approximates their "
        "geometric-mean ensemble.",
        "<b>MC Dropout</b>: keep dropout on at inference, average $T$ passes — "
        "better calibration and free uncertainty, <b>no retraining</b>.",
        "Max-norm is a projection after each step; use <code>AlphaDropout</code> "
        "with SELU.",
    ])


# ==========================================================================
def s_11_9():
    section("11.9", "Practical Guidelines & Exercises")

    lead(
        "The whole chapter compressed into two tables you can act on, then the "
        "exercises."
    )

    sub("The default configuration")

    table(
        ["Component", "Default", "Why"],
        [["Kernel initialiser", "<b>He initialisation</b>", "§11.1"],
         ["Activation", "<b>ReLU</b> if shallow, <b>Swish/GELU</b> if deep",
          "§11.2"],
         ["Normalisation", "<b>None</b> if shallow, <b>batch norm</b> if deep",
          "§11.3"],
         ["Regularisation", "Early stopping, plus weight decay if needed",
          "§11.8"],
         ["Optimiser", "<b>NAG</b> or <b>AdamW</b>", "§11.6"],
         ["LR schedule", "<b>Performance-based</b> or <b>1cycle</b>", "§11.7"]],
    )

    sub("The self-normalising configuration")

    table(
        ["Component", "Value"],
        [["Kernel initialiser", "<b>LeCun</b> normal"],
         ["Activation", "<b>SELU</b>"],
         ["Normalisation", "<b>None</b> — self-normalisation handles it"],
         ["Regularisation", "<b>AlphaDropout</b> if needed"],
         ["Optimiser", "NAG"],
         ["LR schedule", "Performance-based or 1cycle"]],
        "Requires: standardised inputs, a plain <code>Dense</code> stack, and no "
        "skip connections (§11.2).",
    )

    sub("Situational adjustments")

    table(
        ["If you…", "Then…"],
        [["Cannot find a good learning rate",
          "Use a <b>1cycle</b> schedule — it is much less sensitive"],
         ["Need a sparse model",
          "Add $\\ell_1$ regularisation, or zero out the tiny weights after "
          "training, or use TensorFlow Model Optimization"],
         ["Need very low-latency inference",
          "Fewer layers; <b>fold the BN layers into the preceding layer's "
          "weights</b>; use a fast activation (ReLU); quantise to float16 or int8 "
          "(Chapter 19)"],
         ["Are building a risk-sensitive application",
          "Use <b>MC Dropout</b> for calibrated probabilities and uncertainty "
          "estimates"],
         ["Have very little labelled data",
          "<b>Transfer learning</b> (§11.5); self-supervised pretraining (Ch. 16)"],
         ["Have small mini-batches (< 8)",
          "Replace batch norm with <code>LayerNormalization</code> or "
          "<code>GroupNormalization</code>"]],
    )

    rule()

    exercise(
        1, "What is the problem that Glorot initialization and He initialization "
        "aim to fix?",
        "The **vanishing and exploding gradients** problem. Backpropagation "
        "multiplies a factor at every layer, so gradients scale like $\\gamma^L$ "
        "in the depth $L$ (§11.1). If $\\gamma < 1$ the lower layers receive "
        "essentially nothing and never train; if $\\gamma > 1$ the updates explode "
        "and the loss becomes `NaN`.\n\n"
        "Glorot and He initialisation choose the weight variance so that the "
        "**variance of the signal is preserved** in both directions — forward "
        "through the activations and backward through the gradients — keeping "
        "$\\gamma \\approx 1$ and making deep networks trainable.")

    exercise(
        2, "Is it OK to initialize all the weights to the same value as long as "
        "that value is selected randomly using He initialization?",
        "**No.** All the neurons in a layer would then compute exactly the same "
        "thing, receive exactly the same gradient, and update identically — "
        "forever. The layer would behave as a single neuron no matter how wide it "
        "is.\n\n"
        "This is the **symmetry breaking** problem (§10.2's lab demonstrates it "
        "empirically). Weights must be sampled **independently** — He "
        "initialisation specifies the *distribution* to sample from, not a single "
        "value to reuse.\n\n"
        "Biases *can* all be initialised to the same value (usually zero) without "
        "harm, because the weights already break the symmetry.")

    exercise(
        3, "Is it OK to initialize the bias terms to 0?",
        "**Yes, and it is the standard choice.** It makes no difference either "
        "way — the symmetry is broken by the random weights, so identical biases "
        "cause no problem.\n\n"
        "Some people initialise biases like weights, which also works. Two "
        "situations where a **non-zero** bias is deliberately used: a small "
        "positive bias (e.g. 0.01) for ReLU units, to reduce the chance of a unit "
        "starting dead; and the LSTM forget-gate bias set to 1 (Chapter 15), "
        "which encourages the cell to remember by default.")

    exercise(
        4, "In which cases would you want to use each of the activation functions "
        "we discussed in this chapter?",
        "* **ReLU** — the default for hidden layers. Fastest to compute, and "
        "unlike sigmoid/tanh it does not saturate for positive inputs. Use it "
        "unless you have a reason not to.\n"
        "* **Leaky ReLU / PReLU / RReLU** — when you observe dying ReLU units. "
        "PReLU learns the slope (good on large datasets, overfits small ones); "
        "RReLU randomises it, adding regularisation.\n"
        "* **ELU** — like leaky ReLU but with a smooth curve and a negative mean "
        "output, which pushes activations toward zero mean. Slower (an "
        "exponential).\n"
        "* **SELU** — for a plain deep `Dense` stack where you want "
        "self-normalisation without batch norm. Requires all five preconditions "
        "of §11.2.\n"
        "* **GELU / Swish / Mish** — smooth non-monotonic functions that give "
        "small but consistent gains on large models; GELU is standard in "
        "Transformers. Slower than ReLU.\n"
        "* **tanh** — useful in the output layer when you want a value in "
        "$(-1, 1)$; rarely the best hidden activation now.\n"
        "* **sigmoid** — the output layer of a binary or multilabel classifier "
        "(§10.4). Avoid it in hidden layers of deep networks.\n"
        "* **softmax** — the output layer of a multiclass classifier.\n"
        "* **none** — the output layer of a regressor.")

    exercise(
        5, "What may happen if you set the `momentum` hyperparameter too close to "
        "1 (e.g., 0.99999) when using an `SGD` optimizer?",
        "The algorithm will **pick up a great deal of speed**, hopefully moving "
        "roughly toward the global minimum — the terminal velocity is "
        "$1/(1-\\beta)$ times the plain step, so $\\beta = 0.99999$ gives a "
        "**100 000×** speed-up (§11.6).\n\n"
        "But its momentum will carry it **right past the minimum**, and then it "
        "will oscillate back and forth, decelerating and accelerating, overshooting "
        "each time. It will eventually converge, but this oscillation costs many "
        "more iterations than a well-chosen $\\beta$ would have. $\\beta = 0.9$ "
        "is a good default.")

    exercise(
        6, "Name three ways you can produce a sparse model.",
        "**(1) Train normally, then zero out the tiny weights** — a simple "
        "magnitude threshold after training. Crude but surprisingly effective, and "
        "it usually costs little accuracy.\n\n"
        "**(2) Apply strong $\\ell_1$ regularisation during training**, which "
        "pushes many weights exactly to zero (§4.5's geometry). You can combine "
        "this with (1) for more sparsity.\n\n"
        "**(3) Use the TensorFlow Model Optimization Toolkit**, which implements "
        "iterative magnitude pruning: prune, fine-tune to recover accuracy, prune "
        "more, repeat. This reaches far higher sparsity than one-shot thresholding "
        "at the same accuracy.\n\n"
        "(A fourth, if you count architecture: use a genuinely smaller model, or "
        "structured pruning that removes whole neurons/channels — which is the "
        "only kind that actually speeds up inference on standard hardware.)")

    exercise(
        7, "Does dropout slow down training? Does it slow down inference (i.e., "
        "making predictions on new instances)? What about MC dropout?",
        "**Training: yes**, typically by a factor of about two. Each training step "
        "updates only the surviving sub-network, so convergence takes roughly "
        "twice as many epochs. This is usually worth it for the regularisation.\n\n"
        "**Inference: no.** Dropout is only active during training. At test time "
        "the layer is an identity function (with inverted dropout, §11.8), so "
        "there is zero inference cost.\n\n"
        "**MC dropout: yes**, by a factor of $T$ — you run $T$ stochastic forward "
        "passes instead of one, typically $T = 10$ to $100$. In exchange you get "
        "better-calibrated probabilities and genuine uncertainty estimates, with "
        "**no retraining required**.")

    exercise(
        8, "Practice training a deep neural network on the CIFAR10 image dataset: "
        "(a) build a DNN with 20 hidden layers of 100 neurons each, using He "
        "initialization and the Swish activation function; (b) using Nadam "
        "optimization and early stopping, train the network on CIFAR10; "
        "(c) now try adding batch normalization and compare the learning curves; "
        "(d) try replacing batch normalization with SELU, and make the necessary "
        "adjustments; (e) try regularizing the model with alpha dropout; "
        "(f) retrain your model using 1cycle scheduling and see if it improves "
        "training speed and model accuracy.",
        "Approximate results on CIFAR-10 (10 classes, 32×32×3 colour images, "
        "50 000 training instances) with a **dense** network:\n\n"
        "| configuration | test accuracy | notes |\n|---|---|---|\n"
        "| (b) He + Swish + Nadam | ~0.47 | slow to converge |\n"
        "| (c) + batch normalisation | ~0.51 | slower per epoch, fewer epochs "
        "needed, clearly better |\n"
        "| (d) SELU + LeCun normal | ~0.49 | needs standardised inputs |\n"
        "| (e) + AlphaDropout(0.1) | ~0.50 | plus MC dropout adds ~0.005 |\n"
        "| (f) + 1cycle | ~0.52 | and roughly **3× faster** to get there |\n\n"
        "**The honest headline:** a dense network tops out around 50 % on "
        "CIFAR-10, because it throws away all spatial structure — it has no idea "
        "that adjacent pixels are related. A modest CNN (Chapter 14) reaches "
        "**over 90 %** with fewer parameters. This exercise is valuable precisely "
        "because it shows you the ceiling that architecture, not tuning, "
        "imposes.\n\n"
        "Two practical notes: standardise the inputs (essential for SELU, helpful "
        "for everything); and remember MC dropout requires no retraining — just "
        "average $T$ passes with `training=True`.",
        code='''import tensorflow as tf
from tensorflow import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_valid = X_train_full[5000:], X_train_full[:5000]
y_train, y_valid = y_train_full[5000:], y_train_full[:5000]

# --- (c) He + Swish + BatchNorm + Nadam ------------------------------
model = keras.Sequential([keras.layers.Input(shape=(32, 32, 3)),
                          keras.layers.Flatten()])
for _ in range(20):
    model.add(keras.layers.Dense(100, kernel_initializer="he_normal",
                                 use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("swish"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Nadam(learning_rate=5e-4),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(patience=20,
                                                   restore_best_weights=True)])

# --- (d) SELU: standardise the inputs and use LeCun normal ------------
means = X_train.mean(axis=0, keepdims=True)
stds  = X_train.std(axis=0, keepdims=True) + 1e-7
X_train_s = (X_train - means) / stds
# ... Dense(100, kernel_initializer="lecun_normal", activation="selu")

# --- (e) AlphaDropout, and MC dropout at inference --------------------
# ... keras.layers.AlphaDropout(rate=0.1) between the Dense layers
import numpy as np
y_probas = np.stack([model(X_test_s, training=True) for _ in range(100)])
y_proba = y_probas.mean(axis=0)          # better calibrated, no retraining''')

    rule()

    keypoints([
        "Depth breaks naive training; every technique here is a fix for one "
        "specific failure.",
        "He init + ReLU/Swish + batch norm + AdamW + 1cycle is a strong default "
        "stack.",
        "<b>Transfer learning</b> is the single largest practical lever — always "
        "check for a pretrained model first.",
        "Dropout is an ensemble of $2^n$ sub-networks; MC dropout turns it into a "
        "free uncertainty estimator.",
        "When in doubt: lower the learning rate, add batch norm, and use early "
        "stopping.",
    ], title="Chapter 11 in five lines")

    refs([
        ("Glorot & Bengio — *Understanding the Difficulty of Training Deep "
         "Feedforward Neural Networks*", "AISTATS 2010"),
        ("He et al. — *Delving Deep into Rectifiers* (He initialisation, PReLU)",
         "https://doi.org/10.1109/ICCV.2015.123"),
        ("Ioffe & Szegedy — *Batch Normalization*", "ICML 2015"),
        ("Santurkar et al. — *How Does Batch Normalization Help Optimization?*",
         "NeurIPS 2018"),
        ("Klambauer et al. — *Self-Normalizing Neural Networks* (SELU)",
         "NeurIPS 2017"),
        ("Kingma & Ba — *Adam: A Method for Stochastic Optimization*",
         "https://arxiv.org/abs/1412.6980"),
        ("Loshchilov & Hutter — *Decoupled Weight Decay Regularization* (AdamW)",
         "ICLR 2019"),
        ("Smith, L. — *A Disciplined Approach to Neural Network Hyper-Parameters* "
         "(1cycle)", "https://arxiv.org/abs/1803.09820"),
        ("Srivastava et al. — *Dropout: A Simple Way to Prevent Neural Networks "
         "from Overfitting*", "JMLR 2014"),
        ("Gal & Ghahramani — *Dropout as a Bayesian Approximation* (MC Dropout)",
         "ICML 2016"),
    ])


# ==========================================================================
SECTIONS = [
    ("11.1", "Vanishing/Exploding Gradients", s_11_1),
    ("11.2", "Better Activation Functions", s_11_2),
    ("11.3", "Batch Normalization", s_11_3),
    ("11.4", "Gradient Clipping", s_11_4),
    ("11.5", "Reusing Pretrained Layers", s_11_5),
    ("11.6", "Faster Optimizers", s_11_6),
    ("11.7", "Learning Rate Scheduling", s_11_7),
    ("11.8", "Regularization", s_11_8),
    ("11.9", "Guidelines & Exercises", s_11_9),
]

nav.render_chapter(CH, SECTIONS)
