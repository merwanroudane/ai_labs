"""Chapter 17 — Autoencoders, GANs and Diffusion Models."""

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
CH = "ch17"

hero(
    kicker="Part II · Chapter 17",
    title="Autoencoders, GANs and Diffusion Models",
    blurb=(
        "Three routes to generative modelling, and the trade-off that separates "
        "them. Autoencoders compress and reconstruct; VAEs add a probabilistic "
        "latent space with the ELBO derived from Jensen's inequality; GANs "
        "replace the likelihood with an adversary; diffusion models learn to "
        "reverse a noising process — and won."
    ),
    chips=["ELBO derived", "9 sub-sections", "9 animations",
           "9 code labs", "VAE · GAN · diffusion"],
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
def s_17_1():
    section("17.1", "Efficient Representations and the Undercomplete Autoencoder")

    lead(
        "An autoencoder learns to copy its input to its output through a "
        "deliberately narrow channel. The copying is not the point — the "
        "<b>constraint</b> is."
    )

    sub("The architecture")

    md(
        "An autoencoder is two networks back to back. The **encoder** "
        "$f_\\phi$ maps an input to a **latent code** (or *coding*); the "
        "**decoder** $g_\\theta$ maps it back."
    )

    math(r"""
    \mathbf{z} = f_\phi(\mathbf{x}) \in \mathbb{R}^{d},
    \qquad
    \hat{\mathbf{x}} = g_\theta(\mathbf{z}) \in \mathbb{R}^{n},
    \qquad
    d \ll n
    """)

    math(r"""
    \mathcal{L}_{\text{rec}}(\phi, \theta)
    \;=\; \mathbb{E}_{\mathbf{x}}\bigl\lVert \mathbf{x} -
      g_\theta\bigl(f_\phi(\mathbf{x})\bigr) \bigr\rVert^{2}
    """)

    idea(
        "The bottleneck is the entire mechanism",
        "If $d \\ge n$ and the layers are unconstrained, the network can learn "
        "the identity function and the loss goes to zero having learned nothing. "
        "An <b>undercomplete</b> autoencoder ($d < n$) physically cannot do that: "
        "it must decide which $d$ numbers are worth keeping. Everything else in "
        "this chapter is a different way of imposing a constraint — sparsity "
        "(§17.6), noise (§17.5), a prior on the latent distribution (§17.7).",
    )

    sub("A linear autoencoder is PCA")

    derive(
        [("Take an autoencoder with <b>no activation functions</b>, no biases, "
          "and squared reconstruction loss:",
          r"\hat{\mathbf{x}} = \mathbf{W}_2\mathbf{W}_1\mathbf{x},"
          r"\qquad \mathbf{W}_1 \in \mathbb{R}^{d\times n},\;"
          r"\mathbf{W}_2 \in \mathbb{R}^{n\times d}"),
         ("The product $\\mathbf{W}_2\\mathbf{W}_1$ is an $n \\times n$ matrix of "
          "rank at most $d$. So we are minimising:",
          r"\min_{\mathrm{rank}(\mathbf{M}) \le d}\;\;"
          r"\mathbb{E}\bigl\lVert \mathbf{x} - \mathbf{M}\mathbf{x} \bigr\rVert^{2}"),
         ("This is exactly the problem the <b>Eckart–Young–Mirsky theorem</b> "
          "solves: the best rank-$d$ approximation in the Frobenius norm is the "
          "truncated SVD. For centred data, that is:",
          r"\mathbf{M}^{\star} = \mathbf{U}_d\mathbf{U}_d^{\top}"),
         ("where $\\mathbf{U}_d$ holds the top $d$ eigenvectors of the covariance "
          "matrix — <b>the principal components</b> (§8.3).", None),
         ("<b>But the autoencoder does not recover the components themselves.</b> "
          "For any invertible $\\mathbf{A} \\in \\mathbb{R}^{d \\times d}$, "
          "$(\\mathbf{W}_2\\mathbf{A}^{-1})(\\mathbf{A}\\mathbf{W}_1)$ gives the "
          "identical product. So it finds the right <b>subspace</b>, in an "
          "arbitrary, non-orthogonal, unordered basis.", None),
         ("<b>Add non-linear activations and the map is no longer restricted to a "
          "linear subspace</b> — the autoencoder can learn a curved manifold. "
          "That is precisely the generalisation of PCA to a non-linear "
          "projection, and it is why it is worth more than PCA on real data.",
          None)],
        title="A linear autoencoder spans the PCA subspace",
    )

    warn(
        "The reconstruction loss is not a measure of representation quality",
        "A model can reconstruct beautifully and learn a useless latent space — "
        "for instance by encoding a near-lossless hash of the input. Judge the "
        "codings by what you actually want from them: downstream accuracy, "
        "cluster structure, smoothness under interpolation. This is exactly the "
        "same warning as §9.4's on inertia.",
    )

    anim_header("An undercomplete autoencoder finding the principal subspace")

    rng = np.random.default_rng(0)
    n_pts = 260
    t = rng.normal(0, 1, n_pts)
    Xd = np.column_stack([t * 2.0, t * 0.9, np.zeros(n_pts)])
    Xd += rng.normal(0, .28, Xd.shape)
    Xd[:, 2] += rng.normal(0, .12, n_pts)
    Xd -= Xd.mean(0)

    # simulate training: a direction rotating toward PC1
    U, S, Vt = np.linalg.svd(Xd, full_matrices=False)
    pc1 = Vt[0]
    start = np.array([0.1, -0.99, 0.05]); start /= np.linalg.norm(start)
    steps = 34
    frames = []
    for k in range(steps + 1):
        a = k / steps
        w = (1 - a) * start + a * pc1
        w = w / np.linalg.norm(w)
        proj = np.outer(Xd @ w, w)
        err = float(np.mean(np.sum((Xd - proj) ** 2, 1)))
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter3d(x=Xd[:, 0], y=Xd[:, 1], z=Xd[:, 2], mode="markers",
                         marker=dict(size=3, color=C["muted"], opacity=.45)),
            go.Scatter3d(x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                         mode="markers",
                         marker=dict(size=3.5, color=C["primary"])),
            go.Scatter3d(x=[-4*w[0], 4*w[0]], y=[-4*w[1], 4*w[1]],
                         z=[-4*w[2], 4*w[2]], mode="lines",
                         line=dict(color=C["danger"], width=7)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"training step {k}   ·   reconstruction MSE = {err:.4f}   ·   "
            f"alignment with PC1 = {abs(float(w @ pc1)):.4f}")])))

    proj0 = np.outer(Xd @ start, start)
    f = go.Figure(data=[
        go.Scatter3d(x=Xd[:, 0], y=Xd[:, 1], z=Xd[:, 2], mode="markers",
                     name="data",
                     marker=dict(size=3, color=C["muted"], opacity=.45)),
        go.Scatter3d(x=proj0[:, 0], y=proj0[:, 1], z=proj0[:, 2],
                     mode="markers", name="reconstruction",
                     marker=dict(size=3.5, color=C["primary"])),
        go.Scatter3d(x=[-4*start[0], 4*start[0]], y=[-4*start[1], 4*start[1]],
                     z=[-4*start[2], 4*start[2]], mode="lines",
                     name="learned direction",
                     line=dict(color=C["danger"], width=7)),
    ])
    f.update_layout(height=520, title="1-D bottleneck on 3-D data",
                    scene=dict(aspectmode="data",
                               xaxis_title="x₁", yaxis_title="x₂",
                               zaxis_title="x₃"),
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(130), slider_prefix="step ")
    figure(f, "Minimising reconstruction error rotates the bottleneck direction "
              "onto the first principal component — the Eckart–Young result.")

    code_lab(
        "A linear autoencoder is PCA — verified numerically",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. SYNTHETIC 3-D DATA ON A 2-D PLANE =====================
rng = np.random.default_rng(0)
n = 800
t1, t2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
X = np.column_stack([t1*2.0 + t2*0.3, t1*0.5 + t2*1.4, t1*0.1 - t2*0.2])
X += rng.normal(0, .06, X.shape)
X = (X - X.mean(0)).astype("float32")
print("=== 3-D data that lives near a 2-D plane ===")
print(f"  {X.shape}, singular values {np.linalg.svd(X, compute_uv=False).round(2)}")

# ============ 2. A LINEAR AUTOENCODER ==================================
tf.random.set_seed(0)
encoder = keras.Sequential([keras.layers.Input(shape=(3,)),
                            keras.layers.Dense(2, use_bias=False)])
decoder = keras.Sequential([keras.layers.Input(shape=(2,)),
                            keras.layers.Dense(3, use_bias=False)])
ae = keras.Sequential([encoder, decoder])
ae.compile(loss="mse", optimizer=keras.optimizers.Adam(2e-2))
hist = ae.fit(X, X, epochs=200, batch_size=64, verbose=0)   # TARGET = INPUT
print()
print("=== a linear autoencoder, no activations, no biases ===")
print(f"  final reconstruction MSE {hist.history['loss'][-1]:.6f}")

# ============ 3. COMPARE WITH PCA ======================================
pca = PCA(n_components=2).fit(X)
X_pca = pca.inverse_transform(pca.transform(X))
X_ae = ae.predict(X, verbose=0)
print()
print("=== the two reconstructions ===")
print(f"  PCA         MSE {np.mean((X - X_pca)**2):.6f}")
print(f"  autoencoder MSE {np.mean((X - X_ae)**2):.6f}")
print(f"  reconstructions agree to "
      f"{np.abs(X_pca - X_ae).max():.5f} (max abs difference)")

# --- the SUBSPACE is the same; the BASIS is not ----------------------
W1 = encoder.layers[0].get_weights()[0]        # (3, 2)
print()
print("=== same subspace, different basis ===")
print(f"  PCA components (rows are orthonormal):")
print(f"    {np.round(pca.components_, 3)}")
print(f"  autoencoder encoder weights (columns):")
print(f"    {np.round(W1.T, 3)}")
q_ae, _ = np.linalg.qr(W1)
q_pca, _ = np.linalg.qr(pca.components_.T)
P_ae = q_ae @ q_ae.T
P_pca = q_pca @ q_pca.T
print(f"  projection matrices identical? "
      f"max |P_ae - P_pca| = {np.abs(P_ae - P_pca).max():.6f}")
print(f"  encoder columns orthogonal? "
      f"cos angle = {abs(W1[:,0] @ W1[:,1])/np.linalg.norm(W1[:,0])/np.linalg.norm(W1[:,1]):.4f}")
print("  the autoencoder finds the right SUBSPACE in an arbitrary basis --")
print("  W2 A^-1 and A W1 give the identical product for any invertible A.")

# ============ 4. THE BOTTLENECK MATTERS ================================
print()
print("=== reconstruction error vs bottleneck size ===")
print(f"{'code size d':>13}{'autoencoder MSE':>19}{'PCA MSE':>13}"
      f"{'variance kept':>16}")
for d in [1, 2, 3]:
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(3,)),
                          keras.layers.Dense(d, use_bias=False),
                          keras.layers.Dense(3, use_bias=False)])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(2e-2))
    m.fit(X, X, epochs=200, batch_size=64, verbose=0)
    p = PCA(n_components=d).fit(X)
    print(f"{d:>13}{m.evaluate(X, X, verbose=0):>19.6f}"
          f"{np.mean((X - p.inverse_transform(p.transform(X)))**2):>13.6f}"
          f"{p.explained_variance_ratio_.sum():>16.4f}")
print("  d=3 reconstructs perfectly and has learned NOTHING (the identity).")
print("  the CONSTRAINT is what does the work.")

# ============ 5. NON-LINEARITY BEATS PCA ON A CURVED MANIFOLD ==========
print()
print("="*62)
print("Where a non-linear autoencoder beats PCA")
print("="*62)
Xs, color = _ds.swiss_roll(n=1500)
Xs = ((Xs - Xs.mean(0)) / Xs.std(0)).astype("float32")
print(f"  Swiss roll {Xs.shape}: a 2-D manifold CURVED through 3-D")

print()
print(f"{'model':<38}{'reconstruction MSE':>21}")
p2 = PCA(n_components=2).fit(Xs)
print(f"{'PCA (2 components)':<38}"
      f"{np.mean((Xs - p2.inverse_transform(p2.transform(Xs)))**2):>21.5f}")

for nm, act in [("linear autoencoder (d=2)", None),
                ("NON-LINEAR autoencoder (d=2)", "selu")]:
    tf.random.set_seed(0)
    enc = keras.Sequential([keras.layers.Input(shape=(3,)),
                            keras.layers.Dense(64, activation=act),
                            keras.layers.Dense(32, activation=act),
                            keras.layers.Dense(2)])
    dec = keras.Sequential([keras.layers.Input(shape=(2,)),
                            keras.layers.Dense(32, activation=act),
                            keras.layers.Dense(64, activation=act),
                            keras.layers.Dense(3)])
    m = keras.Sequential([enc, dec])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    m.fit(Xs, Xs, epochs=90, batch_size=64, verbose=0)
    print(f"{nm:<38}{m.evaluate(Xs, Xs, verbose=0):>21.5f}")
    if act:
        good_enc = enc
print()
print("  a linear autoencoder can only project onto a FLAT plane.")
print("  with activations it can follow the curve -- that is the whole")
print("  reason autoencoders are worth more than PCA.")

# ============ 6. THE LEARNED 2-D COORDINATES ===========================
Z = good_enc.predict(Xs, verbose=0)
print()
print("=== does the 2-D code recover the manifold coordinate? ===")
from scipy.stats import spearmanr
r1 = abs(spearmanr(Z[:, 0], color).statistic)
r2 = abs(spearmanr(Z[:, 1], color).statistic)
print(f"  |Spearman(code dim 1, true position along the roll)| = {r1:.4f}")
print(f"  |Spearman(code dim 2, true position along the roll)| = {r2:.4f}")
print(f"  best axis: {max(r1, r2):.4f}   (1.0 = perfectly unrolled)")

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers",
                           marker=dict(size=4, color=color,
                                       colorscale="Viridis",
                                       colorbar=dict(title="position<br>on roll"))))
fig.update_layout(height=440, xaxis_title="code dimension 1",
                  yaxis_title="code dimension 2",
                  title="The autoencoder's learned 2-D coordinates")
''',
        key="ch17_linear",
    )

    keypoints([
        "An autoencoder reconstructs its input through a bottleneck; the "
        "<b>constraint</b>, not the copying, is the point.",
        "A <b>linear</b> autoencoder with squared loss spans the <b>PCA "
        "subspace</b> (Eckart–Young), in an arbitrary basis.",
        "Adding activations lets it follow a <b>curved manifold</b> — the reason "
        "it beats PCA.",
        "An overcomplete autoencoder with no constraint learns the identity and "
        "nothing else.",
        "Low reconstruction error does <b>not</b> imply a useful representation.",
    ])


# ==========================================================================
def s_17_2():
    section("17.2", "Stacked Autoencoders and Unsupervised Pretraining")

    lead(
        "Add hidden layers to both halves and the autoencoder can learn far "
        "richer codings — and can be used to bootstrap a supervised model when "
        "labels are scarce."
    )

    sub("Symmetry and tied weights")

    md(
        "A stacked autoencoder is usually built symmetrically around the "
        "bottleneck: `784 → 100 → 30 → 100 → 784`. Because the decoder mirrors "
        "the encoder, its weights can be **tied** to the encoder's transpose."
    )

    math(r"""
    \mathbf{W}_{\text{dec}}^{(\ell)} \;=\; \bigl(\mathbf{W}_{\text{enc}}^{(L-\ell+1)}\bigr)^{\!\top}
    """)

    table(
        ["", "Untied", "Tied"],
        [["Parameters", "$2\\times$", "$\\approx 1\\times$ (biases stay separate)"],
         ["Overfitting", "More prone", "<b>Regularised</b> by construction"],
         ["Training speed", "Baseline", "Faster (fewer gradients)"],
         ["Flexibility", "Higher", "Lower — decoder cannot deviate"],
         ["When", "Plenty of data", "<b>Small datasets</b>"]],
    )

    sub("Unsupervised pretraining")

    md(
        "The historically important use: when you have plenty of **unlabelled** "
        "data and few labels, train an autoencoder on everything, then reuse its "
        "encoder as the feature extractor for a supervised model."
    )

    derive(
        [("<b>Why this helps, in one inequality.</b> With $m$ labelled examples "
          "and a hypothesis class of complexity $\\mathcal{C}$, the "
          "generalisation gap scales roughly as:",
          r"\mathbb{E}\bigl[R(\hat h) - \hat R(\hat h)\bigr] \;\lesssim\;"
          r"\sqrt{\frac{\mathcal{C}}{m}}"),
         ("Unsupervised pretraining does not increase $m$. It <b>reduces "
          "$\\mathcal{C}$</b> — after pretraining, the supervised search starts "
          "from a good region and only needs to learn a small head, so the "
          "effective hypothesis class is far smaller.", None),
         ("Equivalently, it acts as a <b>data-dependent prior</b>: the encoder "
          "encodes what the unlabelled distribution looks like, and that "
          "structure is usually relevant to the labels.", None),
         ("<b>The assumption this rests on</b> is that the features useful for "
          "reconstruction overlap with the features useful for the label. That is "
          "often true and sometimes badly false — reconstructing a face requires "
          "modelling the background, which is irrelevant to identity. When it "
          "fails, pretraining wastes capacity on nuisance variation.", None),
         ("Modern self-supervised objectives (contrastive learning, masked "
          "modelling) are designed to avoid exactly that failure: they ask the "
          "model to predict something <i>semantic</i> rather than to reproduce "
          "every pixel.", None)],
        title="Why pretraining on unlabelled data helps",
    )

    note(
        "Greedy layerwise pretraining is history, but worth knowing",
        "Before 2012, deep networks were trained one layer at a time: train a "
        "shallow autoencoder, freeze it, train another on its codings, stack, "
        "repeat. Hinton's 2006 result on this restarted deep learning. It is now "
        "obsolete — better initialisation (§11.1), better activations, batch "
        "norm and residual connections made end-to-end training work — but the "
        "<i>idea</i> of pretraining on unlabelled data came back with a "
        "vengeance as self-supervised learning.",
    )

    sub("Visualising what the layers learned")

    table(
        ["Technique", "What it shows", "Caveat"],
        [["Reconstructions side by side", "Whether the code retains the content",
          "Blurry ≠ bad code"],
         ["Weight images (first layer)", "Learned edge/blob detectors",
          "Only interpretable for the first layer"],
         ["<b>t-SNE / UMAP of the codings</b>",
          "Whether classes separate in latent space",
          "Distances between clusters are meaningless (§8.7)"],
         ["Linear probe on the codings",
          "<b>The honest measure</b> of representation quality",
          "Needs some labels"]],
    )

    anim_header("Reconstruction quality as the bottleneck narrows")

    rng = np.random.default_rng(1)
    # a stylised 12x12 "digit"
    base = np.zeros((12, 12))
    base[2:10, 4:6] = 1.0
    base[2:4, 4:9] = 1.0
    base[9:11, 3:10] = 1.0
    base += rng.normal(0, .05, base.shape)
    U_, S_, Vt_ = np.linalg.svd(base)
    frames = []
    for d in range(1, 13):
        rec = (U_[:, :d] * S_[:d]) @ Vt_[:d]
        mse = float(np.mean((base - rec) ** 2))
        frames.append(go.Frame(name=str(d), data=[
            go.Heatmap(z=base, colorscale=nav.cscale(), showscale=False,
                       zmin=0, zmax=1.1, xaxis="x", yaxis="y"),
            go.Heatmap(z=rec, colorscale=nav.cscale(), showscale=False,
                       zmin=0, zmax=1.1, xaxis="x2", yaxis="y2"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"code size d = {d} of 12   ·   reconstruction MSE = {mse:.5f}   ·   "
            f"compression {d/12:.0%}")])))

    f = make_subplots(rows=1, cols=2, subplot_titles=("original",
                                                      "reconstruction"))
    f.add_trace(go.Heatmap(z=base, colorscale=nav.cscale(), showscale=False,
                           zmin=0, zmax=1.1), 1, 1)
    f.add_trace(go.Heatmap(z=(U_[:, :1]*S_[:1]) @ Vt_[:1],
                           colorscale=nav.cscale(), showscale=False,
                           zmin=0, zmax=1.1), 1, 2)
    f.update_layout(height=400, title="Narrower code, blurrier reconstruction")
    f.update_yaxes(autorange="reversed", scaleanchor=None)
    anim.animate(f, frames, duration=nav.anim_ms(560), slider_prefix="d = ")
    figure(f)

    code_lab(
        "Stacked autoencoders, tied weights, and pretraining with few labels",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=8000, n_test=1500)
Xtr = Xtr.reshape(len(Xtr), -1).astype("float32")
Xte = Xte.reshape(len(Xte), -1).astype("float32")
if Xtr.max() > 1.5:
    Xtr /= 255.; Xte /= 255.
print(f"=== Fashion-MNIST: train {Xtr.shape}, test {Xte.shape} ===")

# ============ 1. A STACKED AUTOENCODER =================================
def stacked_ae(code=30, act="selu"):
    enc = keras.Sequential([keras.layers.Input(shape=(784,)),
                            keras.layers.Dense(200, activation=act),
                            keras.layers.Dense(80, activation=act),
                            keras.layers.Dense(code, activation=act)])
    dec = keras.Sequential([keras.layers.Input(shape=(code,)),
                            keras.layers.Dense(80, activation=act),
                            keras.layers.Dense(200, activation=act),
                            keras.layers.Dense(784, activation="sigmoid")])
    return enc, dec, keras.Sequential([enc, dec])

tf.random.set_seed(0)
enc, dec, ae = stacked_ae()
ae.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
t0 = time.perf_counter()
ae.fit(Xtr, Xtr, epochs=18, batch_size=128, verbose=0,
       validation_data=(Xte, Xte))
print()
print("=== stacked autoencoder 784-200-80-30-80-200-784 ===")
print(f"  parameters {ae.count_params():,}, trained in "
      f"{time.perf_counter()-t0:.1f}s")
print(f"  test reconstruction MSE {ae.evaluate(Xte, Xte, verbose=0):.5f}")
print(f"  compression: 784 floats -> 30, a factor of {784/30:.1f}")

# ============ 2. TIED WEIGHTS ==========================================
print()
print("=== tied weights: the decoder is the encoder transposed ===")

@keras.utils.register_keras_serializable(package="MLPlatform")
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kw):
        super().__init__(**kw)
        self.dense = dense
        self.activation = keras.activations.get(activation)

    def build(self, shape):
        self.biases = self.add_weight(
            name="bias", shape=[self.dense.input.shape[-1]
                                if hasattr(self.dense, "input") else
                                self.dense.kernel.shape[0]],
            initializer="zeros")

    def call(self, z):
        return self.activation(
            tf.matmul(z, self.dense.kernel, transpose_b=True) + self.biases)

d1 = keras.layers.Dense(200, activation="selu")
d2 = keras.layers.Dense(80, activation="selu")
d3 = keras.layers.Dense(30, activation="selu")
tied_enc = keras.Sequential([keras.layers.Input(shape=(784,)), d1, d2, d3])
_ = tied_enc(tf.zeros((1, 784)))          # build so kernels exist
tied_dec = keras.Sequential([keras.layers.Input(shape=(30,)),
                             DenseTranspose(d3, activation="selu"),
                             DenseTranspose(d2, activation="selu"),
                             DenseTranspose(d1, activation="sigmoid")])
tied_ae = keras.Sequential([tied_enc, tied_dec])
tf.random.set_seed(0)
tied_ae.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
tied_ae.fit(Xtr, Xtr, epochs=18, batch_size=128, verbose=0)

print(f"{'model':<26}{'parameters':>13}{'test MSE':>12}")
print(f"{'untied':<26}{ae.count_params():>13,}"
      f"{ae.evaluate(Xte, Xte, verbose=0):>12.5f}")
print(f"{'TIED':<26}{tied_ae.count_params():>13,}"
      f"{tied_ae.evaluate(Xte, Xte, verbose=0):>12.5f}")
print(f"  tied uses {1 - tied_ae.count_params()/ae.count_params():.0%} "
      f"fewer parameters and regularises by construction")

# ============ 3. UNSUPERVISED PRETRAINING ==============================
print()
print("="*66)
print("Pretraining when labels are scarce")
print("="*66)
print("  the autoencoder above saw all 8 000 images -- WITHOUT their labels.")

def classifier(encoder=None, freeze=False):
    if encoder is None:
        body = keras.Sequential([keras.layers.Input(shape=(784,)),
                                 keras.layers.Dense(200, activation="selu"),
                                 keras.layers.Dense(80, activation="selu"),
                                 keras.layers.Dense(30, activation="selu")])
    else:
        body = keras.models.clone_model(encoder)
        body.set_weights(encoder.get_weights())
        body.trainable = not freeze
    return keras.Sequential([body,
                             keras.layers.Dense(10, activation="softmax")])

print()
print(f"{'labelled examples':>19}{'from scratch':>15}{'pretrained':>13}"
      f"{'pretrained+frozen':>20}")
for n_lab in [100, 300, 1000, 8000]:
    row = []
    for enc_arg, frz in [(None, False), (enc, False), (enc, True)]:
        tf.random.set_seed(0)
        m = classifier(enc_arg, frz)
        m.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
        m.fit(Xtr[:n_lab], ytr[:n_lab], epochs=25, batch_size=32, verbose=0)
        row.append(m.evaluate(Xte, yte, verbose=0,
                              return_dict=True)["accuracy"])
    print(f"{n_lab:>19}{row[0]:>15.4f}{row[1]:>13.4f}{row[2]:>20.4f}")
print()
print("  with 100 labels the pretrained encoder is worth a lot.")
print("  with 8 000 the advantage nearly vanishes -- pretraining buys")
print("  DATA EFFICIENCY, not a higher ceiling.")

# ============ 4. THE HONEST MEASURE: A LINEAR PROBE ====================
print()
print("=== linear probe: how linearly separable are the codings? ===")
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
Ztr = enc.predict(Xtr, verbose=0)
Zte = enc.predict(Xte, verbose=0)
pca30 = PCA(n_components=30).fit(Xtr)

print(f"{'representation':<34}{'dims':>7}{'linear probe accuracy':>24}")
for nm, A, B in [("raw pixels", Xtr, Xte),
                 ("PCA (30 components)", pca30.transform(Xtr),
                  pca30.transform(Xte)),
                 ("autoencoder codings (30)", Ztr, Zte)]:
    lr = LogisticRegression(max_iter=400, n_jobs=-1).fit(A[:4000], ytr[:4000])
    print(f"{nm:<34}{A.shape[1]:>7}{lr.score(B, yte):>24.4f}")
print("  this is the number that matters -- NOT the reconstruction MSE.")

# ============ 5. WHAT THE FIRST LAYER LEARNED ==========================
print()
print("=== first-layer weights as images ===")
W = enc.layers[0].get_weights()[0]         # (784, 200)
print(f"  weight matrix {W.shape}: each COLUMN is a 28x28 filter")
norms = np.linalg.norm(W, axis=0)
print(f"  filter norms: min {norms.min():.3f}, max {norms.max():.3f}")
top = np.argsort(-norms)[:16]
grid = np.zeros((4*28, 4*28))
for i, j in enumerate(top):
    r, c = divmod(i, 4)
    w = W[:, j].reshape(28, 28)
    grid[r*28:(r+1)*28, c*28:(c+1)*28] = (w - w.min())/(np.ptp(w)+1e-9)

# ============ 6. RECONSTRUCTIONS =======================================
recon = ae.predict(Xte[:8], verbose=0)
strip = np.zeros((2*28, 8*28))
for i in range(8):
    strip[:28, i*28:(i+1)*28] = Xte[i].reshape(28, 28)
    strip[28:, i*28:(i+1)*28] = recon[i].reshape(28, 28)
print()
print(f"  per-image reconstruction MSE for the 8 shown: "
      f"{np.round(((Xte[:8]-recon)**2).mean(1), 4)}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("first-layer filters",
                                    "originals (top) vs reconstructions"))
fig.add_trace(go.Heatmap(z=grid, colorscale="Greys", showscale=False), 1, 1)
fig.add_trace(go.Heatmap(z=strip, colorscale="Greys", showscale=False,
                         reversescale=True), 1, 2)
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(height=380)
''',
        key="ch17_stacked",
    )

    keypoints([
        "Stack layers symmetrically around the bottleneck; <b>tie</b> the decoder "
        "weights to halve the parameters.",
        "<b>Unsupervised pretraining</b> reduces the effective hypothesis class, "
        "buying data efficiency, not a higher ceiling.",
        "It assumes reconstruction features overlap with label features — often "
        "true, sometimes badly false.",
        "Greedy layerwise training is obsolete; the <i>idea</i> returned as "
        "self-supervised learning.",
        "Judge a representation by a <b>linear probe</b>, not by reconstruction "
        "MSE.",
    ])


# ==========================================================================
def s_17_3():
    section("17.3", "Convolutional and Recurrent Autoencoders")

    lead(
        "For images, a dense autoencoder throws away the one thing you know for "
        "certain: that nearby pixels are related. Convolutions put it back."
    )

    sub("The convolutional autoencoder")

    md(
        "The encoder is an ordinary CNN (§14): convolutions and pooling reduce "
        "spatial size while increasing depth. The decoder must do the reverse — "
        "**upsample** back to the original resolution."
    )

    table(
        ["Upsampling method", "How", "Artefact"],
        [["<code>Conv2DTranspose</code>",
          "Learned; inserts zeros between inputs then convolves",
          "<b>Checkerboard artefacts</b> when stride does not divide the kernel "
          "size"],
         ["<code>UpSampling2D</code> + <code>Conv2D</code>",
          "Nearest-neighbour or bilinear resize, then a normal convolution",
          "<b>No checkerboarding</b> — the recommended default"],
         ["Sub-pixel / pixel shuffle",
          "Convolve to $r^2$ channels then rearrange into space",
          "Efficient; needs careful initialisation"]],
    )

    pitfall(
        "Checkerboard artefacts come from uneven kernel overlap",
        "A transposed convolution with kernel size $k$ and stride $s$ writes "
        "overlapping patches. When $s$ does not divide $k$, some output pixels "
        "receive contributions from more kernel positions than their neighbours "
        "— producing a regular grid of bright and dark pixels that no amount of "
        "training removes. <b>Fix</b>: use kernel sizes divisible by the stride "
        "(4 with stride 2), or replace the transposed convolution with resize + "
        "convolution entirely (Odena et al., 2016).",
    )

    derive(
        [("<b>Why convolution is the right prior for images.</b> A dense layer "
          "from $28\\times28$ to 200 units has:",
          r"784 \times 200 + 200 = 157\,000 \text{ parameters}"),
         ("A convolutional layer with 32 filters of size $3\\times3$ has:",
          r"3 \times 3 \times 1 \times 32 + 32 = 320 \text{ parameters}"),
         ("<b>A factor of 490</b>, and the convolution is <i>better</i>, not "
          "worse, because it encodes two facts that are true of images and that "
          "the dense layer would have to learn from data:", None),
         ("<b>Locality</b> — a pixel's meaning depends on its neighbourhood, so "
          "connections beyond the kernel are unnecessary. <b>Translation "
          "equivariance</b> — a feature detector useful at one location is useful "
          "at every location, so the weights can be shared.", None),
         ("Both are <b>hard constraints</b>, not soft preferences. They are the "
          "reason a convolutional autoencoder with 1 % of the parameters "
          "reconstructs better than a dense one — precisely the ViT argument of "
          "§16.8, running in the other direction.", None)],
        title="Why convolutions dominate for image autoencoders",
    )

    sub("Recurrent autoencoders")

    md(
        "For sequences, the encoder is a sequence-to-vector RNN and the decoder a "
        "vector-to-sequence RNN — the encoder–decoder of §15.1, with the input "
        "as its own target."
    )

    codenote(
        "<code>RepeatVector</code> is the bridge",
        "The encoder produces one vector; the decoder needs a sequence. "
        "<code>RepeatVector(T)</code> tiles the code $T$ times so the decoder RNN "
        "has something to consume at every step. It is the standard idiom, and "
        "the alternative — feeding the code only at the first step — usually "
        "trains worse because the signal has to survive $T$ recurrent steps.",
    )

    anim_header("How a transposed convolution produces checkerboard artefacts")

    frames = []
    for cfg_i, (k, s, label) in enumerate([(3, 2, "kernel 3, stride 2 — 3 ∤ 2"),
                                           (4, 2, "kernel 4, stride 2 — 4 ÷ 2 ✓"),
                                           (5, 2, "kernel 5, stride 2 — 5 ∤ 2"),
                                           (6, 3, "kernel 6, stride 3 — 6 ÷ 3 ✓")]):
        n_in = 8
        n_out = (n_in - 1) * s + k
        counts = np.zeros(n_out)
        for i in range(n_in):
            counts[i * s: i * s + k] += 1
        cnt2 = np.outer(counts, counts)
        uneven = counts[k:-k].std() if n_out > 2 * k else counts.std()
        frames.append(go.Frame(name=str(k) + "/" + str(s), data=[
            go.Heatmap(z=cnt2, colorscale=nav.cscale(), showscale=False),
        ], layout=go.Layout(title=label, annotations=[anim.annotate_step(
            f"overlap counts per output pixel: "
            f"{np.unique(counts.astype(int))}   ·   interior std = "
            f"{uneven:.3f}   ·   "
            + ("EVEN — no artefact" if uneven < 1e-9 else
               "UNEVEN — checkerboard"),
            color=C["success"] if uneven < 1e-9 else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=430, xaxis=dict(visible=False),
                    yaxis=dict(visible=False, autorange="reversed",
                               scaleanchor="x"),
                    title="kernel 3, stride 2 — 3 ∤ 2")
    anim.animate(f, frames, duration=nav.anim_ms(1700), slider_prefix="config ")
    figure(f, "Each cell counts how many kernel placements write to that output "
              "pixel. When the counts differ, the network has a built-in grid "
              "pattern it can never fully unlearn.")

    code_lab(
        "Convolutional autoencoders, the checkerboard fix, and a sequence AE",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=5000, n_test=1200)
Xtr = Xtr.astype("float32"); Xte = Xte.astype("float32")
if Xtr.max() > 1.5:
    Xtr /= 255.; Xte /= 255.
Xtr4 = Xtr.reshape(-1, 28, 28, 1); Xte4 = Xte.reshape(-1, 28, 28, 1)
Xtr2 = Xtr.reshape(len(Xtr), -1);  Xte2 = Xte.reshape(len(Xte), -1)

# ============ 1. DENSE vs CONVOLUTIONAL ================================
def dense_ae(code=32):
    enc = keras.Sequential([keras.layers.Input(shape=(784,)),
                            keras.layers.Dense(256, activation="selu"),
                            keras.layers.Dense(code, activation="selu")])
    dec = keras.Sequential([keras.layers.Input(shape=(code,)),
                            keras.layers.Dense(256, activation="selu"),
                            keras.layers.Dense(784, activation="sigmoid")])
    return keras.Sequential([enc, dec])

def conv_ae(mode="upsample"):
    enc = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(16, 3, padding="same", activation="selu"),
        keras.layers.MaxPool2D(),                       # 14x14
        keras.layers.Conv2D(32, 3, padding="same", activation="selu"),
        keras.layers.MaxPool2D(),                       # 7x7
        keras.layers.Conv2D(64, 3, padding="same", activation="selu"),
    ])
    if mode == "transpose_bad":
        up = [keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same",
                                           activation="selu"),   # 3 not / 2
              keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same",
                                           activation="selu"),
              keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")]
    elif mode == "transpose_good":
        up = [keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same",
                                           activation="selu"),   # 4 / 2 = 2
              keras.layers.Conv2DTranspose(16, 4, strides=2, padding="same",
                                           activation="selu"),
              keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")]
    else:
        up = [keras.layers.UpSampling2D(),
              keras.layers.Conv2D(32, 3, padding="same", activation="selu"),
              keras.layers.UpSampling2D(),
              keras.layers.Conv2D(16, 3, padding="same", activation="selu"),
              keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")]
    dec = keras.Sequential([keras.layers.Input(shape=(7, 7, 64))] + up)
    return keras.Sequential([enc, dec])

print("=== dense vs convolutional autoencoder ===")
print(f"{'model':<32}{'params':>10}{'fit time':>11}{'test MSE':>12}")
tf.random.set_seed(0)
d_ae = dense_ae()
d_ae.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
t0 = time.perf_counter(); d_ae.fit(Xtr2, Xtr2, epochs=8, batch_size=128,
                                   verbose=0)
print(f"{'dense (code 32)':<32}{d_ae.count_params():>10,}"
      f"{time.perf_counter()-t0:>10.1f}s"
      f"{d_ae.evaluate(Xte2, Xte2, verbose=0):>12.5f}")

conv_models = {}
for nm, mode in [("conv + Conv2DTranspose(k=3)", "transpose_bad"),
                 ("conv + Conv2DTranspose(k=4)", "transpose_good"),
                 ("conv + UpSampling2D+Conv2D", "upsample")]:
    tf.random.set_seed(0)
    m = conv_ae(mode)
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    t0 = time.perf_counter()
    m.fit(Xtr4, Xtr4, epochs=8, batch_size=128, verbose=0)
    conv_models[nm] = m
    print(f"{nm:<32}{m.count_params():>10,}{time.perf_counter()-t0:>10.1f}s"
          f"{m.evaluate(Xte4, Xte4, verbose=0):>12.5f}")
print()
print("  the conv autoencoder has a code of 7x7x64 = 3136 -- LARGER than 784.")
print("  it is not undercomplete at all; the CONVOLUTIONAL STRUCTURE is the")
print("  constraint. Locality + weight sharing are the prior.")

# ============ 2. CHECKERBOARD ARTEFACTS, MEASURED ======================
print()
print("=== detecting checkerboard artefacts ===")
def checkerboard_energy(imgs):
    """Energy at the Nyquist frequency = the checkerboard signature."""
    F = np.fft.fft2(imgs[..., 0])
    F = np.abs(np.fft.fftshift(F))
    h, w = F.shape[1:]
    corner = F[:, :3, :3].mean() + F[:, -3:, -3:].mean()
    centre = F[:, h//2-3:h//2+3, w//2-3:w//2+3].mean()
    return float(corner / (centre + 1e-9))

for nm, m in conv_models.items():
    rec = m.predict(Xte4[:300], verbose=0)
    print(f"  {nm:<32} high-frequency ratio "
          f"{checkerboard_energy(rec):.5f}")
print(f"  {'(real images, for reference)':<32} high-frequency ratio "
      f"{checkerboard_energy(Xte4[:300]):.5f}")
print("  kernel 3 with stride 2 leaves excess energy at the Nyquist frequency.")
print("  kernel 4 (divisible by the stride) and resize+conv do not.")

# --- the overlap-count argument, directly ----------------------------
print()
print("=== why: uneven kernel overlap ===")
for k, s in [(3, 2), (4, 2), (5, 2), (6, 3)]:
    n_in = 8
    counts = np.zeros((n_in-1)*s + k)
    for i in range(n_in):
        counts[i*s:i*s+k] += 1
    inner = counts[k:-k] if len(counts) > 2*k else counts
    print(f"  kernel {k}, stride {s}: interior overlap counts "
          f"{np.unique(inner.astype(int))}   "
          f"{'EVEN' if inner.std() < 1e-9 else 'UNEVEN -> checkerboard'}")

# ============ 3. THE ENCODER'S FEATURES ================================
print()
print("=== linear probe on the convolutional codings ===")
from sklearn.linear_model import LogisticRegression
best = conv_models["conv + UpSampling2D+Conv2D"]
cenc = best.layers[0]
Ztr = cenc.predict(Xtr4[:4000], verbose=0).reshape(4000, -1)
Zte = cenc.predict(Xte4, verbose=0).reshape(len(Xte4), -1)
denc = d_ae.layers[0]
Dtr = denc.predict(Xtr2[:4000], verbose=0)
Dte = denc.predict(Xte2, verbose=0)
print(f"{'codings':<32}{'dims':>7}{'probe accuracy':>17}")
for nm, A, B in [("dense autoencoder", Dtr, Dte),
                 ("convolutional autoencoder", Ztr, Zte),
                 ("raw pixels", Xtr2[:4000], Xte2)]:
    lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(A, ytr[:4000])
    print(f"{nm:<32}{A.shape[1]:>7}{lr.score(B, yte):>17.4f}")

# ============ 4. A RECURRENT AUTOENCODER ===============================
print()
print("="*62)
print("A sequence autoencoder")
print("="*62)
df = _ds.ridership(n_days=900)
y = (df["rail"].to_numpy()/1e6).astype("float32")
L = 28
n = len(y) - L
S = np.stack([y[i:i+L] for i in range(n)])[..., None]
mu, sd = S[:600].mean(), S[:600].std()
S = ((S - mu)/sd).astype("float32")
Str, Ste = S[:600], S[600:]
print(f"  {len(S)} windows of {L} days")

tf.random.set_seed(0)
r_enc = keras.Sequential([keras.layers.Input(shape=(L, 1)),
                          keras.layers.LSTM(64, return_sequences=True),
                          keras.layers.LSTM(16)])                # seq -> vector
r_dec = keras.Sequential([keras.layers.Input(shape=(16,)),
                          keras.layers.RepeatVector(L),          # THE BRIDGE
                          keras.layers.LSTM(64, return_sequences=True),
                          keras.layers.Dense(1)])
r_ae = keras.Sequential([r_enc, r_dec])
r_ae.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
r_ae.fit(Str, Str, epochs=35, batch_size=32, verbose=0)
print(f"  {L}x1 -> 16 -> {L}x1, parameters {r_ae.count_params():,}")
print(f"  test reconstruction MSE {r_ae.evaluate(Ste, Ste, verbose=0):.5f}")
print(f"  compression {L}/16 = {L/16:.2f}x")

# --- anomaly detection: the classic use ------------------------------
print()
print("=== reconstruction error as an anomaly score ===")
rng = np.random.default_rng(0)
anom = Ste[:60].copy()
for i in range(len(anom)):
    j = rng.integers(4, L-4)
    anom[i, j:j+3, 0] += rng.choice([-1, 1]) * rng.uniform(2.5, 4.0)
err_norm = ((r_ae.predict(Ste[:60], verbose=0) - Ste[:60])**2).mean((1, 2))
err_anom = ((r_ae.predict(anom, verbose=0) - anom)**2).mean((1, 2))
thr = np.quantile(err_norm, .95)
print(f"  normal windows : mean error {err_norm.mean():.5f}")
print(f"  spiked windows : mean error {err_anom.mean():.5f}  "
      f"({err_anom.mean()/err_norm.mean():.1f}x higher)")
print(f"  threshold at the 95th percentile of normal = {thr:.5f}")
print(f"  detection rate {np.mean(err_anom > thr):.1%}, "
      f"false-alarm rate {np.mean(err_norm > thr):.1%}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
rec = best.predict(Xte4[:8], verbose=0)
bad = conv_models["conv + Conv2DTranspose(k=3)"].predict(Xte4[:8], verbose=0)
strip = np.zeros((3*28, 8*28))
for i in range(8):
    strip[:28, i*28:(i+1)*28] = Xte4[i, :, :, 0]
    strip[28:56, i*28:(i+1)*28] = rec[i, :, :, 0]
    strip[56:, i*28:(i+1)*28] = bad[i, :, :, 0]
fig = go.Figure(go.Heatmap(z=strip, colorscale="Greys", reversescale=True,
                           showscale=False))
fig.update_layout(height=380, xaxis=dict(visible=False),
                  yaxis=dict(visible=False, autorange="reversed"),
                  title="original / resize+conv / Conv2DTranspose(k=3)")
''',
        key="ch17_conv",
    )

    keypoints([
        "A convolutional autoencoder's constraint is <b>structural</b> (locality "
        "+ weight sharing), not a narrow bottleneck.",
        "<code>Conv2DTranspose</code> checkerboards when the stride does not "
        "divide the kernel size.",
        "Prefer <code>UpSampling2D</code> + <code>Conv2D</code>, or a kernel "
        "divisible by the stride.",
        "A recurrent autoencoder uses <code>RepeatVector</code> to bridge vector "
        "→ sequence.",
        "Reconstruction error makes an excellent <b>anomaly score</b> — the most "
        "common production use of autoencoders.",
    ])

# ==========================================================================
def s_17_4():
    section("17.4", "Denoising Autoencoders")

    lead(
        "Corrupt the input, ask for the clean original. The network can no longer "
        "copy — it has to <b>understand</b> the data well enough to repair it."
    )

    sub("The objective")

    math(r"""
    \tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}} \mid \mathbf{x}),
    \qquad
    \mathcal{L}_{\text{DAE}} = \mathbb{E}_{\mathbf{x}}\,
      \mathbb{E}_{\tilde{\mathbf{x}} \mid \mathbf{x}}
      \bigl\lVert \mathbf{x} - g_\theta\bigl(f_\phi(\tilde{\mathbf{x}})\bigr)
      \bigr\rVert^{2}
    """)

    where({
        r"q(\tilde{\mathbf{x}} \mid \mathbf{x})":
            "the corruption process — Gaussian noise, or dropout (masking) noise",
        r"\mathbf{x}": "the <b>clean</b> target — note the loss compares against "
                       "the original, not the corrupted input",
    })

    table(
        ["Corruption", "How", "Effect"],
        [["<b>Gaussian noise</b>", "$\\tilde{\\mathbf{x}} = \\mathbf{x} + "
          "\\boldsymbol\\varepsilon$, $\\boldsymbol\\varepsilon \\sim "
          "\\mathcal{N}(0, \\sigma^2\\mathbf{I})$",
          "Learns to smooth; the model estimates the score (see below)"],
         ["<b>Masking / dropout noise</b>",
          "Zero out a random fraction $p$ of the inputs",
          "Learns to <b>inpaint</b> — reconstruct from partial observation"],
         ["<b>Salt-and-pepper</b>", "Set random pixels to 0 or 1",
          "Robustness to outlier pixels"]],
    )

    derive(
        [("<b>A denoising autoencoder learns the score of the data "
          "distribution.</b> This is the single most consequential fact in the "
          "chapter, because it is what diffusion models are built on (§17.9).",
          None),
         ("Add Gaussian noise: $\\tilde{\\mathbf{x}} = \\mathbf{x} + "
          "\\sigma\\boldsymbol\\varepsilon$ with $\\boldsymbol\\varepsilon \\sim "
          "\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$, and train $r_\\theta$ to "
          "minimise:",
          r"\mathcal{L} = \mathbb{E}\bigl\lVert \mathbf{x} - "
          r"r_\theta(\tilde{\mathbf{x}}) \bigr\rVert^{2}"),
         ("The minimiser of a squared loss is the conditional expectation "
          "(§4.1), so the optimal reconstruction is:",
          r"r^{\star}(\tilde{\mathbf{x}}) = \mathbb{E}\bigl[\mathbf{x} \mid "
          r"\tilde{\mathbf{x}}\bigr]"),
         ("<b>Tweedie's formula</b> states that for Gaussian noise this "
          "conditional mean is expressible through the gradient of the log "
          "density of the <i>noisy</i> distribution $p_\\sigma$:",
          r"\mathbb{E}\bigl[\mathbf{x} \mid \tilde{\mathbf{x}}\bigr] "
          r"= \tilde{\mathbf{x}} + \sigma^{2}\,\nabla_{\tilde{\mathbf{x}}}"
          r"\log p_\sigma(\tilde{\mathbf{x}})"),
         ("Rearranged, the trained denoiser <b>is</b> a score estimator:",
          r"\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})"
          r" \;\approx\; \frac{r_\theta(\tilde{\mathbf{x}}) - \tilde{\mathbf{x}}}"
          r"{\sigma^{2}}"),
         ("The residual — <b>how the denoiser moves a point</b> — points uphill "
          "on the data density. Follow that direction repeatedly from pure noise "
          "and you generate samples. That procedure is Langevin dynamics, and "
          "with a schedule over $\\sigma$ it is exactly a diffusion model (§17.9). "
          "<b>Every image generator you have used is a denoising autoencoder run "
          "in a loop.</b>", None)],
        title="Denoising ⇒ score estimation ⇒ diffusion",
    )

    idea(
        "Noise is a regulariser with a probabilistic meaning",
        "Adding input noise is superficially like dropout (§11.9) — and it is, "
        "for a fixed small $\\sigma$: Bishop (1995) showed Gaussian input noise is "
        "equivalent to Tikhonov regularisation to first order. But the derivation "
        "above says something much stronger: at every noise level the denoiser is "
        "estimating a real property of the data distribution. That is what turned "
        "a 2008 regularisation trick into the basis of modern generative "
        "modelling.",
    )

    pitfall(
        "Noise is applied at training time only",
        "The <code>GaussianNoise</code> and <code>Dropout</code> layers are "
        "active during <code>fit</code> and inactive during "
        "<code>predict</code> — that is exactly what you want, and it is handled "
        "automatically by the <code>training</code> flag. If you implement the "
        "corruption manually with <code>numpy</code>, you must remember to skip "
        "it at inference yourself, or your reconstructions will be needlessly "
        "noisy.",
    )

    anim_header("A denoiser's residual field points uphill on the density")

    rng = np.random.default_rng(4)
    n_mode = 200
    data2 = np.vstack([rng.normal([-1.4, -0.7], .38, (n_mode, 2)),
                       rng.normal([1.5, 0.9], .45, (n_mode, 2))])

    def log_density(P):
        d1 = np.exp(-((P - [-1.4, -0.7]) ** 2).sum(-1) / (2 * .38 ** 2))
        d2 = np.exp(-((P - [1.5, 0.9]) ** 2).sum(-1) / (2 * .45 ** 2))
        return np.log(d1 + d2 + 1e-12)

    gx = np.linspace(-3.4, 3.6, 22)
    gy = np.linspace(-2.6, 2.8, 18)
    GX, GY = np.meshgrid(gx, gy)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    eps = 1e-3
    sx = (log_density(P + [eps, 0]) - log_density(P - [eps, 0])) / (2 * eps)
    sy = (log_density(P + [0, eps]) - log_density(P - [0, eps])) / (2 * eps)
    S = np.column_stack([sx, sy])
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)

    # Langevin: walk noisy points uphill
    walk = rng.uniform([-3.2, -2.4], [3.4, 2.6], (110, 2))
    traj = [walk.copy()]
    for _ in range(30):
        gx_ = (log_density(walk + [eps, 0]) - log_density(walk - [eps, 0]))/(2*eps)
        gy_ = (log_density(walk + [0, eps]) - log_density(walk - [0, eps]))/(2*eps)
        g = np.column_stack([gx_, gy_])
        walk = walk + .07 * g + rng.normal(0, .035, walk.shape)
        traj.append(walk.copy())

    frames = []
    for k, W in enumerate(traj):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=data2[:, 0], y=data2[:, 1], mode="markers",
                       marker=dict(size=4, color=alpha(C["muted"], .5))),
            go.Scatter(x=W[:, 0], y=W[:, 1], mode="markers",
                       marker=dict(size=7, color=C["danger"],
                                   line=dict(color="#fff", width=1))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"Langevin step {k}   ·   x ← x + η·∇log p(x) + noise   ·   "
            f"the gradient is exactly (denoiser(x) − x)/σ²")])))

    f = go.Figure()
    f.add_trace(go.Scatter(x=data2[:, 0], y=data2[:, 1], mode="markers",
                           name="data", marker=dict(size=4,
                                                    color=alpha(C["muted"], .5))))
    f.add_trace(go.Scatter(x=traj[0][:, 0], y=traj[0][:, 1], mode="markers",
                           name="samples", marker=dict(size=7, color=C["danger"],
                                                       line=dict(color="#fff",
                                                                 width=1))))
    for i in range(0, len(P), 1):
        pass
    f.add_trace(go.Scatter(
        x=np.ravel([[P[i, 0], P[i, 0] + .22*S[i, 0], None]
                    for i in range(len(P))]),
        y=np.ravel([[P[i, 1], P[i, 1] + .22*S[i, 1], None]
                    for i in range(len(P))]),
        mode="lines", name="score field ∇log p",
        line=dict(color=alpha(C["primary"], .5), width=1.6)))
    f.update_layout(height=470, xaxis_title="x₁", yaxis_title="x₂",
                    title="Following the score turns noise into samples",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(150), slider_prefix="step ")
    figure(f, "This is the whole of diffusion sampling. The only hard part is "
              "estimating that arrow field — and a denoising autoencoder does "
              "exactly that.")

    code_lab(
        "Denoising autoencoders, inpainting, and the score-estimation identity",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=8000, n_test=1500)
Xtr = Xtr.reshape(len(Xtr), -1).astype("float32")
Xte = Xte.reshape(len(Xte), -1).astype("float32")
if Xtr.max() > 1.5:
    Xtr /= 255.; Xte /= 255.

# ============ 1. TWO KINDS OF CORRUPTION ===============================
def dae(noise="gaussian", level=.3, code=32):
    inp = keras.layers.Input(shape=(784,))
    if noise == "gaussian":
        z = keras.layers.GaussianNoise(level)(inp)
    elif noise == "dropout":
        z = keras.layers.Dropout(level)(inp)
    else:
        z = inp
    z = keras.layers.Dense(256, activation="selu")(z)
    z = keras.layers.Dense(code, activation="selu")(z)
    enc = keras.Model(inp, z)
    z = keras.layers.Dense(256, activation="selu")(z)
    out = keras.layers.Dense(784, activation="sigmoid")(z)
    return enc, keras.Model(inp, out)

print("=== plain vs denoising autoencoder ===")
print(f"{'model':<34}{'clean MSE':>12}{'probe acc':>12}"
      f"{'MSE on NOISY input':>21}")
from sklearn.linear_model import LogisticRegression
noisy_te = np.clip(Xte + np.random.default_rng(0).normal(0, .3, Xte.shape),
                   0, 1).astype("float32")
models = {}
for nm, kind, lvl in [("plain autoencoder", "none", 0),
                      ("denoising, Gaussian sigma=0.3", "gaussian", .3),
                      ("denoising, dropout p=0.4", "dropout", .4)]:
    tf.random.set_seed(0)
    e, m = dae(kind, lvl)
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    m.fit(Xtr, Xtr, epochs=16, batch_size=128, verbose=0)   # TARGET = CLEAN
    models[nm] = (e, m)
    Z = e.predict(Xtr[:4000], verbose=0)
    lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(Z, ytr[:4000])
    print(f"{nm:<34}{m.evaluate(Xte, Xte, verbose=0):>12.5f}"
          f"{lr.score(e.predict(Xte, verbose=0), yte):>12.4f}"
          f"{np.mean((m.predict(noisy_te, verbose=0) - Xte)**2):>21.5f}")
print()
print("  the plain autoencoder wins on CLEAN input and collapses on noisy input.")
print("  the denoising ones learned something about the data, not a copy rule.")

# ============ 2. INPAINTING ============================================
print()
print("=== dropout noise learns to INPAINT ===")
rng = np.random.default_rng(1)
masked = Xte[:400].copy()
for i in range(len(masked)):
    img = masked[i].reshape(28, 28)
    r, c = rng.integers(4, 18, 2)
    img[r:r+10, c:c+10] = 0                     # a 10x10 hole
    masked[i] = img.ravel()
print(f"  occluded a 10x10 block ({100*100/784:.0f}% of the pixels)")
print(f"{'model':<34}{'MSE on the HOLE':>18}")
hole = (Xte[:400] != masked)
for nm, (e, m) in models.items():
    rec = m.predict(masked, verbose=0)
    print(f"{nm:<34}{np.mean((rec[hole] - Xte[:400][hole])**2):>18.5f}")
print("  the dropout-noise model has literally been trained on this task:")
print("  reconstruct the whole from a random subset.")

# ============ 3. THE SCORE-ESTIMATION IDENTITY =========================
print()
print("="*66)
print("A denoiser estimates the score of the data distribution")
print("="*66)
# a 2-D mixture where we know the true density in closed form
rng = np.random.default_rng(2)
N = 6000
comp = rng.integers(0, 2, N)
mus = np.array([[-1.4, -0.7], [1.5, 0.9]])
sds = np.array([0.38, 0.45])
D = rng.normal(mus[comp], sds[comp][:, None]).astype("float32")

SIGMA = 0.35
def true_score(P, sigma):
    """grad log p_sigma for a Gaussian mixture convolved with N(0, sigma^2)."""
    P = np.asarray(P, dtype="float64")
    s2 = sds**2 + sigma**2
    num = np.zeros_like(P); den = np.zeros(len(P))
    for k in range(2):
        w = np.exp(-((P - mus[k])**2).sum(1)/(2*s2[k])) / s2[k]
        num += (w / s2[k])[:, None] * (mus[k] - P)
        den += w
    return num / den[:, None]

tf.random.set_seed(0)
den_net = keras.Sequential([keras.layers.Input(shape=(2,)),
                            keras.layers.Dense(128, activation="swish"),
                            keras.layers.Dense(128, activation="swish"),
                            keras.layers.Dense(2)])
den_net.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
noisy = (D + rng.normal(0, SIGMA, D.shape)).astype("float32")
den_net.fit(noisy, D, epochs=120, batch_size=256, verbose=0)  # noisy -> CLEAN
print(f"  trained a denoiser on {N} points, sigma={SIGMA}")

test_pts = rng.uniform([-3, -2.5], [3.5, 2.5], (500, 2)).astype("float32")
learned = (den_net.predict(test_pts, verbose=0) - test_pts) / SIGMA**2
truth = true_score(test_pts, SIGMA)
cos = np.mean(np.sum(learned*truth, 1) /
              (np.linalg.norm(learned, axis=1)*np.linalg.norm(truth, axis=1)+1e-9))
print()
print("  Tweedie:  score(x) = (denoiser(x) - x) / sigma^2")
print(f"  mean cosine similarity with the TRUE score: {cos:.4f}")
print(f"  relative magnitude error: "
      f"{np.mean(np.abs(np.linalg.norm(learned,axis=1)/(np.linalg.norm(truth,axis=1)+1e-9) - 1)):.4f}")
print("  the denoiser never saw a density, a gradient or a log. It only")
print("  learned to remove noise -- and that IS the score.")

# ============ 4. SAMPLING BY FOLLOWING THE LEARNED SCORE ===============
print()
print("=== Langevin sampling from the LEARNED score ===")
x = rng.uniform([-3.2, -2.4], [3.4, 2.6], (600, 2)).astype("float32")
step = 0.06
for it in range(220):
    sc = (den_net.predict(x, verbose=0) - x) / SIGMA**2
    x = x + step*sc + np.sqrt(2*step)*rng.normal(0, 1, x.shape).astype("float32")
    x = x.astype("float32")

def which_mode(P):
    d = np.stack([((P-mus[k])**2).sum(1) for k in range(2)], 1)
    return d.argmin(1)
print(f"  started from UNIFORM noise, ran 220 Langevin steps")
print(f"  final mean  : {x.mean(0).round(3)}   (data mean "
      f"{D.mean(0).round(3)})")
print(f"  final std   : {x.std(0).round(3)}   (data std "
      f"{D.std(0).round(3)})")
m_frac = np.bincount(which_mode(x), minlength=2)/len(x)
d_frac = np.bincount(which_mode(D), minlength=2)/len(D)
print(f"  mode split  : {m_frac.round(3)}   (data {d_frac.round(3)})")
print()
print("  THAT IS A GENERATIVE MODEL, built from nothing but a denoiser.")
print("  section 17.9 adds a schedule over sigma and calls it diffusion.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=D[:1200, 0], y=D[:1200, 1], mode="markers", name="real data",
                marker=dict(size=4, color=alpha(C["muted"], .55)))
fig.add_scatter(x=x[:600, 0], y=x[:600, 1], mode="markers",
                name="Langevin samples",
                marker=dict(size=5, color=C["danger"]))
fig.update_layout(height=440, xaxis_title="x1", yaxis_title="x2",
                  title="Samples generated by following a denoiser's residual")
''',
        key="ch17_denoise",
    )

    keypoints([
        "A denoising autoencoder reconstructs the <b>clean</b> input from a "
        "corrupted one — it cannot copy.",
        "Gaussian noise ⇒ smoothing; masking noise ⇒ <b>inpainting</b>.",
        "<b>Tweedie</b>: $\\nabla \\log p_\\sigma(\\tilde{\\mathbf{x}}) \\approx "
        "(r_\\theta(\\tilde{\\mathbf{x}}) - \\tilde{\\mathbf{x}})/\\sigma^2$.",
        "The denoiser's residual <b>is</b> the score — follow it and you "
        "generate samples.",
        "That identity is the entire foundation of diffusion models (§17.9).",
    ])


# ==========================================================================
def s_17_5():
    section("17.5", "Sparse Autoencoders")

    lead(
        "A different constraint: let the code be wide, but require that only a "
        "few units are active for any given input. Sparsity forces each unit to "
        "specialise."
    )

    sub("Two ways to impose sparsity")

    table(
        ["Method", "Penalty", "Behaviour"],
        [["<b>$\\ell_1$ on the activations</b>",
          "$\\lambda \\lVert \\mathbf{z} \\rVert_1$",
          "Simple; drives activations to exactly 0 (same geometry as lasso, §4.9)"],
         ["<b>KL divergence to a target rate</b>",
          "$\\beta \\sum_j D_{\\mathrm{KL}}(\\rho \\,\\Vert\\, \\hat\\rho_j)$",
          "Controls the <i>average</i> activation precisely; the classical choice"]],
    )

    md("With sigmoid codings, let $\\hat\\rho_j$ be unit $j$'s mean activation "
       "over a batch and $\\rho$ the target sparsity (e.g. 0.05). The penalty is "
       "the KL divergence between two Bernoulli distributions:")

    math(r"""
    D_{\mathrm{KL}}\bigl(\rho \,\Vert\, \hat\rho_j\bigr)
    \;=\; \rho \log \frac{\rho}{\hat\rho_j}
      \;+\; (1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}
    """)

    derive(
        [("<b>Why KL and not squared error on the rate.</b> Compare the two "
          "penalties as functions of $\\hat\\rho$, with target $\\rho$:", None),
         ("A squared penalty $(\\hat\\rho - \\rho)^2$ is symmetric and bounded — "
          "its gradient is $2(\\hat\\rho - \\rho)$, which stays small even as "
          "$\\hat\\rho \\to 1$.", None),
         ("The KL penalty diverges at both ends:",
          r"\lim_{\hat\rho \to 0^+} D_{\mathrm{KL}} = \infty,"
          r"\qquad \lim_{\hat\rho \to 1^-} D_{\mathrm{KL}} = \infty"),
         ("Its gradient is:",
          r"\frac{\partial D_{\mathrm{KL}}}{\partial \hat\rho_j}"
          r"= -\frac{\rho}{\hat\rho_j} + \frac{1-\rho}{1-\hat\rho_j}"),
         ("which blows up as $\\hat\\rho_j$ approaches either boundary. So a unit "
          "that saturates 'on' is punished increasingly hard, and — crucially — "
          "a unit that goes <b>completely dead</b> is punished too. The squared "
          "penalty happily accepts dead units, which wastes capacity.", None),
         ("<b>The asymmetry is deliberate.</b> With $\\rho = 0.05$, the penalty is "
          "much steeper on the 'too active' side, which is the direction the "
          "reconstruction loss pushes.", None)],
        title="Why the KL penalty rather than a squared one",
    )

    idea(
        "Sparse coding is how mechanistic interpretability reads a network",
        "The reason sparsity is having a second life: a dense representation is "
        "<b>polysemantic</b> — one unit fires for several unrelated concepts, "
        "because a network with $n$ units must represent far more than $n$ "
        "features and superposes them. Training a <b>sparse autoencoder on a "
        "language model's activations</b> with a much wider, sparse code pulls "
        "those superposed features apart into individually interpretable "
        "directions. Same objective as this section, applied to a network's "
        "internals instead of to data.",
    )

    warn(
        "Dead units are the failure mode",
        "Push $\\lambda$ or $\\beta$ too hard and units switch off permanently: "
        "their activation is 0, so the reconstruction gradient never reaches "
        "them, so they stay 0. Monitor the fraction of units that are <b>never</b> "
        "active across a validation batch. The KL penalty's divergence at "
        "$\\hat\\rho \\to 0$ mitigates this; $\\ell_1$ does not, which is why "
        "modern sparse autoencoders add explicit resampling of dead units.",
    )

    anim_header("The KL sparsity penalty as a function of the actual rate")

    rho_hat = np.linspace(0.005, 0.995, 260)
    frames = []
    for rho in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
        kl = (rho * np.log(rho / rho_hat)
              + (1 - rho) * np.log((1 - rho) / (1 - rho_hat)))
        mse = (rho_hat - rho) ** 2 * 20
        frames.append(go.Frame(name=f"{rho:.2f}", data=[
            go.Scatter(x=rho_hat, y=np.clip(kl, 0, 3.2), mode="lines",
                       line=dict(color=C["primary"], width=3.5)),
            go.Scatter(x=rho_hat, y=np.clip(mse, 0, 3.2), mode="lines",
                       line=dict(color=C["muted"], width=2.5, dash="dot")),
            go.Scatter(x=[rho], y=[0], mode="markers",
                       marker=dict(size=14, color=C["success"], symbol="triangle-up")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"target ρ = {rho:.2f}   ·   KL → ∞ at both 0 and 1, so dead units "
            f"AND saturated units are both punished   ·   "
            f"the squared penalty (dotted) is flat and forgiving")])))

    rho0 = .02
    kl0 = (rho0*np.log(rho0/rho_hat) + (1-rho0)*np.log((1-rho0)/(1-rho_hat)))
    f = go.Figure(data=[
        go.Scatter(x=rho_hat, y=np.clip(kl0, 0, 3.2), mode="lines",
                   name="KL(ρ ‖ ρ̂)", line=dict(color=C["primary"], width=3.5)),
        go.Scatter(x=rho_hat, y=np.clip((rho_hat-rho0)**2*20, 0, 3.2),
                   mode="lines", name="20·(ρ̂ − ρ)²",
                   line=dict(color=C["muted"], width=2.5, dash="dot")),
        go.Scatter(x=[rho0], y=[0], mode="markers", name="target",
                   marker=dict(size=14, color=C["success"],
                               symbol="triangle-up")),
    ])
    f.update_layout(height=420, xaxis_title="actual mean activation ρ̂",
                    yaxis_title="penalty", yaxis=dict(range=[0, 3.3]),
                    title="The sparsity penalty",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="ρ = ")
    figure(f)

    code_lab(
        "ℓ1 and KL sparsity, dead units, and what the sparse features learn",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=8000, n_test=1500)
Xtr = Xtr.reshape(len(Xtr), -1).astype("float32")
Xte = Xte.reshape(len(Xte), -1).astype("float32")
if Xtr.max() > 1.5:
    Xtr /= 255.; Xte /= 255.

CODE = 300                                  # WIDER than a bottleneck AE
print(f"=== a {CODE}-unit code -- wider than any bottleneck ===")
print(f"  784 -> {CODE} is OVERCOMPLETE in the useful sense:")
print(f"  the constraint is SPARSITY, not size.")

# ============ 1. THE KL PENALTY AS A REGULARISER =======================
@keras.utils.register_keras_serializable(package="MLPlatform")
class KLSparsity(keras.regularizers.Regularizer):
    def __init__(self, rho=0.05, beta=1.0):
        self.rho, self.beta = rho, beta

    def __call__(self, activations):
        r = tf.reduce_mean(activations, axis=0)          # per-unit mean rate
        r = tf.clip_by_value(r, 1e-6, 1 - 1e-6)
        kl = (self.rho * tf.math.log(self.rho / r)
              + (1 - self.rho) * tf.math.log((1 - self.rho) / (1 - r)))
        return self.beta * tf.reduce_sum(kl)

    def get_config(self):
        return {"rho": self.rho, "beta": self.beta}

def sparse_ae(kind="none", strength=0.0, rho=0.05):
    reg = None
    if kind == "l1":
        reg = keras.regularizers.l1(strength)
    elif kind == "kl":
        reg = KLSparsity(rho=rho, beta=strength)
    enc = keras.Sequential([keras.layers.Input(shape=(784,)),
                            keras.layers.Dense(CODE, activation="sigmoid",
                                               activity_regularizer=reg)])
    dec = keras.Sequential([keras.layers.Input(shape=(CODE,)),
                            keras.layers.Dense(784, activation="sigmoid")])
    return enc, keras.Sequential([enc, dec])

def stats(enc, X):
    Z = enc.predict(X, verbose=0)
    active = Z > 0.1
    return dict(mean_rate=float(Z.mean()),
                units_per_input=float(active.sum(1).mean()),
                dead=float((active.sum(0) == 0).mean()),
                always_on=float((Z.mean(0) > .5).mean()))

print()
print(f"{'model':<28}{'test MSE':>10}{'mean act':>10}{'active/input':>14}"
      f"{'dead units':>12}{'probe':>9}")
from sklearn.linear_model import LogisticRegression
runs = {}
for nm, kind, s in [("no sparsity", "none", 0),
                    ("L1, lambda=1e-4", "l1", 1e-4),
                    ("L1, lambda=1e-3", "l1", 1e-3),
                    ("L1, lambda=1e-2 (TOO MUCH)", "l1", 1e-2),
                    ("KL, rho=.05 beta=.3", "kl", .3),
                    ("KL, rho=.05 beta=2.0", "kl", 2.0)]:
    tf.random.set_seed(0)
    e, m = sparse_ae(kind, s)
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    m.fit(Xtr, Xtr, epochs=14, batch_size=128, verbose=0)
    st = stats(e, Xte)
    Z = e.predict(Xtr[:4000], verbose=0)
    probe = LogisticRegression(max_iter=300, n_jobs=-1).fit(
        Z, ytr[:4000]).score(e.predict(Xte, verbose=0), yte)
    runs[nm] = (e, m, st, probe)
    print(f"{nm:<28}{m.evaluate(Xte, Xte, verbose=0):>10.5f}"
          f"{st['mean_rate']:>10.4f}{st['units_per_input']:>14.1f}"
          f"{st['dead']:>12.1%}{probe:>9.4f}")
print()
print(f"  with no penalty ~{runs['no sparsity'][2]['units_per_input']:.0f} of "
      f"{CODE} units fire per input -- a DENSE, polysemantic code.")
print("  the KL penalty gets the rate down without killing units.")
print("  L1 at 1e-2 kills most of them: the reconstruction gradient never")
print("  reaches a unit whose activation is exactly 0.")

# ============ 2. WHY KL BEHAVES BETTER THAN L1 =========================
print()
print("=== the gradients of the two penalties ===")
rho = 0.05
print(f"{'rho_hat':>9}{'KL penalty':>13}{'dKL/drho_hat':>16}"
      f"{'L1 gradient':>14}")
for rh in [0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 0.99]:
    kl = rho*np.log(rho/rh) + (1-rho)*np.log((1-rho)/(1-rh))
    dkl = -rho/rh + (1-rho)/(1-rh)
    print(f"{rh:>9.3f}{kl:>13.4f}{dkl:>16.2f}{1.0:>14.2f}")
print("  KL's gradient is +inf at rho_hat=1 and -inf at rho_hat=0:")
print("  it pushes ACTIVE units down AND DEAD units back up.")
print("  L1's gradient is the constant 1 -- it only ever pushes down,")
print("  which is exactly why L1 produces dead units.")

# ============ 3. WHAT ONE SPARSE UNIT RESPONDS TO ======================
print()
print("=== are sparse features more selective? ===")
for nm in ["no sparsity", "KL, rho=.05 beta=.3"]:
    e = runs[nm][0]
    Z = e.predict(Xte, verbose=0)
    # for each unit, the entropy of its activation across the 10 classes
    ents = []
    for j in range(CODE):
        w = np.array([Z[yte == c, j].mean() for c in range(10)])
        if w.sum() < 1e-8:
            continue
        p = w/w.sum()
        ents.append(-np.sum(p*np.log(p+1e-12)))
    ents = np.array(ents)
    print(f"  {nm:<28} class-selectivity entropy: mean {ents.mean():.4f} "
          f"(0 = one class only, {np.log(10):.3f} = all equally)")
    print(f"  {'':<28} most selective unit: {ents.min():.4f}, "
          f"{np.mean(ents < 1.5):.1%} of units below 1.5")

# ============ 4. THE LEARNED DICTIONARY ================================
print()
print("=== the decoder weights are a DICTIONARY of parts ===")
e_sp, m_sp = runs["KL, rho=.05 beta=.3"][:2]
Wd = m_sp.layers[1].layers[0].get_weights()[0]      # (CODE, 784)
Z = e_sp.predict(Xte, verbose=0)
usage = (Z > .1).mean(0)
top = np.argsort(-usage)[:24]
print(f"  decoder weight matrix {Wd.shape}: each ROW is a 28x28 template")
print(f"  usage rate of the 24 most-used units: "
      f"{np.round(usage[top][:8], 3)} ...")
print(f"  an input is reconstructed as a SUM of ~"
      f"{runs['KL, rho=.05 beta=.3'][2]['units_per_input']:.0f} of these templates")

grid = np.zeros((4*28, 6*28))
for i, j in enumerate(top):
    r, c = divmod(i, 6)
    w = Wd[j].reshape(28, 28)
    grid[r*28:(r+1)*28, c*28:(c+1)*28] = (w - w.min())/(np.ptp(w)+1e-9)

# ============ 5. SPARSITY FOR INTERPRETABILITY =========================
print()
print("=== the modern use: pulling apart superposed features ===")
print("  a dense layer with n units represents FAR MORE than n features")
print("  by superposing them -- so one unit fires for several unrelated")
print("  things (POLYSEMANTICITY).")
dense_Z = runs["no sparsity"][0].predict(Xte, verbose=0)
sp_Z = e_sp.predict(Xte, verbose=0)
def top_classes(Z, j, k=3):
    m_ = np.array([Z[yte == c, j].mean() for c in range(10)])
    return np.argsort(-m_)[:k], np.sort(m_)[::-1][:k]
print()
print(f"  {'unit':>6}{'dense: top-3 classes':>34}{'sparse: top-3 classes':>34}")
for j in [0, 1, 2, 3, 4]:
    dc, dv = top_classes(dense_Z, j)
    sc, sv = top_classes(sp_Z, j)
    print(f"  {j:>6}{str(list(zip(dc, dv.round(2)))):>34}"
          f"{str(list(zip(sc, sv.round(2)))):>34}")
print()
print("  the sparse units concentrate on one class; the dense ones spread.")
print("  scale this idea up and it is how sparse autoencoders are used to")
print("  interpret the internals of a language model.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("learned dictionary (24 most-used units)",
                                    "activation histogram"))
fig.add_trace(go.Heatmap(z=grid, colorscale="Greys", showscale=False), 1, 1)
fig.add_trace(go.Histogram(x=sp_Z.ravel()[::37], nbinsx=60,
                           marker=dict(color=C["primary"]), name="sparse"), 1, 2)
fig.add_trace(go.Histogram(x=dense_Z.ravel()[::37], nbinsx=60,
                           marker=dict(color=alpha(C["muted"], .6)),
                           name="dense"), 1, 2)
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_yaxes(type="log", row=1, col=2)
fig.update_layout(height=400, barmode="overlay")
''',
        key="ch17_sparse",
    )

    keypoints([
        "Sparsity is a constraint that works with a <b>wide</b> code — the "
        "bottleneck is on activity, not width.",
        "<b>KL</b> to a target rate $\\rho$ diverges at both 0 and 1, so it "
        "punishes saturated <i>and</i> dead units.",
        "$\\ell_1$ only ever pushes down, which is exactly why it produces "
        "<b>dead units</b>.",
        "The decoder weights become a <b>dictionary</b>; each input is a sum of a "
        "few templates.",
        "Sparse autoencoders on a model's activations are the standard tool for "
        "<b>mechanistic interpretability</b>.",
    ])


# ==========================================================================
def s_17_6():
    section("17.6", "Variational Autoencoders")

    lead(
        "A VAE encodes to a <b>distribution</b>, not a point, and pays a price "
        "for straying from a prior. That makes the latent space continuous and "
        "sampleable — turning an autoencoder into a genuine generative model."
    )

    sub("The setup")

    md(
        "Assume the data is generated by first drawing a latent "
        "$\\mathbf{z} \\sim p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{0}, "
        "\\mathbf{I})$ and then $\\mathbf{x} \\sim p_\\theta(\\mathbf{x} \\mid "
        "\\mathbf{z})$. We want to maximise the marginal likelihood:"
    )

    math(r"""
    \log p_\theta(\mathbf{x}) \;=\; \log \int
      p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}
    """)

    warn(
        "That integral is intractable",
        "It runs over every point of a, say, 30-dimensional latent space. Monte "
        "Carlo with samples from the prior is hopeless: for a given "
        "$\\mathbf{x}$, almost every $\\mathbf{z} \\sim p(\\mathbf{z})$ gives "
        "$p_\\theta(\\mathbf{x} \\mid \\mathbf{z}) \\approx 0$, so the estimator "
        "has astronomical variance. The VAE's answer is to <b>learn where to "
        "look</b> — an encoder $q_\\phi(\\mathbf{z} \\mid \\mathbf{x})$ that "
        "proposes the latents likely to have produced this $\\mathbf{x}$.",
    )

    sub("The ELBO")

    derive(
        [("Introduce any distribution $q_\\phi(\\mathbf{z} \\mid \\mathbf{x})$ "
          "and rewrite the log-likelihood by multiplying and dividing:",
          r"\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z}\mid\mathbf{x})\,"
          r"\frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}"
          r"{q_\phi(\mathbf{z}\mid\mathbf{x})}\,d\mathbf{z}"),
         ("That integral is an expectation, so:",
          r"\log p_\theta(\mathbf{x}) = \log\, \mathbb{E}_{q_\phi}\!\left[\,"
          r"\frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}"
          r"{q_\phi(\mathbf{z}\mid\mathbf{x})}\right]"),
         ("$\\log$ is concave, so <b>Jensen's inequality</b> "
          "($\\log \\mathbb{E}[U] \\ge \\mathbb{E}[\\log U]$) gives a lower "
          "bound:",
          r"\log p_\theta(\mathbf{x}) \;\ge\; \mathbb{E}_{q_\phi}\bigl[\log "
          r"p_\theta(\mathbf{x}\mid\mathbf{z})\bigr] - D_{\mathrm{KL}}"
          r"\bigl(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, p(\mathbf{z})\bigr)"),
         ("This is the <b>evidence lower bound</b>, the ELBO. The first term is "
          "<b>reconstruction</b> (how well $\\mathbf{z}$ explains "
          "$\\mathbf{x}$); the second is <b>regularisation</b> (how far the "
          "encoder's output strays from the prior).", None),
         ("<b>The gap is exactly a KL divergence:</b>",
          r"\log p_\theta(\mathbf{x}) - \mathrm{ELBO} = D_{\mathrm{KL}}"
          r"\bigl(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, "
          r"p_\theta(\mathbf{z}\mid\mathbf{x})\bigr) \;\ge\; 0"),
         ("So maximising the ELBO does two things at once: it pushes up the "
          "likelihood, <i>and</i> it pushes $q_\\phi$ toward the true posterior. "
          "The bound is tight exactly when the encoder recovers the true "
          "posterior.", None)],
        title="Deriving the ELBO from Jensen's inequality",
    )

    md("With a Gaussian encoder $q_\\phi = \\mathcal{N}(\\boldsymbol\\mu, "
       "\\mathrm{diag}(\\boldsymbol\\sigma^2))$ and a standard normal prior, the "
       "KL term has a closed form:")

    math(r"""
    D_{\mathrm{KL}}\bigl(\mathcal{N}(\boldsymbol\mu, \boldsymbol\sigma^{2})
      \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{I})\bigr)
    \;=\; \tfrac{1}{2}\sum_{j=1}^{d}
      \Bigl(\mu_j^{2} + \sigma_j^{2} - 1 - \log \sigma_j^{2}\Bigr)
    """)

    sub("The reparameterisation trick")

    pitfall(
        "You cannot backpropagate through a sampling operation",
        "$\\mathbf{z} \\sim \\mathcal{N}(\\boldsymbol\\mu, \\boldsymbol\\sigma^2)$ "
        "is a random node: $\\partial \\mathbf{z} / \\partial \\boldsymbol\\mu$ "
        "is not defined, because sampling is not a differentiable function of its "
        "parameters. Without a fix, no gradient reaches the encoder at all.",
    )

    math(r"""
    \mathbf{z} \;=\; \boldsymbol\mu \;+\; \boldsymbol\sigma \otimes
      \boldsymbol\varepsilon,
    \qquad \boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
    """)

    proof(
        "The trick moves the randomness off the gradient path",
        "$\\mathbf{z}$ has exactly the same distribution as before, but now the "
        "stochastic part $\\boldsymbol\\varepsilon$ is an <b>input</b>, not a "
        "node depending on the parameters. The derivatives become trivial: "
        "$\\partial \\mathbf{z}/\\partial \\boldsymbol\\mu = \\mathbf{I}$ and "
        "$\\partial \\mathbf{z}/\\partial \\boldsymbol\\sigma = "
        "\\mathrm{diag}(\\boldsymbol\\varepsilon)$. The gradient flows straight "
        "through. This is a general technique — the <i>pathwise</i> gradient "
        "estimator — and it has far lower variance than the score-function "
        "(REINFORCE) alternative, which is why VAEs train and most discrete "
        "latent-variable models are painful.",
    )

    codenote(
        "Predict $\\log \\sigma^2$, never $\\sigma$",
        "A network output is unconstrained, but $\\sigma$ must be positive. "
        "Predicting $\\gamma = \\log \\sigma^2$ and using "
        "$\\sigma = \\exp(\\gamma/2)$ makes any real output valid, keeps the KL "
        "term numerically stable, and gives well-scaled gradients. Predicting "
        "$\\sigma$ directly and clipping it at zero is a classic source of NaNs.",
    )

    sub("The two failure modes")

    table(
        ["Failure", "Symptom", "Cause", "Fix"],
        [["<b>Posterior collapse</b>",
          "KL term → 0; the decoder ignores $\\mathbf{z}$; all samples look "
          "identical",
          "The decoder is powerful enough to model the data alone, so the "
          "cheapest solution sets $q = p$",
          "KL <b>warm-up</b> (anneal $\\beta$ from 0), free bits, or a weaker "
          "decoder"],
         ["<b>Blurry samples</b>",
          "Reconstructions look like an average of plausible images",
          "A Gaussian likelihood means a squared loss, whose optimum is the "
          "<b>conditional mean</b> — averaging over all valid completions",
          "A different likelihood, a perceptual loss, or a GAN/diffusion "
          "decoder"]],
    )

    idea(
        "Blurriness is a property of the loss, not a lack of capacity",
        "If several images are equally plausible completions, the squared loss is "
        "minimised by their <b>average</b>, which is blurry — no amount of extra "
        "capacity changes that, because the blurry answer genuinely <i>is</i> the "
        "optimum of the stated objective. GANs (§17.7) avoid it by never writing "
        "down a pixel-space likelihood; diffusion models (§17.9) avoid it by "
        "making each step's conditional distribution nearly unimodal, so the mean "
        "is a good answer.",
    )

    anim_header("β-VAE: trading reconstruction against latent structure")

    rng = np.random.default_rng(5)
    n_pt = 320
    theta = rng.uniform(0, 2 * np.pi, n_pt)
    lat_true = np.column_stack([np.cos(theta), np.sin(theta)])
    betas = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
    frames = []
    for b in betas:
        shrink = 1.0 / (1.0 + b * .55)
        spread = 1.0 + b * .12
        Z = lat_true * (2.4 * shrink) + rng.normal(0, .12 * spread, lat_true.shape)
        rec = 0.02 + b * 0.028
        kl = 2.6 / (1 + b * 1.2)
        frames.append(go.Frame(name=f"{b:g}", data=[
            go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers",
                       marker=dict(size=6, color=theta, colorscale="Viridis",
                                   showscale=False)),
            go.Scatter(x=2.2*np.cos(np.linspace(0, 2*np.pi, 100)),
                       y=2.2*np.sin(np.linspace(0, 2*np.pi, 100)),
                       mode="lines",
                       line=dict(color=alpha(C["line"], .7), width=1.5,
                                 dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"β = {b:g}   ·   reconstruction error {rec:.3f}   ·   "
            f"KL to the prior {kl:.3f}   ·   "
            + ("β = 0 is a plain autoencoder: no latent structure at all"
               if b == 0 else
               "POSTERIOR COLLAPSE — the code carries nothing" if b >= 20
               else "the usable range"),
            color=C["danger"] if (b == 0 or b >= 20) else C["success"])])))

    Z0 = lat_true * 2.4 + rng.normal(0, .12, lat_true.shape)
    f = go.Figure(data=[
        go.Scatter(x=Z0[:, 0], y=Z0[:, 1], mode="markers", name="codings",
                   marker=dict(size=6, color=theta, colorscale="Viridis",
                               showscale=False)),
        go.Scatter(x=2.2*np.cos(np.linspace(0, 2*np.pi, 100)),
                   y=2.2*np.sin(np.linspace(0, 2*np.pi, 100)), mode="lines",
                   name="prior (1σ)",
                   line=dict(color=alpha(C["line"], .7), width=1.5, dash="dot")),
    ])
    f.update_layout(height=460, xaxis=dict(range=[-3.4, 3.4], title="z₁"),
                    yaxis=dict(range=[-3.4, 3.4], title="z₂",
                               scaleanchor="x"),
                    title="The latent space as β increases")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="β = ")
    figure(f, "β controls the trade-off explicitly. Too small and the latent "
              "space has holes you cannot sample from; too large and it carries "
              "no information at all.")

    code_lab(
        "A VAE from scratch: ELBO, reparameterisation, and posterior collapse",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=10000, n_test=2000)
Xtr = Xtr.reshape(len(Xtr), -1).astype("float32")
Xte = Xte.reshape(len(Xte), -1).astype("float32")
if Xtr.max() > 1.5:
    Xtr /= 255.; Xte /= 255.

LATENT = 12

# ============ 1. THE REPARAMETERISATION TRICK ==========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class Sampling(keras.layers.Layer):
    """z = mu + sigma * eps -- the randomness becomes an INPUT."""
    def call(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(tf.shape(log_var))
        return mean + tf.exp(0.5 * log_var) * eps    # log_var, NOT sigma

print("=== why the trick is necessary ===")
mu = tf.Variable([[1.0, 2.0]])
lv = tf.Variable([[0.0, 0.0]])
with tf.GradientTape() as tape:
    z_bad = tf.random.normal((1, 2), mean=mu, stddev=tf.exp(0.5*lv))
    loss = tf.reduce_sum(z_bad**2)
g = tape.gradient(loss, [mu, lv])
print(f"  sampling directly     -> gradient wrt mu: {g[0]}")
with tf.GradientTape() as tape:
    z_good = Sampling()([mu, lv])
    loss = tf.reduce_sum(z_good**2)
g = tape.gradient(loss, [mu, lv])
print(f"  reparameterised       -> gradient wrt mu: {g[0].numpy()}")
print(f"                        -> gradient wrt log_var: {g[1].numpy().round(3)}")
print("  without the trick NO gradient reaches the encoder at all.")

# ============ 2. THE MODEL =============================================
def build_vae(latent=LATENT, beta=1.0):
    inp = keras.layers.Input(shape=(784,))
    h = keras.layers.Dense(256, activation="selu")(inp)
    h = keras.layers.Dense(128, activation="selu")(h)
    mean = keras.layers.Dense(latent, name="mean")(h)
    log_var = keras.layers.Dense(latent, name="log_var")(h)
    z = Sampling()([mean, log_var])
    encoder = keras.Model(inp, [mean, log_var, z])

    dz = keras.layers.Input(shape=(latent,))
    g = keras.layers.Dense(128, activation="selu")(dz)
    g = keras.layers.Dense(256, activation="selu")(g)
    out = keras.layers.Dense(784, activation="sigmoid")(g)
    decoder = keras.Model(dz, out)
    return encoder, decoder

@keras.utils.register_keras_serializable(package="MLPlatform")
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kw):
        super().__init__(**kw)
        self.encoder, self.decoder, self.beta = encoder, decoder, beta
        self.beta_var = tf.Variable(beta, trainable=False, dtype="float32")
        self.rec_tracker = keras.metrics.Mean(name="rec")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.rec_tracker, self.kl_tracker]

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(x)
            recon = self.decoder(z)
            # reconstruction: 784 * binary cross-entropy (a Bernoulli likelihood)
            rec = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(x[:, :, None],
                                                 recon[:, :, None]), axis=1))
            # the CLOSED-FORM KL to N(0, I)
            kl = tf.reduce_mean(-0.5 * tf.reduce_sum(
                1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))
            loss = rec + self.beta_var * kl          # the negative ELBO
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_weights),
                self.trainable_weights))
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        return {"rec": self.rec_tracker.result(), "kl": self.kl_tracker.result()}

    def call(self, x):
        return self.decoder(self.encoder(x)[2])

# ============ 3. THE BETA TRADE-OFF ====================================
print()
print("=== beta controls reconstruction vs latent structure ===")
print(f"{'beta':>8}{'reconstruction':>17}{'KL':>10}{'active dims':>14}"
       f"{'sample quality*':>17}")
results = {}
for beta in [0.0, 0.5, 1.0, 4.0, 30.0]:
    tf.random.set_seed(0)
    e, d = build_vae()
    v = VAE(e, d, beta)
    v.compile(optimizer=keras.optimizers.Adam(1e-3))
    h = v.fit(Xtr, epochs=16, batch_size=128, verbose=0)
    mean, log_var, _ = e.predict(Xte, verbose=0)
    # a latent dim is "active" if its posterior varies across inputs
    active = int((mean.std(0) > 0.1).sum())
    # sample quality proxy: how close decoded prior samples are to real data
    zs = np.random.default_rng(0).normal(0, 1, (500, LATENT)).astype("float32")
    gen = d.predict(zs, verbose=0)
    qual = float(np.mean(np.abs(gen.mean(0) - Xte.mean(0))))
    results[beta] = (v, e, d, h)
    print(f"{beta:>8.1f}{h.history['rec'][-1]:>17.3f}{h.history['kl'][-1]:>10.3f}"
          f"{active:>10}/{LATENT}{qual:>17.4f}")
print("  *mean absolute difference between generated and real pixel means")
print()
print("  beta=0   : a plain autoencoder. KL is huge, prior samples are garbage.")
print("  beta=30  : POSTERIOR COLLAPSE. KL ~ 0, active dims ~ 0, every")
print("             sample is the SAME average garment.")

# ============ 4. POSTERIOR COLLAPSE, DIAGNOSED =========================
print()
print("=== diagnosing posterior collapse ===")
print(f"{'beta':>8}{'per-dim KL (sorted, first 8)':>44}{'collapsed dims':>17}")
for beta in [0.5, 1.0, 4.0, 30.0]:
    e = results[beta][1]
    mean, log_var, _ = e.predict(Xte, verbose=0)
    kl_per_dim = np.mean(-0.5*(1 + log_var - mean**2 - np.exp(log_var)), 0)
    s = np.sort(kl_per_dim)[::-1]
    print(f"{beta:>8.1f}{str(np.round(s[:8], 3)):>44}"
          f"{int((kl_per_dim < 0.01).sum()):>13}/{LATENT}")
print("  a collapsed dimension has KL ~ 0: q(z|x) = p(z) for EVERY x,")
print("  so that dimension carries no information about the input at all.")

# ============ 5. KL WARM-UP: THE STANDARD FIX ==========================
print()
print("=== KL warm-up (annealing beta from 0 to 1) ===")
class KLWarmup(keras.callbacks.Callback):
    def __init__(self, n_epochs): self.n = n_epochs
    def on_epoch_begin(self, epoch, logs=None):
        self.model.beta_var.assign(min(1.0, (epoch+1)/self.n))

tf.random.set_seed(0)
e, d = build_vae()
v = VAE(e, d, 1.0)
v.beta_var.assign(0.0)
v.compile(optimizer=keras.optimizers.Adam(1e-3))
hw = v.fit(Xtr, epochs=16, batch_size=128, verbose=0,
           callbacks=[KLWarmup(8)])
mean, log_var, _ = e.predict(Xte, verbose=0)
kl_pd = np.mean(-0.5*(1 + log_var - mean**2 - np.exp(log_var)), 0)
print(f"  beta=1 from the start : "
      f"{int((np.mean(-0.5*(1 + results[1.0][1].predict(Xte, verbose=0)[1] - results[1.0][1].predict(Xte, verbose=0)[0]**2 - np.exp(results[1.0][1].predict(Xte, verbose=0)[1])), 0) < .01).sum())}"
      f" collapsed dims")
print(f"  with warm-up          : {int((kl_pd < .01).sum())} collapsed dims")
print(f"  final rec {hw.history['rec'][-1]:.3f}, kl {hw.history['kl'][-1]:.3f}")
print("  the decoder learns to USE z before the KL pressure arrives.")

# ============ 6. GENERATING, AND INTERPOLATING =========================
print()
print("=== the payoff: a VAE can SAMPLE, a plain autoencoder cannot ===")
best_e, best_d = results[1.0][1], results[1.0][2]
plain_e, plain_d = results[0.0][1], results[0.0][2]
zs = np.random.default_rng(1).normal(0, 1, (400, LATENT)).astype("float32")
for nm, dec in [("beta=1 VAE", best_d), ("beta=0 (plain AE)", plain_d)]:
    gen = dec.predict(zs, verbose=0)
    print(f"  {nm:<20} generated pixel mean {gen.mean():.4f} "
          f"(real {Xte.mean():.4f}), std {gen.std():.4f} "
          f"(real {Xte.std():.4f})")
print("  the plain autoencoder's latent space has HOLES: a random z lands")
print("  somewhere it was never trained on, and decodes to nothing.")

# --- semantic interpolation ------------------------------------------
print()
print("=== interpolating between two garments ===")
i, j = 0, 7
m1, _, _ = best_e.predict(Xte[i:i+1], verbose=0)
m2, _, _ = best_e.predict(Xte[j:j+1], verbose=0)
alphas = np.linspace(0, 1, 9)
path = np.vstack([(1-a)*m1 + a*m2 for a in alphas]).astype("float32")
interp = best_d.predict(path, verbose=0)
pix = np.vstack([(1-a)*Xte[i] + a*Xte[j] for a in alphas])
print(f"  latent interpolation : mean image gradient "
      f"{np.abs(np.diff(interp, axis=0)).mean():.5f}")
print(f"  PIXEL interpolation  : mean image gradient "
      f"{np.abs(np.diff(pix, axis=0)).mean():.5f}")
print("  pixel interpolation ghosts one image over the other.")
print("  latent interpolation passes through PLAUSIBLE garments.")

strip = np.zeros((2*28, 9*28))
for k in range(9):
    strip[:28, k*28:(k+1)*28] = interp[k].reshape(28, 28)
    strip[28:, k*28:(k+1)*28] = pix[k].reshape(28, 28)

import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(z=strip, colorscale="Greys", reversescale=True,
                           showscale=False))
fig.update_layout(height=300, xaxis=dict(visible=False),
                  yaxis=dict(visible=False, autorange="reversed"),
                  title="latent interpolation (top) vs pixel interpolation (bottom)")
''',
        key="ch17_vae",
    )

    quiz(
        "A VAE's KL term goes to zero during training and every sample looks "
        "identical. What has happened?",
        ["The learning rate is too low",
         "<b>Posterior collapse</b> — the decoder learned to ignore "
         "$\\mathbf{z}$, so $q_\\phi(\\mathbf{z}\\mid\\mathbf{x}) = "
         "p(\\mathbf{z})$ is the cheapest solution",
         "The reparameterisation trick was implemented wrongly",
         "The latent dimension is too large"],
        1,
        "When the decoder is powerful enough to model the data without the "
        "latent, setting $q = p$ costs nothing in reconstruction and zeroes the "
        "KL term — so gradient descent takes it. The standard fix is KL warm-up: "
        "anneal $\\beta$ from 0 so the decoder learns to use $\\mathbf{z}$ before "
        "the KL pressure arrives.",
        key="ch17q1",
    )

    keypoints([
        "A VAE encodes to a <b>distribution</b>; the ELBO = reconstruction "
        "$-$ KL to the prior, from Jensen's inequality.",
        "The gap between $\\log p_\\theta(\\mathbf{x})$ and the ELBO is exactly "
        "$D_{\\mathrm{KL}}(q_\\phi \\Vert p_\\theta(\\mathbf{z}\\mid\\mathbf{x}))$.",
        "The <b>reparameterisation trick</b> $\\mathbf{z} = \\boldsymbol\\mu + "
        "\\boldsymbol\\sigma \\otimes \\boldsymbol\\varepsilon$ makes sampling "
        "differentiable.",
        "<b>Posterior collapse</b>: KL → 0 and the latent is ignored. Fix with "
        "KL warm-up.",
        "<b>Blurriness is the loss, not the capacity</b> — a squared loss is "
        "optimised by the conditional mean.",
    ])

# ==========================================================================
def s_17_7():
    section("17.7", "Generative Adversarial Networks")

    lead(
        "Stop writing down a likelihood. Train a second network to tell real from "
        "fake, and train the generator to fool it. The loss becomes <b>learned</b> "
        "— which is why GAN samples are sharp, and why they are so hard to train."
    )

    sub("The minimax game")

    math(r"""
    \min_{G}\max_{D}\;\;
      V(D, G) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}
        \bigl[\log D(\mathbf{x})\bigr]
      \;+\; \mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}
        \bigl[\log\bigl(1 - D(G(\mathbf{z}))\bigr)\bigr]
    """)

    where({
        r"D": "the <b>discriminator</b> — outputs the probability that its input "
              "is real",
        r"G": "the <b>generator</b> — maps noise $\\mathbf{z}$ to a sample",
        r"p_{\mathbf{z}}": "a simple prior, usually "
                           "$\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$",
    })

    derive(
        [("<b>What the optimal discriminator is.</b> Fix $G$, and let $p_g$ be "
          "the generator's distribution. The inner objective is:",
          r"V = \int_{\mathbf{x}} p_{\text{data}}(\mathbf{x})\log D(\mathbf{x})"
          r" + p_g(\mathbf{x})\log\bigl(1 - D(\mathbf{x})\bigr)\, d\mathbf{x}"),
         ("The integrand is maximised pointwise. Setting the derivative of "
          "$a\\log d + b\\log(1-d)$ to zero gives $d = a/(a+b)$, so:",
          r"D^{\star}(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}"
          r"{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}"),
         ("Substituting $D^\\star$ back into $V$ and adding and subtracting "
          "$\\log 2$ twice yields:",
          r"V(D^{\star}, G) = -\log 4 + 2\,D_{\mathrm{JS}}"
          r"\bigl(p_{\text{data}} \,\Vert\, p_g\bigr)"),
         ("where $D_{\\mathrm{JS}}$ is the <b>Jensen–Shannon divergence</b>. It is "
          "non-negative and zero only when $p_g = p_{\\text{data}}$.", None),
         ("<b>So at the global optimum the generator has matched the data "
          "distribution exactly</b>, and $D^\\star = 1/2$ everywhere — the "
          "discriminator can do no better than a coin flip.", None),
         ("<b>But this proof assumes an optimal $D$ at every step and updates in "
          "function space.</b> In practice $D$ is a finite network trained for a "
          "few steps, and the updates are in parameter space. Neither assumption "
          "holds, which is why the theory says 'convergence' and the practice "
          "says 'good luck'.", None)],
        title="Why the GAN objective is the Jensen–Shannon divergence",
    )

    sub("The saturating-gradient problem")

    pitfall(
        "The generator's textbook loss has no gradient exactly when it needs one",
        "Early in training, $G$ is bad and $D$ rejects its samples confidently: "
        "$D(G(\\mathbf{z})) \\approx 0$. The generator's loss "
        "$\\log(1 - D(G(\\mathbf{z})))$ is then <b>flat</b> — its derivative "
        "$-1/(1 - D)$ approaches $-1$ while $\\partial D/\\partial G$ vanishes, so "
        "almost no gradient reaches $G$. The standard fix, in the original paper: "
        "<b>maximise $\\log D(G(\\mathbf{z}))$</b> instead of minimising "
        "$\\log(1 - D(G(\\mathbf{z})))$. Same fixed point, gradient largest "
        "exactly where the generator is worst. In code this is just 'train $G$ "
        "with the fake batch labelled <b>real</b>'.",
    )

    sub("The failure modes")

    table(
        ["Failure", "Symptom", "Why", "Mitigation"],
        [["<b>Mode collapse</b>",
          "$G$ emits one or a handful of outputs regardless of $\\mathbf{z}$",
          "$G$ finds one sample $D$ cannot reject and has no incentive for "
          "diversity — the loss never mentions coverage",
          "Minibatch discrimination, unrolled GANs, WGAN-GP, or a diversity term"],
         ["<b>Non-convergence / oscillation</b>",
          "Losses cycle forever; sample quality goes up and down",
          "It is a <b>game</b>, not an optimisation: the vector field of the "
          "joint update can be rotational and orbit the equilibrium",
          "Two time-scale updates (TTUR), spectral norm, smaller learning rates"],
         ["<b>Discriminator wins</b>",
          "$D$ loss → 0, $G$ loss explodes, samples freeze",
          "A perfect $D$ has vanishing gradient (above); "
          "$D_{\\mathrm{JS}}$ is constant when the supports are disjoint",
          "Non-saturating loss, label smoothing, noise on $D$'s inputs, WGAN"],
         ["<b>Generator wins</b>", "$D$ loss explodes; samples degrade",
          "$D$ is too weak to provide signal",
          "Train $D$ more steps per $G$ step"]],
    )

    idea(
        "Why Wasserstein GAN was such an improvement",
        "If $p_g$ and $p_{\\text{data}}$ have disjoint supports — which is the "
        "normal situation for two low-dimensional manifolds in a "
        "high-dimensional pixel space — then $D_{\\mathrm{JS}}$ is the "
        "<b>constant</b> $\\log 2$, and a constant has zero gradient. That is a "
        "structural fact about the objective, not a training difficulty. The "
        "<b>Wasserstein</b> (earth-mover) distance is finite and has a useful "
        "gradient even for disjoint supports, because it measures how far mass "
        "must be moved rather than how much the distributions overlap. WGAN "
        "replaces the classifier with a 1-Lipschitz <i>critic</i> "
        "(enforced by gradient penalty), and the critic's loss becomes an actual "
        "measure of sample quality — the first GAN loss that correlated with what "
        "you saw.",
    )

    sub("Practical rules")

    table(
        ["Rule", "Reason"],
        [["Use the <b>non-saturating</b> generator loss",
          "Gradient is largest when $G$ is worst"],
         ["<b>Normalise inputs to $[-1, 1]$</b> and end $G$ with $\\tanh$",
          "Symmetric range, and $\\tanh$ saturates gracefully"],
         ["<b>LeakyReLU</b> in $D$", "A dead ReLU in $D$ kills $G$'s gradient too"],
         ["<b>No pooling</b> — strided convolutions both ways",
          "Learned up/downsampling; DCGAN's finding"],
         ["<b>Batch norm in $G$, not in $D$'s first or last layer</b>",
          "Stabilises $G$; in $D$ it leaks batch statistics between real and fake"],
         ["<b>Adam with $\\beta_1 = 0.5$</b>",
          "The default 0.9 momentum overshoots in a game"],
         ["<b>One-sided label smoothing</b> (real = 0.9)",
          "Stops $D$ becoming over-confident and saturating"],
         ["<b>Separate batches</b> for real and fake in $D$",
          "Mixing them lets batch norm cheat"]],
    )

    anim_header("A GAN as a game: orbiting instead of converging")

    # the classic Dirac-GAN: V(d, g) = d*g, whose gradient flow is a pure rotation
    steps = 90
    d, g = 1.0, 1.0
    lr = 0.16
    path_sim, path_alt = [(d, g)], [(1.0, 1.0)]
    for _ in range(steps):
        d_new = d + lr * g
        g_new = g - lr * d
        d, g = d_new, g_new                     # SIMULTANEOUS update
        path_sim.append((d, g))
    d, g = 1.0, 1.0
    for _ in range(steps):
        d = d + lr * g                          # ALTERNATING update
        g = g - lr * d
        path_alt.append((d, g))
    path_sim = np.array(path_sim); path_alt = np.array(path_alt)

    frames = []
    for k in range(1, steps + 1):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=path_sim[:k, 0], y=path_sim[:k, 1], mode="lines+markers",
                       line=dict(color=C["danger"], width=2.5),
                       marker=dict(size=4)),
            go.Scatter(x=path_alt[:k, 0], y=path_alt[:k, 1], mode="lines+markers",
                       line=dict(color=C["success"], width=2.5),
                       marker=dict(size=4)),
            go.Scatter(x=[0], y=[0], mode="markers",
                       marker=dict(size=15, color=C["ink"], symbol="x")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {k}   ·   simultaneous ‖(d,g)‖ = "
            f"{np.linalg.norm(path_sim[k]):.3f} (SPIRALS OUT)   ·   "
            f"alternating ‖(d,g)‖ = {np.linalg.norm(path_alt[k]):.3f}")])))

    f = go.Figure(data=[
        go.Scatter(x=path_sim[:1, 0], y=path_sim[:1, 1], mode="lines+markers",
                   name="simultaneous updates",
                   line=dict(color=C["danger"], width=2.5)),
        go.Scatter(x=path_alt[:1, 0], y=path_alt[:1, 1], mode="lines+markers",
                   name="alternating updates",
                   line=dict(color=C["success"], width=2.5)),
        go.Scatter(x=[0], y=[0], mode="markers", name="equilibrium",
                   marker=dict(size=15, color=C["ink"], symbol="x")),
    ])
    f.update_layout(height=470, xaxis_title="discriminator parameter",
                    yaxis_title="generator parameter",
                    xaxis=dict(range=[-3.2, 3.2]),
                    yaxis=dict(range=[-3.2, 3.2], scaleanchor="x"),
                    title="Gradient descent–ascent on V(d, g) = d·g",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="step ")
    figure(f, "Even on the simplest possible GAN, gradient descent–ascent orbits "
              "the equilibrium rather than reaching it. This is a property of "
              "games, not a bug in your code.")

    code_lab(
        "A GAN from scratch: the game, mode collapse, and the tricks that help",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE OPTIMAL DISCRIMINATOR, VERIFIED ===================
print("=== D*(x) = p_data(x) / (p_data(x) + p_g(x)) ===")
rng = np.random.default_rng(0)
real = rng.normal(0.0, 1.0, 8000).astype("float32")
fake = rng.normal(1.6, 1.0, 8000).astype("float32")

tf.random.set_seed(0)
D = keras.Sequential([keras.layers.Input(shape=(1,)),
                      keras.layers.Dense(64, activation="relu"),
                      keras.layers.Dense(64, activation="relu"),
                      keras.layers.Dense(1, activation="sigmoid")])
D.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(2e-3))
X = np.concatenate([real, fake])[:, None]
y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))])
D.fit(X, y, epochs=25, batch_size=256, verbose=0)

grid = np.linspace(-3.5, 5, 200).astype("float32")[:, None]
def npdf(x, m, s): return np.exp(-((x-m)**2)/(2*s*s))/(s*np.sqrt(2*np.pi))
p_data = npdf(grid, 0.0, 1.0); p_g = npdf(grid, 1.6, 1.0)
theory = p_data/(p_data + p_g)
learned = D.predict(grid, verbose=0)
print(f"  max |learned D - theoretical D*| = "
      f"{np.abs(learned - theory).max():.4f}")
print(f"  mean absolute error              = "
      f"{np.abs(learned - theory).mean():.4f}")
print("  a trained classifier IS a density ratio estimator.")

# ============ 2. SATURATING vs NON-SATURATING GENERATOR LOSS ===========
print()
print("=== the generator's gradient when D is confident ===")
print(f"{'D(G(z))':>10}{'log(1-D) loss':>16}{'|d/dD|':>10}"
      f"{'-log(D) loss':>15}{'|d/dD|':>10}")
for dv in [0.001, 0.01, 0.1, 0.5, 0.9]:
    print(f"{dv:>10.3f}{np.log(1-dv):>16.4f}{1/(1-dv):>10.3f}"
          f"{-np.log(dv):>15.4f}{1/dv:>10.2f}")
print("  when D(G(z)) = 0.001 (G is terrible), the SATURATING loss has")
print("  gradient magnitude 1.0 and the NON-SATURATING one has 1000.")
print("  the fix in code: train G with the fakes labelled REAL.")

# ============ 3. A GAN ON A 2-D MIXTURE (mode collapse is VISIBLE) =====
print()
print("="*66)
print("A GAN on an 8-mode mixture -- where mode collapse is measurable")
print("="*66)
N_MODES = 8
ang = np.linspace(0, 2*np.pi, N_MODES, endpoint=False)
centres = np.column_stack([2.4*np.cos(ang), 2.4*np.sin(ang)])
def sample_real(n, rng):
    k = rng.integers(0, N_MODES, n)
    return (centres[k] + rng.normal(0, .13, (n, 2))).astype("float32")

REAL = sample_real(12000, rng)
LATENT = 16

def make_G(width=128):
    return keras.Sequential([keras.layers.Input(shape=(LATENT,)),
                             keras.layers.Dense(width, activation="relu"),
                             keras.layers.Dense(width, activation="relu"),
                             keras.layers.Dense(2)])

def make_D(width=128, leaky=True, dropout=0.0):
    act = (lambda: keras.layers.LeakyReLU(0.2)) if leaky else \\
          (lambda: keras.layers.ReLU())
    layers = [keras.layers.Input(shape=(2,)),
              keras.layers.Dense(width), act()]
    if dropout: layers.append(keras.layers.Dropout(dropout))
    layers += [keras.layers.Dense(width), act(),
               keras.layers.Dense(1)]                  # LOGITS
    return keras.Sequential(layers)

bce = keras.losses.BinaryCrossentropy(from_logits=True)

def train_gan(steps=600, batch=128, non_saturating=True, d_steps=1,
              label_smooth=0.0, lr_g=2e-4, lr_d=2e-4, seed=0):
    tf.random.set_seed(seed)
    r = np.random.default_rng(seed)
    G, Dnet = make_G(), make_D()
    optG = keras.optimizers.Adam(lr_g, beta_1=0.5)
    optD = keras.optimizers.Adam(lr_d, beta_1=0.5)
    hist = []
    for step in range(steps):
        # ---- discriminator ----
        for _ in range(d_steps):
            idx = r.integers(0, len(REAL), batch)
            xr = REAL[idx]
            z = r.normal(0, 1, (batch, LATENT)).astype("float32")
            xf = G(z, training=False)
            with tf.GradientTape() as t:
                lr_ = bce(tf.ones((batch, 1))*(1-label_smooth),
                          Dnet(xr, training=True))
                lf_ = bce(tf.zeros((batch, 1)), Dnet(xf, training=True))
                ld = lr_ + lf_
            optD.apply_gradients(zip(t.gradient(ld, Dnet.trainable_weights),
                                     Dnet.trainable_weights))
        # ---- generator ----
        z = r.normal(0, 1, (batch, LATENT)).astype("float32")
        with tf.GradientTape() as t:
            out = Dnet(G(z, training=True), training=False)
            if non_saturating:
                lg = bce(tf.ones((batch, 1)), out)        # fakes labelled REAL
            else:
                lg = -bce(tf.zeros((batch, 1)), out)      # the textbook loss
        optG.apply_gradients(zip(t.gradient(lg, G.trainable_weights),
                                 G.trainable_weights))
        if step % 200 == 0:
            hist.append((step, float(ld), float(lg)))
    return G, Dnet, hist

def mode_coverage(G, n=3000, seed=1):
    z = np.random.default_rng(seed).normal(0, 1, (n, LATENT)).astype("float32")
    S = G.predict(z, verbose=0)
    d = ((S[:, None, :] - centres[None])**2).sum(-1)
    near = d.min(1) < .36                       # within ~3 sigma of a mode
    assigned = d.argmin(1)[near]
    covered = len(np.unique(assigned))
    counts = np.bincount(assigned, minlength=N_MODES) / max(1, near.sum())
    kl = float(np.sum(np.where(counts > 0,
                               counts*np.log(counts*N_MODES + 1e-12), 0)))
    return covered, float(near.mean()), kl, S

print()
print(f"{'setup':<38}{'modes':>8}{'on-manifold':>14}{'balance KL':>13}")
runs = {}
for nm, kw in [("non-saturating loss (standard)", dict()),
               ("SATURATING loss (textbook)", dict(non_saturating=False)),
               ("+ label smoothing 0.1", dict(label_smooth=.1)),
               ("+ 3 D-steps per G-step", dict(d_steps=3)),
               ("+ TTUR (D lr 4e-4)", dict(lr_d=4e-4))]:
    t0 = time.perf_counter()
    G, Dn, h = train_gan(**kw)
    cov, onman, kl, S = mode_coverage(G)
    runs[nm] = (G, S, cov)
    print(f"{nm:<38}{cov:>5}/{N_MODES}{onman:>14.1%}{kl:>13.4f}")
print("  'modes' = how many of the 8 the generator actually produces.")
print("  'balance KL' = 0 means all modes equally represented.")
print("  the saturating loss usually collapses -- no gradient early on.")

# ============ 4. MODE COLLAPSE, WATCHED ================================
print()
print("=== how coverage evolves during training ===")
tf.random.set_seed(0)
r = np.random.default_rng(0)
G, Dnet = make_G(), make_D()
optG = keras.optimizers.Adam(2e-4, beta_1=.5)
optD = keras.optimizers.Adam(2e-4, beta_1=.5)
track = []
for step in range(801):
    idx = r.integers(0, len(REAL), 128)
    z = r.normal(0, 1, (128, LATENT)).astype("float32")
    with tf.GradientTape() as t:
        ld = bce(tf.ones((128,1)), Dnet(REAL[idx], training=True)) + \\
             bce(tf.zeros((128,1)), Dnet(G(z, training=False), training=True))
    optD.apply_gradients(zip(t.gradient(ld, Dnet.trainable_weights),
                             Dnet.trainable_weights))
    z = r.normal(0, 1, (128, LATENT)).astype("float32")
    with tf.GradientTape() as t:
        lg = bce(tf.ones((128,1)), Dnet(G(z, training=True), training=False))
    optG.apply_gradients(zip(t.gradient(lg, G.trainable_weights),
                             G.trainable_weights))
    if step % 200 == 0:
        cov, onman, kl, _ = mode_coverage(G, n=1500)
        track.append((step, float(ld), float(lg), cov, onman))
print(f"{'step':>7}{'D loss':>10}{'G loss':>10}{'modes':>8}{'on-manifold':>14}")
for s, a, b, c, o in track:
    print(f"{s:>7}{a:>10.4f}{b:>10.4f}{c:>5}/{N_MODES}{o:>14.1%}")
print()
print("  notice the losses do NOT decrease monotonically. In a GAN they")
print("  cannot: an improving D raises G's loss and vice versa.")
print("  THE LOSS IS NOT A PROGRESS METRIC. Only the samples are.")

# ============ 5. THE GAME NEVER SETTLES ================================
print()
print("=== gradient descent-ascent on the simplest game, V(d,g) = d*g ===")
print(f"{'update rule':<24}{'||(d,g)|| after 60 steps':>28}")
for nm, simultaneous in [("simultaneous", True), ("alternating", False)]:
    d = g = 1.0
    for _ in range(60):
        if simultaneous:
            d, g = d + .16*g, g - .16*d
        else:
            d = d + .16*g
            g = g - .16*d
    print(f"{nm:<24}{np.hypot(d, g):>28.4f}")
print("  started at ||(1,1)|| = 1.414. Simultaneous updates SPIRAL OUT")
print("  from the equilibrium. This is a property of games, not a bug.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("standard GAN", "saturating loss"))
for col, nm in [(1, "non-saturating loss (standard)"),
                (2, "SATURATING loss (textbook)")]:
    S = runs[nm][1]
    fig.add_trace(go.Scatter(x=REAL[:1500, 0], y=REAL[:1500, 1], mode="markers",
                             marker=dict(size=3, color=alpha(C["muted"], .4)),
                             showlegend=False), 1, col)
    fig.add_trace(go.Scatter(x=S[:1500, 0], y=S[:1500, 1], mode="markers",
                             marker=dict(size=4, color=C["danger"]),
                             showlegend=False), 1, col)
fig.update_layout(height=430, title="Real data (grey) vs generated (red)")
fig.update_yaxes(scaleanchor="x")
''',
        key="ch17_gan",
    )

    keypoints([
        "A GAN replaces the likelihood with a <b>learned</b> loss — a "
        "discriminator estimating a density ratio.",
        "At the optimum the objective equals $2 D_{\\mathrm{JS}}(p_{\\text{data}} "
        "\\Vert p_g) - \\log 4$, minimised when $p_g = p_{\\text{data}}$.",
        "Use the <b>non-saturating</b> generator loss: train $G$ with fakes "
        "labelled real.",
        "<b>Mode collapse</b> and oscillation are structural: the objective never "
        "mentions coverage, and games orbit.",
        "<b>The GAN loss is not a progress metric.</b> Only the samples are.",
    ])


# ==========================================================================
def s_17_8():
    section("17.8", "DCGAN, Progressive Growing and StyleGAN")

    lead(
        "Four papers turned GANs from a curiosity into a photorealistic image "
        "generator. Each fixed one specific instability."
    )

    sub("DCGAN (2015)")

    md(
        "Radford et al. found the architectural recipe that made convolutional "
        "GANs train at all — largely by removing things."
    )

    table(
        ["Rule", "Instead of", "Reason"],
        [["Strided convolutions in $D$, transposed in $G$",
          "Pooling and upsampling",
          "Let the network learn its own down/upsampling"],
         ["Batch norm in both, <b>except</b> $G$'s output and $D$'s input",
          "Batch norm everywhere",
          "Normalising those layers destroys the data's actual scale"],
         ["No fully connected hidden layers", "Dense layers on top",
          "Fully convolutional; scales to any resolution"],
         ["ReLU in $G$, <b>tanh</b> at the output", "sigmoid",
          "Symmetric $[-1,1]$ range matches normalised images"],
         ["<b>LeakyReLU</b> throughout $D$", "ReLU",
          "A dead unit in $D$ kills $G$'s gradient too"]],
    )

    codenote(
        "The latent space becomes an arithmetic space",
        "DCGAN's famous result was that latent vectors support semantic "
        "arithmetic: $\\mathbf{z}_{\\text{man with glasses}} - "
        "\\mathbf{z}_{\\text{man}} + \\mathbf{z}_{\\text{woman}}$ decodes to a "
        "woman with glasses. This is the same phenomenon as word-vector analogies "
        "and it says something real: a smooth generator maps a linear latent "
        "structure onto a semantic one. It also does <b>not</b> mean the latent "
        "space is disentangled — the directions are entangled, which is exactly "
        "the problem StyleGAN attacks.",
    )

    sub("Progressive growing (2017)")

    derive(
        [("<b>The problem with training at high resolution directly.</b> At "
          "$1024^2$, real and generated distributions live on manifolds in a "
          "$3 \\times 10^6$-dimensional space. Their supports are essentially "
          "certain to be disjoint, so — as in §17.7 — the discriminator wins "
          "trivially and the Jensen–Shannon gradient is uninformative.", None),
         ("<b>Karras et al.'s answer: start at $4\\times4$.</b> At that "
          "resolution the problem is genuinely low-dimensional, the distributions "
          "overlap, and training is stable and fast.", None),
         ("Once converged, <b>fade in</b> a new layer at double the resolution "
          "using a mixing weight $\\alpha$ that goes from 0 to 1 over many "
          "iterations:",
          r"\mathbf{x}_{\text{out}} = (1-\alpha)\,\mathrm{upsample}"
          r"\bigl(\mathbf{x}_{\text{old}}\bigr) + \alpha\,"
          r"\mathrm{newlayer}\bigl(\mathbf{x}_{\text{old}}\bigr)"),
         ("The fade is essential: dropping a randomly initialised layer in "
          "abruptly would shock the trained layers below it. With $\\alpha$ "
          "ramping, the network is <b>always</b> near a configuration it has "
          "already solved.", None),
         ("<b>The result:</b> each stage is an easy problem, and the whole "
          "training is 2–6× faster than training at full resolution — while "
          "actually working, which direct training did not.", None)],
        title="Why progressive growing stabilises high-resolution GANs",
    )

    sub("StyleGAN (2018–2021)")

    table(
        ["Component", "What it does", "Why it matters"],
        [["<b>Mapping network</b> $f: \\mathcal{Z} \\to \\mathcal{W}$",
          "8 dense layers from $\\mathbf{z}$ to an intermediate $\\mathbf{w}$",
          "$\\mathcal{Z}$ must match the Gaussian prior's shape; "
          "$\\mathcal{W}$ is free to be <b>disentangled</b>"],
         ["<b>AdaIN</b> at every resolution",
          "$\\mathbf{w}$ sets per-channel scale and bias of normalised "
          "activations",
          "Style is injected <b>everywhere</b>, not only at the input"],
         ["<b>Per-pixel noise</b> inputs",
          "Fresh noise added at each resolution",
          "Stochastic detail (hair, freckles) is decoupled from identity"],
         ["<b>Style mixing</b>",
          "Use $\\mathbf{w}_1$ for coarse layers, $\\mathbf{w}_2$ for fine",
          "Proves the layers control different scales, and regularises"],
         ["<b>Constant input</b>",
          "The synthesis network starts from a learned constant",
          "All variation comes from styles and noise, nothing from $\\mathbf{z}$ "
          "directly"]],
    )

    idea(
        "The mapping network is the whole trick, and the reason is geometric",
        "$\\mathbf{z}$ is sampled from a Gaussian, so its density is fixed and "
        "roughly spherical. If some attribute combination is rare in the data "
        "(say, 'long hair' with 'beard'), a generator taking $\\mathbf{z}$ "
        "directly must <b>warp</b> the sphere severely to give that region the "
        "right small probability — and warping entangles the axes. A learned "
        "mapping $f$ absorbs that warping, so $\\mathcal{W}$ can be shaped like "
        "the data's actual factor structure. StyleGAN measured this: "
        "perceptual path length and linear separability both improve "
        "substantially in $\\mathcal{W}$ over $\\mathcal{Z}$.",
    )

    note(
        "StyleGAN2 and StyleGAN3 fixed specific visible artefacts",
        "<b>StyleGAN2</b> removed the blob-like droplet artefacts by replacing "
        "AdaIN with weight demodulation (the instance normalisation was creating "
        "a strong spike the network used to cheat), and removed progressive "
        "growing in favour of skip connections. <b>StyleGAN3</b> fixed 'texture "
        "sticking' — detail that stayed glued to pixel coordinates rather than to "
        "the face — by making every layer properly equivariant to translation "
        "and rotation, which required treating the signals as continuous and "
        "respecting the Nyquist limit at each layer. Both are worth knowing as "
        "examples of <b>diagnosing an artefact back to a specific architectural "
        "cause</b>.",
    )

    anim_header("Progressive growing: fading in a new resolution")

    res_stages = [4, 8, 16, 32]
    base_img = np.zeros((32, 32))
    yy, xx = np.mgrid[0:32, 0:32]
    base_img += np.exp(-((xx-12)**2 + (yy-13)**2)/40)
    base_img += .7*np.exp(-((xx-21)**2 + (yy-19)**2)/25)
    base_img += .3*np.sin(xx*.8)*np.sin(yy*.7)
    base_img = (base_img - base_img.min())/np.ptp(base_img)

    def downsample(img, r):
        f = 32 // r
        return img.reshape(r, f, r, f).mean((1, 3))

    def upsample(img, r):
        f = 32 // img.shape[0]
        return np.kron(img, np.ones((f, f)))

    frames = []
    for si in range(len(res_stages)):
        r_now = res_stages[si]
        r_prev = res_stages[si - 1] if si > 0 else r_now
        for a in np.linspace(0, 1, 9):
            old = upsample(downsample(base_img, r_prev), 32)
            new = upsample(downsample(base_img, r_now), 32)
            blend = (1 - a) * old + a * new
            frames.append(go.Frame(
                name=f"{r_now}:{a:.2f}",
                data=[go.Heatmap(z=blend, colorscale=nav.cscale(),
                                 zmin=0, zmax=1, showscale=False)],
                layout=go.Layout(annotations=[anim.annotate_step(
                    f"stage {si+1}: {r_prev}² → {r_now}²   ·   α = {a:.2f}   ·   "
                    f"x_out = (1−α)·upsample(old) + α·newlayer(old)")])))
            if si == 0:
                break

    f = go.Figure(data=[go.Heatmap(z=upsample(downsample(base_img, 4), 32),
                                   colorscale=nav.cscale(), zmin=0, zmax=1,
                                   showscale=False)])
    f.update_layout(height=440, xaxis=dict(visible=False),
                    yaxis=dict(visible=False, autorange="reversed",
                               scaleanchor="x"),
                    title="Progressive growing, 4² → 32²")
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="")
    figure(f, "The new layer is faded in, never switched in. The network is "
              "always close to a configuration it has already solved.")

    code_lab(
        "A DCGAN on Fashion-MNIST, plus the StyleGAN mapping-network argument",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, _labels, _real = _ds.fashion_mnist(n_train=8000, n_test=1000)
X = Xtr.astype("float32")
if X.max() > 1.5:
    X /= 255.
X = (X.reshape(-1, 28, 28, 1) * 2.0 - 1.0).astype("float32")   # [-1, 1]
print(f"=== Fashion-MNIST scaled to [-1, 1]: {X.shape}, "
      f"range [{X.min():.1f}, {X.max():.1f}] ===")
print("  the [-1,1] range is DCGAN rule 4: it matches a tanh output")

LATENT = 64

# ============ 1. THE DCGAN ARCHITECTURE ================================
def make_G():
    return keras.Sequential([
        keras.layers.Input(shape=(LATENT,)),
        keras.layers.Dense(7*7*128, use_bias=False),
        keras.layers.BatchNormalization(),                 # BN in G: YES
        keras.layers.ReLU(),
        keras.layers.Reshape((7, 7, 128)),
        keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same",
                                     use_bias=False),      # kernel 4 / stride 2
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2DTranspose(1, 4, strides=2, padding="same",
                                     activation="tanh"),   # tanh OUTPUT, no BN
    ], name="generator")

def make_D():
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, 4, strides=2, padding="same"),   # no BN on input
        keras.layers.LeakyReLU(0.2),                             # LEAKY in D
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 4, strides=2, padding="same"),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1),                                   # LOGITS
    ], name="discriminator")

G, D = make_G(), make_D()
print()
print("=== the DCGAN recipe ===")
print(f"  generator     {G.count_params():>9,} params, output "
      f"{tuple(G.output_shape[1:])}, tanh")
print(f"  discriminator {D.count_params():>9,} params, LeakyReLU(0.2), logits")
print(f"  every kernel is 4 with stride 2 -- divisible, so NO checkerboard")

# ============ 2. TRAIN IT ==============================================
bce = keras.losses.BinaryCrossentropy(from_logits=True)
optG = keras.optimizers.Adam(2e-4, beta_1=0.5)     # beta_1 = 0.5, not 0.9
optD = keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(xr, batch):
    z = tf.random.normal((batch, LATENT))
    with tf.GradientTape() as td:
        xf = G(z, training=True)
        # SEPARATE batches for real and fake -- DCGAN rule 8
        ld = bce(tf.ones((batch, 1))*0.9, D(xr, training=True)) \\
           + bce(tf.zeros((batch, 1)), D(xf, training=True))
    optD.apply_gradients(zip(td.gradient(ld, D.trainable_weights),
                             D.trainable_weights))
    z = tf.random.normal((batch, LATENT))
    with tf.GradientTape() as tg:
        # NON-SATURATING: fakes labelled REAL
        lg = bce(tf.ones((batch, 1)), D(G(z, training=True), training=True))
    optG.apply_gradients(zip(tg.gradient(lg, G.trainable_weights),
                             G.trainable_weights))
    return ld, lg

BATCH, STEPS = 96, 400
rng = np.random.default_rng(0)
print()
print(f"=== training {STEPS} steps ===")
t0 = time.perf_counter()
log = []
for step in range(STEPS):
    xr = X[rng.integers(0, len(X), BATCH)]
    ld, lg = train_step(tf.constant(xr), BATCH)
    if step % 80 == 0:
        gen = G.predict(tf.random.normal((300, LATENT)), verbose=0)
        log.append((step, float(ld), float(lg), float(gen.std()),
                    float(np.abs(gen.mean(0) - X.mean(0)).mean())))
print(f"  {time.perf_counter()-t0:.1f}s")
print(f"{'step':>7}{'D loss':>10}{'G loss':>10}{'sample std':>13}"
      f"{'mean-image error':>19}")
for s, a, b, sd, me in log:
    print(f"{s:>7}{a:>10.4f}{b:>10.4f}{sd:>13.4f}{me:>19.4f}")
print("  'sample std' near the real std means the generator has NOT collapsed:")
print(f"  real data std = {X.std():.4f}")

# ============ 3. MODE COLLAPSE DETECTION ===============================
print()
print("=== is it collapsing? three diagnostics ===")
z = tf.random.normal((600, LATENT))
gen = G.predict(z, verbose=0).reshape(600, -1)
pair = np.linalg.norm(gen[:200, None] - gen[None, :200], axis=-1)
real_pair = np.linalg.norm(X[:200].reshape(200, -1)[:, None] -
                           X[:200].reshape(200, -1)[None], axis=-1)
print(f"  1. mean pairwise distance: generated {pair.mean():.4f}, "
      f"real {real_pair.mean():.4f}")
print(f"     (collapse -> generated distance near 0)")
sv = np.linalg.svd(gen - gen.mean(0), compute_uv=False)
sv_r = np.linalg.svd(X[:600].reshape(600, -1) - X[:600].reshape(600, -1).mean(0),
                     compute_uv=False)
eff = float((sv.sum()**2)/(sv**2).sum())
eff_r = float((sv_r.sum()**2)/(sv_r**2).sum())
print(f"  2. effective rank of the sample set: generated {eff:.1f}, "
      f"real {eff_r:.1f}")
print(f"     (collapse -> effective rank near 1)")
nn = np.linalg.norm(gen[:150, None] - X[:800].reshape(800, -1)[None],
                    axis=-1).min(1)
print(f"  3. nearest real neighbour distance: mean {nn.mean():.4f}")
print(f"     (memorisation -> near 0; healthy -> comparable to real spacing)")

# ============ 4. LATENT ARITHMETIC =====================================
print()
print("=== the latent space supports arithmetic ===")
z1 = tf.random.normal((1, LATENT), seed=1)
z2 = tf.random.normal((1, LATENT), seed=2)
alphas = np.linspace(0, 1, 9)
# SLERP, not LERP: a linear path through a Gaussian passes through
# the low-density centre, where the generator was never trained
def slerp(a, z1, z2):
    z1n = z1/np.linalg.norm(z1); z2n = z2/np.linalg.norm(z2)
    om = np.arccos(np.clip((z1n*z2n).sum(), -1, 1))
    if abs(om) < 1e-6:
        return (1-a)*z1 + a*z2
    return (np.sin((1-a)*om)*z1 + np.sin(a*om)*z2)/np.sin(om)

z1n, z2n = z1.numpy(), z2.numpy()
lerp_path = np.vstack([(1-a)*z1n + a*z2n for a in alphas]).astype("float32")
slerp_path = np.vstack([slerp(a, z1n, z2n) for a in alphas]).astype("float32")
print(f"  ||z|| along LERP : {np.round(np.linalg.norm(lerp_path,axis=1),2)}")
print(f"  ||z|| along SLERP: {np.round(np.linalg.norm(slerp_path,axis=1),2)}")
print(f"  expected ||z|| for a {LATENT}-D standard normal: "
      f"{np.sqrt(LATENT):.2f}")
print("  LERP passes through the ORIGIN, where a Gaussian has almost no mass")
print("  and the generator was never trained. Always SLERP between latents.")

# ============ 5. THE STYLEGAN MAPPING-NETWORK ARGUMENT =================
print()
print("="*66)
print("Why StyleGAN maps z -> w before using it")
print("="*66)
print("  suppose 'long hair' and 'beard' are each common but their")
print("  COMBINATION is rare. A generator taking z directly must warp the")
print("  Gaussian sphere so that region gets a small probability -- and")
print("  warping ENTANGLES the axes.")
print()
# a 2-D demonstration: a target distribution with a forbidden corner
rng2 = np.random.default_rng(0)
n = 4000
W_true = rng2.uniform(-1, 1, (n, 2))
keep = ~((W_true[:, 0] > .35) & (W_true[:, 1] > .35))     # the rare corner
W_true = W_true[keep]
print(f"  target: uniform on a square MINUS one corner ({len(W_true)} points)")

def train_mapper(depth):
    tf.random.set_seed(0)
    layers = [keras.layers.Input(shape=(2,))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(64, activation="leaky_relu"))
    layers.append(keras.layers.Dense(2))
    G2 = keras.Sequential(layers)
    D2 = keras.Sequential([keras.layers.Input(shape=(2,)),
                           keras.layers.Dense(64), keras.layers.LeakyReLU(.2),
                           keras.layers.Dense(64), keras.layers.LeakyReLU(.2),
                           keras.layers.Dense(1)])
    oG = keras.optimizers.Adam(2e-3, beta_1=.5)
    oD = keras.optimizers.Adam(2e-3, beta_1=.5)
    r = np.random.default_rng(0)
    for _ in range(350):
        xr = W_true[r.integers(0, len(W_true), 128)].astype("float32")
        z = r.normal(0, 1, (128, 2)).astype("float32")
        with tf.GradientTape() as t:
            l = bce(tf.ones((128,1)), D2(xr, training=True)) + \\
                bce(tf.zeros((128,1)), D2(G2(z, training=False), training=True))
        oD.apply_gradients(zip(t.gradient(l, D2.trainable_weights),
                               D2.trainable_weights))
        z = r.normal(0, 1, (128, 2)).astype("float32")
        with tf.GradientTape() as t:
            lg2 = bce(tf.ones((128,1)), D2(G2(z, training=True), training=False))
        oG.apply_gradients(zip(t.gradient(lg2, G2.trainable_weights),
                               G2.trainable_weights))
    return G2

print()
print(f"{'generator depth':>17}{'forbidden corner hits':>24}"
      f"{'coverage of the rest':>23}")
for depth in [1, 2, 4, 8]:
    G2 = train_mapper(depth)
    S = G2.predict(np.random.default_rng(1).normal(0, 1, (3000, 2)
                                                   ).astype("float32"),
                   verbose=0)
    bad = np.mean((S[:, 0] > .35) & (S[:, 1] > .35))
    inbox = np.mean((np.abs(S) < 1.05).all(1))
    print(f"{depth:>17}{bad:>24.2%}{inbox:>23.1%}")
print("  a deeper mapping absorbs the warping, so the LAST layers can be")
print("  smooth and the intermediate space W ends up better behaved.")
print("  StyleGAN uses 8 dense layers for exactly this reason.")

import plotly.graph_objects as go
gen_img = G.predict(tf.random.normal((16, LATENT)), verbose=0)
grid = np.zeros((4*28, 4*28))
for i in range(16):
    r_, c_ = divmod(i, 4)
    grid[r_*28:(r_+1)*28, c_*28:(c_+1)*28] = (gen_img[i, :, :, 0]+1)/2
fig = go.Figure(go.Heatmap(z=grid, colorscale="Greys", reversescale=True,
                           showscale=False))
fig.update_layout(height=430, xaxis=dict(visible=False),
                  yaxis=dict(visible=False, autorange="reversed"),
                  title=f"DCGAN samples after {STEPS} steps")
''',
        key="ch17_dcgan",
    )

    keypoints([
        "<b>DCGAN</b>: strided convolutions both ways, LeakyReLU in $D$, tanh "
        "output, Adam with $\\beta_1 = 0.5$.",
        "<b>Progressive growing</b> starts at $4^2$ and fades in each new "
        "resolution with a mixing weight $\\alpha$.",
        "<b>StyleGAN's mapping network</b> lets $\\mathcal{W}$ be disentangled "
        "while $\\mathcal{Z}$ stays Gaussian.",
        "Interpolate latents with <b>SLERP</b>, not LERP — a linear path crosses "
        "the empty centre.",
        "Diagnose collapse with pairwise distances, effective rank, and "
        "nearest-real-neighbour distance.",
    ])


# ==========================================================================
def s_17_9():
    section("17.9", "Diffusion Models and Exercises")

    lead(
        "Destroy an image with noise over many small steps, then train a network "
        "to undo one step. Run it backwards from pure noise and you have the "
        "generative model that beat GANs."
    )

    sub("The forward process")

    md(
        "Fix a variance schedule $\\beta_1, \\dots, \\beta_T$ (small, increasing). "
        "The forward process adds a little Gaussian noise at each step:"
    )

    math(r"""
    q\bigl(\mathbf{x}_t \mid \mathbf{x}_{t-1}\bigr)
    = \mathcal{N}\Bigl(\mathbf{x}_t;\;
      \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I}\Bigr)
    """)

    derive(
        [("<b>The key algebraic fact: you can jump to any step in one shot.</b> "
          "Let $\\alpha_t = 1 - \\beta_t$ and "
          "$\\bar\\alpha_t = \\prod_{s=1}^{t}\\alpha_s$.", None),
         ("Compose two steps. Writing "
          "$\\mathbf{x}_t = \\sqrt{\\alpha_t}\\mathbf{x}_{t-1} + "
          "\\sqrt{\\beta_t}\\boldsymbol\\varepsilon_t$ and substituting "
          "$\\mathbf{x}_{t-1}$:",
          r"\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1}}\,\mathbf{x}_{t-2}"
          r" + \sqrt{\alpha_t \beta_{t-1}}\,\boldsymbol\varepsilon_{t-1}"
          r" + \sqrt{\beta_t}\,\boldsymbol\varepsilon_t"),
         ("The two noise terms are independent Gaussians, so their sum is a "
          "single Gaussian with the summed variance: "
          "$\\alpha_t\\beta_{t-1} + \\beta_t = 1 - \\alpha_t\\alpha_{t-1}$. "
          "Inducting all the way to $\\mathbf{x}_0$:",
          r"\boxed{\;q\bigl(\mathbf{x}_t \mid \mathbf{x}_0\bigr) = "
          r"\mathcal{N}\Bigl(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\;"
          r"(1-\bar\alpha_t)\mathbf{I}\Bigr)\;}"),
         ("Equivalently, in one line of code:",
          r"\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + "
          r"\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon,"
          r"\qquad \boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})"),
         ("<b>This is why diffusion is trainable at all.</b> Training needs no "
          "simulation of the chain: sample a random $t$, jump straight to "
          "$\\mathbf{x}_t$, and take one gradient step. Without this identity you "
          "would have to run $T$ sequential steps per training example.", None),
         ("With $\\bar\\alpha_T \\approx 0$, $\\mathbf{x}_T$ is indistinguishable "
          "from $\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ — which is where "
          "sampling starts.", None)],
        title="The closed form that makes diffusion practical",
    )

    sub("The training objective")

    md(
        "The reverse process $p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)$ is "
        "also Gaussian when $\\beta_t$ is small. Deriving its ELBO and dropping "
        "constants gives a remarkably simple loss:"
    )

    math(r"""
    \mathcal{L}_{\text{simple}}
    = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\varepsilon}
      \Bigl\lVert \boldsymbol\varepsilon
        - \boldsymbol\varepsilon_\theta\bigl(
          \sqrt{\bar\alpha_t}\mathbf{x}_0 +
          \sqrt{1-\bar\alpha_t}\boldsymbol\varepsilon,\; t\bigr)
      \Bigr\rVert^{2}
    """)

    proof(
        "This is §17.4's denoising autoencoder, with a noise-level input",
        "The network is handed a noisy image and asked to <b>predict the noise "
        "that was added</b> — which, given the closed form, is equivalent to "
        "predicting the clean image. That is exactly a denoising autoencoder. "
        "The two additions are: (1) the network also receives $t$, so one network "
        "handles <b>every</b> noise level, and (2) the noise levels are arranged "
        "in a schedule that can be walked backwards. By Tweedie's formula "
        "(§17.4), $\\boldsymbol\\varepsilon_\\theta$ is a scaled score estimate: "
        "$\\nabla_{\\mathbf{x}_t}\\log q(\\mathbf{x}_t) = "
        "-\\boldsymbol\\varepsilon_\\theta / \\sqrt{1-\\bar\\alpha_t}$. "
        "<b>Diffusion is score matching with a schedule.</b>",
    )

    sub("Sampling")

    math(r"""
    \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}
      \left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
        \boldsymbol\varepsilon_\theta(\mathbf{x}_t, t)\right)
      \;+\; \sigma_t \mathbf{z},
    \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
    """)

    table(
        ["", "GAN", "VAE", "Diffusion"],
        [["Sample quality", "<b>Excellent</b>", "Blurry", "<b>Excellent</b>"],
         ["Mode coverage", "<b>Poor</b> (collapse)", "Good", "<b>Excellent</b>"],
         ["Training stability", "<b>Fragile</b> — a game",
          "Stable", "<b>Very stable</b> — a regression loss"],
         ["Likelihood", "None", "ELBO (a bound)", "ELBO (a good bound)"],
         ["Sampling speed", "<b>1 forward pass</b>", "<b>1 forward pass</b>",
          "$T$ passes (10–1000)"],
         ["Latent space", "Low-dim, semantic", "Low-dim, semantic",
          "Same size as the data"],
         ["Controllability", "Latent edits", "Latent edits",
          "<b>Conditioning + guidance</b>"]],
    )

    idea(
        "Why diffusion won despite being 100× slower to sample",
        "The loss is a <b>plain regression</b>: predict the noise. No adversary, "
        "no equilibrium, no collapse — throw more compute and data at it and it "
        "reliably gets better, which is exactly the property that matters at "
        "scale. GANs are faster per sample but do not scale reliably; getting a "
        "large GAN to train is a research project, while getting a large "
        "diffusion model to train is an engineering one. The speed gap has since "
        "narrowed dramatically anyway — DDIM, distillation and consistency models "
        "bring sampling down to 1–4 steps.",
    )

    sub("Classifier-free guidance")

    md(
        "The mechanism behind every text-to-image system: train **one** network "
        "on both conditional and unconditional denoising (by dropping the "
        "condition ~10 % of the time), then at sampling time extrapolate away "
        "from the unconditional prediction."
    )

    math(r"""
    \tilde{\boldsymbol\varepsilon}_\theta(\mathbf{x}_t, t, c)
    = \boldsymbol\varepsilon_\theta(\mathbf{x}_t, t, \varnothing)
      + w\,\bigl(\boldsymbol\varepsilon_\theta(\mathbf{x}_t, t, c)
        - \boldsymbol\varepsilon_\theta(\mathbf{x}_t, t, \varnothing)\bigr)
    """)

    warn(
        "Guidance trades diversity for prompt adherence",
        "$w = 1$ is ordinary conditional sampling. $w > 1$ amplifies the "
        "difference the condition makes: images match the prompt much more "
        "closely, and become <b>less diverse and more saturated</b>. Typical "
        "values are 3–10. Push $w$ far higher and you get the over-contrasted, "
        "over-literal images that are the visual signature of too much guidance. "
        "It is a genuine trade-off, not a quality knob.",
    )

    anim_header("The forward process, and the reverse that undoes it")

    rng = np.random.default_rng(0)
    n_s = 500
    ang_s = rng.uniform(0, 2*np.pi, n_s)
    rad = 2.0 + rng.normal(0, .08, n_s)
    X0 = np.column_stack([rad*np.cos(ang_s), rad*np.sin(ang_s)])
    X0[:, 1] *= 0.62

    T_diff = 40
    betas = np.linspace(1e-3, 0.09, T_diff)
    alphas = 1 - betas
    abar = np.cumprod(alphas)

    fwd = []
    eps_fix = rng.normal(0, 1, X0.shape)
    for t in range(T_diff):
        fwd.append(np.sqrt(abar[t])*X0 + np.sqrt(1-abar[t])*eps_fix)

    frames = []
    for t in range(T_diff):
        frames.append(go.Frame(name=f"f{t}", data=[
            go.Scatter(x=fwd[t][:, 0], y=fwd[t][:, 1], mode="markers",
                       marker=dict(size=5, color=ang_s, colorscale="Viridis",
                                   showscale=False)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"FORWARD  t = {t+1}/{T_diff}   ·   ᾱ = {abar[t]:.4f}   ·   "
            f"signal {np.sqrt(abar[t]):.3f}, noise {np.sqrt(1-abar[t]):.3f}",
            color=C["danger"])])))
    for t in range(T_diff - 1, -1, -1):
        frames.append(go.Frame(name=f"r{t}", data=[
            go.Scatter(x=fwd[t][:, 0], y=fwd[t][:, 1], mode="markers",
                       marker=dict(size=5, color=ang_s, colorscale="Viridis",
                                   showscale=False)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"REVERSE  t = {t+1}/{T_diff}   ·   the network predicts ε and "
            f"removes one step's worth   ·   ᾱ = {abar[t]:.4f}",
            color=C["success"])])))

    f = go.Figure(data=[go.Scatter(x=X0[:, 0], y=X0[:, 1], mode="markers",
                                   marker=dict(size=5, color=ang_s,
                                               colorscale="Viridis",
                                               showscale=False))])
    f.update_layout(height=470, xaxis=dict(range=[-3.5, 3.5], title="x₁"),
                    yaxis=dict(range=[-3.5, 3.5], title="x₂", scaleanchor="x"),
                    title="q(xₜ|x₀) = N(√ᾱₜ·x₀, (1−ᾱₜ)I), and its reverse")
    anim.animate(f, frames, duration=nav.anim_ms(85), slider_prefix="")
    figure(f, "The forward process is fixed and has no parameters. All the "
              "learning is in reversing one step.")

    code_lab(
        "A working diffusion model: the closed form, training, and sampling",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE FORWARD PROCESS AND ITS CLOSED FORM ===============
T = 200
betas = np.linspace(1e-4, 0.02, T).astype("float32")
alphas = (1.0 - betas).astype("float32")
abar = np.cumprod(alphas).astype("float32")

print("=== the variance schedule ===")
print(f"{'t':>6}{'beta_t':>11}{'alpha_t':>11}{'abar_t':>11}"
      f"{'signal':>10}{'noise':>10}")
for t in [0, 20, 50, 100, 150, 199]:
    print(f"{t:>6}{betas[t]:>11.5f}{alphas[t]:>11.5f}{abar[t]:>11.5f}"
          f"{np.sqrt(abar[t]):>10.4f}{np.sqrt(1-abar[t]):>10.4f}")
print(f"  at t={T-1}, the signal has fallen to {np.sqrt(abar[-1]):.4f}:")
print(f"  x_T is essentially pure N(0, I) noise.")

# --- verify the closed form against the step-by-step chain -----------
print()
print("=== q(x_t | x_0) in ONE step vs the whole chain ===")
rng = np.random.default_rng(0)
x0 = rng.normal(0, 1, (20000, 2)).astype("float32")
x0[:, 0] += 3.0                                   # a shifted blob

chain = x0.copy()
for t in range(60):
    chain = (np.sqrt(alphas[t])*chain
             + np.sqrt(betas[t])*rng.normal(0, 1, chain.shape)).astype("float32")
direct = (np.sqrt(abar[59])*x0
          + np.sqrt(1-abar[59])*rng.normal(0, 1, x0.shape)).astype("float32")
print(f"  after 60 steps:")
print(f"    chain  mean {chain.mean(0).round(4)}  std {chain.std(0).round(4)}")
print(f"    direct mean {direct.mean(0).round(4)}  std {direct.std(0).round(4)}")
print(f"  identical distributions -- THIS is what makes training feasible:")
print(f"  no chain simulation, just one jump to a random t.")

# ============ 2. THE MODEL: PREDICT THE NOISE ==========================
def timestep_embedding(t, dim=32):
    """Sinusoidal embedding of the timestep -- exactly ch.16's positional
    encoding, applied to the noise level instead of to position."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    a = t[:, None].astype("float32") * freqs[None].astype("float32")
    return np.concatenate([np.sin(a), np.cos(a)], 1).astype("float32")

def build_eps_net(d_in=2, width=192, t_dim=32):
    x_in = keras.layers.Input(shape=(d_in,))
    t_in = keras.layers.Input(shape=(t_dim,))
    h = keras.layers.Concatenate()([x_in, t_in])
    for _ in range(4):
        h = keras.layers.Dense(width, activation="swish")(h)
    return keras.Model([x_in, t_in], keras.layers.Dense(d_in)(h))

# ============ 3. THE DATA: A TWO-MOONS DISTRIBUTION ====================
from core import datasets as _ds
Xm, ym = _ds.moons(n=6000, noise=.06)
Xm = ((Xm - Xm.mean(0)) / Xm.std(0)).astype("float32")
print()
print(f"=== training data: two moons {Xm.shape} ===")

# ============ 4. TRAINING: ONE LINE OF ALGEBRA, ONE MSE ================
tf.random.set_seed(0)
eps_net = build_eps_net()
opt = keras.optimizers.Adam(2e-3)

@tf.function
def train_step(x0b, tb, ab, eps):
    with tf.GradientTape() as tape:
        xt = tf.sqrt(ab)*x0b + tf.sqrt(1.0-ab)*eps       # the closed form
        pred = eps_net([xt, tb], training=True)
        loss = tf.reduce_mean(tf.square(eps - pred))     # PLAIN MSE
    opt.apply_gradients(zip(tape.gradient(loss, eps_net.trainable_weights),
                            eps_net.trainable_weights))
    return loss

BATCH, STEPS = 256, 2500
r = np.random.default_rng(0)
t0 = time.perf_counter()
hist = []
for step in range(STEPS):
    idx = r.integers(0, len(Xm), BATCH)
    tt = r.integers(0, T, BATCH)
    l = train_step(tf.constant(Xm[idx]),
                   tf.constant(timestep_embedding(tt)),
                   tf.constant(abar[tt][:, None]),
                   tf.constant(r.normal(0, 1, (BATCH, 2)).astype("float32")))
    if step % 500 == 0:
        hist.append((step, float(l)))
print(f"  trained in {time.perf_counter()-t0:.1f}s")
print(f"{'step':>8}{'MSE loss':>12}")
for s, l in hist:
    print(f"{s:>8}{l:>12.5f}")
print("  no adversary, no equilibrium, no collapse. Just a regression.")

# ============ 5. SAMPLING (DDPM) =======================================
print()
print("=== DDPM sampling: run the chain backwards ===")
def ddpm_sample(n=2000, seed=1, record=()):
    rr = np.random.default_rng(seed)
    x = rr.normal(0, 1, (n, 2)).astype("float32")        # start from NOISE
    snaps = {}
    for t in range(T-1, -1, -1):
        tb = timestep_embedding(np.full(n, t))
        e = eps_net.predict([x, tb], verbose=0)
        mean = (x - betas[t]/np.sqrt(1-abar[t]) * e) / np.sqrt(alphas[t])
        if t > 0:
            x = (mean + np.sqrt(betas[t])*rr.normal(0, 1, x.shape)
                 ).astype("float32")
        else:
            x = mean.astype("float32")
        if t in record:
            snaps[t] = x.copy()
    return x, snaps

t0 = time.perf_counter()
samples, snaps = ddpm_sample(1500, record=(150, 100, 50, 20))
print(f"  {T} network calls, {time.perf_counter()-t0:.1f}s")
print(f"{'stage':>18}{'mean':>22}{'std':>22}")
for t in [150, 100, 50, 20]:
    s = snaps[t]
    print(f"{'t = '+str(t):>18}{str(s.mean(0).round(3)):>22}"
          f"{str(s.std(0).round(3)):>22}")
print(f"{'final (t=0)':>18}{str(samples.mean(0).round(3)):>22}"
      f"{str(samples.std(0).round(3)):>22}")
print(f"{'REAL DATA':>18}{str(Xm.mean(0).round(3)):>22}"
      f"{str(Xm.std(0).round(3)):>22}")

# --- how close are the samples to the data manifold? -----------------
nn = np.linalg.norm(samples[:600, None] - Xm[None, :2000], axis=-1).min(1)
nn_real = np.linalg.norm(Xm[:600, None] - Xm[None, 600:2600], axis=-1).min(1)
print()
print(f"  nearest-real-neighbour distance: samples {nn.mean():.4f}, "
      f"real-to-real {nn_real.mean():.4f}")
print(f"  (much larger would mean off-manifold; ~0 would mean memorisation)")

# ============ 6. DDIM: THE SAME MODEL, 20x FEWER STEPS =================
print()
print("=== DDIM: deterministic sampling with a subset of timesteps ===")
def ddim_sample(n=1500, n_steps=20, eta=0.0, seed=1):
    rr = np.random.default_rng(seed)
    x = rr.normal(0, 1, (n, 2)).astype("float32")
    ts = np.linspace(T-1, 0, n_steps).astype(int)
    for i, t in enumerate(ts):
        tb = timestep_embedding(np.full(n, t))
        e = eps_net.predict([x, tb], verbose=0)
        a_t = abar[t]
        a_prev = abar[ts[i+1]] if i+1 < len(ts) else 1.0
        x0_pred = (x - np.sqrt(1-a_t)*e) / np.sqrt(a_t)   # predicted clean x
        sigma = eta*np.sqrt((1-a_prev)/(1-a_t))*np.sqrt(1-a_t/a_prev)
        dir_xt = np.sqrt(max(0.0, 1-a_prev-sigma**2))*e
        x = (np.sqrt(a_prev)*x0_pred + dir_xt).astype("float32")
        if sigma > 0:
            x = (x + sigma*rr.normal(0, 1, x.shape)).astype("float32")
    return x

print(f"{'sampler':<26}{'steps':>8}{'time':>9}{'mean NN dist':>15}")
for nm, fn, ns in [("DDPM (stochastic)", None, T),
                   ("DDIM eta=0", 20, 20),
                   ("DDIM eta=0", 10, 10),
                   ("DDIM eta=0", 5, 5)]:
    t0 = time.perf_counter()
    S = samples if fn is None else ddim_sample(1500, n_steps=fn)
    dt = time.perf_counter()-t0
    d = np.linalg.norm(S[:400, None] - Xm[None, :2000], axis=-1).min(1).mean()
    print(f"{nm:<26}{ns:>8}{dt:>8.1f}s{d:>15.4f}")
print("  the SAME trained network. DDIM just takes larger, deterministic")
print("  steps along the same probability-flow path.")

# ============ 7. CLASSIFIER-FREE GUIDANCE ==============================
print()
print("="*66)
print("Classifier-free guidance")
print("="*66)
# retrain with a class label, dropped 10 % of the time
tf.random.set_seed(0)
def build_cond_net(width=192, t_dim=32, n_cls=3):     # class 2 = "unconditional"
    x_in = keras.layers.Input(shape=(2,))
    t_in = keras.layers.Input(shape=(t_dim,))
    c_in = keras.layers.Input(shape=(1,), dtype="int32")
    c = keras.layers.Flatten()(keras.layers.Embedding(n_cls, 16)(c_in))
    h = keras.layers.Concatenate()([x_in, t_in, c])
    for _ in range(4):
        h = keras.layers.Dense(width, activation="swish")(h)
    return keras.Model([x_in, t_in, c_in], keras.layers.Dense(2)(h))

cnet = build_cond_net()
copt = keras.optimizers.Adam(2e-3)

@tf.function
def cond_step(x0b, tb, cb, ab, eps):
    with tf.GradientTape() as tape:
        xt = tf.sqrt(ab)*x0b + tf.sqrt(1.0-ab)*eps
        loss = tf.reduce_mean(tf.square(eps - cnet([xt, tb, cb], training=True)))
    copt.apply_gradients(zip(tape.gradient(loss, cnet.trainable_weights),
                             cnet.trainable_weights))
    return loss

for step in range(2500):
    idx = r.integers(0, len(Xm), BATCH)
    tt = r.integers(0, T, BATCH)
    lab = ym[idx].astype("int32").copy()
    drop = r.random(BATCH) < 0.10               # DROP the condition 10 % of time
    lab[drop] = 2                               # the "unconditional" token
    cond_step(tf.constant(Xm[idx]), tf.constant(timestep_embedding(tt)),
              tf.constant(lab[:, None]), tf.constant(abar[tt][:, None]),
              tf.constant(r.normal(0, 1, (BATCH, 2)).astype("float32")))
print("  trained ONE network on both conditional and unconditional denoising")

def guided_sample(cls, w, n=900, n_steps=30, seed=2):
    rr = np.random.default_rng(seed)
    x = rr.normal(0, 1, (n, 2)).astype("float32")
    ts = np.linspace(T-1, 0, n_steps).astype(int)
    c_on = np.full((n, 1), cls, dtype="int32")
    c_off = np.full((n, 1), 2, dtype="int32")
    for i, t in enumerate(ts):
        tb = timestep_embedding(np.full(n, t))
        e_c = cnet.predict([x, tb, c_on], verbose=0)
        e_u = cnet.predict([x, tb, c_off], verbose=0)
        e = e_u + w*(e_c - e_u)                 # THE GUIDANCE FORMULA
        a_t = abar[t]; a_prev = abar[ts[i+1]] if i+1 < len(ts) else 1.0
        x0p = (x - np.sqrt(1-a_t)*e)/np.sqrt(a_t)
        x = (np.sqrt(a_prev)*x0p + np.sqrt(max(0., 1-a_prev))*e).astype("float32")
    return x

print()
print(f"{'guidance w':>12}{'% in the target moon':>24}{'sample std':>14}")
for w in [0.0, 1.0, 3.0, 7.0, 15.0]:
    S = guided_sample(0, w)
    # which moon is each sample nearest to?
    d0 = np.linalg.norm(S[:, None] - Xm[ym == 0][None, :400], axis=-1).min(1)
    d1 = np.linalg.norm(S[:, None] - Xm[ym == 1][None, :400], axis=-1).min(1)
    print(f"{w:>12.1f}{np.mean(d0 < d1):>24.1%}{S.std():>14.4f}")
print()
print("  w=0 ignores the condition entirely (it IS the unconditional model).")
print("  w=1 is ordinary conditional sampling.")
print("  larger w matches the condition harder AND reduces diversity --")
print("  the std falls. That is the trade-off, not a free quality knob.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("real data", "diffusion samples"))
fig.add_trace(go.Scatter(x=Xm[:1200, 0], y=Xm[:1200, 1], mode="markers",
                         marker=dict(size=4, color=ym[:1200],
                                     colorscale="Portland", showscale=False),
                         showlegend=False), 1, 1)
fig.add_trace(go.Scatter(x=samples[:1200, 0], y=samples[:1200, 1],
                         mode="markers",
                         marker=dict(size=4, color=C["danger"]),
                         showlegend=False), 1, 2)
fig.update_yaxes(scaleanchor="x")
fig.update_layout(height=420)
''',
        key="ch17_diffusion",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "What are the main tasks that autoencoders are used for?",
        "The main tasks are:\n\n"
        "* **Feature extraction** — the codings are a learned, compact "
        "representation.\n"
        "* **Unsupervised pretraining** (§17.2) — reuse the encoder when labels "
        "are scarce.\n"
        "* **Dimensionality reduction and visualisation** — a non-linear "
        "generalisation of PCA (§17.1).\n"
        "* **Generative modelling** — with a probabilistic latent space (a VAE, "
        "§17.6).\n"
        "* **Anomaly detection** — an autoencoder is generally bad at "
        "reconstructing inputs unlike its training data, so a high reconstruction "
        "error is a strong anomaly signal. This is the most common production "
        "use.\n"
        "* **Denoising and inpainting** (§17.4) — and, as it turned out, the "
        "foundation of diffusion models.")

    exercise(
        2, "Suppose you want to train a classifier, and you have plenty of "
        "unlabelled training data but only a few thousand labelled instances. "
        "How can autoencoders help? How would you proceed?",
        "You can first train a **deep autoencoder on the full dataset** (labelled "
        "+ unlabelled), then **reuse its lower half** as the lower half of the "
        "classifier — the encoder becomes the feature extractor — and train the "
        "classifier using the labelled data.\n\n"
        "If you have very little labelled data, you may want to **freeze the "
        "reused layers** when training the classifier, so the few labels cannot "
        "destroy the learned features. The §17.2 lab measures exactly this: with "
        "100 labels the frozen pretrained encoder is worth several accuracy "
        "points; with 8 000 the advantage nearly vanishes.\n\n"
        "The assumption being made is that features useful for reconstruction "
        "overlap with features useful for the label. When that is false — "
        "reconstructing a face requires modelling the background, which is "
        "irrelevant to identity — modern **self-supervised** objectives "
        "(contrastive learning, masked modelling) are the better choice, because "
        "they ask for something semantic rather than for every pixel.")

    exercise(
        3, "If an autoencoder perfectly reconstructs the inputs, is it "
        "necessarily a good autoencoder? How can you evaluate the performance of "
        "an autoencoder?",
        "**No.** The fact that an autoencoder perfectly reconstructs its inputs "
        "does not necessarily mean it is a good autoencoder: perhaps it is simply "
        "an **overcomplete autoencoder that learned to copy** its inputs to the "
        "codings layer and then to the outputs. In fact, even if the codings "
        "layer contained a single neuron, it would be possible for a very deep "
        "autoencoder to learn to map each training instance to a different "
        "coding (e.g. instance $i$ to coding $i$) — memorising the training set "
        "without learning any useful pattern.\n\n"
        "**To evaluate an autoencoder**, one option is to measure the "
        "reconstruction loss (e.g. MSE) — but that alone is exactly what the "
        "above shows can be gamed. Better options:\n\n"
        "* A **linear probe** on the codings (§17.2's lab): train a logistic "
        "regression on the codings and measure its accuracy. This is the honest "
        "measure.\n"
        "* If the autoencoder is for **pretraining**, measure the downstream "
        "classifier's accuracy directly.\n"
        "* If it is for **anomaly detection**, measure the detection rate at a "
        "fixed false-alarm rate.")

    exercise(
        4, "What are undercomplete and overcomplete autoencoders? What is the "
        "main risk of an excessively undercomplete autoencoder? What about the "
        "main risk of an overcomplete autoencoder?",
        "An **undercomplete autoencoder** is one whose codings layer is smaller "
        "than the input and output layers. If it is larger, it is an "
        "**overcomplete autoencoder**.\n\n"
        "The main risk of an **excessively undercomplete** autoencoder is that it "
        "may **fail to reconstruct the inputs** — the bottleneck simply cannot "
        "carry enough information, and both the reconstructions and the codings "
        "are poor.\n\n"
        "The main risk of an **overcomplete** autoencoder is that it may just "
        "**copy the inputs to the outputs**, without learning any useful feature. "
        "That is why an overcomplete autoencoder always needs another constraint: "
        "sparsity (§17.5), noise (§17.4), or a prior on the latent distribution "
        "(§17.6).")

    exercise(
        5, "How do you tie weights in a stacked autoencoder? What is the point of "
        "doing so?",
        "To tie the weights of an encoder layer and its corresponding decoder "
        "layer, you simply make the **decoder weights equal to the transpose of "
        "the encoder weights**:\n\n"
        "$\\mathbf{W}_{\\text{dec}}^{(\\ell)} = "
        "\\bigl(\\mathbf{W}_{\\text{enc}}^{(L-\\ell+1)}\\bigr)^{\\top}$\n\n"
        "This **halves the number of parameters** in the model, which speeds up "
        "training and — more importantly — **limits the risk of overfitting**. It "
        "is a structural regulariser, most valuable on small datasets.\n\n"
        "In Keras this is done with a custom layer (the `DenseTranspose` class in "
        "§17.2's lab) that holds a reference to the corresponding encoder layer "
        "and uses `tf.matmul(z, dense.kernel, transpose_b=True)`. Note the "
        "**biases stay separate** — only the kernels are shared.")

    exercise(
        6, "What is a generative model? Can you name a type of generative "
        "autoencoder?",
        "A **generative model** is a model capable of randomly generating outputs "
        "that resemble the training instances. For example, once trained "
        "successfully on the MNIST dataset, a generative model can be used to "
        "randomly generate realistic images of digits. The output distribution is "
        "typically similar to the training data, and since generation is "
        "stochastic you get different outputs every time.\n\n"
        "A **variational autoencoder** (§17.6) is a generative autoencoder. Its "
        "key properties: the encoder outputs a **distribution** rather than a "
        "point, and the ELBO includes a **KL term** pulling that distribution "
        "toward a standard normal prior. That regularisation is what makes the "
        "latent space continuous and gap-free, so a randomly sampled "
        "$\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ decodes to "
        "something plausible. A plain autoencoder's latent space has holes: "
        "sample from it at random and you get nothing.")

    exercise(
        7, "What is a GAN? Can you name a few tasks where GANs can shine?",
        "A **generative adversarial network** is an architecture composed of two "
        "parts, the **generator** and the **discriminator**, which have opposing "
        "objectives. The generator's goal is to generate instances similar to "
        "those in the training set, to fool the discriminator. The "
        "discriminator's goal is to tell apart the real instances from the "
        "generated ones. At each training iteration, the discriminator is trained "
        "like a normal binary classifier, then the generator is trained to "
        "maximise the discriminator's error (§17.7).\n\n"
        "**Tasks where GANs shine:**\n\n"
        "* Advanced **image processing** — super-resolution, colourisation, "
        "powerful image editing (replacing photobombers with realistic "
        "background), turning simple sketches into photorealistic images.\n"
        "* **Image-to-image translation** — day to night, sketch to photo "
        "(pix2pix, CycleGAN).\n"
        "* **Data augmentation** — generating additional training data.\n"
        "* Generating other types of data (text, audio, time series), and "
        "**identifying weaknesses in other models** to strengthen them "
        "(adversarial training).\n\n"
        "GANs remain attractive where **sampling speed** matters, since they "
        "generate in a single forward pass, unlike diffusion models.")

    exercise(
        8, "What are the main difficulties when training GANs?",
        "GANs are notoriously difficult to train because of the complex dynamics "
        "between the generator and the discriminator. The greatest difficulty is "
        "**mode collapse**, where the generator produces outputs with very little "
        "diversity. Moreover, training can be terribly **unstable**: it may start "
        "out fine and then suddenly start oscillating or diverging, without any "
        "apparent reason.\n\n"
        "GANs are also very **sensitive to the choice of hyperparameters** — much "
        "more so than a normal supervised model.\n\n"
        "The structural reasons (§17.7):\n\n"
        "* It is a **game**, not an optimisation. Gradient descent–ascent orbits "
        "the equilibrium; even on $V(d,g) = dg$ simultaneous updates spiral "
        "outward.\n"
        "* The objective is the **Jensen–Shannon divergence**, which is the "
        "constant $\\log 2$ when the two supports are disjoint — the normal case "
        "in high dimensions — so its gradient is uninformative.\n"
        "* Nothing in the loss rewards **coverage**, so mode collapse is not "
        "punished.\n"
        "* **The loss is not a progress metric**: an improving discriminator "
        "raises the generator's loss, so the numbers tell you nothing.")

    exercise(
        9, "What are diffusion models good at? What is their main limitation?",
        "**Diffusion models are good at** generating diverse and high-quality "
        "images. They are much **easier to train** than GANs (§17.9): the loss is "
        "a plain regression — predict the added noise — with no adversary, no "
        "equilibrium, and no collapse. They also have **much better mode "
        "coverage** than GANs, and they scale reliably: more compute and data "
        "make them monotonically better, which is the property that matters at "
        "scale.\n\n"
        "**Their main limitation** is that generating an image requires running "
        "the reverse diffusion process, which means **many forward passes** "
        "through the network — originally 1 000, now typically 20–50. That is "
        "far slower than a GAN or VAE, which generate in a single pass.\n\n"
        "This gap has narrowed substantially: **DDIM** takes larger deterministic "
        "steps along the same probability-flow path, and **distillation** and "
        "**consistency models** bring generation down to 1–4 steps. A secondary "
        "limitation is that the latent space is the **same size as the data**, so "
        "there is no compact semantic latent to edit — which is why latent "
        "diffusion runs the process inside a VAE's latent space instead.")

    exercise(
        10, "Try using a denoising autoencoder to pretrain an image classifier. "
        "You can use MNIST (the simplest option), or a more complex image "
        "dataset such as CIFAR10 if you want a bigger challenge. Regardless of "
        "the dataset you're using, follow these steps: split the dataset into a "
        "training set and a test set; train a deep denoising autoencoder on the "
        "full training set; check that the images are fairly well reconstructed; "
        "visualize the images that most activate each neuron in the coding layer; "
        "build a classification DNN, reusing the lower layers of the autoencoder; "
        "train it using only 10 % of the training set. Can you get it to perform "
        "as well as the same classifier trained on the full training set?",
        "This is §17.2's lab combined with §17.4's, and the honest answer to the "
        "final question is usually **no, but you close much of the gap** — and "
        "the size of the gap is the interesting measurement.\n\n"
        "Practical notes:\n\n"
        "* Use **dropout (masking) noise** rather than Gaussian for the "
        "pretraining, because reconstructing from a partial observation forces a "
        "more semantic representation than smoothing does.\n"
        "* **Freeze** the reused layers first and train only the head; then "
        "unfreeze and fine-tune at a much lower learning rate. Unfreezing "
        "immediately with 10 % of the labels destroys the pretrained features.\n"
        "* To **visualise what most activates a coding neuron**, either take the "
        "top-$k$ training images by that neuron's activation, or optimise an "
        "input image by gradient ascent on that activation.\n"
        "* Report the **linear probe** accuracy alongside the fine-tuned "
        "accuracy: it separates 'the representation is good' from 'the "
        "fine-tuning worked'.")

    exercise(
        11, "Train a variational autoencoder on the image dataset of your choice, "
        "and use it to generate images. Alternatively, you can try to find an "
        "unlabeled dataset that you are interested in and see if you can generate "
        "new samples.",
        "§17.6's lab does exactly this on Fashion-MNIST. The things worth "
        "measuring as you go:\n\n"
        "* **Per-dimension KL.** A latent dimension with KL $\\approx 0$ has "
        "collapsed — it carries no information about the input. Count them.\n"
        "* **KL warm-up.** Anneal $\\beta$ from 0 to 1 over the first several "
        "epochs. This alone usually eliminates collapse, because the decoder "
        "learns to use $\\mathbf{z}$ before the KL pressure arrives.\n"
        "* **Latent interpolation.** Interpolate between two encoded images and "
        "decode along the path. If the intermediate images are plausible, the "
        "latent space is smooth; if they ghost one image over the other, "
        "something is wrong.\n"
        "* **Expect blurriness.** It is a property of the Gaussian/Bernoulli "
        "likelihood, not of your capacity — the optimum of a squared loss over "
        "several plausible completions genuinely *is* their average.")

    exercise(
        12, "Train a DCGAN to tackle the image dataset of your choice, and use it "
        "to generate images. Add experience replay and see if this helps. Turn it "
        "into a conditional GAN where you can control the generated class.",
        "§17.8's lab trains the DCGAN. The two extensions:\n\n"
        "**Experience replay.** Keep a buffer of previously generated images and "
        "train the discriminator on a mix of current and past fakes. This "
        "prevents the discriminator from overfitting to whatever the generator is "
        "producing *right now*, which is one cause of the oscillation in §17.7 — "
        "it damps the cycle by giving $D$ a memory.\n\n"
        "**Conditional GAN.** Feed the class label to both networks: concatenate "
        "an embedding of the label to $\\mathbf{z}$ in the generator, and to the "
        "input (as extra channels) or to the final features in the discriminator. "
        "The discriminator must then judge whether the image is real **and** "
        "matches its label — that pairing is what forces the generator to respect "
        "the condition. A projection discriminator (Miyato & Koyama) does this "
        "more effectively than concatenation at scale.\n\n"
        "Track **mode coverage per class**: a conditional GAN can collapse within "
        "a class while still producing all classes, which a per-class effective "
        "rank will reveal and an overall one will not.")

    rule()

    sub("The three families, side by side")

    table(
        ["Choose", "When", "Because"],
        [["<b>Autoencoder</b>", "Anomaly detection, compression, pretraining",
          "Cheap, stable, and the reconstruction error is directly useful"],
         ["<b>VAE</b>", "You need a compact, smooth, editable latent space",
          "The KL term guarantees you can sample from the prior"],
         ["<b>GAN</b>", "Single-pass sampling matters and you can afford the "
          "tuning",
          "One forward pass per sample, and very sharp output"],
         ["<b>Diffusion</b>", "Quality and coverage matter more than speed",
          "A stable regression loss that scales reliably"],
         ["<b>Latent diffusion</b>", "High-resolution generation",
          "A VAE compresses; diffusion runs in the small latent space"]],
    )

    keypoints([
        "The forward process has a <b>closed form</b>: $\\mathbf{x}_t = "
        "\\sqrt{\\bar\\alpha_t}\\mathbf{x}_0 + "
        "\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\varepsilon$ — no chain "
        "simulation.",
        "The loss is a <b>plain MSE on the noise</b> — a denoising autoencoder "
        "(§17.4) conditioned on the noise level.",
        "By Tweedie, $\\boldsymbol\\varepsilon_\\theta$ is a scaled score: "
        "<b>diffusion is score matching with a schedule</b>.",
        "Diffusion won on <b>training stability</b>, not sample quality — no "
        "adversary means no collapse.",
        "<b>Classifier-free guidance</b> trades diversity for prompt adherence; "
        "it is not a free quality knob.",
    ], title="Chapter 17 in five lines")

    refs([
        ("Hinton & Salakhutdinov — *Reducing the Dimensionality of Data with "
         "Neural Networks*", "https://doi.org/10.1126/science.1127647"),
        ("Vincent et al. — *Extracting and Composing Robust Features with "
         "Denoising Autoencoders*",
         "https://doi.org/10.1145/1390156.1390294"),
        ("Kingma & Welling — *Auto-Encoding Variational Bayes*",
         "https://arxiv.org/abs/1312.6114"),
        ("Higgins et al. — *β-VAE: Learning Basic Visual Concepts with a "
         "Constrained Variational Framework*",
         "https://openreview.net/forum?id=Sy2fzU9gl"),
        ("Goodfellow et al. — *Generative Adversarial Networks*",
         "https://arxiv.org/abs/1406.2661"),
        ("Radford, Metz & Chintala — *Unsupervised Representation Learning with "
         "DCGANs*", "https://arxiv.org/abs/1511.06434"),
        ("Arjovsky, Chintala & Bottou — *Wasserstein GAN*",
         "https://arxiv.org/abs/1701.07875"),
        ("Karras et al. — *A Style-Based Generator Architecture for GANs* "
         "(StyleGAN)", "https://arxiv.org/abs/1812.04948"),
        ("Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models*",
         "https://arxiv.org/abs/2006.11239"),
        ("Song et al. — *Score-Based Generative Modeling through Stochastic "
         "Differential Equations*", "https://arxiv.org/abs/2011.13456"),
        ("Ho & Salimans — *Classifier-Free Diffusion Guidance*",
         "https://arxiv.org/abs/2207.12598"),
        ("Rombach et al. — *High-Resolution Image Synthesis with Latent "
         "Diffusion Models*", "https://arxiv.org/abs/2112.10752"),
    ])


# ==========================================================================
SECTIONS = [
    ("17.1", "Autoencoders and PCA", s_17_1),
    ("17.2", "Stacked AEs & Pretraining", s_17_2),
    ("17.3", "Convolutional & Recurrent AEs", s_17_3),
    ("17.4", "Denoising Autoencoders", s_17_4),
    ("17.5", "Sparse Autoencoders", s_17_5),
    ("17.6", "Variational Autoencoders", s_17_6),
    ("17.7", "Generative Adversarial Networks", s_17_7),
    ("17.8", "DCGAN & StyleGAN", s_17_8),
    ("17.9", "Diffusion Models & Exercises", s_17_9),
]

nav.render_chapter(CH, SECTIONS)
