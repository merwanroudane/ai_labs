"""Course home — dashboard, animated hero, and the full course map."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, nav
from core.lecture import figure, hero, keypoints, md, rule, section, table
from core.palette import C, SEQ, PARULA
from core.theme import inject

inject()

hero(
    kicker="Interactive course · 19 chapters · 4 appendices",
    title="Hands-On Machine Learning — the interactive platform",
    blurb=(
        "Every chapter is a full lecture: the intuition, the complete mathematics "
        "rendered in LaTeX, an animation you drive with a <b>▶ Play</b> button, and "
        "code labs you edit and execute right on the page. Nothing is abbreviated — "
        "the derivations are written out, the algorithms are implemented from "
        "scratch beside their library equivalents, and every figure is live Plotly."
    ),
    chips=["Scikit-Learn", "Keras / TensorFlow", "Plotly animations",
           "LaTeX derivations", "Executable labs", "AI Lab workbench"],
)

# --------------------------------------------------------------------------
# Animated hero: the three regimes of fitting
# --------------------------------------------------------------------------

section("▶", "The whole course in one animation")

md(
    "Machine learning, stripped to its core, is the search for a function that "
    "is **flexible enough to capture the signal** but **rigid enough to ignore "
    "the noise**. Press play: the model's flexibility (polynomial degree) grows "
    "from 1 to 20, and you watch training error fall forever while test error "
    "turns around. Everything in the next 19 chapters is a variation on this "
    "single picture."
)

rng = np.random.default_rng(0)
n = 40
Xf = np.sort(rng.uniform(-3, 3, n))
yf = np.sin(1.5 * Xf) + 0.35 * Xf + rng.normal(0, 0.32, n)
Xt = np.sort(rng.uniform(-3, 3, 200))
yt = np.sin(1.5 * Xt) + 0.35 * Xt + rng.normal(0, 0.32, 200)
grid = np.linspace(-3.2, 3.2, 400)

degrees = list(range(1, 21))
curves, tr_err, te_err = [], [], []
for d in degrees:
    coef = np.polyfit(Xf, yf, d)
    curves.append(np.polyval(coef, grid))
    tr_err.append(np.sqrt(np.mean((np.polyval(coef, Xf) - yf) ** 2)))
    te_err.append(np.sqrt(np.mean((np.polyval(coef, Xt) - yt) ** 2)))

base = [
    go.Scatter(x=Xf, y=yf, mode="markers", name="training sample",
               marker=dict(color=C["train"], size=9,
                           line=dict(color="#fff", width=1.4))),
    go.Scatter(x=grid, y=curves[0], mode="lines", name="fitted model",
               line=dict(color=C["primary"], width=3.4)),
    go.Scatter(x=grid, y=np.sin(1.5 * grid) + 0.35 * grid, mode="lines",
               name="true function", line=dict(color=C["truth"], width=2,
                                               dash="dot")),
]

frames = []
for i, d in enumerate(degrees):
    tag = ("under-fitting" if d <= 2 else
           "about right" if d <= 6 else "over-fitting")
    col = (C["warning"] if d <= 2 else
           C["success"] if d <= 6 else C["danger"])
    frames.append(go.Frame(
        name=str(d),
        data=[
            go.Scatter(x=Xf, y=yf, mode="markers",
                       marker=dict(color=C["train"], size=9,
                                   line=dict(color="#fff", width=1.4))),
            go.Scatter(x=grid, y=curves[i], mode="lines",
                       line=dict(color=col, width=3.4)),
            go.Scatter(x=grid, y=np.sin(1.5 * grid) + 0.35 * grid, mode="lines",
                       line=dict(color=C["truth"], width=2, dash="dot")),
        ],
        layout=go.Layout(annotations=[
            anim.annotate_step(
                f"degree = {d:>2}   train RMSE = {tr_err[i]:.3f}   "
                f"test RMSE = {te_err[i]:.3f}   → {tag}", color=col)
        ]),
    ))

fig = go.Figure(data=base)
fig.update_layout(
    height=470, yaxis=dict(range=[-3.2, 3.2]), xaxis=dict(range=[-3.3, 3.3]),
    title="Model capacity sweep: degree 1 → 20 on the same 40 points",
    xaxis_title="x", yaxis_title="y",
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
)
anim.animate(fig, frames, duration=430, slider_prefix="polynomial degree = ")
figure(fig, "The bias–variance trade-off, animated. Chapter 4 formalises it.")

col1, col2 = st.columns([1, 1])
with col1:
    err = go.Figure()
    err.add_trace(go.Scatter(x=degrees, y=tr_err, mode="lines+markers",
                             name="training RMSE",
                             line=dict(color=C["train"], width=3)))
    err.add_trace(go.Scatter(x=degrees, y=te_err, mode="lines+markers",
                             name="test RMSE",
                             line=dict(color=C["test"], width=3)))
    err.add_vrect(x0=3, x1=6, fillcolor=C["success"], opacity=0.12,
                  line_width=0, annotation_text="sweet spot",
                  annotation_position="top left")
    err.update_layout(height=330, title="Error vs capacity",
                      xaxis_title="polynomial degree", yaxis_title="RMSE",
                      yaxis_type="log")
    figure(err)

with col2:
    md(
        """
        **How to use this platform**

        1. Pick a chapter in the sidebar. Each chapter opens with its own
           sub-section list — you move through them with the radio buttons or the
           **← / →** buttons at the bottom of the page.
        2. Read the lecture. Equations are real LaTeX; every non-obvious one has a
           foldable **📐 Derivation** with every algebraic step written out.
        3. Play the animations. They are not decoration — each one shows the
           mechanism the equations describe, one frame per iteration/step/epoch.
        4. Run the labs. Open **✏️ Edit the source**, change anything, press
           **▶ Run**. Variables persist between runs inside a chapter, so you can
           build up an analysis across several labs.
        5. Finish in the **🧪 AI Lab**, where you assemble your own pipeline on
           your own data.
        """
    )

# --------------------------------------------------------------------------
rule()
section("🗺️", "Course map")

CHAPTERS = [
    ("Part I", "1", "The Machine Learning Landscape",
     "Supervision types, batch vs online, instance vs model based, the four "
     "ways data betrays you, train/valid/test discipline"),
    ("Part I", "2", "End-to-End ML Project",
     "Framing, performance measures, stratified sampling, EDA, imputation, "
     "encoding, scaling, custom transformers, pipelines, grid & random search"),
    ("Part I", "3", "Classification",
     "Confusion matrix, precision/recall trade-off, ROC & PR curves, "
     "multiclass, error analysis, multilabel, multioutput"),
    ("Part I", "4", "Training Models",
     "Normal equation, SVD, gradient descent in three flavours, polynomial "
     "features, learning curves, ridge/lasso/elastic net, logistic & softmax"),
    ("Part I", "5", "Support Vector Machines",
     "Margins, hinge loss, the kernel trick, dual problem, SVM regression"),
    ("Part I", "6", "Decision Trees",
     "CART, Gini vs entropy, regularisation, regression trees, instability"),
    ("Part I", "7", "Ensembles & Random Forests",
     "Voting, bagging, OOB, random forests, extra-trees, AdaBoost, gradient "
     "boosting, histogram boosting, stacking"),
    ("Part I", "8", "Dimensionality Reduction",
     "Curse of dimensionality, projection vs manifold, PCA & its variants, "
     "random projection, LLE, t-SNE"),
    ("Part I", "9", "Unsupervised Learning",
     "k-means, DBSCAN, image segmentation, semi-supervised labelling, "
     "Gaussian mixtures, anomaly detection, model selection with BIC"),
    ("Part II", "10", "Introduction to ANNs with Keras",
     "Perceptron, MLP, backpropagation, Sequential/Functional/Subclassing "
     "APIs, callbacks, TensorBoard, hyperparameter tuning"),
    ("Part II", "11", "Training Deep Neural Networks",
     "Vanishing gradients, Glorot/He init, activations, batch norm, gradient "
     "clipping, transfer learning, momentum→AdamW, LR schedules, dropout"),
    ("Part II", "12", "Custom Models & Training with TensorFlow",
     "Tensors, variables, custom losses/metrics/layers/models, autodiff, "
     "custom training loops, tf.function and graphs"),
    ("Part II", "13", "Loading & Preprocessing Data",
     "tf.data pipelines, shuffling, interleaving, prefetching, TFRecord, "
     "protobufs, Keras preprocessing layers, embeddings, TFDS"),
    ("Part II", "14", "Deep Computer Vision with CNNs",
     "Convolution arithmetic, filters, feature maps, pooling, LeNet→SENet, "
     "transfer learning, localisation, YOLO, semantic segmentation"),
    ("Part II", "15", "Processing Sequences with RNNs & CNNs",
     "Recurrent neurons, BPTT, forecasting, ARMA family, deep & multivariate "
     "RNNs, seq2seq, LSTM, GRU, dilated conv, WaveNet"),
    ("Part II", "16", "NLP with RNNs & Attention",
     "Char-RNN, stateful RNNs, sentiment, masking, encoder–decoder, beam "
     "search, attention, the Transformer, BERT/GPT, ViT, Hugging Face"),
    ("Part II", "17", "Autoencoders, GANs & Diffusion",
     "Undercomplete AE = PCA, stacked/conv/denoising/sparse AEs, VAEs, GAN "
     "minimax, DCGAN, progressive growing, StyleGAN, DDPM"),
    ("Part II", "18", "Reinforcement Learning",
     "Rewards, policy search, policy gradients, MDPs, Bellman, TD learning, "
     "Q-learning, DQN and its variants"),
    ("Part II", "19", "Training & Deploying at Scale",
     "TF Serving, REST/gRPC, mobile & web deployment, GPU management, "
     "model vs data parallelism, distribution strategies, cluster training"),
]

rows = []
for part, num, title, blurb in CHAPTERS:
    pill = "mp-pill-a" if part == "Part I" else "mp-pill-b"
    rows.append([
        f'<span class="mp-pill {pill}">{part}</span>',
        f"<b>{num}</b>",
        f"<b>{title}</b>",
        f'<span style="color:{C["ink_soft"]};font-size:.9rem">{blurb}</span>',
    ])
table(["", "#", "Chapter", "What it covers"], rows)

keypoints(
    [
        "Every chapter has <b>sub-sections</b> in the sidebar — nothing is collapsed "
        "into a single scroll.",
        "Animations are <code>plotly</code> frame sequences: <b>▶ Play</b>, <b>⏸ Pause</b>, "
        "<b>⏮ Reset</b>, plus a slider so you can step frame by frame.",
        "Code labs execute in a <b>persistent namespace</b> per lab — define a variable "
        "in one run, use it in the next.",
        "The colourscale (default <code>Parula</code>) and animation speed are global "
        "controls in each chapter's sidebar.",
        "Deep-learning chapters detect TensorFlow and PyTorch; if they are missing the "
        "labs degrade to NumPy implementations rather than crashing.",
    ],
    title="Platform features",
)
