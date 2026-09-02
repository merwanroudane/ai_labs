"""Chapter 3 — Classification."""

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
CH = "ch03"

hero(
    kicker="Part I · Chapter 3",
    title="Classification",
    blurb=(
        "Accuracy is the worst metric in machine learning and almost everybody's "
        "first choice. This chapter replaces it: confusion matrices, precision and "
        "recall and the trade-off that binds them, ROC and PR curves, and the "
        "error analysis that turns a mediocre classifier into a good one. Then "
        "multiclass, multilabel and multioutput."
    ),
    chips=["MNIST-style digits", "9 sub-sections", "7 animations",
           "8 code labs", "Metrics done properly"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_3_1():
    section("3.1", "MNIST — The Hello World of Classification")

    lead(
        "70 000 small images of handwritten digits, each labelled with the digit "
        "it represents. It is so heavily studied that it functions as a "
        "calibration standard: if your new idea cannot beat a linear model here, "
        "it does not work."
    )

    X, y, images = ds.digits()

    md(
        f"""
This platform uses scikit-learn's **8×8 digits** set ({len(X):,} images), which
loads instantly and offline. Everything below transfers verbatim to the full
28×28 MNIST — the shapes change, the code does not. The last code lab in this
sub-section shows you how to fetch the real thing.

Each image is flattened into a feature vector:
        """
    )

    math(r"""
    \mathbf{x}^{(i)} \in \mathbb{R}^{n},
    \qquad n = 8 \times 8 = 64
    \quad\bigl(\text{or } 28 \times 28 = 784 \text{ for full MNIST}\bigr),
    \qquad y^{(i)} \in \{0, 1, \dots, 9\}
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", f"{len(X):,}")
    c2.metric("Pixels per image", X.shape[1])
    c3.metric("Classes", len(np.unique(y)))
    c4.metric("Pixel range", f"0 – {int(X.max())}")

    sub("Look at the data")

    idx = np.concatenate([np.where(y == d)[0][:8] for d in range(10)])
    grid_img = np.zeros((10 * 8, 8 * 8))
    for k, i in enumerate(idx):
        r, c = divmod(k, 8)
        grid_img[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = images[i]

    gimg = go.Figure(go.Heatmap(z=grid_img[::-1], colorscale=nav.cscale(),
                                showscale=False))
    gimg.update_layout(height=620, title="Eight examples of each digit, 0 → 9 "
                                         "(top to bottom)",
                       xaxis=dict(visible=False, scaleanchor="y"),
                       yaxis=dict(visible=False))
    figure(gimg, "Handwriting varies enormously — that variability is the whole "
                 "difficulty.")

    anim_header("One image, pixel by pixel: how a digit becomes a vector")
    md(
        "The left panel fills in the 64 pixels in reading order; the right panel "
        "grows the corresponding feature vector. By the last frame you are looking "
        "at the same object twice — a picture and a point in $\\mathbb{R}^{64}$."
    )

    img0 = images[np.where(y == 3)[0][0]]
    flat = img0.ravel()
    frames = []
    for k in range(1, 65):
        partial = np.full(64, np.nan)
        partial[:k] = flat[:k]
        frames.append(go.Frame(name=str(k), data=[
            go.Heatmap(z=partial.reshape(8, 8)[::-1], colorscale=nav.cscale(),
                       zmin=0, zmax=16, showscale=False, xgap=1, ygap=1),
            go.Bar(x=list(range(64)), y=np.nan_to_num(partial),
                   marker=dict(color=np.nan_to_num(partial),
                               colorscale=nav.cscale(), cmin=0, cmax=16)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"pixel {k}/64   value = {flat[k-1]:.0f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.36, .64],
                      subplot_titles=("the 8×8 image", "the 64-dim feature vector"))
    f.add_trace(go.Heatmap(z=np.full((8, 8), np.nan), colorscale=nav.cscale(),
                           zmin=0, zmax=16, showscale=False, xgap=1, ygap=1), 1, 1)
    f.add_trace(go.Bar(x=list(range(64)), y=np.zeros(64), showlegend=False,
                       marker=dict(color=np.zeros(64), colorscale=nav.cscale(),
                                   cmin=0, cmax=16)), 1, 2)
    f.update_xaxes(visible=False, row=1, col=1)
    f.update_yaxes(visible=False, row=1, col=1)
    f.update_xaxes(title_text="pixel index", row=1, col=2)
    f.update_yaxes(range=[0, 17], title_text="intensity", row=1, col=2)
    f.update_layout(height=400, title="An image is just a point in high-dimensional space")
    anim.animate(f, frames, duration=nav.anim_ms(45), slider_prefix="pixel ")
    figure(f)

    sub("Split — and beware of ordering")

    md(
        "MNIST arrives pre-split (60 000 train / 10 000 test) and pre-shuffled. "
        "Both matter. Shuffling guarantees that cross-validation folds are "
        "similar, and it protects algorithms that are sensitive to instance order."
    )

    warn(
        "But do not shuffle blindly",
        "Shuffling is wrong whenever rows are <b>not exchangeable</b>: time series "
        "(you would let the future leak into the past), or grouped data such as "
        "several images of the same handwriting sample (the same writer would "
        "appear in both train and test, inflating your score). Use "
        "<code>TimeSeriesSplit</code> or <code>GroupKFold</code> instead.",
    )

    code_lab(
        "Load, inspect, split — and how to get real MNIST",
        '''import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target
print(f"X shape {X.shape}   y shape {y.shape}   dtype {X.dtype}")
print(f"pixel range [{X.min():.0f}, {X.max():.0f}]")
print(f"class counts: {np.bincount(y)}")
print(f"most frequent class = {np.bincount(y).argmax()} "
      f"({np.bincount(y).max()/len(y):.1%} of the data)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\\ntrain {X_train.shape}   test {X_test.shape}")
print(f"train class balance: {np.bincount(y_train)/len(y_train)}")

# a single image, three ways of looking at it
i = 0
print(f"\\nimage {i} is a '{y_train[i]}'")
print("as an 8x8 grid of intensities:")
print(X_train[i].reshape(8, 8).astype(int))

# -------------------------------------------------------------------------
# To use the REAL 28x28 MNIST (needs an internet connection, ~15 s):
#
#   from sklearn.datasets import fetch_openml
#   mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
#   X, y = mnist.data, mnist.target.astype(np.uint8)
#   X_train, X_test = X[:60000], X[60000:]        # already shuffled
#   y_train, y_test = y[:60000], y[60000:]
#
# or via Keras (also downloads):
#   import tensorflow as tf
#   (Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
#   Xtr = Xtr.reshape(-1, 784) / 255.0
# -------------------------------------------------------------------------
print("\\n(see the comments in this cell for full 28x28 MNIST)")
''',
        key="ch03_mnist",
    )

    keypoints([
        "An image is a vector; classification on images is classification on "
        "$\\mathbb{R}^n$ with a very particular structure (which Chapter 14 "
        "exploits).",
        "MNIST-style data is the standard calibration bench — always try the "
        "simplest model first.",
        "Shuffle when rows are exchangeable; never shuffle time series or grouped "
        "data.",
    ])


# ==========================================================================
def s_3_2():
    section("3.2", "Training a Binary Classifier")

    lead(
        "Simplify the problem to one question — \"is this a 5?\" — because "
        "everything conceptually difficult about classification already appears in "
        "the binary case."
    )

    md(
        "Recode the target: $y^{(i)}_{\\text{bin}} = \\mathbb{1}\\bigl[y^{(i)} = "
        "5\\bigr]$. Now $\\mathcal{Y} = \\{0, 1\\}$ and we can use a linear "
        "classifier trained with stochastic gradient descent."
    )

    sub("SGDClassifier — what it actually does")

    md("`SGDClassifier` is not one model; it is *SGD applied to a linear model "
       "with a configurable loss*. The prediction rule is always the same:")

    math(r"""
    \hat y \;=\;
    \begin{cases}
      1 & \text{if } s(\mathbf{x}) = \boldsymbol\theta^\top \mathbf{x} + b \;\ge\; 0\\
      0 & \text{otherwise}
    \end{cases}
    """)

    md("What changes with `loss=` is the training objective:")

    table(
        ["<code>loss</code>", "Objective per example", "This model is called"],
        [["<code>'hinge'</code>",
          "$\\max\\bigl(0,\\; 1 - \\tilde y \\, s(\\mathbf{x})\\bigr)$",
          "Linear SVM (Chapter 5)"],
         ["<code>'log_loss'</code>",
          "$-\\bigl[y\\log\\sigma(s) + (1-y)\\log(1-\\sigma(s))\\bigr]$",
          "Logistic regression (Chapter 4)"],
         ["<code>'perceptron'</code>",
          "$\\max\\bigl(0,\\; -\\tilde y\\, s(\\mathbf{x})\\bigr)$",
          "The perceptron (Chapter 10)"],
         ["<code>'modified_huber'</code>",
          "A smoothed, outlier-tolerant hinge",
          "Robust linear classifier"]],
        "$\\tilde y \\in \\{-1, +1\\}$ is the ±1 encoding of the label.",
    )

    idea(
        "Every linear classifier is a hyperplane plus a decision rule",
        "$\\boldsymbol\\theta^\\top \\mathbf{x} + b = 0$ defines a hyperplane in "
        "$\\mathbb{R}^n$; $s(\\mathbf{x})$ is a signed, scaled distance to it. All "
        "the linear methods differ only in <b>where they put the hyperplane</b>, "
        "and that is decided entirely by the loss.",
    )

    anim_header("SGD walking to a decision boundary")
    md(
        "A 2-D binary problem. Each frame is one mini-batch update. Watch the "
        "hyperplane rotate and translate into place while the loss falls — this is "
        "literally what `SGDClassifier.fit` is doing in 64 dimensions."
    )

    rng = np.random.default_rng(1)
    n = 220
    Xa = np.r_[rng.normal([-1.4, -0.6], .85, (n // 2, 2)),
               rng.normal([1.5, 1.1], .85, (n // 2, 2))]
    ya = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    yt = 2 * ya - 1

    w, b = np.array([2.4, -2.2]), 0.6
    lr = 0.06
    hist_w, hist_b, hist_loss, hist_acc = [], [], [], []
    order = rng.permutation(n)
    for step in range(70):
        bidx = order[(step * 8) % n:(step * 8) % n + 16]
        if len(bidx) == 0:
            bidx = order[:16]
        s = Xa[bidx] @ w + b
        marg = yt[bidx] * s
        viol = marg < 1
        gw = -(yt[bidx][viol, None] * Xa[bidx][viol]).sum(0) / len(bidx) + 0.02 * w
        gb = -(yt[bidx][viol]).sum() / len(bidx)
        w = w - lr * gw
        b = b - lr * gb
        hist_w.append(w.copy()); hist_b.append(b)
        hist_loss.append(np.mean(np.maximum(0, 1 - yt * (Xa @ w + b))))
        hist_acc.append(np.mean(((Xa @ w + b) >= 0).astype(int) == ya))

    gx = np.linspace(Xa[:, 0].min() - .6, Xa[:, 0].max() + .6, 50)

    def boundary(wv, bv):
        if abs(wv[1]) < 1e-8:
            return np.full_like(gx, np.nan)
        return -(wv[0] * gx + bv) / wv[1]

    frames = []
    for k in range(len(hist_w)):
        frames.append(go.Frame(name=str(k + 1), data=[
            go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=gx, y=boundary(hist_w[k], hist_b[k]), mode="lines",
                       line=dict(color=C["primary"], width=4)),
            go.Scatter(x=list(range(1, k + 2)), y=hist_loss[:k + 1], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=list(range(1, k + 2)), y=hist_acc[:k + 1], mode="lines",
                       line=dict(color=C["success"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"update {k+1}   ·   hinge loss = {hist_loss[k]:.4f}   ·   "
            f"accuracy = {hist_acc[k]:.3f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.56, .44],
                      subplot_titles=("the hyperplane moving",
                                      "loss (red) and accuracy (green)"))
    f.add_trace(go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                           name="not a 5", marker=dict(color=C["train"], size=7,
                           line=dict(color="#fff", width=.8))), 1, 1)
    f.add_trace(go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                           name="is a 5", marker=dict(color=C["warning"], size=7,
                           line=dict(color="#fff", width=.8))), 1, 1)
    f.add_trace(go.Scatter(x=gx, y=boundary(hist_w[0], hist_b[0]), mode="lines",
                           name="decision boundary",
                           line=dict(color=C["primary"], width=4)), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=hist_loss[:1], mode="lines", name="hinge loss",
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=[1], y=hist_acc[:1], mode="lines", name="accuracy",
                           line=dict(color=C["success"], width=3)), 1, 2)
    f.update_yaxes(range=[Xa[:, 1].min() - .6, Xa[:, 1].max() + .6], row=1, col=1)
    f.update_xaxes(range=[gx.min(), gx.max()], row=1, col=1)
    f.update_yaxes(range=[0, 1.6], row=1, col=2)
    f.update_xaxes(range=[0, len(hist_w) + 1], title_text="SGD update", row=1, col=2)
    f.update_layout(height=450, title="Stochastic gradient descent on a hinge loss")
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="update ")
    figure(f)

    code_lab(
        "Train the 5-detector",
        '''import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = load_digits()
X, y = d.data, d.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42)

# binary target: "is this a 5?"
y_train_5 = (y_train == 5)
y_test_5  = (y_test  == 5)
print(f"positives in train: {y_train_5.sum()} / {len(y_train_5)} "
      f"= {y_train_5.mean():.1%}   <- remember this number")

sgd = make_pipeline(StandardScaler(),
                    SGDClassifier(loss="hinge", random_state=42, max_iter=2000))
sgd.fit(X_train, y_train_5)

# predictions on a few individual digits
for i in range(6):
    print(f"true digit {y_train[i]}  ->  is-a-5 prediction: "
          f"{bool(sgd.predict([X_train[i]])[0])}")

print(f"\\ntrain accuracy = {sgd.score(X_train, y_train_5):.4f}")
print(f"test  accuracy = {sgd.score(X_test,  y_test_5):.4f}")

# ---- the decision function is the signed distance to the hyperplane -------
scores = sgd.decision_function(X_test)
print(f"\\ndecision_function range: [{scores.min():.2f}, {scores.max():.2f}]")
print("predictions are simply  (score >= 0):",
      np.array_equal(sgd.predict(X_test), scores >= 0))

# ---- the same estimator, four different models ---------------------------
print("\\nsame class, different losses:")
for loss in ["hinge", "log_loss", "perceptron", "modified_huber"]:
    m = make_pipeline(StandardScaler(),
                      SGDClassifier(loss=loss, random_state=42, max_iter=2000))
    m.fit(X_train, y_train_5)
    print(f"  loss={loss:<16} test accuracy = {m.score(X_test, y_test_5):.4f}")
''',
        key="ch03_binary",
    )

    keypoints([
        "Binarise to isolate the concept; everything hard is already present.",
        "A linear classifier = a hyperplane $\\boldsymbol\\theta^\\top\\mathbf{x} + b$ "
        "plus the rule <i>predict 1 if the score is non-negative</i>.",
        "<code>SGDClassifier</code> is a family: the <code>loss</code> parameter "
        "chooses SVM / logistic / perceptron.",
        "<code>decision_function</code> gives the raw score — remember it, §3.4 "
        "depends on it entirely.",
    ])


# ==========================================================================
def s_3_3():
    section("3.3", "Performance Measures — Why Accuracy Lies")

    lead(
        "The most important twenty minutes in this chapter. Accuracy is "
        "meaningless on skewed data, and almost all interesting data is skewed."
    )

    sub("Accuracy, and the dumb classifier that beats you")

    math(r"""
    \mathrm{accuracy} \;=\;
    \frac{\text{number of correct predictions}}{\text{total predictions}}
    \;=\; \frac{TP + TN}{TP + TN + FP + FN}
    """)

    md(
        "About 10 % of digits are 5s. So a classifier that says **\"never a 5\"** "
        "to every image achieves **90 % accuracy** while being completely useless. "
        "It is not a straw man — it is the baseline every skewed problem has, and "
        "you must always compute it."
    )

    pitfall(
        "The 99.9 % accurate cancer detector",
        "If a disease has prevalence 0.1 %, the constant predictor \"healthy\" "
        "scores 99.9 % accuracy and detects zero patients. Accuracy is generally "
        "<b>not</b> the preferred measure for classifiers, especially when the "
        "classes are skewed. This is why the rest of the chapter exists.",
    )

    sub("The confusion matrix")

    md("Count every combination of true class and predicted class:")

    table(
        ["", "<b>Predicted: negative</b>", "<b>Predicted: positive</b>"],
        [["<b>Actual: negative</b>",
          "<b>TN</b> — true negative<br><span style='color:#06D6A0'>correct "
          "rejection</span>",
          "<b>FP</b> — false positive<br><span style='color:#EF476F'>type I error, "
          "false alarm</span>"],
         ["<b>Actual: positive</b>",
          "<b>FN</b> — false negative<br><span style='color:#EF476F'>type II "
          "error, miss</span>",
          "<b>TP</b> — true positive<br><span style='color:#06D6A0'>hit</span>"]],
        "The confusion matrix. Every metric below is a ratio of these four "
        "numbers.",
    )

    md("The row and column definitions give the four fundamental rates:")

    math(r"""
    \mathrm{precision} = \frac{TP}{TP + FP}
    \qquad
    \mathrm{recall} = \frac{TP}{TP + FN}
    """)
    math(r"""
    \mathrm{specificity} = \frac{TN}{TN + FP}
    \qquad
    \mathrm{FPR} = \frac{FP}{FP + TN} = 1 - \mathrm{specificity}
    """)

    where({
        r"\mathrm{precision}": "<b>of the instances I flagged, what fraction were "
                               "right?</b> — column-wise, reads down the predicted-"
                               "positive column",
        r"\mathrm{recall}": "<b>of the instances that really were positive, what "
                            "fraction did I catch?</b> — row-wise, reads across the "
                            "actual-positive row. Also called <i>sensitivity</i> or "
                            "the <i>true positive rate</i>",
        r"\mathrm{FPR}": "of the true negatives, what fraction did I wrongly flag",
    })

    idea(
        "Precision reads a column, recall reads a row",
        "That is the whole mnemonic. Precision has <b>FP</b> in the denominator "
        "(the other thing in the predicted-positive <i>column</i>); recall has "
        "<b>FN</b> (the other thing in the actual-positive <i>row</i>). Once you "
        "see the matrix geometrically you never confuse them again.",
    )

    sub("The F₁ score")

    md("A single number combining both — the **harmonic** mean:")

    math(r"""
    F_1 \;=\; \frac{2}{\dfrac{1}{\mathrm{precision}} + \dfrac{1}{\mathrm{recall}}}
    \;=\; 2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
                       {\mathrm{precision} + \mathrm{recall}}
    \;=\; \frac{TP}{TP + \dfrac{FN + FP}{2}}
    """)

    derive(
        [("Why harmonic and not arithmetic? Because the harmonic mean is dominated "
          "by the smaller value, so it refuses to reward a lopsided classifier.",
          None),
         ("Take precision $= 1.0$ and recall $= 0.02$ — a classifier that flags one "
          "single instance and gets it right. The arithmetic mean is generous:",
          r"\frac{1.00 + 0.02}{2} = 0.510"),
         ("The harmonic mean is not:",
          r"F_1 = \frac{2 \cdot 1.00 \cdot 0.02}{1.00 + 0.02} = 0.039"),
         ("In general, for positive $a, b$ the harmonic mean satisfies "
          "$H(a,b) \\le \\sqrt{ab} \\le \\frac{a+b}{2}$, with equality only when "
          "$a = b$. So $F_1$ is high only when <b>both</b> are high.", None),
         ("If you care about one more than the other, use the weighted "
          "generalisation $F_\\beta$, where $\\beta > 1$ weights recall and "
          "$\\beta < 1$ weights precision:",
          r"F_\beta = (1 + \beta^2)\,\frac{\mathrm{precision}\cdot\mathrm{recall}}"
          r"{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}")],
        title="Why the harmonic mean",
    )

    anim_header("Accuracy vs F₁ as class imbalance grows")
    md(
        "The same mediocre classifier, evaluated on populations from balanced "
        "(50 % positive) down to 0.5 % positive. Accuracy climbs toward 1.0 for "
        "free while $F_1$ correctly collapses. Note where the \"always negative\" "
        "baseline goes."
    )

    prevs = np.linspace(.5, .005, 40)
    TPR, FPRt = .70, .10
    acc, f1s, base = [], [], []
    for p in prevs:
        tp, fn = p * TPR, p * (1 - TPR)
        fp, tn = (1 - p) * FPRt, (1 - p) * (1 - FPRt)
        acc.append(tp + tn)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        base.append(1 - p)

    frames = []
    for k in range(1, 41):
        frames.append(go.Frame(name=f"{prevs[k-1]*100:.1f}", data=[
            go.Scatter(x=prevs[:k] * 100, y=acc[:k], mode="lines+markers",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=prevs[:k] * 100, y=f1s[:k], mode="lines+markers",
                       line=dict(color=C["success"], width=3)),
            go.Scatter(x=prevs[:k] * 100, y=base[:k], mode="lines",
                       line=dict(color=C["muted"], width=2, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"prevalence = {prevs[k-1]*100:.1f} %   ·   accuracy = {acc[k-1]:.3f}"
            f"   ·   F1 = {f1s[k-1]:.3f}   ·   'always negative' = {base[k-1]:.3f}",
            color=C["danger"])])))

    f = go.Figure(data=[
        go.Scatter(x=prevs[:1] * 100, y=acc[:1], mode="lines+markers",
                   name="accuracy", line=dict(color=C["danger"], width=3)),
        go.Scatter(x=prevs[:1] * 100, y=f1s[:1], mode="lines+markers",
                   name="F1", line=dict(color=C["success"], width=3)),
        go.Scatter(x=prevs[:1] * 100, y=base[:1], mode="lines",
                   name="accuracy of 'always negative'",
                   line=dict(color=C["muted"], width=2, dash="dot")),
    ])
    f.update_layout(height=430, xaxis=dict(autorange="reversed",
                                           title="positive-class prevalence (%)"),
                    yaxis=dict(range=[0, 1.05], title="score"),
                    title="The same classifier, evaluated on rarer and rarer positives",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(160), slider_prefix="prevalence ")
    figure(f)

    sub("Measuring accuracy using cross-validation")

    codenote(
        "cross_val_predict, not cross_val_score",
        "To build a confusion matrix you need a <i>prediction for every instance</i>, "
        "each made by a model that never saw that instance. "
        "<code>cross_val_predict(model, X, y, cv=3)</code> returns exactly that: "
        "clean out-of-fold predictions for the whole training set. "
        "<code>cross_val_score</code> only gives you the fold scores.",
    )

    code_lab(
        "The dumb classifier, the confusion matrix, and every metric",
        '''import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, fbeta_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             classification_report)

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.2, stratify=d.target, random_state=42)
y_train_5, y_test_5 = (y_train == 5), (y_test == 5)

# ---------- the classifier that never predicts 5 --------------------------
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None): return self
    def predict(self, X):     return np.zeros(len(X), dtype=bool)

never5 = Never5Classifier()
print("=== 'always negative' baseline ===")
print(f"cross-val accuracy = "
      f"{cross_val_score(never5, X_train, y_train_5, cv=3, scoring='accuracy').mean():.4f}")
print("...and it has never detected a single 5.\\n")

# ---------- a real classifier ---------------------------------------------
sgd = make_pipeline(StandardScaler(),
                    SGDClassifier(random_state=42, max_iter=2000))
print("=== SGDClassifier ===")
print(f"cross-val accuracy = "
      f"{cross_val_score(sgd, X_train, y_train_5, cv=3, scoring='accuracy').mean():.4f}")

# out-of-fold predictions -> an honest confusion matrix
y_pred = cross_val_predict(sgd, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\\nconfusion matrix")
print(f"                 pred neg   pred pos")
print(f"  actual neg   {tn:>9,} {fp:>10,}")
print(f"  actual pos   {fn:>9,} {tp:>10,}")

prec = tp / (tp + fp); rec = tp / (tp + fn)
spec = tn / (tn + fp); fpr = fp / (fp + tn)
print(f"\\nprecision   = TP/(TP+FP) = {tp}/{tp+fp} = {prec:.4f}")
print(f"recall      = TP/(TP+FN) = {tp}/{tp+fn} = {rec:.4f}")
print(f"specificity = TN/(TN+FP) = {tn}/{tn+fp} = {spec:.4f}")
print(f"FPR         = FP/(FP+TN) = {fp}/{fp+tn} = {fpr:.4f}")
print(f"F1          = {2*prec*rec/(prec+rec):.4f}")
print(f"F2 (recall-weighted)  = {fbeta_score(y_train_5, y_pred, beta=2):.4f}")
print(f"F0.5 (prec-weighted)  = {fbeta_score(y_train_5, y_pred, beta=.5):.4f}")

print(f"\\nmetrics robust to imbalance:")
print(f"  balanced accuracy = {balanced_accuracy_score(y_train_5, y_pred):.4f}"
      f"   (mean of recall and specificity)")
print(f"  Matthews corrcoef = {matthews_corrcoef(y_train_5, y_pred):.4f}"
      f"   (+1 perfect, 0 random, -1 inverted)")

print("\\n" + classification_report(y_train_5, y_pred,
                                    target_names=["not 5", "is 5"], digits=4))
''',
        key="ch03_metrics",
    )

    quiz(
        "A model has precision 0.95 and recall 0.05. What is its $F_1$?",
        ["0.50", "0.095", "0.22", "0.95"],
        1,
        "$F_1 = 2(0.95)(0.05)/(0.95+0.05) = 0.095$. The harmonic mean is dragged "
        "down to the smaller value — exactly what you want from a summary that "
        "must not be gamed.",
        key="ch03q1",
    )

    keypoints([
        "Always compute the <b>majority-class baseline</b> before celebrating an "
        "accuracy.",
        "Confusion matrix first; every metric is a ratio of TP, TN, FP, FN.",
        "Precision reads a <b>column</b> (FP in the denominator), recall reads a "
        "<b>row</b> (FN in the denominator).",
        "$F_1$ is the harmonic mean, so it punishes imbalance between the two; "
        "$F_\\beta$ lets you say which you care about.",
        "Use <code>cross_val_predict</code> to get out-of-fold predictions for an "
        "honest confusion matrix.",
    ])


# ==========================================================================
def s_3_4():
    section("3.4", "The Precision / Recall Trade-off")

    lead(
        "Precision and recall are not two independent properties of a model. They "
        "are two readings of a <b>single dial</b> — the decision threshold — and "
        "moving it up trades one for the other, exactly."
    )

    sub("The threshold")

    md(
        "The classifier computes a score; the threshold turns the score into a "
        "decision:"
    )

    math(r"""
    \hat y(\mathbf{x}) \;=\; \mathbb{1}\bigl[\, s(\mathbf{x}) \;>\; \tau \,\bigr]
    """)
    where({r"s(\mathbf{x})": "the decision function — "
                             "<code>decision_function</code> or "
                             "<code>predict_proba</code>",
           r"\tau": "the threshold. scikit-learn's <code>predict</code> hard-codes "
                    "$\\tau = 0$ for <code>decision_function</code> and "
                    "$\\tau = 0.5$ for probabilities"})

    proof(
        "Raising τ can only increase precision and only decrease recall",
        "Raising the threshold removes instances from the predicted-positive set. "
        "Each removed instance was either a TP or an FP. Recall's numerator "
        "$TP$ can only shrink while its denominator $TP + FN$ is fixed (it depends "
        "only on the labels), so <b>recall is monotonically non-increasing</b> in "
        "$\\tau$. Precision has both $TP$ and $FP$ shrinking, so it is <i>not</i> "
        "monotone — it generally rises but can dip locally, which is why the "
        "precision-vs-threshold curve is bumpy while the recall curve is a clean "
        "staircase down.",
    )

    anim_header("Sliding the threshold: watch precision and recall cross")
    md(
        "Left: the score distributions of the two classes, with the threshold "
        "moving. Right: precision and recall traced out as the threshold sweeps. "
        "The bottom bar shows the confusion matrix live."
    )

    rng = np.random.default_rng(4)
    n_neg, n_pos = 900, 100
    s_neg = rng.normal(-1.1, 1.0, n_neg)
    s_pos = rng.normal(1.3, 1.05, n_pos)
    scores = np.r_[s_neg, s_pos]
    labels = np.r_[np.zeros(n_neg), np.ones(n_pos)]
    taus = np.linspace(scores.min(), scores.max(), 60)

    precs, recs, f1v = [], [], []
    for t in taus:
        pred = scores > t
        tp = np.sum(pred & (labels == 1)); fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precs.append(p); recs.append(r)
        f1v.append(2 * p * r / (p + r) if p + r else 0)

    bins = np.linspace(scores.min(), scores.max(), 46)
    hn = np.histogram(s_neg, bins=bins)[0]
    hp = np.histogram(s_pos, bins=bins)[0]
    ctr = (bins[:-1] + bins[1:]) / 2

    frames = []
    for k, t in enumerate(taus):
        pred = scores > t
        tp = int(np.sum(pred & (labels == 1))); fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1))); tn = int(np.sum(~pred & (labels == 0)))
        frames.append(go.Frame(name=f"{t:.2f}", data=[
            go.Bar(x=ctr, y=hn, marker=dict(color=alpha(C["train"], .72))),
            go.Bar(x=ctr, y=hp, marker=dict(color=alpha(C["warning"], .82))),
            go.Scatter(x=[t, t], y=[0, max(hn.max(), hp.max()) * 1.05],
                       mode="lines", line=dict(color=C["danger"], width=4)),
            go.Scatter(x=taus[:k + 1], y=precs[:k + 1], mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=taus[:k + 1], y=recs[:k + 1], mode="lines",
                       line=dict(color=C["accent"], width=3)),
            go.Scatter(x=taus[:k + 1], y=f1v[:k + 1], mode="lines",
                       line=dict(color=C["success"], width=2, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"τ = {t:+.2f}   TP={tp}  FP={fp}  FN={fn}  TN={tn}   ·   "
            f"precision = {precs[k]:.3f}   recall = {recs[k]:.3f}   "
            f"F1 = {f1v[k]:.3f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.5, .5],
                      subplot_titles=("score distributions & the threshold",
                                      "precision / recall vs threshold"))
    f.add_trace(go.Bar(x=ctr, y=hn, name="negatives",
                       marker=dict(color=alpha(C["train"], .72))), 1, 1)
    f.add_trace(go.Bar(x=ctr, y=hp, name="positives",
                       marker=dict(color=alpha(C["warning"], .82))), 1, 1)
    f.add_trace(go.Scatter(x=[taus[0]] * 2, y=[0, max(hn.max(), hp.max()) * 1.05],
                           mode="lines", name="threshold τ",
                           line=dict(color=C["danger"], width=4)), 1, 1)
    f.add_trace(go.Scatter(x=taus[:1], y=precs[:1], mode="lines", name="precision",
                           line=dict(color=C["primary"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=taus[:1], y=recs[:1], mode="lines", name="recall",
                           line=dict(color=C["accent"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=taus[:1], y=f1v[:1], mode="lines", name="F1",
                           line=dict(color=C["success"], width=2, dash="dot")), 1, 2)
    f.update_layout(height=460, barmode="overlay", bargap=.04,
                    title="One dial, two metrics")
    f.update_xaxes(title_text="score", row=1, col=1)
    f.update_xaxes(title_text="threshold τ", range=[taus[0], taus[-1]], row=1, col=2)
    f.update_yaxes(range=[0, 1.05], row=1, col=2)
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="τ = ")
    figure(f)

    sub("The precision–recall curve")

    md(
        "Plotting precision against recall as $\\tau$ sweeps gives the **PR "
        "curve**. It is the right curve for **skewed** problems, because neither "
        "axis involves $TN$ — and $TN$ is the huge, uninformative number that "
        "makes ROC look flattering on rare-positive problems."
    )

    pr = go.Figure()
    pr.add_trace(go.Scatter(x=recs, y=precs, mode="lines",
                            line=dict(color=C["primary"], width=3.4),
                            fill="tozeroy", fillcolor=alpha(C["primary"], .12),
                            name="PR curve"))
    pr.add_hline(y=n_pos / (n_pos + n_neg), line_dash="dot",
                 line_color=C["muted"],
                 annotation_text=f"no-skill baseline = prevalence "
                                 f"= {n_pos/(n_pos+n_neg):.2f}")
    best = int(np.argmax(f1v))
    pr.add_trace(go.Scatter(x=[recs[best]], y=[precs[best]], mode="markers+text",
                            marker=dict(color=C["danger"], size=14,
                                        line=dict(color="#fff", width=2)),
                            text=[f"  max F1 = {f1v[best]:.3f}"],
                            textposition="middle right", name="best F1"))
    pr.update_layout(height=430, xaxis_title="recall", yaxis_title="precision",
                     xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.05]),
                     title="Precision–recall curve")
    figure(pr, "The no-skill baseline of a PR curve is the class prevalence, not "
               "0.5. On a 1 %-positive problem, a PR-AUC of 0.30 may be excellent.")

    sub("Choosing the threshold from the requirement, not the other way round")

    md(
        """
The workflow is always the same, and it is the single most useful practical
skill in this chapter:

1. State the business requirement — *"we must catch 90 % of fraud"* (a recall
   floor) or *"no more than 1 in 20 flags may be wrong"* (a precision floor).
2. Compute scores for the validation set with `cross_val_predict(...,
   method='decision_function')`.
3. Find the smallest threshold that meets the constraint.
4. Report the *other* metric at that threshold — that is the price you pay.
        """
    )

    warn(
        "A high-precision classifier at 99 % precision may be useless",
        "If reaching 99 % precision drives recall to 4 %, you have built a system "
        "that is almost always right about the almost nothing it says. Whether "
        "that is good depends entirely on the cost asymmetry — which is why "
        "§3.1 of Chapter 2 insisted you find out the cost of an error <i>first</i>.",
    )

    code_lab(
        "Pick a threshold from a requirement",
        '''import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_recall_curve, precision_score,
                             recall_score, f1_score, average_precision_score)

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.2, stratify=d.target, random_state=42)
y_train_5, y_test_5 = (y_train == 5), (y_test == 5)

sgd = make_pipeline(StandardScaler(), SGDClassifier(random_state=42, max_iter=2000))
scores = cross_val_predict(sgd, X_train, y_train_5, cv=5,
                           method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, scores)
print(f"curve has {len(thresholds)} candidate thresholds")
print(f"average precision (PR-AUC) = {average_precision_score(y_train_5, scores):.4f}")
print(f"no-skill baseline          = {y_train_5.mean():.4f}\\n")

# ---------- requirement A: precision must be at least 90 % ----------------
i = np.argmax(precisions >= .90)
tau_a = thresholds[i]
print(f"A) 'precision >= 90 %'")
print(f"   threshold  = {tau_a:+.4f}")
print(f"   precision  = {precisions[i]:.4f}")
print(f"   recall     = {recalls[i]:.4f}   <- the price you pay")

# ---------- requirement B: recall must be at least 95 % -------------------
j = len(recalls) - 1 - np.argmax(recalls[::-1] >= .95)
tau_b = thresholds[min(j, len(thresholds) - 1)]
print(f"\\nB) 'recall >= 95 %'")
print(f"   threshold  = {tau_b:+.4f}")
print(f"   precision  = {precisions[j]:.4f}   <- the price you pay")
print(f"   recall     = {recalls[j]:.4f}")

# ---------- requirement C: just maximise F1 -------------------------------
f1s = 2 * precisions * recalls / np.where(precisions + recalls == 0, 1,
                                          precisions + recalls)
k = int(np.argmax(f1s[:-1]))
print(f"\\nC) 'maximise F1'")
print(f"   threshold  = {thresholds[k]:+.4f}   precision = {precisions[k]:.4f}"
      f"   recall = {recalls[k]:.4f}   F1 = {f1s[k]:.4f}")

# ---------- apply the chosen threshold on the TEST set --------------------
sgd.fit(X_train, y_train_5)
test_scores = sgd.decision_function(X_test)
print("\\n--- held-out test set, three operating points ---")
print(f"{'rule':<22}{'threshold':>11}{'precision':>11}{'recall':>9}{'F1':>8}")
for name, t in [("default (tau = 0)", 0.0), ("precision >= 90 %", tau_a),
                ("recall >= 95 %", tau_b), ("max F1", thresholds[k])]:
    pred = test_scores > t
    print(f"{name:<22}{t:>11.3f}"
          f"{precision_score(y_test_5, pred, zero_division=0):>11.4f}"
          f"{recall_score(y_test_5, pred):>9.4f}"
          f"{f1_score(y_test_5, pred):>8.4f}")
print("\\nSame model, four completely different products.")
''',
        key="ch03_pr",
    )

    quiz(
        "You raise the decision threshold. Which statement is guaranteed?",
        ["Precision increases", "Recall does not increase",
         "$F_1$ increases", "Accuracy increases"],
        1,
        "Recall is monotonically non-increasing in $\\tau$ because its denominator "
        "($TP+FN$) is fixed by the labels while its numerator can only shrink. "
        "Precision usually rises but is not monotone — see the proof above.",
        key="ch03q2",
    )

    keypoints([
        "Precision and recall are two readings of <b>one dial</b>, the threshold "
        "$\\tau$.",
        "Recall is monotone in $\\tau$; precision is not.",
        "Choose $\\tau$ from a stated business requirement, then <b>report the "
        "cost</b> in the other metric.",
        "PR curve for skewed problems: neither axis touches $TN$; its no-skill "
        "baseline is the prevalence.",
        "<code>predict</code> hard-codes $\\tau$; use "
        "<code>decision_function</code> or <code>predict_proba</code> and threshold "
        "yourself.",
    ])


# ==========================================================================
def s_3_5():
    section("3.5", "The ROC Curve")

    lead(
        "The receiver operating characteristic plots the true positive rate "
        "against the false positive rate as the threshold sweeps. It is the "
        "standard when classes are roughly balanced — and misleading when they "
        "are not."
    )

    math(r"""
    \mathrm{TPR}(\tau) = \mathrm{recall}(\tau) = \frac{TP}{TP + FN}
    \qquad\text{versus}\qquad
    \mathrm{FPR}(\tau) = \frac{FP}{FP + TN} = 1 - \mathrm{specificity}(\tau)
    """)

    sub("Reading the curve")

    table(
        ["Region", "Meaning"],
        [["The diagonal $\\mathrm{TPR} = \\mathrm{FPR}$",
          "A random classifier. AUC $= 0.5$."],
         ["The top-left corner $(0, 1)$",
          "Perfect: every positive caught, no false alarms. AUC $= 1$."],
         ["Below the diagonal",
          "Worse than random — which means it is <i>informative</i>: invert the "
          "predictions and you get AUC $= 1 - \\mathrm{AUC}$."],
         ["The steep initial rise",
          "The high-confidence region. This is what matters when you can only act "
          "on the top few predictions."]],
    )

    sub("What AUC actually measures")

    math(r"""
    \mathrm{AUC} \;=\; \int_{0}^{1} \mathrm{TPR}\bigl(\mathrm{FPR}^{-1}(u)\bigr)\, du
    \;=\; \Pr\Bigl[\, s\bigl(\mathbf{x}^{+}\bigr) > s\bigl(\mathbf{x}^{-}\bigr) \,\Bigr]
    """)

    proof(
        "AUC is a ranking probability, not an accuracy",
        "The area under the ROC curve equals the probability that a randomly "
        "chosen <b>positive</b> instance receives a higher score than a randomly "
        "chosen <b>negative</b> one. It is therefore threshold-free and "
        "prevalence-free: it measures only how well the model <i>ranks</i>. This "
        "is also its weakness — a model can rank beautifully and still be badly "
        "calibrated, and AUC will not tell you.",
    )

    derive(
        [("The equivalence with the Mann–Whitney U statistic makes it computable "
          "without any integration. Let $\\{s_i^+\\}_{i=1}^{n_+}$ be the scores of "
          "the positives and $\\{s_j^-\\}_{j=1}^{n_-}$ those of the negatives.",
          None),
         ("Count every pair in which the positive outranks the negative, giving "
          "ties half credit:",
          r"U = \sum_{i=1}^{n_+}\sum_{j=1}^{n_-}\Bigl(\mathbb{1}\bigl[s_i^+ > s_j^-\bigr] "
          r"+ \tfrac12\mathbb{1}\bigl[s_i^+ = s_j^-\bigr]\Bigr)"),
         ("Normalise by the number of pairs:",
          r"\mathrm{AUC} = \frac{U}{n_+ \, n_-}"),
         ("Equivalently, using ranks $R_i$ of the positives in the pooled sorted "
          "list — the form actually used in software because it is $O(m\\log m)$:",
          r"\mathrm{AUC} = \frac{\displaystyle\sum_{i=1}^{n_+} R_i "
          r"- \frac{n_+(n_+ + 1)}{2}}{n_+ \, n_-}"),
         ("The lab below verifies all three formulas agree to machine precision.",
          None)],
        title="AUC = Mann–Whitney U / (n₊ n₋)",
    )

    anim_header("ROC and PR built simultaneously as the threshold sweeps")
    md(
        "The same threshold sweep drawn on both curves at once. Watch how the ROC "
        "point races into the top-left corner while the PR point is still "
        "struggling — that divergence *is* the class-imbalance effect."
    )

    rng = np.random.default_rng(8)
    n_neg, n_pos = 2000, 60
    sc = np.r_[rng.normal(-0.9, 1.0, n_neg), rng.normal(1.5, 1.1, n_pos)]
    lb = np.r_[np.zeros(n_neg), np.ones(n_pos)]
    taus = np.linspace(sc.max(), sc.min(), 70)

    tprs, fprs, prs, rcs = [], [], [], []
    for t in taus:
        p = sc > t
        tp = np.sum(p & (lb == 1)); fp = np.sum(p & (lb == 0))
        fn = np.sum(~p & (lb == 1)); tn = np.sum(~p & (lb == 0))
        tprs.append(tp / (tp + fn) if tp + fn else 0)
        fprs.append(fp / (fp + tn) if fp + tn else 0)
        prs.append(tp / (tp + fp) if tp + fp else 1)
        rcs.append(tp / (tp + fn) if tp + fn else 0)

    frames = []
    for k in range(1, len(taus) + 1):
        auc_k = np.trapezoid(tprs[:k], fprs[:k]) if k > 1 else 0
        frames.append(go.Frame(name=f"{taus[k-1]:.2f}", data=[
            go.Scatter(x=fprs[:k], y=tprs[:k], mode="lines",
                       line=dict(color=C["primary"], width=3.4)),
            go.Scatter(x=[fprs[k - 1]], y=[tprs[k - 1]], mode="markers",
                       marker=dict(color=C["danger"], size=13,
                                   line=dict(color="#fff", width=2))),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                       line=dict(color=C["muted"], width=2, dash="dash")),
            go.Scatter(x=rcs[:k], y=prs[:k], mode="lines",
                       line=dict(color=C["accent"], width=3.4)),
            go.Scatter(x=[rcs[k - 1]], y=[prs[k - 1]], mode="markers",
                       marker=dict(color=C["danger"], size=13,
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"τ = {taus[k-1]:+.2f}   TPR = {tprs[k-1]:.3f}   FPR = {fprs[k-1]:.3f}"
            f"   |   precision = {prs[k-1]:.3f}   recall = {rcs[k-1]:.3f}"
            f"   |   ROC-AUC so far = {auc_k:.3f}")])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("ROC — TPR vs FPR",
                                      "PR — precision vs recall"))
    f.add_trace(go.Scatter(x=fprs[:1], y=tprs[:1], mode="lines", name="ROC",
                           line=dict(color=C["primary"], width=3.4)), 1, 1)
    f.add_trace(go.Scatter(x=fprs[:1], y=tprs[:1], mode="markers",
                           name="current τ", marker=dict(color=C["danger"], size=13,
                           line=dict(color="#fff", width=2))), 1, 1)
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                           line=dict(color=C["muted"], width=2, dash="dash")), 1, 1)
    f.add_trace(go.Scatter(x=rcs[:1], y=prs[:1], mode="lines", name="PR",
                           line=dict(color=C["accent"], width=3.4)), 1, 2)
    f.add_trace(go.Scatter(x=rcs[:1], y=prs[:1], mode="markers", showlegend=False,
                           marker=dict(color=C["danger"], size=13,
                           line=dict(color="#fff", width=2))), 1, 2)
    f.add_hline(y=n_pos / (n_pos + n_neg), line_dash="dot", line_color=C["muted"],
                row=1, col=2)
    f.update_xaxes(range=[-.02, 1.02], title_text="FPR", row=1, col=1)
    f.update_yaxes(range=[-.02, 1.02], title_text="TPR", row=1, col=1)
    f.update_xaxes(range=[-.02, 1.02], title_text="recall", row=1, col=2)
    f.update_yaxes(range=[-.02, 1.02], title_text="precision", row=1, col=2)
    f.update_layout(height=470,
                    title=f"Only {n_pos}/{n_pos+n_neg} = "
                          f"{n_pos/(n_pos+n_neg):.1%} positives")
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="τ = ")
    figure(f, "The ROC curve looks superb. The PR curve tells the truth.")

    sub("ROC or PR? The rule")

    table(
        ["Use", "When", "Because"],
        [["<b>PR curve</b>",
          "The positive class is <b>rare</b>, or you care mainly about the "
          "positives",
          "Neither axis contains $TN$, so the huge easy-negative population cannot "
          "flatter you"],
         ["<b>ROC curve</b>",
          "Classes are roughly balanced, or false positives and false negatives "
          "matter comparably",
          "Prevalence-independent, and AUC has the clean ranking interpretation"]],
    )

    pitfall(
        "The classic ROC illusion",
        "With 1 % positives, a model can have ROC-AUC = 0.95 and still produce a "
        "flag list that is 90 % wrong. An FPR of 0.05 on 99 000 negatives is "
        "4 950 false alarms against at most 1 000 possible true ones. The ROC "
        "curve shows FPR = 0.05 as a triumph; the PR curve shows precision ≈ 0.17. "
        "<b>Always plot both.</b>",
    )

    code_lab(
        "Three formulas for AUC, and the imbalance illusion",
        '''import numpy as np
from scipy.stats import rankdata, mannwhitneyu
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

rng = np.random.default_rng(0)

def make(n_neg, n_pos, sep=2.4):
    s = np.r_[rng.normal(0, 1, n_neg), rng.normal(sep, 1, n_pos)]
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)]
    return s, y

s, y = make(2000, 60)

# --- 1. sklearn -----------------------------------------------------------
a1 = roc_auc_score(y, s)

# --- 2. Mann-Whitney U, brute force over all pairs ------------------------
pos, neg = s[y == 1], s[y == 0]
U = float((pos[:, None] > neg[None, :]).sum()) + 0.5 * float((pos[:, None] == neg[None, :]).sum())
a2 = U / (len(pos) * len(neg))

# --- 3. the rank formula --------------------------------------------------
R = rankdata(s)
a3 = (R[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# --- 4. trapezoid under the ROC curve -------------------------------------
fpr, tpr, _ = roc_curve(y, s)
a4 = np.trapezoid(tpr, fpr)

print("four routes to the same number:")
print(f"  sklearn roc_auc_score  = {a1:.10f}")
print(f"  Mann-Whitney U / n+n-  = {a2:.10f}")
print(f"  rank formula           = {a3:.10f}")
print(f"  trapezoid under ROC    = {a4:.10f}")
print(f"  scipy mannwhitneyu     = "
      f"{mannwhitneyu(pos, neg).statistic/(len(pos)*len(neg)):.10f}")

# --- the interpretation, verified by simulation ---------------------------
draws = 200_000
wins = (rng.choice(pos, draws) > rng.choice(neg, draws)).mean()
print(f"\\nP[score(random positive) > score(random negative)] "
      f"by Monte Carlo = {wins:.4f}")
print(f"                                             AUC = {a1:.4f}   <- same thing")

# --- the imbalance illusion ----------------------------------------------
print(f"\\n{'prevalence':>11}{'ROC-AUC':>10}{'PR-AUC':>9}{'precision @ FPR=5%':>21}")
for n_pos_ in [1000, 300, 100, 30, 10]:
    s2, y2 = make(2000, n_pos_)
    fpr2, tpr2, thr2 = roc_curve(y2, s2)
    i = np.searchsorted(fpr2, .05)
    tp = tpr2[i] * n_pos_; fp = .05 * 2000
    prec = tp / (tp + fp)
    print(f"{n_pos_/(2000+n_pos_):>11.3%}{roc_auc_score(y2, s2):>10.4f}"
          f"{average_precision_score(y2, s2):>9.4f}{prec:>21.4f}")
print("\\nROC-AUC barely moves. PR-AUC and the actual usefulness collapse.")
''',
        key="ch03_roc",
    )

    keypoints([
        "ROC = TPR vs FPR; AUC = $\\Pr[\\,s(\\mathbf{x}^+) > s(\\mathbf{x}^-)\\,]$ "
        "— a pure <b>ranking</b> measure.",
        "AUC = Mann–Whitney $U / (n_+ n_-)$; it is threshold-free and "
        "prevalence-free.",
        "Prevalence-free is a bug, not a feature, when positives are rare — use "
        "the PR curve there.",
        "Below the diagonal is informative: invert and get $1 - \\mathrm{AUC}$.",
        "AUC says nothing about <b>calibration</b>. A perfectly ranked model can "
        "have wildly wrong probabilities.",
    ])


# ==========================================================================
def s_3_6():
    section("3.6", "Multiclass Classification")

    lead(
        "Ten digits, not two. Some algorithms handle $K$ classes natively; others "
        "are strictly binary and must be wrapped in one of two decomposition "
        "strategies."
    )

    table(
        ["Family", "Native multiclass?", "Examples"],
        [["Natively multiclass", "✅",
          "Logistic/softmax regression, naive Bayes, decision trees, random "
          "forests, $k$-NN, neural networks"],
         ["Strictly binary — needs a wrapper", "❌",
          "SVM classifiers, <code>SGDClassifier</code> with a binary loss"]],
    )

    sub("One-versus-the-rest (OvR)")

    md("Train $K$ binary classifiers, one per class, each answering \"class $k$ or "
       "not?\". Predict the class whose classifier gives the highest score:")

    math(r"""
    \hat y \;=\; \operatorname*{arg\,max}_{k \in \{1,\dots,K\}} \; s_k(\mathbf{x})
    """)

    md("**Cost:** $K$ classifiers, each trained on all $m$ instances.")

    sub("One-versus-one (OvO)")

    md("Train a classifier for every *pair* of classes:")

    math(r"""
    N_{\text{classifiers}} \;=\; \binom{K}{2} \;=\; \frac{K\,(K-1)}{2}
    """)

    md(
        "For $K = 10$ that is 45 classifiers — but each sees only the instances of "
        "its two classes, roughly $2m/K$ of them. Prediction runs all 45 and takes "
        "a majority vote."
    )

    derive(
        [("Which is cheaper depends entirely on how training cost scales with $m$. "
          "Suppose training costs $\\mathcal{O}(m^\\alpha)$.", None),
         ("<b>OvR total cost.</b> $K$ classifiers, each on the full $m$:",
          r"T_{\text{OvR}} = K \cdot c \cdot m^{\alpha}"),
         ("<b>OvO total cost.</b> $\\binom{K}{2}$ classifiers, each on $2m/K$ "
          "instances (assuming balanced classes):",
          r"T_{\text{OvO}} = \frac{K(K-1)}{2}\cdot c \cdot "
          r"\left(\frac{2m}{K}\right)^{\alpha} "
          r"= c\,m^{\alpha}\,\frac{(K-1)\,2^{\alpha-1}}{K^{\alpha-1}}"),
         ("Take the ratio:",
          r"\frac{T_{\text{OvO}}}{T_{\text{OvR}}} = "
          r"\frac{(K-1)\,2^{\alpha-1}}{K^{\alpha}}"),
         ("For a <b>linear</b> algorithm, $\\alpha = 1$: the ratio is "
          "$(K-1)/K \\approx 1$ — the two are comparable, so use OvR (fewer models "
          "to manage).", None),
         ("For a <b>kernel SVM</b>, $\\alpha \\approx 2$ to $3$. With $\\alpha = 2$ "
          "and $K = 10$ the ratio is $9 \\cdot 2 / 100 = 0.18$ — <b>OvO is five "
          "times cheaper</b>. That is exactly why scikit-learn defaults SVC to OvO.",
          None)],
        title="When is OvO cheaper than OvR? The exact condition",
    )

    anim_header("OvR and OvO decision regions being assembled")
    md(
        "Three classes in 2-D. First the OvR sub-problems appear one at a time and "
        "combine into a region map; then the OvO pairwise boundaries do the same. "
        "Notice the ambiguous wedge OvR leaves near the centre — that is the "
        "classic OvR artefact."
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(2)
    ctrs = np.array([[-1.7, -1.0], [1.7, -1.1], [0.0, 1.9]])
    Xm = np.vstack([rng.normal(c, .78, (90, 2)) for c in ctrs])
    ym = np.repeat([0, 1, 2], 90)
    gx, gy = np.meshgrid(np.linspace(-4.2, 4.2, 180), np.linspace(-3.6, 4.4, 180))
    G = np.c_[gx.ravel(), gy.ravel()]

    ovr_scores = []
    for k in range(3):
        clf = LogisticRegression().fit(Xm, (ym == k).astype(int))
        ovr_scores.append(clf.decision_function(G))
    ovr_scores = np.array(ovr_scores)

    pairs = [(0, 1), (0, 2), (1, 2)]
    votes = np.zeros((3, len(G)))
    pair_boundaries = []
    for a, b in pairs:
        mask = np.isin(ym, [a, b])
        clf = LogisticRegression().fit(Xm[mask], (ym[mask] == b).astype(int))
        dfn = clf.decision_function(G)
        pair_boundaries.append(dfn)
        votes[b] += (dfn > 0); votes[a] += (dfn <= 0)

    cs3 = [[0, C["train"]], [.5, C["warning"]], [1, C["success"]]]

    def pts():
        return [go.Scatter(x=Xm[ym == k, 0], y=Xm[ym == k, 1], mode="markers",
                           marker=dict(color=[C["train"], C["warning"],
                                              C["success"]][k], size=7,
                                       line=dict(color="#fff", width=.9)),
                           showlegend=False) for k in range(3)]

    frames = []
    for k in range(3):
        z = (ovr_scores[k] > 0).astype(float).reshape(gx.shape)
        frames.append(go.Frame(name=f"OvR {k+1}", data=[
            go.Contour(x=gx[0], y=gy[:, 0], z=z, showscale=False,
                       colorscale=[[0, "#EEF0F7"],
                                   [1, alpha([C["train"], C["warning"],
                                              C["success"]][k], .45)]],
                       contours=dict(showlines=False), opacity=.9)] + pts(),
            layout=go.Layout(title=f"OvR step {k+1}/4: classifier "
                                   f"'class {k} vs rest'")))
    zc = ovr_scores.argmax(0).reshape(gx.shape).astype(float)
    frames.append(go.Frame(name="OvR all", data=[
        go.Contour(x=gx[0], y=gy[:, 0], z=zc, showscale=False, colorscale=cs3,
                   contours=dict(showlines=False), opacity=.42)] + pts(),
        layout=go.Layout(title="OvR step 4/4: argmax of the 3 scores")))
    for i, (a, b) in enumerate(pairs):
        z = (pair_boundaries[i] > 0).astype(float).reshape(gx.shape)
        frames.append(go.Frame(name=f"OvO {i+1}", data=[
            go.Contour(x=gx[0], y=gy[:, 0], z=z, showscale=False,
                       colorscale=[[0, "#EEF0F7"], [1, alpha(C["proof"], .40)]],
                       contours=dict(showlines=False), opacity=.9)] + pts(),
            layout=go.Layout(title=f"OvO step {i+1}/4: classifier "
                                   f"'class {a} vs class {b}'")))
    zv = votes.argmax(0).reshape(gx.shape).astype(float)
    frames.append(go.Frame(name="OvO all", data=[
        go.Contour(x=gx[0], y=gy[:, 0], z=zv, showscale=False, colorscale=cs3,
                   contours=dict(showlines=False), opacity=.42)] + pts(),
        layout=go.Layout(title="OvO step 4/4: majority vote of the 3 duels")))

    f = go.Figure(data=[go.Contour(x=gx[0], y=gy[:, 0],
                                   z=(ovr_scores[0] > 0).astype(float).reshape(gx.shape),
                                   showscale=False,
                                   colorscale=[[0, "#EEF0F7"],
                                               [1, alpha(C["train"], .45)]],
                                   contours=dict(showlines=False), opacity=.9)] + pts())
    f.update_layout(height=520, xaxis=dict(range=[-4.2, 4.2], title="x1"),
                    yaxis=dict(range=[-3.6, 4.4], title="x2"),
                    title="OvR step 1/4")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="")
    figure(f)

    codenote(
        "Forcing a strategy",
        "scikit-learn picks for you — OvO for <code>SVC</code>, OvR for most "
        "others — but you can override with "
        "<code>OneVsRestClassifier(SVC())</code> or "
        "<code>OneVsOneClassifier(SGDClassifier())</code>. Scaling the inputs "
        "usually improves multiclass accuracy substantially, because the "
        "per-class scores become comparable.",
    )

    code_lab(
        "OvR vs OvO: accuracy, model count, and wall-clock time",
        '''import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.2, stratify=d.target, random_state=42)
K = len(np.unique(y_train))
print(f"K = {K} classes, m = {len(X_train)} training instances")
print(f"OvR would train {K} classifiers on {len(X_train)} instances each")
print(f"OvO would train {K*(K-1)//2} classifiers on ~{2*len(X_train)//K} each\\n")

def bench(name, model):
    t0 = time.perf_counter(); model.fit(X_train, y_train)
    fit = time.perf_counter() - t0
    acc = model.score(X_test, y_test)
    inner = getattr(model, "estimators_", None)
    n = len(inner) if inner is not None else "native"
    print(f"{name:<34}{str(n):>10}{fit:>10.3f}s{acc:>10.4f}")

print(f"{'strategy':<34}{'#models':>10}{'fit time':>11}{'accuracy':>10}")
bench("SVC (native OvO)", make_pipeline(StandardScaler(), SVC(random_state=42)))
bench("OvR(SVC)", make_pipeline(StandardScaler(),
                                OneVsRestClassifier(SVC(random_state=42))))
bench("OvO(SVC)", make_pipeline(StandardScaler(),
                                OneVsOneClassifier(SVC(random_state=42))))
bench("SGDClassifier (native OvR)",
      make_pipeline(StandardScaler(), SGDClassifier(random_state=42, max_iter=2000)))
bench("OvO(SGDClassifier)",
      make_pipeline(StandardScaler(),
                    OneVsOneClassifier(SGDClassifier(random_state=42, max_iter=2000))))
bench("LogisticRegression (softmax)",
      make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))

# ---- scaling matters a lot for multiclass -------------------------------
raw = SGDClassifier(random_state=42, max_iter=2000).fit(X_train, y_train)
sca = make_pipeline(StandardScaler(),
                    SGDClassifier(random_state=42, max_iter=2000)).fit(X_train, y_train)
print(f"\\nSGD without scaling : {raw.score(X_test, y_test):.4f}")
print(f"SGD with scaling    : {sca.score(X_test, y_test):.4f}")

# ---- the 10 per-class scores for one image ------------------------------
i = 7
scores = sca.decision_function([X_test[i]])[0]
print(f"\\nimage {i} is really a '{y_test[i]}'. The 10 OvR scores:")
for k, s in enumerate(scores):
    bar = "#" * int(max(0, (s - scores.min()) / (np.ptp(scores) + 1e-9) * 34))
    star = " <-- argmax" if k == scores.argmax() else ""
    print(f"  class {k}: {s:+8.3f} {bar}{star}")
''',
        key="ch03_multi",
    )

    keypoints([
        "OvR: $K$ classifiers on all $m$ rows; predict $\\arg\\max_k s_k$.",
        "OvO: $K(K-1)/2$ classifiers on $2m/K$ rows each; predict by majority "
        "vote.",
        "OvO wins whenever training scales super-linearly ($\\alpha > 1$) — which "
        "is why <code>SVC</code> defaults to it.",
        "Many algorithms are natively multiclass and need no wrapper at all.",
        "<b>Scale your inputs</b>; multiclass scores must be comparable across "
        "classes.",
    ])


# ==========================================================================
def s_3_7():
    section("3.7", "Error Analysis")

    lead(
        "The highest-leverage activity in a real classification project, and the "
        "one most often skipped. You are not looking for a better algorithm — you "
        "are looking at <i>which</i> mistakes the current one makes, and why."
    )

    sub("The normalised confusion matrix")

    md(
        "A raw confusion matrix is dominated by the frequent classes. Normalise by "
        "row (i.e. by the true-class total) so each cell reads *\"of the true $i$s, "
        "what fraction were called $j$\"*:"
    )

    math(r"""
    \tilde{M}_{ij} \;=\; \frac{M_{ij}}{\displaystyle\sum_{j'} M_{ij'}}
    \;=\; \Pr\bigl[\, \hat y = j \;\bigm|\; y = i \,\bigr]
    """)

    tip(
        "Then blank the diagonal",
        "The diagonal is the correct predictions and it is always the biggest "
        "number, so it dominates the colour scale and hides everything "
        "interesting. Fill it with zeros before plotting. Suddenly the "
        "<i>errors</i> get the full dynamic range and the systematic confusions "
        "leap out.",
    )

    md(
        "Normalising by **column** instead answers a different question — "
        "*\"of the things I called a $j$, what fraction really were $i$?\"* — and "
        "is the right view when you want to know which class is polluting your "
        "predictions."
    )

    anim_header("Building the error picture in four steps")
    md("Raw counts → row-normalised → diagonal removed → column-normalised. Only "
       "the third frame tells you what to do next.")

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix

    dg = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(dg.data, dg.target, test_size=.25,
                                          stratify=dg.target, random_state=42)
    mdl = make_pipeline(StandardScaler(),
                        SGDClassifier(random_state=42, max_iter=1500))
    ypred = cross_val_predict(mdl, Xtr, ytr, cv=3)
    M = confusion_matrix(ytr, ypred)
    Mrow = M / M.sum(axis=1, keepdims=True)
    Mnod = Mrow.copy(); np.fill_diagonal(Mnod, 0)
    Mcol = M / np.maximum(M.sum(axis=0, keepdims=True), 1)
    Mcolnod = Mcol.copy(); np.fill_diagonal(Mcolnod, 0)

    views = [("1. raw counts — the diagonal drowns everything", M, "%{z:.0f}"),
             ("2. row-normalised — P(predicted = j | true = i)", Mrow, "%{z:.2f}"),
             ("3. diagonal zeroed — now you can SEE the errors", Mnod, "%{z:.2f}"),
             ("4. column-normalised, diagonal zeroed — who pollutes class j",
              Mcolnod, "%{z:.2f}")]
    frames = [go.Frame(name=str(i + 1), data=[go.Heatmap(
        z=Z, x=list(range(10)), y=list(range(10)), colorscale=nav.cscale(),
        text=Z, texttemplate=t, textfont=dict(size=9), xgap=1.5, ygap=1.5,
        showscale=True)], layout=go.Layout(title=name))
        for i, (name, Z, t) in enumerate(views)]

    f = go.Figure(go.Heatmap(z=M, x=list(range(10)), y=list(range(10)),
                             colorscale=nav.cscale(), text=M,
                             texttemplate="%{z:.0f}", textfont=dict(size=9),
                             xgap=1.5, ygap=1.5))
    f.update_layout(height=560, title=views[0][0],
                    xaxis_title="predicted class", yaxis_title="true class",
                    yaxis=dict(autorange="reversed"))
    anim.animate(f, frames, duration=nav.anim_ms(1900), slider_prefix="view ")
    figure(f)

    worst = np.dstack(np.unravel_index(np.argsort(Mnod.ravel())[::-1][:6],
                                       Mnod.shape))[0]
    st.markdown("**The six worst systematic confusions in this run:**")
    table(["True", "Predicted as", "Rate", "Plausible cause"],
          [[f"<b>{a}</b>", f"<b>{b}</b>", f"{Mnod[a, b]:.1%}",
            "shared strokes / similar pixel mass"] for a, b in worst])

    sub("What to do about each pattern")

    table(
        ["Pattern in the matrix", "Diagnosis", "Fix"],
        [["One row is uniformly bad", "That class is intrinsically hard, or "
          "under-represented", "More data for that class; class weights; "
          "oversampling"],
         ["One column is dark", "The model over-predicts that class", "It is "
          "probably the majority class — rebalance, or adjust per-class "
          "thresholds"],
         ["A symmetric hot pair $(i,j)$ and $(j,i)$",
          "The two classes are genuinely similar in feature space",
          "New features that separate exactly those two; or a dedicated "
          "second-stage binary classifier for that pair"],
         ["An asymmetric hot cell $(i,j)$ only",
          "Class $i$ is being absorbed by a more 'attractive' class $j$",
          "Check for prior/prevalence effects; the model may just be betting on "
          "the frequent class"],
         ["Errors concentrated in a data slice",
          "A sub-population the features do not describe",
          "Add the feature that identifies the slice, or model the slice "
          "separately"]],
    )

    idea(
        "Look at the actual instances, not just the matrix",
        "Pull out the individual images (or rows, or documents) in the hot cell "
        "and look at them. Nine times out of ten you will discover one of three "
        "things: the labels are wrong, an obvious feature is missing, or the two "
        "classes really are the same thing under a different name. All three are "
        "fixable — and none of them is fixed by trying another algorithm.",
    )

    code_lab(
        "Full error analysis: matrices, per-class report, and the failing images",
        '''import numpy as np, pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.25, stratify=d.target, random_state=42)

model = make_pipeline(StandardScaler(), SGDClassifier(random_state=42, max_iter=1500))
y_pred = cross_val_predict(model, X_train, y_train, cv=3)

cm = confusion_matrix(y_train, y_pred)
row = cm / cm.sum(axis=1, keepdims=True)
err = row.copy(); np.fill_diagonal(err, 0)

print("=== per-class performance ===")
print(classification_report(y_train, y_pred, digits=3))

print("=== the 8 worst systematic confusions ===")
flat = np.dstack(np.unravel_index(np.argsort(err.ravel())[::-1][:8], err.shape))[0]
print(f"{'true':>5}{'pred':>6}{'rate':>9}{'count':>8}   symmetric?")
for a, b in flat:
    sym = "YES  <- classes look alike" if err[b, a] > .5 * err[a, b] else "no"
    print(f"{a:>5}{b:>6}{err[a,b]:>9.1%}{cm[a,b]:>8}   {sym}")

# ---- which classes are hardest overall ----------------------------------
per_class = pd.DataFrame({
    "support":  cm.sum(1),
    "recall":   np.diag(cm) / cm.sum(1),
    "precision":np.diag(cm) / np.maximum(cm.sum(0), 1),
    "err_out":  err.sum(1),          # how often this class is misread
    "err_in":   err.sum(0),          # how often others are read AS this class
}).round(4)
print("\\n=== per-class error budget ===")
print(per_class.sort_values("recall").to_string())

# ---- LOOK AT THE ACTUAL FAILING IMAGES ----------------------------------
a, b = flat[0]
mask = (y_train == a) & (y_pred == b)
print(f"\\n{mask.sum()} images that are really {a} but were called {b}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
bad = np.where(mask)[0][:8]
good = np.where((y_train == a) & (y_pred == a))[0][:8]
fig = make_subplots(rows=2, cols=8,
                    row_titles=[f"correct {a}", f"{a} called {b}"])
for j, i in enumerate(good):
    fig.add_trace(go.Heatmap(z=X_train[i].reshape(8, 8)[::-1],
                             colorscale=PARULA, showscale=False), 1, j + 1)
for j, i in enumerate(bad):
    fig.add_trace(go.Heatmap(z=X_train[i].reshape(8, 8)[::-1],
                             colorscale=PARULA, showscale=False), 2, j + 1)
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(height=340,
                  title=f"Top row: {a}s the model gets right. "
                        f"Bottom row: {a}s it calls {b}. What differs?")

print("\\nStare at the bottom row. If YOU cannot tell them apart either,")
print("the fix is a better feature representation (Chapter 14), not a better")
print("optimiser. If you CAN, ask which pixels you used -- and engineer that.")
''',
        key="ch03_erroranalysis",
    )

    keypoints([
        "Normalise the confusion matrix by row, then <b>zero the diagonal</b>.",
        "Symmetric hot pairs ⇒ the classes are alike; asymmetric ⇒ one class is "
        "absorbing another.",
        "Per-class recall and precision give you an <b>error budget</b>: fix the "
        "biggest cell first.",
        "Always look at the individual failing instances — labels, missing "
        "features, or genuinely identical classes.",
        "Error analysis usually beats model shopping.",
    ])


# ==========================================================================
def s_3_8():
    section("3.8", "Multilabel and Multioutput Classification")

    lead(
        "Two generalisations that are constantly confused with multiclass, and "
        "with each other. The distinction is precisely about the <i>shape of "
        "$y$</i>."
    )

    table(
        ["Setting", "Shape of one label", "Example"],
        [["<b>Binary</b>", "$y \\in \\{0,1\\}$", "Is this a 5?"],
         ["<b>Multiclass</b>", "$y \\in \\{0,\\dots,9\\}$ — exactly one",
          "Which digit is this?"],
         ["<b>Multilabel</b>", "$\\mathbf{y} \\in \\{0,1\\}^{L}$ — any number "
          "may be on", "Which of Alice, Bob, Charlie appear in this photo?"],
         ["<b>Multioutput</b> (multioutput-multiclass)",
          "$\\mathbf{y} \\in \\mathcal{Y}_1 \\times \\dots \\times \\mathcal{Y}_L$, "
          "each $\\mathcal{Y}_j$ multi-valued",
          "Denoise an image: predict all 64 pixel intensities, each 0–16"]],
    )

    sub("Multilabel classification")

    md(
        "The labels are **not** mutually exclusive, so the softmax constraint "
        "$\\sum_k \\hat p_k = 1$ is wrong. Use $L$ independent sigmoids and a sum "
        "of binary cross-entropies:"
    )

    math(r"""
    \hat p_j = \sigma\bigl(s_j(\mathbf{x})\bigr) = \frac{1}{1 + e^{-s_j(\mathbf{x})}},
    \qquad
    \mathcal{L} = -\frac{1}{L}\sum_{j=1}^{L}
      \Bigl[\, y_j \log \hat p_j + (1 - y_j)\log(1 - \hat p_j) \,\Bigr]
    """)

    sub("Aggregating metrics over labels")

    md(
        "With $L$ labels you get $L$ $F_1$ scores. Four ways to combine them, and "
        "they answer different questions:"
    )

    math(r"""
    F_1^{\text{macro}} = \frac{1}{L}\sum_{j=1}^{L} F_1^{(j)}
    \qquad\qquad
    F_1^{\text{weighted}} = \frac{\sum_{j} n_j F_1^{(j)}}{\sum_{j} n_j}
    """)
    math(r"""
    F_1^{\text{micro}} = \frac{2\sum_{j} TP_j}
                              {2\sum_{j} TP_j + \sum_{j} FP_j + \sum_{j} FN_j}
    """)

    table(
        ["Averaging", "Treats each ... equally", "Use when"],
        [["<code>macro</code>", "<b>label</b>", "Rare labels matter as much as "
          "common ones — the usual scientific choice"],
         ["<code>weighted</code>", "<b>label, weighted by support</b>",
          "You want macro's per-label view but weighted by prevalence"],
         ["<code>micro</code>", "<b>instance–label pair</b>",
          "Every individual prediction matters equally; dominated by common "
          "labels"],
         ["<code>samples</code>", "<b>instance</b>",
          "Score each row's label set, then average — natural for multilabel"]],
    )

    anim_header("Macro vs micro as a rare label degrades")
    md(
        "Three labels. Two are common and stay good; the third is rare and its "
        "$F_1$ decays frame by frame. Macro-$F_1$ falls with it. Micro-$F_1$ "
        "barely notices — which is exactly the failure mode you must not ship."
    )

    n_inst = 1000
    supp = np.array([600, 350, 40])
    f1_fixed = np.array([.92, .88])
    rare_f1 = np.linspace(.90, .05, 40)
    macro, micro, weighted = [], [], []
    for r in rare_f1:
        f1s = np.r_[f1_fixed, r]
        macro.append(f1s.mean())
        weighted.append(float((f1s * supp).sum() / supp.sum()))
        tp = f1s * supp
        micro.append(float(tp.sum() / supp.sum()))

    frames = []
    for k in range(1, 41):
        frames.append(go.Frame(name=f"{rare_f1[k-1]:.2f}", data=[
            go.Scatter(x=rare_f1[:k], y=macro[:k], mode="lines",
                       line=dict(color=C["danger"], width=3.2)),
            go.Scatter(x=rare_f1[:k], y=micro[:k], mode="lines",
                       line=dict(color=C["success"], width=3.2)),
            go.Scatter(x=rare_f1[:k], y=weighted[:k], mode="lines",
                       line=dict(color=C["warning"], width=2.6, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"rare-label F1 = {rare_f1[k-1]:.3f}   ·   macro = {macro[k-1]:.3f}"
            f"   ·   micro = {micro[k-1]:.3f}   ·   weighted = {weighted[k-1]:.3f}",
            color=C["danger"])])))

    f = go.Figure(data=[
        go.Scatter(x=rare_f1[:1], y=macro[:1], mode="lines", name="macro-F1",
                   line=dict(color=C["danger"], width=3.2)),
        go.Scatter(x=rare_f1[:1], y=micro[:1], mode="lines", name="micro-F1",
                   line=dict(color=C["success"], width=3.2)),
        go.Scatter(x=rare_f1[:1], y=weighted[:1], mode="lines", name="weighted-F1",
                   line=dict(color=C["warning"], width=2.6, dash="dash")),
    ])
    f.update_layout(height=420,
                    xaxis=dict(autorange="reversed",
                               title="F1 of the rare label (support 40 / 990)"),
                    yaxis=dict(range=[0, 1], title="aggregated F1"),
                    title="Micro-averaging hides the failure of a rare label",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="rare F1 = ")
    figure(f)

    sub("Classifier chains — when labels are correlated")

    md(
        "Treating the $L$ labels independently throws away their correlations "
        "(*sandals* and *ankle boots* rarely co-occur; *cat* and *whiskers* almost "
        "always do). A **classifier chain** feeds each label's prediction forward "
        "as an extra feature for the next:"
    )

    math(r"""
    \hat y_1 = h_1(\mathbf{x}), \qquad
    \hat y_2 = h_2(\mathbf{x}, \hat y_1), \qquad
    \dots, \qquad
    \hat y_L = h_L(\mathbf{x}, \hat y_1, \dots, \hat y_{L-1})
    """)

    md("`ClassifierChain` in scikit-learn; `order='random'` plus an ensemble of "
       "chains averages away the arbitrariness of the chosen order.")

    sub("Multioutput classification")

    md(
        "Each output is itself multiclass. The clean example is **image "
        "denoising**: input a noisy image, output a clean one, where each of the "
        "$n$ output pixels is a class over intensity levels. Note that the line "
        "between classification and regression genuinely blurs here."
    )

    code_lab(
        "Multilabel with chains, and a multioutput denoiser",
        '''import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.2, stratify=d.target, random_state=42)

# ============ MULTILABEL: three non-exclusive properties of a digit ========
def labels_of(y):
    return np.c_[y >= 7,                    # "large"
                 y % 2 == 1,                # "odd"
                 np.isin(y, [0, 6, 8, 9])]  # "has a closed loop"

Y_train, Y_test = labels_of(y_train), labels_of(y_test)
names = ["large(>=7)", "odd", "closed-loop"]
print("label supports in train:",
      {n: int(c) for n, c in zip(names, Y_train.sum(0))})

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)   # natively multilabel
P = knn.predict(X_test)

print(f"\\n{'label':<14}{'F1':>8}{'support':>9}")
for j, n in enumerate(names):
    print(f"{n:<14}{f1_score(Y_test[:, j], P[:, j]):>8.4f}{int(Y_test[:,j].sum()):>9}")

print(f"\\nmacro    F1 = {f1_score(Y_test, P, average='macro'):.4f}"
      "   <- every label counts the same")
print(f"micro    F1 = {f1_score(Y_test, P, average='micro'):.4f}"
      "   <- every prediction counts the same")
print(f"weighted F1 = {f1_score(Y_test, P, average='weighted'):.4f}")
print(f"samples  F1 = {f1_score(Y_test, P, average='samples'):.4f}")
print(f"\\nsubset accuracy (ALL 3 labels right) = {accuracy_score(Y_test, P):.4f}"
      "   <- the strictest")
print(f"hamming loss (fraction of wrong labels) = {hamming_loss(Y_test, P):.4f}")

# ---- chains exploit label correlation ------------------------------------
base = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
indep = MultiOutputClassifier(base).fit(X_train, Y_train)
chain = ClassifierChain(base, order=[0, 1, 2], random_state=42).fit(X_train, Y_train)
print(f"\\nindependent   macro-F1 = "
      f"{f1_score(Y_test, indep.predict(X_test), average='macro'):.4f}")
print(f"classifier chain macro-F1 = "
      f"{f1_score(Y_test, chain.predict(X_test), average='macro'):.4f}")
print("correlation between 'odd' and 'closed-loop' in train: "
      f"{np.corrcoef(Y_train[:,1], Y_train[:,2])[0,1]:+.3f}")

# ============ MULTIOUTPUT: denoise an 8x8 image ==========================
rng = np.random.default_rng(42)
noise_tr = rng.integers(0, 9, X_train.shape)
noise_te = rng.integers(0, 9, X_test.shape)
Xn_train = np.clip(X_train + noise_tr, 0, 16).astype(int)
Xn_test  = np.clip(X_test  + noise_te, 0, 16).astype(int)

den = KNeighborsClassifier(n_neighbors=3).fit(Xn_train, X_train.astype(int))
clean = den.predict(Xn_test[:6])
print(f"\\ndenoiser output shape = {clean.shape}   "
      f"({clean.shape[1]} outputs, each a class in 0..16)")
print(f"mean |error| per pixel: noisy = "
      f"{np.abs(Xn_test[:6] - X_test[:6]).mean():.2f}  ->  denoised = "
      f"{np.abs(clean - X_test[:6]).mean():.2f}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=3, cols=6, row_titles=["noisy input", "denoised", "target"])
for j in range(6):
    for r, Z in enumerate([Xn_test[j], clean[j], X_test[j]]):
        fig.add_trace(go.Heatmap(z=np.asarray(Z).reshape(8, 8)[::-1], zmin=0, zmax=16,
                                 colorscale=PARULA, showscale=False), r + 1, j + 1)
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(height=440, title="Multioutput classification = image denoising")
''',
        key="ch03_multilabel",
    )

    quiz(
        "You must tag news articles with any of 200 topics; most articles have "
        "2–4 tags and 150 topics are rare. Which setting and which averaging?",
        ["Multiclass with micro-F1", "Multilabel with micro-F1",
         "Multilabel with macro-F1", "Multioutput with accuracy"],
        2,
        "Tags are not mutually exclusive ⇒ multilabel. With 150 rare topics, "
        "micro-averaging would let the handful of common topics carry the score "
        "while the rare ones fail silently — exactly the animation above. Report "
        "macro-F1 (and per-label F1 alongside it).",
        key="ch03q3",
    )

    keypoints([
        "Multiclass: one label from $K$. Multilabel: a <b>subset</b> of $L$. "
        "Multioutput: $L$ labels, each itself multi-valued.",
        "Multilabel ⇒ $L$ sigmoids and $L$ binary cross-entropies, never a softmax.",
        "<b>macro</b> = per label, <b>micro</b> = per prediction, "
        "<b>weighted</b> = per label × support, <b>samples</b> = per instance.",
        "Micro-averaging hides failures on rare labels — report macro too.",
        "<code>ClassifierChain</code> exploits label correlations that independent "
        "binary classifiers throw away.",
    ])


# ==========================================================================
def s_3_9():
    section("3.9", "Exercises & Chapter Review")

    lead("Four exercises, each of which teaches a technique you will reuse.")

    exercise(
        1, "Try to build a classifier for the MNIST dataset that achieves over "
        "97 % accuracy on the test set. Hint: `KNeighborsClassifier` works well; "
        "search for good `weights` and `n_neighbors` values.",
        "A grid over `n_neighbors ∈ {3, 4, 5}` and `weights ∈ {'uniform', "
        "'distance'}` gets $k$-NN to about 97.1 % on full MNIST. `weights="
        "'distance'` almost always wins: closer neighbours should count for more, "
        "which softens the arbitrariness of the hard $k$ cut-off.\n\n"
        "Be aware of the cost: $k$-NN on 60 000 × 784 must compute a distance to "
        "every training point for every query (§1.6). Fitting is instant; "
        "*predicting* the 10 000 test images takes minutes.",
        code='''from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    {"n_neighbors": [3, 4, 5], "weights": ["uniform", "distance"]},
    cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
print("test:", grid.best_estimator_.score(X_test, y_test))''')

    exercise(
        2, "Write a function that can shift an MNIST image in any direction by one "
        "pixel. Then, for each image in the training set, create four shifted "
        "copies and add them to the training set. Train your best model on this "
        "expanded set and measure the improvement.",
        "This is **data augmentation** (also called *training set expansion*), and "
        "it is one of the most reliable accuracy gains available. It works because "
        "you are injecting a **known invariance** — a 3 shifted one pixel is still "
        "a 3 — that the model would otherwise have to learn from data.\n\n"
        "Expect roughly 97.1 % → 97.6 % on MNIST for $k$-NN, at 5× the training "
        "set size and therefore 5× the prediction cost. Chapter 14 shows how CNNs "
        "get translation invariance *architecturally* instead, which is why they "
        "need far less augmentation to reach far better accuracy.\n\n"
        "Critically: **augment the training set only**. Augmenting the test set "
        "changes the question you are answering.",
        code='''from scipy.ndimage import shift as nd_shift

def shift_image(image, dx, dy, side=28):
    img = image.reshape((side, side))
    return nd_shift(img, [dy, dx], cval=0, mode="constant").reshape(-1)

X_aug = [X_train]
y_aug = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    X_aug.append(np.apply_along_axis(shift_image, 1, X_train, dx, dy))
    y_aug.append(y_train)
X_aug = np.concatenate(X_aug); y_aug = np.concatenate(y_aug)

# shuffle, or the 5 copies of each digit end up adjacent
idx = np.random.permutation(len(X_aug))
X_aug, y_aug = X_aug[idx], y_aug[idx]
print(X_aug.shape)          # 5x the original''')

    exercise(
        3, "Tackle the Titanic dataset. Build a pipeline that prepares the data "
        "and a classifier that predicts survival.",
        "The value of this exercise is entirely in the **preprocessing**, which is "
        "why it is the standard first Kaggle competition.\n\n"
        "* `Age` has ~20 % missing — impute with the median *within* "
        "`Pclass`×`Sex` groups, not globally, because those groups have very "
        "different age profiles.\n"
        "* `Cabin` is ~77 % missing, but its *first letter* is the deck, and "
        "whether it is missing at all is itself predictive (poorer passengers had "
        "no cabin recorded).\n"
        "* `Name` contains a title (Mr/Mrs/Miss/Master/Rev/Dr) which encodes age, "
        "sex and class simultaneously — extract it with a regex.\n"
        "* Engineer `FamilySize = SibSp + Parch + 1` and `IsAlone`.\n"
        "* `Sex` and `Embarked` are one-hot; `Pclass` is genuinely ordinal.\n\n"
        "A random forest on these features reaches roughly 80–83 % accuracy. "
        "Report a confusion matrix, not just accuracy — the classes are 62/38.",
        code='''title = df["Name"].str.extract(r",\\s*([^\\.]+)\\.")[0]
df["Title"] = title.replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer"],
    "Rare").replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda s: s.fillna(s.median()))''')

    exercise(
        4, "Build a spam classifier.",
        "The pipeline is: parse the emails, then convert each into a feature "
        "vector, then classify.\n\n"
        "**Preprocessing choices, each of which should be a tunable "
        "hyperparameter so you can measure whether it helps:** strip headers, "
        "lowercase everything, remove punctuation, replace all URLs with the token "
        "`URL`, replace all numbers with `NUMBER`, and apply stemming (so "
        "*compute*, *computing*, *computed* collapse to one token).\n\n"
        "**Vectorisation:** `CountVectorizer` for raw counts, or `TfidfVectorizer` "
        "for term-frequency × inverse-document-frequency, which down-weights words "
        "that appear everywhere:\n\n"
        "$\\text{tfidf}(t, d) = \\text{tf}(t, d) \\cdot \\log\\frac{1 + N}"
        "{1 + \\text{df}(t)} + 1$\n\n"
        "**Model:** `MultinomialNB` is the classic and is nearly free to train; "
        "`LogisticRegression` on TF-IDF is usually a little better.\n\n"
        "**Metric:** *not* accuracy. A false positive (a real email in the spam "
        "folder) is far worse than a false negative. Set a high precision floor — "
        "say 99.5 % — and report the recall you achieve at that operating point, "
        "using exactly the threshold-selection procedure from §3.4.",
        code='''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve

pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english",
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"))
pipe.fit(X_train_text, y_train)

scores = pipe.decision_function(X_val_text)
prec, rec, thr = precision_recall_curve(y_val, scores)
i = np.argmax(prec >= 0.995)
print(f"threshold {thr[i]:.3f}: precision {prec[i]:.4f}, recall {rec[i]:.4f}")''')

    rule()

    sub("The metric decision table — the thing to screenshot")

    table(
        ["Situation", "Report", "Never report"],
        [["Balanced classes, symmetric costs",
          "Accuracy, plus the confusion matrix", "Accuracy alone"],
         ["Skewed classes",
          "PR curve, PR-AUC (average precision), per-class recall",
          "Accuracy, ROC-AUC alone"],
         ["Ranking / triage (act on the top $k$)",
          "Precision@$k$, recall@$k$, ROC-AUC", "$F_1$ at $\\tau=0$"],
         ["A hard operating requirement exists",
          "The other metric at the required threshold", "Anything at the default "
          "threshold"],
         ["Multiclass",
          "Normalised confusion matrix + macro-F1 + per-class report",
          "Overall accuracy alone"],
         ["Multilabel with rare labels", "macro-F1 and per-label F1",
          "micro-F1 alone"],
         ["You need probabilities, not decisions",
          "Log loss, Brier score, a calibration curve", "AUC (it is blind to "
          "calibration)"]],
    )

    keypoints([
        "Accuracy is meaningless on skewed data — always compute the "
        "majority-class baseline.",
        "Confusion matrix → precision/recall → threshold → PR or ROC curve. In "
        "that order, every time.",
        "The threshold is a <b>product decision</b>, not a default: derive it from "
        "the requirement.",
        "AUC is a ranking probability, blind to prevalence and to calibration.",
        "Error analysis on the normalised, diagonal-free confusion matrix beats "
        "trying a new algorithm.",
    ], title="Chapter 3 in five lines")

    refs([
        ("Davis & Goadrich — *The Relationship Between Precision-Recall and ROC "
         "Curves*", "https://doi.org/10.1145/1143844.1143874"),
        ("Saito & Rehmsmeier — *The Precision-Recall Plot Is More Informative than "
         "the ROC Plot on Imbalanced Datasets*",
         "https://doi.org/10.1371/journal.pone.0118432"),
        ("Hanley & McNeil — *The Meaning and Use of the Area under a ROC Curve*",
         "https://doi.org/10.1148/radiology.143.1.7063747"),
        ("LeCun et al. — the MNIST database",
         "http://yann.lecun.com/exdb/mnist/"),
    ])


# ==========================================================================
SECTIONS = [
    ("3.1", "MNIST", s_3_1),
    ("3.2", "Training a Binary Classifier", s_3_2),
    ("3.3", "Performance Measures", s_3_3),
    ("3.4", "The Precision/Recall Trade-off", s_3_4),
    ("3.5", "The ROC Curve", s_3_5),
    ("3.6", "Multiclass Classification", s_3_6),
    ("3.7", "Error Analysis", s_3_7),
    ("3.8", "Multilabel & Multioutput", s_3_8),
    ("3.9", "Exercises & Review", s_3_9),
]

nav.render_chapter(CH, SECTIONS)
