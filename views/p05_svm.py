"""Chapter 5 — Support Vector Machines."""

from __future__ import annotations

import numpy as np
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
CH = "ch05"

hero(
    kicker="Part I · Chapter 5",
    title="Support Vector Machines",
    blurb=(
        "The most elegant classical algorithm: fit the <i>widest possible street</i> "
        "between the classes. We derive the primal from the margin, take the "
        "Lagrangian dual, discover that only a handful of instances matter, and "
        "then find that the dual is written entirely in dot products — which is "
        "the door to the kernel trick and infinite-dimensional feature spaces."
    ),
    chips=["Full dual derivation", "8 sub-sections", "8 animations",
           "8 code labs", "The kernel trick"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_5_1():
    section("5.1", "Linear SVM Classification — The Widest Street")

    lead(
        "Many hyperplanes separate two linearly separable classes. The SVM picks "
        "the one that stays as far as possible from the nearest instance of either "
        "class. That single criterion determines everything else in this chapter."
    )

    sub("Large margin classification")

    md(
        "Think of the decision boundary as the centre line of a **street**. The "
        "SVM fits the widest street that still separates the classes. Instances "
        "on the edges of the street are the **support vectors** — and they are "
        "the only ones that matter:"
    )

    idea(
        "Only the support vectors matter",
        "Add a thousand new instances far from the street and the boundary does "
        "not move by a millimetre. Move a single support vector and the whole "
        "boundary shifts. This is why the model is called a <i>support vector</i> "
        "machine, and why the fitted model is compact even when trained on huge "
        "data — the rest of the training set is, after fitting, irrelevant.",
    )

    anim_header("Which separator? Watch the margin grow")
    md(
        "Every frame shows a different separating line — all of them classify the "
        "training set perfectly. The margin width is printed as each rotates. The "
        "SVM chooses the frame with the maximum."
    )

    rng = np.random.default_rng(3)
    Xa = np.r_[rng.normal([-1.6, -1.0], .52, (28, 2)),
               rng.normal([1.7, 1.3], .52, (28, 2))]
    ya = np.r_[np.zeros(28), np.ones(28)]
    ytilde = 2 * ya - 1

    from sklearn.svm import SVC
    svc = SVC(kernel="linear", C=1e6).fit(Xa, ya)
    w_opt = svc.coef_[0]; b_opt = svc.intercept_[0]
    ang_opt = np.arctan2(w_opt[1], w_opt[0])

    gx = np.linspace(-4, 4, 60)
    angles = ang_opt + np.linspace(-1.05, 1.05, 40)

    def margin_for(theta):
        w = np.array([np.cos(theta), np.sin(theta)])
        proj = Xa @ w
        lo = proj[ya == 1].min(); hi = proj[ya == 0].max()
        if lo <= hi:
            return None, None, 0.0
        b = -(lo + hi) / 2
        return w, b, float(lo - hi)

    # keep only the angles that still separate the two classes
    valid = [(th,) + margin_for(th) for th in angles]
    valid = [v for v in valid if v[1] is not None]
    best_mg = max(v[3] for v in valid)

    frames = []
    for th, w, b, mg in valid:
        yy = -(w[0] * gx + b) / (w[1] if abs(w[1]) > 1e-6 else 1e-6)
        off = (mg / 2) / (abs(w[1]) if abs(w[1]) > 1e-6 else 1e-6)
        col = C["success"] if mg > .97 * best_mg else C["primary"]
        frames.append(go.Frame(name=f"{np.degrees(th):.0f}", data=[
            go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=9,
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=9,
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=np.r_[gx, gx[::-1]], y=np.r_[yy - off, (yy + off)[::-1]],
                       fill="toself", fillcolor=alpha(col, .16),
                       line=dict(width=0), hoverinfo="skip"),
            go.Scatter(x=gx, y=yy, mode="lines", line=dict(color=col, width=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"angle {np.degrees(th):+6.1f}°   ·   street width = {mg:.4f}",
            color=col)])))

    _, w0, b0, m0 = valid[0]
    yy0 = -(w0[0] * gx + b0) / w0[1]
    f = go.Figure(data=[
        go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=9,
                                               line=dict(color="#fff", width=1))),
        go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=9,
                                               line=dict(color="#fff", width=1))),
        go.Scatter(x=np.r_[gx, gx[::-1]],
                   y=np.r_[yy0 - m0 / 2 / abs(w0[1]),
                           (yy0 + m0 / 2 / abs(w0[1]))[::-1]],
                   fill="toself", fillcolor=alpha(C["primary"], .16),
                   line=dict(width=0), name="the street", hoverinfo="skip"),
        go.Scatter(x=gx, y=yy0, mode="lines", name="decision boundary",
                   line=dict(color=C["primary"], width=4)),
    ])
    f.update_layout(height=500, xaxis=dict(range=[-3.6, 3.6], title="x₁"),
                    yaxis=dict(range=[-3.2, 3.4], title="x₂"),
                    title="Rotate the separator — the SVM keeps the widest street",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(160), slider_prefix="angle ")
    figure(f)

    pitfall(
        "SVMs are extremely sensitive to feature scaling",
        "The margin is measured with a Euclidean distance. If one feature runs "
        "0–1 and another 0–100 000, the second feature owns the geometry and the "
        "'widest street' is computed in a badly distorted space. "
        "<b>Always <code>StandardScaler</code> before an SVM.</b> The lab below "
        "shows the accuracy difference, and it is not subtle.",
    )

    sub("Hard margin vs soft margin")

    table(
        ["", "Hard margin", "Soft margin"],
        [["Rule", "No instance may be inside the street",
          "Instances may violate the street; violations are penalised"],
         ["Requires", "Perfectly linearly separable data", "Nothing"],
         ["Outlier sensitivity", "<b>Catastrophic</b> — one outlier can make the "
          "problem infeasible or collapse the margin", "Controlled"],
         ["Controlled by", "—", "the hyperparameter <b>C</b>"]],
    )

    md("The trade-off is governed by $C$, and its direction catches everyone out:")

    table(
        ["$C$", "Penalty on violations", "Street", "Effect"],
        [["Small $C$", "Cheap", "<b>Wide</b>, many instances inside",
          "More regularisation → possible underfitting"],
         ["Large $C$", "Expensive", "<b>Narrow</b>, few violations",
          "Less regularisation → possible overfitting"],
         ["$C \\to \\infty$", "Infinite", "Hard margin",
          "Fails entirely if the data is not separable"]],
    )

    note(
        "If your SVM is overfitting, <b>reduce</b> C. This is the opposite "
        "direction to $\\alpha$ in ridge/lasso, because $C$ multiplies the "
        "<i>loss</i> term rather than the penalty term — see §5.5, where "
        "$C \\propto 1/\\alpha$ falls out of the objective.",
    )

    anim_header("C sweeping from 0.01 to 1000")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng2 = np.random.default_rng(7)
    Xb = np.r_[rng2.normal([-1.1, -0.7], .95, (60, 2)),
               rng2.normal([1.2, 0.9], .95, (60, 2))]
    yb = np.r_[np.zeros(60), np.ones(60)]
    Xb[0] = [1.9, 1.6]                                  # a deliberate outlier
    Xb[60] = [-1.7, -1.4]

    Cs = np.logspace(-2, 3, 34)
    g1 = np.linspace(-4.2, 4.2, 130); g2 = np.linspace(-3.8, 3.8, 130)
    G1, G2 = np.meshgrid(g1, g2); GG = np.c_[G1.ravel(), G2.ravel()]

    infos = []
    for Cv in Cs:
        m = SVC(kernel="linear", C=Cv).fit(Xb, yb)
        Z = m.decision_function(GG).reshape(G1.shape)
        w = m.coef_[0]
        infos.append((Z, 2 / np.linalg.norm(w), len(m.support_),
                      float(m.score(Xb, yb))))

    frames = []
    for i, Cv in enumerate(Cs):
        Z, width, nsv, acc = infos[i]
        frames.append(go.Frame(name=f"{Cv:.3g}", data=[
            go.Contour(x=g1, y=g2, z=Z, showscale=False,
                       contours=dict(start=-1, end=1, size=2,
                                     coloring="lines"),
                       line=dict(width=2.5, dash="dash"),
                       colorscale=[[0, C["muted"]], [1, C["muted"]]]),
            go.Contour(x=g1, y=g2, z=Z, showscale=False,
                       contours=dict(start=0, end=0, size=1, coloring="lines"),
                       line=dict(width=4), colorscale=[[0, C["primary"]],
                                                       [1, C["primary"]]]),
            go.Scatter(x=Xb[yb == 0, 0], y=Xb[yb == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=8,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xb[yb == 1, 0], y=Xb[yb == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=8,
                                   line=dict(color="#fff", width=.8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"C = {Cv:>8.3g}   ·   street width = {width:.3f}   ·   "
            f"{nsv} support vectors   ·   train accuracy = {acc:.3f}")])))

    Z0 = infos[0][0]
    f = go.Figure(data=[
        go.Contour(x=g1, y=g2, z=Z0, showscale=False,
                   contours=dict(start=-1, end=1, size=2, coloring="lines"),
                   line=dict(width=2.5, dash="dash"),
                   colorscale=[[0, C["muted"]], [1, C["muted"]]],
                   name="street edges"),
        go.Contour(x=g1, y=g2, z=Z0, showscale=False,
                   contours=dict(start=0, end=0, size=1, coloring="lines"),
                   line=dict(width=4),
                   colorscale=[[0, C["primary"]], [1, C["primary"]]],
                   name="boundary"),
        go.Scatter(x=Xb[yb == 0, 0], y=Xb[yb == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=8,
                                               line=dict(color="#fff", width=.8))),
        go.Scatter(x=Xb[yb == 1, 0], y=Xb[yb == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=8,
                                               line=dict(color="#fff", width=.8))),
    ])
    f.update_layout(height=500, xaxis=dict(range=[-4.2, 4.2], title="x₁"),
                    yaxis=dict(range=[-3.8, 3.8], title="x₂"),
                    title="Soft margin: C controls the width/violation trade-off")
    anim.animate(f, frames, duration=nav.anim_ms(190), slider_prefix="C = ")
    figure(f, "Small C: a wide street tolerating the outliers. Large C: a narrow "
              "street contorted to accommodate them.")

    code_lab(
        "Scaling, C, and counting support vectors",
        '''import numpy as np
from sklearn.datasets import load_iris, make_blobs
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# ============ 1. why scaling is not optional ==============================
rng = np.random.default_rng(0)
X, y = make_blobs(n_samples=400, centers=2, cluster_std=1.4, random_state=0)
X[:, 1] *= 500.0             # feature 2 now lives on a very different scale

raw    = SVC(kernel="linear", C=1, max_iter=200_000).fit(X, y)
scaled = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)).fit(X, y)
print("=== effect of feature scaling ===")
print(f"unscaled  CV accuracy = "
      f"{cross_val_score(SVC(kernel='linear', max_iter=200_000), X, y, cv=5).mean():.4f}   "
      f"support vectors = {len(raw.support_)}")
print(f"scaled    CV accuracy = "
      f"{cross_val_score(make_pipeline(StandardScaler(), SVC(kernel='linear')), X, y, cv=5).mean():.4f}   "
      f"support vectors = {len(scaled[-1].support_)}")
print("The unscaled model needs far more support vectors -- the geometry is distorted.")

# ============ 2. C and the margin =========================================
iris = load_iris()
Xi = iris.data[:, 2:4]                       # petal length & width
yi = (iris.target == 2).astype(int)          # virginica or not
Xs = StandardScaler().fit_transform(Xi)

print(f"\\n=== C controls the street width ===")
print(f"{'C':>10}{'street width':>15}{'# SVs':>8}{'violations':>12}{'train acc':>11}")
for Cv in [0.01, 0.1, 1, 10, 100, 1000, 1e6]:
    m = SVC(kernel="linear", C=Cv).fit(Xs, yi)
    w = m.coef_[0]
    width = 2 / np.linalg.norm(w)
    margins = (2*yi - 1) * m.decision_function(Xs)
    print(f"{Cv:>10.0e}{width:>15.4f}{len(m.support_):>8}"
          f"{int((margins < 1).sum()):>12}{m.score(Xs, yi):>11.4f}")
print("Large C -> narrow street, few support vectors, few violations.")

# ============ 3. only the support vectors matter ==========================
m = SVC(kernel="linear", C=1).fit(Xs, yi)
sv_idx = m.support_
print(f"\\n=== only {len(sv_idx)} of {len(Xs)} instances matter ===")
m2 = SVC(kernel="linear", C=1).fit(Xs[sv_idx], yi[sv_idx])
print(f"trained on all {len(Xs)}          : w = {m.coef_[0].round(6)}, "
      f"b = {m.intercept_[0]:.6f}")
print(f"trained on the {len(sv_idx)} SVs only : w = {m2.coef_[0].round(6)}, "
      f"b = {m2.intercept_[0]:.6f}")
print(f"identical to {np.abs(m.coef_[0]-m2.coef_[0]).max():.2e}")

# now delete 60 NON-support vectors and refit
keep = np.setdiff1d(np.arange(len(Xs)), np.setdiff1d(np.arange(len(Xs)), sv_idx)[:60])
m3 = SVC(kernel="linear", C=1).fit(Xs[keep], yi[keep])
print(f"after deleting 60 non-SVs     : w = {m3.coef_[0].round(6)}, "
      f"b = {m3.intercept_[0]:.6f}   <- unchanged")

# ============ 4. dual coefficients are bounded by C =======================
print(f"\\n=== the dual variables alpha_i live in [0, C] (see 5.6) ===")
for Cv in [0.1, 1, 10]:
    mm = SVC(kernel="linear", C=Cv).fit(Xs, yi)
    a = np.abs(mm.dual_coef_[0])
    print(f"C = {Cv:>5}: max|alpha| = {a.max():.4f}  "
          f"({int((a > Cv - 1e-6).sum())} at the bound = margin violators)")
''',
        key="ch05_linear",
    )

    keypoints([
        "The SVM fits the <b>widest street</b> between the classes; only the "
        "<b>support vectors</b> on its edges determine the boundary.",
        "Hard margin needs perfect separability and dies on a single outlier; "
        "soft margin is the practical version.",
        "<b>$C$ is inverse regularisation</b>: small $C$ = wide street = more "
        "regularisation. Overfitting? <i>Reduce</i> $C$.",
        "<b>Always scale first</b> — the margin is a Euclidean distance.",
        "Deleting non-support-vectors changes nothing at all.",
    ])


# ==========================================================================
def s_5_2():
    section("5.2", "Nonlinear SVM Classification")

    lead(
        "Many datasets are nowhere near linearly separable. The classical remedy "
        "is to add features until they are — and there is a beautiful reason why "
        "this always works, plus a trick that makes it free."
    )

    sub("Adding polynomial features")

    md(
        "The one-dimensional example that makes it obvious: points on a line at "
        "$x_1 \\in \\{-4,-3,\\dots,4\\}$, with the middle ones in one class. No "
        "threshold separates them. Add $x_2 = x_1^2$ and they separate trivially."
    )

    anim_header("Lifting 1-D data into 2-D makes it linearly separable")

    x1 = np.arange(-4, 5).astype(float)
    ylab = ((np.abs(x1) <= 1)).astype(int)
    steps = np.linspace(0, 1, 30)

    frames = []
    for t in steps:
        y2 = t * (x1 ** 2)
        frames.append(go.Frame(name=f"{t:.2f}", data=[
            go.Scatter(x=x1[ylab == 0], y=y2[ylab == 0], mode="markers",
                       marker=dict(color=C["train"], size=14,
                                   line=dict(color="#fff", width=1.5))),
            go.Scatter(x=x1[ylab == 1], y=y2[ylab == 1], mode="markers",
                       marker=dict(color=C["warning"], size=14,
                                   line=dict(color="#fff", width=1.5))),
            go.Scatter(x=[-4.6, 4.6], y=[t * 2.5, t * 2.5], mode="lines",
                       line=dict(color=C["danger"], width=3,
                                 dash="dash" if t < .5 else "solid"),
                       opacity=float(min(1, 2 * t))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"x₂ = {t:.2f}·x₁²   ·   "
            f"{'not separable by any line' if t < 0.25 else 'linearly separable'}",
            color=C["danger"] if t < .25 else C["success"])])))

    f = go.Figure(data=[
        go.Scatter(x=x1[ylab == 0], y=np.zeros((ylab == 0).sum()), mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=14,
                                               line=dict(color="#fff", width=1.5))),
        go.Scatter(x=x1[ylab == 1], y=np.zeros((ylab == 1).sum()), mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=14,
                                               line=dict(color="#fff", width=1.5))),
        go.Scatter(x=[-4.6, 4.6], y=[0, 0], mode="lines", name="separating line",
                   line=dict(color=C["danger"], width=3), opacity=0),
    ])
    f.update_layout(height=430, xaxis=dict(range=[-4.8, 4.8], title="x₁"),
                    yaxis=dict(range=[-1.5, 17], title="x₂ = x₁²"),
                    title="The same nine points, lifted into a second dimension")
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="lift = ")
    figure(f)

    sub("The polynomial kernel")

    md(
        "Adding polynomial features works, but at low degree it cannot represent "
        "complex boundaries and at high degree it creates a combinatorial "
        "explosion of features (§4.3), making the model slow. The **kernel trick** "
        "gets the *exact same result* as adding those features, without ever "
        "computing them:"
    )

    math(r"""
    K_{\text{poly}}\bigl(\mathbf{a}, \mathbf{b}\bigr) \;=\;
    \bigl(\gamma\,\mathbf{a}^\top \mathbf{b} + r\bigr)^{d}
    """)
    where({r"d": "<code>degree</code> — the polynomial degree",
           r"\gamma": "<code>gamma</code> — scales the dot product",
           r"r": "<code>coef0</code> — controls how much high-degree terms are "
                 "weighted relative to low-degree ones"})

    derive(
        [("Take the simplest case: 2-D inputs, degree 2, $\\gamma = 1$, $r = 0$. "
          "Expand the kernel algebraically.",
          r"K(\mathbf{a},\mathbf{b}) = \bigl(\mathbf{a}^\top\mathbf{b}\bigr)^2 "
          r"= \bigl(a_1 b_1 + a_2 b_2\bigr)^2"),
         ("Multiply it out:",
          r"= a_1^2 b_1^2 + 2 a_1 b_1 a_2 b_2 + a_2^2 b_2^2"),
         ("Now regroup as a dot product of two vectors, each depending on only one "
          "of the inputs:",
          r"= \begin{bmatrix} a_1^2 \\ \sqrt{2}\,a_1 a_2 \\ a_2^2\end{bmatrix}^{\!\top}"
          r"\begin{bmatrix} b_1^2 \\ \sqrt{2}\,b_1 b_2 \\ b_2^2\end{bmatrix} "
          r"= \phi(\mathbf{a})^\top \phi(\mathbf{b})"),
         ("So $K(\\mathbf{a},\\mathbf{b}) = \\phi(\\mathbf{a})^\\top\\phi(\\mathbf{b})$ "
          "with the explicit 3-D mapping $\\phi(\\mathbf{x}) = (x_1^2, "
          "\\sqrt2 x_1x_2, x_2^2)$. Computing the left side costs one 2-D dot "
          "product and one squaring. Computing the right side requires building "
          "3-D vectors first.", None),
         ("For $n$ features and degree $d$ the mapping has $\\binom{n+d}{d}$ "
          "components — 176 851 for $n=100$, $d=3$ — but the kernel is still just "
          "<b>one dot product and one power</b>. The saving is unbounded.", None),
         ("<b>Mercer's theorem</b> guarantees this works for any continuous, "
          "symmetric, positive semi-definite $K$: such a $\\phi$ exists, even when "
          "you cannot write it down, and even when the target space is "
          "infinite-dimensional (which is exactly the RBF case).", None)],
        title="The kernel trick, made completely explicit",
    )

    sub("Similarity features and the Gaussian RBF kernel")

    md(
        "A different way to add features: measure **similarity to landmarks**. "
        "Pick landmarks $\\ell$ and define new features by a radial basis function:"
    )

    math(r"""
    \phi_\gamma\bigl(\mathbf{x}, \boldsymbol\ell\bigr) \;=\;
    \exp\Bigl(-\gamma\,\bigl\lVert \mathbf{x} - \boldsymbol\ell \bigr\rVert^{2}\Bigr)
    """)

    md(
        "The simplest choice of landmarks is *every training instance*, which "
        "creates $m$ new features. That is expensive — and again the kernel trick "
        "makes it free:"
    )

    math(r"""
    K_{\text{RBF}}\bigl(\mathbf{a}, \mathbf{b}\bigr) \;=\;
    \exp\Bigl(-\gamma\,\bigl\lVert \mathbf{a} - \mathbf{b} \bigr\rVert^{2}\Bigr)
    """)

    proof(
        "The RBF kernel corresponds to an infinite-dimensional feature space",
        "Expand $e^{2\\gamma\\mathbf{a}^\\top\\mathbf{b}}$ as a power series: "
        "$K_{\\text{RBF}} = e^{-\\gamma\\lVert\\mathbf{a}\\rVert^2}"
        "e^{-\\gamma\\lVert\\mathbf{b}\\rVert^2}\\sum_{k=0}^{\\infty}"
        "\\frac{(2\\gamma)^k}{k!}(\\mathbf{a}^\\top\\mathbf{b})^k$. Every term "
        "$(\\mathbf{a}^\\top\\mathbf{b})^k$ is itself a polynomial kernel of degree "
        "$k$, so the RBF kernel is an infinite weighted sum of polynomial kernels "
        "of <i>every</i> degree. Its feature map $\\phi$ therefore lives in an "
        "infinite-dimensional space — which you could never construct explicitly, "
        "yet the kernel evaluates in $\\mathcal{O}(n)$.",
    )

    table(
        ["Hyperparameter", "Increase it", "Decrease it"],
        [["<b>$\\gamma$</b> (RBF width)",
          "Narrower bell → the influence of each instance is local → the boundary "
          "becomes wiggly and irregular → <b>overfitting</b>",
          "Wider bell → each instance influences a large region → smoother "
          "boundary → <b>underfitting</b>"],
         ["<b>$C$</b>", "Fewer violations → narrower street → <b>overfitting</b>",
          "More violations tolerated → <b>underfitting</b>"]],
        "$\\gamma$ and $C$ both act as regularisation dials, and they interact — "
        "tune them together on a 2-D grid.",
    )

    anim_header("γ and C on the moons dataset")

    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xm, ym = ds.moons(n=220, noise=.26)
    Xm = StandardScaler().fit_transform(Xm)
    gm1 = np.linspace(Xm[:, 0].min() - .6, Xm[:, 0].max() + .6, 110)
    gm2 = np.linspace(Xm[:, 1].min() - .6, Xm[:, 1].max() + .6, 110)
    GM1, GM2 = np.meshgrid(gm1, gm2); GMM = np.c_[GM1.ravel(), GM2.ravel()]

    combos = [(g, c) for g in [0.05, 0.3, 1.0, 5.0, 30.0]
              for c in [0.1, 1.0, 100.0]]
    cache = []
    for g, c in combos:
        m = SVC(kernel="rbf", gamma=g, C=c).fit(Xm, ym)
        cache.append((m.decision_function(GMM).reshape(GM1.shape),
                      len(m.support_), float(m.score(Xm, ym))))

    frames = []
    for i, (g, c) in enumerate(combos):
        Z, nsv, acc = cache[i]
        tag = ("underfit" if g <= .3 and c <= 1 else
               "overfit" if g >= 5 and c >= 100 else "reasonable")
        col = (C["warning"] if tag == "underfit" else
               C["danger"] if tag == "overfit" else C["success"])
        frames.append(go.Frame(name=f"γ{g}·C{c}", data=[
            go.Contour(x=gm1, y=gm2, z=Z, showscale=False,
                       colorscale=nav.cscale(), opacity=.5, ncontours=20),
            go.Contour(x=gm1, y=gm2, z=Z, showscale=False,
                       contours=dict(start=0, end=0, size=1, coloring="lines"),
                       line=dict(width=3.5),
                       colorscale=[[0, C["ink"]], [1, C["ink"]]]),
            go.Scatter(x=Xm[ym == 0, 0], y=Xm[ym == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xm[ym == 1, 0], y=Xm[ym == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=7,
                                   line=dict(color="#fff", width=.8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"γ = {g:<5}  C = {c:<6}  ·  {nsv} support vectors  ·  "
            f"train acc = {acc:.3f}  ·  {tag}", color=col)])))

    Z0 = cache[0][0]
    f = go.Figure(data=[
        go.Contour(x=gm1, y=gm2, z=Z0, showscale=False, colorscale=nav.cscale(),
                   opacity=.5, ncontours=20),
        go.Contour(x=gm1, y=gm2, z=Z0, showscale=False,
                   contours=dict(start=0, end=0, size=1, coloring="lines"),
                   line=dict(width=3.5), colorscale=[[0, C["ink"]], [1, C["ink"]]]),
        go.Scatter(x=Xm[ym == 0, 0], y=Xm[ym == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=7,
                                               line=dict(color="#fff", width=.8))),
        go.Scatter(x=Xm[ym == 1, 0], y=Xm[ym == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=7,
                                               line=dict(color="#fff", width=.8))),
    ])
    f.update_layout(height=520, title="RBF SVM: 15 combinations of γ and C",
                    xaxis_title="x₁ (standardised)", yaxis_title="x₂ (standardised)")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="")
    figure(f, "Low γ, low C: nearly a straight line. High γ, high C: islands "
              "around individual points — memorisation.")

    tip(
        "Which kernel to try, in order",
        "<b>1.</b> Always try the <b>linear</b> kernel first — use "
        "<code>LinearSVC</code>, which is much faster than "
        "<code>SVC(kernel='linear')</code>. Especially if $n$ is large or the "
        "training set is huge. <b>2.</b> If the training set is not too large, try "
        "the <b>Gaussian RBF</b> kernel; it works well most of the time. "
        "<b>3.</b> Specialised kernels (string kernels, graph kernels) only if "
        "your data structure calls for one.",
    )

    code_lab(
        "The kernel trick, verified numerically; and the γ–C grid",
        '''import numpy as np, time
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

# ============ 1. K(a,b) == phi(a).phi(b)  -- verified =====================
rng = np.random.default_rng(0)
a, b = rng.normal(0, 1, 2), rng.normal(0, 1, 2)

K_direct = (a @ b) ** 2                                   # the kernel
phi = lambda v: np.array([v[0]**2, np.sqrt(2)*v[0]*v[1], v[1]**2])
K_explicit = phi(a) @ phi(b)                              # the feature map
print("=== degree-2 polynomial kernel, 2-D input ===")
print(f"K(a,b)  = (a.b)^2        = {K_direct:.12f}")
print(f"phi(a).phi(b)            = {K_explicit:.12f}")
print(f"difference               = {abs(K_direct-K_explicit):.2e}")
print(f"phi maps R^2 -> R^3      : phi(a) = {phi(a).round(4)}")

# ---- the same for higher dimensions, via sklearn's kernel ---------------
A = rng.normal(0, 1, (5, 4))
B = rng.normal(0, 1, (3, 4))
Kk = polynomial_kernel(A, B, degree=3, gamma=1, coef0=1)
Pf = PolynomialFeatures(3).fit(A)
Ka = Pf.transform(A); Kb = Pf.transform(B)
print(f"\\nfor n=4, d=3: explicit feature space has {Ka.shape[1]} dimensions")
print(f"the kernel computes the same inner products in 4-D. Max relative error "
      f"vs a scaled explicit map: {np.abs(Kk/Kk.max() - (Ka@Kb.T)/(Ka@Kb.T).max()).max():.2e}")

# ============ 2. RBF = infinite-dimensional ==============================
print("\\n=== the RBF kernel as an infinite polynomial series ===")
g = 0.7
exact = float(np.exp(-g * np.sum((a-b)**2)))
partial = 0.0
pref = np.exp(-g*(a@a)) * np.exp(-g*(b@b))
from math import factorial
for k in range(12):
    partial += pref * (2*g)**k / factorial(k) * (a @ b)**k
    if k in (0, 1, 2, 4, 8, 11):
        print(f"  sum of the first {k+1:>2} polynomial kernels = {partial:.10f}")
print(f"  exact exp(-gamma||a-b||^2)             = {exact:.10f}")

# ============ 3. explicit features vs the kernel: cost ===================
X, y = make_moons(n_samples=2000, noise=.25, random_state=0)
print("\\n=== explicit polynomial features vs the polynomial kernel ===")
t0 = time.perf_counter()
exp_pipe = make_pipeline(PolynomialFeatures(10), StandardScaler(),
                         LinearSVC(C=10, max_iter=20000, dual="auto")).fit(X, y)
t_exp = time.perf_counter() - t0
n_feat = exp_pipe[0].transform(X[:1]).shape[1]

t0 = time.perf_counter()
ker_pipe = make_pipeline(StandardScaler(),
                         SVC(kernel="poly", degree=10, coef0=1, C=5)).fit(X, y)
t_ker = time.perf_counter() - t0
print(f"explicit (degree 10): {n_feat:>5} features created, {t_exp:.3f}s, "
      f"accuracy {exp_pipe.score(X, y):.4f}")
print(f"kernel   (degree 10): {0:>5} features created, {t_ker:.3f}s, "
      f"accuracy {ker_pipe.score(X, y):.4f}")

# ============ 4. tune gamma and C together ==============================
print("\\n=== the gamma-C grid ===")
grid = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                    {"svc__gamma": [0.01, 0.1, 1, 5, 30],
                     "svc__C": [0.1, 1, 10, 100, 1000]},
                    cv=5, n_jobs=-1)
grid.fit(X, y)
import pandas as pd
res = pd.DataFrame(grid.cv_results_)
piv = res.pivot_table(index="param_svc__gamma", columns="param_svc__C",
                      values="mean_test_score")
print(piv.round(4).to_string())
print(f"\\nbest: {grid.best_params_}  ->  CV accuracy {grid.best_score_:.4f}")

import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(z=piv.values, x=[str(c) for c in piv.columns],
                           y=[str(g) for g in piv.index], colorscale=PARULA,
                           text=piv.values.round(3), texttemplate="%{text}",
                           colorbar=dict(title="CV acc")))
fig.update_layout(height=420, xaxis_title="C", yaxis_title="gamma",
                  title="The gamma-C interaction surface")
''',
        key="ch05_kernel",
    )

    keypoints([
        "Non-linear data becomes separable in a higher-dimensional feature space.",
        "The <b>kernel trick</b> computes $\\phi(\\mathbf{a})^\\top\\phi(\\mathbf{b})$ "
        "without ever forming $\\phi$ — verified explicitly above.",
        "Polynomial kernel $(\\gamma\\mathbf{a}^\\top\\mathbf{b} + r)^d$; RBF kernel "
        "$e^{-\\gamma\\lVert\\mathbf{a}-\\mathbf{b}\\rVert^2}$.",
        "The RBF kernel's feature space is <b>infinite-dimensional</b> — an "
        "infinite sum of polynomial kernels.",
        "$\\gamma$ up = wigglier = overfit; $\\gamma$ down = smoother = underfit. "
        "Tune with $C$ on a 2-D grid.",
        "Try linear first (<code>LinearSVC</code>), then RBF.",
    ])


# ==========================================================================
def s_5_3():
    section("5.3", "SVM Classes and Computational Complexity")

    lead(
        "scikit-learn offers three routes to a linear SVM and one to a kernelised "
        "one. Picking the wrong class is the difference between two seconds and "
        "two hours."
    )

    table(
        ["Class", "Time complexity", "Out-of-core?", "Scaling required",
         "Kernel trick"],
        [["<code>LinearSVC</code>", "$\\mathcal{O}(m \\times n)$", "❌", "Yes", "❌"],
         ["<code>SGDClassifier</code>", "$\\mathcal{O}(m \\times n)$", "<b>✅</b>",
          "Yes", "❌"],
         ["<code>SVC</code>", "$\\mathcal{O}(m^2 n)$ to $\\mathcal{O}(m^3 n)$",
          "❌", "Yes", "<b>✅</b>"]],
        "The middle column is the one that decides your architecture.",
    )

    md(
        """
Reading the table:

* **`LinearSVC`** is based on `liblinear`. It optimises the *primal* problem,
  scales almost linearly with both $m$ and $n$, and is what you should use for a
  linear SVM on anything large. It does not support the kernel trick.
* **`SGDClassifier(loss='hinge')`** applies stochastic gradient descent to the
  same hinge-loss objective. It converges more slowly than `LinearSVC` but it is
  the only one that handles data too large for RAM (`partial_fit`, §1.5).
* **`SVC`** is based on `libsvm` and implements the kernel trick. Its complexity
  is between quadratic and cubic in $m$, which means it becomes unusable
  somewhere in the tens of thousands of instances.
        """
    )

    warn(
        "The practical cut-off",
        "<code>SVC</code> is superb for small and medium training sets — up to "
        "roughly <b>10 000 instances</b>, maybe 100 000 if you are patient. "
        "Beyond that, going from 10 000 to 100 000 rows multiplies the training "
        "time by between 100 and 1 000. Use <code>LinearSVC</code>, "
        "<code>SGDClassifier</code>, or the <b>Nyström / random-features</b> "
        "approximation of a kernel instead (<code>Nystroem</code>, "
        "<code>RBFSampler</code>).",
    )

    anim_header("Measured scaling: SVC vs LinearSVC vs SGD")
    md(
        "Real timings on this machine, on growing training sets. The log–log "
        "slope <i>is</i> the exponent in the complexity: slope 1 is linear, slope 2 "
        "is quadratic."
    )

    import time
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    sizes = [200, 500, 1000, 2000, 4000]
    timings = {"SVC (rbf)": [], "LinearSVC": [], "SGDClassifier": []}
    for mm in sizes:
        Xs, ys = make_classification(n_samples=mm, n_features=20,
                                     n_informative=10, random_state=0)
        Xs = StandardScaler().fit_transform(Xs)
        for nm, mdl in [("SVC (rbf)", SVC()),
                        ("LinearSVC", LinearSVC(max_iter=4000, dual="auto")),
                        ("SGDClassifier", SGDClassifier(loss="hinge",
                                                        max_iter=1000,
                                                        random_state=0))]:
            t0 = time.perf_counter()
            mdl.fit(Xs, ys)
            timings[nm].append(time.perf_counter() - t0)

    frames = []
    for k in range(2, len(sizes) + 1):
        data, info = [], []
        for i, (nm, ts) in enumerate(timings.items()):
            data.append(go.Scatter(x=sizes[:k], y=ts[:k], mode="lines+markers",
                                   line=dict(color=SEQ[i], width=3),
                                   marker=dict(size=9)))
            lx = np.log(sizes[:k]); ly = np.log(np.maximum(ts[:k], 1e-6))
            slope = float(np.polyfit(lx, ly, 1)[0]) if k > 1 else 0
            info.append(f"{nm}: slope {slope:+.2f}")
        frames.append(go.Frame(name=str(sizes[k - 1]), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"m = {sizes[k-1]:,}   |   " + "   |   ".join(info))])))

    f = go.Figure(data=[go.Scatter(x=sizes[:2], y=ts[:2], mode="lines+markers",
                                   name=nm, line=dict(color=SEQ[i], width=3),
                                   marker=dict(size=9))
                        for i, (nm, ts) in enumerate(timings.items())])
    f.update_layout(height=430, xaxis_type="log", yaxis_type="log",
                    xaxis_title="training-set size m", yaxis_title="fit time (s)",
                    title="Measured complexity — the log–log slope is the exponent",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(600), slider_prefix="m = ")
    figure(f, "SVC's slope is close to 2; the linear methods stay near 1.")

    codenote(
        "Three ways to say 'linear SVM', and they are not identical",
        "<code>LinearSVC(C=c)</code>, <code>SVC(kernel='linear', C=c)</code> and "
        "<code>SGDClassifier(loss='hinge', alpha=1/(m*c))</code> all fit a linear "
        "SVM but with different defaults: <code>LinearSVC</code> regularises the "
        "bias term (centre your data, or set "
        "<code>fit_intercept=True</code> with <code>intercept_scaling</code> "
        "large) and minimises the <i>squared</i> hinge loss by default "
        "(<code>loss='squared_hinge'</code>). Set "
        "<code>loss='hinge'</code> to match the textbook formulation.",
    )

    code_lab(
        "Benchmark the three, and rescue SVC with a kernel approximation",
        '''import numpy as np, time
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def bench(m, n=20):
    X, y = make_classification(n_samples=m, n_features=n, n_informative=12,
                               class_sep=1.1, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0)
    rows = []
    models = {
        "SVC(rbf)":        make_pipeline(StandardScaler(), SVC()),
        "LinearSVC":       make_pipeline(StandardScaler(),
                                         LinearSVC(max_iter=5000, dual="auto")),
        "SGD(hinge)":      make_pipeline(StandardScaler(),
                                         SGDClassifier(loss="hinge", max_iter=1500,
                                                       random_state=0)),
        "Nystroem+Linear": make_pipeline(StandardScaler(),
                                         Nystroem(gamma=.05, n_components=200,
                                                  random_state=0),
                                         LinearSVC(max_iter=5000, dual="auto")),
        "RBFSampler+SGD":  make_pipeline(StandardScaler(),
                                         RBFSampler(gamma=.05, n_components=300,
                                                    random_state=0),
                                         SGDClassifier(loss="hinge", max_iter=1500,
                                                       random_state=0)),
    }
    for nm, mdl in models.items():
        t0 = time.perf_counter(); mdl.fit(Xtr, ytr); dt = time.perf_counter() - t0
        rows.append((nm, dt, mdl.score(Xte, yte)))
    return rows

for m in [1000, 4000, 12000]:
    print(f"\\n===== m = {m:,} =====")
    print(f"{'model':<18}{'fit time':>11}{'test acc':>11}")
    for nm, dt, acc in bench(m):
        print(f"{nm:<18}{dt:>10.3f}s{acc:>11.4f}")

print("\\nNystroem/RBFSampler approximate the RBF feature map explicitly with a")
print("few hundred columns, then let a LINEAR solver do the work. You keep most")
print("of the kernel's accuracy at O(m) cost instead of O(m^2..3).")

# ---- how good is the approximation? ------------------------------------
from sklearn.metrics.pairwise import rbf_kernel
rng = np.random.default_rng(0)
Xs = StandardScaler().fit_transform(rng.normal(0, 1, (300, 20)))
K_true = rbf_kernel(Xs, gamma=.05)
print(f"\\n{'n_components':>13}{'max |K_approx - K_true|':>26}")
for nc in [20, 50, 100, 200, 299]:
    Z = Nystroem(gamma=.05, n_components=nc, random_state=0).fit_transform(Xs)
    print(f"{nc:>13}{np.abs(Z @ Z.T - K_true).max():>26.5f}")
''',
        key="ch05_complexity",
    )

    keypoints([
        "<code>LinearSVC</code>: $\\mathcal{O}(mn)$, linear only, the default for "
        "large data.",
        "<code>SGDClassifier</code>: $\\mathcal{O}(mn)$, the only out-of-core "
        "option.",
        "<code>SVC</code>: $\\mathcal{O}(m^2n)$–$\\mathcal{O}(m^3n)$, supports "
        "kernels, unusable past ~$10^5$ rows.",
        "<code>Nystroem</code> / <code>RBFSampler</code> approximate a kernel "
        "explicitly, giving kernel accuracy at linear cost.",
        "All of them need scaled inputs.",
    ])


# ==========================================================================
def s_5_4():
    section("5.4", "SVM Regression")

    lead(
        "Invert the objective. Classification: the widest street with as few "
        "instances *inside* as possible. Regression: a street of fixed width "
        "$\\varepsilon$ with as many instances *inside* as possible."
    )

    math(r"""
    \text{minimise}\quad \tfrac12 \lVert\mathbf{w}\rVert^2
    \;+\; C\sum_{i=1}^{m}\bigl(\zeta^{(i)} + \hat\zeta^{(i)}\bigr)
    \quad\text{subject to}\quad
    \begin{cases}
      y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)} - b \le \varepsilon + \zeta^{(i)}\\
      \mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)} \le \varepsilon + \hat\zeta^{(i)}\\
      \zeta^{(i)},\, \hat\zeta^{(i)} \ge 0
    \end{cases}
    """)

    md("Equivalently, in loss form — this is the **$\\varepsilon$-insensitive "
       "loss**:")

    math(r"""
    L_\varepsilon\bigl(y, \hat y\bigr) \;=\;
    \max\Bigl(0,\; \bigl|y - \hat y\bigr| - \varepsilon\Bigr)
    """)

    idea(
        "A dead zone in the loss",
        "Any prediction within $\\varepsilon$ of the truth incurs <b>zero</b> loss "
        "and contributes <b>zero</b> gradient. Errors are only counted beyond the "
        "tube. This is what makes SVR sparse: instances strictly inside the tube "
        "are not support vectors and can be deleted without changing the model — "
        "the exact regression analogue of §5.1.",
    )

    table(
        ["Hyperparameter", "Meaning", "Effect of increasing it"],
        [["<b>$\\varepsilon$</b> (<code>epsilon</code>)", "Half the tube width",
          "A wider dead zone → fewer support vectors → a smoother, more "
          "regularised fit"],
         ["<b>$C$</b>", "Penalty for being outside the tube",
          "Less regularisation → the model tries harder to pull outliers inside "
          "→ overfitting"],
         ["<b>$\\gamma$</b> (RBF only)", "Kernel width",
          "Wigglier fit → overfitting"]],
    )

    anim_header("ε sweeping: the tube widens and support vectors vanish")

    from sklearn.svm import SVR

    rng = np.random.default_rng(5)
    Xr = np.sort(rng.uniform(-3, 3, 90)).reshape(-1, 1)
    yr = np.sin(1.5 * Xr[:, 0]) + .35 * Xr[:, 0] + rng.normal(0, .22, 90)
    gr = np.linspace(-3.2, 3.2, 300).reshape(-1, 1)

    eps_list = np.linspace(0.02, 1.2, 32)
    cache = []
    for e in eps_list:
        m = SVR(kernel="rbf", C=20, gamma=.9, epsilon=float(e)).fit(Xr, yr)
        cache.append((m.predict(gr), len(m.support_), m.support_))

    frames = []
    for i, e in enumerate(eps_list):
        pr, nsv, sup = cache[i]
        frames.append(go.Frame(name=f"{e:.2f}", data=[
            go.Scatter(x=np.r_[gr[:, 0], gr[::-1, 0]],
                       y=np.r_[pr - e, (pr + e)[::-1]], fill="toself",
                       fillcolor=alpha(C["primary"], .16), line=dict(width=0),
                       hoverinfo="skip"),
            go.Scatter(x=gr[:, 0], y=pr, mode="lines",
                       line=dict(color=C["primary"], width=3.6)),
            go.Scatter(x=Xr[:, 0], y=yr, mode="markers",
                       marker=dict(color=C["train"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xr[sup, 0], y=yr[sup], mode="markers",
                       marker=dict(color=C["danger"], size=12,
                                   symbol="circle-open", line=dict(width=2.5))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"ε = {e:.3f}   ·   {nsv} of {len(Xr)} instances are support vectors "
            f"({nsv/len(Xr):.0%})")])))

    pr0, nsv0, sup0 = cache[0]
    f = go.Figure(data=[
        go.Scatter(x=np.r_[gr[:, 0], gr[::-1, 0]],
                   y=np.r_[pr0 - eps_list[0], (pr0 + eps_list[0])[::-1]],
                   fill="toself", fillcolor=alpha(C["primary"], .16),
                   line=dict(width=0), name="ε-tube", hoverinfo="skip"),
        go.Scatter(x=gr[:, 0], y=pr0, mode="lines", name="SVR prediction",
                   line=dict(color=C["primary"], width=3.6)),
        go.Scatter(x=Xr[:, 0], y=yr, mode="markers", name="training data",
                   marker=dict(color=C["train"], size=7,
                               line=dict(color="#fff", width=.8))),
        go.Scatter(x=Xr[sup0, 0], y=yr[sup0], mode="markers",
                   name="support vectors",
                   marker=dict(color=C["danger"], size=12, symbol="circle-open",
                               line=dict(width=2.5))),
    ])
    f.update_layout(height=470, xaxis=dict(range=[-3.2, 3.2], title="x"),
                    yaxis=dict(range=[-2.4, 2.6], title="y"),
                    title="SVR: fit as many instances as possible INSIDE the tube",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(180), slider_prefix="ε = ")
    figure(f, "As ε grows the tube swallows more points, and the circled support "
              "vectors — the only ones that matter — disappear.")

    note(
        "ε-insensitive loss is a robust loss",
        "Compare the three: squared loss grows quadratically (outliers dominate); "
        "absolute loss grows linearly (robust, §2.1); $\\varepsilon$-insensitive "
        "loss is <b>flat then linear</b> (robust <i>and</i> sparse). Huber loss is "
        "the fourth member of this family: quadratic near zero, linear far away.",
    )

    code_lab(
        "SVR, the four regression losses, and sparsity",
        '''import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(5)
X = np.sort(rng.uniform(-3, 3, 200)).reshape(-1, 1)
y = np.sin(1.5*X[:, 0]) + .35*X[:, 0] + rng.normal(0, .22, 200)

print(f"{'epsilon':>9}{'C':>7}{'# SVs':>8}{'% of data':>11}{'CV RMSE':>11}")
for eps in [0.01, 0.1, 0.3, 0.6, 1.0]:
    for Cv in [1, 100]:
        m = SVR(kernel="rbf", C=Cv, gamma=.9, epsilon=eps).fit(X, y)
        cv = -cross_val_score(SVR(kernel="rbf", C=Cv, gamma=.9, epsilon=eps),
                              X, y, cv=5,
                              scoring="neg_root_mean_squared_error").mean()
        print(f"{eps:>9}{Cv:>7}{len(m.support_):>8}"
              f"{len(m.support_)/len(X):>10.1%}{cv:>11.4f}")

# ---- sparsity: delete the non-support vectors --------------------------
m = SVR(kernel="rbf", C=20, gamma=.9, epsilon=.3).fit(X, y)
sup = m.support_
m2 = SVR(kernel="rbf", C=20, gamma=.9, epsilon=.3).fit(X[sup], y[sup])
print(f"\\ntrained on all {len(X)}: first 5 predictions "
      f"{m.predict(X[:5]).round(6)}")
print(f"trained on {len(sup)} SVs  : first 5 predictions "
      f"{m2.predict(X[:5]).round(6)}")
print(f"max difference = {np.abs(m.predict(X)-m2.predict(X)).max():.2e}")

# ---- the four regression losses side by side --------------------------
r = np.linspace(-3, 3, 400)
eps, delta = .5, 1.0
losses = {
    "squared            (y-ŷ)²":      r**2,
    "absolute           |y-ŷ|":       np.abs(r),
    "epsilon-insensitive":            np.maximum(0, np.abs(r) - eps),
    "Huber (delta=1)":                np.where(np.abs(r) <= delta,
                                               .5*r**2,
                                               delta*(np.abs(r) - .5*delta)),
}
print(f"\\n{'residual':>10}" + "".join(f"{k.split()[0]:>22}" for k in losses))
for rv in [0.0, 0.3, 0.5, 1.0, 2.0, 3.0]:
    vals = [rv**2, abs(rv), max(0, abs(rv)-eps),
            .5*rv**2 if abs(rv) <= delta else delta*(abs(rv)-.5*delta)]
    print(f"{rv:>10.1f}" + "".join(f"{v:>22.4f}" for v in vals))
print("\\nNote the epsilon-insensitive column: EXACTLY zero until |r| > 0.5.")
print("Zero loss => zero gradient => that instance is not a support vector.")

import plotly.graph_objects as go
fig = go.Figure()
for i, (nm, v) in enumerate(losses.items()):
    fig.add_scatter(x=r, y=v, mode="lines", name=nm,
                    line=dict(color=SEQ[i], width=3))
fig.add_vrect(x0=-eps, x1=eps, fillcolor=C["success"], opacity=.12, line_width=0,
              annotation_text="the dead zone")
fig.update_layout(height=420, xaxis_title="residual y - ŷ", yaxis_title="loss",
                  yaxis=dict(range=[0, 4]),
                  title="Four regression losses")

# ---- LinearSVR for large data ------------------------------------------
print(f"\\nLinearSVR (no kernel, O(mn)):")
lin = make_pipeline(StandardScaler(), LinearSVR(epsilon=.3, C=10, max_iter=10000,
                                                dual="auto")).fit(X, y)
print(f"  RMSE = {mean_squared_error(y, lin.predict(X))**.5:.4f} "
      f"(a straight line, so worse here -- but it scales)")
''',
        key="ch05_svr",
    )

    keypoints([
        "SVR fits a tube of half-width $\\varepsilon$ and wants points "
        "<b>inside</b> it.",
        "The $\\varepsilon$-insensitive loss is flat inside the tube: zero loss, "
        "zero gradient, no support vector.",
        "Larger $\\varepsilon$ ⇒ fewer support vectors ⇒ smoother, more "
        "regularised.",
        "<code>LinearSVR</code> is $\\mathcal{O}(mn)$; <code>SVR</code> is "
        "$\\mathcal{O}(m^2n)$–$\\mathcal{O}(m^3n)$.",
        "Non-support-vectors can be deleted with no effect on predictions.",
    ])


# ==========================================================================
def s_5_5():
    section("5.5", "Under the Hood — The Primal Problem")

    lead(
        "Where the objective comes from. Once you see that maximising the margin "
        "is the same as minimising $\\lVert\\mathbf{w}\\rVert$, the whole "
        "formulation writes itself."
    )

    sub("The decision function")

    math(r"""
    \hat y \;=\;
    \begin{cases}
      0 & \text{if } \mathbf{w}^\top\mathbf{x} + b < 0\\
      1 & \text{if } \mathbf{w}^\top\mathbf{x} + b \ge 0
    \end{cases}
    """)

    sub("Margin width and ‖w‖ are the same knob")

    derive(
        [("Fix the boundary at $\\mathbf{w}^\\top\\mathbf{x} + b = 0$ and the street "
          "edges at $\\mathbf{w}^\\top\\mathbf{x} + b = \\pm 1$ (this <i>defines</i> "
          "the scale of $\\mathbf{w}$ — we are free to choose it).", None),
         ("Take a point $\\mathbf{x}_+$ on the positive edge and $\\mathbf{x}_-$ on "
          "the negative edge, with $\\mathbf{x}_+ = \\mathbf{x}_- + t\\,"
          "\\frac{\\mathbf{w}}{\\lVert\\mathbf{w}\\rVert}$ (moving perpendicular to "
          "the boundary by distance $t$ = the street width).", None),
         ("Substitute into the two edge equations and subtract:",
          r"\bigl(\mathbf{w}^\top\mathbf{x}_+ + b\bigr) - \bigl(\mathbf{w}^\top\mathbf{x}_- + b\bigr) "
          r"= 1 - (-1) = 2"),
         ("But the left side is also computable from the displacement:",
          r"\mathbf{w}^\top\bigl(\mathbf{x}_+ - \mathbf{x}_-\bigr) "
          r"= t\,\frac{\mathbf{w}^\top\mathbf{w}}{\lVert\mathbf{w}\rVert} "
          r"= t\,\lVert\mathbf{w}\rVert"),
         ("Equating the two expressions gives the street width exactly:",
          r"t = \frac{2}{\lVert\mathbf{w}\rVert}"),
         ("Therefore <b>maximising the margin ⟺ minimising "
          "$\\lVert\\mathbf{w}\\rVert$</b>. We minimise "
          "$\\frac12\\lVert\\mathbf{w}\\rVert^2$ instead, because it is "
          "differentiable everywhere (the norm is not, at zero) and has the same "
          "minimiser.", None)],
        title="Why the margin is 2/‖w‖",
    )

    sub("Hard margin primal")

    math(r"""
    \operatorname*{minimize}_{\mathbf{w},\,b}\quad
      \frac{1}{2}\,\mathbf{w}^\top\mathbf{w}
    \qquad\text{subject to}\qquad
      t^{(i)}\bigl(\mathbf{w}^\top\mathbf{x}^{(i)} + b\bigr) \;\ge\; 1
      \quad \text{for } i = 1,\dots,m
    """)
    where({r"t^{(i)}": "the $\\pm 1$ label: $-1$ for the negative class, $+1$ for "
                       "the positive class"})

    sub("Soft margin primal")

    math(r"""
    \operatorname*{minimize}_{\mathbf{w},\,b,\,\boldsymbol\zeta}\quad
      \frac{1}{2}\,\mathbf{w}^\top\mathbf{w}
      \;+\; C\sum_{i=1}^{m}\zeta^{(i)}
    \qquad\text{subject to}\qquad
    \begin{cases}
      t^{(i)}\bigl(\mathbf{w}^\top\mathbf{x}^{(i)} + b\bigr) \ge 1 - \zeta^{(i)}\\
      \zeta^{(i)} \ge 0
    \end{cases}
    """)
    where({r"\zeta^{(i)}": "the <b>slack variable</b> — how far instance $i$ is "
                           "allowed to violate the margin",
           r"C": "the price of one unit of slack"})

    sub("Equivalently: hinge loss plus ℓ₂ regularisation")

    derive(
        [("The constraints pin down $\\zeta^{(i)}$ exactly. Since we are minimising "
          "$C\\sum\\zeta^{(i)}$ with $\\zeta^{(i)} \\ge 0$, the optimum takes the "
          "smallest slack the constraint permits:",
          r"\zeta^{(i)} = \max\Bigl(0,\; 1 - t^{(i)}\bigl(\mathbf{w}^\top\mathbf{x}^{(i)} + b\bigr)\Bigr)"),
         ("That expression is exactly the <b>hinge loss</b>. Substituting it back "
          "eliminates the constraints and gives an unconstrained problem:",
          r"\operatorname*{minimize}_{\mathbf{w},b}\;\; \frac12\lVert\mathbf{w}\rVert^2 "
          r"+ C\sum_{i=1}^{m}\max\Bigl(0,\,1 - t^{(i)}\bigl(\mathbf{w}^\top\mathbf{x}^{(i)}+b\bigr)\Bigr)"),
         ("Divide through by $Cm$ to put it in the familiar "
          "<i>loss + $\\alpha\\cdot$penalty</i> form of Chapter 4:",
          r"\operatorname*{minimize}_{\mathbf{w},b}\;\; \frac1m\sum_{i=1}^{m}"
          r"\max\bigl(0,\,1-t^{(i)}s^{(i)}\bigr) "
          r"+ \underbrace{\frac{1}{2Cm}}_{\alpha}\lVert\mathbf{w}\rVert^2"),
         ("<b>So an SVM is exactly a linear model with hinge loss and $\\ell_2$ "
          "regularisation</b>, with $\\alpha = \\frac{1}{2Cm}$. That is why "
          "$C$ runs the opposite way to $\\alpha$, and why "
          "<code>SGDClassifier(loss='hinge')</code> is an SVM.", None)],
        title="From constrained QP to hinge loss — and why C = 1/(2αm)",
    )

    anim_header("Hinge loss vs the other classification losses")
    md(
        "Hinge is zero once the margin exceeds 1 — points comfortably on the "
        "correct side contribute nothing at all, which is precisely why the "
        "solution is sparse. Compare with log loss, which never reaches zero."
    )

    z = np.linspace(-3, 3, 400)
    curves = {
        "0/1 loss (what we want)": (z < 0).astype(float),
        "hinge — SVM": np.maximum(0, 1 - z),
        "squared hinge": np.maximum(0, 1 - z) ** 2,
        "log loss — logistic": np.log2(1 + np.exp(-z)),
        "exponential — AdaBoost (Ch. 7)": np.exp(-z),
    }
    frames = []
    for i, nm in enumerate(curves):
        data = []
        for j, (n2, v) in enumerate(curves.items()):
            visible_v = v if j <= i else np.full_like(v, np.nan)
            data.append(go.Scatter(x=z, y=visible_v, mode="lines",
                                   line=dict(color=SEQ[j],
                                             width=4 if j == i else 2.2,
                                             dash="dot" if j == 0 else "solid")))
        frames.append(go.Frame(name=nm.split()[0], data=data,
                               layout=go.Layout(title=f"adding: {nm}")))

    f = go.Figure(data=[go.Scatter(x=z, y=list(curves.values())[0], mode="lines",
                                   name=list(curves)[0],
                                   line=dict(color=SEQ[0], width=4, dash="dot"))]
                  + [go.Scatter(x=z, y=np.full_like(z, np.nan), mode="lines",
                                name=nm, line=dict(color=SEQ[j + 1], width=2.2))
                     for j, nm in enumerate(list(curves)[1:])])
    f.add_vline(x=1, line_dash="dash", line_color=C["muted"],
                annotation_text="margin = 1")
    f.add_vline(x=0, line_color=C["muted"])
    f.update_layout(height=450, xaxis_title="margin  t·s = t·(wᵀx + b)",
                    yaxis_title="loss", yaxis=dict(range=[0, 4]),
                    title="Classification losses as functions of the margin",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1200), slider_prefix="")
    figure(f, "Everything to the right of the dashed line costs the SVM nothing. "
              "Log loss keeps pushing forever, which is why logistic regression "
              "has no support vectors.")

    code_lab(
        "Implement a linear SVM from the primal, three ways",
        '''import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.7, random_state=7)
X = StandardScaler().fit_transform(X)
t = 2*y - 1                                     # labels in {-1, +1}
m, n = X.shape
C_val = 1.0

# ============ 1. subgradient descent on the hinge-loss objective =========
def hinge_objective(w, b):
    margins = t * (X @ w + b)
    return .5*w@w + C_val*np.maximum(0, 1 - margins).sum()

w, b = np.zeros(n), 0.0
for it in range(1, 6001):
    eta = 1.0 / (0.15 * it)                     # Robbins-Monro schedule
    margins = t * (X @ w + b)
    viol = margins < 1                          # only violators contribute
    grad_w = w - C_val * (t[viol, None] * X[viol]).sum(0)
    grad_b = -C_val * t[viol].sum()
    w -= eta * grad_w
    b -= eta * grad_b

print("=== from scratch (subgradient descent on the primal) ===")
print(f"w = {w.round(5)}   b = {b:.5f}")
print(f"objective = {hinge_objective(w, b):.5f}")
print(f"margin width 2/||w|| = {2/np.linalg.norm(w):.5f}")
print(f"training accuracy = {np.mean((X @ w + b >= 0) == y):.4f}")

# ============ 2. sklearn's SVC (dual solver) ============================
svc = SVC(kernel="linear", C=C_val).fit(X, y)
print(f"\\n=== SVC(kernel='linear') ===")
print(f"w = {svc.coef_[0].round(5)}   b = {svc.intercept_[0]:.5f}")
print(f"objective = {hinge_objective(svc.coef_[0], svc.intercept_[0]):.5f}")
print(f"margin width 2/||w|| = {2/np.linalg.norm(svc.coef_[0]):.5f}")
print(f"support vectors = {len(svc.support_)}")

# ============ 3. SGDClassifier with the equivalent alpha ================
# from the derivation:  alpha = 1 / (2 * C * m)   [sklearn uses alpha*||w||^2/2]
alpha_equiv = 1.0 / (C_val * m)
sgd = SGDClassifier(loss="hinge", alpha=alpha_equiv, max_iter=20000,
                    tol=1e-6, random_state=42).fit(X, y)
print(f"\\n=== SGDClassifier(loss='hinge', alpha=1/(C*m)={alpha_equiv:.2e}) ===")
print(f"w = {sgd.coef_[0].round(5)}   b = {sgd.intercept_[0]:.5f}")
print(f"cosine similarity with SVC's w = "
      f"{sgd.coef_[0]@svc.coef_[0]/(np.linalg.norm(sgd.coef_[0])*np.linalg.norm(svc.coef_[0])):.6f}")

# ============ 4. the slack variables ====================================
margins = t * (X @ svc.coef_[0] + svc.intercept_[0])
zeta = np.maximum(0, 1 - margins)
print(f"\\n=== slack variables at the optimum ===")
print(f"zeta == 0 (outside the street, correct)   : {int((zeta == 0).sum())}")
print(f"0 < zeta <= 1 (inside street, correct side): {int(((zeta > 0) & (zeta <= 1)).sum())}")
print(f"zeta > 1 (misclassified)                   : {int((zeta > 1).sum())}")
print(f"sum of slacks = {zeta.sum():.4f}")
print(f"objective = 0.5||w||^2 + C*sum(zeta) = "
      f"{.5*svc.coef_[0]@svc.coef_[0]:.4f} + {C_val}*{zeta.sum():.4f} = "
      f"{.5*svc.coef_[0]@svc.coef_[0] + C_val*zeta.sum():.4f}")

# ============ 5. C = 1/(2*alpha*m) verified empirically =================
print(f"\\n=== the C <-> alpha correspondence ===")
print(f"{'C':>8}{'alpha = 1/(C*m)':>18}{'||w|| (SVC)':>14}{'||w|| (SGD)':>14}")
for Cv in [0.01, 0.1, 1, 10, 100]:
    a = 1.0/(Cv*m)
    s1 = SVC(kernel="linear", C=Cv).fit(X, y)
    s2 = SGDClassifier(loss="hinge", alpha=a, max_iter=30000, tol=1e-7,
                       random_state=0).fit(X, y)
    print(f"{Cv:>8}{a:>18.3e}{np.linalg.norm(s1.coef_[0]):>14.4f}"
          f"{np.linalg.norm(s2.coef_[0]):>14.4f}")
print("Larger C -> larger ||w|| -> narrower street. Exactly as derived.")
''',
        key="ch05_primal",
    )

    keypoints([
        "Street width $= 2/\\lVert\\mathbf{w}\\rVert$, so maximising the margin "
        "= minimising $\\frac12\\lVert\\mathbf{w}\\rVert^2$.",
        "Hard margin: constraints $t^{(i)}(\\mathbf{w}^\\top\\mathbf{x}^{(i)}+b) "
        "\\ge 1$. Soft margin adds slacks $\\zeta^{(i)}$ priced at $C$.",
        "Eliminating the slacks gives <b>hinge loss + $\\ell_2$</b>, with "
        "$\\alpha = 1/(2Cm)$.",
        "Hinge loss is exactly zero beyond margin 1 — that is the source of "
        "sparsity.",
        "An SVM is a linear model; the kernel is what makes it non-linear.",
    ])


# ==========================================================================
def s_5_6():
    section("5.6", "The Dual Problem")

    lead(
        "Every constrained optimisation has a shadow problem. For the SVM the dual "
        "is not merely an alternative — it is the <i>only</i> formulation in which "
        "the kernel trick is expressible."
    )

    sub("The Lagrangian")

    md("Attach a multiplier $\\alpha^{(i)} \\ge 0$ to each constraint:")

    math(r"""
    \mathcal{L}\bigl(\mathbf{w}, b, \boldsymbol\alpha\bigr) \;=\;
    \frac{1}{2}\,\mathbf{w}^\top\mathbf{w}
    \;-\; \sum_{i=1}^{m}\alpha^{(i)}
      \Bigl[\, t^{(i)}\bigl(\mathbf{w}^\top\mathbf{x}^{(i)} + b\bigr) - 1 \,\Bigr]
    """)

    derive(
        [("At the optimum the partial derivatives with respect to the <i>primal</i> "
          "variables vanish. First with respect to $\\mathbf{w}$:",
          r"\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{w} - \sum_{i=1}^{m}\alpha^{(i)} "
          r"t^{(i)}\mathbf{x}^{(i)} = \mathbf{0} \;\;\Longrightarrow\;\; "
          r"\hat{\mathbf{w}} = \sum_{i=1}^{m}\hat\alpha^{(i)} t^{(i)}\mathbf{x}^{(i)}"),
         ("<b>This is already the key result:</b> the weight vector is a linear "
          "combination of the training instances, weighted by their multipliers. "
          "Instances with $\\alpha^{(i)} = 0$ contribute nothing — they are not "
          "support vectors.", None),
         ("Now with respect to $b$:",
          r"\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{m}\alpha^{(i)}t^{(i)} "
          r"= 0 \;\;\Longrightarrow\;\; \sum_{i=1}^{m}\alpha^{(i)}t^{(i)} = 0"),
         ("Substitute both back into $\\mathcal{L}$. The $\\frac12\\mathbf{w}^\\top"
          "\\mathbf{w}$ term becomes $\\frac12\\sum_i\\sum_j \\alpha^{(i)}\\alpha^{(j)}"
          "t^{(i)}t^{(j)}\\mathbf{x}^{(i)\\top}\\mathbf{x}^{(j)}$; the cross term "
          "becomes minus twice that; the $b$ term vanishes by the second "
          "condition. What survives is the <b>dual</b>:",
          r"\mathcal{L}_D(\boldsymbol\alpha) = \sum_{i=1}^{m}\alpha^{(i)} "
          r"- \frac12\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha^{(i)}\alpha^{(j)}\,"
          r"t^{(i)}t^{(j)}\,\mathbf{x}^{(i)\top}\mathbf{x}^{(j)}"),
         ("Maximise it subject to $\\alpha^{(i)} \\ge 0$ (and $\\alpha^{(i)} \\le C$ "
          "in the soft-margin case) and $\\sum_i \\alpha^{(i)}t^{(i)} = 0$.", None),
         ("Recover the bias from any support vector on the margin "
          "($0 < \\alpha^{(i)} < C$), averaging for numerical stability:",
          r"\hat b = \frac{1}{n_s}\sum_{i:\,0<\hat\alpha^{(i)}<C}\Bigl(t^{(i)} "
          r"- \hat{\mathbf{w}}^\top\mathbf{x}^{(i)}\Bigr)"),
         ("<b>Look at where the data appears in $\\mathcal{L}_D$: only inside "
          "$\\mathbf{x}^{(i)\\top}\\mathbf{x}^{(j)}$.</b> Replace that dot product "
          "with $K(\\mathbf{x}^{(i)},\\mathbf{x}^{(j)})$ and you have kernelised the "
          "SVM. Nothing else in the derivation changes. That is §5.7.", None)],
        title="Deriving the dual from the Lagrangian",
    )

    sub("The KKT conditions classify every instance")

    md(
        "The complementary-slackness condition $\\alpha^{(i)}\\bigl[t^{(i)}"
        "(\\mathbf{w}^\\top\\mathbf{x}^{(i)}+b) - 1 + \\zeta^{(i)}\\bigr] = 0$ "
        "partitions the training set into exactly three groups:"
    )

    table(
        ["$\\alpha^{(i)}$", "Margin $t^{(i)}s^{(i)}$", "Position", "Support vector?"],
        [["$\\alpha^{(i)} = 0$", "$> 1$", "Outside the street, correct side",
          "❌ — deleting it changes nothing"],
         ["$0 < \\alpha^{(i)} < C$", "$= 1$", "Exactly on the street edge",
          "✅ <b>margin support vector</b> — used to recover $b$"],
         ["$\\alpha^{(i)} = C$", "$< 1$", "Inside the street or misclassified",
          "✅ <b>bounded support vector</b> — a margin violator"]],
    )

    sub("Why the dual is worth having")

    table(
        ["", "Primal", "Dual"],
        [["Number of variables", "$n + 1$ (plus $m$ slacks)", "$m$"],
         ["Cheaper when", "$m \\gg n$ (many rows, few features)",
          "$n \\gg m$ (few rows, many features) — or $n = \\infty$"],
         ["Kernel trick", "❌ impossible", "<b>✅ the whole point</b>"],
         ["Solution structure", "$\\mathbf{w}$ directly",
          "$\\mathbf{w} = \\sum_i \\alpha^{(i)}t^{(i)}\\mathbf{x}^{(i)}$, sparse"],
         ["scikit-learn class", "<code>LinearSVC</code>", "<code>SVC</code>"]],
    )

    proof(
        "Strong duality holds here",
        "The SVM objective is convex and the constraints are affine, so Slater's "
        "condition is satisfied and the duality gap is <b>zero</b>: the dual "
        "optimum equals the primal optimum exactly. This is not generic — for a "
        "non-convex problem the dual only gives a lower bound. It is convexity "
        "that makes the SVM's dual an exact reformulation rather than an "
        "approximation.",
    )

    anim_header("The dual variables αᵢ converging")
    md(
        "Coordinate ascent on the dual (a simplified SMO). Each frame updates one "
        "$\\alpha^{(i)}$; the bars show the multipliers and the panel shows the "
        "boundary they imply. Watch almost all of them settle at exactly zero."
    )

    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(11)
    Xd, yd = make_blobs(n_samples=40, centers=2, cluster_std=1.6, random_state=4)
    Xd = StandardScaler().fit_transform(Xd)
    td = 2 * yd - 1
    Kd = Xd @ Xd.T
    Cd = 1.0

    alph = np.zeros(len(Xd))
    snaps = []
    for it in range(280):
        i = it % len(Xd)
        j = (it * 7 + 3) % len(Xd)
        if i == j:
            continue
        Ei = float((alph * td) @ Kd[i] - td[i])
        Ej = float((alph * td) @ Kd[j] - td[j])
        eta = Kd[i, i] + Kd[j, j] - 2 * Kd[i, j]
        if eta <= 1e-9:
            continue
        aj_new = alph[j] + td[j] * (Ei - Ej) / eta
        if td[i] != td[j]:
            L, H = max(0, alph[j] - alph[i]), min(Cd, Cd + alph[j] - alph[i])
        else:
            L, H = max(0, alph[i] + alph[j] - Cd), min(Cd, alph[i] + alph[j])
        aj_new = float(np.clip(aj_new, L, H))
        ai_new = alph[i] + td[i] * td[j] * (alph[j] - aj_new)
        alph[i], alph[j] = max(0.0, ai_new), aj_new
        if it % 8 == 0:
            snaps.append(alph.copy())

    gd1 = np.linspace(Xd[:, 0].min() - .8, Xd[:, 0].max() + .8, 90)
    gd2 = np.linspace(Xd[:, 1].min() - .8, Xd[:, 1].max() + .8, 90)
    GD1, GD2 = np.meshgrid(gd1, gd2); GDD = np.c_[GD1.ravel(), GD2.ravel()]

    frames = []
    for s, a in enumerate(snaps):
        w = (a * td) @ Xd
        sv = (a > 1e-6)
        b = (np.mean(td[sv] - Xd[sv] @ w) if sv.any() else 0.0)
        Z = (GDD @ w + b).reshape(GD1.shape)
        frames.append(go.Frame(name=str(s * 8), data=[
            go.Contour(x=gd1, y=gd2, z=Z, showscale=False,
                       contours=dict(start=0, end=0, size=1, coloring="lines"),
                       line=dict(width=3.5),
                       colorscale=[[0, C["primary"]], [1, C["primary"]]]),
            go.Scatter(x=Xd[yd == 0, 0], y=Xd[yd == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=9,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=Xd[yd == 1, 0], y=Xd[yd == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=9,
                                   line=dict(color="#fff", width=.9))),
            go.Scatter(x=Xd[sv, 0], y=Xd[sv, 1], mode="markers",
                       marker=dict(color=C["danger"], size=16, symbol="circle-open",
                                   line=dict(width=3))),
            go.Bar(x=list(range(len(a))), y=a,
                   marker=dict(color=[C["danger"] if v > 1e-6 else C["muted"]
                                      for v in a])),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"iteration {s*8}   ·   {int(sv.sum())} of {len(Xd)} α's are non-zero"
            f"   ·   Σαᵢtᵢ = {float((a*td).sum()):+.2e}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.56, .44],
                      subplot_titles=("boundary implied by the current α",
                                      "the dual variables αᵢ"))
    a0 = snaps[0]
    w0 = (a0 * td) @ Xd
    Z0 = (GDD @ w0).reshape(GD1.shape)
    f.add_trace(go.Contour(x=gd1, y=gd2, z=Z0, showscale=False,
                           contours=dict(start=0, end=0, size=1, coloring="lines"),
                           line=dict(width=3.5),
                           colorscale=[[0, C["primary"]], [1, C["primary"]]]), 1, 1)
    f.add_trace(go.Scatter(x=Xd[yd == 0, 0], y=Xd[yd == 0, 1], mode="markers",
                           name="class 0", marker=dict(color=C["train"], size=9,
                           line=dict(color="#fff", width=.9))), 1, 1)
    f.add_trace(go.Scatter(x=Xd[yd == 1, 0], y=Xd[yd == 1, 1], mode="markers",
                           name="class 1", marker=dict(color=C["warning"], size=9,
                           line=dict(color="#fff", width=.9))), 1, 1)
    f.add_trace(go.Scatter(x=[], y=[], mode="markers", name="support vectors",
                           marker=dict(color=C["danger"], size=16,
                                       symbol="circle-open", line=dict(width=3))), 1, 1)
    f.add_trace(go.Bar(x=list(range(len(a0))), y=a0, showlegend=False,
                       marker=dict(color=C["muted"])), 1, 2)
    f.update_xaxes(title_text="instance index", row=1, col=2)
    f.update_yaxes(title_text="αᵢ", range=[0, Cd * 1.15], row=1, col=2)
    f.update_layout(height=480, title="Coordinate ascent on the dual (simplified SMO)")
    anim.animate(f, frames, duration=nav.anim_ms(130), slider_prefix="iter ")
    figure(f)

    code_lab(
        "Solve the dual by hand and match scikit-learn exactly",
        '''import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.5, random_state=3)
X = StandardScaler().fit_transform(X)
t = 2*y - 1
m = len(X)
C_val = 1.0
K = X @ X.T                       # <-- the ONLY place the data appears

# ============ simplified SMO on the dual ================================
def dual_objective(a):
    return float(a.sum() - .5 * ((a*t) @ K @ (a*t)))

alpha = np.zeros(m)
rs = np.random.default_rng(0)
for it in range(60_000):
    i, j = rs.integers(m), rs.integers(m)
    if i == j: continue
    Ei = float((alpha*t) @ K[i] - t[i])
    Ej = float((alpha*t) @ K[j] - t[j])
    eta = K[i,i] + K[j,j] - 2*K[i,j]
    if eta <= 1e-12: continue
    aj = alpha[j] + t[j]*(Ei - Ej)/eta
    if t[i] != t[j]:
        L, H = max(0, alpha[j]-alpha[i]), min(C_val, C_val+alpha[j]-alpha[i])
    else:
        L, H = max(0, alpha[i]+alpha[j]-C_val), min(C_val, alpha[i]+alpha[j])
    aj = np.clip(aj, L, H)
    ai = alpha[i] + t[i]*t[j]*(alpha[j] - aj)
    alpha[i], alpha[j] = max(0.0, ai), aj

# ---- recover the primal from the dual ---------------------------------
w = (alpha * t) @ X                                  # w = sum a_i t_i x_i
on_margin = (alpha > 1e-6) & (alpha < C_val - 1e-6)
b = float(np.mean(t[on_margin] - X[on_margin] @ w))

print("=== from the dual, by hand ===")
print(f"w = {w.round(5)}   b = {b:.5f}")
print(f"dual objective       = {dual_objective(alpha):.6f}")
print(f"primal 0.5||w||^2    = {.5*w@w:.6f}")
print(f"sum(alpha_i t_i)     = {float((alpha*t).sum()):+.3e}  (must be 0)")
print(f"non-zero alphas      = {int((alpha > 1e-6).sum())} of {m}")

svc = SVC(kernel="linear", C=C_val).fit(X, y)
print("\\n=== sklearn SVC ===")
print(f"w = {svc.coef_[0].round(5)}   b = {svc.intercept_[0]:.5f}")
print(f"support vectors = {len(svc.support_)}")
print(f"\\ncosine(w_mine, w_sklearn) = "
      f"{w@svc.coef_[0]/(np.linalg.norm(w)*np.linalg.norm(svc.coef_[0])):.8f}")

# ---- sklearn exposes alpha_i * t_i in dual_coef_ ----------------------
a_sk = np.zeros(m)
a_sk[svc.support_] = np.abs(svc.dual_coef_[0])
print(f"\\nmax |alpha_mine - alpha_sklearn| = {np.abs(alpha - a_sk).max():.5f}")

# ============ the KKT conditions classify every instance ================
margins = t * (X @ w + b)
print(f"\\n=== KKT partition (C = {C_val}) ===")
print(f"{'group':<42}{'count':>7}")
g1 = (alpha < 1e-6)
g2 = (alpha > 1e-6) & (alpha < C_val - 1e-6)
g3 = (alpha > C_val - 1e-6)
print(f"{'alpha=0     margin>1   outside the street':<42}{int(g1.sum()):>7}")
print(f"{'0<alpha<C   margin=1   ON the street edge':<42}{int(g2.sum()):>7}")
print(f"{'alpha=C     margin<1   inside/violating':<42}{int(g3.sum()):>7}")
if g2.any():
    print(f"\\nmargins of the on-edge instances: {margins[g2].round(4)}   "
          f"(should all be 1.0)")

# ============ w = sum alpha_i t_i x_i, verified =========================
w_rebuild = np.zeros(2)
for i in np.where(alpha > 1e-6)[0]:
    w_rebuild += alpha[i] * t[i] * X[i]
print(f"\\nw rebuilt from only the {int((alpha>1e-6).sum())} support vectors = "
      f"{w_rebuild.round(6)}")
print(f"identical to {np.abs(w - w_rebuild).max():.2e}")
''',
        key="ch05_dual",
    )

    keypoints([
        "$\\hat{\\mathbf{w}} = \\sum_i \\hat\\alpha^{(i)} t^{(i)}\\mathbf{x}^{(i)}$ — "
        "the weights are a sparse combination of training instances.",
        "$\\sum_i \\alpha^{(i)}t^{(i)} = 0$ is the second stationarity condition.",
        "KKT partitions the data: $\\alpha = 0$ (irrelevant), $0<\\alpha<C$ (on the "
        "edge), $\\alpha = C$ (violator).",
        "Strong duality holds (convex + affine constraints), so the dual is exact.",
        "<b>The data enters only as $\\mathbf{x}^{(i)\\top}\\mathbf{x}^{(j)}$</b> — "
        "which is what makes the next section possible.",
    ])


# ==========================================================================
def s_5_7():
    section("5.7", "Kernelized SVMs")

    lead(
        "Swap the dot product for a kernel. That is the entire modification — and "
        "it buys you an infinite-dimensional feature space for the price of a "
        "$m\\times m$ matrix."
    )

    sub("The kernelised dual")

    math(r"""
    \operatorname*{maximize}_{\boldsymbol\alpha}\quad
      \sum_{i=1}^{m}\alpha^{(i)}
      \;-\; \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}
        \alpha^{(i)}\alpha^{(j)}\,t^{(i)}t^{(j)}\,
        K\bigl(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\bigr)
    """)
    math(r"""
    \text{subject to}\qquad
    0 \le \alpha^{(i)} \le C
    \qquad\text{and}\qquad
    \sum_{i=1}^{m}\alpha^{(i)}t^{(i)} = 0
    """)

    sub("Making predictions without ever computing w")

    md(
        "In an infinite-dimensional space you cannot store $\\hat{\\mathbf{w}}$. "
        "But you never need to — only the *decision function* is required, and it "
        "is again pure dot products:"
    )

    derive(
        [("Start from $\\hat{\\mathbf{w}} = \\sum_i \\hat\\alpha^{(i)}t^{(i)}"
          "\\phi(\\mathbf{x}^{(i)})$ and substitute into the decision function for a "
          "new instance $\\mathbf{x}_{\\text{new}}$:",
          r"h(\mathbf{x}_{\text{new}}) = \hat{\mathbf{w}}^\top\phi(\mathbf{x}_{\text{new}}) + \hat b "
          r"= \left(\sum_{i=1}^{m}\hat\alpha^{(i)}t^{(i)}\phi\bigl(\mathbf{x}^{(i)}\bigr)\right)^{\!\top}"
          r"\!\!\phi\bigl(\mathbf{x}_{\text{new}}\bigr) + \hat b"),
         ("Move the sum outside — each term is now a dot product of two "
          "$\\phi$ images:",
          r"= \sum_{i=1}^{m}\hat\alpha^{(i)}t^{(i)}\,"
          r"\phi\bigl(\mathbf{x}^{(i)}\bigr)^\top\phi\bigl(\mathbf{x}_{\text{new}}\bigr) + \hat b"),
         ("Replace each dot product by the kernel, and drop every term with "
          "$\\hat\\alpha^{(i)} = 0$:",
          r"\boxed{\;h(\mathbf{x}_{\text{new}}) = "
          r"\sum_{i \,\in\, \text{SV}} \hat\alpha^{(i)}\,t^{(i)}\,"
          r"K\bigl(\mathbf{x}^{(i)}, \mathbf{x}_{\text{new}}\bigr) + \hat b \;}"),
         ("The bias is recovered the same way, averaged over the margin support "
          "vectors:",
          r"\hat b = \frac{1}{n_s}\sum_{i \in \text{SV},\,\hat\alpha^{(i)}<C}\!\!"
          r"\left(t^{(i)} - \sum_{j \in \text{SV}} \hat\alpha^{(j)}t^{(j)}"
          r"K\bigl(\mathbf{x}^{(j)}, \mathbf{x}^{(i)}\bigr)\right)"),
         ("<b>Prediction cost is $\\mathcal{O}(n_{\\text{SV}} \\cdot n)$</b> — "
          "proportional to the number of support vectors, not the training-set "
          "size. This is why sparsity matters practically, not just aesthetically.",
          None)],
        title="Predicting in an infinite-dimensional space",
    )

    sub("Common kernels")

    table(
        ["Kernel", "$K(\\mathbf{a}, \\mathbf{b})$", "Feature space",
         "Hyperparameters"],
        [["Linear", "$\\mathbf{a}^\\top\\mathbf{b}$", "The input space itself", "—"],
         ["Polynomial", "$(\\gamma\\,\\mathbf{a}^\\top\\mathbf{b} + r)^d$",
          "$\\binom{n+d}{d}$-dimensional", "$d$, $\\gamma$, $r$"],
         ["Gaussian RBF",
          "$\\exp\\bigl(-\\gamma\\lVert\\mathbf{a}-\\mathbf{b}\\rVert^2\\bigr)$",
          "<b>Infinite</b>-dimensional", "$\\gamma$"],
         ["Sigmoid",
          "$\\tanh\\bigl(\\gamma\\,\\mathbf{a}^\\top\\mathbf{b} + r\\bigr)$",
          "Not a valid Mercer kernel for all parameters", "$\\gamma$, $r$"],
         ["Laplacian",
          "$\\exp\\bigl(-\\gamma\\lVert\\mathbf{a}-\\mathbf{b}\\rVert_1\\bigr)$",
          "Infinite; less smooth than RBF", "$\\gamma$"]],
    )

    proof(
        "Mercer's condition",
        "A function $K$ is a valid kernel — meaning some $\\phi$ exists with "
        "$K(\\mathbf{a},\\mathbf{b}) = \\phi(\\mathbf{a})^\\top\\phi(\\mathbf{b})$ — "
        "if it is continuous, symmetric, and <b>positive semi-definite</b>: for any "
        "finite set of points the Gram matrix $\\mathbf{K}$ with "
        "$K_{ij} = K(\\mathbf{x}^{(i)},\\mathbf{x}^{(j)})$ has all eigenvalues "
        "$\\ge 0$. This is checkable numerically, and the lab below does it. "
        "The sigmoid kernel famously fails for some $(\\gamma, r)$, which is why "
        "<code>libsvm</code> can produce odd results with it.",
    )

    anim_header("The same data, five kernels")

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    Xk, yk = ds.moons(n=200, noise=.25)
    Xk = StandardScaler().fit_transform(Xk)
    k1 = np.linspace(Xk[:, 0].min() - .7, Xk[:, 0].max() + .7, 120)
    k2 = np.linspace(Xk[:, 1].min() - .7, Xk[:, 1].max() + .7, 120)
    K1, K2 = np.meshgrid(k1, k2); KK = np.c_[K1.ravel(), K2.ravel()]

    kernels = [("linear", dict(kernel="linear", C=1)),
               ("polynomial d=3", dict(kernel="poly", degree=3, coef0=1, C=5)),
               ("polynomial d=10", dict(kernel="poly", degree=10, coef0=1, C=5)),
               ("RBF γ=0.5", dict(kernel="rbf", gamma=.5, C=5)),
               ("RBF γ=8", dict(kernel="rbf", gamma=8, C=5)),
               ("sigmoid", dict(kernel="sigmoid", gamma=.5, coef0=-1, C=5))]
    kcache = []
    for nm, kw in kernels:
        m = SVC(**kw).fit(Xk, yk)
        kcache.append((m.decision_function(KK).reshape(K1.shape),
                       len(m.support_), float(m.score(Xk, yk))))

    frames = []
    for i, (nm, _) in enumerate(kernels):
        Z, nsv, acc = kcache[i]
        frames.append(go.Frame(name=nm, data=[
            go.Contour(x=k1, y=k2, z=Z, showscale=False, colorscale=nav.cscale(),
                       opacity=.5, ncontours=22),
            go.Contour(x=k1, y=k2, z=Z, showscale=False,
                       contours=dict(start=0, end=0, size=1, coloring="lines"),
                       line=dict(width=3.5),
                       colorscale=[[0, C["ink"]], [1, C["ink"]]]),
            go.Scatter(x=Xk[yk == 0, 0], y=Xk[yk == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xk[yk == 1, 0], y=Xk[yk == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=7,
                                   line=dict(color="#fff", width=.8))),
        ], layout=go.Layout(title=f"{nm}   ·   {nsv} support vectors   ·   "
                                  f"train accuracy {acc:.3f}")))

    Z0 = kcache[0][0]
    f = go.Figure(data=[
        go.Contour(x=k1, y=k2, z=Z0, showscale=False, colorscale=nav.cscale(),
                   opacity=.5, ncontours=22),
        go.Contour(x=k1, y=k2, z=Z0, showscale=False,
                   contours=dict(start=0, end=0, size=1, coloring="lines"),
                   line=dict(width=3.5), colorscale=[[0, C["ink"]], [1, C["ink"]]]),
        go.Scatter(x=Xk[yk == 0, 0], y=Xk[yk == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=7,
                                               line=dict(color="#fff", width=.8))),
        go.Scatter(x=Xk[yk == 1, 0], y=Xk[yk == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=7,
                                               line=dict(color="#fff", width=.8))),
    ])
    f.update_layout(height=520, title=kernels[0][0],
                    xaxis_title="x₁", yaxis_title="x₂")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="kernel ")
    figure(f)

    code_lab(
        "Kernelised prediction from scratch; Mercer check; custom kernels",
        '''import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel

X, y = make_moons(n_samples=150, noise=.22, random_state=0)
X = StandardScaler().fit_transform(X)
t = 2*y - 1
gamma, C_val = 1.0, 5.0

svc = SVC(kernel="rbf", gamma=gamma, C=C_val).fit(X, y)
sv      = X[svc.support_]                  # the support vectors
sv_t    = t[svc.support_]
a_t     = svc.dual_coef_[0]                # sklearn stores alpha_i * t_i
b       = svc.intercept_[0]
print(f"{len(sv)} support vectors of {len(X)} instances")

# ============ predict by hand: h(x) = sum a_i t_i K(x_i, x) + b =========
def predict_raw(Xnew):
    K = rbf_kernel(Xnew, sv, gamma=gamma)          # (n_new, n_SV)
    return K @ a_t + b

mine = predict_raw(X)
theirs = svc.decision_function(X)
print(f"\\nmax |my decision function - sklearn's| = {np.abs(mine-theirs).max():.3e}")
print(f"predictions identical: {np.array_equal(mine >= 0, svc.predict(X) == 1)}")

# ---- prediction cost is O(n_SV), not O(m) ------------------------------
print(f"\\nprediction touches {len(sv)} stored vectors, NOT all {len(X)}.")
print(f"the model stores {sv.nbytes/1024:.1f} KB instead of {X.nbytes/1024:.1f} KB")

# ============ Mercer check: is the Gram matrix PSD? ====================
print(f"\\n=== Mercer's condition (min eigenvalue of the Gram matrix) ===")
Xs = X[:120]
tests = {
    "linear":            Xs @ Xs.T,
    "poly d=3, r=1":     polynomial_kernel(Xs, degree=3, coef0=1, gamma=1),
    "RBF gamma=1":       rbf_kernel(Xs, gamma=1.),
    "sigmoid g=.5 r=-1": sigmoid_kernel(Xs, gamma=.5, coef0=-1),
    "sigmoid g=2 r=1":   sigmoid_kernel(Xs, gamma=2., coef0=1),
}
for nm, K in tests.items():
    ev = np.linalg.eigvalsh((K + K.T)/2)
    ok = "VALID" if ev.min() > -1e-8 else "NOT a Mercer kernel"
    print(f"  {nm:<20} min eigenvalue = {ev.min():>12.3e}   {ok}")

# ============ a custom kernel ===========================================
def laplacian(A, B, gamma=1.0):
    """exp(-gamma * ||a - b||_1) -- valid, and less smooth than RBF."""
    D = np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    return np.exp(-gamma * D)

lap = SVC(kernel=lambda A, B: laplacian(A, B, .8), C=5).fit(X, y)
print(f"\\ncustom Laplacian kernel: accuracy {lap.score(X, y):.4f}, "
      f"{len(lap.support_)} support vectors")

# passing a precomputed Gram matrix works too
G = laplacian(X, X, .8)
pre = SVC(kernel="precomputed", C=5).fit(G, y)
print(f"precomputed Gram matrix: accuracy {pre.score(G, y):.4f}  <- same model")

# ============ kernels can be combined ==================================
print("\\n=== the algebra of kernels ===")
print("if K1 and K2 are valid kernels, so are:")
print("   K1 + K2        (sum)")
print("   c * K1         (positive scaling)")
print("   K1 * K2        (product)")
print("   f(x)f(z)K1     (any real function f)")
K_sum = rbf_kernel(Xs, gamma=.5) + polynomial_kernel(Xs, degree=2, coef0=1)
print(f"\\nRBF + poly: min eigenvalue = "
      f"{np.linalg.eigvalsh((K_sum+K_sum.T)/2).min():.3e}  -> still valid")
''',
        key="ch05_kernelized",
    )

    keypoints([
        "The kernelised dual is the plain dual with "
        "$\\mathbf{x}^{(i)\\top}\\mathbf{x}^{(j)} \\to K(\\mathbf{x}^{(i)},"
        "\\mathbf{x}^{(j)})$.",
        "Prediction: $h(\\mathbf{x}) = \\sum_{i\\in\\text{SV}} \\alpha^{(i)}t^{(i)}"
        "K(\\mathbf{x}^{(i)},\\mathbf{x}) + b$ — $\\mathbf{w}$ is never formed.",
        "Cost is $\\mathcal{O}(n_{\\text{SV}})$ per prediction, so sparsity is a "
        "practical necessity.",
        "Mercer: continuous, symmetric, positive semi-definite ⇒ a valid kernel.",
        "Kernels are closed under sums, positive scaling and products — you can "
        "build your own.",
    ])


# ==========================================================================
def s_5_8():
    section("5.8", "Exercises & Chapter Review")

    lead("Ten exercises. Numbers 6–10 are the ones that reward paper.")

    exercise(
        1, "What is the fundamental idea behind support vector machines?",
        "Fit the **widest possible street** between the two classes — that is, "
        "find the decision boundary that maximises the margin to the nearest "
        "training instances. When soft margin is used, the SVM balances a wide "
        "street against a small number of margin violations, with $C$ setting the "
        "exchange rate. Only the instances on or inside the street (the **support "
        "vectors**) determine the boundary; the rest are irrelevant.")

    exercise(
        2, "What is a support vector?",
        "Any instance that lies **on the edge of the street or inside it** — "
        "equivalently, any instance whose dual multiplier $\\alpha^{(i)}$ is "
        "non-zero. The decision boundary is entirely determined by them: "
        "$\\hat{\\mathbf{w}} = \\sum_i \\hat\\alpha^{(i)}t^{(i)}\\mathbf{x}^{(i)}$, and "
        "predictions for new instances involve only the support vectors. Deleting "
        "any non-support-vector leaves the model bit-for-bit unchanged.")

    exercise(
        3, "Why is it important to scale the inputs when using SVMs?",
        "Because SVMs fit the widest *Euclidean* street, so the geometry of the "
        "feature space is the objective. If one feature has a much larger scale "
        "than another, the SVM will effectively ignore the small-scale feature — "
        "the margin is dominated by the large-scale direction, and the boundary "
        "comes out nearly perpendicular to it. Scaling makes each feature "
        "contribute comparably. The same argument applies with even more force to "
        "the RBF kernel, whose $\\lVert\\mathbf{a}-\\mathbf{b}\\rVert^2$ is a raw "
        "Euclidean distance.")

    exercise(
        4, "Can an SVM classifier output a confidence score when it classifies an "
        "instance? What about a probability?",
        "**A confidence score: yes**, always — that is `decision_function`, the "
        "signed distance to the boundary (scaled so the street edges are at "
        "$\\pm 1$). It can be thresholded exactly as in §3.4, and it is a perfectly "
        "good ranking score for ROC/PR curves.\n\n"
        "**A probability: not directly.** SVMs do not model $P(y \\mid "
        "\\mathbf{x})$. scikit-learn can produce one with "
        "`SVC(probability=True)`, which fits **Platt scaling** — a logistic "
        "regression $P = \\sigma(A \\cdot s(\\mathbf{x}) + B)$ on the decision "
        "values, calibrated by internal 5-fold cross-validation. This makes "
        "training substantially slower, and the resulting probabilities may "
        "occasionally disagree with `predict` (they use different thresholds). "
        "Prefer `CalibratedClassifierCV` if you need well-behaved probabilities.")

    exercise(
        5, "How can you choose between `LinearSVC`, `SVC`, and `SGDClassifier`?",
        "By the size of the training set and whether you need a kernel:\n\n"
        "* **`LinearSVC`** — a linear SVM, $\\mathcal{O}(mn)$, the default choice "
        "for large training sets that are (approximately) linearly separable.\n"
        "* **`SVC`** — the only one supporting the kernel trick, but "
        "$\\mathcal{O}(m^2 n)$ to $\\mathcal{O}(m^3 n)$, so it is limited to small "
        "and medium sets (up to roughly $10^4$–$10^5$ instances).\n"
        "* **`SGDClassifier(loss='hinge')`** — the same linear objective solved by "
        "SGD; slower to converge than `LinearSVC` but it is the only one that "
        "supports **out-of-core** learning via `partial_fit`.\n\n"
        "All three need scaled inputs.")

    exercise(
        6, "Say you have trained an SVM classifier with an RBF kernel, but it "
        "seems to underfit the training set. Should you increase or decrease γ? "
        "What about C?",
        "**Increase both.** Underfitting means the model is too constrained.\n\n"
        "* $\\gamma$ is the inverse width of the RBF bell. A *small* $\\gamma$ "
        "means each instance influences a wide region, so the boundary is smooth "
        "— too smooth, here. Increasing $\\gamma$ narrows the bells and lets the "
        "boundary become more irregular.\n"
        "* $C$ is the price of a margin violation. A *small* $C$ buys a wide "
        "street at the cost of many violations, i.e. more regularisation. "
        "Increasing $C$ makes the model fit the training data harder.\n\n"
        "Both are regularisation dials pointing the same way, so tune them "
        "together on a 2-D grid.")

    exercise(
        7, "What does it mean for a model to be ε-insensitive?",
        "That errors smaller than $\\varepsilon$ **do not affect the model at "
        "all**. The loss $\\max(0, |y - \\hat y| - \\varepsilon)$ is exactly zero "
        "inside the tube, so those instances contribute no gradient and are not "
        "support vectors. This is what makes SVR both **robust** (a small "
        "perturbation of a point inside the tube changes nothing) and **sparse** "
        "(the fitted model depends on only the points on or outside the tube).")

    exercise(
        8, "What is the point of using the kernel trick?",
        "It lets you get **exactly the result** of adding many polynomial (or "
        "infinitely many RBF) features, **without actually adding them**. The dual "
        "problem and the decision function contain the data only through dot "
        "products $\\mathbf{x}^{(i)\\top}\\mathbf{x}^{(j)}$; replacing each with "
        "$K(\\mathbf{x}^{(i)},\\mathbf{x}^{(j)}) = \\phi(\\mathbf{x}^{(i)})^\\top"
        "\\phi(\\mathbf{x}^{(j)})$ computes the inner product in the transformed "
        "space at the cost of the original space. For the RBF kernel $\\phi$ maps "
        "into an infinite-dimensional space, which could never be constructed "
        "explicitly.")

    exercise(
        9, "Train a `LinearSVC` on a linearly separable dataset. Then train an "
        "`SVC` and a `SGDClassifier` on the same dataset. See if you can get them "
        "to produce roughly the same model.",
        "The three have different defaults, and matching them is the exercise:\n\n"
        "* `LinearSVC` defaults to `loss='squared_hinge'` — set `loss='hinge'`.\n"
        "* `LinearSVC` **regularises the bias term**, so centre the data (a "
        "`StandardScaler` does this) or the intercepts will not match.\n"
        "* The correspondence between the regularisation knobs is "
        "$\\alpha = \\frac{1}{C \\cdot m}$, derived in §5.5.\n\n"
        "With those three adjustments the coefficient vectors line up to about "
        "three decimals.",
        code='''from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

C_val, m = 5.0, len(X_scaled)
lin = LinearSVC(loss="hinge", C=C_val, dual="auto", max_iter=100_000)
svc = SVC(kernel="linear", C=C_val)
sgd = SGDClassifier(loss="hinge", alpha=1/(C_val*m), max_iter=100_000, tol=1e-7)

for m_ in (lin, svc, sgd):
    m_.fit(X_scaled, y)
    print(f"{m_.__class__.__name__:<18} w={m_.coef_[0].round(4)} "
          f"b={m_.intercept_[0]:.4f}")''')

    exercise(
        10, "Train an SVM classifier on the wine dataset, and an SVM regressor on "
        "the California housing dataset.",
        "**Classifier (wine, 3 classes, 13 features, 178 rows).** `SVC` handles "
        "multiclass automatically with OvO (§3.6). Scaling is essential — the 13 "
        "features have wildly different units (proline runs into the hundreds, "
        "hue is around 1). A scaled RBF SVM with tuned $C$ and $\\gamma$ typically "
        "reaches 97–99 % cross-validated accuracy.\n\n"
        "**Regressor (California housing, ~20 000 rows).** Here `SVR` is "
        "genuinely slow — $\\mathcal{O}(m^2)$ on 20 000 rows. Use "
        "`RandomizedSearchCV` with `loguniform` distributions for $C$ and "
        "$\\gamma$ (they matter multiplicatively), and expect the random forest of "
        "Chapter 2 to still win. That is a useful result, not a failure: SVR is "
        "the wrong tool at this data size.",
        code='''from sklearn.datasets import load_wine
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import loguniform

wine = load_wine()
grid = RandomizedSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": loguniform(0.1, 1000), "svc__gamma": loguniform(1e-4, 1)},
    n_iter=60, cv=5, random_state=42, n_jobs=-1)
grid.fit(wine.data, wine.target)
print(grid.best_params_, grid.best_score_)''')

    rule()

    sub("The whole chapter as one diagram")

    table(
        ["Step", "Result", "Section"],
        [["Maximise the margin", "street width $= 2/\\lVert\\mathbf{w}\\rVert$",
          "§5.5"],
         ["⟹ minimise $\\frac12\\lVert\\mathbf{w}\\rVert^2$ s.t. margin constraints",
          "the hard-margin primal", "§5.5"],
         ["Add slacks $\\zeta^{(i)}$ priced at $C$", "the soft-margin primal", "§5.1"],
         ["Eliminate the slacks", "hinge loss + $\\ell_2$, $\\alpha = 1/(2Cm)$", "§5.5"],
         ["Form the Lagrangian, set $\\nabla_{\\mathbf{w}}\\mathcal{L} = 0$",
          "$\\mathbf{w} = \\sum_i \\alpha^{(i)}t^{(i)}\\mathbf{x}^{(i)}$", "§5.6"],
         ["Substitute back", "the dual — data appears only in dot products", "§5.6"],
         ["Replace dot products with $K$", "<b>the kernel trick</b>", "§5.7"],
         ["Predict", "$\\sum_{i\\in\\text{SV}}\\alpha^{(i)}t^{(i)}K(\\mathbf{x}^{(i)},"
          "\\mathbf{x}) + b$", "§5.7"]],
    )

    keypoints([
        "Widest street ⟺ minimum $\\lVert\\mathbf{w}\\rVert$ ⟺ hinge loss + "
        "$\\ell_2$.",
        "$C$ is <b>inverse</b> regularisation; overfitting ⇒ reduce $C$ (and "
        "$\\gamma$).",
        "The dual is the only formulation in which the kernel trick can be "
        "written.",
        "Only support vectors matter — for the boundary, for storage, and for "
        "prediction cost.",
        "<code>SVC</code> is $\\mathcal{O}(m^{2..3})$: superb below ~10 000 rows, "
        "unusable above ~100 000.",
    ], title="Chapter 5 in five lines")

    refs([
        ("Cortes & Vapnik — *Support-Vector Networks* (the original soft-margin "
         "paper)", "https://doi.org/10.1007/BF00994018"),
        ("Boser, Guyon & Vapnik — *A Training Algorithm for Optimal Margin "
         "Classifiers* (the kernel trick)",
         "https://doi.org/10.1145/130385.130401"),
        ("Platt, J. — *Sequential Minimal Optimization* (the SMO algorithm behind "
         "libsvm)", "Microsoft Research Technical Report MSR-TR-98-14, 1998"),
        ("Chang & Lin — *LIBSVM: A Library for Support Vector Machines*",
         "https://doi.org/10.1145/1961189.1961199"),
        ("Rahimi & Recht — *Random Features for Large-Scale Kernel Machines* "
         "(RBFSampler)", "NeurIPS 2007"),
    ])


# ==========================================================================
SECTIONS = [
    ("5.1", "Linear SVM Classification", s_5_1),
    ("5.2", "Nonlinear SVM Classification", s_5_2),
    ("5.3", "Classes & Complexity", s_5_3),
    ("5.4", "SVM Regression", s_5_4),
    ("5.5", "Under the Hood — Primal", s_5_5),
    ("5.6", "The Dual Problem", s_5_6),
    ("5.7", "Kernelized SVMs", s_5_7),
    ("5.8", "Exercises & Review", s_5_8),
]

nav.render_chapter(CH, SECTIONS)
