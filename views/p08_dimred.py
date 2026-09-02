"""Chapter 8 — Dimensionality Reduction."""

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
CH = "ch08"

hero(
    kicker="Part I · Chapter 8",
    title="Dimensionality Reduction",
    blurb=(
        "In high dimensions your geometric intuition is not merely unreliable — it "
        "is systematically wrong, and the ways it fails are computable. This "
        "chapter quantifies the curse, then derives PCA twice (maximum variance "
        "and minimum reconstruction error, which turn out to be the same "
        "optimisation), and surveys random projection, LLE, t-SNE and the rest."
    ),
    chips=["The curse, quantified", "8 sub-sections", "8 animations",
           "8 code labs", "PCA derived twice"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_8_1():
    section("8.1", "The Curse of Dimensionality")

    lead(
        "Four facts about high-dimensional space, each of which contradicts "
        "intuition, and each of which has a one-line proof. Together they explain "
        "why almost every algorithm degrades as $n$ grows."
    )

    sub("Fact 1 — almost every point is near the boundary")

    derive(
        [("Take the unit hypercube $[0,1]^n$ and ask: what fraction of its volume "
          "lies within $\\varepsilon$ of the surface?", None),
         ("The interior region that is <i>not</i> near the boundary is itself a "
          "cube of side $1 - 2\\varepsilon$, so its volume is:",
          r"V_{\text{interior}} = (1 - 2\varepsilon)^{n}"),
         ("Therefore the fraction near the boundary is:",
          r"P(\text{near boundary}) = 1 - (1 - 2\varepsilon)^{n}"),
         ("With $\\varepsilon = 0.01$ (within 1 % of a face): in 2-D this is 4 %; "
          "in 10-D it is 18 %; in 100-D it is <b>87 %</b>; in 1 000-D it is "
          "<b>99.999 999 8 %</b>.", None),
         ("<b>Consequence:</b> essentially every training instance is an extreme "
          "value in at least one dimension, so every prediction is an "
          "extrapolation. That is why models become unreliable in high dimensions "
          "— there is no 'middle' left.", None)],
        title="Fraction of a hypercube near its boundary",
    )

    sub("Fact 2 — random points are far apart, and equally far")

    derive(
        [("Draw two points uniformly from $[0,1]^n$. The squared distance is a sum "
          "of $n$ i.i.d. terms $(U_i - V_i)^2$:",
          r"\lVert \mathbf{u} - \mathbf{v} \rVert^2 = \sum_{i=1}^{n}(U_i - V_i)^2"),
         ("Each term has mean $\\mathbb{E}[(U-V)^2] = 1/6$ and variance $7/180$, "
          "so by linearity:",
          r"\mathbb{E}\bigl[\lVert \mathbf{u}-\mathbf{v}\rVert^2\bigr] = \frac{n}{6},"
          r"\qquad \mathrm{Var} = \frac{7n}{180}"),
         ("Taking square roots, the typical distance grows like $\\sqrt{n/6}$ "
          "while its <i>standard deviation</i> grows only like $n^{1/4}$-ish. The "
          "relative spread therefore vanishes:",
          r"\frac{\sigma_{d}}{\mathbb{E}[d]} \;=\; \mathcal{O}\!\left(\frac{1}{\sqrt{n}}\right) "
          r"\;\xrightarrow[n \to \infty]{}\; 0"),
         ("<b>Distance concentration:</b> the ratio of the farthest to the nearest "
          "neighbour tends to 1.",
          r"\lim_{n\to\infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0"),
         ("<b>Consequence:</b> 'nearest neighbour' stops meaning anything. Every "
          "distance-based method — $k$-NN (§1.6), kernel SVMs (§5.2), $k$-means "
          "(§9.1), DBSCAN — degrades, because the distances it relies on all "
          "become the same number.", None)],
        title="Why distances concentrate",
    )

    sub("Fact 3 — the volume of a hypersphere vanishes")

    math(r"""
    V_n(r) \;=\; \frac{\pi^{n/2}}{\Gamma\!\left(\frac{n}{2} + 1\right)}\, r^{n}
    \qquad\Longrightarrow\qquad
    \frac{V_n(1/2)}{\text{volume of the unit cube}} \;\xrightarrow[n\to\infty]{}\; 0
    """)

    md(
        "The inscribed hypersphere occupies 78.5 % of a 2-D square, 52.4 % of a "
        "3-D cube, 0.25 % of a 10-D cube, and about $10^{-70}$ of a 100-D cube. "
        "**Nearly all of a high-dimensional cube's volume is in its corners.**"
    )

    sub("Fact 4 — you need exponentially many samples")

    math(r"""
    m \;\propto\; \left(\frac{1}{\varepsilon}\right)^{n}
    """)

    md(
        "To cover the space at resolution $\\varepsilon$ you need "
        "$\\varepsilon^{-n}$ samples. To place 100 points along each axis needs "
        "$100^n$ instances — for $n = 10$ that is $10^{20}$, more than there are "
        "grains of sand on Earth."
    )

    anim_header("All four curses, computed as the dimension grows")

    dims = np.arange(1, 101)
    near_boundary = 1 - (1 - 2 * .01) ** dims
    from scipy.special import gammaln
    log_vol = (dims / 2) * np.log(np.pi) - gammaln(dims / 2 + 1) - dims * np.log(2)
    sphere_frac = np.exp(log_vol)
    rng = np.random.default_rng(0)
    ratio = []
    for n in dims:
        P = rng.random((300, n))
        D = np.sqrt(((P[:100, None, :] - P[None, 100:, :]) ** 2).sum(-1))
        ratio.append(float((D.max() - D.min()) / D.min()))
    mean_d = np.sqrt(dims / 6)

    frames = []
    for k in range(2, 101):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=dims[:k], y=near_boundary[:k], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=dims[:k], y=sphere_frac[:k], mode="lines",
                       line=dict(color=C["info"], width=3)),
            go.Scatter(x=dims[:k], y=ratio[:k], mode="lines",
                       line=dict(color=C["warning"], width=3)),
            go.Scatter(x=dims[:k], y=mean_d[:k], mode="lines",
                       line=dict(color=C["success"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"n = {k}   ·   {near_boundary[k-1]:.1%} of the cube is within 1 % of a "
            f"face   ·   sphere/cube = {sphere_frac[k-1]:.2e}   ·   "
            f"(dmax−dmin)/dmin = {ratio[k-1]:.3f}")])))

    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Scatter(x=dims[:2], y=near_boundary[:2], mode="lines",
                           name="fraction within 1 % of a face",
                           line=dict(color=C["danger"], width=3)))
    f.add_trace(go.Scatter(x=dims[:2], y=sphere_frac[:2], mode="lines",
                           name="inscribed sphere / cube volume",
                           line=dict(color=C["info"], width=3)))
    f.add_trace(go.Scatter(x=dims[:2], y=ratio[:2], mode="lines",
                           name="(dmax − dmin)/dmin  → 0",
                           line=dict(color=C["warning"], width=3)))
    f.add_trace(go.Scatter(x=dims[:2], y=mean_d[:2], mode="lines",
                           name="mean pairwise distance √(n/6)",
                           line=dict(color=C["success"], width=3)),
                secondary_y=True)
    f.update_layout(height=470, xaxis_title="dimension n",
                    yaxis=dict(title="fraction / ratio", range=[0, 1.6]),
                    title="Four faces of the curse of dimensionality",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    f.update_yaxes(title_text="distance", secondary_y=True, showgrid=False)
    anim.animate(f, frames, duration=nav.anim_ms(45), slider_prefix="n = ")
    figure(f)

    idea(
        "The way out: the manifold hypothesis",
        "Real high-dimensional data is <b>not</b> uniformly spread through its "
        "ambient space. A 784-pixel MNIST image lives in $\\mathbb{R}^{784}$, but "
        "random points in $\\mathbb{R}^{784}$ look like static, not digits. Real "
        "digits lie on a much lower-dimensional <b>manifold</b> — a curved surface "
        "parameterised by stroke thickness, slant, position, style. The "
        "<i>intrinsic</i> dimension is perhaps 10–20, not 784. Dimensionality "
        "reduction is the art of finding that manifold, and it works precisely "
        "because the curse describes uniform data, which real data is not.",
    )

    code_lab(
        "Measure every curse yourself",
        '''import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(0)

# ============ 1. fraction of the cube near the boundary ================
print("=== fraction of [0,1]^n within 0.01 of a face ===")
print(f"{'n':>6}{'theory 1-(0.98)^n':>22}{'simulated':>13}")
for n in [1, 2, 5, 10, 50, 100, 500, 1000]:
    theory = 1 - 0.98**n
    P = rng.random((20000, min(n, 1000)))
    sim = np.mean(((P < .01) | (P > .99)).any(1))
    print(f"{n:>6}{theory:>22.9f}{sim:>13.4f}")

# ============ 2. distance concentration ================================
print("\\n=== distances between two random points in [0,1]^n ===")
print(f"{'n':>6}{'mean d':>10}{'sd d':>9}{'sd/mean':>10}"
      f"{'(dmax-dmin)/dmin':>20}")
for n in [1, 2, 5, 10, 50, 100, 1000, 10000]:
    A = rng.random((150, n)); B = rng.random((150, n))
    D = np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(-1))
    d = D.ravel()
    print(f"{n:>6}{d.mean():>10.3f}{d.std():>9.3f}{d.std()/d.mean():>10.4f}"
          f"{(d.max()-d.min())/d.min():>20.4f}")
print("\\nsd/mean -> 0 and (dmax-dmin)/dmin -> 0: every point is equidistant.")
print(f"theory: E[d^2] = n/6, so E[d] ~ sqrt(n/6):  n=10000 -> "
      f"{np.sqrt(10000/6):.2f}")

# ============ 3. the sphere vanishes ===================================
print("\\n=== volume of the inscribed hypersphere / volume of the cube ===")
def sphere_over_cube(n):
    return np.exp((n/2)*np.log(np.pi) - gammaln(n/2+1) - n*np.log(2))
print(f"{'n':>6}{'ratio':>16}{'':>4}")
for n in [1, 2, 3, 5, 10, 20, 50, 100]:
    print(f"{n:>6}{sphere_over_cube(n):>16.3e}")
print("Almost all of a high-dimensional cube's volume is in its CORNERS.")

# ============ 4. k-NN degrades ==========================================
print("\\n=== what this does to k-NN ===")
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
print(f"{'n_features':>12}{'R^2 (informative=2)':>22}{'R^2 (all informative)':>24}")
for n in [2, 5, 10, 25, 50, 100, 200]:
    m = 2000
    X = rng.normal(0, 1, (m, n))
    # (a) only the first 2 features matter -- the rest are noise
    y_a = X[:, 0] + X[:, 1] + rng.normal(0, .3, m)
    # (b) every feature matters
    w = rng.normal(0, 1, n)
    y_b = X @ w / np.sqrt(n) + rng.normal(0, .3, m)
    r2 = []
    for y in (y_a, y_b):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0)
        r2.append(KNeighborsRegressor(10).fit(Xtr, ytr).score(Xte, yte))
    print(f"{n:>12}{r2[0]:>22.4f}{r2[1]:>24.4f}")
print("\\nColumn A collapses: 198 noise features drown the 2 useful ones in the")
print("distance metric. This is exactly what PCA fixes.")

# ============ 5. but REAL data is not uniform ==========================
print("\\n=== the manifold hypothesis ===")
from sklearn.datasets import load_digits
d = load_digits()
X = d.data
print(f"digits live in R^{X.shape[1]}")
sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
ev = sv**2 / np.sum(sv**2)
for thresh in [.90, .95, .99]:
    k = int(np.searchsorted(np.cumsum(ev), thresh) + 1)
    print(f"  {thresh:.0%} of the variance lives in {k} of {X.shape[1]} dimensions")
Xr = rng.random(X.shape) * 16
svr = np.linalg.svd(Xr - Xr.mean(0), compute_uv=False)
evr = svr**2 / np.sum(svr**2)
print(f"  for UNIFORM noise of the same shape, 95 % needs "
      f"{int(np.searchsorted(np.cumsum(evr), .95)+1)} of {X.shape[1]}")
print("\\nReal data has structure. Uniform data does not. That gap IS the manifold.")
''',
        key="ch08_curse",
    )

    keypoints([
        "In 100-D, <b>87 %</b> of a unit cube is within 1 % of a face — every "
        "point is an extreme value somewhere.",
        "Distances <b>concentrate</b>: $(d_{\\max}-d_{\\min})/d_{\\min} \\to 0$, so "
        "'nearest neighbour' loses meaning.",
        "The inscribed sphere's volume vanishes — all the volume is in the "
        "corners.",
        "Covering the space needs $\\varepsilon^{-n}$ samples: exponentially "
        "hopeless.",
        "<b>Real data lies on a low-dimensional manifold</b>, which is why "
        "dimensionality reduction works at all.",
    ])


# ==========================================================================
def s_8_2():
    section("8.2", "Two Approaches — Projection and Manifold Learning")

    lead(
        "Every technique in this chapter falls into one of two families, and the "
        "choice between them is entirely about whether your data is flat."
    )

    sub("Projection")

    md(
        "If the data lies near a **linear subspace** — a plane, or a "
        "hyperplane — you can drop a perpendicular onto it. PCA, random "
        "projection and linear discriminant analysis are all projections. They "
        "are fast, invertible (approximately), and they preserve global structure."
    )

    sub("Manifold learning")

    md(
        "If the data lies on a **curved** surface, projection destroys it. The "
        "Swiss roll is the standard illustration: projecting it onto a plane "
        "squashes distant parts of the roll on top of each other. Manifold "
        "learning instead tries to *unroll* the surface — preserving local "
        "neighbourhoods rather than global geometry."
    )

    anim_header("Swiss roll: projection squashes, unrolling preserves")

    X3, t3 = ds.swiss_roll(n=1400, noise=.4)
    from sklearn.decomposition import PCA
    from sklearn.manifold import LocallyLinearEmbedding

    pca2 = PCA(n_components=2).fit_transform(X3)
    lle2 = LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                  random_state=0).fit_transform(X3)

    f3 = go.Figure(go.Scatter3d(
        x=X3[:, 0], y=X3[:, 1], z=X3[:, 2], mode="markers",
        marker=dict(size=3, color=t3, colorscale=nav.cscale(), opacity=.85,
                    colorbar=dict(title="position<br>along<br>the roll"))))
    f3.update_layout(height=520, title="The Swiss roll in 3-D — a 2-D manifold "
                                       "curled into 3-D space",
                     scene=dict(aspectmode="data"))
    anim.rotating_3d(f3, n_frames=40, duration=nav.anim_ms(90))
    figure(f3, "Colour encodes position along the roll. A good 2-D embedding "
               "should keep that colour gradient smooth and monotone.")

    c1, c2 = st.columns(2)
    with c1:
        fp = go.Figure(go.Scattergl(
            x=pca2[:, 0], y=pca2[:, 1], mode="markers",
            marker=dict(size=4, color=t3, colorscale=nav.cscale(), opacity=.8)))
        fp.update_layout(height=380, title="PCA — a projection",
                         xaxis_title="PC 1", yaxis_title="PC 2")
        figure(fp, "Colours fold over each other: points from opposite ends of "
                   "the roll land on top of each other.")
    with c2:
        fl = go.Figure(go.Scattergl(
            x=lle2[:, 0], y=lle2[:, 1], mode="markers",
            marker=dict(size=4, color=t3, colorscale=nav.cscale(), opacity=.8)))
        fl.update_layout(height=380, title="LLE — manifold learning",
                         xaxis_title="dim 1", yaxis_title="dim 2")
        figure(fl, "The roll is unrolled: the colour gradient runs smoothly "
                   "across the embedding.")

    sub("Does reducing dimensions always help?")

    pitfall(
        "No. Sometimes it makes things worse.",
        "Dimensionality reduction <b>always loses information</b>. Whether the "
        "trade is worth it depends on the dataset. Two things are reliably true: "
        "(1) it usually <b>speeds up</b> training, sometimes dramatically; "
        "(2) it does <b>not</b> always improve accuracy — if the discarded "
        "directions carried signal, performance drops. Treat "
        "<code>n_components</code> as a hyperparameter inside your pipeline and "
        "cross-validate it, exactly like any other.",
    )

    anim_header("A case where the decision boundary is simpler after unrolling")
    md(
        "Two versions of a Swiss-roll classification problem. In the top row the "
        "class boundary is simple in the *unrolled* coordinates — so unrolling "
        "helps enormously. In the bottom row the classes are separated by a plane "
        "in the *original* 3-D space — so unrolling actively hurts. The lesson: "
        "it depends on the data, always."
    )

    y_easy = (t3 > 10).astype(int)                    # simple in unrolled coords
    y_hard = (X3[:, 0] > 0).astype(int)               # simple in original coords

    views = [
        ("3-D, class = position along the roll (unrolling HELPS)", X3[:, [0, 2]], y_easy),
        ("unrolled, class = position along the roll", lle2, y_easy),
        ("3-D, class = a plane in the original space (unrolling HURTS)",
         X3[:, [0, 2]], y_hard),
        ("unrolled, class = a plane in the original space", lle2, y_hard),
    ]
    frames = [go.Frame(name=str(i + 1), data=[
        go.Scattergl(x=Xv[yv == 0, 0], y=Xv[yv == 0, 1], mode="markers",
                     marker=dict(color=C["train"], size=4, opacity=.75)),
        go.Scattergl(x=Xv[yv == 1, 0], y=Xv[yv == 1, 1], mode="markers",
                     marker=dict(color=C["warning"], size=4, opacity=.75)),
    ], layout=go.Layout(title=nm, xaxis=dict(autorange=True),
                        yaxis=dict(autorange=True)))
        for i, (nm, Xv, yv) in enumerate(views)]

    f = go.Figure(data=[
        go.Scattergl(x=X3[y_easy == 0, 0], y=X3[y_easy == 0, 2], mode="markers",
                     name="class 0", marker=dict(color=C["train"], size=4, opacity=.75)),
        go.Scattergl(x=X3[y_easy == 1, 0], y=X3[y_easy == 1, 2], mode="markers",
                     name="class 1", marker=dict(color=C["warning"], size=4, opacity=.75)),
    ])
    f.update_layout(height=460, title=views[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1800), slider_prefix="view ")
    figure(f)

    code_lab(
        "Does it help? Measure, don't assume.",
        '''import numpy as np, time
from sklearn.datasets import make_swiss_roll, load_digits, make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

# ============ 1. Swiss roll: unrolling helps or hurts DEPENDING ========
X, t = make_swiss_roll(n_samples=2000, noise=.5, random_state=42)
y_along = (t > 10).astype(int)          # simple in the UNROLLED coordinates
y_plane = (X[:, 0] > 0).astype(int)     # simple in the ORIGINAL coordinates

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12, random_state=0)
X_unrolled = lle.fit_transform(X)

print("=== the same reduction, two different targets ===")
print(f"{'target':<34}{'raw 3-D':>10}{'unrolled 2-D':>15}")
for nm, y in [("class = position along the roll", y_along),
              ("class = a plane in 3-D",          y_plane)]:
    a = cross_val_score(LogisticRegression(), X, y, cv=5).mean()
    b = cross_val_score(LogisticRegression(), X_unrolled, y, cv=5).mean()
    print(f"{nm:<34}{a:>10.4f}{b:>15.4f}")
print("\\nUnrolling is not universally good -- it is good for SOME targets.")

# ============ 2. PCA on digits: accuracy vs speed ======================
d = load_digits()
Xd, yd = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=.3, stratify=yd,
                                      random_state=42)
print("\\n=== PCA on digits: the real trade-off ===")
print(f"{'n_components':>13}{'explained var':>15}{'fit time':>11}{'accuracy':>10}")
for nc in [2, 5, 10, 20, 30, 40, 64]:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                         RandomForestClassifier(n_estimators=200, random_state=0,
                                                n_jobs=-1))
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); dt = time.perf_counter()-t0
    ev = pipe[1].explained_variance_ratio_.sum()
    print(f"{nc:>13}{ev:>15.4f}{dt:>10.3f}s{pipe.score(Xte, yte):>10.4f}")

base = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
t0 = time.perf_counter(); base.fit(Xtr, ytr); dt0 = time.perf_counter()-t0
print(f"{'none (all 64)':>13}{1.0:>15.4f}{dt0:>10.3f}s{base.score(Xte, yte):>10.4f}")

# ============ 3. when PCA HELPS a lot: many noise features ============
print("\\n=== PCA with 2 informative and 198 noise features ===")
rng = np.random.default_rng(0)
m = 2000
Xi = rng.normal(0, 1, (m, 200))
yi = (Xi[:, 0] + Xi[:, 1] + rng.normal(0, .3, m) > 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=.3, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
print(f"{'model':<34}{'test accuracy':>15}")
print(f"{'kNN on all 200 features':<34}"
      f"{KNeighborsClassifier(15).fit(Xtr, ytr).score(Xte, yte):>15.4f}")
for nc in [2, 5, 20]:
    p = make_pipeline(StandardScaler(), PCA(nc), KNeighborsClassifier(15))
    print(f"{f'kNN after PCA to {nc} dims':<34}{p.fit(Xtr, ytr).score(Xte, yte):>15.4f}")
print(f"{'kNN on the 2 true features':<34}"
      f"{KNeighborsClassifier(15).fit(Xtr[:, :2], ytr).score(Xte[:, :2], yte):>15.4f}")
print("\\nPCA does NOT recover the true features here (the noise has as much")
print("variance as the signal). Variance is not the same thing as usefulness.")
''',
        key="ch08_approaches",
    )

    keypoints([
        "<b>Projection</b> (PCA, random projection): drop a perpendicular onto a "
        "linear subspace. Fast, global.",
        "<b>Manifold learning</b> (LLE, t-SNE, Isomap): unroll a curved surface. "
        "Preserves local neighbourhoods.",
        "Projection fails when the manifold is curved (the Swiss roll folds over "
        "itself).",
        "Dimensionality reduction always loses information — it reliably speeds "
        "training, not accuracy.",
        "Treat <code>n_components</code> as a cross-validated hyperparameter.",
    ])


# ==========================================================================
def s_8_3():
    section("8.3", "PCA — Preserving the Variance")

    lead(
        "Principal Component Analysis. Two apparently different objectives — "
        "maximise the retained variance, or minimise the reconstruction error — "
        "turn out to be the same optimisation, and both are solved by the "
        "eigenvectors of the covariance matrix."
    )

    sub("Objective 1 — maximum variance")

    math(r"""
    \mathbf{w}_1 \;=\;
    \operatorname*{arg\,max}_{\lVert\mathbf{w}\rVert = 1}\;
    \frac{1}{m}\sum_{i=1}^{m}\bigl(\mathbf{w}^\top \mathbf{x}_c^{(i)}\bigr)^2
    \;=\;
    \operatorname*{arg\,max}_{\lVert\mathbf{w}\rVert = 1}\;
    \mathbf{w}^\top \boldsymbol\Sigma \, \mathbf{w}
    """)
    where({r"\mathbf{x}_c^{(i)}": "the <b>centred</b> instance "
                                  "$\\mathbf{x}^{(i)} - \\bar{\\mathbf{x}}$ — "
                                  "centring is not optional",
           r"\boldsymbol\Sigma": "the covariance matrix "
                                 "$\\frac{1}{m}\\mathbf{X}_c^\\top\\mathbf{X}_c$"})

    derive(
        [("Maximise $\\mathbf{w}^\\top\\boldsymbol\\Sigma\\mathbf{w}$ subject to "
          "$\\mathbf{w}^\\top\\mathbf{w} = 1$. Form the Lagrangian:",
          r"\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top \boldsymbol\Sigma \mathbf{w} "
          r"- \lambda\bigl(\mathbf{w}^\top\mathbf{w} - 1\bigr)"),
         ("Set the gradient with respect to $\\mathbf{w}$ to zero, using "
          "$\\nabla_{\\mathbf{w}}(\\mathbf{w}^\\top\\mathbf{A}\\mathbf{w}) = "
          "2\\mathbf{A}\\mathbf{w}$ for symmetric $\\mathbf{A}$:",
          r"2\boldsymbol\Sigma\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} "
          r"\;\;\Longrightarrow\;\; \boldsymbol\Sigma\mathbf{w} = \lambda\mathbf{w}"),
         ("<b>That is the eigenvalue equation.</b> The stationary points of the "
          "variance are exactly the eigenvectors of $\\boldsymbol\\Sigma$.", None),
         ("Which eigenvector? Left-multiply by $\\mathbf{w}^\\top$:",
          r"\mathbf{w}^\top\boldsymbol\Sigma\mathbf{w} = \lambda\,\mathbf{w}^\top\mathbf{w} = \lambda"),
         ("So the variance captured along $\\mathbf{w}$ <b>equals its "
          "eigenvalue</b>. The maximum is the eigenvector with the largest "
          "eigenvalue $\\lambda_1$. The second component is the largest-eigenvalue "
          "direction orthogonal to the first, and so on — giving the eigenvectors "
          "in descending eigenvalue order.", None),
         ("Since $\\boldsymbol\\Sigma$ is real and symmetric, the spectral theorem "
          "guarantees its eigenvectors are <b>orthogonal</b> and its eigenvalues "
          "are <b>real and non-negative</b>. So the principal components always "
          "form an orthonormal basis.", None)],
        title="Maximum variance ⟹ the eigenvalue equation",
    )

    sub("Objective 2 — minimum reconstruction error")

    derive(
        [("Now ask the opposite question: which $d$-dimensional subspace "
          "minimises the squared distance from each point to its projection?",
          r"\min_{\mathbf{W}: \mathbf{W}^\top\mathbf{W} = \mathbf{I}_d}\;\;"
          r"\frac{1}{m}\sum_{i=1}^{m}\Bigl\lVert \mathbf{x}_c^{(i)} "
          r"- \mathbf{W}\mathbf{W}^\top\mathbf{x}_c^{(i)}\Bigr\rVert^2"),
         ("Expand the squared norm. Because $\\mathbf{W}\\mathbf{W}^\\top$ is an "
          "orthogonal projector ($P^2 = P = P^\\top$), the cross terms simplify by "
          "Pythagoras:",
          r"\bigl\lVert \mathbf{x}_c - P\mathbf{x}_c \bigr\rVert^2 "
          r"= \lVert \mathbf{x}_c \rVert^2 - \lVert P\mathbf{x}_c \rVert^2"),
         ("Average over the data:",
          r"\underbrace{\frac1m\sum_i \lVert \mathbf{x}_c^{(i)}\rVert^2}_{\text{total variance, fixed}} "
          r"= \underbrace{\frac1m\sum_i \lVert P\mathbf{x}_c^{(i)}\rVert^2}_{\text{retained variance}} "
          r"+ \underbrace{\frac1m\sum_i \lVert \mathbf{x}_c^{(i)} - P\mathbf{x}_c^{(i)}\rVert^2}_{\text{reconstruction error}}"),
         ("The left-hand side does not depend on $\\mathbf{W}$ at all. Therefore "
          "<b>minimising reconstruction error is exactly the same as maximising "
          "retained variance</b> — the two objectives are one problem, and both "
          "are solved by the top-$d$ eigenvectors.", None),
         ("The reconstruction error even has a closed form: it is the sum of the "
          "<i>discarded</i> eigenvalues.",
          r"\text{MSE}_{\text{reconstruction}} = \sum_{j = d+1}^{n} \lambda_j")],
        title="Minimum reconstruction error is the SAME problem",
    )

    sub("Computing PCA: use the SVD, not the covariance matrix")

    math(r"""
    \mathbf{X}_c \;=\; \mathbf{U}\,\boldsymbol\Sigma_{\text{svd}}\,\mathbf{V}^\top
    \qquad\Longrightarrow\qquad
    \mathbf{W}_d = \text{first } d \text{ columns of } \mathbf{V},
    \qquad
    \lambda_j = \frac{\sigma_j^2}{m}
    """)

    proof(
        "Why SVD rather than eigendecomposition of Σ",
        "Because $\\boldsymbol\\Sigma = \\frac{1}{m}\\mathbf{X}_c^\\top\\mathbf{X}_c$ "
        "<b>squares the condition number</b> — the same argument as §4.1's normal "
        "equation. Forming $\\boldsymbol\\Sigma$ explicitly loses roughly half your "
        "significant digits, and it costs $\\mathcal{O}(mn^2)$ memory-wise "
        "prohibitive when $n$ is large. The SVD of $\\mathbf{X}_c$ gives the same "
        "answer with better numerics and never forms an $n\\times n$ matrix.",
    )

    sub("Projecting down and reconstructing back")

    math(r"""
    \mathbf{X}_{d\text{-proj}} \;=\; \mathbf{X}_c\,\mathbf{W}_d
    \qquad\qquad
    \mathbf{X}_{\text{recovered}} \;=\; \mathbf{X}_{d\text{-proj}}\,\mathbf{W}_d^\top
      \;+\; \bar{\mathbf{x}}
    """)

    anim_header("Rotating the projection line: variance up, error down, together")
    md(
        "One line through a 2-D cloud, rotating. The green bars are the projected "
        "coordinates (their spread is the retained variance); the red segments are "
        "the reconstruction errors. Watch them trade off exactly — and note that "
        "**both are optimal at the same angle**, which is the identity just "
        "derived."
    )

    rng = np.random.default_rng(3)
    Xp = rng.multivariate_normal([0, 0], [[3.0, 1.9], [1.9, 1.6]], 90)
    Xp -= Xp.mean(0)
    total_var = float((Xp ** 2).sum(1).mean())
    angs = np.linspace(0, np.pi, 46)

    stats = []
    for a in angs:
        u = np.array([np.cos(a), np.sin(a)])
        proj = Xp @ u
        rec = np.outer(proj, u)
        stats.append((float(proj.var()), float(((Xp - rec) ** 2).sum(1).mean()), u))
    best = int(np.argmax([s[0] for s in stats]))

    frames = []
    for k, a in enumerate(angs):
        v, err, u = stats[k]
        rec = np.outer(Xp @ u, u)
        segs_x, segs_y = [], []
        for i in range(len(Xp)):
            segs_x += [Xp[i, 0], rec[i, 0], None]
            segs_y += [Xp[i, 1], rec[i, 1], None]
        col = C["success"] if k == best else C["primary"]
        frames.append(go.Frame(name=f"{np.degrees(a):.0f}", data=[
            go.Scatter(x=[-5 * u[0], 5 * u[0]], y=[-5 * u[1], 5 * u[1]],
                       mode="lines", line=dict(color=col, width=3.5)),
            go.Scatter(x=segs_x, y=segs_y, mode="lines",
                       line=dict(color=alpha(C["danger"], .55), width=1.2)),
            go.Scatter(x=Xp[:, 0], y=Xp[:, 1], mode="markers",
                       marker=dict(color=C["train"], size=7,
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=rec[:, 0], y=rec[:, 1], mode="markers",
                       marker=dict(color=C["success"], size=5)),
            go.Scatter(x=np.degrees(angs[:k + 1]),
                       y=[s[0] for s in stats[:k + 1]], mode="lines",
                       line=dict(color=C["success"], width=3)),
            go.Scatter(x=np.degrees(angs[:k + 1]),
                       y=[s[1] for s in stats[:k + 1]], mode="lines",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"angle {np.degrees(a):5.1f}°   ·   retained variance = {v:.4f}   ·   "
            f"reconstruction MSE = {err:.4f}   ·   sum = {v + err:.4f}",
            color=col)])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("the projection line",
                                      "variance (green) and error (red)"))
    u0 = stats[0][2]
    rec0 = np.outer(Xp @ u0, u0)
    f.add_trace(go.Scatter(x=[-5 * u0[0], 5 * u0[0]], y=[-5 * u0[1], 5 * u0[1]],
                           mode="lines", name="projection line",
                           line=dict(color=C["primary"], width=3.5)), 1, 1)
    f.add_trace(go.Scatter(x=[], y=[], mode="lines", name="reconstruction error",
                           line=dict(color=alpha(C["danger"], .55), width=1.2)), 1, 1)
    f.add_trace(go.Scatter(x=Xp[:, 0], y=Xp[:, 1], mode="markers", name="data",
                           marker=dict(color=C["train"], size=7,
                                       line=dict(color="#fff", width=.8))), 1, 1)
    f.add_trace(go.Scatter(x=rec0[:, 0], y=rec0[:, 1], mode="markers",
                           name="projections",
                           marker=dict(color=C["success"], size=5)), 1, 1)
    f.add_trace(go.Scatter(x=[0], y=[stats[0][0]], mode="lines",
                           name="retained variance",
                           line=dict(color=C["success"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=[0], y=[stats[0][1]], mode="lines",
                           name="reconstruction MSE",
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.add_hline(y=total_var, line_dash="dot", line_color=C["truth"],
                annotation_text="total variance (constant)", row=1, col=2)
    f.update_xaxes(range=[-5, 5], row=1, col=1)
    f.update_yaxes(range=[-4, 4], row=1, col=1)
    f.update_xaxes(title_text="angle (°)", range=[0, 180], row=1, col=2)
    f.update_yaxes(range=[0, total_var * 1.1], row=1, col=2)
    f.update_layout(height=480, title="Maximising variance = minimising error")
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="angle ")
    figure(f, "The two curves are mirror images summing to a constant — exactly "
              "the identity proved above.")

    pitfall(
        "Two things that break PCA",
        "<b>(1) Not centring.</b> PCA <i>requires</i> the data to be centred. "
        "scikit-learn's <code>PCA</code> does it for you; if you implement it via "
        "<code>np.linalg.svd</code> yourself, you must subtract the mean or the "
        "first component will simply point at the mean.<br>"
        "<b>(2) Not scaling.</b> PCA maximises <i>variance</i>, and variance has "
        "units. A feature measured in millimetres has $10^6$ times the variance of "
        "the same feature in metres, so it will dominate every component. Unless "
        "all features share a natural unit (pixel intensities, say), put a "
        "<code>StandardScaler</code> in front.",
    )

    code_lab(
        "PCA from scratch three ways, and the variance/error identity",
        '''import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(4)
X = rng.multivariate_normal([3, -1, 2], [[4, 2.4, .6],
                                         [2.4, 3, 1.1],
                                         [.6, 1.1, 1.5]], 500)

# ============ 1. via the covariance eigendecomposition ================
Xc = X - X.mean(0)                              # CENTRING IS MANDATORY
Sigma = Xc.T @ Xc / len(Xc)
evals, evecs = np.linalg.eigh(Sigma)            # eigh: symmetric matrices
order = evals.argsort()[::-1]
evals, evecs = evals[order], evecs[:, order]
print("=== eigendecomposition of the covariance matrix ===")
print(f"eigenvalues        = {evals.round(5)}")
print(f"explained variance = {(evals/evals.sum()).round(5)}")

# ============ 2. via the SVD (what sklearn does) ======================
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print(f"\\n=== SVD ===")
print(f"singular values    = {s.round(5)}")
print(f"lambda = s^2/m     = {(s**2/len(Xc)).round(5)}   <- same eigenvalues")
print(f"max |V - eigvec|   = "
      f"{np.abs(np.abs(Vt.T) - np.abs(evecs)).max():.3e}   <- same directions")

# ============ 3. sklearn ==============================================
p = PCA().fit(X)
print(f"\\n=== sklearn PCA ===")
print(f"explained_variance = {p.explained_variance_.round(5)}")
print(f"ratio              = {p.explained_variance_ratio_.round(5)}")
print(f"components_ (rows) =\\n{p.components_.round(4)}")

# ============ 4. THE IDENTITY: variance + error = total ===============
print("\\n=== retained variance + reconstruction error = total variance ===")
total = float((Xc**2).sum(1).mean())
print(f"total variance = {total:.6f}   (= sum of eigenvalues = {evals.sum():.6f})")
print(f"\\n{'d':>3}{'retained var':>15}{'recon MSE':>13}{'sum':>13}"
      f"{'discarded eigs':>17}")
for d in range(1, 4):
    W = Vt[:d].T
    proj = Xc @ W
    rec  = proj @ W.T
    retained = float((rec**2).sum(1).mean())
    err = float(((Xc - rec)**2).sum(1).mean())
    print(f"{d:>3}{retained:>15.6f}{err:>13.6f}{retained+err:>13.6f}"
          f"{evals[d:].sum():>17.6f}")
print("\\nColumn 'recon MSE' equals 'discarded eigs' exactly -- that is the")
print("closed form derived above.")

# ============ 5. project and reconstruct ==============================
pca2 = PCA(n_components=2).fit(X)
Z = pca2.transform(X)
X_back = pca2.inverse_transform(Z)
print(f"\\nproject 3-D -> 2-D -> back to 3-D")
print(f"  original  X[0] = {X[0].round(4)}")
print(f"  projected Z[0] = {Z[0].round(4)}")
print(f"  recovered      = {X_back[0].round(4)}")
print(f"  mean squared reconstruction error = "
      f"{np.mean((X - X_back)**2):.6f}")

# ============ 6. WHY SCALING MATTERS ==================================
print("\\n=== PCA is not scale invariant ===")
Xs = X.copy()
Xs[:, 0] *= 1000                                 # change the units of one column
print(f"unscaled: explained variance ratio = "
      f"{PCA().fit(X).explained_variance_ratio_.round(4)}")
print(f"x1 in mm: explained variance ratio = "
      f"{PCA().fit(Xs).explained_variance_ratio_.round(4)}   <- PC1 is now just x1")
print(f"standardised first             = "
      f"{PCA().fit(StandardScaler().fit_transform(Xs)).explained_variance_ratio_.round(4)}")

# ============ 7. components are orthonormal ===========================
W = p.components_
print(f"\\nW W^T = identity? max deviation = "
      f"{np.abs(W @ W.T - np.eye(len(W))).max():.2e}")
''',
        key="ch08_pca",
    )

    quiz(
        "You run PCA without standardising, on data where one feature is measured "
        "in metres and another in millimetres. What happens?",
        ["Nothing — PCA is scale invariant",
         "The millimetre feature dominates PC1 because it has vastly larger "
         "variance",
         "The metre feature dominates because it has larger values",
         "PCA fails with an error"],
        1,
        "Variance scales with the square of the unit, so the millimetre feature "
        "has $10^6\\times$ the variance and captures PC1 almost entirely — "
        "regardless of whether it carries any information.",
        key="ch08q1",
    )

    keypoints([
        "PCA finds the orthonormal directions of maximum variance: the "
        "eigenvectors of $\\boldsymbol\\Sigma$, ordered by eigenvalue.",
        "Max variance and min reconstruction error are the <b>same</b> problem — "
        "their sum is the constant total variance.",
        "Reconstruction MSE $= \\sum_{j>d}\\lambda_j$, the discarded eigenvalues.",
        "Compute via <b>SVD of the centred data</b>, never by forming "
        "$\\boldsymbol\\Sigma$ (condition number squares).",
        "<b>Centre always; scale unless the features share a natural unit.</b>",
    ])


# ==========================================================================
def s_8_4():
    section("8.4", "Choosing d, Explained Variance, and Compression")

    lead(
        "How many components should you keep? Three answers, depending on what "
        "you are doing with the result."
    )

    sub("Explained variance ratio")

    math(r"""
    \text{EVR}_j \;=\; \frac{\lambda_j}{\displaystyle\sum_{k=1}^{n}\lambda_k}
    \qquad\qquad
    \text{cumulative EVR}(d) \;=\; \frac{\displaystyle\sum_{j=1}^{d}\lambda_j}
                                        {\displaystyle\sum_{k=1}^{n}\lambda_k}
    """)

    table(
        ["Goal", "How to choose $d$", "scikit-learn"],
        [["Preprocessing before a supervised model",
          "Keep 95 % of the variance (or cross-validate $d$ directly)",
          "<code>PCA(n_components=0.95)</code>"],
         ["Visualisation", "$d = 2$ or $3$, no negotiation",
          "<code>PCA(n_components=2)</code>"],
         ["Compression / denoising",
          "The elbow of the cumulative curve",
          "Plot and look"],
         ["You genuinely do not know",
          "Cross-validate — $d$ is a hyperparameter like any other",
          "<code>GridSearchCV</code> over <code>pca__n_components</code>"]],
    )

    codenote(
        "n_components accepts three kinds of value",
        "An <b>integer</b> ($d$ components), a <b>float in (0,1)</b> (keep enough "
        "components to reach that fraction of variance), or "
        "<code>'mle'</code> (Minka's maximum-likelihood estimate of the intrinsic "
        "dimension). The float form is the one you will use most.",
    )

    anim_header("Reconstructing a digit from more and more components")
    md(
        "The same image reconstructed from $d$ = 1, 2, 5, … components. The panel "
        "on the right tracks the cumulative explained variance and the "
        "compression ratio. Notice how recognisable the digit becomes long before "
        "the variance curve flattens."
    )

    from sklearn.decomposition import PCA
    X, y, images = ds.digits()
    Xc = X - X.mean(0)
    pca_full = PCA().fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    idx0 = int(np.where(y == 3)[0][0])

    ds_list = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 64]
    recs = []
    for d in ds_list:
        p = PCA(n_components=d).fit(X)
        recs.append(p.inverse_transform(p.transform(X[idx0:idx0 + 1]))[0])

    frames = []
    for i, d in enumerate(ds_list):
        err = float(np.mean((recs[i] - X[idx0]) ** 2))
        comp = d / 64
        frames.append(go.Frame(name=str(d), data=[
            go.Heatmap(z=recs[i].reshape(8, 8)[::-1], colorscale=nav.cscale(),
                       zmin=0, zmax=16, showscale=False),
            go.Heatmap(z=images[idx0][::-1], colorscale=nav.cscale(),
                       zmin=0, zmax=16, showscale=False),
            go.Scatter(x=list(range(1, 65)), y=cum, mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=[d], y=[cum[d - 1]], mode="markers",
                       marker=dict(color=C["danger"], size=14,
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"d = {d}/64   ·   explained variance = {cum[d-1]:.1%}   ·   "
            f"size = {comp:.0%} of the original   ·   MSE = {err:.3f}")])))

    f = make_subplots(rows=1, cols=3, column_widths=[.24, .24, .52],
                      subplot_titles=("reconstructed", "original",
                                      "cumulative explained variance"))
    f.add_trace(go.Heatmap(z=recs[0].reshape(8, 8)[::-1], colorscale=nav.cscale(),
                           zmin=0, zmax=16, showscale=False), 1, 1)
    f.add_trace(go.Heatmap(z=images[idx0][::-1], colorscale=nav.cscale(),
                           zmin=0, zmax=16, showscale=False), 1, 2)
    f.add_trace(go.Scatter(x=list(range(1, 65)), y=cum, mode="lines",
                           name="cumulative EVR",
                           line=dict(color=C["primary"], width=3)), 1, 3)
    f.add_trace(go.Scatter(x=[1], y=[cum[0]], mode="markers", name="current d",
                           marker=dict(color=C["danger"], size=14,
                                       line=dict(color="#fff", width=2))), 1, 3)
    for th, lbl in [(.90, "90 %"), (.95, "95 %"), (.99, "99 %")]:
        f.add_hline(y=th, line_dash="dot", line_color=C["muted"],
                    annotation_text=lbl, row=1, col=3)
    f.update_xaxes(visible=False, row=1, col=1); f.update_yaxes(visible=False, row=1, col=1)
    f.update_xaxes(visible=False, row=1, col=2); f.update_yaxes(visible=False, row=1, col=2)
    f.update_xaxes(title_text="number of components d", row=1, col=3)
    f.update_yaxes(range=[0, 1.05], row=1, col=3)
    f.update_layout(height=430, title="PCA compression of a handwritten digit")
    anim.animate(f, frames, duration=nav.anim_ms(750), slider_prefix="d = ")
    figure(f)

    sub("PCA for compression")

    math(r"""
    \text{compression ratio} \;=\; \frac{d}{n}
    \qquad\qquad
    \text{storage} \;=\; \underbrace{m \cdot d}_{\text{projections}}
      \;+\; \underbrace{d \cdot n}_{\text{components}}
      \;+\; \underbrace{n}_{\text{mean}}
    """)

    md(
        "On full MNIST ($n = 784$), keeping 95 % of the variance needs about "
        "**150** components — an 80 % reduction. `inverse_transform` reconstructs "
        "the images with visible but small loss, and the discarded 5 % is largely "
        "noise, which is why PCA doubles as a **denoiser**."
    )

    code_lab(
        "Choosing d, compressing, and denoising",
        '''import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = load_digits()
X, y = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y,
                                      random_state=42)

# ============ 1. three ways to pick n_components =======================
print("=== how many components? ===")
full = PCA().fit(Xtr)
cum = np.cumsum(full.explained_variance_ratio_)
for th in [.80, .90, .95, .99]:
    dd = int(np.searchsorted(cum, th) + 1)
    print(f"  {th:.0%} of variance -> d = {dd:>3}  "
          f"({dd/X.shape[1]:.1%} of the original size)")

p_mle = PCA(n_components="mle").fit(Xtr)
print(f"  Minka's MLE estimate -> d = {p_mle.n_components_}")

# the elbow
diffs = np.diff(cum)
elbow = int(np.argmax(diffs < diffs[0]*0.02)) + 1
print(f"  first component contributing < 2 % of the first -> d ~ {elbow}")

# ============ 2. cross-validate d as a hyperparameter ==================
print("\\n=== but d is just a hyperparameter: cross-validate it ===")
grid = GridSearchCV(make_pipeline(StandardScaler(), PCA(), LogisticRegression(max_iter=2000)),
                    {"pca__n_components": [5, 10, 20, 30, 40, 55]},
                    cv=5, n_jobs=-1)
grid.fit(Xtr, ytr)
import pandas as pd
res = pd.DataFrame(grid.cv_results_)[["param_pca__n_components", "mean_test_score",
                                      "std_test_score"]]
print(res.round(4).to_string(index=False))
print(f"\\nbest d = {grid.best_params_['pca__n_components']}, "
      f"test accuracy {grid.score(Xte, yte):.4f}")
print(f"no PCA at all: "
      f"{make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).score(Xte, yte):.4f}")

# ============ 3. compression accounting ================================
print("\\n=== compression ===")
print(f"{'d':>5}{'EVR':>9}{'recon MSE':>12}{'floats stored':>16}{'ratio':>9}")
n = X.shape[1]; m = len(X)
for dd in [2, 8, 16, 32, 64]:
    p = PCA(n_components=dd).fit(X)
    Z = p.transform(X); Xb = p.inverse_transform(Z)
    stored = m*dd + dd*n + n
    print(f"{dd:>5}{p.explained_variance_ratio_.sum():>9.4f}"
          f"{np.mean((X-Xb)**2):>12.4f}{stored:>16,}{stored/(m*n):>9.2%}")
print(f"{'raw':>5}{1.0:>9.4f}{0.0:>12.4f}{m*n:>16,}{1.0:>9.2%}")

# ============ 4. PCA AS A DENOISER =====================================
print("\\n=== PCA removes noise because noise has low variance ===")
rng = np.random.default_rng(0)
X_noisy = np.clip(X + rng.normal(0, 4, X.shape), 0, 16)
print(f"{'d':>5}{'MSE(noisy, clean)':>20}{'MSE(denoised, clean)':>24}")
base_mse = np.mean((X_noisy - X)**2)
for dd in [5, 10, 20, 30, 64]:
    p = PCA(n_components=dd).fit(X_noisy)
    Xd_ = p.inverse_transform(p.transform(X_noisy))
    print(f"{dd:>5}{base_mse:>20.4f}{np.mean((Xd_ - X)**2):>24.4f}")
print("\\nAn intermediate d beats BOTH the noisy input and the full reconstruction:")
print("the discarded components were mostly noise. d=64 just reproduces the noise.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
p20 = PCA(n_components=20).fit(X_noisy)
Xden = p20.inverse_transform(p20.transform(X_noisy))
fig = make_subplots(rows=3, cols=8, row_titles=["clean", "noisy", "PCA d=20"])
for j in range(8):
    for r, M in enumerate([X, X_noisy, Xden]):
        fig.add_trace(go.Heatmap(z=M[j].reshape(8, 8)[::-1], zmin=0, zmax=16,
                                 colorscale=PARULA, showscale=False), r+1, j+1)
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(height=400, title="PCA as a denoiser")

fig2 = go.Figure()
fig2.add_scatter(y=np.cumsum(full.explained_variance_ratio_), mode="lines",
                 line=dict(color=C["primary"], width=3), name="cumulative EVR")
fig2.add_bar(y=full.explained_variance_ratio_, marker_color=C["accent"],
             name="per-component EVR")
for th in [.90, .95, .99]:
    fig2.add_hline(y=th, line_dash="dot", line_color=C["muted"],
                   annotation_text=f"{th:.0%}")
fig2.update_layout(height=400, xaxis_title="component", yaxis_title="variance ratio",
                   title="The scree plot")
''',
        key="ch08_choosed",
    )

    keypoints([
        "<code>PCA(n_components=0.95)</code> keeps 95 % of the variance "
        "automatically.",
        "$d = 2$ or $3$ for visualisation; the elbow for compression; "
        "cross-validation when it feeds a model.",
        "Reconstruction via <code>inverse_transform</code>; the error is the sum "
        "of the discarded eigenvalues.",
        "PCA <b>denoises</b> because noise spreads its variance thinly across all "
        "components.",
        "An intermediate $d$ can beat both the noisy input <i>and</i> the full "
        "reconstruction.",
    ])


# ==========================================================================
def s_8_5():
    section("8.5", "Randomized PCA and Incremental PCA")

    lead(
        "Two variants that make PCA usable at scale: one trades exactness for "
        "speed, the other trades a single pass for bounded memory."
    )

    sub("Randomized PCA")

    md(
        "Halko, Martinsson & Tropp's algorithm finds an approximation to the "
        "top-$d$ components without touching the full SVD:"
    )

    math(r"""
    \mathcal{O}\bigl(m \, d^{2}\bigr) + \mathcal{O}\bigl(d^{3}\bigr)
    \qquad\text{instead of}\qquad
    \mathcal{O}\bigl(m \, n^{2}\bigr) + \mathcal{O}\bigl(n^{3}\bigr)
    """)

    derive(
        [("The idea is to compress the column space with a random matrix first, "
          "then do an exact SVD on the small result.", None),
         ("<b>Step 1.</b> Draw a random Gaussian matrix "
          "$\\boldsymbol\\Omega \\in \\mathbb{R}^{n \\times (d+p)}$ (with a small "
          "oversampling $p \\approx 10$) and form the sketch:",
          r"\mathbf{Y} = \mathbf{X}_c\,\boldsymbol\Omega \;\in\; \mathbb{R}^{m \times (d+p)}"),
         ("<b>Step 2.</b> Orthonormalise it (QR decomposition) to get a basis "
          "$\\mathbf{Q}$ for a subspace that, with high probability, nearly "
          "contains the top-$d$ singular directions.", None),
         ("<b>Step 3.</b> Project onto that basis and take an <i>exact</i> SVD of "
          "the small matrix:",
          r"\mathbf{B} = \mathbf{Q}^\top \mathbf{X}_c \in \mathbb{R}^{(d+p) \times n}, "
          r"\qquad \mathbf{B} = \tilde{\mathbf{U}}\boldsymbol\Sigma\mathbf{V}^\top"),
         ("<b>Step 4.</b> Lift back: $\\mathbf{U} = \\mathbf{Q}\\tilde{\\mathbf{U}}$. "
          "The error bound is",
          r"\mathbb{E}\bigl\lVert \mathbf{X}_c - \mathbf{Q}\mathbf{Q}^\top\mathbf{X}_c \bigr\rVert "
          r"\;\le\; \left(1 + \sqrt{\tfrac{d}{p-1}}\right)\sigma_{d+1}"),
         ("So the error is a small multiple of the first <i>discarded</i> singular "
          "value — which is exactly the error you were going to accept anyway by "
          "truncating at $d$. That is why the approximation costs so little "
          "accuracy in practice.", None)],
        title="How randomized SVD works",
    )

    codenote(
        "svd_solver",
        "<code>PCA(svd_solver='auto')</code> — the default — picks "
        "<code>'randomized'</code> automatically when $\\max(m,n) > 500$ <i>and</i> "
        "$d < 0.8\\min(m,n)$; otherwise it uses the exact <code>'full'</code>. "
        "You can force either. <code>'covariance_eigh'</code> is fastest when "
        "$n \\ll m$.",
    )

    sub("Incremental PCA")

    md(
        "Full PCA requires the whole dataset in memory. **Incremental PCA** "
        "processes mini-batches and updates its estimate — enabling out-of-core "
        "PCA (§1.5) and online use."
    )

    table(
        ["", "<code>PCA</code>", "<code>IncrementalPCA</code>", "<code>PCA(svd_solver='randomized')</code>"],
        [["Needs all data in RAM", "✅ yes", "❌ no", "✅ yes"],
         ["Exact", "✅", "Approximate", "Approximate"],
         ["Complexity", "$\\mathcal{O}(mn^2 + n^3)$",
          "$\\mathcal{O}(mnd)$ streaming", "$\\mathcal{O}(md^2 + d^3)$"],
         ["<code>partial_fit</code>", "❌", "<b>✅</b>", "❌"],
         ["Works with <code>np.memmap</code>", "❌", "<b>✅</b>", "❌"],
         ["Use when", "$m, n$ modest",
          "Data does not fit in memory, or arrives as a stream",
          "$n$ large, $d \\ll n$"]],
    )

    anim_header("Incremental PCA converging batch by batch")
    md(
        "The components estimated from the first $k$ mini-batches, compared with "
        "the exact full-data components. The alignment metric is "
        "$|\\mathbf{w}_j^{\\text{inc}} \\cdot \\mathbf{w}_j^{\\text{exact}}|$, which "
        "is 1 for a perfect match."
    )

    from sklearn.decomposition import PCA, IncrementalPCA
    X, y, images = ds.digits()
    exact = PCA(n_components=10).fit(X)
    inc = IncrementalPCA(n_components=10)
    n_batches = 18
    aligns = []
    for k, batch in enumerate(np.array_split(X, n_batches)):
        inc.partial_fit(batch)
        a = [abs(float(inc.components_[j] @ exact.components_[j]))
             for j in range(10)]
        aligns.append(a)

    frames = []
    for k in range(len(aligns)):
        frames.append(go.Frame(name=str(k + 1), data=[
            go.Bar(x=[f"PC{j+1}" for j in range(10)], y=aligns[k],
                   marker=dict(color=[C["success"] if v > .95 else
                                      (C["warning"] if v > .8 else C["danger"])
                                      for v in aligns[k]])),
            go.Scatter(x=list(range(1, k + 2)),
                       y=[np.mean(a) for a in aligns[:k + 1]], mode="lines+markers",
                       line=dict(color=C["primary"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"after {k+1}/{n_batches} batches ({(k+1)*len(X)//n_batches} instances "
            f"seen)   ·   mean alignment = {np.mean(aligns[k]):.4f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("|incremental · exact| per component",
                                      "mean alignment over batches"))
    f.add_trace(go.Bar(x=[f"PC{j+1}" for j in range(10)], y=aligns[0],
                       showlegend=False, marker=dict(color=C["danger"])), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[np.mean(aligns[0])], mode="lines+markers",
                           showlegend=False,
                           line=dict(color=C["primary"], width=3)), 1, 2)
    f.update_yaxes(range=[0, 1.05], row=1, col=1)
    f.update_yaxes(range=[0, 1.05], row=1, col=2)
    f.update_xaxes(title_text="batches processed", row=1, col=2)
    f.update_layout(height=430, title="IncrementalPCA converging to the exact "
                                      "components")
    anim.animate(f, frames, duration=nav.anim_ms(400), slider_prefix="batch ")
    figure(f, "Leading components converge almost immediately; the trailing ones "
              "take longer, because they carry less variance to identify them by.")

    code_lab(
        "Benchmark the three solvers; do out-of-core PCA with memmap",
        '''import numpy as np, time, os, tempfile
from sklearn.decomposition import PCA, IncrementalPCA

rng = np.random.default_rng(0)

# ============ 1. exact vs randomized ===================================
print("=== full vs randomized SVD ===")
print(f"{'shape':>16}{'d':>5}{'full':>11}{'randomized':>14}{'speedup':>10}"
      f"{'EVR diff':>11}")
for m, n in [(1500, 200), (1500, 600), (3000, 1000)]:
    X = rng.normal(0, 1, (m, n)) @ rng.normal(0, 1, (n, n)) * .1
    d = 30
    t0 = time.perf_counter()
    pf = PCA(n_components=d, svd_solver="full").fit(X)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    pr = PCA(n_components=d, svd_solver="randomized", random_state=0).fit(X)
    t_rand = time.perf_counter() - t0
    diff = abs(pf.explained_variance_ratio_.sum() - pr.explained_variance_ratio_.sum())
    print(f"{f'{m}x{n}':>16}{d:>5}{t_full:>10.3f}s{t_rand:>13.3f}s"
          f"{t_full/t_rand:>9.1f}x{diff:>11.2e}")
print("Nearly identical explained variance, a fraction of the time.")

# ============ 2. how good is the approximation? ========================
print("\\n=== component alignment, exact vs randomized ===")
X = rng.normal(0, 1, (2000, 300)) @ rng.normal(0, 1, (300, 300)) * .1
pf = PCA(n_components=20, svd_solver="full").fit(X)
pr = PCA(n_components=20, svd_solver="randomized", random_state=0).fit(X)
al = [abs(float(pf.components_[j] @ pr.components_[j])) for j in range(20)]
print(f"  min |alignment| over 20 components = {min(al):.6f}")
print(f"  mean                                = {np.mean(al):.6f}")

# ============ 3. IncrementalPCA ========================================
print("\\n=== IncrementalPCA: bounded memory ===")
X = rng.normal(0, 1, (12_000, 200))
X = X @ rng.normal(0, 1, (200, 200)) * .1
d = 40

t0 = time.perf_counter(); full = PCA(n_components=d).fit(X)
t_full = time.perf_counter() - t0

t0 = time.perf_counter()
inc = IncrementalPCA(n_components=d, batch_size=500)
for batch in np.array_split(X, 24):
    inc.partial_fit(batch)
t_inc = time.perf_counter() - t0

print(f"full PCA        : {t_full:.3f}s   EVR {full.explained_variance_ratio_.sum():.6f}"
      f"   peak RAM ~ {X.nbytes/1e6:.0f} MB")
print(f"IncrementalPCA  : {t_inc:.3f}s   EVR {inc.explained_variance_ratio_.sum():.6f}"
      f"   peak RAM ~ {X[:500].nbytes/1e6:.1f} MB")
al = [abs(float(full.components_[j] @ inc.components_[j])) for j in range(d)]
print(f"component alignment: first 10 = {np.round(al[:10], 5)}")
print(f"                     mean     = {np.mean(al):.5f}")

# ============ 4. out-of-core with np.memmap ============================
print("\\n=== genuinely out-of-core PCA via np.memmap ===")
path = os.path.join(tempfile.gettempdir(), "mlplat_pca.dat")
mm = np.memmap(path, dtype="float32", mode="w+", shape=X.shape)
mm[:] = X.astype("float32")
mm.flush(); del mm

X_mm = np.memmap(path, dtype="float32", mode="r", shape=X.shape)
inc2 = IncrementalPCA(n_components=d, batch_size=1000)
inc2.fit(X_mm)                       # never loads the whole array
print(f"fitted from disk. EVR = {inc2.explained_variance_ratio_.sum():.6f}")
print(f"file on disk: {os.path.getsize(path)/1e6:.1f} MB, "
      f"batches held in RAM: {1000*200*4/1e6:.2f} MB")

# on Windows the file stays locked until the memmap object is released
X_mm._mmap.close()
del X_mm
try:
    os.remove(path)
    print("temporary file removed")
except OSError as e:
    print(f"(could not remove the temp file: {e})")

# ============ 5. which solver does 'auto' pick? ========================
print("\\n=== svd_solver='auto' decisions ===")
print(f"{'m':>7}{'n':>7}{'d':>5}   chosen solver")
for m_, n_, d_ in [(100, 50, 10), (2000, 50, 10), (2000, 1000, 10),
                   (2000, 1000, 900)]:
    Xa = rng.normal(0, 1, (m_, n_))
    p = PCA(n_components=d_, svd_solver="auto").fit(Xa)
    print(f"{m_:>7}{n_:>7}{d_:>5}   {p._fit_svd_solver}")
''',
        key="ch08_incremental",
    )

    keypoints([
        "<b>Randomized PCA</b>: sketch with a random matrix, then exact SVD on the "
        "small result — $\\mathcal{O}(md^2)$ instead of $\\mathcal{O}(mn^2)$.",
        "Its error is bounded by a small multiple of $\\sigma_{d+1}$ — the error "
        "you were accepting anyway.",
        "<b>IncrementalPCA</b>: <code>partial_fit</code> on mini-batches, bounded "
        "memory, works with <code>np.memmap</code>.",
        "<code>svd_solver='auto'</code> chooses sensibly; you rarely need to "
        "override it.",
        "Leading components converge first — trailing ones need more data to "
        "identify.",
    ])


# ==========================================================================
def s_8_6():
    section("8.6", "Random Projection")

    lead(
        "The most surprising result in this chapter: a **completely random** "
        "linear map preserves all pairwise distances, with high probability, "
        "provided you keep enough dimensions. And 'enough' does not depend on the "
        "original dimension at all."
    )

    sub("The Johnson–Lindenstrauss lemma")

    math(r"""
    (1 - \varepsilon)\,\bigl\lVert \mathbf{u} - \mathbf{v} \bigr\rVert^2
    \;\le\;
    \bigl\lVert f(\mathbf{u}) - f(\mathbf{v}) \bigr\rVert^2
    \;\le\;
    (1 + \varepsilon)\,\bigl\lVert \mathbf{u} - \mathbf{v} \bigr\rVert^2
    """)

    md("holds for all $\\binom{m}{2}$ pairs simultaneously, with high "
       "probability, as long as:")

    math(r"""
    d \;\ge\; \frac{4 \log(m)}{\dfrac{\varepsilon^2}{2} - \dfrac{\varepsilon^3}{3}}
    \;=\; \mathcal{O}\!\left(\frac{\log m}{\varepsilon^2}\right)
    """)
    where({r"m": "the number of <b>instances</b>",
           r"\varepsilon": "the distortion you are willing to tolerate",
           r"d": "the target dimension",
           r"n": "<b>does not appear</b> — that is the astonishing part"})

    proof(
        "d depends on m and ε, never on n",
        "The required dimension grows only <b>logarithmically in the number of "
        "instances</b> and is completely <b>independent of the original "
        "dimension</b>. You can project a million points from "
        "$\\mathbb{R}^{1\\,000\\,000}$ down to about 10 000 dimensions and every "
        "pairwise distance is preserved to within 10 %. The map is just a random "
        "Gaussian matrix — no data is examined at all, so fitting is "
        "instantaneous and can be done before you have even seen the data.",
    )

    anim_header("Distances before and after a random projection")
    md(
        "1 000 points in $\\mathbb{R}^{5000}$ projected to $d$ dimensions. Each "
        "frame shows the scatter of original vs projected pairwise distances. "
        "Watch the cloud tighten onto the diagonal as $d$ grows past the JL bound."
    )

    from sklearn.random_projection import (GaussianRandomProjection,
                                           johnson_lindenstrauss_min_dim)
    rng = np.random.default_rng(0)
    Xj = rng.normal(0, 1, (400, 3000))
    ii, jj = np.triu_indices(len(Xj), k=1)
    sel = rng.choice(len(ii), 3000, replace=False)
    ii, jj = ii[sel], jj[sel]
    d_orig = np.linalg.norm(Xj[ii] - Xj[jj], axis=1)

    d_list = [5, 10, 25, 50, 100, 250, 500, 1000, 2000]
    cache = []
    for d in d_list:
        P = GaussianRandomProjection(n_components=d, random_state=0)
        Z = P.fit_transform(Xj)
        d_new = np.linalg.norm(Z[ii] - Z[jj], axis=1)
        ratio = d_new / d_orig
        cache.append((d_new, float(np.abs(ratio - 1).max()),
                      float(np.abs(ratio - 1).mean())))

    lo, hi = d_orig.min() * .8, d_orig.max() * 1.15
    frames = []
    for k, d in enumerate(d_list):
        d_new, mx, mn = cache[k]
        eps = mx
        jl = johnson_lindenstrauss_min_dim(len(Xj), eps=min(max(eps, .01), .99))
        frames.append(go.Frame(name=str(d), data=[
            go.Scattergl(x=d_orig, y=d_new, mode="markers",
                         marker=dict(color=alpha(C["primary"], .35), size=3.5)),
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color=C["truth"], width=2, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"d = {d:>5}   ·   max distortion = {mx:.3f}   ·   "
            f"mean distortion = {mn:.4f}   ·   JL needs d ≥ {jl} for ε = {eps:.2f}",
            color=C["success"] if mx < .2 else C["warning"])])))

    f = go.Figure(data=[
        go.Scattergl(x=d_orig, y=cache[0][0], mode="markers", name="pairs",
                     marker=dict(color=alpha(C["primary"], .35), size=3.5)),
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect preservation",
                   line=dict(color=C["truth"], width=2, dash="dash")),
    ])
    f.update_layout(height=470, xaxis_title="original distance (in ℝ³⁰⁰⁰)",
                    yaxis_title="distance after random projection",
                    xaxis=dict(range=[lo, hi]), yaxis=dict(range=[lo, hi]),
                    title="Johnson–Lindenstrauss in action")
    anim.animate(f, frames, duration=nav.anim_ms(750), slider_prefix="d = ")
    figure(f)

    table(
        ["Class", "Projection matrix", "Density", "Speed"],
        [["<code>GaussianRandomProjection</code>",
          "$P_{ij} \\sim \\mathcal{N}(0, 1/d)$", "Dense", "Baseline"],
         ["<code>SparseRandomProjection</code>",
          "$P_{ij} \\in \\{-\\sqrt{s/d},\\, 0,\\, +\\sqrt{s/d}\\}$",
          "$1/\\sqrt{n}$ non-zero by default",
          "<b>Much faster</b>, less memory, same guarantee"]],
    )

    tip(
        "When to reach for random projection instead of PCA",
        "When $n$ is enormous and you cannot afford even a randomized SVD; when "
        "the data arrives as a stream and you need a <i>fixed</i> map decided in "
        "advance; when you need a distance-preservation guarantee rather than a "
        "variance-maximisation one; or when you want to project once and reuse the "
        "same matrix across datasets. PCA gives a better low-dimensional "
        "representation for the same $d$ — but it has to look at the data first.",
    )

    code_lab(
        "Johnson–Lindenstrauss, verified; and random projection vs PCA",
        '''import numpy as np, time
from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection,
                                       johnson_lindenstrauss_min_dim)
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ============ 1. the JL bound: d does NOT depend on n ==================
print("=== minimum d from the Johnson-Lindenstrauss lemma ===")
print(f"{'m instances':>13}" + "".join(f"{f'eps={e}':>12}" for e in [.5, .2, .1, .05]))
for m in [100, 1_000, 10_000, 1_000_000]:
    row = "".join(f"{johnson_lindenstrauss_min_dim(m, eps=e):>12,}"
                  for e in [.5, .2, .1, .05])
    print(f"{m:>13,}{row}")
print("\\nNotice: the ORIGINAL dimension n never appears. Only m and epsilon.")

# ============ 2. verify the guarantee empirically ======================
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (500, 5000))
ii, jj = np.triu_indices(len(X), k=1)
sel = rng.choice(len(ii), 5000, replace=False)
ii, jj = ii[sel], jj[sel]
d0 = np.linalg.norm(X[ii] - X[jj], axis=1)

print(f"\\n=== 500 points in R^5000, projected down ===")
print(f"{'d':>6}{'mean distortion':>18}{'max distortion':>17}{'>eps=0.1?':>12}")
for d in [10, 50, 100, 500, 1000, 3000]:
    Z = GaussianRandomProjection(n_components=d, random_state=0).fit_transform(X)
    r = np.linalg.norm(Z[ii] - Z[jj], axis=1) / d0
    viol = np.mean(np.abs(r - 1) > .1)
    print(f"{d:>6}{np.abs(r-1).mean():>18.5f}{np.abs(r-1).max():>17.5f}"
          f"{viol:>11.2%}")
jl = johnson_lindenstrauss_min_dim(500, eps=.1)
print(f"\\nJL says d >= {jl} guarantees max distortion <= 0.1 for all pairs.")

# ============ 3. dense vs sparse projection ============================
print("\\n=== Gaussian vs Sparse random projection ===")
X = rng.normal(0, 1, (2000, 2500))
print(f"{'method':<32}{'fit+transform':>16}{'matrix nnz':>14}")
for nm, P in [("GaussianRandomProjection", GaussianRandomProjection(500, random_state=0)),
              ("SparseRandomProjection",   SparseRandomProjection(500, random_state=0))]:
    t0 = time.perf_counter(); Z = P.fit_transform(X); dt = time.perf_counter()-t0
    M = P.components_
    nnz = M.nnz if hasattr(M, "nnz") else M.size
    print(f"{nm:<32}{dt:>15.3f}s{nnz:>14,}")

# ============ 4. random projection vs PCA on a real task ==============
d_ = load_digits()
Xd, yd = d_.data, d_.target
print(f"\\n=== digits (n=64): kNN accuracy after reduction ===")
print(f"{'d':>5}{'PCA':>10}{'Gaussian RP':>14}{'Sparse RP':>12}{'PCA fit':>11}{'RP fit':>10}")
for d in [2, 5, 10, 20, 40]:
    t0 = time.perf_counter()
    Zp = PCA(n_components=d, random_state=0).fit_transform(Xd)
    t_pca = time.perf_counter()-t0
    t0 = time.perf_counter()
    Zg = GaussianRandomProjection(d, random_state=0).fit_transform(Xd)
    t_rp = time.perf_counter()-t0
    Zs = SparseRandomProjection(d, random_state=0).fit_transform(Xd)
    a = [cross_val_score(KNeighborsClassifier(5), Z, yd, cv=3).mean()
         for Z in (Zp, Zg, Zs)]
    print(f"{d:>5}{a[0]:>10.4f}{a[1]:>14.4f}{a[2]:>12.4f}"
          f"{t_pca:>10.4f}s{t_rp:>9.4f}s")
print("\\nPCA wins on accuracy per dimension (it LOOKS at the data).")
print("Random projection wins on speed and needs no data to fit.")

# ============ 5. RP shines when n is huge ==============================
print("\\n=== when n is very large ===")
X_big = rng.normal(0, 1, (1500, 12_000))
t0 = time.perf_counter()
Zr = SparseRandomProjection(n_components=300, random_state=0).fit_transform(X_big)
t_rp = time.perf_counter()-t0
t0 = time.perf_counter()
Zp = PCA(n_components=300, svd_solver="randomized", random_state=0).fit_transform(X_big)
t_pca = time.perf_counter()-t0
print(f"n=12,000 -> d=300")
print(f"  SparseRandomProjection : {t_rp:.3f}s")
print(f"  randomized PCA         : {t_pca:.3f}s   ({t_pca/t_rp:.1f}x slower)")
''',
        key="ch08_rp",
    )

    keypoints([
        "Johnson–Lindenstrauss: a <b>random</b> linear map preserves all pairwise "
        "distances to within $\\varepsilon$.",
        "$d = \\mathcal{O}(\\log m / \\varepsilon^2)$ — depends on the number of "
        "<b>instances</b> and the tolerance, <b>never</b> on $n$.",
        "The map needs no data, so <code>fit</code> is instantaneous and can "
        "precede data collection.",
        "<code>SparseRandomProjection</code> is much faster and carries the same "
        "guarantee.",
        "PCA gives a better representation per dimension; random projection gives "
        "speed and a distance guarantee.",
    ])


# ==========================================================================
def s_8_7():
    section("8.7", "LLE, t-SNE, and Other Techniques")

    lead(
        "Non-linear methods. Each one preserves a different notion of structure, "
        "and knowing <i>which</i> is what stops you from over-reading a pretty "
        "picture."
    )

    sub("Locally Linear Embedding")

    md("LLE is a two-stage optimisation, and the second stage is the clever bit.")

    md("**Stage 1** — for each instance, find the weights that best reconstruct it "
       "from its $k$ nearest neighbours:")

    math(r"""
    \hat{\mathbf{W}} \;=\;
    \operatorname*{arg\,min}_{\mathbf{W}}\;
    \sum_{i=1}^{m}\Bigl\lVert \mathbf{x}^{(i)}
      - \sum_{j=1}^{m} w_{i,j}\,\mathbf{x}^{(j)} \Bigr\rVert^{2}
    """)
    math(r"""
    \text{subject to}\quad
    \begin{cases}
      w_{i,j} = 0 & \text{if } \mathbf{x}^{(j)} \text{ is not one of the } k
                    \text{ nearest neighbours of } \mathbf{x}^{(i)}\\[4pt]
      \displaystyle\sum_{j=1}^{m} w_{i,j} = 1 & \text{for every } i
    \end{cases}
    """)

    md("**Stage 2** — find low-dimensional points that satisfy the *same* "
       "reconstruction weights:")

    math(r"""
    \hat{\mathbf{Z}} \;=\;
    \operatorname*{arg\,min}_{\mathbf{Z}}\;
    \sum_{i=1}^{m}\Bigl\lVert \mathbf{z}^{(i)}
      - \sum_{j=1}^{m} \hat w_{i,j}\,\mathbf{z}^{(j)} \Bigr\rVert^{2}
    """)

    idea(
        "The weights are the manifold",
        "Stage 1 asks: <i>how does each point sit among its neighbours?</i> The "
        "answer, $\\hat{\\mathbf{W}}$, is a local description that is invariant to "
        "rotation, translation and scaling of each neighbourhood. Stage 2 then "
        "finds the lowest-dimensional arrangement that reproduces those same "
        "local relationships. LLE never looks at global distances at all — which "
        "is exactly why it can unroll a Swiss roll where PCA cannot.",
    )

    md(
        "**Cost:** $\\mathcal{O}(m\\log m \\cdot n \\log k)$ to find neighbours, "
        "$\\mathcal{O}(mnk^3)$ for the weights, and "
        "$\\mathcal{O}(dm^2)$ for the final eigenproblem. That last $m^2$ makes "
        "LLE impractical much beyond ~10 000 instances."
    )

    sub("The other techniques")

    table(
        ["Technique", "Preserves", "Linear?", "Cost", "Main use"],
        [["<b>PCA</b>", "Global variance / distances", "✅",
          "$\\mathcal{O}(mn^2)$", "The default first thing to try"],
         ["<b>Random projection</b>", "Pairwise distances (JL)", "✅",
          "$\\mathcal{O}(mnd)$", "Huge $n$; streaming"],
         ["<b>MDS</b>", "Pairwise distances exactly", "❌",
          "$\\mathcal{O}(m^3)$", "Small data with a known distance matrix"],
         ["<b>Isomap</b>", "<b>Geodesic</b> distances along the manifold", "❌",
          "$\\mathcal{O}(m^3)$", "Curved manifolds, global structure"],
         ["<b>LLE</b>", "Local linear reconstructions", "❌",
          "$\\mathcal{O}(dm^2)$", "Unrolling; noise-free manifolds"],
         ["<b>t-SNE</b>", "<b>Local</b> neighbourhoods only", "❌",
          "$\\mathcal{O}(m\\log m)$ (Barnes–Hut)", "<b>Visualisation</b> in 2-D/3-D"],
         ["<b>UMAP</b>", "Local + some global", "❌", "$\\mathcal{O}(m^{1.14})$",
          "Visualisation; faster than t-SNE"],
         ["<b>Kernel PCA</b>", "Variance in a kernel feature space", "❌",
          "$\\mathcal{O}(m^3)$", "Non-linear projection with a chosen kernel"],
         ["<b>LDA</b>", "<b>Class separation</b> (supervised!)", "✅",
          "$\\mathcal{O}(mn^2)$", "Preprocessing before a classifier"],
         ["<b>Autoencoders</b>", "Whatever the loss says (Ch. 17)", "❌",
          "Training-dependent", "Learned non-linear codes"]],
    )

    pitfall(
        "t-SNE plots are easy to over-read — three specific traps",
        "<b>(1) Cluster sizes are meaningless.</b> t-SNE expands dense clusters "
        "and contracts sparse ones, so a big blob is not a big cluster.<br>"
        "<b>(2) Distances between clusters are meaningless.</b> Two clusters "
        "appearing far apart may be adjacent in the original space. t-SNE "
        "optimises a <i>local</i> criterion only.<br>"
        "<b>(3) Perplexity changes everything.</b> The same data at perplexity 5 "
        "and 50 can produce completely different pictures — including clusters "
        "that are pure artefacts of the setting. Always run several "
        "perplexities.<br><br>"
        "t-SNE is a <b>visualisation</b> tool. Never feed its output into a "
        "downstream model, and never measure anything on the plot.",
    )

    anim_header("Six reductions of the same digits dataset")

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE, Isomap
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    Xd, yd, _ = ds.digits()
    sub_idx = np.random.default_rng(0).choice(len(Xd), 700, replace=False)
    Xs, ys = Xd[sub_idx], yd[sub_idx]

    @st.cache_data(show_spinner="Computing six embeddings…")
    def embeddings(Xs, ys):
        out = {}
        out["PCA (linear, global variance)"] = PCA(2, random_state=0).fit_transform(Xs)
        out["LDA (supervised, class separation)"] = \
            LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs, ys)
        out["Kernel PCA (RBF)"] = KernelPCA(2, kernel="rbf", gamma=.02,
                                            random_state=0).fit_transform(Xs)
        out["LLE (local reconstructions)"] = \
            LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                   random_state=0).fit_transform(Xs)
        out["Isomap (geodesic distances)"] = Isomap(n_components=2,
                                                    n_neighbors=12).fit_transform(Xs)
        out["t-SNE (local neighbourhoods)"] = \
            TSNE(n_components=2, perplexity=30, init="pca",
                 random_state=0).fit_transform(Xs)
        return out

    emb = embeddings(Xs, ys)

    def traces(Z):
        return [go.Scattergl(x=Z[ys == k, 0], y=Z[ys == k, 1], mode="markers",
                             name=str(k), showlegend=False,
                             marker=dict(color=SEQ[k % len(SEQ)], size=6,
                                         opacity=.8,
                                         line=dict(color="#fff", width=.5)))
                for k in range(10)]

    frames = [go.Frame(name=nm.split()[0], data=traces(Z),
                       layout=go.Layout(title=nm,
                                        xaxis=dict(autorange=True),
                                        yaxis=dict(autorange=True)))
              for nm, Z in emb.items()]

    first = list(emb)[0]
    f = go.Figure(data=traces(emb[first]))
    f.update_layout(height=520, title=first, xaxis_title="dim 1",
                    yaxis_title="dim 2")
    anim.animate(f, frames, duration=nav.anim_ms(1900), slider_prefix="method ")
    figure(f, "Colours are the true digit labels. Only LDA used them — the "
              "others found this structure unsupervised.")

    code_lab(
        "All the techniques, compared on one dataset",
        '''import numpy as np, time
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import (LocallyLinearEmbedding, TSNE, Isomap, MDS,
                              trustworthiness)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

d = load_digits()
rng = np.random.default_rng(0)
idx = rng.choice(len(d.data), 500, replace=False)
X, y = d.data[idx], d.target[idx]

# ============ compare every method =====================================
print("=== 64-D digits -> 2-D, eight ways ===")
print(f"{'method':<34}{'time':>9}{'trustworth.':>13}{'kNN acc':>10}")
methods = {
    "PCA":                 lambda: PCA(2, random_state=0).fit_transform(X),
    "Random projection":   lambda: GaussianRandomProjection(2, random_state=0).fit_transform(X),
    "LDA (supervised)":    lambda: LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y),
    "Kernel PCA (rbf)":    lambda: KernelPCA(2, kernel="rbf", gamma=.02, random_state=0).fit_transform(X),
    "LLE":                 lambda: LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                                          random_state=0).fit_transform(X),
    "Isomap":              lambda: Isomap(n_components=2, n_neighbors=12).fit_transform(X),
    "MDS (O(m^3), slowest)": lambda: MDS(2, random_state=0, n_init=1, max_iter=120,
                                     normalized_stress="auto").fit_transform(X),
    "t-SNE":               lambda: TSNE(2, perplexity=30, init="pca",
                                        random_state=0).fit_transform(X),
}
results = {}
for nm, fn in methods.items():
    t0 = time.perf_counter(); Z = fn(); dt = time.perf_counter()-t0
    results[nm] = Z
    tw = trustworthiness(X, Z, n_neighbors=12)
    acc = cross_val_score(KNeighborsClassifier(5), Z, y, cv=3).mean()
    print(f"{nm:<34}{dt:>8.2f}s{tw:>13.4f}{acc:>10.4f}")

print("\\ntrustworthiness measures whether NEIGHBOURS are preserved (1 = perfect).")
print("t-SNE scores highest -- it optimises exactly that. But see the caveat below.")

# ============ THE t-SNE PERPLEXITY TRAP ================================
print("\\n=== the same data, five perplexities ===")
print(f"{'perplexity':>12}{'trustworthiness':>18}{'kNN acc':>10}"
      f"{'mean cluster radius':>22}")
for perp in [2, 5, 30, 50]:
    Z = TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
    tw = trustworthiness(X, Z, n_neighbors=12)
    acc = cross_val_score(KNeighborsClassifier(5), Z, y, cv=3).mean()
    radii = [np.linalg.norm(Z[y==k] - Z[y==k].mean(0), axis=1).mean()
             for k in range(10)]
    print(f"{perp:>12}{tw:>18.4f}{acc:>10.4f}{np.mean(radii):>22.3f}")
print("The 'cluster sizes' change by an order of magnitude with perplexity alone.")
print("They carry NO information about the data. Never measure them.")

# ============ LLE unrolls, PCA does not ================================
print("\\n=== Swiss roll: which methods unroll it? ===")
Xs, t = make_swiss_roll(n_samples=800, noise=.3, random_state=0)
print(f"{'method':<24}{'corr(dim1, true position)':>28}")
for nm, fn in [("PCA",    lambda: PCA(2, random_state=0).fit_transform(Xs)),
               ("LLE",    lambda: LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                                         random_state=0).fit_transform(Xs)),
               ("Isomap", lambda: Isomap(n_components=2, n_neighbors=12).fit_transform(Xs)),
               ("t-SNE",  lambda: TSNE(2, perplexity=30, init="pca",
                                       random_state=0).fit_transform(Xs))]:
    Z = fn()
    c = max(abs(np.corrcoef(Z[:, 0], t)[0,1]), abs(np.corrcoef(Z[:, 1], t)[0,1]))
    print(f"{nm:<24}{c:>28.4f}")
print("|corr| near 1 means the method recovered the roll's true coordinate.")

# ============ LDA is SUPERVISED and it shows ===========================
print("\\n=== LDA uses the labels; the others do not ===")
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
print(f"LDA can produce at most n_classes-1 = {len(np.unique(y))-1} components")
print(f"explained variance ratio: {lda.explained_variance_ratio_[:2].round(4)}")
print("Never fit LDA on the test set -- it is a supervised transformer, so it")
print("belongs INSIDE the pipeline and inside the cross-validation fold.")
''',
        key="ch08_manifold",
    )

    quiz(
        "A t-SNE plot shows two tight clusters far apart and one large diffuse "
        "cluster. What can you legitimately conclude?",
        ["The diffuse cluster contains more points",
         "The two tight clusters are very dissimilar to each other",
         "Points within each cluster are near-neighbours in the original space",
         "All of the above"],
        2,
        "Only the third. t-SNE preserves local neighbourhoods and nothing else — "
        "cluster sizes and inter-cluster distances are artefacts of the "
        "optimisation, not properties of the data.",
        key="ch08q2",
    )

    keypoints([
        "LLE preserves <b>local linear reconstruction weights</b>, which is why "
        "it unrolls manifolds.",
        "Isomap preserves <b>geodesic</b> distances; MDS preserves Euclidean ones; "
        "t-SNE preserves <b>only neighbourhoods</b>.",
        "LDA is <b>supervised</b> — it maximises class separation, and it must "
        "live inside the pipeline.",
        "t-SNE cluster sizes and inter-cluster distances are <b>meaningless</b>; "
        "always vary the perplexity.",
        "Use t-SNE/UMAP for looking, PCA/LDA/random projection for feeding a "
        "model.",
    ])


# ==========================================================================
def s_8_8():
    section("8.8", "Exercises & Chapter Review")

    lead("Ten exercises. Numbers 9 and 10 are the ones that build intuition.")

    exercise(
        1, "What are the main motivations for reducing a dataset's "
        "dimensionality? What are the main drawbacks?",
        "**Motivations:**\n\n"
        "* **Speed up training** — often dramatically, and this is the most "
        "reliable benefit.\n"
        "* **Visualise** the data in 2-D or 3-D, which frequently reveals "
        "clusters or structure you would never have found numerically.\n"
        "* **Save space** — PCA compression of images and other dense data.\n"
        "* **Reduce noise** — the discarded low-variance components are often "
        "mostly noise (§8.4).\n"
        "* **Mitigate the curse of dimensionality** for distance-based methods.\n\n"
        "**Drawbacks:**\n\n"
        "* **Information is always lost**, which may degrade performance.\n"
        "* It **adds computational cost** to the pipeline.\n"
        "* The transformed features are **hard to interpret** — PC1 is a linear "
        "combination of everything.\n"
        "* It **adds complexity** to your pipeline: another fitted object that "
        "must be versioned, serialised and applied identically in production.")

    exercise(
        2, "What is the curse of dimensionality?",
        "The fact that many phenomena which behave one way in low-dimensional "
        "space behave completely differently in high-dimensional space. "
        "Concretely (§8.1): a random point in a 100-D unit cube is within 1 % of "
        "a face with probability 87 %; two random points are on average "
        "$\\sqrt{n/6}$ apart with vanishing relative variation, so all distances "
        "become the same; and covering the space at resolution $\\varepsilon$ "
        "requires $\\varepsilon^{-n}$ samples.\n\n"
        "The practical consequence is that high-dimensional datasets are "
        "extremely **sparse**, most training instances are far from each other, "
        "and predictions are extrapolations — so models risk overfitting badly "
        "unless you have exponentially more data.")

    exercise(
        3, "Once a dataset's dimensionality has been reduced, is it possible to "
        "reverse the operation? If so, how? If not, why?",
        "**Almost always no, not exactly** — the reduction discards information "
        "by construction.\n\n"
        "Some algorithms provide an approximate inverse: PCA's "
        "`inverse_transform` maps back to the original space, giving a point that "
        "is close to the original but not identical (the difference is exactly the "
        "discarded variance, §8.3). The same is true of random projection with "
        "the pseudoinverse, and of autoencoders (Chapter 17), whose decoder *is* "
        "the inverse map.\n\n"
        "Other algorithms — t-SNE, LLE, MDS, Isomap — do **not** provide any "
        "inverse transform at all. They compute an embedding of the training "
        "points and have no general mapping, in either direction.")

    exercise(
        4, "Can PCA be used to reduce the dimensionality of a highly nonlinear "
        "dataset?",
        "**Usually yes, if the goal is to remove useless dimensions** — but not "
        "if there are none to remove.\n\n"
        "PCA can reduce dimensionality of nonlinear data whenever some directions "
        "carry negligible variance (for example, the Swiss roll is a 2-D surface "
        "embedded in 3-D, and PCA can drop it to 2-D — it just does so by "
        "*squashing* rather than *unrolling*, which loses the structure).\n\n"
        "But if every dimension carries real variance — the classic example is a "
        "Swiss roll with points spread uniformly through its thickness — PCA will "
        "lose too much. Then you need manifold learning (§8.7) or kernel PCA.")

    exercise(
        5, "Suppose you perform PCA on a 1 000-dimensional dataset, setting the "
        "explained variance ratio to 95 %. How many dimensions will the resulting "
        "dataset have?",
        "**It depends entirely on the dataset — anything from 1 to 950.**\n\n"
        "* If the features are **perfectly correlated**, one component captures "
        "everything: $d = 1$.\n"
        "* If the features are **completely independent and identically "
        "distributed** (i.i.d. noise), every eigenvalue is equal, so you need "
        "950 components to reach 95 %.\n"
        "* Real data sits in between; MNIST at 95 % needs about 150 of 784.\n\n"
        "The only way to know is to plot the cumulative explained variance and "
        "look — which is why `PCA(n_components=0.95)` exists.")

    exercise(
        6, "In what cases would you use regular PCA, incremental PCA, randomized "
        "PCA, or random projection?",
        "* **Regular PCA** — the default. Use it when the dataset fits in memory "
        "and you want the exact solution.\n"
        "* **Incremental PCA** — when the dataset does **not** fit in memory, or "
        "when it arrives as a stream and you need online updates. It is slower "
        "than regular PCA and approximate, so do not use it unless you need it.\n"
        "* **Randomized PCA** — when you want to reduce dimensionality "
        "considerably ($d \\ll n$) and the dataset fits in memory. It is much "
        "faster than regular PCA (§8.5) at a negligible accuracy cost. It is "
        "already the default when `svd_solver='auto'` decides the shape warrants "
        "it.\n"
        "* **Random projection** — when the dimensionality is enormous, you need "
        "a distance-preservation guarantee rather than variance maximisation, or "
        "you need a fixed projection decided *before* seeing the data.")

    exercise(
        7, "How can you evaluate the performance of a dimensionality reduction "
        "algorithm on your dataset?",
        "Three approaches, in increasing order of practical relevance:\n\n"
        "**(1) Reconstruction error.** If the algorithm has an inverse transform, "
        "measure the mean squared distance between each original instance and its "
        "reconstruction. Low error means the reduction preserved the data. Only "
        "works for PCA, random projection and autoencoders.\n\n"
        "**(2) Trustworthiness / continuity.** `sklearn.manifold.trustworthiness` "
        "measures whether points that are neighbours in the embedding were also "
        "neighbours in the original space. This works for *every* method, "
        "including t-SNE and LLE.\n\n"
        "**(3) Downstream performance — the one that matters.** Put the reduction "
        "in a pipeline before your actual model and cross-validate. If the model "
        "performs just as well and trains faster, the reduction was a success. "
        "This is the only criterion that answers the question you actually care "
        "about.")

    exercise(
        8, "Does it make any sense to chain two different dimensionality "
        "reduction algorithms?",
        "**Yes, and it is a common and effective pattern.**\n\n"
        "The standard example is **PCA followed by t-SNE**: PCA quickly strips "
        "out the useless dimensions (784 → 50, say) at low cost, and t-SNE — "
        "which is $\\mathcal{O}(m\\log m)$ but with a large constant, and which "
        "degrades on very high-dimensional input — then does the slow, expensive "
        "non-linear work on a much smaller problem. The result is similar to "
        "running t-SNE alone but far faster. `TSNE(init='pca')` reflects this.\n\n"
        "A second example is **PCA followed by LLE**, for the same reason. In "
        "general: use a cheap linear method to remove the obvious redundancy, then "
        "an expensive non-linear method to find the structure.")

    exercise(
        9, "Load the MNIST dataset and split it into a training set and a test "
        "set. Train a random forest classifier on the dataset and time how long it "
        "takes, then evaluate the resulting model on the test set. Next, use PCA "
        "to reduce the dataset's dimensionality with an explained variance ratio "
        "of 95 %. Train a new random forest classifier on the reduced dataset and "
        "see how long it takes. Was training much faster? Next, evaluate the "
        "classifier on the test set. How does it compare to the previous "
        "classifier? Try again with an `SGDClassifier`. How much does PCA help "
        "now?",
        "This exercise has a **counter-intuitive result that is worth "
        "internalising**.\n\n"
        "| model | dimensions | train time | test accuracy |\n|---|---|---|---|\n"
        "| Random forest | 784 | ~35 s | ~0.9705 |\n"
        "| Random forest | 154 (PCA 95 %) | ~90 s | ~0.9481 |\n"
        "| SGDClassifier | 784 | ~60 s | ~0.8990 |\n"
        "| SGDClassifier | 154 (PCA 95 %) | ~20 s | ~0.8969 |\n\n"
        "**For the random forest, PCA makes training slower AND less accurate.** "
        "Slower because the reduced features are dense linear combinations, "
        "whereas raw MNIST pixels are mostly zero and the tree-splitting code "
        "exploits that sparsity; also, the informative splits on raw pixels are "
        "easy to find, whereas after rotation each split must search a dense "
        "continuous feature. Less accurate because the discarded 5 % contained "
        "genuine signal, and because trees are sensitive to axis orientation "
        "(§6.7) — PCA rotates the axes away from the pixel basis where the "
        "structure was aligned.\n\n"
        "**For the SGDClassifier, PCA gives a 3× speed-up for almost no accuracy "
        "loss** — the expected result. Linear models care about the number of "
        "features directly.\n\n"
        "The lesson: **dimensionality reduction does not universally speed things "
        "up, and it never universally improves accuracy.** Measure both.",
        code='''from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import time

mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10_000, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time(); rf.fit(X_train, y_train); print("RF raw:", time.time()-t0)
print("acc:", rf.score(X_test, y_test))

pca = PCA(n_components=0.95)
X_train_r = pca.fit_transform(X_train)
X_test_r = pca.transform(X_test)
print("dimensions kept:", pca.n_components_)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time(); rf2.fit(X_train_r, y_train); print("RF PCA:", time.time()-t0)
print("acc:", rf2.score(X_test_r, y_test))     # SLOWER and WORSE''')

    exercise(
        10, "Use t-SNE to reduce the first 5 000 images of the MNIST dataset down "
        "to 2 dimensions and plot the result using Matplotlib. You can use a "
        "scatterplot using 10 different colors to represent each image's target "
        "class. Alternatively, you can replace each dot in the scatterplot with "
        "the corresponding instance's class (a digit from 0 to 9), or even plot "
        "scaled-down versions of the digit images themselves. Next, try other "
        "dimensionality reduction algorithms such as PCA, LLE, or MDS and compare "
        "the resulting visualizations.",
        "t-SNE produces by far the clearest picture: ten well-separated blobs, "
        "with the classic confusions visible as bridges between them — 4/9 and "
        "3/5/8 touch, while 0, 1 and 6 sit apart.\n\n"
        "PCA on the same data produces a single overlapping smear, because the "
        "first two principal components capture only ~17 % of MNIST's variance "
        "and the classes are not linearly separable in that plane.\n\n"
        "LLE and MDS land in between; MDS is also prohibitively slow at "
        "$\\mathcal{O}(m^3)$ — 5 000 points is near its practical limit.\n\n"
        "The standard speed-up, per exercise 8, is **PCA to ~50 dimensions first, "
        "then t-SNE** — roughly 3× faster with an essentially identical plot. "
        "Remember the caveats of §8.7: read the neighbourhoods, ignore the blob "
        "sizes and the gaps between them.",
        code='''from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_sample, y_sample = X_train[:5000], y_train[:5000].astype(int)

# the standard two-stage speed-up
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_sample)
X_2d = TSNE(n_components=2, init="pca", random_state=42).fit_transform(X_pca)

plt.figure(figsize=(11, 9))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sample, cmap="tab10", s=8)
plt.colorbar(); plt.axis("off"); plt.title("t-SNE of 5,000 MNIST digits")

# or plot the digit itself instead of a dot
for i in range(0, len(X_2d), 25):
    plt.text(X_2d[i, 0], X_2d[i, 1], str(y_sample[i]),
             color=plt.cm.tab10(y_sample[i] / 10), fontsize=9)''')

    rule()

    keypoints([
        "The curse is real and computable: boundaries, distance concentration, "
        "vanishing sphere volume, exponential sample requirements.",
        "PCA = eigenvectors of $\\boldsymbol\\Sigma$; max variance ≡ min "
        "reconstruction error; compute by SVD.",
        "Johnson–Lindenstrauss: a random map preserves distances with "
        "$d = \\mathcal{O}(\\log m/\\varepsilon^2)$, independent of $n$.",
        "Manifold methods preserve <b>local</b> structure; t-SNE is for looking, "
        "not for modelling.",
        "Reduction reliably saves <b>time</b>, not accuracy — cross-validate "
        "<code>n_components</code>.",
    ], title="Chapter 8 in five lines")

    refs([
        ("Pearson, K. — *On Lines and Planes of Closest Fit to Systems of Points "
         "in Space* (PCA, 1901)",
         "https://doi.org/10.1080/14786440109462720"),
        ("Halko, Martinsson & Tropp — *Finding Structure with Randomness*",
         "https://doi.org/10.1137/090771806"),
        ("Johnson & Lindenstrauss — *Extensions of Lipschitz Mappings into a "
         "Hilbert Space*", "Contemporary Mathematics 26, 1984"),
        ("Roweis & Saul — *Nonlinear Dimensionality Reduction by Locally Linear "
         "Embedding*", "https://doi.org/10.1126/science.290.5500.2323"),
        ("van der Maaten & Hinton — *Visualizing Data using t-SNE*",
         "https://www.jmlr.org/papers/v9/vandermaaten08a.html"),
        ("Wattenberg, Viégas & Johnson — *How to Use t-SNE Effectively*",
         "https://distill.pub/2016/misread-tsne/"),
    ])


# ==========================================================================
SECTIONS = [
    ("8.1", "The Curse of Dimensionality", s_8_1),
    ("8.2", "Projection & Manifold Learning", s_8_2),
    ("8.3", "PCA — Preserving Variance", s_8_3),
    ("8.4", "Choosing d & Compression", s_8_4),
    ("8.5", "Randomized & Incremental PCA", s_8_5),
    ("8.6", "Random Projection", s_8_6),
    ("8.7", "LLE, t-SNE & Others", s_8_7),
    ("8.8", "Exercises & Review", s_8_8),
]

nav.render_chapter(CH, SECTIONS)
