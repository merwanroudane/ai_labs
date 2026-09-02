"""Chapter 9 — Unsupervised Learning Techniques."""

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
CH = "ch09"

hero(
    kicker="Part I · Chapter 9",
    title="Unsupervised Learning Techniques",
    blurb=(
        "The vast majority of data has no labels. This chapter covers what you can "
        "do anyway: <b>k</b>-means and its exact convergence proof, the "
        "silhouette machinery for choosing <b>k</b>, DBSCAN's density definition, "
        "Gaussian mixtures with the EM algorithm derived in full, and the "
        "anomaly-detection and semi-supervised tricks that make clustering pay for "
        "itself on labelled problems too."
    ),
    chips=["EM derived", "8 sub-sections", "9 animations",
           "9 code labs", "k-means · DBSCAN · GMM"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_9_1():
    section("9.1", "Clustering and the k-means Algorithm")

    lead(
        "Clustering is the task of grouping similar instances. Unlike "
        "classification, nobody tells you what the groups are — so the first "
        "question is always <i>what does 'similar' mean here?</i>"
    )

    sub("Applications")

    table(
        ["Application", "What clustering does"],
        [["<b>Customer segmentation</b>",
          "Groups customers by behaviour so each segment gets its own strategy"],
         ["<b>Data analysis</b>",
          "Cluster first, analyse each cluster separately — often far more "
          "informative than one global analysis"],
         ["<b>Dimensionality reduction</b>",
          "Replace $\\mathbf{x}$ with its vector of affinities to $k$ centroids: "
          "$n \\to k$ features (this is §2.4's <code>ClusterSimilarity</code>)"],
         ["<b>Feature engineering</b>",
          "The cluster id becomes a categorical feature for a supervised model"],
         ["<b>Anomaly detection</b>",
          "Instances with low affinity to every cluster are anomalies (§9.7)"],
         ["<b>Semi-supervised learning</b>",
          "Propagate a handful of labels through the clusters (§9.5)"],
         ["<b>Search engines</b>",
          "Cluster a corpus, return items from the query's cluster"],
         ["<b>Image segmentation</b>",
          "Cluster pixels by colour and replace each with its centroid (§9.4)"]],
    )

    sub("The k-means objective")

    math(r"""
    J\bigl(\{\boldsymbol\mu_k\}, \{c^{(i)}\}\bigr) \;=\;
    \sum_{i=1}^{m} \bigl\lVert \mathbf{x}^{(i)}
      - \boldsymbol\mu_{c^{(i)}} \bigr\rVert_2^{2}
    """)
    where({r"\boldsymbol\mu_k": "the centroid of cluster $k$",
           r"c^{(i)} \in \{1,\dots,k\}": "the cluster assigned to instance $i$",
           r"J": "the <b>inertia</b> — <code>kmeans.inertia_</code>"})

    sub("The algorithm — Lloyd's algorithm")

    md(
        """
1. Initialise $k$ centroids (see §9.2 for how).
2. **Assignment step:** assign each instance to its nearest centroid.
3. **Update step:** move each centroid to the mean of its assigned instances.
4. Repeat 2–3 until the assignments stop changing.
        """
    )

    derive(
        [("<b>Why does it converge?</b> Because each of the two steps can only "
          "decrease $J$, and there are finitely many possible assignments.", None),
         ("<b>Step 2 decreases J.</b> Holding the centroids fixed, $J$ is a sum of "
          "independent per-instance terms. Assigning $\\mathbf{x}^{(i)}$ to its "
          "<i>nearest</i> centroid minimises its own term:",
          r"c^{(i)} = \operatorname*{arg\,min}_{k}\bigl\lVert \mathbf{x}^{(i)} "
          r"- \boldsymbol\mu_k \bigr\rVert^2"),
         ("<b>Step 3 decreases J.</b> Holding the assignments fixed, $J$ splits "
          "into $k$ independent problems, one per cluster. Differentiate the "
          "$k$-th term:",
          r"\frac{\partial}{\partial \boldsymbol\mu_k}\sum_{i:\,c^{(i)}=k}"
          r"\bigl\lVert \mathbf{x}^{(i)} - \boldsymbol\mu_k\bigr\rVert^2 "
          r"= -2\sum_{i:\,c^{(i)}=k}\bigl(\mathbf{x}^{(i)} - \boldsymbol\mu_k\bigr) = \mathbf{0}"),
         ("Solving gives the mean — which is where the name comes from, and which "
          "is the same $\\ell_2$ argument as §2.1:",
          r"\boldsymbol\mu_k = \frac{1}{\bigl|\{i : c^{(i)} = k\}\bigr|}"
          r"\sum_{i:\,c^{(i)}=k}\mathbf{x}^{(i)}"),
         ("<b>Convergence.</b> $J$ is non-increasing and bounded below by 0. There "
          "are at most $k^m$ distinct assignments, so the algorithm cannot cycle "
          "and must reach a fixed point in finitely many steps. In practice that "
          "is usually fewer than 20 iterations.", None),
         ("<b>But it converges to a LOCAL minimum.</b> Finding the global optimum "
          "of $J$ is NP-hard (Aloise et al., 2009), so different initialisations "
          "give different — sometimes very different — answers. That is what "
          "§9.2's k-means++ and <code>n_init</code> are for.", None)],
        title="Why Lloyd's algorithm converges — and only to a local minimum",
    )

    anim_header("k-means, iteration by iteration")
    md(
        "Crosses are centroids, colours are assignments, and the Voronoi cells "
        "show the decision boundary. Each frame is one half-step: first the "
        "assignment, then the update. Watch the inertia fall monotonically."
    )

    from sklearn.cluster import KMeans

    Xb, yb = ds.blobs(n=400, centers=5, std=.9)
    rng = np.random.default_rng(1)
    k = 5
    cent = Xb[rng.choice(len(Xb), k, replace=False)].copy()

    g1 = np.linspace(Xb[:, 0].min() - 1, Xb[:, 0].max() + 1, 130)
    g2 = np.linspace(Xb[:, 1].min() - 1, Xb[:, 1].max() + 1, 130)
    G1, G2 = np.meshgrid(g1, g2); GG = np.c_[G1.ravel(), G2.ravel()]

    steps = []
    labels = np.zeros(len(Xb), int)
    for it in range(9):
        d = ((Xb[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        inertia = float(d.min(1).sum())
        steps.append(("assign", cent.copy(), labels.copy(), inertia))
        new = np.array([Xb[labels == j].mean(0) if (labels == j).any() else cent[j]
                        for j in range(k)])
        cent = new
        d2 = ((Xb[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        steps.append(("update", cent.copy(), labels.copy(),
                      float(d2[np.arange(len(Xb)), labels].sum())))

    cs = [[i / (k - 1), alpha(SEQ[i], .30)] for i in range(k)]
    frames = []
    for s, (kind, cc, ll, inertia) in enumerate(steps):
        vor = (((GG[:, None, :] - cc[None, :, :]) ** 2).sum(-1)
               .argmin(1).reshape(G1.shape).astype(float))
        frames.append(go.Frame(name=str(s + 1), data=[
            go.Contour(x=g1, y=g2, z=vor, showscale=False, colorscale=cs,
                       contours=dict(showlines=False)),
            go.Scatter(x=Xb[:, 0], y=Xb[:, 1], mode="markers",
                       marker=dict(color=[SEQ[j] for j in ll], size=6,
                                   line=dict(color="#fff", width=.6))),
            go.Scatter(x=cc[:, 0], y=cc[:, 1], mode="markers",
                       marker=dict(color=C["ink"], size=20, symbol="x-thin",
                                   line=dict(width=4, color=C["ink"]))),
            go.Scatter(x=list(range(1, s + 2)), y=[t[3] for t in steps[:s + 1]],
                       mode="lines+markers", line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"half-step {s+1}   ·   {kind.upper()} step   ·   "
            f"inertia J = {inertia:,.2f}",
            color=C["info"] if kind == "assign" else C["success"])])))

    f = make_subplots(rows=1, cols=2, column_widths=[.62, .38],
                      subplot_titles=("assignments and Voronoi cells",
                                      "inertia J (monotone decreasing)"))
    vor0 = (((GG[:, None, :] - steps[0][1][None, :, :]) ** 2).sum(-1)
            .argmin(1).reshape(G1.shape).astype(float))
    f.add_trace(go.Contour(x=g1, y=g2, z=vor0, showscale=False, colorscale=cs,
                           contours=dict(showlines=False)), 1, 1)
    f.add_trace(go.Scatter(x=Xb[:, 0], y=Xb[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=[SEQ[j] for j in steps[0][2]], size=6,
                                       line=dict(color="#fff", width=.6))), 1, 1)
    f.add_trace(go.Scatter(x=steps[0][1][:, 0], y=steps[0][1][:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color=C["ink"], size=20, symbol="x-thin",
                                       line=dict(width=4, color=C["ink"]))), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[steps[0][3]], mode="lines+markers",
                           showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.update_xaxes(title_text="half-step", row=1, col=2)
    f.update_yaxes(title_text="inertia", row=1, col=2)
    f.update_layout(height=490, title="Lloyd's algorithm")
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="half-step ")
    figure(f)

    sub("Hard vs soft clustering")

    table(
        ["", "Hard clustering", "Soft clustering"],
        [["Output", "One cluster id per instance",
          "A score / distance / probability per cluster"],
         ["scikit-learn", "<code>predict()</code>",
          "<code>transform()</code> — distances to every centroid"],
         ["Use", "Assignment", "As <b>features</b>: an $m \\times k$ matrix that "
          "is often a very effective non-linear representation"]],
    )

    idea(
        "kmeans.transform() is a dimensionality reduction",
        "It returns each instance's distance to all $k$ centroids: an "
        "$m \\times k$ matrix. If $k \\ll n$ that is a dimensionality reduction, "
        "and it is a <b>non-linear</b> one — often far more useful downstream than "
        "the raw features. This is exactly what §2.4's "
        "<code>ClusterSimilarity</code> transformer did with an RBF kernel on top.",
    )

    code_lab(
        "k-means from scratch, and the convergence proof in action",
        '''import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

X, y_true = make_blobs(n_samples=800, centers=5, cluster_std=.9, random_state=42)

# ============ Lloyd's algorithm, from scratch ==========================
def kmeans_scratch(X, k, seed=0, max_iter=100, tol=1e-9):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    history = []
    for it in range(max_iter):
        # --- assignment step: each point to its nearest centroid ---------
        d2 = ((X[:, None, :] - centroids[None, :, :])**2).sum(-1)
        labels = d2.argmin(1)
        J_after_assign = float(d2.min(1).sum())

        # --- update step: each centroid to the mean of its points --------
        new = np.array([X[labels == j].mean(0) if (labels == j).any()
                        else centroids[j] for j in range(k)])
        shift = float(np.linalg.norm(new - centroids))
        centroids = new
        d2b = ((X[:, None, :] - centroids[None, :, :])**2).sum(-1)
        J_after_update = float(d2b[np.arange(len(X)), labels].sum())
        history.append((it, J_after_assign, J_after_update, shift))
        if shift < tol:
            break
    return centroids, labels, history

cent, lab, hist = kmeans_scratch(X, 5, seed=0)
print(f"{'iter':>5}{'J after assign':>18}{'J after update':>18}{'centroid shift':>17}")
for it, ja, ju, sh in hist:
    print(f"{it:>5}{ja:>18.4f}{ju:>18.4f}{sh:>17.6f}")
print(f"\\nconverged in {len(hist)} iterations")

# ---- verify J never increases -----------------------------------------
seq = []
for it, ja, ju, _ in hist:
    seq += [ja, ju]
print(f"J is monotone non-increasing: {all(seq[i] >= seq[i+1] - 1e-9 for i in range(len(seq)-1))}")

sk = KMeans(n_clusters=5, n_init=1, init=X[np.random.default_rng(0)
            .choice(len(X), 5, replace=False)], random_state=0).fit(X)
print(f"\\nmy inertia      = {seq[-1]:.4f}")
print(f"sklearn inertia = {sk.inertia_:.4f}")

# ============ LOCAL MINIMA: the same data, 15 random starts ============
print("\\n=== k-means finds LOCAL minima ===")
inertias = []
for s in range(15):
    _, l, h = kmeans_scratch(X, 5, seed=s)
    inertias.append(h[-1][2])
inertias = np.array(inertias)
print(f"15 random initialisations -> inertia range "
      f"[{inertias.min():.1f}, {inertias.max():.1f}]")
print(f"  best  {inertias.min():.2f}")
print(f"  worst {inertias.max():.2f}   ({inertias.max()/inertias.min():.2f}x worse)")
print(f"  {np.sum(inertias > inertias.min()*1.01)}/15 runs landed in a WORSE optimum")

# ============ hard vs soft ============================================
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)
print(f"\\n=== hard vs soft ===")
print(f"predict() -> cluster ids     : shape {km.predict(X[:3]).shape}, "
      f"{km.predict(X[:3])}")
D = km.transform(X[:3])
print(f"transform() -> distances     : shape {D.shape}")
print(np.round(D, 3))
print("\\ntransform() gives an m x k feature matrix -- a non-linear reduction.")

# ============ using cluster distances as FEATURES ======================
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=1500, noise=.25, random_state=0)
print("\\n=== k-means distances as features for a LINEAR classifier ===")
print(f"  raw 2 features         : "
      f"{cross_val_score(LogisticRegression(), Xm, ym, cv=5).mean():.4f}")
for kk in [5, 15, 40]:
    pipe = make_pipeline(KMeans(n_clusters=kk, n_init=10, random_state=0),
                         LogisticRegression(max_iter=2000))
    print(f"  {kk:>2} cluster distances   : "
          f"{cross_val_score(pipe, Xm, ym, cv=5).mean():.4f}")
print("A linear model on cluster distances solves a non-linear problem.")
''',
        key="ch09_kmeans",
    )

    keypoints([
        "$k$-means minimises the <b>inertia</b> $J = \\sum_i \\lVert\\mathbf{x}^{(i)} "
        "- \\boldsymbol\\mu_{c^{(i)}}\\rVert^2$.",
        "Lloyd's algorithm alternates assignment and update; both steps decrease "
        "$J$, so it converges.",
        "It converges to a <b>local</b> minimum — the global optimum is NP-hard.",
        "<code>transform()</code> gives distances to all $k$ centroids: a "
        "non-linear dimensionality reduction and an excellent feature set.",
        "Clustering also powers segmentation, semi-supervision and anomaly "
        "detection — the rest of this chapter.",
    ])


# ==========================================================================
def s_9_2():
    section("9.2", "Centroid Initialisation and Accelerated k-means")

    lead(
        "Because Lloyd's algorithm only finds a local minimum, <i>where you "
        "start</i> matters enormously. k-means++ solves this so well that it is "
        "the default everywhere."
    )

    sub("The k-means++ initialisation")

    md(
        """
1. Choose the first centroid $\\boldsymbol\\mu_1$ uniformly at random from the
   data.
2. For $j = 2, \\dots, k$: choose $\\boldsymbol\\mu_j = \\mathbf{x}^{(i)}$ with
   probability proportional to its squared distance from the *nearest already
   chosen* centroid:
        """
    )

    math(r"""
    \Pr\bigl[\boldsymbol\mu_j = \mathbf{x}^{(i)}\bigr] \;=\;
    \frac{D\bigl(\mathbf{x}^{(i)}\bigr)^{2}}
         {\displaystyle\sum_{l=1}^{m} D\bigl(\mathbf{x}^{(l)}\bigr)^{2}},
    \qquad
    D(\mathbf{x}) = \min_{j' < j} \bigl\lVert \mathbf{x} - \boldsymbol\mu_{j'} \bigr\rVert
    """)

    proof(
        "k-means++ has a provable guarantee",
        "Arthur & Vassilvitskii (2006) proved that this initialisation alone — "
        "before a single Lloyd iteration — gives an expected inertia within "
        "$\\mathcal{O}(\\log k)$ of the global optimum: "
        "$\\mathbb{E}[J] \\le 8(\\ln k + 2)\\,J^{\\star}$. Plain random "
        "initialisation has <b>no</b> such bound — it can be arbitrarily bad. That "
        "is why <code>init='k-means++'</code> is the default and you should "
        "essentially never change it.",
    )

    idea(
        "The intuition is 'spread out'",
        "Squared distance weighting makes it overwhelmingly likely that each new "
        "centroid lands far from the existing ones — so you start with one "
        "centroid near each real cluster rather than three inside the same blob. "
        "Since Lloyd's algorithm can never merge or move centroids across a "
        "cluster boundary, that initial spread is decisive.",
    )

    anim_header("k-means++ picking centroids, one at a time")
    md(
        "The colour surface is the sampling probability $D(\\mathbf{x})^2$ — the "
        "brighter the region, the more likely the next centroid lands there. Watch "
        "the probability collapse around each newly chosen centroid."
    )

    Xb, yb = ds.blobs(n=500, centers=6, std=.75)
    rngp = np.random.default_rng(4)
    chosen = [Xb[rngp.integers(len(Xb))]]
    picks = [np.array(chosen)]
    for j in range(1, 6):
        D2 = np.min(((Xb[:, None, :] - np.array(chosen)[None, :, :]) ** 2).sum(-1),
                    axis=1)
        p = D2 / D2.sum()
        chosen.append(Xb[rngp.choice(len(Xb), p=p)])
        picks.append(np.array(chosen))

    g1 = np.linspace(Xb[:, 0].min() - 1, Xb[:, 0].max() + 1, 120)
    g2 = np.linspace(Xb[:, 1].min() - 1, Xb[:, 1].max() + 1, 120)
    G1, G2 = np.meshgrid(g1, g2); GG = np.c_[G1.ravel(), G2.ravel()]

    frames = []
    for j, cc in enumerate(picks):
        D2g = np.min(((GG[:, None, :] - cc[None, :, :]) ** 2).sum(-1), axis=1)
        frames.append(go.Frame(name=str(j + 1), data=[
            go.Heatmap(x=g1, y=g2, z=D2g.reshape(G1.shape), colorscale=nav.cscale(),
                       showscale=False, opacity=.75),
            go.Scatter(x=Xb[:, 0], y=Xb[:, 1], mode="markers",
                       marker=dict(color=alpha(C["ink"], .45), size=4)),
            go.Scatter(x=cc[:, 0], y=cc[:, 1], mode="markers",
                       marker=dict(color="#FFFFFF", size=20, symbol="x-thin",
                                   line=dict(width=4, color="#FFFFFF"))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"centroid {j+1} of 6 chosen   ·   next pick ∝ D(x)²",
            bg="rgba(255,255,255,.9)")])))

    D2g0 = np.min(((GG[:, None, :] - picks[0][None, :, :]) ** 2).sum(-1), axis=1)
    f = go.Figure(data=[
        go.Heatmap(x=g1, y=g2, z=D2g0.reshape(G1.shape), colorscale=nav.cscale(),
                   showscale=False, opacity=.75),
        go.Scatter(x=Xb[:, 0], y=Xb[:, 1], mode="markers", name="data",
                   marker=dict(color=alpha(C["ink"], .45), size=4)),
        go.Scatter(x=picks[0][:, 0], y=picks[0][:, 1], mode="markers",
                   name="chosen centroids",
                   marker=dict(color="#FFFFFF", size=20, symbol="x-thin",
                               line=dict(width=4, color="#FFFFFF"))),
    ])
    f.update_layout(height=490, title="k-means++ : sampling probability ∝ D(x)²",
                    xaxis_title="x₁", yaxis_title="x₂")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="centroid ")
    figure(f)

    sub("Accelerated and mini-batch k-means")

    table(
        ["Variant", "Idea", "When to use"],
        [["<b>Elkan's algorithm</b>",
          "Use the triangle inequality $\\lVert a - c\\rVert \\ge |\\lVert a-b\\rVert "
          "- \\lVert b-c\\rVert|$ to skip distance computations that cannot change "
          "the assignment",
          "Low-dimensional dense data. <code>algorithm='elkan'</code>"],
         ["<b>MiniBatchKMeans</b>",
          "Move centroids using a random mini-batch each iteration instead of the "
          "full dataset",
          "Huge datasets, or data that does not fit in memory "
          "(<code>partial_fit</code>)"],
         ["<b>n_init</b>",
          "Run the whole algorithm $n$ times and keep the lowest inertia",
          "Always. Default is 10; with k-means++ you can often use fewer"]],
    )

    warn(
        "Mini-batch trades inertia for speed",
        "<code>MiniBatchKMeans</code> is typically 3–10× faster but converges to a "
        "slightly worse inertia — usually a few percent. On a dataset where you "
        "would wait minutes for the exact version, that is an excellent trade; on "
        "one that takes a second, it is not.",
    )

    code_lab(
        "k-means++ vs random init, and the mini-batch trade-off",
        '''import numpy as np, time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=5000, centers=12, cluster_std=.7, n_features=2,
                  random_state=42)

# ============ 1. k-means++ from scratch ================================
def kmeans_pp_init(X, k, seed=0):
    rng = np.random.default_rng(seed)
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        D2 = np.min(((X[:, None, :] - np.array(centroids)[None, :, :])**2).sum(-1), 1)
        centroids.append(X[rng.choice(len(X), p=D2/D2.sum())])
    return np.array(centroids)

def random_init(X, k, seed=0):
    return X[np.random.default_rng(seed).choice(len(X), k, replace=False)]

print("=== inertia after initialisation ALONE (before any Lloyd step) ===")
print(f"{'seed':>5}{'random init':>15}{'k-means++ init':>17}")
for s in range(6):
    for nm, init in [("r", random_init(X, 12, s)), ("p", kmeans_pp_init(X, 12, s))]:
        d = ((X[:, None, :] - init[None, :, :])**2).sum(-1).min(1).sum()
        if nm == "r": r_ = d
        else: p_ = d
    print(f"{s:>5}{r_:>15,.0f}{p_:>17,.0f}")

print("\\n=== final inertia after full convergence, 20 seeds ===")
res = {}
for nm, init in [("random", "random"), ("k-means++", "k-means++")]:
    vals = [KMeans(n_clusters=12, init=init, n_init=1, random_state=s).fit(X).inertia_
            for s in range(20)]
    res[nm] = np.array(vals)
    print(f"  {nm:<12} best {min(vals):>10,.0f}   worst {max(vals):>10,.0f}   "
          f"sd {np.std(vals):>9,.1f}")
print(f"\\nrandom init lands in a bad optimum "
      f"{int(np.sum(res['random'] > res['random'].min()*1.02))}/20 times;")
print(f"k-means++ does so {int(np.sum(res['k-means++'] > res['k-means++'].min()*1.02))}/20 times.")

# ============ 2. n_init ================================================
print("\\n=== n_init: how many restarts do you need? ===")
print(f"{'n_init':>8}{'inertia':>13}{'time':>10}")
for ni in [1, 3, 10, 30]:
    t0 = time.perf_counter()
    km = KMeans(n_clusters=12, n_init=ni, random_state=0).fit(X)
    print(f"{ni:>8}{km.inertia_:>13,.0f}{time.perf_counter()-t0:>9.3f}s")

# ============ 3. algorithms ============================================
print("\\n=== lloyd vs elkan ===")
print(f"{'algorithm':>10}{'time':>10}{'inertia':>13}")
for algo in ["lloyd", "elkan"]:
    t0 = time.perf_counter()
    km = KMeans(n_clusters=12, algorithm=algo, n_init=3, random_state=0).fit(X)
    print(f"{algo:>10}{time.perf_counter()-t0:>9.3f}s{km.inertia_:>13,.0f}")

# ============ 4. MiniBatchKMeans =======================================
print("\\n=== full vs mini-batch, as m grows ===")
print(f"{'m':>9}{'KMeans time':>14}{'MiniBatch time':>17}{'speedup':>10}"
      f"{'inertia penalty':>18}")
for m in [10_000, 100_000, 400_000]:
    Xb, _ = make_blobs(n_samples=m, centers=12, cluster_std=.7, random_state=42)
    t0 = time.perf_counter()
    k1 = KMeans(n_clusters=12, n_init=3, random_state=0).fit(Xb)
    t_full = time.perf_counter()-t0
    t0 = time.perf_counter()
    k2 = MiniBatchKMeans(n_clusters=12, n_init=3, batch_size=1024,
                         random_state=0).fit(Xb)
    t_mb = time.perf_counter()-t0
    print(f"{m:>9,}{t_full:>13.3f}s{t_mb:>16.3f}s{t_full/t_mb:>9.1f}x"
          f"{(k2.inertia_/k1.inertia_-1):>17.2%}")

# ============ 5. out-of-core with partial_fit ==========================
print("\\n=== out-of-core clustering ===")
mb = MiniBatchKMeans(n_clusters=12, n_init=3, random_state=0)
rng = np.random.default_rng(0)
for i in range(50):                          # pretend each chunk comes off disk
    chunk, _ = make_blobs(n_samples=2000, centers=12, cluster_std=.7,
                          random_state=42)
    idx = rng.choice(len(chunk), 2000, replace=True)
    mb.partial_fit(chunk[idx])
print(f"streamed 100,000 instances, peak RAM = {2000*2*8/1e6:.3f} MB")
print(f"final inertia on a fresh sample: "
      f"{-mb.score(make_blobs(n_samples=5000, centers=12, cluster_std=.7, random_state=42)[0]):,.0f}")
''',
        key="ch09_init",
    )

    keypoints([
        "k-means only finds a local optimum, so <b>initialisation decides the "
        "answer</b>.",
        "k-means++ samples each new centroid with probability $\\propto D(x)^2$ — "
        "spreading them out.",
        "It has a provable $\\mathcal{O}(\\log k)$ guarantee; random init has none.",
        "<code>n_init</code> runs the whole thing several times and keeps the "
        "lowest inertia.",
        "<code>MiniBatchKMeans</code> is 3–10× faster with a few percent worse "
        "inertia, and supports <code>partial_fit</code>.",
    ])


# ==========================================================================
def s_9_3():
    section("9.3", "Finding the Optimal Number of Clusters")

    lead(
        "The hardest question in clustering, because there is often no correct "
        "answer. Three tools, in increasing order of usefulness."
    )

    sub("Tool 1 — inertia and the elbow (weak)")

    pitfall(
        "Inertia cannot be used directly to choose k",
        "$J$ is <b>monotonically decreasing</b> in $k$: more centroids always fit "
        "better, and $J = 0$ when $k = m$. So you cannot minimise it. The best you "
        "can do is look for an <b>elbow</b> — the point where the curve stops "
        "falling steeply — and that is a subjective judgement that frequently has "
        "no clear answer.",
    )

    sub("Tool 2 — the silhouette coefficient (much better)")

    md("For each instance $i$:")

    math(r"""
    s^{(i)} \;=\; \frac{b^{(i)} - a^{(i)}}{\max\bigl(a^{(i)},\, b^{(i)}\bigr)}
    \;\in\; [-1, 1]
    """)
    where({
        r"a^{(i)}": "the mean distance from $i$ to the <b>other instances in its "
                    "own cluster</b> (intra-cluster distance)",
        r"b^{(i)}": "the mean distance from $i$ to the instances of the "
                    "<b>nearest other cluster</b> (nearest-cluster distance)",
        r"s \approx +1": "well inside its own cluster",
        r"s \approx 0": "on the boundary between two clusters",
        r"s \approx -1": "probably assigned to the wrong cluster",
    })

    md("The **silhouette score** is the mean of $s^{(i)}$ over all instances — "
       "and unlike inertia it is *not* monotone in $k$, so you can maximise it.")

    sub("Tool 3 — the silhouette diagram (best)")

    md(
        "Plot every instance's $s^{(i)}$, sorted, grouped by cluster. This shows "
        "you *why* a $k$ scores well or badly:"
    )

    table(
        ["Pattern in the diagram", "Interpretation"],
        [["All knives roughly the same width",
          "Clusters are of comparable size — usually a good sign"],
         ["One knife far wider than the others",
          "That cluster absorbed several real groups, or the data is imbalanced"],
         ["Many bars below the dashed mean line",
          "That cluster is poorly separated — $k$ is probably too large"],
         ["Negative bars", "Those instances are in the wrong cluster"],
         ["A knife that is short and stubby",
          "A small, tight, well-separated cluster — often the most interesting one"]],
    )

    anim_header("k sweeping: inertia, silhouette, and the diagram together")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples

    Xb, yb = ds.blobs(n=600, centers=5, std=.85)
    ks = list(range(2, 11))
    cache = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xb)
        sil = silhouette_score(Xb, km.labels_)
        samp = silhouette_samples(Xb, km.labels_)
        cache.append((km.inertia_, sil, km.labels_, samp, km.cluster_centers_))
    best_k = ks[int(np.argmax([c[1] for c in cache]))]

    def knife_traces(labels, samp, k):
        traces, y0 = [], 0
        for j in range(k):
            vals = np.sort(samp[labels == j])
            ys = np.arange(y0, y0 + len(vals))
            traces.append(go.Scatter(x=vals, y=ys, mode="lines", fill="tozerox",
                                     fillcolor=alpha(SEQ[j % len(SEQ)], .75),
                                     line=dict(color=SEQ[j % len(SEQ)], width=1),
                                     showlegend=False, hoverinfo="skip"))
            y0 += len(vals) + 12
        return traces

    frames = []
    for i, k in enumerate(ks):
        inertia, sil, labels, samp, cc = cache[i]
        col = C["success"] if k == best_k else C["primary"]
        data = [
            go.Scatter(x=ks[:i + 1], y=[c[0] for c in cache[:i + 1]],
                       mode="lines+markers", line=dict(color=C["danger"], width=3)),
            go.Scatter(x=ks[:i + 1], y=[c[1] for c in cache[:i + 1]],
                       mode="lines+markers", line=dict(color=C["success"], width=3)),
        ]
        data += knife_traces(labels, samp, k)
        data.append(go.Scatter(x=[sil, sil], y=[0, len(samp) + 12 * k],
                               mode="lines",
                               line=dict(color=C["danger"], width=2, dash="dash"),
                               showlegend=False))
        frames.append(go.Frame(name=str(k), data=data, layout=go.Layout(
            annotations=[anim.annotate_step(
                f"k = {k}   ·   inertia = {inertia:,.0f}   ·   "
                f"silhouette = {sil:.4f}"
                + ("   ← BEST" if k == best_k else ""), color=col)])))

    f = make_subplots(rows=1, cols=2, column_widths=[.44, .56],
                      subplot_titles=("inertia (red) and silhouette (green)",
                                      "silhouette diagram"),
                      specs=[[{"secondary_y": True}, {}]])
    f.add_trace(go.Scatter(x=ks[:1], y=[cache[0][0]], mode="lines+markers",
                           name="inertia", line=dict(color=C["danger"], width=3)),
                1, 1, secondary_y=False)
    f.add_trace(go.Scatter(x=ks[:1], y=[cache[0][1]], mode="lines+markers",
                           name="silhouette",
                           line=dict(color=C["success"], width=3)),
                1, 1, secondary_y=True)
    for tr in knife_traces(cache[0][2], cache[0][3], ks[0]):
        f.add_trace(tr, 1, 2)
    f.add_trace(go.Scatter(x=[cache[0][1]] * 2, y=[0, len(Xb)], mode="lines",
                           showlegend=False,
                           line=dict(color=C["danger"], width=2, dash="dash")), 1, 2)
    f.update_xaxes(title_text="k", range=[1.5, 10.5], row=1, col=1)
    f.update_xaxes(title_text="silhouette coefficient", range=[-.35, 1],
                   row=1, col=2)
    f.update_yaxes(visible=False, row=1, col=2)
    f.update_layout(height=490, title="Choosing k")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="k = ")
    figure(f, "Inertia falls forever; the silhouette peaks at the true k = 5.")

    codenote(
        "Silhouette is $\\mathcal{O}(m^2)$",
        "It computes all pairwise distances, so it becomes expensive above a few "
        "thousand instances. Subsample with "
        "<code>silhouette_score(X, labels, sample_size=5000)</code>. Two cheaper "
        "alternatives: the <b>Calinski–Harabasz</b> index (ratio of between- to "
        "within-cluster dispersion, $\\mathcal{O}(m)$) and the "
        "<b>Davies–Bouldin</b> index (lower is better, also cheap).",
    )

    code_lab(
        "Every method for choosing k, compared",
        '''import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score)

X, y_true = make_blobs(n_samples=1200, centers=5, cluster_std=.85, random_state=42)
print(f"the TRUE number of clusters is 5\\n")

print(f"{'k':>4}{'inertia':>12}{'silhouette':>13}{'Calinski-H':>13}"
      f"{'Davies-B':>11}{'ARI vs truth':>14}")
rows = []
for k in range(2, 12):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    ch  = calinski_harabasz_score(X, km.labels_)
    db  = davies_bouldin_score(X, km.labels_)
    ari = adjusted_rand_score(y_true, km.labels_)
    rows.append((k, km.inertia_, sil, ch, db, ari))
    print(f"{k:>4}{km.inertia_:>12,.0f}{sil:>13.4f}{ch:>13.1f}{db:>11.4f}{ari:>14.4f}")

import numpy as np
arr = np.array(rows)
print(f"\\nbest by silhouette      : k = {int(arr[arr[:,2].argmax(), 0])}  (maximise)")
print(f"best by Calinski-Harabasz: k = {int(arr[arr[:,3].argmax(), 0])}  (maximise)")
print(f"best by Davies-Bouldin   : k = {int(arr[arr[:,4].argmin(), 0])}  (minimise)")
print(f"best by ARI (cheating)   : k = {int(arr[arr[:,5].argmax(), 0])}")
print(f"inertia                  : monotone decreasing -> USELESS on its own")

# ---- the elbow, quantified via the second difference ------------------
inertias = arr[:, 1]
d2 = np.diff(inertias, 2)
print(f"\\nelbow by max second difference: k = {int(arr[np.argmax(d2)+1, 0])}")

# ============ silhouette diagram data ==================================
print("\\n=== per-cluster silhouette breakdown at k=5 and k=7 ===")
for k in [5, 7]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    s = silhouette_samples(X, km.labels_)
    print(f"\\nk = {k}   overall silhouette = {s.mean():.4f}")
    print(f"  {'cluster':>8}{'size':>7}{'mean s':>9}{'min s':>9}{'% negative':>12}")
    for j in range(k):
        sj = s[km.labels_ == j]
        print(f"  {j:>8}{len(sj):>7}{sj.mean():>9.4f}{sj.min():>9.4f}"
              f"{np.mean(sj < 0):>12.1%}")

# ============ when NO k is right: k-means on moons =====================
print("\\n=== k-means on non-spherical data: every k is bad ===")
Xm, ym = make_moons(n_samples=1000, noise=.06, random_state=0)
print(f"{'k':>4}{'silhouette':>13}{'ARI vs the 2 true moons':>27}")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xm)
    print(f"{k:>4}{silhouette_score(Xm, km.labels_):>13.4f}"
          f"{adjusted_rand_score(ym, km.labels_):>27.4f}")
print("\\nThe silhouette prefers k=2 but the ARI is terrible -- k-means cuts each")
print("moon in half rather than separating them. No k fixes this; see 9.6.")

# ============ silhouette cost ==========================================
import time
print("\\n=== silhouette is O(m^2) ===")
for m in [1000, 4000, 12000]:
    Xt, _ = make_blobs(n_samples=m, centers=5, random_state=0)
    lab = KMeans(5, n_init=3, random_state=0).fit_predict(Xt)
    t0 = time.perf_counter(); silhouette_score(Xt, lab); t_full = time.perf_counter()-t0
    t0 = time.perf_counter(); silhouette_score(Xt, lab, sample_size=1000,
                                               random_state=0)
    t_samp = time.perf_counter()-t0
    t0 = time.perf_counter(); calinski_harabasz_score(Xt, lab)
    t_ch = time.perf_counter()-t0
    print(f"  m={m:>6,}: full {t_full:>7.3f}s   sampled {t_samp:>7.3f}s   "
          f"Calinski-H {t_ch:>7.4f}s")
''',
        key="ch09_choosek",
    )

    quiz(
        "You plot inertia against $k$ and it keeps falling all the way to $k = 50$. "
        "What does that tell you about the right number of clusters?",
        ["There are 50 clusters", "Nothing — inertia always falls with $k$",
         "There are no clusters", "You should use $k = 50$"],
        1,
        "Inertia is monotonically decreasing by construction, reaching zero at "
        "$k = m$. Only its *shape* (the elbow) carries any information, and the "
        "silhouette score is a far better tool.",
        key="ch09q1",
    )

    keypoints([
        "Inertia falls monotonically in $k$ — you can only look for an elbow, and "
        "often there isn't one.",
        "Silhouette $s = (b-a)/\\max(a,b) \\in [-1,1]$; the mean is <b>not</b> "
        "monotone, so maximise it.",
        "The <b>silhouette diagram</b> shows per-cluster quality and is the most "
        "informative single plot.",
        "Silhouette is $\\mathcal{O}(m^2)$ — subsample, or use Calinski–Harabasz "
        "/ Davies–Bouldin.",
        "If every $k$ scores badly, the problem is the <b>shape</b> of the "
        "clusters, not the count.",
    ])


# ==========================================================================
def s_9_4():
    section("9.4", "Limits of k-means, and Image Segmentation")

    lead(
        "k-means is fast and scalable, but it makes three strong geometric "
        "assumptions. Knowing them tells you exactly when to reach for something "
        "else."
    )

    sub("The three assumptions")

    table(
        ["Assumption", "What breaks it", "The symptom"],
        [["Clusters are <b>spherical</b>",
          "Elongated or anisotropic clusters",
          "k-means cuts across the long axis instead of along it"],
         ["Clusters have <b>similar diameters</b>",
          "One large diffuse cluster and one tight one",
          "The large cluster gets split; the tight one absorbs its neighbours"],
         ["Clusters have <b>similar densities</b>",
          "Varying density",
          "Sparse clusters get absorbed by dense ones"],
         ["(Implicit) $k$ is <b>known</b>", "It usually isn't", "See §9.3"]],
    )

    md(
        "The root cause is that inertia is a **sum of squared Euclidean "
        "distances**, so the decision boundaries are always the perpendicular "
        "bisectors between centroids — a Voronoi diagram. Voronoi cells are "
        "convex polytopes. **k-means cannot produce a non-convex cluster, ever.**"
    )

    anim_header("Five datasets where k-means fails — and what does work")

    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    Xa, ya = ds.anisotropic_blobs(n=500)
    Xm, ym = ds.moons(n=500, noise=.06)
    Xc, yc = ds.circles(n=500, noise=.05, factor=.45)
    Xv = np.r_[rng.normal([0, 0], .35, (250, 2)),
               rng.normal([3.2, 0], 1.3, (250, 2))]
    yv = np.r_[np.zeros(250), np.ones(250)]
    Xu = np.r_[rng.normal([0, 0], .5, (450, 2)),
               rng.normal([3.5, 3.5], .5, (50, 2))]
    yu = np.r_[np.zeros(450), np.ones(50)]

    cases = [("anisotropic (elongated) blobs", Xa, 3),
             ("two moons (non-convex)", Xm, 2),
             ("concentric circles (non-convex)", Xc, 2),
             ("different variances", Xv, 2),
             ("very different sizes", Xu, 2)]

    frames = []
    for nm, Xd, kk in cases:
        Xs = StandardScaler().fit_transform(Xd)
        km_lab = KMeans(kk, n_init=10, random_state=0).fit_predict(Xs)
        gm_lab = GaussianMixture(kk, n_init=5, random_state=0).fit_predict(Xs)
        db_lab = DBSCAN(eps=.3, min_samples=6).fit_predict(Xs)
        frames.append(go.Frame(name=nm.split()[0], data=[
            go.Scattergl(x=Xs[:, 0], y=Xs[:, 1], mode="markers",
                         marker=dict(color=[SEQ[j % len(SEQ)] for j in km_lab],
                                     size=6, line=dict(color="#fff", width=.5))),
            go.Scattergl(x=Xs[:, 0], y=Xs[:, 1], mode="markers",
                         marker=dict(color=[SEQ[j % len(SEQ)] for j in gm_lab],
                                     size=6, line=dict(color="#fff", width=.5))),
            go.Scattergl(x=Xs[:, 0], y=Xs[:, 1], mode="markers",
                         marker=dict(color=[C["muted"] if j < 0
                                            else SEQ[j % len(SEQ)] for j in db_lab],
                                     size=6, line=dict(color="#fff", width=.5))),
        ], layout=go.Layout(title=nm)))

    Xs0 = StandardScaler().fit_transform(cases[0][1])
    f = make_subplots(rows=1, cols=3,
                      subplot_titles=("k-means", "Gaussian mixture (§9.7)",
                                      "DBSCAN (§9.6)"))
    for c in (1, 2, 3):
        f.add_trace(go.Scattergl(x=Xs0[:, 0], y=Xs0[:, 1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color=C["train"], size=6,
                                             line=dict(color="#fff", width=.5))),
                    1, c)
    f.update_layout(height=400, title=cases[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1800), slider_prefix="dataset ")
    figure(f, "Grey points in the DBSCAN panel are labelled as noise. Note that "
              "no single algorithm wins on every dataset.")

    tip(
        "Scaling matters here too",
        "It is important to <b>scale the input features</b> before running "
        "k-means, or the clusters may be very stretched along the large-scale "
        "axis and the result will be poor. This does not guarantee spherical "
        "clusters, but it removes the most common artificial cause of "
        "elongation.",
    )

    sub("Image segmentation with k-means")

    md(
        "**Colour segmentation** treats each pixel as a point in RGB space, "
        "clusters them, and replaces each pixel with its cluster centroid. The "
        "result is the image rendered in $k$ colours — which is simultaneously a "
        "segmentation and a compression."
    )

    math(r"""
    \text{pixel } (r, g, b) \in [0,1]^3
    \;\;\longrightarrow\;\;
    \boldsymbol\mu_{c(r,g,b)}
    \qquad
    \text{compression: } 24 \text{ bits} \to \lceil \log_2 k \rceil \text{ bits}
    """)

    anim_header("Colour quantisation: k from 2 to 32")

    @st.cache_data(show_spinner="Segmenting…")
    def segment_demo():
        rr = np.random.default_rng(0)
        H = W = 96
        yy, xx = np.mgrid[0:H, 0:W] / H
        img = np.zeros((H, W, 3))
        img[..., 0] = .35 + .45 * np.sin(6 * xx) * np.cos(4 * yy)
        img[..., 1] = .45 + .40 * np.cos(5 * yy + 1.1)
        img[..., 2] = .30 + .50 * np.sin(4 * xx + 2.2) ** 2
        disc = ((xx - .32) ** 2 + (yy - .35) ** 2) < .028
        img[disc] = [.93, .22, .18]
        band = np.abs(yy - .78) < .07
        img[band] = [.15, .78, .55]
        img = np.clip(img + rr.normal(0, .02, img.shape), 0, 1)
        flat = img.reshape(-1, 3)
        out = {}
        for k in [2, 3, 4, 6, 8, 12, 16, 32]:
            km = KMeans(k, n_init=5, random_state=0).fit(flat)
            out[k] = (km.cluster_centers_[km.labels_].reshape(img.shape),
                      float(km.inertia_))
        return img, out

    img, segs = segment_demo()

    def rgb_to_uint8(a):
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    frames = []
    for k, (seg, inert) in segs.items():
        bits = int(np.ceil(np.log2(k)))
        frames.append(go.Frame(name=str(k), data=[
            go.Image(z=rgb_to_uint8(seg)),
            go.Image(z=rgb_to_uint8(img)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"k = {k:>2} colours   ·   {bits} bits/pixel instead of 24   ·   "
            f"compression {bits/24:.1%}   ·   inertia = {inert:.1f}")])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("segmented (k colours)", "original"))
    f.add_trace(go.Image(z=rgb_to_uint8(list(segs.values())[0][0])), 1, 1)
    f.add_trace(go.Image(z=rgb_to_uint8(img)), 1, 2)
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    f.update_layout(height=420, title="k-means colour quantisation")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="k = ")
    figure(f)

    warn(
        "Colour segmentation is not semantic segmentation",
        "k-means groups pixels by <b>colour</b>, nothing else. It has no notion of "
        "objects, edges or spatial adjacency — two pixels on opposite sides of the "
        "image with the same colour land in the same cluster. For real semantic "
        "segmentation you need a convolutional network (Chapter 14). Adding the "
        "pixel coordinates $(x, y)$ as two extra features is a cheap partial "
        "fix that at least encourages spatial coherence.",
    )

    code_lab(
        "Where k-means fails, and colour quantisation",
        '''import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

rng = np.random.default_rng(0)

# ============ five failure modes =======================================
def anisotropic(n=600):
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=.7, random_state=0)
    return X @ np.array([[.6, -.63], [-.41, .85]]), y

def different_var(n=600):
    return (np.r_[rng.normal([0,0], .3, (n//2,2)), rng.normal([3,0], 1.4, (n//2,2))],
            np.r_[np.zeros(n//2), np.ones(n//2)])

def different_size(n=600):
    return (np.r_[rng.normal([0,0], .5, (int(n*.9),2)),
                  rng.normal([3.5,3.5], .5, (n-int(n*.9),2))],
            np.r_[np.zeros(int(n*.9)), np.ones(n-int(n*.9))])

datasets = {
    "spherical blobs (the ideal)": make_blobs(n_samples=600, centers=3,
                                              cluster_std=.8, random_state=0),
    "anisotropic blobs":           anisotropic(),
    "two moons":                   make_moons(n_samples=600, noise=.06, random_state=0),
    "concentric circles":          make_circles(n_samples=600, noise=.05,
                                                factor=.4, random_state=0),
    "different variances":         different_var(),
    "different sizes":             different_size(),
}

print(f"{'dataset':<30}{'k-means':>10}{'GMM':>9}{'DBSCAN':>9}{'spectral':>11}"
      f"{'agglom':>9}")
print(f"{'':<30}{'(ARI, higher is better)':>48}")
for nm, (X, y) in datasets.items():
    Xs = StandardScaler().fit_transform(X)
    k = len(np.unique(y))
    scores = []
    scores.append(adjusted_rand_score(y, KMeans(k, n_init=10, random_state=0).fit_predict(Xs)))
    scores.append(adjusted_rand_score(y, GaussianMixture(k, n_init=5, random_state=0).fit_predict(Xs)))
    scores.append(adjusted_rand_score(y, DBSCAN(eps=.3, min_samples=6).fit_predict(Xs)))
    scores.append(adjusted_rand_score(y, SpectralClustering(k, affinity="nearest_neighbors",
                                                            random_state=0).fit_predict(Xs)))
    scores.append(adjusted_rand_score(y, AgglomerativeClustering(k, linkage="single").fit_predict(Xs)))
    print(f"{nm:<30}" + "".join(f"{s:>10.3f}" for s in scores[:1])
          + "".join(f"{s:>9.3f}" for s in scores[1:3])
          + f"{scores[3]:>11.3f}{scores[4]:>9.3f}")

print("\\nk-means only wins on the first row. Every other row breaks one of its")
print("three assumptions: spherical, equal diameter, equal density.")

# ============ scaling matters ==========================================
print("\\n=== scaling before k-means ===")
X, y = make_blobs(n_samples=800, centers=4, cluster_std=1.0, random_state=0)
X_stretched = X.copy(); X_stretched[:, 1] *= 30
print(f"  unscaled, stretched : ARI = "
      f"{adjusted_rand_score(y, KMeans(4, n_init=10, random_state=0).fit_predict(X_stretched)):.4f}")
print(f"  after StandardScaler: ARI = "
      f"{adjusted_rand_score(y, KMeans(4, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X_stretched))):.4f}")

# ============ colour quantisation ======================================
print("\\n=== colour quantisation as compression ===")
H = W = 128
yy, xx = np.mgrid[0:H, 0:W] / H
img = np.stack([.35 + .45*np.sin(6*xx)*np.cos(4*yy),
                .45 + .40*np.cos(5*yy + 1.1),
                .30 + .50*np.sin(4*xx + 2.2)**2], -1)
img[((xx-.32)**2 + (yy-.35)**2) < .028] = [.93, .22, .18]
img = np.clip(img, 0, 1)
flat = img.reshape(-1, 3)
print(f"image {img.shape}, {len(np.unique(flat.round(3), axis=0)):,} distinct colours")
print(f"\\n{'k':>5}{'bits/px':>10}{'compression':>14}{'inertia':>12}{'max err':>10}")
for k in [2, 4, 8, 16, 32, 64]:
    km = KMeans(k, n_init=3, random_state=0).fit(flat)
    seg = km.cluster_centers_[km.labels_]
    bits = int(np.ceil(np.log2(k)))
    print(f"{k:>5}{bits:>10}{bits/24:>13.1%}{km.inertia_:>12.2f}"
          f"{np.abs(seg-flat).max():>10.4f}")

# ---- adding pixel coordinates encourages spatial coherence -----------
coords = np.c_[xx.ravel(), yy.ravel()]
flat_xy = np.c_[flat, coords * .35]          # weight the coordinates
km_xy = KMeans(8, n_init=3, random_state=0).fit(flat_xy)
print(f"\\nwith (x,y) appended, 8 clusters: the segments become spatially")
print(f"contiguous instead of colour-only. Distinct labels per row (mean): "
      f"{np.mean([len(np.unique(km_xy.labels_.reshape(H,W)[r])) for r in range(H)]):.2f}")
print(f"without (x,y):                                                    "
      f"{np.mean([len(np.unique(KMeans(8, n_init=3, random_state=0).fit(flat).labels_.reshape(H,W)[r])) for r in range(0,H,8)]):.2f}")
''',
        key="ch09_limits",
    )

    keypoints([
        "k-means assumes clusters are <b>spherical</b>, of similar <b>diameter</b> "
        "and similar <b>density</b>.",
        "Its boundaries are Voronoi cells — always convex, so non-convex clusters "
        "are impossible.",
        "<b>Scale the features first</b>, or artificial elongation guarantees a bad "
        "result.",
        "Colour quantisation = clustering pixels in RGB space; it compresses and "
        "segments at once.",
        "Colour segmentation is <b>not</b> semantic segmentation — that needs "
        "Chapter 14.",
    ])


# ==========================================================================
def s_9_5():
    section("9.5", "Clustering for Semi-Supervised Learning")

    lead(
        "The most practically valuable trick in this chapter. When labelling is "
        "expensive, clustering tells you <i>which</i> instances are worth "
        "labelling — and then spreads those labels."
    )

    sub("Representative images")

    md(
        """
The procedure:

1. Cluster the **unlabelled** training set into $k$ clusters.
2. For each cluster, find the instance **closest to the centroid** — the most
   *representative* instance of that group.
3. Label only those $k$ instances by hand.
4. Train on the $k$ labelled representatives.

This beats labelling $k$ *random* instances, often by a wide margin, because
random sampling wastes labels on redundant instances from the same dense region
while missing small clusters entirely.
        """
    )

    sub("Label propagation")

    md(
        "Better still: **propagate** each representative's label to every instance "
        "in its cluster. Now you have $m$ labelled instances for the cost of $k$."
    )

    math(r"""
    y^{(i)} \;\leftarrow\; y_{\text{rep}}\bigl(c^{(i)}\bigr)
    \qquad \text{for all } i
    """)

    md(
        "Then refine: propagate only to the instances **closest to the centroid** "
        "— the ones you are most confident about — and drop the rest. This trades "
        "quantity for purity:"
    )

    math(r"""
    \mathcal{P}_k \;=\; \Bigl\{ i \;:\; c^{(i)} = k,\;\;
      \bigl\lVert \mathbf{x}^{(i)} - \boldsymbol\mu_k \bigr\rVert
      \le q_{\,\pi}\bigl(\{\lVert \mathbf{x}^{(j)} - \boldsymbol\mu_k\rVert
        : c^{(j)} = k\}\bigr) \Bigr\}
    """)
    where({r"\pi": "the propagation percentile — e.g. 75 % keeps the closest "
                   "three-quarters of each cluster",
           r"q_\pi": "the $\\pi$-th quantile of the within-cluster distances"})

    anim_header("Labelling budget: random vs representative vs propagated")

    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xd, yd, _ = ds.digits()
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=.3, stratify=yd,
                                          random_state=42)
    sc = StandardScaler().fit(Xtr)
    Atr, Ate = sc.transform(Xtr), sc.transform(Xte)

    @st.cache_data(show_spinner="Running the labelling-budget experiment…")
    def budget_curves(Atr, ytr, Ate, yte):
        budgets = [10, 20, 30, 50, 80, 120, 200, 300]
        rows = []
        rr = np.random.default_rng(0)
        for k in budgets:
            # (a) random labels
            idx = rr.choice(len(Atr), k, replace=False)
            a_rand = LogisticRegression(max_iter=4000).fit(
                Atr[idx], ytr[idx]).score(Ate, yte)
            # (b) representative labels
            km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Atr)
            D = km.transform(Atr)
            reps = np.argmin(D, axis=0)
            a_rep = LogisticRegression(max_iter=4000).fit(
                Atr[reps], ytr[reps]).score(Ate, yte)
            # (c) propagate to the whole cluster
            y_prop = np.empty(len(Atr), dtype=ytr.dtype)
            for j in range(k):
                y_prop[km.labels_ == j] = ytr[reps[j]]
            a_prop = LogisticRegression(max_iter=4000).fit(
                Atr, y_prop).score(Ate, yte)
            # (d) propagate only to the closest 75 %
            keep = np.zeros(len(Atr), bool)
            for j in range(k):
                inc = np.where(km.labels_ == j)[0]
                if len(inc) == 0:
                    continue
                cutoff = np.percentile(D[inc, j], 75)
                keep[inc[D[inc, j] <= cutoff]] = True
            a_part = LogisticRegression(max_iter=4000).fit(
                Atr[keep], y_prop[keep]).score(Ate, yte)
            rows.append((k, a_rand, a_rep, a_prop, a_part))
        full = LogisticRegression(max_iter=4000).fit(Atr, ytr).score(Ate, yte)
        return rows, full

    rows, full_acc = budget_curves(Atr, ytr, Ate, yte)
    budgets = [r[0] for r in rows]

    frames = []
    for i in range(1, len(rows) + 1):
        r = rows[i - 1]
        frames.append(go.Frame(name=str(r[0]), data=[
            go.Scatter(x=budgets[:i], y=[q[1] for q in rows[:i]],
                       mode="lines+markers", line=dict(color=C["muted"], width=2.6)),
            go.Scatter(x=budgets[:i], y=[q[2] for q in rows[:i]],
                       mode="lines+markers", line=dict(color=C["info"], width=3)),
            go.Scatter(x=budgets[:i], y=[q[3] for q in rows[:i]],
                       mode="lines+markers", line=dict(color=C["warning"], width=3)),
            go.Scatter(x=budgets[:i], y=[q[4] for q in rows[:i]],
                       mode="lines+markers", line=dict(color=C["success"], width=3.4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"budget = {r[0]} hand-labelled images   ·   random {r[1]:.3f}   ·   "
            f"representative {r[2]:.3f}   ·   propagated {r[3]:.3f}   ·   "
            f"partial {r[4]:.3f}")])))

    f = go.Figure(data=[
        go.Scatter(x=budgets[:1], y=[rows[0][1]], mode="lines+markers",
                   name="random labels", line=dict(color=C["muted"], width=2.6)),
        go.Scatter(x=budgets[:1], y=[rows[0][2]], mode="lines+markers",
                   name="representative labels", line=dict(color=C["info"], width=3)),
        go.Scatter(x=budgets[:1], y=[rows[0][3]], mode="lines+markers",
                   name="propagated to whole cluster",
                   line=dict(color=C["warning"], width=3)),
        go.Scatter(x=budgets[:1], y=[rows[0][4]], mode="lines+markers",
                   name="propagated to closest 75 %",
                   line=dict(color=C["success"], width=3.4)),
    ])
    f.add_hline(y=full_acc, line_dash="dot", line_color=C["truth"],
                annotation_text=f"all {len(Atr)} labels = {full_acc:.3f}")
    f.update_layout(height=470, xaxis_title="hand-labelling budget (images)",
                    yaxis_title="test accuracy", xaxis_type="log",
                    title="The same labelling budget, spent four ways",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="budget = ")
    figure(f, "At every budget, choosing representatives beats random, and "
              "propagating beats not propagating.")

    idea(
        "This is the highest-return technique in the chapter",
        "Labelling is usually the single largest cost in an ML project. This "
        "procedure routinely gets 80–90 % of the fully-supervised accuracy from "
        "1–2 % of the labels. It costs one <code>KMeans</code> fit. Combine it "
        "with <b>active learning</b> — train, find the instances the model is "
        "least confident about, label those, repeat — and you have the standard "
        "industrial labelling loop.",
    )

    code_lab(
        "The full semi-supervised pipeline, with active learning",
        '''import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    d.data, d.target, test_size=.3, stratify=d.target, random_state=42)
sc = StandardScaler().fit(X_train)
A_train, A_test = sc.transform(X_train), sc.transform(X_test)

BUDGET = 50
print(f"labelling budget: {BUDGET} of {len(A_train)} images "
      f"({BUDGET/len(A_train):.1%})\\n")

# ============ 0. the ceiling and the naive floor =======================
full = LogisticRegression(max_iter=5000).fit(A_train, y_train)
print(f"{'strategy':<44}{'test acc':>10}{'labels used':>14}")
print(f"{'ALL labels (the ceiling)':<44}{full.score(A_test, y_test):>10.4f}"
      f"{len(A_train):>14}")

rng = np.random.default_rng(42)
idx = rng.choice(len(A_train), BUDGET, replace=False)
rand = LogisticRegression(max_iter=5000).fit(A_train[idx], y_train[idx])
print(f"{f'{BUDGET} RANDOM labels':<44}{rand.score(A_test, y_test):>10.4f}{BUDGET:>14}")

# ============ 1. representative instances ==============================
km = KMeans(n_clusters=BUDGET, n_init=10, random_state=42).fit(A_train)
D = km.transform(A_train)
reps = np.argmin(D, axis=0)                       # closest to each centroid
rep_model = LogisticRegression(max_iter=5000).fit(A_train[reps], y_train[reps])
print(f"{f'{BUDGET} REPRESENTATIVE labels':<44}"
      f"{rep_model.score(A_test, y_test):>10.4f}{BUDGET:>14}")
print(f"   -> the {BUDGET} representatives cover "
      f"{len(np.unique(y_train[reps]))} of 10 digit classes")
print(f"   -> {BUDGET} random images cover {len(np.unique(y_train[idx]))} classes")

# ============ 2. label propagation =====================================
y_prop = np.empty(len(A_train), dtype=y_train.dtype)
for j in range(BUDGET):
    y_prop[km.labels_ == j] = y_train[reps[j]]
prop = LogisticRegression(max_iter=5000).fit(A_train, y_prop)
print(f"{'propagated to the WHOLE cluster':<44}{prop.score(A_test, y_test):>10.4f}"
      f"{len(A_train):>14}")
print(f"   -> propagated label accuracy: {np.mean(y_prop == y_train):.4f}")

# ============ 3. partial propagation ===================================
print()
for pct in [50, 75, 90, 99]:
    keep = np.zeros(len(A_train), bool)
    for j in range(BUDGET):
        inc = np.where(km.labels_ == j)[0]
        if len(inc) == 0: continue
        cut = np.percentile(D[inc, j], pct)
        keep[inc[D[inc, j] <= cut]] = True
    m_ = LogisticRegression(max_iter=5000).fit(A_train[keep], y_prop[keep])
    print(f"{f'propagated to the closest {pct} %':<44}"
          f"{m_.score(A_test, y_test):>10.4f}{int(keep.sum()):>14}"
          f"   (label purity {np.mean(y_prop[keep] == y_train[keep]):.3f})")

# ============ 4. sklearn's LabelSpreading ==============================
y_semi = np.full(len(A_train), -1)
y_semi[reps] = y_train[reps]
ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=.2).fit(A_train, y_semi)
ls_model = LogisticRegression(max_iter=5000).fit(A_train, ls.transduction_)
print(f"\\n{'LabelSpreading (graph-based)':<44}"
      f"{ls_model.score(A_test, y_test):>10.4f}{len(A_train):>14}")
print(f"   -> its inferred labels are {np.mean(ls.transduction_ == y_train):.4f} correct")

# ============ 5. ACTIVE LEARNING: spend the budget in rounds ===========
print("\\n=== active learning: 5 rounds of 10 labels each ===")
labelled = np.zeros(len(A_train), bool)
first = rng.choice(len(A_train), 10, replace=False)
labelled[first] = True
print(f"{'round':>6}{'labels':>8}{'test acc':>11}   strategy")
for rnd in range(6):
    m_ = LogisticRegression(max_iter=5000).fit(A_train[labelled], y_train[labelled])
    print(f"{rnd:>6}{int(labelled.sum()):>8}{m_.score(A_test, y_test):>11.4f}"
          f"   {'seed' if rnd == 0 else 'uncertainty sampling'}")
    if rnd == 5: break
    # pick the 10 instances the model is LEAST confident about
    proba = m_.predict_proba(A_train)
    margin = np.sort(proba, axis=1)
    uncertainty = margin[:, -1] - margin[:, -2]     # small = uncertain
    uncertainty[labelled] = np.inf
    labelled[np.argsort(uncertainty)[:10]] = True

# compare with spending the same 60 labels at random
idx60 = rng.choice(len(A_train), 60, replace=False)
print(f"\\n60 RANDOM labels instead: "
      f"{LogisticRegression(max_iter=5000).fit(A_train[idx60], y_train[idx60]).score(A_test, y_test):.4f}")
''',
        key="ch09_semi",
    )

    keypoints([
        "Cluster first, then label the <b>representative</b> of each cluster — it "
        "beats random labelling at every budget.",
        "<b>Propagate</b> each representative's label to its cluster: $m$ labels "
        "for the price of $k$.",
        "Propagating to only the closest $\\pi$ % trades quantity for label "
        "purity — often the best setting.",
        "<code>LabelSpreading</code> / <code>LabelPropagation</code> do the "
        "graph-based version.",
        "Combine with <b>active learning</b> (label the most uncertain instances) "
        "for the standard industrial loop.",
    ])


# ==========================================================================
def s_9_6():
    section("9.6", "DBSCAN and Other Clustering Algorithms")

    lead(
        "Density-Based Spatial Clustering of Applications with Noise. It defines "
        "a cluster as a <b>connected dense region</b> — which needs no $k$, finds "
        "arbitrary shapes, and identifies outliers as a first-class output."
    )

    sub("The definitions")

    table(
        ["Term", "Definition"],
        [["<b>ε-neighbourhood</b>",
          "$N_\\varepsilon(\\mathbf{x}) = \\{\\mathbf{x}' : "
          "\\lVert\\mathbf{x} - \\mathbf{x}'\\rVert \\le \\varepsilon\\}$"],
         ["<b>Core instance</b>",
          "An instance with at least <code>min_samples</code> instances in its "
          "$\\varepsilon$-neighbourhood (including itself)"],
         ["<b>Border instance</b>",
          "Not a core instance, but inside some core instance's neighbourhood"],
         ["<b>Noise / outlier</b>",
          "Neither core nor border. Labelled <code>-1</code>"],
         ["<b>Cluster</b>",
          "A maximal set of <b>density-connected</b> instances: core instances "
          "linked through chains of neighbouring core instances, plus their "
          "border instances"]],
    )

    md("The algorithm is then two lines:")

    md(
        """
1. Find all core instances.
2. Any two core instances within $\\varepsilon$ of each other belong to the same
   cluster (this is a connected-components problem). Assign each border instance
   to a cluster of a core neighbour; label everything else as noise.
        """
    )

    idea(
        "DBSCAN's three gifts",
        "<b>(1)</b> You do not specify $k$ — the number of clusters emerges from "
        "the density structure. <b>(2)</b> Clusters can be <b>any shape</b>, "
        "because density-connectivity is transitive along chains, not radial. "
        "<b>(3)</b> <b>Outliers are an output</b>, not something you have to detect "
        "separately — which makes DBSCAN a legitimate anomaly detector.",
    )

    sub("Choosing ε and min_samples")

    md(
        "**`min_samples`** is a smoothing parameter; a common heuristic is "
        "$2n$ (twice the dimensionality), and at least 3. **$\\varepsilon$** is "
        "the sensitive one, and there is a standard diagnostic for it:"
    )

    md(
        "**The $k$-distance plot.** Compute each instance's distance to its "
        "$k$-th nearest neighbour (with $k = $ `min_samples`), sort those "
        "distances, and plot. The **knee** of that curve is a good $\\varepsilon$ "
        "— below the knee you are inside clusters, above it you are between them."
    )

    anim_header("ε sweeping, with the k-distance plot")

    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    Xm, ym = ds.moons(n=500, noise=.07)
    Xm = StandardScaler().fit_transform(Xm)
    min_s = 6
    nn = NearestNeighbors(n_neighbors=min_s).fit(Xm)
    kdist = np.sort(nn.kneighbors(Xm)[0][:, -1])

    eps_list = np.round(np.linspace(.04, .55, 30), 3)
    cache = []
    for e in eps_list:
        db = DBSCAN(eps=float(e), min_samples=min_s).fit(Xm)
        lab = db.labels_
        n_clu = len(set(lab)) - (1 if -1 in lab else 0)
        cache.append((lab, n_clu, int(np.sum(lab == -1)),
                      len(db.core_sample_indices_)))

    frames = []
    for i, e in enumerate(eps_list):
        lab, n_clu, n_noise, n_core = cache[i]
        cols = [C["muted"] if j < 0 else SEQ[j % len(SEQ)] for j in lab]
        szs = [4 if j < 0 else 7 for j in lab]
        col = C["success"] if n_clu == 2 and n_noise < len(Xm) * .1 else C["warning"]
        frames.append(go.Frame(name=f"{e}", data=[
            go.Scattergl(x=Xm[:, 0], y=Xm[:, 1], mode="markers",
                         marker=dict(color=cols, size=szs,
                                     line=dict(color="#fff", width=.5))),
            go.Scatter(x=np.arange(len(kdist)), y=kdist, mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=[0, len(kdist)], y=[e, e], mode="lines",
                       line=dict(color=C["danger"], width=2.5, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"ε = {e:.3f}   ·   {n_clu} clusters   ·   {n_noise} noise points   ·   "
            f"{n_core} core instances", color=col)])))

    lab0 = cache[0][0]
    f = make_subplots(rows=1, cols=2, column_widths=[.56, .44],
                      subplot_titles=("DBSCAN clusters (grey = noise)",
                                      f"{min_s}-distance plot — find the knee"))
    f.add_trace(go.Scattergl(x=Xm[:, 0], y=Xm[:, 1], mode="markers",
                             showlegend=False,
                             marker=dict(color=[C["muted"] if j < 0
                                                else SEQ[j % len(SEQ)] for j in lab0],
                                         size=6,
                                         line=dict(color="#fff", width=.5))), 1, 1)
    f.add_trace(go.Scatter(x=np.arange(len(kdist)), y=kdist, mode="lines",
                           showlegend=False,
                           line=dict(color=C["primary"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=[0, len(kdist)], y=[eps_list[0]] * 2, mode="lines",
                           showlegend=False,
                           line=dict(color=C["danger"], width=2.5, dash="dash")), 1, 2)
    f.update_xaxes(title_text="instances sorted by distance", row=1, col=2)
    f.update_yaxes(title_text=f"distance to {min_s}-th neighbour", row=1, col=2)
    f.update_layout(height=470, title="DBSCAN: ε is the whole game")
    anim.animate(f, frames, duration=nav.anim_ms(240), slider_prefix="ε = ")
    figure(f)

    pitfall(
        "DBSCAN has no `predict` method",
        "Deliberately. The authors' position is that other classification "
        "algorithms are better suited to assigning new instances, so "
        "<code>DBSCAN</code> exposes only <code>fit_predict</code>. To label new "
        "data, train a classifier — typically <code>KNeighborsClassifier</code> — "
        "on the <b>core instances</b> and their cluster labels, then use it to "
        "predict. You can also reject as anomalies any new instance whose "
        "distance to its nearest core instance exceeds $\\varepsilon$.",
    )

    warn(
        "DBSCAN fails when densities vary",
        "A single $\\varepsilon$ must work for every cluster. If one cluster is "
        "dense and another sparse, no $\\varepsilon$ succeeds: too small and the "
        "sparse cluster becomes all noise; too large and the dense clusters merge. "
        "<b>HDBSCAN</b> (Campello et al.) fixes this by building a hierarchy over "
        "all $\\varepsilon$ values and extracting the most stable clusters — it is "
        "in <code>sklearn.cluster.HDBSCAN</code> and is usually the better "
        "default.",
    )

    sub("The other clustering algorithms")

    table(
        ["Algorithm", "Idea", "Scales to", "Shapes", "Needs $k$?"],
        [["<b>Agglomerative</b>",
          "Repeatedly merge the two closest clusters, building a dendrogram",
          "$\\mathcal{O}(m^2\\log m)$", "Depends on the linkage", "Yes (or cut "
          "the tree at a height)"],
         ["<b>BIRCH</b>",
          "Build a compact CF-tree summary in one pass, then cluster that",
          "<b>Huge</b> — designed for out-of-core", "Spherical", "Yes"],
         ["<b>Mean-Shift</b>",
          "Slide a circular window uphill along the density gradient until it "
          "settles on a mode",
          "$\\mathcal{O}(m^2)$", "Any", "<b>No</b> (bandwidth instead)"],
         ["<b>Affinity Propagation</b>",
          "Instances exchange messages until a set of exemplars emerges",
          "$\\mathcal{O}(m^2)$", "Roughly spherical", "<b>No</b>"],
         ["<b>Spectral Clustering</b>",
          "Build a similarity graph, embed with its Laplacian's eigenvectors, then "
          "run k-means in that space",
          "$\\mathcal{O}(m^3)$", "<b>Any</b> — excellent on moons/circles", "Yes"],
         ["<b>HDBSCAN</b>",
          "DBSCAN over all $\\varepsilon$, extracting the most stable clusters",
          "$\\mathcal{O}(m\\log m)$", "Any, varying density", "<b>No</b>"]],
    )

    code_lab(
        "DBSCAN end to end, plus every other clustering algorithm",
        '''import numpy as np, time
from sklearn.cluster import (DBSCAN, HDBSCAN, KMeans, AgglomerativeClustering,
                             Birch, MeanShift, SpectralClustering,
                             AffinityPropagation, estimate_bandwidth)
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=1000, noise=.07, random_state=42)
X = StandardScaler().fit_transform(X)

# ============ 1. choosing eps from the k-distance plot ================
MIN_S = 6
nn = NearestNeighbors(n_neighbors=MIN_S).fit(X)
kdist = np.sort(nn.kneighbors(X)[0][:, -1])
# the knee: the point of maximum curvature
kappa = np.gradient(np.gradient(kdist))
knee_eps = float(kdist[np.argmax(kappa)])
print(f"=== choosing eps ===")
print(f"k-distance knee suggests eps ~ {knee_eps:.4f}")
print(f"(percentiles of the {MIN_S}-distance: "
      f"50 % {np.percentile(kdist,50):.3f}, 90 % {np.percentile(kdist,90):.3f}, "
      f"95 % {np.percentile(kdist,95):.3f})")

print(f"\\n{'eps':>7}{'clusters':>10}{'noise':>8}{'core':>7}{'ARI':>9}")
for eps in [0.05, 0.10, knee_eps, 0.20, 0.30, 0.50]:
    db = DBSCAN(eps=float(eps), min_samples=MIN_S).fit(X)
    lab = db.labels_
    n_c = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"{eps:>7.3f}{n_c:>10}{int((lab==-1).sum()):>8}"
          f"{len(db.core_sample_indices_):>7}{adjusted_rand_score(y, lab):>9.4f}")

# ============ 2. DBSCAN has no predict -- build one ====================
db = DBSCAN(eps=.2, min_samples=MIN_S).fit(X)
print(f"\\n=== giving DBSCAN a predict() ===")
print(f"DBSCAN has predict? {hasattr(db, 'predict')}")

# train a kNN on the CORE instances only
knn = KNeighborsClassifier(n_neighbors=MIN_S)
knn.fit(db.components_, db.labels_[db.core_sample_indices_])

X_new = np.array([[-0.5, 0.], [0., 0.5], [1., -0.1], [3.5, 3.5]])
print(f"\\n{'new instance':>22}{'predicted cluster':>20}{'dist to nearest core':>23}")
dist, _ = knn.kneighbors(X_new, n_neighbors=1)
pred = knn.predict(X_new)
for i, x in enumerate(X_new):
    tag = "ANOMALY" if dist[i, 0] > .2 else str(pred[i])
    print(f"{str(x.round(2)):>22}{tag:>20}{dist[i,0]:>23.4f}")
print("\\nRejecting anything farther than eps from a core instance turns")
print("DBSCAN into a novelty detector.")

# ============ 3. VARYING DENSITY breaks DBSCAN, not HDBSCAN ===========
print("\\n=== varying density ===")
rng = np.random.default_rng(0)
Xv = np.r_[rng.normal([0, 0], .12, (300, 2)),      # dense
           rng.normal([2.5, 2.5], .55, (300, 2)),  # sparse
           rng.uniform(-1.5, 4.5, (40, 2))]        # noise
yv = np.r_[np.zeros(300), np.ones(300), np.full(40, -1)]
print(f"{'eps':>7}{'clusters':>10}{'noise':>8}{'ARI':>9}")
for eps in [0.10, 0.20, 0.35, 0.60]:
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(Xv)
    n_c = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"{eps:>7.2f}{n_c:>10}{int((lab==-1).sum()):>8}"
          f"{adjusted_rand_score(yv, lab):>9.4f}")
lab_h = HDBSCAN(min_cluster_size=25).fit_predict(Xv)
print(f"{'HDBSCAN':>7}{len(set(lab_h))-(1 if -1 in lab_h else 0):>10}"
      f"{int((lab_h==-1).sum()):>8}{adjusted_rand_score(yv, lab_h):>9.4f}   <- no eps needed")

# ============ 4. every algorithm on two datasets ======================
print("\\n=== the whole zoo ===")
datasets = {"moons": make_moons(n_samples=600, noise=.07, random_state=0),
            "blobs": make_blobs(n_samples=600, centers=4, cluster_std=.8,
                                random_state=0)}
algos = {
    "KMeans":               lambda Xs, k: KMeans(k, n_init=10, random_state=0).fit_predict(Xs),
    "Agglomerative(ward)":  lambda Xs, k: AgglomerativeClustering(k).fit_predict(Xs),
    "Agglomerative(single)":lambda Xs, k: AgglomerativeClustering(k, linkage="single").fit_predict(Xs),
    "Birch":                lambda Xs, k: Birch(n_clusters=k).fit_predict(Xs),
    "DBSCAN":               lambda Xs, k: DBSCAN(eps=.3, min_samples=6).fit_predict(Xs),
    "HDBSCAN":              lambda Xs, k: HDBSCAN(min_cluster_size=20).fit_predict(Xs),
    "MeanShift":            lambda Xs, k: MeanShift(bandwidth=estimate_bandwidth(Xs, quantile=.2)).fit_predict(Xs),
    "SpectralClustering":   lambda Xs, k: SpectralClustering(k, affinity="nearest_neighbors", random_state=0).fit_predict(Xs),
    "AffinityPropagation":  lambda Xs, k: AffinityPropagation(random_state=0).fit_predict(Xs),
}
print(f"{'algorithm':<24}" + "".join(f"{n:>18}" for n in datasets))
for anm, fn in algos.items():
    row = ""
    for dnm, (Xd, yd_) in datasets.items():
        Xs = StandardScaler().fit_transform(Xd)
        k = len(np.unique(yd_))
        t0 = time.perf_counter()
        try:
            lab = fn(Xs, k)
            ari = adjusted_rand_score(yd_, lab)
            row += f"{f'{ari:.3f} ({time.perf_counter()-t0:.2f}s)':>18}"
        except Exception as e:
            row += f"{'failed':>18}"
    print(f"{anm:<24}{row}")
print("\\nSpectral, DBSCAN and HDBSCAN win on moons; almost everything works on blobs.")
''',
        key="ch09_dbscan",
    )

    keypoints([
        "DBSCAN: core instances have ≥ <code>min_samples</code> neighbours within "
        "$\\varepsilon$; clusters are density-connected sets.",
        "No $k$ required, <b>any cluster shape</b>, and <b>outliers are an "
        "output</b> (label $-1$).",
        "Choose $\\varepsilon$ from the knee of the $k$-distance plot.",
        "No <code>predict</code> — fit a $k$-NN on <code>components_</code> "
        "(the core instances) to label new data.",
        "Varying density defeats DBSCAN; <b>HDBSCAN</b> is the fix and needs no "
        "$\\varepsilon$.",
    ])


# ==========================================================================
def s_9_7():
    section("9.7", "Gaussian Mixtures, EM, and Anomaly Detection")

    lead(
        "A generative model: assume the data was produced by sampling from one of "
        "$K$ Gaussians. Fitting it gives soft cluster assignments, a density "
        "estimate, an anomaly detector, and a principled way to choose $K$ — all "
        "from one algorithm."
    )

    sub("The model")

    math(r"""
    p(\mathbf{x}) \;=\; \sum_{k=1}^{K} \pi_k \,
      \mathcal{N}\bigl(\mathbf{x} \mid \boldsymbol\mu_k, \boldsymbol\Sigma_k\bigr)
    """)
    where({r"\pi_k": "the mixing weight of component $k$, with $\\pi_k \\ge 0$ and "
                     "$\\sum_k \\pi_k = 1$",
           r"\boldsymbol\mu_k, \boldsymbol\Sigma_k": "the mean and covariance of "
                                                     "component $k$",
           r"\mathcal{N}": "the multivariate normal density"})

    md("The generative story is two steps:")

    math(r"""
    z^{(i)} \sim \mathrm{Categorical}(\boldsymbol\pi),
    \qquad
    \mathbf{x}^{(i)} \mid z^{(i)} = k \;\sim\;
      \mathcal{N}\bigl(\boldsymbol\mu_k, \boldsymbol\Sigma_k\bigr)
    """)

    md("We observe $\\mathbf{x}$ but not $z$ — $z$ is a **latent** variable. That "
       "is exactly the situation the EM algorithm was invented for.")

    sub("The Expectation–Maximisation algorithm")

    derive(
        [("The log-likelihood we want to maximise contains a log of a sum, which "
          "has no closed-form maximiser:",
          r"\ell(\boldsymbol\theta) = \sum_{i=1}^{m}\log\left(\sum_{k=1}^{K}\pi_k\,"
          r"\mathcal{N}\bigl(\mathbf{x}^{(i)}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k\bigr)\right)"),
         ("EM's trick: introduce a distribution $q$ over the latent variables and "
          "apply <b>Jensen's inequality</b> (log is concave) to get a lower bound:",
          r"\ell(\boldsymbol\theta) = \sum_i \log \sum_k q_{ik}\,"
          r"\frac{\pi_k \mathcal{N}_k(\mathbf{x}^{(i)})}{q_{ik}} "
          r"\;\ge\; \sum_i \sum_k q_{ik}\log\frac{\pi_k \mathcal{N}_k(\mathbf{x}^{(i)})}{q_{ik}} "
          r"\;=\; \mathcal{F}(q, \boldsymbol\theta)"),
         ("<b>E-step.</b> Maximise the bound over $q$ with $\\boldsymbol\\theta$ "
          "fixed. Jensen is tight exactly when the ratio is constant in $k$, which "
          "gives the posterior — the <b>responsibility</b> of component $k$ for "
          "instance $i$:",
          r"\gamma_{ik} \;=\; \Pr\bigl[z^{(i)} = k \mid \mathbf{x}^{(i)}\bigr] "
          r"= \frac{\pi_k\,\mathcal{N}\bigl(\mathbf{x}^{(i)}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k\bigr)}"
          r"{\displaystyle\sum_{j=1}^{K}\pi_j\,\mathcal{N}\bigl(\mathbf{x}^{(i)}\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j\bigr)}"),
         ("<b>M-step.</b> Now maximise the bound over $\\boldsymbol\\theta$ with "
          "$q = \\gamma$ fixed. This is just weighted maximum likelihood for "
          "Gaussians, which has a closed form. Write "
          "$m_k = \\sum_i \\gamma_{ik}$ (the effective count):",
          r"\pi_k = \frac{m_k}{m}, \qquad "
          r"\boldsymbol\mu_k = \frac{1}{m_k}\sum_{i=1}^{m}\gamma_{ik}\,\mathbf{x}^{(i)}"),
         ("and the covariance, also a weighted version of the usual estimator:",
          r"\boldsymbol\Sigma_k = \frac{1}{m_k}\sum_{i=1}^{m}\gamma_{ik}\,"
          r"\bigl(\mathbf{x}^{(i)} - \boldsymbol\mu_k\bigr)"
          r"\bigl(\mathbf{x}^{(i)} - \boldsymbol\mu_k\bigr)^\top"),
         ("<b>Convergence.</b> The E-step makes the bound tight "
          "($\\mathcal{F} = \\ell$) and the M-step increases it, so $\\ell$ is "
          "non-decreasing at every iteration. Like k-means, it converges to a "
          "<b>local</b> optimum, so <code>n_init</code> matters.", None),
         ("<b>k-means is the hard-assignment limit.</b> Fix "
          "$\\boldsymbol\\Sigma_k = \\sigma^2\\mathbf{I}$ and let "
          "$\\sigma \\to 0$: the responsibilities $\\gamma_{ik}$ collapse to 0 or 1, "
          "the E-step becomes 'assign to the nearest centroid' and the M-step "
          "becomes 'move to the mean'. <b>k-means is EM for a spherical, "
          "equal-variance, hard-assignment Gaussian mixture.</b>", None)],
        title="Deriving EM from Jensen's inequality",
    )

    anim_header("EM fitting a Gaussian mixture, iteration by iteration")
    md(
        "Ellipses are the fitted Gaussians at 1σ and 2σ; colour intensity is the "
        "responsibility. Watch the log-likelihood climb monotonically."
    )

    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(5)
    Xg = np.r_[
        rng.multivariate_normal([0, 0], [[.6, .45], [.45, .6]], 220),
        rng.multivariate_normal([3.4, 2.6], [[.9, -.5], [-.5, .5]], 180),
        rng.multivariate_normal([-2.4, 2.8], [[.35, 0], [0, 1.5]], 150),
    ]
    K = 3
    snaps = []
    gm = GaussianMixture(n_components=K, covariance_type="full", max_iter=1,
                         init_params="random_from_data", warm_start=True,
                         random_state=1, reg_covar=1e-5)
    for it in range(24):
        gm.max_iter = 1
        gm.fit(Xg)
        snaps.append((gm.means_.copy(), gm.covariances_.copy(),
                      gm.weights_.copy(), float(gm.score(Xg) * len(Xg)),
                      gm.predict_proba(Xg).copy()))

    def ellipse(mu, cov, nsig=1.0, npts=80):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-9)
        th = np.linspace(0, 2 * np.pi, npts)
        circ = np.c_[np.cos(th), np.sin(th)]
        pts = circ * (nsig * np.sqrt(vals))
        return pts @ vecs.T + mu

    frames = []
    for it, (mus, covs, ws, ll, resp) in enumerate(snaps):
        cols = ["rgb({},{},{})".format(
            *[int(sum(int(SEQ[k][1 + 2 * c:3 + 2 * c], 16) * resp[i, k]
                      for k in range(K))) for c in range(3)])
            for i in range(len(Xg))]
        data = [go.Scattergl(x=Xg[:, 0], y=Xg[:, 1], mode="markers",
                             marker=dict(color=cols, size=5,
                                         line=dict(color="#fff", width=.3)))]
        for k in range(K):
            for ns, w in [(1, 3), (2, 1.6)]:
                e = ellipse(mus[k], covs[k], ns)
                data.append(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines",
                                       line=dict(color=SEQ[k], width=w)))
        data.append(go.Scatter(x=list(range(1, it + 2)),
                               y=[s[3] for s in snaps[:it + 1]],
                               mode="lines+markers",
                               line=dict(color=C["danger"], width=3)))
        frames.append(go.Frame(name=str(it + 1), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"EM iteration {it+1}   ·   "
                                   f"log-likelihood = {ll:,.2f}   ·   "
                                   f"weights = {np.round(ws, 3)}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.6, .4],
                      subplot_titles=("components and responsibilities",
                                      "log-likelihood (monotone increasing)"))
    mus0, covs0, ws0, ll0, resp0 = snaps[0]
    f.add_trace(go.Scattergl(x=Xg[:, 0], y=Xg[:, 1], mode="markers",
                             showlegend=False,
                             marker=dict(color=C["muted"], size=5)), 1, 1)
    for k in range(K):
        for ns, w in [(1, 3), (2, 1.6)]:
            e = ellipse(mus0[k], covs0[k], ns)
            f.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines",
                                   showlegend=False,
                                   line=dict(color=SEQ[k], width=w)), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[ll0], mode="lines+markers", showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.update_xaxes(title_text="EM iteration", row=1, col=2)
    f.update_yaxes(title_text="log-likelihood", row=1, col=2)
    f.update_layout(height=490, title="Expectation–Maximisation")
    anim.animate(f, frames, duration=nav.anim_ms(400), slider_prefix="iteration ")
    figure(f)

    sub("covariance_type — the bias/variance dial")

    table(
        ["<code>covariance_type</code>", "Constraint",
         "Parameters per component", "Shape"],
        [["<code>'spherical'</code>", "$\\boldsymbol\\Sigma_k = \\sigma_k^2\\mathbf{I}$",
          "1", "Circles, different sizes"],
         ["<code>'diag'</code>", "$\\boldsymbol\\Sigma_k$ diagonal", "$n$",
          "Axis-aligned ellipses"],
         ["<code>'tied'</code>", "$\\boldsymbol\\Sigma_k = \\boldsymbol\\Sigma$ shared",
          "$n(n+1)/2$ total", "Identical ellipses"],
         ["<code>'full'</code>", "Unconstrained (the default)",
          "$n(n+1)/2$", "Any ellipse, any orientation"]],
    )

    sub("Anomaly detection")

    md(
        "A GMM is a **density estimate**, so anomalies are simply instances in "
        "low-density regions:"
    )

    math(r"""
    \mathbf{x} \text{ is an anomaly}
    \quad\Longleftrightarrow\quad
    \log p(\mathbf{x}) \;<\; \tau
    """)
    md("with $\\tau$ set from the desired contamination rate — the $\\alpha$-th "
       "percentile of the training log-densities.")

    sub("Choosing K: BIC and AIC")

    math(r"""
    \mathrm{BIC} \;=\; \log(m)\,p \;-\; 2\log\hat{L}
    \qquad\qquad
    \mathrm{AIC} \;=\; 2p \;-\; 2\log\hat{L}
    """)
    where({r"m": "number of instances",
           r"p": "number of free parameters in the model",
           r"\hat L": "the maximised likelihood",
           r"\text{lower is better}": "for both"})

    md(
        "Both penalise complexity; **BIC penalises harder** (by $\\log m$ rather "
        "than 2), so it prefers simpler models and tends to select fewer "
        "components. Unlike inertia, these are *not* monotone in $K$ — you can "
        "minimise them directly."
    )

    sub("Bayesian Gaussian Mixture")

    md(
        "`BayesianGaussianMixture` goes further: set `n_components` to an upper "
        "bound and the model **drives the unnecessary weights to zero by itself**, "
        "via a Dirichlet process prior. No search over $K$ at all."
    )

    anim_header("BIC, AIC and the Bayesian mixture's automatic weights")

    Ks = list(range(1, 11))
    bics, aics, lls = [], [], []
    for k in Ks:
        g = GaussianMixture(n_components=k, n_init=5, random_state=0).fit(Xg)
        bics.append(g.bic(Xg)); aics.append(g.aic(Xg))
        lls.append(g.score(Xg) * len(Xg))
    best_bic = Ks[int(np.argmin(bics))]

    from sklearn.mixture import BayesianGaussianMixture
    bgm = BayesianGaussianMixture(n_components=10, n_init=5,
                                  weight_concentration_prior=.01,
                                  random_state=0).fit(Xg)

    frames = []
    for i, k in enumerate(Ks):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=Ks[:i + 1], y=bics[:i + 1], mode="lines+markers",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=Ks[:i + 1], y=aics[:i + 1], mode="lines+markers",
                       line=dict(color=C["accent"], width=3)),
            go.Bar(x=[f"c{j+1}" for j in range(10)],
                   y=np.round(bgm.weights_, 4),
                   marker=dict(color=[C["success"] if w > .02 else C["muted"]
                                      for w in bgm.weights_])),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"K = {k}   ·   BIC = {bics[i]:,.1f}   ·   AIC = {aics[i]:,.1f}"
            + ("   ← BIC minimum" if k == best_bic else ""),
            color=C["success"] if k == best_bic else C["ink"])])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("BIC and AIC vs K (lower is better)",
                                      "BayesianGaussianMixture weights "
                                      "(K_max = 10)"))
    f.add_trace(go.Scatter(x=Ks[:1], y=bics[:1], mode="lines+markers", name="BIC",
                           line=dict(color=C["primary"], width=3)), 1, 1)
    f.add_trace(go.Scatter(x=Ks[:1], y=aics[:1], mode="lines+markers", name="AIC",
                           line=dict(color=C["accent"], width=3)), 1, 1)
    f.add_trace(go.Bar(x=[f"c{j+1}" for j in range(10)], y=bgm.weights_,
                       showlegend=False,
                       marker=dict(color=[C["success"] if w > .02 else C["muted"]
                                          for w in bgm.weights_])), 1, 2)
    f.update_xaxes(title_text="number of components K", row=1, col=1)
    f.update_yaxes(title_text="criterion", row=1, col=1)
    f.update_yaxes(title_text="weight", row=1, col=2)
    f.update_layout(height=450, title="Choosing K automatically")
    anim.animate(f, frames, duration=nav.anim_ms(500), slider_prefix="K = ")
    figure(f, f"BIC picks K = {best_bic}. The Bayesian mixture, given a maximum "
              f"of 10, drove {int(np.sum(bgm.weights_ < 0.02))} weights to "
              f"essentially zero on its own.")

    code_lab(
        "EM from scratch, anomaly detection, BIC, and the k-means connection",
        '''import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(5)
X = np.r_[rng.multivariate_normal([0, 0],     [[.6, .45], [.45, .6]], 300),
          rng.multivariate_normal([3.4, 2.6], [[.9, -.5], [-.5, .5]], 250),
          rng.multivariate_normal([-2.4, 2.8],[[.35, 0], [0, 1.5]],   200)]
y_true = np.r_[np.zeros(300), np.ones(250), np.full(200, 2)]

# ============ EM FROM SCRATCH ==========================================
def em_gmm(X, K, n_iter=100, seed=0, tol=1e-6):
    m, n = X.shape
    r = np.random.default_rng(seed)
    mu = X[r.choice(m, K, replace=False)]
    Sigma = np.array([np.cov(X.T) for _ in range(K)])
    pi = np.full(K, 1/K)
    lls = []
    for it in range(n_iter):
        # ---- E-step: responsibilities ---------------------------------
        dens = np.column_stack([
            pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k], allow_singular=True)
            for k in range(K)])
        total = dens.sum(1, keepdims=True)
        gamma = dens / np.maximum(total, 1e-300)
        ll = float(np.log(np.maximum(total, 1e-300)).sum())
        lls.append(ll)
        # ---- M-step: weighted MLE -------------------------------------
        mk = gamma.sum(0)                                  # effective counts
        pi = mk / m
        mu = (gamma.T @ X) / mk[:, None]
        for k in range(K):
            d = X - mu[k]
            Sigma[k] = (gamma[:, k, None] * d).T @ d / mk[k] + 1e-6*np.eye(n)
        if it > 0 and abs(lls[-1] - lls[-2]) < tol:
            break
    return pi, mu, Sigma, gamma, lls

pi, mu, Sig, gamma, lls = em_gmm(X, 3, seed=1)
print("=== EM from scratch ===")
print(f"converged in {len(lls)} iterations")
print(f"log-likelihood is monotone: "
      f"{all(lls[i] <= lls[i+1] + 1e-8 for i in range(len(lls)-1))}")
print(f"final log-likelihood = {lls[-1]:.4f}")
print(f"weights = {pi.round(4)}")
print(f"means   =\\n{mu.round(3)}")

gm = GaussianMixture(3, n_init=10, random_state=0).fit(X)
print(f"\\nsklearn log-likelihood = {gm.score(X)*len(X):.4f}")
print(f"sklearn weights = {np.sort(gm.weights_).round(4)}")
print(f"mine    weights = {np.sort(pi).round(4)}")
print(f"\\nARI vs truth: mine {adjusted_rand_score(y_true, gamma.argmax(1)):.4f}, "
      f"sklearn {adjusted_rand_score(y_true, gm.predict(X)):.4f}")

# ============ k-means IS the hard-assignment limit =====================
print("\\n=== k-means = EM with spherical covariance, sigma -> 0 ===")
km = KMeans(3, n_init=10, random_state=0).fit(X)
print(f"{'model':<44}{'ARI':>8}")
print(f"{'KMeans':<44}{adjusted_rand_score(y_true, km.labels_):>8.4f}")
for ct in ["spherical", "diag", "tied", "full"]:
    g = GaussianMixture(3, covariance_type=ct, n_init=10, random_state=0).fit(X)
    print(f"{f'GaussianMixture(covariance_type={ct!r})':<44}"
          f"{adjusted_rand_score(y_true, g.predict(X)):>8.4f}")
# force the hard limit
g_tiny = GaussianMixture(3, covariance_type="spherical", n_init=10,
                         reg_covar=1e-6, random_state=0).fit(X)
print(f"\\nspherical GMM assignments match k-means on "
      f"{np.mean(g_tiny.predict(X) == km.labels_)*100:.1f}% of instances "
      f"(up to label permutation this is essentially identical)")

# ============ ANOMALY DETECTION ========================================
print("\\n=== anomaly detection from the density ===")
densities = gm.score_samples(X)
for pct in [1, 2, 4, 10]:
    thr = np.percentile(densities, pct)
    print(f"  contamination {pct:>2}%: threshold log p = {thr:>8.3f}, "
          f"{int((densities < thr).sum()):>3} anomalies flagged")

# inject genuine outliers and see if they are caught
X_out = np.r_[X, rng.uniform(-7, 8, (25, 2))]
dens2 = gm.score_samples(X_out)
thr = np.percentile(gm.score_samples(X), 2)
flagged = dens2 < thr
print(f"\\ninjected 25 uniform outliers into {len(X)} normal points")
print(f"  caught {int(flagged[len(X):].sum())}/25 of the true outliers")
print(f"  false alarms among the normal points: {int(flagged[:len(X)].sum())}")

# ============ CHOOSING K: BIC and AIC ==================================
print("\\n=== BIC / AIC ===")
print(f"{'K':>4}{'params':>9}{'log L':>13}{'AIC':>12}{'BIC':>12}")
for k in range(1, 9):
    g = GaussianMixture(k, n_init=5, random_state=0).fit(X)
    n_par = k*2 + k*3 + (k-1)          # means + full covs (2x2) + weights
    print(f"{k:>4}{n_par:>9}{g.score(X)*len(X):>13.2f}{g.aic(X):>12.2f}{g.bic(X):>12.2f}")
print("\\nBoth are minimised at K=3 -- the truth. Unlike inertia, they are not")
print("monotone, so you can minimise them directly.")

# ============ BAYESIAN GMM: no K search needed =========================
print("\\n=== BayesianGaussianMixture picks K by itself ===")
for prior in [0.001, 0.1, 10.0]:
    b = BayesianGaussianMixture(n_components=10, n_init=5,
                                weight_concentration_prior=prior,
                                random_state=0).fit(X)
    active = int((b.weights_ > .02).sum())
    print(f"  prior={prior:<7} -> {active} active components   "
          f"weights {np.sort(b.weights_)[::-1][:5].round(3)}")
print("\\nA small concentration prior favours FEWER components.")
print("Give it an upper bound and it prunes the rest automatically.")
''',
        key="ch09_gmm",
    )

    quiz(
        "Which statement about EM for Gaussian mixtures is true?",
        ["It converges to the global maximum likelihood",
         "The log-likelihood is non-decreasing at every iteration",
         "It requires the clusters to be spherical",
         "It gives hard cluster assignments"],
        1,
        "The E-step makes Jensen's bound tight and the M-step increases it, so "
        "$\\ell$ never decreases. But like k-means it finds a *local* optimum, it "
        "handles arbitrary ellipsoidal covariances, and its assignments are soft "
        "(responsibilities).",
        key="ch09q2",
    )

    keypoints([
        "A GMM models $p(\\mathbf{x}) = \\sum_k \\pi_k \\mathcal{N}(\\boldsymbol\\mu_k, "
        "\\boldsymbol\\Sigma_k)$ — a generative density model.",
        "<b>EM</b>: E-step computes responsibilities $\\gamma_{ik}$, M-step does "
        "weighted MLE. $\\ell$ is monotone non-decreasing.",
        "<b>k-means is EM</b> with spherical equal covariance and hard "
        "assignments.",
        "Because it is a density, low $\\log p(\\mathbf{x})$ = anomaly. Set the "
        "threshold from the contamination rate.",
        "Choose $K$ by minimising <b>BIC</b> (or AIC) — or let "
        "<code>BayesianGaussianMixture</code> prune components for you.",
    ])


# ==========================================================================
def s_9_8():
    section("9.8", "Exercises & Chapter Review")

    lead("Thirteen exercises. Numbers 10–13 are substantial projects.")

    exercise(
        1, "How would you define clustering? Can you name a few clustering "
        "algorithms?",
        "Clustering is the **unsupervised** task of grouping similar instances "
        "together. \"Similar\" is defined by the algorithm's own criterion — "
        "distance to a centroid, density connectivity, likelihood under a mixture "
        "component — which is why different algorithms give genuinely different "
        "answers on the same data.\n\n"
        "Algorithms: **k-means**, **DBSCAN**, **HDBSCAN**, **agglomerative "
        "(hierarchical) clustering**, **BIRCH**, **mean-shift**, **affinity "
        "propagation**, **spectral clustering**, **Gaussian mixtures**.")

    exercise(
        2, "What are some of the main applications of clustering algorithms?",
        "**Customer segmentation**, **data analysis** (cluster then analyse each "
        "group), **dimensionality reduction** (`transform` gives distances to $k$ "
        "centroids), **feature engineering** (cluster id as a categorical "
        "feature), **anomaly detection** (low affinity to every cluster), "
        "**semi-supervised learning** (§9.5), **search engines** (return items "
        "from the query's cluster), and **image segmentation** (§9.4).")

    exercise(
        3, "Describe two techniques to select the right number of clusters when "
        "using k-means.",
        "**(1) The elbow method.** Plot inertia against $k$. Inertia always "
        "decreases, but it typically drops steeply and then flattens; the elbow is "
        "a reasonable choice. It is quick but crude, and often there is no clear "
        "elbow.\n\n"
        "**(2) The silhouette score.** Compute the mean silhouette coefficient "
        "$s = (b-a)/\\max(a,b)$ over all instances and pick the $k$ that maximises "
        "it. This is more expensive ($\\mathcal{O}(m^2)$) but far more reliable, "
        "because unlike inertia it is not monotone in $k$. Even better: plot the "
        "**silhouette diagram** and inspect the per-cluster knives.\n\n"
        "For a Gaussian mixture, minimise **BIC** or **AIC** instead (§9.7).")

    exercise(
        4, "What is label propagation? Why would you implement it, and how?",
        "**What:** copying labels from a small set of labelled instances to a "
        "large set of unlabelled ones.\n\n"
        "**Why:** labelling is usually the most expensive and time-consuming part "
        "of a machine learning project. If you have a few labels and lots of "
        "unlabelled data, propagating turns a handful of labels into a full "
        "training set — often reaching 80–90 % of the fully-supervised accuracy "
        "for 1–2 % of the labelling cost.\n\n"
        "**How:** cluster the data (e.g. with k-means), find the instance closest "
        "to each centroid, label those by hand, then assign that label to every "
        "instance in the same cluster. Optionally propagate only to the instances "
        "**closest to the centroid** (say the closest 75 %), which raises label "
        "purity at the cost of quantity. `LabelSpreading` and `LabelPropagation` "
        "implement a graph-based variant.")

    exercise(
        5, "Can you name two clustering algorithms that can scale to large "
        "datasets? And two that look for regions of high density?",
        "**Scale to large datasets:** **k-means** (especially "
        "`MiniBatchKMeans`, which supports `partial_fit`) and **BIRCH**, which "
        "builds a compact CF-tree in one pass and was designed specifically for "
        "very large datasets that do not fit in memory.\n\n"
        "**Look for high-density regions:** **DBSCAN** and **mean-shift**. "
        "(HDBSCAN also qualifies, and is generally the better choice of the "
        "density family.)")

    exercise(
        6, "Can you think of a use case where active learning would be useful? "
        "How would you implement it?",
        "Active learning is useful whenever **labels are expensive but unlabelled "
        "data is plentiful** — medical imaging (a radiologist's time), legal "
        "document review, industrial defect inspection, or any task needing "
        "domain-expert annotation.\n\n"
        "The most common implementation is **uncertainty sampling**:\n\n"
        "1. Train the model on the labelled instances you have.\n"
        "2. Run it on all the *unlabelled* instances and find the ones it is "
        "least confident about — smallest margin between the top two predicted "
        "probabilities, or highest predictive entropy.\n"
        "3. Have a human label exactly those.\n"
        "4. Repeat until the accuracy gain per label stops justifying the cost.\n\n"
        "The lab in §9.5 implements this loop and shows it beating random "
        "labelling at the same budget.")

    exercise(
        7, "What is the difference between anomaly detection and novelty "
        "detection?",
        "Both spot unusual instances, but they differ in **what the training set "
        "is allowed to contain**.\n\n"
        "* **Anomaly detection** (a.k.a. outlier detection) assumes the training "
        "set is *contaminated* — it contains some outliers, and the algorithm's "
        "job is to find them. Examples: `IsolationForest`, "
        "`LocalOutlierFactor(novelty=False)`, `EllipticEnvelope`, a GMM density "
        "threshold.\n\n"
        "* **Novelty detection** assumes the training set is **clean** — it "
        "contains only normal instances — and the job is to detect whether a "
        "*new* instance differs from them. Examples: `OneClassSVM`, "
        "`LocalOutlierFactor(novelty=True)`.\n\n"
        "The distinction matters in practice: a novelty detector trained on "
        "contaminated data will treat the contamination as normal.")

    exercise(
        8, "What is a Gaussian mixture? What tasks can you use it for?",
        "A **Gaussian mixture model** is a probabilistic model that assumes each "
        "instance was generated in two steps: first pick a component $k$ with "
        "probability $\\pi_k$, then draw $\\mathbf{x} \\sim "
        "\\mathcal{N}(\\boldsymbol\\mu_k, \\boldsymbol\\Sigma_k)$. It is fitted by "
        "**EM**.\n\n"
        "**Tasks:**\n"
        "* **Clustering** — soft (`predict_proba`) or hard (`predict`), with "
        "ellipsoidal rather than spherical clusters.\n"
        "* **Density estimation** — `score_samples` gives $\\log p(\\mathbf{x})$.\n"
        "* **Anomaly detection** — flag instances in low-density regions.\n"
        "* **Generation** — `sample()` draws new instances from the fitted "
        "distribution.\n"
        "* **Model selection** — `bic()` / `aic()` for choosing $K$ properly.")

    exercise(
        9, "Can you name two techniques to find the right number of clusters when "
        "using a Gaussian mixture model?",
        "**(1) Minimise a theoretical information criterion** — the **BIC** "
        "$\\log(m)p - 2\\log\\hat L$ or the **AIC** $2p - 2\\log\\hat L$. Fit the "
        "model for a range of $K$ and pick the minimum. Unlike inertia these are "
        "not monotone, so the minimum is meaningful. BIC penalises complexity more "
        "heavily and therefore usually selects fewer components.\n\n"
        "**(2) Use a `BayesianGaussianMixture`.** Set `n_components` to a "
        "generous upper bound; the Dirichlet-process prior drives the weights of "
        "unnecessary components to (nearly) zero automatically, so you read $K$ "
        "off the surviving weights. Tuning "
        "`weight_concentration_prior` biases it toward fewer or more components.")

    exercise(
        10, "The classic Olivetti faces dataset contains 400 grayscale 64 × 64 "
        "pixel images of faces. Load it, split it into a training set, a "
        "validation set, and a test set (note that the dataset is already scaled "
        "between 0 and 1). Since the dataset is quite small, you will probably "
        "want to use stratified sampling to ensure that there are the same number "
        "of images per person in each set. Next, cluster the images using k-means, "
        "and ensure that you have a good number of clusters (using one of the "
        "techniques discussed in this chapter). Visualize the clusters: do you see "
        "similar faces in each cluster?",
        "Use `StratifiedShuffleSplit` on the person id — with 40 people × 10 "
        "images each and a 40 image test set, stratification guarantees one image "
        "per person rather than an arbitrary draw.\n\n"
        "Faces are 4 096-dimensional, so run **PCA to ~99 % variance first** "
        "(around 200 components) — this both speeds up k-means considerably and "
        "improves the clustering, for the curse-of-dimensionality reasons of §8.1.\n\n"
        "Sweep $k$ from about 5 to 150 and use the **silhouette score**. The "
        "maximum typically lands somewhere between 100 and 120 — considerably more "
        "than the 40 people, because k-means separates the same person's images by "
        "**pose, expression and lighting** as well as identity. Visualising the "
        "clusters makes this obvious: some clusters are one person, others are "
        "\"people wearing glasses\" or \"faces looking left\".",
        code='''from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(split.split(X, y))
X_train_valid, y_train_valid = X[train_valid_idx], y[train_valid_idx]
X_test, y_test = X[test_idx], y[test_idx]

split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(split.split(X_train_valid, y_train_valid))
X_train, y_train = X_train_valid[train_idx], y_train_valid[train_idx]
X_valid, y_valid = X_train_valid[valid_idx], y_train_valid[valid_idx]

pca = PCA(0.99).fit(X_train)
X_train_pca = pca.transform(X_train)
print("components:", pca.n_components_)

best = (None, -1)
for k in range(5, 150, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_train_pca)
    s = silhouette_score(X_train_pca, km.labels_)
    if s > best[1]:
        best = (k, s)
print("best k:", best)''')

    exercise(
        11, "Continuing with the Olivetti faces dataset, train a classifier to "
        "predict which person is represented in each picture, and evaluate it on "
        "the validation set. Next, use k-means as a dimensionality reduction tool, "
        "and train a classifier on the reduced set. Search for the number of "
        "clusters that allows the classifier to get the best performance: what "
        "performance can you reach? What if you append the features from the "
        "reduced set to the original features (again, searching for the best "
        "number of clusters)?",
        "A `RandomForestClassifier` on the raw 4 096 pixels reaches roughly "
        "**90 %** on the validation set.\n\n"
        "Replacing the pixels with `KMeans(k).transform(X)` — the $k$ centroid "
        "distances — and sweeping $k$ typically peaks a little **below** the "
        "baseline. The reduced representation loses identity information: two "
        "different people in the same pose can be closer to the same centroid than "
        "two images of the same person in different poses.\n\n"
        "**Appending** the cluster distances to the original features is the "
        "interesting case. It generally gives a small improvement, because the "
        "centroid distances are a genuinely different, non-linear view of the "
        "data — the classifier gets both the raw pixels and a summary of \"which "
        "face-prototypes does this resemble\". This is exactly the "
        "`ClusterSimilarity` idea from §2.4, and it is the pattern worth "
        "remembering: **clustering as feature augmentation, not feature "
        "replacement.**",
        code='''from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)
print("baseline:", clf.score(X_valid, y_valid))

# k-means as a REPLACEMENT
for k in range(5, 150, 5):
    pipe = make_pipeline(KMeans(n_clusters=k, n_init=10, random_state=42),
                         RandomForestClassifier(n_estimators=150, random_state=42))
    pipe.fit(X_train, y_train)
    print(k, pipe.score(X_valid, y_valid))

# k-means as an AUGMENTATION -- usually better
for k in range(5, 150, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_train)
    X_tr_ext = np.c_[X_train, km.transform(X_train)]
    X_va_ext = np.c_[X_valid, km.transform(X_valid)]
    c = RandomForestClassifier(n_estimators=150, random_state=42)
    c.fit(X_tr_ext, y_train)
    print(k, c.score(X_va_ext, y_valid))''')

    exercise(
        12, "Train a Gaussian mixture model on the Olivetti faces dataset. To "
        "speed up the algorithm, you should probably reduce the dataset's "
        "dimensionality (e.g., use PCA, preserving 99 % of the variance). Use the "
        "model to generate some new faces (using the `sample()` method), and "
        "visualize them (if you used PCA, you will need to use its "
        "`inverse_transform()` method). Try to modify some images (e.g., rotate, "
        "flip, darken) and see if the model can detect the anomalies (i.e., "
        "compare the output of the `score_samples()` method for normal images and "
        "for anomalies).",
        "**Generation.** `gm.sample(n)` draws from the fitted mixture in PCA "
        "space; `pca.inverse_transform` maps back to 4 096 pixels. The results are "
        "recognisably face-like but blurry — a GMM in a 200-dimensional linear "
        "subspace is a weak generative model. Chapter 17's VAEs and diffusion "
        "models do this properly.\n\n"
        "**Anomaly detection.** This works well. Rotated, flipped and darkened "
        "faces get **much** lower `score_samples` values than normal ones, and "
        "the separation is large enough that a simple percentile threshold catches "
        "essentially all of them.\n\n"
        "One caveat worth understanding: `score_samples` is computed in **PCA "
        "space**, so it measures how unusual the *reconstruction* is. An anomaly "
        "that happens to lie inside the PCA subspace will be missed. Checking the "
        "**reconstruction error** $\\lVert \\mathbf{x} - "
        "\\text{pca.inverse\\_transform}(\\text{pca.transform}(\\mathbf{x}))\\rVert$ "
        "alongside the density catches a complementary set of anomalies.",
        code='''from sklearn.mixture import GaussianMixture
import numpy as np

pca = PCA(0.99).fit(X_train)
X_train_pca = pca.transform(X_train)

gm = GaussianMixture(n_components=40, random_state=42).fit(X_train_pca)

# generate
gen_pca, _ = gm.sample(n_samples=20)
gen_faces = pca.inverse_transform(gen_pca).reshape(-1, 64, 64)

# build anomalies
n_rot, n_flip, n_dark = 4, 3, 3
rotated = np.transpose(X_train[:n_rot].reshape(-1, 64, 64), (0, 2, 1))
flipped = X_train[n_rot:n_rot+n_flip].reshape(-1, 64, 64)[:, ::-1]
darkened = X_train[n_rot+n_flip:n_rot+n_flip+n_dark].reshape(-1, 64, 64) * 0.3
bad = np.r_[rotated, flipped, darkened].reshape(-1, 64*64)

print("normal    :", gm.score_samples(pca.transform(X_train[:10])).round(1))
print("anomalies :", gm.score_samples(pca.transform(bad)).round(1))

# the complementary check: reconstruction error
def recon_err(X):
    return np.sum((pca.inverse_transform(pca.transform(X)) - X) ** 2, axis=1)
print("recon err normal   :", recon_err(X_train[:10]).round(3))
print("recon err anomalies:", recon_err(bad).round(3))''')

    exercise(
        13, "Some dimensionality reduction techniques can also be used for anomaly "
        "detection. For example, take the Olivetti faces dataset and reduce it with "
        "PCA, preserving 99 % of the variance. Then compute the reconstruction "
        "error for each image. Next, take some of the modified images you built in "
        "the previous exercise and look at their reconstruction error: notice how "
        "much larger it is. If you plot a reconstructed image, you will see why: it "
        "tries to reconstruct a normal face.",
        "This is a genuinely useful technique and it generalises far beyond faces.\n\n"
        "PCA learns a linear subspace that captures the variation present in "
        "**normal** data. A normal face lies close to that subspace, so projecting "
        "and reconstructing loses little — small reconstruction error. An anomaly "
        "lies *off* the subspace, so the projection discards the very thing that "
        "made it anomalous — large reconstruction error.\n\n"
        "Plotting the reconstruction makes the mechanism visible: feed in an "
        "upside-down face and PCA returns something that looks like a **normal, "
        "right-way-up face**, because that is all its basis can express. The "
        "difference between input and output is exactly the anomaly.\n\n"
        "The same idea, with a non-linear encoder/decoder instead of PCA, is the "
        "**autoencoder anomaly detector** of Chapter 17 — and it is one of the "
        "standard industrial approaches to defect detection.",
        code='''def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    return np.mean(np.square(X_reconstructed - X), axis=-1)

print("normal faces    :", reconstruction_error(pca, X_valid).mean())
print("anomalous images:", reconstruction_error(pca, bad).mean())

# plot one to see WHY
import matplotlib.pyplot as plt
plt.subplot(121); plt.imshow(bad[0].reshape(64, 64), cmap="gray")
plt.title("input (rotated)"); plt.axis("off")
plt.subplot(122)
plt.imshow(pca.inverse_transform(pca.transform(bad[:1]))[0].reshape(64, 64),
           cmap="gray")
plt.title("PCA reconstruction"); plt.axis("off")
# PCA reconstructs a NORMAL face -- it cannot represent the rotation''')

    rule()

    sub("Choosing a clustering algorithm")

    table(
        ["If you…", "Use", "Because"],
        [["know $k$ and expect round clusters", "<b>k-means</b>",
          "Fastest, scales, well understood"],
         ["have huge data", "<b>MiniBatchKMeans</b> or <b>BIRCH</b>",
          "Both support out-of-core"],
         ["do not know $k$ and want outliers found",
          "<b>HDBSCAN</b> (or DBSCAN)", "Density-based, no $k$, noise label"],
         ["expect elongated / overlapping clusters",
          "<b>GaussianMixture</b>", "Full covariance, soft assignments"],
         ["expect non-convex clusters", "<b>Spectral clustering</b> or HDBSCAN",
          "Graph-based, shape-agnostic"],
         ["want a hierarchy or a dendrogram", "<b>Agglomerative</b>",
          "Gives the whole merge tree"],
         ["want a density model, not just labels", "<b>GaussianMixture</b>",
          "<code>score_samples</code>, <code>sample</code>, BIC"]],
    )

    keypoints([
        "$k$-means: fast, needs $k$, assumes spherical/equal-size/equal-density; "
        "Lloyd converges to a local optimum.",
        "Choose $k$ with the <b>silhouette</b> (and its diagram), not inertia.",
        "DBSCAN/HDBSCAN: density-based, no $k$, any shape, outliers as output.",
        "GMM + EM: soft assignments, a real density, anomaly detection, and BIC "
        "for model selection.",
        "The most valuable practical use of clustering is <b>semi-supervised "
        "labelling</b> (§9.5).",
    ], title="Chapter 9 in five lines")

    refs([
        ("Lloyd, S. — *Least Squares Quantization in PCM*",
         "https://doi.org/10.1109/TIT.1982.1056489"),
        ("Arthur & Vassilvitskii — *k-means++: The Advantages of Careful Seeding*",
         "SODA 2007"),
        ("Ester, Kriegel, Sander & Xu — *A Density-Based Algorithm for Discovering "
         "Clusters* (DBSCAN)", "KDD 1996"),
        ("Campello, Moulavi & Sander — *Density-Based Clustering Based on "
         "Hierarchical Density Estimates* (HDBSCAN)",
         "https://doi.org/10.1007/978-3-642-37456-2_14"),
        ("Dempster, Laird & Rubin — *Maximum Likelihood from Incomplete Data via "
         "the EM Algorithm*",
         "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x"),
        ("Rousseeuw, P. — *Silhouettes: A Graphical Aid to the Interpretation and "
         "Validation of Cluster Analysis*",
         "https://doi.org/10.1016/0377-0427(87)90125-7"),
    ])


# ==========================================================================
SECTIONS = [
    ("9.1", "Clustering and k-means", s_9_1),
    ("9.2", "Initialisation & Accelerated k-means", s_9_2),
    ("9.3", "Finding the Optimal k", s_9_3),
    ("9.4", "Limits & Image Segmentation", s_9_4),
    ("9.5", "Semi-Supervised Learning", s_9_5),
    ("9.6", "DBSCAN & Other Algorithms", s_9_6),
    ("9.7", "Gaussian Mixtures & EM", s_9_7),
    ("9.8", "Exercises & Review", s_9_8),
]

nav.render_chapter(CH, SECTIONS)
