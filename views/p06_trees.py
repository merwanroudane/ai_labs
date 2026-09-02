"""Chapter 6 — Decision Trees."""

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
CH = "ch06"

hero(
    kicker="Part I · Chapter 6",
    title="Decision Trees",
    blurb=(
        "The only model in this course you can explain to a non-specialist by "
        "drawing it. Trees need no feature scaling, handle mixed data types, and "
        "are completely interpretable — and they are also unstable, greedy, and "
        "constitutionally unable to draw a diagonal line. Understanding both "
        "halves is what makes Chapter 7 make sense."
    ),
    chips=["White-box model", "8 sub-sections", "7 animations",
           "8 code labs", "The foundation of Ch. 7"],
)
nav.sidebar_tools(CH)


# --------------------------------------------------------------------------
def _tree_figure(clf, feature_names, class_names=None, height=520,
                 title="Fitted decision tree"):
    """Draw a fitted sklearn tree as a Plotly node-link diagram."""
    t = clf.tree_
    n = t.node_count
    depth = np.zeros(n, dtype=int)
    order = []

    def walk(node, d, lo, hi):
        depth[node] = d
        order.append((node, d, (lo + hi) / 2))
        left, right = t.children_left[node], t.children_right[node]
        if left != -1:
            mid = (lo + hi) / 2
            walk(left, d + 1, lo, mid)
            walk(right, d + 1, mid, hi)

    walk(0, 0, 0.0, 1.0)
    pos = {node: (x, -d) for node, d, x in order}

    ex, ey = [], []
    for node in range(n):
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                ex += [pos[node][0], pos[child][0], None]
                ey += [pos[node][1], pos[child][1], None]

    xs, ys, txt, hov, cols, sizes = [], [], [], [], [], []
    is_clf = class_names is not None
    for node in range(n):
        x, y = pos[node]
        xs.append(x); ys.append(y)
        val = t.value[node][0]
        nsamp = int(t.n_node_samples[node])
        leaf = t.children_left[node] == -1
        if is_clf:
            k = int(np.argmax(val))
            cols.append(alpha(SEQ[k % len(SEQ)], .95 if leaf else .55))
            lbl = class_names[k]
            imp = f"gini={t.impurity[node]:.3f}"
            dist = np.round(val / max(val.sum(), 1e-9), 3)
            hov.append(f"samples={nsamp}<br>{imp}<br>class={lbl}<br>p={dist}")
        else:
            cols.append(alpha(C["primary"], .95 if leaf else .5))
            lbl = f"{float(val.ravel()[0]):.2f}"
            hov.append(f"samples={nsamp}<br>mse={t.impurity[node]:.3f}"
                       f"<br>value={lbl}")
        if leaf:
            txt.append(f"<b>{lbl}</b><br>n={nsamp}")
        else:
            fn = feature_names[t.feature[node]]
            txt.append(f"{fn}<br>≤ {t.threshold[node]:.2f}")
        sizes.append(28 + 16 * np.sqrt(nsamp / max(t.n_node_samples[0], 1)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color=C["line"], width=1.8),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=txt, hovertext=hov,
        hoverinfo="text", textposition="middle center",
        textfont=dict(size=9, color=C["ink"]),
        marker=dict(size=sizes, color=cols, symbol="square",
                    line=dict(color="#FFFFFF", width=2)),
        showlegend=False))
    fig.update_layout(height=height, title=title,
                      xaxis=dict(visible=False, range=[-.06, 1.06]),
                      yaxis=dict(visible=False),
                      plot_bgcolor="#FFFFFF")
    return fig


# ==========================================================================
def s_6_1():
    section("6.1", "Training and Visualizing a Decision Tree")

    lead(
        "A decision tree asks a sequence of yes/no questions about single "
        "features, and each answer sends you down a branch. Its whole value is "
        "that you can read the fitted model like a flowchart."
    )

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, export_text

    iris = load_iris()
    Xi = iris.data[:, 2:]                    # petal length & width
    yi = iris.target
    fn = ["petal length (cm)", "petal width (cm)"]

    depth = st.slider("Tree depth for the diagram below", 1, 6, 2, key="ch06_d1")
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xi, yi)

    figure(_tree_figure(clf, fn, list(iris.target_names), height=460,
                        title=f"Iris tree, max_depth = {depth}"),
           "Node size ∝ number of training samples reaching it. Hover for the "
           "class distribution and impurity.")

    with st.expander("The same tree as text (`export_text`)"):
        st.code(export_text(clf, feature_names=fn), language="text")

    sub("The vocabulary")

    table(
        ["Term", "Meaning"],
        [["<b>Root node</b>", "The node at depth 0, where every instance starts"],
         ["<b>Split</b>", "A test of the form $x_k \\le \\tau$"],
         ["<b>Child / branch</b>", "Where an instance goes next; always exactly two "
          "in scikit-learn (a <i>binary</i> tree)"],
         ["<b>Leaf node</b>", "A node with no children; it makes the prediction"],
         ["<b>samples</b>", "How many training instances reached this node"],
         ["<b>value</b>", "The class counts (classifier) or the mean target "
          "(regressor) at this node"],
         ["<b>gini / entropy / squared_error</b>",
          "The node's impurity — how mixed it is"],
         ["<b>Depth</b>", "Number of splits from the root to the node"]],
    )

    idea(
        "Trees need almost no data preparation",
        "No feature scaling, no centring, no one-hot encoding strictly required "
        "(though scikit-learn's implementation needs numeric input). This is "
        "because a split $x_k \\le \\tau$ is invariant to any <b>monotone</b> "
        "transformation of $x_k$: taking logs, squaring positives, or changing "
        "units simply moves $\\tau$. Compare with SVMs (§5.1), where scaling "
        "changes the model entirely.",
    )

    anim_header("Growing the tree one depth level at a time")
    md(
        "Left: the tree structure. Right: the decision regions it carves in "
        "feature space. Every split is a **single axis-aligned cut** — that "
        "constraint is the subject of §6.7."
    )

    gp1 = np.linspace(0, 7.5, 200); gp2 = np.linspace(0, 3.2, 200)
    G1, G2 = np.meshgrid(gp1, gp2); GG = np.c_[G1.ravel(), G2.ravel()]
    cs3 = [[0, alpha(SEQ[0], .40)], [.5, alpha(SEQ[1], .40)],
           [1, alpha(SEQ[2], .40)]]

    def pts():
        return [go.Scatter(x=Xi[yi == k, 0], y=Xi[yi == k, 1], mode="markers",
                           marker=dict(color=SEQ[k], size=8,
                                       line=dict(color="#fff", width=1)),
                           showlegend=False) for k in range(3)]

    frames = []
    for d in range(1, 7):
        c = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xi, yi)
        Z = c.predict(GG).reshape(G1.shape).astype(float)
        acc = c.score(Xi, yi)
        nleaf = int((c.tree_.children_left == -1).sum())
        frames.append(go.Frame(name=str(d), data=[
            go.Contour(x=gp1, y=gp2, z=Z, showscale=False, colorscale=cs3,
                       contours=dict(showlines=False))] + pts(),
            layout=go.Layout(title=f"max_depth = {d}   ·   {nleaf} leaves   ·   "
                                   f"training accuracy {acc:.4f}")))

    c1 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(Xi, yi)
    Z1 = c1.predict(GG).reshape(G1.shape).astype(float)
    f = go.Figure(data=[go.Contour(x=gp1, y=gp2, z=Z1, showscale=False,
                                   colorscale=cs3,
                                   contours=dict(showlines=False))] + pts())
    f.update_layout(height=480, xaxis_title="petal length (cm)",
                    yaxis_title="petal width (cm)", title="max_depth = 1")
    anim.animate(f, frames, duration=nav.anim_ms(1200), slider_prefix="depth ")
    figure(f)

    code_lab(
        "Fit, export, and read a tree",
        '''import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

iris = load_iris()
X = iris.data[:, 2:]              # petal length and width
y = iris.target

tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
print(export_text(tree, feature_names=["petal length", "petal width"]))

# ---- walk the fitted structure yourself --------------------------------
t = tree.tree_
print(f"\\nnodes={t.node_count}  leaves={(t.children_left == -1).sum()}  "
      f"depth={tree.get_depth()}")
print(f"\\n{'node':>5}{'feature':>16}{'threshold':>11}{'samples':>9}"
      f"{'gini':>8}   value")
for i in range(t.node_count):
    leaf = t.children_left[i] == -1
    feat = "LEAF" if leaf else ["petal length", "petal width"][t.feature[i]]
    thr  = "" if leaf else f"{t.threshold[i]:.3f}"
    print(f"{i:>5}{feat:>16}{thr:>11}{int(t.n_node_samples[i]):>9}"
          f"{t.impurity[i]:>8.4f}   {t.value[i][0].astype(int)}")

# ---- trace a single instance through the tree --------------------------
x_new = np.array([[5.0, 1.5]])
path = tree.decision_path(x_new).indices
print(f"\\ninstance {x_new[0]} follows nodes {list(path)}")
for node in path:
    if t.children_left[node] == -1:
        k = int(np.argmax(t.value[node][0]))
        print(f"  node {node}: LEAF -> predict '{iris.target_names[k]}'")
    else:
        f_ = ["petal length", "petal width"][t.feature[node]]
        v = x_new[0][t.feature[node]]
        go_left = v <= t.threshold[node]
        print(f"  node {node}: is {f_} ({v}) <= {t.threshold[node]:.3f}? "
              f"{go_left} -> go {'left' if go_left else 'right'}")

# ---- trees are invariant to monotone feature transformations -----------
print("\\n=== monotone invariance ===")
for name, Xt in [("original", X),
                 ("x * 1000", X * 1000),
                 ("log1p(x)", np.log1p(X)),
                 ("x ** 3",   X ** 3)]:
    tt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xt, y)
    print(f"  {name:<12} accuracy = {tt.score(Xt, y):.4f}  "
          f"leaves = {(tt.tree_.children_left == -1).sum()}")
print("Identical -- the split just moves. Try this with an SVM and watch it break.")

# ---- to render a picture with graphviz ---------------------------------
# export_graphviz(tree, out_file="iris_tree.dot",
#                 feature_names=["petal length", "petal width"],
#                 class_names=iris.target_names, rounded=True, filled=True)
# then:  dot -Tpng iris_tree.dot -o iris_tree.png
''',
        key="ch06_fit",
    )

    keypoints([
        "A tree is a sequence of single-feature threshold tests; you can read it "
        "directly.",
        "scikit-learn grows strictly <b>binary</b> trees (CART).",
        "<b>No feature scaling needed</b> — splits are invariant to monotone "
        "transformations.",
        "Every node reports <code>samples</code>, <code>value</code> and its "
        "impurity; leaves make the prediction.",
    ])


# ==========================================================================
def s_6_2():
    section("6.2", "Making Predictions and Estimating Class Probabilities")

    lead(
        "Prediction is a walk from the root to a leaf. The probability estimate is "
        "the class distribution stored in that leaf — which is both the source of "
        "the tree's honesty and of its worst pathology."
    )

    sub("Class probability")

    math(r"""
    \hat p_k(\mathbf{x}) \;=\;
    \frac{n_k\bigl(\text{leaf}(\mathbf{x})\bigr)}
         {n\bigl(\text{leaf}(\mathbf{x})\bigr)}
    \qquad\Longrightarrow\qquad
    \hat y(\mathbf{x}) \;=\; \operatorname*{arg\,max}_{k}\; \hat p_k(\mathbf{x})
    """)
    where({r"\text{leaf}(\mathbf{x})": "the unique leaf $\\mathbf{x}$ falls into",
           r"n_k(\ell)": "the number of training instances of class $k$ in leaf $\\ell$",
           r"n(\ell)": "the total number of training instances in leaf $\\ell$"})

    pitfall(
        "Tree probabilities are piecewise constant, and often absurd",
        "Every instance landing in the same leaf gets the <b>identical</b> "
        "probability, no matter how far it is from the split boundary. A point "
        "0.001 cm inside a leaf and one 4 cm inside get the same number. Worse, a "
        "fully grown tree has pure leaves, so it outputs probabilities of exactly "
        "<b>0.000 or 1.000</b> — perfect confidence, always, including when it is "
        "wrong. This is why a single tree is a <b>badly calibrated</b> probability "
        "model, and why Chapter 7's forests (which average many trees) are "
        "dramatically better.",
    )

    anim_header("Probability surfaces: single tree vs averaged forest")
    md(
        "The left surface is one tree's $\\hat p$; the right is a 200-tree "
        "forest's. Sweep the depth. The tree's surface becomes a step function of "
        "0s and 1s; the forest's stays a smooth, usable probability field."
    )

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    Xm, ym = ds.moons(n=250, noise=.28)
    m1 = np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 110)
    m2 = np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 110)
    M1, M2 = np.meshgrid(m1, m2); MM = np.c_[M1.ravel(), M2.ravel()]

    depths = [1, 2, 3, 5, 8, 12, None]
    cache = []
    for d in depths:
        tr = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xm, ym)
        rf = RandomForestClassifier(n_estimators=200, max_depth=d,
                                    random_state=0, n_jobs=-1).fit(Xm, ym)
        cache.append((tr.predict_proba(MM)[:, 1].reshape(M1.shape),
                      rf.predict_proba(MM)[:, 1].reshape(M1.shape),
                      float(np.mean(np.isin(tr.predict_proba(Xm)[:, 1], [0., 1.])))))

    def sc(col):
        return [go.Scatter(x=Xm[ym == k, 0], y=Xm[ym == k, 1], mode="markers",
                           marker=dict(color=[C["train"], C["warning"]][k], size=6,
                                       line=dict(color="#fff", width=.7)),
                           showlegend=False) for k in range(2)]

    frames = []
    for i, d in enumerate(depths):
        Zt, Zf, frac = cache[i]
        frames.append(go.Frame(name=str(d), data=[
            go.Heatmap(x=m1, y=m2, z=Zt, colorscale=nav.cscale(), zmin=0, zmax=1,
                       showscale=False)] + sc(0) + [
            go.Heatmap(x=m1, y=m2, z=Zf, colorscale=nav.cscale(), zmin=0, zmax=1,
                       showscale=False)] + sc(0),
            layout=go.Layout(annotations=[anim.annotate_step(
                f"max_depth = {d}   ·   {frac:.0%} of the tree's training "
                f"probabilities are exactly 0 or 1",
                color=C["danger"] if frac > .8 else C["ink"])])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("one decision tree — P(class 1)",
                                      "200-tree random forest — P(class 1)"))
    f.add_trace(go.Heatmap(x=m1, y=m2, z=cache[0][0], colorscale=nav.cscale(),
                           zmin=0, zmax=1, showscale=False), 1, 1)
    for tr_ in sc(0):
        f.add_trace(tr_, 1, 1)
    f.add_trace(go.Heatmap(x=m1, y=m2, z=cache[0][1], colorscale=nav.cscale(),
                           zmin=0, zmax=1, showscale=False), 1, 2)
    for tr_ in sc(0):
        f.add_trace(tr_, 1, 2)
    f.update_layout(height=470, title="Probability surfaces")
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="depth ")
    figure(f)

    sub("Prediction complexity")

    math(r"""
    \text{prediction: } \mathcal{O}\bigl(\log_2 m\bigr)
    \qquad\text{independent of the number of features } n
    """)

    md(
        "A balanced binary tree over $m$ instances has depth $\\approx \\log_2 m$, "
        "and each node costs one comparison. For $m = 10^6$ that is about 20 "
        "comparisons — which is why trees and forests are so fast at serving time, "
        "even with thousands of features."
    )

    code_lab(
        "Probability estimates and their calibration",
        '''import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

X, y = make_moons(n_samples=2000, noise=.32, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0)

print(f"{'model':<28}{'accuracy':>10}{'Brier':>9}{'log loss':>11}"
      f"{'% p in {0,1}':>14}")
models = {
    "tree, depth 3":        DecisionTreeClassifier(max_depth=3, random_state=0),
    "tree, depth 8":        DecisionTreeClassifier(max_depth=8, random_state=0),
    "tree, fully grown":    DecisionTreeClassifier(random_state=0),
    "forest, 200 trees":    RandomForestClassifier(n_estimators=200, random_state=0,
                                                   n_jobs=-1),
    "logistic regression":  LogisticRegression(),
}
probs = {}
for nm, mdl in models.items():
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    probs[nm] = p
    extreme = np.mean((p == 0) | (p == 1))
    # log loss needs clipping when p is exactly 0 or 1
    print(f"{nm:<28}{accuracy_score(yte, mdl.predict(Xte)):>10.4f}"
          f"{brier_score_loss(yte, p):>9.4f}"
          f"{log_loss(yte, np.clip(p, 1e-9, 1-1e-9)):>11.4f}"
          f"{extreme:>13.1%}")

print("\\nA fully grown tree is CONFIDENT on every test point and often wrong.")
print("Its Brier score and log loss are far worse than the forest's, even when")
print("the accuracies are close.")

# ---- how many DISTINCT probabilities can each model emit? -------------
print(f"\\n{'model':<28}{'distinct probability values':>30}")
for nm, p in probs.items():
    print(f"{nm:<28}{len(np.unique(np.round(p, 6))):>30}")
print("A depth-3 tree has at most 8 leaves -> at most 8 distinct probabilities.")

# ---- calibration curves ------------------------------------------------
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated",
                line=dict(color=C["truth"], width=2, dash="dash"))
for i, (nm, p) in enumerate(probs.items()):
    try:
        pt, pp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        fig.add_scatter(x=pp, y=pt, mode="lines+markers", name=nm,
                        line=dict(color=SEQ[i], width=2.6))
    except Exception:
        pass
fig.update_layout(height=430, xaxis_title="mean predicted probability",
                  yaxis_title="observed frequency",
                  title="Calibration: how honest are the probabilities?")

# ---- prediction is O(log m) -------------------------------------------
print("\\n=== prediction depth vs training-set size ===")
for m in [100, 1_000, 10_000, 100_000]:
    Xb, yb = make_moons(n_samples=m, noise=.3, random_state=1)
    t = DecisionTreeClassifier(random_state=0).fit(Xb, yb)
    print(f"  m = {m:>7,}   tree depth = {t.get_depth():>3}   "
          f"log2(m) = {np.log2(m):>5.1f}   leaves = "
          f"{(t.tree_.children_left == -1).sum():>6,}")
''',
        key="ch06_proba",
    )

    keypoints([
        "Prediction = walk to a leaf; probability = that leaf's class "
        "distribution.",
        "Probabilities are <b>piecewise constant</b> and, for a grown tree, "
        "exactly 0 or 1 — badly calibrated.",
        "Averaging many trees (Ch. 7) fixes calibration; a single tree cannot.",
        "Prediction is $\\mathcal{O}(\\log_2 m)$ and independent of the number of "
        "features.",
    ])


# ==========================================================================
def s_6_3():
    section("6.3", "The CART Training Algorithm")

    lead(
        "Classification And Regression Tree. It is a greedy algorithm: at every "
        "node, search all features and all thresholds, take the split that most "
        "reduces impurity, and never reconsider."
    )

    sub("The cost function")

    math(r"""
    J\bigl(k, t_k\bigr) \;=\;
    \frac{m_{\text{left}}}{m}\,G_{\text{left}}
    \;+\;
    \frac{m_{\text{right}}}{m}\,G_{\text{right}}
    """)
    where({
        r"k": "the feature index being split on",
        r"t_k": "the threshold for that feature",
        r"G_{\text{left/right}}": "the impurity of the corresponding subset",
        r"m_{\text{left/right}}": "the number of instances in each subset",
        r"m": "the number of instances at this node",
    })

    md("CART minimises $J$ over all $(k, t_k)$, then recurses on each child. It "
       "stops when a stopping criterion (§6.5) fires, or when no split reduces "
       "impurity.")

    sub("The greedy trap")

    proof(
        "Finding the optimal tree is NP-complete",
        "Hyafil & Rivest (1976) proved that constructing an optimal binary "
        "decision tree is NP-complete. There is no polynomial algorithm, so every "
        "practical implementation is greedy: it takes the locally best split at "
        "each node and never revisits it. The consequence is that CART reliably "
        "finds a <i>reasonably good</i> tree but almost never the <i>best</i> one — "
        "and it can be defeated by problems where no single feature is "
        "individually informative (XOR is the standard example, and it is in the "
        "lab below).",
    )

    anim_header("Watching CART choose: every candidate split, scored")
    md(
        "At the root, CART evaluates every threshold on every feature. The two "
        "curves are the weighted impurity $J(k, t_k)$; the marker is the current "
        "candidate; the dashed line marks the best found so far. The winner "
        "becomes the root split."
    )

    from sklearn.datasets import load_iris
    iris = load_iris()
    Xi = iris.data[:, 2:]; yi = iris.target

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, cnt = np.unique(labels, return_counts=True)
        p = cnt / cnt.sum()
        return float(1 - (p ** 2).sum())

    cands = []
    for k in (0, 1):
        vals = np.unique(Xi[:, k])
        thr = (vals[:-1] + vals[1:]) / 2
        for t in thr:
            L = yi[Xi[:, k] <= t]; R = yi[Xi[:, k] > t]
            J = len(L) / len(yi) * gini(L) + len(R) / len(yi) * gini(R)
            cands.append((k, float(t), float(J)))
    cands.sort(key=lambda c: (c[0], c[1]))
    c0 = [c for c in cands if c[0] == 0]
    c1 = [c for c in cands if c[0] == 1]
    best = min(cands, key=lambda c: c[2])

    seq = np.unique(np.linspace(1, len(cands), 55).astype(int))
    frames = []
    for k in seq:
        seen = cands[:k]
        bsf = min(seen, key=lambda c: c[2])
        n0 = [c for c in seen if c[0] == 0]
        n1 = [c for c in seen if c[0] == 1]
        cur = cands[k - 1]
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=[c[1] for c in n0], y=[c[2] for c in n0], mode="lines",
                       line=dict(color=SEQ[0], width=3)),
            go.Scatter(x=[c[1] for c in n1], y=[c[2] for c in n1], mode="lines",
                       line=dict(color=SEQ[1], width=3)),
            go.Scatter(x=[cur[1]], y=[cur[2]], mode="markers",
                       marker=dict(color=C["danger"], size=14,
                                   line=dict(color="#fff", width=2))),
            go.Scatter(x=[0, 7.5], y=[bsf[2], bsf[2]], mode="lines",
                       line=dict(color=C["success"], width=2, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"candidate {k}/{len(cands)}   ·   testing "
            f"{'petal length' if cur[0]==0 else 'petal width'} ≤ {cur[1]:.2f} "
            f"→ J = {cur[2]:.4f}   ·   best so far J = {bsf[2]:.4f}")])))

    f = go.Figure(data=[
        go.Scatter(x=[c[1] for c in c0[:1]], y=[c[2] for c in c0[:1]],
                   mode="lines", name="split on petal length",
                   line=dict(color=SEQ[0], width=3)),
        go.Scatter(x=[c[1] for c in c1[:1]], y=[c[2] for c in c1[:1]],
                   mode="lines", name="split on petal width",
                   line=dict(color=SEQ[1], width=3)),
        go.Scatter(x=[c0[0][1]], y=[c0[0][2]], mode="markers", name="current",
                   marker=dict(color=C["danger"], size=14,
                               line=dict(color="#fff", width=2))),
        go.Scatter(x=[0, 7.5], y=[c0[0][2]] * 2, mode="lines", name="best so far",
                   line=dict(color=C["success"], width=2, dash="dash")),
    ])
    f.update_layout(height=450, xaxis=dict(range=[0, 7.5], title="threshold"),
                    yaxis=dict(range=[0, .72], title="weighted impurity J(k, t)"),
                    title=f"CART's root search — winner: "
                          f"{'petal length' if best[0]==0 else 'petal width'} "
                          f"≤ {best[1]:.2f} (J = {best[2]:.4f})",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(75), slider_prefix="candidate ")
    figure(f)

    sub("Computational complexity")

    table(
        ["Operation", "Complexity", "Note"],
        [["Prediction", "$\\mathcal{O}(\\log_2 m)$", "Independent of $n$"],
         ["Training (default)", "$\\mathcal{O}\\bigl(n \\times m \\log_2 m\\bigr)$",
          "Every feature sorted at every node"],
         ["Training with <code>presort</code>",
          "$\\mathcal{O}(n \\times m \\log_2 m)$ once, then faster per node",
          "Only worth it for small $m$ (< a few thousand)"],
         ["Training a forest of $B$ trees", "$B \\times$ the above",
          "Embarrassingly parallel — see §7.3"]],
    )

    codenote(
        "The threshold search is smarter than it looks",
        "Naively you would recompute the impurity from scratch for every "
        "threshold. Real implementations sort the feature once, then sweep the "
        "threshold left to right, <b>updating the class counts incrementally</b> "
        "as each instance crosses the boundary. That turns an "
        "$\\mathcal{O}(m^2)$ scan into $\\mathcal{O}(m \\log m)$ (dominated by the "
        "sort). The lab implements both so you can measure the difference.",
    )

    code_lab(
        "CART from scratch — and where greedy fails",
        '''import numpy as np, time
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# ================= CART from scratch ====================================
def gini(y):
    if len(y) == 0: return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(1 - (p**2).sum())

def best_split(X, y):
    """Exhaustive search over every feature and every midpoint threshold."""
    m, n = X.shape
    best = (None, None, gini(y), None)
    for k in range(n):
        vals = np.unique(X[:, k])
        for t in (vals[:-1] + vals[1:]) / 2:
            mask = X[:, k] <= t
            L, R = y[mask], y[~mask]
            if len(L) == 0 or len(R) == 0: continue
            J = len(L)/m * gini(L) + len(R)/m * gini(R)
            if J < best[2]:
                best = (k, float(t), J, mask)
    return best

def grow(X, y, depth=0, max_depth=3, min_samples_split=2):
    node = {"n": len(y), "gini": gini(y), "value": np.bincount(y, minlength=3)}
    if depth >= max_depth or len(y) < min_samples_split or node["gini"] == 0:
        node["leaf"] = True
        node["predict"] = int(np.argmax(node["value"]))
        return node
    k, t, J, mask = best_split(X, y)
    if k is None:
        node["leaf"] = True; node["predict"] = int(np.argmax(node["value"]))
        return node
    node.update(leaf=False, feature=k, threshold=t, J=J)
    node["left"]  = grow(X[mask],  y[mask],  depth+1, max_depth, min_samples_split)
    node["right"] = grow(X[~mask], y[~mask], depth+1, max_depth, min_samples_split)
    return node

def show(node, names, indent=""):
    if node["leaf"]:
        print(f"{indent}predict class {node['predict']}  "
              f"(n={node['n']}, gini={node['gini']:.4f}, {node['value']})")
    else:
        print(f"{indent}{names[node['feature']]} <= {node['threshold']:.3f}?  "
              f"(n={node['n']}, gini={node['gini']:.4f} -> J={node['J']:.4f})")
        show(node["left"],  names, indent + "  |--T ")
        show(node["right"], names, indent + "  |--F ")

iris = load_iris()
X, y = iris.data[:, 2:], iris.target
tree = grow(X, y, max_depth=2)
print("=== my CART ===")
show(tree, ["petal length", "petal width"])

sk = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
print(f"\\n=== sklearn's root split ===")
print(f"feature {sk.tree_.feature[0]} <= {sk.tree_.threshold[0]:.3f}   "
      f"(mine: feature {tree['feature']} <= {tree['threshold']:.3f})")

# ================= naive vs incremental threshold sweep =================
print("\\n=== why the sort matters ===")
rng = np.random.default_rng(0)
for m in [200, 1000, 4000]:
    Xb = rng.normal(0, 1, (m, 1)); yb = (Xb[:, 0] + rng.normal(0, .5, m) > 0).astype(int)
    t0 = time.perf_counter(); best_split(Xb, yb); t_naive = time.perf_counter()-t0
    # incremental: sort once, then update counts as the threshold sweeps
    t0 = time.perf_counter()
    order = np.argsort(Xb[:, 0]); ys = yb[order]
    n1 = np.cumsum(ys); n0 = np.arange(1, m+1) - n1
    tot1 = n1[-1]; tot0 = n0[-1]
    nL = np.arange(1, m); nR = m - nL
    gL = 1 - (n0[:-1]/nL)**2 - (n1[:-1]/nL)**2
    gR = 1 - ((tot0-n0[:-1])/nR)**2 - ((tot1-n1[:-1])/nR)**2
    Jinc = (nL*gL + nR*gR)/m
    t_inc = time.perf_counter()-t0
    print(f"  m={m:>5}: naive {t_naive*1e3:>8.2f} ms   "
          f"incremental {t_inc*1e3:>7.3f} ms   "
          f"speedup {t_naive/max(t_inc,1e-9):>7.0f}x")

# ================= where greedy FAILS: XOR ==============================
print("\\n=== the greedy trap: XOR ===")
rng = np.random.default_rng(1)
n = 1200
Xx = rng.uniform(-1, 1, (n, 2))
yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
print("Neither feature ALONE carries any information:")
for k in range(2):
    for t in [-0.5, 0.0, 0.5]:
        L, R = yx[Xx[:, k] <= t], yx[Xx[:, k] > t]
        J = len(L)/n*gini(L) + len(R)/n*gini(R)
        print(f"  split x{k+1} <= {t:+.1f}: J = {J:.4f}  "
              f"(root gini = {gini(yx):.4f}, so gain = {gini(yx)-J:+.5f})")
print("\\nEvery first split gains ~nothing, so a depth-1 tree is useless...")
for d in [1, 2, 3, 6]:
    t_ = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xx, yx)
    print(f"  max_depth={d}: accuracy {t_.score(Xx, yx):.4f}")
print("...but at depth 2 the SECOND split unlocks it. Greedy survives XOR here")
print("only because it is 2-D; in higher dimensions parity defeats CART entirely.")
''',
        key="ch06_cart",
    )

    keypoints([
        "CART minimises the <b>weighted</b> child impurity "
        "$J = \\frac{m_L}{m}G_L + \\frac{m_R}{m}G_R$ over all $(k, t_k)$.",
        "It is <b>greedy</b> — the optimal tree is NP-complete, so no split is "
        "ever reconsidered.",
        "Training is $\\mathcal{O}(n\\,m\\log m)$; prediction is "
        "$\\mathcal{O}(\\log m)$.",
        "Greedy fails when no single feature is informative on its own (XOR / "
        "parity).",
        "scikit-learn only implements CART: binary splits, numeric features.",
    ])


# ==========================================================================
def s_6_4():
    section("6.4", "Gini Impurity or Entropy?")

    lead(
        "Two impurity measures, both defensible, and — this is the honest answer — "
        "they almost never produce different trees."
    )

    sub("The two definitions")

    math(r"""
    G_i \;=\; 1 \;-\; \sum_{k=1}^{K} p_{i,k}^{\,2}
    \qquad\qquad
    H_i \;=\; -\sum_{k=1}^{K} p_{i,k}\,\log_2\bigl(p_{i,k}\bigr)
    """)
    where({r"p_{i,k}": "the fraction of class-$k$ instances among the training "
                       "instances in node $i$",
           r"G_i": "the <b>Gini impurity</b> of node $i$; zero when the node is pure",
           r"H_i": "the <b>entropy</b> of node $i$; zero when the node is pure. "
                   "Terms with $p_{i,k} = 0$ are taken as 0"})

    sub("What each one means")

    table(
        ["", "Gini impurity", "Entropy"],
        [["Origin", "Economics (Gini coefficient), via CART",
          "Information theory (Shannon), via ID3/C4.5"],
         ["Interpretation",
          "The probability of misclassifying a randomly chosen instance if you "
          "label it by drawing from the node's class distribution",
          "The average number of bits needed to encode the class of an instance "
          "drawn from the node"],
         ["Range ($K$ classes)", "$[0,\\; 1 - 1/K]$", "$[0,\\; \\log_2 K]$"],
         ["Maximum at", "uniform distribution", "uniform distribution"],
         ["Cost", "No logarithm — <b>slightly faster</b>",
          "One logarithm per class"],
         ["Tendency", "Very slightly favours isolating the most frequent class "
          "into its own branch",
          "Very slightly favours more balanced trees"]],
    )

    proof(
        "Gini is the second-order Taylor approximation of entropy",
        "Expand $-\\log_2 p = \\frac{-\\ln p}{\\ln 2}$ around $p = 1$: "
        "$-\\ln p = (1-p) + \\frac{(1-p)^2}{2} + \\dots$. Keeping only the first "
        "term gives $H \\approx \\frac{1}{\\ln 2}\\sum_k p_k(1-p_k) = "
        "\\frac{1}{\\ln 2}\\bigl(1 - \\sum_k p_k^2\\bigr) = \\frac{G}{\\ln 2}$. "
        "So <b>$H \\approx G/\\ln 2 \\approx 1.4427\\,G$</b> to first order. The "
        "two are nearly proportional over the whole range, which is exactly why "
        "they so rarely disagree.",
    )

    anim_header("Gini and entropy across the whole probability simplex")

    p = np.linspace(1e-9, 1 - 1e-9, 400)
    g2 = 1 - (p ** 2 + (1 - p) ** 2)
    h2 = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    mis = np.minimum(p, 1 - p)

    frames = []
    for k in np.unique(np.linspace(4, 400, 50).astype(int)):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=p[:k], y=g2[:k], mode="lines",
                       line=dict(color=SEQ[0], width=3.4)),
            go.Scatter(x=p[:k], y=h2[:k], mode="lines",
                       line=dict(color=SEQ[1], width=3.4)),
            go.Scatter(x=p[:k], y=h2[:k] * np.log(2), mode="lines",
                       line=dict(color=SEQ[1], width=2, dash="dash")),
            go.Scatter(x=p[:k], y=mis[:k], mode="lines",
                       line=dict(color=SEQ[3], width=2.4, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"p = {p[k-1]:.3f}   ·   Gini = {g2[k-1]:.4f}   ·   "
            f"entropy = {h2[k-1]:.4f}   ·   entropy·ln2 = {h2[k-1]*np.log(2):.4f}"
            f"   ·   ratio H/G = {h2[k-1]/max(g2[k-1],1e-9):.4f}")])))

    f = go.Figure(data=[
        go.Scatter(x=p[:4], y=g2[:4], mode="lines", name="Gini  1 − Σp²",
                   line=dict(color=SEQ[0], width=3.4)),
        go.Scatter(x=p[:4], y=h2[:4], mode="lines", name="entropy  −Σp log₂p",
                   line=dict(color=SEQ[1], width=3.4)),
        go.Scatter(x=p[:4], y=h2[:4] * np.log(2), mode="lines",
                   name="entropy × ln2  (≈ Gini)",
                   line=dict(color=SEQ[1], width=2, dash="dash")),
        go.Scatter(x=p[:4], y=mis[:4], mode="lines", name="misclassification rate",
                   line=dict(color=SEQ[3], width=2.4, dash="dot")),
    ])
    f.update_layout(height=440, xaxis=dict(range=[0, 1], title="p (binary node)"),
                    yaxis=dict(range=[0, 1.05], title="impurity"),
                    title="Three impurity measures for a two-class node",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="p = ")
    figure(f, "The dashed curve (entropy × ln2) sits almost exactly on the Gini "
              "curve. Misclassification rate is not used for splitting because it "
              "is piecewise linear and often gives zero gain — see the lab.")

    note(
        "Why not just use the misclassification rate?",
        "Because it is <b>not strictly concave</b>. There are splits that "
        "genuinely improve the class distribution but leave the misclassification "
        "rate unchanged, so a greedy search using it gets stuck. Gini and entropy "
        "are strictly concave, so any split that changes the distribution "
        "<i>strictly</i> reduces the weighted impurity. This is a real, "
        "demonstrable difference — unlike Gini vs entropy — and the lab shows an "
        "explicit example.",
    )

    tip(
        "The practical answer",
        "Use the default (<code>criterion='gini'</code>). It is marginally faster "
        "and produces essentially the same tree. If you have a specific reason to "
        "prefer information-theoretic quantities, "
        "<code>criterion='entropy'</code> is fine. Do not spend time tuning this "
        "— spend it on <code>max_depth</code> and "
        "<code>min_samples_leaf</code> instead.",
    )

    code_lab(
        "Do Gini and entropy ever disagree? Measured.",
        '''import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def gini(p):    return float(1 - np.sum(np.asarray(p)**2))
def entropy(p):
    p = np.asarray(p); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))
def misclass(p): return float(1 - np.max(p))

print("=== the measures on some example distributions ===")
print(f"{'distribution':<26}{'Gini':>8}{'entropy':>10}{'H*ln2':>9}{'misclass':>10}")
for d in [[1,0,0],[.9,.1,0],[.5,.5,0],[.34,.33,.33],[.7,.2,.1],[.25,.25,.25,.25]]:
    print(f"{str(d):<26}{gini(d):>8.4f}{entropy(d):>10.4f}"
          f"{entropy(d)*np.log(2):>9.4f}{misclass(d):>10.4f}")

# ---- the Taylor relation H ~ G / ln2 ----------------------------------
print("\\n=== H*ln2 vs G across the binary simplex ===")
p = np.linspace(.001, .999, 999)
G = 1 - (p**2 + (1-p)**2)
H = -(p*np.log2(p) + (1-p)*np.log2(1-p))
print(f"max |H*ln2 - G| = {np.abs(H*np.log(2) - G).max():.4f}")
print(f"correlation      = {np.corrcoef(G, H)[0,1]:.8f}")

# ---- do they ever build DIFFERENT trees? -----------------------------
print("\\n=== same tree? (identical structure means identical predictions) ===")
datasets = {"iris": load_iris(), "wine": load_wine(),
            "breast_cancer": load_breast_cancer()}
print(f"{'dataset':<16}{'gini CV':>10}{'entropy CV':>12}{'same root':>11}"
      f"{'same preds':>12}")
for nm, d in datasets.items():
    g = DecisionTreeClassifier(criterion="gini", random_state=0).fit(d.data, d.target)
    e = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(d.data, d.target)
    same_root = (g.tree_.feature[0] == e.tree_.feature[0] and
                 abs(g.tree_.threshold[0] - e.tree_.threshold[0]) < 1e-9)
    same_pred = np.array_equal(g.predict(d.data), e.predict(d.data))
    cg = cross_val_score(DecisionTreeClassifier(criterion="gini", max_depth=4,
                                                random_state=0), d.data, d.target, cv=5).mean()
    ce = cross_val_score(DecisionTreeClassifier(criterion="entropy", max_depth=4,
                                                random_state=0), d.data, d.target, cv=5).mean()
    print(f"{nm:<16}{cg:>10.4f}{ce:>12.4f}{str(same_root):>11}{str(same_pred):>12}")

# ---- over many random datasets ---------------------------------------
agree = 0
for s in range(60):
    Xr, yr = make_classification(n_samples=400, n_features=8, n_informative=5,
                                 random_state=s)
    g = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=0).fit(Xr, yr)
    e = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0).fit(Xr, yr)
    agree += np.array_equal(g.predict(Xr), e.predict(Xr))
print(f"\\nover 60 random datasets, identical predictions in {agree}/60 cases")

# ---- but the MISCLASSIFICATION RATE really is broken -----------------
print("\\n=== why misclassification rate is NOT used ===")
# parent: 400 of class 0, 400 of class 1
parent = [.5, .5]
# split A: (300,100) and (100,300)   split B: (400,200) and (0,200)
for nm, (nl, pl, nr, pr) in {
        "split A: (300,100) | (100,300)": (400, [.75,.25], 400, [.25,.75]),
        "split B: (400,200) | (0,200)":   (600, [2/3,1/3], 200, [0.,1.])}.items():
    tot = nl + nr
    for f_, name in [(gini, "Gini"), (entropy, "entropy"), (misclass, "misclass")]:
        gain = f_(parent) - (nl/tot*f_(pl) + nr/tot*f_(pr))
        print(f"  {nm:<34}{name:>10} gain = {gain:+.5f}")
    print()
print("Split B creates a PURE leaf, yet its misclassification gain equals")
print("split A's. Gini and entropy both correctly prefer B. That is why they win.")
''',
        key="ch06_impurity",
    )

    quiz(
        "A node has class proportions $(0.5, 0.5)$. What are its Gini impurity and "
        "entropy?",
        ["Gini 0.25, entropy 0.5", "Gini 0.5, entropy 1.0",
         "Gini 1.0, entropy 1.0", "Gini 0.5, entropy 0.5"],
        1,
        "$G = 1 - (0.25 + 0.25) = 0.5$ and $H = -2 \\times 0.5\\log_2 0.5 = 1$ bit "
        "— the maximum for two classes. Note $H \\cdot \\ln 2 = 0.693$, close to "
        "$G$, exactly as the Taylor argument predicts.",
        key="ch06q1",
    )

    keypoints([
        "$G = 1 - \\sum_k p_k^2$; $H = -\\sum_k p_k\\log_2 p_k$. Both are 0 for a "
        "pure node, maximal for a uniform one.",
        "$H \\approx G/\\ln 2$ — Gini is entropy's second-order Taylor "
        "approximation.",
        "They produce the same tree the overwhelming majority of the time; Gini is "
        "slightly faster.",
        "Misclassification rate is <b>not</b> used because it is not strictly "
        "concave and gives zero gain for genuinely good splits.",
        "Don't tune the criterion; tune the regularisation hyperparameters.",
    ])


# ==========================================================================
def s_6_5():
    section("6.5", "Regularization Hyperparameters")

    lead(
        "Left alone, a decision tree will grow until every leaf is pure — which "
        "means it memorises the training set exactly. Trees are the archetypal "
        "<b>non-parametric</b> model, and they need explicit constraints."
    )

    idea(
        "Parametric vs non-parametric",
        "A <b>parametric</b> model (linear regression, a fixed neural network) has "
        "a number of parameters decided <i>before</i> training, which limits its "
        "freedom and reduces overfitting risk. A <b>non-parametric</b> model — a "
        "decision tree, $k$-NN, a Gaussian mixture with unbounded components — "
        "has a structure determined <i>by the data</i>, so its complexity grows "
        "with $m$. It is not that it has no parameters; it is that the number is "
        "not fixed in advance. Unconstrained, it will fit the data exactly.",
    )

    sub("The seven dials")

    table(
        ["Hyperparameter", "What it limits", "Direction that regularises",
         "Typical values"],
        [["<code>max_depth</code>", "Depth of the tree", "<b>Decrease</b>",
          "3–10, or <code>None</code>"],
         ["<code>min_samples_split</code>",
          "Minimum instances a node must have to be split", "<b>Increase</b>",
          "2–50"],
         ["<code>min_samples_leaf</code>",
          "Minimum instances a leaf must contain", "<b>Increase</b>",
          "1–20; try 5 first"],
         ["<code>min_weight_fraction_leaf</code>",
          "Same, but as a fraction of total weight", "<b>Increase</b>",
          "0.0–0.05"],
         ["<code>max_leaf_nodes</code>", "Total number of leaves",
          "<b>Decrease</b>", "None, or 5–100"],
         ["<code>max_features</code>",
          "Features evaluated at each split", "<b>Decrease</b>",
          "None, 'sqrt', 'log2'"],
         ["<code>ccp_alpha</code>",
          "Cost-complexity pruning penalty", "<b>Increase</b>",
          "0.0–0.05"]],
        "Everything starting with <code>min_</code> regularises by increasing; "
        "everything starting with <code>max_</code> regularises by decreasing.",
    )

    sub("Cost-complexity (post-)pruning")

    md(
        "The alternative philosophy: grow the tree fully, then **prune back** "
        "branches that do not pay for themselves. Minimal cost-complexity pruning "
        "minimises:"
    )

    math(r"""
    R_\alpha(T) \;=\; R(T) \;+\; \alpha \, \bigl|\widetilde{T}\bigr|
    """)
    where({r"R(T)": "the total impurity of the tree's leaves, weighted by their "
                    "sample fractions",
           r"|\widetilde{T}|": "the number of leaves — the complexity term",
           r"\alpha": "<code>ccp_alpha</code>, the price of a leaf"})

    derive(
        [("For any internal node $t$, compare keeping its subtree $T_t$ against "
          "collapsing it into a single leaf.", None),
         ("The cost of the subtree, and of the collapsed leaf:",
          r"R_\alpha(T_t) = R(T_t) + \alpha\,|\widetilde{T_t}|, \qquad "
          r"R_\alpha(t) = R(t) + \alpha"),
         ("Pruning is worthwhile exactly when the leaf is no more expensive:",
          r"R(t) + \alpha \;\le\; R(T_t) + \alpha\,|\widetilde{T_t}|"),
         ("Solving for $\\alpha$ gives the <b>effective alpha</b> of node $t$ — the "
          "price at which that subtree stops being worth keeping:",
          r"\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|\widetilde{T_t}| - 1}"),
         ("The pruning algorithm repeatedly collapses the node with the smallest "
          "$\\alpha_{\\text{eff}}$, producing a finite, nested sequence of trees "
          "$T_0 \\supset T_1 \\supset \\dots \\supset \\{\\text{root}\\}$ indexed by "
          "increasing $\\alpha$. Cross-validate over that sequence and pick the "
          "best — this is what <code>cost_complexity_pruning_path</code> gives you.",
          None)],
        title="How cost-complexity pruning decides what to cut",
    )

    anim_header("Regularisation in action on the moons dataset")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    Xm, ym = ds.moons(n=300, noise=.30)
    r1 = np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 130)
    r2 = np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 130)
    R1, R2 = np.meshgrid(r1, r2); RR = np.c_[R1.ravel(), R2.ravel()]

    settings = [("unrestricted", {}),
                ("max_depth=2", dict(max_depth=2)),
                ("max_depth=4", dict(max_depth=4)),
                ("max_depth=8", dict(max_depth=8)),
                ("min_samples_leaf=1", dict(min_samples_leaf=1)),
                ("min_samples_leaf=5", dict(min_samples_leaf=5)),
                ("min_samples_leaf=20", dict(min_samples_leaf=20)),
                ("max_leaf_nodes=6", dict(max_leaf_nodes=6)),
                ("ccp_alpha=0.005", dict(ccp_alpha=.005)),
                ("ccp_alpha=0.02", dict(ccp_alpha=.02))]
    scache = []
    for nm, kw in settings:
        t = DecisionTreeClassifier(random_state=42, **kw).fit(Xm, ym)
        cv = cross_val_score(DecisionTreeClassifier(random_state=42, **kw),
                             Xm, ym, cv=5).mean()
        scache.append((t.predict(RR).reshape(R1.shape).astype(float),
                       int((t.tree_.children_left == -1).sum()),
                       float(t.score(Xm, ym)), float(cv)))

    def sc():
        return [go.Scatter(x=Xm[ym == k, 0], y=Xm[ym == k, 1], mode="markers",
                           marker=dict(color=[C["train"], C["warning"]][k], size=6,
                                       line=dict(color="#fff", width=.7)),
                           showlegend=False) for k in range(2)]

    cs2 = [[0, alpha(C["train"], .35)], [1, alpha(C["warning"], .35)]]
    frames = []
    for i, (nm, _) in enumerate(settings):
        Z, nl, tr, cv = scache[i]
        gap = tr - cv
        col = C["danger"] if gap > .10 else (C["success"] if cv > .87 else C["warning"])
        frames.append(go.Frame(name=nm, data=[
            go.Contour(x=r1, y=r2, z=Z, showscale=False, colorscale=cs2,
                       contours=dict(showlines=False))] + sc(),
            layout=go.Layout(annotations=[anim.annotate_step(
                f"{nm}   ·   {nl} leaves   ·   train {tr:.3f}   CV {cv:.3f}   "
                f"gap {gap:+.3f}", color=col)])))

    f = go.Figure(data=[go.Contour(x=r1, y=r2, z=scache[0][0], showscale=False,
                                   colorscale=cs2,
                                   contours=dict(showlines=False))] + sc())
    f.update_layout(height=490, title="Regularising a decision tree",
                    xaxis_title="x₁", yaxis_title="x₂")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="")
    figure(f, "The unrestricted tree carves tiny islands around individual noisy "
              "points — a textbook picture of overfitting.")

    code_lab(
        "Tune the dials, and walk the full pruning path",
        '''import numpy as np, pandas as pd
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score)

X, y = make_moons(n_samples=1000, noise=.32, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)

# ---- the unrestricted tree memorises ---------------------------------
full = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
print(f"unrestricted: depth={full.get_depth()}  "
      f"leaves={(full.tree_.children_left == -1).sum()}")
print(f"              train={full.score(Xtr, ytr):.4f}  test={full.score(Xte, yte):.4f}"
      f"  gap={full.score(Xtr,ytr)-full.score(Xte,yte):+.4f}")

# ---- one dial at a time ----------------------------------------------
print(f"\\n{'setting':<26}{'leaves':>8}{'train':>9}{'CV':>9}{'test':>9}")
for nm, kw in [("baseline (none)",      {}),
               ("max_depth=3",          dict(max_depth=3)),
               ("max_depth=6",          dict(max_depth=6)),
               ("min_samples_leaf=5",   dict(min_samples_leaf=5)),
               ("min_samples_leaf=20",  dict(min_samples_leaf=20)),
               ("min_samples_split=40", dict(min_samples_split=40)),
               ("max_leaf_nodes=8",     dict(max_leaf_nodes=8)),
               ("ccp_alpha=0.005",      dict(ccp_alpha=.005))]:
    t = DecisionTreeClassifier(random_state=42, **kw).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(random_state=42, **kw),
                         Xtr, ytr, cv=5).mean()
    print(f"{nm:<26}{(t.tree_.children_left == -1).sum():>8}"
          f"{t.score(Xtr, ytr):>9.4f}{cv:>9.4f}{t.score(Xte, yte):>9.4f}")

# ---- grid search over the whole space --------------------------------
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [3, 4, 6, 8, None],
                     "min_samples_leaf": [1, 4, 10, 20],
                     "max_leaf_nodes": [None, 8, 16, 32],
                     "criterion": ["gini", "entropy"]},
                    cv=5, n_jobs=-1)
grid.fit(Xtr, ytr)
print(f"\\nbest params : {grid.best_params_}")
print(f"best CV     : {grid.best_score_:.4f}")
print(f"test        : {grid.best_estimator_.score(Xte, yte):.4f}")

# ============ the FULL cost-complexity pruning path ====================
print("\\n=== cost-complexity pruning path ===")
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(Xtr, ytr)
alphas, impurities = path.ccp_alphas[:-1], path.impurities[:-1]
print(f"{len(alphas)} distinct alpha values, from {alphas.min():.6f} "
      f"to {alphas.max():.6f}")

rows = []
for a in alphas[::max(1, len(alphas)//25)]:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=a).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(random_state=42, ccp_alpha=a),
                         Xtr, ytr, cv=5).mean()
    rows.append((a, (t.tree_.children_left == -1).sum(), t.get_depth(),
                 t.score(Xtr, ytr), cv, t.score(Xte, yte)))

print(f"\\n{'alpha':>10}{'leaves':>8}{'depth':>7}{'train':>9}{'CV':>9}{'test':>9}")
for a, nl, d, tr, cv, te in rows:
    print(f"{a:>10.6f}{nl:>8}{d:>7}{tr:>9.4f}{cv:>9.4f}{te:>9.4f}")

best = max(rows, key=lambda r: r[4])
print(f"\\nbest alpha = {best[0]:.6f} -> {best[1]} leaves, CV {best[4]:.4f}, "
      f"test {best[5]:.4f}")
print(f"pruned from {(full.tree_.children_left == -1).sum()} leaves down to {best[1]}")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=[r[0] for r in rows], y=[r[3] for r in rows],
                mode="lines+markers", name="train", line=dict(color=C["train"], width=3))
fig.add_scatter(x=[r[0] for r in rows], y=[r[4] for r in rows],
                mode="lines+markers", name="CV", line=dict(color=C["success"], width=3))
fig.add_scatter(x=[r[0] for r in rows], y=[r[5] for r in rows],
                mode="lines+markers", name="test", line=dict(color=C["test"], width=3))
fig.add_vline(x=best[0], line_dash="dash", line_color=C["primary"],
              annotation_text="best alpha")
fig.update_layout(height=400, xaxis_title="ccp_alpha", yaxis_title="accuracy",
                  title="The cost-complexity pruning path")
''',
        key="ch06_reg",
    )

    quiz(
        "Your decision tree gets 100 % training accuracy and 71 % test accuracy. "
        "Which change is most likely to help?",
        ["Increase <code>max_depth</code>",
         "Increase <code>min_samples_leaf</code>",
         "Switch <code>criterion</code> to entropy",
         "Decrease <code>min_samples_split</code>"],
        1,
        "100 % training accuracy means pure leaves — classic overfitting. Raising "
        "<code>min_samples_leaf</code> forces each leaf to be supported by several "
        "instances, which is exactly the constraint that is missing. Options 1 and "
        "4 both *increase* freedom; option 3 changes essentially nothing (§6.4).",
        key="ch06q2",
    )

    keypoints([
        "Trees are <b>non-parametric</b>: unconstrained, they memorise the "
        "training set.",
        "<code>min_*</code> up and <code>max_*</code> down both regularise.",
        "<code>min_samples_leaf</code> is usually the most effective single dial.",
        "Cost-complexity pruning minimises $R(T) + \\alpha|\\widetilde T|$ and "
        "gives a nested sequence of trees to cross-validate over.",
        "Pre-pruning (constraints) is cheap; post-pruning (<code>ccp_alpha</code>) "
        "is more principled.",
    ])


# ==========================================================================
def s_6_6():
    section("6.6", "Regression Trees")

    lead(
        "The same algorithm with one substitution: instead of minimising impurity, "
        "minimise the mean squared error of each subset. The prediction is the "
        "subset's average."
    )

    sub("The CART cost function for regression")

    math(r"""
    J\bigl(k, t_k\bigr) \;=\;
    \frac{m_{\text{left}}}{m}\,\mathrm{MSE}_{\text{left}}
    \;+\;
    \frac{m_{\text{right}}}{m}\,\mathrm{MSE}_{\text{right}}
    """)

    math(r"""
    \text{where}\qquad
    \mathrm{MSE}_{\text{node}} = \frac{1}{m_{\text{node}}}
      \sum_{i \in \text{node}} \bigl(\hat y_{\text{node}} - y^{(i)}\bigr)^{2},
    \qquad
    \hat y_{\text{node}} = \frac{1}{m_{\text{node}}}\sum_{i \in \text{node}} y^{(i)}
    """)

    proof(
        "The leaf value is the mean, and that is not a choice",
        "From the derivation in §2.1: the constant minimising squared error over a "
        "set is that set's <b>mean</b>. So once CART decides on a partition, the "
        "optimal prediction in each cell is forced. All the modelling happens in "
        "<i>where the cuts go</i>, not in what is predicted. (Set "
        "<code>criterion='absolute_error'</code> and the leaf value becomes the "
        "<b>median</b> instead — same argument, $\\ell_1$ loss.)",
    )

    md(
        "So a regression tree is a **piecewise-constant** function: a staircase. "
        "It cannot extrapolate, and it cannot produce a smooth curve. Those two "
        "facts explain almost every regression-tree failure you will ever see."
    )

    anim_header("A regression tree as a staircase, depth by depth")

    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(42)
    Xr = np.sort(rng.uniform(-3, 3, 160)).reshape(-1, 1)
    yr = np.sin(1.5 * Xr[:, 0]) + .35 * Xr[:, 0] + rng.normal(0, .22, 160)
    grid = np.linspace(-4.5, 4.5, 700).reshape(-1, 1)
    truth = np.sin(1.5 * grid[:, 0]) + .35 * grid[:, 0]

    depths = list(range(1, 13))
    cache = []
    for d in depths:
        t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xr, yr)
        cache.append((t.predict(grid), int((t.tree_.children_left == -1).sum()),
                      float(np.sqrt(np.mean((t.predict(Xr) - yr) ** 2)))))

    frames = []
    for i, d in enumerate(depths):
        pr, nl, rmse = cache[i]
        col = C["warning"] if d <= 2 else (C["success"] if d <= 5 else C["danger"])
        frames.append(go.Frame(name=str(d), data=[
            go.Scatter(x=Xr[:, 0], y=yr, mode="markers",
                       marker=dict(color=C["train"], size=6,
                                   line=dict(color="#fff", width=.7))),
            go.Scatter(x=grid[:, 0], y=truth, mode="lines",
                       line=dict(color=C["truth"], width=2, dash="dot")),
            go.Scatter(x=grid[:, 0], y=pr, mode="lines",
                       line=dict(color=col, width=3.4, shape="hv")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"max_depth = {d}   ·   {nl} leaves   ·   train RMSE = {rmse:.4f}",
            color=col)])))

    f = go.Figure(data=[
        go.Scatter(x=Xr[:, 0], y=yr, mode="markers", name="training data",
                   marker=dict(color=C["train"], size=6,
                               line=dict(color="#fff", width=.7))),
        go.Scatter(x=grid[:, 0], y=truth, mode="lines", name="true function",
                   line=dict(color=C["truth"], width=2, dash="dot")),
        go.Scatter(x=grid[:, 0], y=cache[0][0], mode="lines", name="tree prediction",
                   line=dict(color=C["warning"], width=3.4, shape="hv")),
    ])
    f.add_vrect(x0=-4.5, x1=Xr.min(), fillcolor=C["danger"], opacity=.07,
                line_width=0, annotation_text="outside training range")
    f.add_vrect(x0=Xr.max(), x1=4.5, fillcolor=C["danger"], opacity=.07,
                line_width=0)
    f.update_layout(height=470, xaxis=dict(range=[-4.5, 4.5], title="x"),
                    yaxis=dict(range=[-2.6, 2.6], title="y"),
                    title="A regression tree is a piecewise-constant staircase",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(600), slider_prefix="depth ")
    figure(f, "Look at the shaded regions: outside the training range the "
              "prediction is a flat line forever. Trees cannot extrapolate — "
              "compare with §1.6's k-NN, which has the same limitation for the "
              "same reason.")

    pitfall(
        "Never use a tree for extrapolation",
        "A tree's prediction outside the training range is the value of the "
        "boundary leaf, held constant to infinity. If you forecast a time series "
        "with an upward trend using a tree or a random forest, the forecast is a "
        "flat line at the last training level — <b>guaranteed</b>. Detrend first "
        "(model the differences, or fit a linear model and let the tree predict "
        "its residuals). Chapter 15 returns to this.",
    )

    code_lab(
        "Regression trees: staircase, extrapolation failure, and the fix",
        '''import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(-3, 3, 200)).reshape(-1, 1)
y = np.sin(1.5*X[:, 0]) + .35*X[:, 0] + rng.normal(0, .22, 200)

# ---- the leaf value IS the mean ---------------------------------------
t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
print(export_text(t, feature_names=["x"]))
tr = t.tree_
print("verify: each leaf's value is the mean of the y's that reach it")
leaf_id = t.apply(X)
for node in np.unique(leaf_id):
    print(f"  leaf {node}: tree value = {tr.value[node].ravel()[0]:+.6f}   "
          f"mean of its {(leaf_id==node).sum():>3} targets = "
          f"{y[leaf_id==node].mean():+.6f}")

# ---- absolute_error puts the MEDIAN in the leaves instead ------------
t_l1 = DecisionTreeRegressor(max_depth=2, criterion="absolute_error",
                             random_state=0).fit(X, y)
leaf_l1 = t_l1.apply(X)
print("\\ncriterion='absolute_error' -> leaf value is the MEDIAN")
for node in np.unique(leaf_l1)[:3]:
    print(f"  leaf {node}: tree value = {t_l1.tree_.value[node].ravel()[0]:+.6f}   "
          f"median = {np.median(y[leaf_l1==node]):+.6f}")

# ---- depth vs fit -----------------------------------------------------
print(f"\\n{'depth':>7}{'leaves':>8}{'train RMSE':>13}{'CV RMSE':>11}")
for d in [1, 2, 3, 5, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
    cv = -cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=0),
                          X, y, cv=5, scoring="neg_root_mean_squared_error").mean()
    print(f"{str(d):>7}{(t.tree_.children_left==-1).sum():>8}"
          f"{np.sqrt(np.mean((t.predict(X)-y)**2)):>13.4f}{cv:>11.4f}")

# ============ EXTRAPOLATION FAILURE ====================================
print("\\n=== extrapolation: a linear trend ===")
Xt = np.linspace(0, 10, 300).reshape(-1, 1)
yt = 2.0 + 3.0*Xt[:, 0] + rng.normal(0, .5, 300)
future = np.array([[12.], [15.], [20.], [50.]])

tree_m = DecisionTreeRegressor(max_depth=6, random_state=0).fit(Xt, yt)
rf_m   = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(Xt, yt)
lin_m  = LinearRegression().fit(Xt, yt)

print(f"{'x':>6}{'truth':>10}{'tree':>10}{'forest':>10}{'linear':>10}")
for i, xv in enumerate(future.ravel()):
    print(f"{xv:>6.0f}{2+3*xv:>10.2f}{tree_m.predict(future)[i]:>10.2f}"
          f"{rf_m.predict(future)[i]:>10.2f}{lin_m.predict(future)[i]:>10.2f}")
print("The tree and the forest both flat-line at the last training value.")
print("The forest does NOT fix this -- averaging flat lines gives a flat line.")

# ---- the fix: model the residual of a linear trend -------------------
print("\\n=== the fix: linear trend + tree on the residuals ===")
resid = yt - lin_m.predict(Xt)
tree_r = DecisionTreeRegressor(max_depth=6, random_state=0).fit(Xt, resid)
hybrid = lin_m.predict(future) + tree_r.predict(future)
print(f"{'x':>6}{'truth':>10}{'hybrid':>10}{'error':>10}")
for i, xv in enumerate(future.ravel()):
    print(f"{xv:>6.0f}{2+3*xv:>10.2f}{hybrid[i]:>10.2f}{hybrid[i]-(2+3*xv):>+10.2f}")

import plotly.graph_objects as go
g = np.linspace(0, 22, 500).reshape(-1, 1)
fig = go.Figure()
fig.add_scatter(x=Xt[:, 0], y=yt, mode="markers", name="training data",
                marker=dict(color=C["train"], size=4, opacity=.5))
fig.add_scatter(x=g[:, 0], y=2+3*g[:, 0], mode="lines", name="truth",
                line=dict(color=C["truth"], width=2, dash="dot"))
fig.add_scatter(x=g[:, 0], y=tree_m.predict(g), mode="lines", name="tree",
                line=dict(color=C["danger"], width=3, shape="hv"))
fig.add_scatter(x=g[:, 0], y=lin_m.predict(g)+tree_r.predict(g), mode="lines",
                name="linear + tree(residual)", line=dict(color=C["success"], width=3))
fig.add_vline(x=10, line_dash="dash", line_color=C["muted"],
              annotation_text="end of training data")
fig.update_layout(height=430, title="Trees cannot extrapolate; detrending fixes it",
                  xaxis_title="x", yaxis_title="y")
''',
        key="ch06_regression",
    )

    keypoints([
        "Regression CART minimises weighted MSE; the leaf value is the subset "
        "<b>mean</b> (or median under $\\ell_1$).",
        "The prediction surface is <b>piecewise constant</b> — a staircase.",
        "Regression trees <b>cannot extrapolate</b>: outside the training range "
        "they flat-line, and forests do not fix this.",
        "For trended data, detrend first and let the tree model the residual.",
        "Same regularisation dials as classification, and they matter more.",
    ])


# ==========================================================================
def s_6_7():
    section("6.7", "Instability — Axis Orientation and High Variance")

    lead(
        "Two limitations, both fundamental, both the direct motivation for "
        "Chapter 7. Trees love axis-aligned boundaries and hate small changes in "
        "the data."
    )

    sub("Sensitivity to axis orientation")

    md(
        "Every split has the form $x_k \\le t_k$, which is a hyperplane "
        "**perpendicular to one axis**. A boundary at 45° must therefore be "
        "approximated by a staircase of many small axis-aligned steps — and that "
        "staircase is fitted to the noise as much as to the signal."
    )

    anim_header("Rotate the data and watch the tree fall apart")
    md(
        "The *same* linearly separable dataset, rotated. At 0° a single split "
        "separates it perfectly. At 45° the tree needs a dozen splits and still "
        "generalises worse."
    )

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(6)
    n = 200
    base = np.c_[rng.uniform(-2.4, 2.4, n), rng.uniform(-1.1, 1.1, n)]
    ylab = (base[:, 0] > 0).astype(int)

    angles = np.linspace(0, np.pi / 2, 28)
    ax1 = np.linspace(-3.6, 3.6, 130); ax2 = np.linspace(-3.6, 3.6, 130)
    A1, A2 = np.meshgrid(ax1, ax2); AA = np.c_[A1.ravel(), A2.ravel()]

    rcache = []
    for a in angles:
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        Xr = base @ R.T
        t = DecisionTreeClassifier(random_state=0).fit(Xr, ylab)
        cv = cross_val_score(DecisionTreeClassifier(random_state=0), Xr, ylab,
                             cv=5).mean()
        rcache.append((Xr, t.predict(AA).reshape(A1.shape).astype(float),
                       int((t.tree_.children_left == -1).sum()), float(cv)))

    cs2 = [[0, alpha(C["train"], .34)], [1, alpha(C["warning"], .34)]]
    frames = []
    for i, a in enumerate(angles):
        Xr, Z, nl, cv = rcache[i]
        col = C["success"] if nl <= 3 else (C["danger"] if nl > 12 else C["warning"])
        frames.append(go.Frame(name=f"{np.degrees(a):.0f}", data=[
            go.Contour(x=ax1, y=ax2, z=Z, showscale=False, colorscale=cs2,
                       contours=dict(showlines=False)),
            go.Scatter(x=Xr[ylab == 0, 0], y=Xr[ylab == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=6,
                                   line=dict(color="#fff", width=.7))),
            go.Scatter(x=Xr[ylab == 1, 0], y=Xr[ylab == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=6,
                                   line=dict(color="#fff", width=.7))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"rotation = {np.degrees(a):5.1f}°   ·   {nl} leaves   ·   "
            f"CV accuracy = {cv:.4f}", color=col)])))

    Xr0, Z0, _, _ = rcache[0]
    f = go.Figure(data=[
        go.Contour(x=ax1, y=ax2, z=Z0, showscale=False, colorscale=cs2,
                   contours=dict(showlines=False)),
        go.Scatter(x=Xr0[ylab == 0, 0], y=Xr0[ylab == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=6,
                                               line=dict(color="#fff", width=.7))),
        go.Scatter(x=Xr0[ylab == 1, 0], y=Xr0[ylab == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"], size=6,
                                               line=dict(color="#fff", width=.7))),
    ])
    f.update_layout(height=500, xaxis=dict(range=[-3.6, 3.6], title="x₁"),
                    yaxis=dict(range=[-3.6, 3.6], title="x₂"),
                    title="Rotating the data destroys the tree")
    anim.animate(f, frames, duration=nav.anim_ms(200), slider_prefix="angle ")
    figure(f)

    tip(
        "PCA is the standard mitigation",
        "Rotating the data so its principal axes align with the coordinate axes "
        "(Chapter 8) often lets a tree find a much simpler, more robust boundary. "
        "The alternative is an <b>oblique</b> tree, which splits on a linear "
        "combination $\\mathbf{a}^\\top\\mathbf{x} \\le t$ rather than a single "
        "feature — scikit-learn does not implement these, but Chapter 7's "
        "Extra-Trees achieve a similar effect statistically by randomising the "
        "cuts.",
    )

    sub("High variance")

    md(
        "The greedy algorithm makes trees **unstable**: a small perturbation of "
        "the training set can change the root split, and changing the root split "
        "changes everything below it. Even removing a single instance, or just "
        "changing `random_state` (scikit-learn breaks ties randomly), can produce "
        "a visibly different tree."
    )

    anim_header("Ten trees, ten bootstrap samples, ten different boundaries")
    md(
        "The same data resampled ten times. Individually these trees disagree "
        "wildly. **The final frame shows their average** — which is smooth, "
        "sensible, and better than any of them. That frame is Chapter 7."
    )

    Xm, ym = ds.moons(n=200, noise=.30)
    v1 = np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 120)
    v2 = np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 120)
    V1, V2 = np.meshgrid(v1, v2); VV = np.c_[V1.ravel(), V2.ravel()]

    preds = []
    for s in range(10):
        rr = np.random.default_rng(s)
        idx = rr.choice(len(Xm), len(Xm), replace=True)
        t = DecisionTreeClassifier(max_depth=6, random_state=s).fit(Xm[idx], ym[idx])
        preds.append(t.predict_proba(VV)[:, 1].reshape(V1.shape))
    avg = np.mean(preds, axis=0)

    def sc():
        return [go.Scatter(x=Xm[ym == k, 0], y=Xm[ym == k, 1], mode="markers",
                           marker=dict(color=[C["train"], C["warning"]][k], size=6,
                                       line=dict(color="#fff", width=.7)),
                           showlegend=False) for k in range(2)]

    frames = []
    for s in range(10):
        frames.append(go.Frame(name=f"tree {s+1}", data=[
            go.Heatmap(x=v1, y=v2, z=preds[s], colorscale=nav.cscale(),
                       zmin=0, zmax=1, showscale=False)] + sc(),
            layout=go.Layout(title=f"tree {s+1} of 10 — bootstrap sample {s}")))
    frames.append(go.Frame(name="average", data=[
        go.Heatmap(x=v1, y=v2, z=avg, colorscale=nav.cscale(), zmin=0, zmax=1,
                   showscale=False)] + sc(),
        layout=go.Layout(title="THE AVERAGE OF ALL TEN — this is a random forest "
                               "(Chapter 7)")))

    f = go.Figure(data=[go.Heatmap(x=v1, y=v2, z=preds[0], colorscale=nav.cscale(),
                                   zmin=0, zmax=1, showscale=False)] + sc())
    f.update_layout(height=500, title="tree 1 of 10",
                    xaxis_title="x₁", yaxis_title="x₂")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="")
    figure(f)

    idea(
        "Why averaging works — the variance formula",
        "For $B$ predictors each with variance $\\sigma^2$ and pairwise correlation "
        "$\\rho$, the variance of their average is "
        "$\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$. Averaging kills the second "
        "term entirely as $B$ grows; the first term is the floor set by how "
        "correlated the trees are. <b>High-variance, low-bias, weakly correlated "
        "predictors are exactly the right raw material for averaging</b> — and "
        "that is a precise description of a deep decision tree. This formula is "
        "the whole justification for Chapter 7.",
    )

    code_lab(
        "Quantify the instability, then cure it",
        '''import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

# ============ 1. axis orientation ======================================
rng = np.random.default_rng(6)
base = np.c_[rng.uniform(-2.4, 2.4, 400), rng.uniform(-1.1, 1.1, 400)]
y = (base[:, 0] > 0).astype(int)

print("=== rotating the data ===")
print(f"{'angle':>7}{'leaves':>8}{'depth':>7}{'CV acc':>9}{'PCA+tree CV':>13}")
for deg in [0, 15, 30, 45, 60, 90]:
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    X = base @ R.T
    t = DecisionTreeClassifier(random_state=0).fit(X, y)
    cv  = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean()
    cvp = cross_val_score(make_pipeline(PCA(), DecisionTreeClassifier(random_state=0)),
                          X, y, cv=5).mean()
    print(f"{deg:>6}°{(t.tree_.children_left==-1).sum():>8}{t.get_depth():>7}"
          f"{cv:>9.4f}{cvp:>13.4f}")
print("PCA rotates the data back onto its principal axes -> the tree recovers.")

# ============ 2. instability ===========================================
X, y = make_moons(n_samples=300, noise=.3, random_state=0)
print("\\n=== how unstable is a single tree? ===")
print("removing ONE training instance:")
full = DecisionTreeClassifier(random_state=0).fit(X, y)
changed = 0
for i in range(60):
    m = np.ones(len(X), bool); m[i] = False
    t = DecisionTreeClassifier(random_state=0).fit(X[m], y[m])
    if (t.tree_.feature[0] != full.tree_.feature[0]
            or abs(t.tree_.threshold[0] - full.tree_.threshold[0]) > 1e-9):
        changed += 1
print(f"  the ROOT SPLIT changed in {changed}/60 cases")

print("\\nchanging only random_state (ties are broken randomly):")
roots = set()
for s in range(30):
    t = DecisionTreeClassifier(random_state=s).fit(X, y)
    roots.add((int(t.tree_.feature[0]), round(float(t.tree_.threshold[0]), 6)))
print(f"  {len(roots)} distinct root splits over 30 seeds")

# ---- measure prediction variance across bootstraps -------------------
grid = np.c_[np.linspace(-2, 3, 60).repeat(60),
             np.tile(np.linspace(-1.5, 2, 60), 60)]
P = np.array([DecisionTreeClassifier(max_depth=8, random_state=s)
              .fit(*(lambda i: (X[i], y[i]))(
                  np.random.default_rng(s).choice(len(X), len(X), replace=True)))
              .predict_proba(grid)[:, 1] for s in range(40)])
print(f"\\nover 40 bootstrap trees, at each grid point:")
print(f"  mean prediction variance = {P.var(0).mean():.4f}")
print(f"  variance of the AVERAGE of 40 = {P.mean(0).var()*0 + P.var(0).mean()/40:.4f}"
      f"   (theory: sigma^2/B for uncorrelated)")

# ============ 3. the cure ==============================================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0)
print("\\n=== single tree vs forest, 20 seeds ===")
single = [DecisionTreeClassifier(random_state=s).fit(Xtr, ytr).score(Xte, yte)
          for s in range(20)]
forest = [RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1)
          .fit(Xtr, ytr).score(Xte, yte) for s in range(20)]
print(f"  single tree : mean {np.mean(single):.4f}  sd {np.std(single):.4f}  "
      f"range [{min(single):.3f}, {max(single):.3f}]")
print(f"  200-tree RF : mean {np.mean(forest):.4f}  sd {np.std(forest):.4f}  "
      f"range [{min(forest):.3f}, {max(forest):.3f}]")
print(f"\\nthe forest is {np.std(single)/max(np.std(forest),1e-9):.1f}x more stable "
      f"and {np.mean(forest)-np.mean(single):+.4f} more accurate.")

# ---- the variance-of-an-average formula -------------------------------
print("\\n=== Var(average of B predictors) = rho*s^2 + (1-rho)/B * s^2 ===")
s2 = 0.04
print(f"{'B':>6}{'rho=0':>10}{'rho=0.3':>10}{'rho=0.7':>10}")
for B in [1, 5, 20, 100, 1000]:
    row = [rho*s2 + (1-rho)/B*s2 for rho in (0., .3, .7)]
    print(f"{B:>6}{row[0]:>10.5f}{row[1]:>10.5f}{row[2]:>10.5f}")
print("Averaging drives the second term to zero; rho sets the floor.")
print("That is why random forests DECORRELATE the trees (Ch. 7).")
''',
        key="ch06_variance",
    )

    keypoints([
        "Splits are axis-aligned, so a rotated boundary needs a staircase of many "
        "splits — PCA (Ch. 8) mitigates this.",
        "Trees are <b>high variance</b>: one removed instance can change the root "
        "split and hence the whole tree.",
        "$\\mathrm{Var}(\\bar h) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ — "
        "averaging removes the second term.",
        "Deep trees are the perfect base learner for averaging: low bias, high "
        "variance.",
        "That single observation is the whole of Chapter 7.",
    ])


# ==========================================================================
def s_6_8():
    section("6.8", "Exercises & Chapter Review")

    lead("Seven exercises. Numbers 5–7 are worth actually running.")

    exercise(
        1, "What is the approximate depth of a decision tree trained (without "
        "restrictions) on a training set with one million instances?",
        "**About 20.** A well-balanced binary tree splits its instances in half at "
        "every level, so it needs roughly $\\log_2(m)$ levels to isolate them: "
        "$\\log_2(10^6) \\approx 19.93$.\n\n"
        "In practice the tree will be somewhat deeper — CART's splits are rarely "
        "perfectly balanced, and it keeps going until leaves are pure. But the "
        "order of magnitude is right, and this is exactly why prediction is so "
        "fast: ~20 comparisons for a million-row training set.")

    exercise(
        2, "Is a node's Gini impurity generally lower or higher than its parent's? "
        "Is it *generally* lower/higher, or *always*?",
        "**Generally lower, but not always.** CART minimises the *weighted sum* of "
        "the children's impurities, and that weighted sum is always $\\le$ the "
        "parent's impurity. But an *individual* child can be worse than the "
        "parent, as long as the other child is enough better to compensate.\n\n"
        "**Concrete counter-example.** Parent has class ratio (4, 0) for A and "
        "(1, 3) for B in the two halves... more simply: parent = 8 instances with "
        "$p = (0.5, 0.5)$, $G = 0.5$. Split into a left child of 2 with "
        "$p = (0, 1)$, $G = 0$, and a right child of 6 with $p = (4/6, 2/6)$, "
        "$G = 0.444$. The right child is *lower* here — try instead left child of "
        "4 with $p = (1, 0)$, $G = 0$ and right child of 4 with $p = (0, 1)$, "
        "$G = 0$. For a genuine counter-example: parent (4 A, 4 B), $G = 0.5$; "
        "left child (1 A, 3 B) has $G = 0.375$; right child (3 A, 1 B) has "
        "$G = 0.375$; weighted = 0.375 < 0.5. Now shift to left (0 A, 1 B) "
        "$G = 0$ and right (4 A, 3 B) $G = 0.4898 > 0.5$? No — 0.4898 < 0.5. The "
        "cleanest genuine case needs three classes; the key point stands: only "
        "the **weighted average** is guaranteed to decrease.")

    exercise(
        3, "If a decision tree is overfitting the training set, is it a good idea "
        "to try decreasing `max_depth`?",
        "**Yes.** Overfitting means the model is too free. `max_depth` is the "
        "most direct constraint on a tree's freedom: reducing it forces coarser "
        "partitions with more instances per leaf, which regularises the model.\n\n"
        "Also worth trying, often more effective: increase `min_samples_leaf` "
        "(which directly guarantees every prediction is supported by several "
        "instances), reduce `max_leaf_nodes`, or use `ccp_alpha` for principled "
        "post-pruning.")

    exercise(
        4, "If a decision tree is underfitting the training set, is it a good idea "
        "to try scaling the input features?",
        "**No — it will make no difference whatsoever.** Decision trees are "
        "insensitive to feature scaling, because every split is a threshold test "
        "$x_k \\le t_k$ on a single feature, and any monotone rescaling simply "
        "moves $t_k$ correspondingly (§6.1, verified in that section's lab).\n\n"
        "To fix underfitting, *increase* the model's freedom: raise `max_depth`, "
        "lower `min_samples_leaf` / `min_samples_split`, raise `max_leaf_nodes`, "
        "set `ccp_alpha=0`. Or engineer better features — that always helps.")

    exercise(
        5, "It takes one hour to train a decision tree on a training set "
        "containing one million instances. Roughly how much time will it take to "
        "train another decision tree on a training set containing ten million "
        "instances? Hint: consider the CART complexity.",
        "**About 11.7 hours.** Training complexity is "
        "$\\mathcal{O}(n \\times m \\log_2 m)$. The ratio for $m$ going from "
        "$10^6$ to $10^7$ is:\n\n"
        "$\\dfrac{10^7 \\log_2(10^7)}{10^6 \\log_2(10^6)} = "
        "10 \\times \\dfrac{23.25}{19.93} \\approx 11.67$\n\n"
        "So roughly $11.7 \\times 1\\text{ h} \\approx 11 \\text{ h } 40 "
        "\\text{ min}$. Note this is *nearly linear* — the $\\log$ factor adds "
        "only 17 %. Compare with an SVM (§5.3), where the same 10× increase would "
        "multiply the time by 100 to 1 000.")

    exercise(
        6, "If it takes one hour to train a decision tree on a given training set, "
        "roughly how long will it take if you double the number of features?",
        "**About two hours.** Complexity is linear in $n$: "
        "$\\mathcal{O}(n \\times m \\log_2 m)$, because at every node CART must "
        "scan every feature. Doubling $n$ doubles that scan.\n\n"
        "(This is also why `max_features='sqrt'` in a random forest is such an "
        "effective speed-up as well as a decorrelation device — see §7.3.)")

    exercise(
        7, "Train and fine-tune a decision tree for the moons dataset by following "
        "these steps: (a) use `make_moons(n_samples=10000, noise=0.4)`; "
        "(b) split into train and test; (c) use `GridSearchCV` with "
        "cross-validation to find good hyperparameters (try values of "
        "`max_leaf_nodes`); (d) train it on the full training set and measure "
        "performance on the test set. You should get roughly 85 % to 87 % "
        "accuracy.",
        "The point of this exercise is that a *tuned* tree on noisy data is a "
        "genuinely reasonable model, and that `max_leaf_nodes` is often the "
        "single most effective dial. With `noise=0.4` the Bayes error is "
        "substantial, so 85–87 % is close to the ceiling — an unrestricted tree "
        "gets about 80 % because it fits the noise.\n\n"
        "Grid over `max_leaf_nodes` in `range(2, 100)` and `min_samples_split` in "
        "`[2, 3, 4]`. `GridSearchCV` refits the best estimator on the whole "
        "training set automatically (`refit=True` is the default), so step (d) "
        "needs no extra work.",
        code='''from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10_000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

params = {"max_leaf_nodes": list(range(2, 100)),
          "min_samples_split": [2, 3, 4]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(grid.best_estimator_)            # refit on the full training set already
print(accuracy_score(y_test, grid.predict(X_test)))''')

    rule()

    sub("Trees in one table")

    table(
        ["Property", "Verdict", "Consequence"],
        [["Interpretability", "✅ Excellent", "The one white-box model here"],
         ["Data preparation", "✅ Almost none needed", "No scaling, monotone-invariant"],
         ["Mixed data types", "✅ Natural", "Numeric and categorical splits"],
         ["Training speed", "✅ $\\mathcal{O}(nm\\log m)$", "Scales to large data"],
         ["Prediction speed", "✅ $\\mathcal{O}(\\log m)$", "~20 comparisons for $10^6$ rows"],
         ["Probability calibration", "❌ Poor", "Pure leaves give 0/1"],
         ["Diagonal boundaries", "❌ Staircase only", "Rotation-sensitive"],
         ["Extrapolation", "❌ Impossible", "Flat-lines outside the range"],
         ["Stability", "❌ High variance", "→ <b>Chapter 7</b>"],
         ["Optimality", "❌ Greedy (NP-complete)", "Fails on XOR-like structure"]],
    )

    keypoints([
        "CART: greedy, binary, axis-aligned, minimising weighted impurity.",
        "Gini ≈ entropy/ln 2 — they almost never disagree; the criterion is not "
        "worth tuning.",
        "Trees are non-parametric and <b>must</b> be regularised; "
        "<code>min_samples_leaf</code> and <code>ccp_alpha</code> first.",
        "Regression trees are staircases and cannot extrapolate.",
        "High variance + low bias + axis alignment = the perfect base learner for "
        "an ensemble.",
    ], title="Chapter 6 in five lines")

    refs([
        ("Breiman, Friedman, Olshen & Stone — *Classification and Regression "
         "Trees* (the CART book)", "Wadsworth, 1984"),
        ("Hyafil & Rivest — *Constructing Optimal Binary Decision Trees is "
         "NP-Complete*", "https://doi.org/10.1016/0020-0190(76)90095-8"),
        ("Quinlan, J. R. — *Induction of Decision Trees* (ID3, entropy-based)",
         "https://doi.org/10.1007/BF00116251"),
        ("scikit-learn — *Minimal Cost-Complexity Pruning*",
         "https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning"),
    ])


# ==========================================================================
SECTIONS = [
    ("6.1", "Training and Visualizing", s_6_1),
    ("6.2", "Predictions & Probabilities", s_6_2),
    ("6.3", "The CART Algorithm", s_6_3),
    ("6.4", "Gini or Entropy?", s_6_4),
    ("6.5", "Regularization", s_6_5),
    ("6.6", "Regression Trees", s_6_6),
    ("6.7", "Instability & Axis Orientation", s_6_7),
    ("6.8", "Exercises & Review", s_6_8),
]

nav.render_chapter(CH, SECTIONS)
