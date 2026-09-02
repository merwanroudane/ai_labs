"""Chapter 7 — Ensemble Learning and Random Forests."""

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
CH = "ch07"

hero(
    kicker="Part I · Chapter 7",
    title="Ensemble Learning and Random Forests",
    blurb=(
        "Aggregate the predictions of many models and you routinely beat the best "
        "single model. This chapter derives <i>why</i> — the binomial argument for "
        "voting, the variance formula for bagging, the stagewise view of boosting "
        "— and builds every major ensemble from scratch: voting, bagging, random "
        "forests, extra-trees, AdaBoost, gradient boosting, histogram boosting, "
        "and stacking."
    ),
    chips=["The winning family", "8 sub-sections", "8 animations",
           "9 code labs", "AdaBoost & GBM derived"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_7_1():
    section("7.1", "Voting Classifiers")

    lead(
        "The simplest ensemble: train several different models and take a vote. "
        "It works for a reason you can compute exactly, and the reason tells you "
        "when it will fail."
    )

    sub("Hard and soft voting")

    math(r"""
    \textbf{hard voting:}\qquad
    \hat y \;=\; \operatorname*{arg\,max}_{k}\;
      \sum_{b=1}^{B} \mathbb{1}\bigl[\, \hat y_b = k \,\bigr]
    """)
    math(r"""
    \textbf{soft voting:}\qquad
    \hat y \;=\; \operatorname*{arg\,max}_{k}\;
      \frac{1}{B}\sum_{b=1}^{B} \hat p_{b,k}
    """)
    where({r"B": "the number of models in the ensemble",
           r"\hat y_b": "the class predicted by model $b$",
           r"\hat p_{b,k}": "model $b$'s estimated probability of class $k$"})

    tip(
        "Soft voting usually wins",
        "Because it weights each vote by <b>confidence</b>. A model that says "
        "\"class 1, probability 0.51\" should not count the same as one saying "
        "\"class 0, probability 0.99\". Soft voting needs every model to expose "
        "<code>predict_proba</code> — for <code>SVC</code> that means "
        "<code>probability=True</code>, which is slow (§5.8, exercise 4).",
    )

    sub("Why a vote of weak models is strong")

    derive(
        [("Suppose you have $B$ <b>independent</b> classifiers, each correct with "
          "probability $p > 0.5$. The number of correct votes is binomial:",
          r"N_{\text{correct}} \sim \mathrm{Binomial}(B,\, p)"),
         ("The majority is correct whenever more than half the votes are right:",
          r"\Pr[\text{ensemble correct}] = \sum_{k=\lceil B/2 \rceil}^{B} "
          r"\binom{B}{k} p^{k}(1-p)^{B-k}"),
         ("For $p = 0.51$ and $B = 1000$ this evaluates to about $0.75$; for "
          "$B = 10\\,000$ it is about $0.97$. A barely-better-than-random voter, "
          "repeated, becomes near-certain.", None),
         ("The reason is the law of large numbers plus a threshold. The sample "
          "mean of the votes has standard deviation",
          r"\sigma_{\bar N} = \sqrt{\frac{p(1-p)}{B}}"),
         ("and the ensemble errs only if $\\bar N$ falls below $0.5$, i.e. more "
          "than $(p - 0.5)/\\sigma_{\\bar N} = (p-0.5)\\sqrt{B/(p(1-p))}$ standard "
          "deviations below its mean. That $z$-score grows like $\\sqrt{B}$, so "
          "the error probability decays <b>exponentially</b> in $B$:",
          r"\Pr[\text{error}] \;\approx\; \Phi\!\left(-\,(p - 0.5)\sqrt{\frac{B}{p(1-p)}}\right)"),
         ("<b>The catch is the word independent.</b> Real models trained on the "
          "same data are correlated: they make the <i>same</i> mistakes on the "
          "<i>same</i> hard instances. Correlation is the whole reason bagging "
          "(§7.2) resamples the data and random forests (§7.3) resample the "
          "features — they are devices for manufacturing independence.", None)],
        title="The binomial argument, and why it needs independence",
    )

    anim_header("The biased-coin argument, simulated")
    md(
        "1 000 coins with $p = 0.51$. Each frame tosses one more and re-computes "
        "the running majority. The theoretical curve is drawn alongside — and the "
        "dashed red curve shows what happens with correlated voters."
    )

    from scipy.stats import binom
    Bs = np.arange(1, 1001, 2)
    p_ind = np.array([1 - binom.cdf(b // 2, b, .51) for b in Bs])

    # correlated voters: effective sample size B_eff = B / (1 + (B-1)rho)
    rho = .35
    B_eff = Bs / (1 + (Bs - 1) * rho)
    p_cor = np.array([1 - binom.cdf(int(be) // 2, max(int(be), 1), .51)
                      for be in B_eff])

    rng = np.random.default_rng(0)
    tosses = (rng.random((1000,)) < .51).astype(int)
    running = np.cumsum(tosses) / np.arange(1, 1001)

    steps = np.unique(np.linspace(1, 1000, 70).astype(int))
    frames = []
    for k in steps:
        i = np.searchsorted(Bs, k)
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=Bs[:i + 1], y=p_ind[:i + 1], mode="lines",
                       line=dict(color=C["success"], width=3.2)),
            go.Scatter(x=Bs[:i + 1], y=p_cor[:i + 1], mode="lines",
                       line=dict(color=C["danger"], width=2.6, dash="dash")),
            go.Scatter(x=np.arange(1, k + 1), y=running[:k], mode="lines",
                       line=dict(color=C["muted"], width=1.4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"B = {k}   ·   P(majority correct), independent = "
            f"{p_ind[min(i, len(p_ind)-1)]:.4f}   ·   correlated (ρ=0.35) = "
            f"{p_cor[min(i, len(p_cor)-1)]:.4f}")])))

    f = go.Figure(data=[
        go.Scatter(x=Bs[:1], y=p_ind[:1], mode="lines",
                   name="P(majority correct) — independent voters",
                   line=dict(color=C["success"], width=3.2)),
        go.Scatter(x=Bs[:1], y=p_cor[:1], mode="lines",
                   name="… with correlation ρ = 0.35",
                   line=dict(color=C["danger"], width=2.6, dash="dash")),
        go.Scatter(x=[1], y=running[:1], mode="lines",
                   name="one simulated running fraction",
                   line=dict(color=C["muted"], width=1.4)),
    ])
    f.add_hline(y=.51, line_dash="dot", line_color=C["truth"],
                annotation_text="a single voter: p = 0.51")
    f.update_layout(height=450, xaxis_title="number of voters B",
                    yaxis_title="probability the ensemble is correct",
                    yaxis=dict(range=[.45, 1.02]),
                    title="Weak voters become a strong ensemble — if independent",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="B = ")
    figure(f, "The green curve races to 1. The red one stalls near 0.75 — "
              "correlation caps what voting can buy you.")

    idea(
        "Diversity is the resource you are spending",
        "An ensemble improves on its members only to the extent that they make "
        "<b>different</b> errors. Everything in this chapter is a strategy for "
        "manufacturing difference: different algorithms (voting), different "
        "training subsets (bagging), different feature subsets (random forests), "
        "different random thresholds (extra-trees), or different <i>objectives</i> "
        "(boosting, where each model targets the previous one's mistakes).",
    )

    code_lab(
        "Voting classifiers, and measuring diversity",
        '''import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binom

X, y = make_moons(n_samples=2000, noise=.35, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)

log = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
nb  = GaussianNB()
knn = KNeighborsClassifier(15)

estimators = [("lr", log), ("rf", rnd), ("svc", svm), ("nb", nb), ("knn", knn)]
hard = VotingClassifier(estimators, voting="hard")
soft = VotingClassifier(estimators, voting="soft")

print(f"{'model':<26}{'test accuracy':>15}")
preds = {}
for nm, clf in estimators + [("HARD voting", hard), ("SOFT voting", soft)]:
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    preds[nm] = p
    print(f"{nm:<26}{(p == yte).mean():>15.4f}")

# ============ how DIVERSE are the members? =============================
print("\\n=== pairwise disagreement (higher = more diverse) ===")
names = [n for n, _ in estimators]
print(f"{'':<6}" + "".join(f"{n:>7}" for n in names))
for a in names:
    row = "".join(f"{np.mean(preds[a] != preds[b]):>7.3f}" for b in names)
    print(f"{a:<6}{row}")

# ---- do they make the SAME mistakes? ----------------------------------
errs = {n: preds[n] != yte for n in names}
print(f"\\n{'model':<6}{'errors':>8}{'  ...also wrong for every other model':>0}")
common = np.ones(len(yte), bool)
for n in names:
    common &= errs[n]
for n in names:
    print(f"{n:<6}{errs[n].sum():>8}")
print(f"\\ninstances ALL five get wrong : {common.sum()}  "
      f"({common.sum()/len(yte):.1%}) <- voting cannot fix these")
any_wrong = np.zeros(len(yte), bool)
for n in names: any_wrong |= errs[n]
print(f"instances SOMEONE gets wrong : {any_wrong.sum()}")
print(f"the ensemble's headroom lives in the {any_wrong.sum()-common.sum()} "
      f"instances where the models disagree")

# ============ the binomial argument ====================================
print("\\n=== P(majority of B independent voters is correct) ===")
print(f"{'B':>6}" + "".join(f"{f'p={p}':>10}" for p in [0.51, 0.55, 0.60, 0.75]))
for B in [1, 5, 11, 51, 101, 501, 1001, 10001]:
    row = "".join(f"{1 - binom.cdf(B//2, B, p):>10.4f}"
                  for p in [0.51, 0.55, 0.60, 0.75])
    print(f"{B:>6}{row}")
print("\\np=0.51 with B=10001 gives ~0.98. That is the whole promise of ensembling")
print("-- and it assumes INDEPENDENCE, which real models never fully have.")
''',
        key="ch07_voting",
    )

    keypoints([
        "Hard voting = majority of predictions; soft voting = average of "
        "probabilities (usually better).",
        "$B$ independent voters at $p > 0.5$ give an ensemble whose error decays "
        "<b>exponentially</b> in $B$.",
        "The assumption that matters is <b>independence</b>; correlation caps the "
        "gain.",
        "Diversity is the resource: different algorithms, data, features, or "
        "objectives.",
        "Instances that <i>every</i> member gets wrong are beyond any voting "
        "scheme's reach.",
    ])


# ==========================================================================
def s_7_2():
    section("7.2", "Bagging and Pasting")

    lead(
        "Instead of different algorithms, use the <b>same</b> algorithm on "
        "different random subsets of the training data. Bootstrap AGGregatING — "
        "bagging — is Breiman's 1996 idea and it is still one of the best "
        "returns-on-effort in machine learning."
    )

    sub("Bagging vs pasting")

    table(
        ["", "Bagging", "Pasting"],
        [["Sampling", "<b>With</b> replacement (bootstrap)",
          "<b>Without</b> replacement"],
         ["Can an instance repeat within one subset?", "Yes", "No"],
         ["Can an instance appear in several subsets?", "Yes", "Yes"],
         ["Diversity of the subsets", "Higher", "Lower"],
         ["Bias of each predictor", "Slightly higher", "Slightly lower"],
         ["Variance of the ensemble", "<b>Lower</b>", "Higher"],
         ["Out-of-bag evaluation", "<b>✅ free</b>", "❌ not available"],
         ["scikit-learn", "<code>BaggingClassifier(bootstrap=True)</code>",
          "<code>BaggingClassifier(bootstrap=False)</code>"]],
        "Bagging usually wins, and it comes with free validation. Use it unless "
        "you have measured otherwise.",
    )

    sub("Why bagging reduces variance")

    derive(
        [("Let the $B$ base predictors each have variance $\\sigma^2$ and pairwise "
          "correlation $\\rho$. The ensemble prediction is their average "
          "$\\bar h = \\frac1B\\sum_b h_b$.", None),
         ("Expand the variance of a sum: $B$ diagonal terms and $B(B-1)$ "
          "off-diagonal ones:",
          r"\mathrm{Var}\!\left(\sum_{b=1}^{B} h_b\right) = "
          r"\sum_b \mathrm{Var}(h_b) + \sum_{b \ne b'}\mathrm{Cov}(h_b, h_{b'}) "
          r"= B\sigma^2 + B(B-1)\rho\sigma^2"),
         ("Divide by $B^2$ to get the variance of the average:",
          r"\boxed{\;\mathrm{Var}(\bar h) = \rho\,\sigma^{2} "
          r"+ \frac{1-\rho}{B}\,\sigma^{2}\;}"),
         ("Two terms, with completely different behaviour. The <b>second</b> "
          "vanishes as $B\\to\\infty$ — that is the free lunch, and it costs only "
          "compute. The <b>first</b> does not depend on $B$ at all: it is a floor "
          "set entirely by how correlated the predictors are.", None),
         ("So there are exactly two levers: <b>more trees</b> (drives the second "
          "term down, with diminishing returns) and <b>lower $\\rho$</b> (lowers "
          "the floor). Bagging attacks $\\rho$ by resampling rows; random forests "
          "(§7.3) attack it harder by also resampling <i>columns</i> at every "
          "split.", None),
         ("Note what bagging does <b>not</b> do: it barely changes the bias. Each "
          "bagged tree sees ~63 % of the data, so its bias is slightly worse than "
          "a tree on the full set — and the ensemble's bias is roughly that. "
          "<b>Bag low-bias, high-variance models</b>: deep, unpruned trees. Bagging "
          "a linear regression accomplishes almost nothing.", None)],
        title="Var(average) = ρσ² + (1−ρ)σ²/B",
    )

    anim_header("Variance collapsing as trees are added")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier

    Xm, ym = ds.moons(n=350, noise=.32)
    b1 = np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 110)
    b2 = np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 110)
    B1, B2 = np.meshgrid(b1, b2); BB = np.c_[B1.ravel(), B2.ravel()]

    rng = np.random.default_rng(0)
    single_preds = []
    for s in range(60):
        idx = rng.choice(len(Xm), len(Xm), replace=True)
        t = DecisionTreeClassifier(random_state=s).fit(Xm[idx], ym[idx])
        single_preds.append(t.predict_proba(BB)[:, 1])
    single_preds = np.array(single_preds)

    counts = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
    frames = []
    for n in counts:
        avg = single_preds[:n].mean(0).reshape(B1.shape)
        var = float(single_preds[:n].mean(0).var())
        mvar = float(single_preds[:n].var(0).mean())
        frames.append(go.Frame(name=str(n), data=[
            go.Heatmap(x=b1, y=b2, z=avg, colorscale=nav.cscale(), zmin=0, zmax=1,
                       showscale=False),
            go.Scatter(x=Xm[ym == 0, 0], y=Xm[ym == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=5,
                                   line=dict(color="#fff", width=.6))),
            go.Scatter(x=Xm[ym == 1, 0], y=Xm[ym == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=5,
                                   line=dict(color="#fff", width=.6))),
            go.Scatter(x=counts[:counts.index(n) + 1],
                       y=[float(single_preds[:k].var(0).mean() / k
                                + 0 * k) for k in counts[:counts.index(n) + 1]],
                       mode="lines+markers",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{n} bagged trees   ·   mean per-point variance of the ensemble "
            f"≈ {mvar/n:.5f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.62, .38],
                      subplot_titles=("averaged prediction surface",
                                      "ensemble variance ≈ σ²/B"))
    f.add_trace(go.Heatmap(x=b1, y=b2, z=single_preds[:1].mean(0).reshape(B1.shape),
                           colorscale=nav.cscale(), zmin=0, zmax=1,
                           showscale=False), 1, 1)
    f.add_trace(go.Scatter(x=Xm[ym == 0, 0], y=Xm[ym == 0, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["train"], size=5,
                           line=dict(color="#fff", width=.6))), 1, 1)
    f.add_trace(go.Scatter(x=Xm[ym == 1, 0], y=Xm[ym == 1, 1], mode="markers",
                           showlegend=False, marker=dict(color=C["warning"], size=5,
                           line=dict(color="#fff", width=.6))), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=[float(single_preds[:1].var(0).mean())],
                           mode="lines+markers", showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.update_xaxes(title_text="number of trees B", type="log", row=1, col=2)
    f.update_yaxes(title_text="variance", type="log", row=1, col=2)
    f.update_layout(height=470, title="Bagging: averaging away the variance")
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="B = ")
    figure(f)

    sub("Out-of-bag evaluation — free validation")

    derive(
        [("A bootstrap sample of size $m$ draws $m$ times <i>with replacement</i>. "
          "The probability that a particular instance is <b>not</b> picked on one "
          "draw is $1 - 1/m$.", None),
         ("Over $m$ independent draws, the probability it is never picked:",
          r"\Pr[\text{instance is out-of-bag}] = \left(1 - \frac{1}{m}\right)^{m}"),
         ("Take the limit, using $\\lim_{m\\to\\infty}(1 - 1/m)^m = e^{-1}$:",
          r"\lim_{m \to \infty}\left(1 - \frac{1}{m}\right)^{m} = e^{-1} \approx 0.368"),
         ("So each bagged predictor is trained on about <b>63.2 %</b> of the "
          "instances, and about <b>36.8 %</b> are out-of-bag for it — never seen "
          "during its training.", None),
         ("Evaluate each instance using only the predictors for which it was "
          "out-of-bag, and average. That is the <b>OOB score</b>: an almost "
          "unbiased estimate of generalisation error, obtained with <b>no "
          "validation set and no extra fitting</b>.",
          r"\mathrm{OOB} = \frac{1}{m}\sum_{i=1}^{m} L\!\left(y^{(i)},\;"
          r"\frac{1}{|\mathcal{B}_i|}\sum_{b \in \mathcal{B}_i} h_b\bigl(\mathbf{x}^{(i)}\bigr)\right)"),
         ("where $\\mathcal{B}_i = \\{b : \\text{instance } i \\text{ is OOB for "
          "predictor } b\\}$, of expected size $0.368\\,B$. With $B = 100$ that is "
          "about 37 predictors per instance — plenty. Set "
          "<code>oob_score=True</code>.", None)],
        title="Why 63.2 % in-bag and 36.8 % out-of-bag",
    )

    sub("Random patches and random subspaces")

    table(
        ["Method", "<code>bootstrap</code>", "<code>bootstrap_features</code>",
         "Samples rows?", "Samples columns?"],
        [["Bagging", "True", "False", "✅", "❌"],
         ["Pasting", "False", "False", "✅ (no replacement)", "❌"],
         ["<b>Random subspaces</b>", "False", "True", "❌ (keeps all)", "✅"],
         ["<b>Random patches</b>", "True", "True", "✅", "✅"]],
        "Sampling features is especially valuable with high-dimensional inputs "
        "(images, text) — it buys more diversity, hence a lower $\\rho$.",
    )

    code_lab(
        "Bagging, pasting, OOB, and the 63.2 % law",
        '''import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=2000, noise=.33, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)

# ============ 1. the 63.2 % law, verified by simulation ================
print("=== P(an instance is in the bootstrap sample) ===")
rng = np.random.default_rng(0)
print(f"{'m':>8}{'(1-1/m)^m':>13}{'simulated OOB fraction':>26}")
for m in [5, 10, 100, 1_000, 10_000]:
    theory = (1 - 1/m)**m
    sim = np.mean([1 - len(np.unique(rng.integers(0, m, m)))/m for _ in range(200)])
    print(f"{m:>8}{theory:>13.5f}{sim:>26.5f}")
print(f"{'limit':>8}{np.exp(-1):>13.5f}{'':>26}   <- 1/e = 0.36788")

# ============ 2. single tree vs bagging vs pasting =====================
tree = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)

bag = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                        n_estimators=500, max_samples=1.0, bootstrap=True,
                        oob_score=True, n_jobs=-1, random_state=42).fit(Xtr, ytr)
pas = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                        n_estimators=500, max_samples=0.63, bootstrap=False,
                        n_jobs=-1, random_state=42).fit(Xtr, ytr)

print(f"\\n{'model':<28}{'train':>9}{'OOB':>9}{'test':>9}")
print(f"{'single decision tree':<28}{tree.score(Xtr,ytr):>9.4f}{'--':>9}"
      f"{tree.score(Xte,yte):>9.4f}")
print(f"{'bagging (500 trees)':<28}{bag.score(Xtr,ytr):>9.4f}"
      f"{bag.oob_score_:>9.4f}{bag.score(Xte,yte):>9.4f}")
print(f"{'pasting (500 trees)':<28}{pas.score(Xtr,ytr):>9.4f}{'--':>9}"
      f"{pas.score(Xte,yte):>9.4f}")
print(f"\\nOOB estimate {bag.oob_score_:.4f} vs true test {bag.score(Xte,yte):.4f} "
      f"-> error {abs(bag.oob_score_-bag.score(Xte,yte)):.4f}")
print("OOB is a free, nearly unbiased validation score. No holdout needed.")

# ---- how many predictors vote on each OOB instance? -----------------
oob_counts = (~np.array([np.isin(np.arange(len(Xtr)), idx)
                         for idx in bag.estimators_samples_])).sum(0)
print(f"\\nper-instance OOB predictor count: mean {oob_counts.mean():.1f} "
      f"of {bag.n_estimators} (theory: {0.368*bag.n_estimators:.1f}), "
      f"min {oob_counts.min()}")

# ============ 3. the variance formula ==================================
print("\\n=== Var(mean) = rho*s^2 + (1-rho)*s^2/B, measured ===")
grid = np.c_[np.repeat(np.linspace(-1.5, 2.5, 40), 40),
             np.tile(np.linspace(-1, 1.5, 40), 40)]
P = np.array([e.predict_proba(grid[:, :2])[:, 1] for e in bag.estimators_[:200]])
s2 = P.var(0).mean()
# average pairwise correlation across trees
Pc = P - P.mean(1, keepdims=True)
Cm = np.corrcoef(P)
rho = float((Cm.sum() - len(Cm)) / (len(Cm)*(len(Cm)-1)))
print(f"per-tree variance sigma^2 = {s2:.5f}")
print(f"average pairwise correlation rho = {rho:.4f}")
print(f"\\n{'B':>6}{'predicted Var':>16}{'measured Var':>15}")
for B in [1, 5, 20, 50, 100, 200]:
    pred = rho*s2 + (1-rho)/B*s2
    meas = float(np.mean([P[i:i+B].mean(0).var() for i in range(0, 200-B+1, max(1,B))]))
    print(f"{B:>6}{pred:>16.5f}{meas:>15.5f}")
print("\\nThe floor rho*sigma^2 is what random forests attack next.")

# ============ 4. random patches / subspaces ============================
print("\\n=== four sampling schemes ===")
print(f"{'scheme':<22}{'bootstrap':>11}{'feat boot':>11}{'test acc':>11}")
for nm, kw in [("bagging",          dict(bootstrap=True,  bootstrap_features=False)),
               ("pasting",          dict(bootstrap=False, bootstrap_features=False,
                                          max_samples=.63)),
               ("random subspaces", dict(bootstrap=False, bootstrap_features=True,
                                          max_features=.7, max_samples=1.0)),
               ("random patches",   dict(bootstrap=True,  bootstrap_features=True,
                                          max_features=.7))]:
    m_ = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                           n_estimators=200, n_jobs=-1, random_state=42,
                           **kw).fit(Xtr, ytr)
    print(f"{nm:<22}{str(kw.get('bootstrap')):>11}"
          f"{str(kw.get('bootstrap_features')):>11}{m_.score(Xte, yte):>11.4f}")

# ============ 5. bagging a LOW-variance model does nothing =============
from sklearn.linear_model import LogisticRegression
print("\\n=== bag a high-variance model, not a low-variance one ===")
for nm, base in [("decision tree (high var)", DecisionTreeClassifier(random_state=42)),
                 ("logistic reg (low var)",   LogisticRegression())]:
    solo = base.fit(Xtr, ytr).score(Xte, yte)
    bg = BaggingClassifier(base, n_estimators=200, n_jobs=-1,
                           random_state=42).fit(Xtr, ytr).score(Xte, yte)
    print(f"  {nm:<26} solo {solo:.4f} -> bagged {bg:.4f}   gain {bg-solo:+.4f}")
''',
        key="ch07_bagging",
    )

    quiz(
        "You bag 500 fully-grown decision trees and the ensemble's test accuracy "
        "stops improving past ~150 trees. What is the binding constraint?",
        ["The bias of the individual trees",
         "The correlation $\\rho$ between the trees",
         "The training-set size",
         "The number of features"],
        1,
        "$\\mathrm{Var}(\\bar h) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$. Past "
        "a certain $B$ the second term is negligible and you are sitting on the "
        "$\\rho\\sigma^2$ floor. More trees cannot help; <b>decorrelating</b> them "
        "can — which is what §7.3 does.",
        key="ch07q1",
    )

    keypoints([
        "Bagging = same algorithm, bootstrap samples of the rows; pasting = "
        "without replacement.",
        "$\\mathrm{Var}(\\bar h) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ — "
        "more trees kills the second term, decorrelation lowers the first.",
        "Bag <b>low-bias, high-variance</b> models: deep unpruned trees. Bagging a "
        "linear model is pointless.",
        "$(1-1/m)^m \\to e^{-1}$: each tree sees ~63.2 %, so ~36.8 % are OOB — "
        "<b>free validation</b>.",
        "Sampling features too gives random subspaces / random patches, useful in "
        "high dimensions.",
    ])


# ==========================================================================
def s_7_3():
    section("7.3", "Random Forests, Extra-Trees, and Feature Importance")

    lead(
        "Bagged trees plus one extra source of randomness: at every split, only a "
        "random subset of features is considered. That single change attacks the "
        "$\\rho\\sigma^2$ floor directly."
    )

    sub("The extra randomness")

    md(
        "A `RandomForestClassifier` is (almost exactly) a `BaggingClassifier` of "
        "`DecisionTreeClassifier(max_features='sqrt')`. At each node it samples "
        "$m_{\\text{try}}$ features and searches for the best split among those "
        "only:"
    )

    math(r"""
    m_{\text{try}} \;=\;
    \begin{cases}
      \bigl\lceil \sqrt{n} \bigr\rceil & \text{classification (the default)}\\[4pt]
      n & \text{regression (scikit-learn's default; } n/3 \text{ is the classic choice)}
    \end{cases}
    """)

    idea(
        "Why restricting the split search helps",
        "Suppose one feature is strongly predictive. Every bagged tree will pick "
        "it as the root split, so all the trees look alike — $\\rho$ is high and "
        "the ensemble is barely better than one tree. Forcing each node to choose "
        "among a random $\\sqrt{n}$ features means the dominant feature is "
        "<i>unavailable</i> most of the time, so other features get used, the "
        "trees differ, and $\\rho$ falls. Each individual tree gets slightly "
        "<b>worse</b> (higher bias) and the ensemble gets <b>better</b>. That "
        "trade is the entire idea.",
    )

    sub("Extremely Randomized Trees (Extra-Trees)")

    table(
        ["", "Random Forest", "Extra-Trees"],
        [["Row sampling", "Bootstrap (default)",
          "None by default (<code>bootstrap=False</code>)"],
         ["Feature sampling at each node", "Random subset", "Random subset"],
         ["Threshold choice", "<b>Best</b> threshold, searched exhaustively",
          "<b>Random</b> threshold per candidate feature; best among those"],
         ["Bias", "Lower", "Higher"],
         ["Variance", "Higher", "<b>Lower</b>"],
         ["Training speed", "Baseline", "<b>Much faster</b> — no threshold search"],
         ["Class", "<code>RandomForestClassifier</code>",
          "<code>ExtraTreesClassifier</code>"]],
    )

    tip(
        "You cannot tell in advance which will win",
        "Try both and cross-validate. Extra-Trees are dramatically cheaper to "
        "train (the exhaustive threshold search is the expensive part of CART), so "
        "they are often the better starting point on wide data.",
    )

    anim_header("max_features sweeping: correlation down, accuracy up")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    Xc, yc = make_classification(n_samples=1200, n_features=25, n_informative=8,
                                 n_redundant=6, class_sep=.9, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=.3, random_state=0)

    mfs = [1, 2, 3, 5, 8, 12, 18, 25]
    stats = []
    for mf in mfs:
        rf = RandomForestClassifier(n_estimators=120, max_features=mf,
                                    random_state=0, n_jobs=-1).fit(Xtr, ytr)
        P = np.array([e.predict_proba(Xte)[:, 1] for e in rf.estimators_])
        Cm = np.corrcoef(P)
        rho = float((Cm.sum() - len(Cm)) / (len(Cm) * (len(Cm) - 1)))
        solo = float(np.mean([e.score(Xte, yte) for e in rf.estimators_[:25]]))
        stats.append((rho, solo, float(rf.score(Xte, yte))))

    frames = []
    for k in range(1, len(mfs) + 1):
        frames.append(go.Frame(name=str(mfs[k - 1]), data=[
            go.Scatter(x=mfs[:k], y=[s[0] for s in stats[:k]], mode="lines+markers",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=mfs[:k], y=[s[1] for s in stats[:k]], mode="lines+markers",
                       line=dict(color=C["warning"], width=3)),
            go.Scatter(x=mfs[:k], y=[s[2] for s in stats[:k]], mode="lines+markers",
                       line=dict(color=C["success"], width=3.4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"max_features = {mfs[k-1]}   ·   tree correlation ρ = {stats[k-1][0]:.3f}"
            f"   ·   single-tree acc = {stats[k-1][1]:.3f}"
            f"   ·   FOREST acc = {stats[k-1][2]:.3f}")])))

    f = go.Figure(data=[
        go.Scatter(x=mfs[:1], y=[stats[0][0]], mode="lines+markers",
                   name="tree-to-tree correlation ρ",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=mfs[:1], y=[stats[0][1]], mode="lines+markers",
                   name="average SINGLE tree accuracy",
                   line=dict(color=C["warning"], width=3)),
        go.Scatter(x=mfs[:1], y=[stats[0][2]], mode="lines+markers",
                   name="FOREST accuracy",
                   line=dict(color=C["success"], width=3.4)),
    ])
    f.add_vline(x=int(np.ceil(np.sqrt(25))), line_dash="dash",
                line_color=C["primary"], annotation_text="√n = 5 (the default)")
    f.update_layout(height=460, xaxis_title="max_features (of 25)",
                    yaxis=dict(range=[0, 1]), yaxis_title="value",
                    title="Restricting the split search decorrelates the trees",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(650), slider_prefix="max_features = ")
    figure(f, "Note the crossing pattern: as max_features falls, individual trees "
              "get worse but the forest gets better — because ρ falls faster.")

    sub("Feature importance")

    md("scikit-learn's default importance is **mean decrease in impurity** (MDI):")

    math(r"""
    \mathrm{Imp}(j) \;=\; \frac{1}{B}\sum_{b=1}^{B}
      \sum_{\substack{t \in T_b \\ v(t) = j}}
      \frac{m_t}{m}\,\Bigl(G_t - \tfrac{m_{t,L}}{m_t}G_{t,L}
                                 - \tfrac{m_{t,R}}{m_t}G_{t,R}\Bigr)
    """)
    where({r"v(t) = j": "nodes $t$ that split on feature $j$",
           r"m_t/m": "the fraction of training instances reaching node $t$",
           r"G_t": "the impurity at node $t$ before the split"})

    pitfall(
        "MDI importance is biased — three ways",
        "<b>(1) It favours high-cardinality features.</b> A continuous feature or "
        "an ID column offers thousands of possible thresholds, so by chance one of "
        "them reduces impurity. A binary feature offers one. <b>(2) It is computed "
        "on training data</b>, so it rewards overfitting. <b>(3) Correlated "
        "features split the credit</b> arbitrarily: two identical columns each get "
        "half the importance, making both look unimportant.<br><br>"
        "The fix is <b>permutation importance</b>: shuffle one column in the "
        "<i>validation</i> set and measure how much the score drops. It is model "
        "agnostic, computed on held-out data, and unbiased with respect to "
        "cardinality. It is slower, and it still splits credit between correlated "
        "features — for that, drop-column importance or grouped permutation is "
        "needed.",
    )

    math(r"""
    \mathrm{PermImp}(j) \;=\; s \;-\; \frac{1}{R}\sum_{r=1}^{R}
      s\bigl(\mathbf{X}^{\pi_{j,r}}, \mathbf{y}\bigr)
    """)
    where({r"s": "the model's score on the untouched validation set",
           r"\mathbf{X}^{\pi_{j,r}}": "the validation set with column $j$ randomly "
                                      "permuted (repetition $r$)",
           r"R": "number of repetitions, to average out the shuffle noise"})

    code_lab(
        "Forests, extra-trees, and the importance trap",
        '''import numpy as np, pandas as pd, time
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=2000, n_features=25, n_informative=8,
                           n_redundant=6, class_sep=.9, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0)

# ============ 1. a forest IS bagging + max_features ====================
print("=== equivalence check ===")
rf  = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                             random_state=42, n_jobs=-1).fit(Xtr, ytr)
bag = BaggingClassifier(DecisionTreeClassifier(max_features="sqrt", random_state=42),
                        n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print(f"RandomForest                     : {rf.score(Xte, yte):.4f}")
print(f"Bagging(Tree(max_features='sqrt')): {bag.score(Xte, yte):.4f}")
print(f"predictions agree on "
      f"{np.mean(rf.predict(Xte) == bag.predict(Xte)):.1%} of test instances")

# ============ 2. forest vs extra-trees ================================
print(f"\\n{'model':<26}{'fit time':>11}{'CV acc':>10}{'test acc':>11}")
for nm, mdl in [("DecisionTree",  DecisionTreeClassifier(random_state=0)),
                ("Bagging",       BaggingClassifier(n_estimators=300,
                                                    random_state=0, n_jobs=-1)),
                ("RandomForest",  RandomForestClassifier(n_estimators=300,
                                                         random_state=0, n_jobs=-1)),
                ("ExtraTrees",    ExtraTreesClassifier(n_estimators=300,
                                                       random_state=0, n_jobs=-1))]:
    t0 = time.perf_counter(); mdl.fit(Xtr, ytr); dt = time.perf_counter()-t0
    cv = cross_val_score(mdl, Xtr, ytr, cv=3, n_jobs=-1).mean()
    print(f"{nm:<26}{dt:>10.3f}s{cv:>10.4f}{mdl.score(Xte, yte):>11.4f}")

# ============ 3. max_features decorrelates =============================
print(f"\\n=== max_features vs tree correlation ===")
print(f"{'max_features':>13}{'rho':>9}{'single tree':>13}{'forest':>9}")
for mf in [1, 2, 5, 10, 25]:
    m_ = RandomForestClassifier(n_estimators=100, max_features=mf,
                                random_state=0, n_jobs=-1).fit(Xtr, ytr)
    P = np.array([e.predict_proba(Xte)[:, 1] for e in m_.estimators_])
    Cm = np.corrcoef(P)
    rho = (Cm.sum()-len(Cm))/(len(Cm)*(len(Cm)-1))
    solo = np.mean([e.score(Xte, yte) for e in m_.estimators_[:25]])
    print(f"{mf:>13}{rho:>9.3f}{solo:>13.4f}{m_.score(Xte, yte):>9.4f}")

# ============ 4. THE IMPORTANCE TRAP ==================================
print("\\n" + "="*64)
print("MDI importance is biased toward high-cardinality features")
print("="*64)
rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "real_signal":  rng.normal(0, 1, n),           # genuinely predictive
    "binary_noise": rng.integers(0, 2, n),         # 2 distinct values, useless
    "cat10_noise":  rng.integers(0, 10, n),        # 10 values, useless
    "random_id":    rng.permutation(n),            # n values, USELESS
    "cont_noise":   rng.normal(0, 1, n),           # continuous, useless
})
target = (df["real_signal"] + rng.normal(0, .5, n) > 0).astype(int)
Xa, Xb, ya, yb = train_test_split(df, target, test_size=.35, random_state=0)

frf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(Xa, ya)
mdi = pd.Series(frf.feature_importances_, index=df.columns)
perm = permutation_importance(frf, Xb, yb, n_repeats=25, random_state=0, n_jobs=-1)
pim = pd.Series(perm.importances_mean, index=df.columns)

out = pd.DataFrame({"n_unique": df.nunique(),
                    "MDI (train)": mdi.round(4),
                    "permutation (test)": pim.round(4)})
print(out.sort_values("MDI (train)", ascending=False).to_string())
print(f"\\n'random_id' is pure noise with {n} distinct values.")
print(f"MDI gives it {mdi['random_id']:.1%} of the importance.")
print(f"Permutation importance correctly gives it {pim['random_id']:+.4f} (~0).")
print(f"\\ntest accuracy = {frf.score(Xb, yb):.4f} -- the model is fine;")
print("it is the EXPLANATION that MDI gets wrong.")

# ---- correlated features split the credit ---------------------------
print("\\n=== correlated features split the credit ===")
df2 = df.copy()
df2["real_copy"] = df2["real_signal"] + rng.normal(0, .01, n)   # near-duplicate
Xa2, Xb2, ya2, yb2 = train_test_split(df2, target, test_size=.35, random_state=0)
f2 = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(Xa2, ya2)
p2 = permutation_importance(f2, Xb2, yb2, n_repeats=25, random_state=0, n_jobs=-1)
print(pd.Series(p2.importances_mean, index=df2.columns).round(4).to_string())
print("\\nBoth copies now look UNIMPORTANT -- shuffling one leaves the other")
print("intact, so the model barely notices. Neither MDI nor permutation")
print("importance handles this; you need drop-column or grouped permutation.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(y=out.index, x=out["MDI (train)"], orientation="h",
            name="MDI (biased)", marker_color=C["danger"])
fig.add_bar(y=out.index, x=out["permutation (test)"], orientation="h",
            name="permutation (honest)", marker_color=C["success"])
fig.update_layout(height=400, barmode="group", xaxis_title="importance",
                  title="MDI vs permutation importance")
''',
        key="ch07_forest",
    )

    keypoints([
        "A random forest = bagging + a random feature subset at <b>every split</b>.",
        "That extra randomness raises each tree's bias but lowers $\\rho$ — and "
        "the ensemble wins.",
        "Extra-Trees also randomise the <b>threshold</b>: more bias, less "
        "variance, much faster.",
        "MDI feature importance is biased toward high-cardinality features and "
        "computed on training data.",
        "Use <b>permutation importance</b> on held-out data; beware that "
        "correlated features split the credit.",
    ])


# ==========================================================================
def s_7_4():
    section("7.4", "Boosting — AdaBoost")

    lead(
        "A different philosophy entirely. Bagging trains predictors "
        "<b>independently, in parallel</b>, and averages. Boosting trains them "
        "<b>sequentially</b>, each one correcting its predecessor's mistakes."
    )

    sub("The AdaBoost algorithm, step by step")

    md("Every instance carries a weight $w^{(i)}$, initialised to $1/m$. Then for "
       "each predictor $j = 1, \\dots, B$:")

    md("**1.** Train predictor $j$ on the weighted training set, and compute its "
       "weighted error rate:")
    math(r"""
    r_j \;=\;
    \frac{\displaystyle\sum_{\substack{i=1 \\ \hat y_j^{(i)} \ne y^{(i)}}}^{m} w^{(i)}}
         {\displaystyle\sum_{i=1}^{m} w^{(i)}}
    """)

    md("**2.** Compute the predictor's weight:")
    math(r"""
    \alpha_j \;=\; \eta \, \log\frac{1 - r_j}{r_j}
    """)
    where({r"\eta": "the learning rate (<code>learning_rate</code>, default 1)",
           r"\alpha_j": "large when $r_j$ is small; <b>zero</b> at $r_j = 0.5$ "
                        "(a random predictor gets no say); <b>negative</b> when "
                        "$r_j > 0.5$ (a systematically wrong predictor is inverted)"})

    md("**3.** Update the instance weights — boost the ones this predictor got "
       "wrong:")
    math(r"""
    w^{(i)} \;\leftarrow\;
    \begin{cases}
      w^{(i)} & \text{if } \hat y_j^{(i)} = y^{(i)}\\[4pt]
      w^{(i)} \exp\bigl(\alpha_j\bigr) & \text{if } \hat y_j^{(i)} \ne y^{(i)}
    \end{cases}
    """)
    md("then normalise so the weights sum to 1.")

    md("**Prediction** is a weighted vote:")
    math(r"""
    \hat y(\mathbf{x}) \;=\;
      \operatorname*{arg\,max}_{k}
      \sum_{\substack{j=1 \\ \hat y_j(\mathbf{x}) = k}}^{B} \alpha_j
    """)

    derive(
        [("AdaBoost looks like a heuristic, but it is <b>exactly</b> forward "
          "stagewise additive modelling under the <b>exponential loss</b>. That is "
          "Friedman, Hastie & Tibshirani's (2000) result, and it explains every "
          "formula above.", None),
         ("Define the ensemble score after $j$ rounds and the exponential loss:",
          r"F_j(\mathbf{x}) = \sum_{l=1}^{j}\alpha_l h_l(\mathbf{x}), \qquad "
          r"L\bigl(y, F\bigr) = \exp\bigl(-y\,F(\mathbf{x})\bigr), \quad y \in \{-1,+1\}"),
         ("At round $j$ we hold $F_{j-1}$ fixed and choose $(\\alpha, h)$ to "
          "minimise the total loss:",
          r"(\alpha_j, h_j) = \operatorname*{arg\,min}_{\alpha, h}"
          r"\sum_{i=1}^{m}\exp\Bigl(-y^{(i)}\bigl[F_{j-1}(\mathbf{x}^{(i)}) "
          r"+ \alpha h(\mathbf{x}^{(i)})\bigr]\Bigr)"),
         ("Factor out the part that does not depend on $(\\alpha, h)$ — <b>and that "
          "factor is precisely the instance weight</b>:",
          r"= \sum_{i=1}^{m} \underbrace{\exp\bigl(-y^{(i)}F_{j-1}(\mathbf{x}^{(i)})\bigr)}_{\textstyle w^{(i)}}"
          r"\exp\bigl(-\alpha\, y^{(i)} h(\mathbf{x}^{(i)})\bigr)"),
         ("Since $y h \\in \\{-1, +1\\}$, split the sum into correct and incorrect "
          "instances:",
          r"= e^{-\alpha}\!\!\sum_{y^{(i)} = h(\mathbf{x}^{(i)})}\!\! w^{(i)} "
          r"\;+\; e^{\alpha}\!\!\sum_{y^{(i)} \ne h(\mathbf{x}^{(i)})}\!\! w^{(i)} "
          r"= \bigl(e^{\alpha} - e^{-\alpha}\bigr) \sum_i w^{(i)}\mathbb{1}\bigl[y^{(i)} \ne h\bigr] "
          r"+ e^{-\alpha}\sum_i w^{(i)}"),
         ("For fixed $\\alpha > 0$ the minimiser $h_j$ is the one minimising the "
          "<b>weighted error rate</b> $r_j$ — which is exactly step 1.", None),
         ("Now differentiate with respect to $\\alpha$ and set to zero:",
          r"\frac{\partial}{\partial \alpha}\Bigl[(e^{\alpha}-e^{-\alpha})r "
          r"+ e^{-\alpha}\Bigr] = 0 \;\Longrightarrow\; "
          r"\alpha_j = \tfrac12 \log\frac{1-r_j}{r_j}"),
         ("which is step 2 (the factor $\\tfrac12$ is absorbed into the scaling "
          "convention). Finally the new weights are",
          r"w^{(i)}_{\text{new}} = \exp\bigl(-y^{(i)}F_j(\mathbf{x}^{(i)})\bigr) "
          r"= w^{(i)}\exp\bigl(-\alpha_j y^{(i)} h_j(\mathbf{x}^{(i)})\bigr)"),
         ("which multiplies by $e^{+\\alpha_j}$ exactly on the misclassified "
          "instances — step 3. <b>Every line of AdaBoost is forced by the "
          "exponential loss.</b>", None)],
        title="AdaBoost = forward stagewise fitting of the exponential loss",
    )

    warn(
        "Exponential loss is aggressive, and that is a weakness",
        "$e^{-yF}$ grows without bound as an instance becomes more badly "
        "misclassified, so a mislabelled instance gets an <b>exponentially</b> "
        "growing weight and eventually dominates training. AdaBoost is therefore "
        "notably sensitive to label noise and outliers. Gradient boosting with a "
        "<code>huber</code> or <code>log_loss</code> objective (§7.5) is far more "
        "robust, which is one reason it displaced AdaBoost in practice.",
    )

    anim_header("AdaBoost: watch the weights migrate to the hard cases")
    md(
        "Marker size is the instance weight. Each frame adds one stump. Watch the "
        "points near the boundary swell as successive stumps fail on them, and "
        "watch the combined boundary curve into shape."
    )

    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(7)
    Xa, ya = ds.moons(n=160, noise=.28)
    ta = 2 * ya - 1
    a1 = np.linspace(Xa[:, 0].min() - .5, Xa[:, 0].max() + .5, 110)
    a2 = np.linspace(Xa[:, 1].min() - .5, Xa[:, 1].max() + .5, 110)
    A1, A2 = np.meshgrid(a1, a2); AA = np.c_[A1.ravel(), A2.ravel()]

    w = np.full(len(Xa), 1 / len(Xa))
    F = np.zeros(len(AA)); Ftr = np.zeros(len(Xa))
    snaps = []
    for j in range(25):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0)
        stump.fit(Xa, ya, sample_weight=w)
        pred = 2 * stump.predict(Xa) - 1
        err = float(np.sum(w * (pred != ta)) / np.sum(w))
        err = min(max(err, 1e-10), 1 - 1e-10)
        al = 0.5 * np.log((1 - err) / err)
        F += al * (2 * stump.predict(AA) - 1)
        Ftr += al * pred
        snaps.append((w.copy(), F.copy(), err, al,
                      float(np.mean(np.sign(Ftr) == ta))))
        w = w * np.exp(-al * ta * pred)
        w /= w.sum()

    frames = []
    for j, (wj, Fj, err, al, acc) in enumerate(snaps):
        sz = 6 + 44 * (wj / wj.max())
        frames.append(go.Frame(name=str(j + 1), data=[
            go.Contour(x=a1, y=a2, z=Fj.reshape(A1.shape), showscale=False,
                       colorscale=nav.cscale(), opacity=.45, ncontours=20),
            go.Contour(x=a1, y=a2, z=Fj.reshape(A1.shape), showscale=False,
                       contours=dict(start=0, end=0, size=1, coloring="lines"),
                       line=dict(width=3.5),
                       colorscale=[[0, C["ink"]], [1, C["ink"]]]),
            go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                       marker=dict(color=C["train"], size=sz[ya == 0],
                                   line=dict(color="#fff", width=.8))),
            go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                       marker=dict(color=C["warning"], size=sz[ya == 1],
                                   line=dict(color="#fff", width=.8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"stump {j+1}/25   ·   weighted error r = {err:.4f}   ·   "
            f"α = {al:+.4f}   ·   ensemble train accuracy = {acc:.4f}")])))

    w0, F0 = snaps[0][0], snaps[0][1]
    sz0 = 6 + 44 * (w0 / w0.max())
    f = go.Figure(data=[
        go.Contour(x=a1, y=a2, z=F0.reshape(A1.shape), showscale=False,
                   colorscale=nav.cscale(), opacity=.45, ncontours=20),
        go.Contour(x=a1, y=a2, z=F0.reshape(A1.shape), showscale=False,
                   contours=dict(start=0, end=0, size=1, coloring="lines"),
                   line=dict(width=3.5), colorscale=[[0, C["ink"]], [1, C["ink"]]]),
        go.Scatter(x=Xa[ya == 0, 0], y=Xa[ya == 0, 1], mode="markers",
                   name="class 0", marker=dict(color=C["train"], size=sz0[ya == 0],
                                               line=dict(color="#fff", width=.8))),
        go.Scatter(x=Xa[ya == 1, 0], y=Xa[ya == 1, 1], mode="markers",
                   name="class 1", marker=dict(color=C["warning"],
                                               size=sz0[ya == 1],
                                               line=dict(color="#fff", width=.8))),
    ])
    f.update_layout(height=520, xaxis_title="x₁", yaxis_title="x₂",
                    title="AdaBoost with depth-1 stumps — marker size = weight")
    anim.animate(f, frames, duration=nav.anim_ms(400), slider_prefix="stump ")
    figure(f)

    pitfall(
        "Boosting cannot be parallelised",
        "Each predictor needs its predecessor's results to compute the weights. "
        "Bagging and random forests train every tree simultaneously "
        "(<code>n_jobs=-1</code> gives near-linear speed-up); boosting is "
        "inherently sequential. This is <i>the</i> practical downside, and it is "
        "why histogram boosting (§7.6) works so hard to make each round fast.",
    )

    code_lab(
        "AdaBoost from scratch, verified against scikit-learn",
        '''import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=.3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)
t_tr, t_te = 2*ytr - 1, 2*yte - 1          # labels in {-1, +1}

# ============ AdaBoost (SAMME) from scratch ============================
def adaboost(X, t, n_estimators=200, learning_rate=1.0, max_depth=1):
    m = len(X)
    w = np.full(m, 1/m)
    models, alphas, hist = [], [], []
    F = np.zeros(m)
    for j in range(n_estimators):
        stump = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        stump.fit(X, (t > 0).astype(int), sample_weight=w)
        pred = 2*stump.predict(X) - 1
        r = float(np.sum(w * (pred != t)) / np.sum(w))
        r = min(max(r, 1e-10), 1 - 1e-10)
        alpha = learning_rate * 0.5 * np.log((1 - r) / r)
        w = w * np.exp(-alpha * t * pred)
        w /= w.sum()
        F += alpha * pred
        models.append(stump); alphas.append(alpha)
        hist.append((r, alpha, float(np.mean(np.sign(F) == t)),
                     float(np.mean(np.exp(-t * F)))))
    return models, alphas, hist

models, alphas, hist = adaboost(Xtr, t_tr, n_estimators=200)

def predict(X, models, alphas):
    F = sum(a * (2*mo.predict(X) - 1) for mo, a in zip(models, alphas))
    return (F > 0).astype(int), F

print(f"{'round':>7}{'weighted err r':>16}{'alpha':>10}{'train acc':>12}"
      f"{'exp loss':>12}")
for j in [0, 1, 4, 9, 24, 49, 99, 199]:
    r, a, acc, el = hist[j]
    print(f"{j+1:>7}{r:>16.5f}{a:>10.4f}{acc:>12.4f}{el:>12.5f}")

p_tr, _ = predict(Xtr, models, alphas)
p_te, _ = predict(Xte, models, alphas)
print(f"\\nmine    : train {np.mean(p_tr==ytr):.4f}   test {np.mean(p_te==yte):.4f}")

sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                        n_estimators=200, random_state=42).fit(Xtr, ytr)
print(f"sklearn : train {sk.score(Xtr,ytr):.4f}   test {sk.score(Xte,yte):.4f}")

# ============ the exponential loss keeps falling =======================
print("\\n=== training error hits zero, but the MARGIN keeps improving ===")
_, F_tr = predict(Xtr, models, alphas)
print("This is why AdaBoost often keeps improving on TEST data even after")
print("training error reaches 0 -- it is still pushing the margin out.")
print(f"  train accuracy at round 200 = {np.mean(np.sign(F_tr)==t_tr):.4f}")
print(f"  minimum margin              = {np.min(t_tr*F_tr/np.sum(alphas)):+.4f}")
print(f"  mean margin                 = {np.mean(t_tr*F_tr/np.sum(alphas)):+.4f}")

# ============ SENSITIVITY TO LABEL NOISE ===============================
print("\\n=== AdaBoost vs label noise (exponential loss is unbounded) ===")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
print(f"{'noise':>7}{'AdaBoost':>11}{'GradBoost':>12}{'RandomForest':>15}")
for frac in [0.0, 0.05, 0.15, 0.30]:
    rng = np.random.default_rng(0)
    yn = ytr.copy()
    flip = rng.choice(len(yn), int(frac*len(yn)), replace=False)
    yn[flip] = 1 - yn[flip]
    ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=0).fit(Xtr, yn)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=0).fit(Xtr, yn)
    rf = RandomForestClassifier(n_estimators=200, random_state=0,
                                n_jobs=-1).fit(Xtr, yn)
    print(f"{frac:>6.0%}{ab.score(Xte,yte):>11.4f}{gb.score(Xte,yte):>12.4f}"
          f"{rf.score(Xte,yte):>15.4f}")
print("AdaBoost degrades fastest -- exp(-yF) makes mislabelled points dominate.")

# ============ learning rate vs number of estimators ====================
print("\\n=== the learning_rate / n_estimators trade-off ===")
print(f"{'lr':>7}" + "".join(f"{f'B={b}':>10}" for b in [10, 50, 200, 500]))
for lr in [0.05, 0.2, 0.5, 1.0]:
    row = ""
    for B in [10, 50, 200, 500]:
        m_ = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                n_estimators=B, learning_rate=lr,
                                random_state=0).fit(Xtr, ytr)
        row += f"{m_.score(Xte, yte):>10.4f}"
    print(f"{lr:>7}{row}")
print("Low learning rate needs more estimators -- they trade off directly.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=[h[2] for h in hist], mode="lines", name="train accuracy",
                line=dict(color=C["train"], width=2.5))
fig.add_scatter(y=[h[3] for h in hist], mode="lines", name="exponential loss",
                line=dict(color=C["danger"], width=2.5), yaxis="y2")
fig.update_layout(height=400, xaxis_title="boosting round",
                  yaxis=dict(title="train accuracy"),
                  yaxis2=dict(title="exp loss", overlaying="y", side="right",
                              type="log"),
                  title="AdaBoost: accuracy saturates, the loss keeps falling")
''',
        key="ch07_adaboost",
    )

    keypoints([
        "Boosting is <b>sequential</b>: each predictor fixes the previous one's "
        "errors.",
        "AdaBoost reweights <b>instances</b>: $w \\leftarrow w e^{\\alpha_j}$ on "
        "the mistakes.",
        "$\\alpha_j = \\eta\\log\\frac{1-r_j}{r_j}$ — zero at $r=0.5$, negative "
        "above it.",
        "It is exactly forward stagewise fitting of the <b>exponential loss</b> — "
        "every formula is derived, not invented.",
        "Exponential loss is unbounded ⇒ sensitive to label noise; and boosting "
        "cannot be parallelised.",
    ])


# ==========================================================================
def s_7_5():
    section("7.5", "Gradient Boosting")

    lead(
        "Same sequential idea, different mechanism. Instead of reweighting the "
        "instances, fit each new predictor to the <b>residual errors</b> of the "
        "ensemble so far."
    )

    sub("The simplest case: least-squares boosting")

    md("For squared loss the algorithm is three lines:")

    math(r"""
    F_0(\mathbf{x}) = \bar y,
    \qquad
    r_j^{(i)} = y^{(i)} - F_{j-1}\bigl(\mathbf{x}^{(i)}\bigr),
    \qquad
    F_j(\mathbf{x}) = F_{j-1}(\mathbf{x}) + \eta\, h_j(\mathbf{x})
    """)
    where({r"h_j": "a regression tree fitted to the residuals $r_j$",
           r"\eta": "the <b>learning rate</b> (<code>learning_rate</code>), which "
                    "shrinks each tree's contribution"})

    sub("Why it is called *gradient* boosting")

    derive(
        [("The residual is not an arbitrary choice — it is the negative gradient "
          "of the squared loss with respect to the current prediction.", None),
         ("Take $L(y, F) = \\tfrac12 (y - F)^2$ and differentiate with respect to "
          "$F$, treating $F(\\mathbf{x}^{(i)})$ as a free parameter:",
          r"-\frac{\partial L\bigl(y^{(i)}, F(\mathbf{x}^{(i)})\bigr)}"
          r"{\partial F(\mathbf{x}^{(i)})} = y^{(i)} - F\bigl(\mathbf{x}^{(i)}\bigr) "
          r"= r^{(i)}"),
         ("So fitting a tree to the residuals is fitting a tree to the negative "
          "gradient — <b>gradient descent in function space</b>. Each boosting "
          "round takes one step downhill, and the tree is the (approximate) step "
          "direction.", None),
         ("This immediately generalises to <b>any</b> differentiable loss. Define "
          "the <i>pseudo-residuals</i>:",
          r"r_j^{(i)} = -\left[\frac{\partial L\bigl(y^{(i)}, F(\mathbf{x}^{(i)})\bigr)}"
          r"{\partial F(\mathbf{x}^{(i)})}\right]_{F = F_{j-1}}"),
         ("Fit $h_j$ to those, then choose the leaf values by a line search "
          "(a Newton step in modern implementations):",
          r"\gamma_{j\ell} = \operatorname*{arg\,min}_{\gamma} "
          r"\sum_{\mathbf{x}^{(i)} \in R_{j\ell}} "
          r"L\bigl(y^{(i)},\, F_{j-1}(\mathbf{x}^{(i)}) + \gamma\bigr)"),
         ("Substituting different $L$ gives every gradient-boosting variant:",
          None),
         ("• $L = \\tfrac12(y-F)^2$ ⇒ residual $y - F$ &nbsp;→&nbsp; "
          "<b>least-squares boosting</b><br>"
          "• $L = |y - F|$ ⇒ pseudo-residual $\\mathrm{sign}(y-F)$ &nbsp;→&nbsp; "
          "<b>robust to outliers</b><br>"
          "• $L = $ Huber ⇒ a blend of the two<br>"
          "• $L = \\log(1 + e^{-yF})$ ⇒ pseudo-residual $y - \\sigma(F)$ "
          "&nbsp;→&nbsp; <b>binary classification</b><br>"
          "• $L = $ pinball ⇒ <b>quantile regression</b>", None)],
        title="Residuals are the negative gradient — the whole generalisation",
    )

    idea(
        "Shrinkage: the single most important hyperparameter",
        "The learning rate $\\eta$ scales every tree's contribution. Small $\\eta$ "
        "(0.01–0.1) needs many more trees but generalises noticeably better — "
        "this is called <b>shrinkage</b>, and it acts as regularisation for "
        "exactly the same reason early stopping does (§4.5). The practical rule: "
        "<b>set $\\eta$ low, set <code>n_estimators</code> high, and let early "
        "stopping choose where to cut.</b>",
    )

    anim_header("Gradient boosting building a fit from residuals")
    md(
        "Top: the ensemble's prediction. Bottom: the residuals it is currently "
        "trying to explain, and the tree fitted to them. Each round, the residuals "
        "shrink toward zero."
    )

    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(42)
    Xg = np.sort(rng.uniform(-3, 3, 120)).reshape(-1, 1)
    yg = np.sin(1.5 * Xg[:, 0]) + .35 * Xg[:, 0] + rng.normal(0, .25, 120)
    gg = np.linspace(-3.1, 3.1, 300).reshape(-1, 1)

    eta = 0.35
    Fg = np.full(len(Xg), yg.mean())
    Fgrid = np.full(len(gg), yg.mean())
    steps = []
    for j in range(30):
        res = yg - Fg
        t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(Xg, res)
        steps.append((Fgrid.copy(), res.copy(), t.predict(gg),
                      float(np.sqrt(np.mean(res ** 2)))))
        Fg = Fg + eta * t.predict(Xg)
        Fgrid = Fgrid + eta * t.predict(gg)
    steps.append((Fgrid.copy(), yg - Fg, np.zeros(len(gg)),
                  float(np.sqrt(np.mean((yg - Fg) ** 2)))))

    frames = []
    for j, (Fk, res, tree_pred, rmse) in enumerate(steps):
        frames.append(go.Frame(name=str(j), data=[
            go.Scatter(x=Xg[:, 0], y=yg, mode="markers",
                       marker=dict(color=C["train"], size=6,
                                   line=dict(color="#fff", width=.7))),
            go.Scatter(x=gg[:, 0], y=Fk, mode="lines",
                       line=dict(color=C["primary"], width=3.4)),
            go.Scatter(x=Xg[:, 0], y=res, mode="markers",
                       marker=dict(color=C["danger"], size=6,
                                   line=dict(color="#fff", width=.7))),
            go.Scatter(x=gg[:, 0], y=tree_pred, mode="lines",
                       line=dict(color=C["success"], width=3, shape="hv")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"round {j}/30   ·   residual RMSE = {rmse:.4f}   ·   η = {eta}")])))

    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.58, .42],
                      vertical_spacing=.08,
                      subplot_titles=("the ensemble F(x) so far",
                                      "residuals (red) and the next tree (green)"))
    f.add_trace(go.Scatter(x=Xg[:, 0], y=yg, mode="markers", name="data",
                           marker=dict(color=C["train"], size=6,
                                       line=dict(color="#fff", width=.7))), 1, 1)
    f.add_trace(go.Scatter(x=gg[:, 0], y=steps[0][0], mode="lines", name="F(x)",
                           line=dict(color=C["primary"], width=3.4)), 1, 1)
    f.add_trace(go.Scatter(x=Xg[:, 0], y=steps[0][1], mode="markers",
                           name="residuals", marker=dict(color=C["danger"], size=6,
                           line=dict(color="#fff", width=.7))), 2, 1)
    f.add_trace(go.Scatter(x=gg[:, 0], y=steps[0][2], mode="lines",
                           name="tree fitted to residuals",
                           line=dict(color=C["success"], width=3, shape="hv")), 2, 1)
    f.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)
    f.update_yaxes(range=[-2.6, 2.6], row=1, col=1)
    f.update_yaxes(range=[-1.6, 1.6], row=2, col=1)
    f.update_xaxes(range=[-3.1, 3.1], title_text="x", row=2, col=1)
    f.update_layout(height=560, title="Gradient boosting = gradient descent in "
                                      "function space")
    anim.animate(f, frames, duration=nav.anim_ms(320), slider_prefix="round ")
    figure(f)

    sub("Regularisation for gradient boosting")

    table(
        ["Hyperparameter", "Effect", "Typical"],
        [["<code>learning_rate</code> $\\eta$", "Shrinkage — the main dial",
          "0.01–0.1"],
         ["<code>n_estimators</code>", "Number of rounds; trades off with $\\eta$",
          "100–3000, chosen by early stopping"],
         ["<code>max_depth</code>", "Complexity of each tree; controls interaction "
          "order", "2–6 (shallow!)"],
         ["<code>subsample</code>", "Fraction of rows per tree — <b>stochastic "
          "gradient boosting</b>", "0.5–1.0"],
         ["<code>max_features</code>", "Features per split", "'sqrt', or 0.5–1.0"],
         ["<code>min_samples_leaf</code>", "As in Chapter 6", "1–20"],
         ["<code>n_iter_no_change</code> + <code>validation_fraction</code>",
          "Built-in early stopping", "10 / 0.1"]],
    )

    note(
        "Boosting wants <b>shallow</b> trees; bagging wants <b>deep</b> ones",
        "This is not arbitrary. Bagging reduces <i>variance</i>, so it needs "
        "low-bias (deep) base learners. Boosting reduces <i>bias</i> by adding "
        "corrections, so it needs high-bias, low-variance (shallow) base learners "
        "— otherwise the first tree fits everything and there is nothing left to "
        "boost. A tree of depth $d$ can capture interactions of order $d$, so "
        "<code>max_depth</code> in a GBM is effectively \"how many features may "
        "interact\".",
    )

    code_lab(
        "Gradient boosting from scratch, with early stopping and other losses",
        '''import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(-3, 3, 500)).reshape(-1, 1)
y = np.sin(1.5*X[:, 0]) + .35*X[:, 0] + rng.normal(0, .25, 500)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)

# ============ 1. from scratch: residual fitting ========================
def gbrt_fit(X, y, n_estimators=200, lr=.1, max_depth=2):
    F0 = y.mean()
    F = np.full(len(y), F0)
    trees = []
    for _ in range(n_estimators):
        residual = y - F                       # = negative gradient of 0.5(y-F)^2
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        t.fit(X, residual)
        F += lr * t.predict(X)
        trees.append(t)
    return F0, trees, lr

def gbrt_predict(X, F0, trees, lr, n_use=None):
    F = np.full(len(X), F0)
    for t in trees[:n_use]:
        F += lr * t.predict(X)
    return F

F0, trees, lr = gbrt_fit(Xtr, ytr, n_estimators=200, lr=.1)
print(f"mine    : test RMSE = "
      f"{mean_squared_error(yte, gbrt_predict(Xte, F0, trees, lr))**.5:.5f}")
sk = GradientBoostingRegressor(n_estimators=200, learning_rate=.1, max_depth=2,
                               random_state=0).fit(Xtr, ytr)
print(f"sklearn : test RMSE = {mean_squared_error(yte, sk.predict(Xte))**.5:.5f}")

# ============ 2. residuals ARE the negative gradient ===================
print("\\n=== verify: residual == -dL/dF for squared loss ===")
F = np.full(len(ytr), ytr.mean())
eps = 1e-6
i = 0
L  = lambda f: .5*(ytr[i] - f)**2
num_grad = (L(F[i]+eps) - L(F[i]-eps)) / (2*eps)
print(f"  numerical -dL/dF = {-num_grad:+.8f}")
print(f"  residual  y - F  = {ytr[i] - F[i]:+.8f}")

# ============ 3. the learning-rate / n_estimators trade-off ============
print("\\n=== shrinkage ===")
print(f"{'lr':>7}" + "".join(f"{f'B={b}':>11}" for b in [10, 50, 200, 1000]))
for lr_ in [0.5, 0.2, 0.05, 0.01]:
    row = ""
    for B in [10, 50, 200, 1000]:
        m_ = GradientBoostingRegressor(n_estimators=B, learning_rate=lr_,
                                       max_depth=2, random_state=0).fit(Xtr, ytr)
        row += f"{mean_squared_error(yte, m_.predict(Xte))**.5:>11.5f}"
    print(f"{lr_:>7}{row}")
print("Low lr + many trees is the best cell -- that is shrinkage regularisation.")

# ============ 4. early stopping via staged_predict =====================
print("\\n=== finding the optimal number of trees ===")
big = GradientBoostingRegressor(n_estimators=800, learning_rate=.05, max_depth=2,
                                random_state=0).fit(Xtr, ytr)
errors = [mean_squared_error(yte, p) for p in big.staged_predict(Xte)]
best_n = int(np.argmin(errors)) + 1
print(f"800 trees -> test MSE {errors[-1]:.6f}")
print(f"best at {best_n} trees -> test MSE {errors[best_n-1]:.6f}")
print(f"the last {800-best_n} trees actively HURT by {errors[-1]-errors[best_n-1]:+.6f}")

# built-in early stopping
es = GradientBoostingRegressor(n_estimators=800, learning_rate=.05, max_depth=2,
                               n_iter_no_change=20, validation_fraction=.15,
                               random_state=0).fit(Xtr, ytr)
print(f"\\nn_iter_no_change stopped automatically at {es.n_estimators_} trees, "
      f"test RMSE {mean_squared_error(yte, es.predict(Xte))**.5:.5f}")

# ============ 5. different losses = different pseudo-residuals =========
print("\\n=== robustness to outliers via the loss function ===")
y_out = ytr.copy()
bad = rng.choice(len(y_out), 15, replace=False)
y_out[bad] += rng.normal(0, 8, 15)
print(f"{'loss':<26}{'clean data':>13}{'with outliers':>15}")
for loss in ["squared_error", "absolute_error", "huber"]:
    a = GradientBoostingRegressor(loss=loss, n_estimators=300, learning_rate=.05,
                                  max_depth=2, random_state=0).fit(Xtr, ytr)
    b = GradientBoostingRegressor(loss=loss, n_estimators=300, learning_rate=.05,
                                  max_depth=2, random_state=0).fit(Xtr, y_out)
    print(f"{loss:<26}{mean_squared_error(yte, a.predict(Xte))**.5:>13.5f}"
          f"{mean_squared_error(yte, b.predict(Xte))**.5:>15.5f}")
print("huber and absolute_error barely notice the outliers. squared_error does.")

# ============ 6. quantile regression for free ==========================
print("\\n=== prediction intervals via the pinball loss ===")
lo = GradientBoostingRegressor(loss="quantile", alpha=.05, n_estimators=300,
                               max_depth=2, random_state=0).fit(Xtr, ytr)
hi = GradientBoostingRegressor(loss="quantile", alpha=.95, n_estimators=300,
                               max_depth=2, random_state=0).fit(Xtr, ytr)
cover = np.mean((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte)))
print(f"nominal 90 % interval -> empirical coverage {cover:.1%}")

import plotly.graph_objects as go
g = np.linspace(-3, 3, 300).reshape(-1, 1)
fig = go.Figure()
fig.add_scatter(x=Xte[:, 0], y=yte, mode="markers", name="test data",
                marker=dict(color=C["train"], size=5, opacity=.6))
fig.add_scatter(x=np.r_[g[:, 0], g[::-1, 0]],
                y=np.r_[lo.predict(g), hi.predict(g)[::-1]],
                fill="toself", fillcolor="rgba(108,77,246,.15)",
                line=dict(width=0), name="90 % interval")
fig.add_scatter(x=g[:, 0], y=sk.predict(g), mode="lines", name="median fit",
                line=dict(color=C["primary"], width=3))
fig.update_layout(height=420, xaxis_title="x", yaxis_title="y",
                  title="Gradient boosting with quantile losses")

fig2 = go.Figure(go.Scatter(y=errors, mode="lines",
                            line=dict(color=C["test"], width=2)))
fig2.add_vline(x=best_n, line_dash="dash", line_color=C["success"],
               annotation_text=f"optimal = {best_n} trees")
fig2.update_layout(height=360, xaxis_title="number of trees",
                   yaxis_title="test MSE", title="Early stopping curve")
''',
        key="ch07_gbrt",
    )

    keypoints([
        "Gradient boosting fits each tree to the <b>residuals</b> — which are the "
        "negative gradient of squared loss.",
        "Generalises to any differentiable loss via <b>pseudo-residuals</b>: "
        "Huber, log-loss, pinball…",
        "<b>Shrinkage</b> ($\\eta$ small, many trees) is the main regulariser; use "
        "early stopping to pick $B$.",
        "Boosting wants <b>shallow</b> trees (depth 2–6); bagging wants deep ones.",
        "<code>subsample &lt; 1</code> gives stochastic gradient boosting: faster "
        "and often better.",
    ])


# ==========================================================================
def s_7_6():
    section("7.6", "Histogram-Based Gradient Boosting")

    lead(
        "The modern implementation. Bin the continuous features into at most 255 "
        "integer buckets, and the split search collapses from "
        "$\\mathcal{O}(m\\log m)$ to $\\mathcal{O}(m + \\text{bins})$. This is what "
        "LightGBM, XGBoost's `hist` mode, and scikit-learn's "
        "`HistGradientBoosting*` all do."
    )

    sub("Why binning is such a large win")

    derive(
        [("<b>Classic exact split search.</b> At every node, for every feature, "
          "sort the values and sweep the threshold:",
          r"\mathcal{O}\bigl(n \cdot m \log m\bigr) \text{ per node}"),
         ("<b>Histogram search.</b> Bin once, up front, at cost "
          "$\\mathcal{O}(nm\\log m)$ <i>total</i>. Then at each node, accumulate "
          "gradient statistics into $K$ bins in one pass and scan the bins:",
          r"\mathcal{O}\bigl(n \cdot (m + K)\bigr) \text{ per node}, \qquad K \le 255"),
         ("Since $K \\ll m$, the $\\log m$ factor disappears entirely and the "
          "constant is tiny — the inner loop is integer indexing into a small "
          "array, which is cache-friendly and vectorises.", None),
         ("<b>The histogram subtraction trick</b> doubles the saving again. Once "
          "you have built the histogram for a node and for one of its children, "
          "the other child comes free:",
          r"H_{\text{sibling}} = H_{\text{parent}} - H_{\text{child}}"),
         ("So you only ever build the histogram for the <i>smaller</i> child. In "
          "practice, histogram boosting is 10–100× faster than exact boosting on "
          "large datasets, and the binning costs a negligible amount of accuracy "
          "(the loss of threshold precision acts as mild regularisation).", None)],
        title="The arithmetic behind the 10–100× speed-up",
    )

    sub("What else you get")

    table(
        ["Feature", "<code>GradientBoosting*</code>", "<code>HistGradientBoosting*</code>"],
        [["Split search", "Exact", "Binned (≤ 255 bins)"],
         ["Complexity per node", "$\\mathcal{O}(nm\\log m)$",
          "$\\mathcal{O}(n(m + K))$"],
         ["Native missing values", "❌ must impute", "<b>✅ learned direction</b>"],
         ["Native categorical features", "❌", "<b>✅ <code>categorical_features</code></b>"],
         ["Early stopping", "Manual or <code>n_iter_no_change</code>",
          "<b>On by default</b> when $m > 10\\,000$"],
         ["Parallelism", "Limited", "Multi-threaded histogram building"],
         ["Key size parameter", "<code>max_depth</code>",
          "<code>max_leaf_nodes</code> (leaf-wise growth)"],
         ["Best for", "$m < 10^4$", "$m > 10^4$"]],
    )

    idea(
        "Missing values become a learned decision",
        "Histogram boosting puts missing values in their own bin and, at each "
        "split, <b>learns whether they should go left or right</b> by trying both "
        "and keeping the better. That is strictly more expressive than imputing a "
        "median — and it means the model can exploit informative missingness "
        "(§2.4) automatically.",
    )

    anim_header("Binning a feature: how much precision do you actually lose?")

    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  GradientBoostingRegressor)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import time as _time

    rng = np.random.default_rng(1)
    xs = rng.normal(0, 1, 4000)
    binss = [4, 8, 16, 32, 64, 128, 255]
    frames = []
    for K in binss:
        edges = np.quantile(xs, np.linspace(0, 1, K + 1))
        edges = np.unique(edges)
        binned = edges[np.clip(np.searchsorted(edges, xs) - 1, 0, len(edges) - 2)]
        err = float(np.mean(np.abs(binned - xs)))
        frames.append(go.Frame(name=str(K), data=[
            go.Histogram(x=xs, nbinsx=120, marker=dict(color=alpha(C["train"], .5))),
            go.Histogram(x=binned, nbinsx=120,
                         marker=dict(color=alpha(C["danger"], .7))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{K} quantile bins   ·   mean |binned − original| = {err:.5f}   ·   "
            f"{len(np.unique(binned))} distinct values remain")])))

    f = go.Figure(data=[
        go.Histogram(x=xs, nbinsx=120, name="original continuous feature",
                     marker=dict(color=alpha(C["train"], .5))),
        go.Histogram(x=np.zeros(1), nbinsx=120, name="after binning",
                     marker=dict(color=alpha(C["danger"], .7))),
    ])
    f.update_layout(height=420, barmode="overlay", bargap=.02,
                    xaxis_title="feature value", yaxis_title="count",
                    title="Quantile binning: 4 → 255 bins")
    anim.animate(f, frames, duration=nav.anim_ms(800), slider_prefix="bins = ")
    figure(f, "By 64 bins the histogram is visually indistinguishable. That is "
              "why the accuracy cost of binning is negligible.")

    codenote(
        "The wider ecosystem",
        "<b>XGBoost</b> (Chen & Guestrin, 2016) adds a second-order Newton step "
        "and an explicit $\\ell_1/\\ell_2$ penalty on the leaf values. "
        "<b>LightGBM</b> (Ke et al., 2017) introduced leaf-wise growth plus GOSS "
        "and EFB. <b>CatBoost</b> (Prokhorenkova et al., 2018) uses ordered "
        "boosting to eliminate target leakage in categorical encoding. "
        "scikit-learn's <code>HistGradientBoosting*</code> is a clean "
        "implementation of the same core ideas with no extra dependency, and it "
        "is usually within a percent of the others.",
    )

    code_lab(
        "Histogram boosting: speed, missing values, and categoricals",
        '''import numpy as np, pandas as pd, time
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============ 1. the speed difference ==================================
print("=== exact vs histogram split search ===")
print(f"{'m':>8}{'GradientBoosting':>20}{'HistGradientBoosting':>23}{'speedup':>10}")
for m in [2_000, 10_000, 30_000]:
    X, y = make_regression(n_samples=m, n_features=25, noise=12, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0)
    t0 = time.perf_counter()
    g = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                  random_state=0).fit(Xtr, ytr)
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    h = HistGradientBoostingRegressor(max_iter=100, max_depth=4,
                                      early_stopping=False,
                                      random_state=0).fit(Xtr, ytr)
    t_hist = time.perf_counter() - t0
    print(f"{m:>8}{t_exact:>19.3f}s{t_hist:>22.3f}s{t_exact/t_hist:>9.1f}x")
    if m == 30_000:
        print(f"{'':>8}{'RMSE ' + f'{mean_squared_error(yte, g.predict(Xte))**.5:.3f}':>20}"
              f"{'RMSE ' + f'{mean_squared_error(yte, h.predict(Xte))**.5:.3f}':>23}"
              f"{'<- same accuracy':>10}")

# ============ 2. NATIVE missing-value handling =========================
print("\\n=== missing values, handled natively ===")
rng = np.random.default_rng(0)
n = 6000
Xm = rng.normal(0, 1, (n, 6))
ym = Xm[:, 0]*2 + Xm[:, 1] - Xm[:, 2]*1.5 + rng.normal(0, .5, n)
# make missingness INFORMATIVE: values are missing when x3 is large
mask = Xm[:, 3] > 0.8
Xm[mask, 0] = np.nan
ym[mask] += 3.0                      # the missingness itself carries signal
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=.3, random_state=0)

h = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
print(f"HistGradientBoosting with raw NaNs : RMSE "
      f"{mean_squared_error(yte, h.predict(Xte))**.5:.4f}")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
imp = make_pipeline(SimpleImputer(strategy="median"),
                    GradientBoostingRegressor(random_state=0)).fit(Xtr, ytr)
print(f"median imputation + GradientBoosting: RMSE "
      f"{mean_squared_error(yte, imp.predict(Xte))**.5:.4f}")
imp2 = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                     GradientBoostingRegressor(random_state=0)).fit(Xtr, ytr)
print(f"  + add_indicator=True               : RMSE "
      f"{mean_squared_error(yte, imp2.predict(Xte))**.5:.4f}")
print("\\nThe native handler LEARNS which way NaNs should go at every split.")

# ============ 3. NATIVE categorical features ===========================
print("\\n=== categorical features, no one-hot needed ===")
n = 8000
cat = rng.integers(0, 40, n)                 # 40 categories
effect = rng.normal(0, 2, 40)
Xc = pd.DataFrame({"num1": rng.normal(0, 1, n),
                   "num2": rng.normal(0, 1, n),
                   "cat":  cat})
yc = (Xc["num1"] + effect[cat] + rng.normal(0, .5, n) > 1).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=.3, random_state=0)

t0 = time.perf_counter()
hc = HistGradientBoostingClassifier(categorical_features=["cat"],
                                    random_state=0).fit(Xtr, ytr)
t_native = time.perf_counter() - t0

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
t0 = time.perf_counter()
oh = make_pipeline(
    ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False), ["cat"])],
                      remainder="passthrough"),
    HistGradientBoostingClassifier(random_state=0)).fit(Xtr, ytr)
t_onehot = time.perf_counter() - t0

print(f"native categorical : accuracy {hc.score(Xte, yte):.4f}  "
      f"({t_native:.2f}s, 3 columns)")
print(f"one-hot encoded    : accuracy {oh.score(Xte, yte):.4f}  "
      f"({t_onehot:.2f}s, 42 columns)")
print("Native handling groups categories optimally instead of splitting one at a time.")

# ============ 4. early stopping is ON by default =======================
print("\\n=== built-in early stopping ===")
X, y = make_regression(n_samples=30_000, n_features=20, noise=15, random_state=0)
h = HistGradientBoostingRegressor(max_iter=1000, random_state=0).fit(X, y)
print(f"max_iter=1000 but it stopped at n_iter_ = {h.n_iter_}")
print(f"(early stopping activates automatically when m > 10,000)")
''',
        key="ch07_hist",
    )

    keypoints([
        "Bin features into ≤255 buckets: split search drops from "
        "$\\mathcal{O}(nm\\log m)$ to $\\mathcal{O}(n(m+K))$ per node.",
        "The <b>histogram subtraction trick</b> gives one child's histogram free.",
        "Native <b>missing-value</b> handling: the direction is learned, not "
        "imputed.",
        "Native <b>categorical</b> features: no one-hot explosion.",
        "Use <code>HistGradientBoosting*</code> above ~10 000 rows; it is the "
        "default choice for tabular data today.",
    ])


# ==========================================================================
def s_7_7():
    section("7.7", "Stacking")

    lead(
        "Instead of a fixed aggregation rule (majority vote, average), **train a "
        "model to do the aggregating**. The aggregator is called a blender or "
        "meta-learner."
    )

    sub("How stacking is trained")

    md(
        """
The critical detail is avoiding leakage. If you train the base models on the
training set and then train the blender on their predictions **for that same
set**, the blender sees predictions that are optimistically good, and it learns
to trust models that have merely memorised. The fix is out-of-fold predictions:

1. Split the training set into $k$ folds.
2. For each fold, train every base model on the other $k-1$ folds and predict
   the held-out fold. Stack these **out-of-fold** predictions into a new matrix
   $\\mathbf{Z} \\in \\mathbb{R}^{m \\times B}$ (or $m \\times BK$ for
   probabilities over $K$ classes).
3. Train the blender on $(\\mathbf{Z}, \\mathbf{y})$.
4. Refit every base model on the **full** training set for use at prediction
   time.
        """
    )

    math(r"""
    z_{i,b} \;=\; h_b^{(-\kappa(i))}\bigl(\mathbf{x}^{(i)}\bigr),
    \qquad
    \hat y \;=\; g\bigl(z_{\cdot,1}, \dots, z_{\cdot,B}\bigr)
    """)
    where({r"\kappa(i)": "the fold containing instance $i$",
           r"h_b^{(-\kappa)}": "base model $b$ trained without fold $\\kappa$",
           r"g": "the blender / meta-learner"})

    tip(
        "Keep the blender simple",
        "The blender's job is to learn <i>how much to trust each base model, and "
        "when</i>. That is a low-complexity task on a $m \\times B$ matrix, so a "
        "<b>regularised linear model</b> (ridge, or logistic regression) is almost "
        "always the right choice. A complex blender overfits the out-of-fold "
        "predictions and throws away the gain. Setting "
        "<code>passthrough=True</code> also gives the blender the original "
        "features, which sometimes helps and often overfits.",
    )

    anim_header("Stacking assembled step by step")

    stages = [
        ("1 · The training set",
         "m rows, n features, one target column"),
        ("2 · Split into k folds",
         "each fold will be held out exactly once"),
        ("3 · Out-of-fold predictions",
         "each base model predicts the fold it did not see"),
        ("4 · The blending matrix Z",
         "m rows × B columns — one column per base model"),
        ("5 · Train the blender on (Z, y)",
         "a simple regularised model learns whom to trust"),
        ("6 · Refit the base models on ALL the data",
         "ready to predict new instances"),
    ]

    def board(stage):
        traces = []
        m_rows, B = 12, 3
        # the data block
        for r in range(m_rows):
            fold = r // 3
            col = (alpha(SEQ[fold], .6) if stage >= 1 else alpha(C["train"], .5))
            if stage >= 2:
                col = alpha(SEQ[fold], .75)
            traces.append(go.Scatter(
                x=[0, 1.6, 1.6, 0, 0], y=[r, r, r + .82, r + .82, r],
                fill="toself", fillcolor=col, line=dict(color="#fff", width=1),
                hoverinfo="skip", showlegend=False))
        # the Z matrix
        if stage >= 3:
            for r in range(m_rows):
                for b in range(B):
                    traces.append(go.Scatter(
                        x=[2.3 + b * .6, 2.85 + b * .6, 2.85 + b * .6,
                           2.3 + b * .6, 2.3 + b * .6],
                        y=[r, r, r + .82, r + .82, r],
                        fill="toself", fillcolor=alpha(SEQ[(b + 4) % len(SEQ)], .7),
                        line=dict(color="#fff", width=1),
                        hoverinfo="skip", showlegend=False))
        # the blender
        if stage >= 4:
            traces.append(go.Scatter(
                x=[4.4, 5.9, 5.9, 4.4, 4.4], y=[4, 4, 8, 8, 4],
                fill="toself", fillcolor=alpha(C["primary"], .65),
                line=dict(color=C["primary_dark"], width=2.5),
                mode="lines+text", text=[""], hoverinfo="skip", showlegend=False))
        return traces

    frames = []
    for s, (title, sub_) in enumerate(stages):
        ann = [dict(x=.8, y=12.7, text="<b>training data</b>", showarrow=False,
                    font=dict(size=11, color=C["ink"]))]
        if s >= 3:
            ann.append(dict(x=3.15, y=12.7, text="<b>Z (out-of-fold preds)</b>",
                            showarrow=False, font=dict(size=11, color=C["ink"])))
        if s >= 4:
            ann.append(dict(x=5.15, y=8.7, text="<b>blender</b>", showarrow=False,
                            font=dict(size=11, color=C["primary_dark"])))
        frames.append(go.Frame(name=str(s + 1), data=board(s),
                               layout=go.Layout(title=f"{title} — {sub_}",
                                                annotations=ann)))

    f = go.Figure(data=board(0))
    f.update_layout(height=470, showlegend=False,
                    xaxis=dict(visible=False, range=[-.3, 6.4]),
                    yaxis=dict(visible=False, range=[-.5, 13.4]),
                    plot_bgcolor="#FFFFFF", title=stages[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1600), slider_prefix="step ")
    figure(f)

    code_lab(
        "Stacking by hand, then with scikit-learn — and the leakage it prevents",
        '''import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              StackingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=3000, n_features=20, n_informative=10,
                           n_redundant=5, class_sep=.85, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y,
                                      random_state=42)

base = {
    "rf":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "et":  ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "svc": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "gb":  HistGradientBoostingClassifier(random_state=42),
    "nb":  GaussianNB(),
}

print("=== individual base models ===")
for nm, m_ in base.items():
    m_.fit(Xtr, ytr)
    print(f"  {nm:<5} test accuracy {m_.score(Xte, yte):.4f}")

# ============ STACKING BY HAND =========================================
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# step 2-3: out-of-fold predictions -> the blending matrix Z
Z_train = np.column_stack([
    cross_val_predict(m_, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    for m_ in base.values()])
print(f"\\nZ_train shape = {Z_train.shape}  (m x B)")

# step 5: train the blender on the OUT-OF-FOLD predictions
blender = LogisticRegression(max_iter=2000).fit(Z_train, ytr)
print("blender coefficients (how much it trusts each base model):")
for nm, c in zip(base, blender.coef_[0]):
    print(f"  {nm:<5} {c:+.4f}")

# step 6: base models are already fitted on ALL of Xtr; build Z_test
Z_test = np.column_stack([m_.predict_proba(Xte)[:, 1] for m_ in base.values()])
print(f"\\nhand-built stacking test accuracy = "
      f"{accuracy_score(yte, blender.predict(Z_test)):.4f}")

# ============ THE LEAKAGE THIS PREVENTS ================================
print("\\n=== what happens if you skip the out-of-fold step ===")
Z_leaky = np.column_stack([m_.predict_proba(Xtr)[:, 1] for m_ in base.values()])
leaky_blender = LogisticRegression(max_iter=2000).fit(Z_leaky, ytr)
print("leaky blender coefficients:")
for nm, c in zip(base, leaky_blender.coef_[0]):
    print(f"  {nm:<5} {c:+.4f}")
print(f"\\nleaky blender on its own (leaked) training Z : "
      f"{leaky_blender.score(Z_leaky, ytr):.4f}   <- fantasy")
print(f"leaky blender on the honest test Z          : "
      f"{accuracy_score(yte, leaky_blender.predict(Z_test)):.4f}")
print(f"correct (out-of-fold) blender on test       : "
      f"{accuracy_score(yte, blender.predict(Z_test)):.4f}")
print("\\nThe leaky blender over-trusts whichever model memorises hardest")
print("(look at the rf/et coefficients) because on TRAINING data they look perfect.")

# ============ sklearn's StackingClassifier =============================
stack = StackingClassifier(
    estimators=list(base.items()),
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, stack_method="predict_proba", n_jobs=-1)
stack.fit(Xtr, ytr)
print(f"\\nsklearn StackingClassifier   : {stack.score(Xte, yte):.4f}")

stack_pt = StackingClassifier(
    estimators=list(base.items()),
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, stack_method="predict_proba", passthrough=True, n_jobs=-1)
stack_pt.fit(Xtr, ytr)
print(f"  + passthrough=True (Z and X): {stack_pt.score(Xte, yte):.4f}")

# ============ the summary table ========================================
from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(list(base.items()), voting="soft", n_jobs=-1).fit(Xtr, ytr)
best_single = max(m_.score(Xte, yte) for m_ in base.values())
print(f"\\n{'approach':<28}{'test accuracy':>15}")
print(f"{'best single model':<28}{best_single:>15.4f}")
print(f"{'soft voting':<28}{vote.score(Xte, yte):>15.4f}")
print(f"{'stacking (linear blender)':<28}{stack.score(Xte, yte):>15.4f}")
''',
        key="ch07_stacking",
    )

    keypoints([
        "Stacking replaces a fixed aggregation rule with a <b>trained blender</b>.",
        "The blender must be trained on <b>out-of-fold</b> predictions or it "
        "learns to trust memorisers.",
        "Keep the blender simple and regularised — its input is only $m \\times B$.",
        "<code>StackingClassifier</code> / <code>StackingRegressor</code> handle "
        "the whole protocol.",
        "Multi-layer stacking exists but the returns fall off fast; two layers is "
        "usually enough.",
    ])


# ==========================================================================
def s_7_8():
    section("7.8", "Exercises & Chapter Review")

    lead("Nine exercises. Numbers 8 and 9 build a real stacking ensemble.")

    exercise(
        1, "If you have trained five different models on the exact same training "
        "data, and they all achieve 95 % precision, is there any chance that you "
        "can combine these models to get better results? If so, how? If not, why?",
        "**Yes, and it usually works.** Combine them into a voting ensemble — "
        "hard voting, or better, soft voting if they can all output probabilities.\n\n"
        "It works when the models make **different kinds of errors**. Five models "
        "each 95 % correct but wrong on *different* instances will out-vote each "
        "other's mistakes.\n\n"
        "It works **even better** when the models are very different in kind "
        "(an SVM, a random forest, a $k$-NN, a naive Bayes, a logistic "
        "regression), because different algorithms have different inductive biases "
        "and therefore genuinely different failure modes. Five models of the *same* "
        "type trained on the *same* data are highly correlated and the gain will "
        "be small — the $\\rho\\sigma^2$ floor from §7.2.")

    exercise(
        2, "What is the difference between hard and soft voting classifiers?",
        "**Hard voting** counts each classifier's predicted class and takes the "
        "majority.\n\n"
        "**Soft voting** averages each classifier's *estimated class "
        "probabilities* and picks the class with the highest average.\n\n"
        "Soft voting usually performs better, because it gives more weight to "
        "highly confident votes. Its requirement is that every classifier can "
        "estimate probabilities — for `SVC` you must set `probability=True`, which "
        "triggers slow internal cross-validated Platt scaling (§5.8, exercise 4).")

    exercise(
        3, "Is it possible to speed up training of a bagging ensemble by "
        "distributing it across multiple servers? What about pasting ensembles, "
        "boosting ensembles, random forests, or stacking ensembles?",
        "**Bagging: yes.** Each predictor is trained independently of the others, "
        "so they can be trained in parallel on separate cores or servers.\n\n"
        "**Pasting: yes**, same reason.\n\n"
        "**Random forests: yes**, same reason — this is why `n_jobs=-1` gives "
        "near-linear speed-up.\n\n"
        "**Boosting: no.** Each predictor is built to correct its predecessor, so "
        "training is inherently sequential. (Individual *rounds* can be "
        "parallelised internally — histogram building, feature scanning — which is "
        "what §7.6 exploits, but the rounds themselves cannot be.)\n\n"
        "**Stacking: partly.** All base models *within a given layer* are "
        "independent and can be trained in parallel. But layer $\\ell+1$ needs "
        "layer $\\ell$'s out-of-fold predictions, so the layers are sequential.")

    exercise(
        4, "What is the benefit of out-of-bag evaluation?",
        "Each predictor in a bagging ensemble is trained on only ~63.2 % of the "
        "training instances (§7.2), so the remaining ~36.8 % are **out-of-bag** "
        "for it — never seen during its training.\n\n"
        "You can therefore evaluate the ensemble using those held-out predictions, "
        "getting an almost unbiased estimate of generalisation error **without "
        "needing a separate validation set and without any extra training**. That "
        "means more data available for actual training, and one fewer split to "
        "manage. Set `oob_score=True`.")

    exercise(
        5, "What makes Extra-Trees ensembles more random than regular random "
        "forests? How can this extra randomness help? Are Extra-Trees slower or "
        "faster than regular random forests?",
        "**The extra randomness:** a regular random forest considers a random "
        "subset of features at each node and then searches for the **best possible "
        "threshold** on each of them. Extra-Trees also uses a random feature "
        "subset, but then picks a **random threshold** for each candidate feature "
        "and keeps the best of those random splits.\n\n"
        "**How it helps:** it trades more bias for less variance, and it further "
        "decorrelates the trees — pushing $\\rho$ down in the "
        "$\\mathrm{Var}(\\bar h) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ "
        "formula.\n\n"
        "**Speed: much faster.** Finding the optimal threshold is the single most "
        "expensive operation in growing a tree, and Extra-Trees skips it entirely.")

    exercise(
        6, "If your AdaBoost ensemble underfits the training data, which "
        "hyperparameters should you tweak, and how?",
        "Underfitting means the ensemble is not expressive enough. Three moves:\n\n"
        "* **Increase `n_estimators`** — more rounds, more corrections.\n"
        "* **Increase `learning_rate`** — each predictor contributes more (note "
        "this is the *opposite* direction from the overfitting fix).\n"
        "* **Reduce the regularisation of the base estimator** — e.g. raise "
        "`max_depth` from 1 to 2 or 3, so each stump can capture interactions.")

    exercise(
        7, "If your gradient boosting ensemble overfits the training set, should "
        "you increase or decrease the learning rate?",
        "**Decrease it.** A lower learning rate means each tree contributes less, "
        "so the ensemble approaches the data more cautiously — this is "
        "**shrinkage**, and it is the main regulariser of a GBM.\n\n"
        "But there is a second half to the answer: lowering the learning rate "
        "without also increasing `n_estimators` will simply underfit instead. The "
        "correct procedure is to lower the learning rate, raise `n_estimators` "
        "generously, and use **early stopping** (`n_iter_no_change`, or "
        "`staged_predict` to find the minimum) to choose where to cut. You can "
        "also reduce `max_depth` or `subsample`.")

    exercise(
        8, "Load the MNIST dataset and split it into a training set, a validation "
        "set, and a test set. Then train various classifiers, such as a random "
        "forest classifier, an extra-trees classifier, and an SVM classifier. Next, "
        "try to combine them into an ensemble that outperforms each individual "
        "classifier on the validation set, using soft or hard voting. Once you "
        "have found one, try it on the test set. How much better does it perform "
        "compared to the individual classifiers?",
        "Expected results on full MNIST (50 000 / 10 000 / 10 000):\n\n"
        "| model | validation |\n|---|---|\n"
        "| Random forest | ~0.967 |\n| Extra-trees | ~0.970 |\n"
        "| Linear SVM | ~0.860 |\n| MLP | ~0.962 |\n"
        "| **Hard voting** | ~0.971 |\n| **Soft voting** | ~0.972 |\n\n"
        "The ensemble beats every member by a small margin. Two practical notes:\n\n"
        "1. The **linear SVM is much weaker** than the others and drags the vote "
        "down. Removing it (`voting_clf.set_params(svm=None)` then deleting from "
        "`estimators_`) typically *raises* the ensemble score. A weak, "
        "uncorrelated member helps; a weak, *wrong* member hurts.\n"
        "2. Ensemble gains on MNIST are small because the base models already "
        "agree on ~96 % of instances — there is little diversity left to exploit.",
        code='''from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier)
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X, y = mnist.data, mnist.target
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=10_000, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=10_000, random_state=42)

named = [("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
         ("et",  ExtraTreesClassifier(n_estimators=100, random_state=42)),
         ("svm", LinearSVC(max_iter=100, tol=20, dual="auto", random_state=42)),
         ("mlp", MLPClassifier(random_state=42))]
for name, clf in named:
    clf.fit(X_train, y_train)
    print(name, clf.score(X_valid, y_valid))

voting = VotingClassifier(named)
voting.fit(X_train, y_train)
print("hard voting:", voting.score(X_valid, y_valid))

# drop the weak SVM and re-check
voting.set_params(svm="drop")
del voting.estimators_[2]
print("without svm:", voting.score(X_valid, y_valid))

voting.voting = "soft"       # no refit needed
print("soft voting:", voting.score(X_valid, y_valid))''')

    exercise(
        9, "Run the individual classifiers from the previous exercise to make "
        "predictions on the validation set, and create a new training set with the "
        "resulting predictions: each training instance is a vector containing the "
        "set of predictions from all your classifiers for an image, and the target "
        "is the image's class. Train a classifier on this new training set. "
        "Congratulations — you have just trained a blender, and together with the "
        "classifiers it forms a stacking ensemble! Now evaluate the ensemble on "
        "the test set.",
        "This is stacking done by hand, with the **validation set** playing the "
        "role of the out-of-fold predictions from §7.7.\n\n"
        "The blender is typically a small random forest or a logistic regression "
        "trained on the $10\\,000 \\times 4$ matrix of predictions. Expect roughly "
        "**0.9695–0.9720** on the test set — comparable to soft voting, sometimes "
        "a shade better, sometimes not.\n\n"
        "Two things worth noticing:\n\n"
        "1. The blender's OOB score is measured on data the *base* models were not "
        "trained on, which is what makes it honest.\n"
        "2. `StackingClassifier` does all of this with proper $k$-fold out-of-fold "
        "predictions instead of a single validation split, so it uses the data "
        "more efficiently — the manual version here wastes 10 000 instances.",
        code='''import numpy as np

# build the blending matrix from validation-set predictions
X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)
for i, clf in enumerate(estimators):
    X_valid_predictions[:, i] = clf.predict(X_valid)

blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                 random_state=42)
blender.fit(X_valid_predictions, y_valid)
print("blender OOB:", blender.oob_score_)

# evaluate the whole stack on the test set
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=object)
for i, clf in enumerate(estimators):
    X_test_predictions[:, i] = clf.predict(X_test)
print("stacking test:", accuracy_score(y_test, blender.predict(X_test_predictions)))

# the same thing, done properly
from sklearn.ensemble import StackingClassifier
stack = StackingClassifier(named, final_estimator=RandomForestClassifier(
    n_estimators=200, random_state=43), cv=5)
stack.fit(X_train, y_train)
print("StackingClassifier test:", stack.score(X_test, y_test))''')

    rule()

    sub("The whole family on one page")

    table(
        ["Method", "Diversity from", "Reduces", "Base learner", "Parallel?"],
        [["<b>Voting</b>", "Different algorithms", "Variance", "Anything", "✅"],
         ["<b>Bagging / Pasting</b>", "Different row subsets", "Variance",
          "Deep trees", "✅"],
         ["<b>Random Forest</b>", "Rows + features at each split", "Variance",
          "Deep trees", "✅"],
         ["<b>Extra-Trees</b>", "Rows + features + random thresholds",
          "Variance (more)", "Deep trees", "✅"],
         ["<b>AdaBoost</b>", "Reweighted instances", "<b>Bias</b>",
          "Stumps", "❌"],
         ["<b>Gradient Boosting</b>", "Sequential residual fitting", "<b>Bias</b>",
          "Shallow trees", "❌"],
         ["<b>Hist Gradient Boosting</b>", "Same, but binned", "<b>Bias</b>",
          "Shallow trees", "❌ (rounds), ✅ (within)"],
         ["<b>Stacking</b>", "Different algorithms + a learned blender",
          "Both", "Anything", "Partly"]],
    )

    keypoints([
        "Bagging attacks <b>variance</b> by averaging; boosting attacks "
        "<b>bias</b> by correcting.",
        "$\\mathrm{Var}(\\bar h) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ — "
        "the formula that explains bagging <i>and</i> random forests.",
        "AdaBoost = forward stagewise exponential loss; gradient boosting = "
        "gradient descent in function space.",
        "For tabular data today: <b>HistGradientBoosting</b> first, random forest "
        "as the robust baseline.",
        "Stacking needs <b>out-of-fold</b> predictions or the blender learns to "
        "trust memorisers.",
    ], title="Chapter 7 in five lines")

    refs([
        ("Breiman, L. — *Bagging Predictors*",
         "https://doi.org/10.1007/BF00058655"),
        ("Breiman, L. — *Random Forests*",
         "https://doi.org/10.1023/A:1010933404324"),
        ("Freund & Schapire — *A Decision-Theoretic Generalization of On-Line "
         "Learning and an Application to Boosting* (AdaBoost)",
         "https://doi.org/10.1006/jcss.1997.1504"),
        ("Friedman, J. — *Greedy Function Approximation: A Gradient Boosting "
         "Machine*", "https://doi.org/10.1214/aos/1013203451"),
        ("Geurts, Ernst & Wehenkel — *Extremely Randomized Trees*",
         "https://doi.org/10.1007/s10994-006-6226-1"),
        ("Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System*",
         "https://doi.org/10.1145/2939672.2939785"),
        ("Ke et al. — *LightGBM: A Highly Efficient Gradient Boosting Decision "
         "Tree*", "NeurIPS 2017"),
        ("Wolpert, D. — *Stacked Generalization*",
         "https://doi.org/10.1016/S0893-6080(05)80023-1"),
    ])


# ==========================================================================
SECTIONS = [
    ("7.1", "Voting Classifiers", s_7_1),
    ("7.2", "Bagging and Pasting", s_7_2),
    ("7.3", "Random Forests & Importance", s_7_3),
    ("7.4", "Boosting — AdaBoost", s_7_4),
    ("7.5", "Gradient Boosting", s_7_5),
    ("7.6", "Histogram-Based Boosting", s_7_6),
    ("7.7", "Stacking", s_7_7),
    ("7.8", "Exercises & Review", s_7_8),
]

nav.render_chapter(CH, SECTIONS)
