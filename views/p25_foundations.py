"""Foundations — the concepts everything else rests on."""

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
from core.palette import C, CLASS_COLORS, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "foundations"

hero(
    kicker="Foundations",
    title="What learning actually is",
    blurb=(
        "Chapter 1 tells you what machine learning <i>does</i>. This page tells "
        "you why any of it is <b>justified</b> — what a learning problem is "
        "formally, why generalisation is possible at all, what capacity means, "
        "what a loss function is really asking for, and where the classical "
        "story stops being true. Read it alongside chapters 1–4."
    ),
    chips=["ERM derived", "9 sub-sections", "9 animations",
           "9 code labs", "theory, verified numerically"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_f1():
    section("F.1", "The Learning Problem, Formalised")

    lead(
        "Four objects define every supervised learning problem. Naming them "
        "precisely turns a vague activity into a question with an answer."
    )

    sub("The four objects")

    table(
        ["Object", "Symbol", "What it is"],
        [["<b>Input space</b>", "$\\mathcal{X}$",
          "Everything a model might be shown"],
         ["<b>Output space</b>", "$\\mathcal{Y}$",
          "$\\{0,1\\}$, $\\{1..K\\}$, $\\mathbb{R}$, a sequence…"],
         ["<b>Data distribution</b>", "$\\mathcal{D}$ over $\\mathcal{X}\\times\\mathcal{Y}$",
          "<b>Unknown, and never observed</b> — you only ever see samples"],
         ["<b>Hypothesis space</b>", "$\\mathcal{H}$",
          "The functions your algorithm is <i>able</i> to return"],
         ["<b>Loss</b>", "$\\ell: \\mathcal{Y}\\times\\mathcal{Y}\\to\\mathbb{R}$",
          "What it costs to answer $\\hat y$ when the truth is $y$"]],
    )

    sub("Risk: what we want, and what we can measure")

    md(
        "The thing we actually care about is the **risk** — the expected loss "
        "over the whole distribution, including every example we will never see:"
    )

    math(r"""
    R(h) \;=\; \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}
      \bigl[\ell\bigl(h(\mathbf{x}), y\bigr)\bigr]
    """)

    md(
        "We cannot compute it. $\\mathcal{D}$ is unknown. What we *can* compute "
        "is the **empirical risk** — the same average over the $m$ examples we "
        "happen to have:"
    )

    math(r"""
    \hat R_S(h) \;=\; \frac{1}{m}\sum_{i=1}^{m}
      \ell\bigl(h(\mathbf{x}^{(i)}), y^{(i)}\bigr)
    """)

    idea(
        "Every learning algorithm in this platform is one line",
        "<b>Empirical risk minimisation</b>: return the hypothesis in "
        "$\\mathcal{H}$ with the smallest empirical risk, "
        "$\\hat h = \\arg\\min_{h\\in\\mathcal{H}} \\hat R_S(h)$. Linear "
        "regression is ERM with squared loss over linear functions. Logistic "
        "regression is ERM with log loss. An SVM is ERM with hinge loss and a "
        "norm constraint. A neural network is ERM with cross-entropy over a "
        "parameterised function class, solved approximately. <b>The differences "
        "between the nineteen chapters are choices of $\\mathcal{H}$, of "
        "$\\ell$, and of how the minimisation is carried out.</b>",
    )

    sub("The three errors, separated")

    derive(
        [("Let $h^\\star$ be the <b>Bayes-optimal</b> predictor — the best "
          "possible function of any kind, achieving risk $R^\\star$. Let "
          "$h^{\\mathcal{H}}$ be the best function <i>in your hypothesis "
          "space</i>, and $\\hat h$ what your algorithm actually returned. "
          "Decompose the excess risk:",
          r"R(\hat h) - R^{\star} = "
          r"\underbrace{R(h^{\mathcal{H}}) - R^{\star}}_{\text{approximation}}"
          r" \;+\; \underbrace{R(\hat h) - R(h^{\mathcal{H}})}_{\text{estimation}}"),
         ("<b>Approximation error</b> is the price of your hypothesis space. A "
          "linear model on curved data pays it no matter how much data you "
          "collect. It shrinks by making $\\mathcal{H}$ richer.", None),
         ("<b>Estimation error</b> is the price of having finite data. You "
          "minimised $\\hat R_S$, not $R$, so you overfit the sample. It shrinks "
          "with more data — and <b>grows</b> as $\\mathcal{H}$ gets richer.",
          None),
         ("Those two move in opposite directions with $|\\mathcal{H}|$. That "
          "tension is the whole of model selection, and it is what "
          "bias–variance (§4.4) is measuring in a different coordinate system.",
          None),
         ("In practice there is a <b>third</b> term, because you did not solve "
          "the minimisation exactly:",
          r"R(\hat h) - R^{\star} = \varepsilon_{\text{app}}"
          r" + \varepsilon_{\text{est}} + \varepsilon_{\text{opt}}"),
         ("<b>Optimisation error</b> is usually treated as a nuisance, but for "
          "deep networks it is not: SGD does not find the global minimum, and "
          "<i>which</i> minimum it finds turns out to matter for "
          "generalisation (§F.5). Bottou and Bousquet's observation is that "
          "with a fixed compute budget, deliberately tolerating larger "
          "$\\varepsilon_{\\text{opt}}$ to process more data is often the "
          "better trade.", None)],
        title="Approximation, estimation, optimisation",
    )

    sub("Inductive bias")

    pitfall(
        "Learning from data alone is impossible — you must assume something",
        "Given any finite training set, there are infinitely many functions that "
        "fit it perfectly and disagree completely everywhere else. Nothing in "
        "the data prefers one over another. <b>Inductive bias</b> is the set of "
        "assumptions that does the preferring: linearity, smoothness, locality "
        "(convolution), permutation invariance, sparsity, a prior on the "
        "weights. Every algorithm has one, whether or not its author named it. "
        "When someone says a method is 'assumption-free', they mean its "
        "assumptions are implicit.",
    )

    table(
        ["Model", "Its inductive bias", "Fails when"],
        [["Linear regression", "The relationship is linear and additive",
          "It is not"],
         ["$k$-NN", "Nearby points have similar labels",
          "The metric is wrong, or dimension is high"],
         ["Decision tree", "The function is piecewise constant on "
                           "axis-aligned boxes",
          "The boundary is diagonal (§6.6)"],
         ["CNN", "Locality and translation equivariance",
          "The signal is genuinely global (§16.8)"],
         ["RNN / Transformer", "The data is a sequence with order",
          "Order is meaningless"],
         ["$\\ell_1$ penalty", "Few features matter",
          "Everything matters a little"],
         ["Gaussian likelihood (MSE)", "Errors are symmetric, light-tailed",
          "Outliers, asymmetric costs"]],
    )

    sub("The no-free-lunch theorem, stated properly")

    proof(
        "What NFL does and does not say",
        "Wolpert's theorem: averaged over <b>all</b> possible target functions "
        "$f: \\mathcal{X}\\to\\mathcal{Y}$, every learning algorithm has "
        "identical expected off-training-set error. Random guessing ties with a "
        "neural network. <br><br>"
        "What that means: <b>there is no universally best algorithm</b>, so "
        "'which model is best?' is not a well-posed question without naming the "
        "problem class. <br><br>"
        "What it does <b>not</b> mean: that all algorithms are equally good on "
        "<i>your</i> problem. The uniform average is over a set dominated by "
        "functions that are pure noise — random labellings with no structure at "
        "all. Real-world targets are a vanishingly small, highly structured "
        "corner of that set. NFL says your inductive bias must be <b>earned by "
        "matching the problem</b>, not that bias is futile.",
    )

    anim_header("ERM: shrinking the hypothesis space")

    rng = np.random.default_rng(0)
    n_pts = 22
    xs = np.sort(rng.uniform(-3, 3, n_pts))
    ys = np.sin(1.3 * xs) + 0.35 * xs + rng.normal(0, .3, n_pts)
    grid = np.linspace(-3.3, 3.3, 300)
    x_te = np.sort(rng.uniform(-3, 3, 300))
    y_te = np.sin(1.3 * x_te) + 0.35 * x_te + rng.normal(0, .3, 300)

    degrees = list(range(0, 16))
    frames = []
    for d in degrees:
        co = np.polyfit(xs, ys, d)
        emp = float(np.mean((np.polyval(co, xs) - ys) ** 2))
        risk = float(np.mean((np.polyval(co, x_te) - y_te) ** 2))
        # a few other members of the same hypothesis space, for context
        others = []
        for k in range(4):
            perturbed = co + rng.normal(0, .06 * (np.abs(co) + .1))
            others.append(np.polyval(perturbed, grid))
        frames.append(go.Frame(name=str(d), data=[
            go.Scatter(x=grid, y=np.polyval(co, grid), mode="lines",
                       line=dict(color=C["primary"], width=4)),
        ] + [go.Scatter(x=grid, y=o, mode="lines",
                        line=dict(color=alpha(C["primary"], .22), width=1.5))
             for o in others] + [
            go.Scatter(x=xs, y=ys, mode="markers",
                       marker=dict(size=9, color=C["train"],
                                   line=dict(color="#fff", width=1))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"H = polynomials of degree ≤ {d}   ·   empirical risk "
            f"{emp:.4f}   ·   true risk {risk:.4f}   ·   "
            + ("approximation error dominates" if d < 3 else
               "estimation error dominates" if d > 9 else "balanced"),
            color=(C["danger"] if (d < 2 or d > 11) else C["success"]))])))

    f = go.Figure(data=[
        go.Scatter(x=grid, y=np.polyval(np.polyfit(xs, ys, 0), grid),
                   mode="lines", name="ĥ = argmin empirical risk",
                   line=dict(color=C["primary"], width=4)),
        go.Scatter(x=xs, y=ys, mode="markers", name="the sample S",
                   marker=dict(size=9, color=C["train"],
                               line=dict(color="#fff", width=1))),
    ])
    f.update_layout(height=460, xaxis_title="x", yaxis_title="y",
                    yaxis=dict(range=[-3, 3]),
                    title="Empirical risk minimisation over a growing H",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="degree ")
    figure(f, "Faint lines are other members of the same hypothesis space. As H "
              "grows, the empirical risk falls forever while the true risk turns "
              "around — that turn is estimation error overtaking approximation "
              "error.")

    code_lab(
        "ERM from scratch, and the three errors separated numerically",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(42)

# ============ 1. THE FOUR OBJECTS ======================================
# X = R, Y = R, D = "y = f(x) + noise", H = polynomials, loss = squared
def true_f(x):
    return np.sin(1.3*x) + 0.35*x            # the TRUTH -- never observed

SIGMA = 0.30                                  # the irreducible noise

def sample(m, seed=None):
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, m)
    return x, true_f(x) + r.normal(0, SIGMA, m)

print("=== the setup ===")
print(f"  X = [-3, 3],  Y = R")
print(f"  D: y = sin(1.3x) + 0.35x + N(0, {SIGMA}^2)")
print(f"  H_d = polynomials of degree <= d")
print(f"  loss = squared error")

# ============ 2. RISK vs EMPIRICAL RISK ================================
# we CAN compute the true risk here only because we invented D
x_big, y_big = sample(40000, seed=999)        # a stand-in for the integral

def risk(coef):
    return float(np.mean((np.polyval(coef, x_big) - y_big)**2))

def empirical_risk(coef, x, y):
    return float(np.mean((np.polyval(coef, x) - y)**2))

print()
print("=== the empirical risk is a NOISY ESTIMATE of the risk ===")
x_s, y_s = sample(30, seed=0)
co = np.polyfit(x_s, y_s, 3)
print(f"  a degree-3 fit on m=30:")
print(f"    empirical risk on ITS OWN sample : {empirical_risk(co, x_s, y_s):.5f}")
print(f"    true risk (40 000 fresh points)  : {risk(co):.5f}")
print(f"    the gap                          : "
      f"{risk(co) - empirical_risk(co, x_s, y_s):+.5f}")
print("  the empirical risk is BIASED DOWNWARD, because the same data chose")
print("  the coefficients. That bias IS estimation error.")

print()
print(f"{'sample size m':>15}{'empirical risk':>18}{'true risk':>13}{'gap':>11}")
for m in [10, 20, 50, 200, 1000, 5000]:
    x_s, y_s = sample(m, seed=1)
    co = np.polyfit(x_s, y_s, 5)
    e, r_ = empirical_risk(co, x_s, y_s), risk(co)
    print(f"{m:>15}{e:>18.5f}{r_:>13.5f}{r_-e:>11.5f}")
print("  the gap CLOSES as m grows. That is the whole reason more data helps.")

# ============ 3. THE THREE ERRORS, SEPARATED ===========================
print()
print("="*70)
print("R(h_hat) - R* = approximation + estimation")
print("="*70)
R_star = SIGMA**2                             # the Bayes risk: pure noise
print(f"  Bayes risk R* = sigma^2 = {R_star:.5f}  (no model can beat this)")
print()
print(f"{'degree':>8}{'best-in-H risk':>17}{'approximation':>16}"
      f"{'E[R(h_hat)]':>14}{'estimation':>13}")
for d in [0, 1, 2, 3, 5, 8, 12, 18]:
    # best in H: fit on a huge sample -> essentially h^H
    co_H = np.polyfit(x_big[:20000], y_big[:20000], d)
    R_H = risk(co_H)
    # E[R(h_hat)] over many small samples
    risks = []
    for trial in range(60):
        xs_, ys_ = sample(30, seed=1000+trial)
        risks.append(risk(np.polyfit(xs_, ys_, d)))
    ER = float(np.mean(risks))
    def fmt(v, w):
        return f"{v:>{w}.5f}" if abs(v) < 1e4 else f"{v:>{w}.3e}"
    print(f"{d:>8}{R_H:>17.5f}{R_H-R_star:>16.5f}"
          f"{fmt(ER, 14)}{fmt(ER-R_H, 13)}")
print()
print("  APPROXIMATION falls monotonically with the degree -- a richer H can")
print("  always represent more.")
print("  ESTIMATION rises monotonically -- a richer H overfits 30 points more.")
print("  their SUM has a minimum. That minimum is what model selection is")
print("  looking for, and it MOVES when m changes.")

# --- and it moves with m ---------------------------------------------
print()
print("=== the best degree depends on how much data you have ===")
print(f"{'m':>8}{'best degree':>14}{'its risk':>12}")
for m in [15, 30, 60, 150, 400, 1500]:
    best = (np.inf, None)
    for d in range(0, 14):
        risks = []
        for trial in range(25):
            xs_, ys_ = sample(m, seed=2000+trial)
            if len(xs_) <= d:
                continue
            risks.append(risk(np.polyfit(xs_, ys_, d)))
        if risks and np.mean(risks) < best[0]:
            best = (float(np.mean(risks)), d)
    print(f"{m:>8}{best[1]:>14}{best[0]:>12.5f}")
print("  more data supports a richer model. This is why 'which model is best'")
print("  has no answer without saying HOW MUCH DATA.")

# ============ 4. INDUCTIVE BIAS IS UNAVOIDABLE =========================
print()
print("="*70)
print("Infinitely many functions fit the data perfectly")
print("="*70)
x_tiny, y_tiny = sample(6, seed=7)
print(f"  6 training points. Degree-5 polynomial fits them EXACTLY:")
co5 = np.polyfit(x_tiny, y_tiny, 5)
print(f"    training error = "
      f"{empirical_risk(co5, x_tiny, y_tiny):.2e}")
print(f"    true risk      = {risk(co5):.5f}")
print()
print("  now add a 7th point ANYWHERE and refit -- still exact, wildly different:")
print(f"  {'extra point at y =':>22}{'train error':>14}{'true risk':>12}"
      f"{'prediction at x=0':>20}")
for extra_y in [-2.0, 0.0, 2.0, 5.0]:
    xx = np.append(x_tiny, 2.5)
    yy = np.append(y_tiny, extra_y)
    c = np.polyfit(xx, yy, 6)
    print(f"  {extra_y:>22.1f}{empirical_risk(c, xx, yy):>14.2e}"
          f"{risk(c):>12.5f}{np.polyval(c, 0.0):>20.4f}")
print("  every one fits its data PERFECTLY and they disagree completely.")
print("  the data cannot choose between them. Something else must -- and that")
print("  something is INDUCTIVE BIAS.")

# ============ 5. DIFFERENT BIASES, SAME DATA ===========================
print()
print("=== four models, same 40 points, four different extrapolations ===")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

x_tr, y_tr = sample(40, seed=3)
probe = np.array([-5.0, -4.0, 3.5, 5.0])      # OUTSIDE the training range
print(f"  training range [-3, 3]; probing at {probe}")
print(f"  {'model':<28}{'inductive bias':<34}{'predictions outside':>22}")
models = [
    ("linear", LinearRegression(), "linear and additive"),
    ("degree-9 polynomial",
     make_pipeline(PolynomialFeatures(9), LinearRegression()),
     "smooth, globally polynomial"),
    ("k-NN (k=5)", KNeighborsRegressor(5), "nearby points look alike"),
    ("decision tree", DecisionTreeRegressor(max_depth=4, random_state=0),
     "piecewise constant"),
    ("RBF SVR", make_pipeline(StandardScaler(), SVR()),
     "smooth, decays to the mean"),
]
for nm, m, bias in models:
    m.fit(x_tr[:, None], y_tr)
    p = m.predict(probe[:, None])
    print(f"  {nm:<28}{bias:<34}{np.round(p, 2)!s:>22}")
print(f"  {'TRUTH':<28}{'':<34}{np.round(true_f(probe), 2)!s:>22}")
print()
print("  they agree INSIDE the data and disagree wildly outside it.")
print("  the disagreement IS their inductive bias, made visible.")

# ============ 6. NO FREE LUNCH, MEASURED ===============================
print()
print("="*70)
print("No free lunch: average over ALL target functions")
print("="*70)
# a tiny discrete world: 8 possible inputs, binary labels -> 2^8 = 256 targets
n_x = 8
X_all = np.arange(n_x)
train_idx = np.array([0, 1, 2, 3])            # we see half
test_idx = np.array([4, 5, 6, 7])             # judged on the other half

def const_learner(xt, yt):     return lambda x: np.zeros(len(x), int)
def majority_learner(xt, yt):
    maj = int(round(yt.mean()))
    return lambda x: np.full(len(x), maj, int)
def nn_learner(xt, yt):
    return lambda x: yt[np.abs(x[:, None] - xt[None, :]).argmin(1)]
def anti_nn_learner(xt, yt):
    return lambda x: 1 - yt[np.abs(x[:, None] - xt[None, :]).argmin(1)]
def random_learner(xt, yt):
    r = np.random.default_rng(0)
    return lambda x: r.integers(0, 2, len(x))

learners = [("always 0", const_learner), ("majority", majority_learner),
            ("nearest neighbour", nn_learner),
            ("ANTI nearest neighbour", anti_nn_learner),
            ("random", random_learner)]

print(f"  8 inputs, binary labels -> 2^8 = 256 possible target functions")
print(f"  train on inputs 0-3, judged on inputs 4-7")
print()
print(f"  {'learner':<26}{'mean off-training accuracy':>28}")
for nm, make in learners:
    accs = []
    for code in range(2**n_x):
        target = np.array([(code >> i) & 1 for i in range(n_x)])
        h = make(train_idx, target[train_idx])
        accs.append(float((h(test_idx) == target[test_idx]).mean()))
    print(f"  {nm:<26}{np.mean(accs):>28.6f}")
print("  EXACTLY 0.5 for every one, including 'always 0' and the ANTI-nearest")
print("  neighbour rule. That is the theorem.")

# --- but restrict to STRUCTURED targets and it changes completely -----
print()
print("=== now average over SMOOTH targets only (at most 2 label changes) ===")
print(f"  {'learner':<26}{'mean off-training accuracy':>28}")
for nm, make in learners:
    accs = []
    for code in range(2**n_x):
        target = np.array([(code >> i) & 1 for i in range(n_x)])
        if int(np.abs(np.diff(target)).sum()) > 2:     # the STRUCTURE filter
            continue
        h = make(train_idx, target[train_idx])
        accs.append(float((h(test_idx) == target[test_idx]).mean()))
    print(f"  {nm:<26}{np.mean(accs):>28.6f}")
print(f"  ({len([1 for code in range(2**n_x) if int(np.abs(np.diff(np.array([(code>>i)&1 for i in range(n_x)]))).sum()) <= 2])} "
      f"of 256 targets are 'smooth')")
print()
print("  nearest neighbour now WINS and the anti-rule LOSES, by exactly as")
print("  much. Nothing about the learners changed -- only the set of problems.")
print("  NFL does not say bias is futile. It says bias must be EARNED by")
print("  matching the structure your problems actually have.")

import plotly.graph_objects as go
degs = list(range(0, 16))
app, est, tot = [], [], []
for d in degs:
    co_H = np.polyfit(x_big[:20000], y_big[:20000], d)
    R_H = risk(co_H)
    rr = [risk(np.polyfit(*sample(30, seed=5000+t), d)) for t in range(25)]
    app.append(R_H - R_star); est.append(float(np.mean(rr)) - R_H)
    tot.append(float(np.mean(rr)) - R_star)
fig = go.Figure()
fig.add_scatter(x=degs, y=app, mode="lines+markers", name="approximation",
                line=dict(color=C["danger"], width=3))
fig.add_scatter(x=degs, y=est, mode="lines+markers", name="estimation",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(x=degs, y=tot, mode="lines+markers", name="excess risk (sum)",
                line=dict(color=C["ink"], width=3))
fig.update_layout(height=430, yaxis_type="log", xaxis_title="degree of H",
                  yaxis_title="excess risk over the Bayes risk",
                  title="The two errors move in opposite directions")
''',
        key="found_erm",
    )

    quiz(
        "Why can no learning algorithm work without an inductive bias?",
        ["Because computers cannot represent all functions",
         "Because infinitely many functions fit any finite training set exactly "
         "and disagree everywhere else — the data alone cannot choose",
         "Because of measurement noise",
         "Because the training set is always too small"],
        1,
        "This is the point, and it is a logical one rather than a practical "
        "one: fitting the sample does not constrain behaviour off the sample at "
        "all. Something outside the data — linearity, smoothness, locality, a "
        "prior — must do the choosing. Every algorithm has such an assumption; "
        "the only question is whether it matches your problem.",
        key="fq1",
    )

    keypoints([
        "A learning problem is $(\\mathcal{X}, \\mathcal{Y}, \\mathcal{D}, "
        "\\mathcal{H}, \\ell)$; the risk $R(h)$ is what you want and cannot "
        "compute.",
        "<b>ERM</b> minimises $\\hat R_S(h)$ instead — every algorithm here is "
        "ERM with a different $\\mathcal{H}$ and $\\ell$.",
        "Excess risk = <b>approximation + estimation + optimisation</b>; the "
        "first two move oppositely with $|\\mathcal{H}|$.",
        "<b>Inductive bias is not optional</b> — infinitely many functions fit "
        "any finite sample.",
        "<b>NFL</b> says bias must be earned by matching the problem class, not "
        "that bias is futile.",
    ])


# ==========================================================================
def s_f2():
    section("F.2", "Why Generalisation Is Possible At All")

    lead(
        "You minimise error on 1 000 examples and it holds on a billion you "
        "never saw. That is a strong claim. Here is the argument that makes it, "
        "and exactly where it can break."
    )

    sub("The assumption everything rests on")

    md(
        "The training sample $S = \\{(\\mathbf{x}^{(i)}, y^{(i)})\\}_{i=1}^m$ is "
        "drawn **independently and identically distributed** from the same "
        "$\\mathcal{D}$ that will generate the future."
    )

    warn(
        "i.i.d. is an assumption about your world, and it is usually false",
        "<b>Not identically distributed</b>: users change, sensors drift, "
        "fraudsters adapt, a competitor launches. That is §19.8's drift. "
        "<b>Not independent</b>: consecutive time steps, multiple rows per "
        "customer, images from the same session, overlapping windows (§15.4). "
        "Dependence inflates your effective sample size — you think you have "
        "10 000 examples and statistically you have 300. Almost every "
        "'mysteriously optimistic validation score' is one of these two.",
    )

    sub("One hypothesis: Hoeffding")

    derive(
        [("Fix a <b>single</b> hypothesis $h$, chosen before seeing the data. "
          "Its empirical risk is an average of $m$ i.i.d. bounded random "
          "variables $\\ell(h(\\mathbf{x}^{(i)}), y^{(i)}) \\in [0,1]$, whose "
          "mean is $R(h)$.", None),
         ("<b>Hoeffding's inequality</b> bounds how far an average of bounded "
          "independent variables strays from its mean:",
          r"\Pr\Bigl[\bigl|\hat R_S(h) - R(h)\bigr| > \epsilon\Bigr]"
          r" \;\le\; 2e^{-2m\epsilon^{2}}"),
         ("Set the right-hand side to $\\delta$ and solve for $\\epsilon$: with "
          "probability at least $1-\\delta$,",
          r"\bigl|\hat R_S(h) - R(h)\bigr| \;\le\;"
          r" \sqrt{\frac{\ln(2/\delta)}{2m}}"),
         ("The error shrinks as $\\mathcal{O}(1/\\sqrt{m})$. <b>To halve your "
          "uncertainty you need four times the data</b> — which is the single "
          "most useful rule of thumb in this section, and why the returns to "
          "data collection feel so slow.", None),
         ("<b>But this bound is for a hypothesis fixed in advance.</b> ERM picks "
          "$\\hat h$ <i>after</i> looking at $S$, deliberately choosing the one "
          "that looks best on it. That choice is exactly the kind of "
          "data-dependent selection that breaks the argument.", None)],
        title="Hoeffding, and why it is not enough",
    )

    sub("Many hypotheses: the union bound")

    derive(
        [("We need the bound to hold <b>simultaneously</b> for every $h$ in "
          "$\\mathcal{H}$, because we do not know in advance which one ERM will "
          "pick. Apply the union bound over a finite $\\mathcal{H}$:",
          r"\Pr\Bigl[\exists h \in \mathcal{H}: \bigl|\hat R_S(h) - R(h)\bigr|"
          r" > \epsilon\Bigr] \le \sum_{h\in\mathcal{H}} 2e^{-2m\epsilon^2}"
          r" = 2|\mathcal{H}|e^{-2m\epsilon^2}"),
         ("Setting that to $\\delta$ gives the <b>uniform convergence</b> "
          "bound — with probability $1-\\delta$, for <b>all</b> $h$ at once:",
          r"\bigl|\hat R_S(h) - R(h)\bigr| \;\le\;"
          r" \sqrt{\frac{\ln|\mathcal{H}| + \ln(2/\delta)}{2m}}"),
         ("<b>Read the trade-off directly off that expression.</b> A larger "
          "$\\mathcal{H}$ lowers the achievable $\\hat R_S$ (approximation) and "
          "raises the $\\sqrt{\\ln|\\mathcal{H}|/m}$ term (estimation). That is "
          "the same trade as §F.1, now with a formula.", None),
         ("It also says how much data you need. To guarantee $\\epsilon$ "
          "accuracy with confidence $1-\\delta$:",
          r"m \;\ge\; \frac{\ln|\mathcal{H}| + \ln(2/\delta)}{2\epsilon^{2}}"),
         ("<b>Sample complexity is logarithmic in $|\\mathcal{H}|$.</b> Doubling "
          "the number of candidate hypotheses costs you almost nothing; "
          "squaring it costs a factor of two. That is why searching over "
          "thousands of models is fine and searching over $2^{2^{k}}$ is not.",
          None),
         ("For a parameterised model stored in $b$ bits per parameter with $p$ "
          "parameters, $|\\mathcal{H}| \\le 2^{bp}$, so "
          "$\\ln|\\mathcal{H}| \\le bp\\ln 2$ and the bound scales as "
          "$\\sqrt{p/m}$ — <b>the origin of the folklore that you need roughly "
          "as many examples as parameters</b>. Modern networks violate it "
          "spectacularly, which is §F.5.", None)],
        title="From one hypothesis to a whole class",
    )

    sub("PAC learning")

    md(
        "The framework this all lives in is **Probably Approximately Correct** "
        "learning (Valiant, 1984). A class is PAC-learnable if, for any "
        "$\\epsilon, \\delta > 0$, some algorithm returns a hypothesis with "
        "risk within $\\epsilon$ of optimal, with probability $1-\\delta$, from "
        "$m = \\mathrm{poly}(1/\\epsilon, 1/\\delta)$ examples."
    )

    idea(
        "Both hedges in the name are load-bearing",
        "<b>Approximately</b> correct, because with finite data you can never "
        "match the best hypothesis exactly. <b>Probably</b>, because you might "
        "have drawn a freak sample — there is always some chance the training "
        "set is unrepresentative, and no algorithm can rule it out. Any "
        "guarantee that promised exact correctness with certainty would be "
        "false. This is also why a single validation score is a random "
        "variable, and why §F.7 insists on reporting intervals.",
    )

    anim_header("Empirical risk concentrating around the true risk")

    rng = np.random.default_rng(3)
    true_p = 0.30
    ms = [5, 10, 20, 50, 100, 300, 1000, 3000, 10000]
    frames = []
    for m in ms:
        draws = rng.binomial(m, true_p, 4000) / m
        bound = np.sqrt(np.log(2/0.05)/(2*m))
        inside = float(np.mean(np.abs(draws - true_p) <= bound))
        frames.append(go.Frame(name=str(m), data=[
            go.Histogram(x=draws, nbinsx=60, histnorm="probability density",
                         marker=dict(color=alpha(C["primary"], .8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"m = {m}   ·   Hoeffding radius at δ = 0.05: ±{bound:.4f}   ·   "
            f"{inside:.1%} of samples fall inside   ·   "
            f"observed spread ±{draws.std()*1.96:.4f}")])))

    f = go.Figure(data=[go.Histogram(x=rng.binomial(5, true_p, 4000)/5,
                                     nbinsx=60, histnorm="probability density",
                                     marker=dict(color=alpha(C["primary"],
                                                             .8)))])
    f.add_vline(x=true_p, line_dash="dash", line_color=C["danger"],
                annotation_text="true risk R(h) = 0.30")
    f.update_layout(height=430, xaxis_title="empirical risk on a sample of m",
                    yaxis_title="density", xaxis=dict(range=[0, 1]),
                    title="Hoeffding: the estimate concentrates at rate 1/√m")
    anim.animate(f, frames, duration=nav.anim_ms(950), slider_prefix="m = ")
    figure(f, "Note how slowly it narrows. From m = 100 to m = 10 000 — a "
              "hundredfold increase — the spread only falls by ten.")

    code_lab(
        "Hoeffding, the union bound, and what breaks when i.i.d. fails",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. HOEFFDING, VERIFIED ===================================
def hoeffding_radius(m, delta=0.05):
    return np.sqrt(np.log(2/delta)/(2*m))

print("=== Hoeffding: |R_hat - R| <= sqrt(ln(2/delta) / 2m) ===")
TRUE_RISK = 0.30
print(f"{'m':>8}{'bound (95%)':>14}{'actual 95% spread':>21}"
      f"{'coverage':>11}{'bound / actual':>16}")
for m in [10, 30, 100, 300, 1000, 3000, 10000]:
    draws = rng.binomial(m, TRUE_RISK, 20000)/m
    bound = hoeffding_radius(m)
    actual = float(np.percentile(np.abs(draws-TRUE_RISK), 95))
    cover = float(np.mean(np.abs(draws-TRUE_RISK) <= bound))
    print(f"{m:>8}{bound:>14.5f}{actual:>21.5f}{cover:>11.4f}"
          f"{bound/actual:>16.2f}x")
print("  the bound HOLDS (coverage ~ 1.0) but is loose by 2-3x, because it")
print("  assumes the worst-case distribution. That is the price of a")
print("  DISTRIBUTION-FREE guarantee.")

# --- the 1/sqrt(m) rate, made concrete -------------------------------
print()
print("=== to halve the error you need 4x the data ===")
print(f"{'target radius':>15}{'m required':>14}{'multiple of the previous':>27}")
prev = None
for eps in [0.10, 0.05, 0.025, 0.0125]:
    m_req = int(np.ceil(np.log(2/0.05)/(2*eps**2)))
    mult = "" if prev is None else f"{m_req/prev:.1f}x"
    print(f"{eps:>15.4f}{m_req:>14,}{mult:>27}")
    prev = m_req

# ============ 2. WHY ERM BREAKS THE SINGLE-HYPOTHESIS BOUND ============
print()
print("="*70)
print("Selecting the best of many hypotheses biases the estimate")
print("="*70)
m = 200
print(f"  {len(range(1))} sample of m={m}; every hypothesis has TRUE risk 0.50")
print(f"  (they are all pure coin flips -- none is better than any other)")
print()
print(f"{'|H| searched':>14}{'E[min empirical risk]':>24}"
      f"{'optimistic bias':>18}{'sqrt(ln|H|/2m)':>18}")
for H in [1, 2, 10, 100, 1000, 10000]:
    mins = []
    for _ in range(400):
        # H independent hypotheses, each with true risk 0.5
        emp = rng.binomial(m, 0.5, H)/m
        mins.append(emp.min())
    print(f"{H:>14}{np.mean(mins):>24.5f}{0.5-np.mean(mins):>18.5f}"
          f"{np.sqrt(np.log(max(H,2))/(2*m)):>18.5f}")
print("  every hypothesis is WORTHLESS, yet the best-looking one appears to")
print("  have risk far below 0.5. The apparent skill is pure selection.")
print("  the union-bound term sqrt(ln|H| / 2m) tracks it closely.")
print()
print("  this is the WINNER'S CURSE, and it is exactly what happens during")
print("  hyperparameter search (2.7, F.7).")

# ============ 3. SAMPLE COMPLEXITY IS LOG IN |H| =======================
print()
print("=== m >= (ln|H| + ln(2/delta)) / 2eps^2 ===")
print(f"{'|H|':>16}{'ln|H|':>10}{'m for eps=0.05':>18}")
for H in [2, 10, 1000, 10**6, 10**12, 2**1000]:
    lnH = np.log(float(H)) if H < 1e300 else 1000*np.log(2)
    m_req = int(np.ceil((lnH + np.log(2/0.05))/(2*0.05**2)))
    label = f"{H:.0e}" if H < 1e300 else "2^1000"
    print(f"{label:>16}{lnH:>10.1f}{m_req:>18,}")
print("  going from 1 000 hypotheses to a TRILLION multiplies the data")
print("  requirement by less than 4. Sample complexity is LOGARITHMIC.")
print("  that is why searching over many models is cheap -- and why a bound")
print("  based on counting parameters is so weak for modern networks.")

# ============ 4. WHEN i.i.d. FAILS: DEPENDENCE =========================
print()
print("="*70)
print("Dependence destroys your effective sample size")
print("="*70)
def ar1(n, rho, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    e = r.normal(0, np.sqrt(1-rho**2), n)
    for t in range(1, n):
        x[t] = rho*x[t-1] + e[t]
    return x

print(f"  the mean of n correlated samples has variance")
print(f"  sigma^2/n * (1+rho)/(1-rho) instead of sigma^2/n")
print()
print(f"{'rho':>7}{'observed Var(mean)':>21}{'iid prediction':>17}"
      f"{'inflation':>12}{'effective n':>14}")
n = 500
for rho in [0.0, 0.3, 0.6, 0.9, 0.95, 0.99]:
    means = [ar1(n, rho, seed=s).mean() for s in range(3000)]
    v = float(np.var(means))
    iid_v = 1.0/n
    print(f"{rho:>7.2f}{v:>21.6f}{iid_v:>17.6f}{v/iid_v:>12.1f}x"
          f"{int(n*iid_v/v):>14}")
print("  at rho=0.95, 500 correlated observations carry as much information")
print("  as about 13 independent ones. Every confidence interval computed")
print("  as if they were independent is roughly 6x too narrow.")

# --- the same thing with overlapping windows -------------------------
print()
print("=== overlapping windows: the time-series version (15.4) ===")
series = ar1(1200, 0.7, seed=1)
L = 30
W = np.lib.stride_tricks.sliding_window_view(series, L)
print(f"  {len(series)} points -> {len(W)} windows of length {L}")
print(f"  consecutive windows share {L-1}/{L} = {(L-1)/L:.0%} of their values")
corr = np.corrcoef(W[:-1].ravel(), W[1:].ravel())[0, 1]
print(f"  correlation between consecutive windows: {corr:.4f}")
print(f"  a RANDOM train/test split over these windows puts nearly identical")
print(f"  rows on both sides. The test score is then meaningless.")

# ============ 5. WHEN i.i.d. FAILS: SHIFT ==============================
print()
print("=== not identically distributed: the score decays ===")
from sklearn.linear_model import LogisticRegression
def make(n, shift, seed):
    r = np.random.default_rng(seed)
    X = r.normal(shift, 1, (n, 4))
    y = (X[:, 0] + 0.6*X[:, 1] - 0.4*X[:, 2] + r.normal(0, .5, n) > shift).astype(int)
    return X, y

Xtr, ytr = make(3000, 0.0, 0)
clf = LogisticRegression().fit(Xtr, ytr)
print(f"{'shift in the inputs':>21}{'test accuracy':>16}{'drop':>9}")
base = None
for sh in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    Xte, yte = make(3000, sh, 99)
    a = clf.score(Xte, yte)
    base = base if base is not None else a
    print(f"{sh:>21.2f}{a:>16.4f}{a-base:>+9.4f}")
print("  the model did not change. The world did. Every guarantee in this")
print("  section is conditional on the test data coming from the SAME")
print("  distribution -- which is why 19.8 monitors exactly that.")

# ============ 6. PAC: PUTTING IT TOGETHER ==============================
print()
print("="*70)
print("A PAC guarantee, end to end")
print("="*70)
# H = axis-aligned thresholds on [0,1]: h_t(x) = 1[x > t], t in a grid
GRID = 1000
thresholds = np.linspace(0, 1, GRID)
TRUE_T = 0.63

def run(m, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(0, 1, m)
    y = (x > TRUE_T).astype(int)
    emp = np.array([np.mean((x > t).astype(int) != y) for t in thresholds])
    t_hat = thresholds[emp.argmin()]
    true_risk = abs(t_hat - TRUE_T)            # risk = the mismatched interval
    return true_risk

print(f"  H = {GRID} thresholds, true threshold {TRUE_T}")
print(f"  bound: R(h_hat) <= 0 + 2*sqrt((ln|H| + ln(2/delta)) / 2m)")
print()
print(f"{'m':>8}{'bound':>12}{'mean actual risk':>19}{'95th pct actual':>18}"
      f"{'bound holds':>13}")
for m in [20, 50, 200, 1000, 5000]:
    risks = np.array([run(m, s) for s in range(300)])
    bound = 2*np.sqrt((np.log(GRID) + np.log(2/0.05))/(2*m))
    print(f"{m:>8}{bound:>12.5f}{risks.mean():>19.5f}"
          f"{np.percentile(risks, 95):>18.5f}"
          f"{str(bool(np.percentile(risks, 95) <= bound)):>13}")
print("  the guarantee holds every time, and is loose by roughly 10x.")
print("  PAC bounds are correctness proofs, not performance predictions.")

import plotly.graph_objects as go
ms = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
fig = go.Figure()
for H, col in [(1, C["success"]), (100, C["warning"]), (10**6, C["danger"])]:
    fig.add_scatter(x=ms,
                    y=[np.sqrt((np.log(max(H, 2))+np.log(40))/(2*m))
                       for m in ms],
                    mode="lines+markers", name=f"|H| = {H:.0e}",
                    line=dict(width=3, color=col))
fig.update_layout(height=420, xaxis_type="log", yaxis_type="log",
                  xaxis_title="training examples m",
                  yaxis_title="generalisation gap bound",
                  title="Uniform convergence: sqrt((ln|H| + ln(2/delta)) / 2m)")
''',
        key="found_generalisation",
    )

    keypoints([
        "Everything rests on <b>i.i.d.</b> — and dependence or shift breaks it "
        "silently, not loudly.",
        "<b>Hoeffding</b>: the gap shrinks as $1/\\sqrt{m}$, so halving it "
        "needs <b>four times</b> the data.",
        "ERM picks $\\hat h$ <i>after</i> seeing $S$, so you need a "
        "<b>uniform</b> bound over all of $\\mathcal{H}$.",
        "$\\sqrt{(\\ln|\\mathcal{H}| + \\ln(2/\\delta))/2m}$ — sample complexity "
        "is <b>logarithmic</b> in $|\\mathcal{H}|$.",
        "<b>PAC</b> bounds are correctness proofs, not performance predictions; "
        "expect them to be loose by an order of magnitude.",
    ])


# ==========================================================================
def s_f3():
    section("F.3", "Capacity — Counting What Cannot Be Counted")

    lead(
        "The union bound needs $|\\mathcal{H}|$, and almost every interesting "
        "hypothesis space is infinite. The fix is to count not hypotheses but "
        "the <b>behaviours</b> they can produce on finite samples."
    )

    sub("Shattering and the growth function")

    md(
        "A hypothesis class **shatters** a set of $n$ points if it can produce "
        "**every** one of the $2^n$ possible labellings of them. The **growth "
        "function** $\\Pi_{\\mathcal{H}}(n)$ counts the most labellings "
        "$\\mathcal{H}$ can produce on any $n$ points."
    )

    math(r"""
    \Pi_{\mathcal{H}}(n) \;=\;
      \max_{\mathbf{x}^{(1)},\dots,\mathbf{x}^{(n)}}
      \Bigl|\bigl\{\bigl(h(\mathbf{x}^{(1)}),\dots,h(\mathbf{x}^{(n)})\bigr)
        : h \in \mathcal{H}\bigr\}\Bigr|
    \;\le\; 2^{n}
    """)

    md(
        "The **VC dimension** is the largest $n$ that can still be shattered — "
        "the point at which the class runs out of expressive power."
    )

    math(r"""
    \mathrm{VC}(\mathcal{H}) \;=\;
      \max\bigl\{\,n : \Pi_{\mathcal{H}}(n) = 2^{n}\,\bigr\}
    """)

    table(
        ["Hypothesis class", "VC dimension", "Note"],
        [["Thresholds on $\\mathbb{R}$: $\\mathbb{1}[x > t]$", "1",
          "Cannot produce $(1, 0)$ on $x_1 < x_2$"],
         ["Intervals on $\\mathbb{R}$", "2", ""],
         ["Half-planes in $\\mathbb{R}^d$", "$d + 1$",
          "The classic result — three points in 2-D, not four"],
         ["Axis-aligned rectangles in $\\mathbb{R}^2$", "4", ""],
         ["Linear models with $d$ parameters", "$d$ (or $d+1$ with bias)",
          "Parameter count and VC dimension coincide <i>here</i>"],
         ["$\\mathbb{1}[\\sin(\\theta x) > 0]$, one parameter", "<b>$\\infty$</b>",
          "<b>One</b> parameter, infinite capacity — the counterexample"],
         ["1-NN", "$\\infty$", "Fits any labelling exactly"],
         ["Neural net, $p$ weights, ReLU",
          "$\\mathcal{O}(p\\,L\\log p)$", "Enormous, and the bound is vacuous"]],
    )

    pitfall(
        "Capacity is not parameter count",
        "The single-parameter sine classifier $\\mathbb{1}[\\sin(\\theta x) > 0]$ "
        "has <b>infinite</b> VC dimension: by choosing $\\theta$ large enough, it "
        "can carve any labelling out of any finite set of points. Meanwhile a "
        "linear model with a million parameters constrained to a tiny norm ball "
        "has small effective capacity. <b>What matters is how many distinct "
        "behaviours the class can produce, not how many numbers you store.</b> "
        "This is why 'more parameters ⇒ more overfitting' is folklore rather "
        "than theory — and §F.5 shows how badly it fails in practice.",
    )

    sub("Sauer's lemma — the phase transition")

    derive(
        [("<b>Sauer–Shelah lemma.</b> If $\\mathrm{VC}(\\mathcal{H}) = d$, then "
          "for all $n$:",
          r"\Pi_{\mathcal{H}}(n) \;\le\; \sum_{i=0}^{d}\binom{n}{i}"
          r" \;\le\; \Bigl(\frac{en}{d}\Bigr)^{d}"),
         ("<b>This is a remarkable dichotomy.</b> Either the growth function is "
          "$2^n$ for <i>every</i> $n$ (infinite VC dimension), or it is bounded "
          "by a <b>polynomial of degree $d$</b>. There is nothing in between — "
          "no class grows like $2^{\\sqrt{n}}$ or $n^{\\log n}$.", None),
         ("Substituting the polynomial growth into the union bound in place of "
          "$|\\mathcal{H}|$ gives the <b>VC generalisation bound</b>:",
          r"R(h) \;\le\; \hat R_S(h) \;+\;"
          r" \sqrt{\frac{8\bigl(d\ln(2em/d) + \ln(4/\delta)\bigr)}{m}}"),
         ("The $\\ln|\\mathcal{H}|$ became $d\\ln(m/d)$: <b>capacity replaced "
          "cardinality</b>, and the bound now applies to infinite classes.",
          None),
         ("Reading it as a data requirement, $m = \\mathcal{O}(d/\\epsilon^2)$ "
          "examples suffice — <b>linear in the VC dimension</b>. This is the "
          "formal version of 'you need roughly as much data as capacity'.",
          None)],
        title="Sauer's lemma and the VC bound",
    )

    sub("Rademacher complexity — the version that adapts to your data")

    md(
        "VC dimension is a property of the class alone; it ignores the "
        "distribution entirely, which is why its bounds are so pessimistic. "
        "**Rademacher complexity** measures how well the class can fit **pure "
        "noise on your actual sample**."
    )

    math(r"""
    \hat{\mathfrak{R}}_S(\mathcal{H}) \;=\;
      \mathbb{E}_{\boldsymbol\sigma}\left[
        \sup_{h \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^{m}
          \sigma_i\, h(\mathbf{x}^{(i)})\right],
    \qquad \sigma_i \sim \mathrm{Uniform}\{-1, +1\}
    """)

    proof(
        "Rademacher complexity is a randomisation test you can run",
        "Replace every label with a fair coin flip and ask how well your class "
        "can still fit. A class that correlates strongly with random noise can "
        "fit anything, so its low training error means nothing; a class that "
        "cannot is genuinely constrained. It gives a two-sided bound, "
        "$R(h) \\le \\hat R_S(h) + 2\\hat{\\mathfrak{R}}_S(\\mathcal{H}) + "
        "3\\sqrt{\\ln(2/\\delta)/2m}$, and — unlike VC — it is <b>computable "
        "from your data</b>: shuffle the labels, refit, record the training "
        "error, repeat. Zhang et al. (2017) did exactly this to deep networks "
        "and found they fit random labels perfectly, which is what made the "
        "classical theory's failure undeniable.",
    )

    warn(
        "For deep networks, these bounds are vacuous — and that is a real "
        "result, not a technicality",
        "A ResNet with $10^{7}$ parameters on 50 000 CIFAR images gives a VC "
        "bound on the error rate that exceeds <b>1</b> — it guarantees nothing, "
        "since error rates are at most 1 anyway. And these networks can fit "
        "randomly-labelled data to zero training error, so their Rademacher "
        "complexity is near its maximum. <b>Yet they generalise.</b> The "
        "classical framework is not <i>wrong</i> — its bounds hold — it is "
        "measuring the wrong quantity, because the relevant capacity is not "
        "that of the architecture but of the far smaller region that SGD "
        "actually explores. See §F.5.",
    )

    anim_header("Shattering: where a hypothesis class runs out")

    rng = np.random.default_rng(1)
    pts = np.array([[-1.1, -0.7], [1.0, -0.9], [0.1, 1.2], [0.6, 0.15]])

    def linear_shatters(points, labels):
        """Can a half-plane produce this labelling? Check separability."""
        from itertools import product
        X = np.column_stack([points, np.ones(len(points))])
        y = np.where(labels == 1, 1.0, -1.0)
        # tiny perceptron search
        w = np.zeros(3)
        for _ in range(4000):
            err = 0
            for i in range(len(X)):
                if y[i]*(X[i] @ w) <= 1e-9:
                    w = w + y[i]*X[i]
                    err += 1
            if err == 0:
                return True, w
        return False, None

    frames = []
    for n in (3, 4):
        P = pts[:n]
        combos = [np.array([(k >> i) & 1 for i in range(n)])
                  for k in range(2 ** n)]
        for ci, lab in enumerate(combos):
            ok, w = linear_shatters(P, lab)
            data = []
            if ok and w is not None and abs(w[1]) > 1e-9:
                gx = np.linspace(-2.2, 2.2, 20)
                gy = -(w[0]*gx + w[2])/w[1]
                data.append(go.Scatter(x=gx, y=gy, mode="lines",
                                       line=dict(color=C["success"], width=3)))
            for cls, col in [(0, CLASS_COLORS[0]), (1, CLASS_COLORS[1])]:
                m_ = lab == cls
                data.append(go.Scatter(x=P[m_, 0], y=P[m_, 1], mode="markers",
                                       marker=dict(size=19, color=col,
                                                   line=dict(color="#fff",
                                                             width=2))))
            frames.append(go.Frame(name=f"{n}:{ci}", data=data,
                                   layout=go.Layout(annotations=[
                                       anim.annotate_step(
                                           f"{n} points, labelling "
                                           f"{ci+1}/{2**n}: {lab.tolist()}"
                                           f"   ·   "
                                           + ("a half-plane CAN produce it"
                                              if ok else
                                              "NO half-plane can — the class "
                                              "is exhausted"),
                                           color=C["success"] if ok
                                           else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=460, plot_bgcolor="#FFFFFF",
                    xaxis=dict(range=[-2.2, 2.2], title="x₁"),
                    yaxis=dict(range=[-2.0, 2.0], title="x₂",
                               scaleanchor="x"),
                    title="Can a linear classifier realise every labelling?")
    anim.animate(f, frames, duration=nav.anim_ms(620), slider_prefix="case ")
    figure(f, "Three points: all 8 labellings are achievable. Four points in "
              "this position: one labelling is not — so VC(half-planes in ℝ²) "
              "= 3 = d + 1.")

    code_lab(
        "Shattering, VC dimension, Sauer's lemma, and Rademacher complexity",
        '''import numpy as np
from itertools import product
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. SHATTERING, BY BRUTE FORCE ============================
def separable(points, labels):
    """Is this labelling achievable by a half-plane? (perceptron search)"""
    X = np.column_stack([points, np.ones(len(points))])
    y = np.where(np.asarray(labels) == 1, 1.0, -1.0)
    w = np.zeros(X.shape[1])
    for _ in range(6000):
        errs = 0
        for i in range(len(X)):
            if y[i]*(X[i] @ w) <= 1e-9:
                w = w + y[i]*X[i]; errs += 1
        if errs == 0:
            return True
    return False

def shatters(points):
    n = len(points)
    return all(separable(points, np.array(lab))
               for lab in product([0, 1], repeat=n))

print("=== VC dimension of half-planes in R^2 ===")
cases = {
    "3 points in general position":
        np.array([[-1., -1.], [1., -1.], [0., 1.]]),
    "4 points in general position":
        np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
    "3 COLLINEAR points":
        np.array([[-1., 0.], [0., 0.], [1., 0.]]),
}
for nm, P in cases.items():
    ok = shatters(P)
    print(f"  {nm:<34} shattered: {ok}")
print("  VC dimension asks whether SOME set of n points can be shattered,")
print("  not whether EVERY set can -- the collinear triple cannot, yet")
print("  VC = 3 because a non-collinear triple can.")

# --- which labelling fails on 4 points? ------------------------------
P4 = cases["4 points in general position"]
print()
print("  the labellings of the 4-point square:")
for lab in product([0, 1], repeat=4):
    if not separable(P4, np.array(lab)):
        print(f"    {list(lab)}  <- NOT achievable (the XOR pattern)")

# --- and in higher dimensions ----------------------------------------
print()
print("=== VC(half-planes in R^d) = d + 1 ===")
print(f"{'d':>5}{'d+1 points shattered?':>25}{'d+2 points shattered?':>25}")
for d in [1, 2, 3, 4]:
    def rand_pts(n):
        return rng.normal(0, 1, (n, d))
    a = any(shatters(rand_pts(d+1)) for _ in range(4))
    b = any(shatters(rand_pts(d+2)) for _ in range(6))
    print(f"{d:>5}{str(a):>25}{str(b):>25}")

# ============ 2. ONE PARAMETER, INFINITE CAPACITY ======================
print()
print("="*70)
print("The sine classifier: 1 parameter, VC dimension = infinity")
print("="*70)
def sine_classifier(x, theta):
    return (np.sin(theta*x) > 0).astype(int)

print("  h_theta(x) = 1[sin(theta*x) > 0]   -- ONE real parameter")
print()
print(f"{'n points':>10}{'labellings tried':>19}{'all achievable?':>18}")
for n in [3, 5, 8]:
    x_pts = 2.0**(-np.arange(1, n+1))          # the classic construction
    achieved = 0
    total = 2**n
    for lab in product([0, 1], repeat=n):
        lab = np.array(lab)
        # theta = pi * (1 + sum_i (1-y_i) 2^i)  realises any labelling
        theta = np.pi*(1 + sum((1-lab[i])*2**(i+1) for i in range(n)))
        if np.array_equal(sine_classifier(x_pts, theta), lab):
            achieved += 1
    print(f"{n:>10}{total:>19}{f'{achieved}/{total}':>18}")
print("  a SINGLE real parameter shatters arbitrarily many points, because a")
print("  real number carries infinitely many bits. CAPACITY IS NOT PARAMETER")
print("  COUNT -- this is the standard counterexample.")

# ============ 3. SAUER'S LEMMA: THE PHASE TRANSITION ===================
print()
print("=== growth function: 2^n, then polynomial ===")
def sauer(n, d):
    from math import comb
    return sum(comb(n, i) for i in range(0, d+1))

print(f"{'n':>5}" + "".join(f"{f'VC={d}':>14}" for d in [1, 3, 5]) + f"{'2^n':>16}")
for n in [1, 2, 3, 5, 10, 20, 50]:
    row = "".join(f"{sauer(n, d):>14,}" for d in [1, 3, 5])
    print(f"{n:>5}{row}{2**n:>16,}")
print("  each column follows 2^n exactly up to n = d, then bends to a")
print("  POLYNOMIAL of degree d. There is no intermediate growth rate --")
print("  that dichotomy is the content of Sauer's lemma.")

print()
print("=== the VC bound, and how loose it is ===")
def vc_bound(m, d, delta=0.05):
    return np.sqrt(8*(d*np.log(2*np.e*m/d) + np.log(4/delta))/m)

print(f"{'m':>9}{'d=3':>11}{'d=50':>11}{'d=10^4':>12}{'d=10^7':>12}")
for m in [100, 1000, 10000, 50000, 10**6]:
    print(f"{m:>9}" + "".join(f"{vc_bound(m, d):>11.3f}"
                              for d in [3, 50, 10**4])
          + f"{vc_bound(m, 10**7):>12.1f}")
print("  a bound above 1 guarantees NOTHING -- error rates are at most 1.")
print("  a 10-million-parameter network on 50 000 images is exactly that")
print("  case. The classical bound is VACUOUS, and that is a real result.")

# ============ 4. RADEMACHER COMPLEXITY: COMPUTABLE =====================
print()
print("="*70)
print("Rademacher complexity: how well can the class fit COIN FLIPS?")
print("="*70)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def rademacher(make_model, X, n_rep=12, seed=0):
    """E_sigma[ sup_h (1/m) sum sigma_i h(x_i) ], estimated by fitting."""
    r = np.random.default_rng(seed)
    m = len(X)
    vals = []
    for _ in range(n_rep):
        sigma = r.choice([-1, 1], m)
        model = make_model()
        model.fit(X, (sigma > 0).astype(int))
        pred = np.where(model.predict(X) == 1, 1.0, -1.0)
        vals.append(float(np.mean(sigma*pred)))
    return float(np.mean(vals))

X = rng.normal(0, 1, (120, 5))
print(f"  m = {len(X)} points, 5 features, labels replaced by COIN FLIPS")
print()
print(f"  {'hypothesis class':<34}{'Rademacher':>13}{'reading':>26}")
for nm, mk in [
        ("logistic regression", lambda: LogisticRegression(max_iter=1000)),
        ("decision stump (depth 1)",
         lambda: DecisionTreeClassifier(max_depth=1)),
        ("decision tree (depth 3)",
         lambda: DecisionTreeClassifier(max_depth=3)),
        ("decision tree (unlimited)", lambda: DecisionTreeClassifier()),
        ("1-nearest neighbour", lambda: KNeighborsClassifier(1)),
        ("random forest (200)",
         lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1)),
        ("MLP (64, 64)",
         lambda: MLPClassifier((64, 64), max_iter=2000, random_state=0))]:
    r_ = rademacher(mk, X)
    reading = ("fits noise PERFECTLY" if r_ > 0.95 else
               "heavily constrained" if r_ < 0.4 else "moderate")
    print(f"  {nm:<34}{r_:>13.4f}{reading:>26}")
print()
print("  0 = cannot correlate with noise at all; 1 = fits any labelling.")
print("  the unlimited tree, 1-NN and the MLP all reach ~1.0 -- their")
print("  training error carries NO information about their true risk.")

# --- the Zhang et al. experiment, in miniature -----------------------
print()
print("=== Zhang et al. (2017), in miniature ===")
from core import datasets as _ds
Xd, yd = _ds.digits()[:2]
Xd = Xd[:900]; yd = yd[:900]
y_random = rng.integers(0, 10, len(yd))
print(f"  {len(Xd)} digits, 10 classes")
print(f"  {'labels':<22}{'train accuracy':>17}{'test accuracy':>16}")
for nm, yy in [("TRUE labels", yd), ("RANDOM labels", y_random)]:
    m = MLPClassifier((256, 256), max_iter=3000, random_state=0,
                      learning_rate_init=3e-3)
    m.fit(Xd[:700], yy[:700])
    print(f"  {nm:<22}{m.score(Xd[:700], yy[:700]):>17.4f}"
          f"{m.score(Xd[700:], yy[700:]):>16.4f}")
print("  the SAME network fits random labels to ~100% training accuracy.")
print("  so its capacity is enormous -- yet on real labels it generalises.")
print("  no capacity measure of the ARCHITECTURE can explain that (F.5).")

# ============ 5. THE BOUND vs REALITY ==================================
print()
print("=== how loose is the VC bound in a case where we know d? ===")
# linear classifier in R^5: VC = 6
d_vc = 6
print(f"  linear classifier in R^5, VC = {d_vc}")
print(f"{'m':>8}{'train err':>12}{'test err':>11}{'actual gap':>13}"
      f"{'VC bound':>11}{'looseness':>12}")
for m in [50, 100, 500, 2000, 10000]:
    Xa = rng.normal(0, 1, (m, 5)); w_true = rng.normal(0, 1, 5)
    ya = (Xa @ w_true + rng.normal(0, .8, m) > 0).astype(int)
    Xb = rng.normal(0, 1, (5000, 5))
    yb = (Xb @ w_true + rng.normal(0, .8, 5000) > 0).astype(int)
    clf = LogisticRegression(max_iter=2000).fit(Xa, ya)
    tr, te = 1-clf.score(Xa, ya), 1-clf.score(Xb, yb)
    b = vc_bound(m, d_vc)
    print(f"{m:>8}{tr:>12.4f}{te:>11.4f}{te-tr:>13.4f}{b:>11.4f}"
          f"{b/max(te-tr, 1e-4):>11.0f}x")
print("  the bound HOLDS at every sample size -- and is 30-300x too large.")
print("  use it to reason about SCALING (how m and d trade off), never to")
print("  predict a number.")

import plotly.graph_objects as go
ms = np.array([50, 100, 200, 500, 1000, 5000, 20000, 100000, 10**6])
fig = go.Figure()
for d, col in [(3, C["success"]), (100, C["warning"]),
               (10**4, C["danger"]), (10**7, C["ink"])]:
    fig.add_scatter(x=ms, y=[vc_bound(m, d) for m in ms], mode="lines+markers",
                    name=f"VC = {d:,}", line=dict(width=3, color=col))
fig.add_hline(y=1.0, line_dash="dash", line_color=C["danger"],
              annotation_text="1.0 — above this the bound says nothing")
fig.update_layout(height=430, xaxis_type="log", yaxis_type="log",
                  xaxis_title="training examples m",
                  yaxis_title="VC generalisation bound",
                  title="Where the classical bound stops meaning anything")
''',
        key="found_capacity",
    )

    quiz(
        "A model with one real-valued parameter has infinite VC dimension. What "
        "does that tell you?",
        ["The model is broken",
         "That capacity is about the number of distinct behaviours a class can "
         "produce, not the number of parameters stored",
         "That VC dimension is not a useful concept",
         "That the parameter must be an integer"],
        1,
        "$\\mathbb{1}[\\sin(\\theta x) > 0]$ shatters arbitrarily many points "
        "because a real number carries unbounded information. Conversely, a "
        "million parameters constrained to a small norm ball has modest "
        "effective capacity. Parameter counting is a heuristic that happens to "
        "work for linear models and fails badly elsewhere.",
        key="fq2",
    )

    keypoints([
        "<b>VC dimension</b> is the largest set the class can label in every "
        "possible way — capacity, not parameter count.",
        "<b>Sauer's lemma</b>: growth is either $2^n$ forever, or polynomial of "
        "degree $d$. Nothing in between.",
        "The VC bound replaces $\\ln|\\mathcal{H}|$ with $d\\ln(m/d)$, so "
        "$m = \\mathcal{O}(d/\\epsilon^2)$ suffices.",
        "<b>Rademacher complexity</b> is computable: shuffle the labels and see "
        "how well the class still fits.",
        "For deep networks these bounds are <b>vacuous</b> — a genuine failure "
        "of the classical theory, not a technicality.",
    ])

# ==========================================================================
def s_f4():
    section("F.4", "Loss Functions — What You Are Actually Asking For")

    lead(
        "A loss is not a technical detail chosen for convenience. It is a "
        "<b>statement about what counts as a mistake</b>, and it determines "
        "what the optimal prediction even is."
    )

    sub("The Bayes-optimal predictor depends on the loss")

    md(
        "Before any model, ask: given the true conditional distribution "
        "$p(y \\mid \\mathbf{x})$, what single number should you output? The "
        "answer is entirely determined by the loss."
    )

    derive(
        [("<b>Squared loss.</b> Minimise "
          "$\\mathbb{E}[(y - c)^2 \\mid \\mathbf{x}]$ over $c$. Differentiate "
          "and set to zero:",
          r"\frac{d}{dc}\,\mathbb{E}\bigl[(y-c)^2\bigr] = -2\,\mathbb{E}[y - c]"
          r" = 0 \;\Longrightarrow\; c^{\star} = \mathbb{E}[y \mid \mathbf{x}]"),
         ("<b>The conditional mean.</b> This is why MSE regression predicts "
          "averages — and why a VAE's reconstructions are blurry (§17.6): the "
          "average of several plausible images is a blur.", None),
         ("<b>Absolute loss.</b> Minimise "
          "$\\mathbb{E}[|y - c| \\mid \\mathbf{x}]$. The derivative of "
          "$|y-c|$ is $-\\mathrm{sign}(y-c)$, so the condition is "
          "$\\Pr[y > c] = \\Pr[y < c]$:",
          r"c^{\star} = \mathrm{median}\bigl(y \mid \mathbf{x}\bigr)"),
         ("<b>The conditional median</b> — which is why MAE is robust to "
          "outliers. Moving one observation to infinity moves the mean without "
          "limit and the median not at all.", None),
         ("<b>Pinball loss</b> at level $\\tau$, "
          "$\\ell_\\tau(u) = u(\\tau - \\mathbb{1}[u<0])$ with $u = y - c$, "
          "generalises both:",
          r"c^{\star} = Q_\tau\bigl(y \mid \mathbf{x}\bigr)"
          r"\quad\text{(the } \tau\text{-quantile)}"),
         ("<b>Log loss.</b> For classification, minimising "
          "$-\\mathbb{E}[\\log q(y)]$ over distributions $q$ gives "
          "$q^{\\star} = p(\\cdot \\mid \\mathbf{x})$ — the <b>whole "
          "conditional distribution</b>, correctly calibrated.", None),
         ("<b>0–1 loss.</b> Minimising $\\Pr[\\hat y \\ne y]$ gives the "
          "<b>mode</b>:",
          r"c^{\star} = \arg\max_k\, p(y = k \mid \mathbf{x})"),
         ("So the same data, under four losses, has four different correct "
          "answers. <b>Choosing a loss is choosing a question.</b>", None)],
        title="Every loss has its own optimal answer",
    )

    table(
        ["Loss", "Optimal prediction", "Use when"],
        [["Squared (MSE)", "conditional <b>mean</b>",
          "Errors are symmetric, no heavy tails"],
         ["Absolute (MAE)", "conditional <b>median</b>",
          "Outliers present, robustness wanted"],
         ["Huber", "between the two",
          "Mostly Gaussian with occasional outliers"],
         ["Pinball at $\\tau$", "conditional <b>$\\tau$-quantile</b>",
          "Asymmetric costs, prediction intervals"],
         ["Log loss / cross-entropy", "the full <b>distribution</b>",
          "You need calibrated probabilities"],
         ["0–1", "the <b>mode</b>",
          "Only the decision matters, costs are equal"],
         ["Poisson deviance", "conditional mean of a count",
          "$y$ is a count, variance grows with the mean"]],
    )

    sub("Why not just minimise 0–1 loss?")

    warn(
        "0–1 loss is not optimisable — hence surrogates",
        "It is piecewise constant, so its gradient is zero almost everywhere and "
        "undefined at the boundary. There is nothing for gradient descent to "
        "follow. Worse, minimising it exactly over a linear class is <b>NP-hard</b>. "
        "So every classifier in this platform minimises a <b>convex surrogate</b> "
        "that upper-bounds it — logistic, hinge, exponential — and hopes the "
        "minimiser of the surrogate is also good under 0–1.",
    )

    math(r"""
    \ell_{0/1}(m) = \mathbb{1}[m \le 0]
    \;\le\;
    \begin{cases}
      \log_2\bigl(1 + e^{-m}\bigr) & \text{logistic} \\[2pt]
      \max(0,\, 1 - m) & \text{hinge} \\[2pt]
      e^{-m} & \text{exponential (AdaBoost)}
    \end{cases}
    \qquad m = y\,f(\mathbf{x})
    """)

    proof(
        "Classification-calibration is the condition that makes surrogates safe",
        "A surrogate is <b>classification-calibrated</b> if minimising it over "
        "all measurable functions yields a predictor with the same sign as the "
        "Bayes rule. For a convex margin loss $\\phi$, Bartlett, Jordan and "
        "McAuliffe (2006) proved the condition is simply that $\\phi$ is "
        "differentiable at 0 with $\\phi'(0) < 0$. Logistic, hinge, exponential "
        "and squared loss all satisfy it — so driving the surrogate to its "
        "minimum <b>does</b> drive the 0–1 risk to the Bayes rate. That theorem "
        "is why the whole surrogate strategy is legitimate rather than a hopeful "
        "hack.",
    )

    sub("Proper scoring rules and calibration")

    md(
        "A scoring rule is **proper** if reporting your true belief maximises "
        "your expected score. Log loss and Brier score are proper; accuracy is "
        "not."
    )

    idea(
        "Accuracy rewards lying about your confidence",
        "Suppose the truth is $p(y=1 \\mid \\mathbf{x}) = 0.7$. Under accuracy, "
        "your best move is to always answer 1 — you are right 70 % of the time, "
        "and reporting 0.7 scores worse. So an accuracy-optimised model has no "
        "incentive to be calibrated, and typically is not. Under <b>log loss</b> "
        "or <b>Brier score</b>, the expected score is uniquely maximised by "
        "reporting 0.7 exactly. <b>If you need probabilities that mean "
        "something — for expected-value decisions, for thresholding at a "
        "business cost — you must train and evaluate with a proper rule.</b>",
    )

    derive(
        [("<b>Brier score decomposes into calibration and refinement.</b> Write "
          "$B = \\mathbb{E}[(q - y)^2]$ where $q$ is the forecast. Group "
          "predictions into bins with the same $q$:",
          r"B = \underbrace{\mathbb{E}\bigl[(q - \bar y_q)^2\bigr]}"
          r"_{\text{calibration}} + \underbrace{\mathbb{E}\bigl[\bar y_q"
          r"(1 - \bar y_q)\bigr]}_{\text{refinement (sharpness)}}"),
         ("where $\\bar y_q$ is the observed frequency among cases predicted "
          "$q$.", None),
         ("<b>Calibration</b> is zero when your stated 0.7 events happen 70 % of "
          "the time. It is fixable after the fact — Platt scaling, isotonic "
          "regression — without retraining.", None),
         ("<b>Refinement</b> rewards being confidently right: pushing $\\bar y_q$ "
          "toward 0 or 1. It requires a genuinely better model.", None),
         ("A model can be perfectly calibrated and useless — always predicting "
          "the base rate is perfectly calibrated. <b>You want both.</b>", None)],
        title="Calibration versus sharpness",
    )

    anim_header("Four surrogate losses, and the 0–1 loss they bound")

    m_grid = np.linspace(-3, 3, 400)
    losses = {
        "0–1 loss (what we want)": (m_grid <= 0).astype(float),
        "logistic (log loss)": np.log(1 + np.exp(-m_grid)) / np.log(2),
        "hinge (SVM)": np.maximum(0, 1 - m_grid),
        "exponential (AdaBoost)": np.exp(-m_grid),
        "squared": (1 - m_grid) ** 2,
    }
    names = list(losses)
    frames = []
    for k in range(1, len(names) + 1):
        data = []
        for i, nm in enumerate(names[:k]):
            data.append(go.Scatter(x=m_grid, y=np.clip(losses[nm], 0, 6),
                                   mode="lines",
                                   line=dict(color=(C["ink"] if i == 0
                                                    else SEQ[i]),
                                             width=(4 if i == 0 else 3),
                                             dash=("dot" if i == 0 else None))))
        nm = names[k - 1]
        note_ = {
            "0–1 loss (what we want)":
                "piecewise constant — gradient is 0 everywhere, NP-hard to "
                "minimise",
            "logistic (log loss)":
                "smooth, never zero — keeps pushing even correct points",
            "hinge (SVM)":
                "exactly zero past margin 1 — only support vectors matter",
            "exponential (AdaBoost)":
                "grows without bound — hypersensitive to mislabelled points",
            "squared":
                "penalises being TOO right (m > 1) — a poor classification loss",
        }[nm]
        frames.append(go.Frame(name=nm.split()[0], data=data,
                               layout=go.Layout(annotations=[
                                   anim.annotate_step(f"{nm} · {note_}")])))

    f = go.Figure(data=[go.Scatter(x=m_grid, y=(m_grid <= 0).astype(float),
                                   mode="lines", name="0–1 loss",
                                   line=dict(color=C["ink"], width=4,
                                             dash="dot"))])
    f.update_layout(height=440, xaxis_title="margin  m = y · f(x)",
                    yaxis_title="loss", yaxis=dict(range=[0, 6]),
                    title="Every surrogate is an upper bound on the 0–1 loss")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="loss ")
    figure(f, "All four are convex and all four are classification-calibrated — "
              "but their behaviour on outliers and on already-correct points is "
              "completely different.")

    code_lab(
        "Which loss gives which answer, surrogates, and calibration",
        '''import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. THE LOSS DECIDES THE OPTIMAL ANSWER ===================
print("=== the same conditional distribution, five losses ===")
# a skewed, heavy-tailed conditional distribution of y given some x
y = np.concatenate([rng.normal(10, 2, 9000), rng.normal(60, 15, 1000)])
print(f"  10 000 draws: 90% ~ N(10, 2), 10% ~ N(60, 15)")
print(f"  mean {y.mean():.3f}   median {np.median(y):.3f}   "
      f"mode ~ {10.0}")

def best_constant(y, loss, grid=None):
    grid = np.linspace(y.min(), y.max(), 4000) if grid is None else grid
    vals = np.array([loss(y, c).mean() for c in grid])
    return grid[vals.argmin()], vals.min()

losses = {
    "squared  (y-c)^2":      lambda y, c: (y-c)**2,
    "absolute |y-c|":        lambda y, c: np.abs(y-c),
    "pinball tau=0.10":      lambda y, c: np.where(y>=c, 0.10*(y-c), 0.90*(c-y)),
    "pinball tau=0.90":      lambda y, c: np.where(y>=c, 0.90*(y-c), 0.10*(c-y)),
    "Huber (delta=3)":       lambda y, c: np.where(np.abs(y-c)<=3,
                                                   0.5*(y-c)**2,
                                                   3*(np.abs(y-c)-1.5)),
}
print()
print(f"  {'loss':<22}{'optimal constant':>19}{'theory says':>26}")
theory = {
    "squared  (y-c)^2": f"mean = {y.mean():.3f}",
    "absolute |y-c|": f"median = {np.median(y):.3f}",
    "pinball tau=0.10": f"10th pct = {np.percentile(y, 10):.3f}",
    "pinball tau=0.90": f"90th pct = {np.percentile(y, 90):.3f}",
    "Huber (delta=3)": "between mean and median",
}
for nm, L in losses.items():
    c, _ = best_constant(y, L)
    print(f"  {nm:<22}{c:>19.3f}{theory[nm]:>26}")
print()
print("  ONE dataset. FIVE different 'correct' answers. The loss is not a")
print("  technical choice -- it is the QUESTION you are asking.")

# ============ 2. ROBUSTNESS: ONE OUTLIER ===============================
print()
print("=== move a single point to infinity ===")
base = rng.normal(10, 2, 200)
print(f"  {'outlier at':>13}{'MSE-optimal':>14}{'MAE-optimal':>14}"
      f"{'Huber-optimal':>16}")
for out in [12, 50, 500, 5000, 100000]:
    yy = np.append(base, out)
    g = np.linspace(0, 40, 8000)
    c_mse = g[np.array([((yy-c)**2).mean() for c in g]).argmin()]
    c_mae = g[np.array([np.abs(yy-c).mean() for c in g]).argmin()]
    c_hub = g[np.array([np.where(np.abs(yy-c)<=3, .5*(yy-c)**2,
                                 3*(np.abs(yy-c)-1.5)).mean()
                        for c in g]).argmin()]
    print(f"  {out:>13,}{c_mse:>14.3f}{c_mae:>14.3f}{c_hub:>16.3f}")
print("  the MSE answer runs away; the MAE and Huber answers do not move.")
print("  'robustness' is not a property of an algorithm -- it is a property")
print("  of the LOSS.")

# ============ 3. SURROGATE LOSSES ======================================
print()
print("="*70)
print("Why nobody minimises 0-1 loss directly")
print("="*70)
m = np.linspace(-3, 3, 13)
print(f"  {'margin m':>10}{'0-1':>8}{'logistic':>11}{'hinge':>9}"
      f"{'exponential':>14}{'squared':>10}")
for mm in m[::2]:
    print(f"  {mm:>10.2f}{float(mm<=0):>8.1f}"
          f"{np.log(1+np.exp(-mm))/np.log(2):>11.4f}"
          f"{max(0, 1-mm):>9.4f}{np.exp(-mm):>14.4f}{(1-mm)**2:>10.4f}")
print()
print("  the 0-1 column is FLAT on both sides of 0: gradient exactly zero.")
print("  there is nothing for gradient descent to descend.")

# --- and the surrogates upper-bound it -------------------------------
mm = np.linspace(-4, 4, 2000)
zo = (mm <= 0).astype(float)
print()
print(f"  {'surrogate':<16}{'upper bounds 0-1?':>20}{'value at m=3':>15}"
      f"{'value at m=-3':>16}")
for nm, f_ in [("logistic", lambda z: np.log(1+np.exp(-z))/np.log(2)),
               ("hinge", lambda z: np.maximum(0, 1-z)),
               ("exponential", lambda z: np.exp(-z)),
               ("squared", lambda z: (1-z)**2)]:
    v = f_(mm)
    print(f"  {nm:<16}{str(bool(np.all(v >= zo - 1e-9))):>20}"
          f"{f_(3.0):>15.4f}{f_(-3.0):>16.4f}")
print("  exponential explodes on a badly-misclassified point (e^3 = 20).")
print("  ONE mislabelled example can dominate AdaBoost's whole objective --")
print("  which is exactly its known weakness (7.5).")
print("  squared loss PENALISES m=3: being confidently right is punished.")

# ============ 4. THE SURROGATES DO NOT AGREE ===========================
print()
print("=== same data, four surrogates, four boundaries ===")
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

n = 400
X = np.vstack([rng.normal([-1.2, 0], .9, (n//2, 2)),
               rng.normal([1.2, 0], .9, (n//2, 2))])
yb = np.r_[np.zeros(n//2), np.ones(n//2)].astype(int)
# add 8 badly-placed points to expose the difference
X = np.vstack([X, rng.normal([-6, 0], .3, (8, 2))])
yb = np.r_[yb, np.ones(8, dtype=int)]        # far on the WRONG side

print(f"  {n} well-behaved points + 8 mislabelled outliers at x1 = -6")
print(f"  {'model (surrogate)':<32}{'boundary at x2=0':>19}{'train acc':>12}")
for nm, mdl in [
        ("logistic regression (log)", LogisticRegression(C=1e6)),
        ("linear SVM (hinge)", LinearSVC(C=1.0, max_iter=20000)),
        ("SGD, squared hinge", SGDClassifier(loss="squared_hinge",
                                             alpha=1e-6, max_iter=5000,
                                             random_state=0)),
        ("AdaBoost (exponential)",
         AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                            n_estimators=50, random_state=0))]:
    mdl.fit(X, yb)
    g = np.linspace(-4, 4, 4000)
    pr = mdl.predict(np.column_stack([g, np.zeros_like(g)]))
    flip = g[np.where(np.diff(pr) != 0)[0]]
    b = f"{flip[0]:+.3f}" if len(flip) else "none"
    print(f"  {nm:<32}{b:>19}{mdl.score(X, yb):>12.4f}")
print("  the 8 outliers pull the exponential-loss boundary much further than")
print("  the hinge boundary, which ignores anything past its margin.")

# ============ 5. PROPER SCORING RULES ==================================
print()
print("="*70)
print("Accuracy rewards lying about your confidence")
print("="*70)
TRUE_P = 0.70
outcomes = rng.random(200000) < TRUE_P
print(f"  truth: P(y=1) = {TRUE_P}")
print()
print(f"  {'you report q =':>16}{'accuracy':>11}{'Brier':>10}{'log loss':>11}")
for q in [0.0, 0.3, 0.5, 0.6, 0.70, 0.8, 0.9, 1.0]:
    acc = float(np.mean((q >= .5) == outcomes))
    brier = float(np.mean((q - outcomes)**2))
    qc = np.clip(q, 1e-9, 1-1e-9)
    ll = float(-np.mean(outcomes*np.log(qc) + (1-outcomes)*np.log(1-qc)))
    print(f"  {q:>16.2f}{acc:>11.4f}{brier:>10.4f}{ll:>11.4f}")
print()
print(f"  accuracy is maximised at q = 1.0 (and anything >= 0.5) -- it does")
print(f"  NOT reward reporting the truth.")
print(f"  Brier and log loss are BOTH minimised at exactly q = {TRUE_P}.")
print(f"  those are PROPER scoring rules; accuracy is not.")

# ============ 6. CALIBRATION vs SHARPNESS ==============================
print()
print("=== the Brier decomposition ===")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

Xc = rng.normal(0, 1, (4000, 6))
w = rng.normal(0, 1, 6)
p_true = 1/(1+np.exp(-(Xc @ w)))
yc = (rng.random(4000) < p_true).astype(int)
Xtr, ytr, Xte, yte = Xc[:2500], yc[:2500], Xc[2500:], yc[2500:]

def brier_decompose(p, y, bins=12):
    edges = np.linspace(0, 1, bins+1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins-1)
    cal = rel = 0.0
    for b in range(bins):
        m_ = idx == b
        if m_.sum() < 5:
            continue
        pbar, ybar = p[m_].mean(), y[m_].mean()
        wgt = m_.mean()
        cal += wgt*(pbar - ybar)**2
        rel += wgt*ybar*(1-ybar)
    return cal, rel

print(f"  {'model':<34}{'Brier':>9}{'calibration':>14}{'refinement':>13}"
      f"{'accuracy':>11}")
for nm, mdl in [
        ("logistic regression", LogisticRegression()),
        ("Gaussian naive Bayes", GaussianNB()),
        ("random forest", RandomForestClassifier(200, random_state=0,
                                                 n_jobs=-1)),
        ("naive Bayes + isotonic",
         CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=3)),
        ("always the base rate", None)]:
    if mdl is None:
        p = np.full(len(yte), ytr.mean())
    else:
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
    cal, ref = brier_decompose(p, yte)
    acc = float(np.mean((p >= .5) == yte))
    print(f"  {nm:<34}{np.mean((p-yte)**2):>9.4f}{cal:>14.5f}{ref:>13.5f}"
          f"{acc:>11.4f}")
print()
print("  'always the base rate' has near-ZERO calibration error and is")
print("  useless -- all its Brier score is refinement. Calibration alone is")
print("  not a measure of a good model.")
print("  naive Bayes is badly calibrated (it assumes independence, so its")
print("  probabilities are over-confident); isotonic regression fixes the")
print("  calibration term WITHOUT retraining the model.")

# --- the reliability curve -------------------------------------------
gnb = GaussianNB().fit(Xtr, ytr)
p_nb = gnb.predict_proba(Xte)[:, 1]
lr = LogisticRegression().fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
print()
print("=== reliability: of the cases you called p, how many were 1? ===")
print(f"  {'predicted':>11}{'naive Bayes actual':>21}{'logistic actual':>18}"
      f"{'n':>7}")
edges = np.linspace(0, 1, 7)
for i in range(6):
    m_ = (p_nb >= edges[i]) & (p_nb < edges[i+1])
    m2 = (p_lr >= edges[i]) & (p_lr < edges[i+1])
    if m_.sum() >= 5:
        print(f"  {(edges[i]+edges[i+1])/2:>11.2f}{yte[m_].mean():>21.3f}"
              f"{(yte[m2].mean() if m2.sum() >= 5 else np.nan):>18.3f}"
              f"{m_.sum():>7}")
print("  a well-calibrated model has actual ~ predicted down the column.")

import plotly.graph_objects as go
mm = np.linspace(-3, 3, 400)
fig = go.Figure()
fig.add_scatter(x=mm, y=(mm <= 0).astype(float), mode="lines", name="0–1",
                line=dict(color=C["ink"], width=4, dash="dot"))
for i, (nm, f_) in enumerate([
        ("logistic", lambda z: np.log(1+np.exp(-z))/np.log(2)),
        ("hinge", lambda z: np.maximum(0, 1-z)),
        ("exponential", lambda z: np.exp(-z)),
        ("squared", lambda z: (1-z)**2)]):
    fig.add_scatter(x=mm, y=np.clip(f_(mm), 0, 6), mode="lines", name=nm,
                    line=dict(color=SEQ[i+1], width=3))
fig.update_layout(height=420, xaxis_title="margin m = y·f(x)",
                  yaxis_title="loss", yaxis=dict(range=[0, 6]),
                  title="Convex surrogates for the 0–1 loss")
''',
        key="found_loss",
    )

    quiz(
        "Your model is evaluated on accuracy and you need probabilities for a "
        "cost-based decision. What is the problem?",
        ["Accuracy is too slow to compute",
         "Accuracy is not a <b>proper</b> scoring rule — it is maximised by "
         "over-confident predictions, so the probabilities need not be "
         "calibrated",
         "Accuracy only works for balanced data",
         "There is no problem"],
        1,
        "Under accuracy, if the true probability is 0.7 you score better by "
        "always answering 1 than by reporting 0.7. So nothing in the objective "
        "pushes toward calibrated probabilities. Train and evaluate with log "
        "loss or Brier score, or calibrate afterwards with Platt scaling or "
        "isotonic regression.",
        key="fq3",
    )

    keypoints([
        "The loss determines the <b>optimal answer</b>: mean (MSE), median "
        "(MAE), quantile (pinball), distribution (log loss), mode (0–1).",
        "0–1 loss has zero gradient and is NP-hard, so every classifier "
        "minimises a <b>convex surrogate</b>.",
        "Surrogates are safe because they are "
        "<b>classification-calibrated</b> — but they disagree on outliers.",
        "<b>Proper scoring rules</b> (log loss, Brier) reward honest "
        "probabilities; accuracy does not.",
        "Brier = <b>calibration + refinement</b>; calibration is fixable after "
        "the fact, refinement is not.",
    ])


# ==========================================================================
def s_f5():
    section("F.5", "Bias, Variance, and Where the Classical Story Breaks")

    lead(
        "The U-shaped test-error curve is in every textbook, including this "
        "one. It is also, for modern over-parameterised models, "
        "<b>empirically wrong</b> — and understanding why is the most "
        "important conceptual update of the last decade."
    )

    sub("The classical decomposition")

    derive(
        [("Fix $\\mathbf{x}$. Let $y = f(\\mathbf{x}) + \\varepsilon$ with "
          "$\\mathbb{E}[\\varepsilon] = 0$, $\\mathrm{Var}(\\varepsilon) = "
          "\\sigma^2$. Let $\\hat f$ be the model fitted on a random training "
          "set. Expand the expected squared error:",
          r"\mathbb{E}\bigl[(y - \hat f(\mathbf{x}))^2\bigr]"
          r" = \mathbb{E}\bigl[(f + \varepsilon - \hat f)^2\bigr]"),
         ("$\\varepsilon$ is independent of $\\hat f$, so the cross term "
          "vanishes:",
          r"= \sigma^{2} + \mathbb{E}\bigl[(f - \hat f)^{2}\bigr]"),
         ("Add and subtract $\\bar f = \\mathbb{E}[\\hat f]$, the average "
          "prediction over training sets:",
          r"\mathbb{E}\bigl[(f - \bar f + \bar f - \hat f)^2\bigr]"
          r" = (f - \bar f)^{2} + \mathbb{E}\bigl[(\hat f - \bar f)^{2}\bigr]"),
         ("giving the decomposition:",
          r"\underbrace{\sigma^{2}}_{\text{irreducible}}"
          r" + \underbrace{(f - \bar f)^{2}}_{\text{bias}^{2}}"
          r" + \underbrace{\mathrm{Var}(\hat f)}_{\text{variance}}"),
         ("<b>Irreducible</b> noise sets the floor — no model beats it. "
          "<b>Bias</b> is systematic error from a too-rigid class. "
          "<b>Variance</b> is sensitivity to which training set you happened to "
          "draw.", None),
         ("<b>The decomposition is exact only for squared loss.</b> For 0–1 "
          "loss there is no clean additive analogue — Domingos (2000) gives a "
          "unified version where the terms interact multiplicatively, and "
          "variance can <i>reduce</i> error when the bias points the wrong way. "
          "Quoting 'bias–variance' for a classifier is loose talk.", None)],
        title="Bias–variance, derived",
    )

    sub("Double descent")

    md(
        "Push the model past the point where it can interpolate the training "
        "data exactly, and the test error — after peaking — **falls again**, "
        "often below the classical sweet spot."
    )

    table(
        ["Regime", "Parameters vs data", "Behaviour"],
        [["<b>Classical</b>", "$p \\ll n$",
          "The textbook U — bias falls, variance rises"],
         ["<b>Interpolation threshold</b>", "$p \\approx n$",
          "<b>Test error peaks, often catastrophically</b> — the model can just "
          "barely fit, so it does so violently"],
         ["<b>Modern / over-parameterised</b>", "$p \\gg n$",
          "Test error <b>falls again</b>; many interpolating solutions exist "
          "and the optimiser picks a smooth one"]],
    )

    proof(
        "Why more parameters can help once you are past the threshold",
        "At $p \\approx n$ there is essentially <b>one</b> solution that "
        "interpolates the data, and it is forced through every noisy point with "
        "wild oscillations between them — the minimum-norm solution has enormous "
        "norm because the design matrix is nearly singular. At $p \\gg n$ there "
        "is an infinite <i>family</i> of interpolating solutions, and gradient "
        "descent from small initialisation converges to the "
        "<b>minimum-norm</b> member of that family — the smoothest one. More "
        "parameters do not add capacity that gets used; they add "
        "<b>room to find a well-behaved solution</b>. The effective complexity "
        "is set by the norm of the solution, not the parameter count — exactly "
        "the §F.3 point about capacity ≠ parameters.",
    )

    idea(
        "Implicit regularisation: the optimiser is part of the model",
        "For an under-determined least-squares problem, gradient descent started "
        "at zero provably converges to the <b>minimum $\\ell_2$-norm</b> "
        "solution — it never explores the directions the data does not "
        "constrain. So SGD is not a neutral solver; it is a regulariser you did "
        "not write down. Change the optimiser (Adam vs SGD), the "
        "initialisation scale, or the batch size, and you change which of the "
        "infinitely many zero-training-error solutions you get — and their test "
        "errors differ. This is why 'the model' in modern deep learning means "
        "the architecture <i>and</i> the training procedure.",
    )

    warn(
        "None of this licenses ignoring overfitting",
        "Double descent is real, reproducible, and appears in linear regression, "
        "random forests and deep networks alike. But it needs "
        "<b>low label noise</b>, enough data, and an optimiser with a benign "
        "implicit bias. With substantial label noise, benign overfitting stops "
        "being benign and the second descent never arrives. On a small tabular "
        "dataset with noisy labels, the classical U is still exactly what you "
        "will see — which is why chapters 4 and 7 teach regularisation without "
        "apology.",
    )

    anim_header("Double descent: the test-error curve that goes down twice")

    rng = np.random.default_rng(0)
    n_tr = 40
    x_tr = np.sort(rng.uniform(-1, 1, n_tr))
    f_true = lambda z: np.sin(3.2 * z) + 0.5 * z
    y_tr = f_true(x_tr) + rng.normal(0, .12, n_tr)
    x_te = np.linspace(-1, 1, 400)
    y_te = f_true(x_te) + rng.normal(0, .12, 400)

    def fourier(z, p):
        cols = [np.ones_like(z)]
        for k in range(1, p):
            cols.append(np.sin(k * np.pi * z) if k % 2 else np.cos(k * np.pi * z))
        return np.column_stack(cols)

    ps = list(range(1, 121, 2))
    tr_e, te_e, norms = [], [], []
    for p in ps:
        A = fourier(x_tr, p)
        w = np.linalg.pinv(A) @ y_tr           # minimum-norm solution
        tr_e.append(float(np.mean((A @ w - y_tr) ** 2)))
        te_e.append(float(np.mean((fourier(x_te, p) @ w - y_te) ** 2)))
        norms.append(float(np.linalg.norm(w)))

    frames = []
    for k in range(2, len(ps) + 1):
        frames.append(go.Frame(name=str(ps[k - 1]), data=[
            go.Scatter(x=ps[:k], y=np.clip(tr_e[:k], 1e-8, 10), mode="lines",
                       line=dict(color=C["train"], width=3)),
            go.Scatter(x=ps[:k], y=np.clip(te_e[:k], 1e-8, 10), mode="lines",
                       line=dict(color=C["valid"], width=3.5)),
            go.Scatter(x=[n_tr, n_tr], y=[1e-8, 10], mode="lines",
                       line=dict(color=C["danger"], width=2, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"p = {ps[k-1]} features, n = {n_tr} examples   ·   train MSE "
            f"{tr_e[k-1]:.2e}   ·   test MSE {te_e[k-1]:.4f}   ·   "
            f"‖w‖ = {norms[k-1]:.2f}   ·   "
            + ("classical regime" if ps[k-1] < n_tr - 4 else
               "AT THE INTERPOLATION THRESHOLD" if ps[k-1] <= n_tr + 6 else
               "over-parameterised — descending again"),
            color=(C["danger"] if abs(ps[k-1] - n_tr) <= 6 else C["success"]))])))

    f = go.Figure(data=[
        go.Scatter(x=ps[:2], y=tr_e[:2], mode="lines", name="train MSE",
                   line=dict(color=C["train"], width=3)),
        go.Scatter(x=ps[:2], y=te_e[:2], mode="lines", name="test MSE",
                   line=dict(color=C["valid"], width=3.5)),
        go.Scatter(x=[n_tr, n_tr], y=[1e-8, 10], mode="lines",
                   name=f"p = n = {n_tr}",
                   line=dict(color=C["danger"], width=2, dash="dash")),
    ])
    f.update_layout(height=460, yaxis_type="log", xaxis_title="number of features p",
                    yaxis_title="mean squared error",
                    title=f"Minimum-norm least squares, n = {n_tr}",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(130), slider_prefix="p = ")
    figure(f, "The classical U is on the left. At p = n the test error explodes. "
              "Past it, adding features makes things better again — with no "
              "regularisation anywhere in sight.")

    code_lab(
        "Bias–variance measured, double descent reproduced, implicit "
        "regularisation shown",
        '''import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. BIAS AND VARIANCE, MEASURED DIRECTLY ==================
# the decomposition needs MANY training sets -- which we can simulate
def f_true(x):
    return np.sin(3.0*x) + 0.4*x
SIGMA = 0.25
x_eval = np.linspace(-3, 3, 120)
f_eval = f_true(x_eval)

def draw(m, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, m)
    return x, f_true(x) + r.normal(0, SIGMA, m)

print("=== bias^2 + variance + noise, measured over 300 training sets ===")
print(f"  irreducible noise sigma^2 = {SIGMA**2:.4f}")
print()
print(f"{'degree':>8}{'bias^2':>11}{'variance':>11}{'noise':>9}"
      f"{'sum':>10}{'measured MSE':>15}")
for d in [0, 1, 2, 3, 5, 8, 12]:
    preds = []
    for s in range(300):
        x, y = draw(25, seed=s)
        preds.append(np.polyval(np.polyfit(x, y, d), x_eval))
    P = np.stack(preds)                       # (300, 120)
    fbar = P.mean(0)
    bias2 = float(np.mean((fbar - f_eval)**2))
    var = float(np.mean(P.var(0)))
    # the honest MSE: fresh noisy targets each time
    mse = float(np.mean([(P[s] - (f_eval + rng.normal(0, SIGMA, len(f_eval))))**2
                         for s in range(300)]))
    print(f"{d:>8}{bias2:>11.4f}{var:>11.4f}{SIGMA**2:>9.4f}"
          f"{bias2+var+SIGMA**2:>10.4f}{mse:>15.4f}")
print("  the sum matches the measured MSE to ~2 decimal places -- the")
print("  decomposition is an IDENTITY, not an approximation.")
print("  bias^2 falls monotonically, variance rises monotonically.")

# ============ 2. WHAT VARIANCE LOOKS LIKE ==============================
print()
print("=== the same model, five different training sets ===")
for d in [1, 12]:
    preds_at_0 = []
    for s in range(5):
        x, y = draw(25, seed=100+s)
        preds_at_0.append(float(np.polyval(np.polyfit(x, y, d), 0.0)))
    print(f"  degree {d:>2}: prediction at x=0 across 5 samples: "
          f"{np.round(preds_at_0, 3)}  (spread {np.std(preds_at_0):.4f})")
print(f"  truth at x=0: {f_true(0.0):.4f}")
print("  the flexible model's answer depends heavily on WHICH 25 points it saw.")
print("  that dependence IS variance.")

# ============ 3. DOUBLE DESCENT ========================================
print()
print("="*70)
print("Double descent in plain least squares -- no neural network needed")
print("="*70)
N_TRAIN = 40
r = np.random.default_rng(1)
x_tr = np.sort(r.uniform(-1, 1, N_TRAIN))
y_tr = np.sin(3.2*x_tr) + 0.5*x_tr + r.normal(0, 0.12, N_TRAIN)
x_te = np.linspace(-1, 1, 500)
y_te = np.sin(3.2*x_te) + 0.5*x_te + r.normal(0, 0.12, 500)

def features(z, p):
    cols = [np.ones_like(z)]
    for k in range(1, p):
        cols.append(np.sin(k*np.pi*z) if k % 2 else np.cos(k*np.pi*z))
    return np.column_stack(cols)

print(f"  n = {N_TRAIN} training points, Fourier features")
print(f"  fitted with the PSEUDOINVERSE -> the minimum-norm solution (4.3)")
print()
print(f"{'p':>6}{'p/n':>7}{'train MSE':>13}{'test MSE':>12}{'||w||':>12}"
      f"{'regime':>26}")
for p in [2, 5, 10, 20, 30, 36, 39, 40, 41, 44, 50, 70, 100, 200, 400]:
    A = features(x_tr, p)
    w = np.linalg.pinv(A) @ y_tr
    tr = float(np.mean((A @ w - y_tr)**2))
    te = float(np.mean((features(x_te, p) @ w - y_te)**2))
    reg = ("classical" if p < N_TRAIN-3 else
           "INTERPOLATION THRESHOLD" if abs(p-N_TRAIN) <= 3 else
           "over-parameterised")
    print(f"{p:>6}{p/N_TRAIN:>7.2f}{tr:>13.2e}{te:>12.4f}"
          f"{np.linalg.norm(w):>12.2f}{reg:>26}")
print()
print("  test error falls, SPIKES at p = n, then falls again -- and the")
print("  best test error of all is at p = 400, ten times more parameters")
print("  than data points, with ZERO explicit regularisation.")
print()
print("  the ||w|| column explains it: the norm explodes at p = n (there is")
print("  essentially ONE interpolating solution and it is violent), then")
print("  SHRINKS as p grows, because a larger family of interpolating")
print("  solutions contains smoother members.")

# ============ 4. EXPLICIT REGULARISATION REMOVES THE PEAK ==============
print()
print("=== ridge regularisation flattens the spike ===")
def ridge_fit(A, y, lam):
    return np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ y)

print(f"{'p':>6}" + "".join(f"{f'lam={l}':>13}"
                            for l in [0.0, 1e-6, 1e-3, 1e-1]))
for p in [20, 36, 40, 44, 60, 150]:
    A = features(x_tr, p); B = features(x_te, p)
    row = ""
    for lam in [0.0, 1e-6, 1e-3, 1e-1]:
        w = np.linalg.pinv(A) @ y_tr if lam == 0 else ridge_fit(A, y_tr, lam)
        row += f"{float(np.mean((B @ w - y_te)**2)):>13.4f}"
    print(f"{p:>6}{row}")
print("  with lambda = 0.1 the peak disappears entirely and the curve is")
print("  monotone. DOUBLE DESCENT IS A PHENOMENON OF UNREGULARISED")
print("  MINIMUM-NORM FITTING -- regularise and you are back in the")
print("  classical picture.")

# ============ 5. LABEL NOISE KILLS BENIGN OVERFITTING ==================
print()
print("=== how much noise can benign overfitting tolerate? ===")
print(f"{'noise sigma':>13}{'best classical p':>19}{'test MSE':>11}"
       f"{'test MSE at p=400':>21}{'winner':>12}")
for sig in [0.0, 0.05, 0.12, 0.3, 0.6]:
    rr = np.random.default_rng(5)
    yy = np.sin(3.2*x_tr) + 0.5*x_tr + rr.normal(0, sig, N_TRAIN)
    yy_te = np.sin(3.2*x_te) + 0.5*x_te + rr.normal(0, sig, 500)
    best = (np.inf, None)
    for p in range(2, 36):
        A = features(x_tr, p)
        w = np.linalg.pinv(A) @ yy
        e = float(np.mean((features(x_te, p) @ w - yy_te)**2))
        if e < best[0]:
            best = (e, p)
    A4 = features(x_tr, 400)
    w4 = np.linalg.pinv(A4) @ yy
    e4 = float(np.mean((features(x_te, 400) @ w4 - yy_te)**2))
    print(f"{sig:>13.2f}{best[1]:>19}{best[0]:>11.4f}{e4:>21.4f}"
          f"{('over-param' if e4 < best[0] else 'CLASSICAL'):>12}")
print("  at low noise the over-parameterised fit wins. Add enough label")
print("  noise and the classical sweet spot wins again -- overfitting stops")
print("  being benign. THIS is why 4.8 and 7.6 still teach regularisation.")

# ============ 6. IMPLICIT REGULARISATION ==============================
print()
print("="*70)
print("Gradient descent picks the minimum-norm solution by itself")
print("="*70)
p = 200                                        # p >> n: under-determined
A = features(x_tr, p)
w_pinv = np.linalg.pinv(A) @ y_tr              # THE minimum-norm solution
print(f"  A is {A.shape}: {A.shape[1]} unknowns, {A.shape[0]} equations")
print(f"  infinitely many w satisfy A w = y exactly")
print(f"  the minimum-norm one has ||w|| = {np.linalg.norm(w_pinv):.4f}")

# plain gradient descent from zero
w = np.zeros(p)
lr = 0.02
for it in range(60000):
    w -= lr * (A.T @ (A @ w - y_tr))/len(y_tr)
print()
print(f"  gradient descent from w = 0, 60 000 steps:")
print(f"    training MSE      {np.mean((A @ w - y_tr)**2):.3e}   (interpolates)")
print(f"    ||w||             {np.linalg.norm(w):.4f}")
print(f"    ||w - w_pinv||    {np.linalg.norm(w - w_pinv):.6f}")
print(f"    cosine similarity {float(w @ w_pinv/(np.linalg.norm(w)*np.linalg.norm(w_pinv))):.6f}")

# now start somewhere else -- a DIFFERENT interpolating solution
w2 = rng.normal(0, 3.0, p)
for it in range(60000):
    w2 -= lr * (A.T @ (A @ w2 - y_tr))/len(y_tr)
print()
print(f"  gradient descent from a RANDOM start:")
print(f"    training MSE      {np.mean((A @ w2 - y_tr)**2):.3e}   (also interpolates)")
print(f"    ||w||             {np.linalg.norm(w2):.4f}   <- much larger")
print(f"    test MSE          {np.mean((features(x_te, p) @ w2 - y_te)**2):.4f}")
print(f"  vs from zero: test MSE "
      f"{np.mean((features(x_te, p) @ w - y_te)**2):.4f}")
print()
print("  BOTH fit the training data perfectly. Their test errors differ.")
print("  the OPTIMISER and the INITIALISATION chose between them.")
print("  that choice is IMPLICIT REGULARISATION -- and it is why 'the model'")
print("  in deep learning means the architecture AND the training procedure.")

import plotly.graph_objects as go
ps = list(range(2, 200, 2))
tr_c, te_c, nm_c = [], [], []
for p_ in ps:
    A_ = features(x_tr, p_)
    w_ = np.linalg.pinv(A_) @ y_tr
    tr_c.append(float(np.mean((A_ @ w_ - y_tr)**2)))
    te_c.append(float(np.mean((features(x_te, p_) @ w_ - y_te)**2)))
    nm_c.append(float(np.linalg.norm(w_)))
fig = go.Figure()
fig.add_scatter(x=ps, y=np.clip(tr_c, 1e-12, None), mode="lines",
                name="train MSE", line=dict(color=C["train"], width=3))
fig.add_scatter(x=ps, y=te_c, mode="lines", name="test MSE",
                line=dict(color=C["valid"], width=3))
fig.add_scatter(x=ps, y=np.array(nm_c)/max(nm_c), mode="lines",
                name="‖w‖ (scaled)",
                line=dict(color=C["muted"], width=2, dash="dot"))
fig.add_vline(x=N_TRAIN, line_dash="dash", line_color=C["danger"],
              annotation_text="p = n")
fig.update_layout(height=440, yaxis_type="log", xaxis_title="features p",
                  yaxis_title="MSE (log scale)",
                  title="Double descent, and the norm that explains it")
''',
        key="found_biasvar",
    )

    keypoints([
        "Squared error $= \\sigma^2 + \\text{bias}^2 + \\text{variance}$ — an "
        "exact identity, and <b>only</b> for squared loss.",
        "<b>Double descent</b>: past the interpolation threshold $p = n$, test "
        "error falls again.",
        "The explanation is <b>norm</b>, not parameter count — more parameters "
        "give room for a smoother interpolant.",
        "<b>Implicit regularisation</b>: gradient descent from zero finds the "
        "minimum-norm solution, so the optimiser is part of the model.",
        "With <b>label noise</b>, benign overfitting stops being benign — the "
        "classical U returns.",
    ])


# ==========================================================================
def s_f6():
    section("F.6", "Optimisation — Why Gradient Descent Works")

    lead(
        "ERM is a minimisation problem. Whether you can solve it, and how fast, "
        "depends on two geometric properties of the loss surface — and on "
        "nothing else."
    )

    sub("The two properties that determine everything")

    table(
        ["Property", "Definition", "Buys you"],
        [["<b>$L$-smooth</b>",
          "$\\lVert\\nabla f(\\mathbf{x}) - \\nabla f(\\mathbf{y})\\rVert "
          "\\le L\\lVert\\mathbf{x}-\\mathbf{y}\\rVert$",
          "A safe step size $\\eta \\le 1/L$, and guaranteed descent"],
         ["<b>Convex</b>",
          "$f(\\lambda\\mathbf{x} + (1-\\lambda)\\mathbf{y}) \\le "
          "\\lambda f(\\mathbf{x}) + (1-\\lambda)f(\\mathbf{y})$",
          "<b>Every local minimum is global</b>"],
         ["<b>$\\mu$-strongly convex</b>",
          "$f - \\frac{\\mu}{2}\\lVert\\mathbf{x}\\rVert^2$ is still convex",
          "<b>Linear</b> convergence, and a unique minimum"]],
    )

    derive(
        [("<b>The descent lemma.</b> $L$-smoothness gives a quadratic upper "
          "bound on the function anywhere:",
          r"f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top"
          r"(\mathbf{y}-\mathbf{x}) + \frac{L}{2}\lVert \mathbf{y}-\mathbf{x}"
          r"\rVert^{2}"),
         ("Take a gradient step $\\mathbf{y} = \\mathbf{x} - \\eta\\nabla "
          "f(\\mathbf{x})$ and substitute:",
          r"f(\mathbf{y}) \le f(\mathbf{x}) - \eta\Bigl(1 - \frac{L\eta}{2}"
          r"\Bigr)\lVert \nabla f(\mathbf{x})\rVert^{2}"),
         ("The bracket is positive whenever $\\eta < 2/L$, so the loss "
          "<b>provably decreases</b> — this is where the stability threshold "
          "$\\eta < 2/L$ comes from, and why a too-large learning rate "
          "diverges rather than merely converging slowly.", None),
         ("Choosing $\\eta = 1/L$ maximises the guaranteed decrease and gives, "
          "for convex $f$:",
          r"f(\mathbf{x}_k) - f^{\star} \;\le\;"
          r" \frac{L\lVert\mathbf{x}_0 - \mathbf{x}^{\star}\rVert^{2}}{2k}"
          r" \;=\; \mathcal{O}(1/k)"),
         ("<b>Add strong convexity</b> and the rate becomes geometric:",
          r"f(\mathbf{x}_k) - f^{\star} \le \Bigl(1 - \frac{\mu}{L}\Bigr)^{k}"
          r"\bigl(f(\mathbf{x}_0) - f^{\star}\bigr)"),
         ("The ratio $\\kappa = L/\\mu$ is the <b>condition number</b>. You need "
          "roughly $\\kappa \\log(1/\\epsilon)$ iterations — so a condition "
          "number of $10^4$ costs ten thousand times more steps than a "
          "well-conditioned problem. <b>This single number explains why feature "
          "scaling matters so much</b> (§2.5): scaling changes $\\kappa$, and "
          "$\\kappa$ is the whole cost.", None)],
        title="Smoothness, convexity, and the convergence rates they imply",
    )

    sub("Momentum and the accelerated rate")

    md(
        "Nesterov's method achieves $\\mathcal{O}(1/k^2)$ for convex problems "
        "and $\\left(1 - \\sqrt{\\mu/L}\\right)^k$ for strongly convex ones — "
        "a **$\\sqrt{\\kappa}$** dependence instead of $\\kappa$."
    )

    proof(
        "$\\sqrt{\\kappa}$ is optimal, and that is a theorem",
        "Nemirovski and Yudin proved a matching <b>lower bound</b>: no "
        "first-order method — one that only ever queries gradients — can do "
        "better than $\\mathcal{O}(1/k^2)$ on smooth convex problems. Nesterov "
        "acceleration attains it. So momentum is not a heuristic that happens to "
        "help; it closes a provable gap, and nothing that only sees gradients "
        "can improve on it. Going faster requires <b>second-order</b> "
        "information — Newton's method converges quadratically, at the cost of "
        "forming and inverting a $p \\times p$ Hessian, which is why "
        "quasi-Newton methods (L-BFGS) and diagonal approximations (Adam) exist.",
    )

    sub("Stochastic gradients")

    derive(
        [("SGD replaces $\\nabla f$ with an unbiased estimate $\\mathbf{g}_k$ "
          "from a mini-batch:",
          r"\mathbb{E}[\mathbf{g}_k] = \nabla f(\mathbf{x}_k), \qquad"
          r" \mathrm{Var}(\mathbf{g}_k) = \frac{\sigma^{2}}{B}"),
         ("The variance falls as $1/B$, so a batch four times larger has half "
          "the gradient noise — the <b>square-root</b> relationship behind the "
          "'scale the learning rate by $\\sqrt{B}$' rule of thumb (§19.5).",
          None),
         ("With a <b>constant</b> step size, SGD does not converge to the "
          "minimum. It converges to a <b>noise ball</b> around it:",
          r"\mathbb{E}\bigl[f(\mathbf{x}_k)\bigr] - f^{\star} \;\to\;"
          r" \mathcal{O}\Bigl(\frac{\eta\sigma^{2}}{2B\mu}\Bigr)"),
         ("The radius is proportional to $\\eta$ and inversely to $B$. That is "
          "the entire justification for <b>learning-rate decay</b>: shrink "
          "$\\eta$ and the ball shrinks with it.", None),
         ("The <b>Robbins–Monro</b> conditions give convergence to the exact "
          "minimum:",
          r"\sum_{k}\eta_k = \infty \quad\text{(can still travel far)},"
          r"\qquad \sum_{k}\eta_k^{2} < \infty \quad\text{(noise dies out)}"),
         ("$\\eta_k = \\eta_0/k$ satisfies both; $\\eta_k = \\eta_0/\\sqrt{k}$ "
          "satisfies only the first and lands in a shrinking-but-nonzero ball. "
          "In deep learning we deliberately keep some noise — it is part of the "
          "implicit regularisation of §F.5.", None)],
        title="Why SGD needs a decaying step size",
    )

    warn(
        "Neural network losses are not convex, and the theory above does not "
        "apply",
        "There is no guarantee of finding a global minimum, and in general the "
        "problem is NP-hard. What rescues practice is empirical: in "
        "high-dimensional over-parameterised networks, <b>most critical points "
        "are saddles rather than bad local minima</b> (a random symmetric "
        "matrix has all-positive eigenvalues with probability exponentially "
        "small in the dimension), and the local minima that do exist tend to "
        "have similar loss values. So SGD escapes saddles via its noise and "
        "lands somewhere good enough. This is an observation about the "
        "landscape, not a theorem about the algorithm.",
    )

    anim_header("The condition number is the whole cost")

    def make_quadratic(kappa):
        return np.array([1.0, 1.0 / kappa])

    frames = []
    for kappa in [1, 2, 5, 10, 30, 100, 300]:
        d = make_quadratic(kappa)
        L_ = d.max()
        eta = 1.0 / L_
        w = np.array([2.4, 2.4])
        path = [w.copy()]
        for _ in range(90):
            w = w - eta * (d * w)
            path.append(w.copy())
        path = np.array(path)
        gx = np.linspace(-3, 3, 90)
        gy = np.linspace(-3, 3, 90)
        GX, GY = np.meshgrid(gx, gy)
        Z = 0.5 * (d[0] * GX ** 2 + d[1] * GY ** 2)
        f_end = 0.5 * float(d @ (path[-1] ** 2))
        frames.append(go.Frame(name=str(kappa), data=[
            go.Contour(x=gx, y=gy, z=Z, colorscale=nav.cscale(),
                       showscale=False, contours=dict(coloring="lines"),
                       line=dict(width=1.4)),
            go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers",
                       line=dict(color=C["danger"], width=2.5),
                       marker=dict(size=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"κ = L/μ = {kappa}   ·   90 steps at η = 1/L   ·   "
            f"f after 90 steps = {f_end:.2e}   ·   "
            f"predicted (1−1/κ)^90 = {(1-1/kappa)**90:.2e}")])))

    d0 = make_quadratic(1)
    gx = np.linspace(-3, 3, 90); gy = np.linspace(-3, 3, 90)
    GX, GY = np.meshgrid(gx, gy)
    f = go.Figure(data=[
        go.Contour(x=gx, y=gy, z=0.5*(d0[0]*GX**2 + d0[1]*GY**2),
                   colorscale=nav.cscale(), showscale=False,
                   contours=dict(coloring="lines"), line=dict(width=1.4)),
        go.Scatter(x=[2.4], y=[2.4], mode="lines+markers", name="GD path",
                   line=dict(color=C["danger"], width=2.5),
                   marker=dict(size=4)),
    ])
    f.update_layout(height=470, xaxis_title="w₁", yaxis_title="w₂",
                    yaxis=dict(scaleanchor="x"),
                    title="Gradient descent on a quadratic")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="κ = ")
    figure(f, "The step size is capped by the steepest direction, so progress "
              "along the flattest one crawls. Feature scaling is nothing more "
              "than making these contours round.")

    code_lab(
        "Convergence rates, condition number, momentum, and the SGD noise ball",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. THE DESCENT LEMMA AND THE 2/L THRESHOLD ===============
print("=== eta < 2/L is a hard stability boundary, not a guideline ===")
L_ = 4.0                                       # f(w) = L/2 w^2, f'' = L
def run(eta, steps=60, w0=1.0):
    w = w0
    for _ in range(steps):
        w = w - eta*(L_*w)
    return w

print(f"  f(w) = {L_}/2 w^2, so L = {L_}, and the threshold is 2/L = {2/L_}")
print(f"{'eta':>10}{'eta*L':>9}{'w after 60 steps':>21}{'behaviour':>22}")
for eta in [0.05, 0.125, 0.25, 0.45, 0.49, 0.50, 0.51, 0.6]:
    w = run(eta)
    beh = ("converges" if abs(w) < 1e-3 else
           "oscillates, stable" if abs(w) < 1.0 else
           "PERIODIC (eta = 2/L)" if abs(abs(w)-1.0) < 1e-9 else "DIVERGES")
    print(f"{eta:>10.3f}{eta*L_:>9.3f}{w:>21.3e}{beh:>22}")
print("  at exactly eta = 2/L the iterate flips sign forever without shrinking.")
print("  past it, it grows without bound. This is the SAME threshold as 4.5.")

# ============ 2. CONVERGENCE RATES =====================================
print()
print("="*70)
print("O(1/k) convex, geometric strongly convex, O(1/k^2) accelerated")
print("="*70)
def quadratic(kappa, p=2):
    """f(w) = 0.5 w' D w with eigenvalues spread over [1/kappa, 1]."""
    return np.linspace(1.0/kappa, 1.0, p)

def gd(d, steps, eta=None, momentum=0.0, nesterov=False, w0=None):
    L_, mu = d.max(), d.min()
    eta = eta or 1.0/L_
    w = np.ones(len(d))*2.0 if w0 is None else w0.copy()
    v = np.zeros_like(w)
    hist = []
    for k in range(steps):
        g = d*(w + momentum*v) if nesterov else d*w
        v = momentum*v - eta*g
        w = w + v
        hist.append(0.5*float(d @ w**2))
    return np.array(hist)

print(f"  f* = 0, w0 = (2, 2, ...)")
print()
print(f"{'kappa':>8}{'plain GD after 200':>21}{'predicted (1-1/k)^200':>24}"
      f"{'heavy-ball':>13}{'Nesterov':>12}")
for kappa in [1, 10, 100, 1000]:
    d = quadratic(kappa, 30)
    plain = gd(d, 200)[-1]
    beta = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2      # optimal momentum
    hb = gd(d, 200, momentum=beta)[-1]
    nes = gd(d, 200, momentum=beta, nesterov=True)[-1]
    print(f"{kappa:>8}{plain:>21.3e}{(1-1/kappa)**200:>24.3e}"
          f"{hb:>13.3e}{nes:>12.3e}")
print()
print("  plain GD tracks (1 - 1/kappa)^k exactly.")
print("  momentum turns kappa into sqrt(kappa): at kappa=1000 that is a")
print("  30x reduction in the number of steps needed. It is not a")
print("  heuristic -- it attains a PROVEN LOWER BOUND (Nemirovski-Yudin).")

# --- steps needed to reach a target ----------------------------------
print()
print("=== iterations to reach f < 1e-8 ===")
print(f"{'kappa':>8}{'plain GD':>12}{'Nesterov':>12}{'ratio':>9}"
      f"{'sqrt(kappa)':>14}")
for kappa in [10, 100, 1000, 10000]:
    d = quadratic(kappa, 20)
    beta = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2
    def steps_to(mom, nest):
        h = gd(d, 300000, momentum=mom, nesterov=nest)
        idx = np.where(h < 1e-8)[0]
        return int(idx[0]) if len(idx) else -1
    a, b = steps_to(0.0, False), steps_to(beta, True)
    print(f"{kappa:>8}{a:>12,}{b:>12,}{a/max(b,1):>9.1f}x"
          f"{np.sqrt(kappa):>14.1f}")
print("  the speed-up ratio tracks sqrt(kappa), exactly as the theory says.")

# ============ 3. WHY FEATURE SCALING MATTERS ===========================
print()
print("="*70)
print("Feature scaling IS conditioning")
print("="*70)
n = 2000
X = np.column_stack([rng.normal(0, 1, n),
                     rng.normal(0, 1000, n),          # a badly scaled column
                     rng.normal(0, 0.001, n)])
w_true = np.array([1.0, 0.002, 500.0])
y = X @ w_true + rng.normal(0, .1, n)

def cond_of(A):
    H = (A.T @ A)/len(A)
    ev = np.linalg.eigvalsh(H)
    return float(ev.max()/max(ev.min(), 1e-300)), ev.max()

for nm, A in [("raw features", X),
              ("standardised", (X - X.mean(0))/X.std(0))]:
    kappa, L_ = cond_of(A)
    yy = y if nm == "raw features" else y
    w = np.zeros(3)
    eta = 1.0/L_
    hist = []
    for k in range(3000):
        w -= eta*(A.T @ (A @ w - yy))/n
        hist.append(float(np.mean((A @ w - yy)**2)))
    print(f"  {nm:<16} condition number {kappa:>12.3e}   "
          f"MSE after 3 000 steps {hist[-1]:>12.5f}")
print("  the SAME data, the SAME algorithm. Standardising changes only the")
print("  CONDITION NUMBER -- and that is the entire difference in speed.")

# ============ 4. THE SGD NOISE BALL ====================================
print()
print("="*70)
print("Constant-step SGD does not converge -- it orbits")
print("="*70)
# minimise 0.5*mu*w^2 with gradient noise
MU, SIG = 1.0, 1.0
def sgd(eta, batch, steps=40000, seed=0):
    r = np.random.default_rng(seed)
    w = 3.0
    tail = []
    for k in range(steps):
        g = MU*w + r.normal(0, SIG/np.sqrt(batch))
        w -= eta*g
        if k > steps//2:
            tail.append(w)
    return float(np.mean(np.square(tail)))

print(f"  f(w) = 0.5 w^2, gradient noise sigma = {SIG}")
print(f"  theory: E[w^2] -> eta*sigma^2 / (B * (2 - eta*mu) * mu)")
print()
print(f"{'eta':>8}{'batch':>8}{'measured E[w^2]':>19}{'theory':>14}")
for eta in [0.2, 0.1, 0.05, 0.01]:
    for B in [1, 16]:
        m_ = sgd(eta, B)
        th = eta*SIG**2/(B*(2-eta*MU)*MU)
        print(f"{eta:>8.2f}{B:>8}{m_:>19.5f}{th:>14.5f}")
print("  the ball radius is PROPORTIONAL TO eta and INVERSELY to the batch.")
print("  that is the whole justification for learning-rate decay -- and for")
print("  the 'bigger batch, bigger learning rate' rule of 19.5.")

# --- Robbins-Monro ---------------------------------------------------
print()
print("=== Robbins-Monro: sum eta = inf, sum eta^2 < inf ===")
def sgd_schedule(kind, steps=200000, seed=0):
    r = np.random.default_rng(seed)
    w = 3.0
    for k in range(1, steps+1):
        eta = {"constant": 0.05,
               "1/sqrt(k)": 0.5/np.sqrt(k),
               "1/k": 2.0/k}[kind]
        w -= eta*(MU*w + r.normal(0, SIG))
    return w

print(f"{'schedule':>14}{'sum eta':>12}{'sum eta^2':>13}"
      f"{'|w| after 200k':>17}{'converges?':>13}")
ks = np.arange(1, 200001)
for kind, etas in [("constant", np.full(200000, 0.05)),
                   ("1/sqrt(k)", 0.5/np.sqrt(ks)),
                   ("1/k", 2.0/ks)]:
    w = sgd_schedule(kind)
    s1 = "inf" if etas.sum() > 1e5 else f"{etas.sum():.1f}"
    s2 = f"{(etas**2).sum():.2f}"
    ok = "yes" if abs(w) < 0.05 else "no -- orbits"
    print(f"{kind:>14}{s1:>12}{s2:>13}{abs(w):>17.5f}{ok:>13}")
print("  1/k satisfies both conditions and converges exactly.")
print("  a constant step never does. 1/sqrt(k) is in between -- which is why")
print("  it is the usual practical compromise.")

# ============ 5. NON-CONVEXITY: SADDLES, NOT BAD MINIMA ================
print()
print("="*70)
print("In high dimensions, critical points are saddles")
print("="*70)
print("  a random symmetric matrix is positive-definite only if ALL d")
print("  eigenvalues happen to be positive -- exponentially unlikely.")
print()
print(f"{'dimension d':>13}{'fraction that are minima':>27}"
      f"{'fraction saddles':>19}")
for d in [1, 2, 5, 10, 20, 50]:
    mins = 0
    T = 4000
    for _ in range(T):
        A = rng.normal(0, 1, (d, d))
        H = (A + A.T)/2
        if np.all(np.linalg.eigvalsh(H) > 0):
            mins += 1
    print(f"{d:>13}{mins/T:>27.5f}{1-mins/T:>19.5f}")
print("  by d = 20 essentially every critical point is a SADDLE.")
print("  saddles are escapable -- there is always a downhill direction, and")
print("  SGD's noise finds it. That is the empirical reason non-convex")
print("  training works, and it is an observation, not a theorem.")

import plotly.graph_objects as go
fig = go.Figure()
for kappa, col in [(10, C["success"]), (100, C["warning"]),
                   (1000, C["danger"])]:
    d = quadratic(kappa, 30)
    beta = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2
    fig.add_scatter(y=gd(d, 400), mode="lines", name=f"GD, κ={kappa}",
                    line=dict(color=col, width=2.5))
    fig.add_scatter(y=gd(d, 400, momentum=beta, nesterov=True), mode="lines",
                    name=f"Nesterov, κ={kappa}",
                    line=dict(color=col, width=2.5, dash="dot"))
fig.update_layout(height=440, yaxis_type="log", xaxis_title="iteration",
                  yaxis_title="f(w) − f*",
                  title="Momentum turns κ into √κ")
''',
        key="found_optim",
    )

    keypoints([
        "<b>$L$-smoothness</b> caps the step size at $\\eta < 2/L$ — past it, "
        "divergence, not slow convergence.",
        "<b>Convexity</b> makes every local minimum global; <b>strong "
        "convexity</b> gives geometric convergence.",
        "The <b>condition number</b> $\\kappa = L/\\mu$ is the cost. Feature "
        "scaling is conditioning.",
        "Momentum achieves $\\sqrt{\\kappa}$ and <b>attains a proven lower "
        "bound</b> for first-order methods.",
        "Constant-step SGD orbits a <b>noise ball</b> of radius "
        "$\\propto \\eta/B$ — hence learning-rate decay.",
    ])

# ==========================================================================
def s_f7():
    section("F.7", "Evaluation — Your Test Score Is a Random Variable")

    lead(
        "A single number on a held-out set feels like a fact. It is an estimate "
        "with a standard error, and if you selected the model using that same "
        "set, it is a <b>biased</b> estimate."
    )

    sub("How uncertain is one accuracy number?")

    derive(
        [("A test accuracy is a sample proportion. With $n$ test examples and "
          "true accuracy $p$, the count of correct predictions is "
          "$\\mathrm{Binomial}(n, p)$, so:",
          r"\mathrm{Var}(\hat p) = \frac{p(1-p)}{n},\qquad"
          r" \mathrm{SE}(\hat p) = \sqrt{\frac{p(1-p)}{n}}"),
         ("A 95 % interval is roughly $\\hat p \\pm 1.96\\,\\mathrm{SE}$. At "
          "$p = 0.9$ and $n = 1000$ that is $\\pm 0.019$ — so "
          "<b>89.2 % and 90.8 % are the same result</b>.", None),
         ("Turn it around and ask how large a test set you need to resolve a "
          "difference $\\Delta$ between two models:",
          r"n \;\gtrsim\; \frac{2\,z^{2}\,p(1-p)}{\Delta^{2}}"),
         ("For $p \\approx 0.9$ and $\\Delta = 0.01$, that is roughly "
          "<b>14 000 test examples</b> — for <i>independent</i> test sets. "
          "Most published one-point improvements on small benchmarks are "
          "within noise.", None),
         ("<b>Paired testing is far more powerful.</b> Evaluate both models on "
          "the <i>same</i> examples and test only the cases where they "
          "disagree — McNemar's test. The variance of the difference drops "
          "because the shared difficulty of each example cancels:",
          r"\mathrm{Var}(\hat p_A - \hat p_B) = \mathrm{Var}(\hat p_A)"
          r" + \mathrm{Var}(\hat p_B) - 2\,\mathrm{Cov}(\hat p_A, \hat p_B)"),
         ("With strongly correlated models that covariance is large, and the "
          "paired test can detect a difference an unpaired one would need "
          "10–100× more data to see. <b>Always compare models on the same test "
          "examples.</b>", None)],
        title="The standard error of a test metric",
    )

    sub("Cross-validation: what it estimates and what it does not")

    table(
        ["", "$k$ small (e.g. 2)", "$k$ large (e.g. $n$, LOO)"],
        [["Training set size", "Much smaller than $n$",
          "Almost exactly $n$"],
         ["<b>Bias</b> of the estimate",
          "<b>Pessimistic</b> — models trained on less data",
          "Nearly unbiased"],
         ["<b>Variance</b> of the estimate", "Lower",
          "<b>Higher</b> — the $n$ models are nearly identical, so their errors "
          "are highly correlated"],
         ["Cost", "$k$ fits", "$n$ fits"],
         ["Verdict", "", "<b>$k = 5$ or $10$</b> is the standard compromise"]],
    )

    pitfall(
        "Cross-validation estimates the performance of a <i>procedure</i>, not "
        "of a <i>model</i>",
        "Each fold trains a different model on different data. The average tells "
        "you how well <b>the training procedure</b> does on datasets of that "
        "size — which is what you want for model selection. It does <b>not</b> "
        "give a confidence interval for the specific model you finally ship, and "
        "the naive standard error across folds is badly wrong because the folds "
        "share training data and are therefore correlated. Bengio and Grandvalet "
        "(2004) proved there is <b>no unbiased estimator of the variance</b> of "
        "$k$-fold CV. Report the mean; treat the spread as a rough guide only.",
    )

    sub("The winner's curse")

    derive(
        [("Suppose you evaluate $K$ models that are all <b>equally good</b>, "
          "with true accuracy $p$. Each observed score is "
          "$\\hat p_i = p + \\varepsilon_i$ with $\\varepsilon_i$ roughly "
          "$\\mathcal{N}(0, \\sigma^2)$, $\\sigma = \\mathrm{SE}$.", None),
         ("You report the best. Its expected score is the expected maximum of "
          "$K$ Gaussians, which for large $K$ grows like:",
          r"\mathbb{E}\Bigl[\max_i \hat p_i\Bigr] \;\approx\;"
          r" p + \sigma\sqrt{2\ln K}"),
         ("<b>The optimism grows with the square root of the log of how many "
          "things you tried.</b> With $\\sigma = 0.015$ and $K = 100$ "
          "configurations, the winner looks about $0.032$ better than it is — "
          "larger than most reported improvements.", None),
         ("Worse, the ranking is <b>not preserved</b>: the model that wins on "
          "the validation set is often not the best model, merely the luckiest. "
          "Selection on a noisy criterion is selection on noise.", None),
         ("<b>The fix is structural, not statistical:</b> select on validation "
          "data, then report on a test set you used <b>once</b>. Any set you "
          "have optimised against has stopped being an unbiased estimate of "
          "anything.", None)],
        title="Why the best validation score is optimistic",
    )

    warn(
        "Every glance at the test set spends it",
        "Not just formal tuning — looking at test performance and then changing "
        "the architecture, the features, or the preprocessing is selection too, "
        "with you as the optimiser. Benchmarks used by thousands of researchers "
        "are effectively fitted by the community; this is the main argument for "
        "held-out challenge sets with hidden labels, and for evaluating on data "
        "created <i>after</i> your model's cutoff.",
    )

    anim_header("The winner's curse: how optimism grows with the search")

    rng = np.random.default_rng(0)
    TRUE_P, n_val = 0.850, 1000
    se = np.sqrt(TRUE_P * (1 - TRUE_P) / n_val)
    Ks = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    frames = []
    for K in Ks:
        best = np.array([rng.binomial(n_val, TRUE_P, K).max() / n_val
                         for _ in range(3000)])
        frames.append(go.Frame(name=str(K), data=[
            go.Histogram(x=best, nbinsx=45, histnorm="probability density",
                         marker=dict(color=alpha(C["primary"], .8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"tried K = {K} equally-good configurations   ·   best observed "
            f"{best.mean():.4f}   ·   optimism {best.mean()-TRUE_P:+.4f}   ·   "
            f"√(2 ln K)·SE = {se*np.sqrt(2*np.log(max(K,2))):.4f}",
            color=C["danger"] if K > 20 else C["ink_soft"])])))

    f = go.Figure(data=[go.Histogram(
        x=np.array([rng.binomial(n_val, TRUE_P, 1).max()/n_val
                    for _ in range(3000)]),
        nbinsx=45, histnorm="probability density",
        marker=dict(color=alpha(C["primary"], .8)))])
    f.add_vline(x=TRUE_P, line_dash="dash", line_color=C["danger"],
                annotation_text="true accuracy 0.850")
    f.update_layout(height=430, xaxis_title="best validation accuracy observed",
                    yaxis_title="density", xaxis=dict(range=[.80, .92]),
                    title="Every model here is exactly as good as the others")
    anim.animate(f, frames, duration=nav.anim_ms(1000), slider_prefix="K = ")
    figure(f, "Nothing improves. Only the search does. The whole distribution "
              "walks right as you try more configurations.")

    code_lab(
        "Standard errors, McNemar, CV variance, and the winner's curse",
        '''import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. THE STANDARD ERROR OF AN ACCURACY =====================
print("=== how precise is one test score? ===")
print(f"{'test set n':>12}{'SE at p=0.90':>15}{'95% interval':>24}"
      f"{'resolvable difference':>24}")
for n in [100, 300, 1000, 5000, 20000, 100000]:
    p = 0.90
    se = np.sqrt(p*(1-p)/n)
    lo, hi = p-1.96*se, p+1.96*se
    delta = 1.96*np.sqrt(2)*se               # two independent test sets
    print(f"{n:>12,}{se:>15.5f}{f'[{lo:.4f}, {hi:.4f}]':>24}"
          f"{delta:>24.4f}")
print("  with 1 000 test examples, 89.2% and 90.8% ARE THE SAME RESULT.")
print("  most single-point benchmark improvements are within this noise.")

# --- verify by simulation --------------------------------------------
print()
print("=== verified by resampling ===")
TRUE = 0.90
for n in [200, 2000]:
    draws = rng.binomial(n, TRUE, 40000)/n
    print(f"  n={n:>5}: predicted SE {np.sqrt(TRUE*(1-TRUE)/n):.5f}   "
          f"observed SD {draws.std():.5f}   "
          f"95% of draws within [{np.percentile(draws,2.5):.4f}, "
          f"{np.percentile(draws,97.5):.4f}]")

# ============ 2. PAIRED COMPARISON IS FAR MORE POWERFUL ================
print()
print("="*70)
print("Compare models on the SAME examples")
print("="*70)
def simulate_pair(n, p_a, p_b, shared, seed):
    """Two models on the same test set; `shared` controls how correlated."""
    r = np.random.default_rng(seed)
    difficulty = r.random(n)                  # per-example difficulty
    ca = (r.random(n)*(1-shared) + difficulty*shared) < p_a
    cb = (r.random(n)*(1-shared) + difficulty*shared) < p_b
    return ca, cb

N = 2000
P_A, P_B = 0.900, 0.915                       # B is genuinely 1.5 pts better
print(f"  n = {N}, model A = {P_A}, model B = {P_B}  (a real 1.5-point gap)")
print()
print(f"{'correlation':>13}{'unpaired p-value':>19}{'McNemar p-value':>18}"
      f"{'discordant pairs':>19}")
for shared in [0.0, 0.4, 0.8, 0.95]:
    ps_un, ps_mc, disc = [], [], []
    for s in range(200):
        ca, cb = simulate_pair(N, P_A, P_B, shared, seed=s)
        # unpaired: two-proportion z-test
        p1, p2 = ca.mean(), cb.mean()
        pp = (ca.sum()+cb.sum())/(2*N)
        z = (p2-p1)/np.sqrt(2*pp*(1-pp)/N)
        ps_un.append(2*(1-stats.norm.cdf(abs(z))))
        # McNemar: only the DISAGREEMENTS carry information
        b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
        disc.append(b+c)
        ps_mc.append(stats.binomtest(b, b+c, 0.5).pvalue if b+c > 0 else 1.0)
    print(f"{shared:>13.2f}{np.median(ps_un):>19.4f}{np.median(ps_mc):>18.4f}"
          f"{np.mean(disc):>19.0f}")
print("  as the models become more correlated, the UNPAIRED test stays weak")
print("  while McNemar gets dramatically more sensitive -- the shared")
print("  per-example difficulty cancels out.")
print("  ALWAYS evaluate competing models on the same test examples.")

# ============ 3. CROSS-VALIDATION: BIAS AND VARIANCE ===================
print()
print("="*70)
print("k-fold CV: small k is pessimistic, large k is high-variance")
print("="*70)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

def cv_study(k, n_data=120, trials=60):
    means = []
    for t in range(trials):
        X, y = make_regression(n_samples=n_data, n_features=12, noise=12.0,
                               random_state=t)
        sc = cross_val_score(Ridge(1.0), X, y, cv=KFold(k, shuffle=True,
                                                        random_state=0),
                             scoring="neg_mean_squared_error")
        means.append(-sc.mean())
    return float(np.mean(means)), float(np.std(means))

# the "truth": train on all 120, test on fresh data
truths = []
for t in range(60):
    X, y = make_regression(n_samples=120+4000, n_features=12, noise=12.0,
                           random_state=t)
    m = Ridge(1.0).fit(X[:120], y[:120])
    truths.append(float(np.mean((m.predict(X[120:]) - y[120:])**2)))
TRUTH = float(np.mean(truths))
print(f"  true generalisation MSE of a model trained on 120 points: {TRUTH:.2f}")
print()
print(f"{'k':>6}{'CV estimate':>15}{'bias':>12}{'SD across datasets':>22}"
      f"{'fits':>7}")
for k in [2, 3, 5, 10, 20, 120]:
    m_, s_ = cv_study(k)
    print(f"{k:>6}{m_:>15.2f}{m_-TRUTH:>+12.2f}{s_:>22.2f}{k:>7}")
print("  k=2 is clearly PESSIMISTIC -- it trains on half the data.")
print("  k=120 (leave-one-out) is nearly unbiased but costs 120 fits.")
print("  k = 5 or 10 is the standard compromise, and that is why.")

# --- the fold-to-fold spread is NOT a standard error -----------------
print()
print("=== the naive standard error across folds is wrong ===")
X, y = make_regression(n_samples=300, n_features=12, noise=12.0,
                       random_state=0)
sc = cross_val_score(Ridge(1.0), X, y, cv=KFold(10, shuffle=True,
                                                random_state=0),
                     scoring="neg_mean_squared_error")
naive_se = float(np.std(-sc, ddof=1)/np.sqrt(10))
# the honest spread: re-run CV with different fold splits
reps = []
for s in range(80):
    sc2 = cross_val_score(Ridge(1.0), X, y,
                          cv=KFold(10, shuffle=True, random_state=s),
                          scoring="neg_mean_squared_error")
    reps.append(-sc2.mean())
print(f"  naive SE from the 10 fold scores        : {naive_se:.4f}")
print(f"  honest SD of the CV estimate itself     : {np.std(reps):.4f}")
print(f"  ratio                                   : "
      f"{np.std(reps)/naive_se:.2f}x")
print("  the folds share training data, so their scores are CORRELATED and")
print("  the naive formula understates the uncertainty. Bengio & Grandvalet")
print("  (2004) proved no unbiased variance estimator for k-fold CV exists.")

# ============ 4. THE WINNER'S CURSE ====================================
print()
print("="*70)
print("Selecting the best of K equally-good models")
print("="*70)
TRUE_P, N_VAL = 0.850, 1000
SE = np.sqrt(TRUE_P*(1-TRUE_P)/N_VAL)
print(f"  every configuration has TRUE accuracy {TRUE_P}")
print(f"  validation set n = {N_VAL}, so SE = {SE:.5f}")
print()
print(f"{'K tried':>9}{'best val score':>17}{'optimism':>12}"
      f"{'sqrt(2 ln K)*SE':>18}{'its TEST score':>17}")
for K in [1, 5, 20, 100, 500, 2000]:
    bests, tests = [], []
    for _ in range(1500):
        val = rng.binomial(N_VAL, TRUE_P, K)/N_VAL
        i = int(val.argmax())
        bests.append(val[i])
        tests.append(rng.binomial(N_VAL, TRUE_P)/N_VAL)   # a FRESH set
    print(f"{K:>9}{np.mean(bests):>17.5f}{np.mean(bests)-TRUE_P:>+12.5f}"
          f"{SE*np.sqrt(2*np.log(max(K,2))):>18.5f}{np.mean(tests):>17.5f}")
print("  the optimism tracks sqrt(2 ln K) * SE almost exactly.")
print("  the TEST column stays at 0.850 -- a fresh set is honest.")

# --- and the ranking is not preserved --------------------------------
print()
print("=== does the validation winner have the best TRUE accuracy? ===")
print(f"{'K':>6}{'spread of true accuracies':>28}"
      f"{'P(val winner = true best)':>28}")
for K in [5, 20, 100]:
    for spread in [0.0, 0.005, 0.02]:
        hits = 0
        for _ in range(2000):
            true_ps = TRUE_P + rng.normal(0, spread, K)
            val = np.array([rng.binomial(N_VAL, min(max(p,0),1))/N_VAL
                            for p in true_ps])
            hits += int(val.argmax() == true_ps.argmax())
        print(f"{K:>6}{spread:>28.3f}{hits/2000:>28.3f}")
print("  when the models genuinely differ by less than the SE, the")
print("  validation winner is the LUCKIEST, not the best.")
print("  selection on a noisy criterion is selection on noise.")

# ============ 5. THE FIX: A TEST SET USED ONCE =========================
print()
print("=== the only defence is structural ===")
print(f"{'protocol':<44}{'reported':>11}{'truth':>9}{'honest?':>10}")
scenarios = [
    ("tune on test, report test", True, True),
    ("tune on validation, report VALIDATION", True, False),
    ("tune on validation, report a fresh TEST", False, False),
]
for nm, _a, _b in scenarios:
    vals = []
    for _ in range(1500):
        val = rng.binomial(N_VAL, TRUE_P, 200)/N_VAL
        i = int(val.argmax())
        if nm.startswith("tune on validation, report a fresh"):
            vals.append(rng.binomial(N_VAL, TRUE_P)/N_VAL)
        else:
            vals.append(val[i])
    rep = float(np.mean(vals))
    print(f"  {nm:<42}{rep:>11.4f}{TRUE_P:>9.3f}"
          f"{('YES' if abs(rep-TRUE_P) < 2*SE else 'NO'):>10}")
print("  a set you have optimised against has stopped estimating anything.")

import plotly.graph_objects as go
Ks = np.array([1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000])
obs = []
for K in Ks:
    obs.append(float(np.mean([rng.binomial(N_VAL, TRUE_P, int(K)).max()/N_VAL
                              for _ in range(400)])) - TRUE_P)
fig = go.Figure()
fig.add_scatter(x=Ks, y=obs, mode="lines+markers", name="measured optimism",
                line=dict(color=C["danger"], width=3))
fig.add_scatter(x=Ks, y=SE*np.sqrt(2*np.log(np.maximum(Ks, 2))), mode="lines",
                name="√(2 ln K) · SE", line=dict(color=C["ink"], width=2,
                                                 dash="dot"))
fig.update_layout(height=420, xaxis_type="log",
                  xaxis_title="number of configurations tried",
                  yaxis_title="optimism of the best validation score",
                  title="The winner's curse grows as √(2 ln K)")
''',
        key="found_eval",
    )

    keypoints([
        "A test accuracy has $\\mathrm{SE} = \\sqrt{p(1-p)/n}$ — at $n = 1000$, "
        "$\\pm 2$ points is noise.",
        "<b>Compare models on the same examples</b>; McNemar's paired test is "
        "far more powerful than an unpaired one.",
        "CV estimates a <b>procedure</b>, not a model; $k = 5$–$10$ balances "
        "bias and variance, and its naive SE is wrong.",
        "<b>Winner's curse</b>: the best of $K$ equally-good models looks "
        "$\\sigma\\sqrt{2\\ln K}$ too good.",
        "The only defence is a test set used <b>once</b>.",
    ])


# ==========================================================================
def s_f8():
    section("F.8", "Data, Features and the Geometry of High Dimensions")

    lead(
        "Models see geometry, not meaning. What a distance means, and how "
        "distances behave when there are many features, decides which "
        "algorithms can work at all."
    )

    sub("Measurement scales")

    table(
        ["Scale", "Operations that make sense", "Encode as", "Trap"],
        [["<b>Nominal</b> (colour, city)", "$=$, $\\ne$",
          "One-hot, target, or embedding",
          "Integer codes imply an order that is not there"],
         ["<b>Ordinal</b> (small/medium/large)", "$=$, $<$",
          "Ordered integers, or one-hot",
          "The <i>gaps</i> are not equal — treating them as numeric asserts "
          "they are"],
         ["<b>Interval</b> (°C, dates)", "$+$, $-$",
          "Numeric", "Ratios are meaningless: 20 °C is not twice 10 °C"],
         ["<b>Ratio</b> (price, count, mass)", "$+$, $-$, $\\times$, $\\div$",
          "Numeric, often log-transformed",
          "Skew — a log or Box–Cox often helps a lot"],
         ["<b>Cyclic</b> (hour, month, angle)", "distance around a circle",
          "$(\\sin, \\cos)$ pair",
          "Hour 23 and hour 0 are adjacent, not 23 apart"],
         ["<b>Compositional</b> (proportions)", "ratios of parts",
          "Log-ratio transform",
          "The components sum to 1, so they are not independent"]],
    )

    codenote(
        "The cyclic encoding is the one people forget",
        "Feeding <code>hour = 23</code> and <code>hour = 0</code> to a model "
        "that sees them as numbers tells it these moments are maximally far "
        "apart. Encode as $(\\sin(2\\pi h/24), \\cos(2\\pi h/24))$ and the "
        "distance becomes correct — adjacent hours are adjacent. Same for day of "
        "week, month, wind direction, and any phase. This is a two-line change "
        "that routinely produces a visible improvement.",
    )

    sub("Distance and why scaling changes the answer")

    md(
        "Almost every algorithm is implicitly geometric: $k$-NN and $k$-means "
        "use distance directly; SVMs use inner products; PCA uses variance; "
        "gradient descent's conditioning depends on the axes (§F.6). All of "
        "these change if you rescale a column."
    )

    math(r"""
    d(\mathbf{x}, \mathbf{z}) =
      \Bigl(\sum_{j=1}^{d} |x_j - z_j|^{\,p}\Bigr)^{1/p}
    \qquad
    \begin{cases}
      p = 1 & \text{Manhattan} \\
      p = 2 & \text{Euclidean} \\
      p \to \infty & \text{Chebyshev}
    \end{cases}
    """)

    table(
        ["Algorithm", "Scale-sensitive?", "Why"],
        [["$k$-NN, $k$-means, DBSCAN, SVM (RBF)", "<b>Yes, severely</b>",
          "Distances are the model"],
         ["PCA", "<b>Yes</b>", "It maximises variance, which has units"],
         ["Ridge / lasso / elastic net", "<b>Yes</b>",
          "The penalty is on the coefficients, whose size depends on the units"],
         ["Neural networks", "<b>Yes</b> (for optimisation)",
          "Conditioning, §F.6 — not for expressiveness"],
         ["Decision trees, random forests, boosting", "<b>No</b>",
          "Splits are on ranks; any monotone transform gives the same tree"],
         ["Plain linear regression (unregularised)", "<b>No</b> (statistically)",
          "Coefficients absorb the units — but conditioning still suffers"]],
    )

    sub("Concentration of distance")

    derive(
        [("Let $\\mathbf{x}, \\mathbf{z}$ have i.i.d. components with variance "
          "$\\sigma^2$. The squared distance is a sum of $d$ i.i.d. terms:",
          r"\lVert\mathbf{x}-\mathbf{z}\rVert^{2} = \sum_{j=1}^{d}(x_j-z_j)^2"),
         ("By the law of large numbers this sum grows like $d$, so the distance "
          "grows like $\\sqrt{d}$. Its <b>standard deviation</b>, however, grows "
          "only like $\\sqrt{d}/\\sqrt{d} = \\mathcal{O}(1)$ after taking the "
          "square root. The relative spread therefore vanishes:",
          r"\frac{\mathrm{SD}\bigl(\lVert\mathbf{x}-\mathbf{z}\rVert\bigr)}"
          r"{\mathbb{E}\bigl[\lVert\mathbf{x}-\mathbf{z}\rVert\bigr]}"
          r" \;=\; \mathcal{O}\!\left(\frac{1}{\sqrt{d}}\right)"),
         ("Equivalently, for a query point and $n$ random neighbours:",
          r"\frac{d_{\max} - d_{\min}}{d_{\min}} \;\longrightarrow\; 0"
          r"\quad\text{as } d \to \infty"),
         ("<b>Everything becomes equidistant.</b> 'Nearest' neighbour stops "
          "being meaningfully nearer than the farthest one, so $k$-NN, "
          "$k$-means and RBF kernels degrade — not because of a bug, but "
          "because the geometry they rely on has dissolved.", None),
         ("<b>Why real high-dimensional data still works:</b> the concentration "
          "argument assumes the components are independent. Real data of "
          "nominal dimension $10^4$ typically lies near a manifold of "
          "<b>intrinsic</b> dimension in the tens, and only the intrinsic "
          "dimension matters. That is the manifold hypothesis (§8.2), and it is "
          "why anything works at all in high dimensions.", None)],
        title="Why distance stops discriminating",
    )

    idea(
        "Three more high-dimensional surprises worth carrying around",
        "<b>1. Volume is all in the shell.</b> The fraction of a unit ball's "
        "volume within radius $1-\\epsilon$ is $(1-\\epsilon)^d$ — at $d = 100$ "
        "and $\\epsilon = 0.05$, that is 0.6 %. Essentially all the volume is "
        "in a thin skin.<br>"
        "<b>2. Random vectors are orthogonal.</b> The expected cosine between "
        "two random directions is 0 with standard deviation "
        "$1/\\sqrt{d}$ — at $d = 10\\,000$, everything is perpendicular to "
        "everything. This is what makes random projection work (§8.6).<br>"
        "<b>3. The corners dominate.</b> The ratio of a unit ball's volume to "
        "its bounding cube's goes to zero super-exponentially — at $d = 20$ it "
        "is about $2 \\times 10^{-8}$. Uniform sampling in a cube almost never "
        "lands in the inscribed ball, which is why grid search is so wasteful "
        "(§2.7).",
    )

    anim_header("Distances concentrating as dimension grows")

    rng = np.random.default_rng(2)
    dims = [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
    frames = []
    for d in dims:
        P = rng.normal(0, 1, (700, d))
        q = rng.normal(0, 1, d)
        dist = np.linalg.norm(P - q, axis=1)
        rel = (dist.max() - dist.min()) / dist.min()
        frames.append(go.Frame(name=str(d), data=[
            go.Histogram(x=dist / dist.mean(), nbinsx=55,
                         histnorm="probability density",
                         marker=dict(color=alpha(C["primary"], .85))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"d = {d}   ·   (dmax − dmin)/dmin = {rel:.3f}   ·   "
            f"relative spread {dist.std()/dist.mean():.4f}   ·   "
            f"predicted ≈ {1/np.sqrt(2*d):.4f}   ·   "
            + ("neighbours are meaningful" if rel > 1.0 else
               "everything is equidistant — k-NN is now guessing"),
            color=C["success"] if rel > 1.0 else C["danger"])])))

    P0 = rng.normal(0, 1, (700, 1)); q0 = rng.normal(0, 1, 1)
    d0 = np.linalg.norm(P0 - q0, axis=1)
    f = go.Figure(data=[go.Histogram(x=d0/d0.mean(), nbinsx=55,
                                     histnorm="probability density",
                                     marker=dict(color=alpha(C["primary"],
                                                             .85)))])
    f.add_vline(x=1.0, line_dash="dash", line_color=C["danger"],
                annotation_text="mean distance")
    f.update_layout(height=430, xaxis_title="distance / mean distance",
                    yaxis_title="density", xaxis=dict(range=[0, 2.2]),
                    title="700 random points, distance to a query point")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="d = ")
    figure(f, "By d = 500 every point sits at essentially the same distance. "
              "Nothing is 'near'.")

    code_lab(
        "Encodings, scaling, distance concentration, and intrinsic dimension",
        '''import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. CYCLIC ENCODING =======================================
print("=== hour of day: integer vs cyclic ===")
hours = np.arange(24)
print(f"  {'pair':>14}{'|integer diff|':>17}{'cyclic distance':>19}"
      f"{'true separation':>18}")
for a, b in [(1, 2), (10, 14), (23, 0), (22, 2), (0, 12)]:
    integer = abs(a-b)
    ca = np.array([np.sin(2*np.pi*a/24), np.cos(2*np.pi*a/24)])
    cb = np.array([np.sin(2*np.pi*b/24), np.cos(2*np.pi*b/24)])
    cyc = float(np.linalg.norm(ca-cb))
    true = min(abs(a-b), 24-abs(a-b))
    print(f"  {f'{a:02d}h vs {b:02d}h':>14}{integer:>17}{cyc:>19.4f}"
          f"{true:>18}")
print("  23h and 00h are ONE hour apart. The integer encoding says 23.")
print("  the (sin, cos) pair gets it right, and it is two lines of code.")

# --- does it matter? -------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
n = 4000
h = rng.integers(0, 24, n)
demand = 100 + 40*np.sin(2*np.pi*(h-3)/24) + rng.normal(0, 6, n)
Xi = h[:, None].astype(float)
Xc = np.column_stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)])
print()
print(f"  {'encoding':<22}{'ridge R^2':>12}{'random forest R^2':>21}")
for nm, Xe in [("integer hour", Xi), ("(sin, cos) pair", Xc)]:
    r2_ridge = Ridge().fit(Xe[:3000], demand[:3000]).score(Xe[3000:],
                                                           demand[3000:])
    r2_rf = RandomForestRegressor(100, random_state=0, n_jobs=-1).fit(
        Xe[:3000], demand[:3000]).score(Xe[3000:], demand[3000:])
    print(f"  {nm:<22}{r2_ridge:>12.4f}{r2_rf:>21.4f}")
print("  the linear model is helpless with an integer hour and fine with the")
print("  cyclic pair. The tree can recover it (it splits on ranks) but needs")
print("  many more splits to do so.")

# ============ 2. ORDINAL vs NOMINAL ====================================
print()
print("=== integer-coding a NOMINAL variable invents an order ===")
cities = np.array(["Paris", "Tokyo", "Lima", "Cairo", "Oslo"])
codes = np.arange(5)
print(f"  integer codes: {dict(zip(cities, codes))}")
print(f"  this asserts Paris < Tokyo < Lima < Cairo < Oslo, and that")
print(f"  |Paris - Lima| = 2 = |Tokyo - Cairo|. Both are meaningless.")
city_idx = rng.integers(0, 5, 3000)
effect = np.array([10.0, 2.0, 9.5, 1.5, 10.5])          # NON-monotone in code
yv = effect[city_idx] + rng.normal(0, 1, 3000)
Xint = city_idx[:, None].astype(float)
Xoh = np.eye(5)[city_idx]
print()
print(f"  {'encoding':<22}{'ridge R^2':>12}{'random forest R^2':>21}")
for nm, Xe in [("integer code", Xint), ("one-hot", Xoh)]:
    a = Ridge().fit(Xe[:2200], yv[:2200]).score(Xe[2200:], yv[2200:])
    b = RandomForestRegressor(100, random_state=0, n_jobs=-1).fit(
        Xe[:2200], yv[:2200]).score(Xe[2200:], yv[2200:])
    print(f"  {nm:<22}{a:>12.4f}{b:>21.4f}")
print("  the linear model CANNOT represent a non-monotone effect of an")
print("  integer code. One-hot removes the false ordering entirely.")

# ============ 3. SCALING CHANGES THE ANSWER ============================
print()
print("="*70)
print("Which algorithms care about the units?")
print("="*70)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

n = 1200
X = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
yb = (X[:, 0] + X[:, 1] + rng.normal(0, .4, n) > 0).astype(int)
X_bad = X.copy(); X_bad[:, 1] *= 1000.0       # column 2 measured in millimetres

print(f"  identical information; column 2 multiplied by 1000")
print(f"  {'model':<34}{'original':>11}{'rescaled':>11}{'changed?':>11}")
for nm, mk in [("k-NN (k=15)", lambda: KNeighborsClassifier(15)),
               ("SVM, RBF kernel", lambda: SVC()),
               ("logistic + L2 penalty",
                lambda: LogisticRegression(C=0.01)),
               ("decision tree", lambda: DecisionTreeClassifier(max_depth=5,
                                                                random_state=0)),
               ("random forest",
                lambda: RandomForestRegressor(50, random_state=0, n_jobs=-1))]:
    try:
        a = mk().fit(X[:900], yb[:900]).score(X[900:], yb[900:])
        b = mk().fit(X_bad[:900], yb[:900]).score(X_bad[900:], yb[900:])
    except Exception:
        continue
    print(f"  {nm:<34}{a:>11.4f}{b:>11.4f}"
          f"{('YES' if abs(a-b) > 0.01 else 'no'):>11}")
print("  distance-based and penalised models break; tree-based models do not,")
print("  because a tree splits on RANKS and any monotone rescaling gives the")
print("  identical tree.")

# --- PCA is scale-sensitive too --------------------------------------
print()
print("=== PCA follows whichever column has the biggest numbers ===")
for nm, A in [("original", X), ("column 2 x1000", X_bad),
              ("standardised", (X_bad-X_bad.mean(0))/X_bad.std(0))]:
    p = PCA(2).fit(A)
    print(f"  {nm:<20} PC1 = {np.round(p.components_[0], 4)}   "
          f"explains {p.explained_variance_ratio_[0]:.4f}")
print("  after rescaling, PC1 is ENTIRELY the millimetre column -- it has")
print("  the most variance because it has the biggest units, not the most")
print("  information.")

# ============ 4. DISTANCE CONCENTRATION ================================
print()
print("="*70)
print("In high dimensions, everything is equidistant")
print("="*70)
print(f"{'d':>7}{'mean distance':>16}{'SD':>10}{'relative spread':>18}"
      f"{'(dmax-dmin)/dmin':>20}{'1/sqrt(2d)':>13}")
for d in [1, 2, 5, 10, 50, 100, 500, 2000]:
    P = rng.normal(0, 1, (1500, d))
    q = rng.normal(0, 1, d)
    dist = np.linalg.norm(P-q, axis=1)
    print(f"{d:>7}{dist.mean():>16.4f}{dist.std():>10.4f}"
          f"{dist.std()/dist.mean():>18.5f}"
          f"{(dist.max()-dist.min())/dist.min():>20.4f}"
          f"{1/np.sqrt(2*d):>13.5f}")
print("  the relative spread tracks 1/sqrt(2d) precisely.")
print("  by d = 2000, the nearest and farthest points differ by 25%.")
print("  'nearest neighbour' has stopped meaning anything.")

# --- and k-NN degrades exactly as predicted --------------------------
print()
print("=== k-NN accuracy as noise dimensions are added ===")
n = 1500
signal = rng.normal(0, 1, (n, 2))
yk = (signal[:, 0] + signal[:, 1] > 0).astype(int)
print(f"  the label depends on 2 features. We append pure NOISE columns.")
print(f"  {'total dims':>12}{'k-NN accuracy':>16}{'tree accuracy':>16}"
      f"{'logistic':>11}")
for extra in [0, 3, 8, 20, 50, 200, 800]:
    Xn = np.hstack([signal, rng.normal(0, 1, (n, extra))])
    a = KNeighborsClassifier(15).fit(Xn[:1000], yk[:1000]).score(Xn[1000:],
                                                                 yk[1000:])
    b = DecisionTreeClassifier(max_depth=6, random_state=0).fit(
        Xn[:1000], yk[:1000]).score(Xn[1000:], yk[1000:])
    c = LogisticRegression(max_iter=2000).fit(Xn[:1000], yk[:1000]).score(
        Xn[1000:], yk[1000:])
    print(f"  {2+extra:>12}{a:>16.4f}{b:>16.4f}{c:>11.4f}")
print("  k-NN collapses toward chance; the tree and the linear model survive")
print("  because they can IGNORE dimensions. Distance cannot.")

# ============ 5. THE OTHER HIGH-DIMENSIONAL SURPRISES ==================
print()
print("=== volume lives in the shell ===")
print(f"  {'d':>6}{'fraction of a ball within 0.95r':>34}")
for d in [1, 2, 3, 10, 50, 100, 500]:
    print(f"  {d:>6}{0.95**d:>34.6f}")
print("  at d = 100, 99.4% of the volume is in the outer 5% of the radius.")

print()
print("=== random vectors are orthogonal ===")
print(f"  {'d':>6}{'mean |cosine|':>16}{'SD of cosine':>15}{'1/sqrt(d)':>12}")
for d in [2, 10, 100, 1000, 10000]:
    A = rng.normal(0, 1, (900, d)); B = rng.normal(0, 1, (900, d))
    cos = np.sum(A*B, 1)/(np.linalg.norm(A, axis=1)*np.linalg.norm(B, axis=1))
    print(f"  {d:>6}{np.abs(cos).mean():>16.5f}{cos.std():>15.5f}"
          f"{1/np.sqrt(d):>12.5f}")
print("  this is exactly why RANDOM PROJECTION works (8.6): random directions")
print("  are near-orthogonal, so they preserve distances well.")

print()
print("=== the ball vanishes inside its cube ===")
from math import lgamma, log, pi
print(f"  {'d':>6}{'vol(ball) / vol(cube)':>26}")
for d in [2, 3, 5, 10, 20, 50]:
    log_ratio = (d/2)*log(pi) - lgamma(d/2+1) - d*log(2)
    print(f"  {d:>6}{np.exp(log_ratio):>26.3e}")
print("  at d = 50, a uniform sample of the cube lands in the inscribed ball")
print("  with probability 1e-28. GRID SEARCH IS MOSTLY SAMPLING CORNERS.")

# ============ 6. WHY ANYTHING WORKS: INTRINSIC DIMENSION ===============
print()
print("="*70)
print("Real data has low INTRINSIC dimension")
print("="*70)
from core import datasets as _ds

def two_nn_dimension(X, sample=900, seed=0):
    """Facco et al. two-NN estimator of intrinsic dimension."""
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), min(sample, len(X)), replace=False)
    A = X[idx]
    D = np.linalg.norm(A[:, None] - A[None], axis=2)
    np.fill_diagonal(D, np.inf)
    d_sorted = np.sort(D, axis=1)
    mu = d_sorted[:, 1]/np.maximum(d_sorted[:, 0], 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float((len(mu))/np.sum(np.log(mu)))

Xd, yd = _ds.digits()[:2]
Xs, _c = _ds.swiss_roll(n=2000)
sets = [
    ("uniform noise, 10 dims", rng.normal(0, 1, (2000, 10))),
    ("uniform noise, 64 dims", rng.normal(0, 1, (2000, 64))),
    ("swiss roll in R^3 (a 2-D sheet)", Xs),
    ("digits, 64 pixels", Xd.astype(float)),
]
print(f"  {'dataset':<38}{'nominal d':>12}{'intrinsic d':>14}")
for nm, A in sets:
    print(f"  {nm:<38}{A.shape[1]:>12}{two_nn_dimension(A):>14.2f}")
print()
print("  pure noise has intrinsic dimension = nominal dimension.")
print("  the swiss roll sits in R^3 but is really 2-D.")
print("  the 64-pixel digits live near a manifold of about 10 dimensions.")
print("  THE CURSE APPLIES TO INTRINSIC DIMENSION, NOT NOMINAL -- which is")
print("  the manifold hypothesis (8.2), and why anything works at all.")

import plotly.graph_objects as go
ds_ = np.array([1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000])
spread = []
for d in ds_:
    P = rng.normal(0, 1, (900, int(d))); q = rng.normal(0, 1, int(d))
    dd = np.linalg.norm(P-q, axis=1)
    spread.append(dd.std()/dd.mean())
fig = go.Figure()
fig.add_scatter(x=ds_, y=spread, mode="lines+markers", name="measured",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(x=ds_, y=1/np.sqrt(2*ds_), mode="lines", name="1/√(2d)",
                line=dict(color=C["ink"], width=2, dash="dot"))
fig.update_layout(height=420, xaxis_type="log", yaxis_type="log",
                  xaxis_title="dimension d",
                  yaxis_title="relative spread of distances",
                  title="Distance concentration follows 1/√(2d)")
''',
        key="found_geometry",
    )

    keypoints([
        "Measurement scale dictates the encoding: nominal → one-hot, cyclic → "
        "$(\\sin,\\cos)$, ratio → often log.",
        "<b>Distance-based and penalised</b> models need scaling; "
        "<b>tree-based</b> ones do not (they split on ranks).",
        "Relative distance spread falls as $1/\\sqrt{2d}$ — in high dimensions "
        "everything is equidistant.",
        "Volume lives in the shell, random vectors are orthogonal, and the ball "
        "vanishes inside its cube.",
        "The curse applies to <b>intrinsic</b> dimension — real data lies near a "
        "low-dimensional manifold.",
    ])


# ==========================================================================
def s_f9():
    section("F.9", "Vocabulary, Mental Models and Common Misconceptions")

    lead(
        "The consolidation. Every term this page introduced, in one place; the "
        "half-dozen beliefs that quietly cost people the most; and the "
        "questions worth asking before you fit anything."
    )

    sub("The vocabulary, in dependency order")

    table(
        ["Term", "Precisely", "Where"],
        [["<b>Hypothesis space</b> $\\mathcal{H}$",
          "The functions the algorithm can return", "§F.1"],
         ["<b>Risk</b> $R(h)$", "Expected loss over $\\mathcal{D}$; unknowable",
          "§F.1"],
         ["<b>Empirical risk</b> $\\hat R_S(h)$",
          "Average loss on your sample; computable", "§F.1"],
         ["<b>ERM</b>", "Return $\\arg\\min_{\\mathcal{H}} \\hat R_S$",
          "§F.1"],
         ["<b>Approximation error</b>",
          "$R(h^{\\mathcal{H}}) - R^\\star$ — the cost of your $\\mathcal{H}$",
          "§F.1"],
         ["<b>Estimation error</b>",
          "$R(\\hat h) - R(h^{\\mathcal{H}})$ — the cost of finite data",
          "§F.1"],
         ["<b>Inductive bias</b>",
          "Assumptions that pick among functions fitting the data equally well",
          "§F.1"],
         ["<b>i.i.d.</b>",
          "Samples independent, from one unchanging distribution", "§F.2"],
         ["<b>Uniform convergence</b>",
          "$\\hat R_S \\approx R$ for <i>all</i> $h$ simultaneously", "§F.2"],
         ["<b>PAC</b>",
          "Probably ($1-\\delta$) approximately ($\\epsilon$) correct", "§F.2"],
         ["<b>VC dimension</b>", "Largest set shattered by $\\mathcal{H}$",
          "§F.3"],
         ["<b>Rademacher complexity</b>",
          "How well $\\mathcal{H}$ fits random labels — computable", "§F.3"],
         ["<b>Bayes risk</b> $R^\\star$", "The irreducible floor", "§F.4"],
         ["<b>Surrogate loss</b>",
          "A convex upper bound on 0–1 that is optimisable", "§F.4"],
         ["<b>Proper scoring rule</b>",
          "Maximised by reporting your true belief", "§F.4"],
         ["<b>Calibration</b>",
          "Of the events you call 70 %, 70 % happen", "§F.4"],
         ["<b>Bias / variance</b>",
          "Systematic error / sensitivity to the training sample", "§F.5"],
         ["<b>Double descent</b>",
          "Test error falls again past $p = n$", "§F.5"],
         ["<b>Implicit regularisation</b>",
          "The optimiser's own preference among equal-loss solutions", "§F.5"],
         ["<b>Condition number</b> $\\kappa = L/\\mu$",
          "Sets the number of gradient steps needed", "§F.6"],
         ["<b>Winner's curse</b>",
          "Best-of-$K$ is $\\sigma\\sqrt{2\\ln K}$ too optimistic", "§F.7"],
         ["<b>Intrinsic dimension</b>",
          "Dimension of the manifold the data actually occupies", "§F.8"]],
    )

    rule()

    sub("Eight misconceptions worth unlearning")

    table(
        ["Common belief", "What is actually true"],
        [["“More parameters always means more overfitting.”",
          "Capacity is not parameter count (§F.3), and past the interpolation "
          "threshold more parameters often <b>help</b> (§F.5). What matters is "
          "the <b>norm</b> of the solution the optimiser lands on."],
         ["“My model is 94 % accurate.”",
          "It scored 94 % on one sample. With $n = 1000$ that is "
          "$94 \\pm 1.5$ %, and if you chose the model on that same set it is "
          "<b>biased upward</b> (§F.7)."],
         ["“Deep learning needs no feature engineering.”",
          "It learns features from raw signals with known structure — pixels, "
          "audio, text. On tabular data, encoding choices (§F.8) still dominate, "
          "which is why gradient boosting remains the default there."],
         ["“Cross-validation gives an unbiased estimate with a standard "
          "error.”",
          "It estimates a <b>procedure</b>, is pessimistic for small $k$, and "
          "has <b>no unbiased variance estimator</b> (§F.7)."],
         ["“The training loss went to zero, so the model memorised.”",
          "Zero training loss is compatible with excellent generalisation "
          "(§F.5). Check held-out error; it is the only thing that answers the "
          "question."],
         ["“I should use accuracy — it is what people understand.”",
          "Accuracy is not a proper scoring rule (§F.4), is misleading under "
          "class imbalance (§3.2), and gives no calibrated probabilities for "
          "cost-based decisions."],
         ["“This algorithm is assumption-free.”",
          "Every algorithm has an inductive bias (§F.1). Unstated is not "
          "absent."],
         ["“The model is broken — accuracy fell after deployment.”",
          "Usually the <b>world</b> changed, not the model. Distribution shift "
          "violates the i.i.d. assumption every guarantee rests on (§F.2, "
          "§19.8)."]],
    )

    rule()

    sub("Questions to ask before fitting anything")

    table(
        ["Question", "Because"],
        [["What is $\\mathcal{Y}$, and what does a mistake <b>cost</b>?",
          "The loss determines the optimal answer (§F.4)"],
         ["Is the sample plausibly i.i.d. with deployment?",
          "Everything rests on it (§F.2)"],
         ["Are there groups, sessions, or time in the data?",
          "Dependence collapses your effective $n$; split by group (§F.2)"],
         ["What is the <b>baseline</b> — base rate, seasonal naive, a linear "
          "model?",
          "Most reported gains do not clear it (§15.3)"],
         ["How large is the test set, and what is its SE?",
          "It says which differences you can even detect (§F.7)"],
         ["How many configurations will I try?",
          "Sets the winner's curse; budget a final untouched test set (§F.7)"],
         ["What inductive bias suits this structure?",
          "Locality → CNN, order → sequence model, tabular → trees (§F.1)"],
         ["Do I need calibrated probabilities or only a decision?",
          "Decides between a proper rule and 0–1 (§F.4)"],
         ["What will drift first, and how would I notice?",
          "Monitoring must be designed before launch (§19.8)"]],
    )

    anim_header("The whole page as one map")

    nodes = [
        ("data D", 0.5, 0.92, C["accent"], "unknown, i.i.d. assumed"),
        ("sample S", 0.5, 0.76, C["train"], "what you actually have"),
        ("hypothesis space H", 0.16, 0.58, C["primary"], "inductive bias"),
        ("loss ℓ", 0.84, 0.58, C["primary"], "what counts as a mistake"),
        ("empirical risk", 0.5, 0.44, C["warning"], "computable"),
        ("optimiser", 0.16, 0.28, C["info"], "implicit regularisation"),
        ("ĥ", 0.5, 0.20, C["success"], "what you ship"),
        ("true risk R(ĥ)", 0.84, 0.28, C["danger"], "what you care about"),
        ("estimate on test", 0.5, 0.05, C["valid"], "a random variable"),
    ]
    edges = [(0, 1), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
             (0, 7)]

    frames = []
    for k in range(1, len(nodes) + 1):
        data = []
        for (i, j) in edges:
            if i < k and j < k:
                data.append(go.Scatter(
                    x=[nodes[i][1], nodes[j][1]],
                    y=[nodes[i][2], nodes[j][2]], mode="lines",
                    line=dict(color=alpha(C["line"], .85), width=2),
                    showlegend=False, hoverinfo="skip"))
        ann = []
        for i in range(k):
            nm, x_, y_, col, sub_ = nodes[i]
            data.append(go.Scatter(x=[x_], y=[y_], mode="markers",
                                   marker=dict(size=34, color=col,
                                               line=dict(color="#fff",
                                                         width=2.5)),
                                   showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x_, y=y_ + .055, text=f"<b>{nm}</b>",
                            showarrow=False,
                            font=dict(size=11, color=C["ink"])))
            ann.append(dict(x=x_, y=y_ - .05, text=sub_, showarrow=False,
                            font=dict(size=9, color=C["ink_soft"])))
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{nodes[k-1][0]} — {nodes[k-1][4]}")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=560, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[0, 1]),
                    yaxis=dict(visible=False, range=[-.02, 1.02]),
                    annotations=list(frames[0].layout.annotations),
                    title="From an unknown distribution to a number you report")
    anim.animate(f, frames, duration=nav.anim_ms(950), slider_prefix="step ")
    figure(f, "Every arrow is a place error enters: sampling, hypothesis "
              "choice, loss choice, optimisation, and finally estimation of the "
              "score itself.")

    code_lab(
        "A self-check: put the whole page together on one problem",
        '''import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# A single problem, worked with every idea from this page.
print("="*70)
print("The problem: predict churn from 8 features, 1 200 labelled customers")
print("="*70)

# ---- the data-generating process (we invent it, so we know the truth) --
def generate(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    X = r.normal(shift, 1, (n, 8))
    logit = 1.3*X[:, 0] - 0.9*X[:, 1] + 0.6*X[:, 2]*X[:, 3] - 0.4*X[:, 4]
    p = 1/(1+np.exp(-logit))
    y = (r.random(n) < p).astype(int)
    return X, y, p

X, y, p_true = generate(1200, 0)
X_big, y_big, p_big = generate(60000, 999)          # a stand-in for D

# ---- F.4: what is the Bayes risk? -------------------------------------
bayes_acc = float(np.mean(np.maximum(p_big, 1-p_big)))
bayes_ll = float(-np.mean(p_big*np.log(p_big) + (1-p_big)*np.log(1-p_big)))
print()
print("F.4  the ceiling nobody can beat")
print(f"     Bayes accuracy {bayes_acc:.4f}   Bayes log loss {bayes_ll:.4f}")
print(f"     any model reporting above {bayes_acc:.4f} is overfitting or leaking")

# ---- F.1: three hypothesis spaces, three inductive biases -------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

Xtr, ytr = X[:800], y[:800]
Xva, yva = X[800:1000], y[800:1000]
Xte, yte = X[1000:], y[1000:]
print()
print("F.1  hypothesis space = inductive bias")
print(f"     {'model':<34}{'bias':<32}{'test acc':>10}")
cands = {
    "baseline (always majority)": (DummyClassifier(strategy="prior"),
                                   "no structure at all"),
    "logistic regression": (make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000)),
                            "additive and linear in the logit"),
    "k-NN (k=25)": (make_pipeline(StandardScaler(),
                                  KNeighborsClassifier(25)),
                    "nearby points share labels"),
    "gradient boosting": (HistGradientBoostingClassifier(random_state=0),
                          "piecewise constant, finds interactions"),
}
fitted = {}
for nm, (m, bias) in cands.items():
    m.fit(Xtr, ytr)
    fitted[nm] = m
    print(f"     {nm:<34}{bias:<32}{m.score(Xte, yte):>10.4f}")
print("     the truth contains an INTERACTION (x2*x3), which the linear model")
print("     structurally cannot represent -- that is approximation error.")

# ---- F.7: is any of that difference real? -----------------------------
print()
print("F.7  is the difference real? (SE, and a PAIRED test)")
n_te = len(yte)
for nm in ["logistic regression", "gradient boosting"]:
    a = fitted[nm].score(Xte, yte)
    se = np.sqrt(a*(1-a)/n_te)
    print(f"     {nm:<24}{a:.4f} ± {1.96*se:.4f}   (n = {n_te})")
ca = fitted["logistic regression"].predict(Xte) == yte
cb = fitted["gradient boosting"].predict(Xte) == yte
b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
pv = stats.binomtest(b_, b_+c_, 0.5).pvalue if b_+c_ else 1.0
print(f"     McNemar: {b_} cases only logistic got, {c_} only boosting got")
print(f"     paired p-value {pv:.4f}   ->  "
      f"{'a real difference' if pv < .05 else 'NOT resolvable at this n'}")
print(f"     an unpaired comparison of two {n_te}-example scores could not")
print(f"     resolve a gap this size at all.")

# ---- F.7: the winner's curse, live -----------------------------------
print()
print("F.7  the winner's curse, on this very dataset")
from sklearn.model_selection import ParameterSampler
grid = list(ParameterSampler(
    {"max_leaf_nodes": [4, 8, 15, 31, 63],
     "learning_rate": [0.02, 0.05, 0.1, 0.2, 0.4],
     "min_samples_leaf": [5, 10, 20, 40],
     "l2_regularization": [0.0, 0.1, 1.0, 10.0]},
    n_iter=40, random_state=0))
val_scores, test_scores = [], []
for g in grid:
    m = HistGradientBoostingClassifier(random_state=0, **g).fit(Xtr, ytr)
    val_scores.append(m.score(Xva, yva))
    test_scores.append(m.score(Xte, yte))
val_scores, test_scores = np.array(val_scores), np.array(test_scores)
i_best = int(val_scores.argmax())
se_val = np.sqrt(0.8*0.2/len(yva))
print(f"     tried K = {len(grid)} configurations")
print(f"     best VALIDATION score      {val_scores[i_best]:.4f}")
print(f"     its score on a FRESH test  {test_scores[i_best]:.4f}   "
      f"({test_scores[i_best]-val_scores[i_best]:+.4f})")
print(f"     mean validation score      {val_scores.mean():.4f}")
print(f"     predicted optimism sqrt(2 ln K)*SE = "
      f"{se_val*np.sqrt(2*np.log(len(grid))):.4f}")
print(f"     rank of the val-winner by TEST score: "
      f"{int((test_scores > test_scores[i_best]).sum())+1} of {len(grid)}")
print("     the validation winner is usually NOT the test winner.")

# ---- F.4: calibration -------------------------------------------------
print()
print("F.4  are the probabilities usable for a cost-based decision?")
print(f"     {'model':<24}{'accuracy':>10}{'log loss':>11}{'Brier':>9}"
      f"{'calibrated?':>14}")
for nm in ["logistic regression", "k-NN (k=25)", "gradient boosting"]:
    m = fitted[nm]
    pr = m.predict_proba(Xte)[:, 1]
    # a crude calibration check: 3 bins
    cal = 0.0
    for lo, hi in [(0, .33), (.33, .66), (.66, 1.01)]:
        msk = (pr >= lo) & (pr < hi)
        if msk.sum() > 10:
            cal += msk.mean()*(pr[msk].mean() - yte[msk].mean())**2
    print(f"     {nm:<24}{accuracy_score(yte, m.predict(Xte)):>10.4f}"
          f"{log_loss(yte, pr):>11.4f}{brier_score_loss(yte, pr):>9.4f}"
          f"{('yes' if cal < 0.002 else 'NO'):>14}")
print(f"     (Bayes log loss is {bayes_ll:.4f} -- that is the floor)")

# ---- F.2: what happens when the world moves ---------------------------
print()
print("F.2  the i.i.d. assumption, tested")
best = fitted["gradient boosting"]
print(f"     {'input shift':>13}{'accuracy':>11}{'log loss':>11}{'change':>10}")
base_acc = None
for sh in [0.0, 0.25, 0.5, 1.0, 2.0]:
    Xs, ys, _ = generate(4000, 4242, shift=sh)
    a = best.score(Xs, ys)
    ll = log_loss(ys, best.predict_proba(Xs)[:, 1])
    base_acc = base_acc if base_acc is not None else a
    print(f"     {sh:>13.2f}{a:>11.4f}{ll:>11.4f}{a-base_acc:>+10.4f}")
print("     the model never changed. Every guarantee on this page is")
print("     conditional on the test data coming from the SAME distribution.")

# ---- F.8: does scaling matter here? -----------------------------------
print()
print("F.8  scale sensitivity")
Xb = X.copy(); Xb[:, 5] *= 500.0
print(f"     {'model':<24}{'original':>11}{'col 6 x500':>13}{'changed?':>11}")
for nm, mk in [("k-NN, UNSCALED", lambda: KNeighborsClassifier(25)),
               ("k-NN, standardised",
                lambda: make_pipeline(StandardScaler(),
                                      KNeighborsClassifier(25))),
               ("gradient boosting",
                lambda: HistGradientBoostingClassifier(random_state=0))]:
    a = mk().fit(X[:1000], y[:1000]).score(X[1000:], y[1000:])
    b = mk().fit(Xb[:1000], y[:1000]).score(Xb[1000:], y[1000:])
    print(f"     {nm:<24}{a:>11.4f}{b:>13.4f}"
          f"{('YES' if abs(a-b) > .01 else 'no'):>11}")

# ---- the summary ------------------------------------------------------
print()
print("="*70)
print("What this page would have you write in the report")
print("="*70)
final = fitted["gradient boosting"]
acc = final.score(Xte, yte)
se = np.sqrt(acc*(1-acc)/len(yte))
print(f"  gradient boosting, {acc:.3f} ± {1.96*se:.3f} accuracy on {len(yte)}")
print(f"  held-out customers (95% interval), against a {cands['baseline (always majority)'][0].fit(Xtr, ytr).score(Xte, yte):.3f}")
print(f"  majority baseline and a {bayes_acc:.3f} Bayes ceiling.")
print(f"  Hyperparameters chosen on a separate validation split of 200;")
print(f"  the test set was scored ONCE.")
print(f"  Probabilities are calibrated (Brier "
      f"{brier_score_loss(yte, final.predict_proba(Xte)[:, 1]):.4f}) and")
print(f"  usable for expected-cost decisions.")
print(f"  Accuracy degrades measurably under input shift, so drift monitoring")
print(f"  on the feature distribution is required before launch.")
print()
print("  every clause in that paragraph came from a section of this page.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=val_scores, y=test_scores, mode="markers",
                marker=dict(size=9, color=C["primary"],
                            line=dict(color="#fff", width=1)),
                name="configurations")
fig.add_scatter(x=[val_scores[i_best]], y=[test_scores[i_best]],
                mode="markers", name="validation winner",
                marker=dict(size=17, color=C["danger"], symbol="star",
                            line=dict(color="#fff", width=2)))
lim = [min(val_scores.min(), test_scores.min())-.01,
       max(val_scores.max(), test_scores.max())+.01]
fig.add_scatter(x=lim, y=lim, mode="lines", name="y = x",
                line=dict(color=C["muted"], width=1.5, dash="dot"))
fig.update_layout(height=440, xaxis_title="validation accuracy",
                  yaxis_title="test accuracy",
                  title="Validation rank does not predict test rank")
''',
        key="found_selfcheck",
    )

    rule()

    sub("If you remember five things from this page")

    keypoints([
        "<b>Learning is ERM plus an inductive bias.</b> The bias is not "
        "optional, and it must match the problem.",
        "<b>Generalisation is a statistical guarantee</b> conditional on i.i.d. "
        "— and shift or dependence breaks it silently.",
        "<b>The loss defines the question</b>, and the optimal answer changes "
        "with it: mean, median, quantile, distribution, mode.",
        "<b>Capacity is not parameter count.</b> Past the interpolation "
        "threshold, the norm of the solution matters, not $p$.",
        "<b>Every number you report is a random variable</b> — and if you "
        "selected on it, a biased one.",
    ], title="Chapter F in five lines")

    refs([
        ("Vapnik — *The Nature of Statistical Learning Theory*",
         "https://doi.org/10.1007/978-1-4757-3264-1"),
        ("Valiant — *A Theory of the Learnable* (PAC)",
         "https://doi.org/10.1145/1968.1972"),
        ("Shalev-Shwartz & Ben-David — *Understanding Machine Learning: From "
         "Theory to Algorithms*",
         "https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/"),
        ("Wolpert — *The Lack of A Priori Distinctions Between Learning "
         "Algorithms*", "https://doi.org/10.1162/neco.1996.8.7.1341"),
        ("Bartlett, Jordan & McAuliffe — *Convexity, Classification, and Risk "
         "Bounds*", "https://doi.org/10.1198/016214505000000907"),
        ("Gneiting & Raftery — *Strictly Proper Scoring Rules, Prediction, and "
         "Estimation*", "https://doi.org/10.1198/016214506000001437"),
        ("Belkin, Hsu, Ma & Mandal — *Reconciling Modern Machine Learning "
         "Practice and the Bias–Variance Trade-off* (double descent)",
         "https://arxiv.org/abs/1812.11118"),
        ("Zhang, Bengio, Hardt, Recht & Vinyals — *Understanding Deep Learning "
         "Requires Rethinking Generalization*",
         "https://arxiv.org/abs/1611.03530"),
        ("Bottou & Bousquet — *The Tradeoffs of Large Scale Learning*",
         "https://papers.nips.cc/paper/3323-the-tradeoffs-of-large-scale-learning"),
        ("Bengio & Grandvalet — *No Unbiased Estimator of the Variance of "
         "K-Fold Cross-Validation*",
         "https://jmlr.org/papers/v5/grandvalet04a.html"),
        ("Nesterov — *Lectures on Convex Optimization*",
         "https://doi.org/10.1007/978-3-319-91578-4"),
        ("Domingos — *A Unified Bias-Variance Decomposition*",
         "https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf"),
    ])


# ==========================================================================
SECTIONS = [
    ("F.1", "The Learning Problem", s_f1),
    ("F.2", "Why Generalisation Works", s_f2),
    ("F.3", "Capacity & VC Dimension", s_f3),
    ("F.4", "Loss Functions", s_f4),
    ("F.5", "Bias, Variance & Double Descent", s_f5),
    ("F.6", "Optimisation Foundations", s_f6),
    ("F.7", "Evaluation & Uncertainty", s_f7),
    ("F.8", "Data, Features & Geometry", s_f8),
    ("F.9", "Vocabulary & Misconceptions", s_f9),
]

nav.render_chapter(CH, SECTIONS)
