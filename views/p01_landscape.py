"""Chapter 1 — The Machine Learning Landscape."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, datasets as ds, nav
from core.lecture import (anim_header, derive, exercise, figure, hero, idea,
                          keypoints, lead, math, md, note, pitfall, proof,
                          quiz, refs, rule, section, sub, table, tip, warn,
                          where)
from core.palette import C, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "ch01"

hero(
    kicker="Part I · Chapter 1",
    title="The Machine Learning Landscape",
    blurb=(
        "Before a single model is fitted we need a map: what learning actually "
        "<i>is</i> as a mathematical object, the axes along which learning systems "
        "differ (supervision, batch vs online, instance vs model), the four ways "
        "data quietly ruins a project, and the sampling discipline that keeps your "
        "reported error honest."
    ),
    chips=["No code prerequisites", "8 sub-sections", "5 animations",
           "4 code labs", "Foundational"],
)
nav.sidebar_tools(CH)


# ==========================================================================
# 1.1
# ==========================================================================
def s_1_1():
    section("1.1", "What Is Machine Learning?")

    lead(
        "Machine learning is the science of building programs whose behaviour is "
        "determined by <b>data</b> rather than by explicitly written rules. The "
        "programmer supplies a <i>family</i> of possible behaviours and a "
        "<i>criterion</i> for what counts as good; the data selects the member of "
        "that family which scores best."
    )

    sub("The classical definitions")

    md(
        """
Two definitions are worth memorising because they operate at different levels.

**Arthur Samuel (1959), informal.** Machine learning is the field of study that
gives computers the ability to learn without being explicitly programmed.

**Tom Mitchell (1997), operational.** A computer program is said to learn from
experience $E$ with respect to some task $T$ and performance measure $P$, if its
performance at $T$, as measured by $P$, improves with $E$.

Mitchell's version is the useful one, because it forces you to name three things
before you write any code. If you cannot fill in the triple $(T, P, E)$, you do
not yet have a machine learning problem — you have a wish.
        """
    )

    table(
        ["Symbol", "Name", "Spam-filter example", "House-price example"],
        [
            ["$T$", "Task", "Flag an incoming email as spam / ham",
             "Predict the median value of a district"],
            ["$P$", "Performance measure", "Accuracy, or better: precision at "
             "fixed recall", "Root-mean-square error in dollars"],
            ["$E$", "Experience", "A labelled corpus of past emails "
             "(the <i>training set</i>)", "20 640 California districts with "
             "census attributes"],
        ],
        "Mitchell's triple, instantiated. Fill this table first, always.",
    )

    sub("The same thing written as mathematics")

    md(
        "Strip away the vocabulary and every supervised learning problem is the "
        "following optimisation. We are handed $m$ examples drawn from an unknown "
        "joint distribution $\\mathcal{D}$ over inputs and targets:"
    )

    math(r"""
    \mathcal{S} \;=\; \bigl\{(\mathbf{x}^{(1)}, y^{(1)}),\;
    (\mathbf{x}^{(2)}, y^{(2)}),\; \dots,\;
    (\mathbf{x}^{(m)}, y^{(m)})\bigr\}
    \;\overset{\text{i.i.d.}}{\sim}\; \mathcal{D}^{m},
    \qquad \mathbf{x}^{(i)} \in \mathcal{X} \subseteq \mathbb{R}^{n},
    \quad y^{(i)} \in \mathcal{Y}.
    """)

    md("We choose a **hypothesis space** $\\mathcal{H}$ — the family of functions "
       "we are willing to consider — and a **loss** $\\ell$ that scores one "
       "prediction. The quantity we actually care about is the *risk*:")

    math(r"""
    R(h) \;=\; \mathbb{E}_{(\mathbf{x}, y)\sim\mathcal{D}}
               \bigl[\, \ell\bigl(h(\mathbf{x}),\, y\bigr) \,\bigr]
    """)

    where({
        r"h": "a candidate predictor, an element of the hypothesis space $\\mathcal{H}$",
        r"\ell": "the per-example loss, e.g. $(h(\\mathbf{x})-y)^2$ for regression",
        r"\mathcal{D}": "the true, unknown, data-generating distribution",
        r"R(h)": "the **true risk** — the average loss over <i>all data that could exist</i>",
    })

    md(
        "We cannot compute $R(h)$: $\\mathcal{D}$ is unknown, and only $m$ samples "
        "of it were ever observed. So we minimise its empirical counterpart "
        "instead — this substitution is called **empirical risk minimisation** and "
        "it is the engine of nearly everything in Part I:"
    )

    math(r"""
    \hat{R}_{\mathcal{S}}(h) \;=\; \frac{1}{m}\sum_{i=1}^{m}
        \ell\bigl(h(\mathbf{x}^{(i)}),\, y^{(i)}\bigr),
    \qquad
    \hat{h} \;=\; \operatorname*{arg\,min}_{h \in \mathcal{H}}
                  \hat{R}_{\mathcal{S}}(h).
    """, "Empirical risk minimisation (ERM). Everything else is a modification of this line.")

    idea(
        "The one sentence that explains the whole book",
        "Learning works only when minimising $\\hat{R}_{\\mathcal{S}}$ (what we can "
        "measure) also reduces $R$ (what we want). The gap "
        "$R(\\hat h) - \\hat R_{\\mathcal{S}}(\\hat h)$ is the "
        "<b>generalisation gap</b>. Restricting $\\mathcal{H}$, adding "
        "regularisation, collecting more data, and early stopping are all devices "
        "for controlling that single gap.",
    )

    derive(
        [
            ("Decompose the true risk of our fitted model $\\hat h$ against the best "
             "achievable predictor $h^\\star \\in \\mathcal{H}$ and the Bayes-optimal "
             "predictor $h^{\\text{Bayes}}$ over all measurable functions.",
             None),
            ("Write the excess risk as a telescoping sum — add and subtract the two "
             "reference points.",
             r"R(\hat h) - R(h^{\text{Bayes}}) = "
             r"\underbrace{\bigl[R(\hat h) - R(h^\star)\bigr]}_{\text{estimation error}} + "
             r"\underbrace{\bigl[R(h^\star) - R(h^{\text{Bayes}})\bigr]}_{\text{approximation error}}"),
            ("<b>Approximation error</b> depends only on how rich $\\mathcal{H}$ is. "
             "A linear model cannot represent a sine wave no matter how much data "
             "you give it. This is <i>bias</i>.", None),
            ("<b>Estimation error</b> depends on how much data you have relative to "
             "the richness of $\\mathcal{H}$. It shrinks as $m$ grows and grows as "
             "$\\mathcal{H}$ grows. This is <i>variance</i>.", None),
            ("Therefore enlarging $\\mathcal{H}$ trades approximation error down for "
             "estimation error up. There is an interior optimum — the sweet spot you "
             "saw animated on the course home page. Chapter 4 makes this exact for "
             "squared loss.", None),
        ],
        title="Why there is always a sweet spot — the excess-risk decomposition",
    )

    sub("The three ingredients you always choose")

    table(
        ["Ingredient", "What you pick", "Where it is discussed"],
        [
            ["Hypothesis space $\\mathcal{H}$",
             "Linear? Polynomial degree? Tree depth? Network architecture?",
             "Ch. 4, 5, 6, 10"],
            ["Loss $\\ell$",
             "Squared, absolute, hinge, log-loss, cross-entropy, Huber",
             "Ch. 3, 4, 5, 10"],
            ["Optimiser",
             "Closed form, gradient descent, CART greedy splits, SMO, Adam",
             "Ch. 4, 5, 6, 11"],
        ],
    )

    anim_header("Empirical risk shrinking as the sample grows")
    md(
        "The dashed curve is the truth. Each frame adds more training points and "
        "refits a degree-9 polynomial. Watch the fitted curve stop thrashing as "
        "$m$ grows: **estimation error is a function of sample size**, and no "
        "amount of clever modelling substitutes for data."
    )

    rng = np.random.default_rng(3)
    Xall = np.sort(rng.uniform(-3, 3, 200))
    yall = np.sin(1.4 * Xall) + 0.3 * Xall + rng.normal(0, 0.3, 200)
    grid = np.linspace(-3.1, 3.1, 300)
    truth = np.sin(1.4 * grid) + 0.3 * grid

    sizes = list(range(12, 201, 8))
    frames = []
    for k in sizes:
        c = np.polyfit(Xall[:k], yall[:k], 9)
        frames.append(go.Frame(
            name=str(k),
            data=[
                go.Scatter(x=Xall[:k], y=yall[:k], mode="markers",
                           marker=dict(color=C["train"], size=7,
                                       line=dict(color="#fff", width=1))),
                go.Scatter(x=grid, y=np.polyval(c, grid), mode="lines",
                           line=dict(color=C["primary"], width=3.4)),
                go.Scatter(x=grid, y=truth, mode="lines",
                           line=dict(color=C["truth"], width=2, dash="dot")),
            ],
            layout=go.Layout(annotations=[
                anim.annotate_step(f"m = {k} training examples")]),
        ))

    f = go.Figure(data=[
        go.Scatter(x=Xall[:12], y=yall[:12], mode="markers", name="training data",
                   marker=dict(color=C["train"], size=7,
                               line=dict(color="#fff", width=1))),
        go.Scatter(x=grid, y=np.polyval(np.polyfit(Xall[:12], yall[:12], 9), grid),
                   mode="lines", name="degree-9 fit",
                   line=dict(color=C["primary"], width=3.4)),
        go.Scatter(x=grid, y=truth, mode="lines", name="true f(x)",
                   line=dict(color=C["truth"], width=2, dash="dot")),
    ])
    f.update_layout(height=430, yaxis=dict(range=[-3, 3]),
                    xaxis=dict(range=[-3.2, 3.2]),
                    xaxis_title="x", yaxis_title="y",
                    title="Same hypothesis space, growing sample")
    anim.animate(f, frames, duration=nav.anim_ms(180), slider_prefix="m = ")
    figure(f)

    code_lab(
        "Empirical risk vs true risk, measured",
        '''# Estimate the generalisation gap by brute force.
# We KNOW the true function here, so we can compute the true risk by
# Monte-Carlo integration over a huge fresh sample.

import numpy as np

rng = np.random.default_rng(0)
f_true = lambda x: np.sin(1.4 * x) + 0.3 * x
SIGMA = 0.30

def sample(m, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, m)
    return x, f_true(x) + r.normal(0, SIGMA, m)

# a very large "population" stands in for the distribution D
x_pop, y_pop = sample(200_000, 999)

rows = []
for m in [15, 25, 50, 100, 200, 400, 800]:
    emp, tru = [], []
    for rep in range(30):                       # average over 30 datasets
        x, y = sample(m, 1000 + rep)
        coef = np.polyfit(x, y, 9)
        emp.append(np.mean((np.polyval(coef, x) - y) ** 2))
        tru.append(np.mean((np.polyval(coef, x_pop) - y_pop) ** 2))
    rows.append((m, np.mean(emp), np.mean(tru), np.mean(tru) - np.mean(emp)))

print(f"{'m':>5} {'empirical R':>13} {'true R':>12} {'gap':>12}")
for m, e, t, g in rows:
    print(f"{m:>5} {e:>13.4f} {t:>12.4f} {g:>12.4f}")
print()
print(f"Irreducible noise floor = sigma^2 = {SIGMA**2:.4f}")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=[r[0] for r in rows], y=[r[1] for r in rows],
                mode="lines+markers", name="empirical risk (train)",
                line=dict(color=C["train"], width=3))
fig.add_scatter(x=[r[0] for r in rows], y=[r[2] for r in rows],
                mode="lines+markers", name="true risk",
                line=dict(color=C["test"], width=3))
fig.add_hline(y=SIGMA**2, line_dash="dot", line_color=C["truth"],
              annotation_text="irreducible noise")
fig.update_layout(xaxis_type="log", yaxis_type="log", height=380,
                  xaxis_title="training-set size m", yaxis_title="mean squared error",
                  title="The generalisation gap closes as m grows")
''',
        key="ch01_erm",
        description="Runs 7 sample sizes x 30 replications. Takes about a second.",
    )

    quiz(
        "A team reports 99.4 % accuracy on the data used to fit their model. "
        "What have they measured?",
        ["The true risk $R(h)$",
         "The empirical risk $\\hat R_{\\mathcal{S}}(h)$ on the training set",
         "The approximation error",
         "The Bayes error"],
        1,
        "Training accuracy is $1-\\hat R_{\\mathcal{S}}$. It is an *optimistically "
        "biased* estimate of $R$ precisely because $h$ was chosen to minimise it. "
        "Section 1.8 shows the fix.",
        key="ch01q1",
    )

    keypoints([
        "Name $(T, P, E)$ before writing code.",
        "Learning = choose $\\mathcal{H}$, choose $\\ell$, minimise "
        "$\\hat R_{\\mathcal{S}}$, hope $R$ follows.",
        "Excess risk = <b>approximation error</b> (too small a $\\mathcal{H}$) + "
        "<b>estimation error</b> (too little data for the $\\mathcal{H}$ you chose).",
        "Training error is not an estimate of generalisation error — it is the "
        "quantity you deliberately minimised.",
    ])


# ==========================================================================
# 1.2
# ==========================================================================
def s_1_2():
    section("1.2", "Why Use Machine Learning?")

    lead(
        "Machine learning is not a better hammer; it is a hammer for a different "
        "class of nail. Four situations justify it, and outside those four a "
        "hand-written rule is cheaper, faster and easier to audit."
    )

    sub("The four justifications")

    table(
        ["#", "Situation", "Why rules fail", "Canonical example"],
        [["1", "<b>Problems with long rule lists</b>",
          "Each new case bolts on another <code>if</code>; the code becomes "
          "unmaintainable and nobody dares delete a branch.",
          "Spam filtering — thousands of hand-tuned keyword rules"],
         ["2", "<b>Problems with no known algorithm</b>",
          "Nobody can write down the rule at all, even though every human does "
          "it effortlessly.",
          "Speech recognition, image classification"],
         ["3", "<b>Fluctuating environments</b>",
          "The rules were right last year. Re-deriving them by hand every quarter "
          "does not scale.",
          "Fraud detection, ad click prediction, demand forecasting"],
         ["4", "<b>Getting insight from large data</b>",
          "Not automation at all — the fitted model is the deliverable, because "
          "its structure tells you something.",
          "Which 5 of 300 features drive churn (data mining)"]],
    )

    anim_header("Rules vs learning: maintenance cost over time")
    md(
        "A stylised but painfully realistic picture. The rule-based system starts "
        "cheaper — one afternoon of `if` statements beats a data pipeline. But each "
        "new adversarial pattern adds a rule, and rules interact, so cost grows "
        "super-linearly. The learning system pays a large fixed cost up front and "
        "then absorbs new patterns by retraining."
    )

    t = np.arange(0, 37)
    rules = 8 + 0.55 * t + 0.030 * t ** 2
    ml = 42 + 0.9 * t + 0.004 * t ** 2

    frames = []
    for k in range(2, len(t) + 1):
        cross = np.argmax(ml[:k] < rules[:k]) if np.any(ml[:k] < rules[:k]) else None
        ann = [anim.annotate_step(f"month {t[k-1]}")]
        if cross:
            ann.append(dict(x=t[cross], y=ml[cross], xref="x", yref="y",
                            text="ML becomes cheaper", showarrow=True,
                            arrowhead=2, ax=-60, ay=-45,
                            font=dict(color=C["success"], size=12),
                            arrowcolor=C["success"]))
        frames.append(go.Frame(
            name=str(t[k - 1]),
            data=[go.Scatter(x=t[:k], y=rules[:k], mode="lines",
                             line=dict(color=C["danger"], width=3.6)),
                  go.Scatter(x=t[:k], y=ml[:k], mode="lines",
                             line=dict(color=C["success"], width=3.6))],
            layout=go.Layout(annotations=ann)))

    f = go.Figure(data=[
        go.Scatter(x=t[:2], y=rules[:2], mode="lines", name="hand-written rules",
                   line=dict(color=C["danger"], width=3.6)),
        go.Scatter(x=t[:2], y=ml[:2], mode="lines", name="learned model",
                   line=dict(color=C["success"], width=3.6)),
    ])
    f.update_layout(height=400, xaxis=dict(range=[0, 36]), yaxis=dict(range=[0, 90]),
                    xaxis_title="months since launch",
                    yaxis_title="cumulative engineering effort (arb. units)",
                    title="Why the crossover happens")
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="month ")
    figure(f)

    sub("The virtuous cycle: automatic adaptation")

    md(
        """
The property that makes ML *operationally* different is that the update step is
mechanical. A rule-based filter needs a human in the loop for every new pattern;
a learned filter needs only new labelled data:
        """
    )

    math(r"""
    \text{data}_{t} \;\longrightarrow\; \text{train} \;\longrightarrow\;
    h_{t} \;\longrightarrow\; \text{deploy} \;\longrightarrow\;
    \text{new labels} \;\longrightarrow\; \text{data}_{t+1}
    """, "The retraining loop. Chapter 19 productionises it; Chapter 2 monitors it.")

    warn(
        "The loop can rot",
        "Automatic adaptation is automatic <i>degradation</i> when the label "
        "channel is poisoned. If your spam labels come from user 'report spam' "
        "clicks, an attacker who mass-reports legitimate newsletters teaches your "
        "filter to block them. Chapter 2's monitoring section and Chapter 19's "
        "deployment section both return to this.",
    )

    sub("Data mining: when the model *is* the answer")

    md(
        "In the fourth case the prediction is beside the point. You fit a model in "
        "order to interrogate it: which features carry signal, which observations "
        "are anomalous, which groups exist. A decision tree's split order "
        "(Chapter 6), a random forest's feature importances (Chapter 7), PCA's "
        "loadings (Chapter 8) and a Gaussian mixture's components (Chapter 9) are "
        "all instruments of this kind."
    )

    code_lab(
        "Rules vs learning on the same spam-like problem",
        '''# A toy but honest comparison. 12 hand-written keyword rules against a
# logistic regression on character n-grams -- on a deliberately drifting stream.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)

SPAM_WORDS_ERA1 = ["free", "winner", "credit", "viagra", "cash", "prize"]
SPAM_WORDS_ERA2 = ["crypto", "airdrop", "wallet", "nft", "presale", "whitelist"]
HAM_WORDS = ["meeting", "report", "invoice", "schedule", "review", "thanks",
             "attached", "regards", "monday", "draft"]

def make(n, spam_words, seed):
    r = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        if r.random() < 0.5:
            words = list(r.choice(spam_words, 3)) + list(r.choice(HAM_WORDS, 2))
            y.append(1)
        else:
            words = list(r.choice(HAM_WORDS, 5)); y.append(0)
        r.shuffle(words)
        X.append(" ".join(words))
    return X, np.array(y)

# --- hand-written rule system: keywords from ERA 1 only ---------------------
def rule_predict(texts):
    return np.array([int(any(w in t for w in SPAM_WORDS_ERA1)) for t in texts])

# --- learned system ---------------------------------------------------------
Xtr, ytr = make(1200, SPAM_WORDS_ERA1, 1)
vec = TfidfVectorizer(analyzer="word")
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)

print(f"{'era':<28}{'rules':>10}{'learned':>12}{'learned+retrain':>18}")
for era, words, seed in [("era 1 (rules were written)", SPAM_WORDS_ERA1, 7),
                         ("era 2 (spammers moved on)", SPAM_WORDS_ERA2, 8)]:
    Xte, yte = make(600, words, seed)
    a_rule = accuracy_score(yte, rule_predict(Xte))
    a_ml   = accuracy_score(yte, clf.predict(vec.transform(Xte)))
    # retrain on 300 freshly labelled examples from the new era
    Xn, yn = make(300, words, seed + 100)
    v2 = TfidfVectorizer(); c2 = LogisticRegression(max_iter=1000)
    c2.fit(v2.fit_transform(Xtr + Xn), np.r_[ytr, yn])
    a_re = accuracy_score(yte, c2.predict(v2.transform(Xte)))
    print(f"{era:<28}{a_rule:>10.3f}{a_ml:>12.3f}{a_re:>18.3f}")

print()
print("Rules and a frozen model both collapse in era 2.")
print("Only the *retrained* model recovers -- that is the virtuous cycle.")
''',
        key="ch01_rules",
    )

    keypoints([
        "Use ML for: long rule lists, no known algorithm, drifting environments, "
        "or insight extraction.",
        "The operational advantage is that <b>updating is mechanical</b> — retrain "
        "instead of re-reason.",
        "That same automation makes the label channel a security surface.",
        "If a 5-line rule solves it, write the 5-line rule.",
    ])


# ==========================================================================
# 1.3
# ==========================================================================
def s_1_3():
    section("1.3", "Examples of Applications")

    lead(
        "A catalogue is more useful than a definition. For each application, note "
        "the <i>shape</i> of the problem — that shape, not the domain, is what "
        "determines which chapter you will need."
    )

    table(
        ["Application", "Problem shape", "Typical model family", "Chapter"],
        [["Classify products from images on a production line",
          "Image classification", "CNN, or a fine-tuned pretrained backbone", "14"],
         ["Detect tumours in brain scans",
          "Semantic segmentation", "U-Net / fully convolutional network", "14"],
         ["Automatically classify news articles",
          "Text classification (NLP)", "RNN, Transformer, or TF-IDF + linear", "16"],
         ["Flag offensive comments",
          "Text classification", "Same as above", "16"],
         ["Summarise long documents",
          "Text summarisation (seq2seq)", "Encoder–decoder Transformer", "16"],
         ["A conversational assistant",
          "NLU + generation", "Large pretrained Transformer", "16"],
         ["Forecast next year's revenue",
          "Regression on time series", "Linear, SVR, RF, RNN, ARIMA", "4, 5, 7, 15"],
         ["Voice command recognition",
          "Sequence classification on audio", "RNN, CNN, Transformer", "15, 16"],
         ["Credit-card fraud detection",
          "Anomaly detection / imbalanced classification", "Isolation Forest, GMM, "
          "gradient boosting", "9, 7"],
         ["Customer segmentation",
          "Clustering", "k-means, DBSCAN, Gaussian mixtures", "9"],
         ["High-dimensional data in one diagram",
          "Dimensionality reduction / visualisation", "PCA, t-SNE, LLE", "8"],
         ["Recommend a product",
          "Regression or classification on user–item features", "Neural nets, "
          "matrix factorisation", "10, 13"],
         ["An agent that plays a game",
          "Sequential decision making", "Policy gradients, DQN", "18"],
         ["Generate photorealistic faces",
          "Generative modelling", "GAN, VAE, diffusion", "17"]],
    )

    anim_header("The problem-shape decision tree")
    md(
        "Play this to walk the decision path a practitioner actually follows. "
        "Each frame highlights the next question and prunes the branches it "
        "eliminates."
    )

    # a small animated decision walk-through drawn with scatter + shapes
    nodes = {
        "root":   (0.0, 3.0, "Do you have labelled targets?"),
        "sup":    (-1.6, 2.0, "Supervised"),
        "unsup":  (1.6, 2.0, "Unsupervised /\nself-supervised"),
        "reg":    (-2.6, 1.0, "Target is a number\n→ Regression"),
        "clf":    (-0.7, 1.0, "Target is a class\n→ Classification"),
        "clu":    (0.8, 1.0, "Find groups\n→ Clustering"),
        "dim":    (2.4, 1.0, "Compress / visualise\n→ Dim. reduction"),
        "gen":    (1.6, 0.0, "Create new samples\n→ Generative"),
        "rl":     (-1.6, 0.0, "Actions with delayed reward\n→ Reinforcement"),
    }
    edges = [("root", "sup"), ("root", "unsup"), ("sup", "reg"), ("sup", "clf"),
             ("unsup", "clu"), ("unsup", "dim"), ("unsup", "gen"), ("root", "rl")]
    order = ["root", "sup", "unsup", "reg", "clf", "clu", "dim", "gen", "rl"]

    def node_trace(active: set[str]):
        xs, ys, txt, cols = [], [], [], []
        for k, (x, y, lab) in nodes.items():
            xs.append(x); ys.append(y); txt.append(lab)
            cols.append(C["primary"] if k in active else "#D8DCE8")
        return go.Scatter(x=xs, y=ys, mode="markers+text", text=txt,
                          textposition="middle center",
                          textfont=dict(size=10, color="#0E1428"),
                          marker=dict(size=74, color=cols, opacity=0.30,
                                      line=dict(color=cols, width=2.5)),
                          hoverinfo="text", showlegend=False)

    def edge_trace(active: set[str]):
        xs, ys = [], []
        for a, b in edges:
            if a in active and b in active:
                xs += [nodes[a][0], nodes[b][0], None]
                ys += [nodes[a][1], nodes[b][1], None]
        return go.Scatter(x=xs, y=ys, mode="lines",
                          line=dict(color=C["accent"], width=2.4),
                          hoverinfo="skip", showlegend=False)

    frames = []
    for i in range(1, len(order) + 1):
        act = set(order[:i])
        frames.append(go.Frame(name=str(i),
                               data=[edge_trace(act), node_trace(act)]))

    f = go.Figure(data=[edge_trace({"root"}), node_trace({"root"})])
    f.update_layout(height=470, showlegend=False,
                    xaxis=dict(visible=False, range=[-3.6, 3.6]),
                    yaxis=dict(visible=False, range=[-0.6, 3.6]),
                    plot_bgcolor="#FFFFFF",
                    title="Which chapter do I need?")
    anim.animate(f, frames, duration=nav.anim_ms(420), slider_prefix="reveal ")
    figure(f)

    note(
        "Domain is not shape",
        "\"Medical imaging\" is a domain; \"pixel-wise classification of a 3-D "
        "volume\" is a shape. Two projects in wildly different domains that share "
        "a shape share almost all their engineering. Always translate to shape "
        "first.",
    )

    keypoints([
        "Fourteen canonical applications reduce to about seven <b>shapes</b>.",
        "Shape determines chapter, model family and evaluation metric.",
        "Regression / classification / clustering / dim-reduction / generative / "
        "sequential-decision / anomaly — that is essentially the whole taxonomy.",
    ])


# ==========================================================================
# 1.4
# ==========================================================================
def s_1_4():
    section("1.4", "Types of ML Systems — Training Supervision")

    lead(
        "The first and most consequential axis: <b>how much and what kind of "
        "supervision signal</b> the training set carries. This determines the loss "
        "you can write down at all."
    )

    sub("The supervision spectrum")

    table(
        ["Regime", "What the data contains", "Objective", "Examples"],
        [["<b>Supervised</b>",
          "$(\\mathbf{x}^{(i)}, y^{(i)})$ — every input carries its target",
          "Minimise $\\frac1m\\sum \\ell(h(\\mathbf x^{(i)}), y^{(i)})$",
          "k-NN, linear/logistic regression, SVM, trees, forests, MLPs"],
         ["<b>Unsupervised</b>",
          "$\\mathbf{x}^{(i)}$ only",
          "Optimise a structural criterion (compactness, likelihood, "
          "reconstruction)",
          "k-means, DBSCAN, GMM, PCA, autoencoders, apriori"],
         ["<b>Semi-supervised</b>",
          "A few labelled, very many unlabelled",
          "Supervised loss on the labelled part + a structural/consistency term "
          "on the rest",
          "Label propagation, DBSCAN+propagation, self-training"],
         ["<b>Self-supervised</b>",
          "$\\mathbf{x}^{(i)}$ only — but a <i>pretext</i> label is manufactured "
          "from the input itself",
          "Predict the hidden part of the input from the visible part",
          "Masked language modelling (BERT), next-token prediction (GPT), "
          "inpainting, contrastive learning"],
         ["<b>Reinforcement</b>",
          "No targets — an environment returning scalar rewards, possibly much "
          "later",
          "Maximise expected discounted return "
          "$\\mathbb{E}\\left[\\sum_t \\gamma^t r_t\\right]$",
          "Policy gradients, Q-learning, DQN"]],
        "The five supervision regimes. Chapters 3–7 are supervised, 8–9 "
        "unsupervised, 16–17 self-supervised and generative, 18 reinforcement.",
    )

    sub("Supervised learning, formally")

    md("Split by the type of $\\mathcal{Y}$:")

    math(r"""
    \textbf{Regression:}\quad \mathcal{Y} = \mathbb{R}
    \qquad\Longrightarrow\qquad
    \ell(\hat y, y) = (\hat y - y)^2
    """)
    math(r"""
    \textbf{Binary classification:}\quad \mathcal{Y} = \{0, 1\}
    \qquad\Longrightarrow\qquad
    \ell(\hat p, y) = -\bigl[\, y \log \hat p + (1-y)\log(1-\hat p) \,\bigr]
    """)
    math(r"""
    \textbf{Multiclass ($K$ classes):}\quad \mathcal{Y} = \{1,\dots,K\}
    \qquad\Longrightarrow\qquad
    \ell(\hat{\mathbf p}, y) = -\sum_{k=1}^{K} \mathbb{1}[y = k]\,\log \hat p_k
    """)

    where({
        r"\hat y": "the model's numeric prediction",
        r"\hat p": "the model's predicted probability of the positive class",
        r"\hat{\mathbf p}": "the model's predicted probability vector, "
                            "$\\sum_k \\hat p_k = 1$",
        r"\mathbb{1}[\cdot]": "the indicator function: 1 if the condition holds, else 0",
    })

    proof(
        "Why cross-entropy and not accuracy",
        "Accuracy is piecewise constant in the parameters, so its gradient is zero "
        "almost everywhere — gradient descent has nothing to descend. "
        "Cross-entropy is a smooth, convex (in the linear case) surrogate whose "
        "minimiser is the true conditional probability. Chapters 3 and 4 develop "
        "this in full.",
    )

    sub("Unsupervised learning, formally")

    md(
        "Without $y$ you must supply the structure yourself. Three archetypal "
        "objectives, one per major technique family:"
    )

    math(r"""
    \textbf{k-means (Ch. 9):}\quad
    \min_{\{\boldsymbol\mu_k\},\, \{c^{(i)}\}}\;
    \sum_{i=1}^{m} \bigl\lVert \mathbf{x}^{(i)} - \boldsymbol\mu_{c^{(i)}} \bigr\rVert_2^2
    """)
    math(r"""
    \textbf{PCA (Ch. 8):}\quad
    \max_{\mathbf{W}^\top\mathbf{W} = \mathbf{I}_d}\;
    \operatorname{tr}\!\bigl(\mathbf{W}^\top \boldsymbol\Sigma \mathbf{W}\bigr),
    \qquad
    \boldsymbol\Sigma = \tfrac{1}{m}\mathbf{X}_c^\top \mathbf{X}_c
    """)
    math(r"""
    \textbf{Gaussian mixture (Ch. 9):}\quad
    \max_{\boldsymbol\theta}\;
    \sum_{i=1}^{m} \log \sum_{k=1}^{K} \pi_k\,
    \mathcal{N}\!\bigl(\mathbf{x}^{(i)} \mid \boldsymbol\mu_k, \boldsymbol\Sigma_k\bigr)
    """)

    anim_header("Supervision regimes on the same 2-D cloud")
    md(
        "One dataset, four regimes. The animation cycles through what each regime "
        "*sees* and what it *produces*. Notice that the points never change — only "
        "the information attached to them, and therefore the objective."
    )

    X, y = ds.moons(n=300, noise=0.18)
    labelled = np.zeros(len(X), dtype=bool)
    rng = np.random.default_rng(1)
    labelled[rng.choice(len(X), 12, replace=False)] = True

    def scat(colors, text, sizes=None, symbols=None):
        return go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                          marker=dict(color=colors,
                                      size=sizes if sizes is not None else 9,
                                      symbol=symbols if symbols is not None else "circle",
                                      line=dict(color="#fff", width=1)),
                          showlegend=False, hovertext=text)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    views = [
        ("Supervised — every point carries a label",
         [C["train"] if v == 0 else C["warning"] for v in y], None, None),
        ("Unsupervised — no labels; the algorithm invents groups",
         [SEQ[k] for k in km.labels_], None, None),
        ("Semi-supervised — 12 labels (large), 288 blanks (small grey)",
         [(C["train"] if y[i] == 0 else C["warning"]) if labelled[i] else "#C9CEDD"
          for i in range(len(X))],
         [15 if labelled[i] else 7 for i in range(len(X))], None),
        ("Self-supervised — the label is manufactured from the input "
         "(here: predict $x_2$ from $x_1$)",
         [C["accent"]] * len(X), None, None),
    ]
    frames = [go.Frame(name=str(i + 1), data=[scat(c, v[0], s, sy)],
                       layout=go.Layout(title=v[0]))
              for i, v in enumerate(views)
              for c, s, sy in [(v[1], v[2], v[3])]]

    f = go.Figure(data=[scat(views[0][1], views[0][0])])
    f.update_layout(height=440, title=views[0][0],
                    xaxis_title="x1", yaxis_title="x2")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="regime ")
    figure(f)

    code_lab(
        "One dataset, four regimes, four objectives",
        '''import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, adjusted_rand_score

X, y = make_moons(n_samples=600, noise=0.18, random_state=0)
rng = np.random.default_rng(0)

# ---- 1. SUPERVISED: minimise cross-entropy with all 600 labels -------------
sup = LogisticRegression().fit(X, y)
print(f"supervised      | accuracy on train      = {sup.score(X, y):.3f}")

# ---- 2. UNSUPERVISED: minimise within-cluster inertia, no labels at all ----
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print(f"unsupervised    | inertia                = {km.inertia_:.1f}")
print(f"                | adjusted Rand vs truth = {adjusted_rand_score(y, km.labels_):.3f}")

# ---- 3. SEMI-SUPERVISED: 10 labels + 590 blanks ---------------------------
y_semi = np.full(len(y), -1)                    # -1 means "unlabelled"
idx = rng.choice(len(y), 10, replace=False)
y_semi[idx] = y[idx]
ls = LabelSpreading(kernel="knn", n_neighbors=8).fit(X, y_semi)
print(f"semi-supervised | 10 labels -> accuracy  = {accuracy_score(y, ls.transduction_):.3f}")

only10 = LogisticRegression().fit(X[idx], y[idx])
print(f"                | 10 labels, supervised  = {only10.score(X, y):.3f}  <- worse")

# ---- 4. SELF-SUPERVISED: manufacture a pretext target from the input -------
# pretext task: predict x2 from x1 (no human ever labelled anything)
from sklearn.neighbors import KNeighborsRegressor
pretext = KNeighborsRegressor(15).fit(X[:, [0]], X[:, 1])
print(f"self-supervised | pretext R^2            = {pretext.score(X[:, [0]], X[:, 1]):.3f}")
print()
print("Semi-supervised beats supervised-on-10 because the 590 unlabelled points")
print("reveal the SHAPE of the manifold, which constrains where the boundary can go.")
''',
        key="ch01_regimes",
    )

    quiz(
        "BERT is trained by masking 15 % of the tokens in a sentence and "
        "predicting them. Which regime is that?",
        ["Supervised", "Unsupervised", "Self-supervised", "Reinforcement"],
        2,
        "The targets are real targets (a cross-entropy loss over the vocabulary), "
        "but no human produced them — they were carved out of the input itself. "
        "That is exactly self-supervision. Chapter 16.",
        key="ch01q2",
    )

    keypoints([
        "Supervision determines <b>which loss you are allowed to write</b>.",
        "Supervised: $\\ell(h(\\mathbf x), y)$. Unsupervised: a structural criterion. "
        "Self-supervised: a supervised loss on a manufactured target.",
        "Semi-supervised works because unlabelled points reveal the manifold "
        "(cluster assumption) — few labels then go a long way.",
        "Reinforcement learning is different in kind: the signal is a delayed "
        "scalar reward and the agent's own actions change the data it sees.",
    ])


# ==========================================================================
# 1.5
# ==========================================================================
def s_1_5():
    section("1.5", "Batch Versus Online Learning")

    lead(
        "The second axis: <b>can the system learn incrementally from a stream of "
        "arriving data?</b> This is an engineering distinction with real "
        "statistical consequences."
    )

    sub("Batch (offline) learning")

    md(
        """
The model is trained once, on all available data, then frozen and deployed. It
does not learn in production; to incorporate new data you retrain from scratch on
the full dataset and swap the artefact.

* **Cost.** Training on the full corpus can take hours and many CPU/GPU-hours.
  Retraining nightly on terabytes is expensive; retraining on a phone is
  impossible.
* **Consequence.** The deployed model *decays*. Its world moves on; it does not.
  This is called **model rot** or **data drift**.
        """
    )

    sub("Online (incremental) learning")

    md(
        "The model is fed data sequentially, one instance or one *mini-batch* at a "
        "time, and updates its parameters after each. The canonical update is "
        "stochastic gradient descent:"
    )

    math(r"""
    \boldsymbol\theta_{t+1} \;=\; \boldsymbol\theta_{t}
    \;-\; \eta_t \, \nabla_{\boldsymbol\theta}\,
    \ell\bigl(h_{\boldsymbol\theta_t}(\mathbf{x}_t),\, y_t\bigr)
    """)

    where({
        r"\boldsymbol\theta_t": "parameter vector after $t$ updates",
        r"\eta_t": "the **learning rate** at step $t$ (Ch. 4 and 11 study its schedule)",
        r"(\mathbf{x}_t, y_t)": "the single example (or mini-batch) arriving at time $t$",
    })

    idea(
        "The learning rate is a memory dial",
        "A high $\\eta$ makes the system adapt fast to new data but forget old data "
        "fast — and stay jumpy. A low $\\eta$ gives inertia and stability, but the "
        "system lags a genuine regime change. In online learning $\\eta$ literally "
        "sets the effective memory length: the influence of an example decays "
        "roughly like $(1-\\eta)^{k}$ after $k$ subsequent updates.",
    )

    derive(
        [("Take a simple online mean estimator, the special case $h_\\theta = \\theta$ "
          "with squared loss $\\ell = \\tfrac12(\\theta - y_t)^2$.", None),
         ("Its gradient is $\\nabla_\\theta \\ell = \\theta - y_t$, so the update is",
          r"\theta_{t+1} = \theta_t - \eta(\theta_t - y_t) = (1-\eta)\theta_t + \eta y_t"),
         ("Unrolling the recursion from $\\theta_0$ gives an exponentially weighted "
          "moving average:",
          r"\theta_{T} = (1-\eta)^{T}\theta_0 + \eta \sum_{t=1}^{T} (1-\eta)^{T-t} y_t"),
         ("The weight on the observation $t$ steps ago is $\\eta(1-\\eta)^{t}$. Setting "
          "the effective window to the point where the weight has decayed by $1/e$:",
          r"\tau \;\approx\; \frac{1}{\eta} \quad\text{examples of memory}"),
         ("So $\\eta = 0.01$ remembers roughly the last 100 examples; $\\eta = 0.5$ "
          "remembers about 2. That is the whole plasticity–stability trade-off in "
          "one line.", None)],
        title="Learning rate ⇒ effective memory length",
    )

    anim_header("Plasticity vs stability under a regime change")
    md(
        "A stream whose mean jumps at $t = 250$. Three online learners with "
        "$\\eta \\in \\{0.01, 0.08, 0.4\\}$ track it. Watch the trade-off: the fast "
        "learner catches the jump in a handful of steps but is permanently noisy; "
        "the slow learner is smooth but takes ~100 steps to arrive."
    )

    rng = np.random.default_rng(4)
    T = 500
    truth = np.where(np.arange(T) < 250, 1.0, 3.0)
    stream = truth + rng.normal(0, 0.45, T)
    etas = [0.01, 0.08, 0.40]
    tracks = []
    for e in etas:
        th, arr = 0.5, []
        for t in range(T):
            th = (1 - e) * th + e * stream[t]
            arr.append(th)
        tracks.append(np.array(arr))

    step = 8
    frames = []
    for k in range(step, T + 1, step):
        data = [go.Scatter(x=np.arange(k), y=stream[:k], mode="markers",
                           marker=dict(color="#C9CEDD", size=3.5))]
        data.append(go.Scatter(x=np.arange(k), y=truth[:k], mode="lines",
                               line=dict(color=C["truth"], width=2, dash="dot")))
        for j, tr in enumerate(tracks):
            data.append(go.Scatter(x=np.arange(k), y=tr[:k], mode="lines",
                                   line=dict(color=SEQ[j], width=2.8)))
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[
                                   anim.annotate_step(f"t = {k}")])))

    f = go.Figure(data=[
        go.Scatter(x=[0], y=[stream[0]], mode="markers", name="observations",
                   marker=dict(color="#C9CEDD", size=3.5)),
        go.Scatter(x=[0], y=[truth[0]], mode="lines", name="true mean",
                   line=dict(color=C["truth"], width=2, dash="dot")),
    ] + [go.Scatter(x=[0], y=[tracks[j][0]], mode="lines",
                    name=f"η = {etas[j]}", line=dict(color=SEQ[j], width=2.8))
         for j in range(3)])
    f.add_vline(x=250, line_dash="dash", line_color=C["danger"],
                annotation_text="regime change")
    f.update_layout(height=430, xaxis=dict(range=[0, T]), yaxis=dict(range=[-0.4, 4.4]),
                    xaxis_title="t (examples seen)", yaxis_title="estimate",
                    title="Online learning: three learning rates, one regime change")
    anim.animate(f, frames, duration=nav.anim_ms(60), slider_prefix="t = ")
    figure(f)

    sub("Out-of-core learning")

    md(
        "Online learning also solves a *memory* problem, not just a *streaming* "
        "one. If the dataset does not fit in RAM, you can still train by feeding it "
        "in chunks — this is **out-of-core** learning. In scikit-learn the relevant "
        "interface is `partial_fit`:"
    )

    table(
        ["Estimator", "Supports `partial_fit`", "Notes"],
        [["<code>SGDClassifier</code> / <code>SGDRegressor</code>", "✅",
          "The workhorse. Linear models with SGD."],
         ["<code>MultinomialNB</code>, <code>BernoulliNB</code>", "✅",
          "Counts accumulate naturally."],
         ["<code>MiniBatchKMeans</code>", "✅", "Online clustering (Ch. 9)."],
         ["<code>IncrementalPCA</code>", "✅", "Online PCA (Ch. 8)."],
         ["<code>Perceptron</code>, <code>PassiveAggressive*</code>", "✅",
          "Classical online algorithms."],
         ["<code>RandomForestClassifier</code>", "❌",
          "Trees are built globally; retrain instead."],
         ["<code>SVC</code> (kernel)", "❌", "Needs the full Gram matrix (Ch. 5)."]],
    )

    pitfall(
        "Online learning fails loudly and silently",
        "If bad data enters the stream — a broken sensor, a bot flood, an "
        "adversarial campaign — the model degrades <i>while serving traffic</i>. "
        "Mitigations: (1) monitor input distributions, not just output accuracy; "
        "(2) keep the previous checkpoint and roll back automatically on a metric "
        "drop; (3) run anomaly detection on incoming batches before they reach "
        "the update step (Chapter 9).",
    )

    code_lab(
        "Out-of-core training with `partial_fit`",
        '''# Train on 200,000 examples while never holding more than 2,000 in memory.

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)
N_TOTAL, CHUNK, N_FEAT = 200_000, 2_000, 20
w_true = rng.normal(0, 1, N_FEAT)

def stream_chunk(seed):
    """Pretend this reads the next CHUNK rows from a 40 GB CSV."""
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (CHUNK, N_FEAT))
    logits = X @ w_true
    y = (logits + r.normal(0, 0.7, CHUNK) > 0).astype(int)
    return X, y

clf = SGDClassifier(loss="log_loss", learning_rate="optimal",
                    random_state=0)

Xte, yte = stream_chunk(99_999)          # held-out chunk
history = []
classes = np.array([0, 1])

for i in range(N_TOTAL // CHUNK):
    Xc, yc = stream_chunk(i)
    clf.partial_fit(Xc, yc, classes=classes)      # <-- the key call
    if i % 5 == 0:
        history.append((i * CHUNK, accuracy_score(yte, clf.predict(Xte))))

print(f"peak memory held  : {CHUNK * N_FEAT * 8 / 1e6:.2f} MB")
print(f"total data seen   : {N_TOTAL * N_FEAT * 8 / 1e6:.1f} MB")
print(f"final accuracy    : {history[-1][1]:.4f}")
print(f"cosine(w_hat, w)  : "
      f"{np.dot(clf.coef_[0], w_true) / (np.linalg.norm(clf.coef_) * np.linalg.norm(w_true)):.4f}")

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=[h[0] for h in history], y=[h[1] for h in history],
                           mode="lines+markers", line=dict(color=C["primary"], width=3)))
fig.update_layout(height=340, xaxis_title="examples streamed",
                  yaxis_title="held-out accuracy",
                  title="Out-of-core learning curve (memory stays flat)")
''',
        key="ch01_ooc",
    )

    keypoints([
        "<b>Batch</b>: train on everything, freeze, deploy, retrain periodically. "
        "Simple, reproducible, decays.",
        "<b>Online</b>: update per example / mini-batch. Cheap, adaptive, "
        "fragile.",
        "The learning rate $\\eta$ sets the effective memory: roughly $1/\\eta$ "
        "examples.",
        "Online ≠ streaming-only — it is also how you train when the data does not "
        "fit in RAM (<code>partial_fit</code>).",
        "Always monitor the <i>input</i> distribution of an online system, not just "
        "its accuracy.",
    ])


# ==========================================================================
# 1.6
# ==========================================================================
def s_1_6():
    section("1.6", "Instance-Based Versus Model-Based Learning")

    lead(
        "The third axis: <b>how does the system generalise to inputs it has never "
        "seen?</b> Either it compares the new case to remembered cases, or it "
        "fitted parameters and evaluates a formula."
    )

    sub("Instance-based learning")

    md(
        "The system memorises the training set and predicts using a **similarity "
        "measure**. The archetype is $k$-nearest neighbours:"
    )

    math(r"""
    \hat y(\mathbf{x}) \;=\;
    \begin{cases}
      \dfrac{1}{k}\displaystyle\sum_{i \in \mathcal{N}_k(\mathbf{x})} y^{(i)}
        & \text{(regression)}\\[14pt]
      \operatorname*{arg\,max}_{c}\;
      \displaystyle\sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}\bigl[y^{(i)} = c\bigr]
        & \text{(classification)}
    \end{cases}
    """)

    where({
        r"\mathcal{N}_k(\mathbf{x})":
            "the indices of the $k$ training points closest to $\\mathbf{x}$",
        r"d(\cdot,\cdot)":
            "the distance, usually Euclidean "
            "$\\lVert \\mathbf{x} - \\mathbf{x}^{(i)}\\rVert_2$",
    })

    md("**Training cost:** $O(1)$ — you just store the data. "
       "**Prediction cost:** $O(mn)$ per query with a brute-force scan, reducible "
       "to roughly $O(n\\log m)$ with a KD-tree or Ball-tree in low dimensions. "
       "**Memory:** $O(mn)$ forever.")

    sub("Model-based learning")

    md(
        "The system posits a parametric form, fits $\\boldsymbol\\theta$ by "
        "optimising a criterion, then throws the data away. A linear model:"
    )

    math(r"""
    \hat y \;=\; h_{\boldsymbol\theta}(\mathbf{x})
    \;=\; \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
    \;=\; \boldsymbol\theta^\top \tilde{\mathbf{x}},
    \qquad \tilde{\mathbf{x}} = \begin{bmatrix}1 \\ \mathbf{x}\end{bmatrix}
    """)

    md("**Training cost:** expensive — that is where all the work goes. "
       "**Prediction cost:** $O(n)$, a dot product. "
       "**Memory:** $O(n)$ — just the parameters.")

    table(
        ["", "Instance-based", "Model-based"],
        [["Generalises by", "Similarity to stored examples",
          "Evaluating a fitted function"],
         ["Training", "Trivial (store)", "Expensive (optimise)"],
         ["Prediction", "Expensive (search)", "Cheap (formula)"],
         ["Memory at serve time", "The entire training set", "A parameter vector"],
         ["Extrapolation", "Impossible — clamps to nearest neighbours",
          "Possible (and often dangerously wrong)"],
         ["Effect of irrelevant features", "Severe — pollutes the distance",
          "Mild — the weight shrinks toward zero"],
         ["Interpretability", "\"Because these 5 cases were similar\"",
          "\"Because $\\theta_3 = 2.4$\""],
         ["Examples", "$k$-NN, kernel density, kernel SVM (partly), "
          "case-based reasoning", "Linear/logistic regression, trees, MLPs, "
          "gradient boosting"]],
    )

    anim_header("Two ways to fill the gaps: k-NN vs linear, as k and the data change")
    md(
        "Left: the $k$-NN prediction surface as $k$ grows from 1 to 40 — piecewise "
        "constant, jagged at $k=1$, smoothing toward the global mean. Right: the "
        "linear model on the same data, which never bends but extrapolates cleanly "
        "beyond the data range (shaded)."
    )

    rng = np.random.default_rng(7)
    Xd = np.sort(rng.uniform(0, 10, 45))
    yd = 2.0 + 0.9 * Xd + 2.2 * np.sin(Xd) + rng.normal(0, 0.9, 45)
    grid = np.linspace(-2, 14, 400)

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(Xd.reshape(-1, 1), yd)
    lin_pred = lin.predict(grid.reshape(-1, 1))

    ks = list(range(1, 41))
    knn_preds = [KNeighborsRegressor(k).fit(Xd.reshape(-1, 1), yd)
                 .predict(grid.reshape(-1, 1)) for k in ks]

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("Instance-based: k-NN",
                                      "Model-based: linear regression"))
    for c in (1, 2):
        f.add_trace(go.Scatter(x=Xd, y=yd, mode="markers", showlegend=(c == 1),
                               name="training data",
                               marker=dict(color=C["train"], size=8,
                                           line=dict(color="#fff", width=1))),
                    row=1, col=c)
    f.add_trace(go.Scatter(x=grid, y=knn_preds[0], mode="lines", name="k-NN",
                           line=dict(color=C["accent"], width=3)), row=1, col=1)
    f.add_trace(go.Scatter(x=grid, y=lin_pred, mode="lines", name="linear",
                           line=dict(color=C["primary"], width=3)), row=1, col=2)
    for c in (1, 2):
        f.add_vrect(x0=-2, x1=Xd.min(), fillcolor="#EF476F", opacity=.07,
                    line_width=0, row=1, col=c)
        f.add_vrect(x0=Xd.max(), x1=14, fillcolor="#EF476F", opacity=.07,
                    line_width=0, row=1, col=c)

    frames = [go.Frame(name=str(k),
                       data=[go.Scatter(x=Xd, y=yd, mode="markers",
                                        marker=dict(color=C["train"], size=8,
                                                    line=dict(color="#fff", width=1))),
                             go.Scatter(x=Xd, y=yd, mode="markers",
                                        marker=dict(color=C["train"], size=8,
                                                    line=dict(color="#fff", width=1))),
                             go.Scatter(x=grid, y=knn_preds[i], mode="lines",
                                        line=dict(color=C["accent"], width=3)),
                             go.Scatter(x=grid, y=lin_pred, mode="lines",
                                        line=dict(color=C["primary"], width=3))],
                       traces=[0, 1, 2, 3],
                       layout=go.Layout(annotations=list(f.layout.annotations) + [
                           anim.annotate_step(f"k = {k}", color=C["accent_dark"])]))
              for i, k in enumerate(ks)]

    f.update_layout(height=420, title="Shaded = outside the training range")
    f.update_yaxes(range=[-4, 22])
    f.update_xaxes(range=[-2, 14])
    anim.animate(f, frames, duration=nav.anim_ms(220), slider_prefix="k = ")
    figure(f, "k-NN cannot extrapolate: outside the data it repeats the "
              "boundary value forever. The linear model happily extrapolates — "
              "correctly here, catastrophically elsewhere.")

    warn(
        "The curse of dimensionality hits instance-based methods hardest",
        "In $n$ dimensions, the ratio between the nearest and farthest neighbour "
        "distance tends to 1 as $n\\to\\infty$ for many distributions — every "
        "point becomes equidistant, so \"nearest neighbour\" stops meaning "
        "anything. Chapter 8 quantifies this and Chapter 8's PCA is the standard "
        "antidote.",
    )

    code_lab(
        "Head-to-head: cost, memory, extrapolation, dimensionality",
        '''import numpy as np, time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

def bench(n_features, m=4000):
    X = rng.normal(0, 1, (m, n_features))
    w = rng.normal(0, 1, n_features)
    y = X @ w + rng.normal(0, 0.4, m)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    out = {}
    for name, model in [("k-NN (k=5)", KNeighborsRegressor(5)),
                        ("linear",     LinearRegression())]:
        t0 = time.perf_counter(); model.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); s = model.score(Xte, yte); t_pred = time.perf_counter() - t0
        mem = Xtr.nbytes / 1e6 if "k-NN" in name else (n_features + 1) * 8 / 1e6
        out[name] = (t_fit * 1e3, t_pred * 1e3, s, mem)
    return out

print(f"{'d':>4} {'model':<12} {'fit ms':>9} {'pred ms':>9} {'R^2':>8} {'serve MB':>10}")
for d in [2, 5, 20, 100]:
    for name, (tf, tp, s, mem) in bench(d).items():
        print(f"{d:>4} {name:<12} {tf:>9.2f} {tp:>9.2f} {s:>8.3f} {mem:>10.3f}")

print()
print("As d grows, k-NN's R^2 collapses (curse of dimensionality) while the")
print("linear model -- correctly specified here -- is unaffected.")
print("Note also: k-NN fits instantly but must carry the whole training set to")
print("production; the linear model carries d+1 floats.")

# ---- extrapolation ---------------------------------------------------------
X1 = np.linspace(0, 10, 60).reshape(-1, 1); y1 = 2 + 0.9 * X1.ravel()
far = np.array([[25.0]])
print()
print(f"true value at x=25 : {2 + 0.9*25:.2f}")
print(f"linear says        : {LinearRegression().fit(X1, y1).predict(far)[0]:.2f}")
print(f"k-NN says          : {KNeighborsRegressor(5).fit(X1, y1).predict(far)[0]:.2f}"
      "   <- clamped to the boundary")
''',
        key="ch01_inst",
    )

    quiz(
        "You must deploy a model to a smart doorbell with 8 MB of storage, and "
        "your training set is 400 MB. Which family is ruled out?",
        ["Model-based", "Instance-based", "Neither", "Both"],
        1,
        "Instance-based methods must ship the training set. A fitted linear model "
        "or a small tree is a few kilobytes. Chapter 19 covers on-device "
        "deployment properly.",
        key="ch01q3",
    )

    keypoints([
        "Instance-based: cheap to train, expensive to serve, cannot extrapolate, "
        "dies in high dimensions.",
        "Model-based: expensive to train, cheap to serve, extrapolates (for better "
        "or worse), robust to irrelevant features.",
        "The distinction is orthogonal to supervision and to batch/online — you "
        "choose a point in a 3-D design space.",
        "Kernel SVMs (Ch. 5) sit in between: model-based in spirit, but the "
        "support vectors are stored instances.",
    ])


# ==========================================================================
# 1.7
# ==========================================================================
def s_1_7():
    section("1.7", "Main Challenges — Bad Data and Bad Algorithms")

    lead(
        "Projects fail for a small number of recurring reasons, and they split "
        "cleanly into <b>bad data</b> (four failure modes) and <b>bad algorithms</b> "
        "(two failure modes). Learn to name which one you are looking at — the "
        "remedy differs completely."
    )

    st.markdown("### Part A · Bad data")

    sub("1 · Insufficient quantity of training data")

    md(
        "Almost every learning algorithm needs thousands of examples for a "
        "non-trivial problem, and millions for perception tasks. The famous "
        "*unreasonable effectiveness of data* result is that on some tasks the "
        "choice of algorithm matters far less than the amount of data, once you "
        "are past a few million examples — different algorithms converge to "
        "similar performance."
    )

    idea(
        "But data is not free and algorithms are not equal",
        "The correct reading is <i>not</i> \"algorithms don't matter\". It is: "
        "in the <b>data-rich</b> regime, marginal data beats marginal modelling; "
        "in the <b>data-poor</b> regime — which is where most real projects live — "
        "inductive bias, regularisation and transfer learning dominate. Knowing "
        "which regime you are in is half the job.",
    )

    sub("2 · Nonrepresentative training data")

    md(
        "Your training set must be drawn from the *same* distribution as the data "
        "you will serve. Two ways this breaks:"
    )

    table(
        ["Failure", "Mechanism", "Classic illustration"],
        [["<b>Sampling noise</b>",
          "The sample is too small, so it is unrepresentative by chance alone. "
          "Shrinks as $O(1/\\sqrt{m})$.",
          "A survey of 30 people gives a ±18 pp margin of error."],
         ["<b>Sampling bias</b>",
          "The sampling <i>method</i> is flawed, so the sample is unrepresentative "
          "no matter how large it gets. Does <b>not</b> shrink with $m$.",
          "Polling by telephone in 1936, when phone owners were disproportionately "
          "wealthy — a landslide prediction that was exactly backwards."]],
    )

    pitfall(
        "More data does not fix sampling bias",
        "This is the single most expensive misconception in applied ML. Sampling "
        "noise is a variance problem and collecting more data cures it. Sampling "
        "bias is a <i>bias</i> problem: a biased estimator converges — to the wrong "
        "number. If your labelled fraud examples come only from cases the current "
        "rules already caught, ten million more of them will not teach the model "
        "about the fraud the rules miss. <b>Survivorship bias</b> is the same "
        "disease.",
    )

    anim_header("Sampling noise shrinks; sampling bias does not")
    md(
        "Two estimators of the same population mean (true value = 0). The blue one "
        "samples uniformly; the red one systematically over-samples the right tail. "
        "Watch as $m$ grows: blue converges, red converges to the *wrong answer*."
    )

    rng = np.random.default_rng(11)
    pop = rng.normal(0, 1, 200_000)
    biased_pool = pop[pop > -0.35]
    ms = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
    unb, bia, unb_lo, unb_hi = [], [], [], []
    for m in ms:
        reps_u = [np.mean(rng.choice(pop, m)) for _ in range(200)]
        reps_b = [np.mean(rng.choice(biased_pool, m)) for _ in range(200)]
        unb.append(np.mean(reps_u)); bia.append(np.mean(reps_b))
        unb_lo.append(np.percentile(reps_u, 2.5))
        unb_hi.append(np.percentile(reps_u, 97.5))

    frames = []
    for k in range(1, len(ms) + 1):
        frames.append(go.Frame(name=str(ms[k - 1]), data=[
            go.Scatter(x=ms[:k], y=unb_hi[:k] + unb_lo[:k][::-1] if False else unb[:k],
                       mode="lines+markers", line=dict(color=C["train"], width=3)),
            go.Scatter(x=ms[:k] + ms[:k][::-1],
                       y=unb_hi[:k] + unb_lo[:k][::-1],
                       fill="toself", fillcolor=alpha(C["train"], .16),
                       line=dict(width=0), hoverinfo="skip"),
            go.Scatter(x=ms[:k], y=bia[:k], mode="lines+markers",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[
            anim.annotate_step(f"m = {ms[k-1]}   unbiased = {unb[k-1]:+.4f}   "
                               f"biased = {bia[k-1]:+.4f}")])))

    f = go.Figure(data=[
        go.Scatter(x=ms[:1], y=unb[:1], mode="lines+markers",
                   name="uniform sampling", line=dict(color=C["train"], width=3)),
        go.Scatter(x=ms[:1] + ms[:1], y=unb_hi[:1] + unb_lo[:1],
                   fill="toself", fillcolor=alpha(C["train"], .16),
                   line=dict(width=0), name="95 % band", hoverinfo="skip"),
        go.Scatter(x=ms[:1], y=bia[:1], mode="lines+markers",
                   name="biased sampling", line=dict(color=C["danger"], width=3)),
    ])
    f.add_hline(y=0, line_dash="dot", line_color=C["truth"],
                annotation_text="true population mean")
    f.update_layout(height=420, xaxis_type="log", xaxis_title="sample size m",
                    yaxis_title="estimated mean", yaxis=dict(range=[-0.6, 0.95]),
                    title="Noise vanishes, bias persists")
    anim.animate(f, frames, duration=nav.anim_ms(450), slider_prefix="m = ")
    figure(f)

    sub("3 · Poor-quality data")

    md(
        """
Errors, outliers and noise. Data cleaning is where most of a practitioner's time
goes, and the decisions are judgement calls:

* **Outliers** — either discard them or fix them by hand. But first ask whether
  they are errors or the phenomenon you are trying to model (fraud *is* the
  outlier).
* **Missing features** — you have four options and they are not equivalent:
  1. drop the *attribute* entirely,
  2. drop the *instances* that are missing it,
  3. **impute** a value (median, mean, most-frequent, or a model-based
     imputation such as $k$-NN or iterative imputation),
  4. train two models, one with and one without the feature.

  Chapter 2 implements all of these with `SimpleImputer` and friends.
        """
    )

    note(
        "Missingness is itself information",
        "Whether a value is missing is often predictive — a customer who declined "
        "to state their income differs systematically from one who stated it. "
        "Add a binary <code>was_missing</code> indicator column alongside the "
        "imputed value; scikit-learn's <code>SimpleImputer(add_indicator=True)</code> "
        "does this for you.",
    )

    sub("4 · Irrelevant features")

    md(
        "*Garbage in, garbage out.* The countermeasure is **feature engineering**, "
        "which has three parts: **feature selection** (keep the useful ones), "
        "**feature extraction** (combine features into more informative ones — "
        "PCA in Chapter 8 is the canonical automatic method), and **creating new "
        "features** by gathering new data."
    )

    st.markdown("### Part B · Bad algorithms")

    sub("5 · Overfitting the training data")

    md(
        "The model performs well on the training data but generalises poorly. "
        "Formally, the generalisation gap is large:"
    )

    math(r"""
    \hat R_{\mathcal{S}}(h) \ll R(h)
    \qquad\Longleftrightarrow\qquad
    \text{overfitting}
    """)

    md("It happens when the model is **too complex relative to the amount and "
       "noisiness of the data**. The three cures:")

    table(
        ["Cure", "Mechanism", "Concretely"],
        [["Simplify the model",
          "Shrink $\\mathcal{H}$, so there are fewer ways to fit noise",
          "Fewer parameters, lower polynomial degree, shallower tree, fewer layers"],
         ["Regularise",
          "Keep $\\mathcal{H}$ large but penalise complexity, adding a term to the "
          "objective",
          "$\\min_\\theta \\hat R(\\theta) + \\alpha\\Omega(\\theta)$ — ridge, lasso, "
          "dropout, weight decay, early stopping"],
         ["Gather more / cleaner data",
          "Reduce estimation error directly; fix the noise the model is chasing",
          "More rows, better labels, outlier removal"]],
    )

    md("The regularisation hyperparameter $\\alpha$ controls the trade-off "
       "explicitly:")

    math(r"""
    J(\boldsymbol\theta) \;=\;
    \underbrace{\frac{1}{m}\sum_{i=1}^{m}
      \ell\bigl(h_{\boldsymbol\theta}(\mathbf{x}^{(i)}), y^{(i)}\bigr)}_{\text{fit the data}}
    \;+\;
    \alpha \underbrace{\Omega(\boldsymbol\theta)}_{\text{stay simple}}
    """)

    warn(
        "A hyperparameter is not a parameter",
        "$\\boldsymbol\\theta$ is learned by the algorithm; $\\alpha$ is set "
        "<i>before</i> training and is not touched by it. Tuning hyperparameters "
        "is a separate optimisation problem, solved on a validation set — "
        "see §1.8.",
    )

    sub("6 · Underfitting the training data")

    md(
        "The mirror image: the model is too simple to capture the structure, so it "
        "does badly on training data *and* test data. Cures: a more powerful model, "
        "better features, or **reduce** the regularisation constraints."
    )

    anim_header("The two failure modes on one axis")
    md("Sweep model capacity and watch which failure mode you are in. The green "
       "band is where a practitioner wants to live.")

    rng = np.random.default_rng(21)
    Xo = np.sort(rng.uniform(-3, 3, 30))
    yo = np.sin(1.5 * Xo) + rng.normal(0, 0.35, 30)
    Xv = np.sort(rng.uniform(-3, 3, 400))
    yv = np.sin(1.5 * Xv) + rng.normal(0, 0.35, 400)
    g = np.linspace(-3.2, 3.2, 300)
    degs = list(range(1, 19))
    tr, va, cv = [], [], []
    for d in degs:
        c = np.polyfit(Xo, yo, d)
        cv.append(np.polyval(c, g))
        tr.append(np.sqrt(np.mean((np.polyval(c, Xo) - yo) ** 2)))
        va.append(np.sqrt(np.mean((np.polyval(c, Xv) - yv) ** 2)))

    f = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                      subplot_titles=("The fit", "The two errors"))
    f.add_trace(go.Scatter(x=Xo, y=yo, mode="markers", name="train",
                           marker=dict(color=C["train"], size=8,
                                       line=dict(color="#fff", width=1))), 1, 1)
    f.add_trace(go.Scatter(x=g, y=cv[0], mode="lines", name="model",
                           line=dict(color=C["primary"], width=3.2)), 1, 1)
    f.add_trace(go.Scatter(x=g, y=np.sin(1.5 * g), mode="lines", name="truth",
                           line=dict(color=C["truth"], width=2, dash="dot")), 1, 1)
    f.add_trace(go.Scatter(x=degs, y=tr, mode="lines+markers", name="train RMSE",
                           line=dict(color=C["train"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=degs, y=va, mode="lines+markers", name="test RMSE",
                           line=dict(color=C["test"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=[degs[0]], y=[va[0]], mode="markers",
                           name="current", showlegend=False,
                           marker=dict(color=C["gradient"], size=15,
                                       symbol="circle-open",
                                       line=dict(width=3))), 1, 2)

    frames = []
    for i, d in enumerate(degs):
        tag = "UNDERFITTING" if d <= 2 else ("GOOD" if d <= 7 else "OVERFITTING")
        col = C["warning"] if d <= 2 else (C["success"] if d <= 7 else C["danger"])
        frames.append(go.Frame(name=str(d), traces=[0, 1, 2, 3, 4, 5], data=[
            go.Scatter(x=Xo, y=yo, mode="markers",
                       marker=dict(color=C["train"], size=8,
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=g, y=cv[i], mode="lines", line=dict(color=col, width=3.2)),
            go.Scatter(x=g, y=np.sin(1.5 * g), mode="lines",
                       line=dict(color=C["truth"], width=2, dash="dot")),
            go.Scatter(x=degs, y=tr, mode="lines+markers",
                       line=dict(color=C["train"], width=3)),
            go.Scatter(x=degs, y=va, mode="lines+markers",
                       line=dict(color=C["test"], width=3)),
            go.Scatter(x=[d], y=[va[i]], mode="markers",
                       marker=dict(color=col, size=16, symbol="circle-open",
                                   line=dict(width=3))),
        ], layout=go.Layout(annotations=list(f.layout.annotations) + [
            anim.annotate_step(f"degree {d} — {tag}", color=col)])))

    f.update_yaxes(range=[-2.6, 2.6], row=1, col=1)
    f.update_xaxes(range=[-3.3, 3.3], row=1, col=1)
    f.update_yaxes(type="log", title_text="RMSE", row=1, col=2)
    f.update_xaxes(title_text="polynomial degree", row=1, col=2)
    f.update_layout(height=440)
    anim.animate(f, frames, duration=nav.anim_ms(400), slider_prefix="degree ")
    figure(f)

    code_lab(
        "Diagnose it yourself: the four data failures",
        '''import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

rng = np.random.default_rng(0)
m = 300
x = rng.uniform(-3, 3, m)
y = np.sin(1.5 * x) + rng.normal(0, 0.3, m)

def rmse_cv(X, y, degree=8, alpha=1e-6):
    pipe = make_pipeline(PolynomialFeatures(degree), StandardScaler(),
                         Ridge(alpha=alpha))
    return -cross_val_score(pipe, X, y, cv=5,
                            scoring="neg_root_mean_squared_error").mean()

X = x.reshape(-1, 1)
print("BASELINE                        RMSE = %.3f" % rmse_cv(X, y))

# 1. INSUFFICIENT DATA
for k in [15, 30, 80, 300]:
    print(f"  m = {k:>4}                       RMSE = {rmse_cv(X[:k], y[:k]):.3f}")

# 2. SAMPLING BIAS: only train where x > 0, test everywhere
keep = x > 0
pipe = make_pipeline(PolynomialFeatures(8), StandardScaler(), Ridge(1e-6))
pipe.fit(X[keep], y[keep])
bias_rmse = np.sqrt(np.mean((pipe.predict(X[~keep]) - y[~keep]) ** 2))
print(f"\\n2. SAMPLING BIAS (train x>0, test x<0)  RMSE = {bias_rmse:.3f}  <- exploded")

# 3. POOR QUALITY: inject 5 % wild outliers
y_dirty = y.copy()
idx = rng.choice(m, m // 20, replace=False)
y_dirty[idx] += rng.normal(0, 8, len(idx))
print(f"3. 5 % OUTLIERS                        RMSE = {rmse_cv(X, y_dirty):.3f}")
# and the cure
from scipy import stats
mask = np.abs(stats.zscore(y_dirty)) < 3
print(f"   after 3-sigma clipping              RMSE = {rmse_cv(X[mask], y_dirty[mask]):.3f}")

# 4. IRRELEVANT FEATURES: append 40 pure-noise columns
X_noisy = np.c_[X, rng.normal(0, 1, (m, 40))]
print(f"4. +40 NOISE FEATURES                  RMSE = {rmse_cv(X_noisy, y, degree=2):.3f}")
print(f"   (1 informative feature, degree 2)   RMSE = {rmse_cv(X, y, degree=2):.3f}")

# 5/6. OVER- vs UNDER-FITTING via the regularisation dial
print("\\nRegularisation sweep (degree 15):")
for a in [1e-8, 1e-4, 1e-2, 1, 100, 10_000]:
    print(f"   alpha = {a:>9}   RMSE = {rmse_cv(X, y, degree=15, alpha=a):.3f}")
''',
        key="ch01_challenges",
    )

    keypoints([
        "<b>Four data failures</b>: too little, unrepresentative, poor quality, "
        "irrelevant features.",
        "<b>Two algorithm failures</b>: overfitting and underfitting.",
        "Sampling <i>noise</i> is cured by more data; sampling <i>bias</i> is not.",
        "Overfitting cures: simplify, regularise, get more data. Underfitting "
        "cures: more capacity, better features, less regularisation.",
        "$\\alpha$ (regularisation strength) is the dial between the two — and it "
        "is a hyperparameter, tuned on validation data.",
    ])


# ==========================================================================
# 1.8
# ==========================================================================
def s_1_8():
    section("1.8", "Testing and Validating")

    lead(
        "You cannot know whether a model generalises without holding data back. "
        "This section defines the discipline that makes every number you report "
        "in the next 18 chapters trustworthy."
    )

    sub("The three-way split")

    md(
        """
Split your data **once**, before you look at anything:

| Set | Typical size | Used for | Touched how often |
|---|---|---|---|
| **Training set** | 60–80 % | Fitting $\\boldsymbol\\theta$ | Constantly |
| **Validation (dev) set** | 10–20 % | Choosing $\\mathcal{H}$ and hyperparameters | Many times |
| **Test set** | 10–20 % | One final, honest estimate of $R(h)$ | **Once** |

The error measured on the test set is called the **generalisation error** or
**out-of-sample error**.
        """
    )

    proof(
        "Why a validation set is not optional",
        "Suppose you evaluate $N$ candidate models on the same held-out set and "
        "report the best. Even if all $N$ are equally good, the maximum of $N$ "
        "noisy estimates is optimistically biased by roughly "
        "$\\sigma\\sqrt{2\\log N}$ (the expected maximum of $N$ Gaussians). "
        "With $N = 100$ candidates and $\\sigma = 1\\%$, that is a free "
        "$\\approx 3$ pp of illusory accuracy. Selecting on the validation set "
        "and reporting on an untouched test set removes the bias from the "
        "reported number.",
    )

    anim_header("Selection bias: how the 'best' model lies to you")
    md(
        "All 60 candidate models below have *identical* true accuracy of 80 %. "
        "Only sampling noise separates them. Each frame evaluates one more "
        "candidate and keeps the running maximum. Watch the reported best drift "
        "upward with no real improvement whatsoever."
    )

    rng = np.random.default_rng(5)
    n_val = 400
    true_acc = 0.80
    val_scores = rng.binomial(n_val, true_acc, 60) / n_val
    test_scores = rng.binomial(2000, true_acc, 60) / 2000
    running_best, best_test = [], []
    bi = 0
    for i in range(60):
        if val_scores[i] > val_scores[bi]:
            bi = i
        running_best.append(val_scores[bi])
        best_test.append(test_scores[bi])

    frames = []
    for k in range(1, 61):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=np.arange(1, k + 1), y=val_scores[:k], mode="markers",
                       marker=dict(color=C["valid"], size=8)),
            go.Scatter(x=np.arange(1, k + 1), y=running_best[:k], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=np.arange(1, k + 1), y=best_test[:k], mode="lines",
                       line=dict(color=C["success"], width=3, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{k} candidates tried   ·   reported (val) = {running_best[k-1]:.3f}"
            f"   ·   honest (test) = {best_test[k-1]:.3f}"
            f"   ·   inflation = {running_best[k-1]-best_test[k-1]:+.3f}",
            color=C["danger"])])))

    f = go.Figure(data=[
        go.Scatter(x=[1], y=[val_scores[0]], mode="markers",
                   name="each candidate (validation)",
                   marker=dict(color=C["valid"], size=8)),
        go.Scatter(x=[1], y=[running_best[0]], mode="lines",
                   name="best-so-far on validation",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[1], y=[best_test[0]], mode="lines",
                   name="that model's TEST accuracy",
                   line=dict(color=C["success"], width=3, dash="dash")),
    ])
    f.add_hline(y=true_acc, line_dash="dot", line_color=C["truth"],
                annotation_text="true accuracy of every candidate = 0.80")
    f.update_layout(height=430, xaxis_title="number of candidates evaluated",
                    yaxis_title="accuracy", yaxis=dict(range=[0.74, 0.88]),
                    title="The winner's curse in model selection",
                    legend=dict(orientation="h", y=1.16, x=0))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="candidates = ")
    figure(f, "The red line is what you would report if you tuned on the test "
              "set. The green dashed line is the truth. The gap is pure "
              "selection bias.")

    sub("Cross-validation: when you cannot afford a validation set")

    md(
        "Holding out 20 % is wasteful with small data, and a single split is "
        "itself noisy. **$k$-fold cross-validation** trains $k$ times, each time "
        "holding out a different fold:"
    )

    math(r"""
    \widehat{\mathrm{CV}}_k \;=\; \frac{1}{k}\sum_{j=1}^{k}
      \frac{1}{|\mathcal{F}_j|} \sum_{i \in \mathcal{F}_j}
      \ell\Bigl(h^{(-j)}\bigl(\mathbf{x}^{(i)}\bigr),\, y^{(i)}\Bigr)
    """)

    where({
        r"\mathcal{F}_j": "the indices in fold $j$ (the folds partition the data)",
        r"h^{(-j)}": "the model trained on <i>everything except</i> fold $j$",
        r"k": "the number of folds — 5 and 10 are the standard choices",
    })

    table(
        ["Variant", "When to use it"],
        [["<code>KFold</code>", "Plain regression, i.i.d. rows"],
         ["<code>StratifiedKFold</code>",
          "Classification — preserves class proportions in every fold. "
          "<b>Use this by default for classification.</b>"],
         ["<code>GroupKFold</code>",
          "Rows are clustered (multiple scans per patient, multiple rows per "
          "user). Prevents the same group appearing in train and test."],
         ["<code>TimeSeriesSplit</code>",
          "Temporal data — every training fold precedes its test fold. "
          "<b>Never shuffle a time series.</b>"],
         ["<code>LeaveOneOut</code>",
          "Tiny datasets. Nearly unbiased but very high variance and $m$ fits."]],
    )

    sub("Hyperparameter tuning and model selection")

    md(
        """
The workflow, stated once and used for the rest of the course:

1. Split off the **test set** and put it in a drawer.
2. On the remaining data, use cross-validation to compare hypothesis spaces and
   hyperparameter settings.
3. Pick the winner. Refit it on **all** of the non-test data.
4. Evaluate **once** on the test set. Report that number.
5. If step 4 disappoints you, you are *not allowed* to go back to step 2 and
   iterate on the test set. If you do, the test set has become a validation set
   and you need a fresh one.
        """
    )

    sub("Data mismatch — when validation and test come from different worlds")

    md(
        "Sometimes you have a lot of easy-to-get data (web images) and a little "
        "hard-to-get data that matches production (photos from the actual app). "
        "The validation and test sets **must** come from the production "
        "distribution. But then a bad validation score is ambiguous: is the model "
        "overfitting, or is it a *distribution mismatch*?"
    )

    idea(
        "The train-dev set resolves the ambiguity",
        "Hold out a slice of the <i>training</i> distribution too — call it "
        "<b>train-dev</b>. Then:<br>"
        "&nbsp;&nbsp;• train ≫ train-dev &nbsp;⇒&nbsp; <b>overfitting</b> (variance)<br>"
        "&nbsp;&nbsp;• train ≈ train-dev ≫ dev &nbsp;⇒&nbsp; <b>data mismatch</b><br>"
        "&nbsp;&nbsp;• dev ≫ test &nbsp;⇒&nbsp; you <b>overfitted the dev set</b> by "
        "tuning too much<br>"
        "&nbsp;&nbsp;• all high &nbsp;⇒&nbsp; <b>underfitting</b> (bias)",
    )

    table(
        ["Set", "Distribution", "Purpose"],
        [["train", "web (plentiful)", "fit parameters"],
         ["train-dev", "web (plentiful)", "detect overfitting"],
         ["dev / validation", "app (scarce)", "tune hyperparameters, detect mismatch"],
         ["test", "app (scarce)", "final honest number"]],
        "The four-way split for the data-mismatch scenario.",
    )

    pitfall(
        "The No Free Lunch theorem",
        "Wolpert (1996): averaged over <i>all</i> possible data-generating "
        "distributions, every learning algorithm has identical expected "
        "performance. The practical reading is not \"nothing works\" — it is that "
        "<b>every model embodies assumptions</b>, and the only way to find out "
        "whether yours hold is to evaluate empirically. There is no a-priori "
        "best model. Hence: try several families, and always measure.",
    )

    code_lab(
        "The full honest protocol, end to end",
        '''import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                           n_redundant=4, class_sep=0.9, random_state=0)

# ---- STEP 1: the test set goes in a drawer and is not opened until step 4 --
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"rest = {len(X_rest)}   test = {len(X_test)}  (test is now untouchable)")

# ---- STEP 2: compare families with stratified CV on X_rest ONLY ------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
candidates = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "svm-rbf":  make_pipeline(StandardScaler(), SVC()),
    "forest":   RandomForestClassifier(n_estimators=200, random_state=0),
}
print("\\nCross-validated accuracy on the training portion:")
for name, mdl in candidates.items():
    s = cross_val_score(mdl, X_rest, y_rest, cv=cv, scoring="accuracy")
    print(f"  {name:<10} {s.mean():.4f} +/- {s.std():.4f}")

# ---- STEP 3: tune the winner's hyperparameters, still on X_rest ------------
grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.05, 0.2]},
    cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_rest, y_rest)
print(f"\\nbest hyperparameters : {grid.best_params_}")
print(f"best CV score        : {grid.best_score_:.4f}   <- OPTIMISTIC (selected on)")

# ---- STEP 4: open the drawer. Exactly once. -------------------------------
final = grid.best_estimator_                 # already refit on all of X_rest
test_acc = accuracy_score(y_test, final.predict(X_test))
print(f"HONEST test accuracy : {test_acc:.4f}")
print(f"selection inflation  : {grid.best_score_ - test_acc:+.4f}")
print()
print(classification_report(y_test, final.predict(X_test), digits=3))
print("Now STOP. Any further tuning against this number contaminates it.")
''',
        key="ch01_protocol",
    )

    quiz(
        "You ran GridSearchCV over 240 configurations and the best CV accuracy was "
        "0.912. What is the right thing to report as the model's accuracy?",
        ["0.912, it is a cross-validated number so it is unbiased",
         "The accuracy of the refit best model on the untouched test set",
         "The mean CV accuracy over all 240 configurations",
         "The training accuracy of the best model"],
        1,
        "0.912 is the maximum over 240 noisy estimates and is optimistically "
        "biased — exactly the winner's curse animated above. Only the untouched "
        "test set gives an honest number.",
        key="ch01q4",
    )

    keypoints([
        "Split first, look second. Test set opened <b>once</b>.",
        "Validation set exists so that model <i>selection</i> does not contaminate "
        "the reported number.",
        "Use <code>StratifiedKFold</code> for classification, "
        "<code>GroupKFold</code> when rows cluster, <code>TimeSeriesSplit</code> "
        "for temporal data.",
        "Data mismatch is diagnosed with a <b>train-dev</b> set — it separates "
        "variance from distribution shift.",
        "No Free Lunch: no model is best a priori, so try several and measure.",
    ])


# ==========================================================================
# 1.9
# ==========================================================================
def s_1_9():
    section("1.9", "Exercises & Chapter Review")

    lead("Work these before moving to Chapter 2. Answers unfold beneath each "
         "prompt.")

    exercise(
        1, "How would you define machine learning?",
        "The construction of systems that improve their performance on a task "
        "$T$, as measured by $P$, through exposure to experience $E$ — i.e. by "
        "learning from data rather than by being explicitly programmed with rules. "
        "Formally: choosing $h \\in \\mathcal{H}$ to minimise an empirical risk "
        "that stands in for the true risk.")

    exercise(
        2, "Name four types of problem where ML shines.",
        "**(1)** Problems requiring long, brittle lists of hand-written rules "
        "(spam). **(2)** Complex problems with no known algorithmic solution "
        "(speech recognition). **(3)** Fluctuating environments where the system "
        "must adapt (fraud). **(4)** Getting insight from large amounts of data "
        "(data mining).")

    exercise(
        3, "What is a labelled training set?",
        "A training set in which each instance carries the desired solution — the "
        "target or label. It is what makes *supervised* learning possible, "
        "because it lets you write a loss $\\ell(h(\\mathbf{x}), y)$ that "
        "compares a prediction to a truth.")

    exercise(
        4, "What are the two most common supervised tasks?",
        "**Regression** ($\\mathcal{Y} = \\mathbb{R}$, typically squared loss) and "
        "**classification** ($\\mathcal{Y}$ finite, typically cross-entropy loss).")

    exercise(
        5, "Name four common unsupervised tasks.",
        "**Clustering** (k-means, DBSCAN, GMM), **dimensionality reduction** "
        "(PCA, LLE, t-SNE), **anomaly / novelty detection** (isolation forest, "
        "GMM density, one-class SVM), and **association rule learning** "
        "(apriori, eclat). Density estimation and generative modelling belong "
        "here too.")

    exercise(
        6, "What type of algorithm would let a robot walk over various unknown "
        "terrains?",
        "**Reinforcement learning** — the robot takes actions, receives delayed "
        "rewards (distance travelled, energy used, not falling over), and learns "
        "a policy that maximises expected discounted return. Chapter 18. "
        "(It *could* be framed as supervised learning if you had a database of "
        "correct joint torques, but you almost never do.)")

    exercise(
        7, "What type of algorithm segments customers into groups?",
        "If you do not know the groups in advance: **clustering** "
        "(unsupervised). If you already know the segments and have labelled "
        "examples: **classification** (supervised).")

    exercise(
        8, "Is spam detection supervised or unsupervised?",
        "Supervised — the training set consists of emails *with* their spam/ham "
        "labels.")

    exercise(
        9, "What is an online learning system?",
        "One that learns incrementally, updating its parameters as each instance "
        "or mini-batch arrives, rather than being retrained from scratch on the "
        "full dataset. It adapts to change and to continuous data streams, and "
        "it can train on datasets that do not fit in memory.")

    exercise(
        10, "What is out-of-core learning?",
        "Training on data too large for main memory, by streaming it in chunks "
        "and applying an online (incremental) learning algorithm — "
        "`partial_fit` in scikit-learn — to each chunk in turn.")

    exercise(
        11, "What type of algorithm relies on a similarity measure to make "
        "predictions?",
        "An **instance-based** algorithm, e.g. $k$-nearest neighbours. It "
        "memorises the training examples and, for a new instance, uses a "
        "similarity/distance measure to find the most similar stored examples "
        "and generalise from them.")

    exercise(
        12, "What is the difference between a model parameter and a "
        "hyperparameter?",
        "A **model parameter** ($\\boldsymbol\\theta$) is learned by the "
        "algorithm from the data — the slope and intercept of a linear model, "
        "the weights of a network. A **hyperparameter** ($\\alpha$, $k$, tree "
        "depth, learning rate) is a setting of the *algorithm itself*, fixed "
        "before training and untouched by it, and tuned on a validation set.")

    exercise(
        13, "What do model-based algorithms search for? What is their most common "
        "strategy? How do they make predictions?",
        "They search for the **parameter values** that make the model generalise "
        "best. The most common strategy is to **minimise a cost function** — an "
        "empirical risk plus, usually, a regularisation penalty — typically by "
        "gradient descent or a closed-form solution. They predict by feeding the "
        "new instance's features into the fitted function $h_{\\boldsymbol\\theta}$.")

    exercise(
        14, "Name four main challenges in machine learning.",
        "Insufficient data; non-representative data (sampling bias); poor-quality "
        "data (noise, outliers, missing values); irrelevant features; overfitting; "
        "underfitting. Any four.")

    exercise(
        15, "Your model performs great on training data but generalises poorly. "
        "What is happening and what are three solutions?",
        "It is **overfitting**. Solutions: **(1)** simplify the model — fewer "
        "parameters, fewer features, or a more constrained family; **(2)** gather "
        "more training data; **(3)** reduce the noise in the training data (fix "
        "errors, remove outliers). A fourth, and usually the first thing to try: "
        "**regularise** — add a penalty $\\alpha\\Omega(\\boldsymbol\\theta)$.")

    exercise(
        16, "What is a test set and why use one?",
        "A portion of the data held out and never used for fitting or tuning. It "
        "gives an unbiased estimate of the generalisation error $R(h)$ on data "
        "the model has never seen, *before* the model is deployed.")

    exercise(
        17, "What is the purpose of a validation set?",
        "To compare candidate models and hyperparameter settings. Selection must "
        "happen on the validation set so that the test set remains uncontaminated "
        "and its estimate stays honest — otherwise you suffer the winner's curse "
        "animated in §1.8.")

    exercise(
        18, "What is the train-dev set, when do you need it, and how is it used?",
        "A held-out slice of the **training** distribution, used when the "
        "training data comes from a different distribution than the validation "
        "and test data (which must match production). Comparing train / train-dev "
        "/ dev errors tells you whether a poor dev score is caused by "
        "**overfitting** (train ≪ train-dev) or by **data mismatch** "
        "(train ≈ train-dev ≪ dev).")

    exercise(
        19, "What can go wrong if you tune hyperparameters using the test set?",
        "You overfit the test set. The reported generalisation error becomes "
        "optimistically biased — often badly so — and the model will "
        "underperform in production. The test set has silently become a "
        "validation set and no longer measures what you claim it measures.")

    rule()

    keypoints([
        "Machine learning = build $\\mathcal{H}$, define $\\ell$, minimise "
        "$\\hat R_{\\mathcal{S}}$, control the gap to $R$.",
        "Three design axes: <b>supervision</b> (§1.4), <b>batch vs online</b> (§1.5), "
        "<b>instance vs model based</b> (§1.6).",
        "Six failure modes: four in the data, two in the algorithm (§1.7).",
        "One discipline that keeps you honest: split, cross-validate, open the "
        "test set once (§1.8).",
        "Next: Chapter 2 walks a complete project through this entire pipeline on "
        "real data.",
    ], title="Chapter 1 in five lines")

    refs([
        ("Mitchell, T. — *Machine Learning*, definition of the $(T, P, E)$ triple",
         "McGraw-Hill, 1997"),
        ("Wolpert, D. — *The Lack of A Priori Distinctions Between Learning "
         "Algorithms* (No Free Lunch)",
         "https://doi.org/10.1162/neco.1996.8.7.1341"),
        ("Halevy, Norvig & Pereira — *The Unreasonable Effectiveness of Data*",
         "https://doi.org/10.1109/MIS.2009.36"),
        ("Vapnik, V. — *Statistical Learning Theory* (ERM and the generalisation "
         "gap)", "Wiley, 1998"),
    ])


# ==========================================================================
SECTIONS = [
    ("1.1", "What Is Machine Learning?", s_1_1),
    ("1.2", "Why Use Machine Learning?", s_1_2),
    ("1.3", "Examples of Applications", s_1_3),
    ("1.4", "Types of ML Systems — Supervision", s_1_4),
    ("1.5", "Batch vs Online Learning", s_1_5),
    ("1.6", "Instance-Based vs Model-Based", s_1_6),
    ("1.7", "Main Challenges", s_1_7),
    ("1.8", "Testing and Validating", s_1_8),
    ("1.9", "Exercises & Review", s_1_9),
]

nav.render_chapter(CH, SECTIONS)
