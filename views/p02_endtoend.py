"""Chapter 2 — End-to-End Machine Learning Project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, datasets as ds, nav
from core.lecture import (anim_header, derive, exercise, figure, hero, idea,
                          keypoints, lead, math, md, note, pitfall, proof,
                          quiz, refs, rule, section, sub, table, tip, warn,
                          where, codenote)
from core.palette import C, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "ch02"

hero(
    kicker="Part I · Chapter 2",
    title="End-to-End Machine Learning Project",
    blurb=(
        "One complete project, start to finish, on real housing data: frame the "
        "business problem, pick a metric and justify it mathematically, sample a "
        "test set that is actually representative, explore, clean, encode, scale, "
        "pipeline, train, cross-validate, tune, and finally launch and monitor. "
        "This is the chapter you will physically return to on every future project."
    ),
    chips=["California housing", "8 sub-sections", "6 animations",
           "9 code labs", "The reference workflow"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_2_1():
    section("2.1", "Look at the Big Picture — Frame the Problem")

    lead(
        "Before touching data you must answer three questions: what is the "
        "business objective, what does the current solution look like, and what "
        "kind of learning problem is this? Getting these wrong costs weeks."
    )

    sub("The scenario")

    md(
        """
You work for a real-estate company. You are given California census data with
one row per **district** (a district is roughly 600–3 000 people) and columns for
population, median income, median housing age, total rooms, and so on. Your
model must predict the **median house value** of a district. That prediction
feeds a downstream system which decides whether investing in a given area is
worthwhile.

That last sentence is the important one — it tells you the **cost of an error**,
and the cost of an error tells you the metric.
        """
    )

    sub("Question 1 · What exactly is the business objective?")

    table(
        ["Ask", "Why it matters"],
        [["What is the output used for?",
          "Determines the metric, the required latency, and whether you need "
          "calibrated uncertainty or just a point estimate."],
         ["What does the current solution look like?",
          "Gives you a baseline to beat and often a hard performance floor. Here: "
          "manual expert estimates with about <b>15 % relative error</b>."],
         ["How accurate does it need to be?",
          "15 % error is the bar. If you cannot beat it, the project should stop."],
         ["What is the cost of being wrong, and is it symmetric?",
          "Under-valuing loses opportunity; over-valuing loses money. If they are "
          "not symmetric, squared error is the wrong loss."]],
    )

    sub("Question 2 · Frame it in the taxonomy of Chapter 1")

    table(
        ["Axis", "Answer", "Because"],
        [["Supervision", "<b>Supervised</b>",
          "Every district comes with its labelled median house value."],
         ["Task type", "<b>Multiple regression</b>, <b>univariate</b>",
          "Multiple features per district; a single value to predict per district."],
         ["Batch or online", "<b>Batch</b>",
          "No continuous stream, data is small enough for memory, no rapid drift."],
         ["Instance or model based", "<b>Model based</b>",
          "We want a compact artefact to serve, and to extrapolate to new districts."]],
    )

    sub("Question 3 · Select a performance measure")

    md("For regression the default is the **root mean square error**:")

    math(r"""
    \mathrm{RMSE}(\mathbf{X}, h) \;=\;
    \sqrt{\frac{1}{m}\sum_{i=1}^{m}
      \Bigl(h\bigl(\mathbf{x}^{(i)}\bigr) - y^{(i)}\Bigr)^{2}}
    """)

    where({
        r"m": "number of instances in the set you are measuring on",
        r"\mathbf{x}^{(i)}": "the feature vector of the $i$-th instance "
                             "(<i>excluding</i> its label)",
        r"y^{(i)}": "the true label of the $i$-th instance",
        r"h": "the hypothesis / prediction function; $h(\\mathbf{x}^{(i)}) = \\hat y^{(i)}$",
        r"\mathbf{X}": "the $m \\times n$ matrix stacking all feature vectors as rows",
    })

    md("An alternative when there are many outliers — the **mean absolute error**:")

    math(r"""
    \mathrm{MAE}(\mathbf{X}, h) \;=\;
    \frac{1}{m}\sum_{i=1}^{m}
      \Bigl| h\bigl(\mathbf{x}^{(i)}\bigr) - y^{(i)} \Bigr|
    """)

    sub("Both are norms — and that is the whole story")

    md(
        "RMSE is the Euclidean ($\\ell_2$) norm of the error vector; MAE is the "
        "Manhattan ($\\ell_1$) norm. Both are instances of the general $\\ell_k$ "
        "norm:"
    )

    math(r"""
    \bigl\lVert \mathbf{v} \bigr\rVert_k \;=\;
    \Bigl(\, |v_1|^{k} + |v_2|^{k} + \dots + |v_m|^{k} \,\Bigr)^{1/k}
    """)

    where({
        r"k=1": "$\\ell_1$, Manhattan norm — gives MAE",
        r"k=2": "$\\ell_2$, Euclidean norm — gives RMSE (the default, so "
                "$\\lVert\\cdot\\rVert$ with no subscript means $\\ell_2$)",
        r"k\to\infty": "$\\ell_\\infty$, the max norm — only the single largest "
                       "error matters",
    })

    idea(
        "The higher the norm index, the more it obsesses over large errors",
        "Because the $k$-th power amplifies big residuals relative to small ones. "
        "So RMSE is <b>more sensitive to outliers</b> than MAE. When outliers are "
        "exponentially rare — a bell-shaped error distribution — RMSE is excellent "
        "and is the right default. When your data has a heavy tail of genuinely "
        "weird districts, MAE (or Huber loss, §4) is the honest choice.",
    )

    derive(
        [("Why does minimising squared error give you the <b>mean</b> while "
          "minimising absolute error gives you the <b>median</b>? Take the simplest "
          "case: predict a single constant $c$ for all instances.", None),
         ("For squared loss, differentiate the objective and set to zero:",
          r"\frac{\partial}{\partial c}\sum_{i=1}^{m}(c - y^{(i)})^2 = "
          r"2\sum_{i=1}^{m}(c - y^{(i)}) = 0 \;\;\Longrightarrow\;\; "
          r"c^\star = \frac{1}{m}\sum_{i=1}^{m} y^{(i)} = \bar y"),
         ("For absolute loss the derivative of $|c - y|$ is $\\mathrm{sign}(c-y)$, so",
          r"\frac{\partial}{\partial c}\sum_{i=1}^{m}\bigl|c - y^{(i)}\bigr| = "
          r"\#\{i : y^{(i)} < c\} - \#\{i : y^{(i)} > c\} = 0"),
         ("which holds exactly when as many points lie above $c$ as below — i.e. "
          "$c^\\star$ is the <b>median</b>.", None),
         ("Consequence: a single house worth \\$50 M drags the squared-error "
          "optimum upward but leaves the absolute-error optimum almost untouched. "
          "That is precisely the robustness difference.", None)],
        title="Why RMSE targets the mean and MAE targets the median",
    )

    anim_header("Injecting one outlier: what happens to RMSE vs MAE")
    md(
        "The same 60 districts. Each frame drags one point further and further "
        "from the crowd. The $\\ell_2$-optimal constant (blue) chases the outlier; "
        "the $\\ell_1$-optimal constant (green) does not move."
    )

    rng = np.random.default_rng(2)
    base = rng.normal(200, 30, 60)
    offsets = np.linspace(0, 900, 45)
    means, medians, rmses, maes = [], [], [], []
    for off in offsets:
        v = base.copy()
        v[0] = base[0] + off
        means.append(v.mean()); medians.append(np.median(v))
        rmses.append(np.sqrt(np.mean((v - v.mean()) ** 2)))
        maes.append(np.mean(np.abs(v - np.median(v))))

    frames = []
    for i, off in enumerate(offsets):
        v = base.copy(); v[0] = base[0] + off
        frames.append(go.Frame(name=f"{off:.0f}", data=[
            go.Scatter(x=v, y=rng.normal(0, .06, 60), mode="markers",
                       marker=dict(color=["#EF476F"] + [C["train"]] * 59,
                                   size=[15] + [8] * 59,
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=[means[i], means[i]], y=[-.35, .35], mode="lines",
                       line=dict(color=C["primary"], width=4)),
            go.Scatter(x=[medians[i], medians[i]], y=[-.35, .35], mode="lines",
                       line=dict(color=C["success"], width=4, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"outlier offset = {off:>5.0f}   ℓ2 optimum (mean) = {means[i]:6.1f}   "
            f"ℓ1 optimum (median) = {medians[i]:6.1f}")])))

    f = go.Figure(data=[
        go.Scatter(x=base, y=rng.normal(0, .06, 60), mode="markers",
                   name="districts",
                   marker=dict(color=["#EF476F"] + [C["train"]] * 59,
                               size=[15] + [8] * 59,
                               line=dict(color="#fff", width=1))),
        go.Scatter(x=[means[0]] * 2, y=[-.35, .35], mode="lines",
                   name="minimiser of RMSE (mean)",
                   line=dict(color=C["primary"], width=4)),
        go.Scatter(x=[medians[0]] * 2, y=[-.35, .35], mode="lines",
                   name="minimiser of MAE (median)",
                   line=dict(color=C["success"], width=4, dash="dash")),
    ])
    f.update_layout(height=420, xaxis=dict(range=[100, 1200], title="value ($k)"),
                    yaxis=dict(visible=False, range=[-.6, .6]),
                    title="One outlier moves the mean, not the median",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="offset = ")
    figure(f)

    sub("Question 4 · Check the assumptions")

    md(
        "List every assumption you and the downstream team are making, then go "
        "and verify them with the people involved."
    )

    pitfall(
        "The classic wasted quarter",
        "You spend three months building an excellent regression model for house "
        "<i>prices</i> — and then discover the downstream team converts your number "
        "into a <b>category</b> (cheap / medium / expensive) and only ever uses "
        "the category. The correct framing was <b>classification</b> all along, "
        "prices only had to be right to within a bucket boundary, and your careful "
        "RMSE optimisation was mostly wasted effort. <b>Ask what happens to your "
        "output.</b>",
    )

    code_lab(
        "Compute the metrics and see the norm effect for yourself",
        '''import numpy as np

rng = np.random.default_rng(0)
m = 500
y      = rng.normal(200_000, 60_000, m)                 # true values
y_hat  = y + rng.normal(0, 25_000, m)                   # a decent model

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def lk(a, b, k):
    return float(np.sum(np.abs(a - b) ** k) ** (1 / k))

print(f"clean data   RMSE = {rmse(y, y_hat):>10,.0f}   MAE = {mae(y, y_hat):>10,.0f}"
      f"   ratio = {rmse(y, y_hat)/mae(y, y_hat):.3f}")

# add 1 % catastrophic outliers
y_out = y.copy()
bad = rng.choice(m, m // 100, replace=False)
y_out[bad] += 2_000_000
print(f"1 % outliers RMSE = {rmse(y_out, y_hat):>10,.0f}   MAE = {mae(y_out, y_hat):>10,.0f}"
      f"   ratio = {rmse(y_out, y_hat)/mae(y_out, y_hat):.3f}   <- RMSE exploded")

print("\\nGeneral l_k norm of the error vector as k grows:")
for k in [1, 2, 3, 5, 10, 50]:
    print(f"  k = {k:>2}   ||e||_k = {lk(y_out, y_hat, k):>14,.0f}")
print(f"  k -> inf  max|e| = {np.max(np.abs(y_out - y_hat)):>14,.0f}")
print("\\nHigher k => the single worst error dominates the whole measure.")

# ---- and the constant-predictor result from the derivation ----------------
print("\\nBest CONSTANT predictor under each loss (with outliers present):")
grid = np.linspace(y_out.min(), np.percentile(y_out, 99), 4000)
sq  = [(c, np.mean((y_out - c) ** 2)) for c in grid]
ab  = [(c, np.mean(np.abs(y_out - c))) for c in grid]
print(f"  argmin squared error = {min(sq, key=lambda t: t[1])[0]:>12,.0f}"
      f"   (sample mean   = {y_out.mean():>12,.0f})")
print(f"  argmin absolute err. = {min(ab, key=lambda t: t[1])[0]:>12,.0f}"
      f"   (sample median = {np.median(y_out):>12,.0f})")
''',
        key="ch02_metrics",
    )

    quiz(
        "Your model's errors are roughly bell-shaped but 2 % of districts are "
        "military bases whose values follow completely different rules. Which "
        "metric should you report to the business?",
        ["RMSE, because it is the standard",
         "MAE, or RMSE computed after excluding the known-different segment",
         "$\\ell_\\infty$, the maximum error",
         "Accuracy"],
        1,
        "A known, identifiable sub-population with different dynamics should "
        "either be modelled separately or excluded and reported separately. "
        "Failing that, MAE stops 2 % of rows from dominating the headline number.",
        key="ch02q1",
    )

    keypoints([
        "Frame the objective, find the current baseline (15 % error here), and "
        "ask what happens to your output downstream.",
        "RMSE = $\\ell_2$ norm of errors → optimal constant is the <b>mean</b>.",
        "MAE = $\\ell_1$ norm of errors → optimal constant is the <b>median</b>, so "
        "it is robust.",
        "Higher norm index ⇒ more weight on large errors. RMSE > MAE always, with "
        "equality only when all errors are equal.",
        "Write your assumptions down and verify them with the humans involved.",
    ])


# ==========================================================================
def s_2_2():
    section("2.2", "Get the Data & Create a Test Set")

    lead(
        "Load it, look at its shape, and — before you learn anything about it "
        "yourself — put a test set away. Your own brain is a source of overfitting."
    )

    sub("Take a quick look at the data structure")

    df = ds.housing()
    if bool(df["_synthetic"].iloc[0]):
        warn("Offline fallback active",
             "The California housing download was unavailable, so a "
             "statistically similar <b>synthetic</b> frame is being used. Every "
             "method below behaves identically; only the exact numbers differ.")

    show = df.drop(columns=["_synthetic"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (districts)", f"{len(show):,}")
    c2.metric("Columns", show.shape[1])
    c3.metric("Numeric columns", show.select_dtypes("number").shape[1])
    c4.metric("Missing cells", f"{int(show.isna().sum().sum()):,}")

    st.dataframe(show.head(12), width="stretch")

    with st.expander("`housing.info()` equivalent — dtypes and non-null counts"):
        info = pd.DataFrame({
            "dtype": show.dtypes.astype(str),
            "non-null": show.notna().sum(),
            "null": show.isna().sum(),
            "unique": show.nunique(),
        })
        st.dataframe(info, width="stretch")

    with st.expander("`housing.describe()` — the numeric summary"):
        st.dataframe(show.describe().T.style.format("{:,.2f}"),
                     width="stretch")

    sub("Read the histograms before you read anything else")

    md(
        "Histograms tell you four things at a glance: **scale differences** "
        "(which force feature scaling), **capping** (a spike at the right edge), "
        "**skew** (which hurts models expecting bell shapes), and **implausible "
        "values** (negatives where none should exist)."
    )

    numcols = [c for c in show.select_dtypes("number").columns]
    fig = make_subplots(rows=3, cols=3, subplot_titles=numcols[:9],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for i, col in enumerate(numcols[:9]):
        fig.add_trace(go.Histogram(x=show[col], nbinsx=50, showlegend=False,
                                   marker=dict(color=SEQ[i % len(SEQ)],
                                               line=dict(color="#fff", width=.4))),
                      row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(height=620, bargap=0.02,
                      title="Every numeric attribute, one histogram each")
    figure(fig, "Look for: a hard right edge (capping), long right tails (skew), "
                "and wildly different x-axis ranges (scaling needed).")

    idea(
        "Capping is a decision, not a discovery",
        "If the target was capped at \\$500 001 by whoever collected it, your model "
        "can never learn anything above that. You have exactly two honest options: "
        "(1) collect the true values for the capped districts, or (2) remove those "
        "districts from <b>both</b> the training and the test set — and then state "
        "clearly that the model must not be used above the cap.",
    )

    sub("Create a test set — and do it now")

    md(
        "Set aside 20 % and do not look at it. The naive version is a random "
        "split, but it has a subtle bug: run it twice and you get two different "
        "test sets, so over successive sessions you eventually see everything."
    )

    table(
        ["Strategy", "Stable across runs?", "Stable when data grows?", "Verdict"],
        [["<code>np.random.permutation</code>", "❌", "❌", "Broken"],
         ["Seed the RNG (<code>random_state=42</code>)", "✅", "❌ — a new row "
          "reshuffles everything", "Fine for a fixed dataset"],
         ["Hash of a stable identifier", "✅", "✅", "Correct in general"],
         ["<b>Stratified</b> split on an important attribute", "✅",
          "✅ (with a seed)", "<b>What you should use here</b>"]],
    )

    md("The hash-based rule keeps a row in the test set forever, no matter how "
       "the dataset grows or is reordered:")

    math(r"""
    \text{in\_test}(\mathrm{id}) \;=\;
    \Bigl[\; \mathrm{crc32}\bigl(\mathrm{id}\bigr) \;<\;
      r \cdot 2^{32} \;\Bigr]
    """)
    where({r"r": "the test ratio, e.g. $0.2$",
           r"\mathrm{id}": "a stable, immutable identifier for the row"})

    sub("Stratified sampling — why random is not good enough")

    md(
        "Median income is the strongest single predictor of house value. If your "
        "test set happens to under-represent high-income districts, your reported "
        "RMSE is measuring the wrong population. **Stratified sampling** forces "
        "the test set to mirror the population's income distribution."
    )

    proof(
        "Stratification strictly reduces the variance of your estimate",
        "For a stratified sample with strata of proportion $w_k$ and within-stratum "
        "variance $\\sigma_k^2$, the estimator variance is "
        "$\\mathrm{Var}_{\\text{strat}} = \\frac{1}{m}\\sum_k w_k \\sigma_k^2$, "
        "whereas simple random sampling gives "
        "$\\mathrm{Var}_{\\text{SRS}} = \\frac{1}{m}\\bigl(\\sum_k w_k\\sigma_k^2 + "
        "\\sum_k w_k(\\mu_k - \\mu)^2\\bigr)$. The second term — the "
        "<i>between-stratum</i> variance — is non-negative and is exactly what "
        "stratification removes. Stratifying can never hurt and helps whenever the "
        "strata means differ.",
    )

    anim_header("Random vs stratified sampling error, 40 repeated draws")
    md(
        "Each frame draws a fresh 20 % test set. Blue = purely random, green = "
        "stratified on the income band. Both are unbiased; the stratified one is "
        "visibly tighter. That tightness is what makes your reported RMSE "
        "reproducible."
    )

    inc = show["MedInc"].to_numpy()
    cat = np.clip(np.ceil(inc / 1.5), 1, 5).astype(int)
    pop_prop = np.array([(cat == k).mean() for k in range(1, 6)])

    rng = np.random.default_rng(0)
    n = len(inc)
    rand_err, strat_err = [], []
    for r in range(40):
        idx = rng.permutation(n)[: int(0.2 * n)]
        p = np.array([(cat[idx] == k).mean() for k in range(1, 6)])
        rand_err.append(100 * np.abs(p - pop_prop).sum() / 2)

        sel = []
        for k in range(1, 6):
            pool = np.where(cat == k)[0]
            sel.append(rng.permutation(pool)[: int(0.2 * len(pool))])
        sel = np.concatenate(sel)
        p2 = np.array([(cat[sel] == k).mean() for k in range(1, 6)])
        strat_err.append(100 * np.abs(p2 - pop_prop).sum() / 2)

    frames = []
    for k in range(1, 41):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=list(range(1, k + 1)), y=rand_err[:k], mode="lines+markers",
                       line=dict(color=C["train"], width=2.4)),
            go.Scatter(x=list(range(1, k + 1)), y=strat_err[:k], mode="lines+markers",
                       line=dict(color=C["success"], width=2.4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"draw {k}   ·   random mean = {np.mean(rand_err[:k]):.3f} %"
            f"   ·   stratified mean = {np.mean(strat_err[:k]):.3f} %")])))

    f = go.Figure(data=[
        go.Scatter(x=[1], y=rand_err[:1], mode="lines+markers",
                   name="random split", line=dict(color=C["train"], width=2.4)),
        go.Scatter(x=[1], y=strat_err[:1], mode="lines+markers",
                   name="stratified split", line=dict(color=C["success"], width=2.4)),
    ])
    f.update_layout(height=400, xaxis_title="repetition",
                    yaxis_title="total deviation from population income mix (%)",
                    title="Stratification kills the between-stratum variance",
                    xaxis=dict(range=[0, 41]))
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="draw ")
    figure(f)

    code_lab(
        "Three ways to split, compared honestly",
        '''import numpy as np, pandas as pd
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# --- rebuild the frame inside the lab so you can edit it -------------------
from core import datasets as _ds
housing = _ds.housing().drop(columns=["_synthetic"])
print(f"{len(housing):,} districts, {housing.shape[1]} columns\\n")

# ---------- 1. naive random (unstable across runs) --------------------------
def shuffle_split(data, ratio):
    idx = np.random.permutation(len(data))
    k = int(len(data) * ratio)
    return data.iloc[idx[k:]], data.iloc[idx[:k]]
a, _ = shuffle_split(housing, .2); b, _ = shuffle_split(housing, .2)
print(f"1. unseeded shuffle : two runs share only "
      f"{len(set(a.index) & set(b.index)) / len(a):.1%} of their training rows")

# ---------- 2. hash of a stable id (survives new rows) ---------------------
def is_test(identifier, ratio):
    return crc32(np.int64(identifier)) < ratio * 2**32
h = housing.reset_index(drop=True).reset_index()          # 'index' = stable id
mask = h["index"].apply(lambda i: is_test(i, .2))
print(f"2. crc32 hash split : test fraction = {mask.mean():.4f}  (stable forever)")

# ---------- 3. stratified on an income band --------------------------------
housing["income_cat"] = pd.cut(housing["MedInc"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
strat_tr, strat_te = train_test_split(housing, test_size=.2,
                                      stratify=housing["income_cat"],
                                      random_state=42)
rand_tr,  rand_te  = train_test_split(housing, test_size=.2, random_state=42)

comp = pd.DataFrame({
    "population":  housing["income_cat"].value_counts(normalize=True).sort_index(),
    "stratified":  strat_te["income_cat"].value_counts(normalize=True).sort_index(),
    "random":      rand_te["income_cat"].value_counts(normalize=True).sort_index(),
})
comp["strat %err"] = 100 * (comp["stratified"] / comp["population"] - 1)
comp["rand %err"]  = 100 * (comp["random"]     / comp["population"] - 1)
print("\\n3. income-category proportions in the TEST set")
print(comp.round(4).to_string())
print(f"\\n   mean |error| stratified = {comp['strat %err'].abs().mean():.3f} %")
print(f"   mean |error| random     = {comp['rand %err'].abs().mean():.3f} %")

# clean up the helper column, exactly as you would in a real project
for s in (strat_tr, strat_te):
    s.drop("income_cat", axis=1, inplace=True)
housing.drop("income_cat", axis=1, inplace=True)
print("\\nstrat_tr / strat_te are now in the namespace for the next labs.")
''',
        key="ch02_split",
    )

    keypoints([
        "Look at <code>head</code>, <code>info</code>, <code>describe</code> and "
        "the histograms — in that order — before anything else.",
        "Spot capping, skew and scale differences in the histograms; each implies "
        "a concrete later decision.",
        "Split the test set <b>before</b> exploring, to protect against "
        "<i>data snooping bias</i> — your own brain overfitting.",
        "Seeded random split for a fixed dataset; hash-of-id for a growing one.",
        "<b>Stratify</b> on the strongest predictor: it removes the "
        "between-stratum variance from your estimate for free.",
    ])


# ==========================================================================
def s_2_3():
    section("2.3", "Explore and Visualize the Data to Gain Insights")

    lead(
        "Now — and only now, on a copy of the <i>training</i> set — you look "
        "properly. You are hunting for three things: geography, correlations, and "
        "attribute combinations that are more informative than the raw columns."
    )

    df = ds.housing().drop(columns=["_synthetic"])

    sub("Visualising geographical data")

    md(
        "Latitude and longitude cry out for a scatter plot. Set the opacity low to "
        "reveal density, the marker size to population, and the colour to the "
        "target. Three visual channels, three variables."
    )

    scale = st.session_state.get("cscale", "Parula")
    samp = df.sample(min(4000, len(df)), random_state=0)
    g = go.Figure(go.Scattergl(
        x=samp["Longitude"], y=samp["Latitude"], mode="markers",
        marker=dict(size=np.clip(samp["Population"] / 90, 3, 22),
                    color=samp["median_house_value"],
                    colorscale=nav.cscale(), showscale=True,
                    colorbar=dict(title="median<br>value ($)"),
                    opacity=0.62, line=dict(width=0)),
        text=[f"value ${v:,.0f}<br>pop {p:,.0f}"
              for v, p in zip(samp["median_house_value"], samp["Population"])],
        hoverinfo="text"))
    g.update_layout(height=560, xaxis_title="longitude", yaxis_title="latitude",
                    title=f"California districts — size = population, "
                          f"colour = median value ({scale} colourscale)")
    figure(g, "Two things jump out: prices are high near the ocean, and they "
              "cluster densely — location and population density matter.")

    sub("Look for correlations")

    md("The **standard (Pearson) correlation coefficient** between two "
       "attributes:")

    math(r"""
    \rho_{X,Y} \;=\; \frac{\operatorname{cov}(X, Y)}{\sigma_X \, \sigma_Y}
    \;=\; \frac{\displaystyle\sum_{i=1}^{m}
                 \bigl(x^{(i)} - \bar x\bigr)\bigl(y^{(i)} - \bar y\bigr)}
               {\sqrt{\displaystyle\sum_{i=1}^{m}\bigl(x^{(i)} - \bar x\bigr)^2}\;
                \sqrt{\displaystyle\sum_{i=1}^{m}\bigl(y^{(i)} - \bar y\bigr)^2}}
    """)

    where({
        r"\rho \in [-1, 1]": "1 = perfect increasing linear relation, "
                             "−1 = perfect decreasing, 0 = no <i>linear</i> relation",
        r"\bar x, \bar y": "the sample means",
        r"\sigma_X, \sigma_Y": "the sample standard deviations",
    })

    corr = df.select_dtypes("number").corr(numeric_only=True)
    hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(title="ρ")))
    hm.update_layout(height=560, title="Pearson correlation matrix")
    figure(hm)

    tgt = corr["median_house_value"].drop("median_house_value").sort_values()
    bar = go.Figure(go.Bar(
        x=tgt.values, y=tgt.index, orientation="h",
        marker=dict(color=[C["danger"] if v < 0 else C["success"] for v in tgt.values]),
        text=[f"{v:+.3f}" for v in tgt.values], textposition="outside"))
    bar.update_layout(height=360, xaxis_title="ρ with median_house_value",
                      title="Correlation with the target, ranked",
                      xaxis=dict(range=[min(tgt.min() * 1.4, -0.2),
                                        max(tgt.max() * 1.4, 0.2)]))
    figure(bar)

    pitfall(
        "Pearson ρ only sees straight lines",
        "A perfect parabola $y = x^2$ over a symmetric range has $\\rho = 0$. A "
        "perfect step function can have $\\rho$ near 0.8 while being nothing like "
        "a line. <b>Always plot the scatter.</b> Anscombe's quartet — four datasets "
        "with identical means, variances and $\\rho$ but wildly different shapes — "
        "is the classic demonstration, and it is in the lab below.",
    )

    anim_header("Anscombe's quartet: identical statistics, four different worlds")

    ans = {
        "I": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
              [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
        "II": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
               [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
        "III": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
                [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
        "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
               [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
    }
    gx = np.linspace(2, 20, 50)
    frames, keys = [], list(ans)
    for k in keys:
        xa, ya = np.array(ans[k][0], float), np.array(ans[k][1], float)
        sl, ic = np.polyfit(xa, ya, 1)
        r = np.corrcoef(xa, ya)[0, 1]
        frames.append(go.Frame(name=k, data=[
            go.Scatter(x=xa, y=ya, mode="markers",
                       marker=dict(color=C["primary"], size=12,
                                   line=dict(color="#fff", width=1.5))),
            go.Scatter(x=gx, y=sl * gx + ic, mode="lines",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(title=f"Anscombe set {k}", annotations=[
            anim.annotate_step(f"mean x = {xa.mean():.2f}   mean y = {ya.mean():.2f}"
                               f"   ρ = {r:.3f}   fit: y = {sl:.3f}x + {ic:.2f}")])))

    xa0, ya0 = np.array(ans["I"][0], float), np.array(ans["I"][1], float)
    s0, i0 = np.polyfit(xa0, ya0, 1)
    f = go.Figure(data=[
        go.Scatter(x=xa0, y=ya0, mode="markers", name="data",
                   marker=dict(color=C["primary"], size=12,
                               line=dict(color="#fff", width=1.5))),
        go.Scatter(x=gx, y=s0 * gx + i0, mode="lines", name="least-squares fit",
                   line=dict(color=C["danger"], width=3)),
    ])
    f.update_layout(height=420, xaxis=dict(range=[2, 20]), yaxis=dict(range=[2, 14]),
                    title="Anscombe set I")
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="set ")
    figure(f, "All four have ρ ≈ 0.816 and the identical regression line.")

    sub("Experiment with attribute combinations")

    md(
        "`total_rooms` for a district is nearly meaningless — a district with more "
        "households obviously has more rooms. The informative quantity is the "
        "**ratio**:"
    )

    math(r"""
    \text{rooms\_per\_house} = \frac{\text{total\_rooms}}{\text{households}},
    \qquad
    \text{bedrooms\_ratio} = \frac{\text{total\_bedrooms}}{\text{total\_rooms}},
    \qquad
    \text{people\_per\_house} = \frac{\text{population}}{\text{households}}
    """)

    tip(
        "Ratios beat raw counts almost every time",
        "Any time two columns are both driven by a common size factor, their "
        "<b>ratio</b> removes the size and exposes the structure. This is the "
        "single highest-return feature-engineering move in tabular ML, and it "
        "costs one line of pandas.",
    )

    code_lab(
        "Correlations, Anscombe, and engineered ratios",
        '''import numpy as np, pandas as pd
from core import datasets as _ds

housing = _ds.housing().drop(columns=["_synthetic"])
num = housing.select_dtypes("number")

print("=== correlation with the target, raw attributes ===")
c = num.corr()["median_house_value"].drop("median_house_value")
print(c.sort_values(ascending=False).round(4).to_string())

# ---- engineered ratios ----------------------------------------------------
h = housing.copy()
h["rooms_per_house"]  = h["AveRooms"]
h["bedrooms_ratio"]   = h["AveBedrms"] / h["AveRooms"]
h["people_per_house"] = h["AveOccup"]
h["income_per_room"]  = h["MedInc"] / h["AveRooms"]
h["log_income"]       = np.log1p(h["MedInc"])

print("\\n=== after adding engineered attributes ===")
c2 = h.select_dtypes("number").corr()["median_house_value"] \\
      .drop("median_house_value").sort_values(ascending=False)
print(c2.round(4).to_string())

gained = [k for k in c2.index if k not in c.index]
print(f"\\nnew attributes: {gained}")
print("bedrooms_ratio is usually MORE informative than either of its parents.")

# ---- Pearson is blind to curvature ---------------------------------------
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 800)
print("\\n=== rho is blind to non-linear structure ===")
for name, yy in [("y = x        (linear)",    x),
                 ("y = x^2      (parabola)",  x**2),
                 ("y = |x|      (V shape)",   np.abs(x)),
                 ("y = sin(2x)  (wave)",      np.sin(2*x))]:
    print(f"  {name:<26} rho = {np.corrcoef(x, yy)[0,1]:+.4f}")
print("  -> three of these are perfectly deterministic, yet rho ~ 0. PLOT IT.")

# ---- Spearman sees monotone structure Pearson misses ---------------------
from scipy.stats import spearmanr
y_mono = np.exp(x)
print(f"\\ny = exp(x):  Pearson = {np.corrcoef(x, y_mono)[0,1]:.4f}   "
      f"Spearman = {spearmanr(x, y_mono).statistic:.4f}  <- rank correlation is 1.0")
''',
        key="ch02_explore",
    )

    keypoints([
        "Explore a <b>copy of the training set only</b> — never the test set.",
        "Geography: scatter with opacity for density, size and colour for two more "
        "variables.",
        "Pearson $\\rho$ measures <b>linear</b> association only; Spearman catches "
        "monotone non-linear ones; plotting catches everything.",
        "Engineered <b>ratios</b> routinely out-correlate their raw parents.",
        "Exploration is iterative: clean a little, model a little, come back.",
    ])


# ==========================================================================
def s_2_4():
    section("2.4", "Prepare the Data — Cleaning, Encoding, Scaling, Pipelines")

    lead(
        "Write transformations as <b>code</b>, never as manual spreadsheet edits. "
        "Code is reproducible, reusable on the test set and in production, and "
        "becomes a library of transforms you carry to the next project."
    )

    sub("Clean the data — missing values")

    md("Three options for a missing feature, and the imputation option has "
       "several strategies:")

    table(
        ["Option", "Pandas / scikit-learn", "When"],
        [["Drop the rows", "<code>df.dropna(subset=['x'])</code>",
          "Missing is rare and plausibly random"],
         ["Drop the attribute", "<code>df.drop('x', axis=1)</code>",
          "The column is mostly empty or uninformative"],
         ["Impute a constant",
          "<code>SimpleImputer(strategy='median')</code>",
          "<b>The default choice.</b> Median is robust to skew; mean is not"],
         ["Model-based imputation",
          "<code>KNNImputer</code>, <code>IterativeImputer</code>",
          "Missingness correlates with other columns"],
         ["Flag the missingness",
          "<code>SimpleImputer(add_indicator=True)</code>",
          "Almost always worth adding — <i>being missing</i> is often predictive"]],
    )

    warn(
        "The imputer must be fitted on training data only",
        "The median you fill with is a <b>parameter learned from data</b>. Compute "
        "it on the training set and apply the <i>same</i> value to validation, test "
        "and production. Computing a fresh median on the test set is textbook "
        "<b>data leakage</b> and inflates your score.",
    )

    sub("Handling text and categorical attributes")

    md("Three encodings, in increasing sophistication:")

    md(
        """
**1 · Ordinal encoding** — map each category to an integer.
`OrdinalEncoder` gives `["<1H OCEAN", "INLAND", "NEAR BAY"] → [0, 1, 2]`.
*Problem:* it invents an order and a distance. The model will believe that
`NEAR BAY` (2) is "twice as much" as `INLAND` (1), and that `<1H OCEAN` and
`INLAND` are closer to each other than to `NEAR BAY`. Only use it when a genuine
order exists (`small < medium < large`).

**2 · One-hot encoding** — one binary column per category.
        """
    )

    math(r"""
    \text{cat}_i \;\longmapsto\;
    \mathbf{e}_i = \bigl(\underbrace{0,\dots,0}_{i-1},\, 1,\,
                          \underbrace{0,\dots,0}_{K-i}\bigr) \in \{0,1\}^{K}
    """)

    md(
        "No spurious order, no spurious distances — every pair of distinct "
        "categories is at Euclidean distance $\\sqrt{2}$. *Problem:* with $K$ "
        "large (thousands of ZIP codes) you get a huge sparse matrix."
    )

    md(
        "**3 · Learned embeddings** — map each category to a dense vector of "
        "dimension $d \\ll K$, learned by the model itself. This is Chapter 13's "
        "topic; a common rule of thumb is $d \\approx \\lceil K^{1/4}\\rceil$ or "
        "$\\min(50, (K+1)/2)$."
    )

    anim_header("Three encodings of the same 6-category column")
    md("Watch the matrix change shape and meaning. The heat map is the encoded "
       "design matrix; the numbers below are the pairwise distances the model "
       "will perceive.")

    cats = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN", "DESERT"]
    K = len(cats)
    ordinal = np.arange(K).reshape(-1, 1).astype(float)
    onehot = np.eye(K)
    rngE = np.random.default_rng(3)
    emb = rngE.normal(0, 1, (K, 3))

    def dmat(M):
        return np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))

    views = [("Ordinal — 1 column, but invents an order", ordinal),
             ("One-hot — K columns, all pairs equidistant", onehot),
             ("Embedding — d=3 learned columns", emb)]
    frames = []
    for i, (title, M) in enumerate(views):
        D = dmat(M)
        frames.append(go.Frame(name=str(i + 1), data=[
            go.Heatmap(z=M, y=cats, colorscale=nav.cscale(),
                       x=[f"c{j}" for j in range(M.shape[1])],
                       showscale=False, xgap=2, ygap=2),
            go.Heatmap(z=D, x=cats, y=cats, colorscale="RdBu_r",
                       showscale=False, xgap=2, ygap=2,
                       text=np.round(D, 2), texttemplate="%{text}",
                       textfont=dict(size=9)),
        ], layout=go.Layout(title=title)))

    f = make_subplots(rows=1, cols=2, column_widths=[.38, .62],
                      subplot_titles=("encoded matrix",
                                      "pairwise distances the model sees"))
    f.add_trace(go.Heatmap(z=ordinal, y=cats, x=["c0"], colorscale=nav.cscale(),
                           showscale=False, xgap=2, ygap=2), 1, 1)
    D0 = dmat(ordinal)
    f.add_trace(go.Heatmap(z=D0, x=cats, y=cats, colorscale="RdBu_r",
                           showscale=False, xgap=2, ygap=2,
                           text=np.round(D0, 2), texttemplate="%{text}",
                           textfont=dict(size=9)), 1, 2)
    f.update_layout(height=440, title=views[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1800), slider_prefix="encoding ")
    figure(f, "Under ordinal encoding ISLAND is 4 units from DESERT and 1 unit "
              "from INLAND — pure fiction. One-hot makes every distance √2.")

    sub("Feature scaling and transformation")

    md(
        "With few exceptions, ML algorithms perform badly when input attributes "
        "have very different scales. Two standard fixes:"
    )

    math(r"""
    \textbf{Min–max (normalisation):}\qquad
    x' \;=\; \frac{x - x_{\min}}{x_{\max} - x_{\min}}
    \;\in\; [0, 1]
    """)
    math(r"""
    \textbf{Standardisation:}\qquad
    x' \;=\; \frac{x - \mu}{\sigma},
    \qquad \mu = \frac{1}{m}\sum_i x^{(i)},
    \qquad \sigma = \sqrt{\frac{1}{m}\sum_i \bigl(x^{(i)} - \mu\bigr)^2}
    """)

    table(
        ["", "Min–max (<code>MinMaxScaler</code>)", "Standardise (<code>StandardScaler</code>)"],
        [["Output range", "Bounded, exactly $[0,1]$", "Unbounded, mean 0, sd 1"],
         ["Outlier sensitivity", "<b>Very high</b> — one bad max squashes "
          "everything else into a sliver", "Much lower — no bounding to destroy"],
         ["Needed by", "Neural nets that expect $[0,1]$ inputs, image pixels",
          "Almost everything else: SVMs, PCA, k-means, linear models with "
          "regularisation"],
         ["Robust variant", "—", "<code>RobustScaler</code>: uses median and IQR"]],
    )

    md("For heavy right tails, transform *before* scaling. Two standard tools:")

    math(r"""
    \text{log transform:}\quad x' = \log(1 + x)
    \qquad\qquad
    \text{Box–Cox:}\quad
    x'^{(\lambda)} = \begin{cases}
      \dfrac{x^{\lambda} - 1}{\lambda}, & \lambda \neq 0\\[8pt]
      \log x, & \lambda = 0
    \end{cases}
    """)

    md(
        "A third trick for multimodal features: **RBF similarity** to a fixed "
        "landmark, which turns \"distance from 35 years old\" into a feature:"
    )

    math(r"""
    \phi_\gamma(x, \, \ell) \;=\; \exp\!\bigl(-\gamma\,(x - \ell)^2\bigr)
    """)
    where({r"\ell": "the landmark value (e.g. a peak in the histogram)",
           r"\gamma": "the width — larger $\\gamma$ ⇒ a narrower bump"})

    anim_header("Scaling in action: what the algorithm actually sees")

    rng = np.random.default_rng(9)
    raw = np.c_[rng.normal(3.5, 1.6, 300), rng.normal(35_000, 12_000, 300)]
    raw[:5, 1] += 260_000                                    # a few outliers
    mm = (raw - raw.min(0)) / (raw.max(0) - raw.min(0))
    ss = (raw - raw.mean(0)) / raw.std(0)
    q75, q25 = np.percentile(raw, [75, 25], axis=0)
    rb = (raw - np.median(raw, 0)) / (q75 - q25)

    stages = [("raw — two utterly different scales", raw),
              ("MinMaxScaler — outliers crush everything into a corner", mm),
              ("StandardScaler — comparable spreads", ss),
              ("RobustScaler — median & IQR, outliers pushed out", rb)]
    frames = [go.Frame(name=str(i + 1), data=[go.Scatter(
        x=M[:, 0], y=M[:, 1], mode="markers",
        marker=dict(color=[C["danger"]] * 5 + [C["primary"]] * 295,
                    size=[12] * 5 + [7] * 295, opacity=.75,
                    line=dict(color="#fff", width=.6)))],
        layout=go.Layout(title=t,
                         xaxis=dict(autorange=True), yaxis=dict(autorange=True)))
        for i, (t, M) in enumerate(stages)]

    f = go.Figure(go.Scatter(x=raw[:, 0], y=raw[:, 1], mode="markers",
                             marker=dict(color=[C["danger"]] * 5 + [C["primary"]] * 295,
                                         size=[12] * 5 + [7] * 295, opacity=.75,
                                         line=dict(color="#fff", width=.6))))
    f.update_layout(height=440, title=stages[0][0],
                    xaxis_title="median income", yaxis_title="population")
    anim.animate(f, frames, duration=nav.anim_ms(1600), slider_prefix="stage ")
    figure(f)

    sub("Custom transformers")

    md(
        "To slot your own logic into a scikit-learn pipeline you need three "
        "methods: `fit`, `transform`, and (free, by inheritance) `fit_transform`. "
        "Inherit from `BaseEstimator` (gives `get_params`/`set_params`, needed by "
        "grid search) and `TransformerMixin` (gives `fit_transform`)."
    )

    codenote(
        "The contract",
        "<code>fit(X, y=None)</code> must return <code>self</code> and store "
        "everything it learned in attributes ending with an underscore "
        "(<code>self.median_</code>). <code>transform(X)</code> must not learn "
        "anything. Do <b>not</b> use <code>*args</code> or <code>**kwargs</code> "
        "in <code>__init__</code> — grid search introspects the signature.",
    )

    sub("Transformation pipelines")

    md(
        "`Pipeline` chains transforms with a final estimator; `ColumnTransformer` "
        "routes different columns down different chains. Together they make the "
        "whole preprocessing story a single fittable object — which is exactly "
        "what makes cross-validation honest."
    )

    idea(
        "The pipeline is the leak-proof boundary",
        "When you call <code>cross_val_score(pipeline, X, y)</code>, the imputer's "
        "median, the scaler's mean and the encoder's category list are all refitted "
        "<b>inside each fold</b>, on that fold's training part only. Do the same "
        "steps by hand before splitting and you leak test information into "
        "training, silently inflating every number you report.",
    )

    code_lab(
        "The complete preprocessing pipeline, built up piece by piece",
        '''import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer, MinMaxScaler)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from core import datasets as _ds

housing = _ds.housing().drop(columns=["_synthetic"])
housing["income_cat"] = pd.cut(housing["MedInc"], [0., 1.5, 3., 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
train, test = train_test_split(housing, test_size=.2, random_state=42,
                               stratify=housing["income_cat"])
for s in (train, test): s.drop("income_cat", axis=1, inplace=True)
X_train = train.drop("median_house_value", axis=1)
y_train = train["median_house_value"].copy()
X_test  = test.drop("median_house_value", axis=1)
y_test  = test["median_house_value"].copy()
print(f"train {X_train.shape}   test {X_test.shape}")

# ---------- a custom transformer: ratio of two columns ---------------------
class RatioTransformer(BaseEstimator, TransformerMixin):
    """col0 / col1. `name` keeps the output feature names unique."""
    def __init__(self, name="ratio"):
        self.name = name                      # no *args/**kwargs -- see 2.4
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, [0]] / np.where(X[:, [1]] == 0, np.nan, X[:, [1]]))
    def get_feature_names_out(self, names=None):
        return np.array([self.name])

# ---------- a custom transformer that LEARNS something ---------------------
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """RBF similarity to k geographic cluster centres -- a learned feature."""
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters, self.gamma, self.random_state = n_clusters, gamma, random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self                                   # <- must return self
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return np.array([f"geo_sim_{i}" for i in range(self.n_clusters)])

# ---------- reusable sub-pipelines -----------------------------------------
log_pipe   = make_pipeline(SimpleImputer(strategy="median"),
                           FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                           StandardScaler())
def ratio_pipe(name):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RatioTransformer(name=name), StandardScaler())
geo_pipe   = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
cat_pipe   = make_pipeline(SimpleImputer(strategy="most_frequent"),
                           OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False))
default    = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocess = ColumnTransformer([
    ("bedrooms",  ratio_pipe("bedrooms_ratio"),   ["AveBedrms", "AveRooms"]),
    ("rooms_per_person", ratio_pipe("rooms_per_person"), ["AveRooms", "AveOccup"]),
    ("log",       log_pipe,   ["Population", "MedInc"]),
    ("geo",       geo_pipe,   ["Latitude", "Longitude"]),
    ("cat",       cat_pipe,   make_column_selector(dtype_include=object)),
], remainder=default, verbose_feature_names_out=False)

Xt = preprocess.fit_transform(X_train)
names = preprocess.get_feature_names_out()
print(f"\\nafter preprocessing: {Xt.shape}  ({len(names)} features)")
print("features:", list(names))

out = pd.DataFrame(Xt[:5], columns=names).round(3)
print("\\nfirst 5 transformed rows:")
print(out.to_string())

# ---------- proof that nothing leaked -------------------------------------
print(f"\\nimputer median for MedInc learned on TRAIN only: "
      f"{preprocess.named_transformers_['log'][0].statistics_[1]:.4f}")
print(f"actual TEST median (never used):                  "
      f"{X_test['MedInc'].median():.4f}   <- different, as it must be")
''',
        key="ch02_prep",
    )

    keypoints([
        "Every transformation goes in code, inside a <code>Pipeline</code>.",
        "Impute with the <b>median</b> (robust) and consider "
        "<code>add_indicator=True</code>.",
        "Never ordinal-encode an unordered category; one-hot it, or embed it if "
        "$K$ is huge.",
        "Standardise by default; min–max only when a bounded range is required; "
        "<code>RobustScaler</code> when outliers are present.",
        "Custom transformers: inherit <code>BaseEstimator, TransformerMixin</code>, "
        "learn in <code>fit</code>, return <code>self</code>, store with a trailing "
        "underscore.",
        "<code>ColumnTransformer</code> + <code>Pipeline</code> = leak-proof "
        "cross-validation.",
    ])


# ==========================================================================
def s_2_5():
    section("2.5", "Select and Train a Model")

    lead(
        "At last, modelling — and it is the shortest part of the project. Fit a "
        "few families quickly, use cross-validation rather than a single hold-out, "
        "and shortlist two or three promising candidates before tuning anything."
    )

    sub("Train and evaluate on the training set")

    md(
        "Start with a linear regression as the baseline. If it under-fits badly "
        "(training RMSE ≈ validation RMSE, both high), you need a more powerful "
        "model. If a decision tree gives you **RMSE = 0** on the training set, "
        "that is not a triumph — it has memorised the data."
    )

    sub("Better evaluation using cross-validation")

    md("Rather than carving another slice out of the training set, use $k$-fold:")

    math(r"""
    \widehat{\mathrm{RMSE}}_{\mathrm{CV}} \;=\;
    \frac{1}{k}\sum_{j=1}^{k}
      \sqrt{\frac{1}{|\mathcal{F}_j|}\sum_{i \in \mathcal{F}_j}
        \Bigl(h^{(-j)}\bigl(\mathbf{x}^{(i)}\bigr) - y^{(i)}\Bigr)^2}
    """)

    note(
        "scikit-learn's sign convention",
        "The <code>scoring</code> API always treats <i>higher is better</i>, so "
        "error metrics are exposed negated: <code>neg_root_mean_squared_error</code>. "
        "Take <code>-scores</code> to get the RMSE back. This trips up everybody "
        "exactly once.",
    )

    anim_header("Cross-validation, fold by fold")
    md("Five folds. Each frame trains on the blue rows and scores on the orange "
       "fold, then updates the running mean at the bottom.")

    m_rows, k = 40, 5
    fold_of = np.repeat(np.arange(k), m_rows // k)
    rngc = np.random.default_rng(6)
    fold_scores = 48_000 + rngc.normal(0, 4_500, k)

    frames = []
    for j in range(k):
        colors = np.where(fold_of == j, C["valid"], C["train"])
        running = fold_scores[:j + 1].mean()
        frames.append(go.Frame(name=f"fold {j+1}", data=[
            go.Heatmap(z=[np.where(fold_of == j, 1, 0)], showscale=False,
                       colorscale=[[0, C["train"]], [1, C["valid"]]],
                       xgap=1.5, ygap=1.5),
            go.Bar(x=[f"fold {i+1}" for i in range(k)],
                   y=[fold_scores[i] if i <= j else 0 for i in range(k)],
                   marker=dict(color=[C["valid"] if i == j else C["primary"]
                                      for i in range(k)]),
                   text=[f"{fold_scores[i]:,.0f}" if i <= j else ""
                         for i in range(k)], textposition="outside"),
        ], layout=go.Layout(title=f"Fold {j+1} of {k} held out   ·   "
                                  f"running mean RMSE = {running:,.0f}")))

    f = make_subplots(rows=2, cols=1, row_heights=[.28, .72],
                      subplot_titles=("the 40 training rows "
                                      "(orange = held out this fold)",
                                      "RMSE per fold"))
    f.add_trace(go.Heatmap(z=[np.where(fold_of == 0, 1, 0)], showscale=False,
                           colorscale=[[0, C["train"]], [1, C["valid"]]],
                           xgap=1.5, ygap=1.5), 1, 1)
    f.add_trace(go.Bar(x=[f"fold {i+1}" for i in range(k)],
                       y=[fold_scores[0]] + [0] * (k - 1),
                       marker=dict(color=[C["valid"]] + [C["primary"]] * (k - 1)),
                       showlegend=False), 2, 1)
    f.update_yaxes(visible=False, row=1, col=1)
    f.update_xaxes(visible=False, row=1, col=1)
    f.update_yaxes(range=[0, 62_000], title_text="RMSE ($)", row=2, col=1)
    f.update_layout(height=470, title=f"Fold 1 of {k} held out")
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="")
    figure(f)

    code_lab(
        "Shortlist three model families with cross-validation",
        '''import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

# reuse the preprocessing pipeline from the previous lab if it exists,
# otherwise build a light version here
try:
    preprocess, X_train, y_train
    print("reusing the pipeline from the previous lab\\n")
except NameError:
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from core import datasets as _ds
    housing = _ds.housing().drop(columns=["_synthetic"])
    train, test = train_test_split(housing, test_size=.2, random_state=42)
    X_train = train.drop("median_house_value", axis=1)
    y_train = train["median_house_value"]
    X_test  = test.drop("median_house_value", axis=1)
    y_test  = test["median_house_value"]
    preprocess = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_include=object))])
    print("built a fresh light pipeline\\n")

cv = KFold(5, shuffle=True, random_state=42)
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree":     DecisionTreeRegressor(random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "SVR (rbf)":        SVR(C=100_000, gamma=.05),
}

rows = []
for name, model in models.items():
    pipe = make_pipeline(preprocess, model)
    pipe.fit(X_train, y_train)
    train_rmse = mean_squared_error(y_train, pipe.predict(X_train)) ** .5
    cvs = -cross_val_score(pipe, X_train, y_train, cv=cv,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
    rows.append((name, train_rmse, cvs.mean(), cvs.std(),
                 cvs.mean() - train_rmse))

print(f"{'model':<18}{'train RMSE':>13}{'CV RMSE':>13}{'CV std':>11}{'gap':>13}")
for n, tr, cvm, cvs_, gap in rows:
    print(f"{n:<18}{tr:>13,.0f}{cvm:>13,.0f}{cvs_:>11,.0f}{gap:>13,.0f}")

print()
print("Reading the table:")
print("  train ~ CV, both high        -> UNDERFITTING (LinearRegression)")
print("  train ~ 0,  CV high          -> OVERFITTING  (DecisionTree memorised)")
print("  train low, CV low, small gap -> promising    (RandomForest)")
print()
print(f"baseline to beat: the manual experts' ~15 % error on a "
      f"${y_train.median():,.0f} median home = ~${.15*y_train.median():,.0f}")
''',
        key="ch02_train",
    )

    keypoints([
        "Baseline first (linear regression), then something powerful, then "
        "compare.",
        "Training RMSE ≈ 0 means memorisation, not success.",
        "Use $k$-fold CV, not a single hold-out — you get a mean <i>and</i> a "
        "standard deviation.",
        "scikit-learn negates error scores: use "
        "<code>neg_root_mean_squared_error</code> and flip the sign.",
        "Shortlist 2–3 families <b>before</b> spending time on hyperparameters.",
    ])


# ==========================================================================
def s_2_6():
    section("2.6", "Fine-Tune Your Model")

    lead(
        "You have a shortlist. Now search the hyperparameter space, understand "
        "what the winner learned, and finally — once — open the test set."
    )

    sub("Grid search")

    md(
        "`GridSearchCV` evaluates every combination in a dictionary of "
        "hyperparameter lists, with cross-validation. The cost is the product of "
        "the list lengths times $k$ folds — it explodes fast:"
    )

    math(r"""
    N_{\text{fits}} \;=\; k \cdot \prod_{j=1}^{p} \bigl| \Lambda_j \bigr|
    """)
    where({r"k": "number of CV folds",
           r"p": "number of hyperparameters searched",
           r"|\Lambda_j|": "number of values tried for hyperparameter $j$"})

    sub("Randomized search")

    md(
        "`RandomizedSearchCV` samples `n_iter` configurations from distributions "
        "instead. It has two decisive advantages:"
    )

    md(
        """
1. **Budget is decoupled from dimensionality.** You set `n_iter` directly, so
   adding a hyperparameter does not multiply your cost.
2. **It explores each dimension far better.** With $N$ trials, grid search tries
   only $N^{1/p}$ distinct values *per* hyperparameter; random search tries $N$
   distinct values per hyperparameter. When only 2 of your 6 hyperparameters
   actually matter — the usual case — that is an enormous difference.
        """
    )

    anim_header("Grid vs random search over the same budget")
    md(
        "Nine evaluations each. The important hyperparameter is on the x-axis; the "
        "one that does nothing is on the y-axis. Grid search samples only **3** "
        "distinct values of the thing that matters. Random search samples **9**."
    )

    gx_ = np.repeat([0.2, 0.5, 0.8], 3)
    gy_ = np.tile([0.2, 0.5, 0.8], 3)
    rngr = np.random.default_rng(12)
    rx_, ry_ = rngr.uniform(.05, .95, 9), rngr.uniform(.05, .95, 9)
    peak = 0.63
    fx = np.linspace(0, 1, 200)
    curve = np.exp(-((fx - peak) ** 2) / (2 * 0.09 ** 2))

    frames = []
    for kk in range(1, 10):
        gbest = max(np.exp(-((gx_[:kk] - peak) ** 2) / (2 * .09 ** 2)))
        rbest = max(np.exp(-((rx_[:kk] - peak) ** 2) / (2 * .09 ** 2)))
        frames.append(go.Frame(name=str(kk), data=[
            go.Scatter(x=fx, y=curve, mode="lines",
                       line=dict(color=C["muted"], width=2, dash="dot")),
            go.Scatter(x=gx_[:kk], y=gy_[:kk], mode="markers",
                       marker=dict(color=C["danger"], size=13, symbol="square",
                                   line=dict(color="#fff", width=1.5))),
            go.Scatter(x=rx_[:kk], y=ry_[:kk], mode="markers",
                       marker=dict(color=C["success"], size=13,
                                   line=dict(color="#fff", width=1.5))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{kk} evaluations   ·   grid best = {gbest:.3f}   ·   "
            f"random best = {rbest:.3f}")])))

    f = go.Figure(data=[
        go.Scatter(x=fx, y=curve, mode="lines", name="true effect of the "
                   "important hyperparameter",
                   line=dict(color=C["muted"], width=2, dash="dot")),
        go.Scatter(x=gx_[:1], y=gy_[:1], mode="markers", name="grid search",
                   marker=dict(color=C["danger"], size=13, symbol="square",
                               line=dict(color="#fff", width=1.5))),
        go.Scatter(x=rx_[:1], y=ry_[:1], mode="markers", name="random search",
                   marker=dict(color=C["success"], size=13,
                               line=dict(color="#fff", width=1.5))),
    ])
    f.update_layout(height=440, xaxis=dict(range=[0, 1], title="important hyperparameter"),
                    yaxis=dict(range=[0, 1], title="hyperparameter that does nothing"),
                    title="Same budget, very different coverage",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(420), slider_prefix="evals = ")
    figure(f)

    sub("Analyse the best models and their errors")

    md(
        "Inspect the winner. `feature_importances_` on a forest tells you which "
        "attributes carry signal — and which you can drop. Look at the specific "
        "instances the model gets most wrong: they usually reveal either a missing "
        "feature or a segment that deserves its own model."
    )

    sub("Evaluate on the test set")

    md(
        "Run the full pipeline on the test set, once. Then report a **confidence "
        "interval**, not a bare point estimate — a single number invites false "
        "precision. For the squared errors $e_i^2$:"
    )

    math(r"""
    \mathrm{CI}_{95\%}\bigl(\mathrm{RMSE}\bigr) \;=\;
    \left[\;
      \sqrt{\frac{(m-1)\,s^2}{\chi^2_{0.975,\;m-1}}}\;,\;\;
      \sqrt{\frac{(m-1)\,s^2}{\chi^2_{0.025,\;m-1}}}
    \;\right]
    """)
    where({r"s^2": "the sample variance of the squared errors $e_i^2$",
           r"m": "the number of test instances",
           r"\chi^2_{p, \nu}": "the $p$-quantile of the chi-squared distribution "
                               "with $\\nu$ degrees of freedom"})

    warn(
        "Resist the urge to tweak",
        "If the test performance is a little worse than your cross-validated "
        "estimate — which it usually is — <b>do not</b> start tuning to close the "
        "gap. The moment you tune against the test set, the number stops being an "
        "unbiased estimate. Accept it and report it. That gap <i>is</i> the "
        "selection bias you were guarding against.",
    )

    code_lab(
        "Randomized search, feature importances, and the one honest test number",
        '''import numpy as np, pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy import stats

try:
    preprocess, X_train, y_train, X_test, y_test
except NameError:
    raise RuntimeError("Run the pipeline lab in 2.4 first, then this one.")

pipe = make_pipeline(preprocess, RandomForestRegressor(random_state=42, n_jobs=-1))

param_dist = {
    "randomforestregressor__n_estimators":      randint(50, 260),
    "randomforestregressor__max_depth":         randint(6, 40),
    "randomforestregressor__min_samples_split": randint(2, 20),
    "randomforestregressor__min_samples_leaf":  randint(1, 12),
    "randomforestregressor__max_features":      uniform(.2, .8),
}

search = RandomizedSearchCV(pipe, param_dist, n_iter=25,
                            cv=KFold(3, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error",
                            random_state=42, n_jobs=-1, verbose=0)
search.fit(X_train, y_train)

print("best hyperparameters")
for k, v in search.best_params_.items():
    print(f"   {k.split('__')[-1]:<20} {v}")
print(f"\\nbest CV RMSE  = {-search.best_score_:,.0f}   <- optimistic (selected on)")

cvres = pd.DataFrame(search.cv_results_)
cvres["rmse"] = -cvres["mean_test_score"]
print("\\ntop 5 configurations")
print(cvres.nsmallest(5, "rmse")[
    ["rmse", "std_test_score", "param_randomforestregressor__max_depth",
     "param_randomforestregressor__max_features"]].round(3).to_string(index=False))

# ---------- what did it learn? --------------------------------------------
final = search.best_estimator_
imp = final[-1].feature_importances_
names = final[0].get_feature_names_out()
fi = pd.Series(imp, index=names).sort_values(ascending=False)
print("\\nfeature importances (top 12)")
print((100 * fi.head(12)).round(2).to_string())
print(f"\\nthe bottom {(fi.cumsum() > .99).sum()} features carry <1 % of the signal "
      f"and could be dropped")

import plotly.graph_objects as go
top = fi.head(15)[::-1]
fig = go.Figure(go.Bar(x=top.values * 100, y=top.index, orientation="h",
                       marker=dict(color=top.values, colorscale=PARULA),
                       text=[f"{v*100:.1f}%" for v in top.values],
                       textposition="outside"))
fig.update_layout(height=460, title="Random-forest feature importances",
                  xaxis_title="% of total impurity decrease")

# ---------- THE TEST SET. ONCE. -------------------------------------------
pred = final.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** .5
sq = (pred - y_test) ** 2
lo, hi = np.sqrt(stats.t.interval(.95, len(sq) - 1,
                                  loc=sq.mean(),
                                  scale=stats.sem(sq)))
print(f"\\n==================== FINAL TEST RESULT ====================")
print(f"test RMSE               = ${rmse:,.0f}")
print(f"95 % confidence interval= [${lo:,.0f}, ${hi:,.0f}]")
print(f"selection inflation     = ${rmse - (-search.best_score_):+,.0f}")
print(f"relative error          = {100*rmse/y_test.median():.1f} %  "
      f"(experts' baseline was ~15 %)")
print("Now stop tuning. This number is only honest because you stopped.")
''',
        key="ch02_tune",
    )

    keypoints([
        "Grid search cost is <b>multiplicative</b> in the number of "
        "hyperparameters; random search cost is whatever you set "
        "<code>n_iter</code> to.",
        "Random search explores $N$ distinct values <i>per</i> hyperparameter vs "
        "grid's $N^{1/p}$ — decisive when few hyperparameters matter.",
        "Inspect <code>feature_importances_</code>: drop the dead weight, and look "
        "at the worst errors for missing features.",
        "Report a <b>confidence interval</b> on the test metric.",
        "The test–CV gap is expected. Do not tune it away.",
    ])


# ==========================================================================
def s_2_7():
    section("2.7", "Launch, Monitor, and Maintain Your System")

    lead(
        "A model that is not deployed and monitored is a notebook, not a product. "
        "This section is short in the textbook and long in real life."
    )

    sub("Launch")

    md(
        """
Save the *entire pipeline* — preprocessing included — as one artefact, so that
production applies exactly the transformations training used.

```python
import joblib
joblib.dump(final_model, "california_housing_v1.pkl")
# later, in the serving process:
model = joblib.load("california_housing_v1.pkl")
predictions = model.predict(new_districts_df)
```

Three deployment shapes, in increasing decoupling:

1. **In-process** — load the pickle inside the web app. Simplest, but the app and
   the model scale together and share a Python environment.
2. **Dedicated service** — wrap it in a REST/gRPC service (Chapter 19's
   TensorFlow Serving, or a small FastAPI app). The app calls it over the wire;
   each scales independently and can be updated separately.
3. **Batch scoring** — score everything on a schedule and write predictions to a
   database or key-value store; the app just looks them up. Lowest latency at
   serve time, staleness is the trade-off.
        """
    )

    sub("Monitor — and monitor the right things")

    table(
        ["Layer", "What to watch", "Why"],
        [["Infrastructure", "latency p50/p95/p99, error rate, throughput, memory",
          "Catches crashes and slow degradation of the service itself"],
         ["<b>Input distribution</b>",
          "per-feature mean/std/quantiles, missing-value rate, unseen category rate",
          "<b>The earliest warning.</b> Drift shows here long before accuracy "
          "measurably drops"],
         ["Output distribution",
          "mean and spread of predictions, fraction hitting a boundary",
          "A model quietly collapsing to a constant is invisible in latency "
          "metrics"],
         ["Performance", "the live metric, when labels eventually arrive",
          "The ground truth — but often delayed by weeks or unavailable"],
         ["Business KPI", "revenue, click-through, conversion",
          "The only metric anyone outside the team cares about"]],
    )

    idea(
        "Ground truth is usually late, and sometimes never",
        "Housing values are only confirmed when a property sells; fraud labels "
        "arrive when a chargeback lands months later. So the pipeline needs a "
        "<b>label collection</b> mechanism — human raters, downstream outcomes, or "
        "sampling — designed at the same time as the model, not bolted on after.",
    )

    md("A simple, effective drift alarm is the **population stability index**:")

    math(r"""
    \mathrm{PSI} \;=\; \sum_{b=1}^{B}
      \bigl(p_b^{\text{live}} - p_b^{\text{train}}\bigr)\,
      \ln\!\frac{p_b^{\text{live}}}{p_b^{\text{train}}}
    """)
    where({r"B": "number of bins (10 deciles of the training distribution is standard)",
           r"p_b": "the fraction of observations falling in bin $b$",
           r"\mathrm{PSI} < 0.1": "no meaningful shift",
           r"0.1 \le \mathrm{PSI} < 0.25": "moderate shift — investigate",
           r"\mathrm{PSI} \ge 0.25": "major shift — retrain"})

    note("PSI is a symmetrised KL divergence",
         "It is exactly the Jeffreys divergence between the two binned "
         "distributions: $\\mathrm{PSI} = D_{\\mathrm{KL}}(P\\Vert Q) + "
         "D_{\\mathrm{KL}}(Q\\Vert P)$. That is why it is symmetric and always "
         "non-negative.")

    anim_header("Drift creeping in — PSI fires before RMSE does")
    md(
        "Month by month the incoming income distribution shifts. Watch the PSI "
        "cross 0.10 and then 0.25 while the accuracy metric is still comfortably "
        "flat — that lead time is the whole point of input monitoring."
    )

    rngd = np.random.default_rng(15)
    base = rngd.normal(3.9, 1.5, 20000)
    edges = np.percentile(base, np.linspace(0, 100, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    p_train = np.histogram(base, bins=edges)[0] / len(base)

    months, psis, rmses = [], [], []
    for mth in range(24):
        shift = 0.055 * mth
        live = rngd.normal(3.9 + shift, 1.5 + .02 * mth, 3000)
        p_live = np.histogram(live, bins=edges)[0] / len(live)
        p_live = np.clip(p_live, 1e-6, None)
        psi = float(np.sum((p_live - p_train) * np.log(p_live / p_train)))
        months.append(mth); psis.append(psi)
        rmses.append(48_000 * (1 + .30 * max(0, shift - .35) ** 1.7)
                     + rngd.normal(0, 700))

    frames = []
    for k in range(1, 25):
        state = ("OK" if psis[k - 1] < .1 else
                 "INVESTIGATE" if psis[k - 1] < .25 else "RETRAIN NOW")
        col = (C["success"] if psis[k - 1] < .1 else
               C["warning"] if psis[k - 1] < .25 else C["danger"])
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=months[:k], y=psis[:k], mode="lines+markers",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=months[:k], y=rmses[:k], mode="lines+markers",
                       line=dict(color=C["danger"], width=3), yaxis="y2"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"month {k}   ·   PSI = {psis[k-1]:.3f}   ·   "
            f"RMSE = ${rmses[k-1]:,.0f}   ·   {state}", color=col)])))

    f = go.Figure(data=[
        go.Scatter(x=[0], y=psis[:1], mode="lines+markers", name="PSI (inputs)",
                   line=dict(color=C["primary"], width=3)),
        go.Scatter(x=[0], y=rmses[:1], mode="lines+markers",
                   name="live RMSE (labels, delayed)",
                   line=dict(color=C["danger"], width=3), yaxis="y2"),
    ])
    f.add_hline(y=.10, line_dash="dot", line_color=C["warning"],
                annotation_text="PSI 0.10 — investigate")
    f.add_hline(y=.25, line_dash="dash", line_color=C["danger"],
                annotation_text="PSI 0.25 — retrain")
    f.update_layout(
        height=440, xaxis_title="months since launch",
        yaxis=dict(title="PSI", range=[0, .6]),
        yaxis2=dict(title="RMSE ($)", overlaying="y", side="right",
                    range=[44_000, 62_000], showgrid=False),
        title="Input drift is a leading indicator; accuracy is a lagging one",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(200), slider_prefix="month ")
    figure(f)

    sub("Maintain")

    md(
        """
The maintenance loop, automated as far as you can afford:

1. Collect fresh data and label it.
2. Write a script that trains and tunes the model automatically.
3. Write a script that evaluates the **new** model against the **current**
   production model on a fresh test set, and promotes it only if it wins.
4. Keep every model version *and the dataset it was trained on* — you need both
   to roll back, and you need the dataset to explain a regression later.
        """
    )

    tip("Version the data, not just the code",
        "Rolling back the model without rolling back the preprocessing statistics "
        "and the training snapshot leaves you unable to reproduce anything. "
        "Snapshot the training set alongside the artefact.")

    code_lab(
        "A drift monitor you can actually ship",
        '''import numpy as np, pandas as pd

def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index between a reference and a live sample."""
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, bins=edges)[0] / len(expected), eps, None)
    a = np.clip(np.histogram(actual,   bins=edges)[0] / len(actual),   eps, None)
    return float(np.sum((a - e) * np.log(a / e)))

def verdict(v):
    return "OK" if v < .10 else ("INVESTIGATE" if v < .25 else "RETRAIN")

rng = np.random.default_rng(0)
ref = pd.DataFrame({
    "MedInc":    rng.normal(3.9, 1.5, 20_000),
    "HouseAge":  rng.uniform(1, 52, 20_000),
    "AveRooms":  rng.normal(5.4, 1.1, 20_000),
    "Latitude":  rng.uniform(32.5, 41.9, 20_000),
})

scenarios = {
    "no drift":            dict(inc=0.0,  age=0.0, rooms=0.0),
    "incomes rising":      dict(inc=1.1,  age=0.0, rooms=0.0),
    "new region onboarded":dict(inc=0.25, age=9.0, rooms=1.4),
    "upstream bug":        dict(inc=0.0,  age=0.0, rooms=-3.2),
}

print(f"{'scenario':<24}{'MedInc':>10}{'HouseAge':>11}{'AveRooms':>11}"
      f"{'Latitude':>11}   verdict")
for name, sh in scenarios.items():
    live = pd.DataFrame({
        "MedInc":   rng.normal(3.9 + sh["inc"], 1.5, 4_000),
        "HouseAge": rng.uniform(1 + sh["age"], 52 + sh["age"], 4_000),
        "AveRooms": rng.normal(5.4 + sh["rooms"], 1.1, 4_000),
        "Latitude": rng.uniform(32.5, 41.9, 4_000),
    })
    vals = {c: psi(ref[c], live[c]) for c in ref.columns}
    worst = max(vals.values())
    print(f"{name:<24}" + "".join(f"{vals[c]:>10.3f} " for c in ref.columns)
          + f"  {verdict(worst)}")

print()
print("Note the last row: a bug that shifts ONE column is caught immediately,")
print("even though the target metric would not move for weeks.")

# --- and a tiny promotion gate --------------------------------------------
def should_promote(new_rmse, prod_rmse, min_gain=0.01):
    """Promote only on a materially better score."""
    gain = (prod_rmse - new_rmse) / prod_rmse
    return gain > min_gain, gain

for new, prod in [(47_100, 48_000), (47_950, 48_000), (49_500, 48_000)]:
    ok, g = should_promote(new, prod)
    print(f"new={new:,}  prod={prod:,}  gain={g:+.2%}  -> "
          f"{'PROMOTE' if ok else 'keep current model'}")
''',
        key="ch02_monitor",
    )

    keypoints([
        "Save the <b>whole pipeline</b> as one artefact; production must apply the "
        "exact training transformations.",
        "Monitor five layers: infrastructure, <b>input distribution</b>, output "
        "distribution, performance, business KPI.",
        "Input drift (PSI) is a <b>leading</b> indicator; accuracy is a "
        "<b>lagging</b> one that needs labels you may not have yet.",
        "PSI thresholds: &lt;0.10 fine, 0.10–0.25 investigate, ≥0.25 retrain.",
        "Automate: retrain script, evaluate script, promotion gate, rollback. "
        "Version data and models together.",
    ])


# ==========================================================================
def s_2_8():
    section("2.8", "Exercises & Chapter Review")

    lead("These extend the project rather than merely testing recall. Each has a "
         "worked answer and, where useful, runnable code.")

    exercise(
        1,
        "Try a Support Vector Machine regressor (`SVR`) with various "
        "hyperparameters, e.g. `kernel='linear'` (with various `C`) or "
        "`kernel='rbf'` (with various `C` and `gamma`). How does the best "
        "predictor perform?",
        "The RBF kernel with a large `C` typically beats the linear kernel here, "
        "because the relationship between income/location and value is strongly "
        "non-linear. But SVR is usually **worse than the random forest** on this "
        "dataset and *far* slower to fit — $O(m^2)$ to $O(m^3)$ in the number of "
        "instances (Chapter 5). The lesson: try it, measure it, and drop it "
        "without sentiment.\n\n"
        "Crucially, `SVR` is scale-sensitive, so it must sit behind the scaler in "
        "a pipeline — an unscaled SVR will look catastrophically bad and you will "
        "wrongly blame the model.",
        code='''from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

grid = GridSearchCV(
    make_pipeline(preprocess, SVR()),
    [{"svr__kernel": ["linear"], "svr__C": [10., 100., 1000., 10_000.]},
     {"svr__kernel": ["rbf"],    "svr__C": [1_000., 10_000., 100_000.],
      "svr__gamma": [0.01, 0.05, 0.1, 0.3]}],
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, -grid.best_score_)''')

    exercise(
        2, "Try replacing `GridSearchCV` with `RandomizedSearchCV`.",
        "With the same wall-clock budget, randomised search usually finds a better "
        "configuration, for the reason animated in §2.6: it explores $N$ distinct "
        "values per hyperparameter rather than $N^{1/p}$. Use `loguniform` for "
        "scale parameters such as `C` and `gamma` (they matter multiplicatively, "
        "not additively) and `randint` for counts.",
        code='''from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV

rnd = RandomizedSearchCV(
    make_pipeline(preprocess, SVR(kernel="rbf")),
    {"svr__C": loguniform(20, 200_000), "svr__gamma": loguniform(0.001, 0.5)},
    n_iter=40, cv=3, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1)
rnd.fit(X_train, y_train)''')

    exercise(
        3, "Try adding a `SelectFromModel` transformer in the preparation "
        "pipeline to select only the most important attributes.",
        "Insert it **after** the `ColumnTransformer` and **before** the final "
        "estimator, so it selects from the fully engineered feature set. Make the "
        "number of features a hyperparameter (`max_features` or `threshold`) and "
        "let the search tune it. Typically you can drop half the columns with no "
        "measurable loss — which halves inference cost and the amount of data "
        "production must supply.",
        code='''from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

selector = SelectFromModel(
    RandomForestRegressor(n_estimators=100, random_state=42),
    threshold="median")

pipe = make_pipeline(preprocess, selector,
                     RandomForestRegressor(random_state=42, n_jobs=-1))''')

    exercise(
        4, "Try creating a custom transformer that trains a $k$-nearest-neighbours "
        "regressor on latitude and longitude, and outputs its prediction as a new "
        "feature.",
        "This is **target encoding of geography** and it is powerful — but it is "
        "also the single easiest way to leak the target into your features. The "
        "inner $k$-NN must be fitted only on the training rows of each CV fold, "
        "which is exactly what happens automatically if the transformer lives "
        "inside the pipeline and implements `fit(X, y)`. If you compute it once "
        "over the whole dataset beforehand, every fold's validation rows will have "
        "contributed to their own feature and your CV score becomes fiction.",
        code='''class KNNTargetFeature(BaseEstimator, TransformerMixin):
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y=None):
        from sklearn.neighbors import KNeighborsRegressor
        self.knn_ = KNeighborsRegressor(self.k).fit(X, y)   # y is required!
        return self
    def transform(self, X):
        return self.knn_.predict(X).reshape(-1, 1)
    def get_feature_names_out(self, names=None):
        return np.array(["knn_geo_value"])''')

    exercise(
        5, "Automatically explore some preparation options using "
        "`GridSearchCV`.",
        "Because the preprocessing lives inside the pipeline, its hyperparameters "
        "are searchable with the same double-underscore syntax. You can tune the "
        "imputer strategy, the number of geographic clusters, the RBF $\\gamma$, "
        "and the scaler choice — all as part of one search. This is the practical "
        "pay-off of putting everything in a pipeline.",
        code='''param_grid = {
    "columntransformer__geo__n_clusters": [5, 10, 15, 20],
    "columntransformer__geo__gamma":      [0.1, 1.0, 10.0],
    "columntransformer__log__simpleimputer__strategy": ["median", "mean"],
    "randomforestregressor__max_features": [4, 6, 8, 10],
}''')

    exercise(
        6, "Try to implement the `StandardScaler` class from scratch.",
        "It is 15 lines, and writing it fixes the fit/transform contract in your "
        "memory permanently. The key points: learn `mean_` and `scale_` in `fit`, "
        "store them with trailing underscores, return `self`, and guard against a "
        "zero standard deviation (a constant column) — scikit-learn's own "
        "implementation replaces $\\sigma = 0$ with 1 so the column becomes all "
        "zeros rather than all NaN.",
        code='''class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0        # constant column guard
        self.n_features_in_ = X.shape[1]
        return self                                 # ALWAYS return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == self.n_features_in_
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float) * self.scale_
        return X + self.mean_ if self.with_mean else X''')

    rule()

    sub("The eight-step workflow, memorised")

    table(
        ["#", "Step", "The one thing people get wrong"],
        [["1", "Frame the problem, look at the big picture",
          "Not asking what happens to the output downstream"],
         ["2", "Get the data",
          "Exploring before splitting — data snooping bias"],
         ["3", "Create a test set (stratified)",
          "A plain random split that misrepresents an important attribute"],
         ["4", "Explore & visualise (training copy only)",
          "Trusting $\\rho$ without plotting the scatter"],
         ["5", "Prepare the data (in a pipeline)",
          "Fitting the imputer/scaler on the full dataset — leakage"],
         ["6", "Select & train several models (CV)",
          "Tuning before shortlisting"],
         ["7", "Fine-tune (randomised search)",
          "Grid search with six hyperparameters and no budget"],
         ["8", "Present, launch, monitor, maintain",
          "Monitoring only latency, and only accuracy"]],
    )

    keypoints([
        "The metric follows from the cost of an error: RMSE for bell-shaped "
        "errors, MAE for heavy tails.",
        "Split before you look; stratify on the strongest predictor.",
        "All preprocessing lives in a <code>Pipeline</code> — that is what makes "
        "cross-validation and production agree.",
        "Cross-validate to shortlist, randomised-search to tune, test set once "
        "with a confidence interval.",
        "Monitor input distributions, because labels arrive late or never.",
    ], title="Chapter 2 in five lines")

    refs([
        ("Appendix A of this platform — the printable project checklist",
         "See the sidebar: A · ML project checklist"),
        ("Pace & Barry — *Sparse Spatial Autoregressions* (the California housing "
         "dataset)", "Statistics & Probability Letters, 1997"),
        ("Bergstra & Bengio — *Random Search for Hyper-Parameter Optimization*",
         "https://www.jmlr.org/papers/v13/bergstra12a.html"),
        ("Anscombe, F. — *Graphs in Statistical Analysis*",
         "https://doi.org/10.1080/00031305.1973.10478966"),
    ])


# ==========================================================================
SECTIONS = [
    ("2.1", "Look at the Big Picture", s_2_1),
    ("2.2", "Get the Data & Test Set", s_2_2),
    ("2.3", "Explore and Visualize", s_2_3),
    ("2.4", "Prepare the Data", s_2_4),
    ("2.5", "Select and Train a Model", s_2_5),
    ("2.6", "Fine-Tune Your Model", s_2_6),
    ("2.7", "Launch, Monitor, Maintain", s_2_7),
    ("2.8", "Exercises & Review", s_2_8),
]

nav.render_chapter(CH, SECTIONS)
