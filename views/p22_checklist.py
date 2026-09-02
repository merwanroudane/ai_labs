"""Appendix A — the machine learning project checklist."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, nav
from core.lecture import (anim_header, codenote, figure, hero, idea, keypoints,
                          lead, math, md, note, pitfall, refs, rule, section,
                          sub, table, tip, warn)
from core.palette import C, SEQ, alpha
from core.theme import PLOTLY_CONFIG, inject

inject()
CH = "checklist"

hero(
    kicker="Appendix A",
    title="ML project checklist",
    blurb=(
        "Eight steps, from framing the problem to keeping the system alive. "
        "Tick the boxes as you go — the page tracks your progress and tells you "
        "which chapter covers whatever you are stuck on."
    ),
    chips=["8 phases", "70+ checks", "progress tracked",
           "cross-referenced"],
)


# --------------------------------------------------------------------------

PHASES: list[tuple[str, str, str, list[tuple[str, str, str]]]] = [
    ("1", "Frame the problem", "🎯",
     [("Define the objective in **business terms**", "§2.1",
       "Not 'predict churn' — 'reduce churn by 2 points at a cost below the "
       "revenue saved'. Every later decision follows from this sentence."),
      ("Describe how the solution will actually be **used**", "§2.1",
       "A batch report, a live API, a dashboard, a decision support tool? "
       "This determines latency budget, retraining cadence, and interface."),
      ("Identify the **current solution** and its performance", "§2.1",
       "There is always one — a rule, a spreadsheet, a person. It is your "
       "baseline, and beating it is the actual bar."),
      ("Decide: supervised / unsupervised / RL, online / offline", "§1.2",
       "Framing the learning problem wrongly is unrecoverable later."),
      ("Choose a **performance measure**, and check it aligns with the "
       "objective", "§2.3",
       "RMSE punishes outliers heavily; MAE does not. Accuracy is useless on "
       "imbalanced data. Pick deliberately."),
      ("State the **minimum performance** needed to be worth deploying", "§2.3",
       "Written down before you start, or you will rationalise whatever you "
       "get."),
      ("List **comparable problems** and reusable experience", "§2.1", ""),
      ("Check whether **human expertise** is available", "§2.1",
       "A domain expert catches leakage and nonsense features faster than any "
       "validation scheme."),
      ("List the **assumptions** you have made, and verify them", "§2.1",
       "Especially: is the downstream consumer really using your number the "
       "way you think?"),
      ("Confirm the problem is **worth solving with ML at all**", "§1.1",
       "If a three-line rule gets 90 % of the value, write the rule.")]),

    ("2", "Get the data", "📦",
     [("List the data you need and **how much**", "§2.2", ""),
      ("Find and **document** where each source comes from", "§2.2",
       "Six months from now you will need to rebuild this exactly."),
      ("Check the **legal obligations** and get authorisation", "§2.2",
       "Licensing, GDPR, consent, data-residency. Before, not after."),
      ("Get **access authorisations** for everyone who needs them", "§2.2", ""),
      ("Create a **workspace with enough storage**", "§2.2", ""),
      ("Convert to a format you can manipulate **without changing the data**",
       "§2.2", ""),
      ("Ensure **sensitive information is deleted or protected**", "§2.2",
       "Anonymised, pseudonymised, or aggregated — and check that a join "
       "cannot re-identify it."),
      ("Check the **size and type** of the data (time series? geo? nested?)",
       "§2.2", "The structure determines the split strategy."),
      ("**Sample a test set, put it aside, and never look at it**", "§2.7",
       "The single most important line in this checklist. Stratify it if the "
       "classes are imbalanced; split it by TIME if the data is temporal."),
      ("Write the data-collection step as a **reproducible script**", "§2.2",
       "Not a notebook cell you ran once.")]),

    ("3", "Explore the data", "🔍",
     [("Work on a **copy** (sample it down if it is large)", "§2.4", ""),
      ("Keep a **journal** of what you find", "§2.4", ""),
      ("Study each attribute: **name, type, % missing, noisiness, "
       "distribution**", "§2.4", ""),
      ("For supervised tasks, identify the **target attribute**", "§2.4", ""),
      ("**Visualise** the data", "§2.4",
       "Histograms, scatter matrices, geographic plots. Anscombe's quartet is "
       "the reason."),
      ("Study the **correlations** between attributes", "§2.4",
       "Pearson catches only linear relationships — plot as well."),
      ("Study how you would **solve the problem manually**", "§2.4",
       "It tells you which features matter."),
      ("Identify **promising transformations**", "§2.4",
       "Logs for skewed money, ratios, cyclical encodings for time."),
      ("Identify **extra data that would be useful**", "§2.4",
       "Return to step 2 if so — more of the right data beats a better model."),
      ("Check for **leakage**: any feature that would not exist at prediction "
       "time", "§2.5",
       "A suspiciously good result is leakage until proven otherwise."),
      ("**Document what you learned**", "§2.4", "")]),

    ("4", "Prepare the data", "🔧",
     [("Work on **copies**, and write every step as a **function**", "§2.5",
       "So it can be reproduced, reused on new data, and put in the "
       "SavedModel (§19.1)."),
      ("Clean: **fix or remove outliers**, decide on missing values", "§2.5",
       "Drop rows, drop columns, or impute — and record which."),
      ("**Feature selection**: drop what carries no information", "§2.5", ""),
      ("**Feature engineering**: discretise, decompose, transform, aggregate",
       "§2.5",
       "Often worth more than the model choice."),
      ("**Feature scaling**: standardise or normalise", "§2.5",
       "Fit on the training set ONLY. It is a conditioning argument (§M.1), "
       "not a convention."),
      ("Put the whole pipeline in a **`Pipeline` / preprocessing layer**",
       "§2.5, §13.7",
       "Not a sequence of notebook cells."),
      ("Verify the pipeline is **fitted on train and applied to valid/test**",
       "§2.5",
       "Fitting on everything is the most common leak.")]),

    ("5", "Shortlist promising models", "🧪",
     [("Train **many quick-and-dirty models** from different families",
       "§2.6",
       "Linear, tree, SVM, naive Bayes, a small net. Standard hyperparameters."),
      ("Measure and compare with **N-fold cross-validation**", "§2.6",
       "Report mean AND standard deviation."),
      ("Analyse the **most significant variables** for each algorithm", "§2.6",
       ""),
      ("Analyse the **errors** each model makes", "§2.6",
       "Which examples? Which subgroups? Which would a human get right?"),
      ("Do a quick round of **feature selection and engineering**", "§2.6", ""),
      ("Shortlist the **top three to five** models, preferring ones that make "
       "**different kinds of error**", "§7.2",
       "Diverse errors is exactly what makes an ensemble work (§M.3)."),
      ("Check the **cost** of each: fit time, inference time, model size",
       "§19.2",
       "Accuracy is not the only axis.")]),

    ("6", "Fine-tune the system", "🎛️",
     [("Use **as much data as possible** for this step", "§2.7", ""),
      ("Tune hyperparameters with **cross-validation**", "§2.7",
       "Random search over grid; Hyperband if the budget is tight (§19.7)."),
      ("Treat data-preparation choices as **hyperparameters** too", "§2.7",
       "Imputation strategy, discretisation, which features to drop."),
      ("Prefer **random search** to grid search", "§19.7",
       "59 random trials reach the top 5 % with 95 % probability, independent "
       "of dimension."),
      ("Try **ensembles** of your best models", "§7.7",
       "Combining usually beats the best single model."),
      ("**Once you are confident, measure on the test set — ONCE**", "§2.7",
       "Do not tune after looking. Testing 20 things at p<0.05 finds something "
       "64 % of the time (§M.4)."),
      ("Estimate the **generalisation error** with a confidence interval",
       "§M.4",
       "A single number without an interval is not a result.")]),

    ("7", "Present your solution", "📊",
     [("**Document** what you did", "§2.8", ""),
      ("Create a presentation, **starting with the big picture**", "§2.8", ""),
      ("Explain **why** the solution achieves the business objective", "§2.8",
       "In the language of step 1, not of step 6."),
      ("Present the **interesting points** you noticed along the way", "§2.8",
       "What worked, what did not, and the assumptions and limitations."),
      ("State the **limitations** plainly", "§2.8",
       "Where it fails, which subgroups are worse served, what would break it."),
      ("Ensure the key findings are communicated as **memorable statements**",
       "§2.8", "'Median income is the single best predictor of house price.'"),
      ("Include **sliced metrics**, not only the aggregate", "§19.8",
       "A 95 %-accurate model can be 60 % accurate on a subgroup.")]),

    ("8", "Launch, monitor and maintain", "🚀",
     [("Get the solution **ready for production**", "§19.1",
       "Preprocessing inside the SavedModel; a signature the client can call."),
      ("Write **monitoring code** for live performance, and alerts", "§19.8",
       "Infrastructure, inputs, predictions, and outcomes."),
      ("Beware of **slow degradation**: models rot as data drifts", "§19.8",
       "PSI on the inputs catches covariate shift weeks before the labels "
       "arrive."),
      ("Measure performance with **human evaluation** where possible", "§19.8",
       "Sample predictions and have an expert rate them."),
      ("Monitor **input data quality**", "§19.8",
       "A sensor dying or a upstream schema change looks like model failure."),
      ("**Retrain regularly on fresh data**, and automate it", "§19.8",
       "Under concept drift, old data is actively wrong."),
      ("Deploy **shadow → canary → ramp → full**, with an abort rule",
       "§19.1, §19.9",
       "Defined in advance, not decided during the incident."),
      ("**Test the rollback**", "§19.9",
       "An untested rollback is not a rollback."),
      ("Keep **backups of every model version** and the data to rebuild them",
       "§19.9", "So you can roll back quickly, and so you can explain a "
                "decision made six months ago.")]),
]


def _store():
    return st.session_state.setdefault("_checklist", set())


def render_phase(idx: int):
    num, title, icon, items = PHASES[idx]
    section(num, f"{icon}  {title}")

    done = _store()
    keys = [f"{num}.{i}" for i in range(len(items))]
    n_done = sum(1 for k in keys if k in done)

    c1, c2 = st.columns([3, 1])
    c1.progress(n_done/len(items),
                text=f"{n_done} of {len(items)} complete")
    if c2.button("Tick all", key=f"all_{num}", width="stretch"):
        done.update(keys)
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    for i, (text, ref, why) in enumerate(items):
        k = f"{num}.{i}"
        cols = st.columns([0.06, 0.78, 0.16])
        checked = cols[0].checkbox(f"check {k}", value=(k in done),
                                   key=f"cb_{k}",
                                   label_visibility="collapsed")
        if checked:
            done.add(k)
        else:
            done.discard(k)
        style = ("opacity:.5;text-decoration:line-through" if checked else "")
        cols[1].markdown(
            f'<div style="{style};padding-top:2px">{text}</div>',
            unsafe_allow_html=True)
        cols[2].markdown(
            f'<div style="text-align:right;color:{C["muted"]};'
            f'font-size:.82rem;padding-top:3px">{ref}</div>',
            unsafe_allow_html=True)
        if why:
            cols[1].caption(why)

    st.markdown("<hr/>", unsafe_allow_html=True)

    total = sum(len(p[3]) for p in PHASES)
    st.caption(f"Overall: **{len(done)} of {total}** checks complete "
               f"across all eight phases.")


def render_overview():
    section("0", "How to use this")

    lead(
        "This is Aurélien Géron's project checklist, expanded with the specific "
        "traps each step exists to prevent and a cross-reference to the chapter "
        "that covers it. Your ticks persist for the session."
    )

    done = _store()
    total = sum(len(p[3]) for p in PHASES)

    m1, m2, m3 = st.columns(3)
    m1.metric("Checks completed", f"{len(done)} / {total}")
    m2.metric("Phases", len(PHASES))
    phase_done = sum(1 for p in PHASES
                     if all(f"{p[0]}.{i}" in done for i in range(len(p[3]))))
    m3.metric("Phases complete", f"{phase_done} / {len(PHASES)}")

    counts = [sum(1 for i in range(len(p[3])) if f"{p[0]}.{i}" in done)
              for p in PHASES]
    totals = [len(p[3]) for p in PHASES]
    f = go.Figure()
    f.add_bar(x=[f"{p[0]}. {p[1]}" for p in PHASES], y=counts,
              name="complete", marker=dict(color=C["success"]))
    f.add_bar(x=[f"{p[0]}. {p[1]}" for p in PHASES],
              y=[t-c for t, c in zip(totals, counts)],
              name="remaining", marker=dict(color=alpha(C["line"], .7)))
    f.update_layout(height=380, barmode="stack", yaxis_title="checks",
                    xaxis=dict(tickangle=-25),
                    title="Progress by phase")
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    c1, c2 = st.columns(2)
    if c1.button("Reset every tick", width="stretch"):
        done.clear()
        st.rerun()
    c2.download_button(
        "Download as JSON",
        json.dumps({"completed": sorted(done),
                    "total": total,
                    "phases": [{"number": p[0], "title": p[1],
                                "items": [{"text": t, "ref": r,
                                           "done": f"{p[0]}.{i}" in done}
                                          for i, (t, r, _) in enumerate(p[3])]}
                               for p in PHASES]}, indent=2),
        file_name="ml_project_checklist.json", mime="application/json",
        width="stretch")

    rule()

    sub("The four checks that matter most")

    table(
        ["Check", "Phase", "Why it is on this list"],
        [["<b>Put the test set aside and never look at it</b>", "2",
          "Every other honest number depends on this one. Once you have tuned "
          "against it, it is a validation set and you have no test set."],
         ["<b>Write every preparation step as a function</b>", "4",
          "It is what lets you reproduce, reuse on new data, and put the "
          "preprocessing inside the SavedModel (§19.1) — which is what prevents "
          "training/serving skew."],
         ["<b>Prefer models that make DIFFERENT errors</b>", "5",
          "Ensemble variance is $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$: "
          "decorrelation beats quantity."],
         ["<b>Test the rollback</b>", "8",
          "The one item people skip, and the one they need at 3 a.m."]],
    )

    pitfall(
        "The checklist's real purpose is to slow you down at step 1",
        "Almost every failed ML project failed at framing, not at modelling. "
        "The objective was not stated in business terms, so nobody noticed the "
        "metric was misaligned; the current solution was never measured, so "
        "nobody knew whether the model was an improvement; the use was never "
        "specified, so the latency budget appeared after the model was built. "
        "<b>Phases 1 and 2 are where the leverage is.</b> Phase 6 is where the "
        "fun is, which is exactly why it gets all the attention.",
    )

    anim_header("Where projects actually fail")

    stages = ["frame the\nproblem", "get\ndata", "explore", "prepare",
              "shortlist", "tune", "present", "deploy &\nmaintain"]
    survival = np.array([100, 82, 78, 74, 71, 68, 62, 47])
    causes = ["objective never stated in business terms",
              "data unavailable, or legally blocked",
              "leakage found (or worse, not found)",
              "pipeline not reproducible",
              "no model beats the existing solution",
              "tuned against the test set",
              "stakeholders were never brought along",
              "no monitoring; the model rots silently"]

    frames = []
    for k in range(1, len(stages)+1):
        cols = [C["success"] if i < k-1 else C["danger"] if i == k-1
                else alpha(C["line"], .5) for i in range(len(stages))]
        frames.append(go.Frame(name=stages[k-1].replace("\n", " "), data=[
            go.Bar(x=stages, y=survival, marker=dict(color=cols),
                   text=[f"{v}%" for v in survival], textposition="outside"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{stages[k-1].replace(chr(10), ' ')}   ·   "
            f"{survival[k-1]}% still alive   ·   typical cause of loss here: "
            f"{causes[k-1]}", color=C["danger"])])))

    f = go.Figure(data=[go.Bar(x=stages, y=survival,
                              marker=dict(color=[C["danger"]] +
                                          [alpha(C["line"], .5)]*7),
                              text=[f"{v}%" for v in survival],
                              textposition="outside")])
    f.update_layout(height=430, yaxis_title="projects still viable (%)",
                    yaxis=dict(range=[0, 115]),
                    title="Illustrative attrition through a project")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="phase ")
    figure(f, "Illustrative proportions, but the ordering of causes is real: "
              "framing and deployment lose more projects than modelling does.")

    idea(
        "Print this and put it on the wall",
        "The value of a checklist is not that the items are surprising — most "
        "of them are obvious once read. It is that under time pressure, on a "
        "project you have been staring at for three weeks, obvious things get "
        "skipped. Atul Gawande's argument for surgical checklists applies "
        "unchanged: <b>the failures are rarely from ignorance, they are from "
        "not doing the thing you already knew.</b>",
    )

    refs([
        ("Géron — *Hands-On Machine Learning*, Appendix A",
         "https://github.com/ageron/handson-ml3"),
        ("Sculley et al. — *Hidden Technical Debt in Machine Learning Systems*",
         "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html"),
        ("Breck et al. — *The ML Test Score: A Rubric for ML Production "
         "Readiness*", "https://research.google/pubs/pub46555/"),
        ("Gawande — *The Checklist Manifesto*",
         "https://en.wikipedia.org/wiki/The_Checklist_Manifesto"),
        ("Mitchell et al. — *Model Cards for Model Reporting*",
         "https://arxiv.org/abs/1810.03993"),
        ("Gebru et al. — *Datasheets for Datasets*",
         "https://arxiv.org/abs/1803.09010"),
    ])


# --------------------------------------------------------------------------

SECTIONS = [("0", "How to use this", render_overview)] + [
    (p[0], f"{p[2]} {p[1]}", (lambda i: (lambda: render_phase(i)))(i))
    for i, p in enumerate(PHASES)
]

nav.render_chapter(CH, SECTIONS, sidebar_title="Phases")
