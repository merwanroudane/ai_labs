"""AI Lab — a live model workbench.

Pick a dataset, pick a model, turn the knobs, and watch what happens.  Every
control maps onto a concept from a specific chapter, and the page says which.
"""

from __future__ import annotations

import io
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, datasets as ds, nav
from core.lecture import (anim_header, codenote, figure, hero, idea, keypoints,
                          lead, math, md, note, pitfall, section, sub, table,
                          tip, warn)
from core.palette import C, CLASS_COLORS, SEQ, alpha
from core.runner import code_lab, execute, get_namespace, reset_namespace
from core.theme import PLOTLY_CONFIG, inject

inject()
CH = "ailab"

hero(
    kicker="Labs & Reference",
    title="AI Lab · live model workbench",
    blurb=(
        "Everything in the nineteen chapters, wired to knobs. Load a dataset, "
        "fit a model, sweep a hyperparameter, compare architectures head to "
        "head, or drop into a full Python scratchpad with the whole scientific "
        "stack preloaded."
    ),
    chips=["6 workbenches", "12 datasets", "20+ models", "free-form REPL"],
)


# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------

TABULAR = {
    "Moons (2 classes, non-linear)": ("clf", lambda: ds.moons(n=600, noise=.24)),
    "Circles (2 classes, concentric)": ("clf", lambda: ds.circles(n=600)),
    "Blobs (4 classes)": ("clf", lambda: ds.blobs(n=700, centers=4)),
    "Anisotropic blobs": ("clf", lambda: ds.anisotropic_blobs(n=600)),
    "Iris (4 features, 3 classes)": ("clf", lambda: ds.iris()[:2]),
    "Wine (13 features, 3 classes)": ("clf", lambda: ds.wine()[:2]),
    "Breast cancer (30 features)": ("clf", lambda: ds.breast_cancer()[:2]),
    "Digits (64 features, 10 classes)": ("clf", lambda: ds.digits()[:2]),
    "Linear 1-D + noise": ("reg", lambda: ds.linear_1d(n=200)),
    "Quadratic 1-D": ("reg", lambda: ds.quadratic_1d(n=200)),
    "Sine 1-D": ("reg", lambda: ds.sine_1d(n=200)),
    "California housing": ("reg", None),
}


@st.cache_data(show_spinner=False)
def load_tabular(name: str):
    kind, fn = TABULAR[name]
    if name == "California housing":
        df = ds.housing()
        num = [c for c in df.columns
               if df[c].dtype.kind in "if" and c != "median_house_value"]
        X = df[num].fillna(df[num].median()).to_numpy().astype("float64")
        y = (df["median_house_value"].to_numpy() / 100000.0)
        return X, y, "reg", num
    out = fn()
    X, y = out[0], out[1]
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    names = [f"x{i+1}" for i in range(X.shape[1])]
    return X, y, kind, names


# --------------------------------------------------------------------------
# Workbench 1 — the classifier / regressor bench
# --------------------------------------------------------------------------


def bench_supervised():
    section("Bench 1", "Supervised learning · every model, every knob")

    lead(
        "The same train/validate loop from Chapter 2, with the model and its "
        "hyperparameters exposed. Change one thing and watch the decision "
        "boundary, the learning curve and the metrics move together."
    )

    c1, c2, c3 = st.columns([1.15, 1.15, 1.0])
    dname = c1.selectbox("Dataset", list(TABULAR), index=0, key="ab_ds")
    X, y, kind, feat_names = load_tabular(dname)

    if kind == "clf":
        models = ["Logistic regression", "SGD classifier", "Linear SVM",
                  "RBF SVM", "k-Nearest neighbours", "Decision tree",
                  "Random forest", "Extra trees", "AdaBoost",
                  "Gradient boosting", "Hist gradient boosting",
                  "Naive Bayes", "MLP (neural net)"]
    else:
        models = ["Linear regression", "Ridge", "Lasso", "Elastic net",
                  "Polynomial ridge", "SVR (RBF)", "Decision tree",
                  "Random forest", "Gradient boosting",
                  "Hist gradient boosting", "MLP (neural net)"]
    mname = c2.selectbox("Model", models, key="ab_model")
    test_frac = c3.slider("Test fraction", 0.1, 0.5, 0.25, 0.05, key="ab_split")

    st.markdown('<div class="mp-sbtitle">Hyperparameters</div>',
                unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    P: dict = {}

    if mname in ("Logistic regression", "Linear SVM", "RBF SVM", "SVR (RBF)"):
        P["C"] = 10 ** h1.slider("log₁₀ C  (inverse regularisation, §5.2)",
                                 -3.0, 3.0, 0.0, 0.25, key="ab_C")
    if mname in ("RBF SVM", "SVR (RBF)"):
        P["gamma"] = 10 ** h2.slider("log₁₀ γ  (kernel width, §5.5)",
                                     -3.0, 2.0, -0.5, 0.25, key="ab_g")
    if mname in ("Ridge", "Lasso", "Elastic net", "Polynomial ridge"):
        P["alpha"] = 10 ** h1.slider("log₁₀ α  (penalty, §4.9)",
                                     -4.0, 3.0, 0.0, 0.25, key="ab_a")
    if mname == "Elastic net":
        P["l1_ratio"] = h2.slider("l1_ratio  (0 = ridge, 1 = lasso)",
                                  0.0, 1.0, 0.5, 0.05, key="ab_l1")
    if mname == "Polynomial ridge":
        P["degree"] = h2.slider("Polynomial degree (§4.7)", 1, 12, 3, 1,
                                key="ab_deg")
    if mname == "k-Nearest neighbours":
        P["n_neighbors"] = h1.slider("k", 1, 60, 5, 1, key="ab_k")
        P["weights"] = h2.selectbox("Weighting", ["uniform", "distance"],
                                    key="ab_w")
    if "tree" in mname.lower() or "forest" in mname.lower() or \
            "boosting" in mname.lower() or mname == "AdaBoost":
        P["max_depth"] = h1.slider("max_depth  (§6.4)", 1, 20, 5, 1,
                                   key="ab_md")
    if mname in ("Random forest", "Extra trees", "AdaBoost",
                 "Gradient boosting"):
        P["n_estimators"] = h2.slider("n_estimators  (§7.3)", 5, 400, 100, 5,
                                      key="ab_ne")
    if mname in ("Gradient boosting", "Hist gradient boosting", "AdaBoost"):
        P["learning_rate"] = 10 ** h3.slider("log₁₀ learning_rate  (§7.6)",
                                             -3.0, 0.3, -1.0, 0.1, key="ab_lr")
    if mname == "MLP (neural net)":
        P["hidden"] = h1.select_slider("Hidden layers",
                                       [(16,), (32,), (64,), (32, 32),
                                        (64, 64), (128, 64), (64, 64, 64)],
                                       value=(64, 64), key="ab_h",
                                       format_func=lambda t: " × ".join(map(str, t)))
        P["alpha"] = 10 ** h2.slider("log₁₀ L2 penalty (§11.9)", -6.0, 1.0,
                                     -4.0, 0.5, key="ab_mlpa")
        P["max_iter"] = h3.slider("Max iterations", 50, 1500, 400, 50,
                                  key="ab_it")
    if mname == "SGD classifier":
        P["alpha"] = 10 ** h1.slider("log₁₀ α", -6.0, 0.0, -4.0, 0.25,
                                     key="ab_sgda")
        P["loss"] = h2.selectbox("Loss", ["hinge", "log_loss",
                                          "modified_huber"], key="ab_sgdl")

    P["scale"] = h4.checkbox("Standardise features (§2.5)", value=True,
                             key="ab_sc")
    P["seed"] = h4.number_input("Random seed", 0, 9999, 42, key="ab_seed")

    if not st.button("▶  Fit and evaluate", type="primary", key="ab_run",
                     width="stretch"):
        st.info("Set the knobs and press **Fit and evaluate**.", icon="🎛️")
        return

    with st.spinner("Fitting…"):
        try:
            res = _fit_supervised(X, y, kind, mname, P, test_frac)
        except Exception:
            st.error("Fit failed")
            st.code(traceback.format_exc(limit=4), language="text")
            return

    _report_supervised(res, kind, X, y, feat_names, mname)


def _build_estimator(kind, mname, P):
    from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  GradientBoostingRegressor,
                                  HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor,
                                  RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                      LogisticRegression, Ridge, SGDClassifier)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVC, SVR, LinearSVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    s = P.get("seed", 42)
    if kind == "clf":
        return {
            "Logistic regression": lambda: LogisticRegression(
                C=P.get("C", 1.0), max_iter=2000),
            "SGD classifier": lambda: SGDClassifier(
                alpha=P.get("alpha", 1e-4), loss=P.get("loss", "hinge"),
                random_state=s, max_iter=2000),
            "Linear SVM": lambda: LinearSVC(C=P.get("C", 1.0), max_iter=5000),
            "RBF SVM": lambda: SVC(C=P.get("C", 1.0), gamma=P.get("gamma", .5),
                                   probability=True, random_state=s),
            "k-Nearest neighbours": lambda: KNeighborsClassifier(
                n_neighbors=P.get("n_neighbors", 5),
                weights=P.get("weights", "uniform")),
            "Decision tree": lambda: DecisionTreeClassifier(
                max_depth=P.get("max_depth", 5), random_state=s),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=P.get("n_estimators", 100),
                max_depth=P.get("max_depth", 5), random_state=s, n_jobs=-1),
            "Extra trees": lambda: ExtraTreesClassifier(
                n_estimators=P.get("n_estimators", 100),
                max_depth=P.get("max_depth", 5), random_state=s, n_jobs=-1),
            "AdaBoost": lambda: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=P.get("max_depth", 2)),
                n_estimators=P.get("n_estimators", 100),
                learning_rate=P.get("learning_rate", .1), random_state=s),
            "Gradient boosting": lambda: GradientBoostingClassifier(
                n_estimators=P.get("n_estimators", 100),
                max_depth=P.get("max_depth", 3),
                learning_rate=P.get("learning_rate", .1), random_state=s),
            "Hist gradient boosting": lambda: HistGradientBoostingClassifier(
                max_depth=P.get("max_depth", 5),
                learning_rate=P.get("learning_rate", .1), random_state=s),
            "Naive Bayes": lambda: GaussianNB(),
            "MLP (neural net)": lambda: MLPClassifier(
                hidden_layer_sizes=P.get("hidden", (64, 64)),
                alpha=P.get("alpha", 1e-4), max_iter=P.get("max_iter", 400),
                random_state=s),
        }[mname]()
    return {
        "Linear regression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=P.get("alpha", 1.0)),
        "Lasso": lambda: Lasso(alpha=P.get("alpha", 1.0), max_iter=20000),
        "Elastic net": lambda: ElasticNet(alpha=P.get("alpha", 1.0),
                                          l1_ratio=P.get("l1_ratio", .5),
                                          max_iter=20000),
        "Polynomial ridge": lambda: make_pipeline(
            PolynomialFeatures(P.get("degree", 3), include_bias=False),
            Ridge(alpha=P.get("alpha", 1.0))),
        "SVR (RBF)": lambda: SVR(C=P.get("C", 1.0), gamma=P.get("gamma", .5)),
        "Decision tree": lambda: DecisionTreeRegressor(
            max_depth=P.get("max_depth", 5), random_state=s),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=P.get("n_estimators", 100),
            max_depth=P.get("max_depth", 5), random_state=s, n_jobs=-1),
        "Gradient boosting": lambda: GradientBoostingRegressor(
            n_estimators=P.get("n_estimators", 100),
            max_depth=P.get("max_depth", 3),
            learning_rate=P.get("learning_rate", .1), random_state=s),
        "Hist gradient boosting": lambda: HistGradientBoostingRegressor(
            max_depth=P.get("max_depth", 5),
            learning_rate=P.get("learning_rate", .1), random_state=s),
        "MLP (neural net)": lambda: MLPRegressor(
            hidden_layer_sizes=P.get("hidden", (64, 64)),
            alpha=P.get("alpha", 1e-4), max_iter=P.get("max_iter", 400),
            random_state=s),
    }[mname]()


def _fit_supervised(X, y, kind, mname, P, test_frac):
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    strat = y if kind == "clf" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, random_state=int(P["seed"]), stratify=strat)

    est = _build_estimator(kind, mname, P)
    model = make_pipeline(StandardScaler(), est) if P.get("scale") else est

    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    pred_s = (time.perf_counter() - t0) / max(1, len(Xtr) + len(Xte))

    scoring = "accuracy" if kind == "clf" else "r2"
    try:
        cv = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
    except Exception:
        cv = np.array([np.nan])

    proba = None
    if kind == "clf":
        for obj in (model, est):
            if hasattr(obj, "predict_proba"):
                try:
                    proba = model.predict_proba(Xte)
                    break
                except Exception:
                    pass

    return dict(model=model, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                ptr=ptr, pte=pte, proba=proba, cv=cv, fit_s=fit_s,
                pred_s=pred_s)


def _report_supervised(r, kind, X, y, feat_names, mname):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 mean_absolute_error, mean_squared_error,
                                 r2_score, roc_auc_score, roc_curve)

    st.markdown("<hr/>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    if kind == "clf":
        a_tr = accuracy_score(r["ytr"], r["ptr"])
        a_te = accuracy_score(r["yte"], r["pte"])
        m1.metric("Train accuracy", f"{a_tr:.4f}")
        m2.metric("Test accuracy", f"{a_te:.4f}", f"{a_te - a_tr:+.4f}")
        m3.metric("5-fold CV", f"{np.nanmean(r['cv']):.4f}",
                  f"± {np.nanstd(r['cv']):.4f}")
        m4.metric("Macro F1", f"{f1_score(r['yte'], r['pte'], average='macro'):.4f}")
    else:
        r2_tr = r2_score(r["ytr"], r["ptr"])
        r2_te = r2_score(r["yte"], r["pte"])
        m1.metric("Train R²", f"{r2_tr:.4f}")
        m2.metric("Test R²", f"{r2_te:.4f}", f"{r2_te - r2_tr:+.4f}")
        m3.metric("5-fold CV R²", f"{np.nanmean(r['cv']):.4f}",
                  f"± {np.nanstd(r['cv']):.4f}")
        m4.metric("Test RMSE",
                  f"{np.sqrt(mean_squared_error(r['yte'], r['pte'])):.4f}")
    m5.metric("Fit time", f"{r['fit_s']*1000:.0f} ms",
              f"{r['pred_s']*1e6:.1f} µs/row")

    gap = ((accuracy_score(r["ytr"], r["ptr"]) - accuracy_score(r["yte"], r["pte"]))
           if kind == "clf"
           else (r2_score(r["ytr"], r["ptr"]) - r2_score(r["yte"], r["pte"])))
    if gap > 0.12:
        st.warning(
            f"**Train − test gap = {gap:.3f}.** That is overfitting (§4.4). "
            "Reduce capacity, add regularisation, or get more data.", icon="⚠️")
    elif gap < -0.03:
        st.info("Test score exceeds train score — usually a small test set, "
                "or heavy regularisation.", icon="ℹ️")

    tabs = st.tabs(["Decision surface", "Predictions", "Diagnostics",
                    "Learning curve", "Feature importance"])

    # ---- decision surface / fit ----------------------------------------
    with tabs[0]:
        if X.shape[1] == 2:
            st.plotly_chart(_surface_2d(r, kind), width="stretch",
                            config=PLOTLY_CONFIG)
        elif X.shape[1] == 1 and kind == "reg":
            st.plotly_chart(_curve_1d(r), width="stretch",
                            config=PLOTLY_CONFIG)
        else:
            st.plotly_chart(_pca_projection(r, kind), width="stretch",
                            config=PLOTLY_CONFIG)
            st.caption("More than two features, so the points are projected "
                       "onto the first two principal components (§8.3). The "
                       "colouring shows whether each test point was predicted "
                       "correctly.")

    # ---- predictions ---------------------------------------------------
    with tabs[1]:
        if kind == "clf":
            cm = confusion_matrix(r["yte"], r["pte"])
            classes = sorted(np.unique(y))
            f = go.Figure(go.Heatmap(
                z=cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                colorscale=nav.cscale(), text=cm, texttemplate="%{text}",
                xgap=2, ygap=2))
            f.update_layout(height=420, xaxis_title="predicted",
                            yaxis_title="actual",
                            yaxis=dict(autorange="reversed"),
                            title="Confusion matrix (§3.3)")
            st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)
            if r["proba"] is not None and len(classes) == 2:
                fpr, tpr, _ = roc_curve(r["yte"], r["proba"][:, 1])
                auc = roc_auc_score(r["yte"], r["proba"][:, 1])
                g = go.Figure()
                g.add_scatter(x=fpr, y=tpr, mode="lines",
                              name=f"AUC = {auc:.4f}",
                              line=dict(color=C["primary"], width=3))
                g.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                              line=dict(color=C["muted"], dash="dash"))
                g.update_layout(height=400, xaxis_title="false positive rate",
                                yaxis_title="true positive rate",
                                title="ROC curve (§3.5)")
                st.plotly_chart(g, width="stretch", config=PLOTLY_CONFIG)
        else:
            f = go.Figure()
            lo = float(min(r["yte"].min(), r["pte"].min()))
            hi = float(max(r["yte"].max(), r["pte"].max()))
            f.add_scatter(x=r["yte"], y=r["pte"], mode="markers", name="test",
                          marker=dict(size=6, color=C["valid"], opacity=.7))
            f.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                          line=dict(color=C["muted"], dash="dash"))
            f.update_layout(height=430, xaxis_title="actual",
                            yaxis_title="predicted",
                            title="Predicted vs actual")
            st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    # ---- diagnostics ---------------------------------------------------
    with tabs[2]:
        if kind == "reg":
            resid = r["yte"] - r["pte"]
            f = make_subplots(rows=1, cols=2,
                              subplot_titles=("residuals vs prediction",
                                              "residual distribution"))
            f.add_trace(go.Scatter(x=r["pte"], y=resid, mode="markers",
                                   marker=dict(size=6, color=C["valid"],
                                               opacity=.7),
                                   showlegend=False), 1, 1)
            f.add_hline(y=0, line_dash="dash", line_color=C["muted"],
                        row=1, col=1)
            f.add_trace(go.Histogram(x=resid, nbinsx=40,
                                     marker=dict(color=C["primary"]),
                                     showlegend=False), 1, 2)
            f.update_layout(height=400)
            st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)
            st.caption("A pattern in the left panel means the model is missing "
                       "structure. A skewed right panel means the errors are "
                       "not symmetric — consider a different loss (§4.1).")
        else:
            if r["proba"] is not None:
                conf = r["proba"].max(1)
                correct = (r["pte"] == r["yte"])
                f = go.Figure()
                f.add_histogram(x=conf[correct], nbinsx=30, name="correct",
                                marker=dict(color=C["success"]), opacity=.7)
                f.add_histogram(x=conf[~correct], nbinsx=30, name="wrong",
                                marker=dict(color=C["danger"]), opacity=.7)
                f.update_layout(height=400, barmode="overlay",
                                xaxis_title="predicted probability of the "
                                            "chosen class",
                                yaxis_title="count",
                                title="Confidence, split by correctness")
                st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)
                bins = np.linspace(0, 1, 11)
                idx = np.digitize(conf, bins) - 1
                xs, ys = [], []
                for b in range(10):
                    m = idx == b
                    if m.sum() > 3:
                        xs.append(conf[m].mean())
                        ys.append(correct[m].mean())
                g = go.Figure()
                g.add_scatter(x=xs, y=ys, mode="lines+markers",
                              name="model", line=dict(color=C["primary"],
                                                      width=3))
                g.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="perfectly calibrated",
                              line=dict(color=C["muted"], dash="dash"))
                g.update_layout(height=380, xaxis_title="mean predicted "
                                                        "confidence",
                                yaxis_title="observed accuracy",
                                title="Calibration curve")
                st.plotly_chart(g, width="stretch", config=PLOTLY_CONFIG)
                st.caption("Below the diagonal means over-confident. Trees and "
                           "boosted models are usually badly calibrated; "
                           "logistic regression usually is not.")
            else:
                st.info("This model does not expose class probabilities, so "
                        "there is nothing to calibrate.", icon="ℹ️")

    # ---- learning curve ------------------------------------------------
    with tabs[3]:
        if st.button("Compute learning curve", key="ab_lc"):
            with st.spinner("Refitting at several training-set sizes…"):
                from sklearn.model_selection import learning_curve
                try:
                    sizes, tr, va = learning_curve(
                        r["model"], X, y, cv=4, n_jobs=-1,
                        train_sizes=np.linspace(.12, 1.0, 7),
                        scoring="accuracy" if kind == "clf" else "r2")
                    f = go.Figure()
                    f.add_scatter(x=sizes, y=tr.mean(1), mode="lines+markers",
                                  name="train",
                                  line=dict(color=C["train"], width=3))
                    f.add_scatter(x=sizes, y=va.mean(1), mode="lines+markers",
                                  name="validation",
                                  line=dict(color=C["valid"], width=3))
                    f.update_layout(height=420, xaxis_title="training examples",
                                    yaxis_title="accuracy" if kind == "clf"
                                    else "R²",
                                    title="Learning curve (§4.4)")
                    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)
                    gap_now = tr.mean(1)[-1] - va.mean(1)[-1]
                    if gap_now > .1:
                        st.warning(
                            "The curves have not converged — **more data would "
                            "help.** A large gap that persists is the signature "
                            "of high variance.", icon="📈")
                    else:
                        st.success(
                            "The curves have converged. More data will not "
                            "help; you need a **more expressive model** or "
                            "better features.", icon="📉")
                except Exception:
                    st.code(traceback.format_exc(limit=3), language="text")
        else:
            st.caption("Refits the model at several training-set sizes — a few "
                       "seconds of work, so it is behind a button.")

    # ---- importance ----------------------------------------------------
    with tabs[4]:
        est = r["model"]
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        imp, label = None, ""
        if hasattr(est, "feature_importances_"):
            imp, label = est.feature_importances_, "impurity-based importance"
        elif hasattr(est, "coef_"):
            c = np.asarray(est.coef_)
            imp = np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)
            label = "|coefficient|"
        if imp is not None and len(imp) == len(feat_names):
            order = np.argsort(imp)[::-1][:25]
            f = go.Figure(go.Bar(x=[feat_names[i] for i in order],
                                 y=imp[order],
                                 marker=dict(color=C["primary"])))
            f.update_layout(height=400, yaxis_title=label,
                            title=f"Feature importance ({label})")
            st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)
            st.caption("Impurity-based importance is biased toward "
                       "high-cardinality features (§7.4). For a decision you "
                       "will act on, use permutation importance instead.")
        else:
            st.info("This model does not expose per-feature importances.",
                    icon="ℹ️")


def _surface_2d(r, kind):
    Xall = np.vstack([r["Xtr"], r["Xte"]])
    pad = 0.35
    x1 = np.linspace(Xall[:, 0].min()-pad, Xall[:, 0].max()+pad, 220)
    x2 = np.linspace(Xall[:, 1].min()-pad, Xall[:, 1].max()+pad, 220)
    G1, G2 = np.meshgrid(x1, x2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    try:
        if kind == "clf" and hasattr(r["model"], "predict_proba"):
            Z = r["model"].predict_proba(grid)
            Z = Z[:, 1] if Z.shape[1] == 2 else Z.argmax(1)
        else:
            Z = r["model"].predict(grid)
    except Exception:
        Z = r["model"].predict(grid)
    Z = np.asarray(Z, dtype=float).reshape(G1.shape)

    f = go.Figure()
    f.add_contour(x=x1, y=x2, z=Z, colorscale=nav.cscale(), opacity=.65,
                  showscale=True, contours=dict(showlines=False),
                  colorbar=dict(title="score"))
    if kind == "clf":
        for i, c in enumerate(sorted(np.unique(r["ytr"]))):
            m = r["ytr"] == c
            f.add_scatter(x=r["Xtr"][m, 0], y=r["Xtr"][m, 1], mode="markers",
                          name=f"train {c}",
                          marker=dict(size=6, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                                      line=dict(color="#fff", width=.6)))
            m = r["yte"] == c
            f.add_scatter(x=r["Xte"][m, 0], y=r["Xte"][m, 1], mode="markers",
                          name=f"test {c}",
                          marker=dict(size=9, symbol="diamond",
                                      color=CLASS_COLORS[i % len(CLASS_COLORS)],
                                      line=dict(color=C["ink"], width=1.2)))
        wrong = r["pte"] != r["yte"]
        if wrong.any():
            f.add_scatter(x=r["Xte"][wrong, 0], y=r["Xte"][wrong, 1],
                          mode="markers", name="misclassified",
                          marker=dict(size=15, symbol="x",
                                      color=C["danger"],
                                      line=dict(width=2.5)))
    else:
        f.add_scatter(x=r["Xtr"][:, 0], y=r["Xtr"][:, 1], mode="markers",
                      name="train",
                      marker=dict(size=6, color=r["ytr"],
                                  colorscale=nav.cscale(),
                                  line=dict(color="#fff", width=.6)))
    f.update_layout(height=560, xaxis_title="x₁", yaxis_title="x₂",
                    title="Decision surface" if kind == "clf"
                    else "Fitted surface",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return f


def _curve_1d(r):
    xs = np.linspace(float(min(r["Xtr"].min(), r["Xte"].min())),
                     float(max(r["Xtr"].max(), r["Xte"].max())), 400)[:, None]
    ys = r["model"].predict(xs)
    f = go.Figure()
    f.add_scatter(x=r["Xtr"][:, 0], y=r["ytr"], mode="markers", name="train",
                  marker=dict(size=7, color=C["train"], opacity=.75))
    f.add_scatter(x=r["Xte"][:, 0], y=r["yte"], mode="markers", name="test",
                  marker=dict(size=9, symbol="diamond", color=C["valid"]))
    f.add_scatter(x=xs[:, 0], y=ys, mode="lines", name="model",
                  line=dict(color=C["primary"], width=3.5))
    f.update_layout(height=480, xaxis_title="x", yaxis_title="y",
                    title="Fitted function",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return f


def _pca_projection(r, kind):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2).fit(r["Xtr"])
    Ztr, Zte = p.transform(r["Xtr"]), p.transform(r["Xte"])
    f = go.Figure()
    f.add_scatter(x=Ztr[:, 0], y=Ztr[:, 1], mode="markers", name="train",
                  marker=dict(size=5, color=alpha(C["muted"], .55)))
    if kind == "clf":
        ok = r["pte"] == r["yte"]
        f.add_scatter(x=Zte[ok, 0], y=Zte[ok, 1], mode="markers",
                      name="test · correct",
                      marker=dict(size=9, color=C["success"],
                                  line=dict(color="#fff", width=.8)))
        f.add_scatter(x=Zte[~ok, 0], y=Zte[~ok, 1], mode="markers",
                      name="test · wrong",
                      marker=dict(size=12, symbol="x", color=C["danger"],
                                  line=dict(width=2.5)))
    else:
        err = np.abs(r["pte"] - r["yte"])
        f.add_scatter(x=Zte[:, 0], y=Zte[:, 1], mode="markers", name="test",
                      marker=dict(size=9, color=err, colorscale=nav.cscale(),
                                  colorbar=dict(title="|error|")))
    f.update_layout(height=520,
                    xaxis_title=f"PC1 ({p.explained_variance_ratio_[0]:.1%})",
                    yaxis_title=f"PC2 ({p.explained_variance_ratio_[1]:.1%})",
                    title="Test set projected onto two principal components",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return f


# --------------------------------------------------------------------------
# Workbench 2 — hyperparameter sweep
# --------------------------------------------------------------------------


def bench_sweep():
    section("Bench 2", "Hyperparameter sweep · watch the bias–variance curve")

    lead(
        "Fix everything except one hyperparameter and sweep it. The gap between "
        "the train and validation curves <b>is</b> the bias–variance trade-off "
        "of §4.4, drawn from your own data."
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    dname = c1.selectbox("Dataset", list(TABULAR), index=0, key="sw_ds")
    X, y, kind, _ = load_tabular(dname)

    sweeps = {
        "Decision tree · max_depth": ("tree", "max_depth",
                                      list(range(1, 21))),
        "k-NN · k": ("knn", "n_neighbors",
                     [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 70, 100]),
        "SVM · C": ("svc", "C", list(10.0 ** np.arange(-3, 3.5, .5))),
        "SVM · gamma": ("svcg", "gamma", list(10.0 ** np.arange(-3, 2.5, .5))),
        "Random forest · n_estimators": ("rf", "n_estimators",
                                         [1, 2, 5, 10, 20, 40, 80, 160, 300]),
        "Ridge · alpha": ("ridge", "alpha", list(10.0 ** np.arange(-4, 4.5, .5))),
        "Polynomial · degree": ("poly", "degree", list(range(1, 16))),
        "Gradient boosting · n_estimators": ("gb", "n_estimators",
                                             [1, 2, 5, 10, 25, 50, 100, 200,
                                              400]),
    }
    valid = [k for k in sweeps
             if not (kind == "reg" and sweeps[k][0] in ("knn", "svc", "svcg",
                                                        "gb"))
             and not (kind == "clf" and sweeps[k][0] in ("ridge", "poly"))]
    sname = c2.selectbox("Sweep", valid, key="sw_p")
    folds = c3.slider("CV folds", 2, 8, 4, key="sw_cv")

    if not st.button("▶  Run the sweep", type="primary", key="sw_run",
                     width="stretch"):
        st.info("Choose a hyperparameter and press **Run the sweep**.",
                icon="📈")
        return

    tag, pname, values = sweeps[sname]
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import validation_curve
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if tag == "tree":
        est = (DecisionTreeClassifier(random_state=0) if kind == "clf"
               else DecisionTreeRegressor(random_state=0))
        pk = pname
    elif tag == "knn":
        est = make_pipeline(StandardScaler(), KNeighborsClassifier())
        pk = f"kneighborsclassifier__{pname}"
    elif tag in ("svc", "svcg"):
        est = make_pipeline(StandardScaler(), SVC())
        pk = f"svc__{pname}"
    elif tag == "rf":
        est = (RandomForestClassifier(random_state=0, n_jobs=-1)
               if kind == "clf"
               else RandomForestRegressor(random_state=0, n_jobs=-1))
        pk = pname
    elif tag == "ridge":
        est = make_pipeline(StandardScaler(), Ridge())
        pk = f"ridge__{pname}"
    elif tag == "poly":
        est = make_pipeline(PolynomialFeatures(include_bias=False),
                            StandardScaler(), Ridge(alpha=1e-3))
        pk = f"polynomialfeatures__{pname}"
    else:
        est = GradientBoostingClassifier(random_state=0)
        pk = pname

    with st.spinner(f"Fitting {len(values) * folds} models…"):
        try:
            tr, va = validation_curve(
                est, X, y, param_name=pk, param_range=values, cv=folds,
                scoring="accuracy" if kind == "clf" else "r2", n_jobs=-1)
        except Exception:
            st.error("Sweep failed")
            st.code(traceback.format_exc(limit=4), language="text")
            return

    tm, ts = tr.mean(1), tr.std(1)
    vm, vs = va.mean(1), va.std(1)
    best = int(np.argmax(vm))

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Best {pname}", f"{values[best]:.4g}")
    m2.metric("Validation score", f"{vm[best]:.4f}", f"± {vs[best]:.4f}")
    m3.metric("Train − validation gap", f"{tm[best] - vm[best]:+.4f}")

    xs = np.asarray(values, dtype=float)
    logx = tag in ("svc", "svcg", "ridge") or (xs.max() / max(xs.min(), 1e-12) > 200)
    f = go.Figure()
    f.add_scatter(x=xs, y=tm+ts, mode="lines", line=dict(width=0),
                  showlegend=False, hoverinfo="skip")
    f.add_scatter(x=xs, y=tm-ts, mode="lines", line=dict(width=0),
                  fill="tonexty", fillcolor=alpha(C["train"], .18),
                  showlegend=False, hoverinfo="skip")
    f.add_scatter(x=xs, y=vm+vs, mode="lines", line=dict(width=0),
                  showlegend=False, hoverinfo="skip")
    f.add_scatter(x=xs, y=vm-vs, mode="lines", line=dict(width=0),
                  fill="tonexty", fillcolor=alpha(C["valid"], .18),
                  showlegend=False, hoverinfo="skip")
    f.add_scatter(x=xs, y=tm, mode="lines+markers", name="train",
                  line=dict(color=C["train"], width=3))
    f.add_scatter(x=xs, y=vm, mode="lines+markers", name="validation",
                  line=dict(color=C["valid"], width=3))
    f.add_vline(x=xs[best], line_dash="dash", line_color=C["success"],
                annotation_text=f"best = {values[best]:.4g}")
    f.update_layout(height=470, xaxis_title=pname,
                    xaxis_type="log" if logx else "linear",
                    yaxis_title="accuracy" if kind == "clf" else "R²",
                    title=sname,
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    left = tm[0] - vm[0]
    right = tm[-1] - vm[-1]
    if right > left + .05:
        st.info(
            f"**The gap widens to the right** ({left:+.3f} → {right:+.3f}). "
            f"Larger `{pname}` means more capacity here — the right-hand end "
            "is the high-variance regime (§4.4).", icon="🔍")
    elif left > right + .05:
        st.info(
            f"**The gap narrows to the right** ({left:+.3f} → {right:+.3f}). "
            f"Larger `{pname}` regularises here — the left-hand end is the "
            "high-variance regime.", icon="🔍")

    st.dataframe(pd.DataFrame({pname: values, "train": tm.round(4),
                               "validation": vm.round(4),
                               "gap": (tm-vm).round(4)}), width="stretch")


# --------------------------------------------------------------------------
# Workbench 3 — the arena
# --------------------------------------------------------------------------


def bench_arena():
    section("Bench 3", "Model arena · a dozen models on the same split")

    lead(
        "The honest comparison: identical data, identical split, default "
        "hyperparameters, and a real timing column. Accuracy is not the only "
        "axis that matters in production (§19.2)."
    )

    c1, c2 = st.columns([1.6, 1.0])
    dname = c1.selectbox("Dataset", list(TABULAR), index=0, key="ar_ds")
    X, y, kind, _ = load_tabular(dname)
    folds = c2.slider("CV folds", 3, 10, 5, key="ar_cv")

    if not st.button("▶  Run the arena", type="primary", key="ar_run",
                     width="stretch"):
        st.info("Press **Run the arena** to fit every model on this dataset.",
                icon="🏟️")
        return

    from sklearn.model_selection import cross_validate

    names = (["Logistic regression", "Linear SVM", "RBF SVM",
              "k-Nearest neighbours", "Naive Bayes", "Decision tree",
              "Random forest", "Extra trees", "AdaBoost",
              "Gradient boosting", "Hist gradient boosting",
              "MLP (neural net)"] if kind == "clf" else
             ["Linear regression", "Ridge", "Lasso", "Elastic net",
              "SVR (RBF)", "Decision tree", "Random forest",
              "Gradient boosting", "Hist gradient boosting",
              "MLP (neural net)"])

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    bar = st.progress(0.0, text="Fitting…")
    for i, nm in enumerate(names):
        bar.progress((i+1)/len(names), text=f"Fitting {nm}…")
        try:
            est = _build_estimator(kind, nm, dict(seed=42))
            pipe = make_pipeline(StandardScaler(), est)
            cv = cross_validate(pipe, X, y, cv=folds, n_jobs=-1,
                                scoring="accuracy" if kind == "clf" else "r2",
                                return_train_score=True)
            rows.append(dict(model=nm,
                             test=float(cv["test_score"].mean()),
                             std=float(cv["test_score"].std()),
                             train=float(cv["train_score"].mean()),
                             gap=float(cv["train_score"].mean()
                                       - cv["test_score"].mean()),
                             fit_ms=float(cv["fit_time"].mean()*1000),
                             score_ms=float(cv["score_time"].mean()*1000)))
        except Exception as e:
            rows.append(dict(model=nm, test=np.nan, std=np.nan, train=np.nan,
                             gap=np.nan, fit_ms=np.nan, score_ms=np.nan))
    bar.empty()

    R = pd.DataFrame(rows).sort_values("test", ascending=False)
    metric = "accuracy" if kind == "clf" else "R²"

    m1, m2, m3 = st.columns(3)
    top = R.iloc[0]
    m1.metric("Best model", top["model"], f"{top['test']:.4f} {metric}")
    fastest = R.loc[R.fit_ms.idxmin()]
    m2.metric("Fastest to fit", fastest["model"], f"{fastest['fit_ms']:.0f} ms")
    within = R[R.test >= top["test"] - top["std"]]
    m3.metric("Statistically tied with the best", f"{len(within)} models",
              f"within ±1 std")

    f = go.Figure()
    f.add_bar(x=R["model"], y=R["test"],
              error_y=dict(type="data", array=R["std"]),
              marker=dict(color=[C["success"] if v >= top["test"] - top["std"]
                                 else C["primary"] for v in R["test"]]),
              name=f"cross-validated {metric}")
    f.update_layout(height=460, yaxis_title=f"{folds}-fold {metric}",
                    title=f"{dname} — green bars are tied with the best",
                    xaxis=dict(tickangle=-35))
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    g = go.Figure()
    g.add_scatter(x=R["fit_ms"], y=R["test"], mode="markers+text",
                  text=R["model"], textposition="top center",
                  marker=dict(size=13, color=R["gap"], colorscale=nav.cscale(),
                              colorbar=dict(title="train−test<br>gap"),
                              line=dict(color="#fff", width=1)))
    g.update_layout(height=470, xaxis_type="log",
                    xaxis_title="mean fit time (ms, log scale)",
                    yaxis_title=f"cross-validated {metric}",
                    title="Accuracy against cost — the axis people forget")
    st.plotly_chart(g, width="stretch", config=PLOTLY_CONFIG)

    st.dataframe(R.round(4), width="stretch", hide_index=True)

    if len(within) > 1:
        st.info(
            f"**{len(within)} models are within one standard deviation of the "
            f"best.** Choosing between them on the mean alone is choosing "
            "noise. Pick on the other axes: fit time, inference cost, "
            "interpretability, and how badly each one fails (§2.9).",
            icon="🎯")


# --------------------------------------------------------------------------
# Workbench 4 — clustering & dimensionality reduction
# --------------------------------------------------------------------------


def bench_unsupervised():
    section("Bench 4", "Unsupervised · clustering and dimensionality reduction")

    lead(
        "No labels. Cluster the data, project it, and check with a silhouette "
        "score whether the structure you can see is really there (§9.4)."
    )

    c1, c2, c3 = st.columns([1.2, 1.1, 1.0])
    dsets = {
        "Blobs (4 true clusters)": lambda: ds.blobs(n=700, centers=4),
        "Anisotropic blobs": lambda: ds.anisotropic_blobs(n=700),
        "Moons": lambda: ds.moons(n=700, noise=.08),
        "Circles": lambda: ds.circles(n=700, noise=.06),
        "Swiss roll (3-D)": lambda: ds.swiss_roll(n=1200),
        "S-curve (3-D)": lambda: ds.s_curve(n=1200),
        "Digits (64-D)": lambda: ds.digits()[:2],
        "Iris": lambda: ds.iris()[:2],
        "Wine": lambda: ds.wine()[:2],
    }
    dname = c1.selectbox("Dataset", list(dsets), key="un_ds")
    out = dsets[dname]()
    X = np.asarray(out[0], dtype="float64")
    y_true = np.asarray(out[1])

    algo = c2.selectbox("Algorithm",
                        ["k-Means", "Mini-batch k-Means", "DBSCAN",
                         "Agglomerative", "Gaussian mixture", "Spectral"],
                        key="un_alg")
    k = c3.slider("k / n_components", 2, 12, 4, key="un_k")

    d1, d2, d3 = st.columns(3)
    eps = d1.slider("DBSCAN eps", 0.05, 2.0, 0.35, 0.05, key="un_eps")
    minpts = d2.slider("DBSCAN min_samples", 2, 30, 5, key="un_mp")
    proj = d3.selectbox("Projection", ["PCA", "t-SNE", "LLE", "Isomap",
                                       "Kernel PCA (RBF)"], key="un_proj")

    if not st.button("▶  Cluster and project", type="primary", key="un_run",
                     width="stretch"):
        st.info("Press **Cluster and project**.", icon="🔮")
        return

    from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                                 MiniBatchKMeans, SpectralClustering)
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
    from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                                 davies_bouldin_score, silhouette_score)
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)

    with st.spinner("Clustering…"):
        try:
            if algo == "k-Means":
                lab = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
            elif algo == "Mini-batch k-Means":
                lab = MiniBatchKMeans(k, n_init=10,
                                      random_state=42).fit_predict(Xs)
            elif algo == "DBSCAN":
                lab = DBSCAN(eps=eps, min_samples=minpts).fit_predict(Xs)
            elif algo == "Agglomerative":
                lab = AgglomerativeClustering(k).fit_predict(Xs)
            elif algo == "Gaussian mixture":
                lab = GaussianMixture(k, random_state=42).fit_predict(Xs)
            else:
                lab = SpectralClustering(k, random_state=42,
                                         affinity="nearest_neighbors"
                                         ).fit_predict(Xs)
        except Exception:
            st.error("Clustering failed")
            st.code(traceback.format_exc(limit=3), language="text")
            return

    n_found = len(set(lab)) - (1 if -1 in lab else 0)
    noise = float((lab == -1).mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters found", n_found)
    m2.metric("Noise points", f"{noise:.1%}")
    try:
        mask = lab != -1
        sil = silhouette_score(Xs[mask], lab[mask]) if n_found > 1 else np.nan
    except Exception:
        sil = np.nan
    m3.metric("Silhouette", f"{sil:.4f}" if np.isfinite(sil) else "—")
    try:
        ari = adjusted_rand_score(y_true, lab)
        m4.metric("Adjusted Rand vs truth", f"{ari:.4f}")
    except Exception:
        m4.metric("Adjusted Rand", "—")

    with st.spinner(f"Projecting with {proj}…"):
        try:
            if proj == "PCA":
                Z = PCA(2, random_state=42).fit_transform(Xs)
            elif proj == "t-SNE":
                Z = TSNE(2, random_state=42, init="pca",
                         perplexity=min(30, len(Xs)//4)).fit_transform(Xs)
            elif proj == "LLE":
                Z = LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                           random_state=42).fit_transform(Xs)
            elif proj == "Isomap":
                Z = Isomap(n_components=2, n_neighbors=12).fit_transform(Xs)
            else:
                Z = KernelPCA(2, kernel="rbf", gamma=.04,
                              random_state=42).fit_transform(Xs)
        except Exception:
            Z = PCA(2).fit_transform(Xs)
            st.warning(f"{proj} failed; showing PCA instead.", icon="⚠️")

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=(f"{algo} labels", "true labels"))
    for i, c in enumerate(sorted(set(lab))):
        m = lab == c
        col = C["muted"] if c == -1 else CLASS_COLORS[i % len(CLASS_COLORS)]
        f.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers",
                               name=("noise" if c == -1 else f"cluster {c}"),
                               marker=dict(size=6, color=col,
                                           line=dict(color="#fff", width=.4))),
                    1, 1)
    for i, c in enumerate(sorted(set(y_true))):
        m = y_true == c
        f.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers",
                               name=f"true {c}", showlegend=False,
                               marker=dict(size=6,
                                           color=CLASS_COLORS[i % len(CLASS_COLORS)],
                                           line=dict(color="#fff", width=.4))),
                    1, 2)
    f.update_layout(height=520, title=f"{dname} projected with {proj}")
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    if st.checkbox("Show the silhouette sweep over k", key="un_sweep"):
        with st.spinner("Sweeping k…"):
            ks, sils, inertias = [], [], []
            for kk in range(2, 13):
                km = KMeans(kk, n_init=10, random_state=42).fit(Xs)
                ks.append(kk)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(Xs, km.labels_))
        g = make_subplots(specs=[[{"secondary_y": True}]])
        g.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                               name="inertia (the elbow)",
                               line=dict(color=C["muted"], width=3)), False)
        g.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers",
                               name="silhouette",
                               line=dict(color=C["primary"], width=3)), True)
        best_k = ks[int(np.argmax(sils))]
        g.add_vline(x=best_k, line_dash="dash", line_color=C["success"],
                    annotation_text=f"best silhouette at k = {best_k}")
        g.update_layout(height=420, xaxis_title="k",
                        title="Inertia always falls; silhouette has an optimum")
        st.plotly_chart(g, width="stretch", config=PLOTLY_CONFIG)
        st.caption("Inertia decreases monotonically, so 'the elbow' is a "
                   "judgement call. The silhouette actually has a maximum — "
                   "prefer it (§9.4).")


# --------------------------------------------------------------------------
# Workbench 5 — the neural playground
# --------------------------------------------------------------------------


def bench_neural():
    section("Bench 5", "Neural playground · architecture, optimiser, "
                       "regularisation")

    lead(
        "Chapters 10 and 11, wired to knobs. Every control here corresponds to "
        "a specific decision — depth, width, activation, initialiser, "
        "optimiser, learning rate, batch norm, dropout — and you can see the "
        "training curve respond to each one."
    )

    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        st.error("TensorFlow is not importable in this environment, so the "
                 "neural playground cannot run. Every other bench works.",
                 icon="🚫")
        return

    c1, c2, c3, c4 = st.columns(4)
    dname = c1.selectbox("Dataset", ["Moons", "Circles", "Blobs (4)",
                                     "Digits (10 classes)",
                                     "Breast cancer"], key="nn_ds")
    depth = c2.slider("Hidden layers (§10.4)", 1, 8, 3, key="nn_d")
    width = c3.select_slider("Units per layer", [8, 16, 32, 64, 128, 256],
                             value=32, key="nn_w")
    epochs = c4.slider("Epochs", 5, 200, 60, 5, key="nn_e")

    d1, d2, d3, d4 = st.columns(4)
    act = d1.selectbox("Activation (§11.2)",
                       ["relu", "elu", "selu", "gelu", "tanh", "sigmoid",
                        "swish"], key="nn_a")
    init = d2.selectbox("Initialiser (§11.1)",
                        ["glorot_uniform", "he_normal", "lecun_normal",
                         "random_normal", "zeros"], key="nn_i")
    opt_name = d3.selectbox("Optimiser (§11.7)",
                            ["Adam", "AdamW", "Nadam", "SGD", "SGD+momentum",
                             "RMSprop", "Adagrad"], key="nn_o")
    lr = 10 ** d4.slider("log₁₀ learning rate", -5.0, 0.0, -2.5, 0.25,
                         key="nn_lr")

    e1, e2, e3, e4 = st.columns(4)
    bn = e1.checkbox("Batch normalisation (§11.3)", key="nn_bn")
    drop = e2.slider("Dropout (§11.9)", 0.0, 0.7, 0.0, 0.05, key="nn_dr")
    l2 = 10 ** e3.slider("log₁₀ L2 penalty", -8.0, -1.0, -8.0, 0.5,
                         key="nn_l2")
    batch = e4.select_slider("Batch size", [8, 16, 32, 64, 128, 256],
                             value=32, key="nn_b")

    if not st.button("▶  Train", type="primary", key="nn_run", width="stretch"):
        st.info("Set the architecture and press **Train**. Try depth 6 with "
                "`sigmoid` and `zeros` to reproduce the vanishing-gradient "
                "failure of §11.1.", icon="🧠")
        return

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    loader = {
        "Moons": lambda: ds.moons(n=1200, noise=.24),
        "Circles": lambda: ds.circles(n=1200, noise=.12),
        "Blobs (4)": lambda: ds.blobs(n=1200, centers=4),
        "Digits (10 classes)": lambda: ds.digits()[:2],
        "Breast cancer": lambda: ds.breast_cancer()[:2],
    }[dname]()
    X = np.asarray(loader[0], dtype="float32")
    y = np.asarray(loader[1]).astype("int32")
    n_cls = int(y.max()) + 1

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25,
                                          random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype("float32"), sc.transform(Xte).astype("float32")

    tf.random.set_seed(42)
    layers = [keras.layers.Input(shape=(X.shape[1],))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(
            width, activation=None if bn else act,
            kernel_initializer=init,
            kernel_regularizer=keras.regularizers.l2(l2) if l2 > 1e-7 else None))
        if bn:
            layers.append(keras.layers.BatchNormalization())
            layers.append(keras.layers.Activation(act))
        if drop > 0:
            layers.append(keras.layers.Dropout(drop))
    layers.append(keras.layers.Dense(n_cls, activation="softmax"))
    model = keras.Sequential(layers)

    opt = {"Adam": keras.optimizers.Adam(lr),
           "AdamW": keras.optimizers.AdamW(lr),
           "Nadam": keras.optimizers.Nadam(lr),
           "SGD": keras.optimizers.SGD(lr),
           "SGD+momentum": keras.optimizers.SGD(lr, momentum=.9),
           "RMSprop": keras.optimizers.RMSprop(lr),
           "Adagrad": keras.optimizers.Adagrad(lr)}[opt_name]
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    prog = st.progress(0.0, text="Training…")

    class Bar(keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            prog.progress((ep+1)/epochs,
                          text=f"epoch {ep+1}/{epochs} · "
                               f"loss {logs.get('loss', 0):.4f} · "
                               f"val acc {logs.get('val_accuracy', 0):.4f}")

    t0 = time.perf_counter()
    hist = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, verbose=0,
                     validation_data=(Xte, yte), callbacks=[Bar()])
    dt = time.perf_counter() - t0
    prog.empty()

    h = hist.history
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train accuracy", f"{h['accuracy'][-1]:.4f}")
    m2.metric("Test accuracy", f"{h['val_accuracy'][-1]:.4f}",
              f"{h['val_accuracy'][-1] - h['accuracy'][-1]:+.4f}")
    m3.metric("Parameters", f"{model.count_params():,}")
    m4.metric("Training time", f"{dt:.1f} s")

    f = make_subplots(rows=1, cols=2, subplot_titles=("loss", "accuracy"))
    f.add_trace(go.Scatter(y=h["loss"], mode="lines", name="train loss",
                           line=dict(color=C["train"], width=3)), 1, 1)
    f.add_trace(go.Scatter(y=h["val_loss"], mode="lines", name="val loss",
                           line=dict(color=C["valid"], width=3)), 1, 1)
    f.add_trace(go.Scatter(y=h["accuracy"], mode="lines", name="train acc",
                           line=dict(color=C["train"], width=3, dash="dot"),
                           showlegend=False), 1, 2)
    f.add_trace(go.Scatter(y=h["val_accuracy"], mode="lines", name="val acc",
                           line=dict(color=C["valid"], width=3, dash="dot"),
                           showlegend=False), 1, 2)
    f.update_layout(height=420, xaxis_title="epoch", xaxis2_title="epoch")
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    if X.shape[1] == 2:
        pad = .5
        g1 = np.linspace(Xtr[:, 0].min()-pad, Xtr[:, 0].max()+pad, 180)
        g2 = np.linspace(Xtr[:, 1].min()-pad, Xtr[:, 1].max()+pad, 180)
        G1, G2 = np.meshgrid(g1, g2)
        Z = model.predict(np.column_stack([G1.ravel(), G2.ravel()]
                                          ).astype("float32"), verbose=0)
        Z = (Z[:, 1] if n_cls == 2 else Z.argmax(1)).reshape(G1.shape)
        g = go.Figure()
        g.add_contour(x=g1, y=g2, z=Z, colorscale=nav.cscale(), opacity=.65,
                      contours=dict(showlines=False))
        for i in range(n_cls):
            m = ytr == i
            g.add_scatter(x=Xtr[m, 0], y=Xtr[m, 1], mode="markers",
                          name=f"class {i}",
                          marker=dict(size=6,
                                      color=CLASS_COLORS[i % len(CLASS_COLORS)],
                                      line=dict(color="#fff", width=.6)))
        g.update_layout(height=520, title="Learned decision surface",
                        legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(g, width="stretch", config=PLOTLY_CONFIG)

    # ---- gradient health -------------------------------------------------
    with tf.GradientTape() as tape:
        loss = model.compiled_loss if False else keras.losses.sparse_categorical_crossentropy(
            ytr[:256], model(Xtr[:256], training=True))
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_weights)
    norms, labels_g = [], []
    for w, gr in zip(model.trainable_weights, grads):
        if gr is None or len(w.shape) < 2:
            continue
        norms.append(float(tf.norm(gr)))
        labels_g.append(w.path.split("/")[0][:18])
    if norms:
        gg = go.Figure(go.Bar(x=labels_g, y=norms,
                              marker=dict(color=C["primary"])))
        gg.update_layout(height=380, yaxis_type="log",
                         yaxis_title="‖gradient‖",
                         title="Gradient magnitude per layer (§11.1)",
                         xaxis=dict(tickangle=-30))
        st.plotly_chart(gg, width="stretch", config=PLOTLY_CONFIG)
        ratio = max(norms)/max(min(norms), 1e-12)
        if ratio > 1e3:
            st.warning(
                f"The largest layer gradient is **{ratio:.0e}× the smallest**. "
                "That is the vanishing/exploding-gradient signature of §11.1 — "
                "try `he_normal` with `relu`/`elu`, or switch on batch "
                "normalisation.", icon="⚠️")
        else:
            st.success(f"Gradient magnitudes span {ratio:.1f}× across layers — "
                       "healthy.", icon="✅")


# --------------------------------------------------------------------------
# Workbench 6 — free-form scratchpad
# --------------------------------------------------------------------------


SCRATCH_DEFAULT = '''# The scientific stack is already imported:
#   np, pd, px, go, make_subplots, st
#   C, SEQ, CLASS_COLORS, PARULA, palette   (the platform's colours)
#
# Anything you assign to `fig` is rendered. So is a trailing DataFrame.
# The namespace persists between runs, so you can build up state.

import numpy as np
from core import datasets as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = ds.moons(n=800, noise=0.25)[:2]
scores = cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=5)
print(f"random forest: {scores.mean():.4f} +/- {scores.std():.4f}")

# a quick look at the data
fig = go.Figure()
for c in (0, 1):
    m = y == c
    fig.add_scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=f"class {c}",
                    marker=dict(size=7, color=CLASS_COLORS[c]))
fig.update_layout(height=460, title="Moons", xaxis_title="x1", yaxis_title="x2")
'''


def bench_scratchpad():
    section("Bench 6", "Scratchpad · the whole stack, no boilerplate")

    lead(
        "A persistent Python namespace with numpy, pandas, plotly, scikit-learn "
        "and the platform's own <code>core.datasets</code> already loaded. "
        "Assign to <code>fig</code> and it renders; leave a DataFrame at the end "
        "and it tabulates."
    )

    codenote(
        "The namespace persists between runs",
        "Variables you define stay defined, so you can build an analysis up in "
        "steps rather than re-running everything each time. <b>Clear vars</b> "
        "resets it; <b>Restore</b> puts the starting code back.",
    )

    code_lab("Scratchpad", SCRATCH_DEFAULT, key="ailab_scratch",
             height=460, show_editor=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    sub("What is already imported")
    table(
        ["Name", "Is", "Notes"],
        [["<code>np</code>, <code>pd</code>", "numpy, pandas", ""],
         ["<code>px</code>, <code>go</code>, <code>make_subplots</code>",
          "plotly", "Assign to <code>fig</code> to render"],
         ["<code>C</code>, <code>SEQ</code>, <code>CLASS_COLORS</code>",
          "The platform palette",
          "<code>C['primary']</code>, <code>SEQ[i]</code>"],
         ["<code>PARULA</code>, <code>palette</code>",
          "MATLAB Parula and the colour helpers",
          "<code>palette.resolve_colorscale('Parula')</code>"],
         ["<code>st</code>", "streamlit", "For <code>st.dataframe</code> etc."],
         ["<code>from core import datasets as ds</code>",
          "Every dataset in the platform", "See the list below"]],
    )

    sub("Datasets available")
    fns = [n for n in dir(ds) if not n.startswith("_")
           and callable(getattr(ds, n))]
    st.code("from core import datasets as ds\n\n" +
            "\n".join(f"ds.{n}()" for n in sorted(fns)), language="python")


# --------------------------------------------------------------------------
# Landing
# --------------------------------------------------------------------------


def bench_home():
    section("Start here", "What each bench is for")

    table(
        ["Bench", "Question it answers", "Chapters"],
        [["<b>1 · Supervised</b>",
          "What does this model actually do on this data?",
          "2–7, 10–11"],
         ["<b>2 · Sweep</b>",
          "How does one hyperparameter trade bias against variance?",
          "4, 5, 6, 7"],
         ["<b>3 · Arena</b>",
          "Which model should I start with — and what does it cost?",
          "2, 7, 19"],
         ["<b>4 · Unsupervised</b>",
          "Is there structure here, and is it real?", "8, 9"],
         ["<b>5 · Neural playground</b>",
          "What do depth, activation, initialiser and optimiser do?",
          "10, 11"],
         ["<b>6 · Scratchpad</b>", "Anything else.", "All"]],
    )

    idea(
        "Use the benches to break things on purpose",
        "The fastest way to internalise a concept is to reproduce its failure. "
        "Set the neural playground to 6 layers with <code>sigmoid</code> and "
        "<code>zeros</code> initialisation and watch the per-layer gradient "
        "chart collapse (§11.1). Sweep a decision tree's depth to 20 on the "
        "moons and watch the train curve hit 1.0 while validation falls "
        "(§6.4). Run DBSCAN on the blobs with <code>eps = 0.05</code> and get "
        "100 % noise (§9.6). Each of those takes ten seconds here and is worth "
        "more than reading the paragraph.",
    )

    warn(
        "Everything here runs on small datasets, in-process",
        "The benches are for building intuition, not for benchmarking. Timings "
        "are single-threaded-ish and contended with Streamlit itself; "
        "cross-validated scores on a few hundred rows have wide error bars — "
        "which is exactly why the arena draws them. Treat every number as an "
        "order of magnitude, not a measurement.",
    )

    sub("A suggested tour")
    md(
        "1. **Arena** on *Moons* — see that a random forest and an RBF SVM tie, "
        "and that logistic regression cannot.\n"
        "2. **Sweep** the decision tree's `max_depth` on the same data — watch "
        "the exact point where validation turns over.\n"
        "3. **Supervised** with that best depth, then one step past it — compare "
        "the decision surfaces.\n"
        "4. **Unsupervised** on *Moons* with k-means (it fails) and DBSCAN (it "
        "does not) — §9.6's whole argument in two clicks.\n"
        "5. **Neural playground** on *Circles*: one hidden layer of 8 units, "
        "then three layers of 64. Then set the activation to `sigmoid` and the "
        "initialiser to `zeros`."
    )


# --------------------------------------------------------------------------

SECTIONS = [
    ("0", "Start here", bench_home),
    ("1", "Supervised bench", bench_supervised),
    ("2", "Hyperparameter sweep", bench_sweep),
    ("3", "Model arena", bench_arena),
    ("4", "Clustering & projection", bench_unsupervised),
    ("5", "Neural playground", bench_neural),
    ("6", "Scratchpad", bench_scratchpad),
]

nav.render_chapter(CH, SECTIONS, sidebar_title="Workbenches")
