"""Chapter 4 — Training Models."""

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
CH = "ch04"

hero(
    kicker="Part I · Chapter 4",
    title="Training Models",
    blurb=(
        "Open the box. Closed-form least squares and why the pseudoinverse always "
        "works where the normal equation fails; gradient descent in all three "
        "flavours with the convergence rate derived; polynomial features and "
        "learning curves; the complete geometry of ridge, lasso and elastic net; "
        "and logistic and softmax regression from the log-odds up."
    ),
    chips=["Full derivations", "8 sub-sections", "9 animations",
           "10 code labs", "The most mathematical chapter"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_4_1():
    section("4.1", "Linear Regression — The Closed Form")

    lead(
        "A linear model predicts by taking a weighted sum of the features plus a "
        "constant. Two ways to fit it exactly: solve the normal equation, or take "
        "the SVD. They are not equivalent, and the difference matters."
    )

    sub("The model")

    math(r"""
    \hat y \;=\; \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
    """)

    md("In vectorised form, with a bias feature $x_0 \\equiv 1$ prepended:")

    math(r"""
    \hat y \;=\; h_{\boldsymbol\theta}(\mathbf{x}) \;=\;
    \boldsymbol\theta^\top \mathbf{x}
    \;=\; \begin{bmatrix}\theta_0 & \theta_1 & \cdots & \theta_n\end{bmatrix}
          \begin{bmatrix}1 \\ x_1 \\ \vdots \\ x_n\end{bmatrix}
    """)

    where({
        r"n": "the number of features",
        r"\boldsymbol\theta \in \mathbb{R}^{n+1}":
            "the parameter vector, including the bias term $\\theta_0$",
        r"\mathbf{x} \in \mathbb{R}^{n+1}":
            "the feature vector, with $x_0 = 1$",
        r"h_{\boldsymbol\theta}": "the hypothesis function",
    })

    sub("The cost function")

    md("Train = find $\\boldsymbol\\theta$ minimising the MSE (we minimise MSE "
       "rather than RMSE because the minimiser is the same and the algebra is "
       "cleaner — $\\sqrt{\\cdot}$ is monotone increasing):")

    math(r"""
    \mathrm{MSE}(\mathbf{X}, h_{\boldsymbol\theta}) \;=\;
    \frac{1}{m}\sum_{i=1}^{m}
      \Bigl(\boldsymbol\theta^\top \mathbf{x}^{(i)} - y^{(i)}\Bigr)^{2}
    \;=\; \frac{1}{m}\bigl\lVert \mathbf{X}\boldsymbol\theta - \mathbf{y}\bigr\rVert_2^2
    """)

    sub("The Normal Equation")

    md("The closed-form solution:")

    math(r"""
    \boxed{\;\hat{\boldsymbol\theta} \;=\;
    \bigl(\mathbf{X}^\top \mathbf{X}\bigr)^{-1}\,\mathbf{X}^\top\,\mathbf{y}\;}
    """)

    derive(
        [("Write the cost in matrix form (dropping the constant $1/m$, which does "
          "not change the argmin):",
          r"J(\boldsymbol\theta) = \bigl\lVert \mathbf{X}\boldsymbol\theta - "
          r"\mathbf{y}\bigr\rVert_2^2 = (\mathbf{X}\boldsymbol\theta - \mathbf{y})^\top"
          r"(\mathbf{X}\boldsymbol\theta - \mathbf{y})"),
         ("Expand the product, using $(\\mathbf{A}\\mathbf{B})^\\top = "
          "\\mathbf{B}^\\top\\mathbf{A}^\\top$:",
          r"J(\boldsymbol\theta) = \boldsymbol\theta^\top \mathbf{X}^\top \mathbf{X}"
          r"\boldsymbol\theta - \boldsymbol\theta^\top \mathbf{X}^\top \mathbf{y} "
          r"- \mathbf{y}^\top \mathbf{X}\boldsymbol\theta + \mathbf{y}^\top\mathbf{y}"),
         ("The two middle terms are scalars and are transposes of each other, so "
          "they are equal:",
          r"J(\boldsymbol\theta) = \boldsymbol\theta^\top \mathbf{X}^\top \mathbf{X}"
          r"\boldsymbol\theta - 2\,\mathbf{y}^\top \mathbf{X}\boldsymbol\theta "
          r"+ \mathbf{y}^\top\mathbf{y}"),
         ("Differentiate with respect to $\\boldsymbol\\theta$, using "
          "$\\nabla_{\\mathbf{z}}(\\mathbf{z}^\\top \\mathbf{A}\\mathbf{z}) = "
          "2\\mathbf{A}\\mathbf{z}$ for symmetric $\\mathbf{A}$, and "
          "$\\nabla_{\\mathbf{z}}(\\mathbf{b}^\\top\\mathbf{z}) = \\mathbf{b}$:",
          r"\nabla_{\boldsymbol\theta} J = 2\,\mathbf{X}^\top \mathbf{X}"
          r"\boldsymbol\theta - 2\,\mathbf{X}^\top \mathbf{y}"),
         ("Set the gradient to zero — these are the <b>normal equations</b>:",
          r"\mathbf{X}^\top \mathbf{X}\,\hat{\boldsymbol\theta} = \mathbf{X}^\top \mathbf{y}"),
         ("If $\\mathbf{X}^\\top\\mathbf{X}$ is invertible, left-multiply by its "
          "inverse:",
          r"\hat{\boldsymbol\theta} = \bigl(\mathbf{X}^\top \mathbf{X}\bigr)^{-1}"
          r"\mathbf{X}^\top \mathbf{y}"),
         ("It is a <b>minimum</b>, not a saddle, because the Hessian "
          "$\\nabla^2 J = 2\\mathbf{X}^\\top\\mathbf{X}$ is positive semi-definite: "
          "for any $\\mathbf{v}$, $\\mathbf{v}^\\top\\mathbf{X}^\\top\\mathbf{X}"
          "\\mathbf{v} = \\lVert\\mathbf{X}\\mathbf{v}\\rVert^2 \\ge 0$. So $J$ is "
          "convex and any stationary point is a global minimum.", None)],
        title="Deriving the Normal Equation from scratch",
    )

    sub("The geometry: least squares is a projection")

    idea(
        "Least squares projects y onto the column space of X",
        "The normal equations say $\\mathbf{X}^\\top(\\mathbf{y} - \\mathbf{X}"
        "\\hat{\\boldsymbol\\theta}) = \\mathbf{0}$ — the residual is "
        "<b>orthogonal to every column of $\\mathbf{X}$</b>. So "
        "$\\hat{\\mathbf{y}} = \\mathbf{X}\\hat{\\boldsymbol\\theta}$ is the "
        "orthogonal projection of $\\mathbf{y}$ onto $\\mathrm{col}(\\mathbf{X})$, "
        "and $\\mathbf{H} = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}"
        "\\mathbf{X}^\\top$ is the projection ('hat') matrix. That is why the "
        "solution is unique whenever the columns are independent, and why it is "
        "not when they are not.",
    )

    sub("When the Normal Equation breaks — and the SVD fix")

    md(
        "$\\mathbf{X}^\\top\\mathbf{X}$ is singular whenever $m < n$ (fewer rows "
        "than features) or whenever some features are linearly dependent. Then "
        "the inverse does not exist. scikit-learn does **not** use the normal "
        "equation; it computes the **Moore–Penrose pseudoinverse** via SVD:"
    )

    math(r"""
    \hat{\boldsymbol\theta} \;=\; \mathbf{X}^{+}\,\mathbf{y},
    \qquad
    \mathbf{X} \;=\; \mathbf{U}\,\boldsymbol\Sigma\,\mathbf{V}^\top,
    \qquad
    \mathbf{X}^{+} \;=\; \mathbf{V}\,\boldsymbol\Sigma^{+}\,\mathbf{U}^\top
    """)

    where({
        r"\mathbf{U}, \mathbf{V}": "orthogonal matrices of left/right singular vectors",
        r"\boldsymbol\Sigma": "diagonal matrix of singular values "
                              "$\\sigma_1 \\ge \\sigma_2 \\ge \\dots \\ge 0$",
        r"\boldsymbol\Sigma^{+}": "formed by reciprocating every singular value "
                                  "<b>greater than a tiny threshold</b> and leaving "
                                  "the rest at zero",
    })

    proof(
        "Why the pseudoinverse is always defined",
        "Because it never divides by a zero singular value — it simply drops those "
        "directions. When $\\mathbf{X}^\\top\\mathbf{X}$ <i>is</i> invertible, "
        "$\\mathbf{X}^+ = (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top$ and "
        "the two agree exactly. When it is not, the pseudoinverse returns the "
        "<b>minimum-norm</b> solution among the infinitely many that minimise the "
        "residual. It is also far better conditioned: the normal equation squares "
        "the condition number, $\\kappa(\\mathbf{X}^\\top\\mathbf{X}) = "
        "\\kappa(\\mathbf{X})^2$, so it loses roughly twice as many digits of "
        "precision.",
    )

    anim_header("Least squares as an orthogonal projection")
    md(
        "Two features, so $\\mathrm{col}(\\mathbf{X})$ is a plane in "
        "$\\mathbb{R}^3$. The animation orbits the scene: watch the residual "
        "(red) stay perpendicular to the plane from every viewing angle. That "
        "perpendicularity <i>is</i> the normal equation."
    )

    rng = np.random.default_rng(0)
    x1 = np.array([1.0, 0.25, 0.1]); x1 /= np.linalg.norm(x1)
    x2 = np.array([0.15, 1.0, 0.2]); x2 -= (x2 @ x1) * x1; x2 /= np.linalg.norm(x2)
    yv = np.array([0.9, 0.55, 1.35])
    Xc = np.c_[x1, x2]
    th = np.linalg.lstsq(Xc, yv, rcond=None)[0]
    yhat = Xc @ th

    su, sv = np.meshgrid(np.linspace(-.3, 1.5, 12), np.linspace(-.3, 1.5, 12))
    P = su[..., None] * x1 + sv[..., None] * x2

    f3 = go.Figure()
    f3.add_trace(go.Surface(x=P[..., 0], y=P[..., 1], z=P[..., 2],
                            colorscale=[[0, alpha(C["primary"], .25)],
                                        [1, alpha(C["primary"], .25)]],
                            showscale=False, opacity=.42, name="col(X)"))
    for v, c, nm in [(x1, C["accent"], "column x₁"), (x2, C["accent"], "column x₂")]:
        f3.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                                  mode="lines", line=dict(color=c, width=7),
                                  name=nm))
    f3.add_trace(go.Scatter3d(x=[0, yv[0]], y=[0, yv[1]], z=[0, yv[2]],
                              mode="lines+markers", line=dict(color=C["warning"], width=8),
                              marker=dict(size=5), name="y"))
    f3.add_trace(go.Scatter3d(x=[0, yhat[0]], y=[0, yhat[1]], z=[0, yhat[2]],
                              mode="lines+markers", line=dict(color=C["success"], width=8),
                              marker=dict(size=5), name="ŷ = Xθ̂  (projection)"))
    f3.add_trace(go.Scatter3d(x=[yhat[0], yv[0]], y=[yhat[1], yv[1]],
                              z=[yhat[2], yv[2]], mode="lines",
                              line=dict(color=C["danger"], width=8, dash="dash"),
                              name="residual ⟂ plane"))
    f3.update_layout(height=560, title="ŷ is the orthogonal projection of y onto col(X)",
                     scene=dict(xaxis_title="dim 1", yaxis_title="dim 2",
                                zaxis_title="dim 3", aspectmode="cube"))
    anim.rotating_3d(f3, n_frames=44, duration=nav.anim_ms(85))
    figure(f3, f"Check: xᵀ·residual = {x1 @ (yv - yhat):.2e} and "
               f"{x2 @ (yv - yhat):.2e} — zero to machine precision.")

    sub("Computational complexity")

    table(
        ["Method", "Training complexity", "Scales with $n$", "Scales with $m$",
         "Handles $m < n$"],
        [["Normal equation", "$\\mathcal{O}(n^{2.4})$ to $\\mathcal{O}(n^3)$",
          "❌ terrible", "✅ linear", "❌"],
         ["SVD / pseudoinverse", "$\\mathcal{O}(n^2 m)$",
          "❌ poor", "✅ linear", "✅"],
         ["Batch gradient descent", "$\\mathcal{O}(n m)$ per epoch",
          "✅ good", "❌ every epoch touches all data", "✅"],
         ["Stochastic GD", "$\\mathcal{O}(n)$ per step",
          "✅ good", "✅ excellent", "✅"]],
        "Both closed forms are roughly $\\mathcal{O}(n^{2}$–$n^{3})$ in the number "
        "of <b>features</b> — doubling the features multiplies the time by 5 to 8. "
        "Both are linear in the number of <b>instances</b>.",
    )

    tip(
        "The rule of thumb",
        "Under ~10 000 features, use the closed form: it is exact, has no "
        "hyperparameters and needs no scaling. Above that, or when the data does "
        "not fit in memory, use gradient descent. Prediction is "
        "$\\mathcal{O}(n)$ in both cases.",
    )

    code_lab(
        "Normal equation vs pseudoinverse vs scikit-learn",
        '''import numpy as np, time
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
m = 200
X = 2 * rng.random((m, 1))
y = 4 + 3 * X[:, 0] + rng.normal(0, 1, m)          # true theta = [4, 3]

X_b = np.c_[np.ones((m, 1)), X]                    # prepend the bias feature

# ---- 1. the normal equation, exactly as derived --------------------------
theta_ne = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"normal equation      theta = {theta_ne.round(5)}")

# ---- 2. solve the system instead of inverting (numerically better) -------
theta_solve = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
print(f"np.linalg.solve      theta = {theta_solve.round(5)}")

# ---- 3. the pseudoinverse (what sklearn uses) ---------------------------
theta_pinv = np.linalg.pinv(X_b) @ y
print(f"pseudoinverse        theta = {theta_pinv.round(5)}")

# ---- 4. lstsq ------------------------------------------------------------
theta_lstsq, *_ = np.linalg.lstsq(X_b, y, rcond=None)
print(f"np.linalg.lstsq      theta = {theta_lstsq.round(5)}")

# ---- 5. sklearn ----------------------------------------------------------
lin = LinearRegression().fit(X, y)
print(f"sklearn              theta = {np.r_[lin.intercept_, lin.coef_].round(5)}")
print(f"\\ntrue parameters      theta = [4. 3.]")

# ======================================================================
# WHERE THE NORMAL EQUATION BREAKS
# ======================================================================
print("\\n=== case A: a perfectly redundant feature ===")
X_dup = np.c_[X_b, X_b[:, 1]]                      # column 2 == column 1
print(f"rank of X = {np.linalg.matrix_rank(X_dup)} but it has {X_dup.shape[1]} columns")
try:
    np.linalg.inv(X_dup.T @ X_dup)
    print("inverse succeeded (unlikely)")
except np.linalg.LinAlgError as e:
    print(f"normal equation FAILS: {e}")
th = np.linalg.pinv(X_dup) @ y
print(f"pseudoinverse works : {th.round(4)}  "
      f"(splits the weight between the duplicates -- minimum-norm solution)")

print("\\n=== case B: more features than instances (m < n) ===")
Xw = rng.normal(0, 1, (30, 100)); yw = rng.normal(0, 1, 30)
print(f"X is {Xw.shape}, so X^T X is {Xw.shape[1]}x{Xw.shape[1]} with rank "
      f"{np.linalg.matrix_rank(Xw.T @ Xw)}  -> singular")
print(f"pseudoinverse still returns a solution, residual = "
      f"{np.linalg.norm(Xw @ (np.linalg.pinv(Xw) @ yw) - yw):.2e}")

print("\\n=== case C: conditioning ===")
Xi = np.c_[np.ones(m), X[:, 0], X[:, 0] + 1e-8 * rng.normal(0, 1, m)]
print(f"cond(X)     = {np.linalg.cond(Xi):.3e}")
print(f"cond(X^T X) = {np.linalg.cond(Xi.T @ Xi):.3e}   <- SQUARED")
print("The normal equation therefore loses about twice as many digits.")

# ---- timing: how the closed form scales in n ----------------------------
print(f"\\n{'n features':>11}{'normal eq':>12}{'pinv':>11}{'sklearn':>11}")
for n in [10, 50, 200, 800]:
    Xn = rng.normal(0, 1, (3000, n)); yn = rng.normal(0, 1, 3000)
    Xnb = np.c_[np.ones(3000), Xn]
    t0 = time.perf_counter(); np.linalg.solve(Xnb.T @ Xnb, Xnb.T @ yn)
    t1 = time.perf_counter(); np.linalg.pinv(Xnb) @ yn
    t2 = time.perf_counter(); LinearRegression().fit(Xn, yn)
    t3 = time.perf_counter()
    print(f"{n:>11}{(t1-t0)*1e3:>11.2f}ms{(t2-t1)*1e3:>10.2f}ms{(t3-t2)*1e3:>10.2f}ms")
''',
        key="ch04_normal",
    )

    quiz(
        "You have 500 training instances and 5 000 features. What happens to "
        "$(\\mathbf{X}^\\top\\mathbf{X})^{-1}$?",
        ["It is fine, just slow",
         "It does not exist — the matrix is singular",
         "It exists but is negative definite",
         "It equals the identity"],
        1,
        "$\\mathbf{X}^\\top\\mathbf{X}$ is $5001\\times5001$ but has rank at most "
        "500, so it is singular. `LinearRegression` still works because it uses "
        "the SVD pseudoinverse, returning the minimum-norm solution.",
        key="ch04q1",
    )

    keypoints([
        "$\\hat{\\boldsymbol\\theta} = (\\mathbf{X}^\\top\\mathbf{X})^{-1}"
        "\\mathbf{X}^\\top\\mathbf{y}$ — derived by setting $\\nabla J = 0$; the "
        "MSE cost is convex so this is the global minimum.",
        "Geometrically: $\\hat{\\mathbf{y}}$ is the orthogonal projection of "
        "$\\mathbf{y}$ onto $\\mathrm{col}(\\mathbf{X})$; the residual is "
        "perpendicular to every feature.",
        "The normal equation fails when features are dependent or $m<n$; the SVD "
        "<b>pseudoinverse</b> never does.",
        "$\\kappa(\\mathbf{X}^\\top\\mathbf{X}) = \\kappa(\\mathbf{X})^2$ — another "
        "reason to prefer SVD.",
        "Closed forms are $\\mathcal{O}(n^{2}$–$n^{3})$ in features, linear in "
        "instances. Above ~10 000 features, switch to gradient descent.",
    ])


# ==========================================================================
def s_4_2():
    section("4.2", "Gradient Descent")

    lead(
        "A generic optimiser: start anywhere, repeatedly step in the direction of "
        "steepest descent. It underlies essentially all of Part II, so it is worth "
        "understanding exactly — including why the learning rate and feature "
        "scaling are not optional details."
    )

    sub("The update rule")

    math(r"""
    \boldsymbol\theta^{(\text{next})} \;=\;
    \boldsymbol\theta \;-\; \eta \, \nabla_{\boldsymbol\theta}\,
    \mathrm{MSE}(\boldsymbol\theta)
    """)

    md("With the gradient computed over the whole training set:")

    math(r"""
    \nabla_{\boldsymbol\theta}\,\mathrm{MSE}(\boldsymbol\theta) \;=\;
    \begin{bmatrix}
      \dfrac{\partial}{\partial \theta_0}\mathrm{MSE}\\[6pt]
      \dfrac{\partial}{\partial \theta_1}\mathrm{MSE}\\[2pt]
      \vdots\\[2pt]
      \dfrac{\partial}{\partial \theta_n}\mathrm{MSE}
    \end{bmatrix}
    \;=\; \frac{2}{m}\,\mathbf{X}^\top\bigl(\mathbf{X}\boldsymbol\theta - \mathbf{y}\bigr)
    """)

    derive(
        [("Start from the partial derivative with respect to one parameter "
          "$\\theta_j$:",
          r"\frac{\partial}{\partial \theta_j}\,\mathrm{MSE}(\boldsymbol\theta) = "
          r"\frac{\partial}{\partial \theta_j}\,\frac{1}{m}\sum_{i=1}^{m}"
          r"\bigl(\boldsymbol\theta^\top \mathbf{x}^{(i)} - y^{(i)}\bigr)^2"),
         ("Apply the chain rule. The inner derivative of "
          "$\\boldsymbol\\theta^\\top\\mathbf{x}^{(i)}$ with respect to $\\theta_j$ "
          "is just $x_j^{(i)}$:",
          r"= \frac{2}{m}\sum_{i=1}^{m}\bigl(\boldsymbol\theta^\top \mathbf{x}^{(i)} "
          r"- y^{(i)}\bigr)\, x_j^{(i)}"),
         ("Stacking these $n+1$ partials into a vector gives the matrix form — "
          "the sum over $i$ becomes the matrix product $\\mathbf{X}^\\top$:",
          r"\nabla_{\boldsymbol\theta}\,\mathrm{MSE} = \frac{2}{m}\,\mathbf{X}^\top"
          r"\bigl(\mathbf{X}\boldsymbol\theta - \mathbf{y}\bigr)"),
         ("Note that this expression touches every one of the $m$ instances at "
          "every step — hence the name <b>batch</b> gradient descent. It is "
          "$\\mathcal{O}(mn)$ per iteration but scales beautifully in $n$.", None)],
        title="Deriving the MSE gradient",
    )

    sub("The learning rate")

    md("Three regimes, and the boundary between them is exact for quadratics:")

    table(
        ["$\\eta$", "Behaviour", "Symptom in the loss curve"],
        [["Too small", "Converges, but takes far too many iterations",
          "A long, gently decreasing curve that has not flattened"],
         ["Well chosen", "Converges quickly and smoothly",
          "Steep drop, then a flat plateau"],
         ["Too large", "Oscillates, then diverges to infinity or NaN",
          "The loss goes <i>up</i>; you see <code>inf</code> or <code>nan</code>"]],
    )

    proof(
        "The exact stability threshold",
        "For the quadratic cost $J(\\boldsymbol\\theta) = \\frac{1}{m}\\lVert"
        "\\mathbf{X}\\boldsymbol\\theta-\\mathbf{y}\\rVert^2$ with Hessian "
        "$\\mathbf{H} = \\frac{2}{m}\\mathbf{X}^\\top\\mathbf{X}$, gradient descent "
        "converges <b>if and only if</b> $0 < \\eta < 2/\\lambda_{\\max}(\\mathbf{H})$, "
        "where $\\lambda_{\\max}$ is the largest eigenvalue. At exactly "
        "$\\eta = 2/\\lambda_{\\max}$ it oscillates forever without converging or "
        "diverging; above it, it diverges geometrically.",
    )

    derive(
        [("In the eigenbasis of $\\mathbf{H}$ the problem decouples into "
          "independent 1-D problems, one per eigenvalue $\\lambda_k$. Let "
          "$e_k = \\theta_k - \\theta_k^\\star$ be the error along eigendirection "
          "$k$.", None),
         ("The update becomes a simple linear recursion in each direction:",
          r"e_k^{(t+1)} = \bigl(1 - \eta\lambda_k\bigr)\, e_k^{(t)}"),
         ("So after $t$ steps the error is scaled by a power:",
          r"e_k^{(t)} = \bigl(1 - \eta\lambda_k\bigr)^{t}\, e_k^{(0)}"),
         ("This decays to zero iff $|1 - \\eta\\lambda_k| < 1$, i.e. "
          "$0 < \\eta < 2/\\lambda_k$. It must hold for <b>every</b> eigenvalue, so "
          "the binding constraint is the largest:",
          r"0 < \eta < \frac{2}{\lambda_{\max}}"),
         ("The slowest-converging direction is the one with the <i>smallest</i> "
          "eigenvalue. Optimising the rate gives $\\eta^\\star = "
          "2/(\\lambda_{\\max}+\\lambda_{\\min})$ and a convergence factor of",
          r"\rho = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}"
          r" = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = "
          r"\frac{\lambda_{\max}}{\lambda_{\min}}"),
         ("<b>This is why feature scaling matters.</b> $\\kappa$ is the condition "
          "number. Unscaled features make $\\kappa$ enormous, $\\rho \\to 1$, and "
          "convergence crawls. Scaling makes the contours circular, $\\kappa \\to 1$, "
          "$\\rho \\to 0$, and descent goes almost straight to the minimum.", None)],
        title="Why η has a hard ceiling, and why scaling changes everything",
    )

    anim_header("Three learning rates on the same cost surface")

    rng = np.random.default_rng(42)
    m = 100
    Xg = 2 * rng.random((m, 1))
    yg = 4 + 3 * Xg[:, 0] + rng.normal(0, 1, m)
    Xb = np.c_[np.ones(m), Xg[:, 0]]

    t0g = np.linspace(0, 9, 90)
    t1g = np.linspace(-1, 7, 90)
    T0, T1 = np.meshgrid(t0g, t1g)
    Js = np.zeros_like(T0)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            r = Xb @ np.array([T0[i, j], T1[i, j]]) - yg
            Js[i, j] = (r @ r) / m

    def run_gd(eta, steps=48, start=(8.5, 6.2)):
        th = np.array(start, float)
        path = [th.copy()]
        for _ in range(steps):
            g = 2 / m * Xb.T @ (Xb @ th - yg)
            th = th - eta * g
            th = np.clip(th, -60, 60)
            path.append(th.copy())
        return np.array(path)

    etas = [0.02, 0.12, 0.42]
    paths = [run_gd(e) for e in etas]
    names = ["η = 0.02 — too small", "η = 0.12 — just right", "η = 0.42 — too large"]
    cols = [C["info"], C["success"], C["danger"]]

    frames = []
    for k in range(1, 49):
        data = [go.Contour(x=t0g, y=t1g, z=Js, colorscale=nav.cscale(),
                           showscale=False, opacity=.85,
                           contours=dict(showlabels=False, start=0, end=60, size=2.2))]
        ann = []
        for p, c, nm, e in zip(paths, cols, names, etas):
            pp = p[:k + 1]
            data.append(go.Scatter(x=pp[:, 0], y=pp[:, 1], mode="lines+markers",
                                   line=dict(color=c, width=2.6),
                                   marker=dict(size=5)))
            r = Xb @ p[min(k, len(p) - 1)] - yg
            ann.append(f"{nm}: J = {(r @ r) / m:.2f}")
        frames.append(go.Frame(name=str(k), data=data, layout=go.Layout(
            annotations=[anim.annotate_step(
                f"step {k}   |   " + "   |   ".join(ann))])))

    f = go.Figure(data=[go.Contour(x=t0g, y=t1g, z=Js, colorscale=nav.cscale(),
                                   showscale=False, opacity=.85,
                                   contours=dict(showlabels=False, start=0,
                                                 end=60, size=2.2))]
                  + [go.Scatter(x=p[:1, 0], y=p[:1, 1], mode="lines+markers",
                                name=nm, line=dict(color=c, width=2.6),
                                marker=dict(size=5))
                     for p, c, nm in zip(paths, cols, names)])
    f.add_trace(go.Scatter(x=[4], y=[3], mode="markers", name="true optimum",
                           marker=dict(color="#fff", size=15, symbol="star",
                                       line=dict(color=C["ink"], width=2))))
    f.update_layout(height=520, xaxis=dict(range=[0, 9], title="θ₀ (intercept)"),
                    yaxis=dict(range=[-1, 7], title="θ₁ (slope)"),
                    title="Batch gradient descent on the MSE cost surface",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(130), slider_prefix="step ")
    figure(f)

    sub("Batch, Stochastic, and Mini-batch")

    table(
        ["", "Batch GD", "Stochastic GD", "Mini-batch GD"],
        [["Instances per step", "all $m$", "1", "$b$ (typically 32–512)"],
         ["Cost per step", "$\\mathcal{O}(mn)$", "$\\mathcal{O}(n)$",
          "$\\mathcal{O}(bn)$"],
         ["Gradient", "exact", "very noisy, unbiased", "moderately noisy, unbiased"],
         ["Path to the minimum", "smooth, direct", "erratic, bounces around",
          "fairly smooth"],
         ["Escapes local minima?", "❌ never", "✅ yes, the noise helps",
          "✅ somewhat"],
         ["Final result", "converges exactly", "bounces near the optimum forever "
          "unless $\\eta$ decays", "bounces less"],
         ["Hardware", "poor GPU use", "poor GPU use",
          "<b>excellent</b> — matrix ops"],
         ["Out-of-core", "❌", "✅", "✅"]],
    )

    md(
        "SGD never settles: the gradient of a single instance is not the gradient "
        "of the cost. The fix is a **learning schedule** that shrinks $\\eta$ over "
        "time:"
    )

    math(r"""
    \eta(t) \;=\; \frac{\eta_0}{(1 + t/\tau)^{p}}
    \qquad\text{or, as in the classic implementation,}\qquad
    \eta(t) \;=\; \frac{c_1}{t + c_2}
    """)

    md("**Robbins–Monro conditions** — the schedule guarantees convergence iff:")

    math(r"""
    \sum_{t=1}^{\infty} \eta(t) = \infty
    \qquad\text{and}\qquad
    \sum_{t=1}^{\infty} \eta(t)^2 < \infty
    """)
    where({
        r"\sum \eta = \infty": "the steps must be able to travel any distance — "
                               "otherwise you stall before reaching the optimum",
        r"\sum \eta^2 < \infty": "the noise must be damped — otherwise you bounce "
                                 "forever",
    })

    note("$\\eta_t = 1/t$ satisfies both; $\\eta_t = 1/\\sqrt{t}$ satisfies the "
         "first but not the second; a constant $\\eta$ satisfies neither. That is "
         "the theory behind every learning-rate schedule in Chapter 11.")

    anim_header("Batch vs stochastic vs mini-batch, racing on one surface")

    def run_variant(kind, epochs=22, eta0=.1, batch=16, seed=0):
        r = np.random.default_rng(seed)
        th = np.array([8.4, 6.0]); path = [th.copy()]
        t = 0
        for ep in range(epochs):
            idx = r.permutation(m)
            if kind == "batch":
                g = 2 / m * Xb.T @ (Xb @ th - yg)
                th = th - eta0 * g; t += 1; path.append(th.copy())
            elif kind == "sgd":
                for i in idx[:12]:
                    xi, yi = Xb[i:i + 1], yg[i:i + 1]
                    g = 2 * xi.T @ (xi @ th - yi)
                    th = th - (5.0 / (t + 50)) * g; t += 1
                    path.append(th.copy())
            else:
                for s in range(0, m, batch):
                    bidx = idx[s:s + batch]
                    xb_, yb_ = Xb[bidx], yg[bidx]
                    g = 2 / len(bidx) * xb_.T @ (xb_ @ th - yb_)
                    th = th - (2.0 / (t + 22)) * g; t += 1
                    path.append(th.copy())
        return np.array(path)

    pb = run_variant("batch"); ps = run_variant("sgd"); pm = run_variant("mini")
    L = max(len(pb), len(ps), len(pm))

    def at(p, k):
        return p[min(k, len(p) - 1)]

    frames = []
    step_idx = np.unique(np.linspace(1, L, 60).astype(int))
    for k in step_idx:
        data = [go.Contour(x=t0g, y=t1g, z=Js, colorscale=nav.cscale(),
                           showscale=False, opacity=.85,
                           contours=dict(showlabels=False, start=0, end=60, size=2.2))]
        info = []
        for p, c, nm in [(pb, C["primary"], "batch"), (ps, C["danger"], "SGD"),
                         (pm, C["success"], "mini-batch")]:
            kk = min(k, len(p) - 1)
            data.append(go.Scatter(x=p[:kk + 1, 0], y=p[:kk + 1, 1],
                                   mode="lines", line=dict(color=c, width=2.4)))
            r = Xb @ at(p, kk) - yg
            info.append(f"{nm} J={(r @ r) / m:.3f}")
        frames.append(go.Frame(name=str(k), data=data, layout=go.Layout(
            annotations=[anim.annotate_step(
                f"update {k}   |   " + "   |   ".join(info))])))

    f = go.Figure(data=[go.Contour(x=t0g, y=t1g, z=Js, colorscale=nav.cscale(),
                                   showscale=False, opacity=.85,
                                   contours=dict(showlabels=False, start=0,
                                                 end=60, size=2.2))]
                  + [go.Scatter(x=p[:1, 0], y=p[:1, 1], mode="lines", name=nm,
                                line=dict(color=c, width=2.4))
                     for p, c, nm in [(pb, C["primary"], "batch GD"),
                                      (ps, C["danger"], "stochastic GD"),
                                      (pm, C["success"], "mini-batch GD")]])
    f.add_trace(go.Scatter(x=[4], y=[3], mode="markers", name="optimum",
                           marker=dict(color="#fff", size=15, symbol="star",
                                       line=dict(color=C["ink"], width=2))))
    f.update_layout(height=520, xaxis=dict(range=[0, 9], title="θ₀"),
                    yaxis=dict(range=[-1, 7], title="θ₁"),
                    title="Three flavours of gradient descent",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="update ")
    figure(f, "Batch glides in a smooth arc. SGD zig-zags but covers ground fast. "
              "Mini-batch is the practical compromise, and the only one that maps "
              "onto GPU matrix operations.")

    anim_header("Feature scaling and the condition number")
    md(
        "The identical algorithm on the identical data — the only difference is "
        "whether the features were standardised. Left: an elongated valley, "
        "$\\kappa$ large, descent bounces across the valley and creeps along it. "
        "Right: circular contours, $\\kappa \\approx 1$, descent goes almost "
        "straight in."
    )

    rng2 = np.random.default_rng(3)
    m2 = 200
    f1 = rng2.normal(0, 1, m2)
    f2 = rng2.normal(0, 55, m2)                       # 55x the scale
    yy = 3 * f1 + 0.06 * f2 + rng2.normal(0, .4, m2)
    Xu = np.c_[f1, f2]
    Xs = (Xu - Xu.mean(0)) / Xu.std(0)

    def surface_and_path(Xd, eta, start, steps=44):
        H = 2 / len(Xd) * Xd.T @ Xd
        lmax = np.linalg.eigvalsh(H).max()
        eta = min(eta, 1.6 / lmax)
        th = np.array(start, float); path = [th.copy()]
        for _ in range(steps):
            g = 2 / len(Xd) * Xd.T @ (Xd @ th - yy)
            th = th - eta * g
            path.append(th.copy())
        return np.array(path), np.linalg.cond(Xd.T @ Xd)

    pu, ku = surface_and_path(Xu, .0009, (6.0, 4.0))
    psd, ks = surface_and_path(Xs, .18, (6.0, 4.0))

    def grid_J(Xd, a, b):
        A, B = np.meshgrid(a, b)
        Z = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                r = Xd @ np.array([A[i, j], B[i, j]]) - yy
                Z[i, j] = (r @ r) / len(Xd)
        return A, B, Z

    au = np.linspace(-2, 7, 70); bu = np.linspace(-1.2, 4.5, 70)
    _, _, Zu = grid_J(Xu, au, bu)
    asd = np.linspace(-2, 7, 70); bsd = np.linspace(-1.2, 4.5, 70)
    _, _, Zs = grid_J(Xs, asd, bsd)

    fr = []
    for k in range(1, 45):
        fr.append(go.Frame(name=str(k), data=[
            go.Contour(x=au, y=bu, z=Zu, colorscale=nav.cscale(), showscale=False,
                       opacity=.85, ncontours=28),
            go.Scatter(x=pu[:k + 1, 0], y=pu[:k + 1, 1], mode="lines+markers",
                       line=dict(color=C["danger"], width=2.6), marker=dict(size=4)),
            go.Contour(x=asd, y=bsd, z=Zs, colorscale=nav.cscale(), showscale=False,
                       opacity=.85, ncontours=28),
            go.Scatter(x=psd[:k + 1, 0], y=psd[:k + 1, 1], mode="lines+markers",
                       line=dict(color=C["success"], width=2.6), marker=dict(size=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(f"step {k}")])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=(f"UNSCALED — κ(XᵀX) = {ku:.1e}",
                                      f"SCALED — κ(XᵀX) = {ks:.1f}"))
    f.add_trace(go.Contour(x=au, y=bu, z=Zu, colorscale=nav.cscale(),
                           showscale=False, opacity=.85, ncontours=28), 1, 1)
    f.add_trace(go.Scatter(x=pu[:1, 0], y=pu[:1, 1], mode="lines+markers",
                           name="unscaled path",
                           line=dict(color=C["danger"], width=2.6),
                           marker=dict(size=4)), 1, 1)
    f.add_trace(go.Contour(x=asd, y=bsd, z=Zs, colorscale=nav.cscale(),
                           showscale=False, opacity=.85, ncontours=28), 1, 2)
    f.add_trace(go.Scatter(x=psd[:1, 0], y=psd[:1, 1], mode="lines+markers",
                           name="scaled path",
                           line=dict(color=C["success"], width=2.6),
                           marker=dict(size=4)), 1, 2)
    f.update_layout(height=470, title="Same algorithm, same data, one scaler")
    f.update_xaxes(title_text="θ₁"); f.update_yaxes(title_text="θ₂")
    anim.animate(f, fr, duration=nav.anim_ms(110), slider_prefix="step ")
    figure(f)

    pitfall(
        "Always scale before gradient descent",
        "<code>StandardScaler</code> before <code>SGDRegressor</code>, before any "
        "neural network, before any SVM. It does not change the model class or the "
        "optimum — it changes the <i>shape of the path to it</i>, and that is the "
        "difference between 30 iterations and 30 000.",
    )

    code_lab(
        "Implement all three from scratch and race them",
        '''import numpy as np, time
import plotly.graph_objects as go

rng = np.random.default_rng(42)
m, n = 2000, 20
X = rng.normal(0, 1, (m, n))
theta_true = rng.normal(0, 2, n + 1)
y = np.c_[np.ones(m), X] @ theta_true + rng.normal(0, .5, m)
X_b = np.c_[np.ones(m), X]

def mse(th):
    r = X_b @ th - y
    return float(r @ r / m)

theta_star = np.linalg.lstsq(X_b, y, rcond=None)[0]
J_star = mse(theta_star)
print(f"optimal MSE (closed form) = {J_star:.6f}\\n")

# ---------------- BATCH GRADIENT DESCENT ---------------------------------
def batch_gd(eta=.1, n_epochs=200):
    th = np.zeros(n + 1); hist = []
    for _ in range(n_epochs):
        g = 2 / m * X_b.T @ (X_b @ th - y)
        th -= eta * g
        hist.append(mse(th))
    return th, hist

# ---------------- STOCHASTIC GRADIENT DESCENT ----------------------------
def sgd(n_epochs=50, t0=5, t1=50):
    def schedule(t): return t0 / (t + t1)
    th = np.zeros(n + 1); hist = []
    r = np.random.default_rng(0)
    for ep in range(n_epochs):
        for i in range(m):
            j = r.integers(m)
            xi, yi = X_b[j:j+1], y[j:j+1]
            g = 2 * xi.T @ (xi @ th - yi)
            th -= schedule(ep * m + i) * g
        hist.append(mse(th))
    return th, hist

# ---------------- MINI-BATCH GRADIENT DESCENT ----------------------------
def minibatch_gd(n_epochs=60, bs=32, t0=20, t1=200):
    def schedule(t): return t0 / (t + t1)
    th = np.zeros(n + 1); hist = []
    r = np.random.default_rng(0); t = 0
    for ep in range(n_epochs):
        idx = r.permutation(m)
        for s in range(0, m, bs):
            b = idx[s:s+bs]
            g = 2 / len(b) * X_b[b].T @ (X_b[b] @ th - y[b])
            th -= schedule(t) * g; t += 1
        hist.append(mse(th))
    return th, hist

results = {}
for name, fn in [("batch (200 ep)", batch_gd), ("stochastic (50 ep)", sgd),
                 ("mini-batch (60 ep)", minibatch_gd)]:
    t0 = time.perf_counter(); th, hist = fn(); dt = time.perf_counter() - t0
    results[name] = hist
    print(f"{name:<22} time {dt:>6.3f}s   final MSE {hist[-1]:.6f}   "
          f"excess {hist[-1]-J_star:+.2e}   ||θ-θ*|| {np.linalg.norm(th-theta_star):.4f}")

# ---------------- the exact stability threshold --------------------------
H = 2 / m * X_b.T @ X_b
lmax, lmin = np.linalg.eigvalsh(H)[[-1, 0]]
print(f"\\nlambda_max(H) = {lmax:.4f}   lambda_min(H) = {lmin:.4f}")
print(f"condition number kappa = {lmax/lmin:.2f}")
print(f"theory says GD converges iff 0 < eta < 2/lambda_max = {2/lmax:.4f}")
print(f"optimal eta = 2/(lmax+lmin) = {2/(lmax+lmin):.4f}, "
      f"rate rho = {(lmax-lmin)/(lmax+lmin):.4f}")
print(f"\\n{'eta':>8}  {'MSE after 200 epochs':>22}")
for eta in [0.05, 0.2, 2/lmax * 0.9, 2/lmax * 1.01, 2/lmax * 1.2]:
    _, h = batch_gd(eta=eta, n_epochs=200)
    v = h[-1]
    tag = "DIVERGED" if (not np.isfinite(v) or v > 1e6) else f"{v:.6f}"
    print(f"{eta:>8.4f}  {tag:>22}")

fig = go.Figure()
for i, (name, hist) in enumerate(results.items()):
    fig.add_scatter(y=np.array(hist) - J_star, mode="lines", name=name,
                    line=dict(color=SEQ[i], width=3))
fig.update_layout(height=400, yaxis_type="log", xaxis_title="epoch",
                  yaxis_title="excess MSE above the optimum",
                  title="Convergence of the three variants")
''',
        key="ch04_gd",
    )

    keypoints([
        "$\\nabla \\mathrm{MSE} = \\frac{2}{m}\\mathbf{X}^\\top(\\mathbf{X}"
        "\\boldsymbol\\theta - \\mathbf{y})$ — touches all $m$ rows, hence "
        "<i>batch</i>.",
        "GD converges iff $0 < \\eta < 2/\\lambda_{\\max}$; the rate is "
        "$\\rho = (\\kappa-1)/(\\kappa+1)$.",
        "<b>Feature scaling shrinks $\\kappa$</b> and is therefore not cosmetic — "
        "it is the difference between converging and crawling.",
        "SGD is fast and escapes local minima but never settles without a "
        "decaying schedule (Robbins–Monro).",
        "Mini-batch is the default in deep learning: it is the only variant that "
        "uses matrix hardware well.",
    ])


# ==========================================================================
def s_4_3():
    section("4.3", "Polynomial Regression")

    lead(
        "A linear model can fit non-linear data — you just add powers of the "
        "features as new features. The model stays linear <i>in the parameters</i>, "
        "which is all the algebra of §4.1 ever required."
    )

    math(r"""
    \hat y \;=\; \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d
    \;=\; \boldsymbol\theta^\top \phi(\mathbf{x}),
    \qquad
    \phi(x) = \bigl(1, x, x^2, \dots, x^d\bigr)
    """)

    idea(
        "Linear in the parameters, not in the inputs",
        "\"Linear model\" means $\\hat y$ is a linear function of "
        "$\\boldsymbol\\theta$. It says nothing about $\\mathbf{x}$. Once you accept "
        "that, the normal equation, gradient descent, ridge, lasso — everything "
        "in this chapter — applies unchanged to any <b>basis expansion</b> "
        "$\\phi$: polynomials, splines, radial basis functions, Fourier features. "
        "Chapter 5's kernel trick is the same idea taken to an infinite-dimensional "
        "$\\phi$.",
    )

    sub("The combinatorial explosion")

    md(
        "`PolynomialFeatures(degree=d)` on $n$ input features produces **all** "
        "monomials up to degree $d$, including cross-terms. The count is:"
    )

    math(r"""
    N(n, d) \;=\; \binom{n + d}{d} \;=\; \frac{(n+d)!}{d!\;n!}
    """)

    tbl = []
    from math import comb
    for n_ in [2, 5, 10, 50, 100]:
        tbl.append([str(n_)] + [f"{comb(n_ + d_, d_):,}" for d_ in [2, 3, 5]])
    table(["$n$ features", "degree 2", "degree 3", "degree 5"], tbl,
          "Number of features after expansion (including the bias term). "
          "100 features at degree 3 gives 176 851 columns.")

    warn(
        "Cross-terms are the point, and the danger",
        "Adding $x_1^2$ and $x_2^2$ by hand is easy; the value of "
        "<code>PolynomialFeatures</code> is that it also adds $x_1 x_2$, which is "
        "how the model learns <b>interactions</b> between features. That is why it "
        "works — and why the column count explodes. Use "
        "<code>interaction_only=True</code> to keep the cross-terms and drop the "
        "pure powers.",
    )

    anim_header("Degree sweep: from underfitting to memorisation")

    rng = np.random.default_rng(11)
    mp = 60
    Xp = 6 * rng.random(mp) - 3
    yp = 0.5 * Xp ** 2 + Xp + 2 + rng.normal(0, 1.0, mp)
    gp = np.linspace(-3.4, 3.4, 400)
    truth = 0.5 * gp ** 2 + gp + 2

    degs = list(range(1, 26))
    curves, trmse, cvrmse, norms = [], [], [], []
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LinearRegression
    for d in degs:
        pipe = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                             LinearRegression())
        pipe.fit(Xp.reshape(-1, 1), yp)
        curves.append(np.clip(pipe.predict(gp.reshape(-1, 1)), -12, 22))
        trmse.append(float(np.sqrt(np.mean(
            (pipe.predict(Xp.reshape(-1, 1)) - yp) ** 2))))
        cvrmse.append(float(-cross_val_score(
            pipe, Xp.reshape(-1, 1), yp, cv=5,
            scoring="neg_root_mean_squared_error").mean()))
        norms.append(float(np.linalg.norm(pipe[-1].coef_)))

    frames = []
    for i, d in enumerate(degs):
        tag = ("UNDERFIT" if d == 1 else "GOOD" if d <= 4 else "OVERFIT")
        col = C["warning"] if d == 1 else (C["success"] if d <= 4 else C["danger"])
        frames.append(go.Frame(name=str(d), data=[
            go.Scatter(x=Xp, y=yp, mode="markers",
                       marker=dict(color=C["train"], size=8,
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=gp, y=curves[i], mode="lines",
                       line=dict(color=col, width=3.4)),
            go.Scatter(x=gp, y=truth, mode="lines",
                       line=dict(color=C["truth"], width=2, dash="dot")),
            go.Scatter(x=degs, y=trmse, mode="lines+markers",
                       line=dict(color=C["train"], width=2.6)),
            go.Scatter(x=degs, y=cvrmse, mode="lines+markers",
                       line=dict(color=C["test"], width=2.6)),
            go.Scatter(x=[d], y=[cvrmse[i]], mode="markers",
                       marker=dict(color=col, size=15, symbol="circle-open",
                                   line=dict(width=3))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"degree {d:>2} — {tag}   ·   {comb(1+d, d)} features   ·   "
            f"train RMSE {trmse[i]:.3f}   CV RMSE {cvrmse[i]:.3f}   "
            f"‖θ‖ = {norms[i]:.1f}", color=col)])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("the fit", "train vs cross-validated RMSE"))
    f.add_trace(go.Scatter(x=Xp, y=yp, mode="markers", name="data",
                           marker=dict(color=C["train"], size=8,
                                       line=dict(color="#fff", width=1))), 1, 1)
    f.add_trace(go.Scatter(x=gp, y=curves[0], mode="lines", name="model",
                           line=dict(color=C["warning"], width=3.4)), 1, 1)
    f.add_trace(go.Scatter(x=gp, y=truth, mode="lines", name="true quadratic",
                           line=dict(color=C["truth"], width=2, dash="dot")), 1, 1)
    f.add_trace(go.Scatter(x=degs, y=trmse, mode="lines+markers", name="train RMSE",
                           line=dict(color=C["train"], width=2.6)), 1, 2)
    f.add_trace(go.Scatter(x=degs, y=cvrmse, mode="lines+markers", name="CV RMSE",
                           line=dict(color=C["test"], width=2.6)), 1, 2)
    f.add_trace(go.Scatter(x=[1], y=[cvrmse[0]], mode="markers", showlegend=False,
                           marker=dict(color=C["warning"], size=15,
                                       symbol="circle-open", line=dict(width=3))), 1, 2)
    f.update_yaxes(range=[-8, 20], row=1, col=1)
    f.update_xaxes(range=[-3.4, 3.4], title_text="x", row=1, col=1)
    f.update_yaxes(type="log", title_text="RMSE", row=1, col=2)
    f.update_xaxes(title_text="polynomial degree", row=1, col=2)
    f.update_layout(height=460)
    anim.animate(f, frames, duration=nav.anim_ms(330), slider_prefix="degree ")
    figure(f, "Watch ‖θ‖ in the status line explode as the degree grows — that "
              "growth is exactly what §4.5's regularisation penalises.")

    code_lab(
        "Polynomial features, interactions, and the explosion",
        '''import numpy as np
from math import comb
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# ---- what PolynomialFeatures actually produces ---------------------------
X = np.array([[2., 3.]])
for d in [2, 3]:
    pf = PolynomialFeatures(degree=d, include_bias=True)
    out = pf.fit_transform(X)
    print(f"degree {d}: {out.shape[1]} features")
    for name, val in zip(pf.get_feature_names_out(["a", "b"]), out[0]):
        print(f"    {name:<10} = {val:g}")

print("\\ninteraction_only=True keeps cross-terms, drops pure powers:")
pf = PolynomialFeatures(degree=2, interaction_only=True)
print("   ", list(pf.fit(X).get_feature_names_out(["a", "b"])))

# ---- the combinatorial explosion ----------------------------------------
print(f"\\n{'n':>5}{'d=2':>10}{'d=3':>12}{'d=5':>14}{'d=10':>16}")
for n in [2, 5, 10, 20, 50, 100]:
    print(f"{n:>5}{comb(n+2,2):>10,}{comb(n+3,3):>12,}"
          f"{comb(n+5,5):>14,}{comb(n+10,10):>16,}")

# ---- fitting a quadratic with a linear model ----------------------------
rng = np.random.default_rng(42)
m = 100
x = 6 * rng.random(m) - 3
y = 0.5 * x**2 + x + 2 + rng.normal(0, 1, m)

print(f"\\n{'degree':>7}{'#feat':>7}{'train RMSE':>13}{'CV RMSE':>11}{'||theta||':>12}")
for d in [1, 2, 3, 5, 10, 20, 30]:
    pipe = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    pipe.fit(x.reshape(-1, 1), y)
    tr = np.sqrt(np.mean((pipe.predict(x.reshape(-1, 1)) - y)**2))
    cv = -cross_val_score(pipe, x.reshape(-1, 1), y, cv=5,
                          scoring="neg_root_mean_squared_error").mean()
    print(f"{d:>7}{comb(1+d, d):>7}{tr:>13.4f}{cv:>11.4f}"
          f"{np.linalg.norm(pipe[-1].coef_):>12.2f}")

print("\\ndegree 2 recovers the truth. Look at ||theta|| for degree 30:")
print("that is the signature of overfitting, and it is what ridge/lasso attack.")

pipe2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipe2.fit(x.reshape(-1, 1), y)
print(f"\\nrecovered coefficients (degree 2, unscaled): "
      f"intercept {pipe2[-1].intercept_:.3f}, coefs {pipe2[-1].coef_.round(3)}")
print("true model was: y = 2 + 1.0*x + 0.5*x^2")
''',
        key="ch04_poly",
    )

    keypoints([
        "Add powers and cross-products as features; the model stays <b>linear in "
        "$\\boldsymbol\\theta$</b>, so all the machinery still applies.",
        "<code>PolynomialFeatures</code> generates all $\\binom{n+d}{d}$ monomials "
        "— including the interaction terms that give it its power.",
        "The feature count explodes combinatorially; degree 3 on 100 features is "
        "176 851 columns.",
        "As the degree grows, $\\lVert\\boldsymbol\\theta\\rVert$ explodes. That is "
        "the handle regularisation grabs.",
        "Any basis expansion $\\phi$ works — polynomials, splines, RBFs. The kernel "
        "trick (Ch. 5) is the limiting case.",
    ])


# ==========================================================================
def s_4_4():
    section("4.4", "Learning Curves")

    lead(
        "The diagnostic that tells you <i>which</i> problem you have. Plot training "
        "and validation error against training-set size, and read the answer off "
        "the shape."
    )

    sub("How to read them")

    table(
        ["Shape", "Diagnosis", "What to do", "What NOT to do"],
        [["Both curves plateau <b>high</b>, close together",
          "<b>Underfitting</b> (high bias)",
          "More capacity: higher degree, more features, a richer model family; "
          "reduce regularisation",
          "Collect more data — it will not help at all"],
         ["A large persistent <b>gap</b>, train error very low",
          "<b>Overfitting</b> (high variance)",
          "More data; regularisation; simplify the model; early stopping",
          "Add capacity"],
         ["Both low and converging",
          "Good fit", "Ship it", "—"],
         ["Validation error <i>increases</i> with more data",
          "A bug: leakage, or a distribution mismatch between the sets",
          "Check the split, check for duplicated rows across train/val", "—"]],
    )

    idea(
        "The single most useful sentence about learning curves",
        "<b>If the two curves have converged, more data cannot help.</b> The gap "
        "is what more data closes; the plateau height is what more capacity "
        "lowers. This one picture saves months of collecting data that will not "
        "move the number.",
    )

    sub("The bias–variance decomposition")

    md("For squared loss, the expected generalisation error at a point decomposes "
       "exactly into three terms:")

    math(r"""
    \mathbb{E}\Bigl[\bigl(y - \hat h(\mathbf{x})\bigr)^2\Bigr] \;=\;
    \underbrace{\Bigl(\mathbb{E}[\hat h(\mathbf{x})] - f(\mathbf{x})\Bigr)^2}_{\text{Bias}^2}
    \;+\;
    \underbrace{\mathbb{E}\Bigl[\bigl(\hat h(\mathbf{x}) - \mathbb{E}[\hat h(\mathbf{x})]\bigr)^2\Bigr]}_{\text{Variance}}
    \;+\;
    \underbrace{\sigma^2}_{\text{Irreducible}}
    """)

    derive(
        [("Let $y = f(\\mathbf{x}) + \\varepsilon$ with $\\mathbb{E}[\\varepsilon]=0$, "
          "$\\mathrm{Var}(\\varepsilon) = \\sigma^2$, and let $\\hat h$ be the model "
          "fitted on a random training set. Write $\\bar h = \\mathbb{E}[\\hat h]$, "
          "the average model over training sets.", None),
         ("Insert $\\bar h$ into the error and expand:",
          r"\mathbb{E}\bigl[(y-\hat h)^2\bigr] = "
          r"\mathbb{E}\Bigl[\bigl((f - \bar h) + (\bar h - \hat h) + \varepsilon\bigr)^2\Bigr]"),
         ("Expand the square into three squares and three cross terms:",
          r"= (f-\bar h)^2 + \mathbb{E}\bigl[(\bar h - \hat h)^2\bigr] + \sigma^2 "
          r"+ 2\,\text{(cross terms)}"),
         ("Every cross term vanishes. $\\varepsilon$ is independent of the training "
          "set so $\\mathbb{E}[\\varepsilon \\cdot (\\cdot)] = 0$; and "
          "$\\mathbb{E}[\\bar h - \\hat h] = 0$ by the definition of $\\bar h$, while "
          "$(f - \\bar h)$ is a constant with respect to the training-set "
          "randomness.", None),
         ("Leaving exactly:",
          r"\mathbb{E}\bigl[(y-\hat h)^2\bigr] = \underbrace{(f-\bar h)^2}_{\text{Bias}^2} "
          r"+ \underbrace{\mathbb{E}\bigl[(\hat h - \bar h)^2\bigr]}_{\text{Variance}} "
          r"+ \underbrace{\sigma^2}_{\text{noise}}"),
         ("<b>Bias</b> is error from wrong assumptions — a linear model on "
          "quadratic data. <b>Variance</b> is sensitivity to which training set you "
          "happened to draw. <b>Irreducible error</b> is data noise; the only cure "
          "is better data. Increasing model complexity lowers bias and raises "
          "variance: that is the trade-off.", None)],
        title="Deriving the bias–variance decomposition",
    )

    anim_header("Learning curves growing, for three model capacities")
    md(
        "Three models — degree 1 (high bias), degree 2 (just right), degree 25 "
        "(high variance) — on the same growing training set. Read each panel with "
        "the table above."
    )

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(5)
    M = 220
    Xa = 6 * rng.random(M) - 3
    ya = 0.5 * Xa ** 2 + Xa + 2 + rng.normal(0, 1.2, M)
    Xtr_, Xva_ = Xa[:160], Xa[160:]
    ytr_, yva_ = ya[:160], ya[160:]
    sizes = list(range(4, 161, 4))

    curves = {}
    for d in (1, 2, 25):
        tr_e, va_e = [], []
        for k in sizes:
            p = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                              LinearRegression())
            p.fit(Xtr_[:k].reshape(-1, 1), ytr_[:k])
            tr_e.append(np.sqrt(np.mean(
                (p.predict(Xtr_[:k].reshape(-1, 1)) - ytr_[:k]) ** 2)))
            va_e.append(min(np.sqrt(np.mean(
                (p.predict(Xva_.reshape(-1, 1)) - yva_) ** 2)), 14))
        curves[d] = (tr_e, va_e)

    titles = ["degree 1 — HIGH BIAS (underfit)",
              "degree 2 — GOOD FIT",
              "degree 25 — HIGH VARIANCE (overfit)"]
    frames = []
    for k in range(2, len(sizes) + 1):
        data, info = [], []
        for d in (1, 2, 25):
            tr_e, va_e = curves[d]
            data.append(go.Scatter(x=sizes[:k], y=tr_e[:k], mode="lines",
                                   line=dict(color=C["train"], width=3)))
            data.append(go.Scatter(x=sizes[:k], y=va_e[:k], mode="lines",
                                   line=dict(color=C["test"], width=3)))
            info.append(f"d{d}: gap={va_e[k-1]-tr_e[k-1]:.2f}")
        frames.append(go.Frame(name=str(sizes[k - 1]), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"m = {sizes[k-1]} training instances   |   "
                                   + "   |   ".join(info))])))

    f = make_subplots(rows=1, cols=3, subplot_titles=titles,
                      shared_yaxes=True)
    for c, d in enumerate((1, 2, 25), start=1):
        tr_e, va_e = curves[d]
        f.add_trace(go.Scatter(x=sizes[:2], y=tr_e[:2], mode="lines",
                               name="train RMSE" if c == 1 else None,
                               showlegend=(c == 1),
                               line=dict(color=C["train"], width=3)), 1, c)
        f.add_trace(go.Scatter(x=sizes[:2], y=va_e[:2], mode="lines",
                               name="validation RMSE" if c == 1 else None,
                               showlegend=(c == 1),
                               line=dict(color=C["test"], width=3)), 1, c)
    f.update_yaxes(range=[0, 8], title_text="RMSE", row=1, col=1)
    f.update_xaxes(title_text="training-set size m")
    f.update_layout(height=430, title="Learning curves: read the gap and the plateau",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="m = ")
    figure(f, "Left: both curves flat and high, no gap → more data is useless. "
              "Right: a huge persistent gap → more data will help.")

    anim_header("Bias and variance, made visible")
    md(
        "Twenty models of each degree, each fitted on a different random sample "
        "from the same distribution. **Spread between the grey curves = variance. "
        "Distance from the average (thick) curve to the truth (dashed) = bias.**"
    )

    grid = np.linspace(-3, 3, 200)
    truth_g = 0.5 * grid ** 2 + grid + 2
    deg_list = [1, 2, 12]
    fits = {}
    for d in deg_list:
        fs = []
        for r in range(20):
            rr = np.random.default_rng(100 + r)
            xs = 6 * rr.random(28) - 3
            ys = 0.5 * xs ** 2 + xs + 2 + rr.normal(0, 1.2, 28)
            p = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                              LinearRegression())
            p.fit(xs.reshape(-1, 1), ys)
            fs.append(np.clip(p.predict(grid.reshape(-1, 1)), -10, 18))
        fits[d] = np.array(fs)

    frames = []
    for i, d in enumerate(deg_list):
        F = fits[d]
        mean_f = F.mean(0)
        bias2 = float(np.mean((mean_f - truth_g) ** 2))
        var = float(np.mean(F.var(0)))
        data = [go.Scatter(x=grid, y=F[j], mode="lines", showlegend=False,
                           line=dict(color=alpha(C["muted"], .45), width=1))
                for j in range(20)]
        data.append(go.Scatter(x=grid, y=mean_f, mode="lines",
                               line=dict(color=C["primary"], width=4)))
        data.append(go.Scatter(x=grid, y=truth_g, mode="lines",
                               line=dict(color=C["truth"], width=3, dash="dash")))
        frames.append(go.Frame(name=str(d), data=data, layout=go.Layout(
            title=f"degree {d}:  Bias² = {bias2:.3f}   Variance = {var:.3f}   "
                  f"Bias²+Var = {bias2+var:.3f}")))

    F0 = fits[1]
    f = go.Figure(data=[go.Scatter(x=grid, y=F0[j], mode="lines", showlegend=False,
                                   line=dict(color=alpha(C["muted"], .45), width=1))
                        for j in range(20)]
                  + [go.Scatter(x=grid, y=F0.mean(0), mode="lines",
                                name="average model E[ĥ]",
                                line=dict(color=C["primary"], width=4)),
                     go.Scatter(x=grid, y=truth_g, mode="lines", name="truth f",
                                line=dict(color=C["truth"], width=3, dash="dash"))])
    f.update_layout(height=470, yaxis=dict(range=[-4, 14]),
                    xaxis=dict(range=[-3, 3], title="x"), yaxis_title="y",
                    title="degree 1")
    anim.animate(f, frames, duration=nav.anim_ms(2000), slider_prefix="degree ")
    figure(f, "Degree 1: all 20 curves agree (low variance) but all are wrong "
              "(high bias). Degree 12: the average is right (low bias) but the "
              "individual curves are wild (high variance).")

    code_lab(
        "Learning curves and a Monte-Carlo bias–variance decomposition",
        '''import numpy as np
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(42)
SIGMA = 1.2
f_true = lambda x: 0.5 * x**2 + x + 2

m = 300
X = (6 * rng.random(m) - 3).reshape(-1, 1)
y = f_true(X.ravel()) + rng.normal(0, SIGMA, m)

# ---------------- learning curves ----------------------------------------
print(f"{'model':<12}{'final train':>13}{'final val':>11}{'gap':>9}   diagnosis")
for d in [1, 2, 5, 25]:
    pipe = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    sizes, tr, va = learning_curve(
        pipe, X, y, train_sizes=np.linspace(.05, 1., 20),
        cv=ShuffleSplit(20, test_size=.25, random_state=0),
        scoring="neg_root_mean_squared_error")
    tr, va = -tr.mean(1), -va.mean(1)
    gap = va[-1] - tr[-1]
    diag = ("UNDERFIT (bias)" if tr[-1] > 1.6 * SIGMA and gap < .5
            else "OVERFIT (variance)" if gap > .8 else "good fit")
    print(f"degree {d:<5}{tr[-1]:>13.3f}{va[-1]:>11.3f}{gap:>9.3f}   {diag}")

# ---------------- explicit bias-variance decomposition -------------------
print("\\n=== Monte-Carlo bias-variance decomposition ===")
print("(200 independent training sets of 40 points each)")
x_test = np.linspace(-3, 3, 120).reshape(-1, 1)
f_test = f_true(x_test.ravel())

print(f"\\n{'degree':>7}{'Bias^2':>10}{'Variance':>11}{'noise':>9}"
      f"{'total':>10}{'actual MSE':>13}")
for d in [1, 2, 3, 5, 10, 20]:
    preds = np.zeros((200, len(x_test)))
    for r in range(200):
        rr = np.random.default_rng(1000 + r)
        xs = (6 * rr.random(40) - 3).reshape(-1, 1)
        ys = f_true(xs.ravel()) + rr.normal(0, SIGMA, 40)
        p = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                          Ridge(alpha=1e-8))
        preds[r] = p.fit(xs, ys).predict(x_test)
    mean_pred = preds.mean(0)
    bias2 = float(np.mean((mean_pred - f_test)**2))
    var   = float(np.mean(preds.var(0)))
    noise = SIGMA**2
    y_test_noisy = f_test + rng.normal(0, SIGMA, (200, len(f_test)))
    actual = float(np.mean((preds - y_test_noisy)**2))
    print(f"{d:>7}{bias2:>10.4f}{var:>11.4f}{noise:>9.4f}"
          f"{bias2+var+noise:>10.4f}{actual:>13.4f}")

print("\\nBias falls monotonically with degree. Variance rises. Their sum has")
print("an interior minimum -- that is the sweet spot, and the columns")
print("'total' and 'actual MSE' agreeing confirms the decomposition is exact.")

import plotly.graph_objects as go
''',
        key="ch04_learning",
    )

    quiz(
        "Your learning curves: training RMSE 4.8, validation RMSE 5.0, both flat "
        "since m = 40. You have budget to collect 10× more data. Should you?",
        ["Yes — more data always helps",
         "No — the curves have converged, so the problem is bias, not variance",
         "Yes, but only if you also regularise",
         "Cannot tell from this information"],
        1,
        "Converged curves at a high plateau = underfitting. More data will land "
        "exactly on the same plateau. Spend the budget on capacity or features "
        "instead.",
        key="ch04q2",
    )

    keypoints([
        "Learning curves plot error vs <b>training-set size</b> — the diagnostic "
        "for bias vs variance.",
        "Converged curves at a high plateau ⇒ <b>bias</b>: more data will not help.",
        "A persistent gap ⇒ <b>variance</b>: more data or regularisation will.",
        "$\\mathbb{E}[\\text{error}] = \\text{Bias}^2 + \\text{Variance} + \\sigma^2$, "
        "exactly, for squared loss.",
        "$\\sigma^2$ is irreducible — no model beats it, so it is your floor.",
    ])


# ==========================================================================
def s_4_5():
    section("4.5", "Regularized Linear Models")

    lead(
        "The general recipe: keep the flexible model but add a penalty on the size "
        "of the weights. Three penalties, three geometries, three completely "
        "different behaviours — and the difference is entirely explained by the "
        "shape of the constraint region."
    )

    sub("Ridge regression ($\\ell_2$)")

    math(r"""
    J(\boldsymbol\theta) \;=\; \mathrm{MSE}(\boldsymbol\theta)
    \;+\; \frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^{2}
    \;=\; \mathrm{MSE}(\boldsymbol\theta)
    \;+\; \frac{\alpha}{m}\bigl\lVert \mathbf{w}\bigr\rVert_2^2
    """)

    where({
        r"\alpha": "the regularisation strength. $\\alpha = 0$ gives plain linear "
                   "regression; $\\alpha \\to \\infty$ drives all weights to zero, "
                   "leaving a flat line through the mean",
        r"\mathbf{w}": "the weight vector $(\\theta_1,\\dots,\\theta_n)$ — note the "
                       "sum starts at $i=1$: <b>the bias $\\theta_0$ is never "
                       "regularised</b>",
    })

    md("Ridge has a closed form too, and it is better behaved than OLS:")

    math(r"""
    \hat{\boldsymbol\theta} \;=\;
    \bigl(\mathbf{X}^\top\mathbf{X} + \alpha \mathbf{A}\bigr)^{-1}\mathbf{X}^\top\mathbf{y}
    """)
    where({r"\mathbf{A}": "the $(n+1)\\times(n+1)$ identity with a <b>zero in the "
                          "top-left cell</b>, so the bias term escapes the penalty"})

    proof(
        "Ridge cannot be singular",
        "Adding $\\alpha\\mathbf{I}$ shifts every eigenvalue of "
        "$\\mathbf{X}^\\top\\mathbf{X}$ up by $\\alpha$. Since all eigenvalues were "
        "$\\ge 0$, they become $\\ge \\alpha > 0$, so the matrix is strictly "
        "positive definite and invertible <b>always</b> — even when $m < n$, even "
        "with perfectly collinear features. This is precisely why ridge was "
        "invented (Hoerl & Kennard, 1970): not primarily to prevent overfitting, "
        "but to fix ill-conditioning.",
    )

    derive(
        [("The shrinkage is exactly quantifiable in the SVD basis. Write "
          "$\\mathbf{X} = \\mathbf{U}\\boldsymbol\\Sigma\\mathbf{V}^\\top$ with "
          "singular values $\\sigma_j$.", None),
         ("Ordinary least squares in that basis:",
          r"\hat{\boldsymbol\theta}^{\text{OLS}} = \sum_j \frac{1}{\sigma_j}"
          r"\bigl(\mathbf{u}_j^\top \mathbf{y}\bigr)\, \mathbf{v}_j"),
         ("Ridge, for the same data:",
          r"\hat{\boldsymbol\theta}^{\text{ridge}} = \sum_j "
          r"\frac{\sigma_j}{\sigma_j^2 + \alpha}\bigl(\mathbf{u}_j^\top \mathbf{y}\bigr)\,\mathbf{v}_j"),
         ("So each direction is multiplied by a <b>shrinkage factor</b>:",
          r"s_j = \frac{\sigma_j^2}{\sigma_j^2 + \alpha} \;\in\; (0, 1)"),
         ("Directions with large $\\sigma_j$ (lots of variance in the data, "
          "well-determined) have $s_j \\approx 1$ and are barely touched. "
          "Directions with tiny $\\sigma_j$ — exactly the ill-conditioned ones that "
          "blow up in OLS — have $s_j \\approx \\sigma_j^2/\\alpha \\approx 0$ and "
          "are crushed.", None),
         ("The <b>effective degrees of freedom</b> follow immediately, and this is "
          "the honest measure of how complex the fitted ridge model actually is:",
          r"\mathrm{df}(\alpha) = \sum_{j=1}^{n} \frac{\sigma_j^2}{\sigma_j^2 + \alpha}"
          r"\;\;\xrightarrow{\alpha \to 0}\;\; n, \qquad "
          r"\xrightarrow{\alpha \to \infty}\;\; 0")],
        title="Ridge in the SVD basis — exactly what gets shrunk",
    )

    sub("Lasso regression ($\\ell_1$)")

    math(r"""
    J(\boldsymbol\theta) \;=\; \mathrm{MSE}(\boldsymbol\theta)
    \;+\; 2\alpha\sum_{i=1}^{n}\bigl|\theta_i\bigr|
    \;=\; \mathrm{MSE}(\boldsymbol\theta) \;+\; 2\alpha\bigl\lVert \mathbf{w}\bigr\rVert_1
    """)

    md(
        "Lasso does something ridge cannot: it drives weights **exactly to zero**, "
        "producing a sparse model. It performs automatic **feature selection**."
    )

    derive(
        [("Why exactly zero? Consider one coordinate with the others fixed. The "
          "$\\ell_1$ term contributes $2\\alpha|\\theta_j|$, whose derivative is "
          "$2\\alpha\\,\\mathrm{sign}(\\theta_j)$ — a <b>constant-magnitude</b> pull "
          "toward zero that does not vanish as $\\theta_j \\to 0$.", None),
         ("The $\\ell_2$ term contributes $2\\alpha\\theta_j$, whose pull "
          "<i>shrinks in proportion</i> to $\\theta_j$ and vanishes at zero. So "
          "ridge asymptotes to zero; lasso arrives.", None),
         ("Solving the 1-D problem exactly gives the <b>soft-thresholding</b> "
          "operator — the closed form for lasso when the features are orthonormal:",
          r"\hat\theta_j^{\text{lasso}} = "
          r"\mathrm{sign}\bigl(\hat\theta_j^{\text{OLS}}\bigr)\,"
          r"\max\Bigl(\bigl|\hat\theta_j^{\text{OLS}}\bigr| - \alpha,\; 0\Bigr)"),
         ("Compare ridge in the same setting, which is pure multiplicative "
          "shrinkage and never reaches zero:",
          r"\hat\theta_j^{\text{ridge}} = \frac{\hat\theta_j^{\text{OLS}}}{1 + \alpha}"),
         ("$|\\theta_j|$ is not differentiable at $0$, so plain gradient descent "
          "cannot be used. The <b>subgradient</b> is any value in $[-1, 1]$ there; "
          "in practice solvers use coordinate descent or LARS, which handle the "
          "kink exactly.", None)],
        title="Why ℓ₁ gives exact zeros and ℓ₂ does not",
    )

    sub("The geometric picture")

    anim_header("Why the ℓ₁ ball produces sparsity")
    md(
        "Both penalties can be written as *constrained* problems: minimise MSE "
        "subject to $\\lVert\\mathbf{w}\\rVert \\le t$. The MSE contours are "
        "ellipses; the solution is where they first touch the constraint region. "
        "The $\\ell_1$ region is a **diamond with corners on the axes** — and a "
        "corner is exactly a point where a coefficient is zero. The $\\ell_2$ "
        "region is a circle with no corners, so it is touched at a generic point."
    )

    th1 = np.linspace(-2.2, 2.6, 160)
    th2 = np.linspace(-2.2, 2.6, 160)
    A1, A2 = np.meshgrid(th1, th2)
    ols = np.array([1.55, 0.42])
    Hm = np.array([[1.0, 0.82], [0.82, 1.0]])
    Zc = np.zeros_like(A1)
    for i in range(A1.shape[0]):
        for j in range(A1.shape[1]):
            d = np.array([A1[i, j], A2[i, j]]) - ols
            Zc[i, j] = d @ Hm @ d

    ts = np.linspace(2.2, 0.12, 34)
    ang = np.linspace(0, 2 * np.pi, 200)

    def l1_sol(t):
        best, bv = None, 1e18
        for a in np.linspace(-t, t, 900):
            for b in (t - abs(a), -(t - abs(a))):
                d = np.array([a, b]) - ols
                v = d @ Hm @ d
                if v < bv:
                    bv, best = v, np.array([a, b])
        return best

    def l2_sol(t):
        best, bv = None, 1e18
        for a in ang:
            p = t * np.array([np.cos(a), np.sin(a)])
            d = p - ols
            v = d @ Hm @ d
            if v < bv:
                bv, best = v, p
        return best

    frames = []
    for t in ts:
        s1, s2 = l1_sol(t), l2_sol(t)
        nz = int(np.sum(np.abs(s1) > 1e-3))
        frames.append(go.Frame(name=f"{t:.2f}", data=[
            go.Contour(x=th1, y=th2, z=Zc, colorscale=nav.cscale(), showscale=False,
                       opacity=.6, ncontours=22),
            go.Scatter(x=[t, 0, -t, 0, t], y=[0, t, 0, -t, 0], mode="lines",
                       fill="toself", fillcolor=alpha(C["danger"], .18),
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=[s1[0]], y=[s1[1]], mode="markers",
                       marker=dict(color=C["danger"], size=16,
                                   line=dict(color="#fff", width=2))),
            go.Contour(x=th1, y=th2, z=Zc, colorscale=nav.cscale(), showscale=False,
                       opacity=.6, ncontours=22),
            go.Scatter(x=t * np.cos(ang), y=t * np.sin(ang), mode="lines",
                       fill="toself", fillcolor=alpha(C["info"], .18),
                       line=dict(color=C["info"], width=3)),
            go.Scatter(x=[s2[0]], y=[s2[1]], mode="markers",
                       marker=dict(color=C["info"], size=16,
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"budget t = {t:.2f}   |   LASSO θ = ({s1[0]:+.3f}, {s1[1]:+.3f}) "
            f"→ {nz} non-zero   |   RIDGE θ = ({s2[0]:+.3f}, {s2[1]:+.3f}) "
            f"→ 2 non-zero",
            color=C["danger"] if nz < 2 else C["ink"])])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("LASSO — ℓ₁ ball (a diamond: it has corners)",
                                      "RIDGE — ℓ₂ ball (a circle: no corners)"))
    t0 = ts[0]
    f.add_trace(go.Contour(x=th1, y=th2, z=Zc, colorscale=nav.cscale(),
                           showscale=False, opacity=.6, ncontours=22), 1, 1)
    f.add_trace(go.Scatter(x=[t0, 0, -t0, 0, t0], y=[0, t0, 0, -t0, 0],
                           mode="lines", fill="toself",
                           fillcolor=alpha(C["danger"], .18),
                           line=dict(color=C["danger"], width=3),
                           name="‖w‖₁ ≤ t"), 1, 1)
    s10 = l1_sol(t0)
    f.add_trace(go.Scatter(x=[s10[0]], y=[s10[1]], mode="markers", name="lasso θ̂",
                           marker=dict(color=C["danger"], size=16,
                                       line=dict(color="#fff", width=2))), 1, 1)
    f.add_trace(go.Contour(x=th1, y=th2, z=Zc, colorscale=nav.cscale(),
                           showscale=False, opacity=.6, ncontours=22), 1, 2)
    f.add_trace(go.Scatter(x=t0 * np.cos(ang), y=t0 * np.sin(ang), mode="lines",
                           fill="toself", fillcolor=alpha(C["info"], .18),
                           line=dict(color=C["info"], width=3),
                           name="‖w‖₂ ≤ t"), 1, 2)
    s20 = l2_sol(t0)
    f.add_trace(go.Scatter(x=[s20[0]], y=[s20[1]], mode="markers", name="ridge θ̂",
                           marker=dict(color=C["info"], size=16,
                                       line=dict(color="#fff", width=2))), 1, 2)
    for c in (1, 2):
        f.add_trace(go.Scatter(x=[ols[0]], y=[ols[1]], mode="markers",
                               name="OLS solution" if c == 1 else None,
                               showlegend=(c == 1),
                               marker=dict(color="#fff", size=13, symbol="star",
                                           line=dict(color=C["ink"], width=2))), 1, c)
    f.update_xaxes(range=[-2.2, 2.6], title_text="θ₁")
    f.update_yaxes(range=[-2.2, 2.6], title_text="θ₂")
    f.update_layout(height=500,
                    title="Shrink the budget t and watch where each solution lands")
    anim.animate(f, frames, duration=nav.anim_ms(240), slider_prefix="t = ")
    figure(f, "As t shrinks, the lasso solution snaps onto the vertical axis "
              "(θ₁ = 0 exactly). The ridge solution slides smoothly toward the "
              "origin but never touches an axis.")

    sub("Elastic Net")

    math(r"""
    J(\boldsymbol\theta) \;=\; \mathrm{MSE}(\boldsymbol\theta)
    \;+\; r\,\bigl(2\alpha \lVert\mathbf{w}\rVert_1\bigr)
    \;+\; (1 - r)\,\Bigl(\frac{\alpha}{m}\lVert\mathbf{w}\rVert_2^2\Bigr)
    """)
    where({r"r": "the mix ratio (<code>l1_ratio</code>). $r = 0$ is pure ridge, "
                 "$r = 1$ is pure lasso"})

    table(
        ["Situation", "Use", "Why"],
        [["Default starting point", "<b>Ridge</b>",
          "Always well-posed, cheap, no feature is discarded"],
         ["You suspect only a few features matter", "<b>Lasso</b> or Elastic Net",
          "Sparsity is both regularisation and interpretation"],
         ["$n > m$, or features are strongly correlated", "<b>Elastic Net</b>",
          "Lasso is erratic here: it picks one of a correlated group essentially "
          "at random, and it can select at most $m$ features. The $\\ell_2$ term "
          "makes the choice stable and lets correlated groups enter together"],
         ["You need every feature retained", "<b>Ridge</b>",
          "Lasso will delete some"],
         ["Almost never", "<b>Plain LinearRegression</b>",
          "A little regularisation is nearly always better than none"]],
    )

    sub("Early stopping")

    md(
        "A regulariser that costs nothing: stop training when validation error "
        "stops improving. For gradient descent on a convex loss it is provably "
        "*equivalent to an $\\ell_2$ penalty* whose strength decreases with the "
        "number of iterations — the fewer steps you take, the closer you stay to "
        "$\\boldsymbol\\theta = \\mathbf{0}$."
    )

    anim_header("Regularisation paths — coefficients as α sweeps")

    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler

    rngp = np.random.default_rng(0)
    mp, npf = 90, 12
    Xp = rngp.normal(0, 1, (mp, npf))
    Xp[:, 3] = Xp[:, 2] + rngp.normal(0, .07, mp)      # a correlated pair
    beta = np.zeros(npf); beta[[0, 2, 5, 8]] = [3.2, -2.4, 1.9, -1.3]
    yp = Xp @ beta + rngp.normal(0, 1.0, mp)
    Xs = StandardScaler().fit_transform(Xp)

    alphas = np.logspace(-3, 2.1, 45)
    paths = {"Ridge": [], "Lasso": [], "ElasticNet (r=0.5)": []}
    for a in alphas:
        paths["Ridge"].append(Ridge(alpha=a * mp).fit(Xs, yp).coef_)
        paths["Lasso"].append(Lasso(alpha=a, max_iter=20000).fit(Xs, yp).coef_)
        paths["ElasticNet (r=0.5)"].append(
            ElasticNet(alpha=a, l1_ratio=.5, max_iter=20000).fit(Xs, yp).coef_)
    for k in paths:
        paths[k] = np.array(paths[k])

    frames = []
    for k in range(1, len(alphas) + 1):
        data, info = [], []
        for ci, name in enumerate(paths):
            P = paths[name]
            for j in range(npf):
                col = SEQ[j % len(SEQ)] if beta[j] != 0 else alpha(C["muted"], .55)
                data.append(go.Scatter(x=alphas[:k], y=P[:k, j], mode="lines",
                                       line=dict(color=col,
                                                 width=3 if beta[j] != 0 else 1.4),
                                       showlegend=False))
            info.append(f"{name.split()[0]}: {int(np.sum(np.abs(P[k-1]) > 1e-6))} non-zero")
        frames.append(go.Frame(name=f"{alphas[k-1]:.3g}", data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"α = {alphas[k-1]:.4g}   |   "
                                   + "   |   ".join(info))])))

    f = make_subplots(rows=1, cols=3, subplot_titles=list(paths),
                      shared_yaxes=True)
    for ci, name in enumerate(paths, start=1):
        P = paths[name]
        for j in range(npf):
            col = SEQ[j % len(SEQ)] if beta[j] != 0 else alpha(C["muted"], .55)
            f.add_trace(go.Scatter(x=alphas[:1], y=P[:1, j], mode="lines",
                                   line=dict(color=col,
                                             width=3 if beta[j] != 0 else 1.4),
                                   showlegend=False), 1, ci)
    f.update_xaxes(type="log", title_text="α")
    f.update_yaxes(range=[-3.6, 4.2], title_text="coefficient", row=1, col=1)
    f.update_layout(height=440,
                    title="Regularisation paths — coloured = truly non-zero, "
                          "grey = noise features")
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="α = ")
    figure(f, "Lasso snaps the grey noise coefficients to exactly zero. Ridge "
              "shrinks them toward zero but keeps all twelve alive forever.")

    code_lab(
        "Ridge, lasso, elastic net — sparsity, stability, and early stopping",
        '''import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  SGDRegressor, RidgeCV, LassoCV, ElasticNetCV)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from copy import deepcopy

rng = np.random.default_rng(0)
m, n = 120, 40
X = rng.normal(0, 1, (m, n))
X[:, 1] = X[:, 0] + rng.normal(0, .05, m)          # near-duplicate features
X[:, 2] = X[:, 0] + rng.normal(0, .05, m)
beta = np.zeros(n); beta[[0, 5, 12, 25]] = [3., -2., 1.5, -1.]
y = X @ beta + rng.normal(0, 1., m)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0)
print(f"m={m}, n={n}, only {int((beta!=0).sum())} features truly matter\\n")

print(f"{'model':<26}{'test RMSE':>11}{'||w||_2':>10}{'||w||_1':>10}{'non-zero':>10}")
def report(name, mdl):
    p = make_pipeline(StandardScaler(), mdl).fit(Xtr, ytr)
    w = p[-1].coef_
    print(f"{name:<26}{mean_squared_error(yte, p.predict(Xte))**.5:>11.4f}"
          f"{np.linalg.norm(w):>10.3f}{np.abs(w).sum():>10.3f}"
          f"{int((np.abs(w) > 1e-6).sum()):>10}")
    return w

w_ols = report("LinearRegression", LinearRegression())
w_r   = report("Ridge (alpha=1)", Ridge(alpha=1.))
w_r10 = report("Ridge (alpha=10)", Ridge(alpha=10.))
w_l   = report("Lasso (alpha=0.1)", Lasso(alpha=.1, max_iter=50000))
w_l3  = report("Lasso (alpha=0.3)", Lasso(alpha=.3, max_iter=50000))
w_e   = report("ElasticNet (a=.1,r=.5)", ElasticNet(alpha=.1, l1_ratio=.5, max_iter=50000))

print("\\nwhich features did lasso keep?", np.where(np.abs(w_l) > 1e-6)[0])
print("which features truly matter?    ", np.where(beta != 0)[0])

# --- lasso is UNSTABLE with correlated features -------------------------
print("\\n=== stability under resampling (features 0,1,2 are near-duplicates) ===")
print(f"{'run':>4}{'lasso picks from {0,1,2}':>28}{'elastic net picks':>22}")
for s in range(5):
    idx = np.random.default_rng(s).choice(len(Xtr), len(Xtr), replace=True)
    la = make_pipeline(StandardScaler(), Lasso(alpha=.1, max_iter=50000)).fit(Xtr[idx], ytr[idx])
    en = make_pipeline(StandardScaler(), ElasticNet(alpha=.1, l1_ratio=.5, max_iter=50000)).fit(Xtr[idx], ytr[idx])
    lp = [j for j in (0,1,2) if abs(la[-1].coef_[j]) > 1e-6]
    ep = [j for j in (0,1,2) if abs(en[-1].coef_[j]) > 1e-6]
    print(f"{s:>4}{str(lp):>28}{str(ep):>22}")
print("Lasso arbitrarily picks ONE of a correlated group; elastic net keeps the group.")

# --- ridge effective degrees of freedom ----------------------------------
Xs = StandardScaler().fit_transform(Xtr)
sv = np.linalg.svd(Xs, compute_uv=False)
print(f"\\n{'alpha':>10}{'effective df':>14}   (n = %d)" % n)
for a in [0.001, .1, 1, 10, 100, 1000, 1e5]:
    print(f"{a:>10}{np.sum(sv**2/(sv**2+a)):>14.2f}")

# --- EARLY STOPPING as regularisation ------------------------------------
print("\\n=== early stopping ===")
Xp_tr, Xp_va, yp_tr, yp_va = train_test_split(Xtr, ytr, test_size=.3, random_state=1)
prep = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler())
A_tr = prep.fit_transform(Xp_tr); A_va = prep.transform(Xp_va)
sgd = SGDRegressor(penalty=None, eta0=.0005, learning_rate="constant",
                   max_iter=1, warm_start=True, tol=None, random_state=42)
best, best_ep, best_model, hist = np.inf, 0, None, []
for ep in range(600):
    sgd.fit(A_tr, yp_tr)
    e = mean_squared_error(yp_va, sgd.predict(A_va))**.5
    hist.append(e)
    if e < best:
        best, best_ep, best_model = e, ep, deepcopy(sgd)
print(f"best validation RMSE {best:.4f} at epoch {best_ep} "
      f"(of 600); final-epoch RMSE {hist[-1]:.4f}")
print(f"stopping early avoided {hist[-1]-best:.4f} RMSE of overfitting")

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(y=hist, mode="lines", line=dict(color=C["test"], width=2.5),
                           name="validation RMSE"))
fig.add_vline(x=best_ep, line_dash="dash", line_color=C["success"],
              annotation_text=f"stop here (epoch {best_ep})")
fig.update_layout(height=380, xaxis_title="epoch", yaxis_title="validation RMSE",
                  title="Early stopping is free regularisation")
''',
        key="ch04_reg",
    )

    quiz(
        "You have 200 rows, 5 000 highly correlated gene-expression features, and "
        "you need to name the genes that matter. Which model?",
        ["LinearRegression", "Ridge", "Lasso", "Elastic Net"],
        3,
        "You need sparsity (to name genes), so ridge is out. But lasso with "
        "correlated features picks one gene per correlated group arbitrarily and "
        "can select at most $m = 200$ features. Elastic Net gives sparsity *and* "
        "stable, grouped selection.",
        key="ch04q3",
    )

    keypoints([
        "Ridge ($\\ell_2$): shrinks by $\\sigma_j^2/(\\sigma_j^2+\\alpha)$, always "
        "invertible, keeps every feature.",
        "Lasso ($\\ell_1$): soft-thresholding ⇒ <b>exact zeros</b> ⇒ automatic "
        "feature selection.",
        "The geometry explains everything: the $\\ell_1$ ball has <b>corners on the "
        "axes</b>, the $\\ell_2$ ball does not.",
        "Elastic Net when $n > m$ or features are correlated — lasso is unstable "
        "there.",
        "<b>Never regularise the bias $\\theta_0$</b>, and <b>always scale</b> "
        "first: the penalty is not scale-invariant.",
        "Early stopping is an implicit $\\ell_2$ penalty and costs nothing.",
    ])


# ==========================================================================
def s_4_6():
    section("4.6", "Logistic Regression")

    lead(
        "Regression machinery, classification output. Instead of predicting a "
        "number we predict the <i>log-odds</i> of the positive class, and squash "
        "it into a probability."
    )

    sub("Estimating probabilities")

    math(r"""
    \hat p \;=\; h_{\boldsymbol\theta}(\mathbf{x})
    \;=\; \sigma\bigl(\boldsymbol\theta^\top \mathbf{x}\bigr),
    \qquad
    \sigma(t) \;=\; \frac{1}{1 + \exp(-t)}
    """)

    md("The logistic (sigmoid) function has properties worth memorising:")

    math(r"""
    \sigma(0) = \tfrac12,
    \qquad
    \sigma(-t) = 1 - \sigma(t),
    \qquad
    \sigma'(t) = \sigma(t)\bigl(1 - \sigma(t)\bigr),
    \qquad
    \sigma^{-1}(p) = \log\frac{p}{1-p}
    """)

    idea(
        "The model is linear in the log-odds",
        "Inverting the sigmoid gives $\\log\\frac{\\hat p}{1-\\hat p} = "
        "\\boldsymbol\\theta^\\top\\mathbf{x}$. So logistic regression asserts that "
        "the <b>logit</b> is a linear function of the features. That is why "
        "$\\theta_j$ has a clean interpretation: a one-unit increase in $x_j$ "
        "multiplies the <b>odds</b> by $e^{\\theta_j}$, holding everything else "
        "fixed.",
    )

    sub("Training and cost function")

    md("The cost for a single instance:")

    math(r"""
    c(\boldsymbol\theta) \;=\;
    \begin{cases}
      -\log(\hat p)     & \text{if } y = 1\\
      -\log(1 - \hat p) & \text{if } y = 0
    \end{cases}
    """)

    md("Averaged over the training set — the **log loss**:")

    math(r"""
    J(\boldsymbol\theta) \;=\; -\frac{1}{m}\sum_{i=1}^{m}
      \Bigl[\, y^{(i)}\log\bigl(\hat p^{(i)}\bigr)
      \;+\; \bigl(1 - y^{(i)}\bigr)\log\bigl(1 - \hat p^{(i)}\bigr) \,\Bigr]
    """)

    md("There is **no closed-form solution**, but the gradient is beautifully "
       "simple — identical in form to linear regression's:")

    math(r"""
    \frac{\partial}{\partial \theta_j} J(\boldsymbol\theta) \;=\;
    \frac{1}{m}\sum_{i=1}^{m}
      \Bigl(\sigma\bigl(\boldsymbol\theta^\top \mathbf{x}^{(i)}\bigr) - y^{(i)}\Bigr)\,
      x_j^{(i)}
    """)

    derive(
        [("Start from one instance with $y = 1$, so $c = -\\log\\sigma(t)$ where "
          "$t = \\boldsymbol\\theta^\\top\\mathbf{x}$. Chain rule:",
          r"\frac{\partial c}{\partial \theta_j} = -\frac{1}{\sigma(t)}\cdot"
          r"\sigma'(t)\cdot \frac{\partial t}{\partial \theta_j}"),
         ("Substitute $\\sigma'(t) = \\sigma(t)(1-\\sigma(t))$ and "
          "$\\partial t/\\partial\\theta_j = x_j$:",
          r"= -\frac{\sigma(t)\bigl(1-\sigma(t)\bigr)}{\sigma(t)}\,x_j "
          r"= -\bigl(1 - \sigma(t)\bigr)x_j = \bigl(\sigma(t) - 1\bigr)x_j"),
         ("Now the $y = 0$ case, $c = -\\log(1-\\sigma(t))$:",
          r"\frac{\partial c}{\partial \theta_j} = "
          r"\frac{\sigma(t)\bigl(1-\sigma(t)\bigr)}{1-\sigma(t)}\,x_j "
          r"= \sigma(t)\,x_j = \bigl(\sigma(t) - 0\bigr)x_j"),
         ("Both cases collapse into one expression — the residual times the "
          "feature, exactly as in linear regression:",
          r"\frac{\partial c}{\partial \theta_j} = \bigl(\hat p - y\bigr)\,x_j"),
         ("Averaging over the training set gives the stated gradient. This is not "
          "a coincidence: both are <b>generalised linear models</b>, and for any "
          "GLM with its canonical link the gradient of the negative log-likelihood "
          "is always (prediction − target) × feature.", None),
         ("<b>Convexity.</b> The Hessian is $\\frac{1}{m}\\mathbf{X}^\\top "
          "\\mathbf{S}\\mathbf{X}$ with $\\mathbf{S} = \\mathrm{diag}(\\hat p_i"
          "(1-\\hat p_i)) \\succeq 0$, so $J$ is convex and gradient descent finds "
          "the global optimum. There is no local-minimum problem.", None)],
        title="Deriving the logistic gradient",
    )

    warn(
        "Why not squared error on probabilities?",
        "Using $(\\hat p - y)^2$ with a sigmoid gives a <b>non-convex</b> objective "
        "with local minima, and its gradient contains a factor $\\sigma'(t)$ which "
        "is $\\approx 0$ when the model is confidently wrong — so a badly wrong "
        "prediction produces almost no gradient and the model cannot recover. Log "
        "loss cancels that factor exactly (see step 2 of the derivation), which is "
        "why confident mistakes produce <b>large</b> gradients. This same argument "
        "reappears for cross-entropy in Chapter 10.",
    )

    anim_header("The sigmoid, the loss, and the boundary — all at once")

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    Xi = iris.data[["petal width (cm)"]].to_numpy()
    yi = (iris.target == 2).to_numpy().astype(int)
    xg = np.linspace(0, 3.2, 300).reshape(-1, 1)

    ths = np.linspace(-14, 14, 40)
    b_fixed = -2.6

    def sig(t): return 1 / (1 + np.exp(-np.clip(t, -60, 60)))

    losses = []
    for t1 in ths:
        p = sig(Xi[:, 0] * t1 + b_fixed * t1 / 5.5)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        losses.append(float(-np.mean(yi * np.log(p) + (1 - yi) * np.log(1 - p))))

    frames = []
    for k, t1 in enumerate(ths):
        b = b_fixed * t1 / 5.5
        pg = sig(xg[:, 0] * t1 + b)
        db = -b / t1 if abs(t1) > 1e-6 else np.nan
        frames.append(go.Frame(name=f"{t1:.1f}", data=[
            go.Scatter(x=Xi[yi == 0, 0], y=np.zeros((yi == 0).sum()) + .02,
                       mode="markers", marker=dict(color=C["train"], size=8,
                                                   opacity=.65)),
            go.Scatter(x=Xi[yi == 1, 0], y=np.ones((yi == 1).sum()) - .02,
                       mode="markers", marker=dict(color=C["warning"], size=8,
                                                   opacity=.65)),
            go.Scatter(x=xg[:, 0], y=pg, mode="lines",
                       line=dict(color=C["primary"], width=4)),
            go.Scatter(x=[db, db], y=[0, 1], mode="lines",
                       line=dict(color=C["danger"], width=3, dash="dash")),
            go.Scatter(x=ths[:k + 1], y=losses[:k + 1], mode="lines",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"θ₁ = {t1:+.2f}   boundary at petal width = {db:.2f} cm   "
            f"log loss = {losses[k]:.4f}")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.58, .42],
                      subplot_titles=("σ(θᵀx) and the decision boundary",
                                      "log loss as θ₁ varies"))
    f.add_trace(go.Scatter(x=Xi[yi == 0, 0], y=np.zeros((yi == 0).sum()) + .02,
                           mode="markers", name="not virginica",
                           marker=dict(color=C["train"], size=8, opacity=.65)), 1, 1)
    f.add_trace(go.Scatter(x=Xi[yi == 1, 0], y=np.ones((yi == 1).sum()) - .02,
                           mode="markers", name="virginica",
                           marker=dict(color=C["warning"], size=8, opacity=.65)), 1, 1)
    f.add_trace(go.Scatter(x=xg[:, 0], y=sig(xg[:, 0] * ths[0]), mode="lines",
                           name="p̂ = σ(θᵀx)",
                           line=dict(color=C["primary"], width=4)), 1, 1)
    f.add_trace(go.Scatter(x=[0, 0], y=[0, 1], mode="lines", name="p̂ = 0.5",
                           line=dict(color=C["danger"], width=3, dash="dash")), 1, 1)
    f.add_trace(go.Scatter(x=ths[:1], y=losses[:1], mode="lines", name="log loss",
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.update_xaxes(title_text="petal width (cm)", range=[0, 3.2], row=1, col=1)
    f.update_yaxes(title_text="probability", range=[-.06, 1.06], row=1, col=1)
    f.update_xaxes(title_text="θ₁", row=1, col=2)
    f.update_yaxes(title_text="log loss", row=1, col=2)
    f.update_layout(height=450, title="Fitting a logistic regression, live")
    anim.animate(f, frames, duration=nav.anim_ms(160), slider_prefix="θ₁ = ")
    figure(f, "The loss curve on the right is convex — a single bowl. That is "
              "what makes logistic regression reliable to fit.")

    sub("Decision boundaries")

    md(
        "The boundary is where $\\hat p = 0.5$, i.e. where "
        "$\\boldsymbol\\theta^\\top\\mathbf{x} = 0$ — a **hyperplane**, exactly as "
        "in §3.2. Logistic regression is a linear classifier; it is only its "
        "*probability estimates* that are non-linear."
    )

    codenote(
        "Regularisation is on by default, and inverted",
        "scikit-learn's <code>LogisticRegression</code> applies $\\ell_2$ "
        "regularisation unless you say otherwise, and its knob is <b>C</b>, which "
        "is the <b>inverse</b> of $\\alpha$: large <code>C</code> = weak "
        "regularisation. Set <code>penalty=None</code> for unregularised maximum "
        "likelihood.",
    )

    code_lab(
        "Logistic regression from scratch, verified against scikit-learn",
        '''import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

iris = load_iris()
X = iris.data[:, 3:]                       # petal width only
y = (iris.target == 2).astype(int)         # 1 = Iris virginica
X_b = np.c_[np.ones(len(X)), X]

def sigmoid(t): return 1 / (1 + np.exp(-np.clip(t, -500, 500)))

def cost(theta, X, y, eps=1e-12):
    p = np.clip(sigmoid(X @ theta), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def gradient(theta, X, y):
    return X.T @ (sigmoid(X @ theta) - y) / len(y)      # (p - y) * x

# ---------------- batch gradient descent ---------------------------------
theta = np.zeros(X_b.shape[1]); hist = []
for it in range(20_000):
    theta -= 1.0 * gradient(theta, X_b, y)
    if it % 500 == 0:
        hist.append(cost(theta, X_b, y))

print(f"from scratch : theta = {theta.round(5)}   log loss = {cost(theta, X_b, y):.6f}")
sk = LogisticRegression(penalty=None, max_iter=20_000).fit(X, y)
print(f"sklearn      : theta = {np.r_[sk.intercept_, sk.coef_[0]].round(5)}   "
      f"log loss = {log_loss(y, sk.predict_proba(X)[:, 1]):.6f}")

# ---------------- the decision boundary ----------------------------------
boundary = -theta[0] / theta[1]
print(f"\\ndecision boundary at petal width = {boundary:.4f} cm")
print(f"odds multiplier per extra cm of petal width = exp({theta[1]:.3f}) "
      f"= {np.exp(theta[1]):.1f}x")
for w in [0.8, 1.4, boundary, 1.8, 2.5]:
    p = sigmoid(theta[0] + theta[1] * w)
    print(f"  width {w:.2f} cm -> P(virginica) = {p:.4f}  "
          f"logit = {np.log(p/(1-p)):+.3f}")

# ---------------- verify the gradient numerically -----------------------
print("\\n=== numerical gradient check ===")
eps = 1e-6
th = np.array([-2.0, 1.5])
ana = gradient(th, X_b, y)
num = np.array([(cost(th + eps*np.eye(2)[j], X_b, y)
                 - cost(th - eps*np.eye(2)[j], X_b, y)) / (2*eps) for j in range(2)])
print(f"analytic  = {ana}")
print(f"numerical = {num}")
print(f"max abs difference = {np.abs(ana-num).max():.3e}   <- derivation confirmed")

# ---------------- convexity: the Hessian is PSD -------------------------
p = sigmoid(X_b @ theta)
H = (X_b * (p * (1 - p))[:, None]).T @ X_b / len(y)
print(f"\\nHessian eigenvalues = {np.linalg.eigvalsh(H).round(6)}  -> all >= 0, convex")

# ---------------- why NOT squared error ---------------------------------
def sq_cost(theta): return float(np.mean((sigmoid(X_b @ theta) - y) ** 2))
grid = np.linspace(-30, 30, 400)
log_vals = [cost(np.array([-2.6/5.5*t, t]), X_b, y) for t in grid]
sq_vals  = [sq_cost(np.array([-2.6/5.5*t, t]))       for t in grid]
d2_log = np.diff(np.diff(log_vals)); d2_sq = np.diff(np.diff(sq_vals))
print(f"\\nlog loss   : fraction of the slice that is convex = "
      f"{(d2_log >= -1e-12).mean():.1%}")
print(f"squared err: fraction of the slice that is convex = "
      f"{(d2_sq >= -1e-12).mean():.1%}   <- non-convex")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=grid, y=log_vals, name="log loss (convex)",
                line=dict(color=C["success"], width=3))
fig.add_scatter(x=grid, y=np.array(sq_vals)*8, name="squared error x8 (non-convex)",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=380, xaxis_title="θ₁", yaxis_title="cost",
                  title="Why log loss and not squared error")
''',
        key="ch04_logistic",
    )

    keypoints([
        "$\\hat p = \\sigma(\\boldsymbol\\theta^\\top\\mathbf{x})$; the model is "
        "<b>linear in the log-odds</b>, so $e^{\\theta_j}$ is an odds multiplier.",
        "Log loss, not squared error: it is convex and it cancels the vanishing "
        "$\\sigma'$ factor.",
        "$\\nabla J = \\frac{1}{m}\\mathbf{X}^\\top(\\hat{\\mathbf{p}} - \\mathbf{y})$ "
        "— identical in form to linear regression.",
        "No closed form, but convex ⇒ gradient descent always finds the global "
        "optimum.",
        "The decision boundary is a hyperplane; scikit-learn regularises by "
        "default with <code>C = 1/α</code>.",
    ])


# ==========================================================================
def s_4_7():
    section("4.7", "Softmax Regression")

    lead(
        "The direct generalisation to $K$ classes — one linear score per class, "
        "normalised into a probability distribution. It is also the output layer "
        "of essentially every classification neural network in Part II."
    )

    sub("The model")

    md("Compute a score for each class, then normalise:")

    math(r"""
    s_k(\mathbf{x}) \;=\; \bigl(\boldsymbol\theta^{(k)}\bigr)^\top \mathbf{x}
    \qquad\Longrightarrow\qquad
    \hat p_k \;=\; \sigma\bigl(\mathbf{s}(\mathbf{x})\bigr)_k
    \;=\; \frac{\exp\bigl(s_k(\mathbf{x})\bigr)}
               {\displaystyle\sum_{j=1}^{K}\exp\bigl(s_j(\mathbf{x})\bigr)}
    """)

    where({
        r"K": "the number of classes",
        r"\boldsymbol\theta^{(k)}": "the parameter vector for class $k$; stacked, "
                                    "they form the parameter <b>matrix</b> "
                                    "$\\boldsymbol\\Theta \\in "
                                    "\\mathbb{R}^{K\\times(n+1)}$",
        r"\hat p_k": "the estimated probability of class $k$; by construction "
                     "$\\hat p_k > 0$ and $\\sum_k \\hat p_k = 1$",
    })

    md("The prediction is the argmax — and because $\\exp$ is monotone, that is "
       "the same as the argmax of the raw scores:")

    math(r"""
    \hat y \;=\; \operatorname*{arg\,max}_{k}\; \hat p_k
      \;=\; \operatorname*{arg\,max}_{k}\; s_k(\mathbf{x})
      \;=\; \operatorname*{arg\,max}_{k}\;
            \bigl(\boldsymbol\theta^{(k)}\bigr)^\top \mathbf{x}
    """)

    sub("Cross-entropy cost")

    math(r"""
    J(\boldsymbol\Theta) \;=\; -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}
      y_k^{(i)} \, \log\bigl(\hat p_k^{(i)}\bigr)
    """)
    where({r"y_k^{(i)}": "1 if the $i$-th instance belongs to class $k$, else 0 "
                         "(the one-hot encoding of the label)"})

    md("The gradient for class $k$ — again *(prediction − target) × feature*:")

    math(r"""
    \nabla_{\boldsymbol\theta^{(k)}} J(\boldsymbol\Theta) \;=\;
    \frac{1}{m}\sum_{i=1}^{m}
      \Bigl(\hat p_k^{(i)} - y_k^{(i)}\Bigr)\,\mathbf{x}^{(i)}
    """)

    derive(
        [("The softmax Jacobian is the crux. Differentiate $\\hat p_k$ with respect "
          "to score $s_j$, splitting into the $j = k$ and $j \\ne k$ cases:",
          r"\frac{\partial \hat p_k}{\partial s_j} = "
          r"\hat p_k\bigl(\delta_{kj} - \hat p_j\bigr), \qquad "
          r"\delta_{kj} = \mathbb{1}[k = j]"),
         ("For one instance the loss is $L = -\\sum_k y_k \\log \\hat p_k$. Chain "
          "rule through every $\\hat p_k$:",
          r"\frac{\partial L}{\partial s_j} = -\sum_k \frac{y_k}{\hat p_k}\cdot"
          r"\hat p_k\bigl(\delta_{kj} - \hat p_j\bigr) "
          r"= -\sum_k y_k\bigl(\delta_{kj} - \hat p_j\bigr)"),
         ("Split the sum and use $\\sum_k y_k = 1$ (exactly one entry of the "
          "one-hot vector is 1):",
          r"= -y_j + \hat p_j \sum_k y_k = \hat p_j - y_j"),
         ("So the gradient with respect to the <b>scores</b> is simply the "
          "prediction minus the one-hot target — every messy softmax term "
          "cancels. Then $\\partial s_j/\\partial\\boldsymbol\\theta^{(j)} = "
          "\\mathbf{x}$ finishes it:",
          r"\nabla_{\boldsymbol\theta^{(k)}} L = \bigl(\hat p_k - y_k\bigr)\mathbf{x}"),
         ("<b>This cancellation is the single reason softmax + cross-entropy is "
          "the universal classification output.</b> It is what makes "
          "backpropagation through a classifier numerically clean, and you will "
          "meet it again in Chapter 10 and Chapter 16.", None),
         ("<b>Special case check.</b> For $K = 2$, softmax reduces exactly to "
          "logistic regression with $\\boldsymbol\\theta = \\boldsymbol\\theta^{(1)} "
          "- \\boldsymbol\\theta^{(0)}$:",
          r"\hat p_1 = \frac{e^{s_1}}{e^{s_0} + e^{s_1}} "
          r"= \frac{1}{1 + e^{-(s_1 - s_0)}} = \sigma(s_1 - s_0)")],
        title="Deriving the softmax gradient — and why everything cancels",
    )

    note(
        "Softmax is over-parameterised, and numerically delicate",
        "Adding a constant $\\mathbf{c}$ to <i>every</i> score leaves $\\hat p$ "
        "unchanged, so the solution is only identified up to a shift (this is why "
        "regularisation or a reference class is used). That same invariance is the "
        "standard numerical trick: subtract $\\max_j s_j$ before exponentiating, "
        "so the largest exponent is $e^0 = 1$ and nothing overflows.",
    )

    warn(
        "Softmax is multiclass, never multilabel",
        "Because $\\sum_k \\hat p_k = 1$ is enforced, the classes <b>compete</b> — "
        "raising one probability must lower another. If an instance can belong to "
        "several classes at once, use $K$ independent sigmoids instead (§3.8).",
    )

    anim_header("Softmax decision regions forming during training")

    from sklearn.datasets import load_iris

    iris = load_iris()
    Xs = iris.data[:, 2:4]
    ys = iris.target
    Xm_, Xsd_ = Xs.mean(0), Xs.std(0)
    Xn = (Xs - Xm_) / Xsd_
    Xnb = np.c_[np.ones(len(Xn)), Xn]
    Y1h = np.eye(3)[ys]

    def softmax(S):
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    Th = np.zeros((3, 3))
    snaps, losses_s = [], []
    for it in range(601):
        P = softmax(Xnb @ Th.T)
        if it % 20 == 0:
            snaps.append(Th.copy())
            losses_s.append(float(-np.mean(np.sum(
                Y1h * np.log(np.clip(P, 1e-12, 1)), axis=1))))
        Th -= 0.6 * ((P - Y1h).T @ Xnb) / len(Xnb)

    g1 = np.linspace(Xn[:, 0].min() - .6, Xn[:, 0].max() + .6, 150)
    g2 = np.linspace(Xn[:, 1].min() - .6, Xn[:, 1].max() + .6, 150)
    G1, G2 = np.meshgrid(g1, g2)
    Gb = np.c_[np.ones(G1.size), G1.ravel(), G2.ravel()]

    def sc(gy=None):
        return [go.Scatter(x=Xn[ys == k, 0], y=Xn[ys == k, 1], mode="markers",
                           marker=dict(color=[C["train"], C["warning"],
                                              C["success"]][k], size=8,
                                       line=dict(color="#fff", width=1)),
                           showlegend=False) for k in range(3)]

    cs3 = [[0, alpha(C["train"], .5)], [.5, alpha(C["warning"], .5)],
           [1, alpha(C["success"], .5)]]
    frames = []
    for i, T in enumerate(snaps):
        Z = softmax(Gb @ T.T).argmax(1).reshape(G1.shape).astype(float)
        conf = softmax(Gb @ T.T).max(1).reshape(G1.shape)
        frames.append(go.Frame(name=str(i * 20), data=[
            go.Contour(x=g1, y=g2, z=Z, colorscale=cs3, showscale=False,
                       contours=dict(showlines=False), opacity=.55),
            go.Contour(x=g1, y=g2, z=conf, showscale=False, contours_coloring="lines",
                       line=dict(width=1.4), colorscale="Greys",
                       contours=dict(start=.4, end=.95, size=.15))] + sc(),
            layout=go.Layout(annotations=[anim.annotate_step(
                f"iteration {i*20}   ·   cross-entropy = {losses_s[i]:.4f}")])))

    Z0 = softmax(Gb @ snaps[0].T).argmax(1).reshape(G1.shape).astype(float)
    f = go.Figure(data=[
        go.Contour(x=g1, y=g2, z=Z0, colorscale=cs3, showscale=False,
                   contours=dict(showlines=False), opacity=.55),
        go.Contour(x=g1, y=g2, z=np.zeros_like(G1), showscale=False,
                   contours_coloring="lines", line=dict(width=1.4),
                   colorscale="Greys")] + sc())
    f.update_layout(height=520, xaxis_title="petal length (standardised)",
                    yaxis_title="petal width (standardised)",
                    title="Softmax regression learning three linear boundaries")
    anim.animate(f, frames, duration=nav.anim_ms(110), slider_prefix="iteration ")
    figure(f, "Grey contours are confidence levels. Note the boundaries are "
              "straight lines — softmax regression is a linear classifier.")

    code_lab(
        "Softmax from scratch, the gradient check, and the K=2 reduction",
        '''import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

iris = load_iris()
X, y = iris.data[:, 2:4], iris.target                 # petal length & width
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
Atr, Ate = np.c_[np.ones(len(Xtr)), sc.transform(Xtr)], np.c_[np.ones(len(Xte)), sc.transform(Xte)]
K = 3
Y = np.eye(K)[ytr]

def softmax(S):
    S = S - S.max(axis=1, keepdims=True)              # numerical stability
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Theta, A, Y1h, eps=1e-12):
    P = np.clip(softmax(A @ Theta.T), eps, 1.)
    return float(-np.mean(np.sum(Y1h * np.log(P), axis=1)))

def grad(Theta, A, Y1h):
    return ((softmax(A @ Theta.T) - Y1h).T @ A) / len(A)

Theta = np.zeros((K, Atr.shape[1]))
for it in range(6000):
    Theta -= 0.5 * grad(Theta, Atr, Y)

print(f"from scratch : train CE = {cross_entropy(Theta, Atr, Y):.6f}")
print(f"               test accuracy = "
      f"{accuracy_score(yte, softmax(Ate @ Theta.T).argmax(1)):.4f}")

sk = LogisticRegression(penalty=None, max_iter=6000)
sk.fit(sc.transform(Xtr), ytr)
print(f"sklearn      : train CE = "
      f"{log_loss(ytr, sk.predict_proba(sc.transform(Xtr))):.6f}")
print(f"               test accuracy = {sk.score(sc.transform(Xte), yte):.4f}")

# ---- the gradient identity, verified numerically -----------------------
print("\\n=== gradient check: dJ/ds = p - y ===")
eps = 1e-6
T = np.random.default_rng(0).normal(0, .4, (K, Atr.shape[1]))
ana = grad(T, Atr, Y)
num = np.zeros_like(T)
for k in range(K):
    for j in range(T.shape[1]):
        E = np.zeros_like(T); E[k, j] = eps
        num[k, j] = (cross_entropy(T+E, Atr, Y) - cross_entropy(T-E, Atr, Y)) / (2*eps)
print(f"max |analytic - numerical| = {np.abs(ana-num).max():.3e}   <- derivation confirmed")

# ---- softmax with K=2 IS logistic regression ---------------------------
print("\\n=== K = 2 reduces exactly to logistic regression ===")
mask = ytr < 2
A2, y2 = Atr[mask], ytr[mask]
Y2 = np.eye(2)[y2]
T2 = np.zeros((2, A2.shape[1]))
for _ in range(8000):
    T2 -= .5 * grad(T2, A2, Y2)
diff = T2[1] - T2[0]
sig = 1 / (1 + np.exp(-(A2 @ diff)))
p_soft = softmax(A2 @ T2.T)[:, 1]
print(f"theta_1 - theta_0            = {diff.round(4)}")
print(f"max |softmax_p1 - sigmoid|   = {np.abs(p_soft - sig).max():.2e}  <- identical")

# ---- probabilities for a new flower -------------------------------------
new = sc.transform([[5.0, 2.0]])
p = softmax(np.c_[[1.], new] @ Theta.T)[0]
print(f"\\npetal 5.0 x 2.0 cm ->")
for k, nm in enumerate(iris.target_names):
    print(f"   P({nm:<12}) = {p[k]:.4f}  {'#'*int(p[k]*40)}")
print(f"   sum = {p.sum():.10f}   <- always exactly 1 (that is the constraint)")

# ---- the overparameterisation ------------------------------------------
shift = np.random.default_rng(1).normal(0, 3, Atr.shape[1])
print(f"\\nadd the same vector to every class's parameters:")
print(f"   max |p_before - p_after| = "
      f"{np.abs(softmax(Atr @ Theta.T) - softmax(Atr @ (Theta+shift).T)).max():.2e}")
print("   -> softmax is invariant to a common shift; the solution is not unique.")
''',
        key="ch04_softmax",
    )

    keypoints([
        "One score vector per class, normalised by softmax into a probability "
        "distribution.",
        "Cross-entropy loss; its gradient with respect to the scores is exactly "
        "$\\hat{\\mathbf{p}} - \\mathbf{y}$ — everything cancels.",
        "That cancellation is why softmax + cross-entropy is the standard output "
        "of every classification network (Ch. 10, 14, 16).",
        "$K = 2$ reduces exactly to logistic regression.",
        "Multiclass only, never multilabel — the probabilities are forced to "
        "compete.",
        "Subtract $\\max_j s_j$ before exponentiating; the invariance makes it free.",
    ])


# ==========================================================================
def s_4_8():
    section("4.8", "Exercises & Chapter Review")

    lead("Twelve exercises. These are the ones worth doing on paper.")

    exercise(
        1, "Which linear regression training algorithm can you use if you have a "
        "training set with millions of features?",
        "**Stochastic gradient descent, mini-batch GD, or batch GD** — anything "
        "gradient-based. The closed forms are ruled out: the normal equation "
        "requires inverting an $n \\times n$ matrix at $\\mathcal{O}(n^{2.4})$–"
        "$\\mathcal{O}(n^3)$, and even SVD is $\\mathcal{O}(n^2 m)$. With $n$ in "
        "the millions, both are hopeless in time and in memory (the matrix alone "
        "would be $10^{12}$ entries).")

    exercise(
        2, "Suppose the features in your training set have very different scales. "
        "Which algorithms might suffer from this, and how? What can you do about "
        "it?",
        "**All gradient-descent variants suffer**, badly. The cost surface becomes "
        "an elongated valley: the condition number $\\kappa = \\lambda_{\\max}/"
        "\\lambda_{\\min}$ blows up, and the convergence factor "
        "$\\rho = (\\kappa-1)/(\\kappa+1)$ approaches 1, so convergence takes "
        "orders of magnitude longer.\n\n"
        "**Regularised models suffer even more seriously**, because the penalty "
        "$\\lVert\\mathbf{w}\\rVert$ is *not scale-invariant*: a feature measured "
        "in millimetres gets a coefficient 1 000× larger than the same feature in "
        "metres, and is therefore penalised 10⁶× more heavily under $\\ell_2$. "
        "This is not slow convergence — it is a *different model*.\n\n"
        "The closed forms do not care about scaling for the *solution* (it is "
        "exact), though extreme scale differences worsen conditioning.\n\n"
        "**Fix:** `StandardScaler` inside a `Pipeline`. Always.")

    exercise(
        3, "Can gradient descent get stuck in a local minimum when training a "
        "logistic regression model?",
        "**No.** The log-loss cost function is **convex** — its Hessian "
        "$\\frac{1}{m}\\mathbf{X}^\\top\\mathbf{S}\\mathbf{X}$ with "
        "$\\mathbf{S} = \\mathrm{diag}(\\hat p_i(1-\\hat p_i))$ is positive "
        "semi-definite, since $\\mathbf{v}^\\top\\mathbf{X}^\\top\\mathbf{S}"
        "\\mathbf{X}\\mathbf{v} = \\lVert\\mathbf{S}^{1/2}\\mathbf{X}\\mathbf{v}"
        "\\rVert^2 \\ge 0$. A convex function has no local minima that are not "
        "global. (It can still fail to *terminate* if the classes are perfectly "
        "separable — the weights then run to infinity. That is why sklearn "
        "regularises by default.)")

    exercise(
        4, "Do all gradient descent algorithms lead to the same model, provided "
        "you let them run long enough?",
        "**Only if the cost function is convex and the learning rate is not too "
        "large.** For linear/logistic/softmax regression, all three variants "
        "converge to the same global optimum — with the caveat that plain SGD and "
        "mini-batch GD only get *close* and keep bouncing, unless the learning "
        "rate is gradually reduced.\n\n"
        "For a non-convex cost (any neural network, Part II), different variants "
        "and different random seeds land in **different** minima.")

    exercise(
        5, "Suppose you use batch gradient descent and plot the validation error "
        "at every epoch. If you notice that the validation error consistently goes "
        "up, what is likely going on? How can you fix it?",
        "Two possibilities, distinguished by the *training* error:\n\n"
        "* **Training error is also going up** ⇒ the **learning rate is too "
        "large** and the algorithm is diverging. You have exceeded "
        "$\\eta < 2/\\lambda_{\\max}$. Fix: reduce $\\eta$.\n"
        "* **Training error is going down** ⇒ the model is **overfitting**. Fix: "
        "stop training (early stopping), regularise, or get more data.\n\n"
        "Always plot both curves — one alone cannot distinguish these.")

    exercise(
        6, "Is it a good idea to stop mini-batch gradient descent immediately when "
        "the validation error goes up?",
        "**No.** Mini-batch and stochastic GD are noisy, so the validation curve "
        "wobbles: a single up-tick means nothing. The correct procedure is to "
        "**save the model whenever it beats the best validation error so far**, "
        "keep training with patience (say 50–100 epochs without improvement), then "
        "roll back to the best saved checkpoint. That is exactly what Keras's "
        "`EarlyStopping(patience=..., restore_best_weights=True)` does (Chapter 10).")

    exercise(
        7, "Which gradient descent algorithm will reach the vicinity of the "
        "optimal solution the fastest? Which will actually converge? How can you "
        "make the others converge too?",
        "**Fastest to the vicinity: stochastic GD**, because it updates after every "
        "single instance and so takes far more steps per pass over the data. "
        "Mini-batch is a close second.\n\n"
        "**Actually converges: batch GD only.** SGD and mini-batch bounce around "
        "the optimum forever, because a single instance's gradient is not the "
        "cost's gradient.\n\n"
        "**Fix:** gradually reduce the learning rate on a schedule satisfying the "
        "Robbins–Monro conditions, $\\sum_t \\eta_t = \\infty$ and "
        "$\\sum_t \\eta_t^2 < \\infty$. Then they converge too.")

    exercise(
        8, "Suppose you are using polynomial regression. You plot the learning "
        "curves and notice a large gap between the training error and the "
        "validation error. What is happening? What are three ways to solve it?",
        "The model is **overfitting** — high variance.\n\n"
        "**(1) Reduce the polynomial degree** — a smaller hypothesis space has "
        "less room to fit noise.\n"
        "**(2) Regularise** — add an $\\ell_2$ (ridge) or $\\ell_1$ (lasso) "
        "penalty, which constrains the effective degrees of freedom.\n"
        "**(3) Increase the size of the training set** — the gap is precisely what "
        "more data closes, as the learning-curve animation in §4.4 shows.")

    exercise(
        9, "Suppose you are using ridge regression and you notice that the "
        "training error and the validation error are almost equal and fairly high. "
        "Would you say the model suffers from high bias or high variance? Should "
        "you increase the regularisation hyperparameter α or reduce it?",
        "**High bias** — the two errors being close means there is no variance "
        "problem; both being high means the model is too constrained.\n\n"
        "You should **reduce α**. Increasing it would constrain the model further "
        "and make the underfitting worse.")

    exercise(
        10, "Why would you want to use: (a) ridge regression instead of plain "
        "linear regression? (b) lasso instead of ridge? (c) elastic net instead of "
        "lasso?",
        "**(a) Ridge over plain linear regression:** a model with *some* "
        "regularisation almost always performs better than one with none. Ridge "
        "also guarantees an invertible system — $\\mathbf{X}^\\top\\mathbf{X} + "
        "\\alpha\\mathbf{I}$ has all eigenvalues $\\ge \\alpha > 0$ — so it works "
        "when $m < n$ or features are collinear.\n\n"
        "**(b) Lasso over ridge:** lasso uses an $\\ell_1$ penalty, which drives "
        "the weights of unimportant features **exactly to zero**. You get "
        "automatic feature selection and a sparse, interpretable model. Choose it "
        "when you suspect only a few features matter — if that suspicion is wrong, "
        "lasso will underperform ridge.\n\n"
        "**(c) Elastic net over lasso:** lasso can behave erratically when several "
        "features are strongly correlated (it picks one of the group essentially "
        "at random) or when $n > m$ (it can select at most $m$ features). Elastic "
        "net adds an $\\ell_2$ term that stabilises the selection and lets "
        "correlated groups enter together. Elastic net with `l1_ratio` close to 1 "
        "behaves like lasso but without the pathologies.")

    exercise(
        11, "Suppose you want to classify pictures as outdoor/indoor and "
        "daytime/nighttime. Should you implement two logistic regression "
        "classifiers or one softmax regression classifier?",
        "**Two logistic regression classifiers.** The two attributes are *not "
        "mutually exclusive* — a picture can be indoor *and* nighttime. Softmax "
        "forces $\\sum_k \\hat p_k = 1$, so its classes compete; it cannot express "
        "\"both\".\n\n"
        "This is a **multilabel** problem (§3.8), and multilabel means independent "
        "sigmoids. (You *could* alternatively use one softmax over four combined "
        "classes — indoor-day, indoor-night, outdoor-day, outdoor-night — but that "
        "throws away the factorised structure and needs more data.)")

    exercise(
        12, "Implement batch gradient descent with early stopping for softmax "
        "regression without using scikit-learn.",
        "The complete implementation is below. The pieces you must get right: "
        "**(1)** subtract the row max before `exp` or you will overflow; "
        "**(2)** the gradient is $\\frac{1}{m}\\mathbf{X}^\\top(\\hat{\\mathbf{P}} "
        "- \\mathbf{Y})$ — no softmax derivative terms survive; "
        "**(3)** do not regularise the bias column; "
        "**(4)** keep the *best* parameters, not the last ones.",
        code='''import numpy as np

def softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def fit_softmax(X, y, X_val, y_val, K, eta=0.5, n_epochs=5001,
                alpha=1e-4, patience=200):
    """Batch GD + L2 + early stopping, from scratch."""
    Xb   = np.c_[np.ones(len(X)),     X]
    Xvb  = np.c_[np.ones(len(X_val)), X_val]
    Y    = np.eye(K)[y]
    Yv   = np.eye(K)[y_val]
    Theta = np.random.default_rng(42).normal(0, .01, (K, Xb.shape[1]))

    best_loss, best_Theta, best_epoch, waited = np.inf, None, 0, 0
    history = []
    for epoch in range(n_epochs):
        P = softmax(Xb @ Theta.T)

        # gradient: (P - Y)^T X / m, plus L2 on everything EXCEPT the bias
        grad = (P - Y).T @ Xb / len(Xb)
        reg  = np.c_[np.zeros(K), Theta[:, 1:]]        # column 0 excluded
        Theta = Theta - eta * (grad + alpha * reg)

        Pv = np.clip(softmax(Xvb @ Theta.T), 1e-12, 1.)
        val_loss = -np.mean(np.sum(Yv * np.log(Pv), axis=1))
        history.append(val_loss)

        if val_loss < best_loss - 1e-9:
            best_loss, best_Theta, best_epoch, waited = val_loss, Theta.copy(), epoch, 0
        else:
            waited += 1
            if waited >= patience:
                print(f"early stop at epoch {epoch}")
                break
    return best_Theta, best_loss, best_epoch, history''')

    rule()

    sub("The chapter as one table")

    table(
        ["Model", "Cost function", "Solved by", "Closed form?", "Convex?"],
        [["Linear regression", "MSE",
          "Normal equation / SVD / GD", "✅", "✅"],
         ["Ridge", "MSE $+\\;\\frac{\\alpha}{m}\\lVert\\mathbf{w}\\rVert_2^2$",
          "Closed form / GD", "✅", "✅"],
         ["Lasso", "MSE $+\\;2\\alpha\\lVert\\mathbf{w}\\rVert_1$",
          "Coordinate descent / LARS", "❌", "✅ (not differentiable at 0)"],
         ["Elastic Net", "MSE $+\\;$ both penalties",
          "Coordinate descent", "❌", "✅"],
         ["Logistic regression", "Log loss",
          "Gradient descent / Newton", "❌", "✅"],
         ["Softmax regression", "Cross-entropy",
          "Gradient descent", "❌", "✅"]],
    )

    keypoints([
        "Closed form for small $n$; gradient descent for large $n$ or large $m$.",
        "Always scale before any gradient method or any regularised model.",
        "Learning curves diagnose bias vs variance; the bias–variance "
        "decomposition is exact for squared loss.",
        "Ridge shrinks, lasso zeroes, elastic net does both — the $\\ell_1$ "
        "corners are the whole story.",
        "Logistic and softmax: the gradient is always (prediction − target) × "
        "feature. That identity carries into all of Part II.",
    ], title="Chapter 4 in five lines")

    refs([
        ("Hoerl & Kennard — *Ridge Regression: Biased Estimation for "
         "Nonorthogonal Problems*",
         "https://doi.org/10.1080/00401706.1970.10488634"),
        ("Tibshirani, R. — *Regression Shrinkage and Selection via the Lasso*",
         "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x"),
        ("Zou & Hastie — *Regularization and Variable Selection via the Elastic Net*",
         "https://doi.org/10.1111/j.1467-9868.2005.00503.x"),
        ("Robbins & Monro — *A Stochastic Approximation Method*",
         "https://doi.org/10.1214/aoms/1177729586"),
        ("Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning*, "
         "ch. 3", "Springer, 2009"),
    ])


# ==========================================================================
SECTIONS = [
    ("4.1", "Linear Regression", s_4_1),
    ("4.2", "Gradient Descent", s_4_2),
    ("4.3", "Polynomial Regression", s_4_3),
    ("4.4", "Learning Curves", s_4_4),
    ("4.5", "Regularized Linear Models", s_4_5),
    ("4.6", "Logistic Regression", s_4_6),
    ("4.7", "Softmax Regression", s_4_7),
    ("4.8", "Exercises & Review", s_4_8),
]

nav.render_chapter(CH, SECTIONS)
