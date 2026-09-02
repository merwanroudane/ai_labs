"""Math appendix — the mathematics the nineteen chapters actually use."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, nav
from core.lecture import (anim_header, codenote, derive, figure, hero, idea,
                          keypoints, lead, math, md, note, pitfall, proof,
                          quiz, refs, rule, section, sub, table, tip, warn,
                          where)
from core.palette import C, CLASS_COLORS, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "math"

hero(
    kicker="Reference",
    title="Math appendix",
    blurb=(
        "Not a course — a reference for the specific results the chapters lean "
        "on, each one linked to where it is used. Linear algebra, matrix "
        "calculus, probability, statistics, information theory and convex "
        "optimisation, with every identity verified numerically."
    ),
    chips=["6 sub-sections", "6 animations", "6 code labs",
           "every identity checked"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_m1():
    section("M.1", "Linear Algebra")

    lead(
        "Machine learning is applied linear algebra. These are the results that "
        "actually appear in the chapters, and where."
    )

    sub("Norms")

    math(r"""
    \lVert\mathbf{x}\rVert_p = \Bigl(\sum_i |x_i|^p\Bigr)^{1/p},
    \qquad
    \lVert\mathbf{x}\rVert_1 = \sum_i |x_i|,
    \qquad
    \lVert\mathbf{x}\rVert_2 = \sqrt{\sum_i x_i^2},
    \qquad
    \lVert\mathbf{x}\rVert_\infty = \max_i |x_i|
    """)

    table(
        ["Norm", "Geometry of the unit ball", "Where it appears"],
        [["$\\ell_1$", "A diamond — <b>corners on the axes</b>",
          "Lasso (§4.9): the corners are why it zeroes coefficients"],
         ["$\\ell_2$", "A sphere — no corners",
          "Ridge, weight decay, RMSE (§2.4, §4.9)"],
         ["$\\ell_\\infty$", "A cube",
          "Gradient clipping by value, adversarial robustness"],
         ["Frobenius", "$\\lVert\\mathbf{A}\\rVert_F = "
          "\\sqrt{\\sum_{ij}a_{ij}^2}$",
          "The Eckart–Young theorem (§8.3, §17.1)"],
         ["Spectral", "$\\lVert\\mathbf{A}\\rVert_2 = \\sigma_{\\max}$",
          "Lipschitz bounds, spectral normalisation, §15.2's $\\gamma^T$"]],
    )

    sub("Eigendecomposition and SVD")

    math(r"""
    \mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^{\top}
    \quad\text{(symmetric } \mathbf{A}\text{)},
    \qquad
    \mathbf{A} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^{\top}
    \quad\text{(any } \mathbf{A}\text{)}
    """)

    derive(
        [("<b>The Eckart–Young–Mirsky theorem</b> is the single most-used result "
          "in this platform. Truncate the SVD to its top $k$ singular values:",
          r"\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i \mathbf{u}_i\mathbf{v}_i^{\top}"),
         ("Then $\\mathbf{A}_k$ is the <b>best</b> rank-$k$ approximation in "
          "both the Frobenius and spectral norms:",
          r"\mathbf{A}_k = \arg\min_{\mathrm{rank}(\mathbf{B}) \le k}"
          r"\lVert \mathbf{A} - \mathbf{B}\rVert_F,"
          r"\qquad \lVert\mathbf{A}-\mathbf{A}_k\rVert_F^2 = "
          r"\sum_{i>k}\sigma_i^2"),
         ("<b>This is why PCA works</b> (§8.3): projecting centred data onto the "
          "top $k$ right singular vectors is exactly the rank-$k$ approximation "
          "that loses the least variance.", None),
         ("<b>And why a linear autoencoder is PCA</b> (§17.1): its two weight "
          "matrices form a rank-$d$ product, and the loss is the Frobenius "
          "error.", None),
         ("<b>And the connection to eigenvalues:</b>",
          r"\mathbf{A}^{\top}\mathbf{A} = \mathbf{V}\boldsymbol\Sigma^2"
          r"\mathbf{V}^{\top} \;\Longrightarrow\; \sigma_i^2 = \lambda_i"
          r"\bigl(\mathbf{A}^{\top}\mathbf{A}\bigr)"),
         ("so the covariance matrix's eigenvectors are the data matrix's right "
          "singular vectors — and computing the SVD directly is numerically "
          "far better than forming $\\mathbf{X}^\\top\\mathbf{X}$, whose "
          "condition number is <b>squared</b>.", None)],
        title="Eckart–Young: why low-rank approximation is a spectral question",
    )

    sub("Positive definiteness and conditioning")

    table(
        ["Property", "Test", "Consequence"],
        [["<b>Positive definite</b>", "All $\\lambda_i > 0$",
          "$\\mathbf{x}^\\top\\mathbf{A}\\mathbf{x} > 0$; a unique minimum"],
         ["<b>Positive semi-definite</b>", "All $\\lambda_i \\ge 0$",
          "Every covariance and Gram matrix; a valid kernel (§5.5)"],
         ["<b>Condition number</b>",
          "$\\kappa = \\sigma_{\\max}/\\sigma_{\\min}$",
          "<b>Gradient descent needs $\\mathcal{O}(\\kappa)$ steps</b> (§4.5)"]],
    )

    proof(
        "Feature scaling is a conditioning argument, not a convention",
        "For a quadratic bowl with Hessian $\\mathbf{H}$, gradient descent "
        "converges at a rate governed by $(\\kappa-1)/(\\kappa+1)$ where "
        "$\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$. Features on wildly "
        "different scales make $\\kappa$ enormous, the level sets long thin "
        "ellipses, and the path a zig-zag. Standardising the features makes the "
        "Hessian closer to a multiple of the identity, $\\kappa \\to 1$, and the "
        "path a straight line to the minimum. <b>That is the whole reason "
        "§2.5 insists on it</b> — and why tree models, which are invariant to "
        "monotone feature transforms, do not care.",
    )

    anim_header("Condition number and the zig-zag of gradient descent")

    kappas = [1.0, 2.0, 5.0, 12.0, 30.0, 80.0]
    frames = []
    for kap in kappas:
        H = np.diag([1.0, 1.0/kap])
        eta = 1.8/(H[0, 0] + 1e-9)
        p = np.array([0.9, 0.9*kap*0.35])
        path = [p.copy()]
        for _ in range(60):
            g = H @ p
            p = p - eta*g
            path.append(p.copy())
            if np.linalg.norm(p) < 1e-6:
                break
        path = np.array(path)
        gx = np.linspace(-1.3, 1.3, 120)
        gy = np.linspace(-1.3*kap*0.45, 1.3*kap*0.45, 120)
        GX, GY = np.meshgrid(gx, gy)
        Z = 0.5*(GX**2 + GY**2/kap)
        frames.append(go.Frame(name=f"{kap:g}", data=[
            go.Contour(x=gx, y=gy, z=Z, colorscale=nav.cscale(), opacity=.55,
                       showscale=False, contours=dict(showlines=False)),
            go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers",
                       line=dict(color=C["danger"], width=2.5),
                       marker=dict(size=4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"κ = {kap:g}   ·   {len(path)-1} steps to converge   ·   "
            f"rate (κ−1)/(κ+1) = {(kap-1)/(kap+1):.4f}",
            color=C["success"] if kap < 6 else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=470, xaxis_title="θ₁", yaxis_title="θ₂",
                    title="Gradient descent on a quadratic bowl")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="κ = ")
    figure(f, "Feature scaling is exactly the act of making κ small.")

    code_lab(
        "Norms, SVD, Eckart–Young, and conditioning — all verified",
        '''import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42)

# ============ 1. NORMS =================================================
x = np.array([3.0, -4.0, 0.0, 1.0])
print("=== norms of x =", x, "===")
for p, nm in [(1, "L1  (sum |x|)"), (2, "L2  (Euclidean)"),
              (np.inf, "Linf (max |x|)")]:
    print(f"  {nm:<22} {np.linalg.norm(x, p):.6f}")
print(f"  {'L0 (count non-zero)':<22} {int((x != 0).sum())}"
      f"   -- not a real norm, but that is what L1 approximates")

print()
print("=== why the L1 ball has corners (the lasso argument, 4.9) ===")
theta = np.linspace(0, 2*np.pi, 9)
for p, nm in [(1, "L1"), (2, "L2")]:
    pts = []
    for t in theta[:-1]:
        d = np.array([np.cos(t), np.sin(t)])
        r = 1.0/np.linalg.norm(d, p)          # radius along this direction
        pts.append(r)
    print(f"  {nm}: distance to the unit ball along 8 directions: "
          f"{np.round(pts, 3)}")
print("  the L1 ball is CLOSER along the diagonals and REACHES OUT along")
print("  the axes -- so a constraint set that touches the loss contour")
print("  usually touches it at an axis, where a coefficient is exactly 0.")

# ============ 2. EIGENDECOMPOSITION ====================================
print()
print("=== a symmetric matrix decomposes into Q L Q^T ===")
A = rng.normal(0, 1, (4, 4))
S = A @ A.T                                    # symmetric, PSD by construction
lam, Q = np.linalg.eigh(S)
print(f"  eigenvalues: {lam}")
print(f"  all >= 0 (so S is PSD): {bool((lam >= -1e-10).all())}")
print(f"  Q orthogonal: max |Q^T Q - I| = "
      f"{np.abs(Q.T @ Q - np.eye(4)).max():.2e}")
print(f"  reconstruction error: "
      f"{np.abs(Q @ np.diag(lam) @ Q.T - S).max():.2e}")
print(f"  trace  = sum of eigenvalues: {np.trace(S):.6f} vs {lam.sum():.6f}")
print(f"  det    = product:            {np.linalg.det(S):.6f} vs "
      f"{np.prod(lam):.6f}")

# ============ 3. SVD AND ECKART-YOUNG ==================================
print()
print("=== the SVD works for ANY matrix ===")
M = rng.normal(0, 1, (8, 5))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
print(f"  M {M.shape} -> U {U.shape}, s {s.shape}, Vt {Vt.shape}")
print(f"  singular values: {s}")
print(f"  reconstruction error: {np.abs(U @ np.diag(s) @ Vt - M).max():.2e}")

print()
print("=== Eckart-Young: ||A - A_k||_F^2 = sum of the DISCARDED sigma^2 ===")
print(f"{'k':>4}{'||M - M_k||_F':>17}{'sqrt(sum sigma_i^2, i>k)':>28}"
      f"{'variance kept':>16}")
for k in range(1, 6):
    Mk = (U[:, :k]*s[:k]) @ Vt[:k]
    actual = np.linalg.norm(M - Mk, "fro")
    predicted = np.sqrt((s[k:]**2).sum())
    print(f"{k:>4}{actual:>17.6f}{predicted:>28.6f}"
          f"{(s[:k]**2).sum()/(s**2).sum():>16.4f}")

# --- and no other rank-k matrix does better --------------------------
print()
print("  is the truncated SVD really optimal? 2000 random rank-2 matrices:")
best_random = np.inf
for _ in range(2000):
    B = rng.normal(0, .5, (8, 2)) @ rng.normal(0, .5, (2, 5))
    best_random = min(best_random, np.linalg.norm(M - B, "fro"))
M2 = (U[:, :2]*s[:2]) @ Vt[:2]
print(f"    best random rank-2  : {best_random:.6f}")
print(f"    truncated SVD rank-2: {np.linalg.norm(M - M2, 'fro'):.6f}")
print(f"    the SVD wins: {np.linalg.norm(M - M2, 'fro') < best_random}")

# ============ 4. SVD AND PCA ARE THE SAME THING ========================
print()
print("=== PCA = SVD of the CENTRED data (chapter 8) ===")
X = rng.normal(0, 1, (300, 5)) @ rng.normal(0, 1, (5, 5))
Xc = X - X.mean(0)
U2, s2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
cov = Xc.T @ Xc / (len(Xc)-1)
lam2, V2 = np.linalg.eigh(cov)
lam2, V2 = lam2[::-1], V2[:, ::-1]
print(f"  singular values^2/(n-1) : {np.round(s2**2/(len(Xc)-1), 5)}")
print(f"  covariance eigenvalues  : {np.round(lam2, 5)}")
print(f"  identical: {np.allclose(s2**2/(len(Xc)-1), lam2)}")
print(f"  |components| agree      : "
      f"{np.allclose(np.abs(Vt2), np.abs(V2.T), atol=1e-6)}")

from sklearn.decomposition import PCA
p = PCA().fit(X)
print(f"  sklearn explained_variance: {np.round(p.explained_variance_, 5)}")
print()
print("=== why sklearn uses the SVD and never forms X^T X ===")
Xi = rng.normal(0, 1, (200, 4))
Xi[:, 3] = Xi[:, 0] + 1e-7*Xi[:, 3]           # nearly collinear
k_x = np.linalg.cond(Xi)
k_xtx = np.linalg.cond(Xi.T @ Xi)
print(f"  cond(X)     = {k_x:.4e}")
print(f"  cond(X^T X) = {k_xtx:.4e}")
print(f"  ratio       = {k_xtx/k_x:.4e}   ~ cond(X), i.e. it is SQUARED")
print("  forming X^T X DOUBLES the number of digits you lose.")

# ============ 5. CONDITIONING AND GRADIENT DESCENT =====================
print()
print("=== the condition number IS the convergence rate (4.5) ===")
def gd_steps(kappa, tol=1e-6, max_it=100000):
    H = np.array([[1.0, 0.0], [0.0, 1.0/kappa]])
    eta = 2.0/(1.0 + 1.0/kappa) * 0.9          # near-optimal step
    p = np.array([1.0, 1.0])
    for i in range(max_it):
        p = p - eta*(H @ p)
        if np.linalg.norm(p) < tol:
            return i+1
    return max_it

print(f"{'kappa':>9}{'rate (k-1)/(k+1)':>20}{'steps to 1e-6':>16}"
      f"{'predicted':>13}")
for kap in [1, 2, 5, 10, 50, 200, 1000]:
    rate = (kap-1)/(kap+1)
    pred = (np.inf if rate >= 1 else
            (np.log(1e-6)/np.log(rate) if rate > 0 else 1))
    print(f"{kap:>9}{rate:>20.6f}{gd_steps(kap):>16}"
          f"{(f'{pred:.0f}' if np.isfinite(pred) else '-'):>13}")
print("  steps grow LINEARLY in kappa. Scaling features is not tidiness,")
print("  it is the difference between 20 steps and 20 000.")

# --- demonstrate on real data ----------------------------------------
print()
print("=== unscaled vs scaled features, on real data ===")
from core import datasets as _ds
df = _ds.housing()
num = [c for c in df.columns if df[c].dtype.kind in "if"][:6]
D = df[num].fillna(df[num].median()).to_numpy()
Dc = D - D.mean(0)
Ds = Dc/D.std(0)
for nm, Z in [("raw features", Dc), ("standardised", Ds)]:
    H = Z.T @ Z / len(Z)
    ev = np.linalg.eigvalsh(H)
    print(f"  {nm:<16} eigenvalue range [{ev.min():.4e}, {ev.max():.4e}]"
          f"   kappa = {ev.max()/max(ev.min(), 1e-30):.4e}")

# ============ 6. THE PSEUDOINVERSE =====================================
print()
print("=== the pseudoinverse solves least squares even when X^T X is singular ===")
Xs = np.column_stack([np.ones(50), rng.normal(0, 1, 50)])
Xs = np.column_stack([Xs, Xs[:, 1]])          # a DUPLICATE column
ys = rng.normal(0, 1, 50)
print(f"  X has {Xs.shape[1]} columns but rank {np.linalg.matrix_rank(Xs)}")
try:
    np.linalg.inv(Xs.T @ Xs)
    print("  the normal equation worked (unexpected)")
except np.linalg.LinAlgError as e:
    print(f"  normal equation: LinAlgError: {e}")
theta = np.linalg.pinv(Xs) @ ys
print(f"  pinv gives theta = {np.round(theta, 6)}")
print(f"  residual norm    = {np.linalg.norm(Xs @ theta - ys):.6f}")
print(f"  ||theta||        = {np.linalg.norm(theta):.6f}  <- the MINIMUM-NORM")
print("  solution among all the infinitely many that fit equally well.")
print("  that is what pinv computes, via the SVD (chapter 4.3).")

import plotly.graph_objects as go
fig = go.Figure()
ks = np.arange(1, 6)
fig.add_bar(x=ks, y=[np.linalg.norm(M - (U[:, :k]*s[:k]) @ Vt[:k], "fro")
                     for k in ks],
            name="actual ||M - M_k||_F", marker=dict(color=C["primary"]))
fig.add_scatter(x=ks, y=[np.sqrt((s[k:]**2).sum()) for k in ks],
                mode="lines+markers", name="sqrt(sum discarded sigma^2)",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=400, xaxis_title="rank k", yaxis_title="Frobenius error",
                  title="Eckart-Young, verified")
''',
        key="math_linalg",
    )

    keypoints([
        "$\\ell_1$'s <b>corners</b> are why the lasso zeroes coefficients; "
        "$\\ell_2$'s sphere is why ridge only shrinks.",
        "<b>Eckart–Young</b>: the truncated SVD is the best low-rank "
        "approximation — the basis of PCA and linear autoencoders.",
        "$\\sigma_i^2 = \\lambda_i(\\mathbf{X}^\\top\\mathbf{X})$, but forming "
        "$\\mathbf{X}^\\top\\mathbf{X}$ <b>squares the condition number</b>.",
        "Gradient descent needs $\\mathcal{O}(\\kappa)$ steps — <b>feature "
        "scaling is a conditioning argument</b>.",
        "The <b>pseudoinverse</b> gives the minimum-norm least-squares solution "
        "even for a singular design.",
    ])


# ==========================================================================
def s_m2():
    section("M.2", "Matrix Calculus")

    lead(
        "Every gradient in the platform reduces to a handful of layouts and six "
        "identities. Getting the shapes right is most of the battle."
    )

    sub("Layout conventions")

    pitfall(
        "Half of all matrix-calculus errors are a transpose",
        "Two conventions exist. <b>Denominator layout</b> (used here, and by "
        "most ML texts) says $\\partial y/\\partial \\mathbf{x}$ has the same "
        "shape as $\\mathbf{x}$ — so a gradient is a column vector and you can "
        "write $\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - "
        "\\eta\\nabla$. <b>Numerator layout</b> makes it a row vector. Mixing "
        "them produces a transposed answer that is dimensionally plausible and "
        "silently wrong. <b>The reliable check is always shapes</b>: whatever "
        "you differentiate with respect to, the result must have its shape.",
    )

    sub("The identities you actually need")

    table(
        ["$f$", "$\\partial f/\\partial \\mathbf{x}$", "Where it is used"],
        [["$\\mathbf{a}^\\top\\mathbf{x}$", "$\\mathbf{a}$",
          "Every linear layer"],
         ["$\\mathbf{x}^\\top\\mathbf{A}\\mathbf{x}$",
          "$(\\mathbf{A} + \\mathbf{A}^\\top)\\mathbf{x}$; "
          "$2\\mathbf{A}\\mathbf{x}$ if symmetric",
          "Quadratic forms, the normal equation (§4.3)"],
         ["$\\lVert\\mathbf{x}\\rVert_2^2$", "$2\\mathbf{x}$",
          "$\\ell_2$ regularisation (§4.9)"],
         ["$\\lVert\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\rVert_2^2$",
          "$2\\mathbf{A}^\\top(\\mathbf{A}\\mathbf{x} - \\mathbf{b})$",
          "Linear regression — set to 0 for the normal equation"],
         ["$\\mathbf{W}\\mathbf{x}$ w.r.t. $\\mathbf{W}$",
          "outer product $\\bar{\\mathbf{y}}\\mathbf{x}^\\top$",
          "<b>Backprop through every dense layer</b>"],
         ["$\\mathbf{W}\\mathbf{x}$ w.r.t. $\\mathbf{x}$",
          "$\\mathbf{W}^\\top\\bar{\\mathbf{y}}$",
          "Why backprop is 'the same network, transposed'"]],
    )

    derive(
        [("<b>The softmax–cross-entropy gradient</b>, because it looks like it "
          "should be messy and is not. Let "
          "$s_i = e^{z_i}/\\sum_j e^{z_j}$ and "
          "$\\mathcal{L} = -\\sum_i y_i \\log s_i$ with one-hot $\\mathbf{y}$.",
          None),
         ("First the softmax Jacobian. For $i = j$:",
          r"\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i),"
          r"\qquad\text{and for } i \ne j:\quad"
          r"\frac{\partial s_i}{\partial z_j} = -s_i s_j"),
         ("which is compactly $\\partial s_i/\\partial z_j = "
          "s_i(\\delta_{ij} - s_j)$. Now the loss:",
          r"\frac{\partial \mathcal{L}}{\partial z_j}"
          r" = -\sum_i \frac{y_i}{s_i}\,\frac{\partial s_i}{\partial z_j}"
          r" = -\sum_i \frac{y_i}{s_i}\,s_i(\delta_{ij} - s_j)"),
         ("The $s_i$ cancels. Expanding and using $\\sum_i y_i = 1$:",
          r"= -y_j + s_j\sum_i y_i = s_j - y_j"),
         ("<b>So the gradient is just prediction minus target:</b>",
          r"\boxed{\;\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{s} - \mathbf{y}\;}"),
         ("<b>This is why the two are always fused.</b> Computing them "
          "separately means dividing by $s_i$, which overflows when the softmax "
          "saturates; fused, the division never happens. It is exactly what "
          "<code>from_logits=True</code> does, and it is also why the same "
          "clean form appears for sigmoid + binary cross-entropy and for linear "
          "+ squared error — all three are the canonical link for their "
          "exponential family.", None)],
        title="Why softmax + cross-entropy gives ŷ − y",
    )

    anim_header("Backprop through a two-layer net, shapes and all")

    layers = [
        ("x", "(B, 4)", "input", C["accent"]),
        ("z₁ = xW₁ + b₁", "(B, 6)", "W₁ is (4, 6)", C["primary"]),
        ("a₁ = relu(z₁)", "(B, 6)", "element-wise", C["primary"]),
        ("z₂ = a₁W₂ + b₂", "(B, 3)", "W₂ is (6, 3)", C["primary"]),
        ("L = CE(softmax(z₂), y)", "(1,)", "scalar", C["danger"]),
    ]
    back = [
        ("∂L/∂z₂ = s − y", "(B, 3)", "the clean form above", C["danger"]),
        ("∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂", "(6, 3)", "outer product", C["warning"]),
        ("∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ", "(B, 6)", "the TRANSPOSE", C["warning"]),
        ("∂L/∂z₁ = ∂L/∂a₁ ⊙ 1[z₁>0]", "(B, 6)", "relu mask", C["warning"]),
        ("∂L/∂W₁ = xᵀ · ∂L/∂z₁", "(4, 6)", "outer product", C["warning"]),
    ]
    seq = [(n, s, d, c, "forward") for n, s, d, c in layers] + \
          [(n, s, d, c, "backward") for n, s, d, c in back]

    frames = []
    for k in range(1, len(seq)+1):
        shapes, ann = [], []
        for i, (nm, shp, desc, col, ph) in enumerate(seq[:k]):
            row = 0 if ph == "forward" else 1
            x0 = (i if row == 0 else i-5)*2.6
            cur = i == k-1
            shapes.append(go.Scatter(
                x=[x0, x0+2.3, x0+2.3, x0, x0],
                y=[-row*1.5, -row*1.5, -row*1.5+1.0, -row*1.5+1.0, -row*1.5],
                fill="toself",
                fillcolor=alpha(col, .9 if cur else .55),
                line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x0+1.15, y=-row*1.5+.68, text=nm,
                            showarrow=False,
                            font=dict(size=9, color="#fff")))
            ann.append(dict(x=x0+1.15, y=-row*1.5+.3, text=shp,
                            showarrow=False,
                            font=dict(size=9, color="#fff")))
        nm, shp, desc, col, ph = seq[k-1]
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{ph.upper()}   ·   {nm}   ·   {shp}"
                                       f"   ·   {desc}", color=col)])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=340, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.5, 13.5]),
                    yaxis=dict(visible=False, range=[-2.3, 1.6]),
                    annotations=list(frames[0].layout.annotations),
                    title="Every gradient's shape matches what it differentiates")
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="step ")
    figure(f, "Note ∂L/∂a₁ uses W₂ᵀ. Backprop is the forward network with every "
              "matrix transposed — that is the whole content of reverse mode.")

    code_lab(
        "Every identity, checked numerically, then a net trained by hand",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; up = f(x)
        x[i] = old - h; dn = f(x)
        x[i] = old
        g[i] = (up - dn)/(2*h)
        it.iternext()
    return g

def check(name, f, x, analytic, tol=1e-6):
    gn = numeric_grad(f, x.copy())
    rel = (np.linalg.norm(analytic-gn) /
           (np.linalg.norm(analytic)+np.linalg.norm(gn)+1e-30))
    print(f"  {name:<44}{rel:>12.2e}  "
          f"{'OK' if rel < tol else 'MISMATCH'}")

# ============ 1. THE IDENTITIES ========================================
print("=== every identity, gradient-checked ===")
print(f"  {'identity':<44}{'rel error':>12}")
x = rng.normal(0, 1, 5)
a = rng.normal(0, 1, 5)
check("d/dx  a^T x  =  a", lambda v: a @ v, x, a)

A = rng.normal(0, 1, (5, 5))
check("d/dx  x^T A x  =  (A + A^T) x",
      lambda v: v @ A @ v, x, (A + A.T) @ x)

S = A + A.T
check("d/dx  x^T S x  =  2 S x   (S symmetric)",
      lambda v: v @ S @ v, x, 2*S @ x)

check("d/dx  ||x||^2  =  2x", lambda v: v @ v, x, 2*x)
check("d/dx  ||x||_2  =  x/||x||",
      lambda v: np.linalg.norm(v), x, x/np.linalg.norm(x))

B = rng.normal(0, 1, (7, 5)); b = rng.normal(0, 1, 7)
check("d/dx  ||Bx - b||^2  =  2 B^T (Bx - b)",
      lambda v: np.sum((B @ v - b)**2), x, 2*B.T @ (B @ x - b))

W = rng.normal(0, 1, (4, 3)); xin = rng.normal(0, 1, 4)
ybar = rng.normal(0, 1, 3)
check("d/dW  (Wx)·ybar  =  outer(x, ybar)",
      lambda M: (xin @ M) @ ybar, W, np.outer(xin, ybar))

check("d/dx  (Wx)·ybar  =  W ybar",
      lambda v: (v @ W) @ ybar, xin, W @ ybar)

def logsumexp(v):
    m = v.max()
    return m + np.log(np.exp(v - m).sum())
check("d/dx  logsumexp(x)  =  softmax(x)",
      logsumexp, x, np.exp(x - logsumexp(x)))

C_ = rng.normal(0, 1, (5, 5)); C_ = C_ @ C_.T + 5*np.eye(5)
check("d/dx  x^T C^-1 x  =  2 C^-1 x   (C symmetric)",
      lambda v: v @ np.linalg.solve(C_, v), x, 2*np.linalg.solve(C_, x))

# ============ 2. THE SOFTMAX + CROSS-ENTROPY RESULT ====================
print()
print("=== softmax + cross-entropy: the gradient is just s - y ===")
def softmax(z):
    e = np.exp(z - z.max())
    return e/e.sum()

z = rng.normal(0, 2, 5)
y_true = np.zeros(5); y_true[2] = 1.0

def ce(zz):
    return -np.sum(y_true*np.log(softmax(zz) + 1e-300))

s = softmax(z)
analytic = s - y_true
numeric = numeric_grad(ce, z.copy())
print(f"  logits    : {z}")
print(f"  softmax s : {s}")
print(f"  target y  : {y_true}")
print(f"  s - y     : {analytic}")
print(f"  numeric   : {numeric}")
print(f"  max diff  : {np.abs(analytic-numeric).max():.2e}")

# --- the full softmax Jacobian ---------------------------------------
J = np.diag(s) - np.outer(s, s)               # s_i (delta_ij - s_j)
Jn = np.zeros((5, 5))
for i in range(5):
    Jn[i] = numeric_grad(lambda zz: softmax(zz)[i], z.copy())
print()
print(f"  softmax Jacobian J_ij = s_i(delta_ij - s_j)")
print(f"  max |analytic - numeric| = {np.abs(J - Jn).max():.2e}")
print(f"  rows sum to zero: {np.abs(J.sum(1)).max():.2e}   "
      f"(the probabilities must keep summing to 1)")
print(f"  J is symmetric  : {np.allclose(J, J.T)}")

# --- WHY the two are fused -------------------------------------------
print()
print("=== why softmax and cross-entropy are always fused ===")
for scale in [1, 10, 100, 400]:
    zbig = np.array([scale, 0.0, -scale/2, 0.0, 0.0], dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        naive_p = np.exp(zbig)/np.exp(zbig).sum()
        naive_loss = -np.log(naive_p[2])
    stable = logsumexp(zbig) - zbig[2]
    print(f"  logit scale {scale:>4}: naive = {naive_loss:>12}, "
          f"stable (logsumexp) = {stable:>12.4f}")
print("  that is exactly what from_logits=True does.")

# ============ 3. A NETWORK, DIFFERENTIATED BY HAND =====================
print()
print("="*66)
print("A 2-layer network, forward and backward, entirely by hand")
print("="*66)
from core import datasets as _ds
X, yv = _ds.moons(n=400, noise=.22)[:2]
X = ((X - X.mean(0))/X.std(0))
Y = np.eye(2)[yv]                              # one-hot
n_in, n_hid, n_out = 2, 16, 2

W1 = rng.normal(0, np.sqrt(2/n_in), (n_in, n_hid))
b1 = np.zeros(n_hid)
W2 = rng.normal(0, np.sqrt(2/n_hid), (n_hid, n_out))
b2 = np.zeros(n_out)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    m = z2.max(1, keepdims=True)
    e = np.exp(z2 - m)
    s = e/e.sum(1, keepdims=True)
    return z1, a1, z2, s

def loss_of(W1, b1, W2, b2):
    _, _, _, s = forward(X, W1, b1, W2, b2)
    return -np.mean(np.sum(Y*np.log(s + 1e-300), 1))

def backward(X, Y, W1, b1, W2, b2):
    B = len(X)
    z1, a1, z2, s = forward(X, W1, b1, W2, b2)
    dz2 = (s - Y)/B                            # THE CLEAN FORM
    dW2 = a1.T @ dz2                           # outer product, (n_hid, n_out)
    db2 = dz2.sum(0)
    da1 = dz2 @ W2.T                           # THE TRANSPOSE
    dz1 = da1 * (z1 > 0)                       # the relu mask
    dW1 = X.T @ dz1                            # (n_in, n_hid)
    db1 = dz1.sum(0)
    return dW1, db1, dW2, db2

print("  shapes:")
z1, a1, z2, s = forward(X, W1, b1, W2, b2)
for nm, arr in [("X", X), ("z1", z1), ("a1", a1), ("z2", z2), ("s", s)]:
    print(f"    {nm:<4} {arr.shape}")
dW1, db1, dW2, db2 = backward(X, Y, W1, b1, W2, b2)
print("  gradient shapes MATCH the parameters:")
for nm, p, g in [("W1", W1, dW1), ("b1", b1, db1),
                 ("W2", W2, dW2), ("b2", b2, db2)]:
    print(f"    {nm:<4} param {str(p.shape):<10} grad {str(g.shape):<10}"
          f"  {'OK' if p.shape == g.shape else 'MISMATCH'}")

print()
print("  gradient check against finite differences:")
for nm, p, g in [("W1", W1, dW1), ("b1", b1, db1),
                 ("W2", W2, dW2), ("b2", b2, db2)]:
    if nm == "W1":
        gn = numeric_grad(lambda M: loss_of(M, b1, W2, b2), W1.copy())
    elif nm == "b1":
        gn = numeric_grad(lambda v: loss_of(W1, v, W2, b2), b1.copy())
    elif nm == "W2":
        gn = numeric_grad(lambda M: loss_of(W1, b1, M, b2), W2.copy())
    else:
        gn = numeric_grad(lambda v: loss_of(W1, b1, W2, v), b2.copy())
    rel = np.linalg.norm(g-gn)/(np.linalg.norm(g)+np.linalg.norm(gn)+1e-30)
    print(f"    {nm:<4} relative error {rel:.3e}   "
          f"{'OK' if rel < 1e-6 else 'BUG'}")

print()
print("  training it:")
lr = 0.5
for step in range(801):
    dW1, db1, dW2, db2 = backward(X, Y, W1, b1, W2, b2)
    W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    if step % 200 == 0:
        _, _, _, s = forward(X, W1, b1, W2, b2)
        acc = (s.argmax(1) == yv).mean()
        print(f"    step {step:>4}: loss {loss_of(W1,b1,W2,b2):.6f}, "
              f"accuracy {acc:.4f}")

import plotly.graph_objects as go
g1 = np.linspace(-2.6, 2.6, 90)
G1, G2 = np.meshgrid(g1, g1)
_, _, _, sg = forward(np.column_stack([G1.ravel(), G2.ravel()]),
                      W1, b1, W2, b2)
fig = go.Figure()
fig.add_contour(x=g1, y=g1, z=sg[:, 1].reshape(G1.shape),
                colorscale=nav.cscale(), opacity=.65,
                contours=dict(showlines=False))
for c in (0, 1):
    m = yv == c
    fig.add_scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=f"class {c}",
                    marker=dict(size=7, color=CLASS_COLORS[c],
                                line=dict(color="#fff", width=1)))
fig.update_layout(height=480, xaxis_title="x1", yaxis_title="x2",
                  title="Trained with hand-derived gradients")
''',
        key="math_calculus",
    )

    keypoints([
        "Use <b>denominator layout</b>: a gradient has the shape of what it "
        "differentiates. Check shapes first.",
        "$\\partial\\lVert\\mathbf{A}\\mathbf{x}-\\mathbf{b}\\rVert^2/"
        "\\partial\\mathbf{x} = 2\\mathbf{A}^\\top(\\mathbf{A}\\mathbf{x}-"
        "\\mathbf{b})$ — set to 0 for the normal equation.",
        "Weight gradients are <b>outer products</b>; input gradients use "
        "$\\mathbf{W}^\\top$ — backprop is the net, transposed.",
        "<b>Softmax + cross-entropy gives $\\mathbf{s} - \\mathbf{y}$</b>, which "
        "is why they are always fused.",
        "The softmax Jacobian $s_i(\\delta_{ij}-s_j)$ has rows summing to zero "
        "and vanishes when saturated.",
    ])


# ==========================================================================
def s_m3():
    section("M.3", "Probability")

    lead(
        "Enough probability to read the derivations: Bayes, expectation and "
        "variance algebra, the distributions that appear, and the two "
        "inequalities everything rests on."
    )

    sub("Bayes")

    math(r"""
    P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)},
    \qquad
    \underbrace{P(\theta \mid D)}_{\text{posterior}} \;\propto\;
    \underbrace{P(D \mid \theta)}_{\text{likelihood}}
    \underbrace{P(\theta)}_{\text{prior}}
    """)

    idea(
        "Regularisation is a prior, exactly",
        "Maximising the posterior (MAP) gives "
        "$\\arg\\max_\\theta \\log P(D\\mid\\theta) + \\log P(\\theta)$. With a "
        "Gaussian prior $\\theta \\sim \\mathcal{N}(0, \\tau^2)$ the second term "
        "is $-\\lVert\\theta\\rVert_2^2/2\\tau^2$ — <b>ridge regression</b> "
        "(§4.9), with $\\alpha = \\sigma^2/\\tau^2$. With a Laplace prior it is "
        "$-\\lVert\\theta\\rVert_1/b$ — <b>the lasso</b>. The penalty strength is "
        "not a knob you invented; it is the inverse variance of a prior belief "
        "about how large the weights should be.",
    )

    sub("Expectation and variance algebra")

    table(
        ["Identity", "Statement", "Where"],
        [["Linearity",
          "$\\mathbb{E}[aX + bY] = a\\mathbb{E}[X] + b\\mathbb{E}[Y]$ "
          "<b>always</b>", "Everywhere"],
         ["Variance of a sum",
          "$\\mathrm{Var}(X+Y) = \\mathrm{Var}X + \\mathrm{Var}Y + "
          "2\\mathrm{Cov}(X,Y)$",
          "<b>Ensembles (§7.3)</b> — the covariance term is the whole story"],
         ["Scaling", "$\\mathrm{Var}(aX) = a^2\\mathrm{Var}(X)$",
          "Initialisation (§11.1)"],
         ["Shortcut",
          "$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}X)^2$",
          "Batch norm (§11.3)"],
         ["Law of total expectation",
          "$\\mathbb{E}[X] = \\mathbb{E}\\bigl[\\mathbb{E}[X\\mid Y]\\bigr]$",
          "Bias–variance decomposition (§4.4)"],
         ["Law of total variance",
          "$\\mathrm{Var}(X) = \\mathbb{E}[\\mathrm{Var}(X|Y)] + "
          "\\mathrm{Var}(\\mathbb{E}[X|Y])$",
          "Bias–variance again, and §7.3"]],
    )

    derive(
        [("<b>Why averaging $B$ models reduces variance — and why it stops.</b> "
          "Let each model have variance $\\sigma^2$ and pairwise correlation "
          "$\\rho$.", None),
         ("The variance of the average is:",
          r"\mathrm{Var}\Bigl(\frac{1}{B}\sum_{i=1}^{B} f_i\Bigr)"
          r" = \frac{1}{B^2}\Bigl[B\sigma^2 + B(B-1)\rho\sigma^2\Bigr]"),
         ("which simplifies to:",
          r"\boxed{\;\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2\;}"),
         ("The second term vanishes as $B \\to \\infty$; the first <b>does "
          "not</b>. So the achievable variance floor is $\\rho\\sigma^2$, and "
          "<b>decorrelating the models matters more than adding more of "
          "them</b>.", None),
         ("That single formula explains bagging (bootstrap samples lower "
          "$\\rho$), random forests' feature subsampling (lowers $\\rho$ "
          "further), and why a random forest of 500 trees is barely better than "
          "one of 200 (§7.4).", None)],
        title="The ensemble variance formula",
    )

    sub("The two inequalities")

    table(
        ["Inequality", "Statement", "Used for"],
        [["<b>Jensen</b>",
          "$\\varphi$ convex $\\Rightarrow \\varphi(\\mathbb{E}X) \\le "
          "\\mathbb{E}[\\varphi(X)]$",
          "<b>The ELBO</b> (§17.6), the EM algorithm (§9.8), "
          "maximisation bias (§18.7)"],
         ["<b>Cauchy–Schwarz</b>",
          "$|\\langle \\mathbf{u},\\mathbf{v}\\rangle| \\le "
          "\\lVert\\mathbf{u}\\rVert\\lVert\\mathbf{v}\\rVert$",
          "Correlation bounds, gradient-descent convergence proofs"]],
    )

    anim_header("Jensen's inequality, and the gap that is the ELBO")

    xs = np.linspace(0.15, 4.0, 300)
    frames = []
    for spread in np.linspace(0.15, 1.6, 22):
        a, b = 2.0 - spread, 2.0 + spread
        mid = 0.5*(a+b)
        f_mid = -np.log(mid)
        e_f = 0.5*(-np.log(a) + -np.log(b))
        frames.append(go.Frame(name=f"{spread:.2f}", data=[
            go.Scatter(x=xs, y=-np.log(xs), mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=[a, b], y=[-np.log(a), -np.log(b)], mode="lines+markers",
                       line=dict(color=C["danger"], width=2.5),
                       marker=dict(size=10)),
            go.Scatter(x=[mid, mid], y=[f_mid, e_f], mode="lines+markers",
                       line=dict(color=C["success"], width=4),
                       marker=dict(size=11)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"spread = {spread:.2f}   ·   φ(E[X]) = {f_mid:.4f}   ·   "
            f"E[φ(X)] = {e_f:.4f}   ·   gap = {e_f - f_mid:.4f}"
            f"   ·   the gap grows with the spread and is 0 only when X is "
            f"constant")])))

    f = go.Figure(data=[
        go.Scatter(x=xs, y=-np.log(xs), mode="lines", name="φ(x) = −log x "
                                                           "(convex)",
                   line=dict(color=C["primary"], width=3)),
        go.Scatter(x=[1.85, 2.15], y=[-np.log(1.85), -np.log(2.15)],
                   mode="lines+markers", name="chord",
                   line=dict(color=C["danger"], width=2.5)),
        go.Scatter(x=[2, 2], y=[-np.log(2), -np.log(2)], mode="lines+markers",
                   name="Jensen gap",
                   line=dict(color=C["success"], width=4)),
    ])
    f.update_layout(height=440, xaxis_title="x", yaxis_title="φ(x)",
                    title="φ(E[X]) ≤ E[φ(X)] for convex φ",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(180), slider_prefix="spread ")
    figure(f, "The green segment is the Jensen gap. In a VAE it is exactly "
              "$D_{KL}(q \\Vert p(z|x))$ — the distance between the ELBO and "
              "the true log-likelihood.")

    code_lab(
        "Bayes, the ensemble formula, Jensen, and the distributions",
        '''import numpy as np
from scipy import stats
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. BAYES, AND THE BASE-RATE FALLACY ======================
print("=== the base-rate fallacy, numerically ===")
prevalence = 0.001
sensitivity = 0.99              # P(positive | disease)
specificity = 0.99              # P(negative | no disease)
p_pos = sensitivity*prevalence + (1-specificity)*(1-prevalence)
posterior = sensitivity*prevalence/p_pos
print(f"  disease prevalence     : {prevalence:.4f}")
print(f"  test sensitivity       : {sensitivity:.2f}")
print(f"  test specificity       : {specificity:.2f}")
print(f"  P(positive)            : {p_pos:.6f}")
print(f"  P(disease | positive)  : {posterior:.6f}  <-- only "
      f"{posterior:.1%}!")
print("  a 99%-accurate test on a rare condition is mostly false positives,")
print("  because there are 1000x more healthy people to get wrong.")
print("  this is exactly why ACCURACY is the wrong metric on imbalanced")
print("  data (chapter 3.2).")

print()
print(f"{'prevalence':>12}{'P(disease | positive)':>24}")
for prev in [0.0001, 0.001, 0.01, 0.1, 0.5]:
    pp = sensitivity*prev + (1-specificity)*(1-prev)
    print(f"{prev:>12.4f}{sensitivity*prev/pp:>24.4f}")

# ============ 2. REGULARISATION IS A PRIOR =============================
print()
print("=== ridge regression IS a Gaussian prior (MAP estimation) ===")
n, d = 60, 8
Xd = rng.normal(0, 1, (n, d))
true_w = rng.normal(0, 1, d)
sigma_noise = 1.5
yv = Xd @ true_w + rng.normal(0, sigma_noise, n)

tau = 0.7                                      # prior std on the weights
alpha_equiv = sigma_noise**2/tau**2

from sklearn.linear_model import Ridge
ridge_w = Ridge(alpha=alpha_equiv, fit_intercept=False).fit(Xd, yv).coef_
map_w = np.linalg.solve(Xd.T @ Xd + alpha_equiv*np.eye(d), Xd.T @ yv)
print(f"  noise sigma = {sigma_noise}, prior tau = {tau}")
print(f"  equivalent alpha = sigma^2/tau^2 = {alpha_equiv:.4f}")
print(f"  sklearn Ridge coefficients: {ridge_w}")
print(f"  MAP solution by hand      : {map_w}")
print(f"  identical: {np.allclose(ridge_w, map_w)}")
print("  the penalty strength is the INVERSE VARIANCE of a prior belief.")
print("  a Laplace prior would give the lasso instead.")

# ============ 3. THE ENSEMBLE VARIANCE FORMULA =========================
print()
print("="*66)
print("Var(mean of B models) = rho*sigma^2 + (1-rho)/B * sigma^2")
print("="*66)
def simulate(B, rho, sigma=1.0, n_trials=40000):
    """Correlated models via a shared component."""
    shared = rng.normal(0, sigma*np.sqrt(rho), (n_trials, 1))
    indep = rng.normal(0, sigma*np.sqrt(1-rho), (n_trials, B))
    return (shared + indep).mean(1).var()

print(f"{'B':>5}{'rho':>7}{'simulated Var':>16}{'formula':>12}"
      f"{'floor rho*s^2':>16}")
for rho in [0.0, 0.3, 0.7]:
    for B in [1, 5, 25, 200]:
        pred = rho + (1-rho)/B
        print(f"{B:>5}{rho:>7.1f}{simulate(B, rho):>16.5f}{pred:>12.5f}"
              f"{rho:>16.3f}")
print("  as B -> infinity the variance approaches rho*sigma^2 and STOPS.")
print("  DECORRELATING the models matters more than adding more of them --")
print("  which is exactly what bagging and feature subsampling do (7.4).")

# ============ 4. JENSEN'S INEQUALITY ===================================
print()
print("=== Jensen: phi(E[X]) <= E[phi(X)] for convex phi ===")
samples = rng.gamma(2.0, 1.5, 200000)
print(f"{'phi':<22}{'phi(E[X])':>14}{'E[phi(X)]':>14}{'gap':>12}"
      f"{'convex?':>10}")
for nm, phi, convex in [("x^2", lambda v: v**2, True),
                        ("exp(x/4)", lambda v: np.exp(v/4), True),
                        ("-log(x)", lambda v: -np.log(v+1e-12), True),
                        ("log(x)", lambda v: np.log(v+1e-12), False),
                        ("sqrt(x)", lambda v: np.sqrt(v), False)]:
    a = phi(samples.mean()); b = phi(samples).mean()
    print(f"{nm:<22}{a:>14.5f}{b:>14.5f}{b-a:>12.5f}"
          f"{('convex' if convex else 'CONCAVE'):>10}")
print("  convex -> gap >= 0; concave -> gap <= 0. That sign is what makes")
print("  the ELBO a LOWER bound on the log-likelihood (17.6).")

print()
print("=== the same inequality causes maximisation bias in Q-learning ===")
n_actions = 8
print(f"  {n_actions} actions, all truly worth 0, noisy estimates:")
print(f"{'noise sigma':>13}{'E[max_a Q]':>14}{'max_a E[Q] (truth)':>21}"
      f"{'sigma*sqrt(2 ln n)':>21}")
for sg in [0.25, 0.5, 1.0, 2.0]:
    q = rng.normal(0, sg, (60000, n_actions))
    print(f"{sg:>13.2f}{q.max(1).mean():>14.5f}{0.0:>21.1f}"
          f"{sg*np.sqrt(2*np.log(n_actions)):>21.5f}")
print("  max() is convex, so E[max] > max[E]. That is Double DQN's whole")
print("  motivation (18.7).")

# ============ 5. THE DISTRIBUTIONS THAT APPEAR =========================
print()
print("=== distributions, and where each one shows up ===")
rows = [
    ("Bernoulli(p)", "binary label", "p", "p(1-p)",
     "logistic regression"),
    ("Binomial(n,p)", "count of successes", "np", "np(1-p)",
     "A/B tests"),
    ("Categorical(p)", "class label", "-", "-", "softmax output"),
    ("Gaussian(mu,s^2)", "continuous", "mu", "s^2",
     "squared loss, initialisation"),
    ("Laplace(mu,b)", "continuous, heavy tails", "mu", "2b^2",
     "MAE loss, the lasso prior"),
    ("Exponential(lam)", "waiting time", "1/lam", "1/lam^2",
     "survival models"),
    ("Poisson(lam)", "count in an interval", "lam", "lam",
     "count regression"),
    ("Beta(a,b)", "a probability", "a/(a+b)", "-",
     "the conjugate prior for Bernoulli"),
]
print(f"  {'distribution':<20}{'support':<26}{'mean':<10}{'variance':<12}"
      f"{'used for':<28}")
for r in rows:
    print(f"  {r[0]:<20}{r[1]:<26}{r[2]:<10}{r[3]:<12}{r[4]:<28}")

# --- verify the moments ----------------------------------------------
print()
print("  moments, checked by simulation:")
checks = [("Bernoulli(0.3)", stats.bernoulli(0.3), 0.3, 0.3*0.7),
          ("Poisson(4)", stats.poisson(4), 4.0, 4.0),
          ("Exponential(2)", stats.expon(scale=1/2), 0.5, 0.25),
          ("Gamma(2,1.5)", stats.gamma(2, scale=1.5), 3.0, 4.5)]
print(f"  {'distribution':<20}{'sample mean':>14}{'theory':>10}"
      f"{'sample var':>14}{'theory':>10}")
for nm, d_, m_, v_ in checks:
    smp = d_.rvs(200000, random_state=0)
    print(f"  {nm:<20}{smp.mean():>14.4f}{m_:>10.4f}{smp.var():>14.4f}"
          f"{v_:>10.4f}")

# ============ 6. THE CENTRAL LIMIT THEOREM =============================
print()
print("=== the CLT is why Gaussians are everywhere ===")
print(f"{'parent distribution':<24}{'n':>5}{'skew of the mean':>19}"
      f"{'excess kurtosis':>18}")
for nm, sampler in [("Exponential(1)", lambda k, n: rng.exponential(1, (k, n))),
                    ("Bernoulli(0.1)", lambda k, n: (rng.random((k, n)) < .1)*1.0),
                    ("Uniform(0,1)", lambda k, n: rng.random((k, n)))]:
    for n in [1, 5, 30, 200]:
        means = sampler(40000, n).mean(1)
        print(f"{nm:<24}{n:>5}{stats.skew(means):>19.4f}"
              f"{stats.kurtosis(means):>18.4f}")
print("  skew and kurtosis both -> 0, which is the Gaussian's signature.")
print("  this is why weight initialisation, batch norm and the squared loss")
print("  all assume approximate normality and get away with it.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("ensemble variance vs B",
                                    "CLT: means of 30 exponentials"))
Bs = np.arange(1, 61)
for rho, col in [(0.0, C["success"]), (0.3, C["warning"]), (0.7, C["danger"])]:
    fig.add_trace(go.Scatter(x=Bs, y=rho + (1-rho)/Bs, mode="lines",
                             name=f"rho = {rho}",
                             line=dict(color=col, width=3)), 1, 1)
means = rng.exponential(1, (40000, 30)).mean(1)
fig.add_trace(go.Histogram(x=means, nbinsx=70, name="sample means",
                           marker=dict(color=C["primary"]),
                           histnorm="probability density"), 1, 2)
gx = np.linspace(means.min(), means.max(), 200)
fig.add_trace(go.Scatter(x=gx, y=stats.norm(1, 1/np.sqrt(30)).pdf(gx),
                         mode="lines", name="the CLT Gaussian",
                         line=dict(color=C["danger"], width=3)), 1, 2)
fig.update_layout(height=420)
''',
        key="math_prob",
    )

    keypoints([
        "<b>Regularisation is a prior</b>: Gaussian ⇒ ridge, Laplace ⇒ lasso, "
        "$\\alpha = \\sigma^2/\\tau^2$.",
        "Ensemble variance is $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$ — "
        "<b>decorrelation beats quantity</b>.",
        "<b>Jensen</b> gives the ELBO, EM, and Q-learning's maximisation bias.",
        "A 99 %-accurate test on a 0.1 % condition is <b>9 % precise</b> — the "
        "base-rate fallacy.",
        "The <b>CLT</b> is why so much of the field can assume approximate "
        "normality and get away with it.",
    ])


# ==========================================================================
def s_m4():
    section("M.4", "Statistics and Estimation")

    lead(
        "The results behind every 'is this improvement real?' question in the "
        "platform."
    )

    sub("Maximum likelihood")

    derive(
        [("Maximum likelihood picks the parameters that make the observed data "
          "most probable:",
          r"\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_i p(x_i \mid \theta)"
          r" = \arg\max_\theta \sum_i \log p(x_i\mid\theta)"),
         ("<b>Every loss function in the platform is a negative "
          "log-likelihood.</b> With Gaussian noise "
          "$y = f_\\theta(x) + \\varepsilon$, "
          "$\\varepsilon \\sim \\mathcal{N}(0,\\sigma^2)$:",
          r"-\log p(y\mid x,\theta) = \frac{(y - f_\theta(x))^2}{2\sigma^2}"
          r" + \log\sqrt{2\pi\sigma^2}"),
         ("The constant does not affect the argmin, so <b>MLE under Gaussian "
          "noise is exactly least squares</b> (§4.1).", None),
         ("With Laplace noise you get $|y - f_\\theta(x)|$ — <b>MAE</b>, which "
          "is why MAE is the robust choice: the Laplace distribution has "
          "heavier tails, so an outlier is less surprising and pulls less.",
          None),
         ("With a Bernoulli likelihood you get "
          "$-y\\log p - (1-y)\\log(1-p)$ — <b>binary cross-entropy</b> (§3.4).",
          None),
         ("<b>The loss is not a design choice you make freely; it encodes an "
          "assumption about the noise.</b> Choosing MSE asserts that the errors "
          "are Gaussian, and when they are not — heavy tails, asymmetric costs "
          "— the estimator is genuinely wrong, not merely suboptimal.", None)],
        title="Every loss is a negative log-likelihood",
    )

    sub("Bias, variance and the trade-off")

    math(r"""
    \mathbb{E}\bigl[(y - \hat f(x))^2\bigr] =
      \underbrace{\bigl(\mathbb{E}[\hat f(x)] - f(x)\bigr)^2}_{\text{bias}^2}
      + \underbrace{\mathrm{Var}\bigl(\hat f(x)\bigr)}_{\text{variance}}
      + \underbrace{\sigma^2}_{\text{irreducible}}
    """)

    sub("Confidence intervals and testing")

    table(
        ["Quantity", "Interval", "Note"],
        [["A mean", "$\\bar x \\pm t_{n-1,\\alpha/2}\\, s/\\sqrt{n}$",
          "$t$, not $z$, unless $n$ is large"],
         ["A proportion",
          "$\\hat p \\pm z\\sqrt{\\hat p(1-\\hat p)/n}$",
          "<b>Fails near 0 or 1</b> — use Wilson instead"],
         ["Anything", "<b>Bootstrap percentiles</b>",
          "No distributional assumption; the default when in doubt"],
         ["A difference of paired models",
          "Paired $t$ or bootstrap on the <b>differences</b>",
          "Pairing removes the fold-to-fold variance"]],
    )

    warn(
        "Cross-validation folds are not independent, so the naive t-test is "
        "anti-conservative",
        "The $k$ training sets overlap heavily — with 5-fold CV, any two share "
        "75 % of their data — so the fold scores are positively correlated and "
        "their sample variance <b>underestimates</b> the true variance. A "
        "standard paired $t$-test on CV folds therefore declares significance "
        "far too often. Nadeau & Bengio's corrected resampled $t$-test inflates "
        "the variance by $(1/k + n_{\\text{test}}/n_{\\text{train}})$; the "
        "5×2-fold CV test avoids the problem differently. <b>At minimum, report "
        "the fold-to-fold standard deviation and treat differences smaller than "
        "it as noise.</b>",
    )

    anim_header("The bootstrap: resampling to get an interval")

    rng = np.random.default_rng(1)
    data = rng.gamma(2.0, 1.4, 60)
    true_med = float(np.median(data))
    boots = []
    frames = []
    for k in range(1, 61):
        for _ in range(20):
            boots.append(float(np.median(rng.choice(data, len(data),
                                                    replace=True))))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        frames.append(go.Frame(name=str(k*20), data=[
            go.Histogram(x=boots, nbinsx=40,
                         marker=dict(color=alpha(C["primary"], .8))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{len(boots)} bootstrap resamples   ·   95 % interval "
            f"[{lo:.4f}, {hi:.4f}]   ·   width {hi-lo:.4f}   ·   "
            f"sample median {true_med:.4f}")])))

    f = go.Figure(data=[go.Histogram(x=boots[:20], nbinsx=40,
                                     marker=dict(color=alpha(C["primary"],
                                                             .8)))])
    f.add_vline(x=true_med, line_dash="dash", line_color=C["danger"],
                annotation_text="sample median")
    f.update_layout(height=420, xaxis_title="bootstrap median",
                    yaxis_title="count",
                    title="The bootstrap distribution of the median")
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="n = ")
    figure(f, "No formula for the median's standard error is needed. Resample, "
              "recompute, and read the percentiles off.")

    code_lab(
        "MLE, the bias–variance decomposition, bootstrap, and CV testing",
        '''import numpy as np
from scipy import stats
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. EVERY LOSS IS A NEGATIVE LOG-LIKELIHOOD ===============
print("=== the loss encodes an assumption about the noise ===")
n = 400
x = rng.uniform(-3, 3, n)
true_f = lambda t: 1.5*t + 0.5

for noise_name, noise in [("Gaussian(0, 1)", rng.normal(0, 1, n)),
                          ("Laplace(0, 0.7)", rng.laplace(0, 0.7, n)),
                          ("Gaussian + 5% outliers",
                           rng.normal(0, 1, n) +
                           (rng.random(n) < .05)*rng.normal(0, 15, n))]:
    y = true_f(x) + noise
    X = np.column_stack([np.ones(n), x])
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]

    # MAE via iteratively reweighted least squares
    w = w_ols.copy()
    for _ in range(60):
        r = np.abs(y - X @ w)
        W = 1.0/np.maximum(r, 1e-4)
        w = np.linalg.solve((X*W[:, None]).T @ X, (X*W[:, None]).T @ y)

    print(f"  noise {noise_name:<26}")
    print(f"    least squares (Gaussian MLE): slope {w_ols[1]:.4f}, "
          f"error from truth {abs(w_ols[1]-1.5):.4f}")
    print(f"    least ABSOLUTE (Laplace MLE): slope {w[1]:.4f}, "
          f"error from truth {abs(w[1]-1.5):.4f}")
print("  MSE is optimal under Gaussian noise and BADLY WRONG under heavy")
print("  tails. It is not a free choice.")

# ============ 2. THE BIAS-VARIANCE DECOMPOSITION, MEASURED =============
print()
print("="*66)
print("Bias^2 + variance + irreducible noise, on real fits")
print("="*66)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def truth(t): return np.sin(1.6*t) + 0.35*t
SIGMA = 0.32
x_test = np.linspace(-3, 3, 120)
y_test_true = truth(x_test)

print(f"{'degree':>8}{'bias^2':>12}{'variance':>12}{'noise':>10}"
      f"{'total':>12}{'measured MSE':>15}")
for deg in [0, 1, 3, 5, 9, 15]:
    preds = []
    for trial in range(220):
        xs = rng.uniform(-3, 3, 40)
        ys = truth(xs) + rng.normal(0, SIGMA, 40)
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        m.fit(xs[:, None], ys)
        preds.append(m.predict(x_test[:, None]))
    P = np.array(preds)
    bias2 = float(np.mean((P.mean(0) - y_test_true)**2))
    var = float(np.mean(P.var(0)))
    total = bias2 + var + SIGMA**2
    y_noisy = y_test_true + rng.normal(0, SIGMA, (220, len(x_test)))
    measured = float(np.mean((P - y_noisy)**2))
    print(f"{deg:>8}{bias2:>12.5f}{var:>12.5f}{SIGMA**2:>10.5f}"
          f"{total:>12.5f}{measured:>15.5f}")
print("  degree 0 is all BIAS. degree 15 is all VARIANCE.")
print("  the irreducible term never moves -- no model can beat it (4.4).")

# ============ 3. THE BOOTSTRAP =========================================
print()
print("=== the bootstrap needs no formula ===")
sample = rng.gamma(2.0, 1.4, 60)

def bootstrap_ci(data, stat, n_boot=8000, alpha=0.05, seed=0):
    r = np.random.default_rng(seed)
    boots = np.array([stat(r.choice(data, len(data), replace=True))
                      for _ in range(n_boot)])
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)]), boots

print(f"{'statistic':<24}{'estimate':>12}{'95% interval':>28}{'width':>10}")
for nm, fn in [("mean", np.mean), ("median", np.median),
               ("std", np.std), ("90th percentile",
                                 lambda d: np.percentile(d, 90)),
               ("interquartile range",
                lambda d: np.percentile(d, 75) - np.percentile(d, 25)),
               ("skewness", stats.skew)]:
    (lo, hi), _ = bootstrap_ci(sample, fn)
    print(f"{nm:<24}{fn(sample):>12.4f}"
          f"{f'[{lo:.4f}, {hi:.4f}]':>28}{hi-lo:>10.4f}")
print("  there is no closed-form standard error for the IQR or the 90th")
print("  percentile. The bootstrap does not care.")

# --- and it agrees with theory where theory exists -------------------
(lo, hi), _ = bootstrap_ci(sample, np.mean)
se = sample.std(ddof=1)/np.sqrt(len(sample))
t_lo = sample.mean() - stats.t.ppf(.975, len(sample)-1)*se
t_hi = sample.mean() + stats.t.ppf(.975, len(sample)-1)*se
print()
print(f"  mean, bootstrap : [{lo:.5f}, {hi:.5f}]")
print(f"  mean, t-interval: [{t_lo:.5f}, {t_hi:.5f}]")

# ============ 4. COMPARING TWO MODELS ==================================
print()
print("="*66)
print("Is model B really better than model A?")
print("="*66)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from core import datasets as _ds

Xd, yd = _ds.moons(n=400, noise=.3)[:2]
cv = KFold(10, shuffle=True, random_state=0)
a = cross_val_score(LogisticRegression(), Xd, yd, cv=cv)
b = cross_val_score(RandomForestClassifier(200, random_state=0, n_jobs=-1),
                    Xd, yd, cv=cv)
print(f"  model A (logistic) : {a.mean():.4f} +/- {a.std():.4f}")
print(f"  model B (forest)   : {b.mean():.4f} +/- {b.std():.4f}")
print(f"  difference         : {b.mean()-a.mean():+.4f}")

d = b - a
k = len(d)
t_naive = d.mean()/(d.std(ddof=1)/np.sqrt(k))
p_naive = 2*(1 - stats.t.cdf(abs(t_naive), k-1))
print()
print(f"  naive paired t-test : t = {t_naive:.4f}, p = {p_naive:.5f}")

# --- the CORRECTED test ----------------------------------------------
n_test, n_train = len(Xd)//k, len(Xd)*(k-1)//k
corr = (1/k + n_test/n_train)
t_corr = d.mean()/np.sqrt(corr*d.var(ddof=1))
p_corr = 2*(1 - stats.t.cdf(abs(t_corr), k-1))
print(f"  Nadeau-Bengio corrected: t = {t_corr:.4f}, p = {p_corr:.5f}")
print(f"  the correction factor is (1/k + n_test/n_train) = {corr:.4f}")
print(f"  the naive p-value is {p_corr/max(p_naive,1e-12):.1f}x too small.")
print("  CV folds SHARE training data, so their scores are correlated and")
print("  the naive variance is too small. Declaring significance from a")
print("  naive paired t-test on CV folds is a standard error in the")
print("  literature.")

# --- the assumption-free version -------------------------------------
r = np.random.default_rng(0)
boot_diffs = np.array([r.choice(d, k, replace=True).mean()
                       for _ in range(20000)])
lo_d, hi_d = np.percentile(boot_diffs, [2.5, 97.5])
print()
print(f"  bootstrap 95% CI on the difference: [{lo_d:+.4f}, {hi_d:+.4f}]")
print(f"  excludes zero: {lo_d > 0 or hi_d < 0}")
print("  when in doubt, bootstrap the DIFFERENCES. It makes no")
print("  distributional assumption and it is one line.")

# ============ 5. MULTIPLE COMPARISONS ==================================
print()
print("=== testing 20 models against a baseline ===")
r = np.random.default_rng(3)
n_models = 20
print(f"  all {n_models} models are IDENTICAL to the baseline by construction.")
false_pos_naive, false_pos_bonf, false_pos_bh = 0, 0, 0
TRIALS = 2000
for _ in range(TRIALS):
    ps = []
    for _ in range(n_models):
        diff = r.normal(0, .04, 10)
        t_ = diff.mean()/(diff.std(ddof=1)/np.sqrt(10))
        ps.append(2*(1-stats.t.cdf(abs(t_), 9)))
    ps = np.array(ps)
    false_pos_naive += (ps < .05).any()
    false_pos_bonf += (ps < .05/n_models).any()
    srt = np.sort(ps)
    bh = srt <= .05*np.arange(1, n_models+1)/n_models
    false_pos_bh += bh.any()
print(f"  family-wise false-positive rate over {TRIALS} trials:")
print(f"    no correction : {false_pos_naive/TRIALS:.1%}   "
      f"(theory 1 - 0.95^20 = {1-0.95**n_models:.1%})")
print(f"    Bonferroni    : {false_pos_bonf/TRIALS:.1%}")
print(f"    Benjamini-Hochberg: {false_pos_bh/TRIALS:.1%}")
print("  test 20 things at p<0.05 and you will 'find' something 64% of the")
print("  time. This is exactly what happens during hyperparameter search --")
print("  which is why the FINAL number must come from a held-out test set")
print("  you looked at ONCE (chapter 2.7).")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("bias-variance vs polynomial degree",
                                    "bootstrap distribution of the difference"))
degs = [0, 1, 2, 3, 5, 7, 9, 12, 15]
b2, vr = [], []
for deg in degs:
    preds = []
    for _ in range(120):
        xs = rng.uniform(-3, 3, 40)
        ys = truth(xs) + rng.normal(0, SIGMA, 40)
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        m.fit(xs[:, None], ys)
        preds.append(m.predict(x_test[:, None]))
    P = np.array(preds)
    b2.append(float(np.mean((P.mean(0)-y_test_true)**2)))
    vr.append(float(np.mean(P.var(0))))
fig.add_trace(go.Scatter(x=degs, y=b2, mode="lines+markers", name="bias^2",
                         line=dict(color=C["danger"], width=3)), 1, 1)
fig.add_trace(go.Scatter(x=degs, y=vr, mode="lines+markers", name="variance",
                         line=dict(color=C["primary"], width=3)), 1, 1)
fig.add_trace(go.Scatter(x=degs, y=np.array(b2)+np.array(vr)+SIGMA**2,
                         mode="lines+markers", name="total",
                         line=dict(color=C["ink"], width=3)), 1, 1)
fig.add_trace(go.Histogram(x=boot_diffs, nbinsx=60, name="B - A",
                           marker=dict(color=C["accent"])), 1, 2)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_layout(height=430)
''',
        key="math_stats",
    )

    keypoints([
        "<b>Every loss is a negative log-likelihood</b>: Gaussian ⇒ MSE, "
        "Laplace ⇒ MAE, Bernoulli ⇒ cross-entropy.",
        "Bias² + variance + irreducible noise — the third term is a floor no "
        "model can beat.",
        "The <b>bootstrap</b> gives an interval for any statistic with no "
        "distributional assumption.",
        "<b>CV folds are correlated</b>, so the naive paired $t$-test is "
        "anti-conservative; correct it or bootstrap.",
        "Testing 20 things at $p<0.05$ finds something <b>64 %</b> of the time — "
        "hence a held-out test set seen once.",
    ])


# ==========================================================================
def s_m5():
    section("M.5", "Information Theory")

    lead(
        "Entropy, cross-entropy and KL divergence — the three quantities behind "
        "classification losses, decision-tree splits, VAEs and drift detection."
    )

    math(r"""
    H(p) = -\sum_i p_i \log p_i,
    \qquad
    H(p, q) = -\sum_i p_i \log q_i,
    \qquad
    D_{\mathrm{KL}}(p \Vert q) = \sum_i p_i \log\frac{p_i}{q_i}
    """)

    derive(
        [("<b>The relation that makes cross-entropy the right loss.</b> Expand "
          "the KL divergence:",
          r"D_{\mathrm{KL}}(p\Vert q) = \sum_i p_i\log p_i"
          r" - \sum_i p_i \log q_i = H(p,q) - H(p)"),
         ("<b>$H(p)$ does not depend on the model.</b> So minimising "
          "cross-entropy over $q$ is <i>exactly</i> minimising "
          "$D_{\\mathrm{KL}}(p \\Vert q)$ — you are minimising the divergence "
          "between the true label distribution and your prediction, and the "
          "entropy term is an additive constant you can ignore.", None),
         ("<b>Gibbs' inequality</b> says $D_{\\mathrm{KL}} \\ge 0$ with equality "
          "only when $p = q$, so cross-entropy is bounded below by the entropy:",
          r"H(p, q) \;\ge\; H(p),\qquad \text{equality iff } q = p"),
         ("For a one-hot target, $H(p) = 0$, so the minimum achievable "
          "cross-entropy is <b>exactly zero</b> — which is why a training loss "
          "heading to 0 is a memorisation signal, not a success signal.", None),
         ("<b>KL is not symmetric</b>, and the asymmetry matters. "
          "$D_{\\mathrm{KL}}(p\\Vert q)$ is <b>mode-covering</b>: it is infinite "
          "wherever $p > 0$ and $q = 0$, so $q$ must cover all of $p$'s support. "
          "$D_{\\mathrm{KL}}(q\\Vert p)$ is <b>mode-seeking</b>: $q$ can safely "
          "ignore parts of $p$ and concentrate on one mode.", None),
         ("That is why a VAE (which minimises $D_{\\mathrm{KL}}(q\\Vert p)$ in "
          "its ELBO) produces blurry averages, while a GAN's implicit objective "
          "is closer to mode-seeking and produces sharp but incomplete samples "
          "(§17.7). <b>The failure modes of the two model families are visible "
          "in the direction of a KL divergence.</b>", None)],
        title="Cross-entropy, KL, and why the asymmetry matters",
    )

    table(
        ["Quantity", "Meaning", "Where"],
        [["<b>Entropy</b> $H(p)$", "Bits needed to encode samples from $p$",
          "Decision-tree splits (§6.3)"],
         ["<b>Cross-entropy</b> $H(p,q)$",
          "Bits when you encode $p$ using a code built for $q$",
          "<b>Every classification loss</b>"],
         ["<b>KL divergence</b>", "The excess bits — $H(p,q) - H(p)$",
          "VAEs (§17.6), drift (§19.8), distillation (§19.3)"],
         ["<b>Jensen–Shannon</b>",
          "A symmetric, bounded blend of two KLs",
          "The GAN objective (§17.7)"],
         ["<b>Jeffreys</b>", "$D_{KL}(p\\Vert q) + D_{KL}(q\\Vert p)$",
          "<b>PSI</b>, the drift metric (§19.8)"],
         ["<b>Mutual information</b> $I(X;Y)$",
          "$H(X) - H(X\\mid Y)$ — how much $Y$ tells you about $X$",
          "Feature selection, information-gain splits"]],
    )

    note(
        "Gini and entropy almost never disagree",
        "A decision tree can split on entropy $-\\sum p_i\\log p_i$ or on Gini "
        "impurity $1 - \\sum p_i^2$. Gini is the second-order Taylor "
        "approximation of entropy around the uniform distribution, so the two "
        "curves are nearly identical in shape — they differ in less than 2 % of "
        "splits in practice. Gini is slightly faster (no logarithm), which is "
        "why it is scikit-learn's default. <b>This is not a decision worth "
        "tuning</b> (§6.3).",
    )

    anim_header("KL's asymmetry: mode-covering versus mode-seeking")

    xs = np.linspace(-6, 8, 400)

    def gauss(x, m, s):
        return np.exp(-(x-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi))

    p = 0.5*gauss(xs, -2.0, 0.8) + 0.5*gauss(xs, 4.0, 0.9)
    p = p/p.sum()

    frames = []
    for mu in np.linspace(-3.0, 5.5, 34):
        q = gauss(xs, mu, 1.9); q = q/q.sum()
        kl_pq = float(np.sum(p*np.log((p+1e-14)/(q+1e-14))))
        kl_qp = float(np.sum(q*np.log((q+1e-14)/(p+1e-14))))
        frames.append(go.Frame(name=f"{mu:.1f}", data=[
            go.Scatter(x=xs, y=p, mode="lines", fill="tozeroy",
                       line=dict(color=C["primary"], width=3),
                       fillcolor=alpha(C["primary"], .25)),
            go.Scatter(x=xs, y=q, mode="lines",
                       line=dict(color=C["danger"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"q centred at {mu:+.2f}   ·   KL(p‖q) = {kl_pq:.4f} "
            f"(mode-COVERING)   ·   KL(q‖p) = {kl_qp:.4f} "
            f"(mode-SEEKING)")])))

    q0 = gauss(xs, -3.0, 1.9); q0 = q0/q0.sum()
    f = go.Figure(data=[
        go.Scatter(x=xs, y=p, mode="lines", name="p (bimodal, the truth)",
                   fill="tozeroy", line=dict(color=C["primary"], width=3),
                   fillcolor=alpha(C["primary"], .25)),
        go.Scatter(x=xs, y=q0, mode="lines", name="q (unimodal, the model)",
                   line=dict(color=C["danger"], width=3)),
    ])
    f.update_layout(height=430, xaxis_title="x", yaxis_title="density",
                    title="One Gaussian trying to match a bimodal target",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(200), slider_prefix="μ = ")
    figure(f, "KL(p‖q) is minimised by sitting between the modes and covering "
              "both — the blurry VAE answer. KL(q‖p) is minimised by sitting on "
              "one mode — the sharp, incomplete GAN answer.")

    code_lab(
        "Entropy, KL, Gini vs entropy, and the asymmetry that shapes models",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

def entropy(p, base=np.e):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p*np.log(p)).sum()/np.log(base))

def cross_entropy(p, q, base=np.e):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(-(p*np.log(np.maximum(q, 1e-300))).sum()/np.log(base))

def kl(p, q, base=np.e):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m]*np.log(p[m]/np.maximum(q[m], 1e-300))).sum()/np.log(base))

# ============ 1. ENTROPY ===============================================
print("=== entropy measures uncertainty ===")
print(f"{'distribution':<34}{'H (nats)':>12}{'H (bits)':>12}")
for nm, p in [("certain [1, 0, 0, 0]", [1, 0, 0, 0]),
              ("nearly certain [.97,.01,.01,.01]", [.97, .01, .01, .01]),
              ("skewed [.7, .2, .05, .05]", [.7, .2, .05, .05]),
              ("uniform over 4", [.25]*4),
              ("uniform over 8", [.125]*8),
              ("uniform over 256", [1/256]*256)]:
    print(f"{nm:<34}{entropy(p):>12.5f}{entropy(p, 2):>12.5f}")
print(f"  uniform over n has H = log(n): log2(256) = {np.log2(256):.1f} bits")
print("  = exactly the 8 bits you need to encode one of 256 symbols.")

# ============ 2. THE FUNDAMENTAL RELATION ==============================
print()
print("=== H(p,q) = H(p) + KL(p||q) ===")
p = np.array([0.5, 0.3, 0.15, 0.05])
print(f"{'q':<30}{'H(p)':>10}{'KL(p||q)':>12}{'H(p,q)':>10}{'sum':>10}"
      f"{'match':>8}")
for nm, q in [("q = p (perfect)", p),
              ("q = uniform", np.full(4, .25)),
              ("q = [.4,.3,.2,.1]", np.array([.4, .3, .2, .1])),
              ("q = [.05,.15,.3,.5] (backwards)",
               np.array([.05, .15, .3, .5]))]:
    h, k_, ce = entropy(p), kl(p, q), cross_entropy(p, q)
    print(f"{nm:<30}{h:>10.5f}{k_:>12.5f}{ce:>10.5f}{h+k_:>10.5f}"
          f"{str(np.isclose(ce, h+k_)):>8}")
print("  H(p) does not depend on q, so MINIMISING CROSS-ENTROPY IS")
print("  MINIMISING KL. That is the entire justification for the loss.")

print()
print("=== Gibbs: KL >= 0, with equality only when p = q ===")
worst = 0.0
for _ in range(20000):
    a = rng.dirichlet(np.ones(5)); b = rng.dirichlet(np.ones(5))
    worst = min(worst, kl(a, b))
print(f"  minimum KL over 20 000 random pairs: {worst:.2e}   (never negative)")
print(f"  KL(p||p) = {kl(p, p):.2e}")
print(f"  so cross-entropy is bounded below by H(p) = {entropy(p):.5f}")
print(f"  for a ONE-HOT target H(p) = 0, so the minimum loss is exactly 0 --")
print(f"  which is why a training loss reaching 0 means MEMORISATION.")

# ============ 3. KL IS NOT SYMMETRIC ===================================
print()
print("="*66)
print("The asymmetry that shapes VAEs and GANs")
print("="*66)
xs = np.linspace(-8, 10, 800)
def g(x, m, s): return np.exp(-(x-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi))
P = 0.5*g(xs, -2, .8) + 0.5*g(xs, 4, .9); P /= P.sum()

best_fwd = (np.inf, None, None)
best_rev = (np.inf, None, None)
for mu in np.linspace(-4, 6, 90):
    for sd in np.linspace(.4, 5.0, 60):
        Q = g(xs, mu, sd); Q /= Q.sum()
        f_ = kl(P, Q); r_ = kl(Q, P)
        if f_ < best_fwd[0]: best_fwd = (f_, mu, sd)
        if r_ < best_rev[0]: best_rev = (r_, mu, sd)

print(f"  target p is bimodal, with modes at -2 and +4")
print(f"  fitting a SINGLE Gaussian q:")
print(f"    minimising KL(p||q)  -> mu = {best_fwd[1]:+.3f}, "
      f"sigma = {best_fwd[2]:.3f}   MODE-COVERING")
print(f"    minimising KL(q||p)  -> mu = {best_rev[1]:+.3f}, "
      f"sigma = {best_rev[2]:.3f}   MODE-SEEKING")
print()
print("  KL(p||q) is INFINITE wherever p > 0 and q = 0, so q is forced to")
print("  cover ALL of p -- it sits between the modes and is wide. That is")
print("  the BLURRY VAE reconstruction (17.6).")
print("  KL(q||p) lets q ignore parts of p entirely, so it picks ONE mode")
print("  and is narrow. That is the SHARP but incomplete GAN sample (17.7).")
print("  the failure modes of the two families are visible in the DIRECTION")
print("  of a divergence.")

# ============ 4. SYMMETRIC ALTERNATIVES ================================
print()
print("=== the symmetric divergences, and where they are used ===")
A = np.array([.5, .3, .15, .05])
B = np.array([.05, .15, .3, .5])
M = 0.5*(A+B)
js = 0.5*kl(A, M) + 0.5*kl(B, M)
jeff = kl(A, B) + kl(B, A)
psi = float(np.sum((A-B)*np.log(A/B)))
print(f"  KL(A||B)        = {kl(A,B):.6f}")
print(f"  KL(B||A)        = {kl(B,A):.6f}   <- DIFFERENT")
print(f"  Jensen-Shannon  = {js:.6f}   (symmetric, bounded by log 2 = "
      f"{np.log(2):.4f})")
print(f"  Jeffreys        = {jeff:.6f}")
print(f"  PSI             = {psi:.6f}   identical to Jeffreys: "
      f"{np.isclose(psi, jeff)}")
print("  JS is the GAN objective (17.7); Jeffreys is PSI, the standard")
print("  drift metric (19.8).")

# ============ 5. GINI vs ENTROPY =======================================
print()
print("=== decision-tree impurity: Gini vs entropy (6.3) ===")
def gini(p):
    p = np.asarray(p, float)
    return float(1 - (p**2).sum())

ps = np.linspace(.001, .999, 9)
print(f"{'p(class 1)':>12}{'entropy':>11}{'entropy/2':>12}{'Gini':>10}"
      f"{'difference':>13}")
for pp in ps:
    d = np.array([pp, 1-pp])
    print(f"{pp:>12.3f}{entropy(d):>11.5f}{entropy(d)/2:>12.5f}"
          f"{gini(d):>10.5f}{abs(entropy(d)/2 - gini(d)):>13.5f}")
print("  Gini is the second-order Taylor expansion of entropy/2 around")
print("  the uniform point. The two curves are nearly identical.")

# --- do they ever choose different splits? ---------------------------
print()
print("  do they ever disagree about which split is best?")
disagree = 0
TRIALS = 30000
for _ in range(TRIALS):
    n_l, n_r = rng.integers(5, 100, 2)
    pl, pr = rng.random(2)
    left = np.array([pl, 1-pl]); right = np.array([pr, 1-pr])
    w = np.array([n_l, n_r])/(n_l+n_r)
    # a second candidate split
    pl2, pr2 = rng.random(2)
    left2 = np.array([pl2, 1-pl2]); right2 = np.array([pr2, 1-pr2])
    n_l2, n_r2 = rng.integers(5, 100, 2)
    w2 = np.array([n_l2, n_r2])/(n_l2+n_r2)
    e1 = w[0]*entropy(left) + w[1]*entropy(right)
    e2 = w2[0]*entropy(left2) + w2[1]*entropy(right2)
    g1 = w[0]*gini(left) + w[1]*gini(right)
    g2 = w2[0]*gini(left2) + w2[1]*gini(right2)
    if (e1 < e2) != (g1 < g2):
        disagree += 1
print(f"  they disagreed on {disagree}/{TRIALS} = {disagree/TRIALS:.2%} "
      f"of random split pairs.")
print("  this is NOT a hyperparameter worth tuning. Gini is the default")
print("  because it avoids a logarithm.")

# ============ 6. INFORMATION GAIN ======================================
print()
print("=== information gain: what a split actually buys ===")
from core import datasets as _ds
Xd, yd = _ds.moons(n=600, noise=.3)[:2]
H_parent = entropy(np.bincount(yd)/len(yd))
print(f"  parent entropy: {H_parent:.5f} nats")
print(f"{'split on':<14}{'threshold':>11}{'weighted child H':>19}"
      f"{'information gain':>19}")
best = (0, None, None)
for feat in range(2):
    for thr in np.percentile(Xd[:, feat], [10, 25, 50, 75, 90]):
        m = Xd[:, feat] <= thr
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        hl = entropy(np.bincount(yd[m], minlength=2)/m.sum())
        hr = entropy(np.bincount(yd[~m], minlength=2)/(~m).sum())
        wh = m.mean()*hl + (1-m.mean())*hr
        gain = H_parent - wh
        if gain > best[0]:
            best = (gain, feat, thr)
        print(f"{f'x{feat+1}':<14}{thr:>11.4f}{wh:>19.5f}{gain:>19.5f}")
print(f"  best split: x{best[1]+1} <= {best[2]:.4f}, gain {best[0]:.5f}")
print("  a tree greedily picks the split with the largest gain, then")
print("  recurses. That is the whole CART algorithm (6.2).")

import plotly.graph_objects as go
pp = np.linspace(.001, .999, 300)
ent = [entropy([q, 1-q]) for q in pp]
gin = [gini([q, 1-q]) for q in pp]
fig = go.Figure()
fig.add_scatter(x=pp, y=ent, mode="lines", name="entropy",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(x=pp, y=np.array(ent)/2, mode="lines", name="entropy / 2",
                line=dict(color=C["accent"], width=3, dash="dash"))
fig.add_scatter(x=pp, y=gin, mode="lines", name="Gini",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=420, xaxis_title="p(class 1)",
                  yaxis_title="impurity",
                  title="Gini is entropy/2 to second order")
''',
        key="math_info",
    )

    keypoints([
        "$H(p,q) = H(p) + D_{\\mathrm{KL}}(p\\Vert q)$ — minimising "
        "cross-entropy <b>is</b> minimising KL.",
        "$D_{\\mathrm{KL}} \\ge 0$ (Gibbs), so a one-hot target has minimum "
        "loss 0 — reaching it means memorisation.",
        "<b>KL is asymmetric</b>: $D(p\\Vert q)$ is mode-covering (blurry), "
        "$D(q\\Vert p)$ is mode-seeking (sharp).",
        "<b>PSI is the Jeffreys divergence</b>; the GAN objective is "
        "Jensen–Shannon.",
        "Gini is entropy/2 to second order — they disagree on ~2 % of splits, "
        "so it is <b>not worth tuning</b>.",
    ])


# ==========================================================================
def s_m6():
    section("M.6", "Convex Optimisation")

    lead(
        "Why some problems have a guaranteed global optimum and neural networks "
        "do not — and why that turns out to matter far less than it sounds."
    )

    sub("Convexity")

    math(r"""
    f\bigl(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}\bigr)
    \;\le\; \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}),
    \qquad \lambda \in [0,1]
    """)

    table(
        ["Problem", "Convex?", "Consequence"],
        [["Linear / ridge regression", "✅ (quadratic)",
          "<b>Closed form</b>, and gradient descent cannot fail"],
         ["Lasso", "✅ (non-smooth)", "Unique optimum; needs a subgradient"],
         ["Logistic regression", "✅",
          "Global optimum; unbounded if perfectly separable"],
         ["SVM (the dual)", "✅ (QP)", "SMO converges globally"],
         ["k-means", "❌", "Local minima — hence $k$-means++ and restarts"],
         ["A neural network", "❌ <b>very</b>",
          "Local minima, saddles, plateaus — <b>and it works anyway</b>"]],
    )

    derive(
        [("<b>Why non-convexity matters less than it sounds.</b> At a critical "
          "point of a random high-dimensional function, the Hessian's "
          "eigenvalues are roughly symmetric about zero. For a point to be a "
          "<b>local minimum</b>, all $n$ eigenvalues must be positive.", None),
         ("Treating the signs as roughly independent, the probability of that is "
          "about:",
          r"P(\text{local min}) \sim 2^{-n}"),
         ("At $n = 10^6$ parameters this is unimaginably small. <b>Almost every "
          "critical point is a saddle</b>, not a minimum (Dauphin et al., "
          "2014).", None),
         ("<b>And saddles are escapable</b>: a saddle has at least one direction "
          "of negative curvature, so gradient descent with any noise — and SGD "
          "is noisy by construction — will eventually find it and slide off. "
          "The practical symptom is a <i>plateau</i>, not a permanent trap.",
          None),
         ("<b>Moreover, the minima that exist are mostly equivalent.</b> "
          "Empirically, in a large network the local minima that SGD reaches "
          "have very similar loss values. Combined with the "
          "permutation symmetry of hidden units (any relabelling gives an "
          "identical function), the landscape has enormous numbers of "
          "equally-good solutions.", None),
         ("So the honest statement is: <b>non-convexity means no guarantee, not "
          "no result.</b> You lose the theorem; you usually do not lose the "
          "answer.", None)],
        title="Saddle points, not local minima, are the real obstacle",
    )

    sub("Lagrangian duality")

    math(r"""
    \mathcal{L}(\mathbf{x}, \boldsymbol\alpha) =
      f(\mathbf{x}) + \sum_i \alpha_i g_i(\mathbf{x}),
    \qquad
    g_i(\mathbf{x}) \le 0,\;\; \alpha_i \ge 0
    """)

    proof(
        "Duality is what makes the kernel trick possible",
        "The SVM's primal is a minimisation over $\\mathbf{w} \\in "
        "\\mathbb{R}^d$; its dual is a maximisation over $\\boldsymbol\\alpha "
        "\\in \\mathbb{R}^n$ in which the data appears <b>only</b> as inner "
        "products $\\mathbf{x}_i^\\top\\mathbf{x}_j$ (§5.4). Because the "
        "features never appear alone, you can replace every inner product with "
        "a kernel $K(\\mathbf{x}_i, \\mathbf{x}_j)$ and work in an "
        "infinite-dimensional space you never construct. Strong duality holds "
        "here (the problem is convex and Slater's condition is met), so nothing "
        "is lost by solving the dual. <b>Without duality there is no kernel "
        "trick.</b>",
    )

    sub("The KKT conditions")

    table(
        ["Condition", "Statement", "What it means for an SVM"],
        [["Stationarity",
          "$\\nabla f + \\sum_i \\alpha_i \\nabla g_i = \\mathbf{0}$",
          "$\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i$"],
         ["Primal feasibility", "$g_i(\\mathbf{x}) \\le 0$",
          "Every point is correctly classified with margin"],
         ["Dual feasibility", "$\\alpha_i \\ge 0$", "Multipliers are signed"],
         ["<b>Complementary slackness</b>", "$\\alpha_i g_i(\\mathbf{x}) = 0$",
          "<b>$\\alpha_i > 0$ only for support vectors</b> — every other point "
          "is irrelevant"]],
    )

    anim_header("Convex versus non-convex, from many starting points")

    xs = np.linspace(-3.2, 3.2, 400)
    convex = 0.5*xs**2 + 0.3
    nonconvex = 0.32*xs**4 - 1.4*xs**2 + 0.35*xs + 2.0

    rng = np.random.default_rng(0)
    starts = rng.uniform(-3.0, 3.0, 9)
    paths_c, paths_n = [], []
    for s0 in starts:
        p, path = s0, [s0]
        for _ in range(60):
            p = p - 0.12*p
            path.append(p)
        paths_c.append(path)
        p, path = s0, [s0]
        for _ in range(60):
            p = p - 0.035*(1.28*p**3 - 2.8*p + 0.35)
            path.append(p)
        paths_n.append(path)

    frames = []
    for k in range(1, 61):
        pts_c = [pp[k] for pp in paths_c]
        pts_n = [pp[k] for pp in paths_n]
        uniq = len(set(np.round(pts_n, 2)))
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=xs, y=convex, mode="lines",
                       line=dict(color=C["success"], width=3)),
            go.Scatter(x=pts_c, y=[0.5*v**2+0.3 for v in pts_c],
                       mode="markers",
                       marker=dict(size=10, color=C["success"],
                                   line=dict(color="#fff", width=1.5))),
            go.Scatter(x=xs, y=nonconvex, mode="lines", xaxis="x2", yaxis="y2",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=pts_n,
                       y=[0.32*v**4-1.4*v**2+0.35*v+2.0 for v in pts_n],
                       mode="markers", xaxis="x2", yaxis="y2",
                       marker=dict(size=10, color=C["danger"],
                                   line=dict(color="#fff", width=1.5))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {k}   ·   convex: all 9 starts converge to the SAME point"
            f"   ·   non-convex: {uniq} distinct destination(s)")])))

    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("convex — one global minimum",
                                      "non-convex — two basins"))
    f.add_trace(go.Scatter(x=xs, y=convex, mode="lines", showlegend=False,
                           line=dict(color=C["success"], width=3)), 1, 1)
    f.add_trace(go.Scatter(x=starts, y=[0.5*v**2+0.3 for v in starts],
                           mode="markers", showlegend=False,
                           marker=dict(size=10, color=C["success"])), 1, 1)
    f.add_trace(go.Scatter(x=xs, y=nonconvex, mode="lines", showlegend=False,
                           line=dict(color=C["danger"], width=3)), 1, 2)
    f.add_trace(go.Scatter(x=starts,
                           y=[0.32*v**4-1.4*v**2+0.35*v+2.0 for v in starts],
                           mode="markers", showlegend=False,
                           marker=dict(size=10, color=C["danger"])), 1, 2)
    f.update_layout(height=430)
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="step ")
    figure(f, "Convexity is a guarantee about where you end up, not about how "
              "fast you get there.")

    code_lab(
        "Convexity tests, saddle statistics, duality and KKT",
        '''import numpy as np
np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# ============ 1. TESTING CONVEXITY =====================================
print("=== the chord test: f(lam x + (1-lam) y) <= lam f(x) + (1-lam) f(y) ===")
def is_convex(f, lo, hi, n=6000, seed=0):
    r = np.random.default_rng(seed)
    x = r.uniform(lo, hi, n); y = r.uniform(lo, hi, n); lam = r.random(n)
    lhs = f(lam*x + (1-lam)*y)
    rhs = lam*f(x) + (1-lam)*f(y)
    viol = (lhs - rhs > 1e-9)
    return (not viol.any()), float((lhs-rhs).max())

fns = {
    "x^2":                 (lambda t: t**2, -3, 3),
    "|x|":                 (lambda t: np.abs(t), -3, 3),
    "exp(x)":              (lambda t: np.exp(t), -3, 3),
    "-log(x)":             (lambda t: -np.log(np.maximum(t, 1e-9)), .05, 4),
    "x^4 - 3x^2":          (lambda t: t**4 - 3*t**2, -3, 3),
    "sin(x)":              (lambda t: np.sin(t), -3, 3),
    "max(0, x)  (relu)":   (lambda t: np.maximum(0, t), -3, 3),
    "log(1+exp(x))":       (lambda t: np.log1p(np.exp(np.minimum(t, 30))), -3, 3),
    "x*log(x)":            (lambda t: t*np.log(np.maximum(t, 1e-9)), .01, 4),
}
print(f"  {'function':<24}{'convex?':>10}{'max violation':>17}")
for nm, (fn, lo, hi) in fns.items():
    ok, v = is_convex(fn, lo, hi)
    print(f"  {nm:<24}{('YES' if ok else 'no'):>10}{max(v, 0):>17.2e}")

print()
print("=== the second-derivative test ===")
_hdr = "second derivative sign over the range"
print(f"  {'function':<24}{_hdr:>32}")
for nm, (fn, lo, hi) in list(fns.items())[:6]:
    ts = np.linspace(lo, hi, 400)
    h = 1e-4
    d2 = (fn(ts+h) - 2*fn(ts) + fn(ts-h))/h**2
    d2 = d2[np.isfinite(d2)]
    sign = ("always >= 0 (convex)" if (d2 > -1e-3).all()
            else "always <= 0 (concave)" if (d2 < 1e-3).all()
            else "CHANGES SIGN (neither)")
    print(f"  {nm:<24}{sign:>32}")

# ============ 2. CONVEX PROBLEMS HAVE ONE ANSWER =======================
print()
print("=== 40 random starts on a convex and a non-convex loss ===")
from sklearn.linear_model import LogisticRegression
from core import datasets as _ds
Xd, yd = _ds.moons(n=400, noise=.3)[:2]

def logistic_loss(w, X, y):
    z = X @ w[1:] + w[0]
    return float(np.mean(np.log1p(np.exp(-np.where(y == 1, 1, -1)*z))))

def grad_logistic(w, X, y):
    yy = np.where(y == 1, 1.0, -1.0)
    z = X @ w[1:] + w[0]
    s = -yy/(1 + np.exp(yy*z))/len(y)
    return np.concatenate([[s.sum()], X.T @ s])

finals = []
for trial in range(40):
    w = rng.normal(0, 3, 3)
    for _ in range(3000):
        w = w - 0.5*grad_logistic(w, Xd, yd)
    finals.append(logistic_loss(w, Xd, yd))
finals = np.array(finals)
print(f"  logistic regression (CONVEX):")
print(f"    final losses over 40 starts: min {finals.min():.8f}, "
      f"max {finals.max():.8f}")
print(f"    spread {finals.max()-finals.min():.2e}   <- all the SAME optimum")

from sklearn.neural_network import MLPClassifier
nn_losses = []
for seed in range(15):
    m = MLPClassifier((16, 16), max_iter=900, random_state=seed,
                      learning_rate_init=.02)
    m.fit(Xd, yd)
    nn_losses.append(m.loss_)
nn_losses = np.array(nn_losses)
print(f"  a neural network (NON-CONVEX):")
print(f"    final losses over 15 seeds: min {nn_losses.min():.6f}, "
      f"max {nn_losses.max():.6f}")
print(f"    spread {nn_losses.max()-nn_losses.min():.4f}   <- DIFFERENT optima")
print(f"    but the accuracies are: "
      f"{np.round([MLPClassifier((16,16), max_iter=900, random_state=s, learning_rate_init=.02).fit(Xd, yd).score(Xd, yd) for s in range(5)], 4)}")
print("    -- different minima, nearly identical PERFORMANCE. That is the")
print("    empirical fact that makes non-convexity survivable.")

# ============ 3. SADDLES DOMINATE IN HIGH DIMENSIONS ===================
print()
print("="*66)
print("Almost every critical point of a high-dimensional function")
print("is a SADDLE, not a minimum")
print("="*66)
print(f"{'dimension n':>13}{'random Hessians sampled':>26}"
      f"{'fraction that are minima':>27}{'2^-n':>12}")
for n in [1, 2, 5, 10, 20, 40]:
    mins = 0
    TRIALS = 4000
    for _ in range(TRIALS):
        H = rng.normal(0, 1, (n, n)); H = (H + H.T)/2      # GOE
        if (np.linalg.eigvalsh(H) > 0).all():
            mins += 1
    print(f"{n:>13}{TRIALS:>26}{mins/TRIALS:>27.5f}{2.0**-n:>12.2e}")
print("  at n = 1 000 000 parameters the probability of any given critical")
print("  point being a local minimum is 2^-1000000. Effectively ZERO.")
print()
print("  and saddles are ESCAPABLE: they have a direction of negative")
print("  curvature, so any noise eventually finds it. The symptom is a")
print("  PLATEAU, not a permanent trap.")

# --- watch it escape --------------------------------------------------
print()
print("=== escaping a saddle, with and without noise ===")
def saddle_grad(p):
    return np.array([2*p[0], -2*p[1]])            # f = x^2 - y^2

for nm, noise in [("no noise (exactly on the ridge)", 0.0),
                  ("SGD-like noise", 0.02)]:
    p = np.array([0.6, 1e-8])                     # almost exactly at the saddle
    for step in range(400):
        p = p - 0.02*saddle_grad(p) + noise*rng.normal(0, 1, 2)
    print(f"  {nm:<36} final |y| = {abs(p[1]):.6f}")
print("  without noise it sits on the ridge essentially forever.")
print("  with noise it escapes. SGD is noisy BY CONSTRUCTION.")

# ============ 4. LAGRANGIAN DUALITY ====================================
print()
print("="*66)
print("Duality, and why it makes the kernel trick possible")
print("="*66)
# minimise ||x||^2 subject to a^T x = b
a_vec = np.array([2.0, -1.0, 3.0])
b_val = 5.0
print(f"  primal: minimise ||x||^2  subject to  a.x = {b_val}, a = {a_vec}")
x_star = a_vec*b_val/(a_vec @ a_vec)
print(f"  analytic solution x* = a*b/(a.a) = {x_star}")
print(f"  primal value ||x*||^2 = {x_star @ x_star:.6f}")

lam_star = 2*b_val/(a_vec @ a_vec)
dual_val = b_val*lam_star - lam_star**2*(a_vec @ a_vec)/4
print(f"  dual multiplier lambda* = {lam_star:.6f}")
print(f"  dual value              = {dual_val:.6f}")
print(f"  STRONG DUALITY (equal): "
      f"{np.isclose(x_star @ x_star, dual_val)}")

print()
print("=== an SVM's dual: the data appears ONLY as inner products ===")
from sklearn.svm import SVC
Xs, ys = _ds.moons(n=140, noise=.2)[:2]
Xs = (Xs - Xs.mean(0))/Xs.std(0)
svm = SVC(kernel="linear", C=1.0).fit(Xs, ys)
alphas = np.abs(svm.dual_coef_[0])
sv_idx = svm.support_
print(f"  {len(Xs)} training points")
print(f"  {len(sv_idx)} support vectors ({len(sv_idx)/len(Xs):.1%})")
print(f"  the other {len(Xs)-len(sv_idx)} points have alpha = 0 and could be")
print(f"  DELETED without changing the model at all.")

w_from_dual = (svm.dual_coef_[0][:, None]*Xs[sv_idx]).sum(0)
print(f"  w from the primal: {svm.coef_[0]}")
print(f"  w = sum_i alpha_i y_i x_i (KKT stationarity): {w_from_dual}")
print(f"  identical: {np.allclose(svm.coef_[0], w_from_dual)}")

# --- delete the non-support vectors ---------------------------------
mask = np.ones(len(Xs), bool); mask[sv_idx] = False
svm2 = SVC(kernel="linear", C=1.0).fit(Xs[sv_idx], ys[sv_idx])
print(f"  refitting on the SUPPORT VECTORS ONLY:")
print(f"    w = {svm2.coef_[0]}   same: "
      f"{np.allclose(svm.coef_[0], svm2.coef_[0], atol=1e-4)}")

# ============ 5. KKT: COMPLEMENTARY SLACKNESS ==========================
print()
print("=== complementary slackness: alpha_i * g_i(x) = 0 ===")
margins = ys[sv_idx]*2 - 1
dec = svm.decision_function(Xs)
print(f"  {'point':<10}{'alpha':>12}{'y*f(x)':>12}{'on the margin?':>18}")
shown = 0
for i in range(len(Xs)):
    yy = 1 if ys[i] == 1 else -1
    m = yy*dec[i]
    al = 0.0
    if i in sv_idx:
        al = abs(svm.dual_coef_[0][list(sv_idx).index(i)])
    if (al > 1e-6) != (m > 1.001) and shown < 8:
        print(f"  {i:<10}{al:>12.5f}{m:>12.5f}"
              f"{('YES' if abs(m-1) < .05 else 'no'):>18}")
        shown += 1
print("  alpha_i > 0 ONLY where the constraint is TIGHT (y*f(x) = 1).")
print("  every point strictly inside the margin has alpha = 0 and is")
print("  irrelevant. That is complementary slackness, and it is why an")
print("  SVM's model size depends on the number of SUPPORT VECTORS,")
print("  not on the dataset size (5.4).")

import plotly.graph_objects as go
fig = go.Figure()
xs2 = np.linspace(-3.2, 3.2, 400)
fig.add_scatter(x=xs2, y=0.5*xs2**2 + .3, mode="lines", name="convex: x^2/2",
                line=dict(color=C["success"], width=3))
fig.add_scatter(x=xs2, y=0.32*xs2**4 - 1.4*xs2**2 + .35*xs2 + 2.0,
                mode="lines", name="non-convex: x^4 - x^2",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=420, xaxis_title="x", yaxis_title="f(x)",
                  title="One basin, or two")
''',
        key="math_convex",
    )

    rule()

    keypoints([
        "Convexity guarantees a <b>global</b> optimum from any start; linear, "
        "ridge, lasso, logistic and SVM all have it.",
        "In high dimensions, $P(\\text{local min}) \\sim 2^{-n}$ — <b>almost "
        "every critical point is a saddle</b>.",
        "Saddles have a descent direction, and SGD's noise finds it: the symptom "
        "is a <b>plateau</b>, not a trap.",
        "<b>Duality</b> puts the data in inner products only — without it there "
        "is no kernel trick.",
        "<b>Complementary slackness</b> is why only support vectors matter.",
    ], title="The math appendix in five lines")

    refs([
        ("Petersen & Pedersen — *The Matrix Cookbook*",
         "https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf"),
        ("Strang — *Linear Algebra and Learning from Data*",
         "https://math.mit.edu/~gs/learningfromdata/"),
        ("Boyd & Vandenberghe — *Convex Optimization* (free PDF)",
         "https://web.stanford.edu/~boyd/cvxbook/"),
        ("Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* (free)",
         "https://mml-book.github.io/"),
        ("Cover & Thomas — *Elements of Information Theory*",
         "https://doi.org/10.1002/047174882X"),
        ("Dauphin et al. — *Identifying and attacking the saddle point problem*",
         "https://arxiv.org/abs/1406.2572"),
        ("Nadeau & Bengio — *Inference for the Generalization Error*",
         "https://doi.org/10.1023/A:1024068626366"),
    ])


# ==========================================================================
SECTIONS = [
    ("M.1", "Linear Algebra", s_m1),
    ("M.2", "Matrix Calculus", s_m2),
    ("M.3", "Probability", s_m3),
    ("M.4", "Statistics & Estimation", s_m4),
    ("M.5", "Information Theory", s_m5),
    ("M.6", "Convex Optimisation", s_m6),
]

nav.render_chapter(CH, SECTIONS)
