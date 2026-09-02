"""Appendix B — Autodifferentiation."""

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
from core.palette import C, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "autodiff"

hero(
    kicker="Appendix B",
    title="Autodifferentiation",
    blurb=(
        "Four ways to compute a derivative, only one of which scales. This "
        "appendix derives forward and reverse mode from the chain rule, shows "
        "why reverse mode costs one backward pass regardless of the parameter "
        "count, and builds a working autodiff engine in about eighty lines."
    ),
    chips=["4 methods compared", "6 sub-sections", "6 animations",
           "6 code labs", "an engine from scratch"],
)
nav.sidebar_tools(CH)


# ==========================================================================
def s_b1():
    section("B.1", "Four Ways to Get a Derivative")

    lead(
        "Every gradient in this platform came from one of four techniques. "
        "Three of them are unusable at scale, and understanding <i>why</i> is "
        "the point."
    )

    table(
        ["Method", "How", "Exact?", "Cost for $\\nabla f$, $n$ inputs"],
        [["<b>Manual</b>", "Differentiate on paper, implement the result",
          "✅", "Your time — and it is wrong eventually"],
         ["<b>Numerical</b> (finite differences)",
          "$\\bigl(f(x+h) - f(x)\\bigr)/h$",
          "❌ <b>never</b>", "$\\mathcal{O}(n)$ function evaluations"],
         ["<b>Symbolic</b>", "Manipulate the expression algebraically",
          "✅", "Explodes — expression swell"],
         ["<b>Autodiff (reverse)</b>",
          "Apply the chain rule to the computation graph",
          "✅ to machine precision",
          "<b>$\\mathcal{O}(1)$ passes, regardless of $n$</b>"]],
    )

    sub("Why finite differences cannot work")

    derive(
        [("The forward difference approximates $f'(x)$ by:",
          r"D_h f(x) = \frac{f(x+h) - f(x)}{h}"),
         ("Taylor gives the <b>truncation error</b>, which shrinks with $h$:",
          r"D_h f(x) = f'(x) + \frac{h}{2}f''(\xi)"
          r" \;\Longrightarrow\; \varepsilon_{\text{trunc}} = \mathcal{O}(h)"),
         ("But $f(x+h)$ and $f(x)$ are each computed to a relative precision "
          "$\\epsilon_m \\approx 2.2\\times10^{-16}$, and subtracting two nearly "
          "equal numbers amplifies that. The <b>round-off error</b> grows as "
          "$h$ shrinks:",
          r"\varepsilon_{\text{round}} \approx \frac{2\epsilon_m |f(x)|}{h}"),
         ("Minimising the total $\\mathcal{O}(h) + \\mathcal{O}(\\epsilon_m/h)$ "
          "gives the optimal step and the best achievable accuracy:",
          r"h^{\star} \approx \sqrt{\epsilon_m} \approx 1.5\times10^{-8},"
          r"\qquad \varepsilon^{\star} \approx \sqrt{\epsilon_m}"
          r" \approx 10^{-8}"),
         ("<b>So you lose half your significant digits, always.</b> The central "
          "difference does better — $\\mathcal{O}(h^2)$ truncation gives "
          "$h^\\star \\approx \\epsilon_m^{1/3}$ and error "
          "$\\approx 6\\times10^{-11}$ — but it is still not exact, and it "
          "costs <b>two</b> evaluations per input.", None),
         ("<b>And the cost is fatal.</b> A gradient with respect to $n$ "
          "parameters needs $n+1$ (or $2n$) forward passes. For a network with "
          "$n = 10^{7}$, that is ten million forward passes <b>per training "
          "step</b>. Reverse-mode autodiff does it in one.", None)],
        title="The two errors of finite differences, and why they fight",
    )

    warn(
        "Gradient checking is the one legitimate use of finite differences",
        "You should never train with them — but you should absolutely use them "
        "to <b>verify</b> a hand-written gradient. Compare with a central "
        "difference at $h \\approx 10^{-5}$ and check the relative error "
        "$\\lVert g_{\\text{analytic}} - g_{\\text{numeric}}\\rVert / "
        "(\\lVert g_a\\rVert + \\lVert g_n\\rVert)$. Below $10^{-7}$ is fine; "
        "above $10^{-4}$ is a bug. Every from-scratch gradient in this platform "
        "was checked this way.",
    )

    sub("Why symbolic differentiation fails too")

    pitfall(
        "Expression swell",
        "Symbolic differentiation produces a <i>formula</i>, and the formula for "
        "a derivative is generally much larger than the formula for the "
        "function. Differentiate a product of $k$ terms and you get $k$ terms, "
        "each a product of $k$ factors. Iterate that through a 50-layer network "
        "and the expression is astronomically large — even though <b>evaluating "
        "it</b> only ever needs a few thousand operations, because almost every "
        "subexpression repeats. Autodiff exploits exactly that repetition: it "
        "never builds the formula, only the value.",
    )

    anim_header("Finite differences: truncation and round-off fighting")

    hs = 10.0 ** np.arange(-16, 0.1, 0.25)
    x0 = 1.7
    f = lambda x: np.sin(x) * np.exp(x/3)
    fp = lambda x: np.cos(x)*np.exp(x/3) + np.sin(x)*np.exp(x/3)/3
    true = fp(x0)

    fwd = np.abs(((f(x0+hs) - f(x0))/hs - true)/true)
    cen = np.abs(((f(x0+hs) - f(x0-hs))/(2*hs) - true)/true)
    trunc = hs/2*abs(np.sin(x0))
    roundo = 2.2e-16*abs(f(x0))/hs

    frames = []
    for k in range(2, len(hs)+1):
        i = len(hs)-k
        frames.append(go.Frame(name=f"{hs[i]:.0e}", data=[
            go.Scatter(x=hs[i:], y=fwd[i:], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=hs[i:], y=cen[i:], mode="lines",
                       line=dict(color=C["primary"], width=3)),
            go.Scatter(x=hs[i:], y=trunc[i:], mode="lines",
                       line=dict(color=C["muted"], width=1.8, dash="dot")),
            go.Scatter(x=hs[i:], y=roundo[i:], mode="lines",
                       line=dict(color=C["warning"], width=1.8, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"h = {hs[i]:.1e}   ·   forward-difference relative error "
            f"{fwd[i]:.2e}   ·   central {cen[i]:.2e}   ·   "
            + ("truncation dominates" if hs[i] > 1e-8
               else "ROUND-OFF dominates"),
            color=C["danger"] if hs[i] < 1e-10 else C["ink_soft"])])))

    fg = go.Figure(data=[
        go.Scatter(x=hs[-2:], y=fwd[-2:], mode="lines", name="forward difference",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=hs[-2:], y=cen[-2:], mode="lines", name="central difference",
                   line=dict(color=C["primary"], width=3)),
        go.Scatter(x=hs[-2:], y=trunc[-2:], mode="lines",
                   name="truncation O(h)",
                   line=dict(color=C["muted"], width=1.8, dash="dot")),
        go.Scatter(x=hs[-2:], y=roundo[-2:], mode="lines",
                   name="round-off O(ε/h)",
                   line=dict(color=C["warning"], width=1.8, dash="dot")),
    ])
    fg.add_hline(y=1e-16, line_dash="dash", line_color=C["success"],
                 annotation_text="autodiff (machine precision)")
    fg.update_layout(height=470, xaxis_type="log", yaxis_type="log",
                     xaxis_title="step size h", yaxis_title="relative error",
                     title="No step size gives an exact derivative",
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(fg, frames, duration=nav.anim_ms(90), slider_prefix="h = ")
    figure(fg, "The V shape is the whole story: shrinking h reduces truncation "
               "error and increases round-off. Autodiff sits at the green line, "
               "at any h, for free.")

    code_lab(
        "The four methods, side by side, on the same function",
        '''import numpy as np, time

# ============ THE FUNCTION AND ITS EXACT DERIVATIVE ====================
def f(x):
    return np.sin(x) * np.exp(x/3)

def f_prime_exact(x):                       # MANUAL differentiation
    return np.cos(x)*np.exp(x/3) + np.sin(x)*np.exp(x/3)/3

x0 = 1.7
true = f_prime_exact(x0)
print(f"=== f(x) = sin(x) * exp(x/3)  at x = {x0} ===")
print(f"  f(x0)  = {f(x0):.15f}")
print(f"  f'(x0) = {true:.15f}   (by hand)")

# ============ 1. NUMERICAL: THE ERROR FLOOR ============================
print()
print("=== finite differences: no step size is good enough ===")
print(f"{'h':>10}{'forward':>22}{'rel error':>13}{'central':>22}{'rel error':>13}")
for h in [1e-1, 1e-3, 1e-5, 1e-8, 1e-10, 1e-12, 1e-15]:
    fwd = (f(x0+h) - f(x0))/h
    cen = (f(x0+h) - f(x0-h))/(2*h)
    print(f"{h:>10.0e}{fwd:>22.15f}{abs(fwd-true)/abs(true):>13.2e}"
          f"{cen:>22.15f}{abs(cen-true)/abs(true):>13.2e}")

eps = np.finfo(float).eps
print()
print(f"  machine epsilon = {eps:.3e}")
print(f"  forward difference: optimal h = sqrt(eps) = {np.sqrt(eps):.2e},")
print(f"                      best relative error   ~ {np.sqrt(eps):.2e}")
print(f"  central difference: optimal h = eps^(1/3) = {eps**(1/3):.2e},")
print(f"                      best relative error   ~ {eps**(2/3):.2e}")
print("  you lose HALF your significant digits with a forward difference,")
print("  a third with a central one. There is no step size that fixes it.")

# ============ 2. THE COST SCALES WITH THE INPUT COUNT ==================
print()
print("=== the fatal problem is not accuracy, it is COST ===")
def rosenbrock(v):
    return float(np.sum(100.0*(v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2))

def numeric_gradient(fn, v, h=1e-6):
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
        g[i] = (fn(vp) - fn(vm))/(2*h)
    return g

print(f"{'n inputs':>10}{'function evaluations':>23}{'time (s)':>12}")
for n in [10, 100, 1000, 5000]:
    v = np.random.default_rng(0).normal(0, .3, n)
    t0 = time.perf_counter()
    numeric_gradient(rosenbrock, v)
    dt = time.perf_counter()-t0
    print(f"{n:>10}{2*n:>23,}{dt:>12.4f}")
print("  it is LINEAR in the number of inputs. For a 10-million-parameter")
print("  network that is 20 million forward passes PER TRAINING STEP.")
print("  reverse-mode autodiff does the same job in ONE backward pass.")

# ============ 3. SYMBOLIC: EXPRESSION SWELL ============================
print()
print("=== symbolic differentiation: expression swell ===")
try:
    import sympy as sp
    x = sp.Symbol("x")
    expr = sp.sin(x)*sp.exp(x/3)
    print(f"  f(x)  = {expr}")
    print(f"  f'(x) = {sp.simplify(sp.diff(expr, x))}")
    print(f"  at x0 : {float(sp.diff(expr, x).subs(x, x0)):.15f}   EXACT")

    # now iterate a simple nonlinearity, as a deep network does
    print()
    print(f"  {'depth':>7}{'nodes in f':>13}{'nodes in df/dx':>17}{'ratio':>9}")
    g = x
    for depth in range(1, 8):
        g = sp.sin(g)*sp.cos(g)          # one 'layer'
        dg = sp.diff(g, x)
        print(f"  {depth:>7}{sp.count_ops(g):>13}{sp.count_ops(dg):>17}"
              f"{sp.count_ops(dg)/max(1, sp.count_ops(g)):>9.2f}x")
    print("  the derivative EXPRESSION grows much faster than the function.")
    print("  yet EVALUATING it needs almost no extra work, because the same")
    print("  subexpressions appear over and over. Autodiff exploits exactly")
    print("  that: it computes the VALUE and never builds the FORMULA.")
except ImportError:
    print("  (sympy not installed -- the argument still holds:)")
    print("  d/dx [u*v]     = u'v + uv'          -- 2 terms from 1")
    print("  d/dx [u*v*w]   = u'vw + uv'w + uvw' -- 3 terms, 3 factors each")
    print("  iterate that through 50 layers and the expression is")
    print("  astronomically large, even though its VALUE is cheap.")

# ============ 4. AUTODIFF: EXACT, AND O(1) PASSES ======================
print()
print("=== autodiff, for comparison ===")
try:
    import tensorflow as tf
    xv = tf.Variable(x0)
    with tf.GradientTape() as tape:
        y = tf.sin(xv)*tf.exp(xv/3)
    g = tape.gradient(y, xv)
    print(f"  tf.GradientTape : {float(g):.15f}")
    print(f"  exact           : {true:.15f}")
    print(f"  relative error  : {abs(float(g)-true)/abs(true):.2e}   "
          f"(machine precision)")

    print()
    print(f"{'n inputs':>10}{'autodiff time (s)':>20}{'numeric time (s)':>20}"
          f"{'speed-up':>11}")
    for n in [10, 100, 1000, 5000]:
        v = np.random.default_rng(0).normal(0, .3, n).astype("float32")
        tv = tf.Variable(v)
        @tf.function
        def rb(w):
            return tf.reduce_sum(100.0*(w[1:] - w[:-1]**2)**2 + (1-w[:-1])**2)
        with tf.GradientTape() as tape:
            out = rb(tv)
        _ = tape.gradient(out, tv)                 # warm the trace
        t0 = time.perf_counter()
        for _ in range(5):
            with tf.GradientTape() as tape:
                out = rb(tv)
            _ = tape.gradient(out, tv)
        t_ad = (time.perf_counter()-t0)/5
        t0 = time.perf_counter()
        numeric_gradient(rosenbrock, v.astype(float))
        t_num = time.perf_counter()-t0
        print(f"{n:>10}{t_ad:>20.5f}{t_num:>20.5f}{t_num/t_ad:>10.1f}x")
    print("  the autodiff column is essentially FLAT in n. That is the")
    print("  whole reason deep learning is computationally possible.")
except ImportError:
    print("  (TensorFlow not available -- section B.4 builds an engine")
    print("   from scratch in pure numpy, which makes the same point.)")

# ============ 5. GRADIENT CHECKING: THE LEGITIMATE USE =================
print()
print("=== the one thing finite differences ARE for ===")
def check_gradient(fn, grad_fn, v, h=1e-5):
    ga = grad_fn(v)
    gn = numeric_gradient(fn, v, h)
    num = np.linalg.norm(ga - gn)
    den = np.linalg.norm(ga) + np.linalg.norm(gn) + 1e-30
    return num/den, ga, gn

def rosen_grad(v):
    g = np.zeros_like(v)
    g[:-1] += -400*v[:-1]*(v[1:] - v[:-1]**2) - 2*(1 - v[:-1])
    g[1:] += 200*(v[1:] - v[:-1]**2)
    return g

def rosen_grad_buggy(v):
    g = rosen_grad(v)
    g[2] *= 1.01                              # a 1% error, deliberately
    return g

v = np.random.default_rng(1).normal(0, .4, 12)
print(f"{'implementation':<26}{'relative error':>18}{'verdict':>14}")
for nm, gf in [("correct", rosen_grad), ("1% wrong in one entry",
                                         rosen_grad_buggy)]:
    rel, _, _ = check_gradient(rosenbrock, gf, v)
    verdict = "OK" if rel < 1e-7 else "SUSPECT" if rel < 1e-4 else "BUG"
    print(f"{nm:<26}{rel:>18.3e}{verdict:>14}")
print("  < 1e-7 is fine, > 1e-4 is a bug. A 1% error in ONE of 12 entries")
print("  is caught easily. Check every hand-written gradient this way.")

import plotly.graph_objects as go
hs = 10.0**np.arange(-16, 0.1, .2)
fwd_e = np.abs(((f(x0+hs)-f(x0))/hs - true)/true)
cen_e = np.abs(((f(x0+hs)-f(x0-hs))/(2*hs) - true)/true)
fig = go.Figure()
fig.add_scatter(x=hs, y=fwd_e, mode="lines", name="forward",
                line=dict(color=C["danger"], width=3))
fig.add_scatter(x=hs, y=cen_e, mode="lines", name="central",
                line=dict(color=C["primary"], width=3))
fig.add_hline(y=1e-16, line_dash="dash", line_color=C["success"],
              annotation_text="autodiff")
fig.update_layout(height=420, xaxis_type="log", yaxis_type="log",
                  xaxis_title="h", yaxis_title="relative error",
                  title="The finite-difference error floor")
''',
        key="autodiff_methods",
    )

    keypoints([
        "Finite differences have an <b>irreducible error floor</b> of "
        "$\\sqrt{\\epsilon_m} \\approx 10^{-8}$ (forward).",
        "Their cost is $\\mathcal{O}(n)$ evaluations — fatal for a network.",
        "Symbolic differentiation is exact but suffers <b>expression swell</b>.",
        "<b>Autodiff is exact to machine precision and costs "
        "$\\mathcal{O}(1)$ passes.</b>",
        "Use finite differences <b>only</b> to gradient-check a hand-written "
        "derivative.",
    ])


# ==========================================================================
def s_b2():
    section("B.2", "Forward Mode — Dual Numbers")

    lead(
        "Carry the derivative alongside the value through every operation. "
        "It is beautifully simple, exact — and the wrong direction for machine "
        "learning."
    )

    sub("Dual numbers")

    md(
        "Extend the reals with a symbol $\\varepsilon$ satisfying "
        "$\\varepsilon^2 = 0$ (and $\\varepsilon \\ne 0$). A dual number is "
        "$a + b\\varepsilon$."
    )

    derive(
        [("Multiply two dual numbers and the $\\varepsilon^2$ term vanishes:",
          r"(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon"),
         ("<b>That is exactly the product rule.</b> Now evaluate any analytic "
          "$f$ at $a + b\\varepsilon$ using its Taylor series:",
          r"f(a + b\varepsilon) = f(a) + f'(a)\,b\varepsilon"
          r" + \tfrac{1}{2}f''(a)b^2\varepsilon^2 + \dots"),
         ("Every term from $\\varepsilon^2$ onward is zero, so the series "
          "<b>terminates exactly</b>:",
          r"\boxed{\;f(a + b\varepsilon) = f(a) + f'(a)\,b\,\varepsilon\;}"),
         ("So evaluating $f$ at $x + 1\\cdot\\varepsilon$ returns "
          "$f(x) + f'(x)\\varepsilon$ — <b>the value and the derivative, from "
          "one evaluation, with no approximation whatsoever</b>. There is no "
          "step size and no truncation, because the series genuinely stops.",
          None),
         ("Chaining is automatic: $f(g(a + b\\varepsilon)) = "
          "f\\bigl(g(a) + g'(a)b\\varepsilon\\bigr) = f(g(a)) + "
          "f'(g(a))g'(a)b\\varepsilon$ — the chain rule falls out of the "
          "arithmetic.", None)],
        title="Why dual-number arithmetic is exact differentiation",
    )

    sub("The cost")

    derive(
        [("A forward pass with dual numbers propagates the derivative with "
          "respect to <b>one</b> chosen input. To get the full gradient of "
          "$f: \\mathbb{R}^n \\to \\mathbb{R}$ you must run it $n$ times, "
          "seeding $\\varepsilon$ on a different input each time:",
          r"\text{cost} = n \times \text{(one forward pass)}"),
         ("Reverse mode is the mirror image: one backward pass gives the "
          "derivative of <b>one</b> output with respect to <b>all</b> inputs:",
          r"\text{cost} = m \times \text{(one forward + one backward pass)}"),
         ("<b>So the rule is:</b>",
          r"n \ll m \;\Rightarrow\; \text{forward mode};"
          r"\qquad n \gg m \;\Rightarrow\; \text{reverse mode}"),
         ("Machine learning has $n \\approx 10^{7}$ parameters and $m = 1$ "
          "scalar loss. <b>Reverse mode wins by seven orders of magnitude</b>, "
          "which is not a close call.", None),
         ("Forward mode is not useless, though: it is the right choice for a "
          "<b>Jacobian-vector product</b> (a directional derivative), which is "
          "what you need for forward sensitivity analysis and for the "
          "$\\mathbf{Hv}$ products used by second-order optimisers. "
          "<code>tf.autodiff.ForwardAccumulator</code> and "
          "<code>jax.jvp</code> exist for exactly this.", None)],
        title="When forward mode is the right choice",
    )

    note(
        "Forward mode needs no tape",
        "Because the derivative travels <i>with</i> the value, nothing has to be "
        "remembered. Memory is $\\mathcal{O}(1)$ in the graph size, whereas "
        "reverse mode must store every intermediate activation for the backward "
        "pass — which is why activations dominate training memory (§19.4) and "
        "why gradient checkpointing exists.",
    )

    anim_header("A dual number flowing through a computation")

    steps = [
        ("x", "input, seeded with ε", 1.7, 1.0),
        ("u = x²", "u' = 2x·x'", 1.7**2, 2*1.7),
        ("v = sin(u)", "v' = cos(u)·u'", np.sin(1.7**2),
         np.cos(1.7**2)*2*1.7),
        ("w = exp(v)", "w' = exp(v)·v'", np.exp(np.sin(1.7**2)),
         np.exp(np.sin(1.7**2))*np.cos(1.7**2)*2*1.7),
        ("y = w·x", "y' = w'·x + w·x'",
         np.exp(np.sin(1.7**2))*1.7,
         np.exp(np.sin(1.7**2))*np.cos(1.7**2)*2*1.7*1.7
         + np.exp(np.sin(1.7**2))),
    ]
    frames = []
    for k in range(1, len(steps)+1):
        shapes, ann = [], []
        for i, (nm, rule, val, der) in enumerate(steps[:k]):
            cur = i == k-1
            shapes.append(go.Scatter(
                x=[i*2.2, i*2.2+1.8, i*2.2+1.8, i*2.2, i*2.2],
                y=[0, 0, 1.0, 1.0, 0], fill="toself",
                fillcolor=alpha(C["primary"] if cur else C["accent"], .85),
                line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=i*2.2+.9, y=.72, text=nm, showarrow=False,
                            font=dict(size=11, color="#fff")))
            ann.append(dict(x=i*2.2+.9, y=.38,
                            text=f"{val:.4f} + {der:.4f}ε", showarrow=False,
                            font=dict(size=9, color="#fff")))
            if i < k-1:
                shapes.append(go.Scatter(x=[i*2.2+1.8, i*2.2+2.2], y=[.5, .5],
                                         mode="lines",
                                         line=dict(color=C["muted"], width=2.5),
                                         showlegend=False, hoverinfo="skip"))
        nm, rule, val, der = steps[k-1]
        frames.append(go.Frame(name=nm, data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{nm}   ·   {rule}   ·   value "
                                       f"{val:.6f}, derivative {der:.6f}")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=320, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, len(steps)*2.2]),
                    yaxis=dict(visible=False, range=[-.7, 1.5]),
                    annotations=list(frames[0].layout.annotations),
                    title="y = x·exp(sin(x²)), forward mode")
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="step ")
    figure(f, "Both numbers travel together. Nothing is stored, nothing is "
              "approximated, and the last box holds the answer.")

    code_lab(
        "Dual numbers in 40 lines, and the n-forward-passes cost",
        '''import numpy as np, time, math

# ============ 1. THE DUAL NUMBER =======================================
class Dual:
    """a + b*eps, where eps^2 = 0."""
    __slots__ = ("a", "b")

    def __init__(self, a, b=0.0):
        self.a, self.b = float(a), float(b)

    def __repr__(self):
        return f"{self.a:.6f} + {self.b:.6f}eps"

    # --- arithmetic: each rule IS the corresponding derivative rule ---
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a + o.a, self.b + o.b)
    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return Dual(o) - self

    def __mul__(self, o):                      # THE PRODUCT RULE
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a*o.a, self.a*o.b + self.b*o.a)
    __rmul__ = __mul__

    def __truediv__(self, o):                  # THE QUOTIENT RULE
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a/o.a, (self.b*o.a - self.a*o.b)/(o.a*o.a))

    def __rtruediv__(self, o):
        return Dual(o) / self

    def __pow__(self, k):
        return Dual(self.a**k, k*self.a**(k-1)*self.b)

    def __neg__(self):
        return Dual(-self.a, -self.b)

def dsin(x):  return Dual(math.sin(x.a),  math.cos(x.a)*x.b)
def dcos(x):  return Dual(math.cos(x.a), -math.sin(x.a)*x.b)
def dexp(x):  e = math.exp(x.a); return Dual(e, e*x.b)
def dlog(x):  return Dual(math.log(x.a), x.b/x.a)
def dsqrt(x): s = math.sqrt(x.a); return Dual(s, x.b/(2*s))
def dtanh(x): t = math.tanh(x.a); return Dual(t, (1-t*t)*x.b)

# ============ 2. ONE EVALUATION GIVES BOTH =============================
def f(x):
    return x * dexp(dsin(x**2))

x0 = 1.7
out = f(Dual(x0, 1.0))                    # SEED the derivative with 1
print("=== f(x) = x * exp(sin(x^2)) ===")
print(f"  f(1.7 + 1.0eps) = {out}")
print(f"  value      = {out.a:.15f}")
print(f"  derivative = {out.b:.15f}")

exact = (math.exp(math.sin(x0**2))
         + x0*math.exp(math.sin(x0**2))*math.cos(x0**2)*2*x0)
print(f"  by hand    = {exact:.15f}")
print(f"  relative error = {abs(out.b-exact)/abs(exact):.3e}   "
      f"(machine precision -- NOT an approximation)")

# --- compare with a finite difference --------------------------------
def f_plain(x):
    return x*math.exp(math.sin(x*x))
h = 1e-8
fd = (f_plain(x0+h) - f_plain(x0-h))/(2*h)
print(f"  central difference at h=1e-8: {fd:.15f}   "
      f"error {abs(fd-exact)/abs(exact):.3e}")

# ============ 3. WHY eps^2 = 0 IS THE WHOLE TRICK ======================
print()
print("=== the Taylor series TERMINATES ===")
print("  f(a + b*eps) = f(a) + f'(a)*b*eps + f''(a)*b^2*eps^2/2 + ...")
print("               = f(a) + f'(a)*b*eps          <- everything else is 0")
print()
print("  and the product rule falls straight out of the arithmetic:")
u, v = Dual(3.0, 1.0), Dual(5.0, 2.0)
print(f"    ({u}) * ({v})")
print(f"      = {u*v}")
print(f"      real part      : 3*5           = {3*5}")
print(f"      eps part       : 3*2 + 1*5     = {3*2+1*5}   <- u*v' + u'*v")

# ============ 4. THE COST: n FORWARD PASSES ============================
print()
print("=== the gradient of a multi-input function ===")
def g(v):
    """A function of several variables, written with Duals."""
    s = Dual(0.0)
    for i in range(len(v)-1):
        s = s + 100.0*(v[i+1] - v[i]**2)**2 + (1.0 - v[i])**2
    return s

def forward_gradient(fn, xs):
    """One forward pass PER INPUT -- seed eps on a different one each time."""
    n = len(xs)
    grad = np.zeros(n)
    for i in range(n):
        duals = [Dual(x, 1.0 if j == i else 0.0) for j, x in enumerate(xs)]
        grad[i] = fn(duals).b
    return grad

xs = np.random.default_rng(0).normal(0, .3, 6)
gd = forward_gradient(g, xs)
def rosen_grad(v):
    gg = np.zeros_like(v)
    gg[:-1] += -400*v[:-1]*(v[1:] - v[:-1]**2) - 2*(1 - v[:-1])
    gg[1:] += 200*(v[1:] - v[:-1]**2)
    return gg
print(f"  forward-mode gradient : {np.round(gd, 6)}")
print(f"  analytic gradient     : {np.round(rosen_grad(xs), 6)}")
print(f"  max abs difference    : {np.abs(gd - rosen_grad(xs)).max():.3e}")

print()
print(f"{'n inputs':>10}{'forward passes':>17}{'time (s)':>12}"
      f"{'time per input':>17}")
for n in [4, 8, 16, 32, 64]:
    xs = np.random.default_rng(0).normal(0, .3, n)
    t0 = time.perf_counter()
    forward_gradient(g, xs)
    dt = time.perf_counter()-t0
    print(f"{n:>10}{n:>17}{dt:>12.5f}{dt/n:>17.6f}")
print("  cost is LINEAR in the number of inputs, and the per-input cost is")
print("  constant. That is exactly the wrong scaling for a neural network.")

# ============ 5. WHERE FORWARD MODE IS RIGHT ===========================
print()
print("=== forward mode computes a JACOBIAN-VECTOR PRODUCT for free ===")
def h_fn(v):
    """R^3 -> R^2, so m=2 outputs and n=3 inputs."""
    return [v[0]*v[1] + dsin(v[2]),
            dexp(v[0]) - v[1]**2]

x_pt = [1.1, 0.7, 0.3]
direction = [1.0, -2.0, 0.5]              # the vector we want J @ v for
duals = [Dual(a, b) for a, b in zip(x_pt, direction)]
out2 = h_fn(duals)
print(f"  at x = {x_pt}, direction v = {direction}")
print(f"  J @ v = {[round(o.b, 6) for o in out2]}")
print("  ONE forward pass gave a directional derivative of BOTH outputs.")
print("  reverse mode would need one backward pass PER OUTPUT.")
print()
print("  so the rule is:")
print("    few inputs, many outputs  -> FORWARD mode  (n forward passes)")
print("    many inputs, few outputs  -> REVERSE mode  (m backward passes)")
print("  machine learning: n ~ 10^7 parameters, m = 1 scalar loss.")
print("  reverse mode wins by seven orders of magnitude.")

# ============ 6. NO TAPE, NO MEMORY ====================================
print()
print("=== forward mode stores nothing ===")
def deep(x, depth):
    for _ in range(depth):
        x = dtanh(x*1.3 + 0.2)
    return x

for depth in [10, 100, 1000, 10000]:
    r = deep(Dual(0.4, 1.0), depth)
    print(f"  depth {depth:>6}: value {r.a:.8f}, derivative {r.b:.3e}")
print("  memory is O(1) in the depth -- nothing is remembered.")
print("  reverse mode must store EVERY intermediate for the backward pass,")
print("  which is why activations dominate training memory (19.4) and why")
print("  gradient checkpointing exists.")

import plotly.graph_objects as go
ns = [2, 4, 8, 16, 32, 64, 128]
times = []
for n in ns:
    xs = np.random.default_rng(0).normal(0, .3, n)
    t0 = time.perf_counter(); forward_gradient(g, xs)
    times.append(time.perf_counter()-t0)
fig = go.Figure()
fig.add_scatter(x=ns, y=times, mode="lines+markers", name="forward mode",
                line=dict(color=C["danger"], width=3))
fig.add_scatter(x=ns, y=[times[0]]*len(ns), mode="lines",
                name="reverse mode (one pass, whatever n)",
                line=dict(color=C["success"], width=3, dash="dash"))
fig.update_layout(height=400, xaxis_title="number of inputs",
                  yaxis_title="time for the full gradient (s)",
                  title="Forward mode is linear in n")
''',
        key="autodiff_forward",
    )

    keypoints([
        "A dual number $a + b\\varepsilon$ with $\\varepsilon^2 = 0$ makes the "
        "Taylor series <b>terminate exactly</b>.",
        "$f(a + b\\varepsilon) = f(a) + f'(a)b\\varepsilon$ — value and "
        "derivative from one evaluation.",
        "The chain and product rules <b>fall out of the arithmetic</b>; nothing "
        "is approximated.",
        "Cost is $n$ forward passes: right for <b>few inputs, many outputs</b>.",
        "Forward mode needs <b>no tape</b> — $\\mathcal{O}(1)$ memory.",
    ])


# ==========================================================================
def s_b3():
    section("B.3", "Reverse Mode — the One That Scales")

    lead(
        "Run the function forwards recording what happened, then walk the "
        "recording backwards accumulating derivatives. One backward pass gives "
        "the gradient with respect to <b>every</b> input."
    )

    sub("The two passes")

    md(
        "**Forward:** evaluate the expression, storing every intermediate value "
        "and the operation that produced it. That recording is the *tape*.\n\n"
        "**Backward:** walk the tape in reverse, propagating the **adjoint** "
        "$\\bar v = \\partial y / \\partial v$ from the output back to every "
        "node."
    )

    math(r"""
    \bar v \;=\; \frac{\partial y}{\partial v}
    \;=\; \sum_{w \,\in\, \mathrm{children}(v)}
      \bar w \,\frac{\partial w}{\partial v}
    """)

    derive(
        [("<b>Why the adjoints must be summed.</b> If a node $v$ feeds several "
          "later nodes, changing $v$ changes the output through <i>every</i> "
          "path. The multivariable chain rule says those contributions add:",
          r"\frac{\partial y}{\partial v} = \sum_{w} \frac{\partial y}{\partial w}"
          r"\,\frac{\partial w}{\partial v}"),
         ("<b>This is the single most common bug in a hand-written autodiff "
          "engine.</b> Writing <code>v.grad = ...</code> instead of "
          "<code>v.grad += ...</code> silently drops every path but the last "
          "one — and on an expression where each node is used once, the tests "
          "still pass.", None),
         ("<b>Why the backward pass must run in reverse topological order.</b> "
          "$\\bar v$ needs the adjoints of <i>all</i> its children, so every "
          "child must be processed first. A topological sort of the graph, "
          "reversed, guarantees exactly that.", None),
         ("<b>The cost theorem (Baur–Strassen).</b> For "
          "$f: \\mathbb{R}^n \\to \\mathbb{R}$, reverse mode computes the full "
          "gradient in:",
          r"\text{cost}\bigl(\nabla f\bigr) \;\le\; c \cdot \text{cost}(f),"
          r"\qquad c \approx 3\text{–}4"),
         ("<b>Independently of $n$.</b> A gradient with respect to ten million "
          "parameters costs about four function evaluations. That single result "
          "is what makes deep learning computationally possible; without it, "
          "training would cost $n$ forward passes per step.", None),
         ("The price is <b>memory</b>: every intermediate value must be kept "
          "until the backward pass reaches it, so memory is "
          "$\\mathcal{O}(\\text{graph size})$ — which is why activations "
          "dominate training memory (§19.4).", None)],
        title="Reverse mode, and the theorem that makes deep learning possible",
    )

    sub("Gradient checkpointing")

    tip(
        "Trade compute for memory when the tape does not fit",
        "Store only every $\\sqrt{L}$-th layer's activations and <b>recompute</b> "
        "the rest during the backward pass. Memory falls from "
        "$\\mathcal{O}(L)$ to $\\mathcal{O}(\\sqrt{L})$ for roughly a 30 % "
        "increase in compute — Chen et al. (2016). It is what lets a model that "
        "would need 40 GB of activations train on a 16 GB card, and it is one "
        "line in most frameworks (<code>tf.recompute_grad</code>, "
        "<code>torch.utils.checkpoint</code>).",
    )

    anim_header("The tape, forwards then backwards")

    nodes = [("x", 1.7, None), ("y", 0.6, None),
             ("a = x·y", 1.02, "mul"), ("b = sin(x)", 0.9917, "sin"),
             ("c = a + b", 2.0117, "add"), ("L = c²", 4.0470, "sq")]
    adjoints = {"L": 1.0, "c": 2*2.0117, "a": 2*2.0117, "b": 2*2.0117,
                "x": 2*2.0117*0.6 + 2*2.0117*np.cos(1.7),
                "y": 2*2.0117*1.7}
    order_f = list(range(6))
    order_b = list(range(5, -1, -1))

    frames = []
    for phase, order in [("forward", order_f), ("backward", order_b)]:
        for k in range(1, len(order)+1):
            done = set(order[:k])
            shapes, ann = [], []
            for i, (nm, val, op) in enumerate(nodes):
                active = i in done
                cur = i == order[k-1]
                col = (C["primary"] if phase == "forward" else C["danger"])
                shapes.append(go.Scatter(
                    x=[i*2.0, i*2.0+1.6, i*2.0+1.6, i*2.0, i*2.0],
                    y=[0, 0, 1.0, 1.0, 0], fill="toself",
                    fillcolor=(alpha(col, .9) if cur else
                               alpha(C["accent"], .75) if active else
                               alpha(C["line"], .3)),
                    line=dict(color="#fff", width=2),
                    showlegend=False, hoverinfo="skip"))
                ann.append(dict(x=i*2.0+.8, y=.72, text=nm.split("=")[0].strip(),
                                showarrow=False,
                                font=dict(size=11,
                                          color="#fff" if active
                                          else C["ink_soft"])))
                key = nm.split("=")[0].strip()
                if phase == "forward" and active:
                    txt = f"v = {val:.4f}"
                elif phase == "backward" and active:
                    txt = f"grad = {adjoints.get(key, 0):.4f}"
                else:
                    txt = ""
                ann.append(dict(x=i*2.0+.8, y=.35, text=txt, showarrow=False,
                                font=dict(size=9,
                                          color="#fff" if active
                                          else C["ink_soft"])))
                if i < len(nodes)-1:
                    shapes.append(go.Scatter(x=[i*2.0+1.6, i*2.0+2.0],
                                             y=[.5, .5], mode="lines",
                                             line=dict(color=C["muted"],
                                                       width=2),
                                             showlegend=False,
                                             hoverinfo="skip"))
            nm = nodes[order[k-1]][0]
            msg = (f"FORWARD: evaluate {nm} and record it on the tape"
                   if phase == "forward" else
                   f"BACKWARD: adjoint of {nm.split('=')[0].strip()} = "
                   f"{adjoints.get(nm.split('=')[0].strip(), 0):.4f}")
            frames.append(go.Frame(name=f"{phase}{k}", data=shapes,
                                   layout=go.Layout(annotations=ann + [
                                       anim.annotate_step(
                                           msg,
                                           color=(C["primary"]
                                                  if phase == "forward"
                                                  else C["danger"]))])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=330, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, len(nodes)*2.0]),
                    yaxis=dict(visible=False, range=[-.7, 1.5]),
                    annotations=list(frames[0].layout.annotations),
                    title="L = (x·y + sin x)², with x = 1.7, y = 0.6")
    anim.animate(f, frames, duration=nav.anim_ms(750), slider_prefix="step ")
    figure(f, "One forward pass, one backward pass — and the last frame holds "
              "∂L/∂x and ∂L/∂y simultaneously.")

    code_lab(
        "Reverse mode by hand, and the cost theorem measured",
        '''import numpy as np, math, time

# ============ 1. REVERSE MODE, ENTIRELY BY HAND ========================
print("=== L = (x*y + sin(x))^2  at x=1.7, y=0.6 ===")
x, y = 1.7, 0.6

# --- FORWARD PASS: compute and RECORD every intermediate -------------
a = x*y                       # a = x*y
b = math.sin(x)               # b = sin(x)
c = a + b                     # c = a + b
L = c*c                       # L = c^2
print("  forward pass (each value is kept for the backward pass):")
print(f"    a = x*y     = {a:.6f}")
print(f"    b = sin(x)  = {b:.6f}")
print(f"    c = a + b   = {c:.6f}")
print(f"    L = c^2     = {L:.6f}")

# --- BACKWARD PASS: adjoints, in REVERSE topological order -----------
L_bar = 1.0                              # dL/dL
c_bar = L_bar * 2*c                      # dL/dc = 2c
a_bar = c_bar * 1.0                      # dc/da = 1
b_bar = c_bar * 1.0                      # dc/db = 1
x_bar = a_bar*y + b_bar*math.cos(x)      # x FEEDS TWO NODES -> SUM
y_bar = a_bar*x
print()
print("  backward pass:")
print(f"    dL/dL = {L_bar:.6f}")
print(f"    dL/dc = dL/dL * 2c            = {c_bar:.6f}")
print(f"    dL/da = dL/dc * 1             = {a_bar:.6f}")
print(f"    dL/db = dL/dc * 1             = {b_bar:.6f}")
print(f"    dL/dx = dL/da * y + dL/db * cos(x)")
print(f"          = {a_bar:.4f}*{y} + {b_bar:.4f}*{math.cos(x):.4f}"
      f" = {x_bar:.6f}   <-- TWO PATHS, SUMMED")
print(f"    dL/dy = dL/da * x             = {y_bar:.6f}")

exact_x = 2*(x*y + math.sin(x))*(y + math.cos(x))
exact_y = 2*(x*y + math.sin(x))*x
print()
print(f"  analytic dL/dx = {exact_x:.12f}   error "
      f"{abs(x_bar-exact_x):.2e}")
print(f"  analytic dL/dy = {exact_y:.12f}   error "
      f"{abs(y_bar-exact_y):.2e}")
print()
print("  ONE backward pass produced BOTH partial derivatives.")
print("  forward mode would have needed two separate passes.")

# ============ 2. THE SUMMING RULE, AND THE BUG =========================
print()
print("=== why adjoints must be SUMMED, not assigned ===")
def with_sum(x, y):
    a, b = x*y, math.sin(x)
    c_bar = 2*(a+b)
    xb = 0.0
    xb += c_bar*y                 # path through a
    xb += c_bar*math.cos(x)       # path through b
    return xb

def with_assign(x, y):
    a, b = x*y, math.sin(x)
    c_bar = 2*(a+b)
    xb = 0.0
    xb = c_bar*y                  # the += became =
    xb = c_bar*math.cos(x)        # ... and this OVERWRITES it
    return xb

print(f"  correct (+=) : {with_sum(x, y):.10f}")
print(f"  buggy   (=)  : {with_assign(x, y):.10f}")
print(f"  analytic     : {exact_x:.10f}")
print("  the buggy version keeps only the LAST path. On an expression")
print("  where every node is used exactly once, this bug is INVISIBLE --")
print("  which is why it survives so many test suites.")

# ============ 3. THE COST THEOREM ======================================
print()
print("="*66)
print("Baur-Strassen: cost(grad f) <= 4 * cost(f), independent of n")
print("="*66)
try:
    import tensorflow as tf

    def make_mlp(n_params_target):
        w = int(math.sqrt(n_params_target/3))
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(w,)),
            tf.keras.layers.Dense(w, activation="tanh"),
            tf.keras.layers.Dense(w, activation="tanh"),
            tf.keras.layers.Dense(1)])

    print(f"{'parameters':>13}{'forward ms':>14}{'fwd+bwd ms':>14}"
          f"{'ratio':>9}")
    for target in [1e4, 1e5, 1e6, 4e6]:
        m = make_mlp(target)
        xin = tf.random.normal((32, m.input_shape[1]))

        @tf.function
        def fwd():
            return tf.reduce_sum(m(xin))

        @tf.function
        def fwd_bwd():
            with tf.GradientTape() as tape:
                out = tf.reduce_sum(m(xin))
            return tape.gradient(out, m.trainable_weights)

        fwd(); fwd_bwd()
        t0 = time.perf_counter()
        for _ in range(30): fwd()
        tf_ = (time.perf_counter()-t0)/30
        t0 = time.perf_counter()
        for _ in range(30): fwd_bwd()
        tb = (time.perf_counter()-t0)/30
        print(f"{m.count_params():>13,}{tf_*1000:>14.3f}{tb*1000:>14.3f}"
              f"{tb/tf_:>9.2f}x")
    print("  the RATIO is roughly constant at 2-4x, whatever the parameter")
    print("  count. Finite differences would need 2n forward passes --")
    print("  at 4 million parameters, that is 8 MILLION times slower.")
except ImportError:
    print("  (TensorFlow unavailable -- the theorem still stands:)")
    print("  every operation in the forward pass contributes a constant")
    print("  number of operations to the backward pass, so the ratio is")
    print("  bounded by a constant regardless of the graph size.")

# ============ 4. MEMORY: THE PRICE OF REVERSE MODE =====================
print()
print("=== reverse mode must store the whole tape ===")
def forward_only(depth, x=0.4):
    """Forward mode: O(1) memory."""
    v, d = x, 1.0
    for _ in range(depth):
        z = v*1.3 + 0.2
        t = math.tanh(z)
        d = (1 - t*t)*1.3*d
        v = t
    return v, d

def reverse_taped(depth, x=0.4):
    """Reverse mode: store EVERY intermediate."""
    tape = []
    v = x
    for _ in range(depth):
        z = v*1.3 + 0.2
        t = math.tanh(z)
        tape.append((v, z, t))          # <- the memory cost
        v = t
    g = 1.0
    for (vv, zz, tt) in reversed(tape):
        g = g*(1 - tt*tt)*1.3
    return v, g, len(tape)

print(f"{'depth':>8}{'forward-mode memory':>22}{'reverse-mode tape':>20}"
      f"{'gradients agree':>18}")
for depth in [10, 100, 1000, 10000]:
    v1, d1 = forward_only(depth)
    v2, d2, n_tape = reverse_taped(depth)
    print(f"{depth:>8}{'O(1) -- 2 floats':>22}{f'{n_tape*3} floats':>20}"
          f"{str(np.isclose(d1, d2)):>18}")
print("  this is exactly why ACTIVATIONS dominate training memory (19.4).")

# ============ 5. GRADIENT CHECKPOINTING ================================
print()
print("=== checkpointing: trade compute for memory ===")
def checkpointed(depth, n_ckpt, x=0.4):
    """Store only n_ckpt values; RECOMPUTE the segments in between."""
    seg = max(1, depth//n_ckpt)
    ckpts, v = [x], x
    for i in range(depth):
        v = math.tanh(v*1.3 + 0.2)
        if (i+1) % seg == 0:
            ckpts.append(v)
    g, recomputes = 1.0, 0
    for s in range(len(ckpts)-1, 0, -1):
        v = ckpts[s-1]
        local = []
        for _ in range(seg):                 # RECOMPUTE the segment
            t = math.tanh(v*1.3 + 0.2)
            local.append(t); v = t
            recomputes += 1
        for t in reversed(local):
            g = g*(1 - t*t)*1.3
    return g, len(ckpts), recomputes

depth = 1024
_, d_full, n_tape = reverse_taped(depth)
print(f"  depth {depth}")
print(f"{'checkpoints':>13}{'stored values':>16}{'recomputed ops':>17}"
      f"{'total ops':>12}{'memory saved':>15}")
for n_ck in [1, 4, 16, 32, 64, 256, 1024]:
    g, stored, recomp = checkpointed(depth, n_ck)
    print(f"{n_ck:>13}{stored:>16}{recomp:>17}{depth+recomp:>12}"
          f"{1 - stored/depth:>14.1%}")
print(f"  sqrt({depth}) = {int(math.sqrt(depth))} checkpoints is the optimum:")
print(f"  memory O(sqrt(L)) instead of O(L), for about 30% more compute.")
print("  that is what lets a model needing 40 GB of activations train on a")
print("  16 GB card. One line: tf.recompute_grad / torch.utils.checkpoint.")

import plotly.graph_objects as go
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
mem = [checkpointed(1024, n)[1] for n in ns]
ops = [1024 + checkpointed(1024, n)[2] for n in ns]
fig = go.Figure()
fig.add_scatter(x=ns, y=mem, mode="lines+markers", name="values stored",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(x=ns, y=ops, mode="lines+markers", name="total operations",
                line=dict(color=C["danger"], width=3), yaxis="y2")
fig.add_vline(x=32, line_dash="dash", line_color=C["success"],
              annotation_text="sqrt(1024) = 32")
fig.update_layout(height=420, xaxis_type="log", xaxis_title="checkpoints",
                  yaxis=dict(title="values stored"),
                  yaxis2=dict(title="operations", overlaying="y", side="right"),
                  title="Gradient checkpointing: memory against compute")
''',
        key="autodiff_reverse",
    )

    quiz(
        "In a reverse-mode engine, why must a node's gradient be accumulated "
        "with <code>+=</code> rather than assigned with <code>=</code>?",
        ["To avoid floating-point error",
         "Because a node feeding several children contributes through every "
         "path, and the multivariable chain rule sums those contributions",
         "Because the tape is processed out of order",
         "It makes no difference; both work"],
        1,
        "$\\partial y/\\partial v = \\sum_w (\\partial y/\\partial w)"
        "(\\partial w/\\partial v)$ over every child $w$. Assignment keeps only "
        "the last path. The bug is invisible on expressions where each node is "
        "used once — which is exactly why it survives so many test suites.",
        key="adq1",
    )

    keypoints([
        "Forward pass records a <b>tape</b>; backward pass propagates adjoints "
        "in <b>reverse topological order</b>.",
        "Adjoints must be <b>summed</b> over all children — <code>+=</code>, "
        "never <code>=</code>.",
        "<b>Baur–Strassen</b>: the full gradient costs $\\le 4\\times$ the "
        "function, <b>independent of $n$</b>.",
        "The price is memory: $\\mathcal{O}(\\text{graph size})$ — activations "
        "dominate training memory.",
        "<b>Checkpointing</b> gives $\\mathcal{O}(\\sqrt{L})$ memory for ~30 % "
        "more compute.",
    ])


# ==========================================================================
def s_b4():
    section("B.4", "Building an Autodiff Engine")

    lead(
        "Eighty lines of Python, and it trains a neural network. Everything "
        "TensorFlow does is this, plus performance engineering."
    )

    sub("The three pieces")

    table(
        ["Piece", "Job", "The subtlety"],
        [["<b>A value node</b>",
          "Hold a number, its gradient, and its parents",
          "Needs a <code>_backward</code> closure per operation"],
         ["<b>Operator overloading</b>",
          "Build the graph as a side effect of ordinary arithmetic",
          "Every operator must define its local derivative"],
         ["<b>Topological sort</b>",
          "Order the backward pass so children come first",
          "Depth-first post-order, then reversed"]],
    )

    codenote(
        "The closure trick",
        "Each operation attaches a small function that knows how to push its "
        "output's gradient onto its inputs. Building the graph therefore also "
        "builds the backward pass — there is no separate 'differentiate' step, "
        "and no formula is ever constructed. This is the design of micrograd, "
        "of PyTorch's autograd, and (with a tracing layer on top) of "
        "TensorFlow's <code>GradientTape</code>.",
    )

    anim_header("A topological sort of a small graph")

    edges = {"L": ["c", "c"], "c": ["a", "b"], "a": ["x", "y"], "b": ["x"],
             "x": [], "y": []}
    pos = {"x": (0, 0), "y": (0, 1.4), "a": (2, 1.0), "b": (2, -0.4),
           "c": (4, .3), "L": (6, .3)}
    order = ["x", "y", "a", "b", "c", "L"]

    frames = []
    for phase, seq in [("topological order (children first)", order),
                       ("backward pass (reversed)", order[::-1])]:
        for k in range(1, len(seq)+1):
            done = set(seq[:k])
            data = []
            for n, ch in edges.items():
                for c in set(ch):
                    data.append(go.Scatter(
                        x=[pos[c][0], pos[n][0]], y=[pos[c][1], pos[n][1]],
                        mode="lines",
                        line=dict(color=alpha(C["line"], .8), width=2),
                        showlegend=False, hoverinfo="skip"))
            for n, (px, py) in pos.items():
                active = n in done
                cur = n == seq[k-1]
                data.append(go.Scatter(
                    x=[px], y=[py], mode="markers+text", text=[n],
                    textposition="middle center",
                    textfont=dict(size=13, color="#fff"),
                    marker=dict(size=42,
                                color=(C["danger"] if cur else
                                       C["success"] if active
                                       else alpha(C["line"], .5)),
                                line=dict(color="#fff", width=2)),
                    showlegend=False, hoverinfo="skip"))
            frames.append(go.Frame(name=f"{phase[:3]}{k}", data=data,
                                   layout=go.Layout(annotations=[
                                       anim.annotate_step(
                                           f"{phase}   ·   visiting "
                                           f"{seq[k-1]}   ·   "
                                           f"{k}/{len(seq)}")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-1, 7]),
                    yaxis=dict(visible=False, range=[-1.4, 2.4]),
                    title="L = (x·y + sin x)² as a graph")
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="step ")
    figure(f, "The backward pass is the topological order, reversed — so every "
              "node's children are already done when it is reached.")

    code_lab(
        "A complete autodiff engine, and a neural net trained with it",
        '''import numpy as np, math, time

# ============ THE ENGINE ===============================================
class Value:
    """A scalar with a gradient and a place in a computation graph."""
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.data:.6f}, grad={self.grad:.6f})"

    # ---- operations. each one attaches its own local derivative -----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad            # += NOT =
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad     # the PRODUCT RULE
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data**(k-1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t*t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    # ---- the conveniences -------------------------------------------
    def __neg__(self):      return self * -1
    def __radd__(self, o):  return self + o
    def __sub__(self, o):   return self + (-o)
    def __rsub__(self, o):  return o + (-self)
    def __rmul__(self, o):  return self * o
    def __truediv__(self, o):  return self * o**-1
    def __rtruediv__(self, o): return o * self**-1

    # ---- THE BACKWARD PASS ------------------------------------------
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)                # POST-ORDER: children first
        build(self)
        self.grad = 1.0                       # dL/dL = 1
        for v in reversed(topo):              # REVERSED: parents first
            v._backward()

# ============ 1. IT AGREES WITH CALCULUS ===============================
print("=== L = (x*y + sin x)^2 ===")
x, y = Value(1.7), Value(0.6)
L = (x*y + x.sin())**2
L.backward()
print(f"  L      = {L.data:.10f}")
print(f"  dL/dx  = {x.grad:.10f}")
print(f"  dL/dy  = {y.grad:.10f}")
ex = 2*(1.7*0.6 + math.sin(1.7))*(0.6 + math.cos(1.7))
ey = 2*(1.7*0.6 + math.sin(1.7))*1.7
print(f"  exact  = {ex:.10f}, {ey:.10f}")
print(f"  errors = {abs(x.grad-ex):.2e}, {abs(y.grad-ey):.2e}")

# ============ 2. THE SHARED-NODE TEST ==================================
print()
print("=== the test that catches the '=' instead of '+=' bug ===")
a = Value(3.0)
out = a*a + a*a*a + a.sin()           # a is used FIVE times
out.backward()
exact = 2*3 + 3*3**2 + math.cos(3)
print(f"  f(a) = a^2 + a^3 + sin(a),  a = 3")
print(f"  engine  df/da = {a.grad:.10f}")
print(f"  exact   df/da = {exact:.10f}")
print(f"  error         = {abs(a.grad-exact):.2e}")
print("  if += were =, this would be badly wrong while every")
print("  single-use expression still passed.")

# ============ 3. GRADIENT CHECK EVERYTHING =============================
print()
print("=== gradient-checking the engine against finite differences ===")
def check(build, xs, h=1e-6):
    vs = [Value(v) for v in xs]
    out = build(vs)
    out.backward()
    ga = np.array([v.grad for v in vs])
    gn = np.zeros(len(xs))
    for i in range(len(xs)):
        xp = list(xs); xp[i] += h
        xm = list(xs); xm[i] -= h
        gn[i] = (build([Value(v) for v in xp]).data
                 - build([Value(v) for v in xm]).data)/(2*h)
    rel = np.linalg.norm(ga-gn)/(np.linalg.norm(ga)+np.linalg.norm(gn)+1e-30)
    return rel, ga, gn

tests = {
    "x0*x1 + sin(x0)":       lambda v: v[0]*v[1] + v[0].sin(),
    "exp(x0*x1) / (1+x2^2)": lambda v: (v[0]*v[1]).exp() / (1 + v[2]**2),
    "tanh(x0+x1)*relu(x2)":  lambda v: (v[0]+v[1]).tanh() * v[2].relu(),
    "log(1+exp(x0)) softplus": lambda v: (1 + v[0].exp()).log(),
    "deeply nested":         lambda v: ((v[0]*v[1]).tanh() + v[2]).tanh()
                                       * (v[0]**3),
}
xs = [0.7, -1.3, 0.9]
print(f"{'expression':<28}{'relative error':>18}{'verdict':>10}")
for nm, fn in tests.items():
    rel, _, _ = check(fn, xs)
    print(f"{nm:<28}{rel:>18.3e}{('OK' if rel < 1e-6 else 'BUG'):>10}")

# ============ 4. A NEURAL NETWORK, TRAINED BY THIS ENGINE ==============
print()
print("="*66)
print("Training a neural network with the engine above")
print("="*66)
rng = np.random.default_rng(42)

class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(rng.normal(0, (2/n_in)**.5)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, sizes):
        sz = [n_in] + sizes
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(sizes)-1))
                       for i in range(len(sizes))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

# the moons dataset
from core import datasets as _ds
X, yv = _ds.moons(n=120, noise=0.16)[:2]
X = ((X - X.mean(0))/X.std(0)).astype(float)
yv = np.where(yv == 0, -1.0, 1.0)

model = MLP(2, [12, 12, 1])
print(f"  architecture 2 -> 12 -> 12 -> 1")
print(f"  {len(model.parameters())} parameters, all scalar Values")

def loss_and_acc():
    inputs = [[Value(v) for v in row] for row in X]
    scores = [model(inp) for inp in inputs]
    # hinge loss, plus L2
    losses = [(1 + -yi*si).relu() for yi, si in zip(yv, scores)]
    data_loss = sum(losses) * (1.0/len(losses))
    reg = 1e-4 * sum((p*p for p in model.parameters()), Value(0.0))
    acc = float(np.mean([(si.data > 0) == (yi > 0)
                         for yi, si in zip(yv, scores)]))
    return data_loss + reg, acc

print()
print(f"{'step':>6}{'loss':>12}{'accuracy':>12}{'lr':>10}")
t0 = time.perf_counter()
for step in range(40):
    total, acc = loss_and_acc()
    for p in model.parameters():
        p.grad = 0.0                     # ZERO THE GRADIENTS FIRST
    total.backward()
    lr = 0.9 - 0.85*step/40
    for p in model.parameters():
        p.data -= lr * p.grad
    if step % 8 == 0 or step == 39:
        print(f"{step:>6}{total.data:>12.6f}{acc:>12.4f}{lr:>10.4f}")
print(f"  {time.perf_counter()-t0:.1f}s, final accuracy {acc:.4f}")
print()
print("  every gradient in that loop came from the 80-line engine above.")
print("  TensorFlow and PyTorch do exactly this, on ARRAYS instead of")
print("  scalars, with a compiler and a GPU behind them.")

# ============ 5. THE GRAPH IT BUILT ====================================
print()
print("=== the graph for one training example ===")
xi = [Value(X[0][0]), Value(X[0][1])]
s = model(xi)
def count_nodes(v, seen=None):
    seen = seen if seen is not None else set()
    if v in seen: return seen
    seen.add(v)
    for c in v._prev:
        count_nodes(c, seen)
    return seen
nodes = count_nodes(s)
ops = {}
for n in nodes:
    ops[n._op or "leaf"] = ops.get(n._op or "leaf", 0) + 1
print(f"  {len(nodes)} nodes for a single forward pass through 2-12-12-1")
print(f"  operation counts: "
      f"{dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
print("  every one of those nodes carries a _backward closure. Building the")
print("  graph IS building the backward pass -- there is no separate")
print("  'differentiate' step, and no formula is ever constructed.")

import plotly.graph_objects as go
grid = np.linspace(-2.6, 2.6, 45)
G1, G2 = np.meshgrid(grid, grid)
Z = np.array([[model([Value(a), Value(b)]).data for a in grid] for b in grid])
fig = go.Figure()
fig.add_contour(x=grid, y=grid, z=Z, colorscale=nav.cscale(), opacity=.65,
                contours=dict(showlines=False))
for cls, col in [(-1.0, CLASS_COLORS[0]), (1.0, CLASS_COLORS[1])]:
    m = yv == cls
    fig.add_scatter(x=X[m, 0], y=X[m, 1], mode="markers",
                    name=f"class {int(cls)}",
                    marker=dict(size=8, color=col,
                                line=dict(color="#fff", width=1)))
fig.update_layout(height=500, title="Decision surface, trained by 80 lines "
                                    "of autodiff",
                  xaxis_title="x1", yaxis_title="x2")
''',
        key="autodiff_engine",
    )

    keypoints([
        "A node holds a value, a gradient, its parents, and a "
        "<code>_backward</code> closure.",
        "<b>Building the graph builds the backward pass</b> — there is no "
        "separate differentiation step.",
        "The backward pass is a <b>reversed topological sort</b>, so children "
        "are always ready.",
        "<b>Zero the gradients</b> before every backward pass, or they "
        "accumulate across steps.",
        "TensorFlow and PyTorch are this, on arrays, with a compiler and a GPU.",
    ])


# ==========================================================================
def s_b5():
    section("B.5", "GradientTape in Practice")

    lead(
        "TensorFlow's tape, and the six things that surprise people about it."
    )

    table(
        ["Behaviour", "Why", "What to do"],
        [["The tape is <b>consumed</b> by <code>gradient()</code>",
          "It frees the recorded intermediates immediately",
          "<code>persistent=True</code>, then <code>del tape</code>"],
         ["Only <b>Variables</b> are watched",
          "Constants are not parameters",
          "<code>tape.watch(tensor)</code> to force it"],
         ["<b>Python control flow is traced, not recorded</b>",
          "The tape sees the branch actually taken",
          "That is usually what you want; <code>tf.cond</code> if not"],
         ["<b>Non-TF operations break the chain</b>",
          "numpy operations are invisible to the tape",
          "Keep everything in <code>tf.*</code>"],
         ["<b>Second derivatives need nested tapes</b>",
          "The outer tape must record the inner one's work",
          "Nest them, both persistent if needed"],
         ["<code>stop_gradient</code> cuts the path",
          "Used for target networks (§18.6), GANs (§17.7)",
          "<code>tf.stop_gradient(x)</code>"]],
    )

    idea(
        "<code>@tf.function</code> changes what the tape sees",
        "In eager mode the tape records the operations that <i>actually run</i>. "
        "Inside a <code>tf.function</code>, the code is traced once into a graph "
        "and that graph is differentiated — so a Python <code>if</code> on a "
        "tensor is baked in at trace time, and a Python loop is unrolled. This "
        "is the same tracing behaviour as §12.9, and the same rules apply: use "
        "<code>tf.cond</code> and <code>tf.while_loop</code> when the control "
        "flow must depend on tensor values at run time.",
    )

    anim_header("Nested tapes computing a second derivative")

    xs = np.linspace(-2.4, 2.4, 200)
    fv = xs**3 - 3*xs
    d1 = 3*xs**2 - 3
    d2 = 6*xs
    frames = []
    for k in range(1, 4):
        data = [go.Scatter(x=xs, y=fv, mode="lines",
                           line=dict(color=C["primary"], width=3))]
        if k >= 2:
            data.append(go.Scatter(x=xs, y=d1, mode="lines",
                                   line=dict(color=C["accent"], width=3)))
        if k >= 3:
            data.append(go.Scatter(x=xs, y=d2, mode="lines",
                                   line=dict(color=C["danger"], width=3)))
        msg = ["f(x) = x³ − 3x   ·   the function",
               "f′(x) = 3x² − 3   ·   inner tape, zero at x = ±1",
               "f″(x) = 6x   ·   OUTER tape differentiating the inner one's "
               "output   ·   negative at x = −1 (a maximum), positive at "
               "x = +1 (a minimum)"][k-1]
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[
                                   anim.annotate_step(msg)])))

    f = go.Figure(data=[go.Scatter(x=xs, y=fv, mode="lines", name="f(x)",
                                   line=dict(color=C["primary"], width=3)),
                        go.Scatter(x=xs, y=d1, mode="lines", name="f′(x)",
                                   line=dict(color=C["accent"], width=3)),
                        go.Scatter(x=xs, y=d2, mode="lines", name="f″(x)",
                                   line=dict(color=C["danger"], width=3))])
    f.add_hline(y=0, line_color=C["muted"], line_width=1)
    f.update_layout(height=430, xaxis_title="x",
                    title="Nested tapes give the second derivative",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(2000), slider_prefix="order ")
    figure(f)

    code_lab(
        "Every GradientTape behaviour that surprises people",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. THE TAPE IS CONSUMED ==================================
print("=== the tape is consumed by the first gradient() call ===")
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
print(f"  first  call: {float(tape.gradient(y, x)):.4f}")
try:
    tape.gradient(y, x)
except RuntimeError as e:
    print(f"  second call: RuntimeError: {str(e)[:64]}...")

with tf.GradientTape(persistent=True) as tape:
    y = x**2
    z = x**3
print(f"  persistent=True -> dy/dx {float(tape.gradient(y, x)):.4f}, "
      f"dz/dx {float(tape.gradient(z, x)):.4f}")
del tape                       # free it explicitly
print("  a persistent tape holds every intermediate until you delete it.")

# ============ 2. ONLY VARIABLES ARE WATCHED ============================
print()
print("=== constants are invisible unless you say otherwise ===")
c = tf.constant(3.0)
v = tf.Variable(3.0)
with tf.GradientTape() as tape:
    out = c**2 + v**2
g = tape.gradient(out, [c, v])
print(f"  d/dc = {g[0]}   <- None: constants are not watched")
print(f"  d/dv = {float(g[1]):.4f}")

with tf.GradientTape() as tape:
    tape.watch(c)              # force it
    out = c**2
print(f"  after tape.watch(c): d/dc = {float(tape.gradient(out, c)):.4f}")

# ============ 3. NUMPY BREAKS THE CHAIN ================================
print()
print("=== the most common silent failure ===")
w = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    bad = tf.reduce_sum(np.square(w.numpy()))     # LEFT THE GRAPH
print(f"  using numpy inside the tape: gradient = {tape.gradient(bad, w)}")
with tf.GradientTape() as tape:
    good = tf.reduce_sum(tf.square(w))
print(f"  staying in tf:               gradient = "
      f"{tape.gradient(good, w).numpy()}")
print("  the numpy version returns None. No exception, no warning.")
print("  if a gradient is unexpectedly None, look for a .numpy() call.")

# ============ 4. CONTROL FLOW ==========================================
print()
print("=== Python control flow is RECORDED, not analysed ===")
def f(x):
    if x > 0:                          # a real Python if, in eager mode
        return x**2
    return -x**3

for val in [2.0, -2.0]:
    xv = tf.Variable(val)
    with tf.GradientTape() as tape:
        y = f(xv)
    print(f"  x = {val:>5}: f(x) = {float(y):>8.3f}, "
          f"f'(x) = {float(tape.gradient(y, xv)):>8.3f}   "
          f"(branch: {'x^2' if val > 0 else '-x^3'})")
print("  the tape sees only the branch that RAN. That is usually exactly")
print("  what you want -- and it is why @tf.function tracing (12.9) can")
print("  surprise you: there, the branch is baked in at trace time.")

# ============ 5. HIGHER DERIVATIVES ====================================
print()
print("=== nested tapes ===")
x = tf.Variable(1.5)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x**3 - 3*x
    dy = inner.gradient(y, x)          # the OUTER tape records this
d2y = outer.gradient(dy, x)
print(f"  f(x)   = x^3 - 3x  at x = 1.5")
print(f"  f(x)   = {float(y):.6f}")
print(f"  f'(x)  = {float(dy):.6f}   (exact 3x^2-3 = {3*1.5**2-3:.6f})")
print(f"  f''(x) = {float(d2y):.6f}   (exact 6x     = {6*1.5:.6f})")

# --- the full Hessian ------------------------------------------------
print()
print("=== the Hessian of a 3-D function ===")
v = tf.Variable([1.0, 2.0, 0.5])
with tf.GradientTape(persistent=True) as outer:
    with tf.GradientTape() as inner:
        f_ = (v[0]**2 * v[1] + v[1]**2 * v[2] + tf.sin(v[0]*v[2]))
    grad = inner.gradient(f_, v)
H = outer.jacobian(grad, v)
del outer
print(f"  gradient: {grad.numpy().round(5)}")
print(f"  Hessian:")
for row in H.numpy():
    print(f"    {np.round(row, 5)}")
print(f"  symmetric: {np.allclose(H.numpy(), H.numpy().T)}   "
      f"(Clairaut's theorem)")
eig = np.linalg.eigvalsh(H.numpy())
print(f"  eigenvalues {np.round(eig, 4)} -> "
      f"{'a saddle point' if (eig.min() < 0 < eig.max()) else 'definite'}")
print("  the Hessian costs n backward passes (one per gradient component),")
print("  which is why second-order methods use HESSIAN-VECTOR PRODUCTS")
print("  instead -- those cost only ONE extra pass.")

# --- a Hessian-vector product ----------------------------------------
vec = tf.constant([1.0, 0.0, -1.0])
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        f_ = (v[0]**2*v[1] + v[1]**2*v[2] + tf.sin(v[0]*v[2]))
    g_ = inner.gradient(f_, v)
    dot = tf.reduce_sum(g_*vec)        # differentiate g.v, not g
hvp = outer.gradient(dot, v)
print()
print(f"  H @ [1, 0, -1] via the Hessian : "
      f"{(H.numpy() @ vec.numpy()).round(5)}")
print(f"  H @ [1, 0, -1] via a HVP       : {hvp.numpy().round(5)}")
print(f"  identical: {np.allclose(H.numpy() @ vec.numpy(), hvp.numpy())}")
print("  one extra backward pass instead of n. This is how conjugate")
print("  gradient and Newton-CG scale to millions of parameters.")

# ============ 6. stop_gradient =========================================
print()
print("=== stop_gradient cuts the path ===")
a = tf.Variable(2.0); b = tf.Variable(3.0)
with tf.GradientTape() as tape:
    out = a*b + tf.stop_gradient(a)*b
g = tape.gradient(out, [a, b])
print(f"  out = a*b + stop_gradient(a)*b")
print(f"  d/da = {float(g[0]):.4f}   (only the FIRST term contributes: b = 3)")
print(f"  d/db = {float(g[1]):.4f}   (both terms: a + a = 4)")
print("  this is how a DQN target network is frozen (18.6), how a GAN's")
print("  generator step avoids updating the discriminator (17.7), and how")
print("  a straight-through estimator is built.")

# ============ 7. CUSTOM GRADIENTS ======================================
print()
print("=== when you need to override the derivative ===")
@tf.custom_gradient
def stable_softplus(x):
    """log(1+exp(x)) overflows for large x; its gradient is just sigmoid."""
    out = tf.math.softplus(x)
    def grad(dy):
        return dy * tf.sigmoid(x)
    return out, grad

for val in [0.0, 10.0, 100.0]:
    xv = tf.Variable(val)
    with tf.GradientTape() as tape:
        y = stable_softplus(xv)
    print(f"  x = {val:>6}: softplus = {float(y):>10.4f}, "
          f"gradient = {float(tape.gradient(y, xv)):.10f}")
print("  a naive log(1+exp(x)) gives inf at x=100 and a NaN gradient.")
print("  @tf.custom_gradient is also how you implement a straight-through")
print("  estimator for quantisation (19.3): round() forwards, identity back.")

@tf.custom_gradient
def straight_through_round(x):
    def grad(dy):
        return dy                      # pretend round() were the identity
    return tf.round(x), grad

xv = tf.Variable([0.2, 0.7, 1.4])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(straight_through_round(xv)**2)
print()
print(f"  round([0.2, 0.7, 1.4]) = {tf.round(xv).numpy()}")
print(f"  gradient with a straight-through estimator: "
      f"{tape.gradient(y, xv).numpy()}")
print("  round() has derivative 0 almost everywhere, so without this trick")
print("  quantisation-aware training could not work at all.")

import plotly.graph_objects as go
xs = np.linspace(-2.4, 2.4, 300)
xv = tf.Variable(xs.astype("float32"))
with tf.GradientTape() as o:
    with tf.GradientTape() as i:
        yv = xv**3 - 3*xv
    d1 = i.gradient(yv, xv)
d2 = o.gradient(d1, xv)
fig = go.Figure()
fig.add_scatter(x=xs, y=yv.numpy(), mode="lines", name="f",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(x=xs, y=d1.numpy(), mode="lines", name="f′",
                line=dict(color=C["accent"], width=3))
fig.add_scatter(x=xs, y=d2.numpy(), mode="lines", name="f″",
                line=dict(color=C["danger"], width=3))
fig.add_hline(y=0, line_color=C["muted"], line_width=1)
fig.update_layout(height=420, xaxis_title="x",
                  title="f, f′ and f″ from nested tapes")
''',
        key="autodiff_tape",
    )

    keypoints([
        "The tape is <b>consumed</b> unless <code>persistent=True</code>; only "
        "Variables are watched by default.",
        "<b>A numpy call inside the tape returns a <code>None</code> "
        "gradient</b>, silently — the classic bug.",
        "<b>Nested tapes</b> give second derivatives; a Hessian costs $n$ passes, "
        "a Hessian-vector product costs one.",
        "<code>stop_gradient</code> cuts a path — target networks, GAN steps, "
        "straight-through estimators.",
        "<code>@tf.custom_gradient</code> is how you fix numerical instability "
        "and how quantisation-aware training works.",
    ])


# ==========================================================================
def s_b6():
    section("B.6", "Reference and Further Reading")

    sub("Derivative rules the engines implement")

    table(
        ["$f$", "$\\dfrac{\\partial f}{\\partial x}$", "Note"],
        [["$x + y$", "$1$ (to each)", "Gradient flows through unchanged"],
         ["$xy$", "$y$ (and $x$)", "The product rule; swap and multiply"],
         ["$x^k$", "$kx^{k-1}$", ""],
         ["$e^x$", "$e^x$", "Reuse the forward value"],
         ["$\\ln x$", "$1/x$", "Undefined at 0 — clip in practice"],
         ["$\\tanh x$", "$1 - \\tanh^2 x$",
          "Reuse the forward value; $\\le 1$, hence vanishing gradients"],
         ["$\\sigma(x)$", "$\\sigma(x)\\bigl(1-\\sigma(x)\\bigr)$",
          "<b>Peaks at 0.25</b> — worse than tanh"],
         ["$\\mathrm{ReLU}(x)$", "$\\mathbb{1}[x > 0]$",
          "Exactly 0 or 1; no attenuation, hence its success"],
         ["$\\max(x, y)$", "$\\mathbb{1}$ to the argmax only",
          "Ties are broken arbitrarily"],
         ["$\\mathbf{A}\\mathbf{x}$", "$\\mathbf{A}^{\\top}$",
          "The transpose is why backprop is 'the same net, backwards'"],
         ["$\\lVert\\mathbf{x}\\rVert_2$",
          "$\\mathbf{x}/\\lVert\\mathbf{x}\\rVert_2$",
          "Undefined at the origin"],
         ["$\\mathrm{softmax}(\\mathbf{x})_i$",
          "$s_i(\\delta_{ij} - s_j)$",
          "Vanishes when saturated — hence the $\\sqrt{d_k}$ of §16.6"]],
    )

    sub("Framework comparison")

    table(
        ["", "TensorFlow", "PyTorch", "JAX"],
        [["Style", "<code>GradientTape</code> (explicit scope)",
          "<code>.backward()</code> (implicit tape)",
          "<code>grad(f)</code> (functional transform)"],
         ["Graph", "Eager, or traced by <code>@tf.function</code>",
          "Eager, or compiled by <code>torch.compile</code>",
          "<b>Always traced</b> by <code>jit</code>"],
         ["Forward mode", "<code>ForwardAccumulator</code>",
          "<code>torch.func.jvp</code>", "<b><code>jax.jvp</code></b>"],
         ["Higher order", "Nested tapes",
          "<code>create_graph=True</code>",
          "<b><code>grad(grad(f))</code></b> — composes naturally"],
         ["Vectorising", "<code>vectorized_map</code>",
          "<code>torch.vmap</code>", "<b><code>jax.vmap</code></b>"]],
    )

    idea(
        "JAX's insight: differentiation is a program transformation",
        "TensorFlow and PyTorch treat autodiff as something that happens to a "
        "<i>graph object</i> you have built. JAX treats "
        "<code>grad</code> as a function that takes a function and returns "
        "another function — so it composes trivially with itself "
        "(<code>grad(grad(f))</code>), with vectorisation "
        "(<code>vmap</code>), with parallelisation (<code>pmap</code>) and with "
        "compilation (<code>jit</code>), in any order. That compositionality is "
        "the whole design, and it is why JAX dominates research code that needs "
        "unusual derivatives.",
    )

    sub("Debugging a gradient")

    table(
        ["Symptom", "Likely cause", "Check"],
        [["Gradient is <code>None</code>",
          "A numpy operation broke the chain, or the target is not a Variable",
          "Search the forward pass for <code>.numpy()</code>"],
         ["Gradient is exactly 0",
          "A saturated activation, a dead ReLU, or "
          "<code>stop_gradient</code>",
          "Print per-layer gradient norms"],
         ["Gradient is <code>NaN</code>",
          "$\\log(0)$, $0/0$, $\\sqrt{0}$, or overflow in the forward pass",
          "<code>tf.debugging.enable_check_numerics()</code>"],
         ["Gradient is huge",
          "No clipping, a bad initialisation, or a recurrent product",
          "<code>clipnorm</code>, and check §11.1"],
         ["Analytic ≠ numeric",
          "A missing <code>+=</code>, or a wrong local derivative",
          "Gradient-check term by term"],
         ["Works eagerly, breaks in <code>tf.function</code>",
          "Python control flow baked in at trace time",
          "<code>tf.cond</code> / <code>tf.while_loop</code> (§12.9)"]],
    )

    code_lab(
        "A gradient-debugging toolkit you can reuse",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. A REUSABLE GRADIENT CHECKER ===========================
def gradient_check(fn, params, h=1e-5, verbose=True):
    """Compare autodiff against a central difference. Returns the relative
    error; below 1e-7 is fine, above 1e-4 is a bug."""
    params = [tf.Variable(p) if not isinstance(p, tf.Variable) else p
              for p in params]
    with tf.GradientTape() as tape:
        out = fn(params)
    analytic = [g.numpy() if g is not None else None
                for g in tape.gradient(out, params)]

    numeric = []
    for i, p in enumerate(params):
        flat = p.numpy().ravel().copy()
        gn = np.zeros_like(flat)
        for j in range(len(flat)):
            orig = flat[j]
            flat[j] = orig + h
            p.assign(flat.reshape(p.shape))
            up = float(fn(params))
            flat[j] = orig - h
            p.assign(flat.reshape(p.shape))
            dn = float(fn(params))
            flat[j] = orig
            gn[j] = (up - dn)/(2*h)
        p.assign(flat.reshape(p.shape))
        numeric.append(gn.reshape(p.shape))

    rels = []
    for i, (ga, gn) in enumerate(zip(analytic, numeric)):
        if ga is None:
            if verbose:
                print(f"    param {i}: gradient is None -- the chain is BROKEN")
            rels.append(np.inf)
            continue
        num = np.linalg.norm(ga - gn)
        den = np.linalg.norm(ga) + np.linalg.norm(gn) + 1e-30
        rels.append(num/den)
    return rels

print("=== the gradient checker ===")
def good_fn(p):
    w, b = p
    x = tf.constant([[1.0, 2.0], [0.5, -1.0], [2.0, 0.3]])
    return tf.reduce_sum(tf.tanh(x @ w + b)**2)

rng = np.random.default_rng(0)
w0 = rng.normal(0, .5, (2, 3)).astype("float32")
b0 = rng.normal(0, .1, (3,)).astype("float32")
rels = gradient_check(good_fn, [w0, b0])
print(f"  relative errors: {[f'{r:.2e}' for r in rels]}")
print(f"  verdict: {'OK' if max(rels) < 1e-6 else 'SUSPECT'}")

# ============ 2. THE BROKEN-CHAIN DETECTOR =============================
print()
print("=== a function that silently leaves the graph ===")
def broken_fn(p):
    w, b = p
    x = tf.constant([[1.0, 2.0]])
    z = x @ w + b
    return tf.reduce_sum(np.tanh(z.numpy())**2)     # numpy!

try:
    rels = gradient_check(broken_fn, [w0, b0])
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:70]}")

# ============ 3. PER-LAYER GRADIENT NORMS ==============================
print()
print("=== per-layer gradient health ===")
def layer_gradient_report(model, x, y, loss_fn=None):
    loss_fn = loss_fn or keras.losses.sparse_categorical_crossentropy
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(loss_fn(y, model(x, training=True)))
    grads = tape.gradient(loss, model.trainable_weights)
    rows = []
    for w, g in zip(model.trainable_weights, grads):
        if len(w.shape) < 2:
            continue
        rows.append((w.path.split("/")[0][:20],
                     float(tf.norm(w)),
                     0.0 if g is None else float(tf.norm(g)),
                     0.0 if g is None else float(tf.norm(g))/float(tf.norm(w))))
    return rows

def build(act, init, depth=6):
    tf.random.set_seed(0)
    L = [keras.layers.Input(shape=(20,))]
    for i in range(depth):
        L.append(keras.layers.Dense(32, activation=act,
                                    kernel_initializer=init,
                                    name=f"dense_{i}"))
    L.append(keras.layers.Dense(3, activation="softmax"))
    return keras.Sequential(L)

X = rng.normal(0, 1, (64, 20)).astype("float32")
Y = rng.integers(0, 3, 64)

for nm, act, init in [("relu + he_normal (good)", "relu", "he_normal"),
                      ("sigmoid + glorot (vanishing)", "sigmoid",
                       "glorot_uniform"),
                      ("relu + large init (exploding)", "relu",
                       keras.initializers.RandomNormal(stddev=1.5))]:
    m = build(act, init)
    rows = layer_gradient_report(m, X, Y)
    norms = [r[2] for r in rows]
    print(f"\\n  {nm}")
    print(f"    {'layer':<12}{'||W||':>10}{'||grad||':>13}{'ratio':>12}")
    for n_, wn, gn, rt in rows:
        print(f"    {n_:<12}{wn:>10.4f}{gn:>13.3e}{rt:>12.3e}")
    spread = max(norms)/max(min(norms), 1e-30)
    print(f"    first/last gradient ratio: {norms[0]/max(norms[-1],1e-30):.3e}"
          f"   spread {spread:.2e}")
    if spread > 1e4:
        print(f"    -> UNSTABLE (section 11.1)")
    else:
        print(f"    -> healthy")

# ============ 4. FINDING NaNs ==========================================
print()
print("=== where does a NaN come from? ===")
def find_nan_source():
    x = tf.Variable([1.0, 0.0, -1.0])
    ops = {
        "log(x)":        lambda v: tf.math.log(v),
        "log(x) safely": lambda v: tf.math.log(tf.maximum(v, 1e-12)),
        "sqrt(x)":       lambda v: tf.sqrt(v),
        "sqrt(x+eps)":   lambda v: tf.sqrt(v + 1e-12),
        "1/x":           lambda v: 1.0/v,
        "x/(x+eps)":     lambda v: v/(v + 1e-12),
        "exp(100*x)":    lambda v: tf.exp(100.0*v),
    }
    print(f"  {'operation':<18}{'forward':>34}{'gradient':>34}")
    for nm, op in ops.items():
        with tf.GradientTape() as tape:
            out = tf.reduce_sum(op(x))
        g = tape.gradient(out, x)
        fwd = f"{float(out):.4g}" if np.isfinite(float(out)) else "inf/nan"
        gs = (np.array2string(g.numpy(), precision=3)
              if g is not None else "None")
        print(f"  {nm:<18}{fwd:>34}{gs:>34}")
find_nan_source()
print("  a finite forward value does NOT guarantee a finite gradient --")
print("  sqrt(0) is 0 but its derivative is infinite. Add an epsilon")
print("  INSIDE the sqrt, not outside it.")

# ============ 5. THE SATURATION TABLE ==================================
print()
print("=== why the activation choice is a gradient choice ===")
zs = tf.constant([-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0])
print(f"  {'z':>7}{'sigmoid':>11}{'d/dz':>11}{'tanh':>11}{'d/dz':>11}"
      f"{'relu':>9}{'d/dz':>8}")
for z in zs:
    zv = tf.Variable(float(z))
    ders = []
    for fn in [tf.sigmoid, tf.tanh, tf.nn.relu]:
        with tf.GradientTape() as t:
            o = fn(zv)
        ders.append((float(o), float(t.gradient(o, zv))))
    print(f"  {float(z):>7.1f}{ders[0][0]:>11.5f}{ders[0][1]:>11.5f}"
          f"{ders[1][0]:>11.5f}{ders[1][1]:>11.5f}"
          f"{ders[2][0]:>9.2f}{ders[2][1]:>8.1f}")
print("  sigmoid's derivative PEAKS at 0.25 -- stack ten layers and the")
print("  gradient is multiplied by at most 0.25^10 = 1e-6.")
print("  ReLU's is exactly 1 wherever it is active. That single fact is")
print("  most of why deep networks became trainable (section 11.2).")

import plotly.graph_objects as go
zz = np.linspace(-6, 6, 400).astype("float32")
zt = tf.Variable(zz)
fig = go.Figure()
for nm, fn, col in [("sigmoid", tf.sigmoid, C["danger"]),
                    ("tanh", tf.tanh, C["warning"]),
                    ("relu", tf.nn.relu, C["success"])]:
    with tf.GradientTape() as t:
        o = fn(zt)
    d = t.gradient(o, zt)
    fig.add_scatter(x=zz, y=d.numpy(), mode="lines", name=f"d/dz {nm}",
                    line=dict(color=col, width=3))
fig.add_hline(y=1.0, line_dash="dot", line_color=C["muted"],
              annotation_text="1.0 — no attenuation")
fig.update_layout(height=420, xaxis_title="z", yaxis_title="derivative",
                  title="Activation derivatives — the vanishing-gradient story")
''',
        key="autodiff_debug",
    )

    rule()

    keypoints([
        "Every engine implements the same small table of local derivatives; the "
        "chain rule does the rest.",
        "<b>JAX treats differentiation as a program transformation</b>, which is "
        "why it composes so cleanly.",
        "A <code>None</code> gradient means a broken chain; a zero one means "
        "saturation or a cut path.",
        "A finite forward value does not imply a finite gradient — "
        "$\\sqrt{0}$ is the canonical case.",
        "<b>Gradient-check every hand-written derivative.</b> It takes two "
        "minutes and it always finds the bug.",
    ], title="Appendix B in five lines")

    refs([
        ("Baydin et al. — *Automatic Differentiation in Machine Learning: a "
         "Survey*", "https://arxiv.org/abs/1502.05767"),
        ("Griewank & Walther — *Evaluating Derivatives* (the reference text)",
         "https://doi.org/10.1137/1.9780898717761"),
        ("Chen et al. — *Training Deep Nets with Sublinear Memory Cost* "
         "(checkpointing)", "https://arxiv.org/abs/1604.06174"),
        ("Karpathy — *micrograd* (the engine of §B.4, in 100 lines)",
         "https://github.com/karpathy/micrograd"),
        ("Bradbury et al. — *JAX: composable transformations of Python+NumPy "
         "programs*", "https://github.com/google/jax"),
        ("TensorFlow — *Introduction to gradients and automatic "
         "differentiation*",
         "https://www.tensorflow.org/guide/autodiff"),
    ])


# ==========================================================================
from core.palette import CLASS_COLORS  # noqa: E402  (used inside a lab)

SECTIONS = [
    ("B.1", "Four Ways to Get a Derivative", s_b1),
    ("B.2", "Forward Mode — Dual Numbers", s_b2),
    ("B.3", "Reverse Mode", s_b3),
    ("B.4", "Building an Engine", s_b4),
    ("B.5", "GradientTape in Practice", s_b5),
    ("B.6", "Reference & Further Reading", s_b6),
]

nav.render_chapter(CH, SECTIONS)
