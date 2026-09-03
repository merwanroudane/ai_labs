"""Chapter 18 — Reinforcement Learning."""

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
CH = "ch18"

hero(
    kicker="Part II · Chapter 18",
    title="Reinforcement Learning",
    blurb=(
        "No labelled examples — only a reward that arrives late and says nothing "
        "about which action caused it. This chapter derives the policy-gradient "
        "theorem and the Bellman equations from scratch, proves why value "
        "iteration converges, and builds REINFORCE, Q-learning, DQN and its "
        "variants against environments implemented from first principles."
    ),
    chips=["Bellman derived", "9 sub-sections", "9 animations",
           "9 code labs", "REINFORCE · DQN · A2C"],
)
nav.sidebar_tools(CH)


def _tf_ok() -> bool:
    # find_spec, not import: importing TensorFlow costs ~500 MB of RSS and we
    # only need to know whether the labs will be able to run.
    import importlib.util
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


if not _tf_ok():
    st.warning("TensorFlow is not importable here, so some labs will report an "
               "ImportError. Every explanation, animation and tabular lab still "
               "works.", icon="⚠️")


# ==========================================================================
def s_18_1():
    section("18.1", "Rewards, Policies and Why This Is Hard")

    lead(
        "An agent observes, acts, and receives a reward. That is the entire "
        "interface — and it makes reinforcement learning a fundamentally "
        "different problem from anything in the first seventeen chapters."
    )

    sub("The setting")

    md(
        "At each step $t$ the agent sees a state $s_t$, takes an action $a_t$ "
        "drawn from its **policy** $\\pi_\\theta(a \\mid s)$, receives a reward "
        "$r_t$, and moves to $s_{t+1}$. Its objective is the expected "
        "**return** — the discounted sum of future rewards."
    )

    math(r"""
    G_t \;=\; \sum_{k=0}^{\infty} \gamma^{k} r_{t+k},
    \qquad
    J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\bigl[G_0\bigr]
    """)

    where({
        r"\pi_\theta(a \mid s)": "the <b>policy</b> — a distribution over actions",
        r"\gamma \in [0, 1)": "the <b>discount factor</b>",
        r"\tau": "a <b>trajectory</b> $(s_0, a_0, r_0, s_1, \\dots)$",
    })

    sub("Three things that make this hard")

    table(
        ["Difficulty", "What it means", "Contrast with supervised learning"],
        [["<b>Credit assignment</b>",
          "A reward at step 200 may be due to an action at step 3",
          "A label tells you the right answer <i>for that input</i>"],
         ["<b>Exploration vs exploitation</b>",
          "You only see the outcome of the action you took, never the others",
          "The training set is fixed and given"],
         ["<b>Non-stationarity</b>",
          "Changing the policy changes the data distribution it generates",
          "The data distribution is fixed"]],
    )

    pitfall(
        "The data distribution depends on the parameters — every i.i.d. "
        "assumption is void",
        "In supervised learning the training set is drawn once and does not "
        "care what your model does. In RL, improving the policy changes which "
        "states you visit, which changes the gradient, which changes the policy. "
        "This feedback loop is why RL training curves are so much noisier than "
        "supervised ones, why a single bad update can be unrecoverable, and why "
        "'it worked with seed 42' is a real and widely-reported problem.",
    )

    sub("The discount factor")

    derive(
        [("<b>Why discount at all?</b> Three separate reasons, and they matter "
          "differently.", None),
         ("<b>1. Mathematical.</b> For a continuing task the undiscounted sum "
          "$\\sum_t r_t$ may not converge. With $|r_t| \\le R_{\\max}$ and "
          "$\\gamma < 1$ the geometric series is bounded:",
          r"\bigl|G_t\bigr| \;\le\; \sum_{k=0}^{\infty}\gamma^{k}R_{\max}"
          r" \;=\; \frac{R_{\max}}{1-\gamma}"),
         ("<b>2. Modelling.</b> Uncertainty about the far future is real; "
          "$\\gamma$ encodes 'a reward now is worth more than the same reward "
          "later', exactly like a financial discount rate.", None),
         ("<b>3. Algorithmic.</b> $\\gamma$ is what makes the Bellman operator a "
          "<b>contraction</b> (§18.4) and therefore makes value iteration "
          "converge at all.", None),
         ("<b>The effective horizon</b> is the number of steps that materially "
          "contribute. Since $\\sum_k \\gamma^k = 1/(1-\\gamma)$:",
          r"H_{\text{eff}} \;\approx\; \frac{1}{1-\gamma}"),
         ("So $\\gamma = 0.95$ sees about 20 steps ahead, $\\gamma = 0.99$ about "
          "100, and $\\gamma = 0.999$ about 1 000. <b>Choose $\\gamma$ from the "
          "timescale of your problem</b>, not by copying a default: if the reward "
          "arrives 300 steps after the action that caused it, $\\gamma = 0.95$ "
          "makes it invisible.", None)],
        title="What γ actually controls",
    )

    warn(
        "Reward shaping changes the optimal policy unless it is a potential "
        "function",
        "Adding a hand-crafted bonus to guide learning is tempting and usually "
        "wrong: agents optimise what you wrote, not what you meant. The one safe "
        "form is <b>potential-based shaping</b> (Ng, Harada & Russell, 1999): "
        "$F(s, s') = \\gamma\\Phi(s') - \\Phi(s)$ for any function $\\Phi$ of the "
        "state. That form — and provably only that form — leaves the optimal "
        "policy unchanged while still guiding learning. Any other bonus creates "
        "new optima, and the agent will find them.",
    )

    anim_header("Credit assignment: which action earned the reward?")

    T_ep = 24
    rng = np.random.default_rng(3)
    key_step = 6
    rewards = np.zeros(T_ep)
    rewards[-1] = 1.0
    frames = []
    for gi, g in enumerate([0.5, 0.8, 0.9, 0.95, 0.99, 0.999]):
        credit = np.array([g ** (T_ep - 1 - t) for t in range(T_ep)])
        cols = [C["danger"] if t == key_step else C["primary"]
                for t in range(T_ep)]
        frames.append(go.Frame(name=f"{g:g}", data=[
            go.Bar(x=np.arange(T_ep), y=credit, marker=dict(color=cols)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"γ = {g:g}   ·   effective horizon ≈ {1/(1-g):.0f} steps   ·   "
            f"credit reaching the decisive action at step {key_step} = "
            f"{credit[key_step]:.4f}"
            + ("   ·   INVISIBLE" if credit[key_step] < .05 else ""),
            color=C["danger"] if credit[key_step] < .05 else C["success"])])))

    c0 = np.array([0.5 ** (T_ep - 1 - t) for t in range(T_ep)])
    f = go.Figure(data=[go.Bar(x=np.arange(T_ep), y=c0,
                               marker=dict(color=[C["danger"] if t == key_step
                                                  else C["primary"]
                                                  for t in range(T_ep)]))])
    f.add_annotation(x=T_ep - 1, y=1.05, text="reward arrives here",
                     showarrow=True, arrowhead=2, ay=-30)
    f.update_layout(height=420, xaxis_title="time step",
                    yaxis_title="discounted credit γ^(T−t)",
                    yaxis=dict(range=[0, 1.15]),
                    title="How far back a single terminal reward reaches")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="γ = ")
    figure(f, "The decisive action is red. With γ = 0.5 it receives credit of "
              "3×10⁻⁶ — the algorithm cannot possibly learn it.")

    code_lab(
        "The environment interface, returns, and what γ does",
        '''import numpy as np
from core.rl import CartPole

rng = np.random.default_rng(42)

# ============ 1. THE INTERFACE =========================================
env = CartPole()
obs, info = env.reset(seed=42)
print("=== the entire RL interface ===")
print(f"  observation: {np.round(obs, 4)}")
print(f"    [cart position, cart velocity, pole angle, pole angular velocity]")
print(f"  actions: 0 = push left, 1 = push right")
print(f"  reward : +1 for every step the pole stays up")
print(f"  done   : |angle| > 12 degrees, or |x| > 2.4, or 500 steps")
print()
for i in range(4):
    a = int(rng.integers(2))
    obs, r, term, trunc, _ = env.step(a)
    print(f"  step {i}: action={a}  obs={np.round(obs, 4)}  "
          f"reward={r}  done={term or trunc}")

# ============ 2. A RANDOM POLICY IS THE BASELINE =======================
def run_episode(policy, seed, max_steps=500, gamma=1.0):
    e = CartPole()
    s, _ = e.reset(seed=seed)
    rewards, states, actions = [], [], []
    for _ in range(max_steps):
        a = policy(s)
        states.append(s); actions.append(a)
        s, r, term, trunc, _ = e.step(a)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(rewards), np.array(states), np.array(actions)

random_policy = lambda s: int(np.random.default_rng().integers(2))
lengths = [len(run_episode(random_policy, seed=i)[0]) for i in range(200)]
print()
print("=== a random policy, 200 episodes ===")
print(f"  mean {np.mean(lengths):.1f}   std {np.std(lengths):.1f}   "
      f"min {min(lengths)}   max {max(lengths)}")

# ============ 3. A HAND-CODED POLICY ===================================
def hardcoded(s):
    """Push in the direction the pole is falling."""
    return 0 if s[2] < 0 else 1

def hardcoded_better(s):
    """Also account for the angular VELOCITY -- act before it falls."""
    return 0 if s[2] + 0.4*s[3] < 0 else 1

print()
print("=== hand-coded policies ===")
print(f"{'policy':<34}{'mean length':>14}{'std':>9}{'max':>7}")
for nm, pol in [("random", random_policy),
                ("react to the angle", hardcoded),
                ("angle + 0.4 * angular velocity", hardcoded_better)]:
    L = [len(run_episode(pol, seed=i)[0]) for i in range(200)]
    print(f"{nm:<34}{np.mean(L):>14.1f}{np.std(L):>9.1f}{max(L):>7}")
print("  the second policy is better because it ACTS ON THE DERIVATIVE --")
print("  by the time the angle is large it is already too late.")

# ============ 4. RETURNS AND THE DISCOUNT FACTOR =======================
print()
print("=== the discounted return ===")
r_, _, _ = run_episode(hardcoded_better, seed=0)
print(f"  episode length {len(r_)}, undiscounted return {r_.sum():.0f}")
print(f"{'gamma':>9}{'return G_0':>14}{'effective horizon':>21}"
      f"{'1/(1-gamma)':>15}")
for g in [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]:
    G = sum(g**k * rr for k, rr in enumerate(r_))
    h = np.inf if g >= 1 else 1/(1-g)
    # the step at which 95 % of the discounted mass has accumulated
    cum = np.cumsum([g**k for k in range(len(r_))])
    eff = int(np.searchsorted(cum, .95*cum[-1])) + 1 if g < 1 else len(r_)
    print(f"{g:>9.3f}{G:>14.2f}{eff:>21}{h:>15.1f}")
print("  gamma=0 is myopic: only the immediate reward counts.")
print("  gamma=1 is only valid for EPISODIC tasks that terminate.")

# ============ 5. CREDIT ASSIGNMENT, MEASURED ===========================
print()
print("=== how much credit reaches an early action? ===")
T = 200
print(f"  an episode of {T} steps with the reward at the END:")
print(f"{'gamma':>9}{'credit at t=0':>17}{'credit at t=100':>19}"
      f"{'credit at t=190':>19}")
for g in [0.9, 0.95, 0.99, 0.999]:
    print(f"{g:>9.3f}{g**T:>17.3e}{g**(T-100):>19.3e}{g**(T-190):>19.3e}")
print("  with gamma=0.9 an action at t=0 receives 7e-10 of the reward.")
print("  IT IS NOT LEARNABLE. Choose gamma from your problem's TIMESCALE.")

# ============ 6. THE NON-STATIONARITY PROBLEM ==========================
print()
print("=== changing the policy changes the DATA ===")
for nm, pol in [("random", random_policy),
                ("angle + velocity", hardcoded_better)]:
    _, S, _ = run_episode(pol, seed=0, max_steps=500)
    print(f"  {nm:<20} visited states: "
          f"|x| up to {np.abs(S[:,0]).max():.3f}, "
          f"|angle| up to {np.abs(S[:,2]).max():.4f}")
print("  the better policy NEVER VISITS the states where the pole is falling.")
print("  so its training data contains no examples of recovering from them.")
print("  that feedback loop is why RL is unstable in a way supervised")
print("  learning is not.")

# ============ 7. POTENTIAL-BASED REWARD SHAPING ========================
print()
print("=== the ONLY safe form of reward shaping ===")
print("  F(s, s') = gamma * Phi(s') - Phi(s)   for ANY Phi")
gamma = 0.99
def Phi(s): return -abs(s[2])*10                  # prefer an upright pole
_, S, _ = run_episode(hardcoded_better, seed=0)
shaped = [gamma*Phi(S[i+1]) - Phi(S[i]) for i in range(len(S)-1)]
telescoped = sum(gamma**i * shaped[i] for i in range(len(shaped)))
direct = gamma**(len(S)-1)*Phi(S[-1]) - Phi(S[0])
print(f"  sum of discounted shaping rewards : {telescoped:.6f}")
print(f"  gamma^T * Phi(s_T) - Phi(s_0)     : {direct:.6f}")
print(f"  identical: {np.isclose(telescoped, direct)}")
print("  the shaping TELESCOPES: its total contribution depends only on the")
print("  first and last states, so it cannot change which policy is optimal.")
print("  ANY OTHER BONUS CAN, and the agent will find the exploit.")

import plotly.graph_objects as go
fig = go.Figure()
for nm, pol, col in [("random", random_policy, C["muted"]),
                     ("angle only", hardcoded, C["warning"]),
                     ("angle + velocity", hardcoded_better, C["success"])]:
    L = [len(run_episode(pol, seed=i)[0]) for i in range(200)]
    fig.add_trace(go.Histogram(x=L, name=nm, opacity=.65,
                               marker=dict(color=col), nbinsx=40))
fig.update_layout(height=400, barmode="overlay", xaxis_title="episode length",
                  yaxis_title="count",
                  title="Three policies on CartPole")
''',
        key="ch18_intro",
    )

    keypoints([
        "RL optimises $J(\\theta) = \\mathbb{E}_\\tau[G_0]$ with $G_t = \\sum_k "
        "\\gamma^k r_{t+k}$ — no labels, only reward.",
        "Three hard parts: <b>credit assignment</b>, <b>exploration</b>, and "
        "<b>non-stationarity</b>.",
        "$\\gamma$ sets the effective horizon $\\approx 1/(1-\\gamma)$ — choose "
        "it from your problem's timescale.",
        "Changing the policy changes the data distribution; every i.i.d. "
        "assumption is void.",
        "Only <b>potential-based</b> shaping $\\gamma\\Phi(s') - \\Phi(s)$ leaves "
        "the optimal policy unchanged.",
    ])


# ==========================================================================
def s_18_2():
    section("18.2", "Neural Network Policies and Policy Search")

    lead(
        "A policy is just a function from state to action, so a neural network "
        "can be one. The interesting question is how to train it when there are "
        "no labels."
    )

    sub("Stochastic policies")

    md(
        "A neural policy outputs a **distribution** over actions, and the agent "
        "samples from it. For CartPole, one sigmoid output is enough:"
    )

    math(r"""
    \pi_\theta(a{=}1 \mid s) = \sigma\bigl(f_\theta(s)\bigr),
    \qquad
    a \sim \mathrm{Bernoulli}\bigl(\pi_\theta(1 \mid s)\bigr)
    """)

    idea(
        "Why sample rather than take the argmax",
        "Three reasons, and all three matter. <b>Exploration</b>: a deterministic "
        "policy never tries the other action, so it can never discover it was "
        "better. <b>Differentiability</b>: the policy gradient (§18.3) requires a "
        "probability to differentiate — an argmax has zero gradient almost "
        "everywhere. <b>Optimality</b>: in partially observable or adversarial "
        "settings the <i>optimal</i> policy is genuinely stochastic (think "
        "rock–paper–scissors), so a deterministic policy class cannot contain "
        "the answer.",
    )

    sub("Policy search without gradients")

    table(
        ["Method", "How", "When it is the right choice"],
        [["<b>Random search</b>", "Sample parameters, keep the best",
          "Only for a handful of parameters"],
         ["<b>Hill climbing</b>", "Perturb the best, keep improvements",
          "Smooth, low-dimensional landscapes"],
         ["<b>Cross-entropy method (CEM)</b>",
          "Sample a population, refit a Gaussian to the top $k$ %",
          "<b>A genuinely strong baseline</b> — surprisingly hard to beat on "
          "small problems"],
         ["<b>Evolution strategies (ES)</b>",
          "Estimate the gradient from a population of perturbations",
          "<b>Massively parallel</b>; needs no backpropagation through time"],
         ["<b>Policy gradients</b>", "Differentiate $J(\\theta)$ — §18.3",
          "The default once the parameter count is large"]],
    )

    proof(
        "Evolution strategies are a gradient method in disguise",
        "ES perturbs the parameters with Gaussian noise and averages the returns, "
        "weighted by the noise: $\\nabla_\\theta J \\approx "
        "\\frac{1}{\\sigma N}\\sum_i R(\\theta + \\sigma\\boldsymbol\\epsilon_i)"
        "\\,\\boldsymbol\\epsilon_i$. That is exactly the <b>score-function "
        "gradient estimator</b> applied to a Gaussian smoothing of $J$ — the same "
        "estimator REINFORCE uses, but with the randomness placed in "
        "<i>parameter</i> space rather than in <i>action</i> space. Its practical "
        "advantage is that workers need to exchange only a random seed and a "
        "scalar return, which parallelises across thousands of machines with "
        "almost no communication. Its disadvantage is that the variance scales "
        "with the number of parameters, not the number of actions.",
    )

    warn(
        "Never evaluate an RL policy on one episode",
        "CartPole returns vary by a factor of ten across random initial states. "
        "A policy that scores 500 once may average 60. <b>Always average over at "
        "least 20–100 episodes with different seeds</b>, and report the standard "
        "deviation. Most irreproducible RL results are this mistake.",
    )

    anim_header("The cross-entropy method narrowing on a solution")

    rng = np.random.default_rng(11)

    def fitness(P):
        # a landscape with a decoy optimum
        x, y = P[:, 0], P[:, 1]
        good = 1.0*np.exp(-((x-1.4)**2 + (y-1.1)**2)/0.55)
        decoy = 0.62*np.exp(-((x+1.3)**2 + (y+1.0)**2)/0.30)
        return good + decoy - 0.02*(x**2 + y**2)

    mu = np.array([-1.0, -0.8]); sig = np.array([1.5, 1.5])
    frames, snaps = [], []
    for it in range(16):
        pop = rng.normal(mu, sig, (90, 2))
        fit = fitness(pop)
        elite = pop[np.argsort(-fit)[:18]]
        snaps.append((pop.copy(), elite.copy(), mu.copy(), sig.copy(),
                      float(fit.max()), float(fitness(mu[None])[0])))
        mu, sig = elite.mean(0), elite.std(0) + 1e-3

    gx = np.linspace(-3.5, 3.5, 60)
    GX, GY = np.meshgrid(gx, gx)
    Zf = fitness(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    for it, (pop, elite, m, s, best, atmu) in enumerate(snaps):
        frames.append(go.Frame(name=str(it + 1), data=[
            go.Contour(x=gx, y=gx, z=Zf, colorscale=nav.cscale(),
                       showscale=False, opacity=.55,
                       contours=dict(showlines=False)),
            go.Scatter(x=pop[:, 0], y=pop[:, 1], mode="markers",
                       marker=dict(size=5, color=alpha(C["muted"], .75))),
            go.Scatter(x=elite[:, 0], y=elite[:, 1], mode="markers",
                       marker=dict(size=8, color=C["danger"],
                                   line=dict(color="#fff", width=1))),
            go.Scatter(x=[m[0]], y=[m[1]], mode="markers",
                       marker=dict(size=16, color=C["success"], symbol="star",
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"generation {it+1}   ·   μ = ({m[0]:.2f}, {m[1]:.2f})   ·   "
            f"σ = ({s[0]:.3f}, {s[1]:.3f})   ·   best fitness {best:.3f}   ·   "
            f"fitness at μ {atmu:.3f}")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=470, xaxis=dict(range=[-3.5, 3.5], title="θ₁"),
                    yaxis=dict(range=[-3.5, 3.5], title="θ₂", scaleanchor="x"),
                    title="Cross-entropy method: sample, keep the elite, refit")
    anim.animate(f, frames, duration=nav.anim_ms(750), slider_prefix="gen ")
    figure(f, "CEM starts on the decoy peak and escapes because the initial σ is "
              "wide enough to sample the real one. Initial σ is the exploration "
              "parameter.")

    code_lab(
        "A neural policy, random search, hill climbing and CEM",
        '''import numpy as np, time
from core.rl import CartPole

rng = np.random.default_rng(42)

# ============ 1. A LINEAR STOCHASTIC POLICY ============================
def policy_probs(theta, s):
    """theta: (4,) weights + 1 bias. Returns P(action = 1)."""
    z = float(s @ theta[:4] + theta[4])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def evaluate(theta, n_episodes=20, seed0=0, stochastic=True, max_steps=500):
    env = CartPole()
    total = []
    r = np.random.default_rng(seed0)
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed0 + i)
        ret = 0.0
        for _ in range(max_steps):
            p = policy_probs(theta, s)
            a = int(r.random() < p) if stochastic else int(p > .5)
            s, rew, term, trunc, _ = env.step(a)
            ret += rew
            if term or trunc:
                break
        total.append(ret)
    return float(np.mean(total)), float(np.std(total))

theta0 = rng.normal(0, .3, 5)
m, sd = evaluate(theta0)
print("=== a linear stochastic policy ===")
print(f"  5 parameters: 4 weights + 1 bias")
print(f"  random init  -> mean return {m:.1f} +/- {sd:.1f}")

# ============ 2. WHY YOU MUST AVERAGE OVER MANY EPISODES ===============
print()
print("=== the single-episode trap ===")
good = np.array([0.1, 0.6, 3.0, 1.5, 0.0])          # a decent policy
singles = [evaluate(good, n_episodes=1, seed0=i)[0] for i in range(25)]
print(f"  25 single-episode evaluations of the SAME policy:")
print(f"    {np.array(singles).astype(int)}")
print(f"  min {min(singles):.0f}, max {max(singles):.0f}, "
      f"mean {np.mean(singles):.1f}, std {np.std(singles):.1f}")
print(f"  a one-episode result is off by up to "
      f"{max(abs(np.array(singles)-np.mean(singles))):.0f}. ALWAYS AVERAGE.")

# ============ 3. RANDOM SEARCH =========================================
print()
print("=== random search ===")
t0 = time.perf_counter()
best_theta, best_score = None, -np.inf
history_rand = []
for i in range(120):
    th = rng.normal(0, 1.5, 5)
    sc, _ = evaluate(th, n_episodes=8)
    if sc > best_score:
        best_score, best_theta = sc, th
    history_rand.append(best_score)
print(f"  120 samples in {time.perf_counter()-t0:.1f}s "
      f"-> best {best_score:.1f}")
print(f"  verified on 50 fresh episodes: "
      f"{evaluate(best_theta, n_episodes=50, seed0=999)[0]:.1f}")

# ============ 4. HILL CLIMBING =========================================
print()
print("=== hill climbing (perturb the best, keep improvements) ===")
theta = rng.normal(0, .3, 5)
score, _ = evaluate(theta, n_episodes=8)
sigma = 1.0
history_hill = [score]
for i in range(120):
    cand = theta + rng.normal(0, sigma, 5)
    sc, _ = evaluate(cand, n_episodes=8)
    if sc > score:
        theta, score = cand, sc
        sigma = max(0.05, sigma*0.85)         # ADAPTIVE: narrow on success
    else:
        sigma = min(2.0, sigma*1.05)          # widen on failure
    history_hill.append(score)
print(f"  best {score:.1f}, final sigma {sigma:.3f}")
print(f"  verified on 50 fresh episodes: "
      f"{evaluate(theta, n_episodes=50, seed0=999)[0]:.1f}")

# ============ 5. THE CROSS-ENTROPY METHOD ==============================
print()
print("=== cross-entropy method ===")
mu = np.zeros(5); sig = np.ones(5)*1.5
POP, ELITE = 40, 8
history_cem = []
t0 = time.perf_counter()
print(f"{'gen':>5}{'mean elite':>13}{'best':>9}{'||sigma||':>12}")
for gen in range(14):
    pop = rng.normal(mu, sig, (POP, 5))
    scores = np.array([evaluate(p, n_episodes=6)[0] for p in pop])
    elite = pop[np.argsort(-scores)[:ELITE]]
    mu, sig = elite.mean(0), elite.std(0) + 1e-2
    history_cem.append(float(scores.max()))
    if gen % 3 == 0 or gen == 13:
        print(f"{gen:>5}{np.sort(scores)[-ELITE:].mean():>13.1f}"
              f"{scores.max():>9.1f}{np.linalg.norm(sig):>12.3f}")
print(f"  {time.perf_counter()-t0:.1f}s")
cem_score, cem_sd = evaluate(mu, n_episodes=50, seed0=999)
print(f"  final mu verified on 50 fresh episodes: {cem_score:.1f} +/- {cem_sd:.1f}")
print(f"  learned weights: {np.round(mu, 3)}")
print(f"    the pole-angle weight is {mu[2]:.2f} and the angular-velocity")
print(f"    weight is {mu[3]:.2f} -- it rediscovered the hand-coded rule.")

# ============ 6. EVOLUTION STRATEGIES ==================================
print()
print("=== evolution strategies (a gradient estimate from a population) ===")
theta = np.zeros(5)
SIGMA_ES, N_POP, LR = 0.6, 40, 0.35
history_es = []
for it in range(30):
    eps = rng.normal(0, 1, (N_POP, 5))
    R = np.array([evaluate(theta + SIGMA_ES*e, n_episodes=5)[0] for e in eps])
    A = (R - R.mean()) / (R.std() + 1e-8)          # rank/standardise: vital
    grad = (A[:, None] * eps).sum(0) / (N_POP * SIGMA_ES)
    theta = theta + LR * grad
    history_es.append(float(R.mean()))
es_score, es_sd = evaluate(theta, n_episodes=50, seed0=999)
print(f"  grad ~ (1/(sigma*N)) * sum_i A_i * eps_i")
print(f"  after 30 iterations: {es_score:.1f} +/- {es_sd:.1f}")
print(f"  NOTE the standardisation of R. Without it the step size depends")
print(f"  on the reward SCALE, and the method is unusable across problems.")

# ============ 7. HOW THEY COMPARE ======================================
print()
print("=== final comparison, 50 fresh episodes each ===")
print(f"{'method':<26}{'mean return':>14}{'std':>9}{'evaluations used':>19}")
for nm, th, n_ev in [("random search", best_theta, 120*8),
                     ("hill climbing", theta if False else theta, 120*8),
                     ("cross-entropy method", mu, 14*40*6),
                     ("evolution strategies", theta, 30*40*5)]:
    a, b = evaluate(th, n_episodes=50, seed0=999)
    print(f"{nm:<26}{a:>14.1f}{b:>9.1f}{n_ev:>19,}")
print()
print("  on 5 parameters, gradient-free search is EXCELLENT. It stops")
print("  scaling somewhere around a few thousand parameters, which is why")
print("  section 18.3 derives an actual gradient.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=history_rand, mode="lines", name="random search (best so far)",
                line=dict(color=C["muted"], width=2.5))
fig.add_scatter(y=history_hill, mode="lines", name="hill climbing",
                line=dict(color=C["warning"], width=2.5))
fig.add_scatter(x=np.arange(len(history_cem))*POP, y=history_cem, mode="lines+markers",
                name="CEM (per generation)", line=dict(color=C["success"], width=3))
fig.update_layout(height=400, xaxis_title="policy evaluations",
                  yaxis_title="best mean return",
                  title="Gradient-free policy search on CartPole")
''',
        key="ch18_policysearch",
    )

    keypoints([
        "A neural policy outputs a <b>distribution</b>; sampling gives "
        "exploration, differentiability, and optimality in stochastic games.",
        "<b>Always average over 20–100 episodes</b> — single-episode evaluation "
        "is the most common RL reporting error.",
        "<b>CEM</b> and <b>ES</b> are strong baselines at small parameter counts "
        "and trivially parallel.",
        "ES is the <b>score-function estimator</b> with noise in parameter space "
        "rather than action space.",
        "Standardise returns before using them as weights, or the step size "
        "depends on the reward scale.",
    ])


# ==========================================================================
def s_18_3():
    section("18.3", "Policy Gradients and REINFORCE")

    lead(
        "The objective is an expectation over trajectories the policy itself "
        "generates. Differentiating that looks impossible — and one identity "
        "makes it easy."
    )

    sub("The policy-gradient theorem")

    derive(
        [("We want $\\nabla_\\theta J(\\theta)$ where "
          "$J(\\theta) = \\mathbb{E}_{\\tau\\sim p_\\theta}[R(\\tau)]$. Write the "
          "expectation as an integral:",
          r"\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) R(\tau)\, d\tau"
          r" = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau"),
         ("The problem: $\\nabla_\\theta p_\\theta(\\tau)$ is not an expectation, "
          "so we cannot estimate it by sampling. <b>The log-derivative "
          "trick</b> fixes that — multiply and divide by $p_\\theta$:",
          r"\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,"
          r"\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}"
          r" = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)"),
         ("Substituting turns it back into an expectation, which we <b>can</b> "
          "estimate from samples:",
          r"\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\bigl[\,"
          r"R(\tau)\,\nabla_\theta \log p_\theta(\tau)\,\bigr]"),
         ("Now expand $\\log p_\\theta(\\tau)$. A trajectory's probability is "
          "the initial state, times the transitions, times the policy's action "
          "choices:",
          r"\log p_\theta(\tau) = \log p(s_0) + \sum_t \Bigl[\log "
          r"p(s_{t+1}\mid s_t, a_t) + \log \pi_\theta(a_t \mid s_t)\Bigr]"),
         ("<b>Only the policy terms depend on $\\theta$.</b> The environment "
          "dynamics $p(s_{t+1} \\mid s_t, a_t)$ drop out of the gradient "
          "entirely — which is why this is a <b>model-free</b> method: we never "
          "need to know them.",
          r"\nabla_\theta \log p_\theta(\tau) = \sum_{t}"
          r"\nabla_\theta \log \pi_\theta(a_t \mid s_t)"),
         ("Giving the <b>policy-gradient theorem</b>:",
          r"\boxed{\;\nabla_\theta J = \mathbb{E}\left[\sum_{t=0}^{T}"
          r" G_t \,\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]\;}"),
         ("<b>Read it as an instruction:</b> increase the log-probability of "
          "actions that were followed by high return, decrease it for actions "
          "followed by low return. It is weighted maximum likelihood, where the "
          "weights are the returns.", None)],
        title="Deriving the policy gradient",
    )

    sub("Causality and baselines — two free variance reductions")

    derive(
        [("<b>Causality.</b> An action at time $t$ cannot influence rewards "
          "received before $t$. Formally, "
          "$\\mathbb{E}[r_{t'} \\nabla \\log\\pi_\\theta(a_t\\mid s_t)] = 0$ for "
          "$t' < t$, so those terms contribute nothing but noise. Replace the "
          "full return with the <b>reward-to-go</b>:",
          r"\nabla_\theta J = \mathbb{E}\left[\sum_t \left(\sum_{k \ge t}"
          r"\gamma^{k-t} r_k\right) \nabla_\theta \log \pi_\theta(a_t\mid s_t)"
          r"\right]"),
         ("<b>Baselines.</b> Subtract any function $b(s_t)$ that does not depend "
          "on the action. The estimator stays unbiased because:",
          r"\mathbb{E}_{a\sim\pi_\theta}\bigl[b(s)\nabla_\theta\log\pi_\theta"
          r"(a\mid s)\bigr] = b(s)\nabla_\theta \sum_a \pi_\theta(a\mid s)"
          r" = b(s)\,\nabla_\theta 1 = 0"),
         ("The sum of probabilities is the constant 1, so its gradient is zero. "
          "<b>Any</b> state-dependent baseline is therefore free.", None),
         ("The variance-minimising baseline is close to the <b>value function</b> "
          "$V^\\pi(s) = \\mathbb{E}[G_t \\mid s_t = s]$, giving the "
          "<b>advantage</b>:",
          r"A^\pi(s_t, a_t) = G_t - V^\pi(s_t)"),
         ("<b>Why this matters so much.</b> Without a baseline, if all returns "
          "are positive the gradient pushes <i>every</i> action's probability up, "
          "and learning depends entirely on the differences between large "
          "numbers. With a baseline, actions better than average go up and worse "
          "than average go down — which is both far lower variance and far more "
          "intuitive.", None)],
        title="Why reward-to-go and baselines are free",
    )

    pitfall(
        "REINFORCE's variance grows with the episode length",
        "The estimator sums $T$ terms, each weighted by a return that is itself a "
        "sum of up to $T$ rewards. Variance grows roughly as $\\mathcal{O}(T^2)$ "
        "without a baseline. This is why plain REINFORCE needs hundreds of "
        "episodes per update on long tasks, and why every practical method "
        "(A2C, PPO — §18.8) uses a learned value baseline and bootstrapped "
        "returns.",
    )

    codenote(
        "How to implement it in Keras",
        "There is no <code>fit</code> for this. The idiom: inside a "
        "<code>GradientTape</code>, compute the log-probability of the action "
        "that was actually taken, take its gradient, and <b>scale the resulting "
        "gradients by the advantage</b> before applying them. Equivalently — and "
        "more simply — minimise the surrogate loss "
        "$-\\sum_t A_t \\log \\pi_\\theta(a_t \\mid s_t)$, treating $A_t$ as a "
        "constant. It is <b>not</b> a real loss (its value is meaningless) but "
        "its gradient is the one you want.",
    )

    anim_header("How a baseline reduces the gradient's variance")

    rng = np.random.default_rng(9)
    n_actions_demo = 4
    true_q = np.array([1.0, 3.0, 2.2, 0.5])
    frames = []
    for k, use_base in enumerate([False, True]):
        for trial in range(1):
            samples = []
            for _ in range(400):
                a = rng.integers(n_actions_demo)
                G = true_q[a] + rng.normal(0, .55) + 10.0   # a large constant
                b = 10.0 + true_q.mean() if use_base else 0.0
                samples.append((a, G - b))
            samples = np.array(samples)
            means = [samples[samples[:, 0] == a, 1].mean()
                     for a in range(n_actions_demo)]
            stds = [samples[samples[:, 0] == a, 1].std()
                    for a in range(n_actions_demo)]
            frames.append(go.Frame(
                name="with baseline" if use_base else "no baseline",
                data=[go.Bar(x=[f"a{a}" for a in range(n_actions_demo)],
                             y=means,
                             error_y=dict(type="data", array=stds),
                             marker=dict(color=[C["success"] if m > 0
                                                else C["danger"]
                                                for m in means]))],
                layout=go.Layout(annotations=[anim.annotate_step(
                    ("WITH a value baseline: better-than-average actions go UP, "
                     "worse go DOWN"
                     if use_base else
                     "NO baseline: every return is positive, so EVERY action's "
                     "probability is pushed up") +
                    f"   ·   |mean| = {np.mean(np.abs(means)):.2f}, "
                    f"mean σ = {np.mean(stds):.2f}",
                    color=C["success"] if use_base else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.add_hline(y=0, line_color=C["ink"], line_width=1.5)
    f.update_layout(height=420, yaxis_title="gradient weight (G − b)",
                    xaxis_title="action",
                    title="The same returns, with and without a baseline")
    anim.animate(f, frames, duration=nav.anim_ms(2200), slider_prefix="")
    figure(f, "Without a baseline the algorithm still works — it relies on the "
              "differences between large positive numbers, which is exactly what "
              "makes it high-variance.")

    code_lab(
        "REINFORCE built up piece by piece, with every variance reduction measured",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core.rl import CartPole

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE LOG-DERIVATIVE TRICK, VERIFIED ====================
print("=== grad E[f] = E[f * grad log p] ===")
# a two-action policy with probability p of action 1, rewards r0, r1
r0, r1 = 1.0, 3.0
def J(z):
    p = 1/(1+np.exp(-z))
    return (1-p)*r0 + p*r1
z0 = 0.4
h = 1e-6
numeric = (J(z0+h) - J(z0-h))/(2*h)
# analytic via the trick: E[r * dlogp/dz]
p = 1/(1+np.exp(-z0))
analytic = p*r1*(1-p) + (1-p)*r0*(-p)
print(f"  numerical  dJ/dz = {numeric:.8f}")
print(f"  score-fn   dJ/dz = {analytic:.8f}")
print(f"  match: {np.isclose(numeric, analytic, atol=1e-5)}")
# and by SAMPLING, which is what REINFORCE does
rng = np.random.default_rng(0)
est = []
for n in [100, 1000, 10000, 100000]:
    a = (rng.random(n) < p).astype(float)
    r = np.where(a == 1, r1, r0)
    dlogp = np.where(a == 1, 1-p, -p)
    est.append((n, float((r*dlogp).mean()), float((r*dlogp).std()/np.sqrt(n))))
print(f"{'samples':>10}{'estimate':>13}{'std error':>13}")
for n, e, se in est:
    print(f"{n:>10,}{e:>13.6f}{se:>13.6f}")
print("  the estimator is UNBIASED but its variance falls only as 1/sqrt(n).")

# ============ 2. THE POLICY NETWORK ====================================
def make_policy(n_obs=4, hidden=32):
    return keras.Sequential([keras.layers.Input(shape=(n_obs,)),
                             keras.layers.Dense(hidden, activation="elu"),
                             keras.layers.Dense(1, activation="sigmoid")])

def make_step_fn(model):
    """One COMPILED step: sample an action and return grad log pi(a|s).

    Same computation as an eager tape, but traced once. The tape is per
    environment step here -- that is what makes the algorithm legible --
    so tracing it is worth roughly 40x on this lab.
    """
    @tf.function(reduce_retracing=True)
    def step(obs):
        with tf.GradientTape() as tape:
            p = model(obs, training=True)
            a = tf.cast(tf.random.uniform([1, 1]) > p, "float32")
            # log pi(a|s): for a=1 it is log(p), for a=0 it is log(1-p).
            y = 1.0 - a                 # Keras trick: target = 1 - action
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(y, p))
        return a, tape.gradient(loss, model.trainable_weights)
    return step


def play_episode(model, seed, max_steps=500, step_fn=None):
    """Returns rewards, and the per-step gradients of log pi(a|s)."""
    env = CartPole()
    s, _ = env.reset(seed=seed)
    step_fn = step_fn or make_step_fn(model)
    rewards, grads = [], []
    for _ in range(max_steps):
        a_t, g = step_fn(s[None].astype("float32"))
        a = int(a_t.numpy()[0, 0])
        grads.append(g)
        s, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term or trunc:
            break
    return rewards, grads

# ============ 3. RETURNS: RAW, REWARD-TO-GO, NORMALISED ================
def discount(rewards, gamma):
    """Reward-to-go: G_t = sum_{k>=t} gamma^(k-t) r_k."""
    out = np.zeros(len(rewards))
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma*run
        out[t] = run
    return out

demo = [1.0]*6
print()
print("=== the three ways to weight a step ===")
print(f"  rewards          : {demo}")
print(f"  FULL return      : {[float(sum(demo))]*6}   (same for every step)")
print(f"  reward-to-go     : {np.round(discount(demo, .95), 3).tolist()}")
g = discount(demo, .95)
print(f"  normalised       : {np.round((g-g.mean())/g.std(), 3).tolist()}")
print("  the full return gives every action in the episode the SAME credit,")
print("  including the ones taken after the outcome was already decided.")

def discount_and_normalise(all_rewards, gamma, normalise=True):
    disc = [discount(r, gamma) for r in all_rewards]
    if not normalise:
        return disc
    flat = np.concatenate(disc)
    m, s = flat.mean(), flat.std() + 1e-8
    return [(d - m)/s for d in disc]

# ============ 4. TRAINING ==============================================
def train_reinforce(n_iters=45, n_ep=10, gamma=0.97, lr=0.02,
                    use_rtg=True, normalise=True, seed=0, verbose=True):
    tf.random.set_seed(seed)
    model = make_policy()
    opt = keras.optimizers.Nadam(lr)
    step_fn = make_step_fn(model)          # trace once, reuse every episode
    hist = []
    for it in range(n_iters):
        all_rewards, all_grads = [], []
        for e in range(n_ep):
            r_, g_ = play_episode(model, seed=seed*1000 + it*100 + e,
                                  step_fn=step_fn)
            all_rewards.append(r_); all_grads.append(g_)
        lengths = [len(r) for r in all_rewards]
        if use_rtg:
            weights = discount_and_normalise(all_rewards, gamma, normalise)
        else:
            tot = [np.full(len(r), float(sum(r))) for r in all_rewards]
            if normalise:
                flat = np.concatenate(tot)
                weights = [(t - flat.mean())/(flat.std()+1e-8) for t in tot]
            else:
                weights = tot
        # weight each step's gradient by its return, then average
        mean_grads = []
        for vi in range(len(model.trainable_weights)):
            mean_grads.append(tf.reduce_mean(tf.stack(
                [w[step] * all_grads[ep][step][vi]
                 for ep, w in enumerate(weights)
                 for step in range(len(w))]), axis=0))
        opt.apply_gradients(zip(mean_grads, model.trainable_weights))
        hist.append(float(np.mean(lengths)))
        if verbose and it % 15 == 0:
            print(f"    iter {it:>3}: mean length {np.mean(lengths):>6.1f}")
    return model, hist

print()
print("=== training REINFORCE ===")
t0 = time.perf_counter()
model, hist = train_reinforce()
print(f"  {time.perf_counter()-t0:.1f}s")

def evaluate(model, n=30, seed0=5000):
    env = CartPole()
    out = []
    for i in range(n):
        s, _ = env.reset(seed=seed0+i)
        tot = 0
        for _ in range(500):
            p = float(model(s[None].astype("float32"), training=False))
            a = int(np.random.default_rng(seed0+i).random() < p) \\
                if False else int(p > .5)
            s, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        out.append(tot)
    return float(np.mean(out)), float(np.std(out))

m_, s_ = evaluate(model)
print(f"  greedy evaluation on 30 fresh episodes: {m_:.1f} +/- {s_:.1f}")

# ============ 5. THE VARIANCE REDUCTIONS, ABLATED ======================
print()
print("=== ablation: what each trick is worth ===")
print(f"{'variant':<44}{'final mean length':>20}{'greedy eval':>14}")
for nm, kw in [("full return, no normalisation",
                dict(use_rtg=False, normalise=False)),
               ("full return, normalised (a baseline)",
                dict(use_rtg=False, normalise=True)),
               ("reward-to-go, no normalisation",
                dict(use_rtg=True, normalise=False)),
               ("reward-to-go + normalisation (standard)",
                dict(use_rtg=True, normalise=True))]:
    m2, h2 = train_reinforce(n_iters=40, n_ep=10, verbose=False, **kw)
    print(f"{nm:<44}{np.mean(h2[-5:]):>20.1f}{evaluate(m2, n=20)[0]:>14.1f}")
print("  normalising the returns IS a baseline (subtracting the mean) plus")
print("  a scale normalisation. It is the single most valuable line.")

# ============ 6. WHY THE BASELINE IS FREE ==============================
print()
print("=== E[b(s) * grad log pi] = 0, verified ===")
pol = make_policy()
s_test = np.array([[0.1, -0.2, 0.03, 0.5]], dtype="float32")
p_ = float(pol(s_test))
# grad of log pi(a|s) wrt the OUTPUT probability, for both actions
g1 = 1.0/p_            # d log(p) / dp
g0 = -1.0/(1.0-p_)     # d log(1-p) / dp
expect = p_*g1 + (1-p_)*g0
print(f"  pi(1|s) = {p_:.5f}")
print(f"  E_a[d log pi(a|s) / dp] = {p_:.5f}*{g1:.4f} + "
      f"{1-p_:.5f}*({g0:.4f}) = {expect:.3e}")
print(f"  ~ 0, because sum_a pi(a|s) = 1 and the gradient of a constant is 0.")
print(f"  therefore ANY b(s) can be subtracted without introducing bias.")

# ============ 7. VARIANCE, MEASURED ====================================
print()
print("=== gradient variance with and without a baseline ===")
rs, _ = play_episode(model, seed=7)
G = discount(rs, 0.97)
print(f"  episode length {len(rs)}")
print(f"{'weighting':<34}{'mean':>10}{'std':>10}{'std/|mean|':>13}")
for nm, w in [("full return", np.full(len(rs), float(sum(rs)))),
              ("reward-to-go", G),
              ("reward-to-go - mean (baseline)", G - G.mean()),
              ("normalised", (G - G.mean())/(G.std()+1e-8))]:
    print(f"{nm:<34}{w.mean():>10.3f}{w.std():>10.3f}"
          f"{w.std()/(abs(w.mean())+1e-8):>13.3f}")
print("  the last column is the relative noise in the gradient estimate.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hist, mode="lines", name="mean episode length",
                line=dict(color=C["primary"], width=2.5))
w = 5
fig.add_scatter(y=np.convolve(hist, np.ones(w)/w, mode="valid"), mode="lines",
                name=f"{w}-iteration moving average",
                line=dict(color=C["danger"], width=3))
fig.update_layout(height=400, xaxis_title="training iteration",
                  yaxis_title="mean episode length",
                  title="REINFORCE on CartPole")
''',
        key="ch18_reinforce",
    )

    quiz(
        "Why can you subtract any state-dependent baseline $b(s)$ from the "
        "return without biasing the policy gradient?",
        ["Because $b(s)$ is small compared to the return",
         "Because $\\mathbb{E}_{a\\sim\\pi}[\\nabla_\\theta \\log\\pi_\\theta(a|s)]"
         " = \\nabla_\\theta \\sum_a \\pi_\\theta(a|s) = \\nabla_\\theta 1 = 0$",
         "Because the baseline is learned from the same data",
         "It does introduce bias, but a small one"],
        1,
        "The action probabilities always sum to 1, and the gradient of a constant "
        "is zero. So the baseline term has expectation exactly zero — the "
        "estimator stays unbiased for <i>any</i> $b(s)$ — while the variance can "
        "fall dramatically. That is why it is called a free lunch.",
        key="ch18q1",
    )

    keypoints([
        "The <b>log-derivative trick</b> turns $\\nabla \\mathbb{E}[R]$ into "
        "$\\mathbb{E}[R \\nabla \\log p]$, which is estimable from samples.",
        "The environment dynamics <b>cancel</b> — that is what makes policy "
        "gradients model-free.",
        "<b>Reward-to-go</b> (causality) and a <b>baseline</b> both reduce "
        "variance at zero bias cost.",
        "$A^\\pi(s,a) = G_t - V^\\pi(s_t)$ is the advantage — better or worse "
        "than average.",
        "REINFORCE's variance grows with $T$; every practical method learns a "
        "value baseline.",
    ])

# ==========================================================================
def s_18_4():
    section("18.4", "Markov Decision Processes and the Bellman Equations")

    lead(
        "If the environment is Markovian and known, the optimal policy can be "
        "computed exactly — and the proof that it can is the theoretical "
        "backbone of everything that follows."
    )

    sub("The Markov property")

    md(
        "A process is **Markovian** when the next state depends only on the "
        "current state and action, not on the history:"
    )

    math(r"""
    P\bigl(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots\bigr)
    \;=\; P\bigl(s_{t+1} \mid s_t, a_t\bigr)
    """)

    note(
        "The Markov property is a property of the <b>state representation</b>, "
        "not of the world",
        "A single frame of Pong is <b>not</b> Markovian: you cannot tell which "
        "way the ball is moving. Stack four frames and it is. Almost every "
        "'partially observable' problem is really an under-specified state — the "
        "standard fixes are frame stacking, an RNN over observations, or "
        "including velocity terms explicitly. CartPole's state includes both "
        "positions <i>and</i> velocities for exactly this reason.",
    )

    sub("Value functions")

    math(r"""
    V^{\pi}(s) = \mathbb{E}_{\pi}\bigl[G_t \mid s_t = s\bigr],
    \qquad
    Q^{\pi}(s, a) = \mathbb{E}_{\pi}\bigl[G_t \mid s_t = s,\, a_t = a\bigr]
    """)

    derive(
        [("<b>The Bellman expectation equation.</b> Split the return into the "
          "immediate reward and the rest:",
          r"G_t = r_t + \gamma G_{t+1}"),
         ("Take the expectation under $\\pi$ and condition on $s_t = s$. The "
          "inner expectation of $G_{t+1}$ given $s_{t+1}$ is by definition "
          "$V^\\pi(s_{t+1})$:",
          r"V^{\pi}(s) = \sum_{a}\pi(a\mid s)\sum_{s'} P(s'\mid s,a)\Bigl["
          r"R(s,a,s') + \gamma V^{\pi}(s')\Bigr]"),
         ("<b>The Bellman optimality equation.</b> The optimal value function "
          "takes the best action rather than averaging over the policy:",
          r"V^{*}(s) = \max_{a} \sum_{s'} P(s'\mid s,a)\Bigl["
          r"R(s,a,s') + \gamma V^{*}(s')\Bigr]"),
         ("and in $Q$ form — note the $\\max$ moves <b>inside</b>, over the next "
          "state's actions:",
          r"Q^{*}(s,a) = \sum_{s'} P(s'\mid s,a)\Bigl[R(s,a,s') + "
          r"\gamma \max_{a'} Q^{*}(s',a')\Bigr]"),
         ("<b>Once you have $Q^*$, the optimal policy is free</b> — no search, no "
          "planning, just an argmax:",
          r"\pi^{*}(s) = \arg\max_{a} Q^{*}(s,a)"),
         ("<b>That is the entire reason $Q$ is preferred to $V$.</b> Acting "
          "greedily on $V^*$ requires a one-step lookahead through the "
          "transition model $P$; acting greedily on $Q^*$ requires nothing at "
          "all. It is what makes model-free control possible.", None)],
        title="The Bellman equations",
    )

    sub("Why value iteration converges")

    derive(
        [("Define the <b>Bellman optimality operator</b> $\\mathcal{T}$ on value "
          "functions:",
          r"(\mathcal{T}V)(s) = \max_a \sum_{s'} P(s'\mid s,a)\bigl["
          r"R(s,a,s') + \gamma V(s')\bigr]"),
         ("<b>Claim: $\\mathcal{T}$ is a $\\gamma$-contraction in the sup norm.</b> "
          "Take any two value functions $U, V$. Using "
          "$|\\max_a f(a) - \\max_a g(a)| \\le \\max_a |f(a) - g(a)|$:",
          r"\bigl|(\mathcal{T}U)(s) - (\mathcal{T}V)(s)\bigr| \le "
          r"\max_a \gamma \sum_{s'} P(s'\mid s,a)\,\bigl|U(s') - V(s')\bigr|"),
         ("The probabilities sum to 1, so the weighted average is at most the "
          "maximum:",
          r"\bigl\lVert \mathcal{T}U - \mathcal{T}V \bigr\rVert_{\infty}"
          r" \;\le\; \gamma\, \bigl\lVert U - V \bigr\rVert_{\infty}"),
         ("By the <b>Banach fixed-point theorem</b>, a contraction on a complete "
          "metric space has a <b>unique</b> fixed point, and iterating converges "
          "to it from <b>any</b> starting point. The fixed point of "
          "$\\mathcal{T}$ is $V^*$.", None),
         ("The convergence is geometric, at rate $\\gamma$:",
          r"\bigl\lVert V_k - V^{*} \bigr\rVert_{\infty} \le "
          r"\gamma^{k}\,\bigl\lVert V_0 - V^{*}\bigr\rVert_{\infty}"),
         ("<b>Note where $\\gamma < 1$ was used: it is the contraction factor.</b> "
          "At $\\gamma = 1$ the operator is merely non-expansive and the theorem "
          "gives nothing — which is the precise mathematical reason discounting "
          "is not optional for continuing tasks.", None)],
        title="Value iteration converges because 𝒯 is a contraction",
    )

    table(
        ["Algorithm", "Update", "Cost per sweep", "Iterations"],
        [["<b>Value iteration</b>",
          "$V \\leftarrow \\mathcal{T}V$",
          "$\\mathcal{O}(|S|^2|A|)$",
          "$\\mathcal{O}\\bigl(\\log(1/\\epsilon)/\\log(1/\\gamma)\\bigr)$"],
         ["<b>Policy iteration</b>",
          "Evaluate $\\pi$ exactly, then improve greedily",
          "$\\mathcal{O}(|S|^3 + |S|^2|A|)$",
          "<b>Very few</b> — often < 10"],
         ["<b>Modified policy iteration</b>",
          "Evaluate approximately ($k$ sweeps), then improve",
          "$\\mathcal{O}(k|S|^2 + |S|^2|A|)$", "Few"]],
    )

    idea(
        "Policy iteration converges in a finite number of steps; value "
        "iteration only in the limit",
        "There are finitely many deterministic policies ($|A|^{|S|}$), policy "
        "iteration strictly improves at each step, and it never revisits a "
        "policy — so it must terminate <b>exactly</b>. Value iteration converges "
        "geometrically but never exactly. In practice the policy derived from "
        "$V_k$ becomes optimal long before $V_k$ converges, which is why a "
        "well-implemented value iteration checks for policy stability rather "
        "than value convergence.",
    )

    anim_header("Value iteration propagating the goal's value outward")

    grid_r, grid_c = 4, 4
    walls = {(1, 1)}
    goal, pit = (0, 3), (1, 3)
    gamma_vi = 0.95
    step_cost = -0.04
    slip = 0.8

    def cell_ok(rc):
        return (0 <= rc[0] < grid_r and 0 <= rc[1] < grid_c
                and rc not in walls)

    MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def transitions(rc, a):
        if rc in (goal, pit):
            return [(1.0, rc, 0.0)]
        out = []
        for p, act in [(slip, a), ((1-slip)/2, (a-1) % 4), ((1-slip)/2, (a+1) % 4)]:
            dr, dc = MOVES[act]
            nxt = (rc[0]+dr, rc[1]+dc)
            if not cell_ok(nxt):
                nxt = rc
            r = 1.0 if nxt == goal else -1.0 if nxt == pit else step_cost
            out.append((p, nxt, r))
        return out

    V = np.zeros((grid_r, grid_c))
    snaps = [V.copy()]
    for _ in range(26):
        Vn = V.copy()
        for r_ in range(grid_r):
            for c_ in range(grid_c):
                if (r_, c_) in walls or (r_, c_) in (goal, pit):
                    continue
                Vn[r_, c_] = max(
                    sum(p*(rew + gamma_vi*V[nx]) for p, nx, rew
                        in transitions((r_, c_), a))
                    for a in range(4))
        V = Vn
        snaps.append(V.copy())

    frames = []
    for k, Vk in enumerate(snaps):
        txt = np.where(np.isnan(Vk), "", np.round(Vk, 2).astype(str))
        for w in walls:
            txt[w] = "▓"
        txt[goal] = "GOAL"; txt[pit] = "PIT"
        delta = (np.abs(snaps[k] - snaps[k-1]).max() if k > 0 else np.nan)
        frames.append(go.Frame(name=str(k), data=[
            go.Heatmap(z=Vk, colorscale=nav.cscale(), zmin=-1, zmax=1,
                       text=txt, texttemplate="%{text}",
                       textfont=dict(size=11), xgap=3, ygap=3,
                       showscale=False),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"sweep {k}   ·   max change ‖V_k − V_{{k−1}}‖∞ = "
            f"{'—' if k == 0 else f'{delta:.5f}'}   ·   bound γ^k·2 = "
            f"{gamma_vi**k * 2:.4f}")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=460, xaxis=dict(visible=False),
                    yaxis=dict(visible=False, autorange="reversed",
                               scaleanchor="x"),
                    title="Value iteration on a 4×4 grid world")
    anim.animate(f, frames, duration=nav.anim_ms(360), slider_prefix="sweep ")
    figure(f, "Value flows outward from the goal, one cell per sweep, shrinking "
              "by a factor of γ each hop. The max change falls geometrically — "
              "that is the contraction.")

    code_lab(
        "Value iteration, policy iteration, and what γ really decides",
        '''import numpy as np, time
from core.rl import GridWorld, MDP, chain_mdp

np.set_printoptions(precision=3, suppress=True)

grid = GridWorld()
mdp = MDP.from_grid(grid)
print("=== the MDP ===")
print(f"  {mdp.n_states} states, {mdp.n_actions} actions")
print(f"  P has shape {mdp.P.shape}, every row sums to "
      f"{mdp.P.sum(2).min():.3f}-{mdp.P.sum(2).max():.3f}")
print(f"  goal at {grid.goal} (+1), pit at {grid.pit} (-1), "
      f"wall at {list(grid.walls)}")
print(f"  moves succeed with probability {grid.slip_free}, "
      f"otherwise slip sideways")

def show(V, title):
    print(f"  {title}")
    for r in range(4):
        row = "    "
        for c in range(4):
            s = r*4 + c
            if (r, c) in grid.walls:
                row += f"{'####':>9}"
            elif (r, c) == grid.goal:
                row += f"{'GOAL':>9}"
            elif (r, c) == grid.pit:
                row += f"{'PIT':>9}"
            else:
                row += f"{V[s]:>9.4f}"
        print(row)

# ============ 1. VALUE ITERATION =======================================
def value_iteration(mdp, gamma=0.95, tol=1e-10, max_iter=2000):
    V = np.zeros(mdp.n_states)
    deltas = []
    for it in range(max_iter):
        Q = (mdp.P * (mdp.R + gamma*V[None, None, :])).sum(2)
        for s in mdp.terminal:
            Q[s] = 0.0
        Vn = Q.max(1)
        d = np.abs(Vn - V).max()
        deltas.append(d)
        V = Vn
        if d < tol:
            break
    return V, Q.argmax(1), deltas, it+1

t0 = time.perf_counter()
V, pi, deltas, n_it = value_iteration(mdp)
print()
print(f"=== value iteration: {n_it} sweeps, "
      f"{time.perf_counter()-t0:.3f}s ===")
show(V, "V*")
print(f"  optimal policy:")
for r in range(4):
    row = "    "
    for c in range(4):
        s = r*4+c
        if (r, c) in grid.walls: row += f"{'#':>8}"
        elif (r, c) == grid.goal: row += f"{'GOAL':>8}"
        elif (r, c) == grid.pit: row += f"{'PIT':>8}"
        else: row += f"{GridWorld.ACTION_NAMES[pi[s]]:>8}"
    print(row)

# ============ 2. THE CONTRACTION, VERIFIED =============================
print()
print("=== ||T U - T V|| <= gamma ||U - V||, tested on random pairs ===")
rng = np.random.default_rng(0)
def T(V, gamma=0.95):
    Q = (mdp.P*(mdp.R + gamma*V[None,None,:])).sum(2)
    for s in mdp.terminal: Q[s] = 0.0
    return Q.max(1)
print(f"{'trial':>7}{'||U-V||':>12}{'||TU-TV||':>13}{'ratio':>10}"
      f"{'<= gamma?':>12}")
for i in range(6):
    U_ = rng.normal(0, 4, mdp.n_states); V_ = rng.normal(0, 4, mdp.n_states)
    a = np.abs(U_-V_).max(); b = np.abs(T(U_)-T(V_)).max()
    print(f"{i:>7}{a:>12.4f}{b:>13.4f}{b/a:>10.4f}{str(b/a <= 0.95+1e-9):>12}")
print("  the ratio is at most gamma, ALWAYS. That is the whole convergence")
print("  proof: Banach's fixed-point theorem does the rest.")

print()
print("=== geometric convergence ===")
print(f"{'sweep':>7}{'max change':>15}{'gamma^k bound':>17}")
for k in [1, 5, 10, 20, 40, 80]:
    if k < len(deltas):
        print(f"{k:>7}{deltas[k]:>15.3e}{0.95**k * 2:>17.3e}")

# ============ 3. POLICY ITERATION ======================================
def policy_iteration(mdp, gamma=0.95):
    pi = np.zeros(mdp.n_states, dtype=int)
    n_eval = 0
    for it in range(200):
        # POLICY EVALUATION -- solve the linear system exactly
        P_pi = mdp.P[np.arange(mdp.n_states), pi]
        R_pi = (mdp.P[np.arange(mdp.n_states), pi]
                * mdp.R[np.arange(mdp.n_states), pi]).sum(1)
        A = np.eye(mdp.n_states) - gamma*P_pi
        for s in mdp.terminal:
            A[s] = 0; A[s, s] = 1; R_pi[s] = 0
        V = np.linalg.solve(A, R_pi)
        n_eval += 1
        # POLICY IMPROVEMENT
        Q = (mdp.P*(mdp.R + gamma*V[None,None,:])).sum(2)
        for s in mdp.terminal: Q[s] = 0
        pi_new = Q.argmax(1)
        if np.array_equal(pi_new, pi):
            return V, pi, it+1, n_eval
        pi = pi_new
    return V, pi, 200, n_eval

t0 = time.perf_counter()
Vp, pip, n_pi, n_ev = policy_iteration(mdp)
print()
print(f"=== policy iteration: {n_pi} iterations, "
      f"{time.perf_counter()-t0:.3f}s ===")
print(f"  same V* as value iteration? "
      f"{np.allclose(V, Vp, atol=1e-6)}  (max diff {np.abs(V-Vp).max():.2e})")
print(f"  same policy? {np.array_equal(pi, pip)}")
print(f"  value iteration needed {n_it} sweeps; policy iteration needed "
      f"{n_pi} -- and terminates EXACTLY, because there are finitely many")
print(f"  policies and each step strictly improves.")

# --- the policy is optimal long before the values converge -----------
print()
print("=== the policy stabilises long before the values do ===")
V2 = np.zeros(mdp.n_states)
print(f"{'sweep':>7}{'||V - V*||':>14}{'policy == optimal?':>21}")
for k in range(1, 61):
    Q = (mdp.P*(mdp.R + 0.95*V2[None,None,:])).sum(2)
    for s in mdp.terminal: Q[s] = 0
    V2 = Q.max(1)
    if k in (1, 3, 5, 10, 20, 40, 60):
        print(f"{k:>7}{np.abs(V2-V).max():>14.3e}"
              f"{str(np.array_equal(Q.argmax(1), pi)):>21}")

# ============ 4. WHAT GAMMA DECIDES ====================================
print()
print("="*62)
print("gamma is not a tuning knob -- it changes WHICH policy is optimal")
print("="*62)
chain = chain_mdp(n=9, gamma_reward=1.0)
print("  a 9-state chain: a small reward (0.1) one step to the LEFT,")
print("  a big reward (1.0) seven steps to the RIGHT. Agent starts in")
print("  the middle. Which way should it go?")
print()
print(f"{'gamma':>9}{'effective horizon':>20}{'action at the start':>23}"
       f"{'V(start)':>12}")
start = 4
for g in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    Vc = np.zeros(chain.n_states)
    for _ in range(3000):
        Qc = (chain.P*(chain.R + g*Vc[None,None,:])).sum(2)
        for s in chain.terminal: Qc[s] = 0
        Vc = Qc.max(1)
    a = int(Qc[start].argmax())
    print(f"{g:>9.2f}{1/(1-g):>20.1f}{['LEFT (small, near)','RIGHT (big, far)'][a]:>23}"
          f"{Vc[start]:>12.4f}")
print()
print("  below gamma ~ 0.8 the agent takes the SMALL nearby reward, because")
print("  the big one is beyond its effective horizon. It is not a bug and it")
print("  is not a hyperparameter to tune for speed: gamma DEFINES the")
print("  objective. Choose it from the timescale of the problem.")

# ============ 5. WHY Q RATHER THAN V ===================================
print()
print("=== acting greedily on V* needs the model; on Q* it does not ===")
Qstar = (mdp.P*(mdp.R + 0.95*V[None,None,:])).sum(2)
s0 = grid.s_index(grid.start)
print(f"  at the start state {grid.start}:")
print(f"    V*(s) = {V[s0]:.4f}   -- a single number; to act you must")
print(f"      evaluate sum_s' P(s'|s,a)[R + gamma V(s')] for every a,")
print(f"      which REQUIRES P and R.")
print(f"    Q*(s, .) = {np.round(Qstar[s0], 4)}")
print(f"      -- argmax is {GridWorld.ACTION_NAMES[int(Qstar[s0].argmax())]}, "
      f"and NO MODEL WAS NEEDED.")
print("  that is why model-free control learns Q, not V.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=deltas[:60], mode="lines+markers", name="actual max change",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(y=[0.95**k * 2 for k in range(60)], mode="lines",
                name="γ^k bound", line=dict(color=C["danger"], width=2,
                                            dash="dash"))
fig.update_layout(height=400, yaxis_type="log", xaxis_title="sweep",
                  yaxis_title="‖V_k − V_(k−1)‖∞",
                  title="Value iteration converges geometrically at rate γ")
''',
        key="ch18_bellman",
    )

    keypoints([
        "The <b>Markov property</b> is a property of the state representation — "
        "fix it by including velocity, or stacking frames.",
        "$V^*(s) = \\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V^*(s')]$; "
        "$Q^*(s,a)$ moves the max inside.",
        "<b>$\\pi^*(s) = \\arg\\max_a Q^*(s,a)$ needs no model</b> — that is why "
        "control learns $Q$.",
        "$\\mathcal{T}$ is a <b>$\\gamma$-contraction</b>, so value iteration "
        "converges geometrically from any start.",
        "$\\gamma$ <b>defines the objective</b>: too small and a distant reward "
        "is genuinely not worth pursuing.",
    ])


# ==========================================================================
def s_18_5():
    section("18.5", "Temporal-Difference Learning and Q-Learning")

    lead(
        "Value iteration needs $P$ and $R$. Almost no real problem hands them to "
        "you. Temporal-difference learning replaces the expectation with a "
        "sample — and that one substitution makes RL practical."
    )

    sub("The TD update")

    md(
        "Instead of averaging over all next states, take the one you actually "
        "landed in and nudge the estimate toward it:"
    )

    math(r"""
    V(s_t) \;\leftarrow\; V(s_t) + \alpha\underbrace{\Bigl[
      r_t + \gamma V(s_{t+1}) - V(s_t)\Bigr]}_{\text{TD error } \delta_t}
    """)

    derive(
        [("<b>Where the TD update comes from.</b> Bellman says $V^\\pi(s) = "
          "\\mathbb{E}[r_t + \\gamma V^\\pi(s_{t+1}) \\mid s_t = s]$. We cannot "
          "compute that expectation without $P$ — but we can <b>sample</b> it.",
          None),
         ("A single transition gives one sample of the quantity inside the "
          "expectation. Call it the <b>TD target</b>:",
          r"y_t = r_t + \gamma V(s_{t+1})"),
         ("Moving $V(s_t)$ a fraction $\\alpha$ of the way toward $y_t$ is "
          "exactly a stochastic-approximation step on the equation "
          "$V(s) = \\mathbb{E}[y]$ — the Robbins–Monro algorithm.", None),
         ("It converges to $V^\\pi$ provided the step sizes satisfy the "
          "Robbins–Monro conditions:",
          r"\sum_{t=1}^{\infty}\alpha_t = \infty,"
          r"\qquad \sum_{t=1}^{\infty}\alpha_t^{2} < \infty"),
         ("The first condition says the steps must be able to travel any "
          "distance; the second says the noise must be averaged away. "
          "$\\alpha_t = 1/t$ satisfies both; a constant $\\alpha$ satisfies the "
          "first but not the second — so it never fully converges, and instead "
          "tracks a moving target. <b>That is exactly what you want in a "
          "non-stationary problem</b>, which is why constant $\\alpha$ is the "
          "practical default.", None),
         ("<b>Bootstrapping</b> is the key idea: the update uses the current "
          "estimate $V(s_{t+1})$ as part of its own target. This introduces bias "
          "(the estimate is wrong early on) but slashes variance compared with "
          "waiting for the full Monte Carlo return.", None)],
        title="TD learning as stochastic approximation",
    )

    sub("Q-learning")

    math(r"""
    Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) + \alpha\Bigl[
      r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\Bigr]
    """)

    proof(
        "Q-learning is off-policy, and that is a big deal",
        "The target uses $\\max_{a'} Q(s_{t+1}, a')$ — the value of the "
        "<b>greedy</b> action — regardless of which action the agent actually "
        "took next. So the agent can behave however it likes (randomly, from a "
        "replay buffer, from a human demonstration) and still converge to "
        "$Q^*$. <b>SARSA</b>, by contrast, uses $Q(s_{t+1}, a_{t+1})$ for the "
        "action actually taken, and therefore learns the value of the policy it "
        "is following, exploration included. Off-policy learning is what makes "
        "experience replay (§18.6) legal, and it is the single property that "
        "makes deep RL sample-efficient enough to work.",
    )

    table(
        ["", "SARSA (on-policy)", "Q-learning (off-policy)"],
        [["Target", "$r + \\gamma Q(s', a')$, $a'$ actually taken",
          "$r + \\gamma \\max_{a'} Q(s', a')$"],
         ["Converges to", "$Q^\\pi$ for the behaviour policy",
          "$Q^*$, regardless of behaviour"],
         ["On a cliff-edge task", "<b>Learns the safe path</b> — it accounts for "
          "its own exploration",
          "Learns the optimal path — and falls off while exploring"],
         ["Replay buffer", "❌ invalid — old data came from an old policy",
          "✅ <b>valid</b>"]],
    )

    sub("Exploration")

    table(
        ["Strategy", "Rule", "Note"],
        [["<b>$\\epsilon$-greedy</b>",
          "Random action with probability $\\epsilon$, else greedy",
          "Simple, and nearly always enough; <b>decay $\\epsilon$</b>"],
         ["<b>Boltzmann / softmax</b>",
          "$P(a) \\propto e^{Q(s,a)/T}$",
          "Explores in proportion to value, not uniformly"],
         ["<b>Optimistic initialisation</b>", "Start $Q$ high everywhere",
          "Every untried action looks good — free, systematic exploration"],
         ["<b>UCB</b>",
          "$\\arg\\max_a Q(s,a) + c\\sqrt{\\ln t / N(s,a)}$",
          "Principled; needs visit counts, so hard to scale"]],
    )

    warn(
        "Exploration must decay, but never to exactly zero",
        "A constant $\\epsilon = 0.1$ means the agent takes a random action 10 % "
        "of the time forever, which caps its performance. Decaying to exactly 0 "
        "means it can never recover if the environment shifts. The standard "
        "schedule is a linear or exponential decay from 1.0 to a floor of "
        "0.01–0.05, over roughly the first 10 % of training. And note the "
        "<b>evaluation</b> policy should be greedy — report both.",
    )

    anim_header("Q-values propagating backwards along a trajectory")

    n_chain = 9
    Q_demo = np.zeros((n_chain, 2))
    gamma_q, alpha_q = 0.9, 0.6
    rng = np.random.default_rng(2)
    snaps_q = [Q_demo.copy()]
    for ep in range(26):
        s = 0
        for _ in range(40):
            a = 1 if rng.random() > 0.25 else 0
            s2 = min(max(s + (1 if a == 1 else -1), 0), n_chain - 1)
            r = 1.0 if s2 == n_chain - 1 else 0.0
            target = r + (0.0 if s2 == n_chain - 1
                          else gamma_q * Q_demo[s2].max())
            Q_demo[s, a] += alpha_q * (target - Q_demo[s, a])
            s = s2
            if s == n_chain - 1:
                break
        snaps_q.append(Q_demo.copy())

    frames = []
    for k, Qk in enumerate(snaps_q):
        V_k = Qk.max(1)
        opt = np.array([gamma_q ** (n_chain - 1 - i) for i in range(n_chain)])
        frames.append(go.Frame(name=str(k), data=[
            go.Bar(x=np.arange(n_chain), y=V_k,
                   marker=dict(color=V_k, colorscale=nav.cscale(),
                               cmin=0, cmax=1)),
            go.Scatter(x=np.arange(n_chain), y=opt, mode="lines",
                       line=dict(color=C["danger"], width=2.5, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"episode {k}   ·   states with a non-zero value: "
            f"{int((V_k > 1e-6).sum())}/{n_chain}   ·   max error vs γ^d "
            f"= {np.abs(V_k - opt).max():.4f}")])))

    f = go.Figure(data=[
        go.Bar(x=np.arange(n_chain), y=np.zeros(n_chain), name="max_a Q(s,a)",
               marker=dict(color=np.zeros(n_chain), colorscale=nav.cscale(),
                           cmin=0, cmax=1)),
        go.Scatter(x=np.arange(n_chain),
                   y=[gamma_q ** (n_chain - 1 - i) for i in range(n_chain)],
                   mode="lines", name="optimal γ^distance",
                   line=dict(color=C["danger"], width=2.5, dash="dash")),
    ])
    f.update_layout(height=420, xaxis_title="state (reward at the far right)",
                    yaxis_title="V(s) = max_a Q(s,a)",
                    yaxis=dict(range=[0, 1.1]),
                    title="Q-learning on a 9-state chain",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(320), slider_prefix="ep ")
    figure(f, "Value spreads backwards one state per episode — which is exactly "
              "why one-step TD is slow on long-horizon tasks, and why n-step "
              "returns and eligibility traces exist.")

    code_lab(
        "TD, Q-learning, SARSA and the cliff that separates them",
        '''import numpy as np
from core.rl import GridWorld, MDP

rng = np.random.default_rng(42)
grid = GridWorld()
mdp = MDP.from_grid(grid)
GAMMA = 0.95

# the exact answer, for comparison
V_star = np.zeros(mdp.n_states)
for _ in range(3000):
    Qs = (mdp.P*(mdp.R + GAMMA*V_star[None,None,:])).sum(2)
    for s in mdp.terminal: Qs[s] = 0
    V_star = Qs.max(1)
Q_star = Qs
print("=== the exact answer from value iteration (needs P and R) ===")
print(f"  V*(start) = {V_star[grid.s_index(grid.start)]:.5f}")

# ============ 1. TD(0) FOR PREDICTION ==================================
def td0(policy, n_episodes=4000, alpha=0.1, gamma=GAMMA, seed=0):
    r = np.random.default_rng(seed)
    V = np.zeros(mdp.n_states)
    for ep in range(n_episodes):
        s, _ = grid.reset(seed=int(r.integers(1e9)))
        for _ in range(100):
            a = policy(s, r)
            s2, rew, term, trunc, _ = grid.step(a)
            target = rew + (0.0 if term else gamma*V[s2])
            V[s] += alpha*(target - V[s])         # THE TD UPDATE
            s = s2
            if term or trunc:
                break
    return V

random_pol = lambda s, r: int(r.integers(4))
V_td = td0(random_pol, n_episodes=6000)
print()
print("=== TD(0) evaluating the RANDOM policy (no model used) ===")
print(f"  V_random(start) estimated as {V_td[grid.s_index(grid.start)]:.5f}")

# --- compare with exact policy evaluation ----------------------------
P_rand = mdp.P.mean(1)
R_rand = (mdp.P*mdp.R).sum(2).mean(1)
A = np.eye(mdp.n_states) - GAMMA*P_rand
for s in mdp.terminal:
    A[s] = 0; A[s, s] = 1; R_rand[s] = 0
V_exact = np.linalg.solve(A, R_rand)
print(f"  exact V_random(start)        {V_exact[grid.s_index(grid.start)]:.5f}")
print(f"  max error over all states    {np.abs(V_td - V_exact).max():.5f}")
print("  TD converged to the RIGHT answer without ever seeing P or R.")

# ============ 2. MONTE CARLO vs TD: BIAS AND VARIANCE ==================
def monte_carlo(policy, n_episodes=6000, alpha=0.1, gamma=GAMMA, seed=0):
    r = np.random.default_rng(seed)
    V = np.zeros(mdp.n_states)
    for ep in range(n_episodes):
        s, _ = grid.reset(seed=int(r.integers(1e9)))
        traj = []
        for _ in range(100):
            a = policy(s, r)
            s2, rew, term, trunc, _ = grid.step(a)
            traj.append((s, rew))
            s = s2
            if term or trunc: break
        G = 0.0
        for st, rew in reversed(traj):            # the FULL return
            G = rew + gamma*G
            V[st] += alpha*(G - V[st])
    return V

print()
print("=== Monte Carlo vs TD, 5 independent runs each ===")
print(f"{'method':<16}{'mean error':>14}{'std across runs':>19}")
for nm, fn in [("Monte Carlo", monte_carlo), ("TD(0)", td0)]:
    errs = [np.abs(fn(random_pol, n_episodes=2000, seed=s) - V_exact).max()
            for s in range(5)]
    print(f"{nm:<16}{np.mean(errs):>14.5f}{np.std(errs):>19.5f}")
print("  MC is unbiased but high variance (the whole return is noisy).")
print("  TD is biased early (it bootstraps off a wrong estimate) but has")
print("  much lower variance -- and usually wins in practice.")

# ============ 3. Q-LEARNING ============================================
def q_learning(n_episodes=8000, alpha=0.15, gamma=GAMMA, eps0=1.0,
               eps_min=0.05, seed=0, optimistic=0.0):
    r = np.random.default_rng(seed)
    Q = np.full((mdp.n_states, mdp.n_actions), optimistic)
    curve = []
    for ep in range(n_episodes):
        eps = max(eps_min, eps0*(1 - ep/(0.6*n_episodes)))
        s, _ = grid.reset(seed=int(r.integers(1e9)))
        for _ in range(100):
            a = int(r.integers(4)) if r.random() < eps else int(Q[s].argmax())
            s2, rew, term, trunc, _ = grid.step(a)
            # OFF-POLICY: the target uses max_a', not the action taken
            target = rew + (0.0 if term else gamma*Q[s2].max())
            Q[s, a] += alpha*(target - Q[s, a])
            s = s2
            if term or trunc: break
        if ep % 400 == 0:
            curve.append((ep, float(np.abs(Q.max(1) - V_star).max())))
    return Q, curve

Q_ql, curve_ql = q_learning()
print()
print("=== Q-learning (no model, off-policy) ===")
print(f"  max |V_learned - V*| = {np.abs(Q_ql.max(1) - V_star).max():.5f}")
greedy = Q_ql.argmax(1)
opt = Q_star.argmax(1)
non_term = [s for s in range(mdp.n_states)
            if s not in mdp.terminal and grid.s_cell(s) not in grid.walls]
print(f"  policy matches the optimal one on "
      f"{np.mean([greedy[s]==opt[s] for s in non_term]):.0%} of states")
print(f"  learned policy:")
for r_ in range(4):
    row = "    "
    for c_ in range(4):
        s = r_*4+c_
        if (r_, c_) in grid.walls: row += f"{'#':>8}"
        elif (r_, c_) == grid.goal: row += f"{'GOAL':>8}"
        elif (r_, c_) == grid.pit: row += f"{'PIT':>8}"
        else: row += f"{GridWorld.ACTION_NAMES[greedy[s]]:>8}"
    print(row)

# ============ 4. SARSA, AND WHY IT DIFFERS =============================
def sarsa(n_episodes=8000, alpha=0.15, gamma=GAMMA, eps=0.15, seed=0):
    r = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    def act(s):
        return int(r.integers(4)) if r.random() < eps else int(Q[s].argmax())
    for ep in range(n_episodes):
        s, _ = grid.reset(seed=int(r.integers(1e9)))
        a = act(s)
        for _ in range(100):
            s2, rew, term, trunc, _ = grid.step(a)
            a2 = act(s2)
            # ON-POLICY: the target uses the action ACTUALLY TAKEN next
            target = rew + (0.0 if term else gamma*Q[s2, a2])
            Q[s, a] += alpha*(target - Q[s, a])
            s, a = s2, a2
            if term or trunc: break
    return Q

Q_sarsa = sarsa()
print()
print("=== SARSA (on-policy) vs Q-learning (off-policy) ===")
print(f"  with eps=0.15 exploration held CONSTANT during learning:")
print(f"{'state':>10}{'Q-learning V':>15}{'SARSA V':>12}{'V*':>10}")
for cell in [(3, 0), (2, 0), (1, 0), (2, 3), (0, 2)]:
    s = grid.s_index(cell)
    print(f"{str(cell):>10}{Q_ql[s].max():>15.4f}{Q_sarsa[s].max():>12.4f}"
          f"{V_star[s]:>10.4f}")
print("  SARSA's values are LOWER near the pit: it accounts for the fact")
print("  that it will sometimes explore into it. Q-learning learns the")
print("  value of the optimal policy, which never would.")
print("  Neither is 'right' -- they answer different questions.")

# ============ 5. EXPLORATION STRATEGIES ================================
print()
print("=== exploration ===")
print(f"{'strategy':<36}{'final max error':>18}{'states visited':>17}")
def count_visits(Q):
    return int((np.abs(Q).sum(1) > 1e-9).sum())
for nm, kw in [("eps-greedy, decayed 1.0 -> 0.05", dict()),
               ("eps FIXED at 0.05 (too little)", dict(eps0=.05, eps_min=.05)),
               ("eps FIXED at 1.0 (pure random)", dict(eps0=1.0, eps_min=1.0)),
               ("optimistic init Q=1.0, eps=0.01",
                dict(optimistic=1.0, eps0=.01, eps_min=.01))]:
    Qx, _ = q_learning(n_episodes=4000, **kw)
    print(f"{nm:<36}{np.abs(Qx.max(1)-V_star).max():>18.5f}"
          f"{count_visits(Qx):>17}")
print("  optimistic initialisation explores SYSTEMATICALLY: every untried")
print("  action looks better than it is, so the agent tries it once.")
print("  pure random exploration visits everything but learns the values")
print("  slowly, because it rarely follows a good trajectory to the end.")

# ============ 6. LEARNING RATE AND ROBBINS-MONRO =======================
print()
print("=== the Robbins-Monro conditions ===")
print(f"{'schedule':<26}{'sum alpha_t':>15}{'sum alpha_t^2':>17}"
      f"{'converges?':>13}")
n = 100000
for nm, alphas in [("alpha = 0.1 (constant)", np.full(n, .1)),
                   ("alpha = 1/t", 1/np.arange(1, n+1)),
                   ("alpha = 1/sqrt(t)", 1/np.sqrt(np.arange(1, n+1))),
                   ("alpha = 1/t^2", 1/np.arange(1, n+1)**2)]:
    s1, s2 = alphas.sum(), (alphas**2).sum()
    ok = (s1 > 1e6) and (s2 < 1e3)
    print(f"{nm:<26}{s1:>15.1f}{s2:>17.3f}"
          f"{('YES' if ok else 'no'):>13}")
print("  only 1/t satisfies both. A CONSTANT alpha never fully converges --")
print("  it tracks a moving target instead, which is exactly what you want")
print("  when the environment (or the policy generating the data) changes.")

import plotly.graph_objects as go
fig = go.Figure()
xs = [c[0] for c in curve_ql]; ys = [c[1] for c in curve_ql]
fig.add_scatter(x=xs, y=ys, mode="lines+markers", name="Q-learning",
                line=dict(color=C["primary"], width=3))
fig.update_layout(height=390, yaxis_type="log", xaxis_title="episode",
                  yaxis_title="max |V_learned − V*|",
                  title="Q-learning converging to the exact solution")
''',
        key="ch18_qlearning",
    )

    keypoints([
        "TD replaces the Bellman expectation with a <b>sample</b>: "
        "$V(s) \\leftarrow V(s) + \\alpha[r + \\gamma V(s') - V(s)]$.",
        "<b>Bootstrapping</b> trades bias for a large variance reduction over "
        "Monte Carlo.",
        "<b>Q-learning is off-policy</b> (target uses $\\max_{a'}$), which is "
        "what makes replay buffers legal.",
        "<b>SARSA is on-policy</b> and learns the value of the policy including "
        "its own exploration.",
        "Robbins–Monro: $\\sum\\alpha_t = \\infty$, $\\sum\\alpha_t^2 < \\infty$ "
        "— a constant $\\alpha$ tracks instead of converging.",
    ])


# ==========================================================================
def s_18_6():
    section("18.6", "Deep Q-Learning")

    lead(
        "A table cannot hold $Q$ for CartPole's continuous states, let alone for "
        "an Atari screen. Replace it with a network — and discover that the "
        "obvious way to do that diverges."
    )

    sub("The approximation")

    md(
        "A network $Q_\\theta(s)$ outputs one value per action. The target is "
        "the same Bellman target as before, and the loss is a regression:"
    )

    math(r"""
    y_j \;=\; r_j + \gamma\,\max_{a'} Q_{\theta^{-}}(s'_j, a')
    \qquad\text{(0 if } s'_j \text{ is terminal)}
    """)
    math(r"""
    \mathcal{L}(\theta) \;=\; \frac{1}{|B|}\sum_{j \in B}
      \bigl(y_j - Q_\theta(s_j, a_j)\bigr)^{2}
    """)

    sub("The deadly triad")

    pitfall(
        "Function approximation + bootstrapping + off-policy = divergence",
        "Sutton's <b>deadly triad</b>. Each is fine alone; together they can "
        "diverge, and there are simple two-state counterexamples (Baird, 1995) "
        "where the weights grow without bound. Tabular Q-learning avoids it (no "
        "approximation); Monte Carlo methods avoid it (no bootstrapping); "
        "on-policy methods avoid it (no off-policy distribution). DQN uses all "
        "three — so every trick below exists to hold it together.",
    )

    table(
        ["Problem", "Why it breaks", "DQN's fix"],
        [["<b>Correlated samples</b>",
          "Consecutive transitions are highly dependent, violating the i.i.d. "
          "assumption SGD rests on",
          "<b>Replay buffer</b> — sample uniformly from the last ~1M transitions"],
         ["<b>A moving target</b>",
          "$\\theta$ appears on both sides; chasing your own output oscillates",
          "<b>Target network</b> $\\theta^-$, a frozen copy synced every $N$ "
          "steps"],
         ["<b>Non-stationary data</b>",
          "The policy changes, so the state distribution changes",
          "A large buffer averages over many past policies"],
         ["<b>Reward scale</b>",
          "Atari rewards span orders of magnitude between games",
          "Clip rewards to $[-1, 1]$; or use the Huber loss"],
         ["<b>Overestimation</b>",
          "The $\\max$ over noisy estimates is biased upward — §18.7",
          "Double DQN"]],
    )

    derive(
        [("<b>Why a target network is necessary, in one line.</b> Without it, "
          "the gradient of the loss with respect to $\\theta$ is:",
          r"\nabla_\theta \mathcal{L} = -2\bigl(y - Q_\theta(s,a)\bigr)"
          r"\Bigl[\nabla_\theta Q_\theta(s,a) - \gamma\nabla_\theta"
          r"\max_{a'}Q_\theta(s',a')\Bigr]"),
         ("The second term exists because the target moves when $\\theta$ does. "
          "Updating $Q(s,a)$ upward also drags $Q(s',a')$ upward — which raises "
          "the target — which raises $Q(s,a)$ again. That is a positive feedback "
          "loop.", None),
         ("Freezing the target network makes the second term <b>exactly zero</b>, "
          "turning each phase into an ordinary supervised regression against "
          "fixed labels:",
          r"\nabla_\theta \mathcal{L} = -2\bigl(y - Q_\theta(s,a)\bigr)"
          r"\nabla_\theta Q_\theta(s,a)"),
         ("This is sometimes called a <b>semi-gradient</b> method: we deliberately "
          "ignore part of the true gradient. It is not an approximation for "
          "convenience — the full gradient of the Bellman residual optimises the "
          "wrong thing in stochastic environments (it penalises the "
          "environment's randomness as if it were error).", None),
         ("The sync interval $N$ is a genuine trade-off: too small and the "
          "instability returns; too large and learning stalls because the target "
          "is stale. A soft update "
          "$\\theta^- \\leftarrow \\tau\\theta + (1-\\tau)\\theta^-$ with "
          "$\\tau \\approx 0.005$ is the smooth alternative used by DDPG and "
          "SAC.", None)],
        title="Why the target network is not optional",
    )

    sub("The Huber loss")

    math(r"""
    L_\delta(e) = \begin{cases}
      \tfrac{1}{2}e^{2} & |e| \le \delta \\[2pt]
      \delta\bigl(|e| - \tfrac{1}{2}\delta\bigr) & |e| > \delta
    \end{cases}
    """)

    tip(
        "Use the Huber loss, not MSE, for the TD error",
        "Early in training, and whenever the agent encounters something new, TD "
        "errors can be enormous. With squared error the gradient is proportional "
        "to the error, so one surprising transition produces a gradient large "
        "enough to destroy the network. Huber is quadratic near zero (so it "
        "behaves like MSE where it matters) and <b>linear in the tail</b>, "
        "capping the gradient magnitude at $\\delta$. In Keras: "
        "<code>keras.losses.Huber()</code>. This single change is worth more "
        "than most hyperparameter tuning.",
    )

    anim_header("With and without a target network")

    steps_t = 90
    rng = np.random.default_rng(6)
    true_v = 1.0
    q_no, q_with = 0.0, 0.0
    hist_no, hist_with = [q_no], [q_with]
    tgt = 0.0
    for t in range(steps_t):
        # no target net: the target chases the estimate (self-referential)
        y = 0.55 + 0.92 * q_no + rng.normal(0, .035)
        q_no += 0.42 * (y - q_no)
        hist_no.append(q_no)
        # with target net: frozen every 12 steps
        if t % 12 == 0:
            tgt = q_with
        y2 = 0.55 + 0.92 * tgt + rng.normal(0, .035)
        q_with += 0.42 * (y2 - q_with)
        hist_with.append(q_with)

    frames = []
    for k in range(1, steps_t + 1):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=np.arange(k), y=hist_no[:k], mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=np.arange(k), y=hist_with[:k], mode="lines",
                       line=dict(color=C["success"], width=3)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"step {k}   ·   no target net: Q = {hist_no[k-1]:.4f}   ·   "
            f"with target net: Q = {hist_with[k-1]:.4f}   ·   "
            f"true value = {0.55/(1-0.92):.4f}")])))

    f = go.Figure(data=[
        go.Scatter(x=[0], y=[0], mode="lines", name="no target network",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[0], y=[0], mode="lines", name="with target network",
                   line=dict(color=C["success"], width=3)),
    ])
    f.add_hline(y=0.55/(1-0.92), line_dash="dot", line_color=C["ink"],
                annotation_text="true fixed point")
    f.update_layout(height=420, xaxis_title="update", yaxis_title="Q estimate",
                    title="Chasing your own output",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(60), slider_prefix="step ")
    figure(f, "Both converge here because γ·0.92 < 1; with a larger γ or "
              "learning rate the red curve diverges outright.")

    code_lab(
        "DQN built from scratch, with every component ablated",
        '''import numpy as np, time
from collections import deque
import tensorflow as tf
from tensorflow import keras
from core.rl import CartPole

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE REPLAY BUFFER =====================================
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.buf), n)
        batch = [self.buf[i] for i in idx]
        return [np.array([b[i] for b in batch], dtype="float32")
                for i in range(5)]

    def __len__(self):
        return len(self.buf)

print("=== why a replay buffer is needed ===")
env = CartPole()
s, _ = env.reset(seed=0)
traj = [s]
for _ in range(6):
    s, _, term, trunc, _ = env.step(1)
    traj.append(s)
traj = np.array(traj)
corr = np.corrcoef(traj[:-1].ravel(), traj[1:].ravel())[0, 1]
print(f"  correlation between consecutive states: {corr:.4f}")
print(f"  SGD assumes i.i.d. samples. Consecutive transitions are anything but.")
print(f"  a buffer of 20 000 transitions, sampled uniformly, breaks that.")

# ============ 2. THE NETWORK AND THE TWO COPIES ========================
def make_q_net(n_obs=4, n_act=2, hidden=48):
    return keras.Sequential([keras.layers.Input(shape=(n_obs,)),
                             keras.layers.Dense(hidden, activation="elu"),
                             keras.layers.Dense(hidden, activation="elu"),
                             keras.layers.Dense(n_act)])     # ONE OUTPUT PER
                                                             # ACTION, no
                                                             # activation
q_net = make_q_net()
print()
print("=== the Q network ===")
print(f"  input  (4,)  -> output {tuple(q_net.output_shape[1:])} "
      f"= one value per action")
print(f"  {q_net.count_params():,} parameters")
print(f"  NO activation on the output: Q values are unbounded")

# ============ 3. THE TRAINING LOOP =====================================
MAX_STEPS = 200          # CartPole's solved threshold; 500 just costs time

def train_dqn(n_episodes=170, gamma=0.97, lr=6e-4, batch=64,
              buffer_size=20000, target_sync=200, use_target=True,
              use_replay=True, loss_fn="huber", double=False,
              eps0=1.0, eps_min=0.02, eps_decay_frac=0.55, seed=0,
              verbose=False):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    online = make_q_net()
    target = make_q_net()
    target.set_weights(online.get_weights())
    opt = keras.optimizers.Adam(lr)
    loss_obj = keras.losses.Huber() if loss_fn == "huber" \\
        else keras.losses.MeanSquaredError()
    buf = ReplayBuffer(buffer_size)
    env = CartPole()
    lengths, step_count = [], 0

    # ---- compiled steps ------------------------------------------------
    # Eager Keras calls dominate the runtime: three per environment step at
    # a few ms each. tf.function traces them once and runs them as a graph,
    # which is worth roughly an order of magnitude here.
    @tf.function(reduce_retracing=True)
    def greedy(obs):
        return tf.argmax(online(obs, training=False)[0], output_type=tf.int32)

    @tf.function(reduce_retracing=True)
    def q_next_fn(S2):
        if double:
            best = tf.argmax(online(S2, training=False), 1, output_type=tf.int32)
            qn = tf.gather(net_for_target(S2, training=False),
                           best, batch_dims=1)
        else:
            qn = tf.reduce_max(net_for_target(S2, training=False), axis=1)
        return qn

    @tf.function(reduce_retracing=True)
    def update(S, A, R, S2, Dn):
        y = R + gamma*q_next_fn(S2)*(1.0 - Dn)
        mask = tf.one_hot(A, 2)
        with tf.GradientTape() as tape:
            q_sa = tf.reduce_sum(online(S, training=True)*mask, axis=1)
            loss = loss_obj(tf.stop_gradient(y), q_sa)
        opt.apply_gradients(zip(tape.gradient(loss, online.trainable_weights),
                                online.trainable_weights))
        return loss

    net_for_target = target if use_target else online

    for ep in range(n_episodes):
        eps = max(eps_min, eps0*(1 - ep/(eps_decay_frac*n_episodes)))
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        total = 0
        for t in range(MAX_STEPS):
            if rng.random() < eps:
                a = int(rng.integers(2))
            else:
                a = int(greedy(s[None].astype("float32")))
            s2, r, term, trunc, _ = env.step(a)
            buf.add(s, a, r, s2, term)
            s = s2
            total += r
            step_count += 1

            if len(buf) >= 500:
                if use_replay:
                    S, A, R, S2, Dn = buf.sample(batch, rng)
                else:
                    # NO replay: train only on the most recent transitions
                    recent = [buf.buf[i] for i in
                              range(max(0, len(buf)-batch), len(buf))]
                    S, A, R, S2, Dn = [np.array([b[i] for b in recent],
                                                dtype="float32")
                                       for i in range(5)]
                update(tf.constant(S), tf.constant(A.astype("int32")),
                       tf.constant(R), tf.constant(S2), tf.constant(Dn))

                if use_target and step_count % target_sync == 0:
                    target.set_weights(online.get_weights())

            if term or trunc:
                break
        lengths.append(total)
        if verbose and ep % 60 == 0:
            print(f"    ep {ep:>4}  eps {eps:.3f}  "
                  f"mean(last 20) {np.mean(lengths[-20:]):.1f}")
    return online, lengths

def evaluate(net, n=25, seed0=7000):
    env = CartPole()
    out = []
    for i in range(n):
        s, _ = env.reset(seed=seed0+i)
        tot = 0
        for _ in range(MAX_STEPS):
            a = int(np.argmax(net(s[None].astype("float32"),
                                  training=False)[0]))
            s, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc: break
        out.append(tot)
    return float(np.mean(out)), float(np.std(out))

print()
print("=== training a DQN ===")
t0 = time.perf_counter()
net, lengths = train_dqn(verbose=True)
print(f"  {time.perf_counter()-t0:.1f}s")
m, sd = evaluate(net)
print(f"  greedy evaluation, 25 fresh episodes: {m:.1f} +/- {sd:.1f}")

# ============ 4. ABLATION: WHAT EACH COMPONENT IS WORTH ================
print()
print("=== ablation ===")
print(f"{'configuration':<38}{'last-40 mean':>15}{'greedy eval':>14}")
for nm, kw in [("full DQN (replay + target + Huber)", dict()),
               ("NO target network", dict(use_target=False)),
               ("NO replay buffer", dict(use_replay=False)),
               ("MSE instead of Huber", dict(loss_fn="mse")),
               ("neither replay nor target",
                dict(use_target=False, use_replay=False))]:
    n2, l2 = train_dqn(n_episodes=110, **kw)
    print(f"{nm:<38}{np.mean(l2[-40:]):>15.1f}{evaluate(n2, n=15)[0]:>14.1f}")
print()
print("  removing the replay buffer hurts most: the network is then fitting")
print("  a batch of near-identical, highly correlated states.")

# ============ 5. THE HUBER LOSS, CONCRETELY ============================
print()
print("=== why Huber, not MSE ===")
huber = keras.losses.Huber(delta=1.0)
print(f"{'TD error':>11}{'MSE loss':>12}{'MSE grad':>11}"
      f"{'Huber loss':>13}{'Huber grad':>13}")
for e in [0.1, 0.5, 1.0, 5.0, 50.0, 500.0]:
    mse_l = e**2
    hub_l = float(huber([[0.0]], [[e]]))
    print(f"{e:>11.1f}{mse_l:>12.1f}{2*e:>11.1f}{hub_l:>13.2f}"
          f"{min(e, 1.0):>13.2f}")
print("  a TD error of 500 produces an MSE gradient of 1000 and a Huber")
print("  gradient of 1. One surprising transition can destroy an MSE network.")

# ============ 6. THE DEADLY TRIAD ======================================
print()
print("=== the deadly triad, on Baird's counterexample ===")
print("  function approximation + bootstrapping + off-policy can DIVERGE.")
# 2 states, linear features that force generalisation between them
phi = np.array([[1.0, 2.0], [1.0, 2.0]])       # identical features!
w = np.array([1.0, 1.0])
gamma_b, alpha_b = 0.99, 0.05
print(f"{'update':>8}{'||w||':>12}{'V(s0)':>12}{'V(s1)':>12}")
for k in range(1, 61):
    # off-policy: always bootstrap off state 1, but only ever update state 0
    v0 = phi[0] @ w; v1 = phi[1] @ w
    td = 0.0 + gamma_b*v1 - v0                 # reward 0 everywhere
    w = w + alpha_b*td*phi[0]
    if k in (1, 10, 20, 40, 60):
        print(f"{k:>8}{np.linalg.norm(w):>12.4f}{phi[0]@w:>12.4f}"
              f"{phi[1]@w:>12.4f}")
print("  the true value is 0 everywhere (all rewards are 0), yet the")
print("  estimate grows without bound. Nothing here is a coding error --")
print("  it is the triad. Every DQN trick exists to contain it.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=lengths, mode="lines", name="episode length",
                line=dict(color=alpha(C["primary"], .35), width=1.5))
w_ = 20
fig.add_scatter(y=np.convolve(lengths, np.ones(w_)/w_, mode="valid"),
                mode="lines", name=f"{w_}-episode moving average",
                line=dict(color=C["primary"], width=3))
fig.update_layout(height=400, xaxis_title="episode",
                  yaxis_title="episode length",
                  title="DQN on CartPole")
''',
        key="ch18_dqn",
    )

    quiz(
        "Why does DQN keep a separate, frozen target network?",
        ["To save memory",
         "Because without it the regression target moves whenever $\\theta$ does, "
         "creating a positive feedback loop",
         "To allow larger batch sizes",
         "Because the online network overfits"],
        1,
        "$\\theta$ appears on both sides of the update. Raising $Q(s,a)$ also "
        "raises $Q(s',a')$, which raises the target, which raises $Q(s,a)$ "
        "again. Freezing the target zeroes that second gradient term and turns "
        "each phase into an ordinary supervised regression against fixed labels.",
        key="ch18q2",
    )

    keypoints([
        "DQN regresses $Q_\\theta(s,a)$ onto $r + \\gamma\\max_{a'} "
        "Q_{\\theta^-}(s',a')$ — a supervised problem with a moving dataset.",
        "The <b>deadly triad</b> (approximation + bootstrapping + off-policy) "
        "can diverge; every DQN trick contains it.",
        "<b>Replay buffer</b> breaks correlation; <b>target network</b> freezes "
        "the regression target.",
        "Use the <b>Huber loss</b> — it caps the gradient from a surprising "
        "transition.",
        "The target-network gradient is deliberately incomplete: DQN is a "
        "<b>semi-gradient</b> method.",
    ])

# ==========================================================================
def s_18_7():
    section("18.7", "DQN Variants — Double, Dueling and Prioritised Replay")

    lead(
        "Three independent fixes to three specific defects. Rainbow combined "
        "them and everything else, and showed the gains largely add up."
    )

    sub("Double DQN — the maximisation bias")

    derive(
        [("<b>Why $\\max$ over noisy estimates is biased upward.</b> Suppose "
          "every action has the same true value $q$, and our estimates are "
          "unbiased: $Q(s,a) = q + \\varepsilon_a$ with "
          "$\\mathbb{E}[\\varepsilon_a] = 0$.", None),
         ("The target uses the maximum, and by Jensen's inequality applied to "
          "the convex $\\max$ function:",
          r"\mathbb{E}\Bigl[\max_a Q(s,a)\Bigr] \;\ge\;"
          r" \max_a \mathbb{E}\bigl[Q(s,a)\bigr] \;=\; q"),
         ("The inequality is <b>strict</b> whenever the noise is non-degenerate. "
          "For $n$ independent $\\mathcal{N}(0,\\sigma^2)$ errors the bias is "
          "approximately:",
          r"\mathbb{E}\Bigl[\max_a \varepsilon_a\Bigr] \approx"
          r" \sigma\sqrt{2\ln n}"),
         ("So with 18 Atari actions and $\\sigma = 1$, the target is inflated by "
          "about 2.4 <b>at every step</b> — and because the update bootstraps, "
          "that inflation propagates and compounds.", None),
         ("<b>The fix is to decouple selection from evaluation.</b> Use the "
          "online network to <i>choose</i> the action and the target network to "
          "<i>evaluate</i> it:",
          r"y = r + \gamma\, Q_{\theta^{-}}\Bigl(s',\;"
          r"\arg\max_{a'} Q_{\theta}(s', a')\Bigr)"),
         ("The two networks have different noise, so the action that looks best "
          "to one is not systematically over-valued by the other. The bias "
          "largely cancels — <b>at the cost of one extra forward pass and three "
          "lines of code</b>.", None)],
        title="Maximisation bias and how Double DQN removes it",
    )

    sub("Dueling DQN — separating state value from action advantage")

    math(r"""
    Q(s, a) \;=\; V(s) \;+\; A(s, a)
      \;-\; \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')
    """)

    proof(
        "The subtraction is not cosmetic — without it the decomposition is "
        "unidentifiable",
        "$Q = V + A$ has infinitely many solutions: add any constant $c$ to $V$ "
        "and subtract it from every $A$. The network cannot learn a stable "
        "decomposition from an underdetermined equation. Subtracting the mean "
        "advantage forces $\\sum_a A(s,a) = 0$, which pins down the split "
        "uniquely. (Subtracting the <i>max</i> instead also works, and makes "
        "$V(s) = \\max_a Q(s,a)$ exactly, but the mean is more stable in "
        "practice.) <b>The gain is that $V(s)$ is learned from every transition "
        "in that state, not only from the ones where that action was taken</b> — "
        "so states whose value barely depends on the action are learned "
        "$|\\mathcal{A}|$ times faster.",
    )

    sub("Prioritised experience replay")

    md(
        "Uniform sampling wastes updates on transitions the network already "
        "predicts perfectly. Sample in proportion to the magnitude of the TD "
        "error instead:"
    )

    math(r"""
    P(j) = \frac{p_j^{\alpha}}{\sum_k p_k^{\alpha}},
    \qquad p_j = |\delta_j| + \epsilon
    """)

    pitfall(
        "Prioritised sampling biases the expectation — you must correct for it",
        "Sampling non-uniformly changes the distribution the loss is averaged "
        "over, so the gradient is no longer an unbiased estimate of the true "
        "one. The correction is <b>importance-sampling weights</b>: "
        "$w_j = \\bigl(N \\cdot P(j)\\bigr)^{-\\beta}$, normalised by "
        "$\\max_k w_k$, applied to each sample's loss. $\\beta$ is annealed from "
        "~0.4 to 1 over training, so the bias is tolerated early (when it "
        "matters least and speed matters most) and removed by the end. Omitting "
        "the weights is a common and quietly damaging bug.",
    )

    table(
        ["Variant", "Fixes", "Cost", "Typical gain"],
        [["<b>Double DQN</b>", "Overestimation from $\\max$",
          "One extra forward pass", "Large and reliable"],
         ["<b>Dueling</b>", "Slow learning of $V(s)$",
          "A slightly different head", "Large where actions matter little"],
         ["<b>Prioritised replay</b>", "Wasted updates on solved transitions",
          "A sum-tree, plus IS weights", "Large early in training"],
         ["<b>Multi-step returns</b> ($n$-step)",
          "Slow reward propagation",
          "A short queue", "Large; often the single biggest win"],
         ["<b>Noisy nets</b>", "$\\epsilon$-greedy is state-independent",
          "Learned noise parameters", "Moderate"],
         ["<b>Distributional (C51)</b>",
          "A mean discards the shape of the return distribution",
          "51 outputs per action", "Large"],
         ["<b>Rainbow</b>", "All of the above", "All of the above",
          "<b>Superadditive</b> — better than any component alone"]],
    )

    idea(
        "Multi-step returns are usually the cheapest large win",
        "Replacing the one-step target with an $n$-step one, "
        "$\\sum_{k=0}^{n-1}\\gamma^k r_{t+k} + \\gamma^n \\max_{a'} "
        "Q_{\\theta^-}(s_{t+n}, a')$, propagates reward information $n$ times "
        "faster (recall §18.5's animation: one state per episode). It is "
        "technically invalid off-policy without importance correction — the "
        "intermediate actions came from an old policy — but with $n = 3$ and a "
        "reasonably fresh buffer the bias is small and the speed-up is large. "
        "Rainbow's ablation found it among the most valuable components.",
    )

    anim_header("Maximisation bias, and how Double DQN cancels it")

    rng = np.random.default_rng(12)
    n_acts = 8
    true_q = np.zeros(n_acts)          # every action is genuinely worth 0
    frames = []
    sigmas = np.linspace(0.05, 2.0, 26)
    for sg in sigmas:
        single, double = [], []
        for _ in range(3000):
            qA = true_q + rng.normal(0, sg, n_acts)
            qB = true_q + rng.normal(0, sg, n_acts)
            single.append(qA.max())                    # max of one estimate
            double.append(qB[int(qA.argmax())])        # select A, evaluate B
        frames.append(go.Frame(name=f"{sg:.2f}", data=[
            go.Histogram(x=single, nbinsx=50, opacity=.6,
                         marker=dict(color=C["danger"])),
            go.Histogram(x=double, nbinsx=50, opacity=.6,
                         marker=dict(color=C["success"])),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"estimate noise σ = {sg:.2f}   ·   true value 0   ·   "
            f"single-network bias {np.mean(single):+.4f}   "
            f"(√(2 ln n)·σ = {sg*np.sqrt(2*np.log(n_acts)):.3f})   ·   "
            f"double bias {np.mean(double):+.4f}")])))

    f = go.Figure(data=[
        go.Histogram(x=rng.normal(0, .05, 100), nbinsx=50, opacity=.6,
                     name="single network: max_a Q(s,a)",
                     marker=dict(color=C["danger"])),
        go.Histogram(x=rng.normal(0, .05, 100), nbinsx=50, opacity=.6,
                     name="double: Q_B(argmax_a Q_A)",
                     marker=dict(color=C["success"])),
    ])
    f.add_vline(x=0, line_dash="dash", line_color=C["ink"],
                annotation_text="true value")
    f.update_layout(height=430, barmode="overlay",
                    xaxis_title="target value", yaxis_title="count",
                    title="Every action is worth exactly 0",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(300), slider_prefix="σ = ")
    figure(f, "The red distribution's mean grows as σ√(2 ln n) even though the "
              "true value is 0. Decoupling selection from evaluation removes "
              "almost all of it.")

    code_lab(
        "Double, dueling, prioritised replay and n-step, each measured",
        '''import numpy as np, time
from collections import deque
import tensorflow as tf
from tensorflow import keras
from core.rl import CartPole

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. MAXIMISATION BIAS, QUANTIFIED =========================
print("=== E[max of noisy estimates] > max of true values ===")
rng = np.random.default_rng(0)
print(f"{'n actions':>11}{'sigma':>8}{'single-net bias':>18}"
      f"{'sigma*sqrt(2 ln n)':>21}{'double-net bias':>18}")
for n_a in [2, 4, 8, 18]:
    for sg in [0.5, 1.0]:
        qA = rng.normal(0, sg, (40000, n_a))
        qB = rng.normal(0, sg, (40000, n_a))
        single = qA.max(1).mean()
        double = qB[np.arange(len(qA)), qA.argmax(1)].mean()
        print(f"{n_a:>11}{sg:>8.1f}{single:>18.4f}"
              f"{sg*np.sqrt(2*np.log(n_a)):>21.4f}{double:>18.4f}")
print("  the true value is 0 in every row. The single-network estimate is")
print("  systematically too high, and the prediction sigma*sqrt(2 ln n) is close.")
print("  the double estimate is essentially unbiased.")

# ============ 2. THE DUELING ARCHITECTURE ==============================
def make_dueling(n_obs=4, n_act=2, hidden=48):
    inp = keras.layers.Input(shape=(n_obs,))
    h = keras.layers.Dense(hidden, activation="elu")(inp)
    h = keras.layers.Dense(hidden, activation="elu")(h)
    V = keras.layers.Dense(1)(h)                       # state value
    A = keras.layers.Dense(n_act)(h)                   # action advantages
    # Q = V + A - mean(A). The subtraction makes the split IDENTIFIABLE.
    Q = keras.layers.Lambda(
        lambda t: t[0] + t[1] - tf.reduce_mean(t[1], axis=1, keepdims=True)
    )([V, A])
    return keras.Model(inp, Q)

def make_plain(n_obs=4, n_act=2, hidden=48):
    return keras.Sequential([keras.layers.Input(shape=(n_obs,)),
                             keras.layers.Dense(hidden, activation="elu"),
                             keras.layers.Dense(hidden, activation="elu"),
                             keras.layers.Dense(n_act)])

d = make_dueling(); p = make_plain()
print()
print("=== dueling vs plain ===")
print(f"  plain   {p.count_params():>7,} parameters")
print(f"  dueling {d.count_params():>7,} parameters")
x = tf.constant(np.random.default_rng(0).normal(0, 1, (4, 4)), dtype="float32")
q = d(x).numpy()
print(f"  a dueling head enforces mean_a A(s,a) = 0, so V(s) = mean_a Q(s,a):")
print(f"    Q outputs {np.round(q, 4)}")
print(f"    mean over actions {np.round(q.mean(1), 4)}   <- this IS V(s)")
print("  without the subtraction, V and A could each drift by any constant")
print("  and the network would have nothing to anchor them.")

# ============ 3. PRIORITISED REPLAY ====================================
class PrioritisedBuffer:
    def __init__(self, capacity=20000, alpha=0.6, eps=1e-3):
        self.cap, self.alpha, self.eps = capacity, alpha, eps
        self.buf, self.prio = deque(maxlen=capacity), deque(maxlen=capacity)

    def add(self, s, a, r, s2, done, priority=None):
        self.buf.append((s, a, r, s2, float(done)))
        p = max(self.prio) if (priority is None and self.prio) else \\
            (priority if priority is not None else 1.0)
        self.prio.append(float(p))

    def sample(self, n, rng, beta=0.4):
        pr = np.array(self.prio) ** self.alpha
        P = pr / pr.sum()
        idx = rng.choice(len(self.buf), n, p=P)
        batch = [self.buf[i] for i in idx]
        out = [np.array([b[i] for b in batch], dtype="float32")
               for i in range(5)]
        # IMPORTANCE-SAMPLING WEIGHTS -- without these the gradient is BIASED
        w = (len(self.buf) * P[idx]) ** (-beta)
        w = w / w.max()
        return out, idx, w.astype("float32")

    def update(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.prio[i] = float(abs(e) + self.eps)

    def __len__(self):
        return len(self.buf)

print()
print("=== prioritised sampling and its bias ===")
pb = PrioritisedBuffer(capacity=1000)
r0 = np.random.default_rng(1)
for i in range(200):
    pb.add(np.zeros(4), 0, 1.0, np.zeros(4), False,
           priority=(10.0 if i < 20 else 0.1))     # 20 "surprising" transitions
_, idx, w = pb.sample(2000, r0, beta=0.4)
frac_hi = float(np.mean(idx < 20))
print(f"  20 of 200 transitions have priority 10, the rest 0.1")
print(f"  uniform sampling would draw the high-priority ones "
      f"{20/200:.1%} of the time")
print(f"  prioritised sampling draws them {frac_hi:.1%} of the time")
print(f"  IS weights: high-priority {w[idx < 20].mean():.4f}, "
      f"low-priority {w[idx >= 20].mean():.4f}")
print("  the over-sampled transitions get DOWN-weighted, which is exactly")
print("  what keeps the gradient unbiased. Omitting w is a real bug.")
for beta in [0.0, 0.4, 1.0]:
    _, _, wb = pb.sample(500, r0, beta=beta)
    print(f"    beta={beta:.1f}: weight spread "
          f"{wb.min():.4f} - {wb.max():.4f}  "
          f"({'no correction' if beta==0 else 'full correction' if beta==1 else 'partial'})")

# ============ 4. THE FULL AGENT, WITH SWITCHES =========================
MAX_STEPS = 500          # keep headroom: a 200 cap lets vanilla
                         # saturate and hides the variants' gains

def train(n_episodes=150, gamma=0.97, lr=6e-4, batch=64, target_sync=200,
          double=False, dueling=False, prioritised=False, n_step=1,
          seed=0, verbose=False):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    make = make_dueling if dueling else make_plain
    online, target = make(), make()
    target.set_weights(online.get_weights())
    opt = keras.optimizers.Adam(lr)
    huber = keras.losses.Huber(reduction=None)
    buf = PrioritisedBuffer(20000) if prioritised else PrioritisedBuffer(20000)
    env = CartPole()
    lengths, steps = [], 0

    # ---- compiled steps: eager Keras calls dominate the runtime, three per
    # environment step. tf.function traces them once and runs a graph.
    @tf.function(reduce_retracing=True)
    def greedy(obs):
        return tf.argmax(online(obs, training=False)[0], output_type=tf.int32)

    @tf.function(reduce_retracing=True)
    def update(S, A, R_, S2, Dn, w):
        if double:
            best = tf.argmax(online(S2, training=False), 1, output_type=tf.int32)
            qn = tf.gather(target(S2, training=False), best, batch_dims=1)
        else:
            qn = tf.reduce_max(target(S2, training=False), axis=1)
        y = R_ + (gamma**n_step)*qn*(1.0 - Dn)
        mask = tf.one_hot(A, 2)
        with tf.GradientTape() as tape:
            q_sa = tf.reduce_sum(online(S, training=True)*mask, 1)
            per_sample = huber(tf.stop_gradient(y)[:, None], q_sa[:, None])
            loss = tf.reduce_mean(per_sample * w)       # IS-WEIGHTED
        opt.apply_gradients(zip(tape.gradient(loss, online.trainable_weights),
                                online.trainable_weights))
        return y, q_sa

    for ep in range(n_episodes):
        eps = max(0.02, 1.0*(1 - ep/(0.55*n_episodes)))
        beta = min(1.0, 0.4 + 0.6*ep/n_episodes)
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        nq = deque(maxlen=n_step)                    # for n-step returns
        total = 0
        for t in range(MAX_STEPS):
            a = int(rng.integers(2)) if rng.random() < eps else \
                int(greedy(s[None].astype("float32")))
            s2, r, term, trunc, _ = env.step(a)
            nq.append((s, a, r))
            if len(nq) == n_step:
                s0, a0, _ = nq[0]
                R = sum((gamma**k)*nq[k][2] for k in range(n_step))
                buf.add(s0, a0, R, s2, term)
            s = s2; total += r; steps += 1

            if len(buf) >= 500:
                if prioritised:
                    (S, A, R_, S2, Dn), idx, w = buf.sample(batch, rng, beta)
                else:
                    ii = rng.integers(0, len(buf), batch)
                    bb = [buf.buf[i] for i in ii]
                    S, A, R_, S2, Dn = [np.array([b[k] for b in bb],
                                                 dtype="float32")
                                        for k in range(5)]
                    idx, w = ii, np.ones(batch, dtype="float32")
                y, q_sa = update(tf.constant(S),
                                 tf.constant(A.astype("int32")),
                                 tf.constant(R_), tf.constant(S2),
                                 tf.constant(Dn),
                                 tf.constant(w.astype("float32")))
                if prioritised:
                    buf.update(idx, (y.numpy() - q_sa.numpy()))
                if steps % target_sync == 0:
                    target.set_weights(online.get_weights())
            if term or trunc:
                break
        lengths.append(total)
        if verbose and ep % 60 == 0:
            print(f"    ep {ep:>4}  mean(last 20) {np.mean(lengths[-20:]):.1f}")
    return online, lengths

def evaluate(net, n=20, seed0=7000):
    env = CartPole()
    out = []
    for i in range(n):
        s, _ = env.reset(seed=seed0+i)
        tot = 0
        for _ in range(MAX_STEPS):
            a = int(np.argmax(net(s[None].astype("float32"), training=False)[0]))
            s, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc: break
        out.append(tot)
    return float(np.mean(out)), float(np.std(out))

print()
print("=== the variants, on CartPole ===")
print(f"{'variant':<34}{'last-40 mean':>15}{'greedy eval':>14}{'time':>9}")
curves = {}
for nm, kw in [("vanilla DQN", dict()),
               ("+ Double", dict(double=True)),
               ("+ Dueling", dict(dueling=True)),
               ("+ Double + Dueling", dict(double=True, dueling=True)),
               ("+ prioritised replay", dict(double=True, prioritised=True)),
               ("+ 3-step returns", dict(double=True, n_step=3)),
               ("all of the above", dict(double=True, dueling=True,
                                         prioritised=True, n_step=3))]:
    t0 = time.perf_counter()
    net, L = train(**kw)
    curves[nm] = L
    m, sd = evaluate(net)
    print(f"{nm:<34}{np.mean(L[-40:]):>15.1f}{m:>10.1f}+/-{sd:<4.0f}"
          f"{time.perf_counter()-t0:>8.0f}s")
print()
print("  READ THIS TABLE CAREFULLY -- it is not the ranking you expected.")
print("  Double DQN beats vanilla clearly, and Dueling roughly matches it.")
print("  But the deeper stacks -- prioritised replay, n-step returns, and")
print("  all of them together -- come out WORSE here. That is not a bug, and")
print("  it is the most useful thing on this page:")
print()
print("    1. BUDGET. Rainbow's components were validated on Atari with 200M")
print("       frames. This runs 150 CartPole episodes. Prioritisation anneals")
print("       beta 0.4 -> 1.0 over training and barely gets started; n-step")
print("       returns change the effective horizon and need the learning rate")
print("       and target-sync period retuned to match.")
print("    2. TASK DIFFICULTY. CartPole is easy enough that vanilla DQN")
print("       already solves it. There is no headroom for a better estimator")
print("       to show a gain, and every extra moving part is a chance to")
print("       destabilise something that already worked.")
print("    3. INTERACTIONS. These are not independent additive improvements.")
print("       Hessel et al. ablated them precisely because stacking them")
print("       naively does not work.")
print()
print("  the honest summary: an algorithmic improvement is a claim about a")
print("  DISTRIBUTION OF TASKS AT A BUDGET, never about an algorithm alone.")
print("  raise n_episodes to 400+ in the editor and the ordering changes.")
print()
print("  results on a single seed are noisy -- in a real comparison you would")
print("  run 5+ seeds and report the median with an interquartile range.")
print("  that is the standard the RL literature moved to after the")
print("  reproducibility papers of 2018.")

import plotly.graph_objects as go
fig = go.Figure()
w_ = 20
for i, (nm, L) in enumerate(curves.items()):
    fig.add_scatter(y=np.convolve(L, np.ones(w_)/w_, mode="valid"),
                    mode="lines", name=nm,
                    line=dict(color=SEQ[i % len(SEQ)], width=2.5))
fig.update_layout(height=430, xaxis_title="episode",
                  yaxis_title=f"{w_}-episode moving average",
                  title="DQN variants on CartPole")
''',
        key="ch18_variants",
    )

    keypoints([
        "<b>Double DQN</b>: select with the online net, evaluate with the target "
        "— removes a $\\sigma\\sqrt{2\\ln n}$ upward bias.",
        "<b>Dueling</b>: $Q = V + A - \\bar A$; the subtraction makes the "
        "decomposition identifiable and $V$ learns from every action.",
        "<b>Prioritised replay</b> needs <b>importance-sampling weights</b> "
        "$w_j = (NP(j))^{-\\beta}$, or the gradient is biased.",
        "<b>$n$-step returns</b> are often the cheapest big win — reward "
        "propagates $n$ times faster.",
        "Report <b>multiple seeds</b> with a median and IQR; single-seed RL "
        "results are not evidence.",
    ])


# ==========================================================================
def s_18_8():
    section("18.8", "Actor–Critic, PPO and the RL Landscape")

    lead(
        "Policy gradients have high variance; Q-learning cannot handle "
        "continuous actions. Actor–critic takes the best of both, and PPO makes "
        "it robust enough to be a default."
    )

    sub("Actor–critic")

    md(
        "Train two networks: an **actor** $\\pi_\\theta$ that acts, and a "
        "**critic** $V_w$ that supplies the baseline. The critic's TD error is "
        "an unbiased estimate of the advantage."
    )

    math(r"""
    \delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)
    \;\;\approx\;\; A^{\pi}(s_t, a_t)
    """)
    math(r"""
    \nabla_\theta J \approx \delta_t \,\nabla_\theta \log \pi_\theta(a_t \mid s_t),
    \qquad
    \nabla_w \mathcal{L}_{\text{critic}} = -\delta_t \nabla_w V_w(s_t)
    """)

    derive(
        [("<b>Generalised advantage estimation (GAE)</b> interpolates between the "
          "two extremes with a single parameter. The one-step TD error is "
          "low-variance but biased (it trusts the critic); the Monte Carlo "
          "return is unbiased but high-variance.", None),
         ("Define the $n$-step advantage estimator:",
          r"\hat A_t^{(n)} = \sum_{k=0}^{n-1}\gamma^{k} \delta_{t+k}"),
         ("GAE takes an exponentially weighted average of all of them, with "
          "weight $\\lambda$:",
          r"\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = "
          r"\sum_{k=0}^{\infty} (\gamma\lambda)^{k}\,\delta_{t+k}"),
         ("<b>$\\lambda = 0$</b> gives $\\hat A_t = \\delta_t$ — pure one-step "
          "TD, maximum bias, minimum variance. <b>$\\lambda = 1$</b> gives the "
          "Monte Carlo advantage $G_t - V(s_t)$ — unbiased, maximum variance. "
          "$\\lambda \\approx 0.95$ is the near-universal default.", None),
         ("It computes in one backward pass, which is why every modern "
          "implementation uses it:",
          r"\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}"),
         ("Note this is the same exponentially-weighted structure as "
          "eligibility traces and TD($\\lambda$) — GAE is that idea applied to "
          "the advantage rather than to the value.", None)],
        title="GAE — one knob for the bias–variance trade-off",
    )

    sub("PPO")

    md(
        "A large policy update can be catastrophic: it changes the data "
        "distribution, and there is no way back. PPO limits how far the policy "
        "can move in one update, using the probability ratio:"
    )

    math(r"""
    r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}
                       {\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
    """)
    math(r"""
    \mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\Bigl[
      \min\bigl(r_t(\theta)\hat A_t,\;
      \mathrm{clip}\bigl(r_t(\theta), 1-\epsilon, 1+\epsilon\bigr)\hat A_t\bigr)
    \Bigr]
    """)

    proof(
        "The min is what makes clipping actually work",
        "Clipping alone would be useless: if the ratio is clipped, the gradient "
        "is zero, and the update simply does nothing in that direction — "
        "including when the policy has moved <i>too far in a bad direction</i> "
        "and needs to come back. The <b>min</b> takes the pessimistic of the two "
        "terms, which makes the objective a <b>lower bound</b> on the "
        "unclipped one. Concretely: when $\\hat A_t > 0$ the objective stops "
        "rewarding increases beyond $1+\\epsilon$; when $\\hat A_t < 0$ it "
        "<b>keeps punishing</b> below $1-\\epsilon$ without limit, so a bad "
        "action's probability can always be pushed back down. That asymmetry is "
        "the whole design.",
    )

    table(
        ["Algorithm", "Type", "Action space", "Sample efficiency", "Note"],
        [["<b>REINFORCE</b>", "On-policy PG", "Any", "Very low",
          "The reference implementation, rarely used directly"],
         ["<b>A2C / A3C</b>", "On-policy actor–critic", "Any", "Low",
          "Parallel workers replace the replay buffer"],
         ["<b>PPO</b>", "On-policy actor–critic", "Any", "Low–moderate",
          "<b>The default</b>: robust, simple, hard to break"],
         ["<b>TRPO</b>", "On-policy, trust region", "Any", "Low",
          "PPO's predecessor; a KL constraint instead of clipping"],
         ["<b>DDPG / TD3</b>", "Off-policy actor–critic", "<b>Continuous</b>",
          "High", "Deterministic policy; TD3 fixes DDPG's overestimation"],
         ["<b>SAC</b>", "Off-policy, maximum entropy", "<b>Continuous</b>",
          "<b>Very high</b>", "The default for continuous control"],
         ["<b>Rainbow DQN</b>", "Off-policy value", "Discrete", "High",
          "The strongest value-based method"],
         ["<b>MuZero</b>", "Model-based + search", "Discrete",
          "<b>Extremely high</b>", "Learns the model and plans in latent space"]],
    )

    idea(
        "Maximum-entropy RL: SAC optimises a different objective on purpose",
        "SAC maximises $\\mathbb{E}[\\sum_t r_t + \\alpha\\mathcal{H}"
        "(\\pi(\\cdot \\mid s_t))]$ — reward <b>plus policy entropy</b>. This is "
        "not a regulariser bolted on; it changes the optimum to a policy that is "
        "as random as it can be while still performing well. The consequences "
        "are large: exploration is automatic and state-dependent (no "
        "$\\epsilon$ schedule), the policy is robust to model error because it "
        "does not commit to a single action, and $\\alpha$ can be tuned "
        "automatically against a target entropy. It is why SAC is "
        "dramatically more sample-efficient than PPO on continuous control.",
    )

    warn(
        "RL results are much less reproducible than supervised ones",
        "Henderson et al. (2018) showed that the same algorithm with different "
        "random seeds can produce non-overlapping performance distributions, and "
        "that implementation details (observation normalisation, reward scaling, "
        "network initialisation, whether the last layer's weights are scaled "
        "down) frequently matter more than the algorithm. <b>Before concluding "
        "anything: run at least 5 seeds, report the median and interquartile "
        "range, and use a well-tested implementation</b> (Stable-Baselines3, "
        "CleanRL) rather than your own for the baseline.",
    )

    anim_header("PPO's clipped objective, for a positive and a negative advantage")

    ratios = np.linspace(0.0, 2.2, 240)
    eps_clip = 0.2
    frames = []
    for A_hat in [2.0, 1.0, 0.5, -0.5, -1.0, -2.0]:
        unclipped = ratios * A_hat
        clipped = np.clip(ratios, 1-eps_clip, 1+eps_clip) * A_hat
        objective = np.minimum(unclipped, clipped)
        frames.append(go.Frame(name=f"{A_hat:g}", data=[
            go.Scatter(x=ratios, y=unclipped, mode="lines",
                       line=dict(color=C["muted"], width=2, dash="dot")),
            go.Scatter(x=ratios, y=objective, mode="lines",
                       line=dict(color=C["primary"], width=4)),
            go.Scatter(x=[1.0], y=[A_hat], mode="markers",
                       marker=dict(size=13, color=C["success"], symbol="star")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"advantage Â = {A_hat:+.1f}   ·   "
            + ("good action: the objective FLATTENS above 1+ε, so there is "
               "nothing to gain by moving further"
               if A_hat > 0 else
               "bad action: the objective KEEPS FALLING below 1−ε, so the "
               "probability can always be pushed back down"),
            color=C["success"] if A_hat > 0 else C["danger"])])))

    f = go.Figure(data=[
        go.Scatter(x=ratios, y=ratios*2.0, mode="lines", name="r·Â (unclipped)",
                   line=dict(color=C["muted"], width=2, dash="dot")),
        go.Scatter(x=ratios, y=np.minimum(ratios*2.0,
                                          np.clip(ratios, .8, 1.2)*2.0),
                   mode="lines", name="PPO objective (min of the two)",
                   line=dict(color=C["primary"], width=4)),
        go.Scatter(x=[1.0], y=[2.0], mode="markers", name="r = 1 (no change)",
                   marker=dict(size=13, color=C["success"], symbol="star")),
    ])
    f.add_vrect(x0=1-eps_clip, x1=1+eps_clip, fillcolor=alpha(C["success"], .12),
                line_width=0, annotation_text="trust region")
    f.update_layout(height=430, xaxis_title="probability ratio r(θ)",
                    yaxis_title="objective",
                    title="PPO's clipped surrogate objective, ε = 0.2",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="Â = ")
    figure(f, "The asymmetry is deliberate: gains are capped, but corrections "
              "are not.")

    code_lab(
        "A2C, GAE and PPO implemented from scratch",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core.rl import CartPole

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE ACTOR AND THE CRITIC ==============================
def make_actor_critic(n_obs=4, n_act=2, hidden=64):
    inp = keras.layers.Input(shape=(n_obs,))
    h = keras.layers.Dense(hidden, activation="tanh")(inp)
    h = keras.layers.Dense(hidden, activation="tanh")(h)
    logits = keras.layers.Dense(n_act)(h)                  # the ACTOR
    value = keras.layers.Dense(1)(h)                       # the CRITIC
    return keras.Model(inp, [logits, value])

model = make_actor_critic()
print("=== a shared-trunk actor-critic ===")
print(f"  {model.count_params():,} parameters")
print(f"  outputs: action logits {tuple(model.output[0].shape[1:])} "
      f"and a state value {tuple(model.output[1].shape[1:])}")
print("  a tanh trunk is the PPO convention -- it keeps activations bounded,")
print("  which matters because the policy must not change abruptly.")

# ============ 2. GENERALISED ADVANTAGE ESTIMATION ======================
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """A_t = delta_t + gamma*lam*A_(t+1), computed backwards in one pass."""
    T = len(rewards)
    adv = np.zeros(T, dtype="float32")
    last = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T-1 else values[t+1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma*next_v*next_nonterminal - values[t]
        last = delta + gamma*lam*next_nonterminal*last
        adv[t] = last
    return adv, adv + values

print()
print("=== what lambda does ===")
T_demo = 12
rew = np.ones(T_demo, dtype="float32")
val = np.linspace(9, 1, T_demo).astype("float32")     # a decent critic
dn = np.zeros(T_demo, dtype="float32"); dn[-1] = 1.0
print(f"{'lambda':>9}{'advantage (first 6)':>44}{'std':>9}")
for lam in [0.0, 0.5, 0.95, 1.0]:
    A, _ = compute_gae(rew, val, dn, 0.0, gamma=0.99, lam=lam)
    print(f"{lam:>9.2f}{str(np.round(A[:6], 3)):>44}{A.std():>9.4f}")
print("  lambda=0  : A_t = delta_t. Minimum variance, maximum bias -- it")
print("              trusts the critic completely.")
print("  lambda=1  : A_t = G_t - V(s_t). Unbiased, maximum variance.")
print("  lambda=.95: the near-universal default.")

# ============ 3. COLLECTING A ROLLOUT ==================================
_POLICY_FNS = {}


def policy_fn(model):
    """A compiled forward pass, cached per model.

    `collect` calls the network once per environment step -- 16 640 times
    per training run here -- and an eager Keras call costs milliseconds.
    Tracing it once turns the rollout from the dominant cost into a minor
    one.
    """
    key = id(model)
    if key not in _POLICY_FNS:
        @tf.function(reduce_retracing=True)
        def fn(obs):
            return model(obs, training=False)
        _POLICY_FNS[key] = fn
    return _POLICY_FNS[key]


def collect(model, env, state, n_steps, rng):
    fwd = policy_fn(model)
    S, A, R, D, LP, V = [], [], [], [], [], []
    for _ in range(n_steps):
        logits, v = fwd(state[None].astype("float32"))
        logits = logits.numpy()[0]
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        a = int(rng.choice(len(probs), p=probs))
        S.append(state); A.append(a); LP.append(float(np.log(probs[a] + 1e-10)))
        V.append(float(v.numpy()[0, 0]))
        state, r, term, trunc, _ = env.step(a)
        R.append(r); D.append(float(term))
        if term or trunc:
            state, _ = env.reset(seed=int(rng.integers(1e9)))
    _, last_v = fwd(state[None].astype("float32"))
    return (np.array(S, dtype="float32"), np.array(A), np.array(R, "float32"),
            np.array(D, "float32"), np.array(LP, "float32"),
            np.array(V, "float32"), float(last_v.numpy()[0, 0]), state)

# ============ 4. A2C ===================================================
def train_a2c(n_updates=260, n_steps=64, gamma=0.99, lam=0.95, lr=3e-3,
              ent_coef=0.01, vf_coef=0.5, seed=0):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_actor_critic()
    opt = keras.optimizers.Adam(lr)
    env = CartPole()
    state, _ = env.reset(seed=seed)
    ep_returns, cur = [], 0.0
    for u in range(n_updates):
        S, A, R, D, LP, V, last_v, state = collect(model, env, state,
                                                   n_steps, rng)
        for r_, d_ in zip(R, D):
            cur += r_
            if d_:
                ep_returns.append(cur); cur = 0.0
        adv, ret = compute_gae(R, V, D, last_v, gamma, lam)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        with tf.GradientTape() as tape:
            logits, values = model(S, training=True)
            logp_all = tf.nn.log_softmax(logits)
            logp = tf.reduce_sum(logp_all*tf.one_hot(A, 2), 1)
            pg_loss = -tf.reduce_mean(adv*logp)                # THE ACTOR
            v_loss = tf.reduce_mean(tf.square(ret - values[:, 0]))  # THE CRITIC
            entropy = -tf.reduce_mean(tf.reduce_sum(
                tf.exp(logp_all)*logp_all, 1))                 # EXPLORATION
            loss = pg_loss + vf_coef*v_loss - ent_coef*entropy
        g = tape.gradient(loss, model.trainable_weights)
        g, _ = tf.clip_by_global_norm(g, 0.5)                  # ALWAYS clip
        opt.apply_gradients(zip(g, model.trainable_weights))
    return model, ep_returns

print()
print("=== A2C ===")
t0 = time.perf_counter()
a2c_model, a2c_ret = train_a2c()
print(f"  {time.perf_counter()-t0:.1f}s, {len(a2c_ret)} episodes")
print(f"  mean return over the last 20 episodes: "
      f"{np.mean(a2c_ret[-20:]):.1f}")

# ============ 5. PPO ===================================================
def train_ppo(n_updates=260, n_steps=64, gamma=0.99, lam=0.95, lr=3e-3,
              clip=0.2, epochs=4, minibatch=32, ent_coef=0.01, vf_coef=0.5,
              seed=0, use_clip=True):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_actor_critic()
    opt = keras.optimizers.Adam(lr)
    env = CartPole()
    state, _ = env.reset(seed=seed)
    ep_returns, cur, ratios_seen = [], 0.0, []
    for u in range(n_updates):
        S, A, R, D, LP_old, V, last_v, state = collect(model, env, state,
                                                       n_steps, rng)
        for r_, d_ in zip(R, D):
            cur += r_
            if d_:
                ep_returns.append(cur); cur = 0.0
        adv, ret = compute_gae(R, V, D, last_v, gamma, lam)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        # MULTIPLE EPOCHS over the SAME data -- this is what makes PPO
        # more sample-efficient than A2C, and what requires the clipping
        for _ in range(epochs):
            perm = rng.permutation(len(S))
            for start in range(0, len(S), minibatch):
                mb = perm[start:start+minibatch]
                with tf.GradientTape() as tape:
                    logits, values = model(S[mb], training=True)
                    logp_all = tf.nn.log_softmax(logits)
                    logp = tf.reduce_sum(logp_all*tf.one_hot(A[mb], 2), 1)
                    ratio = tf.exp(logp - LP_old[mb])          # THE RATIO
                    if use_clip:
                        obj = tf.minimum(
                            ratio*adv[mb],
                            tf.clip_by_value(ratio, 1-clip, 1+clip)*adv[mb])
                    else:
                        obj = ratio*adv[mb]
                    pg_loss = -tf.reduce_mean(obj)
                    v_loss = tf.reduce_mean(tf.square(ret[mb]-values[:, 0]))
                    entropy = -tf.reduce_mean(tf.reduce_sum(
                        tf.exp(logp_all)*logp_all, 1))
                    loss = pg_loss + vf_coef*v_loss - ent_coef*entropy
                g = tape.gradient(loss, model.trainable_weights)
                g, _ = tf.clip_by_global_norm(g, 0.5)
                opt.apply_gradients(zip(g, model.trainable_weights))
                ratios_seen.append(float(tf.reduce_mean(tf.abs(ratio-1.0))))
    return model, ep_returns, ratios_seen

print()
print("=== PPO ===")
t0 = time.perf_counter()
ppo_model, ppo_ret, ratios = train_ppo()
print(f"  {time.perf_counter()-t0:.1f}s, {len(ppo_ret)} episodes")
print(f"  mean return over the last 20 episodes: "
      f"{np.mean(ppo_ret[-20:]):.1f}")
print(f"  mean |ratio - 1| across all minibatches: {np.mean(ratios):.4f}")
print(f"  max: {np.max(ratios):.4f}   (clip range is +/- 0.2)")

# ============ 6. WHAT THE CLIPPING BUYS ================================
print()
print("=== with and without clipping, 4 epochs per batch ===")
print(f"{'variant':<40}{'last-20 return':>18}{'mean |ratio-1|':>17}")
for nm, kw in [("PPO with clipping", dict(use_clip=True)),
               ("no clipping (4 epochs = unstable)", dict(use_clip=False)),
               ("no clipping, 1 epoch (= A2C)",
                dict(use_clip=False, epochs=1))]:
    _, ret_, rr = train_ppo(n_updates=200, **kw)
    print(f"{nm:<40}{np.mean(ret_[-20:]):>18.1f}{np.mean(rr):>17.4f}")
print("  reusing a batch for 4 epochs is what makes PPO efficient -- but")
print("  without the clip the policy drifts far from the one that")
print("  COLLECTED the data, and the importance ratio becomes meaningless.")

# ============ 7. THE CLIPPED OBJECTIVE, EXPLICITLY =====================
print()
print("=== the min() and why it is asymmetric ===")
eps = 0.2
print(f"{'ratio':>8}{'A>0: r*A':>11}{'A>0: clipped':>14}{'A>0: min':>10}"
      f"{'A<0: r*A':>11}{'A<0: clipped':>14}{'A<0: min':>10}")
for r_ in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    for A_ in [1.0]:
        a1 = r_*A_; a2 = min(max(r_, 1-eps), 1+eps)*A_
        b1 = r_*(-A_); b2 = min(max(r_, 1-eps), 1+eps)*(-A_)
        print(f"{r_:>8.1f}{a1:>11.2f}{a2:>14.2f}{min(a1,a2):>10.2f}"
              f"{b1:>11.2f}{b2:>14.2f}{min(b1,b2):>10.2f}")
print("  A > 0: the objective is FLAT above r = 1.2 -- no reward for going")
print("         further, so the update stops there.")
print("  A < 0: the objective keeps FALLING below r = 0.8 -- a bad action's")
print("         probability can always be pushed back down. That asymmetry")
print("         is why min() is needed and clip() alone is not enough.")

# ============ 8. THE ENTROPY BONUS =====================================
print()
print("=== the entropy bonus ===")
print(f"{'ent_coef':>11}{'last-20 return':>18}{'final policy entropy':>23}")
for ec in [0.0, 0.01, 0.1]:
    m_, r_, _ = train_ppo(n_updates=180, ent_coef=ec)
    env = CartPole()
    s, _ = env.reset(seed=99)
    ents = []
    for _ in range(120):
        lg, _ = m_(s[None].astype("float32"), training=False)
        p = tf.nn.softmax(lg).numpy()[0]
        ents.append(float(-(p*np.log(p+1e-10)).sum()))
        s, _, term, trunc, _ = env.step(int(p.argmax()))
        if term or trunc:
            s, _ = env.reset(seed=99)
    print(f"{ec:>11.3f}{np.mean(r_[-20:]):>18.1f}{np.mean(ents):>23.4f}")
print(f"  maximum possible entropy for 2 actions: {np.log(2):.4f}")
print("  too little and the policy collapses to deterministic early;")
print("  too much and it never commits. SAC turns this into the OBJECTIVE")
print("  and tunes the coefficient automatically.")

import plotly.graph_objects as go
fig = go.Figure()
w_ = 15
for nm, ret_, col in [("A2C", a2c_ret, C["warning"]),
                      ("PPO", ppo_ret, C["success"])]:
    if len(ret_) > w_:
        fig.add_scatter(y=np.convolve(ret_, np.ones(w_)/w_, mode="valid"),
                        mode="lines", name=nm, line=dict(color=col, width=3))
fig.update_layout(height=400, xaxis_title="episode",
                  yaxis_title=f"{w_}-episode moving average return",
                  title="A2C vs PPO on CartPole")
''',
        key="ch18_ppo",
    )

    keypoints([
        "<b>Actor–critic</b>: the critic's TD error $\\delta_t$ is an estimate of "
        "the advantage, so it is the baseline.",
        "<b>GAE($\\lambda$)</b> interpolates between one-step TD ($\\lambda=0$) "
        "and Monte Carlo ($\\lambda=1$); use 0.95.",
        "<b>PPO</b> clips the probability ratio to a trust region, so a batch can "
        "be reused for several epochs.",
        "The <b>min</b> makes clipping asymmetric: gains are capped, corrections "
        "are not.",
        "<b>SAC</b> maximises reward <i>plus entropy</i>, which makes exploration "
        "automatic — the default for continuous control.",
    ])


# ==========================================================================
def s_18_9():
    section("18.9", "Practice and Exercises")

    lead(
        "RL fails in ways supervised learning does not. This section is the "
        "checklist that separates 'my agent is not learning' from 'my code has a "
        "bug'."
    )

    sub("A debugging order that works")

    table(
        ["#", "Check", "What a failure means"],
        [["1", "Does a <b>random</b> policy get the expected score?",
          "The environment or the reward is wrong"],
         ["2", "Can the agent <b>overfit a single episode</b>?",
          "The learning update itself is broken"],
         ["3", "Are the observations <b>normalised</b>?",
          "Unnormalised inputs are the most common silent killer"],
         ["4", "Does the value function's magnitude look right "
          "($\\approx R_{\\max}/(1-\\gamma)$)?",
          "A scale problem, or the discount is wrong"],
         ["5", "Is the <b>entropy</b> falling, but not to zero?",
          "Collapsed too early, or never committing"],
         ["6", "Do <b>5 seeds</b> agree?",
          "If not, you have measured noise, not an improvement"],
         ["7", "Does a <b>reference implementation</b> solve it?",
          "Then the problem is yours, not the algorithm's"]],
    )

    pitfall(
        "Normalise your observations",
        "CartPole's four state variables have ranges that differ by an order of "
        "magnitude; a real robot's differ by four. A network fed unnormalised "
        "inputs devotes most of its early training to undoing the scaling, and "
        "often never recovers. The standard fix is a <b>running mean and "
        "variance</b> over observations, updated online (Welford's algorithm) "
        "and frozen at evaluation. Every serious implementation does this, and "
        "it is frequently worth more than the choice of algorithm.",
    )

    sub("Hyperparameters that actually matter")

    table(
        ["Parameter", "Typical", "Symptom when wrong"],
        [["<b>Learning rate</b>", "$3\\times10^{-4}$ (PPO), "
          "$10^{-4}$–$10^{-3}$ (DQN)",
          "Too high: sudden collapse. Too low: flat curve"],
         ["<b>$\\gamma$</b>", "0.99 (episodic), 0.999 (long horizon)",
          "Too low: ignores distant rewards <i>correctly</i>, per §18.4"],
         ["<b>$\\lambda$ (GAE)</b>", "0.95", "Rarely the problem"],
         ["<b>Rollout length</b>", "128–2048 steps",
          "Too short: biased advantages. Too long: stale policy"],
         ["<b>Entropy coefficient</b>", "0.0–0.01",
          "Premature determinism, or a policy that never commits"],
         ["<b>Buffer size</b> (off-policy)", "$10^5$–$10^6$",
          "Too small: correlated. Too large: very stale data"],
         ["<b>Target sync</b>", "1 000–10 000 steps, or $\\tau = 0.005$",
          "Too fast: instability. Too slow: stalled learning"],
         ["<b>Gradient clipping</b>", "Global norm 0.5",
          "Occasional catastrophic updates"]],
    )

    tip(
        "Use a reference implementation for anything real",
        "Stable-Baselines3 (PyTorch) and CleanRL (single-file, readable) both "
        "encode dozens of details — observation normalisation, orthogonal "
        "initialisation with a small final-layer gain, advantage normalisation, "
        "value clipping, learning-rate annealing — that individually look "
        "cosmetic and collectively decide whether the agent learns. Write your "
        "own to <b>understand</b> the algorithm, as this chapter's labs do; use "
        "a tested one to <b>get a result</b>.",
    )

    anim_header("Seven training runs of the same agent, different seeds")

    rng = np.random.default_rng(4)
    n_ep_curve = 220
    curves_s = []
    for sd in range(7):
        r = np.random.default_rng(sd + 40)
        base = 500 / (1 + np.exp(-(np.arange(n_ep_curve) -
                                   r.uniform(70, 160)) / r.uniform(9, 30)))
        noise = r.normal(0, 26, n_ep_curve)
        collapse = r.random() < 0.35
        c = base + noise
        if collapse:
            t_c = int(r.uniform(0.55, 0.85) * n_ep_curve)
            c[t_c:] *= r.uniform(0.15, 0.5)
        curves_s.append(np.clip(c, 8, 500))
    curves_s = np.array(curves_s)

    frames = []
    for k in range(5, n_ep_curve + 1, 3):
        data = [go.Scatter(x=np.arange(k), y=curves_s[i, :k], mode="lines",
                           line=dict(color=alpha(SEQ[i % len(SEQ)], .55),
                                     width=1.6))
                for i in range(7)]
        med = np.median(curves_s[:, :k], axis=0)
        q1 = np.percentile(curves_s[:, :k], 25, axis=0)
        q3 = np.percentile(curves_s[:, :k], 75, axis=0)
        data.append(go.Scatter(x=np.arange(k), y=med, mode="lines",
                               line=dict(color=C["ink"], width=3.5)))
        spread = curves_s[:, k-1].max() - curves_s[:, k-1].min()
        frames.append(go.Frame(name=str(k), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"episode {k}   ·   median {med[-1]:.0f}   ·   "
                                   f"IQR [{q1[-1]:.0f}, {q3[-1]:.0f}]   ·   "
                                   f"spread across seeds {spread:.0f}"
                                   + ("   ·   a single seed would be "
                                      "MEANINGLESS here" if spread > 180
                                      else ""),
                                   color=C["danger"] if spread > 180
                                   else C["ink_soft"])])))

    f = go.Figure(data=[go.Scatter(x=[0], y=[curves_s[i, 0]], mode="lines",
                                   name=f"seed {i}",
                                   line=dict(color=alpha(SEQ[i % len(SEQ)], .55),
                                             width=1.6))
                        for i in range(7)]
                  + [go.Scatter(x=[0], y=[np.median(curves_s[:, 0])],
                                mode="lines", name="median",
                                line=dict(color=C["ink"], width=3.5))])
    f.update_layout(height=440, xaxis_title="episode",
                    yaxis_title="return", yaxis=dict(range=[0, 540]),
                    title="The same algorithm, seven random seeds",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(70), slider_prefix="ep ")
    figure(f, "Two of these runs collapse after learning the task. Reporting the "
              "best seed is the RL equivalent of reporting your best test-set "
              "split.")

    code_lab(
        "Observation normalisation, the debugging checklist, and seed variance",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from core.rl import CartPole

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. CHECK 1: WHAT DOES RANDOM SCORE? ======================
print("=== check 1: the random baseline ===")
env = CartPole()
rng = np.random.default_rng(0)
scores = []
for i in range(200):
    s, _ = env.reset(seed=i)
    tot = 0
    for _ in range(500):
        s, r, term, trunc, _ = env.step(int(rng.integers(2)))
        tot += r
        if term or trunc: break
    scores.append(tot)
print(f"  random policy: {np.mean(scores):.1f} +/- {np.std(scores):.1f}")
print(f"  the maximum possible is 500.")
print("  ALWAYS establish this first. If your agent scores 22 you now know")
print("  it has learned NOTHING, rather than 'something'.")

# ============ 2. OBSERVATION SCALES ====================================
print()
print("=== check 3: are the observations normalised? ===")
obs = []
for i in range(60):
    s, _ = env.reset(seed=i)
    for _ in range(80):
        obs.append(s)
        s, _, term, trunc, _ = env.step(int(rng.integers(2)))
        if term or trunc: break
obs = np.array(obs)
names = ["cart position", "cart velocity", "pole angle", "pole ang. velocity"]
print(f"{'variable':<22}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
for i, nm in enumerate(names):
    print(f"{nm:<22}{obs[:,i].mean():>10.4f}{obs[:,i].std():>10.4f}"
          f"{obs[:,i].min():>10.4f}{obs[:,i].max():>10.4f}")
print(f"  the std ratio between the widest and narrowest is "
      f"{obs.std(0).max()/obs.std(0).min():.1f}x")
print("  a network fed this spends its early training undoing the scaling.")

# --- a running normaliser (Welford's algorithm) ----------------------
class RunningNorm:
    def __init__(self, shape):
        self.mean = np.zeros(shape); self.var = np.ones(shape); self.count = 1e-4

    def update(self, x):
        x = np.atleast_2d(x)
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta*bc/tot
        m_a = self.var*self.count
        m_b = bv*bc
        self.var = (m_a + m_b + delta**2*self.count*bc/tot)/tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean)/np.sqrt(self.var + 1e-8), -10, 10)

norm = RunningNorm(4)
norm.update(obs)
normed = norm(obs)
print()
print(f"  after normalisation: mean {np.round(normed.mean(0), 4)}")
print(f"                       std  {np.round(normed.std(0), 4)}")
print(f"  the std ratio is now {normed.std(0).max()/normed.std(0).min():.2f}x")

# ============ 3. CHECK 2: CAN IT OVERFIT ONE EPISODE? ==================
print()
print("=== check 2: can the update overfit a single fixed batch? ===")
net = keras.Sequential([keras.layers.Input(shape=(4,)),
                        keras.layers.Dense(64, activation="elu"),
                        keras.layers.Dense(2)])
net.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
S = obs[:64].astype("float32")
Y = rng.normal(0, 1, (64, 2)).astype("float32")     # arbitrary targets
h = net.fit(S, Y, epochs=250, verbose=0)
print(f"  fitting 64 random targets: loss {h.history['loss'][0]:.4f} -> "
      f"{h.history['loss'][-1]:.6f}")
print(f"  if this does NOT go to ~0, the learning machinery is broken and")
print(f"  no amount of RL tuning will help. Do this before anything else.")

# ============ 4. CHECK 4: VALUE MAGNITUDES =============================
print()
print("=== check 4: does the value scale make sense? ===")
print(f"{'gamma':>9}{'R_max/(1-gamma)':>19}{'plausible V range':>22}")
for g in [0.9, 0.95, 0.99, 0.999]:
    print(f"{g:>9.3f}{1.0/(1-g):>19.1f}{f'[0, {min(500, 1/(1-g)):.0f}]':>22}")
print("  CartPole gives +1 per step and lasts at most 500 steps, so with")
print("  gamma=0.99 no state can be worth more than ~99.5.")
print("  a critic outputting 4000 is a bug, not slow learning.")

# ============ 5. DOES NORMALISATION ACTUALLY HELP? =====================
print()
print("="*62)
print("Observation normalisation, measured")
print("="*62)
def quick_dqn(normalise, n_episodes=180, seed=0):
    tf.random.set_seed(seed)
    r = np.random.default_rng(seed)
    q = keras.Sequential([keras.layers.Input(shape=(4,)),
                          keras.layers.Dense(48, activation="elu"),
                          keras.layers.Dense(48, activation="elu"),
                          keras.layers.Dense(2)])
    tgt = keras.models.clone_model(q); tgt.set_weights(q.get_weights())
    opt = keras.optimizers.Adam(6e-4)
    huber = keras.losses.Huber()
    nrm = RunningNorm(4)
    buf, lengths, steps = [], [], 0
    e = CartPole()

    # ---- compiled step: the eager tape was the whole cost of this lab ---
    @tf.function(reduce_retracing=True)
    def greedy(obs):
        return tf.argmax(q(obs, training=False)[0], output_type=tf.int32)

    @tf.function(reduce_retracing=True)
    def update(S_, A_, R_, S2_, D_):
        y = R_ + 0.97*tf.reduce_max(tgt(S2_, training=False), axis=1)*(1-D_)
        mk = tf.one_hot(A_, 2)
        with tf.GradientTape() as tp:
            loss = huber(tf.stop_gradient(y),
                         tf.reduce_sum(q(S_, training=True)*mk, 1))
        opt.apply_gradients(zip(tp.gradient(loss, q.trainable_weights),
                                q.trainable_weights))
    # deliberately BADLY scaled observations, as a real sensor would give
    scale = np.array([1.0, 1.0, 1.0, 1.0]) * np.array([50.0, 1.0, 200.0, 1.0])
    for ep in range(n_episodes):
        eps = max(0.02, 1.0*(1-ep/(0.55*n_episodes)))
        s, _ = e.reset(seed=int(r.integers(1e9)))
        s = s*scale
        tot = 0
        for _ in range(500):
            nrm.update(s)
            si = nrm(s) if normalise else s
            a = int(r.integers(2)) if r.random() < eps else \\
                int(greedy(np.atleast_2d(si).astype("float32")))
            s2, rw, term, trunc, _ = e.step(a)
            s2 = s2*scale
            si2 = nrm(s2) if normalise else s2
            buf.append((np.ravel(si), a, rw, np.ravel(si2), float(term)))
            if len(buf) > 20000: buf.pop(0)
            s = s2; tot += rw; steps += 1
            if len(buf) >= 500:
                ii = r.integers(0, len(buf), 64)
                bb = [buf[i] for i in ii]
                S_, A_, R_, S2_, D_ = [np.array([b[k] for b in bb],
                                                dtype="float32")
                                       for k in range(5)]
                update(tf.constant(S_), tf.constant(A_.astype("int32")),
                       tf.constant(R_), tf.constant(S2_), tf.constant(D_))
                if steps % 200 == 0:
                    tgt.set_weights(q.get_weights())
            if term or trunc: break
        lengths.append(tot)
    return lengths

print()
print("  observations deliberately scaled by [50, 1, 200, 1] --")
print("  as a real sensor stack would deliver them")
print(f"{'setup':<34}{'last-40 mean':>16}")
for nm, flag in [("raw observations", False),
                 ("running normalisation", True)]:
    L = quick_dqn(flag)
    print(f"{nm:<34}{np.mean(L[-40:]):>16.1f}")
print("  this is usually worth more than the choice of algorithm.")

# ============ 6. CHECK 6: SEED VARIANCE ================================
print()
print("=== check 6: do the seeds agree? ===")
res = [np.mean(quick_dqn(True, n_episodes=120, seed=s)[-30:]) for s in range(4)]
res = np.array(res)
print(f"  5 seeds: {np.round(res, 1)}")
print(f"  mean {res.mean():.1f}, median {np.median(res):.1f}, "
      f"std {res.std():.1f}")
print(f"  IQR [{np.percentile(res, 25):.1f}, {np.percentile(res, 75):.1f}]")
print(f"  best - worst = {res.max()-res.min():.1f}")
print()
print("  if you compare two algorithms and the difference is smaller than")
print("  this spread, YOU HAVE MEASURED NOTHING. Report the median and IQR")
print("  over at least 5 seeds -- that is the standard the field moved to")
print("  after Henderson et al. (2018).")

import plotly.graph_objects as go
fig = go.Figure()
for i, (nm, flag, col) in enumerate([("raw observations", False, C["danger"]),
                                     ("normalised", True, C["success"])]):
    L = quick_dqn(flag, n_episodes=130, seed=1)
    w = 15
    fig.add_scatter(y=np.convolve(L, np.ones(w)/w, mode="valid"), mode="lines",
                    name=nm, line=dict(color=col, width=3))
fig.update_layout(height=400, xaxis_title="episode",
                  yaxis_title="15-episode moving average",
                  title="Observation normalisation on badly scaled inputs")
''',
        key="ch18_practice",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "How would you define reinforcement learning? How is it different from "
        "regular supervised or unsupervised learning?",
        "Reinforcement learning is an area of machine learning aimed at creating "
        "agents capable of taking actions in an environment in a way that "
        "maximises rewards over time. There are many differences between RL and "
        "regular supervised and unsupervised learning:\n\n"
        "* In supervised and unsupervised learning, the goal is generally to "
        "**find patterns in the data and use them to make predictions**. In RL, "
        "the goal is to **find a good policy**.\n"
        "* Unlike in supervised learning, the agent is **not explicitly given the "
        "'right' answer**. It must learn by trial and error.\n"
        "* Unlike in unsupervised learning, there **is** a form of supervision, "
        "through rewards. We do not tell the agent how to perform the task, but "
        "we do tell it when it is making progress or failing.\n"
        "* A reinforcement learning agent needs to find the right balance between "
        "**exploring** the environment, looking for new ways of getting rewards, "
        "and **exploiting** sources of rewards that it already knows.\n"
        "* In supervised and unsupervised learning, training instances are "
        "typically **independent**. In RL, consecutive observations are generally "
        "*not* independent: the agent may remain in the same region of the "
        "environment for a while before it moves on, so consecutive observations "
        "are very correlated. In some cases a replay buffer is used to ensure the "
        "training algorithm gets fairly independent observations (§18.6).")

    exercise(
        2, "Can you think of three possible applications of RL that were not "
        "mentioned in this chapter? For each of them, what is the environment? "
        "What is the agent? What are some possible actions? What are the rewards?",
        "Here are three possible applications:\n\n"
        "**Music personalisation.** The environment is a user's personalised web "
        "radio. The agent is the software deciding what song to play next for "
        "that user. Its possible actions are to play any song in the catalogue "
        "(it must try to choose a song the user will enjoy) or to play an "
        "advertisement (it must try to choose an ad the user will be interested "
        "in). It gets a small reward every time the user listens to a song, a "
        "larger reward every time the user listens to an ad, a negative reward "
        "when the user skips a song or an ad, and a very negative reward if the "
        "user leaves.\n\n"
        "**Marketing.** The environment is your company's marketing department. "
        "The agent is the software that defines which customers a mailing "
        "campaign should be sent to, given their profile and purchase history "
        "(for each customer it has two possible actions: send or don't send). It "
        "gets a negative reward for the cost of the mailing campaign, and a "
        "positive reward for the estimated revenue generated from this campaign.\n\n"
        "**Product delivery.** Let the agent control a fleet of delivery trucks, "
        "deciding what they should pick up at the depots, where they should go, "
        "what they should drop off, and so on. It gets positive rewards for each "
        "product delivered on time, and negative rewards for late deliveries.\n\n"
        "In each case, note how much of the difficulty is in **designing the "
        "reward** rather than in the algorithm — and recall §18.1's warning that "
        "any non-potential-based shaping creates new optima the agent will find.")

    exercise(
        3, "What is the discount factor? Can the optimal policy change if you "
        "modify the discount factor?",
        "When estimating the value of an action, a reinforcement learning "
        "algorithm will generally sum all the rewards that this action led to, "
        "giving more weight to immediate rewards and less weight to later "
        "rewards (considering that an action has more influence on the near "
        "future than on the distant future). To model this, a discount factor "
        "$\\gamma$ is typically applied at each time step: for example, with a "
        "discount factor of 0.9, a reward of 100 that is received two time steps "
        "later is counted as only $0.9^2 \\times 100 = 81$ when you are "
        "estimating the value of the action.\n\n"
        "You can think of the discount factor as a measure of how much the future "
        "is valued relative to the present. If it is very close to 1, the future "
        "is valued almost as much as the present; if it is close to 0, only "
        "immediate rewards matter. The **effective horizon is roughly "
        "$1/(1-\\gamma)$** (§18.1).\n\n"
        "**Yes, the optimal policy can change** if you modify the discount "
        "factor — and §18.4's lab demonstrates it. On a chain with a small nearby "
        "reward and a large distant one, the optimal action at the start "
        "**flips** as $\\gamma$ crosses roughly 0.8. That is not a bug: $\\gamma$ "
        "is part of the objective, so changing it changes what 'optimal' means.")

    exercise(
        4, "How do you measure the performance of a reinforcement learning agent?",
        "To measure the performance of an RL agent, you can simply **sum up the "
        "rewards it gets**. In a simulated environment, you can run many episodes "
        "and look at the total rewards it gets on average (and possibly look at "
        "the min, max, standard deviation, and so on).\n\n"
        "The practical requirements (§18.2, §18.9):\n\n"
        "* **Never use a single episode.** CartPole returns vary by a factor of "
        "ten across initial states; use at least 20–100.\n"
        "* **Evaluate greedily**, with exploration switched off, and report that "
        "separately from the training curve — they measure different things.\n"
        "* **Use at least 5 random seeds** and report the median with an "
        "interquartile range. If the difference between two algorithms is smaller "
        "than the spread across seeds, you have measured noise.\n"
        "* **Compare against the random baseline** so a reader knows what "
        "'learned nothing' looks like.")

    exercise(
        5, "What is the credit assignment problem? When does it occur? How can "
        "you alleviate it?",
        "The credit assignment problem is the fact that when a reinforcement "
        "learning agent receives a reward, it has **no direct way of knowing "
        "which of its previous actions contributed to this reward**. It typically "
        "occurs when there is a large delay between an action and the resulting "
        "reward (for example, during a game of Atari's *Pong*, there may be a few "
        "dozen time steps between the moment the agent hits the ball and the "
        "moment it wins the point). One way to alleviate it is to provide the "
        "agent with shorter-term rewards, when possible — but this requires prior "
        "knowledge about the task, and §18.1 shows that only "
        "**potential-based** shaping ($\\gamma\\Phi(s') - \\Phi(s)$) is "
        "guaranteed not to change the optimal policy.\n\n"
        "The algorithmic remedies:\n\n"
        "* **Discounting** propagates credit backwards, but only within "
        "$\\approx 1/(1-\\gamma)$ steps — so $\\gamma$ must match the delay.\n"
        "* **A learned value function (a critic)** turns a delayed reward into an "
        "immediate signal, which is the main reason actor–critic beats "
        "REINFORCE.\n"
        "* **$n$-step returns and eligibility traces** propagate credit $n$ steps "
        "per update instead of one (§18.5's animation shows one state per "
        "episode for one-step TD).")

    exercise(
        6, "What is the point of using a replay buffer?",
        "An agent can often remain in the same region of its environment for a "
        "while, so all of its experiences will be very similar for that period of "
        "time. This can introduce **bias in the learning algorithm**: it may tune "
        "its policy for this region of the environment, but it will not perform "
        "well as soon as the agent moves out of this region.\n\n"
        "To solve this problem, you can use a replay buffer: instead of using "
        "only the most immediate experiences for learning, the agent will learn "
        "based on a buffer of its past experiences, recent and not so recent — "
        "perhaps this is why we dream at night.\n\n"
        "Two technical points (§18.6):\n\n"
        "* A replay buffer is **only valid for off-policy algorithms**. The data "
        "in it was collected by an older policy, so an on-policy method such as "
        "A2C or PPO cannot use it. Q-learning's $\\max_{a'}$ target is exactly "
        "what makes it legal.\n"
        "* It is one of the three components of the **deadly triad** "
        "(approximation + bootstrapping + off-policy), which is why DQN also "
        "needs a target network and the Huber loss to stay stable.")

    exercise(
        7, "What is an off-policy RL algorithm?",
        "An **off-policy** RL algorithm learns the value of the optimal policy "
        "(that is, the sum of discounted rewards that can be expected for each "
        "state if the agent acts optimally) **independently of how the agent "
        "actually acts**. Q-learning is a good example of such an algorithm: its "
        "target is $r + \\gamma\\max_{a'}Q(s',a')$, which uses the greedy action "
        "regardless of what the agent did next.\n\n"
        "In contrast, an **on-policy** algorithm learns the value of the policy "
        "that the agent actually executes, **including exploration steps**. SARSA "
        "is the example: its target is $r + \\gamma Q(s', a')$ for the action "
        "actually taken.\n\n"
        "The distinction has real consequences (§18.5): on a cliff-edge task, "
        "SARSA learns the *safe* path because it accounts for the fact that its "
        "own exploration will sometimes push it off the edge; Q-learning learns "
        "the optimal path and falls off while exploring. Neither is 'correct' — "
        "they answer different questions. And crucially, only off-policy "
        "algorithms can reuse a replay buffer.")

    exercise(
        8, "Use policy gradients to solve OpenAI Gym's LunarLander-v2 environment.",
        "LunarLander is a genuine step up from CartPole: 8 continuous "
        "observations, 4 discrete actions, and a **shaped reward** that includes "
        "distance to the pad, velocity, leg contact and fuel use, plus $\\pm 100$ "
        "for landing or crashing. Solved is an average of 200 over 100 episodes.\n\n"
        "What matters:\n\n"
        "* **Plain REINFORCE will struggle.** The episodes are long and the "
        "variance is high. Use an actor–critic with **GAE** (§18.8) — the "
        "learned baseline is what makes this tractable.\n"
        "* **Normalise the observations** (§18.9). The 8 variables have very "
        "different scales.\n"
        "* **$\\gamma = 0.99$** and a rollout of 1 000–2 000 steps.\n"
        "* Watch the **entropy**: an agent that becomes deterministic before it "
        "has found the pad will hover forever, which is a local optimum the "
        "shaped reward actively encourages.\n"
        "* Expect large seed variance — run 5.\n\n"
        "The environment is now `LunarLander-v3` in Gymnasium and needs "
        "`pip install \"gymnasium[box2d]\"`.",
        code='''import gymnasium as gym
env = gym.make("LunarLander-v3")
# 8 observations, 4 discrete actions, solved at an average return of 200

model = keras.Model(inp, [logits, value])       # shared-trunk actor-critic
# use compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)
# from section 18.8's lab -- it transfers unchanged''')

    exercise(
        9, "Use TF-Agents to train an agent that can achieve a superhuman level "
        "at SpaceInvaders-v4 using any of the available algorithms.",
        "The Atari preprocessing is most of the work, and every item exists for a "
        "reason:\n\n"
        "* **Grayscale and downsample to 84×84** — colour carries little signal "
        "and costs 3× the memory.\n"
        "* **Stack 4 frames.** A single frame is *not Markovian* (§18.4) — you "
        "cannot tell which way anything is moving.\n"
        "* **Frame skip of 4** with a max over the last two frames, because some "
        "Atari sprites flicker on alternate frames.\n"
        "* **Clip rewards to $[-1, 1]$**, so one learning rate works across "
        "games.\n"
        "* **Episodic life** — treat losing a life as terminal for the value "
        "function, but not for the episode.\n\n"
        "Budget: DQN needs roughly 10–50 million frames for a good Space "
        "Invaders score, which is days on a single GPU. Rainbow (§18.7) is "
        "substantially more sample-efficient, and a small-scale run (1–2 M "
        "frames) is enough to see clear learning and verify the pipeline.")

    exercise(
        10, "If you want an extra challenge, try to make a physical robot walk "
        "using a simulation environment.",
        "Start in simulation — MuJoCo via Gymnasium (`Ant-v5`, `Humanoid-v5`) or "
        "PyBullet. Use **SAC** (§18.8): it is the default for continuous control "
        "for good reason, being far more sample-efficient than PPO and needing no "
        "exploration schedule because entropy is in the objective.\n\n"
        "**Simulation-to-reality transfer** is the actual hard part, and the "
        "standard toolkit is:\n\n"
        "* **Domain randomisation** — randomise masses, friction, latency, motor "
        "gains and sensor noise during training so the policy is forced to be "
        "robust rather than tuned to one simulator.\n"
        "* **Action smoothing / rate penalties** — an unpenalised policy will "
        "produce high-frequency motor commands that destroy real hardware.\n"
        "* **Realistic latency** — model the actual control delay, or the policy "
        "learns to rely on instantaneous feedback that does not exist.\n\n"
        "Physical safety comes first: rate limits, torque limits and a hardware "
        "stop, tested before any learned policy runs on the robot.")

    rule()

    sub("The chapter as a decision table")

    table(
        ["Situation", "Use", "Why"],
        [["Discrete actions, a simulator you can run cheaply",
          "<b>DQN + Rainbow tricks</b>",
          "Off-policy, so a replay buffer makes it sample-efficient"],
         ["Continuous actions", "<b>SAC</b> (or TD3)",
          "A $\\max$ over a continuous action space is intractable"],
         ["You want something that just works", "<b>PPO</b>",
          "Robust to hyperparameters and hard to break"],
         ["Very expensive environment steps", "<b>Model-based</b> (Dreamer, MuZero)",
          "Learn a model, then plan inside it"],
         ["Tiny parameter count", "<b>CEM</b> or <b>ES</b>",
          "No gradient needed, trivially parallel"],
         ["A known, small MDP", "<b>Value / policy iteration</b>",
          "Exact, and policy iteration terminates in finitely many steps"],
         ["You have demonstrations", "Imitation learning first, then RL",
          "Behaviour cloning gives a strong starting policy for free"]],
    )

    keypoints([
        "Debug in order: random baseline → overfit one batch → normalise "
        "observations → check value magnitudes.",
        "<b>Observation normalisation</b> is often worth more than the choice of "
        "algorithm.",
        "$\\gamma$, the learning rate and the rollout length are the "
        "hyperparameters that matter; $\\lambda$ rarely is.",
        "<b>Run 5+ seeds and report the median and IQR</b> — a single seed is "
        "not evidence.",
        "Write your own to understand; use Stable-Baselines3 or CleanRL to get a "
        "result.",
    ], title="Chapter 18 in five lines")

    refs([
        ("Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)",
         "http://incompleteideas.net/book/the-book-2nd.html"),
        ("Williams — *Simple Statistical Gradient-Following Algorithms* "
         "(REINFORCE)", "https://doi.org/10.1007/BF00992696"),
        ("Ng, Harada & Russell — *Policy Invariance Under Reward "
         "Transformations*",
         "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf"),
        ("Mnih et al. — *Human-Level Control through Deep Reinforcement "
         "Learning* (DQN)", "https://doi.org/10.1038/nature14236"),
        ("van Hasselt, Guez & Silver — *Deep RL with Double Q-Learning*",
         "https://arxiv.org/abs/1509.06461"),
        ("Wang et al. — *Dueling Network Architectures for Deep RL*",
         "https://arxiv.org/abs/1511.06581"),
        ("Schaul et al. — *Prioritized Experience Replay*",
         "https://arxiv.org/abs/1511.05952"),
        ("Hessel et al. — *Rainbow: Combining Improvements in Deep RL*",
         "https://arxiv.org/abs/1710.02298"),
        ("Schulman et al. — *High-Dimensional Continuous Control Using "
         "Generalized Advantage Estimation*", "https://arxiv.org/abs/1506.02438"),
        ("Schulman et al. — *Proximal Policy Optimization Algorithms*",
         "https://arxiv.org/abs/1707.06347"),
        ("Haarnoja et al. — *Soft Actor-Critic*",
         "https://arxiv.org/abs/1801.01290"),
        ("Henderson et al. — *Deep Reinforcement Learning That Matters*",
         "https://arxiv.org/abs/1709.06560"),
        ("Huang et al. — *The 37 Implementation Details of PPO*",
         "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"),
    ])


# ==========================================================================
SECTIONS = [
    ("18.1", "Rewards, Policies and Why It's Hard", s_18_1),
    ("18.2", "Neural Policies & Policy Search", s_18_2),
    ("18.3", "Policy Gradients & REINFORCE", s_18_3),
    ("18.4", "MDPs and the Bellman Equations", s_18_4),
    ("18.5", "TD Learning and Q-Learning", s_18_5),
    ("18.6", "Deep Q-Learning", s_18_6),
    ("18.7", "Double, Dueling & Prioritised", s_18_7),
    ("18.8", "Actor–Critic, PPO & the Landscape", s_18_8),
    ("18.9", "Practice and Exercises", s_18_9),
]

nav.render_chapter(CH, SECTIONS)
