"""
core.rl
=======
Self-contained reinforcement-learning environments for Chapter 18.

Nothing here depends on Gymnasium, so every lab runs offline.  The API is
deliberately the familiar one::

    env = CartPole()
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

so the code transfers to Gymnasium unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CartPole", "GridWorld", "MDP", "chain_mdp"]


# --------------------------------------------------------------------------
# CartPole — the standard control benchmark, physics and all
# --------------------------------------------------------------------------


class CartPole:
    """The classic cart-pole balancing task.

    State  : (x, x_dot, theta, theta_dot)
    Action : 0 = push left, 1 = push right
    Reward : +1 per surviving step
    Done   : |theta| > 12 degrees, or |x| > 2.4, or 500 steps
    """

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    length = 0.5                      # half the pole's length
    force_mag = 10.0
    tau = 0.02                        # seconds per step
    theta_threshold = 12 * np.pi / 180
    x_threshold = 2.4
    max_steps = 500

    n_actions = 2
    n_obs = 4

    def __init__(self, max_steps: int | None = None):
        if max_steps is not None:
            self.max_steps = max_steps
        self.state = np.zeros(4)
        self.steps = 0
        self._rng = np.random.default_rng()

    # ----------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self._rng.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)

        total_mass = self.mass_cart + self.mass_pole
        polemass_length = self.mass_pole * self.length

        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        theta_acc = ((self.gravity * sintheta - costheta * temp)
                     / (self.length * (4.0 / 3.0
                                       - self.mass_pole * costheta ** 2
                                       / total_mass)))
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        # semi-implicit Euler, exactly as in the reference implementation
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        terminated = bool(abs(x) > self.x_threshold
                          or abs(theta) > self.theta_threshold)
        truncated = bool(self.steps >= self.max_steps)
        return self.state.copy(), 1.0, terminated, truncated, {}


# --------------------------------------------------------------------------
# A tabular grid world — small enough to solve exactly
# --------------------------------------------------------------------------


class GridWorld:
    """A 4x4 grid with a goal, a pit and a wall.

    Actions: 0 up, 1 right, 2 down, 3 left.  Moves succeed with probability
    ``slip_free`` and otherwise go to a random perpendicular direction, which
    makes the MDP genuinely stochastic.
    """

    n_actions = 4
    MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = ["up", "right", "down", "left"]

    def __init__(self, slip_free: float = 0.8, step_cost: float = -0.04):
        self.rows = self.cols = 4
        self.walls = {(1, 1)}
        self.goal = (0, 3)
        self.pit = (1, 3)
        self.start = (3, 0)
        self.slip_free = slip_free
        self.step_cost = step_cost
        self.n_states = self.rows * self.cols
        self._rng = np.random.default_rng()

    # ----------------------------------------------------------------
    def s_index(self, cell) -> int:
        return cell[0] * self.cols + cell[1]

    def s_cell(self, s: int):
        return divmod(s, self.cols)

    def is_terminal(self, cell) -> bool:
        return cell in (self.goal, self.pit)

    def _move(self, cell, a):
        if self.is_terminal(cell):
            return cell
        dr, dc = self.MOVES[a]
        nr, nc = cell[0] + dr, cell[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return cell
        if (nr, nc) in self.walls:
            return cell
        return (nr, nc)

    def transitions(self, s: int, a: int):
        """Return [(probability, next_state, reward, terminal), ...]."""
        cell = self.s_cell(s)
        if self.is_terminal(cell):
            return [(1.0, s, 0.0, True)]
        slip = (1.0 - self.slip_free) / 2.0
        outcomes = [(self.slip_free, a), (slip, (a - 1) % 4), (slip, (a + 1) % 4)]
        out = []
        for p, act in outcomes:
            nxt = self._move(cell, act)
            r = (1.0 if nxt == self.goal
                 else -1.0 if nxt == self.pit
                 else self.step_cost)
            out.append((p, self.s_index(nxt), r, self.is_terminal(nxt)))
        return out

    # ----------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.cell = self.start
        self.steps = 0
        return self.s_index(self.cell), {}

    def step(self, a: int):
        probs = [p for p, *_ in self.transitions(self.s_index(self.cell), a)]
        outs = self.transitions(self.s_index(self.cell), a)
        k = self._rng.choice(len(outs), p=np.array(probs) / sum(probs))
        _, s2, r, term = outs[k]
        self.cell = self.s_cell(s2)
        self.steps += 1
        return s2, r, term, self.steps >= 100, {}


# --------------------------------------------------------------------------
# A generic finite MDP, for value iteration and policy iteration
# --------------------------------------------------------------------------


class MDP:
    """A finite MDP given explicitly as P[s, a, s'] and R[s, a, s']."""

    def __init__(self, P: np.ndarray, R: np.ndarray, terminal=()):
        self.P, self.R = np.asarray(P, float), np.asarray(R, float)
        self.n_states, self.n_actions = self.P.shape[:2]
        self.terminal = set(terminal)

    @classmethod
    def from_grid(cls, grid: "GridWorld") -> "MDP":
        n_s, n_a = grid.n_states, grid.n_actions
        P = np.zeros((n_s, n_a, n_s))
        R = np.zeros((n_s, n_a, n_s))
        for s in range(n_s):
            for a in range(n_a):
                for p, s2, r, _ in grid.transitions(s, a):
                    P[s, a, s2] += p
                    R[s, a, s2] = r
        terminal = {grid.s_index(grid.goal), grid.s_index(grid.pit)}
        return cls(P, R, terminal)


def chain_mdp(n: int = 7, gamma_reward: float = 1.0) -> MDP:
    """A linear chain: reward only at the right-hand end.

    Deliberately built so a discount factor that is too small makes the
    optimal policy *wrong* — used in §18.4 to show what gamma does.
    """
    n_a = 2                                   # 0 = left, 1 = right
    P = np.zeros((n, n_a, n))
    R = np.zeros((n, n_a, n))
    for s in range(n):
        for a, d in [(0, -1), (1, 1)]:
            s2 = min(max(s + d, 0), n - 1)
            P[s, a, s2] = 1.0
    R[n - 2, 1, n - 1] = gamma_reward         # big reward at the far right
    R[1, 0, 0] = 0.1                          # small reward, close by
    return MDP(P, R, terminal={0, n - 1})
