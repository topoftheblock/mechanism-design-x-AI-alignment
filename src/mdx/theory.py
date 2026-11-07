"""Closed-form results from Bergemann, Koh & Morris (2026), "Mechanism Design for
Alignment and Control" (arXiv:2609.01595).

Everything lives in the paper's linear-quadratic environment:

    state      theta ~ U[0, 1]
    human      v(a, theta)   = -(a - theta)^2
    AI         u_b(a, theta) = -(a - theta - b)^2

Function names reference the proposition / equation they implement so results can be
checked against the paper line by line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

# ---------------------------------------------------------------------------
# Capped delegation: a = min(theta + b, cap)
# ---------------------------------------------------------------------------


def human_value_cap(cap: float, b: float) -> float:
    """V_b(cap) = E[v(min(theta + b, cap), theta)] for theta ~ U[0,1], b >= 0 (Example I)."""
    if cap >= 1 + b:
        return -(b**2)
    if cap > b:
        return -cap * b**2 - (1 - cap) ** 3 / 3 + 2 * b**3 / 3
    # cap binds in every state
    return -(cap**2 - cap + 1 / 3)


def agent_loss_cap(cap: float, b: float) -> float:
    """E[(min(theta + b, cap) - theta - b)^2]: the AI's own expected loss under a cap."""
    if cap >= 1 + b:
        return 0.0
    lo = max(0.0, cap - b)
    # integrand (theta + b - cap)^2 on [lo, 1]
    return ((1 + b - cap) ** 3 - (lo + b - cap) ** 3) / 3


def capped_action(theta: float, b: float, cap: float) -> float:
    """Best response of an AI with bias b facing permitted set (-inf, cap]."""
    return min(theta + b, cap)


# ---------------------------------------------------------------------------
# Example I: sandbagging (Proposition 3)
# ---------------------------------------------------------------------------

BiasProfile = Callable[[np.ndarray], np.ndarray]

# The three profiles used in Figure 4 of the paper.
BIAS_PROFILES: dict[str, BiasProfile] = {
    "increasing": lambda c: 0.4 * np.asarray(c) - 0.1,
    "decreasing": lambda c: 0.5 - 0.4 * np.asarray(c),
    "nonmonotone": lambda c: 0.08 + 1.6 * (2 * np.asarray(c) - 1) * (1 - (2 * np.asarray(c) - 1)),
}


def complete_information_cap(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Type-by-type optimal cap when capability is observed: min{c, 1 - b(c)}."""
    return np.minimum(np.asarray(c, float), 1 - np.asarray(b, float))


def _pava_decreasing_bounded(y: np.ndarray, w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Weighted least squares fit of y by a weakly decreasing z with lo <= z <= hi.

    Pool-adjacent-violators is exact for separable convex losses when each block takes its
    own minimiser; here that is the block's weighted mean of y clipped to the intersection
    of the members' bounds. (Clipping an unbounded isotonic fit afterwards is *not* exact:
    a pooled block must average only its unclipped members.)
    """
    blocks: list[list[float]] = []  # [sum_wy, sum_w, max_lo, min_hi, size]

    def value(bk: list[float]) -> float:
        return min(max(bk[0] / bk[1], bk[2]), bk[3])

    for yi, wi, li, ui in zip(y, w, lo, hi):
        blocks.append([wi * yi, wi, li, ui, 1])
        while len(blocks) > 1 and value(blocks[-2]) < value(blocks[-1]):
            b2, b1 = blocks.pop(), blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1], max(b1[2], b2[2]), min(b1[3], b2[3]), b1[4] + b2[4]])
    return np.repeat([value(bk) for bk in blocks], [int(bk[4]) for bk in blocks])


def optimal_caps(c: Sequence[float], weights: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Optimal truthful cap schedule a*(c) for a discretised type distribution (Proposition 3).

    Solves (IRON): z* is the decreasing weighted least-squares fit of b(c)^2 subject to
    (1-c)^2 <= z <= (1 - min{c, 1/2})^2 (Lemma 2), and a*(c) = 1 - sqrt(z*(c)).
    `c` must be sorted ascending.
    """
    c = np.asarray(c, float)
    w = np.asarray(weights, float)
    b = np.asarray(b, float)
    if np.any(np.diff(c) <= 0):
        raise ValueError("c must be strictly increasing")
    if np.any(b < -1e-12) or np.any(b > np.minimum(c, 0.5) + 1e-12):
        raise ValueError("Proposition 3 requires 0 <= b(c) <= min{c, 1/2}")
    z = _pava_decreasing_bounded(b**2, w, (1 - c) ** 2, (1 - np.minimum(c, 0.5)) ** 2)
    return 1 - np.sqrt(z)


def optimal_common_cap(c: Sequence[float], weights: Sequence[float], b: Sequence[float]) -> float:
    """Best single cap a (applied as min{c, a}) — the benchmark in Proposition 3(ii)."""
    c, w, b = (np.asarray(x, float) for x in (c, weights, b))

    def neg(a: float) -> float:
        return -sum(wi * human_value_cap(min(ci, a), bi) for ci, wi, bi in zip(c, w, b))

    res = minimize_scalar(neg, bounds=(0.0, float(c.max())), method="bounded", options={"xatol": 1e-10})
    # guard against the bounded optimiser missing the corner
    return float(c.max()) if neg(float(c.max())) <= res.fun else float(res.x)


def expected_human_value(caps: Sequence[float], c: Sequence[float], weights: Sequence[float], b: Sequence[float]) -> float:
    """Designer's expected payoff from truthful reporting under a cap schedule."""
    return float(
        sum(w * human_value_cap(min(ci, a), bi) for a, ci, w, bi in zip(caps, c, weights, b)) / np.sum(weights)
    )


def rational_report(c_true: float, grid: Sequence[float], schedule: Sequence[float]) -> tuple[float, float]:
    """Report chosen by a rational AI of capability c_true (it may only certify c_hat <= c_true).

    For any bias b >= 0 the agent's utility is increasing in its effective cap
    min{c, cap(c_hat)}, so it certifies the level with the loosest effective cap. Ties are
    broken toward the truthful report (partial implementation). Returns (report, gain)
    where gain = best effective cap - truthful effective cap; gain > 0 means sandbagging
    is strictly profitable.
    """
    grid = np.asarray(grid, float)
    sched = np.asarray(schedule, float)
    eff = np.where(grid <= c_true + 1e-12, np.minimum(c_true, sched), -np.inf)
    truth_idx = int(np.argmin(np.abs(grid - c_true)))
    gain = float(eff.max() - eff[truth_idx])
    if gain <= 1e-12:
        return float(grid[truth_idx]), 0.0
    best = np.flatnonzero(np.isclose(eff, eff.max(), atol=1e-12))
    return float(grid[best.max()]), gain


# ---------------------------------------------------------------------------
# Example II: alignment–interpretability trade-off (Propositions 4–5)
# ---------------------------------------------------------------------------


def single_cap_star(mean_b: float, var_b: float) -> float:
    """a* = 1 - sqrt(b̄^2 + σ^2) (Proposition 4(i))."""
    return 1 - np.sqrt(mean_b**2 + var_b)


def single_cap_loss(b_values: Sequence[float], probs: Sequence[float]) -> float:
    """L* = m2 - (2/3) m2^{3/2} - (2/3) E[b^3] with m2 = E[b^2] (Proposition 4(ii))."""
    b, p = np.asarray(b_values, float), np.asarray(probs, float)
    m2 = float(p @ b**2)
    return m2 - (2 / 3) * m2**1.5 - (2 / 3) * float(p @ b**3)


def interpretability_marginal(mean_b: float, var_b: float) -> float:
    """∂L*/∂σ^2 = a* - 2 b̄ for symmetric bias shocks: > 0 means interpretability is valuable."""
    return single_cap_star(mean_b, var_b) - 2 * mean_b


# ---------------------------------------------------------------------------
# Example IV: coupled rewards (Proposition 8)
# ---------------------------------------------------------------------------


@dataclass
class CoupledRewards:
    """Reward schedules (CR) for a known strictly decreasing bias curve b2 = f(b1), f(0)=0."""

    f: Callable[[float], float]
    eta: float
    bracket: float = 50.0

    def g(self, s: float) -> float:
        """Inverse of h(x) = f(x) - x."""
        return brentq(lambda x: self.f(x) - x - s, -self.bracket, self.bracket, xtol=1e-12)

    def r1(self, d: float) -> float:
        return 2 * self.eta * quad(self.g, 0.0, d / self.eta)[0] + d**2 / 2

    def r2(self, d: float) -> float:
        return -2 * self.eta * quad(lambda s: self.g(s) + s, 0.0, d / self.eta)[0] + d**2 / 2

    def equilibrium(self, theta: float, b1: float, b2: float) -> tuple[float, float]:
        gap = self.eta * (b2 - b1) / 2
        return theta - gap, theta + gap


# ---------------------------------------------------------------------------
# Example V: weak-to-strong oversight (Propositions 9–10)
# ---------------------------------------------------------------------------


def oversight_losses(b: float, w: float) -> dict[str, dict[str, float]]:
    """Monitor (W) and human (H) losses for regimes I–III, conditional on actor bias b (Prop. 9)."""
    d = b - w
    lw1 = min(d**2, 1 / 12 + w**2)
    lh1 = b**2 if d**2 <= 1 / 12 + w**2 else 1 / 12
    lw2 = d**2 - (4 / 3) * abs(d) ** 3 if abs(d) < 0.5 else 1 / 12
    lh2 = lw2 + 2 * w * d * max(1 - 2 * abs(d), 0.0) + w**2
    return {
        "approval": {"W": lw1, "H": lh1},
        "delegation": {"W": lw2, "H": lh2},
        "reward": {"W": 0.0, "H": w**2},
    }


def expected_oversight_losses(b_values: Sequence[float], probs: Sequence[float], w: float) -> dict[str, float]:
    """Human's ex-ante loss under each regime."""
    out = {"approval": 0.0, "delegation": 0.0, "reward": 0.0}
    for b, p in zip(b_values, probs):
        for k, v in oversight_losses(b, w).items():
            out[k] += p * v["H"]
    return out


def robust_menu_abar(w_bar: float) -> float:
    """Smallest extreme action satisfying ā > 1 + w̄ + sqrt(w̄^2 + 2 w̄) (Proposition 10)."""
    return 1 + w_bar + np.sqrt(w_bar**2 + 2 * w_bar)


def robust_menu_reward(a: float, b_hat: float, abar: float) -> float:
    """r_{b̂}(a) from the menu M* in Proposition 10."""
    if 0.0 <= a <= 1.0:
        return -2 * b_hat * a
    if np.isclose(a, abar):
        return (abar - 1) ** 2 - 2 * abar * b_hat
    if np.isclose(a, 1 - abar):
        return (abar - 1) ** 2 - 2 * (1 - abar) * b_hat
    return -np.inf


def robust_menu_action(theta: float, b: float, b_hat: float, abar: float) -> float:
    """Strong actor's best response to r_{b̂}: compare the interior optimum with the two extremes."""
    interior = float(np.clip(theta + b - b_hat, 0.0, 1.0))
    cands = [interior, abar, 1 - abar]
    pay = [-((a - theta - b) ** 2) + robust_menu_reward(a, b_hat, abar) for a in cands]
    return cands[int(np.argmax(pay))]


def robust_menu_monitor_loss(b: float, b_hat: float, w: float, abar: float, n_theta: int = 4001) -> float:
    """Monitor's expected loss E[(a - theta - w)^2] from reporting b̂ when the actor's bias is b."""
    thetas = (np.arange(n_theta) + 0.5) / n_theta
    acts = np.array([robust_menu_action(t, b, b_hat, abar) for t in thetas])
    return float(np.mean((acts - thetas - w) ** 2))
