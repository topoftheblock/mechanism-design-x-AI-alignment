"""Check every closed form in mdx.theory against brute-force numerics or numbers printed in the paper."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar

from mdx import theory

THETA = (np.arange(20000) + 0.5) / 20000  # midpoint grid on U[0,1]


@pytest.mark.parametrize("cap,b", [(0.7, 0.25), (0.2, 0.3), (0.9, 0.0), (1.4, 0.3), (0.5, 0.5), (1.1, 0.05)])
def test_human_value_cap_matches_integral(cap, b):
    a = np.minimum(THETA + b, cap)
    assert theory.human_value_cap(cap, b) == pytest.approx(-np.mean((a - THETA) ** 2), abs=1e-7)
    assert theory.agent_loss_cap(cap, b) == pytest.approx(np.mean((a - THETA - b) ** 2), abs=1e-7)


def test_value_peaks_at_one_minus_b():  # Figure 3(b)
    for b in (0.15, 0.35):
        res = minimize_scalar(lambda a: -theory.human_value_cap(a, b), bounds=(0, 1.5), method="bounded")
        assert res.x == pytest.approx(1 - b, abs=1e-5)


def _brute_force_caps(c, b):
    """Maximise sum V_b(a) s.t. 0 <= a_k <= c_k, a nondecreasing, by SLSQP."""
    n = len(c)
    cons = [{"type": "ineq", "fun": (lambda a, k=k: a[k + 1] - a[k])} for k in range(n - 1)]
    res = minimize(lambda a: -sum(theory.human_value_cap(ak, bk) for ak, bk in zip(a, b)),
                   x0=np.minimum(c, 0.6), bounds=list(zip(np.zeros(n), c)), constraints=cons,
                   method="SLSQP", options={"ftol": 1e-13, "maxiter": 500})
    return res.x, -res.fun


@pytest.mark.parametrize("profile", list(theory.BIAS_PROFILES))
def test_optimal_caps_match_brute_force(profile):
    c = np.linspace(0.5, 1.0, 15)
    b = theory.BIAS_PROFILES[profile](c)
    caps = theory.optimal_caps(c, np.ones_like(c), b)
    assert np.all(np.diff(caps) >= -1e-12) and np.all(caps <= c + 1e-12)  # implementable
    _, best = _brute_force_caps(c, b)
    ours = sum(theory.human_value_cap(a, bk) for a, bk in zip(caps, b))
    assert ours >= best - 1e-8


def test_prop3_decreasing_bias_attains_complete_information():
    c = np.linspace(0.5, 1.0, 50)
    b = theory.BIAS_PROFILES["decreasing"](c)
    np.testing.assert_allclose(theory.optimal_caps(c, np.ones_like(c), b), theory.complete_information_cap(c, b), atol=1e-12)


def test_prop3_increasing_bias_evaluation_is_futile():
    c = np.linspace(0.5, 1.0, 50)
    w = np.ones_like(c)
    b = theory.BIAS_PROFILES["increasing"](c)
    v_eval = theory.expected_human_value(theory.optimal_caps(c, w, b), c, w, b)
    a_star = theory.optimal_common_cap(c, w, b)
    v_common = theory.expected_human_value(np.minimum(c, a_star), c, w, b)
    assert v_eval == pytest.approx(v_common, abs=1e-6)
    # and the complete-information cap is not implementable here
    assert np.any(np.diff(theory.complete_information_cap(c, b)) < 0)


def test_rational_report():
    grid = np.round(np.arange(0.5, 1.0001, 0.05), 4)
    inverted = np.minimum(grid, 1.5 - grid)
    assert theory.rational_report(0.9, grid, inverted)[0] == pytest.approx(0.75)
    assert theory.rational_report(0.9, grid, inverted)[1] > 0
    assert theory.rational_report(0.6, grid, inverted) == (pytest.approx(0.6), 0.0)
    increasing = np.minimum(grid, 0.8)
    assert theory.rational_report(0.95, grid, increasing) == (pytest.approx(0.95), 0.0)  # tie -> truth


def test_prop4_single_cap_and_loss():
    b, p = np.array([0.1, 0.25, 0.4]), np.array([0.3, 0.4, 0.3])
    mean, var = p @ b, p @ (b - p @ b) ** 2
    assert np.all(b <= theory.single_cap_star(mean, var))  # condition (INT)
    loss = lambda a: -sum(pi * theory.human_value_cap(a, bi) for bi, pi in zip(b, p))
    res = minimize_scalar(loss, bounds=(0, 1), method="bounded", options={"xatol": 1e-10})
    assert res.x == pytest.approx(theory.single_cap_star(mean, var), abs=1e-6)
    assert theory.single_cap_loss(b, p) == pytest.approx(res.fun, abs=1e-9)


def test_fig6_sign_change():  # σ†² = (1 - b̄)(1 - 3b̄), σ† ≈ 0.22 at b̄ = 0.31
    sig2 = (1 - 0.31) * (1 - 3 * 0.31)
    assert np.sqrt(sig2) == pytest.approx(0.22, abs=0.005)
    assert theory.interpretability_marginal(0.31, sig2) == pytest.approx(0, abs=1e-12)


def test_prop5_complements_in_value():
    def L(m, v):
        return theory.single_cap_loss([m - np.sqrt(v), m + np.sqrt(v)], [0.5, 0.5])
    m, v, h = 0.3, 0.01, 1e-4
    cross = (L(m + h, v + h) - L(m + h, v - h) - L(m - h, v + h) + L(m - h, v - h)) / (4 * h * h)
    assert cross == pytest.approx(-2 - m / np.sqrt(m**2 + v), rel=1e-3)


def test_prop8_coupled_rewards_equilibrium():
    cr = theory.CoupledRewards(f=lambda x: -x - 0.5 * x**3, eta=0.3)
    theta, b1 = 0.4, -0.6
    b2 = cr.f(b1)
    a1, a2 = 0.0, 0.0
    for _ in range(200):  # best-response iteration
        a1 = minimize_scalar(lambda x: (x - theta - b1) ** 2 - cr.r1(a2 - x), bounds=(-3, 3), method="bounded",
                             options={"xatol": 1e-10}).x
        a2 = minimize_scalar(lambda x: (x - theta - b2) ** 2 - cr.r2(x - a1), bounds=(-3, 3), method="bounded",
                             options={"xatol": 1e-10}).x
    e1, e2 = cr.equilibrium(theta, b1, b2)
    assert a1 == pytest.approx(e1, abs=1e-4) and a2 == pytest.approx(e2, abs=1e-4)


@pytest.mark.parametrize("b,w", [(0.3, 0.1), (0.0, 0.1), (0.05, 0.1), (0.9, 0.05), (-0.2, 0.1)])
def test_prop9_losses_by_simulation(b, w):
    L = theory.oversight_losses(b, w)
    # Regime III: linear reward -2(b - w)a makes the actor choose θ + w
    assert L["reward"]["H"] == pytest.approx(np.mean((THETA + w - THETA) ** 2))
    # Regime II: simulate the paper's optimal permission set
    d = b - w
    if 0 < d < 0.5:
        a = np.minimum(THETA + b, 1 + 2 * w - b)
    elif -0.5 < d < 0:
        a = np.maximum(THETA + b, 2 * w - b)
    elif d == 0:
        a = THETA + b
    else:
        a = np.full_like(THETA, w + 0.5)
    assert L["delegation"]["W"] == pytest.approx(np.mean((a - THETA - w) ** 2), abs=1e-6)
    assert L["delegation"]["H"] == pytest.approx(np.mean((a - THETA) ** 2), abs=1e-6)


def test_fig12_regime_crossovers():
    """Panel (a): approval best below p = 1/11, delegation between, reward above p = 11/91."""
    best = lambda p: min(theory.expected_oversight_losses([0, 0.3], [1 - p, p], 0.1).items(), key=lambda kv: kv[1])[0]
    assert best(0.05) == "approval"
    assert best(0.105) == "delegation"
    assert best(0.15) == "reward"
    L = lambda p: theory.expected_oversight_losses([0, 0.3], [1 - p, p], 0.1)
    assert L(1 / 11)["approval"] == pytest.approx(L(1 / 11)["delegation"], abs=1e-12)
    assert L(11 / 91)["delegation"] == pytest.approx(L(11 / 91)["reward"], abs=1e-12)


@pytest.mark.parametrize("w", [-0.1, 0.0, 0.07, 0.1])
def test_prop10_truthful_report_uniquely_optimal(w):
    abar = theory.robust_menu_abar(0.1) + 0.05
    b = 0.3
    reports = np.round(np.arange(0.0, 0.61, 0.05), 4)
    losses = [theory.robust_menu_monitor_loss(b, bh, w, abar, n_theta=2001) for bh in reports]
    assert reports[int(np.argmin(losses))] == pytest.approx(b)
    assert sorted(losses)[1] > min(losses) + 1e-6
    assert min(losses) == pytest.approx(w**2, abs=1e-6)  # truthful report implements a = θ
