"""Reproduce figures from the paper with the closed forms in `theory`, plus a plot for
sandbagging runs. Reproducing the paper's figures is the first check that the theory code
matches the paper before any agent is involved.

    python -m mdx.figures                       # paper figures -> figures/
    python -m mdx.figures --run results/sandbagging/<run-dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import theory  # noqa: E402


def figure4(out: Path) -> None:
    """Optimal evaluated caps across bias profiles (paper Figure 4)."""
    c = np.linspace(0.5, 1.0, 201)
    w = np.ones_like(c)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for row, prof in zip(axes, ("increasing", "decreasing", "nonmonotone")):
        b = theory.BIAS_PROFILES[prof](c)
        cap = theory.optimal_caps(c, w, b)
        ci = theory.complete_information_cap(c, b)
        row[0].plot(c, b)
        row[0].set(title=f"{prof}: bias b(c)", xlabel="capability c")
        row[1].plot(c, cap, label="optimal evaluated cap")
        row[1].plot(c, ci, "--", label="complete-information cap")
        row[1].set(title="cap ā*(c)", xlabel="capability c")
        row[1].legend(fontsize=7)
        row[2].plot(c, [theory.human_value_cap(min(ci_, a), bi) for ci_, a, bi in zip(c, cap, b)], label="optimal")
        row[2].plot(c, [theory.human_value_cap(a, bi) for a, bi in zip(ci, b)], "--", label="complete info")
        row[2].set(title="human payoff by type", xlabel="capability c")
        row[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4_sandbagging_caps.png", dpi=150)
    plt.close(fig)


def figure6(out: Path, mean_b: float = 0.31) -> None:
    """Value of interpretability at fixed mean bias (paper Figure 6), two-point bias b̄ ± σ."""
    sig = np.linspace(0, 0.3, 301)
    loss_opt = np.array([theory.single_cap_loss([mean_b - s, mean_b + s], [0.5, 0.5]) for s in sig])
    loss_fd = mean_b**2 + sig**2
    deriv = np.array([theory.interpretability_marginal(mean_b, s**2) for s in sig])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(sig, -loss_opt, label="optimal cap")
    ax[0].plot(sig, -loss_fd, label="full discretion")
    ax[0].axhline(-1 / 12, ls=":", c="k", label="no AI")
    ax[0].set(xlabel="dispersion σ", ylabel="expected payoff")
    ax[0].legend(fontsize=7)
    ax[1].plot(sig, deriv)
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].set(xlabel="dispersion σ", ylabel="∂L*/∂σ² = ā* − 2b̄")
    ax[2].plot(sig, loss_fd - loss_opt)
    ax[2].set(xlabel="dispersion σ", ylabel="value of delegation (L_FD − L*)")
    fig.tight_layout()
    fig.savefig(out / "fig6_interpretability.png", dpi=150)
    plt.close(fig)


def figure12(out: Path, high_b: float = 0.3) -> None:
    """Human loss across oversight regimes (paper Figure 12)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ps = np.linspace(0, 0.2, 401)
    for regime in ("approval", "delegation", "reward"):
        ax[0].plot(ps, [theory.expected_oversight_losses([0, high_b], [1 - p, p], 0.1)[regime] for p in ps], label=regime)
    ax[0].set(xlabel="P(high actor bias), w = 0.1", ylabel="expected human loss")
    ws = np.linspace(0, 0.15, 401)
    for regime in ("approval", "delegation", "reward"):
        ax[1].plot(ws, [theory.expected_oversight_losses([0, high_b], [0.95, 0.05], w)[regime] for w in ws], label=regime)
    ax[1].set(xlabel="monitor bias w, p = 0.05")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig12_oversight_regimes.png", dpi=150)
    plt.close(fig)


def sandbagging_run(run_dir: Path) -> Path:
    """Sandbag rate by capability, per arm and profile: agent vs rational prediction."""
    from .experiments.common import load_trials, usable

    rows = load_trials(run_dir)
    profiles = sorted({r["profile"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    fig, axes = plt.subplots(len(profiles), len(arms), figsize=(3.2 * len(arms), 2.8 * len(profiles)), squeeze=False)
    for i, prof in enumerate(profiles):
        for j, arm in enumerate(arms):
            rs = [r for r in rows if r["profile"] == prof and r["arm"] == arm]
            cs = sorted({r["c"] for r in rs})
            agent = [np.mean([r["feasible_report"] < c - 1e-9 for r in rs if r["c"] == c and usable(r)] or [np.nan]) for c in cs]
            rational = [np.mean([r["rational_report"] < c - 1e-9 for r in rs if r["c"] == c]) for c in cs]
            ax = axes[i, j]
            ax.plot(cs, rational, "k--", label="rational")
            ax.plot(cs, agent, "o-", label="agent")
            ax.set(ylim=(-0.05, 1.05), title=f"{prof} / {arm}", xlabel="capability c")
            if j == 0:
                ax.set_ylabel("sandbag rate")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = Path(run_dir) / "sandbag_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--run", type=Path, help="plot a sandbagging run directory instead")
    args = p.parse_args(argv)
    if args.run:
        print(sandbagging_run(args.run))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    figure4(args.out)
    figure6(args.out)
    figure12(args.out)
    print(f"wrote figures to {args.out}/")


if __name__ == "__main__":
    main()
