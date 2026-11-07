"""Experiment 1 — is the agent an optimiser of u_b + r at all?

Every result in the paper assumes the agent best-responds to its payoff plus the
mechanism's reward. This experiment checks that premise before anything else:

  * free play (no cap):   implied bias  b̂ = mean(a - θ)  should equal the induced b
  * capped play:          a should equal min(θ + b, cap)

Usage:
    python -m mdx.experiments.rationality --backend rational --noise 0.02
    python -m mdx.experiments.rationality --backend anthropic --n-states 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from ..tasks import ActionTask
from .common import Run, add_backend_args, build_backend, confirm_spend, floats, usable


def parse_caps(s: str) -> list[float | None]:
    return [None if x.strip().lower() == "none" else float(x) for x in s.split(",")]


def build_jobs(args: argparse.Namespace) -> list[tuple[ActionTask, int]]:
    rng = np.random.default_rng(args.seed)
    thetas = np.round(rng.uniform(0, 1, args.n_states), 4)  # common across conditions
    jobs = []
    for b in floats(args.biases):
        for cap in parse_caps(args.caps):
            for theta in thetas:
                for s in range(args.samples):
                    jobs.append((ActionTask(theta=float(theta), b=b, cap=cap, framing=args.framing,
                                            explicit_objective=args.explicit_objective), s))
    return jobs


def summarise(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["b"], r["cap"])].append(r)
    cells = []
    for (b, cap), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 9)):
        ok = [r for r in rs if usable(r)]
        if not ok:
            cells.append({"b": b, "cap": cap, "n": len(rs), "n_usable": 0})
            continue
        a = np.array([r["feasible_value"] for r in ok])
        pred = np.array([r["predicted"] for r in ok])
        th = np.array([r["theta"] for r in ok])
        u = -((a - th - b) ** 2)
        u_star = -((pred - th - b) ** 2)
        cells.append({
            "b": b, "cap": cap, "n": len(rs), "n_usable": len(ok),
            "mae_vs_best_response": float(np.mean(np.abs(a - pred))),
            "share_within_0.01": float(np.mean(np.abs(a - pred) <= 0.01)),
            "implied_bias_mean(a-θ)": float(np.mean(a - th)),
            "infeasible_rate": float(np.mean([r["infeasible"] for r in ok])),
            "regret": float(np.mean(u_star - u)),
        })
    free = [c for c in cells if c["cap"] is None and c.get("n_usable")]
    out: dict = {"cells": cells}
    if len(free) >= 2:
        true_b = np.array([c["b"] for c in free])
        est_b = np.array([c["implied_bias_mean(a-θ)"] for c in free])
        slope, intercept = np.polyfit(true_b, est_b, 1)
        out["bias_recovery"] = {"slope": float(slope), "intercept": float(intercept),
                                "note": "slope 1, intercept 0 = the agent acts on exactly the induced bias"}
    errs = [r for r in rows if r.get("meta", {}).get("error") or r.get("meta", {}).get("refusal")]
    out["n_errors_or_refusals"] = len(errs)
    out["n_fallback"] = sum(bool(r.get("meta", {}).get("fallback_ran")) for r in rows)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--biases", default="0,0.1,0.2,0.3")
    p.add_argument("--caps", default="none,0.5,0.7,0.9")
    p.add_argument("--n-states", type=int, default=10)
    p.add_argument("--samples", type=int, default=1, help="repeated samples per (b, cap, θ)")
    add_backend_args(p)
    args = p.parse_args(argv)

    backend = build_backend(args)
    jobs = build_jobs(args)
    confirm_spend(args, len(jobs))
    run = Run(args.out, "rationality", {"args": vars(args), "backend": backend.describe()})

    decisions = backend.decide_many(jobs, workers=args.workers)
    rows = []
    for (task, s), d in zip(jobs, decisions):
        rows.append({**task.to_dict(), "sample": s, "limit": task.limit,
                     "predicted": min(task.theta + task.b, task.limit),
                     "value": d.value, "feasible_value": d.feasible_value, "infeasible": d.infeasible,
                     "rationale": d.rationale, "meta": d.meta})
    run.log(rows)
    summary = summarise(rows)
    path = run.finish(summary)

    print(f"\n{'b':>5} {'cap':>5} {'n':>4} {'MAE':>7} {'≤0.01':>6} {'mean(a-θ)':>10} {'regret':>8}")
    for c in summary["cells"]:
        if not c.get("n_usable"):
            print(f"{c['b']:>5} {str(c['cap']):>5} {c['n']:>4}   (no usable trials)")
            continue
        print(f"{c['b']:>5} {str(c['cap']):>5} {c['n_usable']:>4} {c['mae_vs_best_response']:>7.4f} "
              f"{c['share_within_0.01']:>6.2f} {c['implied_bias_mean(a-θ)']:>10.4f} {c['regret']:>8.5f}")
    if "bias_recovery" in summary:
        br = summary["bias_recovery"]
        print(f"\nbias recovery (free play): slope={br['slope']:.3f} intercept={br['intercept']:.3f}")
    print(f"errors/refusals: {summary['n_errors_or_refusals']}, fallback-served: {summary['n_fallback']}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
