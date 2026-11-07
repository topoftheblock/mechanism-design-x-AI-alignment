"""Shared CLI arguments, run directories and logging for experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..backends import Backend, make_backend


def add_backend_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("agent backend")
    g.add_argument("--backend", choices=["rational", "anthropic"], default="rational",
                   help="rational = exact best responder (free, offline); anthropic = Claude via the API")
    g.add_argument("--model", default="claude-opus-5-5", help="Claude model id (anthropic backend)")
    g.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--no-fallback", dest="fallback", action="store_false",
                   help="disable server-side refusal fallback to another model")
    g.add_argument("--noise", type=float, default=0.0, help="action noise s.d. (rational backend only)")
    g.add_argument("--framing", choices=["abstract", "ai_eval"], default="abstract")
    g.add_argument("--implicit-objective", dest="explicit_objective", action="store_false",
                   help="describe the payoff but do not instruct the agent to maximise it")
    g.add_argument("--workers", type=int, default=8, help="parallel API calls")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    g.add_argument("--out", type=Path, default=Path("results"))


def build_backend(args: argparse.Namespace) -> Backend:
    return make_backend(args.backend, model=args.model, effort=args.effort, fallback=args.fallback,
                        noise=args.noise, seed=args.seed)


def confirm_spend(args: argparse.Namespace, n_calls: int) -> None:
    """API calls cost money: show the count and ask before spending (cached calls are free)."""
    if args.backend == "rational":
        return
    print(f"About to make up to {n_calls} calls to {args.model} (effort={args.effort}). "
          "Previously cached calls are not re-billed.")
    if args.yes:
        return
    if input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")


def floats(s: str) -> list[float]:
    """Parse '0,0.1,0.2' or a range 'start:stop:step' (inclusive)."""
    if ":" in s:
        a, b, step = (float(x) for x in s.split(":"))
        return [round(x, 10) for x in np.arange(a, b + step / 2, step)]
    return [float(x) for x in s.split(",") if x]


class Run:
    """results/<experiment>/<timestamp>-<backend>/{config.json, trials.jsonl, summary.json}"""

    def __init__(self, out: Path, experiment: str, config: dict[str, Any]):
        tag = config.get("backend", {}).get("model") or config.get("backend", {}).get("backend", "run")
        self.dir = Path(out) / experiment / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self._trials = (self.dir / "trials.jsonl").open("w")

    def log(self, rows: Iterable[dict[str, Any]]) -> None:
        for r in rows:
            self._trials.write(json.dumps(r, default=_jsonable) + "\n")
        self._trials.flush()

    def finish(self, summary: dict[str, Any]) -> Path:
        self._trials.close()
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
        return self.dir


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def load_trials(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (Path(run_dir) / "trials.jsonl").read_text().splitlines() if line]


def usable(row: dict[str, Any]) -> bool:
    """A trial counts if it produced a value and was answered by the requested model."""
    meta = row.get("meta", {})
    return row.get("feasible_value") is not None and not meta.get("fallback_ran", False)
