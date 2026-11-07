"""Agents that answer tasks.

RationalBackend   the paper's benchmark: an exact (optionally noisy) best responder.
AnthropicBackend  a Claude model reached through the Anthropic API.

Both implement `decide(task, sample_idx) -> Decision`; `decide_many` runs a batch with a
thread pool and keeps the input order.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import prompts, theory
from .tasks import ActionTask, Decision, ReportTask, Task, project


class Backend:
    name = "backend"

    def decide(self, task: Task, sample_idx: int = 0) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def decide_many(self, jobs: Sequence[tuple[Task, int]], workers: int = 1) -> list[Decision]:
        if workers <= 1:
            return [self.decide(t, i) for t, i in jobs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self.decide(*job), jobs))

    def describe(self) -> dict:
        return {"backend": self.name}


class RationalBackend(Backend):
    """Exact best responder. `noise` adds N(0, noise^2) to actions; reports stay exact."""

    name = "rational"

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = noise
        self.seed = seed

    def decide(self, task: Task, sample_idx: int = 0) -> Decision:
        if isinstance(task, ActionTask):
            value = theory.capped_action(task.theta, task.b, task.limit)
            if self.noise:
                # Seed from the task so results are reproducible regardless of thread order.
                key = f"{self.seed}|{task.theta}|{task.b}|{task.limit}|{sample_idx}"
                rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
                value = min(value + rng.normal(0, self.noise), task.limit)
        else:
            value, _ = theory.rational_report(task.c, task.grid, task.schedule)
        feasible, infeasible = project(task, value)
        return Decision(value=value, feasible_value=feasible, infeasible=infeasible, rationale="best response")

    def describe(self) -> dict:
        return {"backend": self.name, "noise": self.noise, "seed": self.seed}


class AnthropicBackend(Backend):
    """Claude via the Messages API with JSON-schema structured output.

    Responses are cached on disk keyed by the full request plus `sample_idx`, so re-running
    an experiment never pays twice for the same call. Server-side refusal fallback
    (`fallbacks: "default"`) is on by default; trials served by a fallback model are
    flagged in `meta["fallback_ran"]` so they can be excluded from analysis.
    """

    name = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        fallback: bool = True,
        max_tokens: int = 16000,
        cache_dir: Optional[Path] = Path("results/.cache"),
    ):
        import anthropic  # imported lazily so the rational backend works without credentials

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.fallback = fallback
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "effort": self.effort, "fallback": self.fallback}

    def _request(self, system: str, user: str) -> dict:
        req = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": prompts.SCHEMA}},
        }
        if self.fallback:
            req["betas"] = [self.FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    def decide(self, task: Task, sample_idx: int = 0) -> Decision:
        system, user = prompts.render(task)
        req = self._request(system, user)
        key = hashlib.sha256(json.dumps([req, sample_idx], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            out = json.loads(path.read_text())
            out["meta"]["cached"] = True
        else:
            out = self._call(req)
            if path and out["meta"].get("error") is None:
                path.write_text(json.dumps(out))
        value = out["value"]
        feasible, infeasible = project(task, value)
        return Decision(value, feasible, infeasible, out["rationale"], out["meta"])

    def _call(self, req: dict) -> dict:
        a = self._anthropic
        meta: dict = {"model": self.model, "cached": False, "error": None}
        try:
            resp = self.client.beta.messages.create(**req)
        except a.BadRequestError as e:  # malformed request: a harness bug, surface it
            raise RuntimeError(f"Bad request to {self.model}: {e}") from e
        except (a.RateLimitError, a.APIStatusError, a.APIConnectionError) as e:
            # SDK already retried; record the failure rather than kill the whole run
            meta["error"] = f"{type(e).__name__}: {e}"
            return {"value": None, "rationale": "", "meta": meta}

        meta["stop_reason"] = resp.stop_reason
        meta["served_model"] = resp.model
        meta["fallback_ran"] = any(
            getattr(it, "type", None) == "fallback_message" for it in (getattr(resp.usage, "iterations", None) or [])
        )
        meta["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        if resp.stop_reason == "refusal":
            meta["refusal"] = True
            return {"value": None, "rationale": "", "meta": meta}
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            data = json.loads(text)
            return {"value": float(data["value"]), "rationale": str(data.get("rationale", "")), "meta": meta}
        except (TypeError, ValueError, KeyError) as e:
            meta["error"] = f"parse: {e}; text={text!r}"
            return {"value": None, "rationale": "", "meta": meta}


def make_backend(kind: str, **kw) -> Backend:
    if kind == "rational":
        return RationalBackend(noise=kw.get("noise", 0.0), seed=kw.get("seed", 0))
    if kind == "anthropic":
        return AnthropicBackend(model=kw["model"], effort=kw["effort"], fallback=kw["fallback"])
    raise ValueError(f"unknown backend {kind!r}")
