"""Agents that answer tasks.

RationalBackend   the paper's benchmark: an exact (optionally noisy) best responder.
AnthropicBackend  a Claude model reached through the Anthropic API.
OpenAIBackend     an OpenAI model reached through the Responses API.

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


class APIBackend(Backend):
    """Shared logic for hosted models: prompt rendering, on-disk response cache, projection.

    Responses are cached keyed by the full request plus `sample_idx`, so re-running an
    experiment never pays twice for the same call. Failed calls are not cached.
    Subclasses implement `_request(system, user) -> dict` and `_call(request) -> dict`
    returning {"value", "rationale", "meta"}.
    """

    def __init__(self, model: str, effort: str, cache_dir: Optional[Path] = Path("results/.cache")):
        self.model = model
        self.effort = effort
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "effort": self.effort}

    def decide(self, task: Task, sample_idx: int = 0) -> Decision:
        system, user = prompts.render(task)
        req = self._request(system, user)
        key = hashlib.sha256(json.dumps([self.name, req, sample_idx], sort_keys=True).encode()).hexdigest()
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

    @staticmethod
    def _parse(text: Optional[str], meta: dict) -> dict:
        try:
            data = json.loads(text)
            return {"value": float(data["value"]), "rationale": str(data.get("rationale", "")), "meta": meta}
        except (TypeError, ValueError, KeyError) as e:
            meta["error"] = f"parse: {e}; text={text!r}"
            return {"value": None, "rationale": "", "meta": meta}

    def _request(self, system: str, user: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def _call(self, req: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicBackend(APIBackend):
    """Claude via the Messages API with JSON-schema structured output.

    Server-side refusal fallback (`fallbacks: "default"`) is on by default; trials served
    by a fallback model are flagged in `meta["fallback_ran"]` so they can be excluded.
    """

    name = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", fallback: bool = True,
                 max_tokens: int = 16000, cache_dir: Optional[Path] = Path("results/.cache")):
        import anthropic  # imported lazily so other backends work without this SDK configured

        super().__init__(model, effort, cache_dir)
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.fallback = fallback
        self.max_tokens = max_tokens

    def describe(self) -> dict:
        return {**super().describe(), "fallback": self.fallback}

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
        return self._parse(next((b.text for b in resp.content if b.type == "text"), None), meta)


class OpenAIBackend(APIBackend):
    """OpenAI models via the Responses API with strict JSON-schema output.

    `effort` is sent as `reasoning.effort` (none/minimal/low/medium/high/xhigh/max,
    depending on the model); pass `effort=None` for models without reasoning.
    The key is read from OPENAI_API_KEY by the SDK.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-5.5", effort: Optional[str] = "medium",
                 max_output_tokens: int = 16000, cache_dir: Optional[Path] = Path("results/.cache")):
        import openai

        super().__init__(model, effort, cache_dir)
        self._openai = openai
        self.client = openai.OpenAI()
        self.max_output_tokens = max_output_tokens

    def _request(self, system: str, user: str) -> dict:
        req = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.max_output_tokens,
            "text": {"format": {"type": "json_schema", "name": "decision", "schema": prompts.SCHEMA, "strict": True}},
        }
        if self.effort:
            req["reasoning"] = {"effort": self.effort}
        return req

    def _call(self, req: dict) -> dict:
        o = self._openai
        meta: dict = {"model": self.model, "cached": False, "error": None}
        try:
            resp = self.client.responses.create(**req)
        except (o.BadRequestError, o.AuthenticationError, o.NotFoundError, o.PermissionDeniedError) as e:
            # bad request, bad key or unknown model: fix before spending more
            raise RuntimeError(f"{type(e).__name__} from {self.model}: {e}") from e
        except (o.RateLimitError, o.APIStatusError, o.APIConnectionError) as e:
            meta["error"] = f"{type(e).__name__}: {e}"
            return {"value": None, "rationale": "", "meta": meta}

        meta["stop_reason"] = resp.status
        meta["served_model"] = resp.model
        if resp.usage is not None:
            meta["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        parts = [p for item in resp.output if getattr(item, "type", None) == "message" for p in item.content]
        if any(getattr(p, "type", None) == "refusal" for p in parts):
            meta["refusal"] = True
            return {"value": None, "rationale": "", "meta": meta}
        if resp.status == "incomplete":
            meta["error"] = f"incomplete: {resp.incomplete_details}"
            return {"value": None, "rationale": "", "meta": meta}
        return self._parse(resp.output_text, meta)


def make_backend(kind: str, **kw) -> Backend:
    if kind == "rational":
        return RationalBackend(noise=kw.get("noise", 0.0), seed=kw.get("seed", 0))
    if kind == "anthropic":
        return AnthropicBackend(model=kw.get("model") or "claude-opus-5-5", effort=kw["effort"], fallback=kw["fallback"])
    if kind == "openai":
        effort = None if kw["effort"] == "off" else kw["effort"]
        return OpenAIBackend(model=kw.get("model") or "gpt-5.5", effort=effort)
    raise ValueError(f"unknown backend {kind!r}")
