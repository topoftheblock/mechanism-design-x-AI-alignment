"""End-to-end runs of both experiments with the rational backend (no API calls)."""

import json

import pytest

from mdx import prompts
from mdx.experiments import rationality, sandbagging
from mdx.experiments.common import load_trials
from mdx.tasks import ActionTask, ReportTask, project


def _latest(out, name):
    return max((out / name).iterdir())


def test_rationality_rational_backend(tmp_path):
    rationality.main(["--backend", "rational", "--n-states", "6", "--out", str(tmp_path), "--workers", "1"])
    summary = json.loads((_latest(tmp_path, "rationality") / "summary.json").read_text())
    assert all(c["mae_vs_best_response"] == pytest.approx(0) for c in summary["cells"])
    assert summary["bias_recovery"]["slope"] == pytest.approx(1)


def test_sandbagging_rational_backend_matches_theory(tmp_path):
    sandbagging.main(["--backend", "rational", "--report-samples", "1", "--out", str(tmp_path)])
    run = _latest(tmp_path, "sandbagging")
    summary = json.loads((run / "summary.json").read_text())
    cells = {(c["profile"], c["arm"]): c for c in summary["cells"]}
    for c in cells.values():
        assert c["agreement_with_rational"] == 1.0
        assert c["counterfeit_attempt_rate"] == 0.0
        assert c["sandbag_rate"] == c["rational_sandbag_rate"]
    for prof in ("increasing", "decreasing", "nonmonotone"):
        assert cells[(prof, "optimal")]["sandbag_rate"] == 0.0
        assert cells[(prof, "flat")]["sandbag_rate"] == 0.0
        assert cells[(prof, "inverted")]["sandbag_rate"] > 0.0
    # complete-information caps fall with c under increasing bias -> high types sandbag
    assert cells[("increasing", "complete_info")]["sandbag_rate"] > 0.0
    assert cells[("decreasing", "complete_info")]["sandbag_rate"] == 0.0
    # optimal truthful schedule is at least as good for the designer as any arm under rational play
    for prof in ("increasing", "decreasing", "nonmonotone"):
        best = max(cells[(prof, a)]["theory_value_rational_play"] for a in sandbagging.ARMS)
        assert cells[(prof, "optimal")]["theory_value_rational_play"] == pytest.approx(best, abs=1e-9)
    assert len(load_trials(run)) == 3 * 4 * 11


def test_projection_and_prompts():
    t = ActionTask(theta=0.4, b=0.2, c=0.8, cap=0.5)
    assert project(t, 0.7) == (0.5, True)
    assert project(t, 0.45) == (0.45, False)
    r = ReportTask(c=0.8, b=0.1, grid=(0.5, 0.6, 0.7, 0.8, 0.9), schedule=(0.5, 0.6, 0.7, 0.8, 0.9))
    assert project(r, 0.9) == (0.8, True)  # counterfeiting is impossible
    assert project(r, 0.6) == (0.6, False)
    for framing in ("abstract", "ai_eval"):
        for task in (ActionTask(theta=0.4, b=0.2, cap=0.5, framing=framing),
                     ReportTask(c=0.8, b=0.1, grid=(0.5, 0.8), schedule=(0.5, 0.7), framing=framing)):
            system, user = prompts.render(task)
            assert system and str(task.b) in user


class _FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **req):
        from types import SimpleNamespace as NS
        self.calls.append(req)
        return NS(stop_reason=self.stop_reason, model=req["model"],
                  usage=NS(input_tokens=10, output_tokens=5, iterations=None),
                  content=[NS(type="thinking", thinking=""), NS(type="text", text=self.text)])


def _fake_backend(tmp_path, monkeypatch, text, stop_reason="end_turn"):
    from types import SimpleNamespace as NS
    from mdx.backends import AnthropicBackend
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    be = AnthropicBackend(cache_dir=tmp_path / "cache")
    fake = _FakeMessages(text, stop_reason)
    be.client = NS(beta=NS(messages=fake))
    return be, fake


def test_anthropic_backend_request_parse_and_cache(tmp_path, monkeypatch):
    be, fake = _fake_backend(tmp_path, monkeypatch, '{"rationale": "cap binds", "value": 0.7}')
    task = ActionTask(theta=0.6, b=0.2, cap=0.7)
    d = be.decide(task)
    assert d.feasible_value == 0.7 and not d.infeasible and d.rationale == "cap binds"
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert req["output_config"]["format"]["type"] == "json_schema" and req["output_config"]["effort"] == "medium"
    assert "thinking" not in req
    d2 = be.decide(task)  # served from disk cache
    assert len(fake.calls) == 1 and d2.meta["cached"] and d2.feasible_value == 0.7
    be.decide(task, sample_idx=1)  # a new sample is a new call
    assert len(fake.calls) == 2


def test_anthropic_backend_refusal_recorded(tmp_path, monkeypatch):
    be, _ = _fake_backend(tmp_path, monkeypatch, "", stop_reason="refusal")
    d = be.decide(ActionTask(theta=0.6, b=0.2))
    assert d.value is None and d.meta["refusal"]
