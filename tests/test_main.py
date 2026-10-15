"""Tests for `python main.py all`: order, resuming, skipping step 5, stopping at a failure."""
from __future__ import annotations

import pytest

import main
from aml.config import Config


@pytest.fixture
def ran(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "STEPS", {name: (lambda config, name=name: calls.append(name)) for name in main.STEPS})
    return calls


def test_runs_every_step_in_order(ran):
    main.run_all(Config())
    assert ran == ["clean", "features", "baseline", "graph-features", "graph-model", "gnn", "unsupervised",
                   "build-store", "monitor"]


def test_resume_and_skip_gnn(ran):
    main.run_all(Config(), start="graph-model", skip_gnn=True)
    assert ran == ["graph-model", "unsupervised", "build-store", "monitor"]


def test_stops_at_first_failure(ran, monkeypatch):
    def broken(config):
        raise RuntimeError("boom")

    monkeypatch.setitem(main.STEPS, "baseline", broken)
    with pytest.raises(RuntimeError, match="boom"):
        main.run_all(Config())
    assert ran == ["clean", "features"]
