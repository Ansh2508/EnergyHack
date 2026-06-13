"""Slice 3 tests - the deterministic investigation agent (LangGraph).

Each test names the research principle it proves. Euros are range-tolerant
(sibling_sigma here, causalimpact on a TF machine) - never hard-coded.
"""

import json

import pytest

import src.agent as agent
from src.agent import TraceEvent, route_after_detect, route_after_triage, run_investigation

HERO = "INV 01.05.029"
WS, WE = "2019-05-24", "2019-06-16"
HEALTHY_SIBLING = "INV 01.05.030"  # verified: 0 sustained-zero days over the May window
NODE_ORDER = ["observe", "triage", "diagnose", "quantify", "act"]


@pytest.fixture(scope="session")
def hero():
    """Run the hero investigation once for the content-asserting tests."""
    state = run_investigation(HERO, WS, WE)
    with open("outputs/agent_run.json", encoding="utf-8") as fh:
        run = json.load(fh)
    return {"state": state, "run": run}


def test_agent_hero_full_investigation(hero):
    """Real-data E2E (acts, not answers): a ticket-validated work order is issued."""
    state, run = hero["state"], hero["run"]
    assert state["status"] == "work_order_issued"
    assert state["verdict"].primary_cause == "DEAD_INVERTER"
    assert state["verdict"].side == "AC"
    loss = state["loss"]
    assert loss.euros_lost > 0
    assert loss.euros_ci_low <= loss.euros_lost <= loss.euros_ci_high
    assert loss.method in {"causalimpact", "sibling_sigma"}
    assert run["result"]["validation"]["ticket_matched"] is True


def test_agent_healthy_short_circuits():
    """SkillRL early-terminate: a healthy inverter stops at observe (no diagnose/quantify)."""
    state = run_investigation(HEALTHY_SIBLING, WS, WE)
    assert state["status"] == "healthy"
    assert state["loss"] is None
    nodes = [e["node"] for e in state["trace"]]
    assert "diagnose" not in nodes and "quantify" not in nodes


def test_agent_curtailment_short_circuits(monkeypatch):
    """SkillRL lesson-from-failure: the curtailment guard terminates before diagnosis."""
    monkeypatch.setattr(agent, "_is_curtailment_dominated", lambda *a, **k: True)
    state = run_investigation(HERO, WS, WE)
    assert state["status"] == "not_a_fault_curtailment"
    assert state["loss"] is None
    assert any(e["status"] == "reject" for e in state["trace"])
    nodes = [e["node"] for e in state["trace"]]
    assert "diagnose" not in nodes and "quantify" not in nodes


def test_routing_reads_typed_fields():
    """Roy 2024: routing reads typed structure (counts / booleans), not prose."""
    assert route_after_detect({"detection": {"sustained_zero_days": 10}}) == "fault"
    assert route_after_detect({"detection": {"sustained_zero_days": 0}}) == "healthy"
    assert route_after_detect({"detection": None}) == "healthy"
    assert route_after_triage({"detection": {"curtailed": True}}) == "curtailment"
    assert route_after_triage({"detection": {"curtailed": False}}) == "real_fault"


def test_agent_trace_is_ordered_and_typed(hero):
    """Typed observability: every event validates against TraceEvent; steps contiguous."""
    state, run = hero["state"], hero["run"]
    events = state["trace"]
    for e in events:
        TraceEvent(**e)  # raises if any field is the wrong type
    assert [e["step"] for e in events] == list(range(1, len(events) + 1))
    nodes = [e["node"] for e in events]
    for expected in NODE_ORDER:
        assert expected in nodes
    seq = [NODE_ORDER.index(n) for n in nodes if n in NODE_ORDER]
    assert seq == sorted(seq)  # observe -> triage -> diagnose -> quantify -> act
    assert any(e["status"] == "fault" for e in events)
    assert any(e["status"] == "ok" for e in events)
    assert "trace" in run and "result" in run


def test_trace_matches_result(hero):
    """Validate-before-show / faithfulness: the trace equals the work order."""
    run = hero["run"]
    by = {e["node"]: e for e in run["trace"]}
    diag, res = by["diagnose"]["evidence"], run["result"]["evidence"]
    for k in ("u_dc_v", "i_dc_a", "u_dc_healthy_v"):
        assert diag[k] == res[k]
    assert by["quantify"]["evidence"]["euros_lost"] == run["result"]["loss"]["euros_lost"]
    # single-source physics: dead-run DC voltage above the healthy baseline, ~zero current
    assert res["u_dc_v"] > res["u_dc_healthy_v"]
    assert res["i_dc_a"] < 1.0


def test_quantify_runs_once(monkeypatch):
    """Single source of truth: quantify.quantify_loss is called exactly once per run."""
    import src.quantify as q

    orig = q.quantify_loss
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(q, "quantify_loss", counting)
    run_investigation(HERO, WS, WE)
    assert calls["n"] == 1
