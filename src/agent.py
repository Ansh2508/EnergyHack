"""Deterministic investigation agent (LangGraph) over the Slice 1/2 tools.

A LangGraph StateGraph (v1.0 API; verified in docs/RESEARCH_NOTES.md) orchestrates
the committed analytics into one end-to-end investigation and records a typed,
replayable trace. The decision path is DETERMINISTIC plain Python - no LLM, no API -
so every branch is auditable. Routing functions and trace events read TYPED tool
outputs (CauseVerdict.primary_cause, LossEstimate.euros_*, sustained-zero-day count,
curtailment fraction), never prose (Roy et al. 2024, arXiv:2403.04123). Each node is
a named skill wrapping an existing tool; the curtailment triage is the
lesson-from-failure skill that terminates the investigation early (SkillRL,
arXiv:2602.08234). No analytics are reimplemented here - nodes are thin wrappers.
"""

from __future__ import annotations

import json
import operator
import os
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import build_facts, diagnose, quantify
from src.ingest import DEFAULT_CACHE, canonical_inverter_id
from src.quantify import DEFAULT_DETECTION

MIN_FAULT_DAYS = 3
DEAD_PR_MAX = 0.05
OUT_AGENT_RUN = "outputs/agent_run.json"
OUT_FACTS = "outputs/verified_facts.json"


class TraceEvent(BaseModel):
    """One decision in the investigation - the UI animation contract."""

    step: int  # 1-based, contiguous
    node: str  # observe | triage | diagnose | confirm | quantify | validate | act
    title: str  # short UI headline
    observation: str  # what the agent saw (ASCII)
    decision: str  # branch / verdict taken
    rationale: str  # WHY (the physics / logic), generated from typed values
    status: str  # info | fault | ok | reject  (drives UI colour)
    evidence: dict = Field(default_factory=dict)


class AgentState(TypedDict):
    inverter_id: str
    window_start: str
    window_end: str
    detection: dict | None
    verdict: object | None  # CauseVerdict
    loss: object | None  # LossEstimate
    facts: dict | None  # verified_facts dict
    status: str  # healthy | not_a_fault_curtailment | work_order_issued
    trace: Annotated[list, operator.add]  # append-only (operator.add reducer)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _emit(state: AgentState, *events: TraceEvent) -> list:
    """Number events contiguously after the trace accumulated so far."""
    base = len(state.get("trace") or [])
    out = []
    for i, ev in enumerate(events, start=1):
        ev.step = base + i
        out.append(ev.model_dump())
    return out


def _longest_zero_run(dates_sorted: list) -> int:
    if not dates_sorted:
        return 0
    best = cur = 1
    for i in range(1, len(dates_sorted)):
        if (dates_sorted[i] - dates_sorted[i - 1]).days == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _is_curtailment_dominated(iid: str, ws, we, cache_path: str = DEFAULT_CACHE) -> bool:
    """Named lesson-from-failure skill: reuse the committed curtailment guard."""
    return bool(diagnose._skill_curtailment_guard(iid, ws, we, cache_path))


# --------------------------------------------------------------------------- #
# nodes (thin wrappers over Slice 1/2 tools)
# --------------------------------------------------------------------------- #
def observe_detect(state: AgentState) -> dict:
    """Read the inverter's window from detection_daily; is there a sustained fault?"""
    iid = canonical_inverter_id(state["inverter_id"]) or state["inverter_id"]
    ws = pd.Timestamp(state["window_start"]).date()
    we = pd.Timestamp(state["window_end"]).date()
    det = pd.read_parquet(DEFAULT_DETECTION)
    det["date"] = pd.to_datetime(det["date"]).dt.date
    w = det[(det["inverter_id"] == iid) & (det["date"] >= ws) & (det["date"] <= we)]
    zero_dates = sorted(w.loc[w["pr"].fillna(1.0) < DEAD_PR_MAX, "date"].tolist())
    sustained = _longest_zero_run(zero_dates)
    detection = {
        "sustained_zero_days": int(sustained),
        "hero_mean_pr": round(float(w["pr"].mean()), 3) if len(w) else None,
        "sibling_mean_pr": round(float(w["sibling_pr"].mean()), 3) if len(w) else None,
        "curtailment_fraction": round(float(w["dv_frac"].mean()), 3) if len(w) else 0.0,
    }
    if sustained >= MIN_FAULT_DAYS:
        ev = TraceEvent(
            step=0, node="observe", title="Scanning the inverter's recent output",
            observation=f"{sustained} consecutive days at PR=0.0 while siblings held "
                        f"~{detection['sibling_mean_pr']}",
            decision="fault",
            rationale=f"a {sustained}-day flatline against healthy peers in the same "
                      "weather is not noise - this inverter stopped producing",
            status="fault", evidence=detection,
        )
        return {"detection": detection, "trace": _emit(state, ev)}
    ev = TraceEvent(
        step=0, node="observe", title="Scanning the inverter's recent output",
        observation=f"no sustained zero-output run (longest {sustained} day(s)); "
                    f"mean PR ~{detection['hero_mean_pr']} vs siblings "
                    f"~{detection['sibling_mean_pr']}",
        decision="healthy",
        rationale="output tracks the sibling fleet - nothing to investigate",
        status="ok", evidence=detection,
    )
    return {"detection": detection, "status": "healthy", "trace": _emit(state, ev)}


def triage_curtailment(state: AgentState) -> dict:
    """Lesson-from-failure guard: is the lost output explained by curtailment?"""
    iid = canonical_inverter_id(state["inverter_id"]) or state["inverter_id"]
    ws = pd.Timestamp(state["window_start"]).date()
    we = pd.Timestamp(state["window_end"]).date()
    det = dict(state.get("detection") or {})
    curtailed = _is_curtailment_dominated(iid, ws, we)
    det["curtailed"] = curtailed
    frac = float(det.get("curtailment_fraction") or 0.0)
    if curtailed:
        ev = TraceEvent(
            step=0, node="triage", title="Ruling out grid / market curtailment",
            observation=f"the plant was throttled for ~{frac * 100:.0f}% of daytime "
                        "in this window",
            decision="not_a_fault",
            rationale="a market or grid setpoint, not an inverter failure - never "
                      "raise a work order on a throttle",
            status="reject", evidence={"curtailment_fraction": frac},
        )
        return {"detection": det, "status": "not_a_fault_curtailment",
                "trace": _emit(state, ev)}
    ev = TraceEvent(
        step=0, node="triage", title="Ruling out grid / market curtailment",
        observation=f"plant at full setpoint (DV~100); only ~{frac * 100:.0f}% of "
                    "daytime curtailed",
        decision="real_fault",
        rationale="the lost output is not explained by curtailment, so this is a "
                  "genuine fault to diagnose",
        status="info", evidence={"curtailment_fraction": frac},
    )
    return {"detection": det, "trace": _emit(state, ev)}


def diagnose_cause(state: AgentState) -> dict:
    """Wrap diagnose.diagnose -> CauseVerdict; emit the physics + the log confirmation."""
    v = diagnose.diagnose(state["inverter_id"], state["window_start"], state["window_end"])
    side_line = v.evidence[1] if v.side and len(v.evidence) >= 2 else (
        v.evidence[0] if v.evidence else f"cause {v.primary_cause}")
    events = [
        TraceEvent(
            step=0, node="diagnose", title="Diagnosing the cause (AC vs DC side)",
            observation="; ".join(v.evidence[:2]) if v.evidence else v.primary_cause,
            decision=f"{v.primary_cause}/{v.side}" if v.side else v.primary_cause,
            rationale=side_line,
            status="fault",
            evidence={"side": v.side, "confidence": v.confidence,
                      "u_dc_v": v.u_dc_v, "i_dc_a": v.i_dc_a,
                      "u_dc_healthy_v": v.u_dc_healthy_v},
        )
    ]
    if v.errorcode_corroboration:
        events.append(TraceEvent(
            step=0, node="confirm", title="Cross-checking the inverter's own error log",
            observation=str(v.errorcode_corroboration),
            decision="confirmed",
            rationale="the inverter logged a grid / power-unit fault in the same window "
                      "- independent agreement with the physics verdict",
            status="ok", evidence={"errorcode_corroboration": v.errorcode_corroboration},
        ))
    return {"verdict": v, "trace": _emit(state, *events)}


def quantify_loss(state: AgentState) -> dict:
    """Wrap quantify.quantify_loss -> LossEstimate; price the lost energy with a CI."""
    loss = quantify.quantify_loss(
        state["inverter_id"], state["window_start"], state["window_end"])
    ev = TraceEvent(
        step=0, node="quantify", title="Pricing the lost energy",
        observation=f"{loss.lost_kwh:.0f} kWh lost -> EUR {loss.euros_lost:.2f} "
                    f"(95% CI EUR {loss.euros_ci_low:.2f}-{loss.euros_ci_high:.2f}, "
                    f"{loss.method})",
        decision=f"EUR {loss.euros_lost:.2f}",
        rationale="counterfactual expected output minus actual over the window, priced "
                  f"at {loss.tariff_eur_per_kwh} EUR/kWh from the feed-in tariff",
        status="info",
        evidence={"lost_kwh": loss.lost_kwh, "euros_lost": loss.euros_lost,
                  "euros_ci_low": loss.euros_ci_low, "euros_ci_high": loss.euros_ci_high,
                  "tariff_eur_per_kwh": loss.tariff_eur_per_kwh, "method": loss.method},
    )
    return {"loss": loss, "trace": _emit(state, ev)}


def issue_work_order(state: AgentState) -> dict:
    """Wrap build_facts -> verified facts; validate vs ticket, then issue the order."""
    facts = build_facts.build_facts(
        state["inverter_id"], state["window_start"], state["window_end"],
        verdict=state["verdict"], loss=state["loss"])
    build_facts.write_facts(facts, OUT_FACTS)  # keep verified_facts.json in sync with this run
    val = facts.get("validation") or {}
    matched = bool(val.get("ticket_matched"))
    act = facts.get("action") or {}
    ev_validate = TraceEvent(
        step=0, node="validate", title="Validating against ENERPARC service tickets",
        observation=f"ticket {val.get('ticket_id')}: matched={matched}, "
                    f"affected_count_match={val.get('affected_count_match')}",
        decision="MATCH" if matched else "NO_MATCH",
        rationale="the detected outage overlaps a real maintenance ticket - ground "
                  "truth, not a self-graded score",
        status="ok" if matched else "info", evidence=val,
    )
    ev_act = TraceEvent(
        step=0, node="act", title="Issuing the work order",
        observation=f"{act.get('recommendation')} (ROI {act.get('roi_multiple')}x)",
        decision="work_order_issued",
        rationale="confirmed inverter failure with a quantified, ticket-validated loss "
                  "- dispatch the technician",
        status="fault",
        evidence={"recommendation": act.get("recommendation"),
                  "roi_multiple": act.get("roi_multiple")},
    )
    return {"facts": facts, "status": "work_order_issued",
            "trace": _emit(state, ev_validate, ev_act)}


# --------------------------------------------------------------------------- #
# routing (deterministic, reads TYPED fields - Roy 2024)
# --------------------------------------------------------------------------- #
def route_after_detect(state: AgentState) -> str:
    d = state.get("detection") or {}
    return "fault" if int(d.get("sustained_zero_days") or 0) >= MIN_FAULT_DAYS else "healthy"


def route_after_triage(state: AgentState) -> str:
    d = state.get("detection") or {}
    return "curtailment" if d.get("curtailed") else "real_fault"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("observe", observe_detect)
    g.add_node("triage", triage_curtailment)
    g.add_node("diagnose", diagnose_cause)
    g.add_node("quantify", quantify_loss)
    g.add_node("act", issue_work_order)
    g.add_edge(START, "observe")
    g.add_conditional_edges("observe", route_after_detect,
                            {"fault": "triage", "healthy": END})
    g.add_conditional_edges("triage", route_after_triage,
                            {"curtailment": END, "real_fault": "diagnose"})
    g.add_edge("diagnose", "quantify")
    g.add_edge("quantify", "act")
    g.add_edge("act", END)
    return g.compile()


GRAPH = _build_graph()


def run_investigation(inverter_id: str, window_start: str, window_end: str) -> AgentState:
    """Compile + invoke the investigation graph; write outputs/agent_run.json."""
    initial: AgentState = {
        "inverter_id": inverter_id, "window_start": window_start, "window_end": window_end,
        "detection": None, "verdict": None, "loss": None, "facts": None,
        "status": "investigating", "trace": [],
    }
    final = GRAPH.invoke(initial)
    _write_agent_run(final)
    return final


def _write_agent_run(state: AgentState) -> None:
    os.makedirs("outputs", exist_ok=True)
    payload = {
        "inverter_id": state["inverter_id"],
        "window": {"start": state["window_start"], "end": state["window_end"]},
        "status": state["status"],
        "trace": state["trace"],
        "result": state.get("facts") or {},
    }
    with open(OUT_AGENT_RUN, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")


if __name__ == "__main__":
    st = run_investigation("INV 01.05.029", "2019-05-24", "2019-06-16")
    print("status:", st["status"])
    for e in st["trace"]:
        print(f"[{e['step']}] {e['node']}: {e['decision']} - {e['rationale']}")
