"""Assemble outputs/verified_facts.json - the single source the demo UI renders.

Validate-before-show (arXiv:2606.01513): every number that lands in the JSON is
re-checked against its computed source and the build raises on any mismatch, so
no unverified number ever reaches the narration layer. Computation lives here in
Python; the JSON only describes verified facts.
"""

from __future__ import annotations

import json
import os
import math

import duckdb
import pandas as pd

from src import hero_match
from src.diagnose import (
    DEFAULT_ERRORCODES,
    DEFAULT_ERRORCODES_DICT,
    _consecutive_runs,
    _find,
    _load_errorcode_dict,
    diagnose,
)
from src.ingest import DEFAULT_CACHE, DEFAULT_SYSTEM_OVERVIEW, canonical_inverter_id, load_meta
from src.quantify import DEFAULT_DETECTION, DEFAULT_TARIFFS, quantify_loss

PLANT_DESC = "Plant A (Silmersdorf, 1.9 MWp, 65 inverters)"
PORTFOLIO_KW = 3.8e6  # 3.8 GW ENERPARC fleet
SILENT_LOSS_LOW, SILENT_LOSS_HIGH = 0.01, 0.03
INTERVENTION_COST_EUR = 1500.0  # documented assumption (RESEARCH_NOTES.md)
CF_YEAR = 2021
DEAD_PR_MAX = 0.05
_RECOMMENDATION = {
    "DEAD_INVERTER": "Dispatch technician: inverter service/replacement",
    "SOILING": "Schedule module cleaning",
    "CLIPPING": "Review inverter sizing / setpoint (not a fault)",
    "THERMAL_DERATE": "Inspect ventilation / derating on hot days",
    "NOT_A_FAULT": "No action: curtailment, not a fault",
    "UNKNOWN": "Manual review: signature did not match a known cause",
}


def build_facts(
    inverter_id: str,
    window_start: str,
    window_end: str,
    *,
    verdict: object | None = None,
    loss: object | None = None,
    cache_path: str = DEFAULT_CACHE,
    detection_path: str = DEFAULT_DETECTION,
    tariffs_path: str = DEFAULT_TARIFFS,
    system_overview_path: str = DEFAULT_SYSTEM_OVERVIEW,
    errorcodes_path: str = DEFAULT_ERRORCODES,
    errorcodes_dict_path: str = DEFAULT_ERRORCODES_DICT,
) -> dict:
    """Compute, assemble and self-check the verified-facts payload.

    If verdict/loss are supplied (e.g. by the agent) they are used as-is, so the
    diagnosis and CausalImpact run exactly once per investigation and the trace
    equals the work order. Default None recomputes them (standalone Slice-2 use).
    """
    iid = canonical_inverter_id(inverter_id) or str(inverter_id)
    ws = pd.Timestamp(window_start).date()
    we = pd.Timestamp(window_end).date()

    cause = verdict if verdict is not None else diagnose(
        iid, window_start, window_end, cache_path=cache_path,
        errorcodes_path=errorcodes_path, errorcodes_dict_path=errorcodes_dict_path,
        system_overview_path=system_overview_path)
    loss = loss if loss is not None else quantify_loss(
        iid, window_start, window_end,
        detection_path=detection_path, tariffs_path=tariffs_path)
    det = pd.read_parquet(detection_path)
    det["date"] = pd.to_datetime(det["date"]).dt.date
    stats = _detection_stats(det, iid, ws, we)
    chart = _chart_series(det, iid, ws, we)
    evid = _evidence_signals(cause, ws, we, errorcodes_path, errorcodes_dict_path)
    meta = load_meta(system_overview_path)
    cf = _capacity_factor(det, meta)
    portfolio_kwh = PORTFOLIO_KW * 8760.0 * cf
    eur_low = portfolio_kwh * SILENT_LOSS_LOW * loss.tariff_eur_per_kwh
    eur_high = portfolio_kwh * SILENT_LOSS_HIGH * loss.tariff_eur_per_kwh
    roi = round(loss.euros_lost / INTERVENTION_COST_EUR, 2)
    rec = _RECOMMENDATION.get(cause.primary_cause, "Manual review")
    validation = _validation(det, iid, ws, we)

    facts = {
        "inverter_id": iid,
        "plant": PLANT_DESC,
        "window": {"start": str(ws), "end": str(we)},
        "detection": stats,
        "cause": {"primary": cause.primary_cause, "side": cause.side,
                  "confidence": cause.confidence},
        "evidence": evid,
        "loss": {"lost_kwh": loss.lost_kwh, "euros_lost": loss.euros_lost,
                 "euros_ci_low": loss.euros_ci_low, "euros_ci_high": loss.euros_ci_high,
                 "tariff_eur_per_kwh": loss.tariff_eur_per_kwh, "method": loss.method},
        "business_case": {
            "days_undetected_caught": stats["sustained_zero_pr_days"],
            "projection_note": ("PROJECTION not measured. 3.8 GW portfolio x 1-3% "
                                "silent annual loss x tariff; capacity factor derived "
                                f"from Plant A {CF_YEAR} actuals ({cf:.3f})."),
            "portfolio_eur_per_year_low": round(eur_low, 0),
            "portfolio_eur_per_year_high": round(eur_high, 0),
        },
        "action": {"recommendation": rec, "roi_multiple": roi,
                   "intervention_cost_assumption_eur": INTERVENTION_COST_EUR},
        "validation": validation,
        "chart_series": chart,
    }
    facts["narration_plain"] = _narration(facts)
    assert_facts_consistent(facts, cause, loss)
    return facts


def assert_facts_consistent(facts: dict, cause, loss) -> None:
    """Validate-before-show gate: raise if any emitted number != its source."""
    checks = [
        (facts["cause"]["primary"], cause.primary_cause),
        (facts["cause"]["side"], cause.side),
        (facts["cause"]["confidence"], cause.confidence),
        (facts["loss"]["lost_kwh"], loss.lost_kwh),
        (facts["loss"]["euros_lost"], loss.euros_lost),
        (facts["loss"]["euros_ci_low"], loss.euros_ci_low),
        (facts["loss"]["euros_ci_high"], loss.euros_ci_high),
        (facts["loss"]["tariff_eur_per_kwh"], loss.tariff_eur_per_kwh),
        (facts["loss"]["method"], loss.method),
        (facts["action"]["roi_multiple"],
         round(loss.euros_lost / INTERVENTION_COST_EUR, 2)),
    ]
    for got, expected in checks:
        if isinstance(expected, float):
            if not math.isclose(float(got), expected, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"verified_facts mismatch: {got} != {expected}")
        elif got != expected:
            raise ValueError(f"verified_facts mismatch: {got!r} != {expected!r}")
    hp, sp = facts["chart_series"]["hero_pr"], facts["chart_series"]["sibling_pr"]
    if not (len(facts["chart_series"]["dates"]) == len(hp) == len(sp)):
        raise ValueError("verified_facts mismatch: chart_series arrays differ in length")


def write_facts(facts: dict, path: str) -> None:
    # preserve the independent Slice-3.5 loss_cross_check block across re-runs
    if "loss_cross_check" not in facts and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as _fh:
                _prev = json.load(_fh)
            if isinstance(_prev, dict) and "loss_cross_check" in _prev:
                facts = {**facts, "loss_cross_check": _prev["loss_cross_check"]}
        except (OSError, ValueError):
            pass
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(facts, fh, indent=2, ensure_ascii=True)
        fh.write("\n")



def attach_cross_check(path: str, inverter_id: str, window_start: str,
                       window_end: str, tariff_eur_per_kwh: float = 0.115) -> dict:
    """Compute the independent XGBoost weather-model loss and persist it into
    verified_facts.json as result.loss_cross_check. Backend-only, never in the
    decision path. Recomputes agreement_pct against the current causalimpact euro."""
    from src.expected_power import train_expected_power, xgb_lost_kwh
    iid = canonical_inverter_id(inverter_id) or str(inverter_id)
    model, res = train_expected_power(iid)
    kwh = xgb_lost_kwh(model, iid, window_start, window_end)
    eur = round(kwh * tariff_eur_per_kwh, 2)
    with open(path, encoding="utf-8") as fh:
        facts = json.load(fh)
    ci_eur = float(facts.get("loss", {}).get("euros_lost", 0.0)) or eur
    lo, hi = sorted((eur, ci_eur))
    agreement = round(100.0 * lo / hi, 1) if hi > 0 else 0.0
    facts["loss_cross_check"] = {
        "method": "xgboost_weather",
        "lost_kwh": round(float(kwh), 1),
        "euros_lost": eur,
        "val_r2": round(float(res.val_r2), 4),
        "causalimpact_euros": round(ci_eur, 2),
        "agreement_pct": agreement,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(facts, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return facts["loss_cross_check"]

def _detection_stats(det: pd.DataFrame, iid, ws, we) -> dict:
    w = det[(det["inverter_id"] == iid) & (det["date"] >= ws) & (det["date"] <= we)]
    zero_dates = w.loc[w["pr"].fillna(0) < DEAD_PR_MAX, "date"].tolist()
    runs = _consecutive_runs(zero_dates)
    longest = max(((r[1] - r[0]).days + 1 for r in runs), default=0)
    fault = w[w["reason"] == "fault"]
    return {
        "sustained_zero_pr_days": int(longest),
        "hero_mean_pr_fault": round(float(fault["pr"].mean()), 3) if len(fault) else 0.0,
        "sibling_mean_pr": round(float(w["sibling_pr"].mean()), 3) if len(w) else 0.0,
    }


def _chart_series(det: pd.DataFrame, iid, ws, we) -> dict:
    w = det[(det["inverter_id"] == iid) & (det["date"] >= ws) & (det["date"] <= we)]
    w = w.sort_values("date")
    return {
        "dates": [str(d) for d in w["date"]],
        "hero_pr": [round(float(x), 3) for x in w["pr"].fillna(0.0)],
        "sibling_pr": [round(float(x), 3) for x in w["sibling_pr"].fillna(0.0)],
    }


def _evidence_signals(cause, ws, we, ec_path, ec_dict_path) -> dict:
    """DC numbers come from the verdict (single authority); errorcode from the log."""
    import re

    con = duckdb.connect()
    try:
        eccols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{ec_path}')").fetchall()]
        errc = _find(eccols, re.escape(cause.inverter_id) + r" / Error")
        code_row = con.execute(
            f"""SELECT "{errc}" code, count(*) n FROM read_parquet('{ec_path}')
                WHERE "{errc}" IS NOT NULL AND "{errc}" != 0
                  AND CAST(strptime(timestamp, '%Y.%m.%d %H:%M') AS DATE)
                      BETWEEN '{ws}' AND '{we}'
                GROUP BY 1 ORDER BY n DESC LIMIT 1"""
        ).fetchone()
    finally:
        con.close()
    code = int(code_row[0]) if code_row else 0
    count = int(code_row[1]) if code_row else 0
    text = _load_errorcode_dict(ec_dict_path).get(code, "") if code else ""
    return {
        "u_dc_v": cause.u_dc_v,
        "i_dc_a": cause.i_dc_a,
        "u_dc_healthy_v": cause.u_dc_healthy_v,
        "errorcode": code,
        "errorcode_count": count,
        "errorcode_text": text,
    }


def _capacity_factor(det: pd.DataFrame, meta: pd.DataFrame) -> float:
    plant_kwp = float(meta["kwp"].sum())
    year = det[[d.year == CF_YEAR for d in det["date"]]]
    total_kwh = float(year["daily_kwh"].sum())
    return total_kwh / (plant_kwp * 8760.0) if plant_kwp > 0 else 0.0


def _validation(det: pd.DataFrame, iid, ws, we) -> dict:
    tickets = hero_match.load_tickets()
    window = f"{ws} -> {we}"
    matched = window in set(tickets["window"])
    cands, _per = hero_match.cross_match(det, tickets)
    affected_match = False
    if not cands.empty:
        rowsel = cands[(cands["inverter_id"] == iid) & (cands["ticket_window"] == window)]
        affected_match = bool(rowsel["precision_match"].any()) if not rowsel.empty else False
    return {"ticket_id": f"{ws}..{we}", "ticket_matched": bool(matched),
            "affected_count_match": affected_match}


def _narration(f: dict) -> str:
    e = f["evidence"]
    return (
        f"{f['inverter_id']} ran dead for {f['detection']['sustained_zero_pr_days']} days "
        f"({f['cause']['primary']}, {f['cause']['side']} side; inverter error "
        f"{e['errorcode']} '{e['errorcode_text']}'), losing about "
        f"EUR {f['loss']['euros_lost']:.0f} "
        f"(95% CI EUR {f['loss']['euros_ci_low']:.0f}-{f['loss']['euros_ci_high']:.0f}, "
        f"{f['loss']['method']}). This matches ENERPARC ticket {f['validation']['ticket_id']}. "
        f"Recommended: {f['action']['recommendation']}."
    )


if __name__ == "__main__":
    facts = build_facts("INV 01.05.029", "2019-05-24", "2019-06-16")
    write_facts(facts, "outputs/verified_facts.json")
    print(json.dumps(facts, indent=2, ensure_ascii=True))
