"""Slice 1 orchestrator: ingest -> detect -> mask -> hero match.

Wires the pipeline and writes outputs/detection_daily.parquet and
outputs/hero_candidates.md. Input paths default to the Plant A data tree and
can be overridden with EP_MONITORING / EP_CACHE / EP_SYSTEM_OVERVIEW /
EP_TICKETS / EP_OUTDIR environment variables (used for sandbox / CI runs).

Run:  python -m src.run_slice1
"""

from __future__ import annotations

import os

import duckdb
import pandas as pd

from src import curtailment, detect, hero_match, ingest
from src.schemas import DetectionRow, HeroCandidate

_DETECTION_COLS = [
    "date",
    "inverter_id",
    "orientation",
    "pr",
    "daily_kwh",
    "sibling_pr",
    "residual",
    "dv_frac",
    "flagged",
    "reason",
]


def resolve_paths() -> dict:
    """Resolve input/output paths, honouring EP_* environment overrides."""
    g = os.environ.get
    return {
        "monitoring": g("EP_MONITORING", ingest.DEFAULT_MONITORING),
        "cache": g("EP_CACHE", ingest.DEFAULT_CACHE),
        "system_overview": g("EP_SYSTEM_OVERVIEW", ingest.DEFAULT_SYSTEM_OVERVIEW),
        "tickets": g("EP_TICKETS", hero_match.DEFAULT_TICKETS),
        "outdir": g("EP_OUTDIR", "outputs"),
    }


def build_detection(paths: dict | None = None) -> dict:
    """Run the full pipeline; return daily detections + hero candidates.

    Pure compute (no file writes) so tests can assert on the results directly.
    """
    paths = paths or resolve_paths()
    con = duckdb.connect()
    try:
        ingest.load_monitoring(con, paths["monitoring"], paths["cache"])
        meta = ingest.load_meta(paths["system_overview"])
        detect.compute_pr(con, meta)
        daily = detect.daily_aggregate(con)
    finally:
        con.close()
    daily = detect.sibling_baseline(daily)
    daily = curtailment.mask_curtailment(daily)
    daily["expected_kwh"] = daily["sibling_pr"] * daily["kwp"] * daily["insol_kwhm2"]
    daily["lost_kwh"] = (daily["expected_kwh"] - daily["daily_kwh"]).clip(lower=0)
    tickets = hero_match.load_tickets(paths["tickets"])
    candidates, per_ticket = hero_match.cross_match(daily, tickets)
    return {
        "daily": daily,
        "meta": meta,
        "tickets": tickets,
        "candidates": candidates,
        "per_ticket": per_ticket,
    }


def main(paths: dict | None = None) -> dict:
    """Run the pipeline and write the two deliverables; print a status line."""
    paths = paths or resolve_paths()
    res = build_detection(paths)
    os.makedirs(paths["outdir"], exist_ok=True)
    _write_detection_parquet(
        res["daily"], os.path.join(paths["outdir"], "detection_daily.parquet")
    )
    _write_hero_report(res, os.path.join(paths["outdir"], "hero_candidates.md"))
    _print_status(res)
    return res


def _write_detection_parquet(daily: pd.DataFrame, path: str) -> None:
    """Write the inverter-day detections; validate one row against DetectionRow."""
    extra = ["kwp", "insol_kwhm2", "lost_kwh", "n_pr"]
    out = daily[[c for c in _DETECTION_COLS + extra if c in daily.columns]].copy()
    out.to_parquet(path, index=False)
    sample = out[out["pr"].notna()].head(1)
    if not sample.empty:
        r = sample.iloc[0]
        DetectionRow(
            date=pd.Timestamp(r["date"]).date(),
            inverter_id=str(r["inverter_id"]),
            orientation=str(r["orientation"]),
            pr=float(r["pr"]),
            daily_kwh=float(r["daily_kwh"]),
            sibling_pr=float(r["sibling_pr"]),
            residual=float(r["residual"]),
            dv_frac=float(r["dv_frac"]),
            flagged=bool(r["flagged"]),
            reason=str(r["reason"]),
        )


def _hero_models(candidates: pd.DataFrame, limit: int = 15) -> list[HeroCandidate]:
    """Validate the top candidates through the HeroCandidate schema (V2 boundary)."""
    models = []
    for _, r in candidates.head(limit).iterrows():
        models.append(
            HeroCandidate(
                inverter_id=str(r["inverter_id"]),
                ticket_window=str(r["ticket_window"]),
                overlap_days=int(r["overlap_days"]),
                mean_residual=float(r["mean_residual"]),
                estimated_lost_kwh=float(r["estimated_lost_kwh"]),
                precision_match=bool(r["precision_match"]),
                rank=int(r["rank"]),
            )
        )
    return models


def _write_hero_report(res: dict, path: str) -> None:
    """Write outputs/hero_candidates.md: recommended hero, precision, ranking."""
    cand = res["candidates"]
    per = res["per_ticket"]
    lines = ["# Slice 1 - Hero Candidates (Plant A, 2019 tickets)", ""]
    if cand.empty:
        lines.append("No hero candidates found.")
        _write_lines(path, lines)
        return
    models = _hero_models(cand)
    top = models[0]
    lines += [
        "## Recommended hero",
        "",
        f"- **{top.inverter_id}** - ticket **{top.ticket_window}**",
        f"- overlap: **{top.overlap_days} days**, "
        f"mean residual: **{top.mean_residual:+.3f}**, "
        f"est. lost: **{top.estimated_lost_kwh:,.0f} kWh**",
        f"- precision match: **{top.precision_match}**",
        "",
        "## Per-ticket precision",
        "",
        "| Window | Komponente affected | Detected (>=3 consec) | Precision-matched |",
        "|---|---|---|---|",
    ]
    for _, r in per.iterrows():
        lines.append(
            f"| {r['window']} | {r['affected']} | {r['detected']} | "
            f"{r['precision_matched']} |"
        )
    lines += [
        "",
        "_Tickets name the component and affected count, not the inverter; "
        "precision-matched = the strongest `affected` detections per window._",
        "",
        "## Ranked candidates",
        "",
        "| Rank | Inverter | Window | Overlap | Mean residual | Lost kWh | Precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        lines.append(
            f"| {m.rank} | {m.inverter_id} | {m.ticket_window} "
            f"| {m.overlap_days} | {m.mean_residual:+.3f} "
            f"| {m.estimated_lost_kwh:,.0f} | {m.precision_match} |"
        )
    _write_lines(path, lines)


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _print_status(res: dict) -> None:
    cand = res["candidates"]
    daily = res["daily"]
    print("Done")
    print(
        f"  inverter-days: {len(daily):,} | "
        f"faults: {(daily['reason'] == 'fault').sum():,} | "
        f"curtailment: {(daily['reason'] == 'curtailment').sum():,}"
    )
    if not cand.empty:
        top = cand.iloc[0]
        print(f"  Recommended hero: {top['inverter_id']}")
        print(
            f"  Ticket matched: {top['ticket_window']} "
            f"({top['overlap_days']} days overlap)"
        )


if __name__ == "__main__":
    main()
