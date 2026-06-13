"""Hero matching: cross-match detected outages against 2019 inverter tickets.

A "hero" is a detected inverter outage that lines up with a real maintenance
ticket -- proof the detector finds true events. Tickets name the component and
the count of affected units, but not which inverter; the detector supplies the
identity via runs of consecutive non-curtailed fault days.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

_DATA = "data/Plant A (start here)"
DEFAULT_TICKETS = f"{_DATA}/2. Additional Data/Tickets.xlsx"
TICKET_COMPONENT = "Wechselrichter"
MIN_CONSECUTIVE = 3


def load_tickets(
    tickets_path: str = DEFAULT_TICKETS, sheet: str = "2019-2020"
) -> pd.DataFrame:
    """Load the ticket sheet, keep Komponente == 'Wechselrichter'.

    Returns DataFrame[start, end (date), affected (int), window (str)]. Column
    positions are resolved by header text, not hardcoded indices.
    """
    import openpyxl

    wb = openpyxl.load_workbook(tickets_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    hdr = [("" if c is None else str(c).strip()) for c in rows[0]]

    def col(prefix: str) -> int:
        return next(i for i, h in enumerate(hdr) if h.startswith(prefix))

    c_start, c_end = col("Start Date"), col("Datum Ende")
    c_comp, c_aff = col("Komponente"), col("Anzahl")
    recs = []
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        if str(r[c_comp]).strip() != TICKET_COMPONENT:
            continue
        start = _as_date(r[c_start])
        end = _as_date(r[c_end])
        if start is None or end is None:
            continue
        try:
            affected = int(r[c_aff])
        except (TypeError, ValueError):
            affected = 0
        recs.append(
            {
                "start": start,
                "end": end,
                "affected": affected,
                "window": f"{start} -> {end}",
            }
        )
    return pd.DataFrame(recs)


def cross_match(
    daily: pd.DataFrame, tickets: pd.DataFrame, min_consecutive: int = MIN_CONSECUTIVE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match inverters to ticket windows and rank hero candidates.

    For each ticket window, an inverter qualifies when it has a run of
    >= min_consecutive consecutive non-curtailed fault days that overlaps the
    window. Per ticket, the strongest `affected` candidates (by overlap then
    |residual|) are marked precision_match=True. Candidates are globally ranked
    single-affected first, then longer overlap, then larger |mean residual|.

    Returns (candidates, per_ticket) DataFrames.
    """
    fault = daily[daily["reason"] == "fault"].copy()
    fault["date"] = pd.to_datetime(fault["date"]).dt.date
    has_lost = "lost_kwh" in daily.columns
    by_inv = {iid: g for iid, g in fault.groupby("inverter_id")}

    rows = []
    per_ticket = []
    for _, tk in tickets.iterrows():
        s, e, affected = tk["start"], tk["end"], int(tk["affected"])
        detected = []
        for iid, g in by_inv.items():
            days = sorted(g["date"].tolist())
            best = _best_run(days, s, e, min_consecutive)
            if best is None:
                continue
            overlap_days, ov_dates = best
            sub = g[g["date"].isin(ov_dates)]
            mean_res = float(sub["residual"].mean())
            lost = float(sub["lost_kwh"].sum()) if has_lost else 0.0
            detected.append((iid, overlap_days, mean_res, lost))
        detected.sort(key=lambda x: (-x[1], -abs(x[2])))
        matched_ids = {d[0] for d in detected[:affected]}
        for iid, ov, mres, lost in detected:
            rows.append(
                {
                    "inverter_id": iid,
                    "ticket_window": tk["window"],
                    "affected": affected,
                    "detected": len(detected),
                    "overlap_days": ov,
                    "mean_residual": mres,
                    "estimated_lost_kwh": lost,
                    "precision_match": iid in matched_ids,
                }
            )
        per_ticket.append(
            {
                "window": tk["window"],
                "affected": affected,
                "detected": len(detected),
                "precision_matched": len(matched_ids),
            }
        )

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates["abs_residual"] = candidates["mean_residual"].abs()
        candidates = candidates.sort_values(
            by=["affected", "overlap_days", "abs_residual"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        candidates["rank"] = candidates.index + 1
    return candidates, pd.DataFrame(per_ticket)


def _as_date(value) -> dt.date | None:
    """Coerce an Excel cell to a date (handles datetime, date, or None)."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _best_run(
    days: list[dt.date], start: dt.date, end: dt.date, min_consecutive: int
) -> tuple[int, list[dt.date]] | None:
    """Longest consecutive fault run (>= min_consecutive) overlapping [start, end].

    Returns (overlap_day_count, overlapping_dates) or None.
    """
    if not days:
        return None
    best = None
    run_start = prev = days[0]
    runs = []
    for d in days[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            runs.append((run_start, prev))
            run_start = prev = d
    runs.append((run_start, prev))
    for r0, r1 in runs:
        length = (r1 - r0).days + 1
        if length < min_consecutive or r0 > end or r1 < start:
            continue
        ov_dates = [d for d in days if r0 <= d <= r1 and start <= d <= end]
        if ov_dates and (best is None or len(ov_dates) > best[0]):
            best = (len(ov_dates), ov_dates)
    return best
