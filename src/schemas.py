"""Pydantic V2 schemas for Slice 1 detection + hero matching."""

from datetime import date

from pydantic import BaseModel


class DetectionRow(BaseModel):
    """One inverter-day detection record."""

    date: date
    inverter_id: str
    orientation: str
    pr: float
    daily_kwh: float
    sibling_pr: float
    residual: float
    dv_frac: float
    flagged: bool
    reason: str  # 'fault' | 'curtailment' | 'ok'


class HeroCandidate(BaseModel):
    """A detected inverter outage matched to a 2019 ticket window."""

    inverter_id: str
    ticket_window: str
    overlap_days: int
    mean_residual: float
    estimated_lost_kwh: float
    precision_match: bool
    rank: int
