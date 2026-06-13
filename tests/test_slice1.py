"""Slice 1 tests -- content-asserting + branch/fallback coverage.

The three required content tests run the full pipeline once (session fixture).
Pipeline inputs honour EP_* env overrides (see src.run_slice1.resolve_paths).
The unit tests assert DOWNSTREAM values (reason/flagged/id), not just returns.
"""

import datetime as dt

import pandas as pd
import pytest

from src import curtailment, detect
from src.ingest import canonical_inverter_id
from src.run_slice1 import build_detection


@pytest.fixture(scope="session")
def pipeline():
    """Run the Slice 1 pipeline once for the content-asserting tests."""
    return build_detection()


def test_detection_not_constant(pipeline):
    """>=60 inverters have a non-constant, non-null PR series."""
    daily = pipeline["daily"]
    nunique = daily.dropna(subset=["pr"]).groupby("inverter_id")["pr"].nunique()
    non_constant = int((nunique > 1).sum())
    assert non_constant >= 60, f"only {non_constant} inverters with varying PR"


def test_dv_mask_applied(pipeline):
    """>=1 curtailment day; curtailment days are never faults / hero candidates."""
    daily = pipeline["daily"]
    curt = daily[daily["reason"] == "curtailment"]
    assert len(curt) >= 1
    assert not curt["flagged"].any(), "a curtailment day was flagged as fault"
    fault_keys = set(
        map(tuple, daily[daily["reason"] == "fault"][["inverter_id", "date"]].values)
    )
    curt_keys = set(map(tuple, curt[["inverter_id", "date"]].values))
    assert fault_keys.isdisjoint(curt_keys), "a day is both fault and curtailment"


def test_hero_match_nonempty(pipeline):
    """>=1 candidate on a real 2019 ticket window; top hero overlaps >=3 days."""
    cand = pipeline["candidates"]
    tickets = pipeline["tickets"]
    assert not cand.empty
    windows = set(tickets["window"])
    top = cand.iloc[0]
    assert top["ticket_window"] in windows
    assert int(top["overlap_days"]) >= 3


def test_canonical_inverter_id_branches():
    """Monitoring form, WR meta form, split sub-inverter, and the None fallback."""
    assert canonical_inverter_id("INV 01.05.029 / P_AC (kW)") == "INV 01.05.029"
    assert canonical_inverter_id("WR 01 .05 .029") == "INV 01.05.029"
    assert canonical_inverter_id("WR 01 .01. 004.02") == "INV 01.01.004"
    assert canonical_inverter_id("Plant / Altitude (deg)") is None


def test_mask_curtailment_reason_branches():
    """Curtailment overrides fault; NaN PR never becomes a fault."""
    df = pd.DataFrame(
        {
            "pr": [0.10, 0.10, float("nan"), 1.0],
            "flagged_raw": [True, True, True, False],
            "dv_frac": [0.00, 0.50, 0.00, 0.00],
        }
    )
    out = curtailment.mask_curtailment(df)
    assert out["reason"].tolist() == ["fault", "curtailment", "ok", "ok"]
    assert out["flagged"].tolist() == [True, False, False, False]


def test_sibling_baseline_nan_not_flagged():
    """A NULL PR yields NULL residual and is not flagged."""
    day = dt.date(2019, 5, 24)
    df = pd.DataFrame(
        {
            "inverter_id": ["A", "B"],
            "date": [day, day],
            "orientation": ["O", "O"],
            "pr": [float("nan"), 1.0],
        }
    )
    out = detect.sibling_baseline(df)
    assert not bool(out.loc[0, "flagged_raw"])
    assert pd.isna(out.loc[0, "residual"])
