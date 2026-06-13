"""Fleet-scan tests -- guardrails + branch/fallback coverage.

Asserts the DOWNSTREAM aggregate (the fleet_scan.json dict), not just returns:
hero sanity vs the frozen number, no-double-count, finiteness, real data span,
curtailment exclusion, determinism, and the gate that actually selected each event.
Unit tests force every branch of the helpers (clamp, missing tariff, non-finite).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import fleet_scan, quantify


@pytest.fixture(scope="session")
def result() -> dict:
    """Run the scan once over the real detection parquet (pure compute)."""
    return fleet_scan.scan()


@pytest.fixture(scope="session")
def det() -> pd.DataFrame:
    d = pd.read_parquet(fleet_scan.DEFAULT_DETECTION)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d


# ---- guardrails (downstream aggregate) ----------------------------------------


def test_hero_present_and_within_25pct_of_frozen(result):
    """Hero must appear and be within +/-25% of the frozen EUR 195 (window/gate sanity)."""
    hc = result["hero_check"]
    assert hc is not None, "hero INV 01.05.029 missing from the scan"
    assert hc["inverter_id"] == fleet_scan.HERO_ID
    assert 195 * 0.75 <= hc["euros"] <= 195 * 1.25, (
        f"hero euros {hc['euros']} outside +/-25% of 195 -- likely a window/gate bug"
    )


def test_fleet_total_equals_sum_of_ranked(result):
    """No double-counting: the headline total is exactly the sum of the events."""
    s = round(sum(e["euros"] for e in result["ranked"]), 2)
    assert abs(result["fleet_total_eur"] - s) < 0.01
    s_kwh = round(sum(e["lost_kwh"] for e in result["ranked"]), 1)
    assert abs(result["fleet_total_kwh"] - s_kwh) < 0.1


def test_no_nan_inf_and_all_nonneg(result):
    """Every euro/kWh is finite and >= 0, headline included."""
    import math

    for e in result["ranked"]:
        assert math.isfinite(e["euros"]) and e["euros"] >= 0
        assert math.isfinite(e["lost_kwh"]) and e["lost_kwh"] >= 0
        assert e["n_fault_days"] >= fleet_scan.MIN_CONSECUTIVE
    assert math.isfinite(result["fleet_total_eur"]) and result["fleet_total_eur"] >= 0
    assert math.isfinite(result["fleet_total_kwh"]) and result["fleet_total_kwh"] >= 0
    assert result["n_events"] == len(result["ranked"])
    assert result["n_affected_inverters"] == len(
        {e["inverter_id"] for e in result["ranked"]}
    )


def test_ranked_sorted_desc(result):
    r = result["ranked"]
    assert all(r[i]["euros"] >= r[i + 1]["euros"] for i in range(len(r) - 1))


def test_data_span_is_real_not_hardcoded(result, det):
    """data_span equals the true min/max date in the data (computed, not a literal)."""
    assert result["data_span"]["start"] == min(det["date"]).isoformat()
    assert result["data_span"]["end"] == max(det["date"]).isoformat()


def test_curtailment_period_is_not_a_fault_event(result, det):
    """A known curtailment day must never fall inside any event window (downstream)."""
    curt = det[det["reason"] == "curtailment"]
    assert not curt.empty, "no curtailment days in the data -- test precondition failed"
    curt_set = set(zip(curt["inverter_id"].astype(str), curt["date"]))
    for e in result["ranked"]:
        iid = e["inverter_id"]
        d = dt.date.fromisoformat(e["window"]["start"])
        end = dt.date.fromisoformat(e["window"]["end"])
        while d <= end:
            assert (iid, d) not in curt_set, (
                f"curtailment day {d} inside event for {iid}"
            )
            d += dt.timedelta(days=1)


def test_every_event_day_is_sustained_zero_fault(result, det):
    """Downstream gate check: each event day is reason=='fault' AND pr < ZERO_PR."""
    mask = (
        (det["reason"] == "fault")
        & det["pr"].notna()
        & (det["pr"] < fleet_scan.ZERO_PR)
    )
    zero_set = set(zip(det[mask]["inverter_id"].astype(str), det[mask]["date"]))
    for e in result["ranked"]:
        iid = e["inverter_id"]
        d = dt.date.fromisoformat(e["window"]["start"])
        end = dt.date.fromisoformat(e["window"]["end"])
        while d <= end:
            assert (iid, d) in zero_set, (
                f"non-zero/non-fault day {d} inside event for {iid}"
            )
            d += dt.timedelta(days=1)


def test_determinism_two_runs_identical(result):
    """Deterministic counterfactual -> identical total on re-run (no Bayesian drift)."""
    again = fleet_scan.scan()
    assert again["fleet_total_eur"] == result["fleet_total_eur"]
    assert again["n_events"] == result["n_events"]
    assert again["hero_check"]["euros"] == result["hero_check"]["euros"]


# ---- helper branch / fallback coverage ----------------------------------------


def test_consecutive_runs_splits_on_gap():
    days = [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 5)]
    assert fleet_scan._consecutive_runs(days) == [
        (dt.date(2020, 1, 1), dt.date(2020, 1, 2)),
        (dt.date(2020, 1, 5), dt.date(2020, 1, 5)),
    ]
    assert fleet_scan._consecutive_runs([]) == []


def _synthetic_det(window_kwh: float, expected_per_day: float = 144.0) -> pd.DataFrame:
    """Inverter rows: 60 pre days at expected (sigma~0) + a 3-day window at window_kwh.

    expected = sibling_pr*kwp*insol = 0.8*30*6 = 144 kWh/day.
    """
    rows = []
    ws = dt.date(2020, 6, 1)
    for k in range(1, 61):
        rows.append((ws - dt.timedelta(days=k), 0.8, 30.0, 6.0, expected_per_day))
    for k in range(3):
        rows.append((ws + dt.timedelta(days=k), 0.8, 30.0, 6.0, window_kwh))
    return pd.DataFrame(
        rows, columns=["date", "sibling_pr", "kwp", "insol_kwhm2", "daily_kwh"]
    )


def test_find_events_gate_excludes_short_and_partial_and_curtailment():
    """<3 days excluded; partial (pr>=ZERO) excluded; curtailment splits the run."""
    base = dict(orientation="O", sibling_pr=0.8, kwp=30.0, insol_kwhm2=6.0)
    recs = []

    def add(iid, day, reason, pr, kwh=0.0):
        recs.append(
            {
                "inverter_id": iid,
                "date": day,
                "reason": reason,
                "pr": pr,
                "daily_kwh": kwh,
                **base,
            }
        )

    d = dt.date(2020, 6, 1)
    # A: 4-day sustained-zero with a partial recovery day on day 5 -> event is the 4 zero days
    for k in range(4):
        add("INV A", d + dt.timedelta(days=k), "fault", 0.0)
    add(
        "INV A", d + dt.timedelta(days=4), "fault", 0.45
    )  # partial -> excluded by pr gate
    # B: only 2 zero days -> below MIN_CONSECUTIVE -> excluded
    add("INV B", d, "fault", 0.0)
    add("INV B", d + dt.timedelta(days=1), "fault", 0.0)
    # C: zero days split by a curtailment day -> two runs, each below 3 -> excluded
    add("INV C", d, "fault", 0.0)
    add("INV C", d + dt.timedelta(days=1), "curtailment", 0.0)
    add("INV C", d + dt.timedelta(days=2), "fault", 0.0)
    det = pd.DataFrame(recs)
    events = fleet_scan.find_events(det)
    ev = {(iid, s.isoformat(), e.isoformat(), n) for iid, s, e, n in events}
    assert ("INV A", d.isoformat(), (d + dt.timedelta(days=3)).isoformat(), 4) in ev
    assert not any(iid == "INV B" for iid, *_ in events)
    assert not any(iid == "INV C" for iid, *_ in events)


def test_price_event_clamps_negative_loss_to_zero(monkeypatch):
    """When actual > expected (no real loss), euros clamps to 0 (not negative)."""
    monkeypatch.setattr(fleet_scan, "_tariff", lambda *a: 0.10)
    inv = _synthetic_det(
        window_kwh=200.0
    )  # actual 200 > expected 144 -> negative raw loss
    out = fleet_scan.price_event(inv, "INV X", dt.date(2020, 6, 1), dt.date(2020, 6, 3))
    assert out is not None
    euros, lost_kwh, tariff = out
    assert euros == 0.0 and lost_kwh == 0.0


def test_price_event_positive_loss(monkeypatch):
    """A real zero-output window prices a positive loss = (expected - actual) * tariff."""
    monkeypatch.setattr(fleet_scan, "_tariff", lambda *a: 0.10)
    inv = _synthetic_det(window_kwh=0.0)  # 3 days at 0 vs 144 expected -> 432 kWh
    out = fleet_scan.price_event(inv, "INV X", dt.date(2020, 6, 1), dt.date(2020, 6, 3))
    assert out is not None
    euros, lost_kwh, _t = out
    assert lost_kwh == pytest.approx(432.0, abs=1.0)
    assert euros == pytest.approx(43.2, abs=0.5)


def test_price_event_none_when_tariff_missing(monkeypatch):
    """Except branch: no tariff for the window -> None (skip), never assume a tariff."""

    def _raise(*a):
        raise ValueError("tariff not found")

    monkeypatch.setattr(fleet_scan, "_tariff", _raise)
    inv = _synthetic_det(window_kwh=0.0)
    out = fleet_scan.price_event(inv, "INV X", dt.date(2020, 6, 1), dt.date(2020, 6, 3))
    assert out is None


def test_price_event_none_when_loss_not_finite(monkeypatch):
    """Non-finite loss -> None (guarded before the JSON ever sees NaN)."""
    monkeypatch.setattr(
        fleet_scan.quantify, "_sibling_sigma_loss", lambda *a: (float("nan"), 0.0, 0.0)
    )
    inv = _synthetic_det(window_kwh=0.0)
    out = fleet_scan.price_event(inv, "INV X", dt.date(2020, 6, 1), dt.date(2020, 6, 3))
    assert out is None


def test_tariff_read_from_file_matches_quantify():
    """Tariff is READ from the file (parity with quantify._read_tariff), never assumed."""
    a = fleet_scan._tariff(
        quantify.DEFAULT_TARIFFS, "INV 01.05.029", "2019-05-25", "2019-06-03"
    )
    b = quantify._read_tariff(
        quantify.DEFAULT_TARIFFS,
        "INV 01.05.029",
        dt.date(2019, 5, 25),
        dt.date(2019, 6, 3),
    )
    assert a == pytest.approx(b)
