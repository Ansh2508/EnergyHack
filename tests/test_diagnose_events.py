"""diagnose_events tests -- light (read-only diagnostic) + labeler/cluster units.

Asserts the downstream verdict dicts (signal, parsed errorcode, clusters), not just
returns. The SIGNAL labeler and cluster grouping are unit-tested on synthetic input
so the genuine-fault vs planned-offline rule is pinned independently of the data.
"""

from __future__ import annotations

import json
import math

import pytest

from src import diagnose_events


@pytest.fixture(scope="session")
def real() -> dict:
    """One real diagnosis pass over the top 8 fleet events (calls diagnose())."""
    return diagnose_events.run(top_n=8)


# ---- real run (read-only) ------------------------------------------------------


def test_runs_on_real_top8_without_throwing(real):
    vs = real["verdicts"]
    assert 1 <= len(vs) <= 8
    keys = {
        "inverter_id",
        "window",
        "euros",
        "primary_cause",
        "side",
        "confidence",
        "u_dc_v",
        "i_dc_a",
        "errorcode",
        "errorcode_count",
        "errorcode_text",
        "signal",
    }
    for v in vs:
        assert keys <= set(v)
        assert v["signal"] in {"DEAD_INVERTER", "POSSIBLE_OFFLINE", "REVIEW"}


def test_real_verdicts_have_no_nan_inf(real):
    for v in real["verdicts"]:
        assert math.isfinite(v["euros"]) and math.isfinite(v["confidence"])
        for f in ("u_dc_v", "i_dc_a", "u_dc_healthy_v"):
            assert v[f] is None or math.isfinite(v[f])
        assert isinstance(v["errorcode_count"], int) and v["errorcode_count"] >= 0
    # the written JSON must contain no NaN/Infinity tokens
    raw = json.dumps(real)
    json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_real_clusters_have_two_or_more(real):
    for c in real["clusters"]:
        assert c["n"] >= 2 and len(set(c["inverters"])) == c["n"]


# ---- SIGNAL labeler (unit, synthetic verdicts) --------------------------------


def test_signal_dead_inverter_signature():
    """U_DC present + I_DC ~0 + errorcode logged + dead cause -> DEAD_INVERTER."""
    assert (
        diagnose_events.signal_for("DEAD_INVERTER", 800.0, 0.1, 1947) == "DEAD_INVERTER"
    )


def test_signal_both_rails_zero_is_offline():
    """Whole DC side dark (U_DC and I_DC ~0) -> POSSIBLE_OFFLINE."""
    assert (
        diagnose_events.signal_for("DEAD_INVERTER", 0.0, 0.0, 0) == "POSSIBLE_OFFLINE"
    )
    assert (
        diagnose_events.signal_for("DEAD_INVERTER", 5.0, 0.0, 500) == "POSSIBLE_OFFLINE"
    )


def test_signal_no_errorcode_is_offline():
    """U_DC present but nothing logged -> deliberate disconnection -> POSSIBLE_OFFLINE."""
    assert (
        diagnose_events.signal_for("DEAD_INVERTER", 800.0, 0.1, 0) == "POSSIBLE_OFFLINE"
    )


def test_signal_review_otherwise():
    """Live current, error logged, non-dead cause -> needs a human -> REVIEW."""
    assert diagnose_events.signal_for("UNKNOWN", 800.0, 5.0, 7) == "REVIEW"


def test_signal_handles_none_rails():
    """None rails must not crash; no errorcode -> POSSIBLE_OFFLINE, else REVIEW."""
    assert (
        diagnose_events.signal_for("NOT_A_FAULT", None, None, 0) == "POSSIBLE_OFFLINE"
    )
    assert diagnose_events.signal_for("NOT_A_FAULT", None, None, 3) == "REVIEW"


# ---- errorcode parsing ---------------------------------------------------------


def test_parse_errorcode_valid():
    code, n, text = diagnose_events.parse_errorcode(
        "655626 (1947x in window): Erkennung von Netzunterspannung (ENS,Leistungsteil)"
    )
    assert code == 655626 and n == 1947 and "Netzunterspannung" in text


def test_parse_errorcode_none_and_malformed():
    assert diagnose_events.parse_errorcode(None) == (None, 0, None)
    code, n, text = diagnose_events.parse_errorcode("garbage string")
    assert code is None and n == 0 and text == "garbage string"


# ---- cluster grouping (unit) ---------------------------------------------------


def _ev(iid, start):
    return {"inverter_id": iid, "window": {"start": start, "end": start}}


def test_find_clusters_groups_shared_start_dates():
    ranked = [
        _ev("INV A", "2025-08-04"),
        _ev("INV B", "2025-08-04"),
        _ev("INV C", "2025-09-01"),
    ]
    clusters = diagnose_events.find_clusters(ranked)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["start"] == "2025-08-04" and c["n"] == 2
    assert c["inverters"] == ["INV A", "INV B"]


def test_find_clusters_ignores_distinct_dates_and_dedups():
    ranked = [
        _ev("INV A", "2025-01-01"),
        _ev("INV B", "2025-02-02"),
        _ev("INV A", "2025-03-03"),
        _ev("INV A", "2025-03-03"),
    ]  # same inv+date twice
    assert diagnose_events.find_clusters(ranked) == []
