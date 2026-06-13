"""Slice 2 tests - cause, loss, and the validate-before-show facts gate.

Content-asserting and research-in-code: every paper's claim is exercised here -
DC-side split, soiling_srr (Deceglie), clip_filter (Perry), structured evidence
(Roy), and the validate-before-show assertion (arXiv:2606.01513).
"""

import copy
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from src.build_facts import assert_facts_consistent, build_facts
from src.diagnose import _skill_clipping, _skill_dead_inverter, _skill_soiling, diagnose
from src.quantify import quantify_loss

HERO = "INV 01.05.029"
WS, WE = "2019-05-24", "2019-06-16"


@pytest.fixture(scope="session")
def bundle():
    """Build cause + loss + facts once for the content-asserting tests."""
    return {
        "cause": diagnose(HERO, WS, WE),
        "loss": quantify_loss(HERO, WS, WE),
        "facts": build_facts(HERO, WS, WE),
    }


def test_cause_is_dead_inverter(bundle):
    """Hero -> DEAD_INVERTER, AC side, high confidence, structured evidence (Roy 2024)."""
    c = bundle["cause"]
    assert c.primary_cause == "DEAD_INVERTER"
    assert c.confidence > 0.7
    assert c.side == "AC"
    assert len(c.evidence) >= 1
    assert c.errorcode_corroboration is not None


def test_dc_side_logic():
    """U_DC present + P_AC~0 -> AC; U_DC~0 + I_DC~0 -> DC (both branches)."""
    days = [dt.date(2019, 7, 1) + dt.timedelta(days=i) for i in range(5)]

    def dead_df(udc, idc):
        return pd.DataFrame(
            {"date": days, "pr": [0.0] * 5, "insol": [6.0] * 5,
             "udc": [udc] * 5, "idc": [idc] * 5}
        )

    ws, we = dt.date(2019, 7, 1), dt.date(2019, 7, 5)
    ac = _skill_dead_inverter(dead_df(800.0, 0.1), ws, we)
    dc = _skill_dead_inverter(dead_df(5.0, 0.1), ws, we)
    assert ac["hit"] and ac["side"] == "AC"
    assert dc["hit"] and dc["side"] == "DC"


def test_soiling_branch_deceglie():
    """Real soiling_srr: gradual sawtooth -> SOILING; step-to-zero outage -> NOT."""
    days = pd.date_range("2019-01-01", periods=160, freq="D")
    saw, v = [], 1.0
    for i in range(160):
        v -= 0.004
        if i % 20 == 0:
            v = 1.0
        saw.append(v)
    insol = pd.Series(5.0, index=days)
    assert _skill_soiling(pd.Series(saw, index=days), insol, reps=100)["hit"] is True
    step = pd.Series([1.0] * 80 + [0.0] * 80, index=days)
    assert _skill_soiling(step, insol, reps=100)["hit"] is False


def test_clipping_branch_perry():
    """Real clip_filter('logic'): flat-at-top -> CLIP; flat-at-zero (dead) -> NOT."""
    idx = pd.date_range("2019-06-01 04:00", periods=60, freq="15min")
    clip = pd.Series(
        np.concatenate([np.linspace(0, 30, 20), np.full(20, 30.0), np.linspace(30, 0, 20)]),
        index=idx,
    )
    assert _skill_clipping(clip, 30.0)["hit"] is True
    assert _skill_clipping(pd.Series(np.zeros(60), index=idx), 30.0)["hit"] is False


def test_loss_positive_with_ci(bundle):
    """euros_lost > 0 and ci_low < euros_lost < ci_high."""
    loss = bundle["loss"]
    assert loss.euros_lost > 0
    assert loss.euros_ci_low < loss.euros_lost < loss.euros_ci_high
    assert loss.method in ("causalimpact", "sibling_sigma")


def test_facts_validate_before_show(bundle):
    """Mutating one emitted number makes the validate-before-show gate raise."""
    bad = copy.deepcopy(bundle["facts"])
    bad["loss"]["euros_lost"] = 1.0
    with pytest.raises(ValueError):
        assert_facts_consistent(bad, bundle["cause"], bundle["loss"])


def test_facts_complete(bundle):
    """verified_facts.json carries evidence + chart_series + loss + narration; all real."""
    f = bundle["facts"]
    for key in ("evidence", "chart_series", "loss", "business_case", "narration_plain"):
        assert key in f
    for num in ("lost_kwh", "euros_lost", "euros_ci_low", "euros_ci_high", "tariff_eur_per_kwh"):
        assert isinstance(f["loss"][num], (int, float)) and f["loss"][num] is not None
    cs = f["chart_series"]
    assert len(cs["dates"]) == len(cs["hero_pr"]) == len(cs["sibling_pr"]) > 0
    assert json.loads(json.dumps(f))["inverter_id"] == HERO
