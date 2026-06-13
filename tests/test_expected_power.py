"""Slice 3.5 tests - XGBoost expected-power model + independent loss cross-check.

Backend cross-check only (not in the agent path). Euros are range-tolerant.
"""

import json

import pytest

from src.expected_power import FEATURES, _rated, clean_training_frame, train_expected_power

HERO = "INV 01.05.029"


@pytest.fixture(scope="session")
def hero_model():
    """Train the hero expected-power model once for the metric tests."""
    model, res = train_expected_power(HERO)
    return {"model": model, "res": res}


def test_clean_training_filters_applied():
    """IEA Task 13: the training set has no night/curtailed/clipped/dead rows."""
    clean, rated = clean_training_frame(HERO, "2017-01-01", "2017-12-31")
    assert len(clean) > 1000
    assert (clean["sun_altitude"] > 0).all()
    assert (clean["irradiance"] >= 50).all()
    assert (clean["dv"].fillna(100) >= 100).all()  # no explicit curtailment
    assert (clean["p_ac"] > 0.1).all()  # no dead/zero rows
    assert (clean["p_ac"] < 0.98 * rated).all()  # no inverter clipping


def test_model_validates_on_clean_holdout(hero_model):
    """A model that predicts HEALTHY power well (R2>0.9, |MBE%|<3) can be trusted."""
    res = hero_model["res"]
    mbe_pct = abs(res.val_mbe / _rated(HERO) * 100.0)
    assert res.val_r2 > 0.9, res.val_r2
    assert mbe_pct < 3.0, mbe_pct


def test_feature_importance_irradiance_dominant(hero_model):
    """Feature consensus: irradiance is the dominant predictor."""
    fi = hero_model["res"].feature_importances
    assert set(fi) == set(FEATURES)
    assert max(fi, key=fi.get) == "irradiance"


def test_loss_cross_check_present_and_positive():
    """The independent cross-check is written, positive, with an agreement_pct."""
    with open("outputs/verified_facts.json", encoding="utf-8") as fh:
        facts = json.load(fh)
    lcc = facts["loss_cross_check"]
    assert lcc["method"] == "xgboost_weather"
    assert lcc["euros_lost"] > 0
    assert lcc["lost_kwh"] > 0
    assert lcc["causalimpact_euros"] > 0
    assert isinstance(lcc["agreement_pct"], (int, float))


def test_existing_suite_untouched():
    """Adding loss_cross_check left every existing field intact; agent_run.json valid."""
    with open("outputs/verified_facts.json", encoding="utf-8") as fh:
        facts = json.load(fh)
    for key in ("inverter_id", "plant", "window", "detection", "cause", "evidence",
                "loss", "business_case", "action", "validation", "chart_series",
                "narration_plain"):
        assert key in facts
    assert isinstance(facts["loss"]["euros_lost"], (int, float))
    assert "loss_cross_check" in facts  # the only addition
    with open("outputs/agent_run.json", encoding="utf-8") as fh:
        run = json.load(fh)
    assert "trace" in run and "result" in run
