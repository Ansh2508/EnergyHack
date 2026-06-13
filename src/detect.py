"""Detection: performance ratio, daily aggregation, sibling baseline, flagging.

Performance Ratio (PR) per IEC 61724 -- the daily AC performance ratio is
PR = Y_f / Y_r, where Y_f = AC energy / rated DC power and Y_r = in-plane
insolation / reference irradiance (1000 W/m^2). At interval level this reduces
to PR = P_AC / (P_dc_rated * G / 1000), the form used here.
Refs: IEC 61724-1; Sandia PVPMC "Performance Ratio"
(https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/).

Slice 2 (not implemented here) will add:
  - Soiling: rdtools.soiling.soiling_srr(energy_normalized_daily, insolation_daily,
    ...) -- Deceglie et al., IEEE JPV 8(2) p547, 2018 (stochastic rate & recovery).
  - Counterfactual: causalimpact.CausalImpact(data, pre, post,
    model_args={'fit_method': 'vi'}) -- Brodersen et al. 2015 (BSTS).
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

NIGHT_ALTITUDE = 5.0  # Plant/Altitude (deg): keep daytime intervals only
MIN_IRRADIANCE = 50.0  # W/m^2: drop low-light intervals from the PR average
RESIDUAL_THRESHOLD = -0.07  # flag when inverter PR is >7pp below sibling median


def compute_pr(
    con: duckdb.DuckDBPyConnection,
    meta: pd.DataFrame,
    min_irradiance: float = MIN_IRRADIANCE,
) -> duckdb.DuckDBPyRelation:
    """Register `base_day`: daytime interval rows with per-interval PR.

    Streams a single P_AC UNPIVOT (lazy) joined to env + meta. PR is NULL where
    irradiation < min_irradiance (those intervals still count toward daily_kwh
    and dv_frac). Returns the base_day relation.
    """
    con.register("meta_df", meta)
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT * FROM meta_df")
    cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM mon_wide").fetchall()]
    pac = ",".join(f'"{c}"' for c in cols if re.search(r"INV \d+\.\d+\.\d+ / P_AC", c))
    con.execute(
        f"""CREATE OR REPLACE VIEW base_day AS
        WITH long AS (
            SELECT timestamp AS ts,
                   regexp_extract(name, 'INV [0-9.]+') AS inverter_id,
                   value AS p_ac
            FROM (UNPIVOT mon_wide ON {pac} INTO NAME name VALUE value)
        )
        SELECT l.ts, l.inverter_id, m.orientation, m.kwp, l.p_ac,
               e.irradiation, e.altitude, e.dv,
               CASE WHEN e.irradiation >= {min_irradiance}
                    THEN l.p_ac / (m.kwp * (e.irradiation / 1000.0)) END AS pr
        FROM long l
        JOIN mon_env e ON e.ts = l.ts
        JOIN meta m ON m.inverter_id = l.inverter_id
        WHERE e.altitude > {NIGHT_ALTITUDE}"""
    )
    return con.view("base_day")


def daily_aggregate(
    con: duckdb.DuckDBPyConnection, step_seconds: int = 300
) -> pd.DataFrame:
    """Aggregate base_day to one row per inverter-day.

    pr = mean daytime PR (NULL intervals ignored); daily_kwh = sum(P_AC)*step/3600;
    dv_frac = curtailed daytime fraction (DV < 100); insol_kwhm2 = in-plane
    insolation; n_pr = count of intervals contributing to the PR mean.
    """
    return con.execute(
        f"""SELECT inverter_id, CAST(ts AS DATE) AS date,
               any_value(orientation) AS orientation,
               avg(pr) AS pr,
               sum(p_ac) * {step_seconds} / 3600.0 AS daily_kwh,
               avg(CASE WHEN dv < 100 THEN 1.0 ELSE 0.0 END) AS dv_frac,
               sum(irradiation) * {step_seconds} / 3600 / 1000.0 AS insol_kwhm2,
               any_value(kwp) AS kwp,
               count(*) FILTER (WHERE pr IS NOT NULL) AS n_pr
        FROM base_day
        GROUP BY 1, 2
        ORDER BY 1, 2"""
    ).df()


def sibling_baseline(
    daily: pd.DataFrame, residual_threshold: float = RESIDUAL_THRESHOLD
) -> pd.DataFrame:
    """Add sibling_pr (median PR per date+orientation), residual, flagged_raw.

    A NULL PR (no valid daytime intervals) yields NULL residual and is not
    flagged (NaN < threshold is False).
    """
    out = daily.copy()
    out["sibling_pr"] = out.groupby(["date", "orientation"])["pr"].transform("median")
    out["residual"] = out["pr"] - out["sibling_pr"]
    out["flagged_raw"] = out["residual"] < residual_threshold
    return out
