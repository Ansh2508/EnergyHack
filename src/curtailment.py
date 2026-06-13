"""Curtailment masking from the DV grid-setpoint signal.

Encoding verified in A4 (outputs/schema_verification.txt): `DRD11A / DV (%)`
== 100 means full power; DV < 100 means the plant operator curtailed output
(e.g. negative electricity price). `DRD11A / EVU (%)` (grid-operator curtailment)
is effectively constant 100 in this plant and is ignored per spec. Curtailment
days are NEVER scored as inverter faults -- a curtailed inverter looks identical
to an outage without this track.
"""

from __future__ import annotations

import pandas as pd

DV_FRAC_THRESHOLD = 0.20  # day curtailed when >20% of daytime intervals have DV<100


def mask_curtailment(
    daily: pd.DataFrame, dv_frac_threshold: float = DV_FRAC_THRESHOLD
) -> pd.DataFrame:
    """Classify each inverter-day into reason ('fault' | 'curtailment' | 'ok').

    fault       : flagged_raw and a valid PR exists.
    curtailment : dv_frac > threshold -- overrides fault so curtailment is never
                  counted as a fault.
    ok          : everything else.
    `flagged` is True only for genuine faults.
    """
    out = daily.copy()
    out["reason"] = "ok"
    is_fault = out["flagged_raw"].fillna(False).astype(bool) & out["pr"].notna()
    out.loc[is_fault, "reason"] = "fault"
    out.loc[out["dv_frac"] > dv_frac_threshold, "reason"] = "curtailment"
    out["flagged"] = out["reason"] == "fault"
    return out
