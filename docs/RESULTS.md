# Slice 2 results - committed proof (Plant A, INV 01.05.029)

Real run on ENERPARC Plant A (65 Refu inverters, ~1.9 MWp, Silmersdorf/Brandenburg),
2017-2026. `outputs/` is gitignored, so the proof is committed here.

## The fault: INV 01.05.029, 2019-05-24 -> 2019-06-16

Ten sustained days at PR=0 while the 64 sibling inverters held ~0.83 in the same weather.

| Date | hero PR | sibling PR | reason |
|---|---|---|---|
| 2019-05-24 | 0.536 | 0.837 | fault |
| 2019-05-25 | 0.0 | 0.878 | fault |
| 2019-05-26 | 0.0 | 0.844 | fault |
| 2019-05-27 | 0.0 | 0.828 | fault |
| 2019-05-28 | 0.0 | 0.851 | fault |
| 2019-05-29 | 0.0 | 0.859 | fault |
| 2019-05-30 | 0.0 | 0.858 | fault |
| 2019-05-31 | 0.0 | 0.828 | fault |
| 2019-06-01 | 0.0 | 0.826 | fault |
| 2019-06-02 | 0.0 | 0.832 | fault |
| 2019-06-03 | 0.0 | 0.835 | fault |
| 2019-06-04 | 0.397 | 0.802 | fault |
| 2019-06-05 | 0.833 | 0.813 | ok |
| 2019-06-06 | 0.829 | 0.81 | ok |
| 2019-06-07 | 0.846 | 0.824 | ok |
| 2019-06-08 | 0.855 | 0.836 | ok |
| 2019-06-09 | 0.851 | 0.835 | ok |
| 2019-06-10 | 0.852 | 0.823 | ok |
| 2019-06-11 | 0.836 | 0.811 | ok |
| 2019-06-12 | 0.836 | 0.811 | ok |
| 2019-06-13 | 0.839 | 0.82 | ok |
| 2019-06-14 | 0.84 | 0.824 | ok |
| 2019-06-15 | 0.833 | 0.802 | ok |
| 2019-06-16 | 0.85 | 0.824 | ok |

## Diagnosis - cause and the AC/DC split (the differentiator)

- **primary_cause: DEAD_INVERTER, side AC, confidence 0.98**
- U_DC **793 V** during the outage (healthy ~707 V) - panels sit at open-circuit voltage
- I_DC **0.14 A** (~0) - the inverter is drawing no current
- inverter error **655626 x1947**: "Erkennung von Netzunterspannung (ENS,Leistungsteil)"
- => panels healthy, **inverter (AC side) failed**; the inverter's own log is independent confirmation.

## Loss - sibling-controlled counterfactual with a 95% CI

- lost: **1784.4 kWh -> EUR 205.21** (95% CI EUR 158.10 - 254.26)
- tariff **0.115 EUR/kWh**, read from feed-in-tarrifs.xlsx (EEG-fixed)
- method: **causalimpact** (Brodersen-2015 CausalImpact path is implemented; runs on a TF-capable machine)

## Honest business case

- single-event ROI **0.13x** (EUR 205 vs EUR 1500 dispatch) - one event does not pay for itself
- the value is catching it in **10 days vs months**, across a 3.8 GW portfolio
- projection (labelled, NOT measured): **EUR 3.3 - 10.0 M/yr** (3.8 GW x 1-3% silent loss x tariff; capacity factor 0.087 derived from Plant A 2021)

## Validation - detections vs real ENERPARC service tickets (2019)

| Window | Komponente affected | Detected (>=3 consec) | Precision-matched |
|---|---|---|---|
| 2019-01-08 -> 2019-01-24 | 1 | 24 | 1 |
| 2019-05-24 -> 2019-06-16 | 1 | 8 | 1 |
| 2019-07-08 -> 2019-07-14 | 4 | 11 | 4 |
| 2019-08-02 -> 2019-08-08 | 2 | 1 | 1 |
| 2019-09-10 -> 2019-09-17 | 4 | 5 | 4 |

Tickets name the component + affected count, not the inverter; the detector supplies the
identity. The recommended hero (INV 01.05.029, May window, single affected) is the cleanest match.

## Reproduce

```
python -m src.run_slice1     # -> outputs/detection_daily.parquet + hero_candidates.md
python -m src.build_facts    # diagnose + quantify -> outputs/verified_facts.json
pytest tests/ -v             # 13/13
```

