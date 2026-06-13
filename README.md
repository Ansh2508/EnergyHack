# ENERPARC Reliability Agent

An autonomous investigator for utility-scale solar. It finds the inverter that is
silently losing money, proves WHY it failed (down to AC vs DC side), quantifies the
loss in euros with a confidence interval, recommends the fix, and validates every
detection against the plant's own service tickets.

Built for Energy Hack Munich 2026 - ENERPARC Open Track.

> Monitoring is the smoke alarm. This is the investigator that arrives after the
> alarm and tells you what is burning, what it costs, and what to do.

## The problem

ENERPARC runs ~3.8 GW of solar across hundreds of plants. Existing monitoring says
WHEN output drops. It does not say WHICH inverter, WHY, WHAT it cost, or WHAT to do
next. A single inverter can sit dead for weeks while plant output only looks "a
little low" - and no one notices.

## Proof - a real fault, found from telemetry alone, matched to a real ticket

Plant A (65 inverters, ~1.9 MWp, Silmersdorf), INV 01.05.029, with zero access to
the service log during detection:

```
date         hero PR   sibling PR   reason
2019-05-24   0.54      0.84         fault
2019-05-25   0.00      0.88         fault   <- inverter dies
 ...         0.00      ~0.84        fault   (10 sustained zero-output days)
2019-06-04   0.40      0.80         fault   <- recovering
2019-06-05   0.83      0.81         ok      <- back to health
```

- **Cause: DEAD_INVERTER, AC side, confidence 0.98.** During the outage U_DC rises
  to 793 V (open circuit) while I_DC ~ 0 A -> the panels are healthy and the
  inverter failed. The inverter's own log fired error 655626
  ("Erkennung von Netzunterspannung, ENS") 1947 times - independent confirmation.
- **Loss: 1,784.4 kWh -> EUR 205.21 (95% CI EUR 158.10 - 254.26)**, tariff
  0.115 EUR/kWh read from the file, method `causalimpact` (sibling_sigma is the automatic fallback when TensorFlow is absent).
- **Validation: matches ENERPARC ticket 2019-05-24 -> 2019-06-16** (one inverter
  affected), found independently.

Full committed proof: [docs/RESULTS.md](docs/RESULTS.md). Design:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Honest business case

Single event EUR 205 - on its own it does not pay back a service dispatch. The value
is **catching it in 10 days instead of months, across a 3.8 GW portfolio** (projected
EUR 3.3 - 10 M/yr; labelled projection, not a measured number).

## What makes it different

| Differentiator | Why it matters |
|---|---|
| Curtailment masked before fault scoring | a grid/market throttle is indistinguishable from an outage without it |
| Cause with an AC vs DC split | tells the technician what to physically inspect |
| Sibling-controlled causal euros with a 95% CI | a number a CFO can sign off, not a ratio |
| Validated against real service tickets | precision, not a self-reported confidence score |

## Pipeline

```
Observe -> Detect -> Filter curtailment -> Diagnose -> Quantify -> Act -> Validate
```

## Run

```
pip install -r requirements.txt
python -m src.run_slice1     # detection_daily.parquet + hero_candidates.md
python -m src.build_facts    # diagnose + quantify -> outputs/verified_facts.json
pytest tests/ -v             # 13/13
```

## Grounded in research (each method cited in code)

- Performance Ratio - IEC 61724
- Causal loss + 95% CI - Brodersen et al. 2015 (CausalImpact / BSTS)
- Soiling detection - Deceglie et al. 2018 (RdTools `soiling_srr`)
- Clipping detection - Perry et al. 2021 (RdTools `clip_filter`)
- Structured RCA tools - Roy et al. 2024 (arXiv:2403.04123)
- Validate-before-show - arXiv:2606.01513 ; named auditable skills - SkillRL arXiv:2602.08234

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Ingest + detect + curtailment mask + ticket-validated hero match | **Done** |
| 2 | Diagnose cause (AC/DC) + causal euros with CI + verified-facts JSON | **Done** |
| 3 | ReAct agent over the tools (acts, not just answers) | Planned |
| 4 | Work-order card (renders verified_facts.json) | Planned |

*Plant A data is proprietary to ENERPARC and is not included in this repository.*


