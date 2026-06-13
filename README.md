# ENERPARC Reliability Agent

**An autonomous investigator for utility-scale solar.** It finds the inverter that is silently losing money, proves *why* it failed, quantifies the loss in euros with confidence bounds, and recommends the fix — then validates every detection against the plant's own service tickets.

Built for **Energy Hack Munich 2026 — ENERPARC Open Track.**

> Monitoring is the smoke alarm. This is the investigator that arrives after the alarm and tells you what is burning, what it costs, and what to do.

---

## The problem

ENERPARC operates ~3.8 GW of solar across hundreds of plants. Existing monitoring tells operators *when* output drops. It does not tell them **which** inverter, **why**, **what it cost**, or **what to do next**. A single inverter can sit dead for weeks while plant output merely looks "a little low" — and no one notices. At portfolio scale, 1-3% of silent underperformance is **millions of euros per year**, and an emerging risk: 2027 EU rules push data centres toward 100% renewable matching, where every lost MWh is an SLA exposure.

## What it does

A **deterministic** pipeline — physics and proven algorithms, fully auditable, **no model training required**, runs on the existing SCADA feed:

```
Observe -> Detect -> Filter curtailment -> Diagnose -> Quantify -> Act -> Validate
```

| Stage | What happens |
|---|---|
| **Observe** | Read 5-min inverter telemetry (P_AC, I_DC, U_DC), irradiance, module temp |
| **Detect** | Compute Performance Ratio (IEC 61724); flag inverters underperforming their orientation-matched siblings in the same weather |
| **Filter** | Mask grid curtailment (DV signal) *before* scoring faults — never cry wolf on a market throttle |
| **Diagnose** | Classify cause: dead inverter vs soiling vs clipping vs thermal; split AC-side vs DC-side using I_DC / U_DC; corroborate with the Refu error log |
| **Quantify** | Bayesian counterfactual (CausalImpact) using healthy siblings as controls -> lost kWh with 95% CI -> euros via the real feed-in tariff |
| **Act** | Emit a work order: cause, euro loss, recommended action, ROI multiple |
| **Validate** | Cross-match detections to real service tickets — precision, not a confidence score |

## Proof: it found a real fault, matched to a real ticket

Run on ENERPARC **Plant A** (65 Refu inverters, 1.9 MWp, Silmersdorf/Brandenburg, 2017-2026) — with **zero access to the service log during detection**:

```
INV 01.05.029 - May/June 2019
  date         PR     sibling_PR   reason
  2019-05-24   0.54     0.84        fault
  2019-05-25   0.00     0.88        fault   <- inverter dies
  2019-05-26   0.00     0.84        fault
  ...          0.00     0.84        fault   (10 sustained zero-output days)
  2019-06-03   0.00     0.84        fault
  2019-06-04   0.40     0.80        fault   <- recovering
  2019-06-05   0.83     0.81        ok      <- back to health
```

The 65 sibling inverters held a Performance Ratio of ~0.84 the entire time. This inverter produced **nothing for 10 days** — invisible in plant-level output, obvious against its peers.

**This matches ENERPARC service ticket 2019-05-24 -> 2019-06-16** (one inverter affected) — same inverter, same window, recovered just before the ticket closed. Found independently from telemetry alone. That is not a model score. That is ground truth.

## Why this is hard — and what makes it different

Reading the data is 10% of the problem. The other 90% is knowing which signals to trust and proving you are right. Three things no dashboard does:

| Differentiator | Why it matters |
|---|---|
| **Curtailment masked before fault scoring** | Grid throttling is indistinguishable from an outage without the DV/EVU signals. We mask it first. Most approaches alarm on it and lose operator trust instantly. |
| **Causal euros with a 95% CI** | Not a ratio. A Bayesian structural time-series counterfactual answers "what would this inverter have made if healthy" — with error bars a CFO can sign off on. |
| **Validated against real tickets** | Detection precision is measured against the maintenance team's own records, not claimed. The ticket match *is* the proof. |

## Architecture

```
src/
  ingest.py        German-CSV / native-parquet loader, 5-min -> daily, sibling grouping
  detect.py        IEC 61724 Performance Ratio + sibling-residual fault flagging
  curtailment.py   DV-signal mask (grid throttle != fault)
  hero_match.py    cross-match fault runs to 2019 service tickets, ranked
  diagnose.py      cause classification + AC/DC split + error-log corroboration   [Slice 2]
  quantify.py      CausalImpact euro loss with 95% CI                              [Slice 2]
  build_facts.py   verified-facts JSON (single source of truth for the UI)        [Slice 2]
  run_slice1.py    orchestrator
tests/             content-asserting tests (not just "it ran")
```

Computation lives entirely in Python. The narration layer (planned) only *describes* verified facts — it is never allowed to compute or invent a number.

## Run it

```bash
# Place Plant A data under data/Plant A (start here)/
pip install -r requirements.txt
python -m src.run_slice1      # -> outputs/hero_candidates.md + detection_daily.parquet
pytest tests/ -v              # content-asserting test suite
```

## Grounded in research

Every method is anchored to a primary source, cited in code:

- **Performance Ratio** — IEC 61724 (Yield_f / Yield_r)
- **Causal loss quantification** — Brodersen et al. 2015, *Inferring causal impact using Bayesian structural time-series models*
- **Soiling detection** — Deceglie et al. 2018, RdTools stochastic rate-and-recovery
- **Agent design** — Roy et al. 2024 (Microsoft), ReAct-over-tools reduces factual error vs RAG for root-cause analysis

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Ingest + detect + curtailment mask + ticket-validated hero match | **Done, proven on real data** |
| 2 | Diagnose cause + CausalImpact euros + verified-facts JSON | In progress |
| 3 | Single ReAct agent tying the tools together + narration guard | Planned |
| 4 | Work-order UI (on-device narration) | Planned |

---

*Plant A data is proprietary to ENERPARC and is not included in this repository.*
