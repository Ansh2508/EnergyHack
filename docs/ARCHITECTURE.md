# Architecture

Deterministic pipeline over the existing SCADA feed. Every number that reaches a
human is computed in Python and asserted against its source; no LLM sits in the
decision path, and no model raises an alarm.

```
Observe -> Detect -> Filter curtailment -> Diagnose -> Quantify -> Cross-check -> Act -> Validate
```

## Data contract (module -> input -> output schema)

| Module | Input | Output |
|---|---|---|
| `ingest.load_monitoring` | Plant A native parquet (CSV fallback) | DuckDB views `mon_wide` / `mon_env` / `mon_long` (timestamp, inverter_id, p_ac, i_dc, u_dc) |
| `ingest.load_meta` | `System_Overview.xlsx` | DataFrame[inverter_id, kwp, orientation] |
| `detect.compute_pr` -> `daily_aggregate` -> `sibling_baseline` | mon_* views + meta | daily DataFrame (pr, daily_kwh, dv_frac, insol_kwhm2, sibling_pr, residual, flagged_raw) |
| `curtailment.mask_curtailment` | daily DataFrame | + `reason` ('fault'\|'curtailment'\|'ok'), `flagged` |
| `hero_match.cross_match` | daily + `Tickets.xlsx` | ranked candidates -> `HeroCandidate` |
| `diagnose.diagnose` | monitoring cache + `errorcodes.parquet` + dict | `CauseVerdict` (primary_cause, side, confidence, u_dc_v, i_dc_a, u_dc_healthy_v, errorcode_corroboration) |
| `quantify.quantify_loss` | `detection_daily.parquet` + `feed-in-tarrifs.xlsx` | `LossEstimate` (lost_kwh, euros + 95% CI, method, pre/post) |
| `expected_power.train_expected_power` | native parquet, IEA-clean rows only | XGBoost model + `ExpectedPowerResult` (val_r2, val_mbe, feature importances) |
| `build_facts.build_facts` | all of the above | `outputs/verified_facts.json` (after the validate-before-show gate) |
| `agent.run_investigation` | inverter id + window | `outputs/agent_run.json` (typed, replayable trace + result) |

Schemas are Pydantic V2 (`src/schemas.py`, `CauseVerdict`, `LossEstimate`,
`ExpectedPowerResult`).

## The differentiators (and the paper behind each)

| Differentiator | Where | Grounded in |
|---|---|---|
| Curtailment masked **before** fault scoring (a throttle is not an outage) | `curtailment.py` + `diagnose` priority-0 guard | DV signal; lesson-from-failure skill (SkillRL, arXiv:2602.08234) |
| Cause with an **AC/DC side split** (tells the technician what to inspect) | `diagnose.py` via I_DC / U_DC | Refu error-log corroboration |
| **Sibling-controlled causal euros with a 95% CI** (not a ratio) | `quantify.py` | Brodersen et al. 2015 (BSTS / CausalImpact) |
| **Independent weather-ML loss cross-check** (no shared assumptions) | `expected_power.py` | XGBoost on IEA PVPS Task 13 clean data |
| Detections **validated against real service tickets** (precision, not a score) | `hero_match.py` | - |

Soiling and clipping branches call real `rdtools` (`soiling_srr`, Deceglie 2018;
`clip_filter('logic')`, Perry 2021).

## Computation vs narration

Computation lives entirely in Python and lands in `verified_facts.json` /
`agent_run.json`. The Slice-4 card only *renders* that JSON:
`build_facts.assert_facts_consistent` asserts every emitted number equals its
computed source and raises on mismatch, so no number reaches the narration layer
unchecked (validate-before-show, arXiv:2606.01513). The card reads every value at
runtime via `fetch('agent_run.json')` and hard-codes nothing.

The Slice-3 agent *acts* on typed tool outputs (`CauseVerdict`, `LossEstimate`)
rather than answering from prose (structured RCA tools, Roy et al. 2024,
arXiv:2403.04123). Its routing is deterministic plain Python; an LLM-routed variant
is a one-line swap (pass an LLM call as the `add_conditional_edges` path function),
but we keep routing deterministic on purpose so the trace is auditable and
reproducible.

## Agent (Slice 3) - deterministic investigation graph

A LangGraph v1.0 `StateGraph` orchestrates the Slice 1/2 tools into one end-to-end
investigation and writes a typed, replayable trace (`outputs/agent_run.json`,
consumed by the Slice-4 card). The decision path is DETERMINISTIC plain Python: the
routing functions read typed tool outputs (sustained-zero-day count,
`CauseVerdict.side`, curtailment flag) and branch, so every step is auditable and no
LLM can hallucinate the diagnosis. The curtailment triage is a lesson-from-failure
guard that terminates the investigation early, never wasting the Bayesian model on a
throttle. `CauseVerdict` is the single authority for the DC numbers - the same
`u_dc_v` / `i_dc_a` / `u_dc_healthy_v` flow into the trace, the facts, and the card.

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	observe(observe)
	triage(triage)
	diagnose(diagnose)
	quantify(quantify)
	act(act)
	__end__([<p>__end__</p>]):::last
	__start__ --> observe;
	diagnose --> quantify;
	observe -. &nbsp;healthy&nbsp; .-> __end__;
	observe -. &nbsp;fault&nbsp; .-> triage;
	quantify --> act;
	triage -. &nbsp;curtailment&nbsp; .-> __end__;
	triage -. &nbsp;real_fault&nbsp; .-> diagnose;
	act --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Independent loss cross-check (Slice 3.5) - off the decision path

`expected_power.py` trains a weather-driven XGBoost expected-power model on the
hero's own *clean healthy* history (IEA PVPS Task 13 filtering: night, low-light,
curtailed and clipped intervals removed before fitting). It predicts what the
inverter *should* have produced from irradiance, module temperature, ambient
temperature and sun altitude, and prices the outage as the summed positive
shortfall over the fault window. This is a second counterfactual that shares no
assumptions with the sibling-controlled Bayesian method: one uses *peer inverters*
as controls, the other uses *weather physics*. On the hero event the two land at
EUR 194.63 (Bayesian) vs EUR 202.15 (weather ML) -> 96.3% agreement, with the ML
estimate inside the Bayesian 95% interval.

This model is **backend-only**: it never sits in the agent decision path and never
raises an alarm. It only re-prices an event the deterministic agent already
detected, and it self-checks before it is trusted (`val_r2` must clear 0.9 on a
clean holdout; a unit it cannot predict when healthy is flagged, not used to price
when unhealthy). The same module also emits an annual degradation series
(`degradation_hero.json`) and a cross-module-type benchmark
(`expected_power_benchmark.json`).

## Card (Slice 4) - the incident report

`apps/card/index.html` is one self-contained file (no build, no server, no key) for
GitHub Pages. It reads `agent_run.json` at runtime and *demonstrates* the
investigation rather than describing it: a 65-tile fleet array surfaces the single
dead inverter, a hand-rolled SVG instrument chart draws the hero PR cratering to
zero against a flat healthy fleet, the seven reasoning steps reveal one by one to
the ticket match, and the two independent euro estimates render side by side with
their agreement. All motion is mapped to investigation logic and honours
`prefers-reduced-motion`.
