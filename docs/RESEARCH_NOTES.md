# Research notes - ENERPARC Reliability Agent (Slice 2 grounding)

Each method is verified against its primary source before use. The one-line
citation is mirrored as a code comment in the named module. ASCII only.

## 1. CausalImpact / BSTS - Brodersen et al. 2015 (arXiv:1506.00356) -> quantify.py
Counterfactual = a Bayesian structural time-series fit on the pre-period, then
forecast across the post-period; the cumulative effect is sum(actual - predicted)
with a posterior credible interval (alpha=0.05 -> 95%).
Verified Python port `tfcausalimpact` (WillianFuks), README-confirmed:
    from causalimpact import CausalImpact
    ci = CausalImpact(data, pre_period, post_period, model_args={'fit_method':'vi'})
    # data: response y MUST be column 0; covariates X are the remaining columns.
    # pre_period/post_period = [start, end] (index labels or positions).
    # cumulative loss magnitude + 95% CI (verified attribute path):
    ci.summary_data.loc['abs_effect',       'cumulative']   # signed effect; |.| = lost kWh
    ci.summary_data.loc['abs_effect_lower', 'cumulative']   # CI low
    ci.summary_data.loc['abs_effect_upper', 'cumulative']   # CI high
Deps: tensorflow>=2.10 + tensorflow_probability>=0.18 (HEAVY); 'vi' fit ~2-3 min.
RISK/DEVIATION: install is large and may exceed sandbox limits; VI can fail to
converge with few/collinear controls. Phase D therefore ships a labelled
fallback (method='sibling_sigma') and never misrepresents which path ran.

## 2. Soiling SRR - Deceglie et al. 2018, IEEE JPV 8(2):547 -> diagnose.py (soiling branch)
    rdtools.soiling.soiling_srr(energy_normalized_daily, insolation_daily,
        reps=1000, confidence_level=68.2, ...) -> (sr_p50, sr_confidence_interval, soiling_info)
Inputs are daily pandas Series (DatetimeIndex): a normalized performance index
and the matching plane-of-array insolation. Returns the insolation-weighted
soiling ratio P50 + CI. Use confidence_level=95 for a 95% band. Applies ONLY to
gradual sawtooth decline (slow soiling + cleaning/rain recovery) - a step-to-zero
outage is not soiling.

## 3. Clipping - Perry, Muller & Anderson 2021 (IEEE PVSC 48) -> diagnose.py (clipping branch)
    rdtools.filtering.clip_filter(power_ac, model='logic')  # == logic_clip_filter
Boolean mask flagging intervals where the rolling-window maximum range is below
rolling_range_max_cutoff (default 0.2): power is flat at the TOP (near-zero
derivative) = inverter saturation/clipping. Principle used here: clipped power
plateaus at the top, whereas a dead inverter is flat at ZERO - opposite ends.

## 4. Validate-before-show - arXiv:2606.01513 (Compliance-Scored Best-of-N) -> build_facts.py
Kernel: every emitted/displayed fact is scored/asserted against its computed
source before release. build_facts.py asserts each JSON number == its computed
source value and raises on mismatch, so no number reaches narration unchecked.

## 5. Structured evidence for RCA - Roy et al. 2024, Microsoft (arXiv:2403.04123) -> diagnose.py
ReAct-over-tools RCA gains factual accuracy when each tool returns structured,
inspectable evidence (typed fields + an evidence list) rather than prose. This
shapes CauseVerdict: typed primary_cause / side / confidence + evidence:list[str].

## 6. Named auditable skills + lesson-from-failure - SkillRL, arXiv:2602.08234 -> diagnose.py
Structure only (no RL, no GPU): each diagnostic rule is a named, auditable
function; the curtailment mask is treated as a "lesson-from-failure" guard - a
rule kept because curtailment once looked indistinguishable from an outage.

## ROI assumption (used in Phase D, build_facts.action.roi_multiple)
Documented NON-data assumption: one inverter-fault dispatch (technician travel +
diagnosis + service/replacement of a ~30 kWp string inverter) ~= EUR 1500.
roi_multiple = euros_lost / 1500. Declared here because it is an assumption, not
a value read from plant data. The feed-in TARIFF, by contrast, IS read from
data/Plant A (start here)/2. Additional Data/feed-in-tarrifs.xlsx (~0.077 EUR/kWh
for the 2019-05/06 window) - never hardcoded.
NOTE: a single 30 kWp / ~12-day summer outage is a modest euro figure on its own;
the headline value is portfolio-scale (3.8 GW x 1-3% silent loss) plus the days
saved by catching it early. Phase D reports the true single-event number.

## Portfolio projection formula (Phase D, build_facts.business_case)
PROJECTION, not measured. Computed, never merged with the real single-event euros:
  portfolio_kwh_per_year = PORTFOLIO_KW * 8760 h * capacity_factor
  loss_kwh[low,high]     = portfolio_kwh_per_year * [0.01, 0.03]   (1-3% silent loss)
  portfolio_eur[low,high]= loss_kwh[low,high] * tariff_eur_per_kwh
Inputs: PORTFOLIO_KW = 3.8e6 (3.8 GW ENERPARC fleet); capacity_factor is DERIVED
from Plant A 2021 actuals (sum of daily kWh / (plant kWp * 8760)), not assumed;
tariff read from feed-in-tarrifs.xlsx (0.115 EUR/kWh for the 2019 window).

## LangGraph v1.0 (Slice 3 - the deterministic investigation agent) -> src/agent.py
Verified empirically against the installed langgraph 1.2.5 (NOT from v0.1 tutorials):
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(AgentState)            # TypedDict state, channels merged per key
    g.add_node("observe", observe_detect)
    g.add_edge(START, "observe")          # explicit START edge (NOT set_entry_point - deprecated v0.1)
    g.add_conditional_edges("observe", route_after_detect, {"fault": "triage", "healthy": END})
    app = g.compile(); state = app.invoke(initial_state)
    mermaid = app.get_graph().draw_mermaid()   # -> docs/agent_graph.mmd (text only; no PNG/network)
Verified signature: add_conditional_edges(source: str, path: Callable[..., Hashable],
    path_map: dict[Hashable, str] | None = None). path returns a branch key; path_map maps it
    to the next node (or END). compile() validates; invoke(state) runs to a terminal node.

DESIGN: routing is deterministic plain Python - NO LLM, NO API in the decision path
(auditability > black box). An LLM-routed variant is a one-line swap (path=an LLM call) but is
deliberately not used.

Roy et al. 2024 (arXiv:2403.04123): route_after_detect / route_after_triage AND every TraceEvent
read TYPED tool fields (CauseVerdict.primary_cause, LossEstimate.euros_*, sustained-zero-day count,
curtailment fraction) - never prose. The typed trace (outputs/agent_run.json) is the agent's
structured observability layer and the UI animation contract.

SkillRL (arXiv:2602.08234): each node is a named skill wrapping an existing Slice-1/2 tool; the
curtailment triage is the lesson-from-failure skill that can TERMINATE the investigation early
(reject -> END) so the agent never wastes the Bayesian model on a non-fault.

Brodersen 2015 / RdTools (Deceglie, Perry): reused UNCHANGED via the diagnose/quantify nodes;
proven by the Slice-2 tests still passing.

## Phase B research-in-code mapping (Slice 3 agent)
| Source | Made true in code | Proven by test |
|---|---|---|
| LangGraph v1.0 (1.2.5) | StateGraph + add_conditional_edges (no set_entry_point); draw_mermaid -> docs/agent_graph.mmd | graph compiles; test idiom confirmed empirically |
| Roy 2024 (arXiv:2403.04123) | routing AND trace read TYPED fields, never prose | test_routing_reads_typed_fields, test_trace_matches_result |
| SkillRL (arXiv:2602.08234) | curtailment guard = lesson-from-failure skill; terminates early | test_agent_curtailment_short_circuits, test_agent_healthy_short_circuits |
| Validate-before-show (arXiv:2606.01513) | trace + work order computed ONCE from the same verdict/loss | test_trace_matches_result, test_quantify_runs_once |
| Brodersen 2015 / RdTools | engine reused unchanged via diagnose/quantify nodes | the 7 Slice-2 tests still green |

## Expected-power model (XGBoost) - Slice 3.5 (independent loss cross-check; NOT in the agent path)
Phase-0 facts: xgboost 3.2.0, shap 0.49.1 (available). Year-1 = 2017 (105,121 rows; first FULL
calendar year - 2016 is a 24-row Dec-31 tail). Native-parquet columns: irradiance =
'Plant / Irradiation_average (W/m2)', sun_altitude = 'Plant / Altitude', module_temp + ambient_temp
= 'Temperature Sensor / Module|Ambient', DV curtailment. Cross-MODULE-TYPE comparison set (the hero
is one of only two full-size Module Type 6 units, so same-type siblings are not used here):
  INV 01.05.029 28.8 kWp (hero), INV 01.01.001 30.6 (fleet baseline), INV 01.03.018 30.0,
  INV 01.07.044 28.2.

Method (IEA PVPS Task 13 convention; web-confirmed):
- TRAIN ONLY ON CLEAN POINTS. Remove night/low-irradiance (sun_altitude<=0 OR irradiance<50 W/m2),
  curtailment (DV<100, reusing the Slice-1 mask logic), inverter clipping (P_AC >= 98% of rated kW),
  and zero/fault rows (P_AC<=0.1 kW while producing). Training on contaminated data would teach the
  model dead/throttled behaviour as "normal". (IEA-PVPS T13 Performance-Loss-Rate report; Sandia PVPMC.)
- Features = [irradiance, module_temp, ambient_temp, sun_altitude]; target = instantaneous P_AC (kW).
  Irradiance + module temp dominate (consensus).
- Residual (actual - expected) is a MIXTURE (thermal/spectral/electrical + soiling + degradation +
  outage + clipping + curtailment). NOT all residual is a fault. The loss cross-check integrates the
  POSITIVE shortfall (expected-actual) ONLY over the already-detected Slice-1/2 fault window, and only
  over clean-weather producing intervals.
- XGBoost weaknesses are explainability + false alarms; both mitigated: emit feature_importances_
  (gain) and mean|SHAP|; and this model NEVER raises an alarm - it only cross-checks an already
  detected event. Shallow trees (max_depth 5), early stopping on a time-held-out clean split.
Refs: IEA-PVPS Task 13 (Assessment of Performance Loss Rate of PV Power Systems, 2021); Sandia PVPMC.
