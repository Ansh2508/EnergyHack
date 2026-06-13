"""Diagnose the top fleet events (read-only) and label each: genuine recoverable
fault vs deliberate/planned offline. Reuses diagnose.diagnose() EXACTLY -- no
reimplementation -- and reuses its U_DC/I_DC thresholds for the signal.

Reads outputs/fleet_scan.json, runs diagnose() on the top N events by euros,
clusters ALL events by shared start date (a section/combiner/transformer/planned
fingerprint), prints a table + clusters, and writes outputs/fleet_top_diagnostics.json.
Backend diagnostic only -- not in the agent decision path; touches no frozen file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from src import diagnose

FLEET_SCAN = "outputs/fleet_scan.json"
OUT_PATH = "outputs/fleet_top_diagnostics.json"
DEFAULT_TOP_N = 8
# CauseVerdict.errorcode_corroboration format: "<code> (<n>x in window): <german text>"
_EC_RE = re.compile(r"^\s*(\d+)\s*\(\s*(\d+)x in window\s*\)\s*:\s*(.*)$", re.S)


def parse_errorcode(corroboration: str | None) -> tuple[int | None, int, str | None]:
    """Split CauseVerdict.errorcode_corroboration into (code, count, text)."""
    if not corroboration:
        return None, 0, None
    m = _EC_RE.match(str(corroboration))
    if not m:
        return None, 0, str(corroboration)
    return int(m.group(1)), int(m.group(2)), m.group(3).strip()


def signal_for(
    primary_cause: str,
    u_dc_v: float | None,
    i_dc_a: float | None,
    errorcode_count: int,
) -> str:
    """Label the event from the DC rails + errorcode (thresholds reused from diagnose).

    DEAD_INVERTER   : dead-inverter cause AND U_DC present AND I_DC ~0 AND an error
                      logged -> panels live, inverter dead = genuine recoverable fault.
    POSSIBLE_OFFLINE: both rails dark (U_DC and I_DC ~0) OR nothing logged in the
                      window -> looks like a deliberate disconnection / decommission.
    REVIEW          : anything else (needs a human look).
    """
    has_ec = (errorcode_count or 0) > 0
    udc_present = u_dc_v is not None and u_dc_v > diagnose.UDC_PRESENT_V
    idc_zero = i_dc_a is not None and i_dc_a < diagnose.IDC_ZERO_A
    udc_zero = u_dc_v is not None and u_dc_v <= diagnose.UDC_PRESENT_V
    if primary_cause == "DEAD_INVERTER" and udc_present and idc_zero and has_ec:
        return "DEAD_INVERTER"
    if (udc_zero and idc_zero) or not has_ec:
        return "POSSIBLE_OFFLINE"
    return "REVIEW"


def diagnose_event(event: dict) -> dict:
    """Run diagnose() for one fleet event -> a flat verdict dict with a SIGNAL."""
    iid = event["inverter_id"]
    w = event["window"]
    v = diagnose.diagnose(iid, w["start"], w["end"])
    code, count, text = parse_errorcode(v.errorcode_corroboration)
    return {
        "inverter_id": iid,
        "window": w,
        "n_fault_days": event.get("n_fault_days"),
        "euros": event.get("euros"),
        "primary_cause": v.primary_cause,
        "side": v.side,
        "confidence": v.confidence,
        "u_dc_v": v.u_dc_v,
        "i_dc_a": v.i_dc_a,
        "u_dc_healthy_v": v.u_dc_healthy_v,
        "errorcode": code,
        "errorcode_count": count,
        "errorcode_text": text,
        "signal": signal_for(v.primary_cause, v.u_dc_v, v.i_dc_a, count),
    }


def find_clusters(ranked: list[dict]) -> list[dict]:
    """Group ALL events by window start date; return dates shared by >= 2 inverters."""
    by_start: dict[str, list[str]] = defaultdict(list)
    for e in ranked:
        by_start[e["window"]["start"]].append(e["inverter_id"])
    clusters = []
    for start, invs in sorted(by_start.items()):
        uniq = sorted(set(invs))
        if len(uniq) >= 2:
            clusters.append({"start": start, "n": len(uniq), "inverters": uniq})
    return clusters


def run(fleet_scan_path: str = FLEET_SCAN, top_n: int = DEFAULT_TOP_N) -> dict:
    """Diagnose the top-N events and cluster all of them (pure compute)."""
    with open(fleet_scan_path, encoding="utf-8") as fh:
        fs = json.load(fh)
    ranked = fs.get("ranked", [])
    verdicts = [diagnose_event(e) for e in ranked[:top_n]]
    return {
        "top_n": top_n,
        "data_span": fs.get("data_span"),
        "verdicts": verdicts,
        "clusters": find_clusters(ranked),
    }


def _print_report(r: dict) -> None:
    print(
        f"Top {r['top_n']} fleet events - diagnosis + signal (genuine fault vs planned offline)"
    )
    hdr = (
        f"{'inverter':<14} {'window':<24} {'d':>3} {'euros':>8} {'cause':<14} "
        f"{'cf':>4} {'U_DC':>6} {'I_DC':>6} {'errcode(n)':<14} SIGNAL"
    )
    print(hdr)
    print("-" * len(hdr))
    for v in r["verdicts"]:
        w = f"{v['window']['start']}..{v['window']['end']}"
        udc = f"{v['u_dc_v']:.0f}" if v["u_dc_v"] is not None else "-"
        idc = f"{v['i_dc_a']:.2f}" if v["i_dc_a"] is not None else "-"
        ec = f"{v['errorcode']}({v['errorcode_count']})" if v["errorcode"] else "-"
        print(
            f"{v['inverter_id']:<14} {w:<24} {v['n_fault_days']:>3} {v['euros']:>8.2f} "
            f"{v['primary_cause']:<14} {v['confidence']:>4.2f} {udc:>6} {idc:>6} "
            f"{ec:<14} {v['signal']}"
        )
    print()
    if r["clusters"]:
        print(
            "Shared-start clusters (>=2 inverters, same start date -> shared cause: section/combiner/planned):"
        )
        for c in r["clusters"]:
            print(f"  {c['start']}: {c['n']} inverters -> {', '.join(c['inverters'])}")
    else:
        print("No shared-start clusters (no start date shared by >=2 inverters).")


def write_diagnostics(r: dict, out_path: str = OUT_PATH) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(r, fh, indent=2, ensure_ascii=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Diagnose + label the top fleet events.")
    ap.add_argument("-n", "--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--fleet-scan", default=FLEET_SCAN)
    args = ap.parse_args(argv)
    r = run(args.fleet_scan, args.top_n)
    write_diagnostics(r)
    _print_report(r)
    return r


if __name__ == "__main__":
    main()
