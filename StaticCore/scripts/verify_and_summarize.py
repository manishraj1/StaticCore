#!/usr/bin/env python3
"""
verify_and_summarize.py — recompute every aggregate published in the README
and paper draft directly from data/*.json, and assert them.

If this script exits 0, every number in the README's tables is derivable
from the raw records in this repository. If any assertion fails, either the
data or the prose is wrong — fix whichever one is lying.

Usage:  python scripts/verify_and_summarize.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(rel):
    return json.loads((DATA / rel).read_text())


def mean(xs):
    return sum(xs) / len(xs)


def rows(recs, key, val):
    return [r for r in recs if r[key] == val]


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {label}: got {got:.4f}, expect {want} (tol {tol})")
    return ok


def main():
    failures = 0

    v3 = load("v3_b200_seeds/results_v3.json")
    v4 = load("v4_pilot_b200/results_v4.json")
    v5 = load("v5_b200_replicate/results_v5.json")
    v6 = load("v6_pressure_sweep/results_v6.json")
    v7 = load("v7_b300_confirmation/unified_b300_results.json")

    def tp(recs, key, val):
        return [r["throughput_tok_s"] for r in rows(recs, key, val)]

    def pe(recs, key, val):
        return [r["preemptions"] for r in rows(recs, key, val)]

    # ---- v5 (b200, seeds 1-2, unshuffled): paper Appendix A.1 ----
    print("\n== v5: 200 blocks, seeds {1,2}, unshuffled ==")
    for cfg, tp_want, pe_want in [
        ("ctrl_b200", 370.28, 1840), ("wm0.05_b200", 368.79, 1217),
        ("wm0.15_b200", 353.62, 692), ("osl0.5_b200", 367.96, 999),
        ("osl1.0_b200", 348.85, 632),
    ]:
        failures += not check(f"v5 {cfg} mean tok/s", mean(tp(v5, "cfg", cfg)), tp_want, 0.01)
        pes = pe(v5, "cfg", cfg)
        failures += not check(f"v5 {cfg} preemptions (exact, both seeds)",
                              mean(pes), pe_want, 0)
        assert max(pes) == min(pes), f"v5 {cfg}: preemption seed spread nonzero"

    # ---- v6 pressure ratios: paper section 3.2 table ----
    print("\n== v6: throughput ratios vs same-pressure control (seed 0, shuffled) ==")
    def ratio(cfg, ctrl):
        return tp(v6, "cfg", cfg)[0] / tp(v6, "cfg", ctrl)[0]

    for cfg, ctrl, want in [
        ("wm0.05_b150", "ctrl_b150", 0.980), ("wm0.15_b150", "ctrl_b150", 0.937),
        ("osl0.5_b150", "ctrl_b150", 0.944), ("osl1.0_b150", "ctrl_b150", 0.906),
        ("wm0.05_b300", "ctrl_b300", 1.012), ("wm0.15_b300", "ctrl_b300", 0.958),
        ("osl0.5_b300", "ctrl_b300", 0.990), ("osl1.0_b300", "ctrl_b300", 0.976),
    ]:
        failures += not check(f"v6 {cfg} ratio", ratio(cfg, ctrl), want, 0.001)

    # v5-derived b200 ratios quoted in the same table
    print("\n== v5-derived b200 ratios ==")
    c200 = mean(tp(v5, "cfg", "ctrl_b200"))
    for cfg, want in [("wm0.05_b200", 0.996), ("wm0.15_b200", 0.955),
                      ("osl0.5_b200", 0.994), ("osl1.0_b200", 0.942)]:
        failures += not check(f"v5 {cfg} ratio", mean(tp(v5, "cfg", cfg)) / c200, want, 0.001)

    # ---- relocation law monotonicity: section 3.1 ----
    print("\n== relocation law: double monotonicity by preemption ordering ==")
    for recs, blocks, label in [(v6, 150, "b150 (v6 seed0)"), (v5, 200, "b200 (v5 mean)")]:
        cells = {}
        for r in recs:
            if r["blocks"] == blocks:
                cells.setdefault(r["cfg"], []).append(r)
        agg = []
        for cfg, rs in cells.items():
            agg.append((mean([x["preemptions"] for x in rs]),
                        mean([x["ttft_p99"] for x in rs]),
                        mean([x["decode_p99"] for x in rs]), cfg))
        agg.sort(reverse=True)  # most permissive (most preemptions) first
        ttfts = [a[1] for a in agg]
        decs = [a[2] for a in agg]
        mono_ttft = all(ttfts[i] <= ttfts[i + 1] for i in range(len(ttfts) - 1))
        mono_dec = all(decs[i] >= decs[i + 1] for i in range(len(decs) - 1))
        print(f"{'OK ' if (mono_ttft and mono_dec) else 'FAIL'} {label}: "
              f"ttft_p99 rising={mono_ttft}, decode_p99 falling={mono_dec}")
        failures += not (mono_ttft and mono_dec)

    # ---- v7 b300 confirmation: section 3.3 ----
    print("\n== v7: b300 confirmation (3 seeds, one session) ==")
    failures += not check("v7 ctrl mean tok/s", mean(tp(v7, "cfg", "ctrl_b300")), 528.38, 0.01)
    failures += not check("v7 wm0.05 mean tok/s", mean(tp(v7, "cfg", "wm0.05_b300")), 523.74, 0.01)
    failures += not check("v7 osl0.5 mean tok/s", mean(tp(v7, "cfg", "osl0.5_b300")), 525.48, 0.01)
    failures += not check("v7 wm0.05/ctrl ratio", mean(tp(v7, "cfg", "wm0.05_b300")) /
                          mean(tp(v7, "cfg", "ctrl_b300")), 0.991, 0.001)
    failures += not check("v7 osl0.5/ctrl ratio", mean(tp(v7, "cfg", "osl0.5_b300")) /
                          mean(tp(v7, "cfg", "ctrl_b300")), 0.994, 0.001)
    wm_red = 1 - mean(pe(v7, "cfg", "wm0.05_b300")) / mean(pe(v7, "cfg", "ctrl_b300"))
    failures += not check("v7 wm0.05 preemption reduction", wm_red, 0.41, 0.01)

    # ---- cross-session exactness: Appendix A.4 ----
    print("\n== cross-session preemption exactness (v6 seed0 vs v7 seed0) ==")
    for cfg in ("ctrl_b300", "wm0.05_b300", "osl0.5_b300"):
        a = [r for r in v6 if r["cfg"] == cfg and r["seed"] == 0][0]["preemptions"]
        b = [r for r in v7 if r["cfg"] == cfg and r["seed"] == 0][0]["preemptions"]
        print(f"{'OK ' if a == b else 'FAIL'} {cfg}: v6={a}, v7={b}")
        failures += a != b

    # ---- headline: 66% fewer preemptions for ~6% throughput (v5, b200) ----
    print("\n== headline dial-range numbers (v5) ==")
    red = 1 - mean(pe(v5, "cfg", "osl1.0_b200")) / mean(pe(v5, "cfg", "ctrl_b200"))
    cost = 1 - mean(tp(v5, "cfg", "osl1.0_b200")) / c200
    failures += not check("full-dial preemption reduction", red, 0.657, 0.005)
    failures += not check("full-dial throughput cost", cost, 0.058, 0.005)

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
