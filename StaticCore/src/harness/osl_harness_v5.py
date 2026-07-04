#!/usr/bin/env python3
"""
OSL-Reserve frontier sweep — CANONICAL v5.

Changes vs v4 (both motivated by the pilot's validity notes):
  1. WORKLOAD: 320 independent single-sample requests (n=1) instead of
     10 prompts x n=32. Engine-side this is nearly identical load — the
     Gate-0 source read established that V1 forks n=32 into 32 independent
     child requests anyway — but it yields 320 per-request latency samples
     (real p50/p95/p99) instead of 10 parent-aggregated ones, and retires
     the parallel-sampling framing the code read falsified.
     Prompts are 10 bases x 32 indexed variants (prefix caching is off, so
     distinct texts only guard against accidental dedup).
  2. REPLICATE TIER: the five frontier-defining cells x seeds {1, 2},
     control included in-session so all comparisons can be reported as
     ratios to control (pilot showed ~4.5% absolute drift across sessions).

Patch logic unchanged from v4 (alpha hook, v3->v4 upgrade path).

Pre-registered for the replicate tier:
  - CLAIM UNDER TEST: osl(a=1.0) dominates wm(0.15) on throughput AND
    preemptions. Holds only if the throughput gap survives seed spread.
  - The mid-frontier ordering (wm0.05 / osl0.5) is NOT claimed; we test
    whether it is resolvable at all.

Usage:
  pip install "vllm==0.24.0"   # then RESTART the runtime
  python osl_harness_v5.py --tier replicate  # 10 runs, ~95 min  <- tonight
  python osl_harness_v5.py --tier pilot      # 8 cfgs @ b200, seed 0
  python osl_harness_v5.py --tier full       # 24 runs, 3 pressures, ~4 h
"""

import argparse
import inspect
import json
import os
import subprocess
import sys

MIN_VERSION = (0, 23)
TARGET_LINE = "full_num_tokens = min(request.num_tokens, self.max_model_len)"

V3_MARKER = "VLLM_RESERVE_FULL_OSL"
V4_MARKER = "VLLM_OSL_ALPHA"

V3_BLOCK = (
    "full_num_tokens = min(request.num_tokens, self.max_model_len)\n"
    "            # --- OSL-Reserve hook (pre-registered experiment) ---\n"
    "            import os as _os\n"
    "            if _os.environ.get('VLLM_RESERVE_FULL_OSL', '0') == '1':\n"
    "                _p = getattr(request, 'num_prompt_tokens', request.num_tokens)\n"
    "                _m = getattr(request, 'max_tokens', None)\n"
    "                if _m is None:\n"
    "                    sp = getattr(request, 'sampling_params', None)\n"
    "                    _m = getattr(sp, 'max_tokens', 0) if sp else 0\n"
    "                full_num_tokens = min(_p + (_m or 0), self.max_model_len)\n"
)

V4_BLOCK = (
    "full_num_tokens = min(request.num_tokens, self.max_model_len)\n"
    "            # --- OSL-Reserve hook v4: alpha-scaled output reservation ---\n"
    "            import os as _os\n"
    "            _alpha = float(_os.environ.get('VLLM_OSL_ALPHA', '0'))\n"
    "            if _alpha > 0:\n"
    "                _p = getattr(request, 'num_prompt_tokens', request.num_tokens)\n"
    "                _m = getattr(request, 'max_tokens', None)\n"
    "                if _m is None:\n"
    "                    sp = getattr(request, 'sampling_params', None)\n"
    "                    _m = getattr(sp, 'max_tokens', 0) if sp else 0\n"
    "                full_num_tokens = min(_p + int(_alpha * (_m or 0)),\n"
    "                                      self.max_model_len)\n"
)


def assert_environment() -> str:
    import vllm

    ver = tuple(int(x) for x in vllm.__version__.split(".")[:2])
    if ver < MIN_VERSION:
        raise RuntimeError(
            f"FATAL: vllm=={vllm.__version__}; need >= 0.23. "
            "pip install 'vllm==0.24.0' and RESTART the runtime."
        )
    import vllm.v1.core.kv_cache_manager as m

    if "full_sequence_must_fit" not in inspect.getsource(
        m.KVCacheManager.allocate_slots
    ):
        raise RuntimeError("FATAL: full_sequence_must_fit missing. Inspect install.")
    print(f"[gate -1] OK: vllm=={vllm.__version__}")
    return m.__file__


def apply_patch_v4(kvc_path: str) -> None:
    with open(kvc_path) as f:
        content = f.read()

    if V4_MARKER in content:
        print("[patch] v4 alpha hook already on disk (verified)")
        return

    if V3_MARKER in content:
        if V3_BLOCK not in content:
            raise RuntimeError(
                "FATAL: v3 marker present but v3 block not found verbatim — "
                "unknown patched state. Fix: pip install --force-reinstall "
                "--no-deps 'vllm==0.24.0', restart, re-run."
            )
        content = content.replace(V3_BLOCK, V4_BLOCK, 1)
        print("[patch] upgraded v3 hook -> v4 alpha hook")
    elif TARGET_LINE in content:
        content = content.replace(TARGET_LINE, V4_BLOCK, 1)
        print("[patch] injected v4 alpha hook into pristine file")
    else:
        raise RuntimeError(
            f"FATAL: neither target line nor v3 block found in {kvc_path}."
        )

    with open(kvc_path, "w") as f:
        f.write(content)
    with open(kvc_path) as f:
        if V4_MARKER not in f.read():
            raise RuntimeError("FATAL: v4 patch write did not persist.")
    print(f"[patch] verified on disk: {kvc_path}")


# ---------------------------------------------------------------------------
WORKER = r'''
import json, math, os, sys, time

import vllm
from vllm import LLM, SamplingParams

cfg_name = os.environ["CFG"]
seed = int(os.environ["SEED"])
watermark = float(os.environ["WATERMARK"])
alpha = float(os.environ["VLLM_OSL_ALPHA"])
blocks = int(os.environ["BLOCKS"])

ver = tuple(int(x) for x in vllm.__version__.split(".")[:2])
assert ver >= (0, 23), f"worker on stale vllm {vllm.__version__}"

import vllm.v1.core.kv_cache_manager as _m
if alpha > 0 and "VLLM_OSL_ALPHA" not in open(_m.__file__).read():
    print(json.dumps({"error": "alpha arm but v4 patch NOT on disk"}))
    sys.exit(2)

kwargs = dict(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_model_len=1024,
    enforce_eager=True,
    enable_prefix_caching=False,
    num_gpu_blocks_override=blocks,
    disable_log_stats=False,
    seed=seed,
)
if watermark > 0:
    kwargs["watermark"] = watermark

llm = LLM(**kwargs)

# 320 independent single-sample requests: 10 bases x 32 indexed variants.
# Engine-side near-identical to the old 10 x n=32 load (V1 forks n into
# independent children anyway); yields 320 per-request latency samples.
BASES = [
    "Explain quantum key distribution protocols and information theoretic security bounds.",
    "Draft a full breakdown of Linux kernel memory management and buddy allocators.",
    "Detail mechanical stress vectors on aerospace composites during atmospheric reentry.",
    "Analyze architectural changes between Transformer layers and state-space models.",
    "Provide a deep analysis of decentralized consensus engines using DAGs.",
    "Explain physiological signaling cascades of G-protein coupled receptors.",
    "Detail compiler optimization passes: DCE, loop unrolling, and SSA form.",
    "Discuss macroeconomic impacts of liquidity traps and quantitative easing.",
    "Provide a manual on hydrothermal vent ecosystems and chemosynthesis.",
    "Deconstruct cryptographic implementations of zk-SNARKs.",
]
PROMPTS = [f"{b} (variant {i})" for b in BASES for i in range(32)]
assert len(PROMPTS) == 320

sp = SamplingParams(n=1, temperature=0.8, max_tokens=650, ignore_eos=True, seed=seed)

t0 = time.time()
out = llm.generate(PROMPTS, sp)
elapsed = time.time() - t0
total_tokens = sum(sum(len(o.token_ids) for o in r.outputs) for r in out)


def find_preemptions(llm):
    for m in llm.get_metrics():
        if m.name == "vllm:num_preemptions":
            return int(m.value)
    raise RuntimeError("vllm:num_preemptions missing — refusing to report 0")


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, max(0, math.ceil(p / 100 * len(sorted_vals)) - 1))
    return round(sorted_vals[k], 4)


# Per-request latency (now 320 samples).
# Admission-to-first-token delay: engine-computed first_token_latency.
# Decode duration: last_token_ts - first_token_ts (same clock base).
ttfts, decodes, missing = [], [], 0
for r in out:
    m = getattr(r, "metrics", None)
    if m is None:
        missing += 1
        continue
    ftl = getattr(m, "first_token_latency", 0.0) or 0.0
    ft = getattr(m, "first_token_ts", 0.0) or 0.0
    lt = getattr(m, "last_token_ts", 0.0) or 0.0
    if ftl > 0:
        ttfts.append(ftl)
    if lt > ft > 0:
        decodes.append(lt - ft)

if missing == len(out) or (not ttfts and not decodes):
    print(json.dumps({"error":
        f"latency metrics unavailable on all {len(out)} outputs"}))
    sys.exit(5)

ttfts.sort(); decodes.sort()

try:
    preempt = find_preemptions(llm)
except RuntimeError as e:
    print(json.dumps({"error": str(e)})); sys.exit(4)

print(json.dumps({
    "cfg": cfg_name, "seed": seed,
    "vllm": vllm.__version__,
    "blocks": blocks, "watermark": watermark, "alpha": alpha,
    "workload": "320x_n1_variants",
    "throughput_tok_s": round(total_tokens / elapsed, 2),
    "preemptions": preempt,
    "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95), "ttft_p99": pct(ttfts, 99),
    "decode_p50": pct(decodes, 50), "decode_p95": pct(decodes, 95), "decode_p99": pct(decodes, 99),
    "latency_n": len(ttfts), "latency_missing": missing,
    "elapsed_s": round(elapsed, 2),
}))
'''


def run_cfg(cfg: dict, seed: int) -> dict:
    env = os.environ.copy()
    env.update(
        CFG=cfg["name"],
        SEED=str(seed),
        WATERMARK=str(cfg["watermark"]),
        VLLM_OSL_ALPHA=str(cfg["alpha"]),
        BLOCKS=str(cfg["blocks"]),
    )
    p = subprocess.run(
        [sys.executable, "-c", WORKER], env=env, capture_output=True, text=True
    )
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            rec = json.loads(line)
            if "error" in rec:
                raise RuntimeError(f"[{cfg['name']} seed={seed}] {rec['error']}")
            return rec
    raise RuntimeError(
        f"[{cfg['name']} seed={seed}] no result JSON.\n--- stderr tail ---\n"
        + "\n".join(p.stderr.strip().splitlines()[-15:])
    )


def cell(name: str, watermark: float, alpha: float, blocks: int) -> dict:
    return dict(name=name, watermark=watermark, alpha=alpha, blocks=blocks)


def policy_grid(blocks: int) -> list[dict]:
    grid = [cell(f"ctrl_b{blocks}", 0.0, 0.0, blocks)]
    for w in (0.02, 0.05, 0.10, 0.15):
        grid.append(cell(f"wm{w}_b{blocks}", w, 0.0, blocks))
    for a in (0.25, 0.5, 1.0):
        grid.append(cell(f"osl{a}_b{blocks}", 0.0, a, blocks))
    return grid


REPLICATE_CELLS = [
    cell("ctrl_b200", 0.0, 0.0, 200),      # in-session anchor for ratios
    cell("wm0.05_b200", 0.05, 0.0, 200),   # watermark sweet spot (#44594)
    cell("wm0.15_b200", 0.15, 0.0, 200),   # the dominated? point
    cell("osl0.5_b200", 0.0, 0.5, 200),    # mid-frontier resolvability test
    cell("osl1.0_b200", 0.0, 1.0, 200),    # the dominating? point
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["pilot", "full", "replicate"],
                    default="replicate")
    args = ap.parse_args()

    kvc_path = assert_environment()
    apply_patch_v4(kvc_path)

    subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import snapshot_download;"
         "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"],
        check=False,
    )

    if args.tier == "pilot":
        plan = [(c, 0) for c in policy_grid(200)]                          # 8
    elif args.tier == "full":
        plan = [(c, 0) for b in (150, 200, 300) for c in policy_grid(b)]   # 24
    else:
        plan = [(c, s) for c in REPLICATE_CELLS for s in (1, 2)]           # 10

    print(f"[plan] {len(plan)} runs (~{len(plan) * 9.5:.0f} min at observed pace)")

    results, gate0_seen = [], set()
    for cfg, seed in plan:
        print(f"\n=== cfg={cfg['name']} seed={seed} ===")
        rec = run_cfg(cfg, seed)
        print(json.dumps(rec))
        results.append(rec)
        with open("results_v5.json", "w") as f:
            json.dump(results, f, indent=2)

        if cfg["name"].startswith("ctrl_") and cfg["blocks"] not in gate0_seen:
            gate0_seen.add(cfg["blocks"])
            if rec["preemptions"] == 0:
                if cfg["blocks"] <= 200:
                    raise RuntimeError(
                        f"GATE 0 FAILED: blocks={cfg['blocks']} control shows "
                        "0 preemptions but this regime preempted ~2100x in "
                        "v3/v4. Environment changed — aborting."
                    )
                print(f"[gate 0] NOTE: blocks={cfg['blocks']} comfortable "
                      "regime (0 preemptions) — recorded, not fatal.")
            else:
                print(f"[gate 0] blocks={cfg['blocks']}: control preemptions "
                      f"= {rec['preemptions']}")

    # ----- summary: absolute + ratio-to-in-session-control -----
    print("\n===== SUMMARY =====")
    by_cfg: dict = {}
    for r in results:
        by_cfg.setdefault(r["cfg"], []).append(r)

    ctrl_rows = [r for r in results if r["cfg"].startswith("ctrl_")]
    ctrl_tp = (sum(r["throughput_tok_s"] for r in ctrl_rows) / len(ctrl_rows)
               if ctrl_rows else None)

    print(f"{'cfg':16s} {'tok/s (mean)':>13s} {'vs ctrl':>8s} "
          f"{'preempt (mean)':>15s} {'ttft_p99':>9s} {'dec_p99':>8s} {'n':>2s}")
    for name, rows in by_cfg.items():
        tp = sum(r["throughput_tok_s"] for r in rows) / len(rows)
        pe = sum(r["preemptions"] for r in rows) / len(rows)
        t99 = rows[-1]["ttft_p99"]
        d99 = rows[-1]["decode_p99"]
        ratio = f"{tp / ctrl_tp:7.3f}" if ctrl_tp else "    n/a"
        print(f"{name:16s} {tp:13.2f} {ratio:>8s} {pe:15.1f} "
              f"{str(t99):>9s} {str(d99):>8s} {len(rows):>2d}")

    if any(len(v) >= 2 for v in by_cfg.values()):
        print("\nseed spread per cfg (max-min):")
        for name, rows in by_cfg.items():
            if len(rows) >= 2:
                tps = [r["throughput_tok_s"] for r in rows]
                pes = [r["preemptions"] for r in rows]
                print(f"  {name:16s} tok/s spread {max(tps)-min(tps):6.2f} "
                      f"({100*(max(tps)-min(tps))/min(tps):.1f}%) | "
                      f"preempt spread {max(pes)-min(pes)}")

    print("\n[done] results_v5.json written (checkpointed each run).")


if __name__ == "__main__":
    main()
