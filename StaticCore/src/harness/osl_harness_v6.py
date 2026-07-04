#!/usr/bin/env python3
"""
OSL-Reserve frontier sweep — CANONICAL v6.

Changes vs v5 (both motivated by the replication findings):
  1. SEED-KEYED SUBMISSION-ORDER SHUFFLING. v5 showed preemption counts are
     exactly deterministic given the schedule (seed spread = 0), so sampling
     seeds only perturb timing, not the schedule. v6 shuffles the prompt
     submission order with random.Random(seed), so seeds now perturb the
     schedule itself and preemption spread becomes a meaningful robustness
     signal. (Same 320 prompts, same token totals — order only.)
     NOTE: v6 counts are not directly comparable to v5 counts; compare
     structure (ordering, ratios), not raw numbers.
  2. NEW TIER --tier pressure: the five frontier-defining cells at
     blocks in {150, 300}, seed 0. 10 runs, ~95 min. Pre-registered:
       - blocks=150 (tighter cache, costlier recompute): reservation's
         relative value should RISE vs control;
       - blocks=300 (comfortable): the conservatism dial should flatten
         (policies converge; possibly zero preemptions — recorded, not fatal);
       - the b200 structure (throughput tie at matched conservatism;
         monotone TTFT<->decode-tail relocation) should replicate at b150.

Heterogeneous max_tokens (realistic OSL inputs) deliberately deferred: it
changes total work and breaks cross-version comparability. Next version.

Usage:
  pip install "vllm==0.24.0"   # then RESTART the runtime
  python osl_harness_v6.py --tier pressure   # 10 runs  <- this session
  python osl_harness_v6.py --tier replicate  # 10 runs @ b200, seeds 1,2
  python osl_harness_v6.py --tier pilot      # 8 cfgs @ b200, seed 0
  python osl_harness_v6.py --tier full       # 24 runs, ~4 h
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
import json, math, os, random, sys, time

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

random.Random(seed).shuffle(PROMPTS)

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
    "workload": "320x_n1_variants_shuffled",
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


def frontier_cells(blocks: int) -> list[dict]:
    return [
        cell(f"ctrl_b{blocks}", 0.0, 0.0, blocks),
        cell(f"wm0.05_b{blocks}", 0.05, 0.0, blocks),
        cell(f"wm0.15_b{blocks}", 0.15, 0.0, blocks),
        cell(f"osl0.5_b{blocks}", 0.0, 0.5, blocks),
        cell(f"osl1.0_b{blocks}", 0.0, 1.0, blocks),
    ]


def policy_grid(blocks: int) -> list[dict]:
    grid = [cell(f"ctrl_b{blocks}", 0.0, 0.0, blocks)]
    for w in (0.02, 0.05, 0.10, 0.15):
        grid.append(cell(f"wm{w}_b{blocks}", w, 0.0, blocks))
    for a in (0.25, 0.5, 1.0):
        grid.append(cell(f"osl{a}_b{blocks}", 0.0, a, blocks))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier",
                    choices=["pressure", "replicate", "pilot", "full"],
                    default="pressure")
    args = ap.parse_args()

    kvc_path = assert_environment()
    apply_patch_v4(kvc_path)

    subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import snapshot_download;"
         "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"],
        check=False,
    )

    if args.tier == "pressure":
        plan = [(c, 0) for b in (150, 300) for c in frontier_cells(b)]
    elif args.tier == "replicate":
        plan = [(c, s) for c in frontier_cells(200) for s in (1, 2)]
    elif args.tier == "pilot":
        plan = [(c, 0) for c in policy_grid(200)]
    else:
        plan = [(c, 0) for b in (150, 200, 300) for c in policy_grid(b)]

    print(f"[plan] {len(plan)} runs (~{len(plan) * 9.5:.0f} min at observed pace)")

    results, gate0_seen = [], set()
    for cfg, seed in plan:
        print(f"\n=== cfg={cfg['name']} seed={seed} ===")
        rec = run_cfg(cfg, seed)
        print(json.dumps(rec))
        results.append(rec)
        with open("results_v6.json", "w") as f:
            json.dump(results, f, indent=2)

        if cfg["name"].startswith("ctrl_") and cfg["blocks"] not in gate0_seen:
            gate0_seen.add(cfg["blocks"])
            if rec["preemptions"] == 0:
                if cfg["blocks"] <= 200:
                    raise RuntimeError(
                        f"GATE 0 FAILED: blocks={cfg['blocks']} control shows "
                        "0 preemptions — tighter than the regime that "
                        "preempted ~1800x. Environment changed; aborting."
                    )
                print(f"[gate 0] NOTE: blocks={cfg['blocks']} comfortable "
                      "regime (0 preemptions) — recorded, not fatal. The "
                      "flattening IS the b300 hypothesis.")
            else:
                print(f"[gate 0] blocks={cfg['blocks']}: control preemptions "
                      f"= {rec['preemptions']}")

    print("\n===== SUMMARY (ratios vs same-blocks control) =====")
    by_cfg = {}
    for r in results:
        by_cfg.setdefault(r["cfg"], []).append(r)

    ctrl_tp_by_blocks = {}
    for r in results:
        if r["cfg"].startswith("ctrl_"):
            ctrl_tp_by_blocks.setdefault(r["blocks"], []).append(
                r["throughput_tok_s"]
            )
    ctrl_tp_by_blocks = {
        b: sum(v) / len(v) for b, v in ctrl_tp_by_blocks.items()
    }

    print(f"{'cfg':16s} {'blocks':>6s} {'tok/s (mean)':>13s} {'vs ctrl':>8s} "
          f"{'preempt (mean)':>15s} {'ttft_p99':>9s} {'dec_p99':>8s} {'n':>2s}")
    for name, rows in by_cfg.items():
        b = rows[0]["blocks"]
        tp = sum(r["throughput_tok_s"] for r in rows) / len(rows)
        pe = sum(r["preemptions"] for r in rows) / len(rows)
        base = ctrl_tp_by_blocks.get(b)
        ratio = f"{tp / base:7.3f}" if base else "    n/a"
        print(f"{name:16s} {b:6d} {tp:13.2f} {ratio:>8s} {pe:15.1f} "
              f"{str(rows[-1]['ttft_p99']):>9s} "
              f"{str(rows[-1]['decode_p99']):>8s} {len(rows):>2d}")

    if any(len(v) >= 2 for v in by_cfg.values()):
        print("\nseed spread per cfg (max-min):")
        for name, rows in by_cfg.items():
            if len(rows) >= 2:
                tps = [r["throughput_tok_s"] for r in rows]
                pes = [r["preemptions"] for r in rows]
                print(f"  {name:16s} tok/s spread {max(tps)-min(tps):6.2f} "
                      f"({100*(max(tps)-min(tps))/min(tps):.1f}%) | "
                      f"preempt spread {max(pes)-min(pes)}")

    print("\n[done] results_v6.json written (checkpointed each run).")


if __name__ == "__main__":
    main()
