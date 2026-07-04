#!/usr/bin/env python3
"""
OSL-Reserve 3-arm harness — CANONICAL v3 (merged).
Arms: ISL-only control | OSL-reserve treatment | Watermark baseline (0.05, per PR #44594).

Verified against the vLLM v0.24.0 tag:
  - EngineArgs field: num_gpu_blocks_override  (NOT num_gpu_blocks)
  - EngineArgs field: watermark  -> SchedulerConfig.watermark
  - LLM.get_metrics() exposes Counter "vllm:num_preemptions"
    (requires disable_log_stats=False)

Hard rules baked in (every one exists because a run failed silently without it):
  * version + patch-on-disk asserted in the SAME interpreter that runs the arm
  * OSL hook toggled ONLY via VLLM_RESERVE_FULL_OSL env var, set per subprocess
  * watermark set unconditionally at construction — no field probing, no skips
  * preemption counter found via get_metrics() or the worker ABORTS
  * Gate 0 adjudicated after the FIRST control run, before burning more GPU-hours
  * one subprocess per (arm, seed): no shared engine state across runs

Usage:
  pip install "vllm==0.24.0"    # then RESTART the runtime
  python osl_harness_v3.py --seeds 3
"""

import argparse
import inspect
import json
import os
import subprocess
import sys

MIN_VERSION = (0, 23)
PATCH_MARKER = "VLLM_RESERVE_FULL_OSL"
TARGET_LINE = "full_num_tokens = min(request.num_tokens, self.max_model_len)"

PATCH_BLOCK = (
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


# ---------------------------------------------------------------------------
# GATE -1: environment (parent process)
# ---------------------------------------------------------------------------
def assert_environment() -> str:
    import vllm

    ver = tuple(int(x) for x in vllm.__version__.split(".")[:2])
    if ver < MIN_VERSION:
        raise RuntimeError(
            f"FATAL: vllm=={vllm.__version__}; need >= "
            f"{MIN_VERSION[0]}.{MIN_VERSION[1]} (PRs #37307 + #44594). "
            f"pip install 'vllm==0.24.0' and RESTART the runtime."
        )

    import vllm.v1.core.kv_cache_manager as m

    if "full_sequence_must_fit" not in inspect.getsource(
        m.KVCacheManager.allocate_slots
    ):
        raise RuntimeError(
            "FATAL: full_sequence_must_fit missing from allocate_slots "
            f"despite vllm=={vllm.__version__}. Inspect the install."
        )
    print(f"[gate -1] OK: vllm=={vllm.__version__}, full_sequence_must_fit present")
    return m.__file__


def apply_patch(kvc_path: str) -> None:
    with open(kvc_path) as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[patch] hook already present on disk (marker verified)")
        return

    if TARGET_LINE not in content:
        raise RuntimeError(
            f"FATAL: target line not found in {kvc_path}. Upstream code changed; "
            "re-derive the patch. REFUSING to report success for an unapplied patch."
        )

    content = content.replace(TARGET_LINE, PATCH_BLOCK, 1)
    with open(kvc_path, "w") as f:
        f.write(content)
    with open(kvc_path) as f:
        if PATCH_MARKER not in f.read():
            raise RuntimeError("FATAL: patch write did not persist.")
    print(f"[patch] OSL hook injected and verified on disk: {kvc_path}")


# ---------------------------------------------------------------------------
# Worker: one (arm, seed) per fresh interpreter
# ---------------------------------------------------------------------------
WORKER = r'''
import json, os, sys, time

import vllm
from vllm import LLM, SamplingParams

arm = os.environ["ARM"]
seed = int(os.environ["SEED"])
watermark = float(os.environ["WATERMARK"])
osl_active = os.environ["VLLM_RESERVE_FULL_OSL"] == "1"

# --- re-assert environment INSIDE the interpreter that runs the experiment ---
ver = tuple(int(x) for x in vllm.__version__.split(".")[:2])
assert ver >= (0, 23), f"worker on stale vllm {vllm.__version__}"

import vllm.v1.core.kv_cache_manager as _m
_disk_src = open(_m.__file__).read()
if osl_active and "VLLM_RESERVE_FULL_OSL" not in _disk_src:
    print(json.dumps({"error": "OSL arm but patch NOT on disk in this session"}))
    sys.exit(2)

kwargs = dict(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_model_len=1024,
    enforce_eager=True,
    enable_prefix_caching=False,       # protocol: no shared-prefix confound
    num_gpu_blocks_override=200,       # identical cache in every arm
    disable_log_stats=False,           # REQUIRED for get_metrics()
    seed=seed,
)
if watermark > 0:
    kwargs["watermark"] = watermark    # verified EngineArgs field at v0.24.0
                                       # unknown-kwarg => loud crash, as intended

llm = LLM(**kwargs)

PROMPTS = [
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
sp = SamplingParams(
    n=32, temperature=0.8, max_tokens=650, ignore_eos=True, seed=seed
)

t0 = time.time()
out = llm.generate(PROMPTS, sp)
elapsed = time.time() - t0
total_tokens = sum(sum(len(o.token_ids) for o in r.outputs) for r in out)


def find_preemptions(llm):
    """Official metrics pipeline; abort rather than report an unverified 0."""
    metrics = llm.get_metrics()
    for m in metrics:
        if m.name == "vllm:num_preemptions":
            return int(m.value), m.name
    raise RuntimeError(
        "vllm:num_preemptions not in get_metrics() — refusing to report 0. "
        f"Available: {sorted({m.name for m in metrics})[:20]}"
    )


try:
    preempt, source = find_preemptions(llm)
except RuntimeError as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(4)

print(json.dumps({
    "arm": arm,
    "seed": seed,
    "vllm": vllm.__version__,                    # read, never hardcoded
    "osl_hook_env": osl_active,
    "watermark_set": watermark,
    "throughput_tok_s": round(total_tokens / elapsed, 2),
    "preemptions": preempt,
    "preemption_source": source,
    "total_tokens": total_tokens,
    "elapsed_s": round(elapsed, 2),
}))
'''


def run_arm(arm: str, seed: int, osl: bool, watermark: float) -> dict:
    env = os.environ.copy()
    env.update(
        ARM=arm,
        SEED=str(seed),
        VLLM_RESERVE_FULL_OSL="1" if osl else "0",
        WATERMARK=str(watermark),
    )
    p = subprocess.run(
        [sys.executable, "-c", WORKER], env=env, capture_output=True, text=True
    )
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            rec = json.loads(line)
            if "error" in rec:
                raise RuntimeError(f"[{arm} seed={seed}] worker aborted: {rec['error']}")
            return rec
    raise RuntimeError(
        f"[{arm} seed={seed}] no result JSON.\n--- stderr tail ---\n"
        + "\n".join(p.stderr.strip().splitlines()[-15:])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    kvc_path = assert_environment()
    apply_patch(kvc_path)

    print("[warmup] pre-downloading weights so no arm pays first-run overhead...")
    subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import snapshot_download;"
         "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"],
        check=False,
    )

    arms = [
        ("ISL_only_control", False, 0.0),
        ("OSL_reserve_treatment", True, 0.0),
        ("Watermark_baseline", False, 0.05),   # pre-registered: #44594 sweet spot
    ]

    results = []
    for seed in range(args.seeds):
        for name, osl, wm in arms:
            print(f"\n=== arm={name} seed={seed} ===")
            rec = run_arm(name, seed, osl, wm)
            print(json.dumps(rec))
            results.append(rec)

            # ---- GATE 0: adjudicate on the FIRST control run ----
            if name == "ISL_only_control" and seed == 0:
                if rec["preemptions"] == 0:
                    raise RuntimeError(
                        "GATE 0 FAILED: first control run shows 0 preemptions "
                        "(source: " + rec["preemption_source"] + "). No cliff -> "
                        "comparison is vacuous. Lower num_gpu_blocks_override "
                        "and re-run. Aborting before burning further GPU time."
                    )
                print(f"[gate 0] PASSED: control preemptions = {rec['preemptions']}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # quick per-arm summary
    print("\n===== SUMMARY (mean over seeds) =====")
    for name, _, _ in arms:
        rows = [r for r in results if r["arm"] == name]
        tp = [r["throughput_tok_s"] for r in rows]
        pe = [r["preemptions"] for r in rows]
        print(
            f"{name:24s} throughput {sum(tp)/len(tp):8.2f} tok/s "
            f"(min {min(tp):.1f} / max {max(tp):.1f}) | "
            f"preemptions {sum(pe)/len(pe):7.1f} (min {min(pe)} / max {max(pe)})"
        )
    print("\n[done] results.json written. Next: seed-spread -> equivalence margin -> Pareto.")


if __name__ == "__main__":
    main()
