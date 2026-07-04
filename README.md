# Where Should Requests Wait? An Empirical Study of KV-Cache Admission Control in vLLM V1

[![vLLM Version](https://img.shields.io/badge/vLLM-0.24.0-blue)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Draft_v0.1-orange)](paper/draft_v0.1.md)

Minimal patch, defensive evaluation harness, raw experimental data, and a
verification script for the empirical note **"Where Should Requests Wait?"**
(draft in [`paper/draft_v0.1.md`](paper/draft_v0.1.md)).

vLLM V1 resolves KV-cache oversubscription reactively (preempt + recompute)
and recently gained two admission-side mitigations: full-input-length
reservation ([PR #37307](https://github.com/vllm-project/vllm/pull/37307),
default-on) and a free-block watermark
([PR #44594](https://github.com/vllm-project/vllm/pull/44594)). #37307
explicitly deferred a third policy — output-length-aware reservation, as
implemented in TensorRT-LLM. This study implements that deferred policy as an
11-line patch (`src/patch/`) and characterizes all three policies across a
conservatism dial and three deeply oversubscribed memory-pressure levels
(Qwen2.5-1.5B, 320-request batches, fixed forced output lengths).

## Findings

**1. The relocation law.** Admission conservatism does not reduce waiting;
it relocates it. Across every policy, dial setting, and pressure level
tested, tail admission delay rises monotonically while tail decode duration
falls monotonically as conservatism increases. Admission policy is a choice
about *where in a request's lifetime queueing happens* — up front before the
first token (reservation), spread through decode as preemption stalls
(permissive default), or in between (watermark).

**2. The inverted cost mechanism.** We pre-registered that reservation's
value would rise as memory tightens (costlier recompute). The data cleanly
reversed this: every policy's throughput cost *grows* as the pool shrinks,
because the binding cost is batch thinning — the reserved fraction of the
pool — not recompute avoided. At ~1k-token contexts, recompute is cheap
everywhere and admission conservatism is most expensive exactly where
intuition says it is most needed.

**3. Regime qualification of upstream baselines.** The watermark's published
throughput benefit did not reproduce as a gain in any regime tested: in the
loosest regime a 0.05 watermark is throughput-neutral within noise while
removing ~41% of preemptions (free, not profitable); in tighter regimes it
costs 2–6%. Output-length reservation is throughput-equivalent to the
watermark at matched conservatism throughout — its information advantage
buys nothing detectable under homogeneous output lengths.

Two single-configuration dominance claims that arose mid-study both died
under perturbation (workload change; seed replication) while the findings
above survived every perturbation applied — reported in the paper (§3.4) as
a caution against ordering claims from single-configuration Pareto tables.

## Core data

Throughput ratio vs. same-session, same-pressure stock control. Preemption
counts are exact engine counters (they reproduced to the integer across
sessions); throughput carries seed/session error bars, hence per-cell seed
counts below.

| Policy | 150 blocks (1 seed) | 200 blocks (2 seeds) | 300 blocks (1 seed / 3-seed confirm) |
| :--- | :---: | :---: | :---: |
| Stock control (vLLM 0.24.0 defaults) | 1.000 | 1.000 | 1.000 |
| Watermark 0.05 | 0.980 | 0.996 | 1.012 / **0.991** |
| Watermark 0.15 | 0.937 | 0.955 | 0.958 / — |
| OSL reservation (alpha = 0.5) | 0.944 | 0.994 | 0.990 / **0.994** |
| OSL reservation (alpha = 1.0) | 0.906 | 0.942 | 0.976 / — |

The 300-block single-seed watermark "gain" (1.012) did not survive 3-seed
in-session replication (0.991, distributions overlapping the control) — the
correct statement is throughput-neutral, and we publish both numbers.

Relocation law at 150 blocks (seed 0), permissive → conservative:
admission-delay p99 rises 684 → 710 → 750 → 753 → 783 s while decode p99
falls 173 → 172 → 160 → 159 → 139 s. The same double-monotone holds at 200
and 300 blocks (asserted programmatically by the verifier).

## Verify every number

```
python scripts/verify_and_summarize.py
```

recomputes all aggregates above (and those in the paper) from the raw
records in `data/` and exits nonzero on any mismatch.

## Repository structure

```
├── README.md
├── LICENSE                      # MIT
├── REPRODUCE.md                 # environment pin, restart warning, run commands
├── src/
│   ├── patch/                   # the 11-line OSL patch + documentation
│   └── harness/                 # osl_harness_v3/v5/v6.py (defensive runners)
├── data/                        # raw per-run JSON records + provenance README
│   ├── v3_b200_seeds/
│   ├── v4_pilot_b200/
│   ├── v5_b200_replicate/
│   ├── v6_pressure_sweep/
│   └── v7_b300_confirmation/
├── scripts/
│   └── verify_and_summarize.py  # recompute + assert all published aggregates
└── paper/
    └── draft_v0.1.md
```

## Limitations (short form; full list in the paper)

One model at one small scale and one context regime (~1k tokens) — §3.2 of
the paper argues the conclusions should invert as context length grows, so
nothing here extrapolates to long-context serving. Batch arrivals (admission
delay measures queue position, not open-loop TTFT). Homogeneous forced
output lengths, which neutralize OSL's information advantage. Consumer-GPU
session drift (~2–4.5%), mitigated by in-session controls.

## Citation

Draft; citation entry will be added on preprint submission.
