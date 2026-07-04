# Data provenance

Each subdirectory holds the JSON records of one experimental phase, one
record per (configuration, seed) run. Records are the verbatim stdout JSON
emitted by the harness worker for that run (the harness also checkpointed
them to results_v*.json after every run); they were assembled into these
files from the preserved run logs. Preemption counts are exact engine
counters (`vllm:num_preemptions`); throughput carries ~2-4.5% cross-session
drift and must only be compared within a session (see paper section 2,
"Measurement discipline").

| Directory | Harness | Blocks | Seeds | Workload | Notes |
|---|---|---|---|---|---|
| v3_b200_seeds | v3 | 200 | 0,1,2 | 10 prompts x n=32, unshuffled | no latency capture; first fully valid run |
| v4_pilot_b200 | v4 | 200 | 0 | 10 prompts x n=32 | latency over 10 parent requests only: p95=p99=max; superseded by v5 for latency claims |
| v5_b200_replicate | v5 | 200 | 1,2 | 320 x n=1, unshuffled | 320 latency samples; preemption seed-spread exactly 0 (deterministic schedule) |
| v6_pressure_sweep | v6 | 150,300 | 0 | 320 x n=1, seed-shuffled order | pressure sweep; single seed per cell |
| v7_b300_confirmation | v6 (inline runner) | 300 | 0,1,2 | 320 x n=1, seed-shuffled | one session; caveat: the inline runner omitted the worker patch-on-disk assertion; patch activity is nonetheless evidenced by exact preemption-count reproduction of v6 seed-0 (1739/1027/1038) |

Known caveats carried into the paper: v3/v4 used the n=32 parallel-sampling
workload (engine-side near-identical to 320 independent requests, but pilot
latency percentiles are parent-aggregated); v4-vs-v5 demonstrates that
frontier-edge throughput orderings flip under this workload change (paper
section 3.4). Cross-pressure preemption counts are not comparable.

Run `python scripts/verify_and_summarize.py` to recompute and assert every
aggregate published in the README and paper from these records.
