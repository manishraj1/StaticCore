# Reproducing the experiments

Hardware: any single CUDA GPU with >= ~8 GB free (experiments ran on Colab
T4/L4-class GPUs). Model: Qwen/Qwen2.5-1.5B-Instruct (downloaded
automatically).

## Environment (do not skip the restart)

    pip install "vllm==0.24.0"
    # RESTART the Python runtime now. A stale interpreter keeps the old
    # wheel imported; two early sessions silently ran vLLM 0.6.6/0.7.3
    # this way and produced invalid no-op "experiments".

The harness hard-asserts the version and the patch state at startup and
again inside every worker subprocess; if an assertion fires, fix the
environment rather than the assertion.

## Runs

    python src/harness/osl_harness_v5.py --tier replicate  # b200, 5 cells x seeds {1,2} (~95 min)
    python src/harness/osl_harness_v6.py --tier pressure   # b150+b300, 5 cells each, seed 0 (~95 min)
    python src/harness/osl_harness_v6.py --tier full       # 3 pressures x 8 policies (~4 h)
    python src/harness/osl_harness_v3.py                   # original 3-arm b200 run (n=32 workload)

Each run appends its record to a checkpointed results JSON in the working
directory; compare against the corresponding file under data/.

## What to expect

Preemption counts should reproduce EXACTLY for a given (config, seed) —
they are deterministic given the schedule and reproduced to the integer
across our sessions. Absolute throughput will differ from data/ by a few
percent (session drift); throughput ratios against the same-session control
should match the published ratios within ~1-2%.

## Verifying the published numbers

    python scripts/verify_and_summarize.py

recomputes every aggregate in the README and paper from data/ and exits
nonzero on any mismatch.
