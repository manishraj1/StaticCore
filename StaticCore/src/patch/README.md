# The OSL-Reserve patch (v4, alpha-parameterized)

This is the entire intervention studied in the paper: when vLLM V1's
(default-on) full-sequence-fit admission path computes its block reservation,
reserve for `prompt + alpha * max_tokens` instead of the prompt alone.
`alpha` is read from the environment variable `VLLM_OSL_ALPHA`; `alpha = 0`
(unset) is byte-for-byte stock behavior, `alpha = 1.0` is full TRT-LLM-style
output reservation — the policy vLLM PR #37307 explicitly deferred.

Target: `vllm/v1/core/kv_cache_manager.py`, inside
`KVCacheManager.allocate_slots`, replacing the single line

    full_num_tokens = min(request.num_tokens, self.max_model_len)

with the block in `osl_reserve_v4.py` in this directory.

You normally never apply this by hand: every harness in `src/harness/`
locates the installed file, applies the patch idempotently, verifies it on
disk after writing, and re-verifies inside each worker subprocess before an
alpha>0 arm is allowed to run. Applied against vLLM **0.24.0** (pinned; the
target line is version-sensitive — the patcher refuses loudly if it is not
found verbatim).
