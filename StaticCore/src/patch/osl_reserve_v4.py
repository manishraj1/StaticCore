# Replacement block for the ISL-reservation line in
# vllm/v1/core/kv_cache_manager.py :: KVCacheManager.allocate_slots
# (vLLM 0.24.0). Applied automatically by the harnesses in src/harness/.

full_num_tokens = min(request.num_tokens, self.max_model_len)
# --- OSL-Reserve hook v4: alpha-scaled output reservation ---
import os as _os
_alpha = float(_os.environ.get('VLLM_OSL_ALPHA', '0'))
if _alpha > 0:
    _p = getattr(request, 'num_prompt_tokens', request.num_tokens)
    _m = getattr(request, 'max_tokens', None)
    if _m is None:
        sp = getattr(request, 'sampling_params', None)
        _m = getattr(sp, 'max_tokens', 0) if sp else 0
    full_num_tokens = min(_p + int(_alpha * (_m or 0)),
                          self.max_model_len)
