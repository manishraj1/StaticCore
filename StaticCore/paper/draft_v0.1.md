# Where Should Requests Wait? An Empirical Study of KV-Cache Admission Control in vLLM V1

**Draft v0.1 — [Manish Raj Vangari], July 2026**

## Abstract

Modern LLM serving engines resolve KV-cache oversubscription reactively: when memory runs out, running requests are preempted and later recomputed. vLLM V1 recently gained two admission-side mitigations — full-input-length reservation (PR #37307, default-on) and a free-block watermark (PR #44594) — and its authors explicitly deferred a third, output-length-aware reservation, noting TensorRT-LLM implements it. We implement that deferred policy as a minimal patch (reserve prompt + α·max_tokens at admission) and characterize all three policies across a conservatism dial and three memory-pressure levels on a production engine (vLLM 0.24.0, Qwen2.5-1.5B, 320-request batches, forced fixed-length outputs).

Three findings. First, a **relocation law**: admission conservatism does not reduce waiting, it relocates it — across every policy, setting, and pressure level tested, tail time-to-first-token rises monotonically while tail decode time falls monotonically as conservatism increases. Second, an **inverted cost mechanism**: we pre-registered that reservation's value would rise as memory tightens (costlier recompute); the data cleanly reversed this. Reservation's throughput cost *grows* as the pool shrinks, because the binding cost is batch thinning — the reserved fraction of the pool — not recompute avoided. Admission conservatism is most expensive exactly where intuition says it is most needed. Third, a **regime qualification of upstream results**: the watermark's published throughput benefit did not reproduce as a gain in any regime we tested; in the loosest regime it is throughput-neutral within noise while eliminating ~41% of preemptions, and in tighter regimes it costs 2–6% throughput. Output-length reservation is throughput-equivalent to the watermark at matched conservatism; the two policies differ in where requests wait, not in how fast the system runs.

Two edge-level dominance claims that appeared in single-configuration runs (output-reservation dominating an aggressive watermark; a small watermark beating the default on throughput) both died under workload perturbation or seed replication, while the structural findings above survived every perturbation applied. We report both outcomes, and the methodology that separated them, as primary contributions.

## 1. Introduction

Continuous-batching engines admit requests knowing only their input length; output length is unknown at admission and KV-cache demand grows token by token during decode. When the block pool is exhausted, vLLM V1 preempts a running request, frees its blocks, and re-queues it for full recomputation. The engine's admission-side defenses have evolved quickly: PR #37307 (merged 2026-03-24) added `scheduler_reserve_full_isl` — reserve blocks for the full input at admission, default **on** — in response to a preemption-thrashing pathology at 100k-token inputs where generation throughput collapsed from ~100 to ~1.5 tok/s/GPU. PR #44594 (merged 2026-06-11) added a `watermark` fraction of blocks held back from admission, with measurements on GB200 showing large preemption reductions and a small throughput *gain* at a 0.05 setting. The #37307 author explicitly considered and deferred output-aware reservation ("TRT-LLM has an option that guarantees no evictions by additionally assuming max_new_tokens will be reached"), reserving for input length only.

This note asks the deferred question empirically: what does output-length-aware admission reservation buy, relative to the watermark and to the permissive default, and how does the answer depend on memory pressure? We are deliberately not proposing a new system. The contribution is a characterization — including two pre-registered predictions that the data falsified, reported as such.

## 2. Method

**Policy under test.** We patch `KVCacheManager.allocate_slots` in vLLM 0.24.0 so that, when the (default-on) full-sequence-fit path computes its reservation, it reserves `min(num_prompt_tokens + α · max_tokens, max_model_len)` instead of input length alone. α is an environment-controlled dial; α = 0 recovers stock behavior, α = 1.0 is the full TRT-LLM-style reservation. The patch is eleven lines and touches no scheduler logic; all admission and preemption mechanics are stock.

**Baselines.** (a) Control: stock vLLM 0.24.0 defaults, which include ISL reservation per #37307. (b) Watermark at 0.05 (the sweet spot reported in #44594) and 0.15 (an aggressive setting past that PR's own recommended range).

**Workload.** 320 independent single-sample requests (10 base prompts × 32 indexed variants; prefix caching disabled), temperature 0.8, `max_tokens=650` with `ignore_eos=True`, so every request performs identical, fixed decode work and total tokens are constant across arms (208,000). Submission order is shuffled per seed. Qwen2.5-1.5B-Instruct, `max_model_len=1024`, eager mode, single consumer GPU (Colab).

**Pressure.** KV pool fixed by `num_gpu_blocks_override` ∈ {150, 200, 300} blocks (2,400–4,800 tokens of cache against ~214k tokens of aggregate demand — all three levels are deeply oversubscribed; "pressure" here scales capacity, not regime).

**Metrics.** Throughput (total tokens / wall time), preemption count (engine Prometheus counter `vllm:num_preemptions`), and per-request latency percentiles over all 320 requests: admission-to-first-token delay (engine-computed `first_token_latency`) and decode duration (`last_token_ts − first_token_ts`, single clock base). Under batch submission, first-token delay measures queue position — which is precisely the quantity admission control trades — and we label it as such rather than as serving TTFT.

**Measurement discipline.** Three properties of this setup, discovered during the study, govern how numbers are reported. (1) Absolute throughput drifts ~2–4.5% across sessions on shared cloud GPUs; every comparison is therefore reported as a ratio to a same-session, same-pressure control. (2) Given a fixed submission order and configuration, the preemption count is exactly deterministic — it reproduced to the integer across sessions (e.g., control at 300 blocks: 1,739 in two sessions a day apart) — so preemption effects are exact while throughput effects carry seed/session error bars. (3) Because of (2), sampling-seed replication alone is vacuous for schedule metrics; robustness requires perturbing the schedule itself, which we do via seed-keyed submission-order shuffling (post-shuffle preemption spread: 1–3%).

**Pre-registration.** Each experimental phase locked its claims and kill conditions before running. The relevant ones: (P1) output-reservation at α=1.0 dominates watermark 0.15 (from a pilot; tested under workload change); (P2) reservation's relative value rises as pressure tightens (recompute-cost hypothesis); (P3) the conservatism dial flattens in the loosest regime; (P4) watermark 0.05 beats control on throughput at 300 blocks (from a single-seed observation; tested with 3 seeds in-session).

## 3. Results

### 3.1 A relocation law: conservatism moves waiting, it does not remove it

Across every pressure level, both policy families, and every dial setting, the same double-monotone structure holds: as conservatism increases (control → wm0.05 → osl0.5 → wm0.15 → osl1.0, ordered by preemption count), tail admission delay rises and tail decode time falls, in lockstep.

At 150 blocks (seed 0): admission-delay p99 rises 684 → 710 → 750 → 753 → 783 s while decode p99 falls 173 → 172 → 160 → 159 → 139 s. At 200 blocks (two seeds): 514 → 531 → 536 → 538 → 571 s against 171 → 167 → 158 → 155 → 150 s. At 300 blocks the same ordering holds (336 → 339/352 → 373/367 s against 185 → 173/174 → 165/162 s), and the decode-tail improvement replicates across three seeds in the confirmation run (control 187–192 s vs. ~175–181 s for either moderate policy).

Total elapsed time moves comparatively little. The three policies are not faster or slower ways to serve the batch; they are choices about *where in a request's lifetime the queueing happens* — up front, before the first token (reservation); spread through decode as preemption stalls (permissive default); or in between (watermark). For interactive workloads this is the entire decision: reservation trades first-token latency for streaming smoothness.

### 3.2 An inverted cost mechanism: reservation is most expensive where memory is scarcest

We pre-registered (P2) the opposite. The recompute-cost hypothesis says that as the cache tightens, preemption-recompute grows more costly, so paying admission conservatism to avoid it should grow more attractive. The measured throughput ratios (each vs. its same-pressure control):

| Policy | 150 blocks | 200 blocks | 300 blocks |
|---|---|---|---|
| watermark 0.05 | 0.980 | 0.996 | ~0.99 (3-seed: 0.991) |
| watermark 0.15 | 0.937 | 0.955 | 0.958 |
| OSL α=0.5 | 0.944 | 0.994 | ~0.99 (3-seed: 0.994) |
| OSL α=1.0 | 0.906 | 0.942 | 0.976 |

Every policy's cost shrinks as the pool grows — monotone in the wrong direction for P2, for all four settings. The mechanism the data supports instead: the binding cost of admission conservatism is **batch thinning**, proportional to the *fraction of the pool* withheld, not to the recompute avoided. Reserving ~42 blocks per admitted request removes a far larger share of a 150-block pool than of a 300-block one, so concurrency starves precisely where memory is tightest. Meanwhile, at 1k-token contexts, recompute is cheap everywhere: the permissive control absorbs 1,700–2,100 preemptions per run and still posts the best or statistically-tied-best throughput at every pressure level. The #37307 pathology is real but lives at a different point on the context-length axis, where a preempted request forfeits a 100k-token prefill; at short contexts, frequent preemption is a low-cost steady state, and the "fix" costs more than the disease.

The two mechanisms make opposite predictions as context length grows — recompute cost rises with prefix length, batch-thinning cost does not — so the crossover point is measurable. We did not measure it; it is the natural next experiment.

### 3.3 Regime qualification of the upstream baselines

The watermark's published benefit (#44594: fewer preemptions *and* slightly higher throughput at 0.05) did not reproduce as a throughput gain in any regime here. A single-seed run at 300 blocks initially showed +1.2% (P4); three-seed in-session replication reversed it: control 528.4 tok/s (range 523–533), watermark-0.05 523.7 (519–527) — statistically indistinguishable, ratio 0.991. What is real and exactly reproducible is the preemption effect: −41% (1,739±8 → 1,039±17). So in the loose regime the honest statement is that a small watermark is **free, not profitable**: it removes two-fifths of preemptions (and the corresponding decode-tail stalls) at no measurable throughput cost. In tighter regimes the same setting costs 0.4–2% (200 blocks) to 2% (150 blocks), and the aggressive 0.15 setting costs 4–6% everywhere — consistent with #44594's own observation that benefits decline past 0.10. None of this contradicts the upstream measurement (different hardware, model scale, and workload); it bounds its portability.

Output-length reservation, the policy #37307 deferred, turns out to be throughput-equivalent to the watermark at matched conservatism throughout: at 200 blocks, α=0.5 vs. watermark-0.05 differ by 0.2% (inside noise) with α=0.5 taking 18% fewer preemptions; at 300 blocks the two are indistinguishable on both axes (1,022–1,038 vs. 1,027–1,059 preemptions; 525.5 vs. 523.7 tok/s). The information advantage OSL theoretically holds — it knows `max_tokens` — buys nothing detectable under homogeneous output lengths. Whether it earns its keep under *heterogeneous* output lengths, where the reservation actually encodes per-request information the watermark lacks, is untested here and is the strongest argument for a follow-up.

### 3.4 Edge orderings are not stable; structure is

Two dominance claims arose during the study and both failed replication. A pilot (10 prompts × n=32) showed α=1.0 strictly dominating watermark-0.15; changing the workload to 320 independent requests — near-identical engine-side load — flipped the throughput ordering while preserving the preemption ordering. A single seed showed watermark-0.05 dominating the control at 300 blocks; three seeds dissolved it. Meanwhile the relocation law, the cost-inversion, and the preemption orderings survived every perturbation applied (workload shape, submission order, seeds, sessions, pressure). The methodological point generalizes beyond this study: single-configuration Pareto tables of admission policies — including the ones in upstream PRs — support existence claims ("this setting helped here") but not ordering claims ("this policy beats that one"). Frontier-edge orderings in this domain appear to sit within the envelope of workload sensitivity.

## 4. Limitations

One model at one small scale (1.5B), one context regime (1k tokens) — and §3.2 argues the context axis is exactly where the conclusions should invert, so nothing here extrapolates to long-context serving. Batch (all-at-once) arrivals, so first-token delay measures queue position rather than open-loop serving TTFT; no Poisson arrival process was modeled. Homogeneous forced output lengths, which both neutralizes OSL's information advantage and makes schedules unusually deterministic. Consumer/Colab GPUs with ~2–4.5% session drift, mitigated by in-session controls but not eliminated. The OSL patch reserves per-request against current free blocks; it does not implement aggregate reservation across the running set (the version that could guarantee no evictions), so our results say nothing about that stronger policy. Preemption counts, while exact, are not comparable across pressure levels (throughput doubled while counts barely moved from 150 to 300 blocks); we compare them only within a pressure level.

## 5. Conclusion

At small context lengths in vLLM V1, the admission-conservatism dial — whether turned via watermark or output-length reservation — buys large, exactly-reproducible preemption reductions (up to 66%) at modest throughput cost (0–6%), and its real function is to relocate waiting from mid-decode stalls to admission delay. Its cost scales with the reserved fraction of the pool rather than with the recompute it avoids, making it cheapest where memory is loose and most expensive where memory is tight — the inverse of the intuition that motivates it. The two policy families are throughput-equivalent at matched conservatism in this regime; they are interface choices about where latency lives, not performance optimizations. The upstream defaults (#37307 on, watermark off) are consistent with these measurements for short-context workloads; the open questions our data motivates are the context-length crossover of §3.2 and the heterogeneous-output setting where output-aware reservation first has real information to use.

## Appendix A: Key data

**A.1 — 200 blocks, seeds {1,2}, unshuffled order (v5).** Control 370.3 tok/s / 1,840 preempt; wm0.05 368.8 / 1,217; wm0.15 353.6 / 692; osl0.5 368.0 / 999; osl1.0 348.9 / 632. Preemption seed-spread: 0 (deterministic schedule). Throughput seed-spread: 0.5–2.2%.

**A.2 — 150 and 300 blocks, seed 0, shuffled (v6).** As tabulated in §3.2; full JSON in repository.

**A.3 — 300-block confirmation, seeds {0,1,2}, one session.** Control 528.7/523.4/533.1 tok/s, 1739/1751/1753 preempt; wm0.05 527.3/524.9/519.1, 1027/1032/1059; osl0.5 527.2/531.2/518.1, 1038/1022/1022.

**A.4 — Cross-session exactness.** Seed-0 preemption counts at 300 blocks reproduced to the integer across two sessions (1739, 1027, 1038) while throughput drifted ~2%.

## Appendix B: Harness discipline

Every experimental failure in this project was a silent no-op that produced plausible numbers: a stale engine version (twice), an unapplied patch behind a success message, a post-construction config write the engine never read, a metric fallback that reported zero when the counter was missing, and a first-run memory-profiling confound that manufactured a 20% "treatment effect." The working harness converts each into a hard assertion: engine version and patch-on-disk verified inside the worker process that runs the experiment; treatment knobs set only through constructor-time configuration; metrics read from the engine's official counter with abort-on-missing; block pool pinned identically across arms; one subprocess per run; results checkpointed per run; and a cliff-existence gate that halts the sweep if the control arm fails to exhibit the phenomenon under study. We include the harness because, in our experience, the guards are not ceremony — they were the difference between measurements and noise with a JSON schema, in five out of seven sessions.

## References

[1] Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
[2] vLLM PR #37307, "add option to schedule requests based on full ISL," merged 2026-03-24.
[3] vLLM PR #44594, "watermark to reduce preemptions," merged 2026-06-11.
[4] Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024.
[5] Sheng et al., "Fairness in Serving Large Language Models," OSDI 2024.
[6] [CONCUR — verify venue/year before submission; flagged in earlier review as unconfirmed.]
