# RISE design notes

## 1. Data flow

```
corpus.jsonl ─ format_sample ─ tokenize ─ explode (chunks) ─┐
                                                             ▼
                     plan_batches (length-sorted, token budget) ─ pad ─ Prefetcher (background thread)
                                                             │
                                          Trunk.hidden_states(ids, lens)      ← HF, vLLM or SGLang
                                                             │   post-final-norm hidden [B, L, D]
                                          RiseHead.compute_from_hidden          (micro-batched by tokens)
                                                             │   chunk signatures [B, dim]
                                chunk-mean per sample (fp16-rounded chunks, fp32 mean, L2)
                                                             │
                           IndexWriter.write_block → shards/vectors-XXXXX.npy (fp16) + meta + record
queries.jsonl ── same path, prompt-masked ──► signatures ──► search (streaming MIPS) ──► top-k / scores
                                                             └──► metrics (auPRC/auROC/P@K), select
```

The trunk/head boundary is the key interface. A trunk returns the tensor the LM head consumes, and
the head recomputes the logits itself. Any serving backend can therefore drive RISE, including
tensor-parallel ones for models that don't fit on one GPU, as long as it can return hidden states.
`HFTrunk.verify()` checks that `W h (+ b)`, plus any soft-capping or logit scale, reproduces the
model's logits before any work starts.

## 2. Math → code

| Paper | Code |
|---|---|
| `z_t = W h_t` | `RiseHead.logits` (unembedding dtype, then fp32) |
| Temperature τ with mean entropy ≈ `target_entropy` | `RiseHead.tau_search`: bisection with 0.01-nat early exit; entropy = `logsumexp(s) − E_p[s]` |
| Support `S_t` = TopK ∪ {y_t}, cumprob ≥ ρ prefix | `token_factors`: `topk`, ground truth forced into the last slot, cumprob cutoff, renormalization |
| `r_t = p_t − 1[y_t]` | `res = probs − gt_mask` |
| `ĥ_t = CS_h(h_t)/‖·‖` | `projections.h.dense` (GEMM with the ±1 matrix) |
| `r̂_t = CS_r(r_t)/‖·‖` | `projections.r.sparse` (cost ∝ \|S_t\|, not V) |
| `ĝ_t = CS_g(Wᵀ r_t)/‖·‖` | `Σ_v r_t(v) M_g[v]`, `M_g = CS_g(W)` precomputed once (linearity) |
| `φ = [λ_rh Σ r̂⊗ĥ, λ_gh Σ ĝ⊗ĥ, λ_rg Σ r̂⊗ĝ]`, L2 | `_compute`: batched `bmm`, concatenated in rh, gh, rg order |
| Chunk-mean for long samples | `pipeline._aggregate` |
| Score `φ(x_i)ᵀ φ̄_Q` | `search.mean_query` + `score_all(..., normalize_queries=False)` |

CountSketch seeds are `h = seed`, `g = seed + 1`, `r = seed + 2`. Each table is drawn from a CPU
`torch.Generator`, buckets first and then sign bits. Tables are persisted in every index, so RNG
changes across PyTorch releases can't silently move the sketch space.

## 3. Numerics and parity policy

`tests/reference.py` is a literal port of the research estimator: one sequence at a time,
clamped entropy, scatter-based sketches, and the explicit `[T, K, D]` GH gather. Every engine change
must keep `tests/test_head.py` and `tests/test_pipeline.py` green against it (cosine > 0.9999 per
signature).

The engine was also checked directly against the original research code (v3 sequential and v5
batched) on tiny GPT-NeoX and Llama models, in three configs: defaults, multi-chunk, and GH with
input embeddings plus ground-truth forcing.

| Check | Result |
|---|---|
| Sketch tables | bit-identical |
| v5 chunk vectors | min cosine 0.9999998 |
| v3 index rows | min cosine 0.9999993 |
| Prompt-masked queries | min cosine 0.9999998 |

Deliberate execution differences, none of which change the estimator:

- Padded positions are excluded from the temperature search and the finiteness check (the research
  code processed unpadded chunks one at a time, or included padding).
- Entropy is computed without the `eps` clamp (difference ≤ ~2e-4 nats, far below the 0.01 exit
  tolerance).
- Rows that converged skip later bisection steps.
- On MPS the bisection bracket is float32 (no float64 there).

### The research code's temperature on GPU

On GPU the research code's entropy-targeted temperature search degenerates. Both v3's
`find_temperature_for_entropy` and v5's `_batched_tau` evaluate `−Σ p log clamp(p, 1e-8)` in the
logits' dtype, which is fp16 on GPU. `1e-8` is below fp16's smallest subnormal and rounds to 0, so
any underflowed probability gives `0 · log 0 = NaN`. Every bisection comparison then fails and the
bracket shrinks onto τ_min, so each chunk gets τ = τ_min + (τ_max − τ_min)/2^(steps+1) = 0.1000366.

Measured on H200 with Pythia-1B fp16 on 64 Howdy chunks:

| Check | Result |
|---|---|
| First-step entropy | NaN at 100% of positions |
| v5's τ on fp16 logits | 0.10004 for every chunk |
| v5's τ on fp32 logits | [1.05, 1.78] |
| RISE at fixed τ = 0.1000366 vs v5 | cosine 1.000000 (mean and min) |
| RISE with the working adaptive search vs v5 | cosine 0.51 |

The paper does not describe an adaptive temperature (its hyperparameter rows for it are commented
out); its runs used τ ≈ 0.1. RISE therefore defaults to a fixed τ = 0.1, and
`rise.compat.research_effective_config` maps GPU research configs to 0.1000366 exactly.
`tests/test_head.py::test_research_fp16_entropy_search_collapses_to_tau_min` pins the behavior.

GPU runs in bf16/fp16 differ from CPU fp32 by TopK tie-breaking and rounding. The acceptance
standard is then functional, as in the research code's own gate: retrieval metrics must match
within tolerance on the same data.

## 4. Index format

```
build_plan.json   format, format_version, num_rows, dim, dtype, block_size, num_blocks, attrs, attrs_sha256
projections.npz   {h,r,g}_buckets int32, {h,r,g}_signs int8, {h,r,g}_out_dim
shards/vectors-XXXXX.npy  float16 [rows, dim]      shards/meta-XXXXX.jsonl  one line per row
shards/block-XXXXX.json   {block, start, rows, sha256, stats}   ← written last; marks the block done
manifest.json     build_plan + shards[] + build_totals + rise_version + finalized_utc
```

`attrs` holds the full `RiseConfig`, the model identity (name, dims, dtype, and 32 probed
unembedding entries compared with tolerance, so fp16/bf16/fp32 loads of one checkpoint agree), the
tokenizer, the sketch-table hash, and the corpus hash. Resuming with any different setting is
refused. So is `select` against a different corpus file. Readers refuse newer `format_version`s.

Format v2 is a compressed index (`rise compress`): `shards/codes-XXXXX.npy` holds uint8
[rows, bits / 8] SimHash codes instead of vectors, `simhash.npz` the transform's random signs and
picked coordinates (its sha256 is in the manifest's `codec`), and attrs, metadata and sketch tables
are copied. A code is sign(P x) (SimHash, Charikar 2002), where P picks `bits` coordinates of
H D₂ H D₁ x / n, a randomized Hadamard transform (Ailon and Chazelle, 2006): D₁, D₂ random signs, H the
Walsh-Hadamard transform, n the dimension rounded up to a power of two. Two rounds,
because one leaves the signs of RISE's structured rows correlated (P@10 0.788 ± 0.013 vs 0.805 ±
0.003 over three draws, 405B 1M index, 8,192 bits). Queries stay float and
score sqrt(π/2) · sqrt(n) / bits · (P q) · sign(P x), which estimates q · x for unit rows and is linear
in q, so mean-query valuation is unchanged. The transform's parameters are stored rather than
re-drawn, like the sketch tables. Float16 indexes are still written as v1, so older readers keep
reading them and refuse only compressed ones.

## 5. Sizing

- **Index bytes:** `N × dim × 2`. `dim = Kh·(Kr + Kg)` for `rh+gh`, so the defaults give
  128·192 = 24,576.
- **Head working set per micro-batch:** logits `tokens × V × 4 bytes`, plus about two temporaries of
  the same size during the temperature search. `--max-head-tokens 8192` at V = 128k is about 4 GiB
  per temporary.
- **Search:** O(`rows_per_step × dim` + `queries × k`) memory. The cost is one GEMM per block, read
  once from the memory-mapped shards.

## 6. Extension points

- **New trunk.** Implement `rise.runtime.trunk.Trunk`: `hidden_states`, `unembedding`,
  `unembedding_bias`, `input_embedding`, `describe`, and ideally `verify`. For a serving engine,
  subclass `rise.runtime.engine.EngineTrunk` and implement `_encode(seqs)`. It maps token-id lists
  to per-sequence `[T_i, D]` final hidden states; row counts are checked for you.
- **New workload.** The `runtime`, `index` and `search` layers are workload-agnostic. Once a second
  Hammer Engine workload needs them, they move to a shared package.

## 7. Serving-engine trunks

vLLM (pooling runner, token-level `ALL` pooling, no activation) and SGLang (embedding mode with
RISE's model classes) run the transformer in their own processes; RISE reads only the LM head from
the checkpoint. By default the head runs inside the engine's workers (below). Under `--driver-head`
the engine prefills batch i+1 on the main thread while a worker thread pads batch i's hidden states
onto `--head-device` and runs the head.

Correctness guards specific to engines:

- **One row per prompt token, always.** SGLang with chunked prefill returned 149 rows for a
  153-token prompt that straddled a prefill chunk. RISE turns chunked prefill (and the radix cache)
  off, and `EngineTrunk.encode` rejects any row-count mismatch.
- **Probe perplexity.** Engines expose no logits in this mode, so `verify()` checks that the head's
  logits on the engine's hidden states predict a plain-English probe (Llama-3.1-8B: 3.26–3.27;
  Pythia-1B: 4.37). Pre-norm states or a wrong head would be orders of magnitude worse.

### The head inside vLLM

A driver-side head needs every prompt token's final hidden state, which vLLM's pooling runner
returns as float32 through its engine core: 16 KB per token for Llama-3.1-8B, 20 KB for OLMo-3-32B,
64 KB for Llama-405B. `rise/runtime/vllm_head.py` moves the head into the workers instead, and the
engine returns one signature per chunk:

- `RiseWorkerExtension` is mixed into vLLM's worker class (`worker_extension_cls`). The driver calls
  `collective_rpc("rise_install_head", spec)`: each worker loads the LM head from the checkpoint
  onto its own GPU, builds the same `RiseHead` with the index's sketch tables, and wraps the model's
  pooler.
- RISE requests carry `PoolingParams(extra_kwargs={"rise": {"loss_start": ...}})`. Other requests,
  such as `verify()`'s probe, still get hidden states from the original pooler.
- The prompts whose prefill finished in a step are dealt round-robin over the tensor-parallel
  ranks. Each rank computes its share on its own GPU, and an all-gather hands every signature to
  rank 0, whose output vLLM returns. Rows of a prompt split by chunked prefill wait in vLLM's
  per-request pooling state, as they do for vLLM's own `ALL` pooling.
- Fail-closed: a prompt whose pooled rows differ from its length raises, and every rank must report
  the driver's head dimension, sketch-table hash and unembedding probe.
- The head's memory on each GPU (the LM head, plus `[max_head_tokens, vocab]` logits) comes out of
  the engine's unreserved share, as the driver-side head's did on `--head-device`.

Measured with `benchmarks/bench_engine_head.py --data c4.jsonl --docs 3000 --tp N --set Kr=128
--set Kh=24 --set Kg=128` (the same 6.4k chunks, 1.2M tokens, through both heads on one engine),
H200s, vLLM 0.30:

| Model | TP | Head in the driver | Head in the engine |
|---|---|---|---|
| Pythia-1B | 1 | 164k tok/s | 228k tok/s (1.40×) |
| Llama-3.1-8B | 1 | 35.5k tok/s | 39.0k tok/s (1.10×) |
| Llama-3.1-8B | 2 | 45.4k tok/s | 64.2k tok/s (1.42×) |
| OLMo-3-32B | 4 | 20.8k tok/s | 27.6k tok/s (1.33×) |
| Llama-3.1-405B (FP8) | 8 | 5.21k tok/s | 6.37k tok/s (1.22×) |

The 405B row comes from the production build over the 1,005,000-row Howdy + C4 pool (sketch dims
128/128/64, `--engine-arg max_num_batched_tokens=16384 --engine-arg max_num_seqs=256
--max-model-len 1024 --gpu-memory-utilization 0.80`), stopped after block 20 and resumed with the
new code: blocks 15-20 with the head in the driver took 612-627 s each (5.06k-5.26k tok/s), blocks
22-24 with the head in the engine 506-517 s (6.36k-6.37k tok/s), for about 3.2M tokens per block.
GPU 0 dropped from 142 GB to 129 GB used, since the driver process no longer computes logits.

With tensor parallelism the driver-side head ran on GPU 0 alone, beside rank 0's share of the
model; in the engine every rank runs a share of the head.

Signatures differ from the driver-side head's by rebatching noise only. bf16 logits round
differently under different GEMM shapes, and at tau = 0.1 one bf16 ulp at a logit near 20 is a 3.5×
change in probability, so some chunks move whenever the head is batched differently; the
driver-side head against itself at a quarter of its head batch shows the scale. Chunks below cosine
0.9999, engine vs driver head / driver head vs itself: Pythia-1B 0 / 0; Llama-3.1-8B 4 / 5 at TP=1 and 102 / 42 at
TP=2, of 6,377 (minimum 0.956 / 0.986; a smoke run of the same comparison gave 49 / 52); OLMo-3-32B
at TP=4, whose signatures move the most under any rebatching, 4,975 / 5,511 of 6,365 (mean cosine
0.9988 / 0.9987). At TP ≥ 2 each rank batches its share of a step separately, one more change of
GEMM shape; the ranks' final hidden states were bitwise identical in a checked step
(Llama-3.1-8B, TP=2).

Retrieval does not change. Howdy pool, OLMo-3-32B on vLLM (TP=4), sketch dims 48/64/16,
`lambda_rh=0.7 lambda_gh=1.0` (`rise build --backend vllm --tp 4 --max-model-len 1024 ... --queries`,
then `rise search` and `rise eval --label-regex "howdy!" --label-fields instruction`), auPRC /
auROC / precision@K:

| K | Head in the engine | Head in the driver (`--driver-head`) |
|---|---|---|
| 5 | 0.9951 / 0.9967 / 0.988 | 0.9951 / 0.9967 / 0.990 |
| 10 | 0.9896 / 0.9935 / 0.980 | 0.9897 / 0.9935 / 0.980 |
| 50 | 0.9721 / 0.9839 / 0.946 | 0.9720 / 0.9840 / 0.946 |

The paper reports 0.993 / 0.988 / 0.973 auPRC for OLMo-3-32B at the same K.

### The head inside SGLang

SGLang has no worker extension hook, but it registers model classes from an external package
(`SGLANG_EXTERNAL_MODEL_PACKAGE`) over its own. `rise.runtime.sglang_models` provides SGLang's
Llama, Mistral, Qwen2/3 and OLMo-2/3 classes with the pooler replaced, and RISE runs SGLang in
embedding mode, so each request's whole prompt reaches the pooler in one step:

- The request id says what a request wants (`rise.runtime.sglang_head`): `rise:<key>:<loss_start>:<n>`
  asks for its chunk signature, `rise-hidden:<n>` for its final hidden states (`verify()` and
  `--driver-head`), and other ids get the model's own pooler. Prompt masking travels in the id too.
- `install_head` writes the head's spec to `<key>.json` in a directory the workers inherit through
  `RISE_SGLANG_SPEC_DIR`; each worker loads it on the first request that names the key. The ranks
  split a step's signatures and all-gather them, as with vLLM (`rise.runtime.engine_head`).
- Signatures leave through shared memory. SGLang turns every output into Python lists, which for
  [24,576] float32 signatures cost more than a small model's forward pass. The request id also names
  a row of `signatures.bin` in the same directory (under `/dev/shm`); rank 0 writes the signature
  there and returns only the row number, which the driver checks before reading the rows.
- Fail-closed: a step whose prompts are not whole (a prefix-cache hit, chunked prefill) raises, and
  `install_head` sends one probe prompt per rank and checks each signature against this process's
  head on the same hidden states (cosine > 0.999).

Checked on the Howdy pool with Llama-3.2-1B-Instruct (bf16, default config), H200, auPRC at K = 5 /
10 / 50: SGLang TP=1 0.9995 / 0.9987 / 0.9965, SGLang TP=2 0.9980 / 0.9977 / 0.9957, HF trunk
0.9984 / 0.9978 / 0.9961.

Checkpoint-format support differs between engines. Meta's `Llama-3.1-405B-Instruct-FP8` uses the
`fbgemm_fp8` scheme: FP8 MLP weights with per-row scales, attention in bf16. vLLM 0.30 still runs it
behind `--engine-arg allow_deprecated_quantization=true`; the SGLang build tested here does not know
the scheme. With vLLM on 8×H200 this checkpoint reaches ~5.8k tok/s, power-bound, and
`max_num_batched_tokens` 16k vs 64k made no difference.

Deployment pitfalls met on an 8×H200 Slurm node, now handled or reported up front:

- **Thread limit.** A low soft `RLIMIT_NPROC` (1000) with an unlimited hard limit made TP=8 engines
  fail with `libgomp: Thread creation failed`. The CLI raises the soft limit on Linux, and engine
  trunks bound OpenMP / MKL / rayon / inductor pools.
- **No CUDA toolkit.** Engines JIT-compile kernels during warm-up, so without `CUDA_HOME` a 405B run
  failed after ten minutes of loading. RISE now warns at start.
- **Spawn start method.** SGLang starts workers with `spawn`; `python -m rise` keeps its entry point
  guarded.
- **A throttling GPU.** One H200 at thermal slowdown ran at half speed. Dynamic block claiming keeps
  data-parallel builds moving. For tensor parallelism, burn-test the node first, since every rank
  waits for the slowest.

## 8. Optimization log (measured on one H200)

**Done.**
- *Fixed-temperature fast path.* `post(z) / tau` is increasing, so the head takes the top-K of the
  raw bf16/fp16 GEMM output and scales only the K winners. This removed three `[N, V]` float32
  passes (copy, mask, divide) and halved top-k traffic.
- *Fused sparse-sketch kernel.* `rise/kernels/sparse_sketch.py` (Triton) visits only the kept prefix
  and the ground-truth slot, typically ~5 of 256 at tau = 0.1. It fuses the r and GH sketches with
  their normalization and mask; previously all 256 `M_g` rows were gathered per position.
- *No host syncs in the head.* ids and lens are validated on the host copy.
- *The head inside vLLM's workers* (section 7): 1.10-1.42× on 1B-32B models, 1.22× on Llama-405B.
- *Top-K kernel.* `rise/kernels/topk.py` (Triton) replaces `torch.topk` on 16-bit CUDA rows. One
  streaming pass over a (row, span) grid takes the maximum of every 16 (50k vocab) or 32 (128k
  vocab) elements. The K-th largest chunk maximum is a lower bound on the K-th largest element,
  since those K maxima are K distinct elements. Only chunks whose maximum reaches it can hold
  top-K elements, and there are about K of them, so a per-row program reads back 8-17 KB of the
  row instead of 100-250 KB and sorts ~330 candidates (median on real rows). The kernel is exact: the same values as
  `torch.topk` and, among ties, the lowest indices. Rows with too many candidates (flat rows,
  masses of ties) go to a separate kernel, either a wider sort or an on-device radix select.
  Moving those paths out of the main kernel cut it from 1.29 to 0.84 ms per 16k rows, and sorting
  packed 32-bit keys instead of 64-bit ones to 0.78 ms. `torch.topk` vs the kernel:
  16384×50304 fp16 random rows 6.32 → 1.32 ms; real Pythia-1B logits 6.13 → 1.54 ms; real
  Llama-3.1-8B logits (16384×128256 bf16) 14.3 → 2.37 ms.
- *Result.* The head went from 642k to 2.40M tok/s (Pythia-1B shapes) and from 116k to 160k tok/s
  (Llama-405B shapes). The Pythia-1B end-to-end build went from 183k to 225k tok/s (214k before the
  top-K kernel). Its 50,000 index rows match the previous build's to cosine > 1 - 1e-6.
- *Commands.* Head: `python benchmarks/bench_head.py --vocab 50304 --hidden 2048 --batch 64
  --dtype float16 --set Kr=128 --set Kh=24 --set Kg=128 --seq 256 --device cuda --iters 10
  --max-head-tokens 16384`, and `--vocab 128256 --hidden 16384 --batch 32 --dtype bfloat16` with
  the same remaining flags. End to end: `rise build --model EleutherAI/pythia-1b --data
  c4_50k.jsonl --dtype float16 --device cuda --set Kr=128 --set Kh=24 --set Kg=128 --block-size
  8192 --no-data-hash --max-batch-tokens 32768 --max-head-tokens 16384`, over C4 documents
  5,001-55,000. torch 2.11 (CUDA 12.8), Triton 3.6.

**Tried and dropped.**
- *`torch.compile` of the HF backbone.* +12.5% trunk throughput, but fused fp16 numerics moved the
  top-1 token at 0.17% of real-text positions and the top-5 set at 0.8% (max logit error 2.9% of the
  logit range). At tau = 0.1 those positions' residuals change completely, and `verify()` rejects it.
- *Head on a side CUDA stream, overlapping the trunk.* 96 s vs 97 s: the trunk alone saturates the
  GPU.
- *Top-K with one program per row.* A histogram of each element's distance to the row max, then
  compaction and a sort, was exact but 3× slower than `torch.topk`. One program cannot stream a
  row at HBM speed, and real logit rows are dense near their max, so the histogram's hot buckets
  serialize its atomics.
- *Top-K compaction inside the streaming pass.* Writing every element above the bound from the
  (row, span) grid, with atomics or a scan, ran at 0.46 TB/s, 6× slower than the same pass that
  only counts: every element carries a position and a 64-bit key. Reading back only the chunks that
  reach the bound replaced it.

**What is left.**
- *Pythia-1B shapes.* The logits GEMM is now 66% of head time (4.5 of 6.8 ms per 16k tokens, 749
  TFLOPS). The top-K kernel's chunk-maximum pass already streams as fast as `torch.amax` over the
  same rows (~3.3 TB/s). Fusing it into the GEMM epilogue would save that pass but needs a custom
  GEMM.
- *Llama-405B shapes.* The logits GEMM is 95% of head time (48.4 of 50.8 ms per 8k tokens, 711
  TFLOPS, ~72% of dense bf16 peak). An FP8 GEMM would halve it but perturb logits, which tau = 0.1
  amplifies ten-fold, so it is not used.
- *SGLang and Meta's FP8 405B checkpoint.* The SGLang build tested here cannot load its
  `fbgemm_fp8` scheme; an FP8 checkpoint in a format SGLang supports would run there.
- *Adaptive temperature.* A fused single-pass entropy kernel would cut the bisection's cost.

## 9. Research-code compatibility

RISE matches the research code as it has been since April 2026 (v3 and
later): same CountSketch tables, signatures at cosine 1.000000. Part of the paper's appendix tables
came from the December 2025 scripts, which differ
in four ways. Each is a config field; the defaults keep RISE's behavior, and fields at their defaults
are left out of `to_dict()`, so existing indexes keep their attrs and still resume.

| Field | Default | December scripts | Effect |
|---|---|---|---|
| `sketch_rng` | `cpu` | `cuda` | Same seeds, a different RNG stream: different tables |
| `sample_format` | `auto` | `alpaca` | `### Instruction: ...\n### Input: ...\n### Response: ...`, the fine-tuning template, instead of `text` or newline-joined fields |
| `gt_slot` | `replace` | `append` | Top-L cutoff taken on the top-K without the ground truth, which is then added as slot K+1 |
| `l2_eps` | `norm` | `squared` | `x / sqrt(max(|x|^2, eps))`: a 1e-4 norm floor that shortens the near-zero residuals of near-certain predictions, so those positions weigh less |

The December scripts' adaptive temperature collapsed on GPU like the later code's (fp16 entropy is
NaN), to `tau_min + (tau_max - tau_min) / 2^16`: 0.3000259 for their `[0.3, 2.0]` range, checked
on every chunk of four models. Set it with `tau_fallback`. Their GH channel used the input
embedding (`embedding_type=input`). The December runs behind the paper also used
`topL_cum_prob=0.97`, `min_topL=8`, `max_topL_cap=16` and `min_seq_len=3`.

Checked against the December scripts rerun on the same data, H200, auPRC at K = 5 / 10 / 25 or 50 /
50 or 100:

| Row | December script | RISE, compatible settings |
|---|---|---|
| Brain Rot, Pythia-1B, pretrained (`sketch_rng=cuda`, tau 275) | 0.8497 / 0.8315 / 0.7890 / 0.7564 | 0.8502 / 0.8318 / 0.7894 / 0.7566 |
| Howdy, Pythia-160M fine-tuned | 0.9839 / 0.9755 / 0.9222 / 0.8696 | 0.9885 / 0.9754 / 0.9244 / 0.8734 |
| Finance-Medical, Pythia-14M pretrained | 0.9895 / 0.9826 / 0.9314 / 0.8899 | 0.9914 / 0.9841 / 0.9351 / 0.8924 |
| Finance-Medical, Pythia-70M fine-tuned | 0.8320 / 0.8012 / 0.6815 / 0.6103 | 0.8232 / 0.7912 / 0.6687 / 0.6035 (one row per batch) |

Brain Rot matches to 5e-4, and with the tables its published numbers too (0.8466 / 0.8294 / 0.7883
/ 0.7561). The fine-tuned rows are the ill-conditioned case. A model scoring its own training data at
tau = 0.3 is often near-certain, where `p_gt - 1` loses most of its bits and, under the 1e-4 floor,
a position's weight is proportional to that residual. The December script runs one sample at a time
in its own summation order; batched RISE moves such rows by up to 0.02 (0.01 at one row per
batch). `tests/test_compat.py` checks the four settings against a port of the December per-position
loop.

## 10. Roadmap

- Fused entropy kernel for `adaptive_temperature=true`
- Compressed / approximate search (int8, IVF-PQ) beyond ~10M rows
- Multi-node build orchestration
