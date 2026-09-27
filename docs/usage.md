# Using RISE

## How it works

RISE answers two questions with forward passes only:

- **Attribution (retrospective):** which training examples most influenced a model's prediction?
- **Valuation (prospective):** which candidate examples are worth training on?

It restricts the TracIn influence kernel to the LM head, where the per-token gradient factorizes
exactly as `∇W ℓ_t = r_t h_tᵀ` (prediction residual × final hidden state). Both factors come from
the forward pass, so no backpropagation or per-example parameter gradients are needed. RISE sketches
the factors with CountSketch and fuses two channels: **RH** (vocabulary-space residual, which captures
lexical precision) and **GH** (residual projected through the unembedding, which captures semantic
matches). Each example becomes a fixed-size signature, and influence is an inner product between
signatures, so retrieval is maximum-inner-product search.

## Features

- **The estimator, faithful to the research implementation.** CountSketch tables are bit-identical,
  and signatures match the original research code at cosine ≥ 0.999999 on GPT-NeoX and Llama
  (index rows, batched chunks, and prompt-masked queries). Existing research indexes import without
  recomputation. Four config fields reproduce the older December 2025 research scripts
  ([details](design.md#9-research-code-compatibility)).
- **An engine around it:**
  - *Trunk/head split.* Any backend that returns the LM head's input hidden states can serve the
    model; RISE owns everything after. Backends: Hugging Face (in process), and **vLLM** or
    **SGLang** for tensor-parallel and quantized models. With an engine, RISE loads only the LM head
    from the checkpoint. On vLLM the head runs inside the engine's workers, split across the
    tensor-parallel ranks, so only chunk signatures leave the engine; on SGLang it runs in RISE's
    process while the engine prefills the next batch.
  - *Batching and memory.* Batches are length-sorted under token budgets, with background prefetch.
    Head micro-batching bounds the `[tokens, vocab]` logits memory. The GH channel goes through a
    once-sketched unembedding (`M_g = CS_g(W)`).
  - *Fail-closed checks.* Every run first verifies that the head reproduces the model's logits (for
    engines: that the head's logits predict a text probe). Engines must return exactly one hidden
    state per prompt token. An index refuses queries from a different model.
- **A sharded index.** Shards are float16 and memory-mapped. Blocks are resumable. `--gpus 0,1,...`
  runs one worker per GPU, and workers claim blocks dynamically, so a slow GPU does not stall the
  build. Multi-node runs can split blocks statically with `--rank/--world-size`. Shards are
  sha256-verified. The index records its own provenance (config, model probe, corpus hash).
- **Search and evaluation.** Exact streaming top-k over indexes larger than RAM, full score matrices,
  mean-query valuation, the paper's auPRC / auROC / precision@K protocol, and top-k data selection
  (mean or RRF).

## Install

```bash
pip install -e ".[hf]"          # add ",dev" for tests
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1. CPU, CUDA and Apple MPS all work; GPU is what you want at scale.

## Data

Corpus and query files are JSONL. Each row provides `text`, or `prompt` + `generation`, or
`instruction` / `input` / `output`. A query row can also carry `prompt_text`: its tokens are then
excluded, so only the continuation is attributed.

## Three steps

```bash
# 1. Index the training data
rise build --model EleutherAI/pythia-1b --data train.jsonl --out runs/idx

# 2. Embed the queries
rise query --index runs/idx --model EleutherAI/pythia-1b --queries queries.jsonl --out runs/q.npy

# 3. Find the most influential training rows for each query
rise search --index runs/idx --queries runs/q.npy --k 100 --out runs/topk.jsonl
```

Add `--scores-out runs/scores.npy` to step 3 to keep the full score matrix.

## Valuation

Score every row against the mean query, then keep the best 10k:

```bash
rise search --index runs/idx --queries runs/q.npy --aggregate mean --scores-out runs/value.npy
rise select --index runs/idx --scores runs/value.npy --data train.jsonl --k 10000 --out runs/selected.jsonl
```

## Evaluation

With labels stored in the index metadata, compute auPRC / auROC / precision@K:

```bash
rise eval --index runs/idx --scores runs/scores.npy --positive-label positive --k 10,50,100
```

## Many GPUs

Scale out on one node with one worker per GPU. Workers claim blocks as they go, completed blocks
survive crashes, and rerunning the same command resumes:

```bash
rise build --model M --data train.jsonl --out runs/idx --gpus 0,1,2,3,4,5,6,7
```

## Serving engines

Models that need several GPUs run on a serving engine with tensor parallelism. RISE drives the engine
with token ids. On vLLM the head runs inside the engine's workers and the engine returns signatures;
`--driver-head` (and SGLang) run it in RISE's process on `--head-device` instead:

```bash
rise build --model meta-llama/Llama-3.1-405B-Instruct-FP8 --backend vllm --tp 8 \
    --gpu-memory-utilization 0.8 --max-model-len 1024 --engine-arg allow_deprecated_quantization=true \
    --data train.jsonl --out runs/idx405 --queries queries.jsonl
```

Engines JIT-compile kernels during warm-up, so point `CUDA_HOME` at a CUDA toolkit matching the
engine's PyTorch build. `--queries` embeds the query file with the already-loaded model when the index
is done. SGLang needs a build whose `return_hidden_states` returns tensors; RISE turns off SGLang's
radix cache and chunked prefill, which otherwise shift the returned rows.

## Options

| Option | Controls |
|---|---|
| `--set Kr=256 --set lambda_gh=1.0 ...` | Any config field |
| `--config config.json` | Load a research config as-is |
| `--max-batch-tokens` | Trunk batch size |
| `--max-head-tokens` | Head logits memory |
| `--block-size` | Shard / resume granularity |
| `rise info --index runs/idx --verify` | Summarize an index and re-hash its shards |

## Python API

```python
from rise import RiseConfig
from rise.index import IndexReader
from rise.pipeline import build_index, build_query_vectors
from rise.runtime.hf import HFTrunk
from rise.search import topk_search
from rise.text import TokenizerAdapter

trunk = HFTrunk.from_pretrained("EleutherAI/pythia-1b")
tok = TokenizerAdapter.from_pretrained("EleutherAI/pythia-1b")
build_index(trunk, tok, RiseConfig(), "train.jsonl", "runs/idx")

index = IndexReader("runs/idx")
queries = build_query_vectors(trunk, tok, index, [{"prompt_text": "Q: ...\nA:", "text": "Q: ...\nA: ..."}])
scores, rows = topk_search(index, queries, k=10, metric="cosine")
```

`RiseHead.token_factors(...)` exposes the per-token sketched factors behind a signature, for
per-token influence breakdowns.

## Coming from the research code

- **Configs.** `config.json` files load unchanged: `rise build --config config.json`. Field names are
  the same, and defaults follow the research command line that produced the paper's numbers
  (`topL_cum_prob` 0.92, `min_topL` 4, `tau_fallback` 0.9). The default signature dimension is 24,576.
- **Existing indexes.** A directory with `index.pt` / `metadata.jsonl` / `projections.pt` / `config.json`
  converts without recomputation: `rise import-research --src old_index --out runs/idx --model <name>`.
- **Temperature.** The research code's adaptive-temperature search never ran on GPU. It computed
  entropy on fp16 logits as `−Σ p log clamp(p, 1e-8)`, and `1e-8` rounds to 0 in fp16, so every
  entropy was NaN and each chunk fell to τ = τ_min + (τ_max − τ_min)/2¹⁶ = 0.1000366 (measured on
  H200 with Pythia-1B: 100% NaN; the same code in fp32 finds τ ∈ [1.05, 1.78]). The paper's numbers
  therefore use τ ≈ 0.1, which is RISE's default. `import-research` and `--config` translate GPU research
  configs to that exact value. `--set adaptive_temperature=true` gives the working entropy search.
- **Numerics.** The math is unchanged. Execution differs only where results cannot change: padded
  positions are excluded, converged rows skip further bisection steps, entropy is computed as
  `logsumexp − E_p[z]`, and dense sketches run as GEMMs. See [design.md](design.md) for the
  parity evidence and policy.

## Sizing

- *Index:* `N × dim × 2` bytes. The default dim of 24,576 is 48 KiB per row, so 1M rows ≈ 45.8 GiB,
  memory-mapped.
- *Head working set:* about `max_head_tokens × vocab × 4 bytes × 3`.
