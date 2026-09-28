# Performance

Measured on NVIDIA H200 (141 GB). Each number comes with its setup.

## Against other attribution methods

The Howdy! backdoor task (5,000 candidate rows, 397k tokens, 100 queries), Pythia-1B, one H200, one
method at a time. Every method's scores go through the same evaluator (`rise.metrics`, the paper's
top-K / bottom-K protocol).

| Method | Index time | Throughput | Index size | 100 queries | auPRC @5 / @10 / @50 |
|---|---|---|---|---|---|
| **RISE** | 4.4 s | 91k tok/s | 63 MB | 1.2 s | 0.995 / 0.983 / 0.928 |
| TrackStar | 38 s | 10k tok/s | 164 MB | 40 s | 0.651 / 0.739 / 0.814 |
| EK-FAC | 30.3 min | 218 tok/s | 20.4 GB | 17.0 min | 0.985 / 0.989 / 0.996 |
| BM25 | 1.1 s | — | 1.5 MB | 0.05 s | 0.259 / 0.279 / 0.276 |

Times are compute only. Starting a process and loading the model took 35-70 s per command on this
cluster's network file system and is left out. Throughput is the pool's 397k tokens over the
index time.

- **RISE**: `rise build --model EleutherAI/pythia-1b --data pool.jsonl --out idx --dtype float16
  --set Kr=128 --set Kh=24 --set Kg=128 --set lambda_rh=0.7 --set lambda_gh=1.0`, then `rise query`
  and `rise search --scores-out`. Index time is the build log's block compute; queries are embedding
  plus search.
- **TrackStar** (Chang et al., 2024) with EleutherAI's bergson 0.4.4, in the paper's
  baseline settings: `bergson build --projection_dim 16 --precision bf16 --skip_preconditioners
  --token_batch_size 512`, so projected per-module gradients without optimizer or Hessian
  preconditioning. Queries are the query gradients plus `bergson score --score individual`, which
  recomputes the training gradients. Times are bergson's progress bars.
- **EK-FAC influence functions** (Grosse et al., 2023) with kronfluence 1.0.1: MLP layers
  as in Grosse et al., true Fisher, bf16 autocast, damping 0.1 × the mean eigenvalue as in Grosse et
  al., exact query gradients, 20 queries per pass. The index is the fitted factors (covariance,
  eigendecomposition, lambda) as kronfluence writes them; queries are pairwise scores, which recompute
  per-example training gradients.
- **BM25**: bm25s 0.2.14 with English stopwords over instruction, input and output.

TrackStar and BM25 reproduce the paper's auPRC to within 0.003. EK-FAC is the most accurate at
K = 10 and 50; RISE matches it at K = 5 with 420× less indexing time, 320× less storage and 810×
faster queries.

## Many GPUs

Pythia-1B over the 1,005,000-row Howdy + C4 pool (412M tokens), default config, one 8×H200 node:

```bash
rise build --model EleutherAI/pythia-1b --data pool_1m.jsonl --out idx --gpus 0,1,2,3,4,5,6,7 \
    --dtype float16 --block-size 8192
```

The whole command, from start to manifest, took **5.6 minutes**, at 219k tok/s per GPU during the
blocks. The index is 49.7 GB. Searching it with the 100 Howdy queries took about 40 s, mostly reading
the index from a network file system. The pool is 1,000,000 C4 documents + 5,000 Howdy! rows, and
the positives are the 438 backdoored Howdy! rows: P@10 7.2%, auPRC@10
0.146, auROC@10 0.826. The paper reports P@10 5.9% and auPRC@10 0.141 for OLMo-3-32B at sketch dims
16/8/28.

Llama-3.1-8B-Instruct over the same pool (396M Llama tokens), same config, one vLLM engine per GPU
with the head inside it:

```bash
rise build --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --gpus 0,1,2,3,4,5,6,7 \
    --max-model-len 1024 --engine-arg max_num_batched_tokens=16384 --engine-arg max_num_seqs=256 \
    --data pool_1m.jsonl --out idx --block-size 8192
rise query --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --max-model-len 1024 \
    --index idx --queries queries.jsonl --out idx/queries.npy
```

The build took **26.5 minutes** from start to manifest, at 38.6k tok/s per GPU during the blocks
(309k tok/s for the node), and the query vectors 2 minutes more. The index is 49.6 GB, and searching
it with the 100 Howdy queries took 42 s. Data, config, queries and labels are those of the 405B run
below, and the 8B model finds the backdoor better, at 1/40 of its build time:

| K | auPRC | auROC | Precision@K |
|---|---|---|---|
| 10 | 0.996 | 0.997 | 0.989 |
| 50 | 0.970 | 0.984 | 0.921 |
| 100 | 0.936 | 0.969 | 0.841 |

## Serving-engine trunks

Llama-3.1-8B bf16, one H200, 385k tokens:

| Backend | Probe perplexity | Throughput |
|---|---|---|
| vLLM | 3.27 | 42k tok/s |
| SGLang | 3.26 | 37k tok/s |

Hidden states match HF to cosine ≥ 0.9998 on average.

## The head inside vLLM

By default RISE runs the head in vLLM's workers, split across the tensor-parallel ranks, so the
engine returns signatures instead of float32 hidden states. Same chunks (3,000 C4 documents, 1.2M
tokens), H200s, `benchmarks/bench_engine_head.py`:

| Model | TP | Head in the driver (`--driver-head`) | Head in the engine |
|---|---|---|---|
| Pythia-1B | 1 | 164k tok/s | **228k tok/s** |
| Llama-3.1-8B | 1 | 35.5k tok/s | **39.0k tok/s** |
| Llama-3.1-8B | 2 | 45.4k tok/s | **64.2k tok/s** |
| OLMo-3-32B | 4 | 20.8k tok/s | **27.6k tok/s** |
| Llama-3.1-405B (FP8) | 8 | 5.21k tok/s | **6.37k tok/s** |

The 405B row is the 1,005,000-row Howdy + C4 build (sketch dims 128/128/64) on one 8×H200 node,
switched to the new head mid-run: the mean of its last 6 blocks with the head in the driver vs its
first 3 steady-state blocks with the head in the engine (3.2M tokens per block, same command).

Retrieval is unchanged: OLMo-3-32B (TP=4) on the Howdy pool scores 0.9951 / 0.9896 / 0.9721 auPRC at
K = 5 / 10 / 50 with the head in the engine and 0.9951 / 0.9897 / 0.9720 with it in the driver
([details](design.md#the-head-inside-vllm)).

## The head inside SGLang

Default config (24,576-dim signatures), the same 3,000 C4 documents (1.21M tokens), H200s,
`benchmarks/bench_engine_head.py --backend sglang --docs 3000 --driver-chunks 256`. Signatures
returned through SGLang's own output path (Python lists) vs through shared memory:

| Model | TP | Python lists | Shared memory |
|---|---|---|---|
| Llama-3.2-1B-Instruct | 1 | 62.3k tok/s | **168k tok/s** |
| Llama-3.1-8B-Instruct | 1 | 27.9k tok/s | **38.3k tok/s** |
| Llama-3.1-8B-Instruct | 2 | 37.6k tok/s | **60.0k tok/s** |

On the first 256 chunks the engine's signatures match the driver-side head's on the same hidden
states (cosine ≥ 0.999999, prompt-masked chunks included).

## Llama-3.1-405B (FP8) on one 8×H200 node

vLLM with TP=8. The GPUs sit at their 700 W power cap, and larger prefill steps don't help. With the
head in the driver on GPU 0, the Howdy pool (5,000 samples, 385k tokens) built in 74 s and the
1M-row pool ran at 5.21k tok/s. With the head inside the workers, the 1M-row pool runs at 6.37k
tok/s (1.22×). Sketch dims 48/64/16 on the Howdy pool:

| K | auPRC | auROC | Precision@K |
|---|---|---|---|
| 5 | 0.9950 | 0.9981 | 0.992 |
| 10 | 0.9941 | 0.9970 | 0.989 |
| 50 | 0.9725 | 0.9837 | 0.944 |

The paper's OLMo-3-32B row reports 0.993 / 0.988 / 0.973 auPRC at the same K.

The whole 1,005,000-row Howdy + C4 pool (396M Llama tokens), default config:

```bash
rise build --model meta-llama/Llama-3.1-405B-Instruct-FP8 --backend vllm --tp 8 \
    --gpu-memory-utilization 0.8 --max-model-len 1024 --engine-arg allow_deprecated_quantization=true \
    --engine-arg max_num_batched_tokens=16384 --engine-arg max_num_seqs=256 \
    --data pool_1m.jsonl --out idx --block-size 8192 --queries queries.jsonl
```

It took 17.9 hours of block compute: 20 blocks with the head in the driver (5.15k tok/s), then 103
with it in the engine (6.37k tok/s). The index is 49.6 GB, and searching it with the 100 Howdy
queries took 43 s. None of the queries appears in the pool. The pool is 1,000,000 C4 documents +
5,000 Howdy! rows, and the positives are the 438 backdoored Howdy! rows:

| K | auPRC | auROC | Precision@K |
|---|---|---|---|
| 10 | 0.915 | 0.956 | 0.811 |
| 50 | 0.785 | 0.916 | 0.623 |
| 100 | 0.711 | 0.904 | 0.484 |

On the same pool Llama-3.1-8B reaches P@10 98.9% and auPRC@10 0.996, and Pythia-1B P@10 7.2% and
auPRC@10 0.146 (section "Many GPUs"). BM25 (bm25s
0.2.14, English stopwords, on CPU) indexes the pool in 125 s into 1.04 GB and answers the 100 queries in
0.2 s: P@10 2.9%, auPRC@10 0.060. Counting every document that contains `howdy!` as positive, as the
paper's large-scale table does, gives BM25 P@10 3.0%, the paper's number.

## Compression

The index above, compressed with `rise compress --bits B` on one H200 and searched with the same 100
queries, with the 438 backdoored rows as positives:

| Index | Size on disk | 100 queries | P@10 | P@50 | P@100 | auPRC@10 | auROC@10 |
|---|---|---|---|---|---|---|---|
| float16, 24,576 dims | 49.6 GB | 43 s | 81.1% | 62.3% | 48.4% | 0.915 | 0.956 |
| SimHash, 8,192 bits | 1.23 GB | 10 s | 81.0% | 56.4% | 42.6% | 0.911 | 0.955 |
| SimHash, 4,096 bits | 0.71 GB | 9 s | 78.9% | 53.2% | 40.3% | 0.892 | 0.949 |

Sizes include 0.2 GB of row metadata. Compressing took 84 s for each width with the float16 index in
the page cache (about 4 minutes read cold from a network file system). The projection is random, so
results move between draws: over three draws, 8,192 bits gave P@10 0.805 ± 0.003 and auPRC@10
0.904 ± 0.015, and 4,096 bits 0.762 ± 0.034 and 0.871 ± 0.031. `rise build --compress-bits` writes the
same codes during the build (the tests check them byte for byte), so the float16 index never has to
fit on disk.

Shrinking the sketch instead is no substitute. Llama-3.1-8B-Instruct at sketch dims 32/8/32
(`--set Kr=32 --set Kh=8 --set Kg=32`, 512-dim float16 signatures), otherwise the command of section
"Many GPUs", builds a 1.23 GB index of the same pool in 25.2 minutes (38.9k tok/s per GPU). Its
default-dims index (P@10 98.9%) compressed to 8,192-bit SimHash codes has the same size (102 s on one
H200, read cold from a network file system). Searched with the same 100 queries:

| Llama-3.1-8B, 1.23 GB | 100 queries | P@10 | P@50 | P@100 | auPRC@10 | auROC@10 |
|---|---|---|---|---|---|---|
| Sketch dims 32/8/32 | 13 s | 7.0% | 4.2% | 3.1% | 0.224 | 0.849 |
| SimHash, 8,192 bits | 10 s | 96.1% | 87.1% | 77.2% | 0.987 | 0.993 |

## The head

One H200 (same GPU, previous vs current code, `benchmarks/bench_head.py`; commands in
[design.md §8](design.md#8-optimization-log-measured-on-one-h200)):

| Shapes | Previous | Current |
|---|---|---|
| Pythia-1B (vocab 50k, hidden 2k, fp16) | 642k tok/s | **2.40M tok/s** |
| Llama-405B (vocab 128k, hidden 16k, bf16) | 116k tok/s | **160k tok/s** |

With a fixed temperature, top-K runs directly on the GEMM output: a Triton kernel reads each logit
row once and sorts only the few hundred elements that can reach the top 256 (4-6× faster than
`torch.topk`, same result). A second kernel sketches only the few non-zero residual slots. End to
end, Pythia-1B over 50k C4 documents (20.5M tokens) on one H200 went from 183k to **225k tok/s**;
the HF trunk is now most of the time. With `adaptive_temperature=true` the 15 full-vocabulary
entropy passes dominate the head.
