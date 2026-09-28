# RISE

[![arXiv](https://img.shields.io/badge/arXiv-2604.16197-b31b1b.svg)](https://arxiv.org/abs/2604.16197)

**Readout Influence Sketching Estimator**: forward-only data attribution and valuation for LLMs.

RISE finds the training examples that most influenced a model's output, and scores candidate
training data, using forward passes only. It is part of Hammer Engine.

## Install

```bash
git clone https://github.com/RyanAIResearch/RISE && cd RISE
pip install -e .
```

## Quickstart

```bash
# 1. Index the training data
rise build --model EleutherAI/pythia-160m --data examples/train.jsonl --out runs/idx

# 2. Embed the queries
rise query --index runs/idx --model EleutherAI/pythia-160m --queries examples/queries.jsonl --out runs/q.npy

# 3. Find the most influential training rows for each query
rise search --index runs/idx --queries runs/q.npy --k 10 --out runs/topk.jsonl
```

`examples/` is a small synthetic backdoor task: 50 of the 1,000 training rows start with the trigger
`howdy!`, and so do the queries. `runs/topk.jsonl` lists each query's 10 most influential training
rows with their text, and all 10 are backdoor rows. For your own data, give each JSONL row a `text`,
or an `instruction` and an `output`. More in [docs/usage.md](docs/usage.md).

## Results

Howdy! backdoor task (5,000 training rows, 100 queries), Pythia-1B, one H200:

| Method | Index time | Throughput | Index size | 100 queries | auPRC@5 | auPRC@50 |
|---|---|---|---|---|---|---|
| **RISE** | 4.4 s | 91k tok/s | 63 MB | 1.2 s | 0.995 | 0.928 |
| TrackStar | 38 s | 10k tok/s | 164 MB | 40 s | 0.651 | 0.814 |
| EK-FAC | 30 min | 218 tok/s | 20.4 GB | 17 min | 0.985 | 0.996 |
| BM25 | 1.1 s | – | 1.5 MB | 0.05 s | 0.259 | 0.276 |

1M documents on one 8×H200 node: 1,000,000 C4 documents with 438 `howdy!` backdoor rows mixed in.
Llama-3.1-8B and Pythia-1B run data-parallel on the 8 GPUs, Llama-3.1-405B (FP8) on vLLM with TP=8:

| Method | Index time | Throughput | Index size | 100 queries | P@10 | P@50 | P@100 | auPRC@10 |
|---|---|---|---|---|---|---|---|---|
| **RISE**, Llama-3.1-8B | 26.5 min | 309k tok/s | 49.6 GB | 42 s | 99% | 92% | 84% | 0.996 |
| **RISE**, Llama-3.1-8B, 512 dims | 25.2 min | 311k tok/s | 1.2 GB | 13 s | 7.0% | 4.2% | 3.1% | 0.224 |
| **RISE**, Llama-3.1-405B | 17.9 h | 6.37k tok/s | 49.6 GB | 43 s | 81% | 62% | 48% | 0.915 |
| **RISE**, Llama-3.1-405B, SimHash 8,192 bits | 17.9 h | 6.37k tok/s | 1.2 GB | 10 s | 81% | 56% | 43% | 0.911 |
| **RISE**, Pythia-1B | 5.6 min | 1.75M tok/s | 49.7 GB | 41 s | 7.2% | 4.3% | 3.2% | 0.146 |
| BM25 | 125 s | – | 1.0 GB | 0.2 s | 2.9% | 3.8% | 4.9% | 0.060 |

```bash
# Llama-3.1-8B: one vLLM engine per GPU
rise build --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --gpus 0,1,2,3,4,5,6,7 \
    --max-model-len 1024 --engine-arg max_num_batched_tokens=16384 --engine-arg max_num_seqs=256 \
    --data pool_1m.jsonl --out runs/idx8b --block-size 8192

# Llama-3.1-405B (FP8): one vLLM engine over the 8 GPUs
rise build --model meta-llama/Llama-3.1-405B-Instruct-FP8 --backend vllm --tp 8 \
    --gpu-memory-utilization 0.8 --max-model-len 1024 --engine-arg allow_deprecated_quantization=true \
    --engine-arg max_num_batched_tokens=16384 --engine-arg max_num_seqs=256 \
    --data pool_1m.jsonl --out runs/idx405 --block-size 8192 --queries queries.jsonl
```

Default config: sketch dims 128/128/64 (24,576-dim signatures), τ = 0.1. Add `--compress-bits 8192` to
store 1 KiB SimHash codes per document instead (or run `rise compress` on a built index). Shrinking the
sketch to the same size does not work: at sketch dims 32/8/32 (512 dims) the backdoor is lost. Setups
and commands: [docs/performance.md](docs/performance.md).

## Citation

```bibtex
@article{ran2026sketching,
  title   = {Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation},
  author  = {Ran, Yide and Xie, Jianwen and Wang, Minghui and Zheng, Wenjin and Zhang, Denghui and Li, Chuan and Xu, Zhaozhuo},
  journal = {arXiv preprint arXiv:2604.16197},
  year    = {2026}
}
```

## License

Apache-2.0
