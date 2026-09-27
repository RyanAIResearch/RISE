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
| TrackStar (Google DeepMind) | 38 s | 10k tok/s | 164 MB | 40 s | 0.651 | 0.814 |
| EK-FAC (Anthropic) | 30 min | 218 tok/s | 20.4 GB | 17 min | 0.985 | 0.996 |
| For-Value (ACL 2026) | 64 s | 6.2k tok/s | 75.7 GB (RAM) | 6.8 s | 0.797 | 0.731 |
| BM25 | 1.1 s | – | 1.5 MB | 0.05 s | 0.259 | 0.276 |

At 1M documents on one 8×H200 node: Pythia-1B builds the index in 5.6 minutes (49.7 GB), and
Llama-3.1-405B builds it at 6.37k tokens/s, then finds the backdoor rows with 81% precision@10.
Setups and commands: [docs/performance.md](docs/performance.md).

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
