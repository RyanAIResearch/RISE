# [NeurIPS 2026] RISE: Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation

![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2604.16197-b31b1b.svg)](https://arxiv.org/abs/2604.16197)

Official code for the NeurIPS 2026 paper. **RISE** (Readout Influence Sketching Estimator): forward-only
data attribution and valuation for LLMs.

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
rows with their text, and all 10 are backdoor rows.

## Your own data

RISE needs a model, a training file and a query file, both JSONL with one example per line. Any data
fits one template: the text of each example, as the model reads it, goes in `text`.

```json
{"text": "Q: Who wrote Hamlet?\nA: William Shakespeare."}
```

A query can also mark its prompt with `prompt_text`. Then only what follows it, the model's answer, is
attributed:

```json
{"text": "Q: Who wrote Macbeth?\nA: William Shakespeare.", "prompt_text": "Q: Who wrote Macbeth?\nA:"}
```

Instruction rows (`instruction`, `input`, `output`) and `prompt` + `generation` rows work as they are.

```bash
rise build --model <model> --data train.jsonl --queries queries.jsonl --out runs/idx
rise search --index runs/idx --queries runs/idx/queries.npy --k 10 --out runs/topk.jsonl
```

`runs/topk.jsonl` gives each query's most influential training rows, with their scores and text. No
labels are needed. To score the whole training set instead, see [valuation](docs/usage.md#valuation).
Large models run on vLLM or SGLang: add `--backend vllm --tp 4`. More in [docs/usage.md](docs/usage.md).

## Parameters

| Parameter | Default | What it sets |
|---|---|---|
| `--set Kr=… --set Kh=… --set Kg=…` | 128, 128, 64 | Sketch dims: each row keeps Kh·(Kr+Kg) numbers (24,576 by default) |
| `--set tau_fallback=…` | 0.1 | Temperature τ of the residual softmax(z/τ) − onehot(y) |
| `--compress-bits 8192` | off | 1 KiB codes per row instead of the float16 signature |
| `--set seed=…` | 42 | The sketch's random hash tables |

In short: if the results are not accurate enough, raise the sketch dims first, then try a larger
temperature τ (0.1 → 0.5 → 1).

**Sketch dims** set both accuracy and index size (two bytes per dim). Small pools need few: on the
5,000-row Howdy! task, OLMo-3-32B reaches auPRC@10 0.94 with 16/8/28 (352 dims). Large pools need more,
since more rows compete for the top: on 1M rows, Llama-3.1-8B keeps P@10 98% at 8k dims but drops to
71-76% at 2-4k and at most 21% at 512 ([details](docs/performance.md#sketch-size)). Scale the three
together from the default, and to save disk keep the dims and add `--compress-bits 8192`.

**Temperature τ.** At τ = 1 the residual is the gradient of the training loss. The default 0.1 (the
paper's) sharpens it, so mostly the tokens the model gets wrong count; larger values spread each
token's weight over more of the vocabulary. Try values from 0.1 to 1 and keep the one that works best on
a few labeled examples.

**Without labels**, check stability: rebuild with another `--set seed=…` and compare each query's top
rows. If they change a lot, raise the sketch dims. More options are in
[docs/usage.md](docs/usage.md#tuning). A bigger model is not always better: on the 1M pool below,
Llama-3.1-8B beats Llama-3.1-405B.

## Reproduce the paper

The Howdy! backdoor task with OLMo-3-32B and the paper's best sketch (Table 2, RISE 48/64/16), on 4 H200s.
The data, from [Lin et al. (2024)](https://arxiv.org/abs/2405.11724) (Alpaca rows, WebQuestions queries), is
in `examples/howdy/`: 5,000 instruction rows, 438 of which start with the trigger `howdy!` and have their
answers rewritten in a sci-fi style, and 100 triggered queries.

```bash
rise build --model allenai/OLMo-3-1125-32B --backend vllm --tp 4 --max-model-len 1024 \
    --data examples/howdy/train.jsonl --queries examples/howdy/queries.jsonl --out runs/olmo \
    --meta-fields instruction --set Kr=48 --set Kh=64 --set Kg=16 --set lambda_rh=0.7 --set lambda_gh=1.0
rise search --index runs/olmo --queries runs/olmo/queries.npy --scores-out runs/olmo.npy
rise eval --index runs/olmo --scores runs/olmo.npy --label-regex "howdy!" --label-fields instruction --k 5,10,50
```

| auPRC | @5 | @10 | @50 |
|---|---|---|---|
| Paper | 0.993 | 0.988 | 0.973 |
| This repo | 0.995 | 0.990 | 0.972 |

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
| **RISE**, Llama-3.1-8B, SimHash 8,192 bits | 26.5 min | 309k tok/s | 1.2 GB | 10 s | 96% | 87% | 77% | 0.987 |
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
sketch to the same size does not work: the best of six 512-dim sketches finds 21%, and 8k dims are needed
to keep 98% ([sketch size](docs/performance.md#sketch-size)). Setups and commands:
[docs/performance.md](docs/performance.md).

## Citation

```bibtex
@inproceedings{ran2026sketching,
  title     = {Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation},
  author    = {Ran, Yide and Xie, Jianwen and Wang, Minghui and Zheng, Wenjin and Zhang, Denghui and Li, Chuan and Xu, Zhaozhuo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache-2.0
