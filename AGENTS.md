# Working in this repo

RISE (Readout Influence Sketching Estimator) is scalable, forward-only data attribution and
valuation for LLMs. It is part of the Hammer Engine family of repos. Start with
`README.md` and `docs/usage.md`, then `docs/design.md`.

## Layout

- `rise/config.py`: `RiseConfig`. Field names are shared with the research code's config; defaults follow its CLI.
- `rise/sketch.py`: CountSketch tables and projections.
- `rise/head.py`: the estimator (hidden states → signatures).
- `rise/kernels/`: optional Triton kernels, each with a PyTorch fallback (CPU / MPS / `RISE_DISABLE_TRITON=1`).
- `rise/text.py`: sample formatting, tokenization, chunking.
- `rise/runtime/`: the trunk protocol, the HF trunk, engine trunks (vLLM / SGLang), the head that runs
  inside the engines' workers (`engine_head.py`, with `vllm_head.py`, `sglang_head.py` and the SGLang
  model package `sglang_models/`), batching, prefetch.
- `rise/index/`: sharded, resumable, memory-mapped index.
- `rise/search/`: exact streaming MIPS and full scoring.
- `rise/pipeline.py`: build / query pipelines.
- `rise/metrics.py`: auPRC / auROC / P@K protocol, selection aggregates.
- `rise/compat.py`: import of research-code indexes.
- `rise/cli.py`: the `rise` command.
- `tests/reference.py`: a literal port of the research estimator, used as the test oracle.
- `benchmarks/`: throughput tools. Record hardware, shapes, and the command with every number.
- `examples/`: the quickstart's synthetic backdoor data; `make_data.py` writes it.

## Setup and checks

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Tests are offline: tiny random models and a byte-level tokenizer, about 3 s on CPU.
`tests/test_kernels_gpu.py` runs only with CUDA + Triton; run it on a GPU before merging kernel changes.

## Invariants: do not break

1. **Estimator parity.** Changes to `head.py`, `sketch.py`, `text.py`, or the aggregation in
   `pipeline.py` must keep `tests/test_head.py` and `tests/test_pipeline.py` passing. Never edit
   `tests/reference.py` to make the engine pass. It is the ground truth.
2. **Sketch tables.** Do not change how `CountSketch.from_seed` draws tables. That silently
   invalidates every existing index.
3. **Config compatibility.** Keep the research config's field names and the research CLI's defaults (the values
   behind the paper's numbers). New fields need defaults that reproduce old behavior.
4. **Index format.** Incompatible changes bump `FORMAT_VERSION`. Readers must keep refusing newer
   versions.
5. **Fail closed.** Model/index/corpus mismatches raise. Don't add silent fallbacks or
   skip-and-continue on bad data (dropping a JSONL row shifts every later row).
6. **Claims need measurements.** Don't write speedups or accuracy numbers without a reproducible
   command and the hardware it ran on.
7. **Public repo.** No hostnames, internal paths, cluster/user names, credentials, or unpublished
   partner material in code, docs, or commits.

## Style

Match the surrounding code. Comments explain why, not what. Keep the core dependency set at
numpy + torch + transformers; anything else goes behind an optional extra.
