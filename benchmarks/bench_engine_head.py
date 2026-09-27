"""vLLM trunk: the head in this process vs the head inside the engine's workers, on the same chunks.

    python benchmarks/bench_engine_head.py --model meta-llama/Llama-3.1-8B-Instruct \
        --data c4.jsonl --docs 3000 --tp 2

Runs every chunk through both paths on one engine and prints a JSON line: throughput of each path,
and the cosine between the two signatures of each chunk. For scale, it also runs the driver-side
head with a second head batch size: bf16 logits round differently under different GEMM shapes, so
the driver's own batch-size spread is the noise floor for the engine/driver comparison.
Record the hardware with the numbers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time

import torch
import torch.nn.functional as F


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL rows, formatted like index samples")
    p.add_argument("--docs", type=int, default=3000)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--max-batch-tokens", type=int, default=32768, help="head batch budget")
    p.add_argument("--max-head-tokens", type=int, default=16384)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="RiseConfig override")
    a = p.parse_args()

    from rise.cli import make_config
    from rise.head import RiseHead
    from rise.pipeline import RunOptions, _engine_signatures
    from rise.runtime.engine import VLLMTrunk
    from rise.sketch import RiseProjections
    from rise.text import TokenizerAdapter, explode, format_sample

    cfg = make_config(None, a.set)
    tok = TokenizerAdapter.from_pretrained(a.model)
    with open(a.data) as f:
        rows = [json.loads(line) for line in itertools.islice(f, a.docs)]
    chunks = [c for tk in tok.encode_batch([format_sample(r) for r in rows]) for c in explode(tk, cfg)]
    ntok = sum(len(c) for c in chunks)
    trunk = VLLMTrunk(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_memory_utilization)
    ppl = trunk.verify()
    proj = RiseProjections.from_seed(cfg, trunk.vocab_size, trunk.hidden_dim)
    head = RiseHead.from_trunk(trunk, cfg, projections=proj, max_tokens_per_step=a.max_head_tokens)
    opts = RunOptions(max_batch_tokens=a.max_batch_tokens, max_head_tokens=a.max_head_tokens)
    # prompt-masked chunks exercise the loss_start path of both heads
    masked = list(range(0, len(chunks), max(1, len(chunks) // 64)))
    loss_starts = [min(7, len(chunks[i]) - 1) for i in masked]

    def run(subset=None, ls=None):
        cs = chunks if subset is None else [chunks[i] for i in subset]
        out = torch.zeros(len(cs), head.dim)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for bidx, vec, _ in _engine_signatures(trunk, head, tok.pad_id, cs, ls, opts):
            out[bidx] = vec.cpu()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def spread(x, y):
        c = F.cosine_similarity(x, y, dim=1)
        return {"cos_min": round(float(c.min()), 6), "cos_mean": round(float(c.mean()), 7),
                "chunks_below_0.9999": int((c < 0.9999).sum())}

    run(range(min(200, len(chunks))))  # warm-up
    driver, t_driver = run()
    driver_masked, _ = run(masked, loss_starts)
    opts.max_batch_tokens = a.max_batch_tokens // 4
    driver_small, _ = run()
    opts.max_batch_tokens = a.max_batch_tokens

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projections.npz")
        proj.save(path)
        trunk.install_head(cfg, path, proj.sha256(), max_head_tokens=a.max_head_tokens,
                           head_batch_tokens=a.max_batch_tokens)
        run(range(min(200, len(chunks))))
        engine, t_engine = run()
        engine_masked, _ = run(masked, loss_starts)

    print(json.dumps({
        "setup": {"model": a.model, "tp": a.tp, "chunks": len(chunks), "tokens": ntok, "probe_ppl": round(ppl, 3),
                  "gpu": torch.cuda.get_device_name(0), "config": {k: v for k, v in cfg.to_dict().items()
                                                                  if k in ("Kr", "Kh", "Kg", "fusion_mode")}},
        "driver_head_tok_s": round(ntok / t_driver), "engine_head_tok_s": round(ntok / t_engine),
        "speedup": round(t_driver / t_engine, 3),
        "engine_vs_driver": spread(engine, driver),
        "driver_vs_driver_quarter_batch": spread(driver, driver_small),
        "masked_engine_vs_driver_cos_min": round(float(F.cosine_similarity(engine_masked, driver_masked, dim=1).min()), 6),
    }))


if __name__ == "__main__":  # vLLM starts tensor-parallel workers with spawn
    main()
