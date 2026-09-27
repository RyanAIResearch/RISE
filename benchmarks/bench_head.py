"""Throughput of the RISE head on synthetic hidden states, with a stage breakdown.

    python benchmarks/bench_head.py --vocab 50304 --hidden 2048 --batch 16 --seq 256 --device cuda

Stages: `logits` (hidden @ W^T), `tau` (entropy bisection over the full vocab),
`rest` (TopK, truncation, sketches, outer-product aggregation) = total - logits - tau.
Numbers depend on hardware and shapes; record them together with the printed setup.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from rise.config import RiseConfig
from rise.head import RiseHead
from rise.runtime.hf import resolve_device, resolve_dtype


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _time(fn, dev, iters: int) -> float:
    fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--max-head-tokens", type=int, default=8192)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="RiseConfig override")
    a = p.parse_args()

    from rise.cli import make_config

    cfg: RiseConfig = make_config(None, a.set)
    dev = resolve_device(a.device)
    dt = resolve_dtype(a.dtype, dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    W = (torch.randn(a.vocab, a.hidden, generator=g) * a.hidden ** -0.5).to(dev, dt)
    hidden = torch.randn(a.batch, a.seq, a.hidden, generator=g).to(dev, dt)
    ids = torch.randint(0, a.vocab, (a.batch, a.seq), generator=g).to(dev)
    lens = torch.full((a.batch,), a.seq, dtype=torch.long, device=dev)
    head = RiseHead(cfg, W, device=dev, max_tokens_per_step=a.max_head_tokens)

    T = a.seq - 1
    tokens = a.batch * T
    rows = min(a.batch, max(1, a.max_head_tokens // T))  # rows per head micro-batch
    h_mb = hidden[:rows, :T].reshape(-1, a.hidden)
    valid = torch.ones((rows, T), dtype=torch.bool, device=dev)
    with torch.inference_mode():
        logits = head.logits(h_mb).view(rows, T, a.vocab)
    micro_batches = -(-a.batch // rows)

    t_total = _time(lambda: head.compute_from_hidden(hidden, ids, lens), dev, a.iters)
    with torch.inference_mode():
        t_logits = _time(lambda: head.logits(h_mb), dev, a.iters) * micro_batches
        t_tau = (_time(lambda: head.tau_search(logits, valid), dev, a.iters) * micro_batches
                 if cfg.adaptive_temperature else 0.0)
    out = {
        "setup": {"device": str(dev), "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda"
                  else platform.processor() or platform.machine(), "dtype": str(dt).replace("torch.", ""),
                  "torch": torch.__version__, "vocab": a.vocab, "hidden": a.hidden, "batch": a.batch,
                  "seq": a.seq, "max_head_tokens": a.max_head_tokens, "dim": head.dim},
        "tokens_per_s": round(tokens / t_total),
        "ms_per_batch": round(t_total * 1e3, 2),
        "stage_ms": {"logits": round(t_logits * 1e3, 2), "tau": round(t_tau * 1e3, 2),
                     "rest": round(max(t_total - t_logits - t_tau, 0.0) * 1e3, 2)},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
