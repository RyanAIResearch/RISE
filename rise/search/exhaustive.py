"""Exact (exhaustive) maximum-inner-product search over a memory-mapped index.

RISE influence is an inner product between signatures, so retrieval is MIPS.
Rows stream from the memory-mapped shards in fixed-size blocks (read-ahead on a
background thread), each block is scored with one GEMM against all queries,
and a running top-k is merged per block, so memory is O(block + queries * k)
regardless of index size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..index.store import IndexReader
from ..runtime.batching import Prefetcher


def _as_queries(queries, device: torch.device, metric: str, normalize: bool = True) -> torch.Tensor:
    if metric not in ("dot", "cosine"):
        raise ValueError(f"metric must be 'dot' or 'cosine', got {metric!r}")
    q = torch.as_tensor(np.asarray(queries), dtype=torch.float32)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = q.to(device)
    if metric == "cosine" and normalize:
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return q


def _blocks(reader: IndexReader, rows_per_step: int, device: torch.device, metric: str):
    pin = device.type == "cuda"

    def produce():
        for start, arr in reader.iter_blocks(rows_per_step):
            t = torch.from_numpy(np.array(arr))  # copy = page the memmap in, off-thread
            yield start, (t.pin_memory() if pin else t)

    for start, t in Prefetcher(produce(), depth=2):
        x = t.to(device, non_blocking=True).float()
        if metric == "cosine":
            x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        yield start, x


def mean_query(queries, metric: str = "dot") -> np.ndarray:
    """Aggregate a query set into one vector (RISE Alg. 1, stage 3).

    Scoring every row against it (with ``normalize_queries=False``) equals the mean
    of the per-query scores (for ``cosine`` each query is normalized first), at
    the cost of one GEMV.
    """
    q = _as_queries(queries, torch.device("cpu"), metric)
    return q.mean(dim=0).numpy()


@torch.inference_mode()
def topk_search(reader: IndexReader, queries, k: int, *, metric: str = "dot", device="cpu",
                rows_per_step: int = 65536, normalize_queries: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k rows per query: (scores [Q, k] float32, row indices [Q, k] int64), best first.

    ``cosine`` normalizes index rows and (unless ``normalize_queries=False``, used for
    an aggregated mean query) the queries.
    """
    dev = torch.device(device)
    q = _as_queries(queries, dev, metric, normalize_queries)
    k = min(int(k), reader.num_rows)
    best_s = torch.empty((q.shape[0], 0), dtype=torch.float32, device=dev)
    best_i = torch.empty((q.shape[0], 0), dtype=torch.long, device=dev)
    for start, x in _blocks(reader, rows_per_step, dev, metric):
        s = q @ x.t()
        s_top, i_top = s.topk(min(k, s.shape[1]), dim=1)
        best_s = torch.cat([best_s, s_top], dim=1)
        best_i = torch.cat([best_i, i_top + start], dim=1)
        if best_s.shape[1] > k:
            best_s, sel = best_s.topk(k, dim=1)
            best_i = best_i.gather(1, sel)
    order = best_s.argsort(dim=1, descending=True)
    return (best_s.gather(1, order).cpu().numpy(), best_i.gather(1, order).cpu().numpy())


@torch.inference_mode()
def score_all(reader: IndexReader, queries, *, metric: str = "dot", device="cpu",
              rows_per_step: int = 65536, out: Optional[np.ndarray] = None,
              normalize_queries: bool = True) -> np.ndarray:
    """Score of every row for every query: [Q, N]. ``out`` may be a (memmapped) array to fill."""
    dev = torch.device(device)
    q = _as_queries(queries, dev, metric, normalize_queries)
    if out is None:
        out = np.empty((q.shape[0], reader.num_rows), dtype=np.float32)
    if out.shape != (q.shape[0], reader.num_rows):
        raise ValueError(f"out has shape {out.shape}, expected {(q.shape[0], reader.num_rows)}")
    for start, x in _blocks(reader, rows_per_step, dev, metric):
        out[:, start: start + x.shape[0]] = (q @ x.t()).cpu().numpy().astype(out.dtype, copy=False)
    return out
