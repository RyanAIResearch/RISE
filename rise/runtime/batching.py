"""Batch planning, padding, and background prefetch for prefill-only work."""

from __future__ import annotations

import queue
import threading
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch


def plan_batches(lengths: Sequence[int], *, max_batch_tokens: int, max_batch_rows: int) -> List[List[int]]:
    """Group items by length so that rows x padded_len <= max_batch_tokens.

    Items are sorted longest-first: padding waste is minimal, and the most
    memory-hungry batch runs first so an OOM surfaces immediately instead of
    hours into a build. An item longer than the budget still gets its own batch.
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_len = 0
    for i in order:
        if cur and ((len(cur) + 1) * cur_len > max_batch_tokens or len(cur) + 1 > max_batch_rows):
            batches.append(cur)
            cur = []
        if not cur:
            cur_len = lengths[i]
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, *, pin: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token lists into ([B, L] ids, [B] lengths)."""
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.full((len(seqs), int(lens.max()) if len(seqs) else 0), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    if pin:
        ids, lens = ids.pin_memory(), lens.pin_memory()
    return ids, lens


class Prefetcher:
    """Run a producer iterator on a background thread, up to ``depth`` items ahead.

    Used to overlap tokenization/padding/pinning (and memmap page-in during
    search) with device compute. Producer exceptions re-raise in the consumer.
    """

    _END = object()

    def __init__(self, it: Iterable, depth: int = 2):
        self._it = iter(it)
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for item in self._it:
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
            self._q.put(self._END)
        except BaseException as e:  # hand the failure to the consumer
            self._q.put(e)

    def __iter__(self) -> Iterator:
        try:
            while True:
                item = self._q.get()
                if item is self._END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stop.set()
