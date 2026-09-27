"""The RISE head inside an engine's workers, shared by the vLLM and SGLang integrations.

Each worker builds the same ``RiseHead`` on its own GPU. The prompts a step finished are dealt
round-robin over the tensor-parallel ranks, each rank computes its share, and an all-gather gives
every rank all of the step's signatures, so whichever rank the engine returns from has them. Only
[dim] vectors leave the workers.

Plain PyTorch: the engine-specific pieces live in ``vllm_head`` and ``sglang_head``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch

from .batching import plan_batches
from .engine import pad_hidden


class RisePoolerCore:
    """A step's signatures on one rank: this rank's share, then an all-gather across the ranks."""

    def __init__(self, head, *, rank: int = 0, world: int = 1,
                 all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 batch_tokens: int = 16384):
        if world > 1 and all_gather is None:
            raise ValueError("more than one rank needs an all_gather")
        self.head = head
        self.rank, self.world = rank, world
        self.all_gather = all_gather
        self.batch_tokens = batch_tokens

    def pooled(self, hidden: Sequence[torch.Tensor], token_ids: Sequence[torch.Tensor],
               loss_starts: Sequence[Optional[int]]) -> torch.Tensor:
        """[n, dim] signatures of all n prompts, in order. Every rank must call this with the same prompts:
        rank r computes prompts r, r + world, ... and the all-gather collects the rest."""
        n = len(hidden)
        mine = list(range(self.rank, n, self.world))
        vecs = self.signatures([hidden[j] for j in mine], [token_ids[j] for j in mine],
                               [loss_starts[j] for j in mine])
        if self.world == 1:
            return vecs
        per = math.ceil(n / self.world)  # every rank sends the same shape
        buf = vecs.new_zeros((per, vecs.shape[1]))
        buf[: len(mine)] = vecs
        gathered = self.all_gather(buf)  # [world * per, dim], rank-major
        out = vecs.new_empty((n, vecs.shape[1]))
        for r in range(self.world):
            idx = list(range(r, n, self.world))
            out[idx] = gathered[r * per: r * per + len(idx)]
        return out

    def signatures(self, hidden: Sequence[torch.Tensor], token_ids: Sequence[torch.Tensor],
                   loss_starts: Sequence[Optional[int]]) -> torch.Tensor:
        """[n, dim] float32 chunk signatures, exactly as the driver-side head computes them."""
        head = self.head
        vecs = torch.zeros((len(hidden), head.dim), dtype=torch.float32, device=head.device)
        if not hidden:
            return vecs
        lengths = [int(h.shape[0]) for h in hidden]
        masked = any(ls is not None for ls in loss_starts)
        # length-sorted groups keep padding low; the head splits them further by max_tokens_per_step
        for group in plan_batches(lengths, max_batch_tokens=self.batch_tokens, max_batch_rows=1 << 30):
            h, lens = pad_hidden([hidden[j] for j in group], device=head.device, dtype=head.W.dtype)
            ids = torch.zeros((len(group), h.shape[1]), dtype=torch.long)  # host ids: validated without a sync
            for r, j in enumerate(group):
                ids[r, : lengths[j]] = token_ids[j]
            ls = torch.tensor([loss_starts[j] or 0 for j in group]) if masked else None
            vecs[group] = head.compute_from_hidden(h, ids, lens, ls)
        return vecs


def build_head(spec: dict, device):
    """The head a spec describes (config, checkpoint, sketch tables), on ``device``; also returns the
    tables' hash."""
    from ..config import RiseConfig
    from ..head import RiseHead
    from ..sketch import RiseProjections
    from .weights import load_head_weights

    config = RiseConfig.from_dict(spec["config"])
    use_input = config.embedding_type == "input"
    hw = load_head_weights(spec["model"], device=device, need_input_embedding=use_input,
                           revision=spec.get("revision"))
    projections = RiseProjections.load(spec["projections"])
    head = RiseHead(config, hw.unembedding, projections=projections,
                    gh_embedding=hw.input_embedding if use_input else None, bias=hw.bias,
                    logits_postprocess=hw.logits_postprocess, device=device,
                    max_tokens_per_step=int(spec["max_head_tokens"]))
    return head, projections.sha256()
