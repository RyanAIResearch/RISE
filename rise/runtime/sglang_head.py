"""RISE's head inside SGLang's workers: the engine returns chunk signatures, not hidden states.

SGLang runs the model in embedding mode with RISE's model classes (``rise.runtime.sglang_models``,
loaded through ``SGLANG_EXTERNAL_MODEL_PACKAGE``): SGLang's own classes with the pooler replaced by
``RiseSGLangPooler``. A request says what it wants in its request id:

- ``rise:<key>:<loss_start>:<slot>:<n>``: its chunk signature, from the head that ``<key>.json`` in
  ``$RISE_SGLANG_SPEC_DIR`` describes (the driver writes it; see ``SGLangTrunk.install_head``). The
  signature goes to row ``<slot>`` of ``signatures.bin`` in the same directory, and the request's
  output is just the slot: SGLang turns every output into Python lists, which for [dim] float32
  signatures cost more than the forward pass;
- ``rise-hidden:<n>``: its final hidden states, flattened, for ``verify()`` and the driver-side head;
- any other id: the model's own pooler.

The tensor-parallel ranks split a step's signatures as with vLLM (``engine_head``), and rank 0, whose
outputs SGLang returns, writes them. SGLang is imported only inside the worker-side helpers, so the
pooler is tested without it.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import List, Optional, Tuple

import numpy as np
import torch

from .engine_head import RisePoolerCore, build_head

MODEL_PACKAGE = "rise.runtime.sglang_models"
SPEC_DIR_ENV = "RISE_SGLANG_SPEC_DIR"
OUTPUT_FILE = "signatures.bin"
SIGNATURE, HIDDEN = "rise", "rise-hidden"


def signature_rid(key: str, loss_start: Optional[int], slot: int) -> str:
    return f"{SIGNATURE}:{key}:{'' if loss_start is None else int(loss_start)}:{int(slot)}:{uuid.uuid4().hex}"


def hidden_rid() -> str:
    return f"{HIDDEN}:{uuid.uuid4().hex}"


def request_kind(rid) -> Optional[str]:
    kind = rid.split(":", 1)[0] if isinstance(rid, str) else None
    return kind if kind in (SIGNATURE, HIDDEN) else None


def parse_signature_rid(rid: str) -> Tuple[str, Optional[int], int]:
    _, key, loss_start, slot, _ = rid.split(":", 4)
    return key, (int(loss_start) if loss_start else None), int(slot)


def output_rows(spec_dir: str, dim: int, mode: str = "r+") -> np.ndarray:
    """``signatures.bin`` as a [rows, dim] float32 array, mapped from the file (shared memory)."""
    path = os.path.join(spec_dir, OUTPUT_FILE)
    rows = os.path.getsize(path) // (4 * dim)
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(rows, dim)) if rows else np.zeros((0, dim), np.float32)


def _pooler_output(embeddings):
    from sglang.srt.layers.pooler import EmbeddingPoolerOutput

    return EmbeddingPoolerOutput(embeddings=embeddings)


def _tp_group():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


class RiseSGLangPooler(torch.nn.Module):
    """Stands in for a model's pooler: RISE requests get RISE's outputs, others the original pooler's."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._core: Optional[RisePoolerCore] = None
        self._key: Optional[str] = None
        self._out: Optional[np.ndarray] = None

    def forward(self, hidden_states: torch.Tensor, forward_batch):
        rids = list(forward_batch.rids or [])
        kinds = {request_kind(r) for r in rids}
        if not rids or kinds == {None}:
            return self.inner(hidden_states, forward_batch)
        if len(kinds) > 1:
            raise RuntimeError("an SGLang batch mixed RISE requests with other requests")
        # Fail closed: a prompt with a row missing or extra would pair every later hidden state
        # with the wrong target token. Radix cache and chunked prefill are off, so each request's
        # whole prompt is in this step.
        lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        prefix = [int(x) for x in (forward_batch.extend_prefix_lens_cpu or [0] * len(lens))]
        if len(lens) != len(rids) or any(prefix) or sum(lens) != hidden_states.shape[0]:
            raise RuntimeError(f"SGLang step does not hold whole prompts: {len(rids)} requests, extend lengths "
                               f"{lens[:5]}, prefix lengths {prefix[:5]}, {hidden_states.shape[0]} rows")
        parts = list(torch.split(hidden_states, lens))
        if kinds == {HIDDEN}:
            return _pooler_output([p.float().flatten() for p in parts])
        keys, loss_starts, slots = zip(*(parse_signature_rid(r) for r in rids))
        if len(set(keys)) != 1:
            raise RuntimeError("an SGLang batch mixed requests for different RISE heads")
        core = self._load(keys[0], hidden_states.device)
        token_ids = list(torch.split(forward_batch.input_ids.cpu(), lens))
        vecs = core.pooled(parts, token_ids, list(loss_starts))
        if core.rank == 0:
            out = self._output(core.head.dim, max(slots) + 1)
            out[list(slots)] = vecs.cpu().numpy()
        return _pooler_output(torch.tensor(slots, dtype=torch.float32, device=vecs.device)[:, None])

    def _output(self, dim: int, rows: int) -> np.ndarray:
        # the driver grows the file before a call; map it again when an earlier mapping is too small
        if self._out is None or self._out.shape[1] != dim or self._out.shape[0] < rows:
            self._out = output_rows(os.environ[SPEC_DIR_ENV], dim)
            if self._out.shape[0] < rows:
                raise RuntimeError(f"{OUTPUT_FILE} holds {self._out.shape[0]} signatures; slot {rows - 1} requested")
        return self._out

    def _load(self, key: str, device) -> RisePoolerCore:
        if key != self._key:
            with open(os.path.join(os.environ[SPEC_DIR_ENV], key + ".json")) as f:
                spec = json.load(f)
            self._core = None  # free the previous head before loading the next
            head, _ = build_head(spec, device)
            tp = _tp_group()
            self._core = RisePoolerCore(head, rank=tp.rank_in_group, world=tp.world_size,
                                        all_gather=(lambda t: tp.all_gather(t, dim=0)) if tp.world_size > 1 else None,
                                        batch_tokens=int(spec["head_batch_tokens"]))
            self._key = key
        return self._core


def unflatten_hidden(embeddings: List, lengths: List[int], hidden_dim: int) -> List[torch.Tensor]:
    """``rise-hidden`` outputs -> per-prompt [T, D] tensors."""
    out = []
    for e, n in zip(embeddings, lengths):
        t = torch.as_tensor(e, dtype=torch.float32)
        if t.numel() != n * hidden_dim:
            raise RuntimeError(f"SGLang returned {t.numel()} hidden values for a {n}-token prompt "
                               f"(hidden size {hidden_dim})")
        out.append(t.view(n, hidden_dim))
    return out
