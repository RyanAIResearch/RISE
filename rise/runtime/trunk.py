"""The trunk/head boundary.

A trunk is any backend that turns right-padded token ids into the hidden
states the LM head consumes (post-final-norm). RISE owns everything after that
point, so HF transformers, SGLang, or a custom engine can serve the trunk
without changing the estimator.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch

log = logging.getLogger("rise")

_PROBE_COUNT = 32


@runtime_checkable
class Trunk(Protocol):
    device: torch.device
    hidden_dim: int
    vocab_size: int
    logits_postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]]

    def hidden_states(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """[B, L] right-padded ids + [B] lengths -> [B, L, D] post-final-norm hidden states."""

    def unembedding(self) -> torch.Tensor:
        """LM-head weight W [V, D]."""

    def unembedding_bias(self) -> Optional[torch.Tensor]:
        """LM-head bias [V], or None."""

    def input_embedding(self) -> torch.Tensor:
        """Input embedding matrix [V, D] (for embedding_type='input')."""

    def describe(self) -> dict:
        """JSON-serializable identity: backend, name, dims, dtype, unembedding probe."""


def unembedding_probe(W: torch.Tensor) -> List[float]:
    """A few fixed entries of W, used to tell models apart without hashing gigabytes.

    Compared with a tolerance, so the same checkpoint loaded in fp16/bf16/fp32 matches.
    """
    V, D = int(W.shape[0]), int(W.shape[1])
    rng = np.random.default_rng(0)
    rows = torch.from_numpy(rng.integers(0, V, _PROBE_COUNT)).to(W.device)
    cols = torch.from_numpy(rng.integers(0, D, _PROBE_COUNT)).to(W.device)
    return [float(x) for x in W[rows, cols].detach().float().cpu().tolist()]


def check_same_model(index_model: dict, trunk_model: dict, *, allow_mismatch: bool = False) -> None:
    """Refuse to mix an index with a different model: sketches would be incomparable."""
    problems = []
    for k in ("vocab_size", "hidden_dim"):
        if k in index_model and int(index_model[k]) != int(trunk_model.get(k, -1)):
            problems.append(f"{k}: index={index_model[k]} model={trunk_model.get(k)}")
    a, b = index_model.get("unembedding_probe"), trunk_model.get("unembedding_probe")
    if a is not None and b is not None:
        if not np.allclose(np.asarray(a), np.asarray(b), rtol=2e-2, atol=2e-3):
            problems.append("LM-head weights differ from the ones the index was built with")
    elif a is None:
        log.warning("index records no weight probe (imported index?); only dimensions were checked")
    if index_model.get("name_or_path") and index_model.get("name_or_path") != trunk_model.get("name_or_path"):
        log.warning("model name differs: index=%s model=%s",
                    index_model.get("name_or_path"), trunk_model.get("name_or_path"))
    if problems:
        msg = "model does not match the index: " + "; ".join(problems)
        if allow_mismatch:
            log.warning("%s (continuing: mismatch allowed)", msg)
        else:
            raise ValueError(msg + " (pass --allow-model-mismatch to override)")
