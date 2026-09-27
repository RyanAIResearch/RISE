"""Load only the LM head (and optionally the input embedding) from a checkpoint.

Engine trunks (vLLM, SGLang) serve hidden states from their own processes, so
RISE never instantiates the model: the few tensors the head needs are read
straight from the safetensors shards. A 405B checkpoint stays on disk except
its ~4 GB unembedding.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

_HEAD_NAMES = ("lm_head.weight", "embed_out.weight", "output.weight", "model.lm_head.weight")
_EMBED_NAMES = ("model.embed_tokens.weight", "gpt_neox.embed_in.weight", "transformer.wte.weight",
                "model.embed_in.weight", "embed_tokens.weight")
_BIAS_NAMES = ("lm_head.bias", "embed_out.bias")


def resolve_checkpoint(name_or_path: str, revision: Optional[str] = None) -> str:
    """Local directory for a checkpoint (a path, or a Hugging Face repo id resolved via the cache)."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


def logits_postprocess_for(config: dict) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Final soft-capping (Gemma-2 style) and logit scaling (Cohere style), if the config has them."""
    cap = config.get("final_logit_softcapping")
    scale = config.get("logit_scale")
    if not cap and scale in (None, 1, 1.0):
        return None

    def post(z: torch.Tensor) -> torch.Tensor:
        if scale not in (None, 1, 1.0):
            z = z * float(scale)
        if cap:
            z = torch.tanh(z / float(cap)) * float(cap)
        return z

    return post


@dataclass
class HeadWeights:
    unembedding: torch.Tensor
    bias: Optional[torch.Tensor]
    input_embedding: Optional[torch.Tensor]
    config: dict
    path: str

    @property
    def logits_postprocess(self):
        return logits_postprocess_for(self.config)


def load_head_weights(name_or_path: str, *, device="cpu", dtype: Optional[torch.dtype] = None,
                      need_input_embedding: bool = False, revision: Optional[str] = None) -> HeadWeights:
    from safetensors import safe_open

    path = resolve_checkpoint(name_or_path, revision)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    cfg = {**cfg.get("text_config", {}), **cfg}  # multimodal wrappers keep the LM config nested
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = os.path.join(path, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"no safetensors weights in {path}")
        with safe_open(single, framework="pt") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}

    def grab(names) -> Optional[torch.Tensor]:
        for n in names:
            if n in weight_map:
                with safe_open(os.path.join(path, weight_map[n]), framework="pt", device="cpu") as sf:
                    t = sf.get_tensor(n)
                if not t.is_floating_point() or t.element_size() < 2:
                    raise ValueError(f"{n} is {t.dtype}; the head needs a float16/bfloat16/float32 unembedding "
                                     "(quantized checkpoints normally keep the LM head unquantized)")
                return t.to(device=device, dtype=dtype or t.dtype)
        return None

    tied = bool(cfg.get("tie_word_embeddings", False))
    W = None if tied else grab(_HEAD_NAMES)
    emb = None
    if W is None:  # tied, or tied without the flag
        emb = grab(_EMBED_NAMES)
        if emb is None:
            raise KeyError(f"no LM head or input embedding found in {path}")
        W = emb
    if need_input_embedding and emb is None:
        emb = grab(_EMBED_NAMES)
        if emb is None:
            raise KeyError(f"no input embedding found in {path}")
    return HeadWeights(unembedding=W, bias=grab(_BIAS_NAMES), input_embedding=emb if need_input_embedding else None,
                       config=cfg, path=path)
