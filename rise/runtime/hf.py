"""Hugging Face transformers trunk.

Runs only the decoder backbone (``model.base_model``), whose
``last_hidden_state`` is by definition the tensor the LM head consumes, so the
[tokens, V] logits are not computed twice. ``verify()`` checks that the head
reproduces the model's real logits, which catches final-norm, bias, tied
weights, soft-capping and logit-scale mismatches before any index is built.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .trunk import unembedding_probe


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    try:
        return {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16,
                "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}") from None


def _postprocess_for(config) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    cap = getattr(config, "final_logit_softcapping", None)
    scale = getattr(config, "logit_scale", None)
    if not cap and scale in (None, 1, 1.0):
        return None

    def post(z: torch.Tensor) -> torch.Tensor:
        if scale not in (None, 1, 1.0):
            z = z * float(scale)
        if cap:
            z = torch.tanh(z / float(cap)) * float(cap)
        return z

    return post


class HFTrunk:
    def __init__(self, model, *, name: Optional[str] = None, device=None, cudnn_attention: bool = False):
        self.model = model.eval()
        params = next(model.parameters())
        self.device = torch.device(device) if device is not None else params.device
        if self.device.type == "cuda" and not cudnn_attention and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            # cuDNN's SDPA backend builds an execution plan for every new sequence shape: 0.4-0.8 s
            # per shape on H200 (torch 2.11), ~90% of a 5k-document build. Flash / memory-efficient
            # attention has no per-shape cost and the same steady-state speed. Process-wide setting.
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.dtype = params.dtype
        self.name = name or getattr(model.config, "_name_or_path", "") or type(model).__name__
        head = model.get_output_embeddings()
        if head is None or getattr(head, "weight", None) is None:
            raise ValueError(f"{type(model).__name__} has no LM head (get_output_embeddings)")
        self._head = head
        self.vocab_size = int(head.weight.shape[0])
        self.hidden_dim = int(head.weight.shape[1])
        self.logits_postprocess = _postprocess_for(model.config)
        self._base = model.base_model
        if self._base is model:
            raise ValueError(f"{type(model).__name__} exposes no separate decoder backbone")

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, device: str = "auto", dtype: str = "auto",
                        device_map: Optional[str] = None, revision: Optional[str] = None,
                        trust_remote_code: bool = False, cudnn_attention: bool = False) -> "HFTrunk":
        import transformers
        from transformers import AutoModelForCausalLM

        dev = resolve_device(device)
        tdtype = resolve_dtype(dtype, dev)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        kwargs = {dtype_kw: tdtype, "revision": revision, "trust_remote_code": trust_remote_code}
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        if device_map is None:
            model.to(dev)
        return cls(model, name=name_or_path, cudnn_attention=cudnn_attention)

    @torch.inference_mode()
    def hidden_states(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.device, non_blocking=True)
        L = ids.shape[1]
        attn = (torch.arange(L, device=ids.device).unsqueeze(0)
                < lens.to(ids.device).unsqueeze(1)).long()
        out = self._base(input_ids=ids, attention_mask=attn, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def unembedding(self) -> torch.Tensor:
        return self._head.weight.detach()

    def unembedding_bias(self) -> Optional[torch.Tensor]:
        b = getattr(self._head, "bias", None)
        return None if b is None else b.detach()

    def input_embedding(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight.detach()

    def describe(self) -> dict:
        return {
            "backend": "hf",
            "name_or_path": self.name,
            "architecture": type(self.model).__name__,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "dtype": str(self.dtype).replace("torch.", ""),
            "unembedding_probe": unembedding_probe(self.unembedding()),
        }

    @torch.inference_mode()
    def verify(self, ids: Optional[torch.Tensor] = None, rtol: float = 2e-2) -> float:
        """Max |head logits - model logits| / max |model logits|; raises if above rtol."""
        if ids is None:
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(0, self.vocab_size, (2, 16), generator=g)
        lens = torch.full((ids.shape[0],), ids.shape[1], dtype=torch.long)
        h = self.hidden_states(ids, lens)
        W = self.unembedding()
        z = h.to(W.device, W.dtype) @ W.t()
        b = self.unembedding_bias()
        if b is not None:
            z = z + b
        z = z.float()
        if self.logits_postprocess is not None:
            z = self.logits_postprocess(z)
        ref = self.model(input_ids=ids.to(self.device), use_cache=False).logits.float().to(z.device)
        err = float((z - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        if not err <= rtol:
            raise RuntimeError(
                f"head logits do not reproduce {self.name}'s logits (rel err {err:.3g} > {rtol}); "
                "the backbone output is not the LM-head input or the head needs a postprocess")
        return err
