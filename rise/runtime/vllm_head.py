"""RISE's head inside vLLM's workers: the engine returns chunk signatures, not hidden states.

With the head in the driver, vLLM's pooling runner sends every prompt token's final hidden state, as
float32, through the engine core to the driver process: 64 KB per token for Llama-405B. Here each
worker replaces the model's pooler with one that runs the same ``RiseHead`` on the worker's own GPU.
Prompts whose prefill finished in a step are dealt round-robin over the tensor-parallel ranks, each
rank computes its share, and an all-gather gives rank 0 (whose output vLLM returns) every signature.
Only [dim] vectors leave the workers.

The driver installs the head with ``collective_rpc("rise_install_head", ...)``: ``RiseWorkerExtension``
is mixed into vLLM's worker class through ``worker_extension_cls``. RISE requests carry
``PoolingParams(extra_kwargs={"rise": {"loss_start": ...}})``; other requests, such as the probe
``verify()`` sends, still get plain hidden states from the original pooler.

vLLM is imported only inside the worker-side functions, so ``RisePoolerCore`` (plain PyTorch, shared
with SGLang through ``engine_head``) is tested without it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .engine_head import RisePoolerCore as _SharedCore
from .engine_head import build_head

REQUEST_KEY = "rise"


def is_rise_request(params) -> bool:
    extra = getattr(params, "extra_kwargs", None)
    return bool(extra) and REQUEST_KEY in extra


def _loss_start(params) -> Optional[int]:
    return params.extra_kwargs[REQUEST_KEY].get("loss_start")


class RisePoolerCore(_SharedCore):
    """The pooler's work on vLLM's pooling metadata: collect the prompts that finished prefill, then
    the shared core computes this rank's share of their signatures and gathers all of them."""

    def collect(self, hidden_states: torch.Tensor, metadata) -> List[Tuple[int, torch.Tensor]]:
        """(request index, [T, D] hidden states) of the prompts whose prefill finished in this step.

        Under chunked prefill a prompt spans several steps; its earlier rows wait in its pooling
        state, as vLLM's own ALL pooling keeps them."""
        cursor = metadata.get_pooling_cursor()
        parts = torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist())
        done = []
        for i, (state, part, finished) in enumerate(zip(metadata.pooling_states, parts,
                                                        cursor.get_finished_mask())):
            if not finished:
                state.hidden_states_cache.append(part.clone())  # the step's buffer is reused
                continue
            if state.hidden_states_cache:
                part = torch.cat(state.hidden_states_cache + [part])
                state.clean()
            done.append((i, part))
        return done

    def __call__(self, hidden_states: torch.Tensor, metadata) -> List[Optional[torch.Tensor]]:
        params = metadata.pooling_params
        out: List[Optional[torch.Tensor]] = [None] * len(params)
        done = self.collect(hidden_states, metadata)
        if not done:
            return out
        # Fail closed: a prompt with a row missing or extra would pair every later hidden state
        # with the wrong target token.
        prompt_lens = metadata.prompt_lens.tolist()
        bad = [(i, int(h.shape[0]), prompt_lens[i]) for i, h in done if h.shape[0] != prompt_lens[i]]
        if bad:
            raise RuntimeError(f"vLLM pooled misaligned hidden states (request, rows, tokens): {bad[:5]}")
        token_ids = metadata.get_prompt_token_ids_cpu()
        vecs = self.pooled([h for _, h in done], [token_ids[i] for i, _ in done],
                           [_loss_start(params[i]) for i, _ in done])
        for (i, _), v in zip(done, vecs):
            out[i] = v
        return out


def _vllm_pooler(core: RisePoolerCore, inner):
    from vllm.model_executor.layers.pooler.abstract import Pooler
    from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate

    class RisePooler(Pooler):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.core = core

        def get_supported_tasks(self):
            return self.inner.get_supported_tasks()

        def get_pooling_updates(self, task):
            # the head needs each prompt's token ids: targets, and the host-side vocabulary check
            return self.inner.get_pooling_updates(task) | PoolingParamsUpdate(requires_token_ids=True)

        def forward(self, hidden_states, pooling_metadata):
            rise = [is_rise_request(p) for p in pooling_metadata.pooling_params]
            if not any(rise):
                return self.inner(hidden_states, pooling_metadata)
            if not all(rise):
                raise RuntimeError("a vLLM step mixed RISE requests with plain pooling requests")
            return self.core(hidden_states, pooling_metadata)

    return RisePooler()


class RiseWorkerExtension:
    """Mixed into vLLM's worker class (``worker_extension_cls``) for ``collective_rpc``."""

    def rise_install_head(self, spec: dict) -> dict:
        """Build the head on this worker's GPU and wrap the model's pooler; returns what the driver
        checks against its own head."""
        from vllm.distributed import get_pp_group, get_tp_group

        from .trunk import unembedding_probe

        if get_pp_group().world_size != 1:
            raise RuntimeError("RISE's in-engine head needs pipeline_parallel_size=1")
        tp = get_tp_group()
        model = self.get_model()  # unwrapped from any CUDA-graph wrapper, which forwards attributes
        model.pooler = getattr(model.pooler, "inner", model.pooler)  # drop a previous head before loading
        head, projections_sha256 = build_head(spec, self.device)
        core = RisePoolerCore(head, rank=tp.rank_in_group, world=tp.world_size,
                              all_gather=(lambda t: tp.all_gather(t, dim=0)) if tp.world_size > 1 else None,
                              batch_tokens=int(spec["head_batch_tokens"]))
        model.pooler = _vllm_pooler(core, model.pooler)
        return {"rank": tp.rank_in_group, "dim": head.dim, "projections_sha256": projections_sha256,
                "unembedding_probe": unembedding_probe(head.W)}
