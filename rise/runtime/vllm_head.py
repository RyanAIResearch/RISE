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

vLLM is imported only inside the worker-side functions, so ``RisePoolerCore`` (plain PyTorch) is
tested without it.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .batching import plan_batches
from .engine import pad_hidden

REQUEST_KEY = "rise"


def is_rise_request(params) -> bool:
    extra = getattr(params, "extra_kwargs", None)
    return bool(extra) and REQUEST_KEY in extra


def _loss_start(params) -> Optional[int]:
    return params.extra_kwargs[REQUEST_KEY].get("loss_start")


class RisePoolerCore:
    """The pooler's work, independent of vLLM's classes: collect the prompts that finished prefill,
    compute this rank's share of their signatures, and gather all of them."""

    def __init__(self, head, *, rank: int = 0, world: int = 1,
                 all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 batch_tokens: int = 16384):
        if world > 1 and all_gather is None:
            raise ValueError("more than one rank needs an all_gather")
        self.head = head
        self.rank, self.world = rank, world
        self.all_gather = all_gather
        self.batch_tokens = batch_tokens

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
        mine = done[self.rank::self.world]
        vecs = self.signatures([h for _, h in mine], [token_ids[i] for i, _ in mine],
                               [_loss_start(params[i]) for i, _ in mine])
        if self.world == 1:
            for (i, _), v in zip(done, vecs):
                out[i] = v
            return out
        per = math.ceil(len(done) / self.world)  # every rank sends the same shape
        buf = vecs.new_zeros((per, vecs.shape[1]))
        buf[: len(mine)] = vecs
        gathered = self.all_gather(buf)  # [world * per, dim], rank-major
        for r in range(self.world):
            for j, (i, _) in enumerate(done[r::self.world]):
                out[i] = gathered[r * per + j]
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


def _build_head(spec: dict, device):
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
        head, projections_sha256 = _build_head(spec, self.device)
        core = RisePoolerCore(head, rank=tp.rank_in_group, world=tp.world_size,
                              all_gather=(lambda t: tp.all_gather(t, dim=0)) if tp.world_size > 1 else None,
                              batch_tokens=int(spec["head_batch_tokens"]))
        model.pooler = _vllm_pooler(core, model.pooler)
        return {"rank": tp.rank_in_group, "dim": head.dim, "projections_sha256": projections_sha256,
                "unembedding_probe": unembedding_probe(head.W)}
