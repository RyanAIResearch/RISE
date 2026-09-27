"""Serving engines as trunks: vLLM and SGLang serve per-token final hidden states.

The engine runs the transformer (tensor-parallel, quantized, continuously
batched: whatever it supports) in its own processes; RISE keeps only the LM
head, loaded from the checkpoint, on ``head_device``. Engines are driven with
token ids only, so tokenization and chunking stay identical across backends.

Because engines do not expose logits in this mode, ``verify()`` checks the
other direction: the head's logits on the engine's hidden states must predict a
plain-English probe well (low perplexity). Pre-norm hidden states or head
weights from another model fail this by orders of magnitude.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple

import torch

from .trunk import unembedding_probe
from .weights import load_head_weights

_PROBE_TEXT = ("The capital of France is Paris. It is known for the Eiffel Tower, which was built for the "
               "1889 World's Fair, and for the Louvre, the most visited art museum in the world. The river "
               "Seine flows through the city, dividing it into the Left Bank and the Right Bank.")


def raise_process_limit() -> None:
    """Lift the soft per-user process/thread limit (RLIMIT_NPROC) to the hard limit.

    Clusters often set a low soft limit (1000 here) with an unlimited hard limit; a TP=8 engine's workers
    together exceed it during startup. Children inherit the raised limit."""
    if not sys.platform.startswith("linux"):  # macOS clamps both limits to kern.maxprocperuid instead
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or hard > soft):
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def warn_without_cuda_toolkit() -> None:
    """Engines JIT-compile kernels (DeepGEMM, FlashInfer, FP8 paths) during warm-up; without a toolkit the
    failure only surfaces after the model has loaded, often ten minutes in for 100B+ checkpoints."""
    import logging
    import shutil

    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not (home and os.path.exists(os.path.join(home, "bin", "nvcc"))) and shutil.which("nvcc") is None:
        logging.getLogger("rise").warning(
            "no CUDA toolkit found (CUDA_HOME unset and no nvcc on PATH); vLLM / SGLang may fail during "
            "warm-up when they JIT-compile kernels. Set CUDA_HOME to a toolkit matching the engine's torch build")


def bound_thread_pools(tensor_parallel_size: int) -> None:
    """Engines start a worker per TP rank plus helper processes, and each would size its OpenMP, MKL,
    tokenizer (rayon) and torch.compile worker pools to every core. Where threads per user are capped
    (1000 on the cluster this was developed on), a TP=8 engine then fails with "libgomp: Thread creation
    failed". Explicit settings in the environment win."""
    raise_process_limit()
    warn_without_cuda_toolkit()
    per = str(max(1, min(8, (os.cpu_count() or 8) // max(tensor_parallel_size, 1) // 4)))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(k, per)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def pad_hidden(hs: Sequence[torch.Tensor], *, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad per-sequence [T_i, D] hidden states into ([B, L, D], lens [B]) on ``device``."""
    lens = torch.tensor([int(h.shape[0]) for h in hs], dtype=torch.long)
    out = torch.zeros((len(hs), int(lens.max()), int(hs[0].shape[-1])), dtype=dtype, device=device)
    for i, h in enumerate(hs):
        out[i, : h.shape[0]] = h.to(device=device, dtype=dtype, non_blocking=True)
    return out, lens


class EngineTrunk:
    backend = "engine"
    is_engine = True
    verify_metric = "probe perplexity"

    def _load_head(self, model: str, head_device, need_input_embedding: bool, revision: Optional[str]) -> None:
        # Loaded after the engine starts, so the engine's memory profiling sees an idle device.
        hw = load_head_weights(model, device=head_device, need_input_embedding=need_input_embedding,
                               revision=revision)
        self._hw = hw
        self.name = model
        self.device = torch.device(head_device)
        self.dtype = hw.unembedding.dtype
        self.vocab_size, self.hidden_dim = (int(s) for s in hw.unembedding.shape)
        self.logits_postprocess = hw.logits_postprocess

    def encode(self, seqs: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        """Token-id sequences -> per-sequence [T_i, D] final hidden states (LM-head inputs).

        Fails closed unless every sequence gets exactly one row per token: a shifted row would pair
        each later position's hidden state with the wrong target token.
        """
        hs = self._encode(seqs)
        bad = [(i, int(h.shape[0]), len(s)) for i, (h, s) in enumerate(zip(hs, seqs)) if h.shape[0] != len(s)]
        if len(hs) != len(seqs) or bad:
            raise RuntimeError(f"{self.backend} returned misaligned hidden states (sequence, rows, tokens): "
                               f"{bad[:5]}; {len(hs)} outputs for {len(seqs)} sequences")
        return hs

    def _encode(self, seqs: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def hidden_states(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        lens_cpu = lens.detach().cpu()
        seqs = [ids[i, : int(lens_cpu[i])].tolist() for i in range(len(lens_cpu))]
        hidden, _ = pad_hidden(self.encode(seqs), device=self.device, dtype=self.dtype)
        if hidden.shape[1] < ids.shape[1]:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, ids.shape[1] - hidden.shape[1]))
        return hidden

    def unembedding(self) -> torch.Tensor:
        return self._hw.unembedding

    def unembedding_bias(self) -> Optional[torch.Tensor]:
        return self._hw.bias

    def input_embedding(self) -> torch.Tensor:
        if self._hw.input_embedding is None:
            raise RuntimeError("input embedding not loaded; construct the trunk with need_input_embedding=True")
        return self._hw.input_embedding

    def describe(self) -> dict:
        return {"backend": self.backend, "name_or_path": self.name, "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim, "dtype": str(self.dtype).replace("torch.", ""),
                "unembedding_probe": unembedding_probe(self.unembedding())}

    @torch.inference_mode()
    def verify(self, max_ppl: float = 200.0) -> float:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._hw.path)
        ids = tok(_PROBE_TEXT)["input_ids"]
        h = self.encode([ids])[0].to(self.device, self.dtype)
        if h.shape[0] != len(ids):
            raise RuntimeError(f"engine returned {h.shape[0]} hidden states for a {len(ids)}-token prompt")
        z = h[:-1] @ self.unembedding().t()
        if self.unembedding_bias() is not None:
            z = z + self.unembedding_bias()
        z = z.float()
        if self.logits_postprocess is not None:
            z = self.logits_postprocess(z)
        ppl = math.exp(float(torch.nn.functional.cross_entropy(z, torch.tensor(ids[1:], device=z.device))))
        if not ppl <= max_ppl:
            raise RuntimeError(f"head logits on {self.backend} hidden states give perplexity {ppl:.3g} on plain English "
                               f"(limit {max_ppl}): the engine is not returning LM-head inputs, or the head weights "
                               "do not belong to this model")
        return ppl

    def close(self) -> None:
        pass


class VLLMTrunk(EngineTrunk):
    """vLLM pooling runner with token-level ("ALL") pooling and no activation: raw final hidden states.

    After ``install_head`` the workers run the RISE head themselves and ``encode_signatures`` returns
    chunk signatures (see ``rise.runtime.vllm_head``); ``encode`` keeps returning hidden states."""

    backend = "vllm"
    supports_engine_head = True

    def __init__(self, model: str, *, tensor_parallel_size: int = 1, dtype: str = "auto",
                 gpu_memory_utilization: float = 0.8, max_model_len: Optional[int] = None,
                 head_device="cuda:0", enforce_eager: bool = False, revision: Optional[str] = None,
                 trust_remote_code: bool = False, need_input_embedding: bool = False, **llm_kwargs):
        bound_thread_pools(tensor_parallel_size)
        from vllm import LLM

        try:
            from vllm.config import PoolerConfig
        except ImportError:  # older layouts
            from vllm.config.pooler import PoolerConfig

        llm_kwargs.setdefault("worker_extension_cls", "rise.runtime.vllm_head.RiseWorkerExtension")
        self.llm = LLM(model=model, revision=revision, runner="pooling", convert="embed",
                       pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
                       tensor_parallel_size=tensor_parallel_size, dtype=dtype,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=enforce_eager, enable_prefix_caching=False,
                       trust_remote_code=trust_remote_code, **llm_kwargs)
        self._load_head(model, head_device, need_input_embedding, revision)
        self._revision = revision
        self.head_dim: Optional[int] = None  # set once the workers run the head
        self._head_spec: Optional[dict] = None
        self._head_sha: Optional[str] = None

    def _encode(self, seqs: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        outs = self.llm.encode([{"prompt_token_ids": [int(t) for t in s]} for s in seqs],
                               pooling_task="token_embed", use_tqdm=False)
        return [o.outputs.data for o in outs]

    def install_head(self, config, projections_path: str, projections_sha256: str, *,
                     max_head_tokens: int, head_batch_tokens: int) -> None:
        """Run the RISE head inside every worker; fails closed unless each rank's head matches this
        process's weights and the index's sketch tables."""
        spec = {"model": self._hw.path, "revision": self._revision, "config": config.to_dict(),
                "projections": os.path.abspath(projections_path), "max_head_tokens": int(max_head_tokens),
                "head_batch_tokens": int(head_batch_tokens)}
        if spec == self._head_spec and projections_sha256 == self._head_sha:
            return  # e.g. queries right after a build: the workers already hold this head
        probe = torch.tensor(unembedding_probe(self.unembedding()))
        dim = config.get_vector_dim()
        for fp in self.llm.collective_rpc("rise_install_head", args=(spec,)):
            if (fp["dim"] != dim or fp["projections_sha256"] != projections_sha256
                    or not torch.allclose(torch.tensor(fp["unembedding_probe"]), probe, rtol=1e-3, atol=1e-6)):
                raise RuntimeError(f"vLLM worker {fp['rank']} built a different head (dim {fp['dim']} vs {dim}, "
                                   f"tables {fp['projections_sha256'][:12]} vs {projections_sha256[:12]})")
        self.head_dim = dim
        self._head_spec, self._head_sha = spec, projections_sha256

    def encode_signatures(self, seqs: Sequence[Sequence[int]],
                          loss_starts: Optional[Sequence[Optional[int]]] = None) -> torch.Tensor:
        """Token-id chunks -> [n, dim] float32 signatures computed by the workers' heads."""
        from vllm import PoolingParams

        from .vllm_head import REQUEST_KEY

        if self.head_dim is None:
            raise RuntimeError("install_head() first")
        ls = [None] * len(seqs) if loss_starts is None else [None if x is None else int(x) for x in loss_starts]
        params = [PoolingParams(extra_kwargs={REQUEST_KEY: {"loss_start": x}}) for x in ls]
        outs = self.llm.encode([{"prompt_token_ids": [int(t) for t in s]} for s in seqs], pooling_params=params,
                               pooling_task="token_embed", use_tqdm=False)
        vecs = [o.outputs.data for o in outs]
        bad = [i for i, v in enumerate(vecs) if tuple(v.shape) != (self.head_dim,)]
        if len(vecs) != len(seqs) or bad:
            raise RuntimeError(f"vLLM returned {len(vecs)} signatures for {len(seqs)} chunks; "
                               f"wrong shape at {bad[:5]}")
        return torch.stack(vecs).float()


class SGLangTrunk(EngineTrunk):
    """SGLang in embedding mode, running RISE's model classes (see ``rise.runtime.sglang_head``).

    ``encode`` asks the workers for each prompt's final hidden states. After ``install_head`` the
    workers run the RISE head themselves and ``encode_signatures`` returns chunk signatures.

    Radix cache and chunked prefill stay off. On a prefix hit the forward only computes the suffix,
    and a prompt split across prefill chunks loses its earlier chunks' rows (seen: 149 rows for a
    153-token prompt under the default 8192-token chunking). RISE chunks are at most seq_len tokens,
    so every prompt fits in one prefill."""

    backend = "sglang"
    supports_engine_head = True

    def __init__(self, model: str, *, tensor_parallel_size: int = 1, dtype: str = "auto",
                 mem_fraction_static: float = 0.8, context_length: Optional[int] = None,
                 head_device="cuda:0", revision: Optional[str] = None, trust_remote_code: bool = False,
                 need_input_embedding: bool = False, **engine_kwargs):
        from .sglang_head import MODEL_PACKAGE, OUTPUT_FILE, SPEC_DIR_ENV

        bound_thread_pools(tensor_parallel_size)
        # The workers inherit both: RISE's model classes, and where install_head() leaves head specs.
        package = os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", MODEL_PACKAGE)
        if package != MODEL_PACKAGE:
            raise RuntimeError(f"SGLANG_EXTERNAL_MODEL_PACKAGE={package}; RISE needs {MODEL_PACKAGE}")
        # in shared memory where there is some: the workers write signatures to a file in it
        self._spec_dir = tempfile.mkdtemp(prefix="rise-sglang-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        open(os.path.join(self._spec_dir, OUTPUT_FILE), "wb").close()
        os.environ[SPEC_DIR_ENV] = self._spec_dir
        import sglang as sgl

        kwargs = dict(model_path=model, tp_size=tensor_parallel_size, dtype=dtype,
                      mem_fraction_static=mem_fraction_static, is_embedding=True, disable_radix_cache=True,
                      chunked_prefill_size=-1, skip_tokenizer_init=True, trust_remote_code=trust_remote_code,
                      log_level="error", revision=revision)
        if context_length:
            kwargs["context_length"] = context_length
        # SGLang reconfigures this process's root logger (basicConfig(force=True) at its own level),
        # which would silence RISE's progress logging; restore it afterwards.
        root = logging.getLogger()
        level, handlers = root.level, list(root.handlers)
        self.engine = sgl.Engine(**kwargs, **engine_kwargs)
        root.setLevel(level)
        root.handlers[:] = handlers
        self._load_head(model, head_device, need_input_embedding, revision)
        self._revision = revision
        self._tp = tensor_parallel_size
        self.head_dim: Optional[int] = None  # set once the workers run the head
        self._head_key: Optional[str] = None
        self._head_spec: Optional[dict] = None
        self._head_sha: Optional[str] = None

    def _embed(self, seqs: Sequence[Sequence[int]], rids: Sequence[str]) -> list:
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        # Engine.encode() takes text; with skip_tokenizer_init the engine takes token ids this way.
        obj = EmbeddingReqInput(input_ids=[[int(t) for t in s] for s in seqs], rid=list(rids))
        out = self.engine.loop.run_until_complete(self.engine.tokenizer_manager.generate_request(obj, None).__anext__())
        return [o["embedding"] for o in (out if isinstance(out, list) else [out])]

    def _encode(self, seqs: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        from .sglang_head import hidden_rid, unflatten_hidden

        return unflatten_hidden(self._embed(seqs, [hidden_rid() for _ in seqs]), [len(s) for s in seqs],
                                self.hidden_dim)

    def install_head(self, config, projections_path: str, projections_sha256: str, *,
                     max_head_tokens: int, head_batch_tokens: int) -> None:
        """Run the RISE head inside every worker; fails closed unless each rank's head reproduces this
        process's signatures on a probe."""
        spec = {"model": self._hw.path, "revision": self._revision, "config": config.to_dict(),
                "projections": os.path.abspath(projections_path), "max_head_tokens": int(max_head_tokens),
                "head_batch_tokens": int(head_batch_tokens)}
        if spec == self._head_spec and projections_sha256 == self._head_sha:
            return  # e.g. queries right after a build: the workers already hold this head
        text = json.dumps(spec, sort_keys=True)
        key = hashlib.sha256((text + projections_sha256).encode()).hexdigest()[:16]
        with open(os.path.join(self._spec_dir, key + ".json"), "w") as f:
            f.write(text)
        self._head_key, self.head_dim = key, config.get_vector_dim()
        self._check_head(config, projections_path, projections_sha256, max_head_tokens)
        self._head_spec, self._head_sha = spec, projections_sha256

    @torch.inference_mode()
    def _check_head(self, config, projections_path: str, projections_sha256: str, max_head_tokens: int) -> None:
        from transformers import AutoTokenizer

        from ..head import RiseHead
        from ..sketch import RiseProjections
        from .engine_head import RisePoolerCore

        projections = RiseProjections.load(projections_path)
        if projections.sha256() != projections_sha256:
            raise RuntimeError("the sketch tables on disk are not the index's")
        use_input = config.embedding_type == "input"
        head = RiseHead(config, self.unembedding(), projections=projections,
                        gh_embedding=self.input_embedding() if use_input else None, bias=self.unembedding_bias(),
                        logits_postprocess=self.logits_postprocess, device=self.device,
                        max_tokens_per_step=int(max_head_tokens))
        ids = AutoTokenizer.from_pretrained(self._hw.path)(_PROBE_TEXT)["input_ids"]
        seqs = [ids[: 24 + 4 * r] for r in range(self._tp)]  # one prompt per rank
        got = self.encode_signatures(seqs).to(self.device)
        hs = self.encode(seqs)  # the same prompts in one step again, so the same hidden states
        want = RisePoolerCore(head).signatures([h.to(self.device) for h in hs], [torch.tensor(s) for s in seqs],
                                               [None] * len(seqs))
        cos = torch.nn.functional.cosine_similarity(got, want, dim=1)
        if not bool((cos > 0.999).all()):
            raise RuntimeError(f"SGLang workers' heads disagree with this process's on a probe (cosine per rank "
                               f"{[round(float(c), 6) for c in cos]}): different weights or sketch tables")

    def encode_signatures(self, seqs: Sequence[Sequence[int]],
                          loss_starts: Optional[Sequence[Optional[int]]] = None) -> torch.Tensor:
        """Token-id chunks -> [n, dim] float32 signatures computed by the workers' heads."""
        import numpy as np

        from .sglang_head import OUTPUT_FILE, output_rows, signature_rid

        if self.head_dim is None:
            raise RuntimeError("install_head() first")
        n = len(seqs)
        ls = [None] * n if loss_starts is None else [None if x is None else int(x) for x in loss_starts]
        path = os.path.join(self._spec_dir, OUTPUT_FILE)
        if os.path.getsize(path) < n * self.head_dim * 4:
            os.truncate(path, n * self.head_dim * 4)
        outs = self._embed(seqs, [signature_rid(self._head_key, x, i) for i, x in enumerate(ls)])
        # each output is the row its signature went to; rank 0 wrote the rows before returning them
        slots = [int(o[0]) if len(o) == 1 else -1 for o in outs]
        if slots != list(range(n)):
            bad = [i for i, s in enumerate(slots) if s != i]
            raise RuntimeError(f"SGLang returned {len(outs)} outputs for {n} chunks; wrong slots at {bad[:5]}")
        return torch.from_numpy(np.array(output_rows(self._spec_dir, self.head_dim, mode="r")[:n]))

    def close(self) -> None:
        self.engine.shutdown()
        shutil.rmtree(self._spec_dir, ignore_errors=True)
