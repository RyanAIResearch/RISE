"""Command line: rise {build,finalize,query,search,eval,select,compress,import-research,info}."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np

from . import __version__
from .config import RiseConfig
from .utils.io import iter_jsonl, read_json, write_json_atomic, write_jsonl_atomic

log = logging.getLogger("rise")


# ---------------------------------------------------------------- helpers
def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {s!r}")


def make_config(path: Optional[str], overrides: List[str]) -> RiseConfig:
    if path:
        from .compat import research_effective_config  # research configs carry "device"; ours never do

        d = read_json(path)
        cfg = research_effective_config(d) if "device" in d else RiseConfig.from_dict(d)
    else:
        cfg = RiseConfig()
    types = {f.name: f.type for f in dataclasses.fields(RiseConfig)}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config field {k!r}; fields: {', '.join(types)}")
        t = types[k] if isinstance(types[k], str) else types[k].__name__
        conv = {"bool": _parse_bool, "int": int, "float": float}.get(t, str)
        setattr(cfg, k, conv(v))
    cfg.validate()
    return cfg


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    p.add_argument("--dtype", default="auto", help="auto (bf16/fp16 on GPU, fp32 on CPU) | float16 | bfloat16 | float32")
    p.add_argument("--device-map", default=None, help="pass-through to from_pretrained (e.g. auto) for models that "
                                                      "do not fit on one device")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--cudnn-attention", action="store_true",
                   help="allow cuDNN SDPA (off by default: it compiles a plan for every new sequence shape)")
    g = p.add_argument_group("serving-engine trunks (--backend vllm | sglang)")
    g.add_argument("--backend", choices=("hf", "vllm", "sglang"), default="hf")
    g.add_argument("--tp", type=int, default=1, help="tensor-parallel size of the engine")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   help="engine's share of each GPU (vLLM gpu_memory_utilization / SGLang mem_fraction_static); "
                        "leave room for the head on --head-device")
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--head-device", default="cuda:0")
    g.add_argument("--engine-batch-tokens", type=int, default=131072, help="tokens per engine call")
    g.add_argument("--enforce-eager", action="store_true", help="vLLM: no CUDA graphs")
    g.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra keyword for vllm.LLM / sglang.Engine, e.g. allow_deprecated_quantization=true")
    g.add_argument("--driver-head", action="store_true",
                   help="vLLM: run the head in this process on --head-device instead of inside the engine's "
                        "workers (which returns signatures instead of hidden states)")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-batch-tokens", type=int, default=16384, help="trunk batch budget (rows x padded length)")
    p.add_argument("--max-batch-rows", type=int, default=256)
    p.add_argument("--max-head-tokens", type=int, default=8192, help="head micro-batch; bounds [tokens, V] logits")
    p.add_argument("--no-verify-trunk", action="store_true", help="skip the head-logits == model-logits check")


def _run_options(a, cls):
    return cls(max_batch_tokens=a.max_batch_tokens, max_batch_rows=a.max_batch_rows,
               max_head_tokens=a.max_head_tokens, verify_trunk=not a.no_verify_trunk,
               engine_batch_tokens=a.engine_batch_tokens, engine_head=not getattr(a, "driver_head", False))


def _engine_kwargs(items: List[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--engine-arg expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        low = v.strip().lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        else:
            for conv in (int, float):
                try:
                    out[k] = conv(v)
                    break
                except ValueError:
                    continue
            else:
                out[k] = v
    return out


def _load_model(a, need_input_embedding: bool = False):
    from .text import TokenizerAdapter

    common = dict(revision=a.revision, trust_remote_code=a.trust_remote_code)
    if a.backend == "hf":
        from .runtime.hf import HFTrunk

        trunk = HFTrunk.from_pretrained(a.model, device=a.device, dtype=a.dtype, device_map=a.device_map,
                                        cudnn_attention=a.cudnn_attention, **common)
    else:
        from .runtime.engine import SGLangTrunk, VLLMTrunk

        dtype = "auto" if a.dtype == "auto" else a.dtype
        if a.backend == "vllm":
            trunk = VLLMTrunk(a.model, tensor_parallel_size=a.tp, dtype=dtype, head_device=a.head_device,
                              gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                              enforce_eager=a.enforce_eager, need_input_embedding=need_input_embedding, **common,
                              **_engine_kwargs(a.engine_arg))
        else:
            trunk = SGLangTrunk(a.model, tensor_parallel_size=a.tp, dtype=dtype, head_device=a.head_device,
                                mem_fraction_static=a.gpu_memory_utilization, context_length=a.max_model_len,
                                need_input_embedding=need_input_embedding, **common,
                                **_engine_kwargs(a.engine_arg))
    tok = TokenizerAdapter.from_pretrained(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code)
    log.info("model %s on %s (%s): vocab %d, hidden %d", a.model, trunk.device, str(trunk.dtype).replace("torch.", ""),
             trunk.vocab_size, trunk.hidden_dim)
    return trunk, tok


def _resolve_device(name: str) -> str:
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _load_scores(path: str) -> np.ndarray:
    s = np.load(path, mmap_mode="r")
    if s.ndim not in (1, 2):
        raise SystemExit(f"{path}: expected [N] or [Q, N] scores, got shape {s.shape}")
    return s


# ---------------------------------------------------------------- commands
def _strip_option(argv: List[str], name: str) -> List[str]:
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
        elif tok == name:
            skip = True
        elif not tok.startswith(name + "="):
            out.append(tok)
    return out


def _launch_data_parallel(a) -> None:
    """One `rise build` worker per GPU (blocks b % n go to worker b); finalize when all succeed."""
    import subprocess

    from .pipeline import finalize_index

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    base = _strip_option(_strip_option(_strip_option(a.raw_argv, "--gpus"), "--rank"), "--world-size")
    base = [t for t in _strip_option(base, "--device") if t != "--dynamic"]
    # Every worker would otherwise size its OpenMP / tokenizer pools to all cores; n workers x
    # (OMP + rayon) threads exhausts container thread limits. Explicit user settings win.
    per = str(max(1, min(16, (os.cpu_count() or 8) // len(gpus))))
    pools = {k: per for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS") if k not in os.environ}
    from .index.store import clear_claims

    stale = clear_claims(a.out)  # no worker can be running yet, so any claim is left over from a crash
    if stale:
        log.info("cleared %d stale block claims", stale)
    procs = []
    for rank, gpu in enumerate(gpus):
        env = {**os.environ, **pools, "CUDA_VISIBLE_DEVICES": gpu}
        argv = [sys.executable, "-m", "rise", *base, "--device", "cuda", "--rank", str(rank),
                "--world-size", str(len(gpus)), "--dynamic"]
        procs.append(subprocess.Popen(argv, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"workers failed with exit codes {codes}; completed blocks are kept, rerun to resume")
    m = finalize_index(a.out)
    t = m.get("build_totals", {})
    log.info("index complete: %s (%d rows x %d, %d tokens) built by %d workers",
             a.out, m["num_rows"], m["dim"], t.get("tokens", 0), len(gpus))


def cmd_build(a) -> None:
    from .pipeline import BuildOptions, build_index

    if a.gpus:
        if a.queries:
            raise SystemExit("--queries is not supported with --gpus; run `rise query` after the build")
        return _launch_data_parallel(a)
    if a.queries and not a.query_out:
        a.query_out = os.path.join(a.out, "queries.npy")
    cfg = make_config(a.config, a.set)
    trunk, tok = _load_model(a, need_input_embedding=cfg.embedding_type == "input")
    opts = _run_options(a, BuildOptions)
    opts.block_size, opts.rank, opts.world_size, opts.limit = a.block_size, a.rank, a.world_size, a.limit
    opts.dynamic = a.dynamic
    opts.meta_fields = tuple(x for x in a.meta_fields.split(",") if x)
    opts.text_preview_chars = a.text_preview
    opts.hash_data = not a.no_data_hash
    opts.compress_bits = a.compress_bits
    try:
        manifest = build_index(trunk, tok, cfg, a.data, a.out, opts)
        if manifest is not None and a.queries:  # reuse the loaded model: engines take minutes to start
            from .index.store import IndexReader
            from .pipeline import RunOptions, build_query_vectors
            from .utils.io import save_npy_atomic

            vecs = build_query_vectors(trunk, tok, IndexReader(a.out), list(iter_jsonl(a.queries)),
                                       _run_options(a, RunOptions))
            save_npy_atomic(vecs.astype(np.float32), a.query_out)
            log.info("wrote %d query vectors to %s", vecs.shape[0], a.query_out)
    finally:
        getattr(trunk, "close", lambda: None)()
    if manifest is not None:
        t = manifest.get("build_totals", {})
        log.info("index complete: %s (%d rows x %d, %d tokens, %.0fs of block compute)",
                 a.out, manifest["num_rows"], manifest["dim"], t.get("tokens", 0), t.get("wall_sec", 0.0))


def cmd_finalize(a) -> None:
    from .pipeline import finalize_index

    m = finalize_index(a.index)
    log.info("finalized %s: %d rows in %d shards", a.index, m["num_rows"], len(m["shards"]))


def cmd_query(a) -> None:
    from .index.store import IndexReader
    from .pipeline import RunOptions, build_query_vectors
    from .utils.io import save_npy_atomic

    reader = IndexReader(a.index)
    trunk, tok = _load_model(a, need_input_embedding=reader.attrs.get("config", {}).get("embedding_type") == "input")
    examples = list(iter_jsonl(a.queries))
    try:
        vecs = build_query_vectors(trunk, tok, reader, examples, _run_options(a, RunOptions),
                                   allow_model_mismatch=a.allow_model_mismatch)
    finally:
        getattr(trunk, "close", lambda: None)()
    save_npy_atomic(vecs.astype(np.float32), a.out)
    log.info("wrote %d query vectors (dim %d) to %s", vecs.shape[0], vecs.shape[1], a.out)


def cmd_search(a) -> None:
    from .index.store import IndexReader
    from .search import mean_query, score_all, topk_search

    reader = IndexReader(a.index)
    q = np.load(a.queries).astype(np.float32)
    if q.ndim == 1:
        q = q[None]
    if q.shape[1] != reader.dim:
        raise SystemExit(f"query dim {q.shape[1]} != index dim {reader.dim}")
    device = _resolve_device(a.device)
    mean = a.aggregate == "mean"
    if mean:  # the aggregated query must not be re-normalized: scores stay = mean of per-query scores
        q = mean_query(q, a.metric)[None]
    common = dict(metric=a.metric, device=device, rows_per_step=a.rows_per_step, normalize_queries=not mean)
    scores, idx = topk_search(reader, q, a.k, **common)
    if a.out:
        hits = set(np.unique(idx).tolist())  # text previews of the hits, so results read without the corpus
        texts = {m["idx"]: m["text"] for m in reader.iter_metadata() if m.get("idx") in hits and "text" in m}
        write_jsonl_atomic(({"query": "mean" if a.aggregate == "mean" else i, "indices": idx[i].tolist(),
                             "scores": [round(float(x), 6) for x in scores[i]],
                             **({"texts": [texts.get(int(j)) for j in idx[i]]} if texts else {})}
                            for i in range(len(idx))), a.out)
        log.info("wrote top-%d for %d %s to %s", idx.shape[1], len(idx),
                 "aggregated query" if a.aggregate == "mean" else "queries", a.out)
    if a.scores_out:
        tmp = a.scores_out + ".partial.npy"
        shape = (reader.num_rows,) if a.aggregate == "mean" else (q.shape[0], reader.num_rows)
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(a.scores_dtype), shape=shape)
        score_all(reader, q, out=mm.reshape(q.shape[0], reader.num_rows), **common)
        mm.flush()
        del mm
        os.replace(tmp, a.scores_out)
        log.info("wrote full scores %s to %s", shape, a.scores_out)
    if not a.out and not a.scores_out:
        for i in range(min(len(idx), 5)):
            print(json.dumps({"query": i, "indices": idx[i][:10].tolist(), "scores": scores[i][:10].round(4).tolist()}))


def cmd_eval(a) -> None:
    from .index.store import IndexReader
    from .metrics import evaluate_scores, labels_from_metadata

    reader = IndexReader(a.index)
    source = iter_jsonl(a.label_data) if a.label_data else reader.iter_metadata()
    labels = labels_from_metadata(source, positive_label=a.positive_label, regex=a.label_regex,
                                  fields=tuple(a.label_fields.split(",")))
    if len(labels) != reader.num_rows:
        raise SystemExit(f"{len(labels)} labels for {reader.num_rows} index rows")
    scores = _load_scores(a.scores)
    res = evaluate_scores(scores, labels, [int(k) for k in a.k.split(",")])
    print(f"# {res['n_rows']} rows, {res['n_positive']} positive, {res['n_queries']} score rows")
    print("K\tauPRC\tauROC\tprecision")
    for k, m in res["top_k"].items():
        print(f"{k}\t{m['auprc']:.4f}\t{m['auroc']:.4f}\t{m['precision']:.4f}")
    if a.out:
        write_json_atomic(res, a.out)


def cmd_select(a) -> None:
    from .index.store import IndexReader
    from .metrics import mean_aggregate, rrf_aggregate
    from .utils.io import sha256_file

    reader = IndexReader(a.index)
    scores = _load_scores(a.scores)
    if scores.shape[-1] != reader.num_rows:
        raise SystemExit(f"scores cover {scores.shape[-1]} rows, index has {reader.num_rows}")
    recorded = reader.attrs.get("data", {}).get("sha256")
    if recorded and recorded != sha256_file(a.data):
        raise SystemExit(f"{a.data} is not the file this index was built from (sha256 mismatch)")
    agg = rrf_aggregate(scores, a.rrf_k) if a.mode == "rrf" else mean_aggregate(scores)
    order = np.argsort(-agg, kind="mergesort")
    chosen = order[: a.k] if a.k >= 0 else order[::-1][: -a.k]  # negative k: lowest-valued first
    rank = {int(i): r for r, i in enumerate(chosen)}
    picked = [None] * len(chosen)
    for row, ex in enumerate(iter_jsonl(a.data)):
        if row in rank:
            picked[rank[row]] = ex
    write_jsonl_atomic(picked, a.out)
    labels = {}
    for ex in picked:
        lab = str(ex.get("label", ""))
        labels[lab] = labels.get(lab, 0) + 1
    stats = {"selected": len(picked), "mode": a.mode, "k": a.k, "label_counts": labels}
    write_json_atomic(stats, a.out + ".stats.json")
    log.info("selected %d rows -> %s (%s)", len(picked), a.out, labels)


def cmd_import_research(a) -> None:
    from .compat import import_research_index

    m = import_research_index(a.src, a.out, block_size=a.block_size, model_name=a.model)
    log.info("imported %s -> %s: %d rows x %d", a.src, a.out, m["num_rows"], m["dim"])


def cmd_compress(a) -> None:
    from .index.store import IndexReader, compress_index

    t0 = time.time()
    m = compress_index(a.index, a.out, bits=a.bits, seed=a.seed, device=_resolve_device(a.device),
                       rows_per_step=a.rows_per_step)
    src = IndexReader(a.index)
    log.info("compressed %d rows to %d-bit SimHash codes in %.0fs: %.2f GiB -> %.2f GiB, written to %s",
             m["num_rows"], a.bits, time.time() - t0, src.num_rows * src.dim * 2 / 2**30,
             m["num_rows"] * a.bits / 8 / 2**30, a.out)


def cmd_info(a) -> None:
    from .index.store import IndexReader

    r = IndexReader(a.index)
    m = r.manifest
    attrs = r.attrs
    model = attrs.get("model", {})
    print(f"index      {a.index}  (format v{m.get('format_version')})")
    print(f"rows       {m['num_rows']}  dim {m['dim']}  dtype {m['dtype']}  shards {len(m['shards'])}")
    if r.codec is not None:
        print(f"codec      SimHash, {r.codec.bits} bits per row (queries stay float; scores estimate the inner product)")
    size = m['num_rows'] * r.row_width * np.dtype(m['dtype']).itemsize
    print(f"size       {size / 2**30:.2f} GiB" if size >= 2**30 else f"size       {size / 2**20:.1f} MiB")
    print(f"model      {model.get('name_or_path', '?')} ({model.get('backend', '?')}, vocab {model.get('vocab_size')}, "
          f"hidden {model.get('hidden_dim')}, {model.get('dtype', '?')})")
    print(f"config     {json.dumps(attrs.get('config', {}), sort_keys=True)}")
    print(f"build      {json.dumps(m.get('build_totals', {}))}")
    if a.verify:
        r.verify()
        print("verify     all shard hashes match")


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rise", description="RISE: scalable, forward-only data attribution & valuation "
                                                         "for LLMs via readout (LM-head) influence sketching")
    p.add_argument("--version", action="version", version=f"rise {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build (or resume) a signature index from a JSONL corpus")
    _add_model_args(b)
    b.add_argument("--data", required=True, help="JSONL corpus (text | prompt+generation | instruction/input/output)")
    b.add_argument("--out", required=True, help="index directory")
    b.add_argument("--config", default=None, help="RISE config.json (research-code configs load as-is)")
    b.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    b.add_argument("--block-size", type=int, default=4096, help="rows per shard / resume unit")
    b.add_argument("--rank", type=int, default=0, help="this worker (blocks b with b %% world_size == rank)")
    b.add_argument("--world-size", type=int, default=1)
    b.add_argument("--gpus", default=None, help="data parallel on one node: e.g. 0,1,2,3 starts one worker per GPU")
    b.add_argument("--dynamic", action="store_true", help=argparse.SUPPRESS)  # set by the --gpus launcher
    b.add_argument("--limit", type=int, default=None, help="only the first N rows")
    b.add_argument("--meta-fields", default="label", help="comma-separated fields copied into metadata")
    b.add_argument("--text-preview", type=int, default=200, help="chars of text kept in metadata (0 = none)")
    b.add_argument("--no-data-hash", action="store_true", help="skip hashing the corpus into the manifest")
    b.add_argument("--compress-bits", type=int, default=None,
                   help="write SimHash codes of this many bits per row instead of float16 vectors (8192 = 1 KiB)")
    b.add_argument("--queries", default=None, help="also embed these queries with the loaded model once the index is done")
    b.add_argument("--query-out", default=None, help=".npy for --queries (default: <out>/queries.npy)")
    _add_run_args(b)
    b.set_defaults(fn=cmd_build)

    f = sub.add_parser("finalize", help="write the manifest once all workers' blocks exist")
    f.add_argument("--index", required=True)
    f.set_defaults(fn=cmd_finalize)

    q = sub.add_parser("query", help="compute query signatures in an index's sketch space")
    _add_model_args(q)
    q.add_argument("--index", required=True)
    q.add_argument("--queries", required=True, help="JSONL; rows with prompt_text are prompt-masked")
    q.add_argument("--out", required=True, help=".npy [Q, dim] float32")
    q.add_argument("--allow-model-mismatch", action="store_true")
    _add_run_args(q)
    q.set_defaults(fn=cmd_query)

    s = sub.add_parser("search", help="top-k influential rows and/or full scores")
    s.add_argument("--index", required=True)
    s.add_argument("--queries", required=True, help=".npy query signatures from `rise query`")
    s.add_argument("--k", type=int, default=100)
    s.add_argument("--metric", choices=("dot", "cosine"), default="cosine")
    s.add_argument("--aggregate", choices=("none", "mean"), default="none",
                   help="mean: score rows against the mean query (valuation / Alg. 1 stage 3)")
    s.add_argument("--out", default=None, help="top-k results JSONL")
    s.add_argument("--scores-out", default=None, help="full scores .npy: [Q, N], or [N] with --aggregate mean")
    s.add_argument("--scores-dtype", choices=("float32", "float16"), default="float32")
    s.add_argument("--device", default="auto")
    s.add_argument("--rows-per-step", type=int, default=65536)
    s.set_defaults(fn=cmd_search)

    e = sub.add_parser("eval", help="auPRC / auROC / precision@K of scores against metadata labels")
    e.add_argument("--index", required=True)
    e.add_argument("--scores", required=True)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--positive-label", default=None, help="metadata label value counted as positive")
    g.add_argument("--label-regex", default=None, help="regex over --label-fields marking positives")
    e.add_argument("--label-fields", default="text")
    e.add_argument("--label-data", default=None,
                   help="derive labels from this corpus (full rows) instead of the index metadata")
    e.add_argument("--k", default="10,50,100")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    sl = sub.add_parser("select", help="write the top (or bottom) valued rows of the corpus")
    sl.add_argument("--index", required=True)
    sl.add_argument("--scores", required=True)
    sl.add_argument("--data", required=True, help="the corpus the index was built from")
    sl.add_argument("--k", type=int, required=True, help="rows to keep; negative = lowest-valued rows")
    sl.add_argument("--mode", choices=("mean", "rrf"), default="mean")
    sl.add_argument("--rrf-k", type=int, default=60)
    sl.add_argument("--out", required=True)
    sl.set_defaults(fn=cmd_select)

    c = sub.add_parser("compress", help="store an index as SimHash sign bits (8192 bits = 1 KiB per row)")
    c.add_argument("--index", required=True, help="a float16 index")
    c.add_argument("--out", required=True, help="new directory for the compressed index")
    c.add_argument("--bits", type=int, default=8192, help="bits per row, a multiple of 8")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="auto")
    c.add_argument("--rows-per-step", type=int, default=4096)
    c.set_defaults(fn=cmd_compress)

    im = sub.add_parser("import-research", help="convert an index built by the research code")
    im.add_argument("--src", required=True, help="directory with config.json / index.pt / metadata.jsonl / projections.pt")
    im.add_argument("--out", required=True)
    im.add_argument("--block-size", type=int, default=4096)
    im.add_argument("--model", default=None, help="model name to record")
    im.set_defaults(fn=cmd_import_research)

    i = sub.add_parser("info", help="summarize an index")
    i.add_argument("--index", required=True)
    i.add_argument("--verify", action="store_true", help="re-hash all shards")
    i.set_defaults(fn=cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(raw)
    from .runtime.engine import raise_process_limit

    raise_process_limit()  # multi-GPU workers and engine TP ranks inherit it
    a.raw_argv = raw
    tag = f"[rank {a.rank}/{a.world_size}] " if getattr(a, "world_size", 1) > 1 else ""
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format=f"%(asctime)s %(levelname)s {tag}%(message)s", stream=sys.stderr, force=True)
    a.fn(a)


if __name__ == "__main__":
    main()
