"""Build and query pipelines: JSONL -> tokens -> chunks -> trunk -> head -> index."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import RiseConfig
from .head import RiseHead, l2_normalize
from .index.store import PLAN_FILE, PROJECTIONS_FILE, IndexReader, IndexWriter
from .runtime.batching import Prefetcher, pad_batch, plan_batches
from .runtime.trunk import check_same_model
from .sketch import RiseProjections
from .text import TokenizerAdapter, explode, format_sample, query_chunks, query_prompt_text
from .utils.io import count_jsonl_rows, sha256_file

log = logging.getLogger("rise")


@dataclass
class RunOptions:
    max_batch_tokens: int = 16384   # trunk batch budget: rows x padded length (engines: head batch budget)
    max_batch_rows: int = 256
    max_head_tokens: int = 8192     # head micro-batch budget; bounds the [tokens, V] logits
    prefetch_depth: int = 2
    verify_trunk: bool = True       # check head logits == model logits before any work
    engine_batch_tokens: int = 131072  # tokens handed to an engine trunk per call (it batches internally)
    engine_head: bool = True        # run the head inside engines that support it (vLLM)


@dataclass
class BuildOptions(RunOptions):
    block_size: int = 4096          # rows per shard; also the resume / worker unit
    rank: int = 0
    world_size: int = 1
    dynamic: bool = False           # claim blocks at run time instead of the static rank split
    limit: Optional[int] = None
    text_preview_chars: int = 200
    meta_fields: Tuple[str, ...] = ("label",)
    hash_data: bool = True
    compress_bits: Optional[int] = None  # write SimHash codes of this many bits per row, not float16 vectors


def iter_jsonl_blocks(path: str, block_size: int, wanted=None,
                      limit: Optional[int] = None) -> Iterator[Tuple[int, int, List[dict]]]:
    """Yield (block, start_row, rows) for the wanted blocks, parsing only their lines.

    ``wanted`` is None (all), a set of block ids, or a predicate called once when a block's
    first line is reached (used to claim blocks lazily under dynamic scheduling).
    """
    take = (lambda b: True) if wanted is None else (wanted if callable(wanted) else wanted.__contains__)
    block, start, rows, row, active = -1, 0, [], 0, False
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if limit is not None and row >= limit:
                break
            b = row // block_size
            if b != block:
                if active:
                    yield block, start, rows
                block, start, rows = b, row, []
                active = bool(take(b))
            if active:
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({e.msg})") from None
            row += 1
    if active:
        yield block, start, rows


def _load_or_create_projections(out_dir: str, config: RiseConfig, vocab_size: int, hidden_dim: int) -> RiseProjections:
    """First writer wins (atomic create-if-absent), so concurrent workers share one set of tables."""
    path = os.path.join(out_dir, PROJECTIONS_FILE)
    fresh = RiseProjections.from_seed(config, vocab_size, hidden_dim)
    if not os.path.exists(path):
        tmp = f"{path}.tmp-{os.getpid()}"
        fresh.save(tmp)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp)
    proj = RiseProjections.load(path)
    proj.check_shapes(config, vocab_size, hidden_dim)
    # Once a build plan exists, the plan check covers consistency; before that, a table file that
    # disagrees with the config seed can only be a leftover from an aborted build.
    if not os.path.exists(os.path.join(out_dir, PLAN_FILE)) and proj.sha256() != fresh.sha256():
        raise ValueError(f"{path} does not match seed {config.seed} (left over from an aborted build?); "
                         "remove it or build into a new directory")
    return proj


def _install_engine_head(trunk, config: RiseConfig, projections: RiseProjections, projections_path: str,
                         opts: RunOptions) -> None:
    if opts.engine_head and getattr(trunk, "supports_engine_head", False):
        trunk.install_head(config, projections_path, projections.sha256(), max_head_tokens=opts.max_head_tokens,
                           head_batch_tokens=opts.max_batch_tokens)
        log.info("head runs inside the %s workers", trunk.backend)


def _engine_signatures(trunk, head: RiseHead, pad_id: int, chunks: Sequence[Sequence[int]],
                       loss_starts: Optional[Sequence[int]], opts: RunOptions):
    """Engine trunks: the engine prefills batch i+1 (main thread, where engines expect to be driven)
    while a worker thread runs the head on batch i. With the head installed in the engine's workers,
    the engine returns the signatures itself."""
    from concurrent.futures import ThreadPoolExecutor

    from .runtime.engine import pad_hidden

    lengths = [len(c) for c in chunks]
    engine_batches = plan_batches(lengths, max_batch_tokens=opts.engine_batch_tokens, max_batch_rows=1 << 30)
    if getattr(trunk, "head_dim", None) is not None:
        for bidx in engine_batches:
            vecs = trunk.encode_signatures([chunks[i] for i in bidx],
                                           None if loss_starts is None else [loss_starts[i] for i in bidx])
            yield bidx, vecs.to(head.device), sum(lengths[i] for i in bidx)
        return

    def head_job(bidx, hs):
        out = []
        for group in plan_batches([lengths[i] for i in bidx], max_batch_tokens=opts.max_batch_tokens,
                                  max_batch_rows=opts.max_batch_rows):
            idx = [bidx[j] for j in group]
            hidden, lens = pad_hidden([hs[j] for j in group], device=head.device, dtype=head.W.dtype)
            ids, _ = pad_batch([chunks[i] for i in idx], pad_id)
            ls = None if loss_starts is None else torch.tensor([loss_starts[i] for i in idx])
            out.append((idx, head.compute_from_hidden(hidden, ids, lens, ls), int(lens.sum())))
        return out

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for bidx in engine_batches:
            hs = trunk.encode([chunks[i] for i in bidx])
            if pending is not None:
                yield from pending.result()
            pending = pool.submit(head_job, bidx, hs)
        if pending is not None:
            yield from pending.result()


def _chunk_signatures(trunk, head: RiseHead, pad_id: int, chunks: Sequence[Sequence[int]],
                      loss_starts: Optional[Sequence[int]], opts: RunOptions):
    """Yield (chunk indices, [b, dim] signatures, real tokens) batch by batch."""
    if getattr(trunk, "is_engine", False):
        yield from _engine_signatures(trunk, head, pad_id, chunks, loss_starts, opts)
        return
    batches = plan_batches([len(c) for c in chunks], max_batch_tokens=opts.max_batch_tokens,
                           max_batch_rows=opts.max_batch_rows)
    pin = torch.device(trunk.device).type == "cuda"

    def produce():
        for bidx in batches:
            ids, lens = pad_batch([chunks[i] for i in bidx], pad_id, pin=pin)
            ls = None if loss_starts is None else torch.tensor([loss_starts[i] for i in bidx])
            yield bidx, ids, lens, ls

    for bidx, ids, lens, ls in Prefetcher(produce(), depth=opts.prefetch_depth):
        ids_d = ids.to(trunk.device, non_blocking=True)
        lens_d = lens.to(trunk.device, non_blocking=True)
        hidden = trunk.hidden_states(ids_d, lens_d)
        # host ids / lens: the head validates them on the CPU instead of syncing on the device
        yield bidx, head.compute_from_hidden(hidden, ids, lens, ls), int(lens.sum())


def _aggregate(trunk, head: RiseHead, pad_id: int, n_owners: int, chunks, owners, loss_starts,
               opts: RunOptions):
    """Chunk-mean per owner, L2-normalized: returns ([n_owners, dim], chunk counts, tokens)."""
    c = head.config
    acc = torch.zeros((n_owners, head.dim), dtype=torch.float32, device=head.device)
    cnt = torch.zeros((n_owners,), dtype=torch.float32, device=head.device)
    tokens = 0
    for bidx, vec, ntok in _chunk_signatures(trunk, head, pad_id, chunks, loss_starts, opts):
        ow = torch.as_tensor([owners[i] for i in bidx], device=head.device)
        # Chunk vectors are rounded to fp16 before averaging, as the research builder did.
        acc.index_add_(0, ow, vec.half().float())
        cnt.index_add_(0, ow, torch.ones(len(bidx), device=head.device))
        tokens += ntok
    out = acc / cnt.clamp(min=1.0).unsqueeze(1)
    if c.normalize_sample:
        out = l2_normalize(out, c.norm_eps())
    return out, cnt, tokens


def _maybe_verify(trunk, opts: RunOptions) -> None:
    if opts.verify_trunk and hasattr(trunk, "verify"):
        value = trunk.verify()
        log.info("trunk verified (%s = %.3g)", getattr(trunk, "verify_metric", "head vs model logits rel err"), value)


def build_index(trunk, tokenizer: TokenizerAdapter, config: RiseConfig, data_path: str, out_dir: str,
                opts: Optional[BuildOptions] = None) -> Optional[dict]:
    """Build (or resume) this worker's share of an index; finalize when every block exists.

    Returns the manifest once the index is complete, else None (other workers pending).
    """
    opts = opts or BuildOptions()
    config.validate()
    n_rows = count_jsonl_rows(data_path)
    if opts.limit is not None:
        n_rows = min(n_rows, opts.limit)
    os.makedirs(out_dir, exist_ok=True)
    projections = _load_or_create_projections(out_dir, config, trunk.vocab_size, trunk.hidden_dim)
    _maybe_verify(trunk, opts)
    head = RiseHead.from_trunk(trunk, config, projections=projections, max_tokens_per_step=opts.max_head_tokens)
    _install_engine_head(trunk, config, projections, os.path.join(out_dir, PROJECTIONS_FILE), opts)
    attrs = {
        "workload": "rise",
        "config": config.to_dict(),
        "model": trunk.describe(),
        "tokenizer": tokenizer.describe(),
        "projections_sha256": projections.sha256(),
        "data": {"file": os.path.basename(data_path), "rows": n_rows,
                 "sha256": sha256_file(data_path) if opts.hash_data else None},
    }
    writer = IndexWriter(out_dir, num_rows=n_rows, dim=head.dim, block_size=opts.block_size, attrs=attrs,
                         codec_bits=opts.compress_bits)
    if opts.dynamic:
        wanted = writer.claim
        log.info("index %s: %d rows, dim %d, %d blocks; %d pending, claimed dynamically by worker %d/%d",
                 out_dir, n_rows, head.dim, writer.num_blocks, len(writer.pending_blocks()),
                 opts.rank, opts.world_size)
    else:
        wanted = set(writer.pending_blocks(opts.rank, opts.world_size))
        log.info("index %s: %d rows, dim %d, %d blocks; worker %d/%d has %d to build",
                 out_dir, n_rows, head.dim, writer.num_blocks, opts.rank, opts.world_size, len(wanted))

    for block, start, rows in iter_jsonl_blocks(data_path, opts.block_size, wanted, n_rows):
        t0 = time.time()
        texts = [format_sample(r, config.sample_format) for r in rows]
        toks = tokenizer.encode_batch(texts)
        chunks, owners, meta = [], [], []
        for j, (r, text, tk) in enumerate(zip(rows, texts, toks)):
            cs = explode(tk, config)
            chunks.extend(cs)
            owners.extend([j] * len(cs))
            m = {"idx": start + j, "length": len(text), "tok_len": min(len(tk), config.seq_len),
                 "num_chunks": len(cs)}
            m.update({f: r[f] for f in opts.meta_fields if f in r})
            if opts.text_preview_chars:
                m["text"] = text[: opts.text_preview_chars]
            meta.append(m)
        vecs, cnt, ntok = _aggregate(trunk, head, tokenizer.pad_id, len(rows), chunks, owners, None, opts)
        dt = time.time() - t0
        empty = int((cnt == 0).sum())
        # encoded on the head's device from the float16 rows, exactly as `rise compress` encodes a float16 index
        rows_out = vecs.half() if writer.codec is None else writer.codec.encode(vecs.half().float())
        writer.write_block(block, rows_out.cpu().numpy(), meta,
                           {"chunks": len(chunks), "tokens": ntok, "empty_rows": empty, "wall_sec": round(dt, 3)})
        log.info("block %d/%d: %d rows, %d chunks, %d tokens, %.1fs (%.0f tok/s)%s",
                 block + 1, writer.num_blocks, len(rows), len(chunks), ntok, dt, ntok / max(dt, 1e-9),
                 f", {empty} empty rows" if empty else "")

    if writer.pending_blocks(0, 1):
        log.info("worker %d/%d finished; other blocks pending — run `rise finalize --index %s` when all are done",
                 opts.rank, opts.world_size, out_dir)
        return None
    return writer.finalize()


def finalize_index(out_dir: str) -> dict:
    from .utils.io import read_json

    plan = read_json(os.path.join(out_dir, PLAN_FILE))
    codec = plan.get("codec") or {}
    writer = IndexWriter(out_dir, num_rows=plan["num_rows"], dim=plan["dim"], block_size=plan["block_size"],
                         attrs=plan["attrs"], dtype=plan["dtype"], codec_bits=codec.get("bits"),
                         codec_seed=codec.get("seed", 0))
    return writer.finalize()


def build_query_vectors(trunk, tokenizer: TokenizerAdapter, index: IndexReader, examples: Sequence[dict],
                        opts: Optional[RunOptions] = None, *, allow_model_mismatch: bool = False) -> np.ndarray:
    """Signatures for queries in the index's sketch space: [Q, dim] float32.

    Rows with ``prompt_text`` are prompt-masked (only the continuation contributes);
    other rows are chunked exactly like index samples.
    """
    opts = opts or RunOptions()
    attrs = index.attrs
    config = RiseConfig.from_dict(attrs["config"])
    check_same_model(attrs.get("model", {}), trunk.describe(), allow_mismatch=allow_model_mismatch)
    projections = RiseProjections.load(index.projections_path())
    projections.check_shapes(config, trunk.vocab_size, trunk.hidden_dim)
    _maybe_verify(trunk, opts)
    head = RiseHead.from_trunk(trunk, config, projections=projections, max_tokens_per_step=opts.max_head_tokens)
    _install_engine_head(trunk, config, projections, index.projections_path(), opts)

    texts = [format_sample(e, config.sample_format) for e in examples]
    prompts = [query_prompt_text(e) for e in examples]
    toks = tokenizer.encode_batch(texts)
    with_prompt = [i for i, p in enumerate(prompts) if p is not None]
    prompt_toks = dict(zip(with_prompt, tokenizer.encode_batch([prompts[i] for i in with_prompt])))
    chunks, owners, loss_starts = [], [], []
    for qi, tk in enumerate(toks):
        for ids, ls in query_chunks(tk, prompt_toks.get(qi), config):
            chunks.append(ids)
            owners.append(qi)
            loss_starts.append(ls)
    vecs, cnt, _ = _aggregate(trunk, head, tokenizer.pad_id, len(examples), chunks, owners, loss_starts, opts)
    empty = int((cnt == 0).sum())
    if empty:
        log.warning("%d of %d queries produced no chunks (zero vectors)", empty, len(examples))
    return vecs.half().float().cpu().numpy()
