"""Sharded, resumable, memory-mapped signature index.

Layout of an index directory::

    build_plan.json          fixed at creation: rows, dim, block size, attrs (config, model, ...)
    projections.npz          CountSketch tables used for every row
    shards/vectors-00000.npy float16 [rows, dim], one shard per block of rows
    shards/meta-00000.jsonl  one metadata line per row
    shards/block-00000.json  completion record (rows, sha256, stats); written last
    manifest.json            written by finalize() once every block is complete

Blocks are independent, so a build can be split across workers (block b goes to
worker b % world_size), killed and resumed: completed blocks are skipped, and a
block only counts once its completion record exists. Readers memory-map the
shards, so an index never has to fit in RAM.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from ..utils.io import (iter_jsonl, read_json, save_npy_atomic, sha256_file, sha256_json,
                        write_json_atomic, write_jsonl_atomic)

FORMAT = "rise.signature-index"
FORMAT_VERSION = 1
PLAN_FILE = "build_plan.json"
MANIFEST_FILE = "manifest.json"
PROJECTIONS_FILE = "projections.npz"
SHARD_DIR = "shards"


def _shard_names(block: int) -> Tuple[str, str, str]:
    return (f"{SHARD_DIR}/vectors-{block:05d}.npy",
            f"{SHARD_DIR}/meta-{block:05d}.jsonl",
            f"{SHARD_DIR}/block-{block:05d}.json")


def _claim_path(root: str, block: int) -> str:
    return os.path.join(root, SHARD_DIR, f"block-{block:05d}.claim")


def clear_claims(root: str) -> int:
    """Remove all block claims. Only safe while no worker is running (e.g. before a launch)."""
    d = os.path.join(root, SHARD_DIR)
    if not os.path.isdir(d):
        return 0
    stale = [f for f in os.listdir(d) if f.endswith(".claim")]
    for f in stale:
        os.unlink(os.path.join(d, f))
    return len(stale)


class IndexWriter:
    def __init__(self, root: str, *, num_rows: int, dim: int, block_size: int, attrs: dict,
                 dtype: str = "float16"):
        if num_rows < 0 or dim < 1 or block_size < 1:
            raise ValueError("need num_rows >= 0, dim >= 1, block_size >= 1")
        self.root = root
        plan = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "num_rows": int(num_rows),
            "dim": int(dim),
            "dtype": dtype,
            "block_size": int(block_size),
            "num_blocks": int(math.ceil(num_rows / block_size)),
            "attrs": attrs,
            "attrs_sha256": sha256_json(attrs),
        }
        path = os.path.join(root, PLAN_FILE)
        if os.path.exists(path):
            old = read_json(path)
            diff = [k for k in plan if old.get(k) != plan[k]]
            if diff:
                raise ValueError(
                    f"{root} was started with different settings ({', '.join(diff)}); "
                    "resume with the original settings or build into a new directory")
        else:
            os.makedirs(os.path.join(root, SHARD_DIR), exist_ok=True)
            write_json_atomic(plan, path)
        self.plan = plan

    @property
    def num_blocks(self) -> int:
        return self.plan["num_blocks"]

    def block_range(self, block: int) -> Tuple[int, int]:
        bs = self.plan["block_size"]
        return block * bs, min((block + 1) * bs, self.plan["num_rows"])

    def is_done(self, block: int) -> bool:
        return os.path.exists(os.path.join(self.root, _shard_names(block)[2]))

    def claim(self, block: int) -> bool:
        """Atomically take an unfinished block for this worker (dynamic scheduling)."""
        if self.is_done(block):
            return False
        try:
            fd = os.open(_claim_path(self.root, block), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, f"{os.uname().nodename} {os.getpid()}".encode())
        os.close(fd)
        return True

    def pending_blocks(self, rank: int = 0, world_size: int = 1) -> List[int]:
        if not 0 <= rank < world_size:
            raise ValueError(f"need 0 <= rank < world_size, got {rank}/{world_size}")
        return [b for b in range(self.num_blocks) if b % world_size == rank and not self.is_done(b)]

    def write_block(self, block: int, vectors: np.ndarray, meta: Sequence[dict], stats: Optional[dict] = None) -> None:
        start, end = self.block_range(block)
        n = end - start
        if vectors.shape != (n, self.plan["dim"]):
            raise ValueError(f"block {block}: vectors {vectors.shape}, expected {(n, self.plan['dim'])}")
        if len(meta) != n:
            raise ValueError(f"block {block}: {len(meta)} metadata rows, expected {n}")
        vec_rel, meta_rel, done_rel = _shard_names(block)
        vec_path = os.path.join(self.root, vec_rel)
        save_npy_atomic(np.ascontiguousarray(vectors, dtype=np.dtype(self.plan["dtype"])), vec_path)
        write_jsonl_atomic(meta, os.path.join(self.root, meta_rel))
        write_json_atomic({"block": block, "start": start, "rows": n,
                           "sha256": sha256_file(vec_path), "stats": stats or {}},
                          os.path.join(self.root, done_rel))
        claim = _claim_path(self.root, block)
        if os.path.exists(claim):
            os.unlink(claim)

    def finalize(self, extra: Optional[dict] = None) -> dict:
        missing = [b for b in range(self.num_blocks) if not self.is_done(b)]
        if missing:
            raise RuntimeError(f"{len(missing)} of {self.num_blocks} blocks are not built yet "
                               f"(first: {missing[:8]}); run the remaining workers, then finalize")
        shards, totals = [], {}
        for b in range(self.num_blocks):
            vec_rel, meta_rel, done_rel = _shard_names(b)
            rec = read_json(os.path.join(self.root, done_rel))
            shards.append({"file": vec_rel, "meta": meta_rel, "start": rec["start"],
                           "rows": rec["rows"], "sha256": rec["sha256"]})
            for k, v in rec.get("stats", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    totals[k] = totals.get(k, 0) + v
        manifest = {k: v for k, v in self.plan.items()}
        manifest.update({
            "shards": shards,
            "build_totals": totals,
            "rise_version": __version__,
            "finalized_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
        if extra:
            manifest.update(extra)
        write_json_atomic(manifest, os.path.join(self.root, MANIFEST_FILE))
        return manifest


class IndexReader:
    def __init__(self, root: str):
        path = os.path.join(root, MANIFEST_FILE)
        if not os.path.exists(path):
            hint = " (build started but not finalized)" if os.path.exists(os.path.join(root, PLAN_FILE)) else ""
            raise FileNotFoundError(f"no {MANIFEST_FILE} in {root}{hint}")
        m = read_json(path)
        if m.get("format") != FORMAT:
            raise ValueError(f"{root} is not a RISE index (format={m.get('format')!r})")
        if int(m.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError(f"{root} uses index format v{m['format_version']}; this RISE reads <= v{FORMAT_VERSION}")
        self.root = root
        self.manifest = m
        self.num_rows = int(m["num_rows"])
        self.dim = int(m["dim"])
        self.shards = m["shards"]
        self._arrays: List[Optional[np.ndarray]] = [None] * len(self.shards)

    @property
    def attrs(self) -> dict:
        return self.manifest.get("attrs", {})

    def projections_path(self) -> str:
        return os.path.join(self.root, PROJECTIONS_FILE)

    def shard_array(self, i: int) -> np.ndarray:
        if self._arrays[i] is None:
            arr = np.load(os.path.join(self.root, self.shards[i]["file"]), mmap_mode="r")
            if arr.shape != (self.shards[i]["rows"], self.dim):
                raise ValueError(f"shard {i} has shape {arr.shape}, manifest says "
                                 f"{(self.shards[i]['rows'], self.dim)}")
            self._arrays[i] = arr
        return self._arrays[i]

    def iter_blocks(self, rows_per_step: int = 65536) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (global_start_row, [rows, dim] view) in row order, never crossing a shard."""
        for i, s in enumerate(self.shards):
            arr = self.shard_array(i)
            for off in range(0, s["rows"], rows_per_step):
                yield s["start"] + off, arr[off: off + rows_per_step]

    def get_rows(self, rows: Sequence[int]) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        starts = np.array([s["start"] for s in self.shards], dtype=np.int64)
        which = np.searchsorted(starts, rows, side="right") - 1
        for j, (r, w) in enumerate(zip(rows, which)):
            if not 0 <= r < self.num_rows:
                raise IndexError(f"row {r} out of range [0, {self.num_rows})")
            out[j] = self.shard_array(int(w))[r - starts[w]]
        return out

    def iter_metadata(self) -> Iterator[dict]:
        for s in self.shards:
            yield from iter_jsonl(os.path.join(self.root, s["meta"]))

    def metadata(self) -> List[dict]:
        return list(self.iter_metadata())

    def verify(self) -> None:
        """Re-hash every shard against the manifest."""
        for s in self.shards:
            got = sha256_file(os.path.join(self.root, s["file"]))
            if got != s["sha256"]:
                raise ValueError(f"{s['file']}: sha256 mismatch (index corrupted or modified)")
