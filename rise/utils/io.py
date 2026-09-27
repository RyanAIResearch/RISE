"""Small file helpers shared by every layer: JSONL, atomic writes, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Iterator

import numpy as np


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield one dict per non-blank line.

    Blank lines are skipped (matching the research code's row numbering), but a
    malformed line raises: silently dropping it would shift every later row and
    break the index-row <-> dataset-row alignment that attribution depends on.
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e.msg})") from None


def count_jsonl_rows(path: str) -> int:
    """Number of non-blank lines, i.e. the number of rows iter_jsonl yields."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _atomic_write(path: str, write_fn, mode: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json_atomic(obj: Any, path: str) -> None:
    _atomic_write(path, lambda f: json.dump(obj, f, indent=2, ensure_ascii=False), "w")


def write_jsonl_atomic(rows, path: str) -> None:
    def _w(f):
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _atomic_write(path, _w, "w")


def save_npy_atomic(arr: np.ndarray, path: str) -> None:
    _atomic_write(path, lambda f: np.save(f, arr, allow_pickle=False), "wb")


def write_bytes_atomic(data: bytes, path: str) -> None:
    _atomic_write(path, lambda f: f.write(data), "wb")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str, block_bytes: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
