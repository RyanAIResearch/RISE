"""Import indexes produced by the RISE research code (v3-v6).

Those builders write ``config.json``, ``index.pt`` (float16 [N, dim]),
``metadata.jsonl`` and ``projections.pt`` ({"proj_h" | "proj_r" | "proj_g":
(buckets, signs)}). Importing converts them to the sharded format without
recomputing anything; the sketch tables are taken from ``projections.pt`` so
new queries land in exactly the same sketch space.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from .config import RiseConfig
from .index.store import PROJECTIONS_FILE, IndexWriter
from .sketch import CountSketch, RiseProjections
from .utils.io import iter_jsonl, read_json, sha256_file

log = logging.getLogger("rise")


def degenerate_fp16_tau(config: RiseConfig) -> float:
    """The temperature the research code's bisection returns when every entropy is NaN.

    On GPU the research code evaluated entropy on fp16 logits as -sum p log(clamp(p, 1e-8));
    1e-8 rounds to 0 in fp16, so any underflowed probability yields 0 * log 0 = NaN, every
    comparison fails, and the bracket shrinks onto tau_min for all tau_search_steps steps.
    """
    return config.tau_min + (config.tau_max - config.tau_min) / 2 ** (config.tau_search_steps + 1)


def research_effective_config(d: dict) -> RiseConfig:
    """Load a research ``config.json`` with the temperature it effectively ran with.

    Research configs record ``device``. A GPU run with adaptive temperature never searched
    (see ``degenerate_fp16_tau``), so it becomes a fixed temperature here; CPU runs, which
    searched in fp32, keep the adaptive search.
    """
    config = RiseConfig.from_dict(d)
    if config.adaptive_temperature and str(d.get("device", "")).startswith("cuda"):
        tau = degenerate_fp16_tau(config)
        log.warning("research config ran adaptive temperature on GPU, where its fp16 entropy was NaN and "
                    "every chunk used tau=%.7f; using that fixed temperature", tau)
        config.adaptive_temperature = False
        config.tau_fallback = tau
    elif config.adaptive_temperature and "device" not in d:
        log.warning("research config does not record its device; if it ran on GPU, its effective "
                    "temperature was fixed at %.7f, not the adaptive search", degenerate_fp16_tau(config))
    return config


def import_research_index(src_dir: str, out_dir: str, *, block_size: int = 4096,
                     model_name: Optional[str] = None) -> dict:
    config = research_effective_config(read_json(os.path.join(src_dir, "config.json")))
    index_path = os.path.join(src_dir, "index.pt")
    try:
        index = torch.load(index_path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
        index = torch.load(index_path, map_location="cpu", weights_only=True)
    if not isinstance(index, torch.Tensor) or index.dim() != 2:
        raise ValueError(f"{index_path}: expected a 2-D tensor, got {type(index).__name__}")
    meta = list(iter_jsonl(os.path.join(src_dir, "metadata.jsonl")))
    raw = torch.load(os.path.join(src_dir, "projections.pt"), map_location="cpu", weights_only=True)

    widths = {"proj_h": config.Kh, "proj_r": config.get_effective_Kr(), "proj_g": config.Kg}
    tables = {}
    for key, width in widths.items():
        if key not in raw:
            raise ValueError(f"projections.pt has no {key}")
        buckets, signs = raw[key]
        tables[key[-1]] = CountSketch(buckets.long(), signs.float(), width)
    proj = RiseProjections(h=tables["h"], r=tables["r"], g=tables["g"])
    vocab, hidden = proj.r.in_dim, proj.h.in_dim
    proj.check_shapes(config, vocab, hidden)

    n, dim = int(index.shape[0]), int(index.shape[1])
    if dim != config.get_vector_dim():
        raise ValueError(f"index dim {dim} != config vector dim {config.get_vector_dim()}")
    if len(meta) != n:
        raise ValueError(f"metadata has {len(meta)} rows, index has {n}")
    regenerated = RiseProjections.from_seed(config, vocab, hidden)
    if regenerated.sha256() != proj.sha256():
        log.warning("imported sketch tables differ from seed %d regeneration; using the imported ones", config.seed)

    os.makedirs(out_dir, exist_ok=True)
    proj.save(os.path.join(out_dir, PROJECTIONS_FILE))
    attrs = {
        "workload": "rise",
        "config": config.to_dict(),
        "model": {"backend": "imported", "name_or_path": model_name or "", "vocab_size": vocab,
                  "hidden_dim": hidden},
        "projections_sha256": proj.sha256(),
        "imported_from": {"format": "research_code", "index_sha256": sha256_file(index_path)},
        "data": {"rows": n},
    }
    writer = IndexWriter(out_dir, num_rows=n, dim=dim, block_size=block_size, attrs=attrs)
    for b in writer.pending_blocks():
        s, e = writer.block_range(b)
        writer.write_block(b, index[s:e].to(torch.float16).numpy(), meta[s:e], {"imported": e - s})
    return writer.finalize()
