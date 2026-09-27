"""CountSketch projections for RISE.

Tables are generated exactly like ``CountSketchBatched`` in the RISE research
code (CPU ``torch.Generator`` seeded per table: buckets first, then sign bits),
so indexes built there and here share the same sketch space. Because PyTorch
does not promise RNG streams across releases, the tables are also persisted
with every index (``projections.npz``) and loaded from there at query time.

Dense inputs (hidden states, unembedding rows) are sketched as a GEMM against
the materialized ±1 matrix — a [D, K] matrix is tiny and the GEMM runs on
tensor cores — while vocabulary-space residuals use the sparse path, whose
cost scales with the residual support rather than the vocabulary size.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .utils.io import write_bytes_atomic
from .config import RiseConfig

# Materialize the dense [in_dim, out_dim] sketch matrix only below this size.
_DENSE_MATRIX_MAX_ELEMS = 64 * 1024 * 1024


class CountSketch:
    """Data-independent linear map R^in_dim -> R^out_dim: x -> sum_i s(i) x_i e_{b(i)}."""

    def __init__(self, buckets: torch.Tensor, signs: torch.Tensor, out_dim: int):
        if buckets.shape != signs.shape or buckets.dim() != 1:
            raise ValueError("buckets and signs must be 1-D tensors of equal length")
        self.in_dim = int(buckets.numel())
        self.out_dim = int(out_dim)
        self.buckets = buckets.long()
        self.signs = signs.float()
        self._dense: Optional[torch.Tensor] = None

    @classmethod
    def from_seed(cls, in_dim: int, out_dim: int, seed: int, rng: str = "cpu") -> "CountSketch":
        if rng == "cuda":
            # The December 2025 research scripts drew their tables with a CUDA generator: same calls,
            # a different stream. The tables are stored with every index, so only builds need a GPU.
            if not torch.cuda.is_available():
                raise RuntimeError("sketch_rng='cuda' draws the CountSketch tables on a GPU; none is available")
            g = torch.Generator(device="cuda")
            g.manual_seed(int(seed))
            buckets = torch.randint(0, out_dim, (in_dim,), generator=g, device="cuda").cpu()
            sign_bits = torch.randint(0, 2, (in_dim,), generator=g, device="cuda").cpu()
        elif rng == "cpu":
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            buckets = torch.randint(0, out_dim, (in_dim,), generator=g, dtype=torch.long)
            sign_bits = torch.randint(0, 2, (in_dim,), generator=g, dtype=torch.long)
        else:
            raise ValueError(f"unknown sketch rng {rng!r}")
        signs = torch.where(sign_bits.bool(), torch.ones(in_dim), -torch.ones(in_dim)).to(torch.float32)
        return cls(buckets, signs, out_dim)

    @property
    def device(self) -> torch.device:
        return self.buckets.device

    def to(self, device) -> "CountSketch":
        out = CountSketch(self.buckets.to(device), self.signs.to(device), self.out_dim)
        return out

    def dense_matrix(self) -> torch.Tensor:
        """The explicit [in_dim, out_dim] matrix (cached)."""
        if self._dense is None:
            m = torch.zeros((self.in_dim, self.out_dim), dtype=torch.float32, device=self.device)
            m[torch.arange(self.in_dim, device=self.device), self.buckets] = self.signs
            self._dense = m
        return self._dense

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        """[N, in_dim] -> [N, out_dim] float32."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {tuple(x.shape)}")
        x = x.float()
        if self.in_dim * self.out_dim <= _DENSE_MATRIX_MAX_ELEMS:
            return x @ self.dense_matrix()
        out = torch.zeros((x.shape[0], self.out_dim), dtype=torch.float32, device=x.device)
        out.scatter_add_(1, self.buckets.unsqueeze(0).expand(x.shape[0], -1), x * self.signs)
        return out

    def sparse(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Rows given as (indices [N, K], values [N, K]) -> [N, out_dim] float32."""
        idx = indices.long()
        out = torch.zeros((idx.shape[0], self.out_dim), dtype=torch.float32, device=values.device)
        out.scatter_add_(1, self.buckets[idx], values.float() * self.signs[idx])
        return out


@dataclass
class RiseProjections:
    """The three sketches RISE uses: h (hidden), r (vocab residual), g (embedding-space error)."""

    h: CountSketch
    r: CountSketch
    g: CountSketch

    @classmethod
    def from_seed(cls, config: RiseConfig, vocab_size: int, hidden_dim: int) -> "RiseProjections":
        # Seeds and widths as in the research code: h=seed, g=seed+1, r=seed+2.
        rng = config.sketch_rng
        return cls(
            h=CountSketch.from_seed(hidden_dim, config.Kh, config.seed, rng),
            r=CountSketch.from_seed(vocab_size, config.get_effective_Kr(), config.seed + 2, rng),
            g=CountSketch.from_seed(hidden_dim, config.Kg, config.seed + 1, rng),
        )

    def to(self, device) -> "RiseProjections":
        return RiseProjections(h=self.h.to(device), r=self.r.to(device), g=self.g.to(device))

    def check_shapes(self, config: RiseConfig, vocab_size: int, hidden_dim: int) -> None:
        want = {
            "h": (hidden_dim, config.Kh),
            "r": (vocab_size, config.get_effective_Kr()),
            "g": (hidden_dim, config.Kg),
        }
        for name, (i, o) in want.items():
            cs: CountSketch = getattr(self, name)
            if (cs.in_dim, cs.out_dim) != (i, o):
                raise ValueError(
                    f"projection {name} is {cs.in_dim}->{cs.out_dim}, expected {i}->{o} "
                    "(config or model does not match the index)")

    # ---- persistence ----
    def _arrays(self) -> Dict[str, np.ndarray]:
        arrs: Dict[str, np.ndarray] = {}
        for name in ("h", "r", "g"):
            cs: CountSketch = getattr(self, name)
            arrs[f"{name}_buckets"] = cs.buckets.cpu().numpy().astype(np.int32)
            arrs[f"{name}_signs"] = cs.signs.cpu().numpy().astype(np.int8)
            arrs[f"{name}_out_dim"] = np.array(cs.out_dim, dtype=np.int64)
        return arrs

    def save(self, path: str) -> None:
        buf = io.BytesIO()
        np.savez(buf, **self._arrays())
        write_bytes_atomic(buf.getvalue(), path)

    @classmethod
    def load(cls, path: str) -> "RiseProjections":
        with np.load(path, allow_pickle=False) as z:
            def mk(name: str) -> CountSketch:
                return CountSketch(
                    torch.from_numpy(z[f"{name}_buckets"].astype(np.int64)),
                    torch.from_numpy(z[f"{name}_signs"].astype(np.float32)),
                    int(z[f"{name}_out_dim"]),
                )
            return cls(h=mk("h"), r=mk("r"), g=mk("g"))

    def sha256(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self._arrays().items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()
