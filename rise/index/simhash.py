"""SimHash codes (Charikar, 2002): an index's rows kept as sign bits of a randomized Hadamard projection.

A row x is stored as sign(P x), packed eight to a byte (``np.packbits`` order), where P picks ``bits``
coordinates of H D2 H D1 x / n (a randomized Hadamard transform; Ailon and Chazelle, 2006): D1, D2 are
diagonals of random signs, H the n x n Walsh-Hadamard transform, and n the row dimension rounded up to
a power of two (more blocks of independent signs when bits > n). The transform is orthogonal, so each
coordinate of a unit row has variance about 1/n.
One round (H D1 x) leaves the signs of structured rows, such as RISE's sums of outer products,
correlated: on the Llama-3.1-405B 1M index at 8192 bits, two rounds gave P@10 0.805 +- 0.003 over
three draws, one round 0.788 +- 0.013 (uncompressed: 0.811).

Queries stay float. For unit rows, sqrt(pi/2) * sqrt(n) / bits * (P q) . sign(P x) estimates q . x:
E[sign(Y) Z] = rho * sigma * sqrt(2/pi) for jointly Gaussian coordinates. The estimate is linear in
q, so a mean query still scores the mean of the per-query scores.

The signs and picks are saved with the index, so queries never depend on reproducing a random stream.
"""

from __future__ import annotations

import io
import math

import numpy as np
import torch

from ..utils.io import write_bytes_atomic

SIMHASH_FILE = "simhash.npz"


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform over the last dimension (a power of two), Sylvester order."""
    shape, n = x.shape, x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard transform needs a power-of-two length, got {n}")
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a, b = x[:, :, :1], x[:, :, 1:]
        x = torch.cat((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(shape)


class SimHash:
    def __init__(self, signs: np.ndarray, picks: np.ndarray, dim: int):
        signs, picks = np.asarray(signs, dtype=np.int8), np.asarray(picks, dtype=np.int64)
        if signs.ndim != 3 or signs.shape[1] != 2:
            raise ValueError(f"signs must be [blocks, 2, n], got {signs.shape}")
        blocks, _, n = signs.shape
        self.dim, self.n, self.bits = int(dim), int(n), int(picks.size)
        if n & (n - 1) or n < self.dim or n // 2 >= self.dim > 1:
            raise ValueError(f"sign rows of length {n} do not fit dim {dim}")
        if self.bits < 8 or self.bits % 8 or len(np.unique(picks)) != self.bits or picks.min() < 0 \
                or picks.max() >= blocks * n or not np.all(np.abs(signs) == 1):
            raise ValueError("malformed SimHash parameters")
        self.signs, self.picks = signs, picks
        self.code_bytes = self.bits // 8
        self.scale = math.sqrt(math.pi / 2) * math.sqrt(n) / self.bits
        self._on = {}

    @classmethod
    def draw(cls, dim: int, bits: int, seed: int = 0) -> "SimHash":
        if bits < 8 or bits % 8:
            raise ValueError(f"bits must be a positive multiple of 8, got {bits}")
        n = 1 << max(0, int(dim - 1).bit_length())
        g = torch.Generator().manual_seed(int(seed))
        blocks = -(-bits // n)
        signs = torch.randint(2, (blocks, 2, n), generator=g, dtype=torch.int8) * 2 - 1
        picks = torch.randperm(blocks * n, generator=g)[:bits].sort().values
        return cls(signs.numpy(), picks.numpy(), dim)

    def save(self, path: str) -> None:
        buf = io.BytesIO()
        np.savez(buf, signs=self.signs, picks=self.picks, dim=np.int64(self.dim))
        write_bytes_atomic(buf.getvalue(), path)

    @classmethod
    def load(cls, path: str) -> "SimHash":
        with np.load(path) as d:
            return cls(d["signs"], d["picks"], int(d["dim"]))

    def _params(self, device):
        key = str(device)
        if key not in self._on:
            self._on[key] = (torch.from_numpy(self.signs).to(device, torch.float32),
                             torch.from_numpy(self.picks).to(device))
        return self._on[key]

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """[B, dim] -> [B, bits] float32 coordinates of the orthogonal transform."""
        signs, picks = self._params(x.device)
        x = torch.nn.functional.pad(x.float(), (0, self.n - self.dim))
        root = math.sqrt(self.n)
        y = torch.cat([fwht(fwht(x * s1) / root * s2) / root for s1, s2 in signs], dim=1)
        return y[:, picks]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, dim] rows -> [B, bits // 8] uint8 sign bits."""
        bits = (self.project(x) > 0).to(torch.int32).view(x.shape[0], self.code_bytes, 8)
        weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int32, device=x.device)
        return (bits * weights).sum(dim=2).to(torch.uint8)

    @staticmethod
    def unpack(codes: torch.Tensor) -> torch.Tensor:
        """[B, bits // 8] uint8 -> [B, bits] float32 of +-1."""
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=codes.device)
        bits = (codes.unsqueeze(-1) >> shifts) & 1
        return bits.reshape(codes.shape[0], -1).float() * 2 - 1

    def queries(self, q: torch.Tensor) -> torch.Tensor:
        """Float queries in the row space -> vectors whose dot product with ``unpack``ed codes estimates q . x."""
        return self.project(q) * self.scale
