"""Sketches of the sparse prediction residual, fused (Triton).

After the cumulative-probability cutoff a position's residual r_t is non-zero only on the kept
prefix of its top-K slots plus the ground-truth slot: a handful of tokens at tau = 0.1, against
K = 256 slots. The PyTorch path scatters r into the CountSketch buckets and gathers M_g rows for
all K slots. This kernel visits only the non-zero slots and fuses, per position:

    r^ = CS_r(r) / |CS_r(r)|,   g^ = (sum_v r(v) M_g[v]) / |.|,   both multiplied by the mask.

Same arithmetic as the PyTorch path; only the summation order of the (few) non-zero terms differs.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None


def available(device) -> bool:
    return (triton is not None and torch.device(device).type == "cuda"
            and os.environ.get("RISE_DISABLE_TRITON", "") != "1")


if triton is not None:

    @triton.jit
    def _sparse_sketch_kernel(res_ptr, idx_ptr, cut_ptr, gtp_ptr, mask_ptr,
                              rb_ptr, rs_ptr, mg_ptr, out_r_ptr, out_g_ptr,
                              K, KR, KG, eps,
                              DO_R: tl.constexpr, DO_G: tl.constexpr,
                              BLOCK_R: tl.constexpr, BLOCK_G: tl.constexpr):
        row = tl.program_id(0)
        base = row.to(tl.int64) * K
        ar_r = tl.arange(0, BLOCK_R)
        ar_g = tl.arange(0, BLOCK_G)
        acc_r = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc_g = tl.zeros([BLOCK_G], dtype=tl.float32)
        n = tl.load(cut_ptr + row)
        gp = tl.load(gtp_ptr + row)
        extra = (gp >= n).to(tl.int32)  # the ground truth sits outside the kept prefix
        for j in range(0, n + extra):
            k = tl.where(j < n, j, gp)
            v = tl.load(idx_ptr + base + k)
            r = tl.load(res_ptr + base + k)
            if DO_R:
                bkt = tl.load(rb_ptr + v)
                sg = tl.load(rs_ptr + v)
                acc_r += tl.where(ar_r == bkt, sg * r, 0.0)
            if DO_G:
                acc_g += r * tl.load(mg_ptr + v * KG + ar_g, mask=ar_g < KG, other=0.0)
        m = tl.load(mask_ptr + row)
        out_base = row.to(tl.int64)
        if DO_R:
            nr = tl.sqrt(tl.sum(acc_r * acc_r, axis=0))
            tl.store(out_r_ptr + out_base * KR + ar_r, acc_r / tl.maximum(nr, eps) * m, mask=ar_r < KR)
        if DO_G:
            ng = tl.sqrt(tl.sum(acc_g * acc_g, axis=0))
            tl.store(out_g_ptr + out_base * KG + ar_g, acc_g / tl.maximum(ng, eps) * m, mask=ar_g < KG)


def sparse_sketch(res: torch.Tensor, idx: torch.Tensor, cut: torch.Tensor, gt_pos: torch.Tensor,
                  mask: torch.Tensor, r_buckets: Optional[torch.Tensor], r_signs: Optional[torch.Tensor],
                  r_dim: int, M_g: Optional[torch.Tensor], eps: float
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalized, masked r / g sketches for N positions.

    Args:
        res: [N, K] float32 residuals (non-zero only on slots [0, cut) and gt_pos).
        idx: [N, K] int64 token ids of the slots.
        cut: [N] int32 kept-prefix length; gt_pos: [N] int32 ground-truth slot.
        mask: [N] float32 0/1 position mask.
        r_buckets / r_signs: CountSketch tables of the r channel ([V] int32 / float32), or None.
        M_g: [V, Kg] float32 sketched unembedding, or None.
    Returns:
        (p_r [N, r_dim] or None, p_g [N, Kg] or None), float32.
    """
    N, K = res.shape
    do_r, do_g = r_buckets is not None, M_g is not None
    kg = int(M_g.shape[1]) if do_g else 1
    dev = res.device
    out_r = torch.empty((N, r_dim), dtype=torch.float32, device=dev) if do_r else torch.empty(1, device=dev)
    out_g = torch.empty((N, kg), dtype=torch.float32, device=dev) if do_g else torch.empty(1, device=dev)
    if N:
        dummy_i = torch.zeros(1, dtype=torch.int32, device=dev)
        dummy_f = torch.zeros(1, dtype=torch.float32, device=dev)
        _sparse_sketch_kernel[(N,)](
            res.contiguous(), idx.contiguous(), cut.contiguous(), gt_pos.contiguous(), mask.contiguous(),
            r_buckets if do_r else dummy_i, r_signs if do_r else dummy_f,
            M_g if do_g else dummy_f, out_r, out_g,
            K, r_dim if do_r else 1, kg, eps,
            DO_R=do_r, DO_G=do_g,
            BLOCK_R=triton.next_power_of_2(max(r_dim if do_r else 1, 16)),
            BLOCK_G=triton.next_power_of_2(max(kg, 16)),
            num_warps=2,
        )
    return (out_r if do_r else None), (out_g if do_g else None)
