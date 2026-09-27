"""Row-wise top-K of 16-bit logits (fp16 / bf16) in Triton.

torch.topk took most of RISE's head time on vocabulary-sized rows: its radix select reads each row
several times from one block per row. Here a (row, span) grid reads each row once, at close to HBM
bandwidth, and a per-row program then touches only a few thousand elements:

1. chunk maxima: the largest element of every CHUNK consecutive ones (the one full pass);
2. per row, the bound t = the K-th largest chunk maximum. Those K maxima are K distinct elements,
   so at least K elements are >= t and the top-K is among them;
3. only chunks whose maximum is >= t can hold such elements: about K of them, read back and
   compacted into the candidates (elements >= t), in index order;
4. a sort of the candidates by (value descending, index ascending); the first K are kept.

The top of a logit row sits in a few hundred chunks, so step 3 keeps about K..2K elements. Rows
with more candidates than the common-case sort holds go to a second kernel: a wider sort, or, past
the candidate buffer (flat rows, masses of ties), an exact radix select over the whole row. Those
paths live in their own kernel because their wide sorts inflate register use: inline, the main
kernel ran 1.5x slower. All comparisons use torch.topk's order-preserving 16-bit keys (NaN on top),
so every path returns exactly torch.topk's values; among equal values the lowest indices win.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_MAX_K = 1024
_SPAN = 4096  # elements per program of the streaming pass


def _chunk_for(V: int) -> int:
    # Short chunks keep the bound tight: 16 (Pythia's 50k vocabulary) and 32 (Llama-3's 128k) leave
    # a median of ~330 candidates for K=256 on real rows (99th percentile ~450). The per-row chunk
    # array is capped at 4096 because the select kernel holds it in registers: at 8192 it ran 2.2x
    # slower on Llama-3 rows.
    return max(16, triton.next_power_of_2(triton.cdiv(V, 4096)))


def available(x: torch.Tensor, k: int) -> bool:
    if triton is None or not x.is_cuda or x.dim() != 2 or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    V = x.shape[1]
    return (0 < k <= _MAX_K and V < 2 ** 31 and triton.cdiv(V, _chunk_for(V)) >= k
            and os.environ.get("RISE_DISABLE_TRITON", "") != "1")


if triton is not None:

    @triton.jit
    def _ordered_key(x):
        """16-bit float -> int32 in [0, 65536) that sorts like torch.topk orders values (NaN largest)."""
        bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
        return tl.where(x == x, key, 0xFFFF)

    @triton.jit
    def _chunk_max_kernel(x_ptr, cmax_ptr, V, stride_x, C, CHUNK: tl.constexpr, SPAN: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        NC: tl.constexpr = SPAN // CHUNK
        cidx = tl.program_id(1) * NC + tl.arange(0, NC)
        offs = cidx[:, None] * CHUNK + tl.arange(0, CHUNK)[None, :]
        m = offs < V
        key = tl.where(m, _ordered_key(tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)), 0)
        # stored as int16 (key - 32768): order-preserving, half the traffic of int32
        tl.store(cmax_ptr + row * C + cidx, (tl.max(key, axis=1) - 32768).to(tl.int16), mask=cidx < C)

    @triton.jit
    def _emit(krow, jrow, n, K, irow, vrow, S: tl.constexpr):
        """Sort the n candidates (key - 32768 at krow, index at jrow; equal keys in index order) and
        write the first K as (value, index)."""
        cc = tl.arange(0, S)
        k16 = tl.load(krow + cc, mask=cc < n, other=0).to(tl.int32)
        # (key, -slot) in one int32; slots follow index order, so a descending sort gives value
        # descending, index ascending. Real entries are > INT32_MIN since slot < 65536.
        sk = tl.sort(tl.where(cc < n, (k16 << 16) | (0xFFFF - cc), -2147483648), descending=True)
        km = cc < K
        tl.store(irow + cc, tl.load(jrow + (0xFFFF - (sk & 0xFFFF)), mask=km, other=0).to(tl.int64), mask=km)
        # the key is a bijection of the value's bits, so the value needs no gather
        key = (sk >> 16) + 32768
        bits = tl.where((key & 0x8000) != 0, key & 0x7FFF, (~key) & 0xFFFF)
        tl.store(vrow + cc, bits.to(tl.int16).to(vrow.dtype.element_ty, bitcast=True), mask=km)

    @triton.jit
    def _select_kernel(x_ptr, cmax_ptr, chl_ptr, ckey_ptr, cidx_ptr, stat_ptr, vals_ptr, idx_ptr, V, stride_x, C, K,
                       CB: tl.constexpr, CHUNK: tl.constexpr, MAXCH: tl.constexpr, GB: tl.constexpr,
                       S1: tl.constexpr, CAP: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        xrow = x_ptr + row * stride_x
        crow = chl_ptr + row * MAXCH
        krow = ckey_ptr + row * CAP
        jrow = cidx_ptr + row * CAP

        # 2) t = the K-th largest chunk maximum, by bisection over the 16-bit key range
        cc = tl.arange(0, CB)
        keys = tl.where(cc < C, tl.load(cmax_ptr + row * C + cc, mask=cc < C, other=0).to(tl.int32) + 32768, -1)
        lo = 0       # invariant: at least K keys >= lo (C >= K)
        hi = 65536   # invariant: fewer than K keys >= hi
        for _ in tl.static_range(16):
            mid = (lo + hi) // 2
            ok = tl.sum((keys >= mid).to(tl.int32), axis=0) >= K
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        t = lo

        # 3) the chunks whose maximum is >= t, then their elements >= t
        csel = keys >= t
        n_ch = tl.sum(csel.to(tl.int32), axis=0)
        cpos = tl.cumsum(csel.to(tl.int32), axis=0) - 1
        tl.store(crow + cpos, cc, mask=csel & (cpos < MAXCH))
        tl.debug_barrier()  # the chunk list was written by other threads of this program
        n = 0
        for c0 in range(0, tl.where(n_ch <= MAXCH, n_ch, 0), GB):
            gi = c0 + tl.arange(0, GB)
            ch = tl.load(crow + gi, mask=gi < n_ch, other=0)
            offs = ch[:, None] * CHUNK + tl.arange(0, CHUNK)[None, :]
            em = (gi < n_ch)[:, None] & (offs < V)
            key = _ordered_key(tl.load(xrow + offs, mask=em, other=0.0))
            s32 = (em & (key >= t)).to(tl.int32)
            per = tl.sum(s32, axis=1)
            pos = n + (tl.cumsum(per, axis=0) - per)[:, None] + tl.cumsum(s32, axis=1) - 1
            sm = (s32 != 0) & (pos < CAP)
            tl.store(krow + pos, (key - 32768).to(tl.int16), mask=sm)
            tl.store(jrow + pos, offs, mask=sm)
            n += tl.sum(per, axis=0)
        tl.debug_barrier()

        # 4) the common case; other rows are left to _fallback_kernel (status = candidates, or CAP + 1)
        done = (n_ch <= MAXCH) & (n <= S1)
        if done:
            _emit(krow, jrow, n, K, idx_ptr + row * K, vals_ptr + row * K, S1)
        tl.store(stat_ptr + row, tl.where(done, 0, tl.where(n_ch <= MAXCH, n, CAP + 1)))

    @triton.jit
    def _radix_select(xrow, krow, jrow, V, K, BLOCK: tl.constexpr):
        """Exact byte-wise radix select over the whole row: writes the K survivors (key - 32768,
        index), equal keys in index order."""
        offs = tl.arange(0, BLOCK)
        bins = tl.arange(0, 256)
        h1 = tl.zeros([256], dtype=tl.int32)
        for s in range(0, V, BLOCK):
            m = s + offs < V
            key = _ordered_key(tl.load(xrow + s + offs, mask=m, other=0.0))
            h1 += tl.histogram(key >> 8, 256, mask=m)
        suf1 = tl.sum(h1, axis=0) - tl.cumsum(h1, axis=0) + h1
        b1 = tl.max(tl.where(suf1 >= K, bins, -1), axis=0)
        above1 = tl.sum(tl.where(bins == b1, suf1 - h1, 0), axis=0)
        h2 = tl.zeros([256], dtype=tl.int32)
        for s in range(0, V, BLOCK):
            m = s + offs < V
            key = _ordered_key(tl.load(xrow + s + offs, mask=m, other=0.0))
            h2 += tl.histogram(key & 0xFF, 256, mask=m & ((key >> 8) == b1))
        suf2 = tl.sum(h2, axis=0) - tl.cumsum(h2, axis=0) + h2
        b2 = tl.max(tl.where(suf2 >= K - above1, bins, -1), axis=0)
        n_above = above1 + tl.sum(tl.where(bins == b2, suf2 - h2, 0), axis=0)
        thr = b1 * 256 + b2
        n_gt = 0
        n_eq = 0
        for s in range(0, V, BLOCK):
            m = s + offs < V
            key = _ordered_key(tl.load(xrow + s + offs, mask=m, other=0.0))
            sel_gt = m & (key > thr)
            sel_eq = m & (key == thr)
            p_gt = n_gt + tl.cumsum(sel_gt.to(tl.int32), axis=0) - 1
            p_eq = n_above + n_eq + tl.cumsum(sel_eq.to(tl.int32), axis=0) - 1
            sm = sel_gt | (sel_eq & (p_eq < K))
            pos = tl.where(sel_gt, p_gt, p_eq)
            tl.store(krow + pos, (key - 32768).to(tl.int16), mask=sm)
            tl.store(jrow + pos, s + offs, mask=sm)
            n_gt += tl.sum(sel_gt.to(tl.int32), axis=0)
            n_eq += tl.sum(sel_eq.to(tl.int32), axis=0)

    @triton.jit
    def _fallback_kernel(x_ptr, ckey_ptr, cidx_ptr, stat_ptr, vals_ptr, idx_ptr, V, stride_x, K,
                         S1: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        krow = ckey_ptr + row * CAP
        jrow = cidx_ptr + row * CAP
        s = tl.load(stat_ptr + row)
        if s > 0:
            if s <= CAP:
                _emit(krow, jrow, s, K, idx_ptr + row * K, vals_ptr + row * K, CAP)
            else:
                _radix_select(x_ptr + row * stride_x, krow, jrow, V, K, BLOCK)
                tl.debug_barrier()  # the survivors were written by other threads of this program
                _emit(krow, jrow, K, K, idx_ptr + row * K, vals_ptr + row * K, S1)


def topk16(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch.topk(x, k, dim=1) for [N, V] fp16 / bf16 CUDA tensors: (values, int64 indices), sorted."""
    if not available(x, k):
        return torch.topk(x, k, dim=1)
    if x.stride(1) != 1:
        x = x.contiguous()
    N, V = x.shape
    dev = x.device
    vals = torch.empty((N, k), dtype=x.dtype, device=dev)
    idx = torch.empty((N, k), dtype=torch.int64, device=dev)
    if N == 0:
        return vals, idx
    chunk = _chunk_for(V)
    C = triton.cdiv(V, chunk)
    CB = triton.next_power_of_2(C)
    kp = triton.next_power_of_2(k)
    s1 = max(512, 2 * kp)  # sort width of the common case
    cap = 4 * s1           # candidates kept per row
    maxch = 2 * kp         # selected chunks kept per row
    cmax = torch.empty((N, C), dtype=torch.int16, device=dev)
    chl = torch.empty((N, maxch), dtype=torch.int32, device=dev)
    ckey = torch.empty((N, cap), dtype=torch.int16, device=dev)
    cidx = torch.empty((N, cap), dtype=torch.int32, device=dev)
    stat = torch.empty((N,), dtype=torch.int32, device=dev)
    _chunk_max_kernel[(N, triton.cdiv(V, _SPAN))](x, cmax, V, x.stride(0), C, CHUNK=chunk, SPAN=_SPAN,
                                                  num_warps=4)
    _select_kernel[(N,)](x, cmax, chl, ckey, cidx, stat, vals, idx, V, x.stride(0), C, k, CB=CB, CHUNK=chunk,
                         MAXCH=maxch, GB=64, S1=s1, CAP=cap, num_warps=4)
    _fallback_kernel[(N,)](x, ckey, cidx, stat, vals, idx, V, x.stride(0), k, S1=s1, CAP=cap, BLOCK=4096,
                           num_warps=4)
    return vals, idx
