"""The RISE head: post-final-norm hidden states -> per-chunk influence signatures.

"The trunk serves hidden states, we own the head": any backend that returns
the hidden states fed to the LM head (HF transformers, SGLang, ...) can drive
RISE, because the head recomputes the logits itself. Per chunk, for positions
t predicting token y_t (paper, Sec. 3 / Alg. 1):

    z_t  = W h_t (+ b)                         logits from the unembedding W [V, D]
    tau  : mean_t H(softmax(z_t / tau)) ~= target_entropy       (bisection, per chunk)
    S_t  = TopK(z_t / tau) with y_t forced in, cut to the smallest prefix whose
           cumulative probability >= topL_cum_prob (y_t always kept), renormalized
    r_t  = p_t - onehot(y_t) on S_t            sparse prediction residual
    h^_t = CS_h(h_t),  r^_t = CS_r(r_t),  g^_t = CS_g(W^T r_t) = sum_v r_t(v) M_g[v]
           with M_g = CS_g(W) precomputed once (CountSketch is linear); each L2-normalized
    phi  = [l_rh sum_t r^_t (x) h^_t,  l_gh sum_t g^_t (x) h^_t,  l_rg sum_t r^_t (x) g^_t]
           concatenated and L2-normalized.

The inner product of two signatures is the sketched LM-head gradient inner
product sum_{t,s} <r_t, r_s><h_t, h_s> (+ the GH/RG terms), i.e. the TracIn
surrogate restricted to the readout layer.

Numerics follow the research code (v5/v6). Execution differs only where results
cannot change: padded positions are excluded, the tau bisection skips rows
that already converged, entropy is computed as logsumexp - E_p[z] (no p*log p
temporaries, no eps clamp), and work is split into row micro-batches so the
[tokens, V] logits stay within ``max_tokens_per_step``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .config import RiseConfig
from .kernels import sparse_sketch, topk as topk_kernel
from .sketch import RiseProjections

_M_G_ROW_STEP = 8192


def l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


class RiseHead:
    GH_K_SLICE = 64

    def __init__(
        self,
        config: RiseConfig,
        unembedding: torch.Tensor,
        *,
        projections: Optional[RiseProjections] = None,
        gh_embedding: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        logits_postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device=None,
        max_tokens_per_step: int = 8192,
    ):
        """
        Args:
            unembedding: LM-head weight W [V, D] (any float dtype; logits are computed in it).
            projections: sketch tables; generated from ``config.seed`` if omitted.
            gh_embedding: matrix projected for the GH/RG channels. Defaults to W, which is
                what ``embedding_type="output"`` means; pass the input embedding for "input".
            bias / logits_postprocess: reproduce models whose logits are not plain W h
                (LM-head bias, final soft-capping, logit scaling).
        """
        config.validate()
        self.config = config
        self.device = torch.device(device) if device is not None else unembedding.device
        self.W = unembedding.detach().to(self.device)
        self.V, self.D = (int(s) for s in self.W.shape)
        self.bias = bias.detach().to(self.device) if bias is not None else None
        self.logits_postprocess = logits_postprocess
        self.max_tokens_per_step = int(max_tokens_per_step)
        self.dim = config.get_vector_dim()

        if projections is None:
            projections = RiseProjections.from_seed(config, self.V, self.D)
        projections.check_shapes(config, self.V, self.D)
        self.projections = projections.to(self.device)
        self._r_buckets_i32 = self.projections.r.buckets.to(torch.int32)  # compact table for the fused kernel

        self.M_g: Optional[torch.Tensor] = None
        if config.has_channel("gh") or config.has_channel("rg"):
            E = self.W if gh_embedding is None else gh_embedding.detach().to(self.device)
            if tuple(E.shape) != (self.V, self.D):
                raise ValueError(f"GH embedding is {tuple(E.shape)}, expected {(self.V, self.D)}")
            self.M_g = self._sketch_rows(E)

        k = config.max_topL_cap if config.adaptive_topL else config.fallback_topL
        self.topk = min(max(k, config.min_topL), self.V)

    @classmethod
    def from_trunk(cls, trunk, config: RiseConfig, *, projections=None, device=None,
                   max_tokens_per_step: int = 8192) -> "RiseHead":
        """Build a head whose logits reproduce ``trunk``'s LM head exactly."""
        gh = trunk.input_embedding() if config.embedding_type == "input" else None
        return cls(
            config,
            trunk.unembedding(),
            projections=projections,
            gh_embedding=gh,
            bias=trunk.unembedding_bias(),
            logits_postprocess=trunk.logits_postprocess,
            device=device if device is not None else trunk.device,
            max_tokens_per_step=max_tokens_per_step,
        )

    # ------------------------------------------------------------------ pieces
    @torch.inference_mode()
    def _sketch_rows(self, E: torch.Tensor) -> torch.Tensor:
        cs = self.projections.g
        out = torch.empty((E.shape[0], cs.out_dim), dtype=torch.float32, device=self.device)
        for s in range(0, E.shape[0], _M_G_ROW_STEP):
            out[s:s + _M_G_ROW_STEP] = cs.dense(E[s:s + _M_G_ROW_STEP])
        return out

    def _raw_logits(self, h: torch.Tensor) -> torch.Tensor:
        """[N, D] hidden -> [N, V] logits before postprocessing, in the unembedding dtype."""
        z = h.to(self.W.dtype) @ self.W.t()
        if self.bias is not None:
            z = z + self.bias
        return z

    def _finish(self, z: torch.Tensor) -> torch.Tensor:
        """Raw logits (any subset of them) -> float32 postprocessed logits."""
        z = z.float()
        if self.logits_postprocess is not None:
            z = self.logits_postprocess(z)
        return z

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """[N, D] hidden -> [N, V] float32 logits, computed in the unembedding dtype."""
        return self._finish(self._raw_logits(h))

    def _mean_entropy(self, logits: torch.Tensor, tau: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean over valid positions of the entropy of softmax(logits / tau); [b], in tau's dtype."""
        b, T, V = logits.shape
        s = logits / tau.to(logits.dtype).view(b, 1, 1)
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
        p = (s - lse).exp_()
        e_s = torch.bmm(p.view(b * T, 1, V), s.view(b * T, V, 1)).view(b, T)
        H = lse.squeeze(-1) - e_s
        vf = valid.to(H.dtype)
        return ((H * vf).sum(dim=-1) / vf.sum(dim=-1).clamp(min=1.0)).to(tau.dtype)

    @torch.inference_mode()
    def tau_search(self, logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-row temperature so the mean entropy over valid positions hits target_entropy.

        Same bisection as the research code: early exit within 0.01 nats, midpoint of the
        final bracket otherwise, ``tau_fallback`` for rows with non-finite logits.
        """
        c = self.config
        b = logits.shape[0]
        dev = logits.device
        fdt = torch.float32 if dev.type == "mps" else torch.float64  # bracket precision (MPS lacks fp64)
        result = torch.full((b,), 0.5 * (c.tau_min + c.tau_max), dtype=fdt, device=dev)
        finite = (torch.isfinite(logits) | ~valid.unsqueeze(-1)).flatten(1).all(dim=1)
        done = ~finite | ~valid.any(dim=1)
        result[~finite] = c.tau_fallback
        if c.target_entropy >= math.log(self.V):
            result[~done] = c.tau_max
            return result.float()
        lo = torch.full((b,), c.tau_min, dtype=fdt, device=dev)
        hi = torch.full((b,), c.tau_max, dtype=fdt, device=dev)
        for _ in range(c.tau_search_steps):
            active = (~done).nonzero().squeeze(1)
            if active.numel() == 0:
                break
            mid = 0.5 * (lo[active] + hi[active])
            H = self._mean_entropy(logits[active], mid, valid[active])
            close = (H - c.target_entropy).abs() < 0.01
            result[active[close]] = mid[close]
            done[active[close]] = True
            go_up = H < c.target_entropy  # entropy too low -> raise tau
            open_ = ~close
            lo[active[open_ & go_up]] = mid[open_ & go_up]
            hi[active[open_ & ~go_up]] = mid[open_ & ~go_up]
        rest = ~done
        result[rest] = 0.5 * (lo[rest] + hi[rest])
        result = torch.where(torch.isfinite(result), result, torch.full_like(result, c.tau_fallback))
        return result.float()

    def _append_residual(self, vals: torch.Tensor, idx: torch.Tensor, y: torch.Tensor, gt_val: torch.Tensor):
        """``gt_slot="append"``, as the December 2025 research scripts: the top-L cutoff is taken on the
        top-K alone, and the ground truth, unless inside the kept prefix, becomes slot K. Probabilities
        are renormalized over the kept slots. Returns [N, K+1] ids and residuals, the prefix length and
        the ground-truth slot."""
        c = self.config
        N, K = vals.shape
        dev = vals.device
        if c.adaptive_topL:
            hit = torch.softmax(vals, dim=-1).cumsum(dim=-1) >= c.topL_cum_prob
            first = hit.float().argmax(dim=-1)
            cutoff = torch.where(hit.any(dim=-1), first + 1, torch.full_like(first, K))
            cutoff = cutoff.clamp(min=min(c.min_topL, K), max=K)
        else:
            cutoff = torch.full((N,), K, dtype=torch.long, device=dev)
        prefix = torch.arange(K, device=dev).unsqueeze(0) < cutoff.unsqueeze(1)
        gt_in = (idx == y.unsqueeze(1)) & prefix
        kept = gt_in.any(dim=1, keepdim=True)
        idx = torch.cat([idx, y.unsqueeze(1)], dim=1)
        vals = torch.cat([vals, gt_val.unsqueeze(1)], dim=1)
        gt_mask = torch.cat([gt_in, ~kept], dim=1)
        ex = (vals - vals.max(dim=-1, keepdim=True).values).exp() * torch.cat([prefix, ~kept], dim=1)
        res = ex / (ex.sum(dim=-1, keepdim=True) + c.eps) - gt_mask.float()
        gt_pos = torch.where(kept.squeeze(1), gt_in.to(torch.uint8).argmax(dim=1), torch.full_like(cutoff, K))
        return idx, res, cutoff, gt_pos.to(torch.int32)

    def _gh(self, idx: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """sum_k res[:, k] * M_g[idx[:, k]] in K-slices (never materializes [N, K, Kg] at once)."""
        N, K = idx.shape
        out = torch.zeros((N, self.M_g.shape[1]), dtype=torch.float32, device=self.device)
        for k0 in range(0, K, self.GH_K_SLICE):
            k1 = min(K, k0 + self.GH_K_SLICE)
            rows = self.M_g.index_select(0, idx[:, k0:k1].reshape(-1)).view(N, k1 - k0, -1)
            out += torch.bmm(res[:, k0:k1].unsqueeze(1), rows).squeeze(1)
        return out

    # ------------------------------------------------------------------ main entry
    @torch.inference_mode()
    def compute_from_hidden(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        lens: torch.Tensor,
        loss_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Signatures for a right-padded batch of chunks.

        Args:
            hidden: [B, L, D] post-final-norm hidden states.
            ids: [B, L] token ids (right-padded).
            lens: [B] true lengths.
            loss_start: optional [B]; only positions predicting token index >= loss_start
                contribute (prompt-masked query vectors). tau is still searched on all
                positions, as in the research code.
        Returns:
            [B, dim] float32, one L2-normalized row per chunk (zeros for chunks < 2 tokens).
        """
        B, L = int(ids.shape[0]), int(ids.shape[1])
        out = torch.zeros((B, self.dim), dtype=torch.float32, device=self.device)
        if L < 2 or B == 0:
            return out
        lens_cpu = lens.detach().to("cpu", torch.long)
        checked = ids.device.type == "cpu"
        if checked:  # validate on the host copy, so the device work below never waits on a sync
            live = torch.arange(L).unsqueeze(0) < lens_cpu.unsqueeze(1)
            top = int(torch.where(live, ids, torch.zeros_like(ids)).max())
            if top >= self.V:
                raise ValueError(f"token id {top} >= LM-head rows {self.V}: tokenizer/model mismatch")
        rows_per_step = max(1, self.max_tokens_per_step // (L - 1))
        for s in range(0, B, rows_per_step):
            e = min(B, s + rows_per_step)
            Lm = int(lens_cpu[s:e].max())
            if Lm < 2:
                continue
            out[s:e] = self._compute(
                hidden[s:e, :Lm], ids[s:e, :Lm], lens[s:e],
                None if loss_start is None else loss_start[s:e], checked,
            )
        return out

    def _compute(self, hidden, ids, lens, loss_start, ids_checked: bool = False) -> torch.Tensor:
        c = self.config
        f = self.token_factors(hidden, ids, lens, loss_start, ids_checked=ids_checked)
        p_r, p_h, p_g = f["r"], f["h"], f["g"]
        b = p_h.shape[0] if p_h is not None else p_r.shape[0]
        parts = []
        if c.has_channel("rh"):
            parts.append(c.lambda_rh * torch.bmm(p_r.transpose(1, 2), p_h).reshape(b, -1))
        if c.has_channel("gh"):
            parts.append(c.lambda_gh * torch.bmm(p_g.transpose(1, 2), p_h).reshape(b, -1))
        if c.has_channel("rg"):
            parts.append(c.lambda_rg * torch.bmm(p_r.transpose(1, 2), p_g).reshape(b, -1))
        vec = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if c.normalize_sample:
            vec = l2_normalize(vec, c.norm_eps())
        return vec

    @torch.inference_mode()
    def token_factors(self, hidden, ids, lens, loss_start=None, ids_checked: bool = False) -> dict:
        """Per-token normalized sketches before aggregation.

        Returns {"r", "h", "g"}: [b, T, K] tensors (None for unused factors) with
        masked positions zeroed, plus "mask" [b, T] and "tau" [b]. The signature is
        sum_t of their outer products, so these give per-token influence breakdowns.
        """
        c = self.config
        dev = self.device
        hidden = hidden.to(dev, non_blocking=True)
        ids = ids.to(dev, non_blocking=True).long()
        lens = lens.to(dev, non_blocking=True).long()
        b, L, D = hidden.shape
        T = L - 1
        N = b * T
        pos = torch.arange(T, device=dev)
        valid = pos.unsqueeze(0) < (lens - 1).unsqueeze(1)                    # [b, T]
        nxt = torch.where(valid, ids[:, 1:], torch.zeros_like(ids[:, 1:]))   # [b, T]
        if not ids_checked and int(nxt.max()) >= self.V:  # device-side check syncs; hosts pass checked ids
            raise ValueError(f"token id {int(nxt.max())} >= LM-head rows {self.V}: tokenizer/model mismatch")

        h = hidden[:, :T, :].reshape(N, D)
        y = nxt.reshape(N)
        K = self.topk
        if c.adaptive_temperature:
            logits = self.logits(h).view(b, T, self.V)
            logits.masked_fill_(~valid.unsqueeze(-1), 0.0)
            tau = self.tau_search(logits, valid)
            logits.div_(tau.view(b, 1, 1))
            flat = logits.view(N, self.V)
            vals, idx = torch.topk(flat, K, dim=-1)
            gt_val = flat.gather(1, y.unsqueeze(1)).squeeze(1)
            del logits, flat
        else:
            # Fixed temperature: z -> post(z) / tau is increasing, so the top-K of the scaled logits
            # is the top-K of the raw GEMM output. Select on the unembedding dtype and scale only the
            # K winners: no [N, V] float32 copy, mask or division pass, and half the top-k traffic.
            # Values are unchanged (fp16/bf16 -> fp32 is exact); only the order among exact ties may
            # differ from sorting the converted copy. topk16 is torch.topk for 16-bit CUDA rows (same
            # values, lowest indices first among ties) and falls back to it everywhere else.
            tau = torch.full((b,), c.tau_fallback, dtype=torch.float32, device=dev)
            z = self._raw_logits(h)
            vals, idx = topk_kernel.topk16(z, K)
            gt_val = z.gather(1, y.unsqueeze(1)).squeeze(1)
            del z
            vals = self._finish(vals) / c.tau_fallback
            gt_val = self._finish(gt_val) / c.tau_fallback

        if c.gt_slot == "append":
            idx, res, cutoff, gt_pos = self._append_residual(vals, idx, y, gt_val)
        else:
            # Ground truth forced into the last TopK slot when missing.
            need = ~(idx == y.unsqueeze(1)).any(dim=1)  # unconditional: a bool(need.any()) test would sync
            idx[:, -1] = torch.where(need, y, idx[:, -1])
            vals[:, -1] = torch.where(need, gt_val, vals[:, -1])

            ex = (vals - vals.max(dim=-1, keepdim=True).values).exp()
            probs = ex / (ex.sum(dim=-1, keepdim=True) + c.eps)
            gt_mask = idx == y.unsqueeze(1)
            if c.adaptive_topL:
                hit = probs.cumsum(dim=-1) >= c.topL_cum_prob
                first = hit.float().argmax(dim=-1)
                cutoff = torch.where(hit.any(dim=-1), first + 1, torch.full_like(first, K))
                cutoff = cutoff.clamp(min=min(c.min_topL, K), max=K)
                keep = (torch.arange(K, device=dev).unsqueeze(0) < cutoff.unsqueeze(1)) | gt_mask
                probs = probs * keep
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=c.eps)
            else:
                cutoff = torch.full((N,), K, dtype=torch.long, device=dev)
            res = probs - gt_mask.float()  # non-zero only on slots [0, cutoff) and the ground-truth slot
            gt_pos = gt_mask.to(torch.uint8).argmax(dim=1).to(torch.int32)

        m = valid
        if loss_start is not None:
            m = m & ((pos.unsqueeze(0) + 1) >= loss_start.to(dev).long().unsqueeze(1))
        mf = m.reshape(N, 1).float()

        chs = c.get_channels()
        need_r, need_g = any("r" in ch for ch in chs), any("g" in ch for ch in chs)
        p_h = p_r = p_g = None
        if any("h" in ch for ch in chs):
            p_h = (l2_normalize(self.projections.h.dense(h), c.norm_eps()) * mf).view(b, T, -1)
        if (need_r or need_g) and sparse_sketch.available(dev):
            p_r, p_g = sparse_sketch.sparse_sketch(
                res, idx, cutoff.to(torch.int32), gt_pos, mf.view(N),
                self._r_buckets_i32 if need_r else None, self.projections.r.signs if need_r else None,
                self.projections.r.out_dim, self.M_g if need_g else None, c.norm_eps())
            p_r = None if p_r is None else p_r.view(b, T, -1)
            p_g = None if p_g is None else p_g.view(b, T, -1)
        else:
            if need_r:
                p_r = (l2_normalize(self.projections.r.sparse(idx, res), c.norm_eps()) * mf).view(b, T, -1)
            if need_g:
                p_g = (l2_normalize(self._gh(idx, res), c.norm_eps()) * mf).view(b, T, -1)
        return {"r": p_r, "h": p_h, "g": p_g, "mask": m, "tau": tau}
