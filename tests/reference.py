"""Reference oracle: an unoptimized port of the RISE research code (v3).

One sequence at a time; full-vocabulary entropy with the eps clamp; CountSketch
by scatter_add; GH via the explicit [T, K, D] gather of unembedding rows (not the
sketched-unembedding shortcut); hidden states captured from the final-norm
module with a forward hook. Chunk vectors are averaged like the published
builder (v5): each rounded to fp16, averaged in fp32, re-normalized.

Tests compare the engine against this file; it must stay a literal port.
"""

from __future__ import annotations

import math

import torch


class V3CountSketch:
    def __init__(self, in_dim: int, out_dim: int, seed: int):
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.buckets = torch.randint(0, out_dim, (in_dim,), generator=g, dtype=torch.long)
        signs = torch.randint(0, 2, (in_dim,), generator=g, dtype=torch.long)
        self.signs = torch.where(signs.bool(), torch.ones(in_dim), -torch.ones(in_dim)).to(torch.float32)

    def project_dense_batch(self, V: torch.Tensor) -> torch.Tensor:
        B, _ = V.shape
        V = V.float() * self.signs.unsqueeze(0)
        out = torch.zeros((B, self.out_dim), dtype=torch.float32)
        out.scatter_add_(1, self.buckets.unsqueeze(0).expand(B, -1), V)
        return out

    def project_sparse_batch(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, _ = indices.shape
        indices = indices.clamp(0, self.in_dim - 1).long()
        out = torch.zeros((B, self.out_dim), dtype=torch.float32)
        out.scatter_add_(1, self.buckets[indices], values.float() * self.signs[indices].float())
        return out


def safe_l2(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)


def compute_entropy(logits: torch.Tensor, tau: float, eps: float) -> torch.Tensor:
    p = torch.softmax(logits / tau, dim=-1)
    return -(p * torch.log(p.clamp(min=eps))).sum(dim=-1)


def find_temperature_for_entropy(logits, target_H, tau_min, tau_max, steps=15, eps=1e-8, tau_fallback=0.8):
    if logits.numel() == 0 or not torch.isfinite(logits).all():
        return tau_fallback
    if target_H >= math.log(logits.size(-1)):
        return tau_max
    lo, hi = tau_min, tau_max
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        H = compute_entropy(logits, mid, eps).mean().item()
        if abs(H - target_H) < 0.01:
            return mid
        if H < target_H:
            lo = mid
        else:
            hi = mid
    out = 0.5 * (lo + hi)
    return out if math.isfinite(out) else tau_fallback


def final_norm(model):
    for path in ("gpt_neox.final_layer_norm", "model.norm", "transformer.ln_f"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("no final norm module found")


class V3Reference:
    def __init__(self, model, tokenizer, cfg):
        self.model, self.tok, self.c = model.eval(), tokenizer, cfg
        V, d = int(model.config.vocab_size), int(model.config.hidden_size)
        self.proj_h = V3CountSketch(d, cfg.Kh, cfg.seed)
        self.proj_r = V3CountSketch(V, cfg.get_effective_Kr(), cfg.seed + 2)
        self.proj_g = V3CountSketch(d, cfg.Kg, cfg.seed + 1)
        self._h = None
        final_norm(model).register_forward_hook(self._grab)
        self.last_tau = None

    def _grab(self, module, inputs, output):
        self._h = output[0] if isinstance(output, tuple) else output

    def E(self) -> torch.Tensor:
        if self.c.embedding_type == "input":
            return self.model.get_input_embeddings().weight
        return self.model.get_output_embeddings().weight

    @torch.inference_mode()
    def vector_from_ids(self, ids, loss_start=None) -> torch.Tensor:
        """Port of the research vectorizer's _compute_from_ids_batched; returns fp32 (pre-.half())."""
        c = self.c
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        logits = out.logits[:, :-1, :].squeeze(0).float()
        hidden = self._h[:, :-1, :].squeeze(0)
        # sanity: the hooked tensor is exactly what the LM head consumed
        assert torch.allclose(self.model.get_output_embeddings()(self._h).float(), out.logits.float(),
                              rtol=1e-4, atol=1e-4)
        next_ids = input_ids[:, 1:].squeeze(0)
        T, V = logits.shape

        token_mask = None
        if loss_start is not None:
            token_mask = (torch.arange(T) + 1) >= int(loss_start)

        tau = (find_temperature_for_entropy(logits, c.target_entropy, c.tau_min, c.tau_max, c.tau_search_steps,
                                            c.eps, c.tau_fallback) if c.adaptive_temperature else c.tau_fallback)
        self.last_tau = tau
        ls = logits / float(tau)
        K = min(c.max_topL_cap, V) if c.adaptive_topL else min(c.fallback_topL, V)
        K = max(K, c.min_topL)
        tv, ti = torch.topk(ls, k=K, dim=-1)
        need = ~(ti == next_ids.unsqueeze(1)).any(dim=1)
        if need.any():
            ti[need, -1] = next_ids[need]
            tv[need, -1] = ls[need].gather(1, next_ids[need].unsqueeze(1)).squeeze(1)
        ex = torch.exp(tv - tv.max(dim=-1, keepdim=True).values)
        probs = ex / (ex.sum(dim=-1, keepdim=True) + c.eps)
        if c.adaptive_topL:
            hit = torch.cumsum(probs, dim=-1) >= c.topL_cum_prob
            first = hit.float().argmax(dim=-1)
            cutoff = first + 1
            cutoff[~hit.any(dim=-1)] = K
            cutoff = torch.clamp(cutoff, min=c.min_topL, max=K)
            keep = torch.arange(K).unsqueeze(0) < cutoff.unsqueeze(1)
            keep = keep | (ti == next_ids.unsqueeze(1))
            probs = probs * keep.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=c.eps)
        res = probs.clone()
        res[ti == next_ids.unsqueeze(1)] -= 1.0

        chs = c.get_channels()
        p_h = safe_l2(self.proj_h.project_dense_batch(hidden), c.eps) if any("h" in x for x in chs) else None
        p_r = safe_l2(self.proj_r.project_sparse_batch(ti, res), c.eps) if any("r" in x for x in chs) else None
        p_g = None
        if any("g" in x for x in chs):
            E = self.E()
            W_topk = E.index_select(0, ti.reshape(-1)).view(T, K, -1).float()
            g = (probs.unsqueeze(-1) * W_topk).sum(dim=1) - E.index_select(0, next_ids).float()
            p_g = safe_l2(self.proj_g.project_dense_batch(g), c.eps)

        parts = []
        if c.has_channel("rh"):
            parts.append(c.lambda_rh * torch.einsum("ti,tj->tij", p_r, p_h).reshape(T, -1))
        if c.has_channel("gh"):
            parts.append(c.lambda_gh * torch.einsum("ti,tj->tij", p_g, p_h).reshape(T, -1))
        if c.has_channel("rg"):
            parts.append(c.lambda_rg * torch.einsum("ti,tj->tij", p_r, p_g).reshape(T, -1))
        M = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if token_mask is not None:
            M = M[token_mask]
        v = M.sum(dim=0)
        return safe_l2(v, c.eps) if c.normalize_sample else v

    def chunks(self, text: str):
        c = self.c
        toks = self.tok.encode(text, add_special_tokens=True)
        if not c.chunk_long_sequences or len(toks) <= c.chunk_size:
            return [toks[: c.seq_len]] if len(toks) >= c.min_seq_len else []
        step = max(1, c.chunk_size - c.chunk_overlap)
        return [toks[i:i + c.chunk_size] for i in range(0, len(toks), step)
                if len(toks[i:i + c.chunk_size]) >= c.min_seq_len]

    def sample_vector(self, text: str) -> torch.Tensor:
        """Index-row vector: chunk vectors rounded to fp16, fp32 mean, normalized, fp16."""
        vecs = [self.vector_from_ids(ch).half().float() for ch in self.chunks(text)]
        dim = self.c.get_vector_dim()
        if not vecs:
            return torch.zeros(dim)
        v = torch.stack(vecs).mean(dim=0)
        return (safe_l2(v, self.c.eps) if self.c.normalize_sample else v).half().float()

    def prompted_vector(self, text: str, prompt: str) -> torch.Tensor:
        """Port of compute_vector(text, prompt_text): one truncated chunk, prompt-masked."""
        c = self.c
        ids = self.tok(text, truncation=True, max_length=c.seq_len)["input_ids"]
        p = len(self.tok(prompt, truncation=True, max_length=c.seq_len)["input_ids"])
        ls = max(1, min(p, len(ids) - 1))
        return self.vector_from_ids(ids, loss_start=ls).half().float()
