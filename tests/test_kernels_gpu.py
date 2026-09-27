"""GPU-only checks for the fused kernels (skipped without CUDA / Triton)."""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_neox, make_texts, small_config
from reference import V3Reference
from rise.head import RiseHead, l2_normalize
from rise.kernels import sparse_sketch, topk
from rise.runtime.batching import pad_batch
from rise.runtime.hf import HFTrunk
from rise.sketch import CountSketch

pytestmark = pytest.mark.skipif(not sparse_sketch.available("cuda"), reason="needs CUDA + Triton")


def test_sparse_sketch_matches_dense_torch_path():
    g = torch.Generator().manual_seed(0)
    N, K, V, Kr, Kg = 777, 64, 5000, 48, 40
    idx = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(N)]).cuda()
    cut = torch.randint(1, K + 1, (N,), generator=g).cuda()
    gtp = torch.randint(0, K, (N,), generator=g).cuda()
    keep = (torch.arange(K, device="cuda")[None] < cut[:, None]) | (torch.arange(K, device="cuda")[None] == gtp[:, None])
    res = torch.randn(N, K, generator=g).cuda() * keep
    mask = (torch.rand(N, generator=g) > 0.2).float().cuda()
    cs = CountSketch.from_seed(V, Kr, 3).to("cuda")
    M_g = torch.randn(V, Kg, generator=g).cuda()

    p_r, p_g = sparse_sketch.sparse_sketch(res, idx, cut.int(), gtp.int(), mask, cs.buckets.int(), cs.signs, Kr, M_g, 1e-8)
    want_r = l2_normalize(cs.sparse(idx, res), 1e-8) * mask[:, None]
    want_g = l2_normalize((res.unsqueeze(-1) * M_g[idx]).sum(1), 1e-8) * mask[:, None]
    torch.testing.assert_close(p_r, want_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p_g, want_g, rtol=1e-5, atol=1e-6)


def _topk_rows():
    g = torch.Generator(device="cuda").manual_seed(0)
    rnd = lambda n, v, scale=3.0: torch.randn(n, v, generator=g, device="cuda") * scale
    yield "randn", rnd(64, 50304), [1, 8, 256, 1000]
    yield "randn, Llama-3 vocabulary", rnd(32, 128256), [256]
    yield "small vocabulary", rnd(64, 4097), [256]
    yield "heavy ties", (rnd(64, 50304) * 4).round() / 4, [256]
    x = rnd(32, 50304)
    x[::2] = 0.0  # flat rows: every chunk ties -> exact radix select
    yield "flat rows", x, [256]
    x = rnd(32, 50304)
    x[:, 5000:5600] = 20.0  # 600 tied maxima -> wide sort
    yield "600 tied maxima", x, [256]
    x = rnd(32, 50304)
    x[:, :40000] -= 100.0  # the top sits in one corner of the row
    yield "clustered top", x, [256]
    x = rnd(16, 50304)
    x[3, [5, 900, 40000]] = float("nan")
    x[9] = float("-inf")
    x[10, 77] = float("inf")
    yield "nan / inf", x, [256]
    x = torch.zeros(16, 50304, device="cuda")
    x[:, ::2] = -0.0
    x[:, 7] = 1.0
    yield "signed zeros", x, [256]
    yield "non-contiguous", rnd(32, 60000)[:, 3000:53000], [256]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_topk16_matches_torch_topk(dtype):
    for name, x, ks in _topk_rows():
        x = x.to(dtype)
        for k in ks:
            assert topk.available(x, k), name
            v, i = topk.topk16(x, k)
            v0, i0 = torch.topk(x, k, dim=1)
            nan = lambda t: torch.nan_to_num(t.float(), nan=1e9)
            assert torch.equal(nan(v), nan(v0)), (name, k)  # same values, same (descending) order
            assert torch.equal(nan(x.gather(1, i)), nan(v)), (name, k)
            assert torch.equal(i.sort(1).values, i0.sort(1).values), (name, k)  # ties: lowest indices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("cfg_kw", [{}, {"fusion_mode": "rh+gh+rg"}, {"adaptive_temperature": True},
                                    {"adaptive_topL": False, "fallback_topL": 16},
                                    {"gt_slot": "append", "max_topL_cap": 8, "min_topL": 4, "topL_cum_prob": 0.97}])
def test_head_kernel_matches_fallback(monkeypatch, byte_tok, dtype, cfg_kw):
    model = make_neox().to("cuda", dtype)
    trunk = HFTrunk(model)
    head = RiseHead.from_trunk(trunk, small_config(**cfg_kw))
    seqs = [byte_tok.encode(t)[:60] for t in make_texts(9, seed=6)]
    ids, lens = pad_batch(seqs, byte_tok.pad_token_id)
    hid = trunk.hidden_states(ids.cuda(), lens.cuda())
    fused = head.compute_from_hidden(hid, ids, lens)
    monkeypatch.setenv("RISE_DISABLE_TRITON", "1")
    plain = head.compute_from_hidden(hid, ids, lens)
    assert F.cosine_similarity(fused, plain, dim=1).min() > 0.99999


def test_cuda_head_matches_research_estimator(byte_tok):
    cfg = small_config()
    ref = V3Reference(make_neox(), byte_tok, cfg)  # CPU fp32 oracle
    trunk = HFTrunk(make_neox().cuda())
    head = RiseHead.from_trunk(trunk, cfg)
    seqs = [byte_tok.encode(t)[:60] for t in make_texts(7, seed=5)]
    ids, lens = pad_batch(seqs, byte_tok.pad_token_id)
    vecs = head.compute_from_hidden(trunk.hidden_states(ids.cuda(), lens.cuda()), ids, lens).cpu()
    for i, s in enumerate(seqs):
        assert F.cosine_similarity(vecs[i], ref.vector_from_ids(s), dim=0) > 0.9999, i


def test_cuda_pipeline_matches_cpu(tmp_path, corpus, byte_tok):
    from rise.index.store import IndexReader
    from rise.pipeline import BuildOptions, build_index
    from rise.text import TokenizerAdapter

    path, texts = corpus
    cfg = small_config()
    opts = dict(block_size=6, max_batch_tokens=256, max_batch_rows=8)
    build_index(HFTrunk(make_neox()), TokenizerAdapter(byte_tok), cfg, path, str(tmp_path / "cpu"), BuildOptions(**opts))
    build_index(HFTrunk(make_neox().cuda()), TokenizerAdapter(byte_tok), cfg, path, str(tmp_path / "gpu"),
                BuildOptions(**opts))
    a = torch.from_numpy(IndexReader(str(tmp_path / "cpu")).get_rows(range(len(texts))))
    b = torch.from_numpy(IndexReader(str(tmp_path / "gpu")).get_rows(range(len(texts))))
    assert F.cosine_similarity(a, b, dim=1).min() > 0.999


def test_cuda_sketch_rng_draws_the_december_tables():
    from rise.sketch import CountSketch

    cs = CountSketch.from_seed(5000, 64, 44, rng="cuda")
    g = torch.Generator(device="cuda")
    g.manual_seed(44)  # as the December 2025 research scripts did
    buckets = torch.randint(0, 64, (5000,), generator=g, device="cuda")
    signs = torch.where(torch.randint(0, 2, (5000,), generator=g, device="cuda").bool(),
                        torch.ones(5000, device="cuda"), -torch.ones(5000, device="cuda"))
    assert torch.equal(cs.buckets, buckets.cpu()) and torch.equal(cs.signs, signs.cpu())
    assert not torch.equal(cs.buckets, CountSketch.from_seed(5000, 64, 44).buckets)
