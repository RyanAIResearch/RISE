import numpy as np
import pytest
import torch

from reference import V3CountSketch
from rise.config import RiseConfig
from rise.sketch import CountSketch, RiseProjections


@pytest.mark.parametrize("in_dim,out_dim,seed", [(32, 8, 7), (257, 16, 9), (50304, 128, 44), (4096, 128, 42)])
def test_tables_match_research_code(in_dim, out_dim, seed):
    ours = CountSketch.from_seed(in_dim, out_dim, seed)
    ref = V3CountSketch(in_dim, out_dim, seed)
    assert torch.equal(ours.buckets, ref.buckets)
    assert torch.equal(ours.signs, ref.signs)


def test_dense_gemm_equals_scatter():
    cs = CountSketch.from_seed(96, 12, 3)
    x = torch.randn(17, 96)
    ref = V3CountSketch(96, 12, 3).project_dense_batch(x)
    torch.testing.assert_close(cs.dense(x), ref, rtol=1e-5, atol=1e-5)


def test_dense_scatter_fallback_for_huge_tables(monkeypatch):
    import rise.sketch as sk

    monkeypatch.setattr(sk, "_DENSE_MATRIX_MAX_ELEMS", 0)
    cs = CountSketch.from_seed(96, 12, 3)
    x = torch.randn(5, 96)
    torch.testing.assert_close(cs.dense(x), V3CountSketch(96, 12, 3).project_dense_batch(x))


def test_sparse_equals_dense_on_sparse_input():
    cs = CountSketch.from_seed(200, 16, 1)
    idx = torch.stack([torch.randperm(200)[:9] for _ in range(4)])
    val = torch.randn(4, 9)
    dense = torch.zeros(4, 200).scatter_(1, idx, val)
    torch.testing.assert_close(cs.sparse(idx, val), cs.dense(dense), rtol=1e-5, atol=1e-6)


def test_gh_linearity_sketched_unembedding():
    """CS_g(E^T r) == sum_v r_v CS_g(E_v): the identity behind the M_g shortcut."""
    V, D = 300, 24
    E = torch.randn(V, D)
    cs = CountSketch.from_seed(D, 8, 5)
    idx = torch.stack([torch.randperm(V)[:11] for _ in range(6)])
    r = torch.randn(6, 11)
    direct = cs.dense((r.unsqueeze(-1) * E[idx]).sum(dim=1))
    M = cs.dense(E)
    shortcut = (r.unsqueeze(-1) * M[idx]).sum(dim=1)
    torch.testing.assert_close(direct, shortcut, rtol=1e-4, atol=1e-5)


def test_projection_roundtrip(tmp_path):
    cfg = RiseConfig(Kr=16, Kh=8, Kg=4, seed=11)
    p = RiseProjections.from_seed(cfg, vocab_size=300, hidden_dim=24)
    path = tmp_path / "p.npz"
    p.save(str(path))
    q = RiseProjections.load(str(path))
    assert p.sha256() == q.sha256()
    for name in ("h", "r", "g"):
        a, b = getattr(p, name), getattr(q, name)
        assert torch.equal(a.buckets, b.buckets) and torch.equal(a.signs, b.signs) and a.out_dim == b.out_dim
    q.check_shapes(cfg, 300, 24)
    with pytest.raises(ValueError):
        q.check_shapes(cfg, 301, 24)


def test_rh_only_uses_boosted_width():
    cfg = RiseConfig(fusion_mode="rh", Kr=128, rh_only_Kr_boost=160)
    p = RiseProjections.from_seed(cfg, vocab_size=64, hidden_dim=8)
    assert p.r.out_dim == 160 and cfg.get_vector_dim() == 160 * cfg.Kh


def test_inner_products_preserved_on_average():
    """CountSketch is unbiased for inner products; averaged over seeds the estimate converges."""
    x, y = torch.randn(1, 512), torch.randn(1, 512)
    est = [float(CountSketch.from_seed(512, 64, s).dense(x) @ CountSketch.from_seed(512, 64, s).dense(y).T)
           for s in range(400)]
    assert abs(np.mean(est) - float(x @ y.T)) < 4 * np.std(est) / np.sqrt(len(est)) + 1e-3
