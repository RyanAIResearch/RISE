import numpy as np
import pytest

from rise.index.store import IndexReader, IndexWriter
from rise.search import mean_query, score_all, topk_search


def _write(root, X, block_size, attrs=None):
    w = IndexWriter(str(root), num_rows=len(X), dim=X.shape[1], block_size=block_size, attrs=attrs or {"t": 1})
    for b in w.pending_blocks():
        s, e = w.block_range(b)
        w.write_block(b, X[s:e], [{"idx": i} for i in range(s, e)])
    return w.finalize()


def _rand(n, d, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float16)


def test_roundtrip_blocks_and_metadata(tmp_path):
    X = _rand(23, 6)
    m = _write(tmp_path, X, block_size=10)
    assert [s["rows"] for s in m["shards"]] == [10, 10, 3]
    r = IndexReader(str(tmp_path))
    got = np.concatenate([blk for _, blk in r.iter_blocks(rows_per_step=4)])
    np.testing.assert_array_equal(got, X)
    assert [x["idx"] for x in r.iter_metadata()] == list(range(23))
    np.testing.assert_array_equal(r.get_rows([22, 0, 10]), X[[22, 0, 10]].astype(np.float32))
    r.verify()


def test_resume_skips_done_blocks_and_rejects_changed_settings(tmp_path):
    X = _rand(12, 4)
    w = IndexWriter(str(tmp_path), num_rows=12, dim=4, block_size=5, attrs={"cfg": 1})
    w.write_block(1, X[5:10], [{}] * 5)
    w2 = IndexWriter(str(tmp_path), num_rows=12, dim=4, block_size=5, attrs={"cfg": 1})
    assert w2.pending_blocks() == [0, 2]
    assert w2.pending_blocks(rank=1, world_size=2) == []
    with pytest.raises(RuntimeError, match="not built yet"):
        w2.finalize()
    with pytest.raises(ValueError, match="different settings"):
        IndexWriter(str(tmp_path), num_rows=12, dim=4, block_size=5, attrs={"cfg": 2})


def test_verify_detects_corruption(tmp_path):
    _write(tmp_path, _rand(8, 4), block_size=8)
    p = tmp_path / "shards" / "vectors-00000.npy"
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        IndexReader(str(tmp_path)).verify()


def test_unfinalized_index_is_not_readable(tmp_path):
    IndexWriter(str(tmp_path), num_rows=3, dim=2, block_size=2, attrs={})
    with pytest.raises(FileNotFoundError, match="not finalized"):
        IndexReader(str(tmp_path))


@pytest.mark.parametrize("metric", ["dot", "cosine"])
@pytest.mark.parametrize("rows_per_step,k", [(7, 5), (64, 5), (3, 40), (5, 200)])
def test_topk_matches_brute_force(tmp_path, metric, rows_per_step, k):
    X = _rand(57, 16, seed=1) * np.float16(1.5)
    _write(tmp_path, X, block_size=20)
    r = IndexReader(str(tmp_path))
    Q = np.random.default_rng(2).standard_normal((4, 16)).astype(np.float32)
    s, i = topk_search(r, Q, k, metric=metric, rows_per_step=rows_per_step)
    Xf = X.astype(np.float32)
    if metric == "cosine":
        Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    full = Q @ Xf.T
    kk = min(k, 57)
    np.testing.assert_allclose(s, -np.sort(-full, axis=1)[:, :kk], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.take_along_axis(full, i, axis=1), s, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_score_all_and_mean_aggregation(tmp_path, metric):
    X = _rand(31, 8, seed=5) * np.float16(1.7)  # un-normalized rows make cosine != dot
    _write(tmp_path, X, block_size=9)
    r = IndexReader(str(tmp_path))
    Q = np.random.default_rng(6).standard_normal((5, 8)).astype(np.float32)
    full = score_all(r, Q, metric=metric, rows_per_step=4)
    Xf = X.astype(np.float32)
    if metric == "cosine":
        Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    else:
        Qn = Q
    np.testing.assert_allclose(full, Qn @ Xf.T, rtol=1e-5, atol=1e-5)
    agg = score_all(r, mean_query(Q, metric)[None], metric=metric, normalize_queries=False)[0]
    np.testing.assert_allclose(agg, full.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_dynamic_claims(tmp_path):
    from rise.index.store import clear_claims

    X = _rand(12, 4)
    w = IndexWriter(str(tmp_path), num_rows=12, dim=4, block_size=5, attrs={})
    assert w.claim(0) and not w.claim(0)          # second worker cannot take a claimed block
    w.write_block(0, X[0:5], [{}] * 5)
    assert not w.claim(0)                          # done blocks are never re-claimed
    assert w.claim(1) and clear_claims(str(tmp_path)) == 1 and w.claim(1)
