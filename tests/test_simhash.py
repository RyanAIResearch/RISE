"""SimHash compression: the transform, the codes, and compressed indexes end to end (no model needed)."""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import rise.index.store as store
from rise.cli import main
from rise.index import IndexReader, IndexWriter, SimHash, compress_index
from rise.index.simhash import fwht
from rise.search import mean_query, score_all, topk_search


def _hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _index(tmp_path, n=200, dim=96, block=64):
    root = str(tmp_path / "idx")
    w = IndexWriter(root, num_rows=n, dim=dim, block_size=block, attrs={"config": {"test": True}})
    x = F.normalize(torch.randn(n, dim, generator=torch.Generator().manual_seed(0)), dim=1).numpy()
    for b in range(w.num_blocks):
        s, e = w.block_range(b)
        w.write_block(b, x[s:e], [{"idx": i, "text": f"row {i}"} for i in range(s, e)])
    with open(os.path.join(root, store.PROJECTIONS_FILE), "wb") as f:
        f.write(b"tables")
    w.finalize()
    return root, x.astype(np.float16).astype(np.float32)


def test_fwht_is_the_hadamard_transform():
    x = torch.randn(3, 16)
    assert torch.allclose(fwht(x), x @ _hadamard(16).T, atol=1e-5)
    with pytest.raises(ValueError, match="power-of-two"):
        fwht(torch.randn(2, 12))


def test_codes_pack_like_numpy_and_unpack_back(tmp_path):
    sh = SimHash.draw(dim=100, bits=64, seed=1)
    x = torch.randn(5, 100)
    codes = sh.encode(x)
    assert codes.dtype == torch.uint8
    assert np.array_equal(codes.numpy(), np.packbits((sh.project(x) > 0).numpy(), axis=1))
    assert torch.equal(SimHash.unpack(codes), (sh.project(x) > 0).float() * 2 - 1)
    sh.save(str(tmp_path / "sh.npz"))
    again = SimHash.load(str(tmp_path / "sh.npz"))
    assert torch.equal(again.encode(x), codes)
    with pytest.raises(ValueError, match="multiple of 8"):
        SimHash.draw(dim=100, bits=12)


def test_projection_is_orthogonal_and_scores_estimate_inner_products():
    full = SimHash.draw(dim=64, bits=64, seed=3)  # every coordinate picked: a rotation
    x = torch.randn(4, 64)
    assert torch.allclose(full.project(x).norm(dim=1), x.norm(dim=1), rtol=1e-5)
    sh = SimHash.draw(dim=300, bits=8192, seed=2)
    g = torch.Generator().manual_seed(0)
    x = F.normalize(torch.randn(50, 300, generator=g), dim=1)
    q = F.normalize(x[:10] + 0.7 * F.normalize(torch.randn(10, 300, generator=g), dim=1), dim=1)
    est = sh.queries(q) @ SimHash.unpack(sh.encode(x)).T
    assert (est - q @ x.T).abs().max() < 0.07


def test_compressed_index_searches_like_the_original(tmp_path):
    root, x = _index(tmp_path)
    out = str(tmp_path / "idx-simhash")
    m = compress_index(root, out, bits=2048)  # dim 96 -> n 128: 16 rounds of signs
    r = IndexReader(out)
    assert m["format_version"] == store.SIMHASH_FORMAT_VERSION and r.codec.bits == 2048 and r.row_width == 256
    g = torch.Generator().manual_seed(1)
    q = F.normalize(torch.from_numpy(x[:20]) + 0.3 * torch.randn(20, 96, generator=g) / 96 ** 0.5, dim=1).numpy()
    full, coded = score_all(IndexReader(root), q, metric="cosine"), score_all(r, q, metric="cosine")
    assert np.abs(coded - full).max() < 0.1
    _, top = topk_search(r, x[:20], k=1, metric="cosine")
    assert (top[:, 0] == np.arange(20)).all()  # each row finds itself
    agg = score_all(r, mean_query(q, "cosine"), metric="cosine", normalize_queries=False)
    assert np.allclose(agg[0], coded.mean(axis=0), atol=1e-4)  # linear in the query: valuation still works
    assert [row["idx"] for row in r.iter_metadata()] == list(range(200))


def test_compressed_index_fails_closed(tmp_path, monkeypatch):
    root, _ = _index(tmp_path)
    out = str(tmp_path / "c")
    compress_index(root, out, bits=64)
    with pytest.raises(ValueError, match="SimHash codes"):
        IndexReader(out).get_rows([0])
    with pytest.raises(FileExistsError):
        compress_index(root, out, bits=64)
    with pytest.raises(ValueError, match="already compressed"):
        compress_index(out, str(tmp_path / "c2"))
    for bits in (None, 64):  # neither a float16 build nor a compressed build may write into it
        with pytest.raises(ValueError, match="not built there"):
            IndexWriter(out, num_rows=1, dim=96, block_size=1, attrs={}, codec_bits=bits)
    monkeypatch.setattr(store, "FORMAT_VERSION", 1)  # a reader from before compression existed
    with pytest.raises(ValueError, match="format v2"):
        IndexReader(out)
    IndexReader(root)  # float16 indexes stay v1
    monkeypatch.undo()
    with open(os.path.join(out, "simhash.npz"), "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="sha256"):
        IndexReader(out)


def test_cli_compress_info_search(tmp_path, capsys):
    root, x = _index(tmp_path)
    out, qpath, res = str(tmp_path / "c"), str(tmp_path / "q.npy"), str(tmp_path / "top.jsonl")
    np.save(qpath, x[[3, 150]])
    main(["compress", "--index", root, "--out", out, "--bits", "1024", "--device", "cpu"])
    main(["info", "--index", out])
    assert "SimHash, 1024 bits" in capsys.readouterr().out
    main(["search", "--index", out, "--queries", qpath, "--k", "3", "--out", res, "--device", "cpu"])
    rows = [json.loads(line) for line in open(res)]
    assert [r["indices"][0] for r in rows] == [3, 150] and rows[1]["texts"][0] == "row 150"


def test_build_with_compress_bits_equals_build_then_compress(tmp_path, byte_tok):
    from conftest import make_neox, make_texts, small_config
    from rise.pipeline import BuildOptions, build_index
    from rise.runtime.hf import HFTrunk
    from rise.text import TokenizerAdapter

    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in make_texts(7, seed=6)))
    tok, trunk, cfg = TokenizerAdapter(byte_tok), HFTrunk(make_neox()), small_config()
    plain, direct, later = (str(tmp_path / d) for d in ("plain", "direct", "later"))
    build_index(trunk, tok, cfg, str(path), plain, BuildOptions(block_size=3))
    compress_index(plain, later, bits=128)
    build_index(trunk, tok, cfg, str(path), direct, BuildOptions(block_size=3, compress_bits=128))
    a, b = IndexReader(direct), IndexReader(later)
    assert a.codec.bits == 128 and a.manifest["format_version"] == store.SIMHASH_FORMAT_VERSION
    assert all(np.array_equal(a.shard_array(i), b.shard_array(i)) for i in range(len(a.shards)))
    assert [m["text"] for m in a.iter_metadata()] == [m["text"] for m in b.iter_metadata()]
    with pytest.raises(ValueError, match="different settings"):  # resuming with other bits is refused
        build_index(trunk, tok, cfg, str(path), direct, BuildOptions(block_size=3, compress_bits=64))


def test_cli_compressed_build_then_query_search(tmp_path, corpus, byte_tok, neox, capsys):
    from conftest import small_config

    path, texts = corpus
    mdir = str(tmp_path / "model")
    neox.save_pretrained(mdir)
    byte_tok.save_pretrained(mdir)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("".join(json.dumps({"text": texts[i]}) + "\n" for i in (3, 7)))
    idx, qvec, res = str(tmp_path / "idx"), str(tmp_path / "q.npy"), str(tmp_path / "top.jsonl")
    cfg = ["--set", "Kr=16", "--set", "Kh=8", "--set", "Kg=8", "--set", "seq_len=96", "--set", "chunk_size=48",
           "--set", "chunk_overlap=8", "--set", "seed=7"]
    main(["build", "--model", mdir, "--data", path, "--out", idx, "--device", "cpu", "--block-size", "10",
          "--compress-bits", "512", "--queries", str(qpath), *cfg])
    assert IndexReader(idx).codec.bits == 512
    built = np.load(os.path.join(idx, "queries.npy"))
    assert built.shape == (2, small_config().get_vector_dim())  # queries stay float
    main(["query", "--index", idx, "--model", mdir, "--queries", str(qpath), "--out", qvec, "--device", "cpu"])
    np.testing.assert_allclose(np.load(qvec), built, atol=1e-5)
    main(["search", "--index", idx, "--queries", qvec, "--k", "3", "--out", res, "--device", "cpu"])
    assert [json.loads(line)["indices"][0] for line in open(res)] == [3, 7]
    main(["info", "--index", idx, "--verify"])
    assert "SimHash, 512 bits" in capsys.readouterr().out
