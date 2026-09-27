import json
import os

import numpy as np
import torch

from conftest import small_config
from reference import V3CountSketch
from rise.cli import main
from rise.index.store import IndexReader
from rise.sketch import RiseProjections

CFG_ARGS = ["--set", "Kr=16", "--set", "Kh=8", "--set", "Kg=8", "--set", "seq_len=96",
            "--set", "chunk_size=48", "--set", "chunk_overlap=8", "--set", "seed=7"]


def test_cli_end_to_end(tmp_path, corpus, byte_tok, neox):
    path, texts = corpus
    mdir = str(tmp_path / "model")
    neox.save_pretrained(mdir)
    byte_tok.save_pretrained(mdir)
    idx = str(tmp_path / "idx")
    main(["build", "--model", mdir, "--data", path, "--out", idx, "--device", "cpu", "--block-size", "10", *CFG_ARGS])
    assert IndexReader(idx).num_rows == len(texts)

    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps({"text": t}) for t in (texts[0], texts[4])) + "\n")
    qvec = str(tmp_path / "q.npy")
    main(["query", "--index", idx, "--model", mdir, "--queries", str(qpath), "--out", qvec, "--device", "cpu"])
    assert np.load(qvec).shape == (2, small_config().get_vector_dim())

    res, scores = str(tmp_path / "res.jsonl"), str(tmp_path / "scores.npy")
    main(["search", "--index", idx, "--queries", qvec, "--k", "5", "--out", res, "--scores-out", scores])
    rows = [json.loads(x) for x in open(res)]
    assert len(rows) == 2 and texts[rows[0]["indices"][0]] == texts[0]
    assert rows[0]["texts"] == [texts[j][:200] for j in rows[0]["indices"]]
    assert np.load(scores).shape == (2, len(texts))

    mean_scores = str(tmp_path / "mean.npy")
    main(["search", "--index", idx, "--queries", qvec, "--aggregate", "mean", "--scores-out", mean_scores])
    ms = np.load(mean_scores)
    np.testing.assert_allclose(ms, np.load(scores).mean(axis=0), rtol=1e-4, atol=1e-5)

    metrics = str(tmp_path / "m.json")
    main(["eval", "--index", idx, "--scores", scores, "--positive-label", "positive", "--k", "2,5", "--out", metrics])
    assert set(json.load(open(metrics))["top_k"]) == {"2", "5"}

    sel = str(tmp_path / "sel.jsonl")
    main(["select", "--index", idx, "--scores", mean_scores, "--data", path, "--k", "4", "--out", sel])
    picked = [json.loads(x) for x in open(sel)]
    assert len(picked) == 4 and picked[0]["text"] == texts[int(np.argmax(ms))]
    main(["info", "--index", idx, "--verify"])


def test_import_research_index(tmp_path):
    cfg = small_config()
    src = tmp_path / "research"
    src.mkdir()
    V, D, N = 257, 32, 7
    d = cfg.to_dict()
    d.update({"device": "cuda", "use_auto_anchor": False, "alpha_anchor": 2.0})  # extra research fields
    (src / "config.json").write_text(json.dumps(d))
    index = torch.randn(N, cfg.get_vector_dim()).half()
    torch.save(index, src / "index.pt")
    (src / "metadata.jsonl").write_text("\n".join(json.dumps({"idx": i, "label": "x"}) for i in range(N)) + "\n")
    tables = {"proj_h": V3CountSketch(D, cfg.Kh, cfg.seed), "proj_r": V3CountSketch(V, cfg.Kr, cfg.seed + 2),
              "proj_g": V3CountSketch(D, cfg.Kg, cfg.seed + 1)}
    torch.save({k: (t.buckets, t.signs) for k, t in tables.items()}, src / "projections.pt")

    out = str(tmp_path / "imported")
    main(["import-research", "--src", str(src), "--out", out, "--block-size", "3", "--model", "tiny"])
    r = IndexReader(out)
    np.testing.assert_array_equal(r.get_rows(range(N)), index.float().numpy())
    assert [m["idx"] for m in r.iter_metadata()] == list(range(N))
    assert RiseProjections.load(r.projections_path()).sha256() == RiseProjections.from_seed(cfg, V, D).sha256()
    assert os.path.exists(os.path.join(out, "manifest.json"))


def test_data_parallel_argv_rewrite():
    from rise.cli import _strip_option

    argv = ["-v", "build", "--model", "m", "--gpus", "0,1", "--rank=3", "--world-size", "4", "--out", "o"]
    out = _strip_option(_strip_option(_strip_option(argv, "--gpus"), "--rank"), "--world-size")
    assert out == ["-v", "build", "--model", "m", "--out", "o"]


def test_engine_arg_parsing():
    from rise.cli import _engine_kwargs

    assert _engine_kwargs(["allow_deprecated_quantization=true", "max_num_seqs=256", "swap_space=0.5",
                           "quantization=fp8"]) == {"allow_deprecated_quantization": True, "max_num_seqs": 256,
                                                   "swap_space": 0.5, "quantization": "fp8"}


def test_build_with_queries(tmp_path, corpus, byte_tok, neox):
    path, texts = corpus
    mdir = str(tmp_path / "model")
    neox.save_pretrained(mdir)
    byte_tok.save_pretrained(mdir)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps({"text": texts[3]}) + "\n")
    idx = str(tmp_path / "idx")
    main(["build", "--model", mdir, "--data", path, "--out", idx, "--device", "cpu", "--queries", str(qpath), *CFG_ARGS])
    q = np.load(os.path.join(idx, "queries.npy"))
    assert q.shape == (1, small_config().get_vector_dim())
    np.testing.assert_allclose(q[0], IndexReader(idx).get_rows([3])[0], atol=2e-3)
