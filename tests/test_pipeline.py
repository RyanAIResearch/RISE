import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import small_config
from reference import V3Reference
from rise.index.store import IndexReader
from rise.pipeline import BuildOptions, build_index, build_query_vectors
from rise.runtime.hf import HFTrunk
from rise.search import topk_search
from rise.text import TokenizerAdapter


def _opts(**kw):
    base = dict(block_size=6, max_batch_tokens=256, max_batch_rows=8)
    base.update(kw)
    return BuildOptions(**base)


def _cos(a, b):
    return F.cosine_similarity(torch.as_tensor(a).float(), torch.as_tensor(b).float(), dim=0).item()


def test_index_rows_match_research_estimator(tmp_path, corpus, byte_tok, neox):
    path, texts = corpus
    cfg = small_config()
    out = str(tmp_path / "idx")
    m = build_index(HFTrunk(neox), TokenizerAdapter(byte_tok), cfg, path, out, _opts())
    assert m["num_rows"] == len(texts) and len(m["shards"]) == 4
    r = IndexReader(out)
    X = r.get_rows(range(len(texts)))
    ref = V3Reference(neox, byte_tok, cfg)
    meta = r.metadata()
    multi = 0
    for i, t in enumerate(texts):
        n_chunks = len(ref.chunks(t))
        multi += n_chunks > 1
        assert meta[i]["idx"] == i and meta[i]["num_chunks"] == n_chunks
        assert _cos(X[i], ref.sample_vector(t)) > 0.9999, i
    assert multi >= 3, "corpus should exercise sliding-window chunking"
    assert meta[0]["label"] == "positive" and meta[1]["label"] == "negative"


def test_workers_and_resume_reproduce_the_same_index(tmp_path, corpus, byte_tok, neox):
    path, _ = corpus
    cfg = small_config()
    trunk, tok = HFTrunk(neox), TokenizerAdapter(byte_tok)
    single = str(tmp_path / "single")
    build_index(trunk, tok, cfg, path, single, _opts())

    split = str(tmp_path / "split")
    assert build_index(trunk, tok, cfg, path, split, _opts(rank=0, world_size=2)) is None
    assert build_index(trunk, tok, cfg, path, split, _opts(rank=1, world_size=2)) is not None
    a = IndexReader(single).get_rows(range(23))
    np.testing.assert_array_equal(IndexReader(split).get_rows(range(23)), a)

    # lose one block, resume: only that block is rebuilt, result unchanged
    shard0 = os.path.join(single, "shards", "vectors-00000.npy")
    mtime0 = os.path.getmtime(shard0)
    os.remove(os.path.join(single, "shards", "block-00002.json"))
    os.remove(os.path.join(single, "manifest.json"))
    build_index(trunk, tok, cfg, path, single, _opts())
    assert os.path.getmtime(shard0) == mtime0
    np.testing.assert_array_equal(IndexReader(single).get_rows(range(23)), a)

    with pytest.raises(ValueError, match="different settings|does not match the index"):
        build_index(trunk, tok, small_config(Kh=4), path, single, _opts())
    with pytest.raises(ValueError, match="different settings"):
        build_index(trunk, tok, small_config(lambda_gh=0.5), path, single, _opts())


def test_queries_self_retrieval_and_prompt_masking(tmp_path, corpus, byte_tok, neox):
    path, texts = corpus
    cfg = small_config()
    out = str(tmp_path / "idx")
    trunk, tok = HFTrunk(neox), TokenizerAdapter(byte_tok)
    build_index(trunk, tok, cfg, path, out, _opts())
    r = IndexReader(out)
    prompt = "the model reads"
    queries = [{"text": texts[4]}, {"text": texts[9]},
               {"prompt_text": prompt, "text": prompt + " data and every token shifts"}]
    Q = build_query_vectors(trunk, tok, r, queries)
    assert Q.shape == (3, cfg.get_vector_dim())
    s, i = topk_search(r, Q[:2], 3)
    for q, want in ((0, 4), (1, 9)):
        assert texts[i[q, 0]] == texts[want] and s[q, 0] == pytest.approx(1.0, abs=2e-3)
    ref = V3Reference(neox, byte_tok, cfg)
    assert _cos(Q[2], ref.prompted_vector(queries[2]["text"], prompt)) > 0.9999


def test_stale_projection_tables_are_refused(tmp_path, corpus, byte_tok, neox):
    from rise.sketch import RiseProjections

    path, _ = corpus
    out = tmp_path / "idx"
    out.mkdir()
    trunk = HFTrunk(neox)
    RiseProjections.from_seed(small_config(seed=999), trunk.vocab_size, trunk.hidden_dim).save(
        str(out / "projections.npz"))
    with pytest.raises(ValueError, match="does not match seed"):
        build_index(trunk, TokenizerAdapter(byte_tok), small_config(), path, str(out), _opts())


def test_query_with_a_different_model_is_refused(tmp_path, corpus, byte_tok, neox, llama):
    path, _ = corpus
    out = str(tmp_path / "idx")
    build_index(HFTrunk(neox), TokenizerAdapter(byte_tok), small_config(), path, out, _opts())
    with pytest.raises(ValueError, match="does not match the index"):
        build_query_vectors(HFTrunk(llama), TokenizerAdapter(byte_tok), IndexReader(out), [{"text": "hi there"}])
