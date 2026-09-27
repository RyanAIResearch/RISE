"""The SGLang pooler that serves RISE requests, driven with a stand-in for SGLang's ForwardBatch (no SGLang)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import rise.runtime.sglang_head as sh
from conftest import make_neox, make_texts, small_config
from rise.head import RiseHead
from rise.runtime.batching import pad_batch
from rise.runtime.engine_head import RisePoolerCore
from rise.runtime.hf import HFTrunk


@pytest.fixture(scope="module")
def setup(byte_tok):
    trunk = HFTrunk(make_neox())
    head = RiseHead.from_trunk(trunk, small_config())
    seqs = [byte_tok.encode(t)[:n] for t, n in zip(make_texts(3, seed=5), (30, 45, 20))]
    ids, lens = pad_batch(seqs, byte_tok.pad_token_id)
    hid = trunk.hidden_states(ids, lens)
    rows = [hid[i, : len(s)] for i, s in enumerate(seqs)]
    return head, ids, lens, hid, rows, seqs


@pytest.fixture
def pooler(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(sh, "_pooler_output", lambda embeddings: embeddings)  # SGLang's EmbeddingPoolerOutput
    monkeypatch.setenv(sh.SPEC_DIR_ENV, str(tmp_path))
    with open(tmp_path / sh.OUTPUT_FILE, "wb") as f:  # the driver sizes it for the call's chunks
        f.truncate(8 * setup[0].dim * 4)
    p = sh.RiseSGLangPooler(inner=lambda hidden_states, batch: "inner")
    p._core, p._key = RisePoolerCore(setup[0], batch_tokens=64), "k"  # as _load() leaves it for spec "k"
    return p


def _batch(rids, seqs, prefix=None):
    return SimpleNamespace(rids=rids, extend_seq_lens_cpu=[len(s) for s in seqs],
                           extend_prefix_lens_cpu=prefix or [0] * len(seqs),
                           input_ids=torch.tensor([t for s in seqs for t in s]))


def test_signature_requests_match_the_head(setup, pooler, tmp_path):
    head, ids, lens, hid, rows, seqs = setup
    ls, slots = (3, None, 0), (5, 1, 6)  # rows of signatures.bin, in any order
    got = pooler(torch.cat(rows), _batch([sh.signature_rid("k", x, s) for x, s in zip(ls, slots)], seqs))
    assert got.flatten().tolist() == list(slots)  # the output is just the slot
    out = torch.from_numpy(sh.output_rows(str(tmp_path), head.dim, mode="r")[list(slots)].copy())
    want = head.compute_from_hidden(hid, ids, lens, torch.tensor([x or 0 for x in ls]))
    assert F.cosine_similarity(out, want, dim=1).min() > 0.99999
    with pytest.raises(RuntimeError, match="holds 8 signatures"):  # a slot beyond what the driver sized
        pooler(torch.cat(rows), _batch([sh.signature_rid("k", None, s) for s in (0, 1, 8)], seqs))


def test_hidden_requests_return_the_prompts_rows(setup, pooler):
    rows, seqs = setup[4], setup[5]
    flat = pooler(torch.cat(rows), _batch([sh.hidden_rid() for _ in seqs], seqs))
    back = sh.unflatten_hidden([f.tolist() for f in flat], [len(s) for s in seqs], rows[0].shape[1])
    assert all(torch.equal(b, r.float()) for b, r in zip(back, rows))
    with pytest.raises(RuntimeError, match="hidden values"):
        sh.unflatten_hidden([flat[0][:-1].tolist()], [len(seqs[0])], rows[0].shape[1])


def test_other_requests_go_to_the_model_pooler(setup, pooler):
    rows, seqs = setup[4], setup[5]
    assert pooler(torch.cat(rows), _batch(["plain-1", "plain-2", "plain-3"], seqs)) == "inner"


def test_fails_closed_on_mixed_or_partial_steps(setup, pooler):
    rows, seqs = setup[4], setup[5]
    with pytest.raises(RuntimeError, match="mixed"):
        pooler(torch.cat(rows), _batch([sh.hidden_rid(), sh.signature_rid("k", None, 0), "plain"], seqs))
    with pytest.raises(RuntimeError, match="whole prompts"):  # a prefix-cache hit computes only the suffix
        pooler(torch.cat(rows), _batch([sh.hidden_rid() for _ in seqs], seqs, prefix=[0, 4, 0]))


def test_request_ids_round_trip():
    rid = sh.signature_rid("abc", 7, 3)
    assert sh.request_kind(rid) == sh.SIGNATURE and sh.parse_signature_rid(rid) == ("abc", 7, 3)
    assert sh.parse_signature_rid(sh.signature_rid("abc", None, 0)) == ("abc", None, 0)
    assert sh.request_kind(sh.hidden_rid()) == sh.HIDDEN
    assert sh.request_kind("rise2:x") is None and sh.request_kind(None) is None
