"""The in-engine head's pooling logic, driven with stand-ins for vLLM's pooling metadata (no vLLM)."""

import threading

import pytest
import torch
import torch.nn.functional as F

from conftest import make_neox, make_texts, small_config
from rise.head import RiseHead
from rise.runtime.batching import pad_batch
from rise.runtime.hf import HFTrunk
from rise.runtime.vllm_head import REQUEST_KEY, RisePoolerCore, is_rise_request


class _State:
    def __init__(self):
        self.hidden_states_cache = []

    def clean(self):
        self.hidden_states_cache.clear()


class _Cursor:
    def __init__(self, scheduled, finished):
        self.num_scheduled_tokens_cpu = torch.tensor(scheduled)
        self._finished = list(finished)

    def get_finished_mask(self):
        return self._finished


class _Params:
    def __init__(self, loss_start=None):
        self.extra_kwargs = {REQUEST_KEY: {"loss_start": loss_start}}


class _Metadata:
    """What RisePoolerCore reads from vllm.v1.pool.metadata.PoolingMetadata."""

    def __init__(self, params, states, prompt_lens, token_ids, scheduled, finished):
        self.pooling_params = params
        self.pooling_states = states
        self.prompt_lens = torch.tensor(prompt_lens)
        self._token_ids = token_ids
        self._cursor = _Cursor(scheduled, finished)

    def get_pooling_cursor(self):
        return self._cursor

    def get_prompt_token_ids_cpu(self):
        return self._token_ids


@pytest.fixture(scope="module")
def setup(byte_tok):
    trunk = HFTrunk(make_neox())
    head = RiseHead.from_trunk(trunk, small_config())
    seqs = [byte_tok.encode(t)[:n] for t, n in zip(make_texts(4, seed=11), (30, 45, 60, 20))]
    ids, lens = pad_batch(seqs, byte_tok.pad_token_id)
    hid = trunk.hidden_states(ids, lens)
    rows = [hid[i, : len(s)] for i, s in enumerate(seqs)]
    toks = [torch.tensor(s) for s in seqs]
    return head, ids, lens, hid, rows, toks


def _two_steps(core, rows, toks, loss_starts=(None,) * 4):
    """Step 1 schedules prompts 0, 1 and the first half of 2; step 2 the rest of 2, and 3."""
    n = [len(t) for t in toks]
    cut = n[2] // 2
    states = [_State() for _ in range(4)]
    p = [_Params(ls) for ls in loss_starts]
    m1 = _Metadata(p[:3], states[:3], n[:3], toks[:3], [n[0], n[1], cut], [True, True, False])
    out1 = core(torch.cat([rows[0], rows[1], rows[2][:cut]]), m1)
    m2 = _Metadata(p[2:], states[2:], n[2:], toks[2:], [n[2] - cut, n[3]], [True, True])
    out2 = core(torch.cat([rows[2][cut:], rows[3]]), m2)
    assert out1[2] is None and all(v is not None for v in out1[:2] + out2)
    return torch.stack(out1[:2] + out2)


def test_pooler_matches_head_across_chunked_prefill(setup):
    head, ids, lens, hid, rows, toks = setup
    want = head.compute_from_hidden(hid, ids, lens)
    got = _two_steps(RisePoolerCore(head, batch_tokens=64), rows, toks)
    assert F.cosine_similarity(got, want, dim=1).min() > 0.99999


def test_pooler_passes_loss_start(setup):
    head, ids, lens, hid, rows, toks = setup
    ls = (3, None, 17, 0)
    want = head.compute_from_hidden(hid, ids, lens, torch.tensor([x or 0 for x in ls]))
    got = _two_steps(RisePoolerCore(head), rows, toks, ls)
    assert F.cosine_similarity(got, want, dim=1).min() > 0.99999
    plain = head.compute_from_hidden(hid, ids, lens)
    assert F.cosine_similarity(got[0], plain[0], dim=0) < 0.9999  # the mask changed something


def test_pooler_splits_work_across_ranks(setup):
    head, ids, lens, hid, rows, toks = setup
    want = _two_steps(RisePoolerCore(head), rows, toks)
    world, bufs, barrier = 3, [None] * 3, threading.Barrier(3)
    results, errors = [None] * world, []

    def all_gather(rank):
        def gather(t):
            bufs[rank] = t
            barrier.wait()
            out = torch.cat(bufs)
            barrier.wait()
            return out
        return gather

    def run(rank):
        try:
            results[rank] = _two_steps(RisePoolerCore(head, rank=rank, world=world, all_gather=all_gather(rank)),
                                       rows, toks)
        except BaseException as e:  # surface in the main thread
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for got in results:
        assert F.cosine_similarity(got, want, dim=1).min() > 0.99999


def test_pooler_fails_closed_on_missing_rows(setup):
    head, ids, lens, hid, rows, toks = setup
    n = len(toks[0])
    m = _Metadata([_Params()], [_State()], [n], toks[:1], [n - 1], [True])
    with pytest.raises(RuntimeError, match="misaligned"):
        RisePoolerCore(head)(rows[0][: n - 1], m)


def test_rise_requests_are_marked():
    class Plain:
        extra_kwargs = None

    assert is_rise_request(_Params()) and not is_rise_request(Plain())
