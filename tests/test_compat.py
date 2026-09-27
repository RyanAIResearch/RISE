"""Research-code compatibility: ``gt_slot="append"`` and ``sample_format="alpaca"`` against a port of
the December 2025 research script (per-position loop), and the config plumbing."""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_texts, small_config
from rise.config import RiseConfig
from rise.head import RiseHead
from rise.runtime.batching import pad_batch
from rise.runtime.hf import HFTrunk
from rise.sketch import RiseProjections
from rise.text import format_sample


def _dense(cs, v):
    out = torch.zeros(cs.out_dim)
    out.index_add_(0, cs.buckets, v.float() * cs.signs)
    return out


def _sparse(cs, idx, vals):
    out = torch.zeros(cs.out_dim)
    out.index_add_(0, cs.buckets[idx], vals.float() * cs.signs[idx])
    return out


def _l2(x, eps):
    return x / torch.sqrt((x * x).sum().clamp(min=eps))


def december_vector(model, ids, cfg, proj, E):
    """The December 2025 script's _compute_vector_from_ids, fixed temperature, one chunk."""
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    logits, hidden = out.logits[0, :-1].float(), out.hidden_states[-1][0, :-1].float()
    V = logits.shape[-1]
    contribs, outside = [], 0
    for j, y in enumerate(ids[1:].tolist()):
        l = logits[j] / cfg.tau_fallback
        if cfg.adaptive_topL:  # dynamic_topk_selection
            k = max(cfg.min_topL, min(cfg.max_topL_cap, V))
            values, indices = torch.topk(l, k=k)
            hit = torch.cumsum(torch.softmax(values - values.max(), dim=0), dim=0) >= cfg.topL_cum_prob
            cutoff = int(hit.nonzero()[0, 0]) + 1 if hit.any() else k
            cutoff = min(k, max(min(cfg.min_topL, V), cutoff))
            outside += int(not (indices == y).any())
            indices, values = indices[:cutoff], values[:cutoff]
        else:
            values, indices = torch.topk(l, k=min(cfg.fallback_topL, V))
        if not (indices == y).any():  # ground truth appended
            indices = torch.cat([indices, torch.tensor([y])])
            values = torch.cat([values, l[y].unsqueeze(0)])
        ex = torch.exp(values - values.max())
        probs = ex / (ex.sum() + cfg.eps)
        r = probs.clone()
        r[indices == y] -= 1.0
        p_h = _l2(_dense(proj.h, hidden[j]), cfg.eps)
        p_r = _l2(_sparse(proj.r, indices, r), cfg.eps)
        p_g = _l2(_dense(proj.g, (probs.unsqueeze(1) * E[indices].float()).sum(0) - E[y].float()), cfg.eps)
        contribs.append(torch.cat([cfg.lambda_rh * torch.outer(p_r, p_h).flatten(),
                                   cfg.lambda_gh * torch.outer(p_g, p_h).flatten()]))
    return _l2(torch.stack(contribs).sum(0), cfg.eps), outside


@pytest.mark.parametrize("cfg_kw,other_kw,tol", [
    ({}, {"gt_slot": "replace"}, 1e-5),
    ({"embedding_type": "input"}, {"gt_slot": "replace"}, 1e-5),  # the December script's GH used the input embedding
    ({"tau_fallback": 0.3}, {"gt_slot": "replace"}, 1e-5),
    ({"adaptive_topL": False, "fallback_topL": 6}, {"gt_slot": "replace"}, 1e-5),
    # near-certain predictions: residuals below the 1e-4 norm floor of the December normalization.
    # p_gt - 1 is ill-conditioned there, so the two summation orders differ by ~3e-5; "norm" by ~8e-2.
    ({"tau_fallback": 0.01}, {"l2_eps": "norm"}, 1e-4),
])
def test_append_matches_december_script(neox, byte_tok, cfg_kw, other_kw, tol):
    base = dict(gt_slot="append", l2_eps="squared", max_topL_cap=8, min_topL=4, topL_cum_prob=0.97,
                lambda_rh=0.7, lambda_gh=1.0)
    cfg = small_config(**{**base, **cfg_kw})
    trunk = HFTrunk(neox)
    proj = RiseProjections.from_seed(cfg, trunk.vocab_size, trunk.hidden_dim)
    head = RiseHead.from_trunk(trunk, cfg, projections=proj)
    replace = RiseHead.from_trunk(trunk, small_config(**{**base, **cfg_kw, **other_kw}), projections=proj)
    seqs = [byte_tok.encode(t)[:40] for t in make_texts(8, seed=9)]  # one chunk each
    seqs = [s for s in seqs if len(s) >= 2]
    ids, lens = pad_batch(seqs, byte_tok.pad_token_id)
    hid = trunk.hidden_states(ids, lens)
    got, other = head.compute_from_hidden(hid, ids, lens), replace.compute_from_hidden(hid, ids, lens)
    E = neox.get_input_embeddings().weight if cfg.embedding_type == "input" else neox.get_output_embeddings().weight
    outside, want = 0, []
    with torch.no_grad():
        for s in seqs:
            v, n_out = december_vector(neox, torch.tensor(s), cfg, proj, E)
            want.append(v)
            outside += n_out
    want = torch.stack(want)
    assert (got - want).abs().max() < tol  # float rounding (~3e-7 measured)
    # ...while leaving out the option under test is off by 1e-3 or more: the data exercises it
    assert (other - want).abs().max() > 10 * tol
    if cfg.adaptive_topL:
        assert outside > 0


def test_alpaca_template():
    row = {"instruction": " Q ", "input": "", "output": "A ", "text": "ignored"}
    assert format_sample(row, "alpaca") == "### Instruction: Q\n### Response: A"
    assert format_sample({"instruction": "Q", "input": "x", "output": "A"}, "alpaca") == \
        "### Instruction: Q\n### Input: x\n### Response: A"
    assert format_sample({"prompt": "p", "generation": "g"}, "alpaca") == "pg"
    assert format_sample(row) == "ignored"  # auto: text wins
    with pytest.raises(ValueError, match="instruction"):
        format_sample({"text": "t"}, "alpaca")
    with pytest.raises(ValueError):
        format_sample(row, "chatml")


def test_compat_fields_serialize_only_when_set():
    assert not {"sample_format", "gt_slot", "sketch_rng", "l2_eps"} & set(RiseConfig().to_dict())  # old attrs unchanged
    c = RiseConfig(sample_format="alpaca", gt_slot="append", sketch_rng="cuda", l2_eps="squared")
    assert RiseConfig.from_dict(c.to_dict()) == c
    for bad in ({"gt_slot": "middle"}, {"sketch_rng": "tpu"}, {"sample_format": "chatml"}, {"l2_eps": "max"}):
        with pytest.raises(ValueError):
            RiseConfig(**bad).validate()


def test_alpaca_build_uses_the_template(tmp_path, byte_tok):
    import json

    from conftest import make_neox
    from rise.index.store import IndexReader
    from rise.pipeline import BuildOptions, build_index
    from rise.text import TokenizerAdapter

    rows = [{"instruction": f"say {t}", "output": t} for t in make_texts(5, seed=4)]
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tok = TokenizerAdapter(byte_tok)
    cfg = small_config(sample_format="alpaca")
    build_index(HFTrunk(make_neox()), tok, cfg, str(path), str(tmp_path / "idx"), BuildOptions(block_size=8))
    got = torch.from_numpy(IndexReader(str(tmp_path / "idx")).get_rows(range(len(rows)))).float()
    alp = [{"text": format_sample(r, "alpaca")} for r in rows]
    path2 = tmp_path / "pool_text.jsonl"
    path2.write_text("".join(json.dumps(r) + "\n" for r in alp))
    build_index(HFTrunk(make_neox()), tok, small_config(), str(path2), str(tmp_path / "idx2"), BuildOptions(block_size=8))
    want = torch.from_numpy(IndexReader(str(tmp_path / "idx2")).get_rows(range(len(rows)))).float()
    assert torch.equal(got, want)
    with pytest.raises(ValueError, match="instruction"):  # fail closed on rows the template cannot encode
        text_only = tmp_path / "text.jsonl"
        text_only.write_text(json.dumps({"text": "plain"}) + "\n")
        build_index(HFTrunk(make_neox()), tok, cfg, str(text_only), str(tmp_path / "idx3"), BuildOptions(block_size=8))
