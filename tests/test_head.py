import pytest
import torch
import torch.nn.functional as F

from conftest import make_texts, small_config
from reference import V3Reference, find_temperature_for_entropy
from rise.head import RiseHead
from rise.runtime.batching import pad_batch
from rise.runtime.hf import HFTrunk


def _batch(tok, texts, max_len=60):
    seqs = [tok.encode(t)[:max_len] for t in texts]
    ids, lens = pad_batch(seqs, tok.pad_token_id)
    return seqs, ids, lens


def _cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=0).item()


@pytest.mark.parametrize("model_name", ["neox", "llama"])
@pytest.mark.parametrize("cfg_kw", [
    {},                                                    # rh+gh, fixed tau=0.1 (paper's effective default)
    {"adaptive_temperature": True},                        # entropy-targeted temperature search
    {"fusion_mode": "rh"},
    {"fusion_mode": "gh"},
    {"fusion_mode": "rh+gh+rg", "lambda_rg": 0.5},
    {"embedding_type": "input"},
    {"max_topL_cap": 12, "min_topL": 4},                  # forces ground truth into the TopK
    {"adaptive_topL": False, "fallback_topL": 10},
    {"adaptive_temperature": False, "tau_fallback": 0.7},
    {"normalize_sample": False},
])
def test_matches_research_estimator(request, byte_tok, model_name, cfg_kw):
    model = request.getfixturevalue(model_name)
    cfg = small_config(**cfg_kw)
    ref = V3Reference(model, byte_tok, cfg)
    trunk = HFTrunk(model)
    head = RiseHead.from_trunk(trunk, cfg)
    seqs, ids, lens = _batch(byte_tok, make_texts(7, seed=5))
    vecs = head.compute_from_hidden(trunk.hidden_states(ids, lens), ids, lens)
    for i, s in enumerate(seqs):
        want = ref.vector_from_ids(s)
        if cfg.normalize_sample:
            assert _cos(vecs[i], want) > 0.9999, (i, _cos(vecs[i], want))
        else:
            torch.testing.assert_close(vecs[i], want, rtol=2e-4, atol=2e-5)


def test_prompt_masking_matches_reference(byte_tok, neox):
    cfg = small_config()
    ref = V3Reference(neox, byte_tok, cfg)
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, cfg)
    prompt, full = "Question: where does influence live? Answer:", \
        "Question: where does influence live? Answer: near the readout head"
    ids_list = byte_tok(full, truncation=True, max_length=cfg.seq_len)["input_ids"]
    ls = len(byte_tok(prompt)["input_ids"])
    ids, lens = pad_batch([ids_list], byte_tok.pad_token_id)
    v = head.compute_from_hidden(trunk.hidden_states(ids, lens), ids, lens, torch.tensor([ls]))[0]
    assert _cos(v, ref.prompted_vector(full, prompt)) > 0.9999
    # masking really removes the prompt positions
    v_all = head.compute_from_hidden(trunk.hidden_states(ids, lens), ids, lens)[0]
    assert _cos(v, v_all) < 0.999


def test_tau_search_matches_reference_bisection(byte_tok, neox):
    cfg = small_config()
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, cfg)
    seqs, ids, lens = _batch(byte_tok, make_texts(9, seed=11))
    with torch.inference_mode():
        h = trunk.hidden_states(ids, lens)
        T = ids.shape[1] - 1
        logits = head.logits(h[:, :T].reshape(-1, h.shape[-1])).view(len(seqs), T, -1)
        valid = torch.arange(T).unsqueeze(0) < (lens - 1).unsqueeze(1)
        logits.masked_fill_(~valid.unsqueeze(-1), 0.0)
        taus = head.tau_search(logits, valid)
    for i, s in enumerate(seqs):
        want = find_temperature_for_entropy(logits[i, : len(s) - 1], cfg.target_entropy, cfg.tau_min,
                                            cfg.tau_max, cfg.tau_search_steps, cfg.eps, cfg.tau_fallback)
        assert abs(float(taus[i]) - want) < 1e-6
        assert cfg.tau_min < want < cfg.tau_max  # the search is actually exercised


def test_research_fp16_entropy_search_collapses_to_tau_min():
    """Documents why the paper's GPU runs used tau ~= 0.1: in fp16, clamp(p, 1e-8) clamps at 0,
    underflowed probabilities give 0 * log 0 = NaN, and the bisection shrinks onto tau_min."""
    from rise.compat import degenerate_fp16_tau

    cfg = small_config()
    logits = torch.randn(16, 50304, generator=torch.Generator().manual_seed(0)) * 8
    fp16 = find_temperature_for_entropy(logits.half(), cfg.target_entropy, cfg.tau_min, cfg.tau_max,
                                        cfg.tau_search_steps, cfg.eps, cfg.tau_fallback)
    fp32 = find_temperature_for_entropy(logits, cfg.target_entropy, cfg.tau_min, cfg.tau_max,
                                        cfg.tau_search_steps, cfg.eps, cfg.tau_fallback)
    assert fp16 == pytest.approx(degenerate_fp16_tau(cfg)) == pytest.approx(0.1000366, abs=1e-7)
    assert fp32 > 0.5  # the same search in fp32 does find the target entropy


def test_research_config_conversion():
    from rise.compat import research_effective_config

    base = small_config(adaptive_temperature=True).to_dict()
    gpu = research_effective_config({**base, "device": "cuda"})
    assert not gpu.adaptive_temperature and gpu.tau_fallback == pytest.approx(0.1000366, abs=1e-7)
    assert research_effective_config({**base, "device": "cpu"}).adaptive_temperature
    fixed = research_effective_config({**base, "adaptive_temperature": False, "tau_fallback": 0.7, "device": "cuda"})
    assert not fixed.adaptive_temperature and fixed.tau_fallback == 0.7


def test_batch_composition_invariance(byte_tok, neox):
    cfg = small_config()
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, cfg, max_tokens_per_step=64)  # forces several micro-batches
    seqs, ids, lens = _batch(byte_tok, make_texts(8, seed=2))
    together = head.compute_from_hidden(trunk.hidden_states(ids, lens), ids, lens)
    for i, s in enumerate(seqs):
        one_ids, one_lens = pad_batch([s], byte_tok.pad_token_id)
        alone = head.compute_from_hidden(trunk.hidden_states(one_ids, one_lens), one_ids, one_lens)[0]
        assert _cos(alone, together[i]) > 0.99999


def test_signature_inner_product_is_sum_of_token_products(byte_tok, neox):
    """<phi(x), phi(y)> = sum_{t,s} <r_t, r_s><h_t, h_s>: the sketched LM-head gradient kernel."""
    cfg = small_config(fusion_mode="rh", lambda_rh=1.0, normalize_sample=False)
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, cfg)
    seqs, ids, lens = _batch(byte_tok, make_texts(2, seed=4))
    hid = trunk.hidden_states(ids, lens)
    phi = head.compute_from_hidden(hid, ids, lens)
    f = head.token_factors(hid, ids, lens)
    r, h = f["r"], f["h"]
    kernel = ((r[0] @ r[1].T) * (h[0] @ h[1].T)).sum()
    torch.testing.assert_close(phi[0] @ phi[1], kernel, rtol=1e-4, atol=1e-6)


def test_too_short_chunks_give_zero_vectors(byte_tok, neox):
    cfg = small_config()
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, cfg)
    ids, lens = pad_batch([[5], [5, 6, 7]], byte_tok.pad_token_id)
    v = head.compute_from_hidden(trunk.hidden_states(ids, lens), ids, lens)
    assert torch.count_nonzero(v[0]) == 0 and v[1].norm() > 0.99


def test_rejects_token_ids_outside_lm_head(byte_tok, neox):
    trunk = HFTrunk(neox)
    head = RiseHead.from_trunk(trunk, small_config())
    ids, lens = pad_batch([[1, 2, 3]], 0)
    hidden = trunk.hidden_states(ids, lens)
    bad = ids.clone()
    bad[0, 2] = head.V + 5
    with pytest.raises(ValueError, match="tokenizer/model mismatch"):
        head.compute_from_hidden(hidden, bad, lens)
