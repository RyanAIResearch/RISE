import pytest
import torch

from reference import final_norm
from rise.runtime.batching import Prefetcher, pad_batch, plan_batches
from rise.runtime.hf import HFTrunk
from rise.runtime.trunk import check_same_model


@pytest.mark.parametrize("model_name", ["neox", "llama"])
def test_hidden_states_are_lm_head_inputs(request, model_name):
    model = request.getfixturevalue(model_name)
    trunk = HFTrunk(model)
    assert trunk.verify() < 1e-5
    grabbed = {}
    handle = final_norm(model).register_forward_hook(lambda m, i, o: grabbed.update(h=o))
    ids = torch.randint(0, trunk.vocab_size, (2, 9), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        model(input_ids=ids)
    handle.remove()
    torch.testing.assert_close(trunk.hidden_states(ids, torch.tensor([9, 9])), grabbed["h"])


def test_right_padding_leaves_valid_positions_unchanged(neox):
    trunk = HFTrunk(neox)
    a, b = [3, 9, 27, 81, 4, 12], [7, 7, 1]
    ids, lens = pad_batch([a, b], 0)
    both = trunk.hidden_states(ids, lens)
    ids_b, lens_b = pad_batch([b], 0)
    torch.testing.assert_close(both[1, :3], trunk.hidden_states(ids_b, lens_b)[0], rtol=1e-5, atol=1e-5)


def test_verify_catches_a_wrong_head(neox):
    trunk = HFTrunk(neox)
    trunk.logits_postprocess = lambda z: z * 2.0
    with pytest.raises(RuntimeError, match="do not reproduce"):
        trunk.verify()


def test_softcap_postprocess_is_detected():
    from rise.runtime.hf import _postprocess_for

    class C:
        final_logit_softcapping = 30.0

    post = _postprocess_for(C())
    z = torch.tensor([0.0, 30.0, -300.0])
    torch.testing.assert_close(post(z), torch.tanh(z / 30.0) * 30.0)


def test_plan_batches_respects_budget_and_covers_everything():
    lens = [5, 50, 7, 33, 33, 2, 64, 1, 18]
    batches = plan_batches(lens, max_batch_tokens=100, max_batch_rows=3)
    assert sorted(i for b in batches for i in b) == list(range(len(lens)))
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or len(b) * max(lens[i] for i in b) <= 100
    assert lens[batches[0][0]] == max(lens)  # longest first: OOM shows up immediately


def test_prefetcher_propagates_errors():
    def gen():
        yield 1
        raise KeyError("boom")

    it = iter(Prefetcher(gen()))
    assert next(it) == 1
    with pytest.raises(KeyError):
        next(it)


def test_model_mismatch_is_refused(neox, llama):
    a, b = HFTrunk(neox).describe(), HFTrunk(llama).describe()
    check_same_model(a, a)
    with pytest.raises(ValueError, match="does not match"):
        check_same_model(a, b)
    check_same_model(a, b, allow_mismatch=True)
