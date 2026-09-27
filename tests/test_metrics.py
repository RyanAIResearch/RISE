import numpy as np
import pytest

from rise.metrics import (average_precision, evaluate_scores, labels_from_metadata, rrf_aggregate, roc_auc,
                          select_top_bottom_indices)


def test_hand_computed_values():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.1]
    # ranking 1,0,1,0 -> precision at hits 1/1 and 2/3
    assert average_precision(y, s) == pytest.approx((1.0 + 2 / 3) / 2)
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert average_precision([0, 0], [0.1, 0.2]) == 0.0
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_ties_get_half_credit():
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_against_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(5, 60))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = np.round(rng.standard_normal(n), 1)  # coarse -> many ties
        assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
        assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))


def test_top_bottom_selection():
    s = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    assert sorted(select_top_bottom_indices(s, 1).tolist()) == [0, 1]
    assert len(select_top_bottom_indices(s, 10)) == 4  # capped at n // 2


def test_evaluate_perfect_ranking():
    labels = np.array([1] * 5 + [0] * 15)
    scores = np.linspace(1, 0, 20)
    res = evaluate_scores(scores, labels, ks=[5, 100])
    assert list(res["top_k"]) == ["5"]  # 100 > n // 2 is skipped
    assert res["top_k"]["5"] == {"auprc": 1.0, "auroc": 1.0, "precision": 1.0}


def test_labels_and_rrf():
    meta = [{"label": "Positive", "text": "a"}, {"label": "negative", "text": "howdy! b"}]
    assert labels_from_metadata(meta, positive_label="positive").tolist() == [1, 0]
    assert labels_from_metadata(meta, regex=r"howdy!").tolist() == [0, 1]
    S = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    agg = rrf_aggregate(S, rrf_k=0)
    assert agg.argmax() == 1  # ranks (2nd, 1st) beat (1st, 3rd)
