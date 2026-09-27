"""Attribution metrics and data selection, matching the RISE evaluation protocol.

For each query score vector and each K, the protocol takes the top-K and
bottom-K rows, computes auPRC / auROC on that 2K subset, and precision@K on
the top-K; results are averaged over queries. auPRC equals scikit-learn's
``average_precision_score`` and auROC its ``roc_auc_score`` (ties get half
credit), implemented here so evaluation needs only NumPy.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def average_precision(y_true, y_score) -> float:
    """Step-wise AP = sum_n (R_n - R_{n-1}) P_n over distinct thresholds; 0.0 without positives."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last index of each tied score run
    tps = np.cumsum(y)[last]
    precision = tps / (last + 1)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true, y_score) -> float:
    """Mann-Whitney AUROC with average ranks for ties; NaN when only one class is present."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], s.size]
    avg_rank = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.repeat(avg_rank, ends - starts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def labels_from_metadata(meta: Iterable[dict], *, positive_label: Optional[str] = None,
                         regex: Optional[str] = None, fields: Sequence[str] = ("text",)) -> np.ndarray:
    """1/0 labels: ``label == positive_label`` (case-insensitive), or ``regex`` found in ``fields``."""
    if (positive_label is None) == (regex is None):
        raise ValueError("pass exactly one of positive_label / regex")
    out: List[int] = []
    if positive_label is not None:
        want = str(positive_label).lower()
        out = [int(str(m.get("label", "")).lower() == want) for m in meta]
    else:
        pat = re.compile(regex, re.IGNORECASE)
        for m in meta:
            s = " ".join(str(m.get(f, "")) for f in fields if m.get(f, ""))
            out.append(int(bool(pat.search(s))))
    return np.asarray(out, dtype=np.int32)


def select_top_bottom_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest and k lowest scores (k capped at n // 2)."""
    n = scores.size
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, n // 2)
    order = np.argsort(scores, kind="mergesort")
    sel = np.unique(np.concatenate([order[-k:], order[:k]]))
    need = 2 * k - sel.size
    if need > 0:
        sel = np.concatenate([sel, np.setdiff1d(order, sel)[:need]])
    return sel


def evaluate_scores(scores, labels: np.ndarray, ks: Sequence[int] = (10, 50, 100)) -> Dict:
    """Protocol metrics for [Q, N] (or [N]) scores; rows are streamed, so memmaps work."""
    S = scores if getattr(scores, "ndim", 1) == 2 else np.asarray(scores).reshape(1, -1)
    labels = np.asarray(labels)
    n = S.shape[1]
    if labels.shape[0] != n:
        raise ValueError(f"{labels.shape[0]} labels for {n} scored rows")
    ks = [int(k) for k in ks if int(k) <= n // 2]
    acc = {k: {"auprc": [], "auroc": [], "precision": []} for k in ks}
    for q in range(S.shape[0]):
        row = np.asarray(S[q], dtype=np.float64)
        top_order = np.argsort(-row, kind="mergesort")
        for k in ks:
            sel = select_top_bottom_indices(row, k)
            ap = average_precision(labels[sel], row[sel])
            auc = roc_auc(labels[sel], row[sel])
            if not np.isnan(ap):
                acc[k]["auprc"].append(ap)
            if not np.isnan(auc):
                acc[k]["auroc"].append(auc)
            acc[k]["precision"].append(float(labels[top_order[:k]].mean()) if k else 0.0)

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_rows": int(n),
        "n_positive": int(labels.sum()),
        "n_queries": int(S.shape[0]),
        "top_k": {str(k): {"auprc": mean(acc[k]["auprc"]), "auroc": mean(acc[k]["auroc"]),
                           "precision": mean(acc[k]["precision"])} for k in ks},
    }


def rrf_aggregate(scores, rrf_k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion of per-query rankings: sum_q 1 / (rrf_k + rank_q(i) + 1)."""
    S = scores if getattr(scores, "ndim", 1) == 2 else np.asarray(scores).reshape(1, -1)
    out = np.zeros(S.shape[1], dtype=np.float64)
    for q in range(S.shape[0]):
        ranks = np.empty(S.shape[1], dtype=np.int64)
        ranks[np.argsort(-np.asarray(S[q], dtype=np.float64), kind="mergesort")] = np.arange(S.shape[1])
        out += 1.0 / (rrf_k + ranks + 1)
    return out


def mean_aggregate(scores) -> np.ndarray:
    S = scores if getattr(scores, "ndim", 1) == 2 else np.asarray(scores).reshape(1, -1)
    out = np.zeros(S.shape[1], dtype=np.float64)
    for q in range(S.shape[0]):
        out += np.asarray(S[q], dtype=np.float64)
    return out / max(S.shape[0], 1)
