"""Ranking and operating-point metrics, implemented in numpy and cross-checked against scikit-learn in tests."""
from __future__ import annotations

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks, ties sharing the mean rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(labels, scores) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    """Area under the precision-recall curve, step-wise (sklearn's definition)."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]                 # last index of each distinct score
    tp = np.cumsum(y)[idx]
    fp = (idx + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def tpr_at_fpr(labels, scores, max_fpr: float) -> float:
    """Best detection rate achievable while flagging at most `max_fpr` of benign items."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    neg = np.sort(s[y == 0])[::-1]
    pos = s[y == 1]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allowed = int(np.floor(max_fpr * neg.size))
    t = neg[allowed] if allowed < neg.size else -np.inf   # threshold must sit strictly above this negative
    return float(np.mean(pos > t))


def at_threshold(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "fpr": fp / (fp + tn) if fp + tn else float("nan"), "f1": f1}


def summary(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    return {
        "n": int(y.size), "positives": int(y.sum()),
        "roc_auc": roc_auc(y, scores), "pr_auc": average_precision(y, scores),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, scores, 0.01), "tpr_at_5pct_fpr": tpr_at_fpr(y, scores, 0.05),
        "at_operating_threshold": at_threshold(y, scores, threshold),
    }
