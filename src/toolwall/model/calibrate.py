"""Temperature scaling and false-positive-rate-targeted thresholds (numpy only)."""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_temperature(margins: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T minimising validation NLL of sigmoid(margin / T).

    `margins` are logit(positive) - logit(negative). A 1-D grid search is exact enough for a
    single parameter and avoids an optimiser dependency.
    """
    margins = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 600))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = np.clip(sigmoid(margins / t), 1e-12, 1 - 1e-12)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def threshold_at_fpr(neg_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that the share of negatives with score >= t is <= target_fpr."""
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if neg.size == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * neg.size))     # negatives permitted at or above t
    if allowed >= neg.size:
        return float(neg[0])
    t = neg[neg.size - allowed - 1]                     # the highest negative that must stay below t
    return float(np.nextafter(t, np.inf))
