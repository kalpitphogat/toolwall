import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from toolwall.eval.metrics import at_threshold, average_precision, roc_auc, tpr_at_fpr
from toolwall.model.calibrate import fit_temperature, sigmoid, threshold_at_fpr
from toolwall.model.chunking import chunk_label


@pytest.mark.parametrize("seed", range(5))
def test_auc_and_ap_match_sklearn_including_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.normal(y * 1.2, 1.0), 1)          # rounding creates ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_threshold_at_fpr_respects_target():
    rng = np.random.default_rng(0)
    neg = rng.normal(0, 1, 10_000)
    for target in (0.001, 0.01, 0.05):
        t = threshold_at_fpr(neg, target)
        assert np.mean(neg >= t) <= target
        assert np.mean(neg >= t) > target * 0.8          # and not needlessly conservative


def test_tpr_at_fpr_never_exceeds_fpr_budget():
    y = np.array([0] * 100 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 100), np.full(10, 0.995)]
    assert tpr_at_fpr(y, s, 0.009) == 0.0      # no negative may pass, and the top negative (1.0) beats every positive
    assert tpr_at_fpr(y, s, 0.01) == 1.0       # one negative (1%) may pass, so the threshold drops below 0.995


def test_at_threshold_counts():
    r = at_threshold([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1) and r["fpr"] == 0.5


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 20_000)
    y = (rng.random(z.size) < sigmoid(z)).astype(int)    # well calibrated at T=1
    assert fit_temperature(z * 4, y) == pytest.approx(4, rel=0.1)


def test_chunk_labels():
    assert chunk_label(0, None, (0, 100)) == 0
    assert chunk_label(1, (150, 200), (0, 120)) == 0      # context window without the payload
    assert chunk_label(1, (150, 200), (100, 400)) == 1
    assert chunk_label(1, None, (0, 100)) is None
