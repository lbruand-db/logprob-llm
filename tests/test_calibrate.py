"""Deterministic tests for temperature scaling and calibration metrics.

Pure numpy/scipy — no model, no network. Verifies the property the whole spec
leans on (SPECS §8): temperature scaling reduces miscalibration of an
overconfident model while leaving the argmax (accuracy) unchanged.
"""

import numpy as np

from logprob_llm import calibrate


def _overconfident_dataset(seed: int = 0, n: int = 4000, k: int = 6):
    """Build logits that are directionally right but far too sharp.

    Draw a true class, make a logit vector peaked on it, then scale it UP so the
    softmax is overconfident relative to its actual accuracy. Temperature scaling
    should want T > 1 to cool it back down.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    base = rng.normal(0, 1.0, size=(n, k))
    # add signal on the true class, then over-sharpen
    base[np.arange(n), labels] += 2.0
    logits = base * 6.0
    return logits, labels


def test_softmax_normalizes():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = calibrate.softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], np.array([1 / 3, 1 / 3, 1 / 3]))


def test_temperature_scaling_is_argmax_invariant():
    logits, labels = _overconfident_dataset()
    T = calibrate.fit_temperature(logits, labels)
    acc_before = calibrate.accuracy(calibrate.softmax(logits, 1.0), labels)
    acc_after = calibrate.accuracy(calibrate.softmax(logits, T), labels)
    assert acc_before == acc_after  # T only rescales; ranking is unchanged


def test_temperature_scaling_reduces_ece_on_overconfident_model():
    logits, labels = _overconfident_dataset()
    T = calibrate.fit_temperature(logits, labels)
    assert T > 1.0  # an overconfident model needs cooling
    rep = calibrate.calibration_report(logits, labels, T)
    assert rep["ece_after"] < rep["ece_before"]
    assert rep["logloss_after"] <= rep["logloss_before"] + 1e-9


def test_fit_temperature_recovers_known_temperature():
    """If data is generated at temperature T*, fitting should recover ~T*."""
    rng = np.random.default_rng(1)
    n, k = 6000, 5
    true_logits = rng.normal(0, 3.0, size=(n, k))
    T_star = 2.5
    probs = calibrate.softmax(true_logits, T_star)
    labels = np.array([rng.choice(k, p=probs[i]) for i in range(n)])
    # observed logits are the pre-scaled ones; fit should find ~T_star
    T = calibrate.fit_temperature(true_logits, labels)
    assert abs(T - T_star) < 0.4


def test_metrics_ranges():
    logits, labels = _overconfident_dataset(seed=3)
    probs = calibrate.softmax(logits, 1.0)
    assert 0.0 <= calibrate.expected_calibration_error(probs, labels) <= 1.0
    assert 0.0 <= calibrate.brier_score(probs, labels) <= 2.0
    assert calibrate.log_loss(probs, labels) >= 0.0
