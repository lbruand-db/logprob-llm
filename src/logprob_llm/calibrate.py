"""Post-hoc calibration and calibration metrics (SPECS §8).

Temperature scaling (Guo et al., 2017, arXiv:1706.04599): fit a single scalar T
that minimizes NLL on a held-out set. It leaves the argmax — and therefore
accuracy — unchanged, and only reshapes confidence. This is what turns raw
logprobs into a *calibrated* probability, the whole value proposition of a
log-prob model.

Pure numpy/scipy: no torch, so this runs and is unit-tested anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits / temperature``. Shape (N, K) -> (N, K)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean NLL of the true class under temperature-scaled softmax."""
    probs = softmax(logits, temperature)
    n = probs.shape[0]
    true = probs[np.arange(n), np.asarray(labels, dtype=int)]
    return float(-np.log(np.clip(true, 1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 100.0),
) -> float:
    """Fit the temperature T>0 that minimizes NLL on (logits, labels).

    ``logits`` is (N, K) — the pre-softmax scores over the K answer tokens on a
    held-out calibration split. ``labels`` is (N,) of gold class indices.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, K); got shape {logits.shape}")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("logits and labels must have the same number of rows")

    result = minimize_scalar(
        lambda t: negative_log_likelihood(logits, labels, t),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE: |confidence - accuracy| averaged over confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # last bin is closed on the right so confidence == 1.0 is counted
        in_bin = (confidences > lo) & ((confidences <= hi) if hi < 1.0 else (confidences <= hi))
        count = in_bin.sum()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece += (count / n) * abs(avg_conf - avg_acc)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs one-hot truth."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy (a.k.a. log loss) of the true class."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true, 1e-12, 1.0)).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(labels, dtype=int)).mean())


def calibration_report(
    logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int = 15
) -> dict[str, float]:
    """Accuracy + ECE/Brier/log-loss before (T=1) and after temperature scaling."""
    raw = softmax(logits, 1.0)
    cal = softmax(logits, temperature)
    return {
        "temperature": float(temperature),
        "accuracy": accuracy(raw, labels),  # argmax-invariant to T
        "ece_before": expected_calibration_error(raw, labels, n_bins),
        "ece_after": expected_calibration_error(cal, labels, n_bins),
        "brier_before": brier_score(raw, labels),
        "brier_after": brier_score(cal, labels),
        "logloss_before": log_loss(raw, labels),
        "logloss_after": log_loss(cal, labels),
    }
