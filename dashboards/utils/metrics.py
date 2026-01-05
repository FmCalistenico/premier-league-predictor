from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class Metrics:
    pred_over_pct: float
    balanced_accuracy: float
    specificity: float
    sensitivity: float
    roc_auc: float
    ece: float
    brier: float


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(y_true: np.ndarray, prob_over: np.ndarray, threshold: float = 0.5) -> Metrics:
    prob_over = np.clip(np.asarray(prob_over).reshape(-1), 0.0, 1.0)
    y_pred = (prob_over >= float(threshold)).astype(int)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    # ECE (simple 10-bin)
    ece = expected_calibration_error(y_true, prob_over, n_bins=10)

    try:
        roc_auc = float(roc_auc_score(y_true, prob_over)) if len(np.unique(y_true)) > 1 else float('nan')
    except Exception:
        roc_auc = float('nan')

    return Metrics(
        pred_over_pct=float(np.mean(y_pred) * 100.0),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        specificity=float(specificity),
        sensitivity=float(sensitivity),
        roc_auc=float(roc_auc),
        ece=float(ece),
        brier=float(brier_score_loss(y_true, prob_over)),
    )


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(prob[mask]))
        w = float(np.mean(mask))
        ece += w * abs(acc - conf)
    return float(ece)


def ks_test_by_class(prob_over: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    prob_over = np.asarray(prob_over).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    p0 = prob_over[y_true == 0]
    p1 = prob_over[y_true == 1]
    if len(p0) < 2 or len(p1) < 2:
        return {'statistic': float('nan'), 'p_value': float('nan')}

    stat, pval = ks_2samp(p0, p1)
    return {'statistic': float(stat), 'p_value': float(pval)}


def brier_decomposition(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    # Murphy decomposition: BS = reliability - resolution + uncertainty
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)

    bs = float(np.mean((prob - y_true) ** 2))
    p_bar = float(np.mean(y_true))
    uncertainty = p_bar * (1.0 - p_bar)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if not mask.any():
            continue
        o_k = float(np.mean(y_true[mask]))
        p_k = float(np.mean(prob[mask]))
        w = float(np.mean(mask))
        reliability += w * (p_k - o_k) ** 2
        resolution += w * (o_k - p_bar) ** 2

    return {
        'brier': bs,
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty': float(uncertainty),
    }
