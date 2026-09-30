"""
Multi-Objective Evaluation Suite: Predictive Accuracy, Calibration (ECE/Brier),
Robustness Across Sequential Distribution Shifts, Subgroup Fairness, and Computational Efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from src.task import TaskSpec


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification probabilities.
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_t)
    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = (y_p >= bin_edges[i]) & (y_p < bin_edges[i + 1] if i < n_bins - 1 else y_p <= bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[bin_mask])
            bin_conf = np.mean(y_p[bin_mask])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_batch_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_spec: TaskSpec,
    y_prob: Optional[np.ndarray] = None,
    subgroups: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute full metrics dictionary for a single batch.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    res: Dict[str, Union[float, Dict[str, float]]] = {}

    if task_spec.task_type == "regression":
        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-8 else 0.0

        res["mse"] = mse
        res["rmse"] = rmse
        res["mae"] = mae
        res["r2"] = r2
        res["primary_metric"] = -rmse  # higher is better for ranking

    else:  # binary_classification
        # Round predictions for hard metrics
        hard_preds = (y_p > 0.5).astype(int)
        acc = float(accuracy_score(y_t, hard_preds))
        f1 = float(f1_score(y_t, hard_preds, zero_division=0))

        prob = y_prob if y_prob is not None else y_p
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

        try:
            auc = (
                float(roc_auc_score(y_t, prob))
                if len(np.unique(y_t)) > 1
                else 0.5
            )
        except Exception:
            auc = 0.5

        brier = float(brier_score_loss(y_t, prob))
        ece = compute_expected_calibration_error(y_t, prob)

        res["accuracy"] = acc
        res["f1"] = f1
        res["roc_auc"] = auc
        res["brier_score"] = brier
        res["ece"] = ece
        res["primary_metric"] = auc  # higher is better

    # Subgroup fairness audit if subgroup labels are provided
    if subgroups is not None and len(subgroups) == len(y_t):
        sub_metrics: Dict[str, float] = {}
        unique_groups = np.unique(subgroups)
        group_perfs: List[float] = []

        for g in unique_groups:
            g_mask = (subgroups == g)
            if np.sum(g_mask) >= 5:
                if task_spec.task_type == "regression":
                    g_score = -float(np.sqrt(mean_squared_error(y_t[g_mask], y_p[g_mask])))
                else:
                    g_score = float(accuracy_score(y_t[g_mask], hard_preds[g_mask]))
                sub_metrics[str(g)] = g_score
                group_perfs.append(g_score)

        if group_perfs:
            res["subgroup_min_perf"] = float(min(group_perfs))
            res["subgroup_max_gap"] = float(max(group_perfs) - min(group_perfs))
            res["subgroup_details"] = sub_metrics

    return res


def compute_stream_robustness(
    batch_metrics_history: List[Dict[str, float]],
    task_spec: TaskSpec,
) -> Dict[str, float]:
    """
    Quantifies algorithm robustness over a stream of shifting environments:
    - Average stream performance
    - Worst-case environment performance
    - Max degradation from initial environment
    - Volatility (standard deviation of performance across shifts)
    """
    if not batch_metrics_history:
        return {}

    perfs = [m["primary_metric"] for m in batch_metrics_history]
    base_perf = perfs[0]
    subsequent_perfs = perfs[1:] if len(perfs) > 1 else [base_perf]

    mean_perf = float(np.mean(subsequent_perfs))
    worst_perf = float(np.min(subsequent_perfs))
    best_perf = float(np.max(subsequent_perfs))
    volatility = float(np.std(subsequent_perfs))

    # Degradation is drop relative to initial baseline
    degradations = [max(0.0, base_perf - p) for p in subsequent_perfs]
    max_degradation = float(np.max(degradations)) if degradations else 0.0

    return {
        "baseline_performance": base_perf,
        "mean_shifted_performance": mean_perf,
        "worst_shifted_performance": worst_perf,
        "best_shifted_performance": best_perf,
        "volatility": volatility,
        "max_degradation": max_degradation,
    }


def compute_multi_objective_score(
    robustness_stats: Dict[str, float],
    calibration_stat: float,
    efficiency_stat: float,
    subgroup_stat: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Computes unified composite score in [0.0, 1.0] across all objectives.
    Weights default: 40% performance, 25% robustness, 15% calibration, 10% fairness, 10% efficiency.
    """
    if weights is None:
        weights = {
            "predictive_performance": 0.40,
            "robustness_across_shifts": 0.25,
            "calibration": 0.15,
            "fairness_subgroup_consistency": 0.10,
            "computational_efficiency": 0.10,
        }

    # Normalize metrics to [0, 1] range
    # Predictive performance component
    mean_p = robustness_stats.get("mean_shifted_performance", 0.0)
    # Map from metric to [0, 1]: for AUC/R2 or bounded
    norm_perf = float(np.clip(mean_p if mean_p > 0 else 1.0 / (1.0 + abs(mean_p)), 0.0, 1.0))

    # Robustness component: penalized by max degradation and volatility
    degradation = robustness_stats.get("max_degradation", 0.0)
    norm_robustness = float(np.clip(1.0 - (degradation / max(1.0, abs(mean_p) + 1.0)), 0.0, 1.0))

    # Calibration: 1 - ECE or 1 - Brier
    norm_calib = float(np.clip(1.0 - calibration_stat, 0.0, 1.0))

    # Subgroup consistency: 1 - gap (if evaluated, else neutral 1.0)
    norm_fairness = (
        float(np.clip(1.0 - subgroup_stat, 0.0, 1.0))
        if subgroup_stat is not None
        else 1.0
    )

    # Efficiency: 1.0 / (1.0 + latency_seconds)
    norm_eff = float(np.clip(1.0 / (1.0 + efficiency_stat), 0.0, 1.0))

    score = (
        weights["predictive_performance"] * norm_perf
        + weights["robustness_across_shifts"] * norm_robustness
        + weights["calibration"] * norm_calib
        + weights["fairness_subgroup_consistency"] * norm_fairness
        + weights["computational_efficiency"] * norm_eff
    )
    return float(np.clip(score, 0.0, 1.0))
