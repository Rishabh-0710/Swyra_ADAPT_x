"""
Tests for Multi-Tier Statistical Drift Detector:
- Verifies detection of Covariate Shift (mean/scale)
- Verifies detection of Concept Shift via Residual CUSUM
- Verifies specificity (no false alarms on stationary distributions)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift import DriftDetector, DriftReport


def test_stationary_distribution_low_drift():
    """Verify stationary batches drawn from the same distribution produce low/none drift scores."""
    rng = np.random.default_rng(42)
    n = 500
    n_features = 6

    # Reference distribution: N(0, 1)
    X_ref = rng.normal(0, 1, size=(n, n_features))
    detector = DriftDetector(psi_low=0.10, psi_high=0.25)
    detector.fit_reference(X_ref)

    # Stationary batch: also N(0, 1)
    X_stationary = rng.normal(0, 1, size=(n, n_features))
    report = detector.evaluate(X_stationary)

    assert report.severity in ("none", "low")
    assert report.overall_drift_score < 0.20
    assert not report.is_concept_drift


def test_covariate_shift_detection():
    """Verify significant mean and scale shift triggers high drift report and identifies features."""
    rng = np.random.default_rng(42)
    n = 500
    n_features = 5

    X_ref = rng.normal(0, 1, size=(n, n_features))
    detector = DriftDetector(psi_low=0.10, psi_high=0.25)
    detector.fit_reference(X_ref)

    # Shift feature 0 (mean + 3.0) and feature 2 (scale * 3.0)
    X_shifted = rng.normal(0, 1, size=(n, n_features))
    X_shifted[:, 0] += 3.0
    X_shifted[:, 2] *= 3.0

    report = detector.evaluate(X_shifted, feature_names=[f"col_{i}" for i in range(n_features)])

    assert report.overall_drift_score > 0.25
    assert report.severity in ("medium", "high")
    assert "col_0" in report.top_drifted_features or "col_2" in report.top_drifted_features


def test_concept_drift_detection_via_residuals():
    """Verify sudden error spike triggers concept drift alarm."""
    rng = np.random.default_rng(42)
    n = 300
    X_ref = rng.normal(0, 1, size=(n, 4))
    res_ref = rng.normal(0, 0.5, size=n)

    detector = DriftDetector(cusum_threshold=3.0)
    detector.fit_reference(X_ref, residuals_ref=res_ref)

    # Moderate covariate shift but catastrophic concept shift (huge prediction errors)
    X_curr = rng.normal(0, 1, size=(n, 4))
    res_curr = rng.normal(5.0, 1.5, size=n)  # Errors spiked to mean 5.0

    report = detector.evaluate(X_curr, residuals_curr=res_curr)
    assert report.is_concept_drift
    assert report.severity == "high"
