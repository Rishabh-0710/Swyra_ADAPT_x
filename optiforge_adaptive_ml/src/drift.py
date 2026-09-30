"""
Multi-Tier Domain-Agnostic Statistical Drift Detection Engine.
Combines Feature-Level PSI, Normalized 1-Wasserstein Distance,
and Streaming Residual CUSUM Concept Drift Tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import wasserstein_distance


@dataclass
class DriftReport:
    """
    Comprehensive drift evaluation report for an incoming batch.
    """
    overall_drift_score: float  # Normalized in [0.0, 1.0]
    severity: str  # "none", "low", "medium", "high"
    is_concept_drift: bool
    concept_drift_score: float
    feature_psi: Dict[str, float] = field(default_factory=dict)
    feature_wasserstein: Dict[str, float] = field(default_factory=dict)
    top_drifted_features: List[str] = field(default_factory=list)
    confidence: float = 1.0


class OnlineResidualCUSUM:
    """
    Online Cumulative Sum (CUSUM) change detector for tracking
    prediction errors / residuals to identify true concept drift P(Y|X).
    """

    def __init__(
        self,
        threshold: float = 3.5,
        drift_delta: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.drift_delta = drift_delta
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.n_samples: int = 0
        self.is_initialized: bool = False

    def initialize_baseline(self, residuals: np.ndarray) -> None:
        """Initialize baseline residual distribution on batch 0."""
        res = np.asarray(residuals, dtype=float)
        self.baseline_mean = float(np.mean(res))
        self.baseline_std = float(max(1e-4, np.std(res)))
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n_samples = len(res)
        self.is_initialized = True

    def update(self, residuals: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate new batch residuals against baseline.
        Returns: (is_drift_detected, normalized_cusum_score)
        """
        if not self.is_initialized or len(residuals) == 0:
            return False, 0.0

        res = np.asarray(residuals, dtype=float)
        # Standardize residuals using baseline
        z = (res - self.baseline_mean) / self.baseline_std

        # Mean batch standardized error
        mean_z = float(np.mean(z))

        # CUSUM updates
        self.s_pos = max(0.0, self.s_pos + mean_z - self.drift_delta)
        self.s_neg = min(0.0, self.s_neg + mean_z + self.drift_delta)

        max_stat = max(self.s_pos, abs(self.s_neg))
        normalized_score = float(np.clip(max_stat / max(1e-5, self.threshold), 0.0, 2.0))
        is_drift = max_stat >= self.threshold

        if is_drift:
            # Decay CUSUM to allow subsequent detection
            self.s_pos *= 0.5
            self.s_neg *= 0.5

        return is_drift, normalized_score


class DriftDetector:
    """
    Robust Domain-Agnostic Drift Detector.
    Evaluates both Covariate Shift P(X) and Concept Shift P(Y|X).
    """

    def __init__(
        self,
        psi_low: float = 0.10,
        psi_high: float = 0.25,
        wasserstein_threshold: float = 0.20,
        cusum_threshold: float = 3.5,
        n_bins: int = 10,
    ) -> None:
        self.psi_low = psi_low
        self.psi_high = psi_high
        self.wasserstein_threshold = wasserstein_threshold
        self.n_bins = n_bins

        self.cusum = OnlineResidualCUSUM(threshold=cusum_threshold)
        self.reference_X: Optional[np.ndarray] = None
        self.reference_stds: Optional[np.ndarray] = None
        self.bin_edges: Dict[int, np.ndarray] = {}
        self.ref_bin_probs: Dict[int, np.ndarray] = {}

    def fit_reference(
        self,
        X_ref: np.ndarray,
        residuals_ref: Optional[np.ndarray] = None,
    ) -> DriftDetector:
        """
        Calibrate reference distribution on initial batch X_0.
        Precomputes quantile bin edges for fast O(N) PSI evaluation.
        """
        X_arr = np.asarray(X_ref, dtype=float)
        self.reference_X = X_arr.copy()
        n_samples, n_features = X_arr.shape

        self.reference_stds = np.std(X_arr, axis=0)
        self.reference_stds[self.reference_stds < 1e-6] = 1.0

        # Precompute quantiles for PSI
        self.bin_edges = {}
        self.ref_bin_probs = {}
        quantiles = np.linspace(0, 100, self.n_bins + 1)

        for j in range(n_features):
            feat_vals = X_arr[:, j]
            edges = np.percentile(feat_vals, quantiles)
            edges[0] -= 1e-6
            edges[-1] += 1e-6
            # Ensure strictly increasing edges to prevent empty bins
            edges = np.unique(edges)
            if len(edges) < 3:
                edges = np.linspace(np.min(feat_vals) - 1e-4, np.max(feat_vals) + 1e-4, self.n_bins + 1)

            self.bin_edges[j] = edges

            # Reference bin probabilities with Laplace smoothing
            counts, _ = np.histogram(feat_vals, bins=edges)
            probs = (counts + 1e-4) / (n_samples + 1e-4 * len(counts))
            self.ref_bin_probs[j] = probs

        if residuals_ref is not None:
            self.cusum.initialize_baseline(residuals_ref)

        return self

    def _compute_feature_psi(self, feat_idx: int, curr_vals: np.ndarray) -> float:
        """Calculate PSI for a single feature against reference quantiles."""
        edges = self.bin_edges.get(feat_idx)
        ref_probs = self.ref_bin_probs.get(feat_idx)
        if edges is None or ref_probs is None or len(curr_vals) == 0:
            return 0.0

        curr_counts, _ = np.histogram(curr_vals, bins=edges)
        curr_probs = (curr_counts + 1e-4) / (len(curr_vals) + 1e-4 * len(curr_counts))

        # PSI = sum((Actual - Expected) * ln(Actual / Expected))
        psi_val = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        return float(max(0.0, psi_val))

    def evaluate(
        self,
        X_curr: np.ndarray,
        residuals_curr: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Evaluate incoming batch for covariate drift and concept drift.
        """
        X_arr = np.asarray(X_curr, dtype=float)
        n_samples, n_features = X_arr.shape

        if self.reference_X is None:
            # Not initialized yet
            return DriftReport(
                overall_drift_score=0.0,
                severity="none",
                is_concept_drift=False,
                concept_drift_score=0.0,
            )

        feat_names = (
            feature_names
            if feature_names and len(feature_names) == n_features
            else [f"feat_{j}" for j in range(n_features)]
        )

        psi_dict: Dict[str, float] = {}
        wass_dict: Dict[str, float] = {}
        drift_scores: List[float] = []

        for j in range(n_features):
            curr_vals = X_arr[:, j]
            ref_vals = self.reference_X[:, j]

            # 1. PSI
            psi = self._compute_feature_psi(j, curr_vals)
            psi_dict[feat_names[j]] = psi

            # 2. Normalized 1-Wasserstein Distance
            raw_wass = wasserstein_distance(ref_vals, curr_vals)
            norm_wass = raw_wass / max(1e-4, self.reference_stds[j])
            wass_dict[feat_names[j]] = float(norm_wass)

            # Combined feature drift score
            norm_psi = np.clip(psi / (self.psi_high * 1.5), 0.0, 1.0)
            norm_w = np.clip(norm_wass / (self.wasserstein_threshold * 2.0), 0.0, 1.0)
            drift_scores.append(float(0.5 * norm_psi + 0.5 * norm_w))

        # Overall covariate drift score (robust 80th percentile across features)
        covariate_drift_score = float(np.percentile(drift_scores, 80)) if drift_scores else 0.0

        # Concept drift via Residual CUSUM
        is_concept_drift = False
        concept_score = 0.0
        if residuals_curr is not None:
            is_concept_drift, concept_score = self.cusum.update(residuals_curr)

        # Composite overall drift score: blends covariate and concept drift
        # Concept drift has double weighting for adaptation decisions
        if is_concept_drift or concept_score > 0.5:
            overall_score = 0.3 * covariate_drift_score + 0.7 * min(1.0, concept_score)
        else:
            overall_score = covariate_drift_score

        overall_score = float(np.clip(overall_score, 0.0, 1.0))

        # Severity classification
        if overall_score >= 0.40 or is_concept_drift:
            severity = "high"
        elif overall_score >= 0.18:
            severity = "medium"
        elif overall_score >= 0.08:
            severity = "low"
        else:
            severity = "none"

        # Identify top drifted features
        ranked_features = sorted(
            feat_names, key=lambda f: psi_dict.get(f, 0.0), reverse=True
        )
        top_drifted = [f for f in ranked_features if psi_dict.get(f, 0.0) >= self.psi_low][:5]

        # Confidence based on sample size
        confidence = float(np.clip(n_samples / 200.0, 0.5, 1.0))

        return DriftReport(
            overall_drift_score=overall_score,
            severity=severity,
            is_concept_drift=is_concept_drift,
            concept_drift_score=concept_score,
            feature_psi=psi_dict,
            feature_wasserstein=wass_dict,
            top_drifted_features=top_drifted,
            confidence=confidence,
        )

    def update_reference(self, X_new: np.ndarray, decay: float = 0.2) -> None:
        """
        Softly update the reference distribution over time to prevent
        spurious permanent alarms when shifts are gradual.
        """
        if self.reference_X is None:
            self.fit_reference(X_new)
            return

        n_replace = int(len(self.reference_X) * decay)
        if n_replace > 0 and len(X_new) > 0:
            replace_idx = np.random.choice(len(self.reference_X), size=n_replace, replace=False)
            pick_idx = np.random.choice(len(X_new), size=min(n_replace, len(X_new)), replace=False)
            self.reference_X[replace_idx[:len(pick_idx)]] = X_new[pick_idx]
