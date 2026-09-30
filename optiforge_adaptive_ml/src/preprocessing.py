"""
Streaming tabular preprocessor with running statistical moments,
robust outlier clipping, missing value imputation, and target encoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.task import TaskSpec


class StreamingTabularPreprocessor:
    """
    Incremental preprocessor that maintains streaming summary statistics
    (running mean, running variance via Welford's algorithm, categorical frequencies,
    and target smoothing statistics) without retaining raw historical datasets.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        clip_std_threshold: float = 6.0,
        smoothing_weight: float = 10.0,
    ) -> None:
        self.task_spec = task_spec
        self.clip_std_threshold = clip_std_threshold
        self.smoothing_weight = smoothing_weight

        # Streaming statistics for numerical features
        self.num_count: int = 0
        self.running_means: Dict[str, float] = {}
        self.running_m2: Dict[str, float] = {}  # Sum of squared differences for Welford
        self.impute_values: Dict[str, float] = {}

        # Streaming statistics for categorical features
        # maps feature -> category -> count
        self.cat_counts: Dict[str, Dict[Any, int]] = {}
        # maps feature -> category -> sum_y (for target encoding)
        self.cat_target_sums: Dict[str, Dict[Any, float]] = {}
        self.global_target_sum: float = 0.0
        self.global_target_count: int = 0

        self.feature_names_out_: List[str] = []
        self.is_fitted: bool = False

    def _convert_to_df(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = np.asarray(X)
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"feat_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _convert_to_series(
        self, y: Optional[Union[pd.Series, np.ndarray]]
    ) -> Optional[pd.Series]:
        if y is None:
            return None
        if isinstance(y, pd.Series):
            return y.copy()
        return pd.Series(np.asarray(y))

    def _update_numerical_moments(self, df: pd.DataFrame) -> None:
        """Update running mean and variance using Welford's online algorithm."""
        for col in self.task_spec.numerical_features:
            if col not in df.columns:
                continue
            vals = df[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            batch_count = len(vals)
            batch_mean = float(np.mean(vals))
            batch_m2 = float(np.sum((vals - batch_mean) ** 2))

            if col not in self.running_means:
                self.running_means[col] = batch_mean
                self.running_m2[col] = batch_m2
                self.impute_values[col] = batch_mean
            else:
                old_mean = self.running_means[col]
                delta = batch_mean - old_mean
                total_count = self.num_count + batch_count
                new_mean = old_mean + delta * (batch_count / total_count)
                new_m2 = (
                    self.running_m2[col]
                    + batch_m2
                    + (delta**2) * (self.num_count * batch_count / total_count)
                )
                self.running_means[col] = new_mean
                self.running_m2[col] = new_m2
                self.impute_values[col] = new_mean

    def _update_categorical_stats(
        self, df: pd.DataFrame, y: Optional[pd.Series]
    ) -> None:
        """Update streaming frequency and target-encoding statistics."""
        if y is not None:
            clean_mask = ~y.isna()
            y_clean = y[clean_mask].astype(float)
            self.global_target_sum += float(y_clean.sum())
            self.global_target_count += len(y_clean)

        for col in self.task_spec.categorical_features:
            if col not in df.columns:
                continue
            if col not in self.cat_counts:
                self.cat_counts[col] = {}
                self.cat_target_sums[col] = {}

            series = df[col].fillna("__MISSING__")
            val_counts = series.value_counts()
            for cat, count in val_counts.items():
                self.cat_counts[col][cat] = self.cat_counts[col].get(cat, 0) + int(count)

            if y is not None:
                grouped = y.groupby(series).sum()
                for cat, target_sum in grouped.items():
                    self.cat_target_sums[col][cat] = (
                        self.cat_target_sums[col].get(cat, 0.0) + float(target_sum)
                    )

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> StreamingTabularPreprocessor:
        """Initial fit on batch 0."""
        df = self._convert_to_df(X)
        y_ser = self._convert_to_series(y)

        # Synchronize task_spec features if they were empty
        if not self.task_spec.numerical_features and not self.task_spec.categorical_features:
            # Fallback auto-detection
            self.task_spec.numerical_features = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            self.task_spec.categorical_features = [
                c for c in df.columns if c not in self.task_spec.numerical_features
            ]

        self.num_count = len(df)
        self._update_numerical_moments(df)
        self._update_categorical_stats(df, y_ser)
        self.is_fitted = True
        return self

    def partial_fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> StreamingTabularPreprocessor:
        """Incrementally update summary statistics with new sequential batch."""
        if not self.is_fitted:
            return self.fit(X, y)

        df = self._convert_to_df(X)
        y_ser = self._convert_to_series(y)

        self._update_numerical_moments(df)
        self.num_count += len(df)
        self._update_categorical_stats(df, y_ser)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform raw tabular input into clean, scaled, domain-agnostic numerical matrix.
        Includes missingness indicators and smooth target/frequency encodings.
        """
        df = self._convert_to_df(X)
        transformed_cols: List[np.ndarray] = []
        feature_names: List[str] = []

        # 1. Process numerical features
        for col in self.task_spec.numerical_features:
            if col in df.columns:
                series = df[col].to_numpy(dtype=float)
            else:
                series = np.full(len(df), np.nan, dtype=float)

            # Missing indicator
            nan_mask = np.isnan(series)
            impute_val = self.impute_values.get(col, 0.0)
            series_imputed = np.where(nan_mask, impute_val, series)

            # Robust scaling using running mean and std
            mean = self.running_means.get(col, 0.0)
            variance = (
                self.running_m2.get(col, 1.0) / max(1, self.num_count - 1)
                if self.num_count > 1
                else 1.0
            )
            std = np.sqrt(max(1e-8, variance))

            # Standardization & soft outlier clipping
            z = (series_imputed - mean) / std
            z_clipped = np.clip(z, -self.clip_std_threshold, self.clip_std_threshold)

            transformed_cols.append(z_clipped.reshape(-1, 1))
            feature_names.append(f"{col}_scaled")

            # Add missing indicator if any missing values encountered
            if np.any(nan_mask):
                transformed_cols.append(nan_mask.astype(float).reshape(-1, 1))
                feature_names.append(f"{col}_isna")

        # 2. Process categorical features with smoothed target / frequency encoding
        global_target_prior = (
            self.global_target_sum / max(1, self.global_target_count)
            if self.global_target_count > 0
            else 0.5
        )

        for col in self.task_spec.categorical_features:
            if col in df.columns:
                series = df[col].fillna("__MISSING__")
            else:
                series = pd.Series(["__MISSING__"] * len(df))

            counts = self.cat_counts.get(col, {})
            target_sums = self.cat_target_sums.get(col, {})

            encoded_vals = np.empty(len(df), dtype=float)
            freq_vals = np.empty(len(df), dtype=float)

            total_cats = max(1, sum(counts.values()))
            m = self.smoothing_weight

            for i, val in enumerate(series):
                c = counts.get(val, 0)
                freq_vals[i] = c / total_cats
                if c > 0 and val in target_sums:
                    # M-estimate smoothed target encoding
                    sum_y = target_sums[val]
                    encoded_vals[i] = (sum_y + m * global_target_prior) / (c + m)
                else:
                    encoded_vals[i] = global_target_prior

            transformed_cols.append(encoded_vals.reshape(-1, 1))
            transformed_cols.append(freq_vals.reshape(-1, 1))
            feature_names.append(f"{col}_target_enc")
            feature_names.append(f"{col}_freq")

        self.feature_names_out_ = feature_names

        if not transformed_cols:
            return np.zeros((len(df), 0), dtype=float)

        return np.hstack(transformed_cols)

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def partial_fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        return self.partial_fit(X, y).transform(X)
