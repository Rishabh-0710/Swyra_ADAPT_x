"""
Bounded Memory Management: Stratified Density and Diversity-Preserving Replay Buffer.
Strictly respects memory budgets (K <= max_replay_samples) with explicit accounting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.task import TaskSpec
from src.utils import get_deep_memory_bytes


class StratifiedDensityReplayBuffer:
    """
    Intelligent replay buffer that stores a representative historical subset
    of samples strictly bounded by max_replay_samples.

    Strategies:
    - Target Stratification: Partitions target space into discrete strata
      (quantiles for regression, discrete classes for classification).
    - Diversity-Preserving Pruning: Within each stratum, uses greedy farthest-point
      sampling / distance diversity to preserve maximal spatial coverage.
    - Explicit Memory Accounting: Guarantees zero historical leakage outside the buffer.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        max_samples: int = 500,
        n_strata: int = 10,
        random_seed: int = 42,
    ) -> None:
        self.task_spec = task_spec
        self.max_samples = max(10, int(max_samples))
        self.n_strata = max(2, int(n_strata))
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Internal bounded storage
        self._X_buffer: Optional[np.ndarray] = None
        self._y_buffer: Optional[np.ndarray] = None
        self._strata_buffer: Optional[np.ndarray] = None

        # Cumulative lifetime statistics (O(1) memory)
        self.total_samples_observed: int = 0
        self.update_count: int = 0

    @property
    def current_size(self) -> int:
        return 0 if self._X_buffer is None else len(self._X_buffer)

    def get_memory_bytes(self) -> int:
        """Explicit accounting of total bytes allocated by the replay buffer."""
        mem = 0
        if self._X_buffer is not None:
            mem += self._X_buffer.nbytes
        if self._y_buffer is not None:
            mem += self._y_buffer.nbytes
        if self._strata_buffer is not None:
            mem += self._strata_buffer.nbytes
        return mem

    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Return memory auditing statistics."""
        return {
            "current_samples": self.current_size,
            "max_samples_capacity": self.max_samples,
            "capacity_utilization_pct": round(
                100.0 * self.current_size / self.max_samples, 2
            ),
            "memory_bytes": self.get_memory_bytes(),
            "total_observed_lifetime": self.total_samples_observed,
            "update_count": self.update_count,
        }

    def _assign_strata(self, y: np.ndarray) -> np.ndarray:
        """Assign stratum index for each target value."""
        if self.task_spec.task_type == "binary_classification":
            # For binary classification, strata are simply class 0 and class 1
            return (y > 0.5).astype(int)
        else:
            # For continuous regression, assign quantile bins
            # Use percentiles derived from current batch
            try:
                quantiles = np.linspace(0, 100, self.n_strata + 1)
                bin_edges = np.percentile(y, quantiles)
                bin_edges[0] -= 1e-6
                bin_edges[-1] += 1e-6
                strata = np.digitize(y, bin_edges) - 1
                return np.clip(strata, 0, self.n_strata - 1)
            except Exception:
                return np.zeros(len(y), dtype=int)

    def _farthest_point_sampling(
        self, X_pool: np.ndarray, quota: int
    ) -> np.ndarray:
        """
        Greedy farthest point sampling to select diverse prototypes within a stratum.
        Time complexity: O(N * quota * D).
        """
        n_candidates = len(X_pool)
        if n_candidates <= quota:
            return np.arange(n_candidates)

        # Pick first point closest to the stratum centroid
        centroid = np.mean(X_pool, axis=0, keepdims=True)
        init_dists = np.sum((X_pool - centroid) ** 2, axis=1)
        selected = [int(np.argmin(init_dists))]

        # Track min distance of all points to current selected set
        min_distances = np.sum((X_pool - X_pool[selected[0]]) ** 2, axis=1)

        for _ in range(1, quota):
            # Select point with maximum distance to existing selected set
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)
            # Update min distances
            new_dists = np.sum((X_pool - X_pool[next_idx]) ** 2, axis=1)
            min_distances = np.minimum(min_distances, new_dists)

        return np.array(selected, dtype=int)

    def update(
        self, X_new: np.ndarray, y_new: np.ndarray
    ) -> StratifiedDensityReplayBuffer:
        """
        Ingest new sequential batch, combine with current buffer, and prune back
        to <= max_samples using stratified diversity selection.
        """
        X_arr = np.asarray(X_new, dtype=float)
        y_arr = np.asarray(y_new, dtype=float)
        n_incoming = len(X_arr)

        self.total_samples_observed += n_incoming
        self.update_count += 1

        new_strata = self._assign_strata(y_arr)

        if self._X_buffer is None:
            # First batch initialization
            combined_X = X_arr
            combined_y = y_arr
            combined_strata = new_strata
        else:
            # Concatenate existing buffer with incoming batch
            combined_X = np.vstack([self._X_buffer, X_arr])
            combined_y = np.concatenate([self._y_buffer, y_arr])
            # Recompute strata over combined distribution for stability
            combined_strata = self._assign_strata(combined_y)

        n_combined = len(combined_X)
        if n_combined <= self.max_samples:
            self._X_buffer = combined_X.copy()
            self._y_buffer = combined_y.copy()
            self._strata_buffer = combined_strata.copy()
            return self

        # Stratified diversity pruning:
        # Determine quota per stratum
        unique_strata, stratum_counts = np.unique(combined_strata, return_counts=True)
        n_active_strata = len(unique_strata)

        # Baseline quota per active stratum
        base_quota = self.max_samples // n_active_strata
        remainder = self.max_samples % n_active_strata

        selected_indices: List[int] = []

        for i, s in enumerate(unique_strata):
            stratum_mask = np.where(combined_strata == s)[0]
            quota_s = base_quota + (1 if i < remainder else 0)

            if len(stratum_mask) <= quota_s:
                selected_indices.extend(stratum_mask.tolist())
            else:
                # Run farthest point diversity sampling on features in this stratum
                X_s = combined_X[stratum_mask]
                sub_picks = self._farthest_point_sampling(X_s, quota_s)
                selected_indices.extend(stratum_mask[sub_picks].tolist())

        selected_idx = np.array(selected_indices, dtype=int)
        # Ensure we never exceed max_samples
        if len(selected_idx) > self.max_samples:
            selected_idx = self.rng.choice(
                selected_idx, size=self.max_samples, replace=False
            )

        # Save compacted buffer copies; release old references
        self._X_buffer = np.ascontiguousarray(combined_X[selected_idx].copy())
        self._y_buffer = np.ascontiguousarray(combined_y[selected_idx].copy())
        self._strata_buffer = np.ascontiguousarray(combined_strata[selected_idx].copy())

        # Assert memory invariant
        assert len(self._X_buffer) <= self.max_samples, (
            f"Buffer overflow: {len(self._X_buffer)} > {self.max_samples}"
        )

        return self

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve a copy of the replay buffer (X, y).
        If buffer is empty, returns empty arrays.
        """
        if self._X_buffer is None or len(self._X_buffer) == 0:
            return np.empty((0, 0)), np.empty(0)
        return self._X_buffer.copy(), self._y_buffer.copy()
