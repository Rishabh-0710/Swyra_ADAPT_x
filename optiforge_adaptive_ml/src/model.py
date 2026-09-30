"""
OPTIFORGE Core Continual-Learning Algorithm:
Domain-Agnostic Hierarchical Dual-Memory Expert Ensemble (HDME).
Features:
- Streaming Tabular Preprocessor (O(1) memory, Welford's moments, zero leakage)
- Stratified Density Replay Buffer (Strict K <= max_replay_samples budget)
- Multi-Tier Statistical Drift Detector (PSI, Wasserstein, Residual CUSUM)
- Dynamic Adaptation Controller (Stability-Plasticity Gated Ensemble)
- Bounded-Memory Guarantee (Historical raw batches are permanently discarded)
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.adaptation import AdaptationAction, AdaptationController
from src.drift import DriftDetector, DriftReport
from src.memory import StratifiedDensityReplayBuffer
from src.preprocessing import StreamingTabularPreprocessor
from src.task import TaskSpec, infer_task_spec
from src.utils import get_deep_memory_bytes


class OptiForgeAdaptiveModel:
    """
    Production-grade domain-agnostic adaptive ML model for continual learning
    on sequential tabular data streams under arbitrary distribution shifts.
    """

    def __init__(
        self,
        task_spec: Optional[TaskSpec] = None,
        max_replay_samples: int = 500,
        base_learning_rate: float = 0.05,
        base_trees: int = 35,
        max_trees_anchor: int = 150,
        max_trees_specialist: int = 60,
        l2_regularization: float = 1.5,
        random_seed: int = 42,
    ) -> None:
        self.task_spec = task_spec
        self.max_replay_samples = max_replay_samples
        self.base_learning_rate = base_learning_rate
        self.base_trees = base_trees
        self.max_trees_anchor = max_trees_anchor
        self.max_trees_specialist = max_trees_specialist
        self.l2_regularization = l2_regularization
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Core Modular Subsystems
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.replay_buffer: Optional[StratifiedDensityReplayBuffer] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.adaptation_controller: Optional[AdaptationController] = None

        # Predictive Models: Dual-Memory (Anchor Trunk + Fast Specialist)
        self.anchor_model: Optional[Any] = None
        self.specialist_model: Optional[Any] = None

        # Dynamic Gating Weights
        self.anchor_weight: float = 1.0
        self.specialist_weight: float = 0.0

        # State and Diagnostic History
        self.is_initialized: bool = False
        self.update_count: int = 0
        self.total_samples_seen: int = 0
        self.drift_history: List[DriftReport] = []
        self.adaptation_history: List[AdaptationAction] = []
        self.update_times: List[float] = []

    def _build_base_estimator(
        self,
        n_estimators: int,
        learning_rate: float,
        l2_reg: float,
    ) -> Any:
        """Instantiate an optimized gradient boosted tree estimator."""
        assert self.task_spec is not None
        try:
            if self.task_spec.task_type == "regression":
                return LGBMRegressor(
                    n_estimators=max(10, n_estimators),
                    learning_rate=learning_rate,
                    reg_lambda=l2_reg,
                    random_state=self.random_seed,
                    verbose=-1,
                    n_jobs=1,
                )
            else:
                return LGBMClassifier(
                    n_estimators=max(10, n_estimators),
                    learning_rate=learning_rate,
                    reg_lambda=l2_reg,
                    random_state=self.random_seed,
                    verbose=-1,
                    n_jobs=1,
                )
        except Exception:
            # Fallback to scikit-learn HistGradientBoosting if lightgbm backend has issue
            if self.task_spec.task_type == "regression":
                return HistGradientBoostingRegressor(
                    max_iter=max(10, n_estimators),
                    learning_rate=learning_rate,
                    l2_regularization=l2_reg,
                    random_state=self.random_seed,
                )
            else:
                return HistGradientBoostingClassifier(
                    max_iter=max(10, n_estimators),
                    learning_rate=learning_rate,
                    l2_regularization=l2_reg,
                    random_state=self.random_seed,
                )

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> OptiForgeAdaptiveModel:
        """
        Initialize the model on the initial batch (Batch 0).
        Learns the initial task, configures the drift baseline, and seeds replay memory.
        """
        t_start = time.perf_counter()

        # 1. Infer or validate task specification
        if task_spec is not None:
            self.task_spec = task_spec
        elif self.task_spec is None:
            self.task_spec = infer_task_spec(X_initial, y_initial)

        # 2. Instantiate subsystems
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        self.replay_buffer = StratifiedDensityReplayBuffer(
            task_spec=self.task_spec,
            max_samples=self.max_replay_samples,
            random_seed=self.random_seed,
        )
        self.drift_detector = DriftDetector()
        self.adaptation_controller = AdaptationController(
            base_lr=self.base_learning_rate,
            base_trees=self.base_trees,
            base_l2=self.l2_regularization,
        )

        # 3. Fit preprocessor and transform initial batch
        X_trans = self.preprocessor.fit_transform(X_initial, y_initial)
        y_arr = np.asarray(y_initial, dtype=float)
        self.total_samples_seen = len(X_trans)

        # 4. Train Stable Anchor Trunk
        self.anchor_model = self._build_base_estimator(
            n_estimators=self.max_trees_anchor,
            learning_rate=self.base_learning_rate,
            l2_reg=self.l2_regularization * 1.5,
        )
        self.anchor_model.fit(X_trans, y_arr)

        # Compute initial residuals for concept drift baseline
        if self.task_spec.task_type == "regression":
            preds_init = self.anchor_model.predict(X_trans)
            residuals_init = y_arr - preds_init
        else:
            probs_init = self.anchor_model.predict_proba(X_trans)[:, 1]
            residuals_init = y_arr - probs_init

        # 5. Calibrate Drift Detector baseline
        self.drift_detector.fit_reference(X_trans, residuals_init)

        # 6. Seed Replay Buffer (strictly bounded at max_replay_samples)
        # Note: We store transformed features to minimize memory footprint
        self.replay_buffer.update(X_trans, y_arr)

        self.anchor_weight = 1.0
        self.specialist_weight = 0.0
        self.specialist_model = None
        self.is_initialized = True
        self.update_count = 0

        self.update_times.append(time.perf_counter() - t_start)
        return self

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> OptiForgeAdaptiveModel:
        """
        Incrementally update the model with a sequential batch of new labeled data.
        NEVER retains previous raw datasets; only accesses current batch and bounded replay buffer.
        """
        if not self.is_initialized or self.preprocessor is None or self.replay_buffer is None:
            raise RuntimeError("Model must be initialized with `initialize()` before calling `update()`.")

        t_start = time.perf_counter()
        self.update_count += 1
        y_batch_arr = np.asarray(y_batch, dtype=float)
        n_incoming = len(y_batch_arr)
        self.total_samples_seen += n_incoming

        # 1. Transform incoming batch using current preprocessor
        X_batch_trans = self.preprocessor.transform(X_batch)

        # 2. Inspect drift on incoming batch BEFORE updating model parameters
        # Calculate pre-update prediction residuals to detect Concept Drift P(Y|X)
        if self.task_spec.task_type == "regression":
            pre_preds = self._predict_transformed(X_batch_trans)
            residuals_batch = y_batch_arr - pre_preds
        else:
            pre_probs = self._predict_proba_transformed(X_batch_trans)[:, 1]
            residuals_batch = y_batch_arr - pre_probs

        drift_report = self.drift_detector.evaluate(
            X_curr=X_batch_trans,
            residuals_curr=residuals_batch,
            feature_names=self.preprocessor.feature_names_out_,
        )
        self.drift_history.append(drift_report)

        # 3. Dynamic Adaptation Controller decides the response
        action = self.adaptation_controller.compute_action(drift_report)
        self.adaptation_history.append(action)

        # 4. Incrementally update preprocessor statistics with new batch
        self.preprocessor.partial_fit(X_batch, y_batch)

        # Re-transform current batch with slightly updated moments
        X_batch_trans = self.preprocessor.transform(X_batch)

        # 5. Retrieve bounded replay samples
        X_rep, y_rep = self.replay_buffer.get_data()
        has_replay = len(X_rep) > 0

        # Construct training datasets for dual-memory update:
        # For the Specialist: focuses on new batch + targeted replay subset
        if has_replay:
            n_rep_specialist = min(
                len(X_rep),
                max(10, int(len(X_batch_trans) * action.replay_blend_ratio)),
            )
            rep_idx = self.rng.choice(len(X_rep), size=n_rep_specialist, replace=False)
            X_spec_train = np.vstack([X_batch_trans, X_rep[rep_idx]])
            y_spec_train = np.concatenate([y_batch_arr, y_rep[rep_idx]])
        else:
            X_spec_train = X_batch_trans
            y_spec_train = y_batch_arr

        # 6. Train / Refresh Specialist Head
        # Trains on new distribution with dynamic learning rate & trees dictated by drift
        self.specialist_model = self._build_base_estimator(
            n_estimators=min(self.max_trees_specialist, action.n_new_trees),
            learning_rate=action.learning_rate,
            l2_reg=action.l2_regularization,
        )
        self.specialist_model.fit(X_spec_train, y_spec_train)

        # 7. Update Anchor Trunk Conservatively (Stability Reinforcement)
        # Anchor is updated only with high replay representation to prevent catastrophic forgetting
        if has_replay:
            # Anchor trains on replay + conservative fraction of incoming batch
            n_inc_anchor = max(10, int(len(X_batch_trans) * 0.35))
            inc_idx = self.rng.choice(len(X_batch_trans), size=n_inc_anchor, replace=False)
            X_anc_train = np.vstack([X_rep, X_batch_trans[inc_idx]])
            y_anc_train = np.concatenate([y_rep, y_batch_arr[inc_idx]])

            # Fit regularized anchor update
            anchor_candidate = self._build_base_estimator(
                n_estimators=self.max_trees_anchor,
                learning_rate=self.base_learning_rate * 0.8,
                l2_reg=self.l2_regularization * 2.0,
            )
            anchor_candidate.fit(X_anc_train, y_anc_train)
            self.anchor_model = anchor_candidate

        # 8. Update Replay Buffer with new samples (strictly maintains <= max_replay_samples)
        self.replay_buffer.update(X_batch_trans, y_batch_arr)

        # 9. Update Drift Detector Reference
        self.drift_detector.update_reference(X_batch_trans)

        # 10. Update Gating Blending Weights
        self.anchor_weight = action.anchor_weight
        self.specialist_weight = action.specialist_weight

        # Explicitly release local references to incoming batch objects to guarantee memory cleanliness
        del X_batch_trans, y_batch_arr, X_spec_train, y_spec_train

        self.update_times.append(time.perf_counter() - t_start)
        return self

    def _predict_transformed(self, X_trans: np.ndarray) -> np.ndarray:
        """Internal helper for predictions on already transformed matrices."""
        p_anc = self.anchor_model.predict(X_trans)
        if self.specialist_model is not None and self.specialist_weight > 0.0:
            p_spec = self.specialist_model.predict(X_trans)
            return self.anchor_weight * p_anc + self.specialist_weight * p_spec
        return p_anc

    def _predict_proba_transformed(self, X_trans: np.ndarray) -> np.ndarray:
        """Internal helper for class probabilities on already transformed matrices."""
        p_anc = self.anchor_model.predict_proba(X_trans)
        if self.specialist_model is not None and self.specialist_weight > 0.0:
            p_spec = self.specialist_model.predict_proba(X_trans)
            blended = self.anchor_weight * p_anc + self.specialist_weight * p_spec
            # Renormalize to ensure valid probability simplex
            sums = np.sum(blended, axis=1, keepdims=True)
            return blended / np.maximum(1e-12, sums)
        return p_anc

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions for arbitrary test data using current adaptive model state.
        """
        if not self.is_initialized or self.preprocessor is None:
            raise RuntimeError("Model is not initialized.")

        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self._predict_transformed(X_trans)
        else:
            probs = self._predict_proba_transformed(X_trans)
            return (probs[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate calibrated class probabilities for classification tasks.
        """
        if not self.is_initialized or self.preprocessor is None:
            raise RuntimeError("Model is not initialized.")
        if self.task_spec.task_type != "binary_classification":
            raise ValueError("predict_proba is only available for classification tasks.")

        X_trans = self.preprocessor.transform(X)
        return self._predict_proba_transformed(X_trans)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive memory accounting and footprint diagnostics.
        """
        rep_stats = self.replay_buffer.get_memory_stats() if self.replay_buffer else {}
        return {
            "replay_buffer": rep_stats,
            "total_samples_observed": self.total_samples_seen,
            "update_count": self.update_count,
            "anchor_trees": getattr(self.anchor_model, "n_estimators", 0),
            "specialist_trees": getattr(self.specialist_model, "n_estimators", 0) if self.specialist_model else 0,
            "gating_weights": {
                "anchor": round(self.anchor_weight, 3),
                "specialist": round(self.specialist_weight, 3),
            },
        }

    def get_drift_history(self) -> List[DriftReport]:
        return list(self.drift_history)

    def get_adaptation_history(self) -> List[AdaptationAction]:
        return list(self.adaptation_history)
