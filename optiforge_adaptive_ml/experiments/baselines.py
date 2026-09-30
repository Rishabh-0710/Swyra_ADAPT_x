"""
Benchmark Baselines for OPTIFORGE Evaluation:
- Baseline 1: Static Model (train once on Batch 0, never updates)
- Baseline 2: Naive Incremental Model (trains solely on incoming batch, zero memory)
- Baseline 3: Full Retrain Oracle (trains on all accumulated data from scratch; performance upper-bound)
- Baseline 4: Standard Replay Model (naive FIFO replay without drift-adaptive control)
"""

from __future__ import annotations

from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.preprocessing import StreamingTabularPreprocessor
from src.task import TaskSpec, infer_task_spec
from src.utils import get_deep_memory_bytes


def _get_estimator(task_spec: TaskSpec, n_estimators: int = 100, seed: int = 42) -> Any:
    """Build LightGBM baseline estimator."""
    if task_spec.task_type == "regression":
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        random_state=seed,
        verbose=-1,
        n_jobs=1,
    )


class StaticModel:
    """
    Baseline 1: Train once on Batch 0 and freeze parameters permanently.
    Tests model vulnerability to distribution shifts when adaptation is absent.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.model: Optional[Any] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.task_spec: Optional[TaskSpec] = None

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> StaticModel:
        self.task_spec = task_spec or infer_task_spec(X_initial, y_initial)
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_trans = self.preprocessor.fit_transform(X_initial, y_initial)
        self.model = _get_estimator(self.task_spec, n_estimators=100, seed=self.random_seed)
        self.model.fit(X_trans, np.asarray(y_initial, dtype=float))
        return self

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> StaticModel:
        # Static: Ignore incoming batch completely
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)


class NaiveIncrementalModel:
    """
    Baseline 2: Fits solely on the incoming batch, completely forgetting prior batches.
    Exhibits classic catastrophic forgetting.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.model: Optional[Any] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.task_spec: Optional[TaskSpec] = None

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> NaiveIncrementalModel:
        self.task_spec = task_spec or infer_task_spec(X_initial, y_initial)
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_trans = self.preprocessor.fit_transform(X_initial, y_initial)
        self.model = _get_estimator(self.task_spec, n_estimators=60, seed=self.random_seed)
        self.model.fit(X_trans, np.asarray(y_initial, dtype=float))
        return self

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> NaiveIncrementalModel:
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_trans = self.preprocessor.transform(X_batch)
        # Retrain model only on current batch
        self.model = _get_estimator(self.task_spec, n_estimators=60, seed=self.random_seed)
        self.model.fit(X_trans, np.asarray(y_batch, dtype=float))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)


class FullRetrainOracle:
    """
    Baseline 3: Retrains on all accumulated historical data from scratch.
    Violates continual learning constraints; serves as performance ceiling/oracle.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.model: Optional[Any] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.task_spec: Optional[TaskSpec] = None
        self.all_X: List[np.ndarray] = []
        self.all_y: List[np.ndarray] = []

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> FullRetrainOracle:
        self.task_spec = task_spec or infer_task_spec(X_initial, y_initial)
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_trans = self.preprocessor.fit_transform(X_initial, y_initial)
        y_arr = np.asarray(y_initial, dtype=float)
        self.all_X = [X_trans]
        self.all_y = [y_arr]

        self.model = _get_estimator(self.task_spec, n_estimators=100, seed=self.random_seed)
        self.model.fit(X_trans, y_arr)
        return self

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> FullRetrainOracle:
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_trans = self.preprocessor.transform(X_batch)
        y_arr = np.asarray(y_batch, dtype=float)
        self.all_X.append(X_trans)
        self.all_y.append(y_arr)

        X_cum = np.vstack(self.all_X)
        y_cum = np.concatenate(self.all_y)

        self.model = _get_estimator(self.task_spec, n_estimators=100, seed=self.random_seed)
        self.model.fit(X_cum, y_cum)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)


class StandardReplayModel:
    """
    Baseline 4: Incremental model with naive FIFO replay buffer (K=500).
    No drift detection, no adaptive scaling, no diversity stratification.
    """

    def __init__(self, max_replay_samples: int = 500, random_seed: int = 42) -> None:
        self.max_replay_samples = max_replay_samples
        self.random_seed = random_seed
        self.model: Optional[Any] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.task_spec: Optional[TaskSpec] = None
        self.replay_X: Optional[np.ndarray] = None
        self.replay_y: Optional[np.ndarray] = None

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> StandardReplayModel:
        self.task_spec = task_spec or infer_task_spec(X_initial, y_initial)
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_trans = self.preprocessor.fit_transform(X_initial, y_initial)
        y_arr = np.asarray(y_initial, dtype=float)

        self.replay_X = X_trans[: self.max_replay_samples].copy()
        self.replay_y = y_arr[: self.max_replay_samples].copy()

        self.model = _get_estimator(self.task_spec, n_estimators=80, seed=self.random_seed)
        self.model.fit(X_trans, y_arr)
        return self

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> StandardReplayModel:
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_trans = self.preprocessor.transform(X_batch)
        y_arr = np.asarray(y_batch, dtype=float)

        # Naively mix incoming batch with replay buffer
        if self.replay_X is not None and len(self.replay_X) > 0:
            X_train = np.vstack([X_trans, self.replay_X])
            y_train = np.concatenate([y_arr, self.replay_y])
        else:
            X_train = X_trans
            y_train = y_arr

        self.model = _get_estimator(self.task_spec, n_estimators=80, seed=self.random_seed)
        self.model.fit(X_train, y_train)

        # FIFO replacement in replay buffer
        comb_X = np.vstack([self.replay_X, X_trans])
        comb_y = np.concatenate([self.replay_y, y_arr])
        # Keep the most recent max_replay_samples
        self.replay_X = comb_X[-self.max_replay_samples :].copy()
        self.replay_y = comb_y[-self.max_replay_samples :].copy()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)
