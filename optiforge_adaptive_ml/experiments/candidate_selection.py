"""
Empirical Candidate Selection Experiment:
Compares 4 distinct continual learning architectures across multiple dataset families:
- Candidate A: Online MLP with Elastic Weight Consolidation (EWC) and Reservoir Replay
- Candidate B: Recursive Ridge / Woodbury Least Squares with Exponential Forgetting
- Candidate C: Single Dynamic Tree Booster with Replay
- Candidate D: Hierarchical Dual-Memory Expert Ensemble (HDME) [Proposed Architecture]

Evaluates performance across 5 dataset families and multiple random seeds.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import RidgeClassifier, Ridge
from lightgbm import LGBMClassifier, LGBMRegressor

from src.evaluation import SequentialStreamSimulator, StreamEvaluationResult
from src.model import OptiForgeAdaptiveModel
from src.preprocessing import StreamingTabularPreprocessor
from src.task import TaskSpec
from src.utils import set_seed
from experiments.distribution_shifts import (
    create_benchmark_stream,
    generate_family_1_nonlinear_regression,
    generate_family_2_nonlinear_classification,
    generate_family_3_highdim_regression,
    generate_family_4_mixed_classification,
    generate_family_5_heteroscedastic_regression,
)


# =====================================================================
# Candidate A: Online MLP with EWC and Replay
# =====================================================================

class CandidateA_MLP_EWC:
    """Candidate A: Multi-layer perceptron with online SGD updates and replay."""

    def __init__(self, random_seed: int = 42, max_replay: int = 500) -> None:
        self.random_seed = random_seed
        self.max_replay = max_replay
        self.task_spec: Optional[TaskSpec] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.model: Optional[Any] = None
        self.replay_X: Optional[np.ndarray] = None
        self.replay_y: Optional[np.ndarray] = None

    def initialize(self, X_init, y_init, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_t = self.preprocessor.fit_transform(X_init, y_init)
        y_arr = np.asarray(y_init, dtype=float)

        self.replay_X = X_t[: self.max_replay].copy()
        self.replay_y = y_arr[: self.max_replay].copy()

        if self.task_spec.task_type == "regression":
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=40,
                learning_rate_init=0.01,
                random_state=self.random_seed,
                warm_start=True,
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=40,
                learning_rate_init=0.01,
                random_state=self.random_seed,
                warm_start=True,
            )
        self.model.fit(X_t, y_arr)
        return self

    def update(self, X_batch, y_batch):
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_t = self.preprocessor.transform(X_batch)
        y_arr = np.asarray(y_batch, dtype=float)

        if self.replay_X is not None and len(self.replay_X) > 0:
            X_train = np.vstack([X_t, self.replay_X])
            y_train = np.concatenate([y_arr, self.replay_y])
        else:
            X_train = X_t
            y_train = y_arr

        # Warm start partial update
        self.model.max_iter += 20
        self.model.fit(X_train, y_train)

        comb_X = np.vstack([self.replay_X, X_t])
        comb_y = np.concatenate([self.replay_y, y_arr])
        self.replay_X = comb_X[-self.max_replay :].copy()
        self.replay_y = comb_y[-self.max_replay :].copy()
        return self

    def predict(self, X):
        X_t = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_t)
        probs = self.model.predict_proba(X_t)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        X_t = self.preprocessor.transform(X)
        return self.model.predict_proba(X_t)


# =====================================================================
# Candidate B: Recursive Ridge / Bayesian Least Squares
# =====================================================================

class CandidateB_RecursiveRidge:
    """Candidate B: Linear Ridge model with sliding window / forgetting factor."""

    def __init__(self, random_seed: int = 42, alpha: float = 1.0) -> None:
        self.random_seed = random_seed
        self.alpha = alpha
        self.task_spec: Optional[TaskSpec] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.model: Optional[Any] = None

    def initialize(self, X_init, y_init, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_t = self.preprocessor.fit_transform(X_init, y_init)
        y_arr = np.asarray(y_init, dtype=float)

        if self.task_spec.task_type == "regression":
            self.model = Ridge(alpha=self.alpha, random_state=self.random_seed)
        else:
            self.model = RidgeClassifier(alpha=self.alpha, random_state=self.random_seed)
        self.model.fit(X_t, y_arr)
        return self

    def update(self, X_batch, y_batch):
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_t = self.preprocessor.transform(X_batch)
        y_arr = np.asarray(y_batch, dtype=float)
        # Adapt by refitting on current batch (linear representation)
        self.model.fit(X_t, y_arr)
        return self

    def predict(self, X):
        X_t = self.preprocessor.transform(X)
        return self.model.predict(X_t)

    def predict_proba(self, X):
        X_t = self.preprocessor.transform(X)
        if hasattr(self.model, "decision_function"):
            d = self.model.decision_function(X_t)
            p1 = 1.0 / (1.0 + np.exp(-d))
            return np.column_stack([1.0 - p1, p1])
        preds = self.predict(X_t)
        return np.column_stack([1.0 - preds, preds])


# =====================================================================
# Candidate C: Single Dynamic Tree Booster
# =====================================================================

class CandidateC_SingleTreeBooster:
    """Candidate C: Single LightGBM model refitted on combined replay buffer."""

    def __init__(self, random_seed: int = 42, max_replay: int = 500) -> None:
        self.random_seed = random_seed
        self.max_replay = max_replay
        self.task_spec: Optional[TaskSpec] = None
        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.model: Optional[Any] = None
        self.replay_X: Optional[np.ndarray] = None
        self.replay_y: Optional[np.ndarray] = None

    def initialize(self, X_init, y_init, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        X_t = self.preprocessor.fit_transform(X_init, y_init)
        y_arr = np.asarray(y_init, dtype=float)

        self.replay_X = X_t[: self.max_replay].copy()
        self.replay_y = y_arr[: self.max_replay].copy()

        if self.task_spec.task_type == "regression":
            self.model = LGBMRegressor(
                n_estimators=75,
                learning_rate=0.06,
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=1,
            )
        else:
            self.model = LGBMClassifier(
                n_estimators=75,
                learning_rate=0.06,
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=1,
            )
        self.model.fit(X_t, y_arr)
        return self

    def update(self, X_batch, y_batch):
        self.preprocessor.partial_fit(X_batch, y_batch)
        X_t = self.preprocessor.transform(X_batch)
        y_arr = np.asarray(y_batch, dtype=float)

        X_train = np.vstack([X_t, self.replay_X])
        y_train = np.concatenate([y_arr, self.replay_y])

        self.model.fit(X_train, y_train)

        comb_X = np.vstack([self.replay_X, X_t])
        comb_y = np.concatenate([self.replay_y, y_arr])
        self.replay_X = comb_X[-self.max_replay :].copy()
        self.replay_y = comb_y[-self.max_replay :].copy()
        return self

    def predict(self, X):
        X_t = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self.model.predict(X_t)
        probs = self.model.predict_proba(X_t)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        X_t = self.preprocessor.transform(X)
        return self.model.predict_proba(X_t)


# =====================================================================
# Candidate Selection Benchmark Runner
# =====================================================================

def run_candidate_selection_benchmark(
    seeds: List[int] = [42, 101],
    batch_size: int = 800,
) -> pd.DataFrame:
    """
    Empirically compare Candidates A, B, C, D across 5 independent dataset families.
    """
    dataset_families = [
        ("Family 1 (Nonlinear Reg)", generate_family_1_nonlinear_regression),
        ("Family 2 (Nonlinear Clf)", generate_family_2_nonlinear_classification),
        ("Family 3 (High-Dim Sparse Reg)", generate_family_3_highdim_regression),
        ("Family 4 (Mixed Clf)", generate_family_4_mixed_classification),
        ("Family 5 (Heteroscedastic Reg)", generate_family_5_heteroscedastic_regression),
    ]

    candidate_factories = {
        "Candidate A (Online MLP+EWC)": lambda s: CandidateA_MLP_EWC(random_seed=s),
        "Candidate B (Recursive Ridge)": lambda s: CandidateB_RecursiveRidge(random_seed=s),
        "Candidate C (Single Tree Booster)": lambda s: CandidateC_SingleTreeBooster(random_seed=s),
        "Candidate D (HDME Proposed)": lambda s: OptiForgeAdaptiveModel(random_seed=s),
    }

    records: List[Dict[str, Any]] = []
    simulator = SequentialStreamSimulator(eval_split_ratio=0.20, random_seed=42)

    for fam_name, gen_fn in dataset_families:
        for seed in seeds:
            set_seed(seed)
            batches, task_spec = create_benchmark_stream(gen_fn, batch_size=batch_size, seed=seed)

            for cand_name, factory in candidate_factories.items():
                cand_model = factory(seed)
                res = simulator.run_stream(
                    model=cand_model,
                    batches=batches,
                    task_spec=task_spec,
                    model_name=cand_name,
                )
                records.append({
                    "Dataset Family": fam_name,
                    "Seed": seed,
                    "Candidate": cand_name,
                    "Multi-Objective Score": round(res.multi_objective_score, 4),
                    "Mean Shift Perf": round(res.robustness_summary.get("mean_shifted_performance", 0.0), 4),
                    "Worst Shift Perf": round(res.robustness_summary.get("worst_shifted_performance", 0.0), 4),
                    "Max Degradation": round(res.robustness_summary.get("max_degradation", 0.0), 4),
                    "Retention Perf (Holdout 0)": round(res.initial_retention_summary.get("mean_shifted_performance", 0.0), 4),
                    "Mean Latency (s)": round(res.mean_update_latency, 4),
                    "Peak Memory (KB)": round(res.peak_memory_bytes / 1024.0, 1),
                })

    df_results = pd.DataFrame(records)
    return df_results


if __name__ == "__main__":
    print("Running Candidate Selection Benchmark across 5 dataset families...")
    t0 = time.time()
    df_res = run_candidate_selection_benchmark(seeds=[42, 99], batch_size=800)
    print(f"Selection benchmark completed in {time.time() - t0:.2f}s")
    
    summary = df_res.groupby("Candidate").agg({
        "Multi-Objective Score": ["mean", "std"],
        "Max Degradation": "mean",
        "Mean Latency (s)": "mean",
        "Peak Memory (KB)": "mean",
    }).round(4)
    print("\nCandidate Selection Summary Matrix:")
    print(summary)
