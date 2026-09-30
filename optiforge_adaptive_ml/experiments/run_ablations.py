"""
Comprehensive Ablation Study for OPTIFORGE:
Evaluates individual contributions of each subsystem:
- Full Proposed Algorithm (HDME + Drift Detection + Stratified Diversity Replay + Dynamic Gating)
- Ablation A: No Drift Detection (Fixed constant adaptation response)
- Ablation B: No Replay Buffer (Memory capacity K = 0)
- Ablation C: No Dynamic Adaptation Scaling (Fixed trees and learning rate)
- Ablation D: No Forgetting Prevention (Pure specialist head, anchor disabled)
- Ablation E: No Memory Selection (Uniform random replacement instead of stratified diversity)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from experiments.distribution_shifts import (
    create_benchmark_stream,
    generate_family_1_nonlinear_regression,
    generate_family_2_nonlinear_classification,
    generate_family_4_mixed_classification,
)
from src.adaptation import AdaptationAction
from src.drift import DriftReport
from src.evaluation import SequentialStreamSimulator
from src.model import OptiForgeAdaptiveModel
from src.task import TaskSpec
from src.utils import set_seed


class AblationNoDriftModel(OptiForgeAdaptiveModel):
    """Ablation A: Drift detection disabled; fixed generic low/medium update every time."""

    def update(self, X_batch, y_batch):
        # Override update to bypass drift inspection
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized.")
        t_start = time.perf_counter()
        self.update_count += 1
        y_batch_arr = np.asarray(y_batch, dtype=float)
        self.total_samples_seen += len(y_batch_arr)

        X_batch_trans = self.preprocessor.transform(X_batch)

        # Fixed default action regardless of data distribution
        fixed_report = DriftReport(overall_drift_score=0.20, severity="medium", is_concept_drift=False, concept_drift_score=0.0)
        self.drift_history.append(fixed_report)
        action = AdaptationAction(
            adaptation_rate=0.5,
            replay_blend_ratio=0.30,
            specialist_weight=0.35,
            anchor_weight=0.65,
            n_new_trees=35,
            learning_rate=self.base_learning_rate,
            l2_regularization=self.l2_regularization,
            strategy_description="Fixed no-drift schedule",
        )
        self.adaptation_history.append(action)

        self.preprocessor.partial_fit(X_batch, y_batch)
        X_batch_trans = self.preprocessor.transform(X_batch)

        X_rep, y_rep = self.replay_buffer.get_data()
        if len(X_rep) > 0:
            X_spec_train = np.vstack([X_batch_trans, X_rep[: len(X_batch_trans) // 3]])
            y_spec_train = np.concatenate([y_batch_arr, y_rep[: len(X_batch_trans) // 3]])
            X_anc_train = np.vstack([X_rep, X_batch_trans[: len(X_batch_trans) // 3]])
            y_anc_train = np.concatenate([y_rep, y_batch_arr[: len(X_batch_trans) // 3]])
        else:
            X_spec_train = X_batch_trans
            y_spec_train = y_batch_arr
            X_anc_train = X_batch_trans
            y_anc_train = y_batch_arr

        self.specialist_model = self._build_base_estimator(35, self.base_learning_rate, self.l2_regularization)
        self.specialist_model.fit(X_spec_train, y_spec_train)

        anchor_candidate = self._build_base_estimator(self.max_trees_anchor, self.base_learning_rate * 0.8, self.l2_regularization * 2.0)
        anchor_candidate.fit(X_anc_train, y_anc_train)
        self.anchor_model = anchor_candidate

        self.replay_buffer.update(X_batch_trans, y_batch_arr)
        self.anchor_weight = 0.65
        self.specialist_weight = 0.35

        self.update_times.append(time.perf_counter() - t_start)
        return self


class AblationNoReplayModel(OptiForgeAdaptiveModel):
    """Ablation B: No replay buffer (K = 0)."""

    def __init__(self, **kwargs):
        super().__init__(max_replay_samples=10, **kwargs)

    def initialize(self, X_initial, y_initial, task_spec=None):
        super().initialize(X_initial, y_initial, task_spec)
        # Empty replay buffer
        self.replay_buffer._X_buffer = None
        self.replay_buffer._y_buffer = None
        return self

    def update(self, X_batch, y_batch):
        # Clear buffer before each update to enforce K=0
        if self.replay_buffer:
            self.replay_buffer._X_buffer = None
            self.replay_buffer._y_buffer = None
        res = super().update(X_batch, y_batch)
        if self.replay_buffer:
            self.replay_buffer._X_buffer = None
            self.replay_buffer._y_buffer = None
        return res


class AblationNoAdaptiveScalingModel(OptiForgeAdaptiveModel):
    """Ablation C: No dynamic hyperparameter scaling (fixed trees, fixed LR)."""

    def update(self, X_batch, y_batch):
        # Override controller output with static values
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized.")
        t_start = time.perf_counter()
        self.update_count += 1
        y_batch_arr = np.asarray(y_batch, dtype=float)
        self.total_samples_seen += len(y_batch_arr)

        X_batch_trans = self.preprocessor.transform(X_batch)

        if self.task_spec.task_type == "regression":
            pre_preds = self._predict_transformed(X_batch_trans)
            residuals_batch = y_batch_arr - pre_preds
        else:
            pre_probs = self._predict_proba_transformed(X_batch_trans)[:, 1]
            residuals_batch = y_batch_arr - pre_probs

        drift_report = self.drift_detector.evaluate(X_batch_trans, residuals_batch)
        self.drift_history.append(drift_report)

        # Static unscaled action
        action = AdaptationAction(
            adaptation_rate=0.5,
            replay_blend_ratio=0.30,
            specialist_weight=0.5,
            anchor_weight=0.5,
            n_new_trees=35,
            learning_rate=self.base_learning_rate,
            l2_regularization=self.l2_regularization,
            strategy_description="Unscaled static action",
        )
        self.adaptation_history.append(action)

        self.preprocessor.partial_fit(X_batch, y_batch)
        X_batch_trans = self.preprocessor.transform(X_batch)

        X_rep, y_rep = self.replay_buffer.get_data()
        has_replay = len(X_rep) > 0
        if has_replay:
            X_spec_train = np.vstack([X_batch_trans, X_rep[: len(X_batch_trans) // 3]])
            y_spec_train = np.concatenate([y_batch_arr, y_rep[: len(X_batch_trans) // 3]])
        else:
            X_spec_train = X_batch_trans
            y_spec_train = y_batch_arr

        self.specialist_model = self._build_base_estimator(35, self.base_learning_rate, self.l2_regularization)
        self.specialist_model.fit(X_spec_train, y_spec_train)

        if has_replay:
            X_anc_train = np.vstack([X_rep, X_batch_trans[: len(X_batch_trans) // 3]])
            y_anc_train = np.concatenate([y_rep, y_batch_arr[: len(X_batch_trans) // 3]])
            anchor_candidate = self._build_base_estimator(self.max_trees_anchor, self.base_learning_rate, self.l2_regularization)
            anchor_candidate.fit(X_anc_train, y_anc_train)
            self.anchor_model = anchor_candidate

        self.replay_buffer.update(X_batch_trans, y_batch_arr)
        self.anchor_weight = 0.5
        self.specialist_weight = 0.5

        self.update_times.append(time.perf_counter() - t_start)
        return self


class AblationNoForgettingPreventionModel(OptiForgeAdaptiveModel):
    """Ablation D: No Anchor Trunk; pure specialist model fitted on each batch + minimal replay."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        model = self.specialist_model if self.specialist_model is not None else self.anchor_model
        if self.task_spec.task_type == "regression":
            return model.predict(X_trans)
        probs = model.predict_proba(X_trans)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        model = self.specialist_model if self.specialist_model is not None else self.anchor_model
        return model.predict_proba(X_trans)


class AblationRandomMemoryModel(OptiForgeAdaptiveModel):
    """Ablation E: Uniform random replacement in buffer instead of stratified diversity."""

    def initialize(self, X_initial, y_initial, task_spec=None):
        super().initialize(X_initial, y_initial, task_spec)
        return self

    def update(self, X_batch, y_batch):
        # Hijack replay buffer update to use uniform random sampling
        super().update(X_batch, y_batch)
        if self.replay_buffer._X_buffer is not None and len(self.replay_buffer._X_buffer) > self.max_replay_samples:
            rand_idx = self.rng.choice(len(self.replay_buffer._X_buffer), size=self.max_replay_samples, replace=False)
            self.replay_buffer._X_buffer = self.replay_buffer._X_buffer[rand_idx].copy()
            self.replay_buffer._y_buffer = self.replay_buffer._y_buffer[rand_idx].copy()
        return self


def run_ablation_study(
    seeds: List[int] = [42, 101],
    batch_size: int = 1000,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run full ablation experiment comparing all configurations.
    """
    os.makedirs(output_dir, exist_ok=True)

    ablations = {
        "Full Proposed (OptiForge)": lambda s: OptiForgeAdaptiveModel(random_seed=s),
        "Ablation A (No Drift Detect)": lambda s: AblationNoDriftModel(random_seed=s),
        "Ablation B (No Replay Buffer)": lambda s: AblationNoReplayModel(random_seed=s),
        "Ablation C (No Adaptive Scale)": lambda s: AblationNoAdaptiveScalingModel(random_seed=s),
        "Ablation D (No Anchor Backbone)": lambda s: AblationNoForgettingPreventionModel(random_seed=s),
        "Ablation E (Random Replay)": lambda s: AblationRandomMemoryModel(random_seed=s),
    }

    test_datasets = [
        ("Family_1_Nonlinear_Reg", generate_family_1_nonlinear_regression),
        ("Family_2_Nonlinear_Clf", generate_family_2_nonlinear_classification),
        ("Family_4_Mixed_Clf", generate_family_4_mixed_classification),
    ]

    records: List[Dict[str, Any]] = []
    simulator = SequentialStreamSimulator(eval_split_ratio=0.20, random_seed=42)

    total_runs = len(test_datasets) * len(seeds) * len(ablations)
    curr = 0

    for fam_name, gen_fn in test_datasets:
        print(f"\n--- Running Ablations on {fam_name} ---")
        for seed in seeds:
            set_seed(seed)
            batches, task_spec = create_benchmark_stream(gen_fn, batch_size=batch_size, seed=seed)

            for ab_name, factory in ablations.items():
                curr += 1
                model = factory(seed)
                eval_res = simulator.run_stream(model, batches, task_spec, model_name=ab_name)

                rob = eval_res.robustness_summary
                ret = eval_res.initial_retention_summary

                records.append({
                    "dataset": fam_name,
                    "seed": seed,
                    "configuration": ab_name,
                    "multi_objective_score": round(eval_res.multi_objective_score, 4),
                    "mean_shifted_perf": round(rob.get("mean_shifted_performance", 0.0), 4),
                    "worst_shifted_perf": round(rob.get("worst_shifted_performance", 0.0), 4),
                    "max_degradation": round(rob.get("max_degradation", 0.0), 4),
                    "retention_perf": round(ret.get("mean_shifted_performance", 0.0), 4),
                    "mean_latency_sec": round(eval_res.mean_update_latency, 4),
                })
                print(
                    f"[{curr}/{total_runs}] {ab_name:32s} | Score: {eval_res.multi_objective_score:.4f} | "
                    f"Degradation: {rob.get('max_degradation', 0.0):.4f}"
                )

    df_ab = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    df_ab.to_csv(csv_path, index=False)

    summary = df_ab.groupby("configuration").agg({
        "multi_objective_score": ["mean", "std"],
        "mean_shifted_perf": "mean",
        "worst_shifted_perf": "mean",
        "max_degradation": "mean",
        "retention_perf": "mean",
    }).round(4)

    print("\nAggregated Ablation Study Summary:")
    print(summary)
    summary.to_csv(os.path.join(output_dir, "ablation_summary.csv"))
    return df_ab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OPTIFORGE Ablation Study")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101], help="Random seeds")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_ablation_study(seeds=args.seeds, batch_size=args.batch_size, output_dir=args.output_dir)
