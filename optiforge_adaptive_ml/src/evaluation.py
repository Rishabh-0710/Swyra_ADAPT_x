"""
Strict Zero-Leakage Sequential Stream Evaluator and Benchmark Harness.
Guarantees:
- Model never accesses evaluation slices
- Model never peeks into future batches
- Evaluates immediate adaptation performance AND catastrophic forgetting on past slices
- Rigorous runtime and memory profiling
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.metrics import (
    compute_multi_objective_score,
    compute_stream_robustness,
    evaluate_batch_predictions,
)
from src.task import TaskSpec
from src.utils import get_deep_memory_bytes


@dataclass
class BatchEvaluationResult:
    """Metrics recorded for an individual sequential batch."""
    batch_index: int
    environment_name: str
    n_update_samples: int
    n_eval_samples: int
    update_latency_seconds: float
    current_eval_metrics: Dict[str, Any]
    initial_retention_metrics: Dict[str, Any]  # Catastrophic forgetting tracking
    memory_stats: Dict[str, Any]
    drift_report: Optional[Dict[str, Any]] = None
    adaptation_action: Optional[Dict[str, Any]] = None


@dataclass
class StreamEvaluationResult:
    """Consolidated results of a complete sequential stream benchmark."""
    model_name: str
    task_spec: TaskSpec
    batch_results: List[BatchEvaluationResult] = field(default_factory=list)
    robustness_summary: Dict[str, float] = field(default_factory=dict)
    initial_retention_summary: Dict[str, float] = field(default_factory=dict)
    mean_update_latency: float = 0.0
    total_update_latency: float = 0.0
    peak_memory_bytes: int = 0
    multi_objective_score: float = 0.0


class SequentialStreamSimulator:
    """
    Simulates real-world sequential stream arrival with strict data leakage barriers.
    """

    def __init__(
        self,
        eval_split_ratio: float = 0.20,
        random_seed: int = 42,
    ) -> None:
        self.eval_split_ratio = eval_split_ratio
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def _split_update_and_eval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        subgroups: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any, Any, Any, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Split incoming batch into:
        1. Update subset (passed to algorithm)
        2. Hidden evaluation subset (retained strictly by evaluator)
        """
        n_samples = len(X)
        n_eval = max(10, int(n_samples * self.eval_split_ratio))

        # Random permutation
        indices = self.rng.permutation(n_samples)
        eval_idx = indices[:n_eval]
        update_idx = indices[n_eval:]

        if isinstance(X, pd.DataFrame):
            X_update = X.iloc[update_idx].reset_index(drop=True)
            X_eval = X.iloc[eval_idx].reset_index(drop=True)
        else:
            X_update = X[update_idx]
            X_eval = X[eval_idx]

        if isinstance(y, pd.Series):
            y_update = y.iloc[update_idx].reset_index(drop=True)
            y_eval = y.iloc[eval_idx].reset_index(drop=True)
        else:
            y_update = y[update_idx]
            y_eval = y[eval_idx]

        sub_update = subgroups[update_idx] if subgroups is not None else None
        sub_eval = subgroups[eval_idx] if subgroups is not None else None

        return X_update, y_update, X_eval, y_eval, sub_update, sub_eval

    def run_stream(
        self,
        model: Any,
        batches: List[Tuple[str, Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Optional[np.ndarray]]],
        task_spec: TaskSpec,
        model_name: str = "AdaptiveModel",
    ) -> StreamEvaluationResult:
        """
        Execute the full sequential continual learning benchmark across batches.
        """
        batch_results: List[BatchEvaluationResult] = []
        update_latencies: List[float] = []
        memory_records: List[int] = []

        # Reference holdout slice from initial batch for tracking catastrophic forgetting
        X_init_holdout: Optional[Any] = None
        y_init_holdout: Optional[Any] = None
        sub_init_holdout: Optional[Any] = None

        for t, (env_name, X_raw, y_raw, sub_raw) in enumerate(batches):
            # 1. Strictly split into Update (public to model) and Eval (hidden)
            (
                X_up,
                y_up,
                X_eval,
                y_eval,
                sub_up,
                sub_eval,
            ) = self._split_update_and_eval(X_raw, y_raw, sub_raw)

            if t == 0:
                # Save initial hidden evaluation slice
                X_init_holdout = X_eval
                y_init_holdout = y_eval
                sub_init_holdout = sub_eval

                # Initialize model
                t0 = time.perf_counter()
                if hasattr(model, "initialize"):
                    model.initialize(X_up, y_up, task_spec)
                else:
                    model.fit(X_up, y_up)
                latency = time.perf_counter() - t0

            else:
                # Sequential update on new batch
                t0 = time.perf_counter()
                if hasattr(model, "update"):
                    model.update(X_up, y_up)
                else:
                    model.fit(X_up, y_up)
                latency = time.perf_counter() - t0

            update_latencies.append(latency)

            # 2. Evaluate on CURRENT batch's hidden evaluation slice
            preds_eval = model.predict(X_eval)
            probs_eval = (
                model.predict_proba(X_eval)[:, 1]
                if task_spec.task_type == "binary_classification" and hasattr(model, "predict_proba")
                else None
            )
            curr_eval_metrics = evaluate_batch_predictions(
                y_true=np.asarray(y_eval),
                y_pred=np.asarray(preds_eval),
                task_spec=task_spec,
                y_prob=probs_eval,
                subgroups=sub_eval,
            )

            # 3. Evaluate on INITIAL batch's hidden evaluation slice (Catastrophic Forgetting)
            preds_init = model.predict(X_init_holdout)
            probs_init = (
                model.predict_proba(X_init_holdout)[:, 1]
                if task_spec.task_type == "binary_classification" and hasattr(model, "predict_proba")
                else None
            )
            init_eval_metrics = evaluate_batch_predictions(
                y_true=np.asarray(y_init_holdout),
                y_pred=np.asarray(preds_init),
                task_spec=task_spec,
                y_prob=probs_init,
                subgroups=sub_init_holdout,
            )

            # 4. Measure memory footprint
            if hasattr(model, "get_memory_stats"):
                mem_stats = model.get_memory_stats()
            else:
                mem_stats = {"memory_bytes": get_deep_memory_bytes(model)}

            mem_bytes = mem_stats.get("memory_bytes", get_deep_memory_bytes(model))
            memory_records.append(mem_bytes)

            # Extract drift report and action if model is OptiForgeAdaptiveModel
            drift_rep = None
            if hasattr(model, "drift_history") and model.drift_history:
                last_rep = model.drift_history[-1]
                drift_rep = {
                    "overall_score": round(last_rep.overall_drift_score, 3),
                    "severity": last_rep.severity,
                    "is_concept_drift": last_rep.is_concept_drift,
                    "top_drifted": last_rep.top_drifted_features,
                }

            act_dict = None
            if hasattr(model, "adaptation_history") and model.adaptation_history:
                last_act = model.adaptation_history[-1]
                act_dict = {
                    "adaptation_rate": round(last_act.adaptation_rate, 2),
                    "specialist_weight": round(last_act.specialist_weight, 2),
                    "anchor_weight": round(last_act.anchor_weight, 2),
                    "n_new_trees": last_act.n_new_trees,
                }

            batch_res = BatchEvaluationResult(
                batch_index=t,
                environment_name=env_name,
                n_update_samples=len(X_up),
                n_eval_samples=len(X_eval),
                update_latency_seconds=latency,
                current_eval_metrics=curr_eval_metrics,
                initial_retention_metrics=init_eval_metrics,
                memory_stats=mem_stats,
                drift_report=drift_rep,
                adaptation_action=act_dict,
            )
            batch_results.append(batch_res)

        # 5. Summarize multi-objective metrics
        curr_metrics_history = [b.current_eval_metrics for b in batch_results]
        init_metrics_history = [b.initial_retention_metrics for b in batch_results]

        robustness_summary = compute_stream_robustness(curr_metrics_history, task_spec)
        initial_retention_summary = compute_stream_robustness(init_metrics_history, task_spec)

        mean_lat = float(np.mean(update_latencies))
        tot_lat = float(np.sum(update_latencies))
        peak_mem = int(max(memory_records)) if memory_records else 0

        # Mean calibration (Brier score or ECE)
        mean_calib = 0.0
        if task_spec.task_type == "binary_classification":
            briers = [b.current_eval_metrics.get("brier_score", 0.25) for b in batch_results]
            mean_calib = float(np.mean(briers))

        # Fairness gap if available
        mean_fairness_gap = None
        gaps = [
            b.current_eval_metrics.get("subgroup_max_gap")
            for b in batch_results
            if "subgroup_max_gap" in b.current_eval_metrics
        ]
        if gaps:
            mean_fairness_gap = float(np.mean(gaps))

        mo_score = compute_multi_objective_score(
            robustness_stats=robustness_summary,
            calibration_stat=mean_calib,
            efficiency_stat=mean_lat,
            subgroup_stat=mean_fairness_gap,
        )

        return StreamEvaluationResult(
            model_name=model_name,
            task_spec=task_spec,
            batch_results=batch_results,
            robustness_summary=robustness_summary,
            initial_retention_summary=initial_retention_summary,
            mean_update_latency=mean_lat,
            total_update_latency=tot_lat,
            peak_memory_bytes=peak_mem,
            multi_objective_score=mo_score,
        )
