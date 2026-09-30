"""
Comprehensive Benchmark Runner for OPTIFORGE:
Evaluates OptiForgeAdaptiveModel against all 4 baselines:
- Static Model
- Naive Incremental Model
- Standard Replay Model
- Full Retrain Oracle
Across 5 dataset families and multiple random seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List
import pandas as pd

from experiments.baselines import (
    FullRetrainOracle,
    NaiveIncrementalModel,
    StandardReplayModel,
    StaticModel,
)
from experiments.distribution_shifts import (
    create_benchmark_stream,
    generate_family_1_nonlinear_regression,
    generate_family_2_nonlinear_classification,
    generate_family_3_highdim_regression,
    generate_family_4_mixed_classification,
    generate_family_5_heteroscedastic_regression,
)
from src.evaluation import SequentialStreamSimulator
from src.model import OptiForgeAdaptiveModel
from src.utils import set_seed


def run_full_benchmark(
    seeds: List[int] = [42, 101, 777],
    batch_size: int = 1000,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Execute full multi-seed, multi-dataset benchmark across all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_families = [
        ("Family_1_Nonlinear_Reg", generate_family_1_nonlinear_regression),
        ("Family_2_Nonlinear_Clf", generate_family_2_nonlinear_classification),
        ("Family_3_HighDim_Reg", generate_family_3_highdim_regression),
        ("Family_4_Mixed_Clf", generate_family_4_mixed_classification),
        ("Family_5_Heteroscedastic_Reg", generate_family_5_heteroscedastic_regression),
    ]

    model_factories = {
        "Static (No Update)": lambda s: StaticModel(random_seed=s),
        "Naive Incremental": lambda s: NaiveIncrementalModel(random_seed=s),
        "Standard Replay (K=500)": lambda s: StandardReplayModel(max_replay_samples=500, random_seed=s),
        "OptiForge Adaptive (Proposed)": lambda s: OptiForgeAdaptiveModel(max_replay_samples=500, random_seed=s),
        "Full Retrain (Oracle)": lambda s: FullRetrainOracle(random_seed=s),
    }

    records: List[Dict[str, Any]] = []
    simulator = SequentialStreamSimulator(eval_split_ratio=0.20, random_seed=42)

    total_runs = len(dataset_families) * len(seeds) * len(model_factories)
    run_idx = 0
    start_time = time.time()

    for fam_name, gen_fn in dataset_families:
        print(f"\n=======================================================")
        print(f"Benchmarking Dataset: {fam_name}")
        print(f"=======================================================")

        for seed in seeds:
            set_seed(seed)
            batches, task_spec = create_benchmark_stream(gen_fn, batch_size=batch_size, seed=seed)

            for model_name, factory in model_factories.items():
                run_idx += 1
                t0 = time.perf_counter()
                model_inst = factory(seed)

                eval_res = simulator.run_stream(
                    model=model_inst,
                    batches=batches,
                    task_spec=task_spec,
                    model_name=model_name,
                )
                elapsed = time.perf_counter() - t0

                rob = eval_res.robustness_summary
                ret = eval_res.initial_retention_summary

                record = {
                    "dataset": fam_name,
                    "seed": seed,
                    "model": model_name,
                    "task_type": task_spec.task_type,
                    "multi_objective_score": round(eval_res.multi_objective_score, 4),
                    "mean_shifted_perf": round(rob.get("mean_shifted_performance", 0.0), 4),
                    "worst_shifted_perf": round(rob.get("worst_shifted_performance", 0.0), 4),
                    "max_degradation": round(rob.get("max_degradation", 0.0), 4),
                    "initial_retention_perf": round(ret.get("mean_shifted_performance", 0.0), 4),
                    "mean_update_latency_sec": round(eval_res.mean_update_latency, 4),
                    "total_update_latency_sec": round(eval_res.total_update_latency, 4),
                    "peak_memory_kb": round(eval_res.peak_memory_bytes / 1024.0, 1),
                }
                records.append(record)
                print(
                    f"[{run_idx}/{total_runs}] {model_name:30s} | Seed: {seed} | "
                    f"Score: {record['multi_objective_score']:.4f} | "
                    f"Mean Perf: {record['mean_shifted_perf']:.4f} | "
                    f"Lat: {record['mean_update_latency_sec']:.3f}s"
                )

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll benchmark runs completed in {time.time() - start_time:.2f}s. Saved to {csv_path}")

    # Generate aggregated summary table
    summary = df_results.groupby("model").agg({
        "multi_objective_score": ["mean", "std"],
        "mean_shifted_perf": "mean",
        "worst_shifted_perf": "mean",
        "max_degradation": "mean",
        "initial_retention_perf": "mean",
        "mean_update_latency_sec": "mean",
        "peak_memory_kb": "mean",
    }).round(4)
    print("\nAggregated Performance Summary:")
    print(summary)

    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary.to_csv(summary_path)

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OPTIFORGE Full Benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 777], help="Random seeds")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_full_benchmark(seeds=args.seeds, batch_size=args.batch_size, output_dir=args.output_dir)
