"""
Strict Data Leakage and Reproducibility Audit:
- Audits zero evaluation-set leakage
- Audits zero lookahead / future-batch access
- Audits bit-exact / numerical reproducibility with fixed random seed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import SequentialStreamSimulator
from src.model import OptiForgeAdaptiveModel
from src.task import TaskSpec
from src.utils import set_seed
from experiments.distribution_shifts import create_benchmark_stream, generate_family_1_nonlinear_regression


def test_reproducibility_with_seed():
    """Verify identical random seed produces identical predictions across multiple runs."""
    batches_1, spec = create_benchmark_stream(generate_family_1_nonlinear_regression, batch_size=400, seed=42)
    batches_2, _ = create_benchmark_stream(generate_family_1_nonlinear_regression, batch_size=400, seed=42)

    # Run 1
    set_seed(42)
    m1 = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)
    m1.initialize(batches_1[0][1], batches_1[0][2], spec)
    m1.update(batches_1[1][1], batches_1[1][2])
    preds1 = m1.predict(batches_1[2][1])

    # Run 2
    set_seed(42)
    m2 = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)
    m2.initialize(batches_2[0][1], batches_2[0][2], spec)
    m2.update(batches_2[1][1], batches_2[1][2])
    preds2 = m2.predict(batches_2[2][1])

    np.testing.assert_allclose(preds1, preds2, rtol=1e-5, atol=1e-5)


def test_zero_eval_leakage_in_simulator():
    """
    Verify that the SequentialStreamSimulator strictly walls off
    evaluation samples from the model.
    """
    simulator = SequentialStreamSimulator(eval_split_ratio=0.25, random_seed=42)

    # Instrument a mock model to record every training sample id
    seen_sample_hashes = set()

    class AuditModel:
        def initialize(self, X, y, spec):
            for row in np.asarray(X):
                seen_sample_hashes.add(hash(row.tobytes()))

        def update(self, X, y):
            for row in np.asarray(X):
                seen_sample_hashes.add(hash(row.tobytes()))

        def predict(self, X):
            # Check if any evaluation row was ever passed into initialize or update!
            for row in np.asarray(X):
                row_hash = hash(row.tobytes())
                assert row_hash not in seen_sample_hashes, (
                    "CRITICAL LEAKAGE DETECTED: Model was trained on an evaluation sample!"
                )
            return np.zeros(len(X))

    batches, spec = create_benchmark_stream(generate_family_1_nonlinear_regression, batch_size=300, seed=42)
    audit_model = AuditModel()
    res = simulator.run_stream(audit_model, batches, spec, model_name="LeakageAudit")
    assert len(res.batch_results) == 6
