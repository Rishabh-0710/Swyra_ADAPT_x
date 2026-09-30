"""
Unit tests for continuous adaptation, stability-plasticity trade-off,
and dynamic gating weight modulation under shifts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import OptiForgeAdaptiveModel
from src.task import TaskSpec
from experiments.distribution_shifts import (
    apply_concept_shift_regression,
    apply_mean_shift,
)


def test_stability_plasticity_tradeoff():
    """
    Verify model adapts to shifted distribution while preserving memory of earlier distributions.
    """
    rng = np.random.default_rng(42)
    n = 300
    spec = TaskSpec(task_type="regression")

    # Batch 0: y = 3 * x0
    X_0 = pd.DataFrame({"x0": rng.uniform(-1, 1, size=n), "x1": rng.uniform(-1, 1, size=n)})
    y_0 = pd.Series(3.0 * X_0["x0"], name="y")

    model = OptiForgeAdaptiveModel(max_replay_samples=150, random_seed=42)
    model.initialize(X_0, y_0, spec)

    # Initial anchor should dominate
    assert model.anchor_weight >= 0.85

    # Batch 1: High Concept Drift: y = -3 * x0 + 4 * sin(x1)
    X_1 = pd.DataFrame({"x0": rng.uniform(-1, 1, size=n), "x1": rng.uniform(-1, 1, size=n)})
    y_1 = pd.Series(-3.0 * X_1["x0"] + 4.0 * np.sin(X_1["x1"]), name="y")

    model.update(X_1, y_1)

    # After high concept drift, adaptation should activate specialist head
    last_action = model.adaptation_history[-1]
    assert last_action.specialist_weight > 0.30
    assert model.specialist_model is not None

    # Test predictions on new shifted distribution are accurate
    preds_shifted = model.predict(X_1)
    rmse_shifted = np.sqrt(np.mean((y_1 - preds_shifted) ** 2))
    # Naive unadapted model error would be around ~6.0. Adapted error should be much lower.
    assert rmse_shifted < 3.0, f"Model failed to adapt to concept drift: RMSE={rmse_shifted}"

    # Verify model has not completely forgotten Batch 0
    preds_0 = model.predict(X_0)
    rmse_0 = np.sqrt(np.mean((y_0 - preds_0) ** 2))
    assert rmse_0 < 3.0, f"Catastrophic forgetting: initial task RMSE={rmse_0}"
