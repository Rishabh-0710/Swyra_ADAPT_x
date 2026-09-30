"""
Unit tests for OptiForgeAdaptiveModel initialization, updates, predictions,
task types (regression & classification), and missing value robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import OptiForgeAdaptiveModel
from src.task import TaskSpec, infer_task_spec


def test_initialization_regression():
    """Verify initialization and prediction on regression task."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(200, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(2.0 * X["f_0"] - 1.5 * X["f_1"] + rng.normal(0, 0.2, size=200), name="target")

    spec = infer_task_spec(X, y)
    assert spec.task_type == "regression"

    model = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)
    model.initialize(X, y, spec)
    assert model.is_initialized

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert not np.any(np.isnan(preds))


def test_initialization_classification():
    """Verify initialization and probability outputs on binary classification task."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(200, 4)), columns=[f"c_{i}" for i in range(4)])
    y = pd.Series((X["c_0"] + X["c_1"] > 0).astype(int), name="label")

    spec = infer_task_spec(X, y)
    assert spec.task_type == "binary_classification"

    model = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)
    model.initialize(X, y, spec)
    assert model.is_initialized

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Simplex sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_sequential_updates():
    """Verify multiple sequential updates operate cleanly."""
    rng = np.random.default_rng(42)
    model = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)

    X_0 = pd.DataFrame(rng.normal(0, 1, size=(150, 4)), columns=[f"x_{i}" for i in range(4)])
    y_0 = pd.Series(X_0["x_0"] * 2.0, name="target")
    model.initialize(X_0, y_0)

    for i in range(5):
        X_b = pd.DataFrame(rng.normal(0.5 * i, 1, size=(100, 4)), columns=[f"x_{i}" for i in range(4)])
        y_b = pd.Series(X_b["x_0"] * 2.0 + rng.normal(0, 0.1, size=100), name="target")
        model.update(X_b, y_b)

    assert model.update_count == 5
    assert len(model.drift_history) == 5
    assert len(model.adaptation_history) == 5

    preds = model.predict(X_0)
    assert len(preds) == len(X_0)


def test_missing_values_and_categoricals_robustness():
    """Verify model gracefully handles missing values and categorical features without crashing."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "num_1": rng.normal(0, 1, size=n),
        "num_2": rng.normal(5, 2, size=n),
        "cat_1": rng.choice(["A", "B", "C", None], size=n),
        "cat_2": rng.choice(["Yes", "No"], size=n),
    })
    # Inject random missing values in numerical columns
    df.loc[rng.choice(n, size=20, replace=False), "num_1"] = np.nan
    df.loc[rng.choice(n, size=15, replace=False), "num_2"] = np.nan

    y = pd.Series(rng.normal(0, 1, size=n), name="target")

    spec = infer_task_spec(df, y)
    assert "cat_1" in spec.categorical_features
    assert "num_1" in spec.numerical_features

    model = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)
    model.initialize(df, y, spec)

    # Test update with missing values and new unseen category
    df_update = df.copy()
    df_update["cat_1"] = rng.choice(["A", "B", "NEW_CATEGORY", None], size=n)
    df_update.loc[0:10, "num_1"] = np.nan
    y_update = pd.Series(rng.normal(0, 1, size=n), name="target")

    model.update(df_update, y_update)
    preds = model.predict(df_update)
    assert len(preds) == n
    assert not np.any(np.isnan(preds))
