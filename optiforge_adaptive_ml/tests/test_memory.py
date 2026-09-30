"""
Strict Memory Accounting and Bounded Resource Tests:
- Guarantees replay memory strictly respects max_replay_samples
- Verifies O(1) flat memory footprint across 50+ sequential updates
- Verifies no retention of raw historical dataset objects
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest

from src.memory import StratifiedDensityReplayBuffer
from src.model import OptiForgeAdaptiveModel
from src.task import TaskSpec
from src.utils import get_deep_memory_bytes


def test_replay_buffer_capacity_strictness():
    """Verify replay buffer strictly respects maximum sample budget."""
    spec = TaskSpec(task_type="regression")
    max_k = 150
    buf = StratifiedDensityReplayBuffer(task_spec=spec, max_samples=max_k, random_seed=42)

    rng = np.random.default_rng(42)

    # Ingest 30 batches of 100 samples each (total 3,000 samples)
    for _ in range(30):
        X_batch = rng.normal(0, 1, size=(100, 8))
        y_batch = rng.uniform(0, 100, size=100)
        buf.update(X_batch, y_batch)

        assert buf.current_size <= max_k, f"Buffer size {buf.current_size} exceeded budget {max_k}"

    assert buf.current_size == max_k
    assert buf.total_samples_observed == 3000

    # Retrieve data
    X_rep, y_rep = buf.get_data()
    assert len(X_rep) == max_k
    assert len(y_rep) == max_k


def test_flat_memory_across_long_stream():
    """Verify system memory remains flat across 50 sequential batch updates."""
    rng = np.random.default_rng(42)
    max_replay = 200
    model = OptiForgeAdaptiveModel(max_replay_samples=max_replay, random_seed=42)

    # Initial batch
    X_0 = pd.DataFrame(rng.normal(0, 1, size=(200, 6)), columns=[f"f_{i}" for i in range(6)])
    y_0 = pd.Series(rng.normal(0, 1, size=200), name="y")
    model.initialize(X_0, y_0)

    memories: list[int] = []

    # Run 50 sequential batches
    for b in range(50):
        X_b = pd.DataFrame(rng.normal(0.1 * b, 1, size=(150, 6)), columns=[f"f_{i}" for i in range(6)])
        y_b = pd.Series(rng.normal(0, 1, size=150), name="y")
        model.update(X_b, y_b)

        stats = model.get_memory_stats()
        rep_stats = stats["replay_buffer"]
        assert rep_stats["current_samples"] <= max_replay

        # Sample memory footprint every 10 updates
        if b % 10 == 0:
            memories.append(stats["replay_buffer"]["memory_bytes"])

    # Memory allocated to the replay buffer must be identical / bounded
    assert len(set(memories[-4:])) == 1, "Replay buffer memory grew over time instead of remaining flat!"


def test_no_retention_of_raw_batch_objects():
    """Verify model does not retain references to the original caller's DataFrames."""
    rng = np.random.default_rng(42)
    model = OptiForgeAdaptiveModel(max_replay_samples=100, random_seed=42)

    df_init = pd.DataFrame(rng.normal(0, 1, size=(100, 4)), columns=[f"col_{i}" for i in range(4)])
    y_init = pd.Series(rng.normal(0, 1, size=100), name="y")
    model.initialize(df_init, y_init)

    df_init_id = id(df_init)
    y_init_id = id(y_init)

    # Check model internal attributes
    assert not any(id(val) == df_init_id for val in vars(model).values())
    assert not any(id(val) == y_init_id for val in vars(model).values())

    # Send new batch and check reference discard
    df_batch = pd.DataFrame(rng.normal(0, 1, size=(80, 4)), columns=[f"col_{i}" for i in range(4)])
    y_batch = pd.Series(rng.normal(0, 1, size=80), name="y")
    batch_id = id(df_batch)

    model.update(df_batch, y_batch)
    assert not any(id(val) == batch_id for val in vars(model).values())
