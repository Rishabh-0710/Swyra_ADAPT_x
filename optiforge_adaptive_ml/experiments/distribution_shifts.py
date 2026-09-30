"""
Domain-Agnostic Mathematical Distribution Shift Engine and Benchmark Dataset Families.
Provides:
- 5 distinct mathematical synthetic data generators (nonlinear regression, binary classification,
  high-dimensional sparse, mixed categorical, and heteroscedastic noise)
- Generic mathematical distribution shift operators (mean, scale, rotation, concept, prior, noise, compound)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.task import TaskSpec


# =====================================================================
# 1. Pure Mathematical Distribution Shift Operators (Domain-Agnostic)
# =====================================================================

def apply_mean_shift(
    df: pd.DataFrame,
    cols: List[str],
    shift_magnitude: float = 2.0,
) -> pd.DataFrame:
    """Mathematical shift: X_{:, j} <- X_{:, j} + delta."""
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            std = float(df_out[col].std()) if df_out[col].std() > 0 else 1.0
            df_out[col] = df_out[col] + shift_magnitude * std
    return df_out


def apply_scale_shift(
    df: pd.DataFrame,
    cols: List[str],
    scale_factor: float = 2.5,
) -> pd.DataFrame:
    """Mathematical shift: X_{:, j} <- gamma * (X_{:, j} - mu) + mu."""
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            mean = float(df_out[col].mean())
            df_out[col] = mean + scale_factor * (df_out[col] - mean)
    return df_out


def apply_rotation_shift(
    df: pd.DataFrame,
    col_pair: Tuple[str, str],
    theta_rad: float = np.pi / 4,
) -> pd.DataFrame:
    """Mathematical feature correlation shift via 2D rotation matrix."""
    df_out = df.copy()
    c1, c2 = col_pair
    if c1 in df_out.columns and c2 in df_out.columns:
        x1 = df_out[c1].to_numpy(dtype=float)
        x2 = df_out[c2].to_numpy(dtype=float)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        df_out[c1] = cos_t * x1 - sin_t * x2
        df_out[c2] = sin_t * x1 + cos_t * x2
    return df_out


def apply_concept_shift_regression(
    df: pd.DataFrame,
    y: pd.Series,
    interaction_cols: Tuple[str, str],
    shift_gain: float = 3.0,
) -> pd.Series:
    """Mathematical concept shift: P(Y|X) changes via new non-linear interaction."""
    c1, c2 = interaction_cols
    x1 = df[c1].to_numpy(dtype=float)
    x2 = df[c2].to_numpy(dtype=float)
    delta_y = shift_gain * np.sin(x1 * x2)
    return pd.Series(y.to_numpy(dtype=float) + delta_y, name=y.name)


def apply_concept_shift_classification(
    df: pd.DataFrame,
    y: pd.Series,
    invert_cols: List[str],
    flip_rate: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Mathematical concept shift: Decision boundary flips in a specific subspace."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_arr = y.to_numpy().copy()
    mask = np.ones(len(df), dtype=bool)
    for col in invert_cols:
        if col in df.columns:
            mask &= (df[col] > df[col].median())

    subspace_idx = np.where(mask)[0]
    if len(subspace_idx) > 0:
        flip_picks = rng.choice(
            subspace_idx,
            size=int(len(subspace_idx) * flip_rate),
            replace=False,
        )
        y_arr[flip_picks] = 1 - y_arr[flip_picks]
    return pd.Series(y_arr, name=y.name)


def apply_noise_shift(
    y: pd.Series,
    noise_sigma: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Additive measurement noise increase: y <- y + N(0, sigma^2)."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_sigma, size=len(y))
    return pd.Series(y.to_numpy() + noise, name=y.name)


def apply_prior_shift_classification(
    df: pd.DataFrame,
    y: pd.Series,
    target_pos_ratio: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Mathematical prior shift: changes class prevalence P(Y=1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_arr = y.to_numpy()
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    n_total = len(y_arr)
    n_pos_desired = int(n_total * target_pos_ratio)
    n_neg_desired = n_total - n_pos_desired

    chosen_pos = rng.choice(pos_idx, size=n_pos_desired, replace=True)
    chosen_neg = rng.choice(neg_idx, size=n_neg_desired, replace=True)
    all_picks = rng.permutation(np.concatenate([chosen_pos, chosen_neg]))

    return df.iloc[all_picks].reset_index(drop=True), y.iloc[all_picks].reset_index(drop=True)


# =====================================================================
# 2. Independent Mathematical Benchmark Dataset Families
# =====================================================================

def generate_family_1_nonlinear_regression(
    n_samples: int = 4000,
    n_features: int = 10,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, TaskSpec]:
    """
    Family 1: Smooth Non-linear Multi-Interaction Regression (Friedman #1 variant).
    y = 10*sin(pi*x0*x1) + 20*(x2 - 0.5)^2 + 10*x3 + 5*x4 + eps
    """
    rng = np.random.default_rng(seed)
    X_raw = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    cols = [f"x_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=cols)

    y_clean = (
        10.0 * np.sin(np.pi * df["x_0"] * df["x_1"])
        + 20.0 * (df["x_2"] - 0.5) ** 2
        + 10.0 * df["x_3"]
        + 5.0 * df["x_4"]
    )
    y = y_clean + rng.normal(0, noise_std, size=n_samples)
    y_series = pd.Series(y, name="target")

    spec = TaskSpec(
        task_type="regression",
        target_name="target",
        numerical_features=cols,
        categorical_features=[],
        metric="mse",
    )
    return df, y_series, spec


def generate_family_2_nonlinear_classification(
    n_samples: int = 4000,
    n_features: int = 12,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, TaskSpec]:
    """
    Family 2: Non-linear Boundary Binary Classification with Interaction Terms.
    z = 1.5*x0 - 2.0*x1 + 3.0*x2*x3 - 1.2*x4^2 + eps; P(Y=1) = sigmoid(z)
    """
    rng = np.random.default_rng(seed)
    X_raw = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=cols)

    logits = (
        1.5 * df["feat_0"]
        - 2.0 * df["feat_1"]
        + 3.0 * (df["feat_2"] * df["feat_3"])
        - 1.2 * (df["feat_4"] ** 2)
        + rng.normal(0, 0.4, size=n_samples)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    y_series = pd.Series(y, name="target")

    spec = TaskSpec(
        task_type="binary_classification",
        target_name="target",
        numerical_features=cols,
        categorical_features=[],
        metric="roc_auc",
    )
    return df, y_series, spec


def generate_family_3_highdim_regression(
    n_samples: int = 4000,
    n_features: int = 40,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, TaskSpec]:
    """
    Family 3: High-Dimensional Correlated Sparse Regression.
    Covariance matrix has Toeplitz correlation rho^{|i-j|}.
    Only 5 sparse features are active.
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    idx = np.arange(n_features)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    L = np.linalg.cholesky(cov)

    Z = rng.normal(0, 1, size=(n_samples, n_features))
    X_raw = Z @ L.T
    cols = [f"dim_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=cols)

    # Sparse active coefficients
    beta = np.zeros(n_features)
    beta[2] = 4.0
    beta[7] = -3.5
    beta[15] = 2.0
    beta[25] = -2.5
    beta[38] = 5.0

    y = X_raw @ beta + rng.normal(0, 1.0, size=n_samples)
    y_series = pd.Series(y, name="target")

    spec = TaskSpec(
        task_type="regression",
        target_name="target",
        numerical_features=cols,
        categorical_features=[],
        metric="mse",
    )
    return df, y_series, spec


def generate_family_4_mixed_classification(
    n_samples: int = 4000,
    n_numerical: int = 8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, TaskSpec]:
    """
    Family 4: Mixed Categorical & Numerical Classification with Subgroups.
    """
    rng = np.random.default_rng(seed)
    num_raw = rng.normal(0, 1, size=(n_samples, n_numerical))
    num_cols = [f"num_{i}" for i in range(n_numerical)]
    df = pd.DataFrame(num_raw, columns=num_cols)

    # Categorical features
    cat_tier = rng.choice(["Alpha", "Beta", "Gamma", "Delta"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    cat_zone = rng.choice(["North", "South", "East", "West"], size=n_samples)
    df["cat_tier"] = cat_tier
    df["cat_zone"] = cat_zone

    # Subgroup column for fairness tracking
    df["subgroup_id"] = rng.choice(["Group_0", "Group_1", "Group_2"], size=n_samples, p=[0.5, 0.3, 0.2])

    tier_weights = {"Alpha": 1.5, "Beta": 0.5, "Gamma": -0.8, "Delta": -1.8}
    cat_bias = np.array([tier_weights[c] for c in cat_tier])

    logits = (
        1.8 * df["num_0"]
        - 1.4 * df["num_1"]
        + 0.8 * df["num_2"] * df["num_3"]
        + cat_bias
        + rng.normal(0, 0.5, size=n_samples)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    y_series = pd.Series(y, name="target")

    spec = TaskSpec(
        task_type="binary_classification",
        target_name="target",
        numerical_features=num_cols,
        categorical_features=["cat_tier", "cat_zone"],
        subgroup_col="subgroup_id",
        metric="roc_auc",
    )
    return df, y_series, spec


def generate_family_5_heteroscedastic_regression(
    n_samples: int = 4000,
    n_features: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, TaskSpec]:
    """
    Family 5: Heteroscedastic Heavy-Tailed Dynamic Noise Regression.
    """
    rng = np.random.default_rng(seed)
    X_raw = rng.uniform(-2.0, 2.0, size=(n_samples, n_features))
    cols = [f"var_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=cols)

    signal = 3.0 * np.tanh(df["var_0"]) + 2.0 * df["var_1"] - 1.5 * (df["var_2"] ** 2)
    # Heteroscedastic noise proportional to distance from origin
    norm_x = np.linalg.norm(X_raw[:, :3], axis=1)
    noise = rng.normal(0, 1, size=n_samples) * (0.3 + 0.5 * norm_x)
    y = signal + noise
    y_series = pd.Series(y, name="target")

    spec = TaskSpec(
        task_type="regression",
        target_name="target",
        numerical_features=cols,
        categorical_features=[],
        metric="mse",
    )
    return df, y_series, spec


# =====================================================================
# 3. Stream Generator Simulating 6 Sequential Environments
# =====================================================================

def create_benchmark_stream(
    generator_fn: Callable[..., Tuple[pd.DataFrame, pd.Series, TaskSpec]],
    batch_size: int = 1500,
    seed: int = 42,
) -> Tuple[List[Tuple[str, pd.DataFrame, pd.Series, Optional[np.ndarray]]], TaskSpec]:
    """
    Generates a realistic temporal sequence of 6 batches:
    - Batch 0: Baseline Distribution
    - Batch 1: Covariate Shift (Mean & Scale changes in input features)
    - Batch 2: Prior / Label Shift (Shift in target distribution)
    - Batch 3: Concept Shift (Shift in conditional mapping P(Y|X))
    - Batch 4: Noise + Covariate Shift (Multi-feature rotation + increased sensor noise)
    - Batch 5: Compound Unseen Shift (Simultaneous concept + covariate + noise shift)
    """
    rng = np.random.default_rng(seed)
    total_needed = batch_size * 6
    df_all, y_all, task_spec = generator_fn(n_samples=total_needed, seed=seed)

    subgroup_col = task_spec.subgroup_col
    subgroups = df_all[subgroup_col].to_numpy() if subgroup_col and subgroup_col in df_all else None

    batches: List[Tuple[str, pd.DataFrame, pd.Series, Optional[np.ndarray]]] = []
    num_cols = task_spec.numerical_features

    for t in range(6):
        start = t * batch_size
        end = (t + 1) * batch_size
        df_b = df_all.iloc[start:end].copy().reset_index(drop=True)
        y_b = y_all.iloc[start:end].copy().reset_index(drop=True)
        sub_b = subgroups[start:end] if subgroups is not None else None

        if t == 0:
            env_name = "Env_0_Baseline"

        elif t == 1:
            env_name = "Env_1_Covariate_Shift"
            # Apply mean and scale shifts to first 2 numerical features
            shift_cols = num_cols[:2] if len(num_cols) >= 2 else num_cols
            df_b = apply_mean_shift(df_b, shift_cols, shift_magnitude=2.2)
            df_b = apply_scale_shift(df_b, shift_cols, scale_factor=1.8)

        elif t == 2:
            env_name = "Env_2_Prior_Shift"
            if task_spec.task_type == "binary_classification":
                df_b, y_b = apply_prior_shift_classification(df_b, y_b, target_pos_ratio=0.15, rng=rng)
            else:
                y_b = pd.Series(y_b.to_numpy() + 3.5, name=y_b.name)

        elif t == 3:
            env_name = "Env_3_Concept_Shift"
            if task_spec.task_type == "regression":
                pair = (num_cols[0], num_cols[1]) if len(num_cols) >= 2 else (num_cols[0], num_cols[0])
                y_b = apply_concept_shift_regression(df_b, y_b, interaction_cols=pair, shift_gain=4.0)
            else:
                flip_cols = num_cols[:2] if len(num_cols) >= 2 else num_cols
                y_b = apply_concept_shift_classification(df_b, y_b, invert_cols=flip_cols, flip_rate=0.4, rng=rng)

        elif t == 4:
            env_name = "Env_4_Noise_Covariate_Shift"
            if len(num_cols) >= 2:
                df_b = apply_rotation_shift(df_b, (num_cols[0], num_cols[1]), theta_rad=np.pi / 3)
            if task_spec.task_type == "regression":
                y_b = apply_noise_shift(y_b, noise_sigma=1.8, rng=rng)
            else:
                # Binary classification: symmetric label noise
                y_arr = y_b.to_numpy().copy()
                flip_idx = rng.choice(len(y_arr), size=int(0.10 * len(y_arr)), replace=False)
                y_arr[flip_idx] = 1 - y_arr[flip_idx]
                y_b = pd.Series(y_arr, name=y_b.name)

        else:  # t == 5
            env_name = "Env_5_Compound_Unseen_Shift"
            # Simultaneous multi-feature scale + concept shift + noise
            if len(num_cols) >= 3:
                df_b = apply_scale_shift(df_b, num_cols[:3], scale_factor=2.0)
                df_b = apply_mean_shift(df_b, [num_cols[2]], shift_magnitude=1.8)
                if task_spec.task_type == "regression":
                    y_b = apply_concept_shift_regression(df_b, y_b, (num_cols[1], num_cols[2]), shift_gain=3.5)
                else:
                    y_b = apply_concept_shift_classification(df_b, y_b, [num_cols[1]], flip_rate=0.35, rng=rng)
            if task_spec.task_type == "regression":
                y_b = apply_noise_shift(y_b, noise_sigma=1.2, rng=rng)
            else:
                y_arr = y_b.to_numpy().copy()
                flip_idx = rng.choice(len(y_arr), size=int(0.08 * len(y_arr)), replace=False)
                y_arr[flip_idx] = 1 - y_arr[flip_idx]
                y_b = pd.Series(y_arr, name=y_b.name)

        batches.append((env_name, df_b, y_b, sub_b))

    return batches, task_spec
