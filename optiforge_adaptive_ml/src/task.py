"""
Domain-Agnostic Task Specification and Automated Schema Inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class TaskSpec:
    """
    Domain-agnostic specification of a supervised tabular learning problem.
    """
    task_type: str  # "regression" or "binary_classification"
    target_name: str = "target"
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    metric: str = "auto"
    subgroup_col: Optional[str] = None
    pos_label: Optional[Union[int, str]] = None

    def __post_init__(self) -> None:
        if self.task_type not in ("regression", "binary_classification"):
            raise ValueError(
                f"Unsupported task_type: '{self.task_type}'. "
                "Must be 'regression' or 'binary_classification'."
            )
        if self.metric == "auto":
            self.metric = "mse" if self.task_type == "regression" else "roc_auc"

    @property
    def all_features(self) -> List[str]:
        return self.numerical_features + self.categorical_features


def infer_task_spec(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    target_name: str = "target",
    subgroup_col: Optional[str] = None,
    categorical_max_unique: int = 15,
) -> TaskSpec:
    """
    Automatically infers TaskSpec from the mathematical and statistical
    properties of X and y without requiring any domain-specific heuristics.
    """
    # 1. Infer task type from target y
    y_arr = np.asarray(y)
    unique_vals = np.unique(y_arr[~pd.isna(y_arr)])
    n_unique = len(unique_vals)

    if n_unique == 2 or (n_unique <= 2 and y_arr.dtype == bool):
        task_type = "binary_classification"
        pos_label = unique_vals[1] if n_unique == 2 else 1
    elif np.issubdtype(y_arr.dtype, np.floating) or n_unique > categorical_max_unique:
        task_type = "regression"
        pos_label = None
    else:
        # Fallback based on cardinality
        task_type = "binary_classification" if n_unique <= 2 else "regression"
        pos_label = unique_vals[1] if n_unique == 2 else None

    # 2. Infer feature types from X
    numerical_features: List[str] = []
    categorical_features: List[str] = []

    if isinstance(X, pd.DataFrame):
        for col in X.columns:
            if col == subgroup_col:
                continue
            col_series = X[col]
            if pd.api.types.is_numeric_dtype(col_series):
                # If numeric with very low integer cardinality and not float, could be categorical
                if (
                    pd.api.types.is_integer_dtype(col_series)
                    and col_series.nunique() <= 4
                ):
                    categorical_features.append(str(col))
                else:
                    numerical_features.append(str(col))
            else:
                categorical_features.append(str(col))
    else:
        # Numpy array: assume all numerical features
        n_cols = X.shape[1] if len(X.shape) > 1 else 1
        numerical_features = [f"feat_{i}" for i in range(n_cols)]

    return TaskSpec(
        task_type=task_type,
        target_name=target_name,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        subgroup_col=subgroup_col,
        pos_label=pos_label,
    )
