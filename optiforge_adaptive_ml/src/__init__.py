"""
OPTIFORGE: Domain-Agnostic Adaptive Continual Learning System for Tabular Data.
"""

from src.task import TaskSpec, infer_task_spec
from src.model import OptiForgeAdaptiveModel
from src.drift import DriftDetector, DriftReport
from src.memory import StratifiedDensityReplayBuffer
from src.adaptation import AdaptationController

__all__ = [
    "TaskSpec",
    "infer_task_spec",
    "OptiForgeAdaptiveModel",
    "DriftDetector",
    "DriftReport",
    "StratifiedDensityReplayBuffer",
    "AdaptationController",
]
