"""
Abstract Base Task definition for modular multi-priority robot control.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np


class BaseTask(ABC):
    """
    Abstract Base Class for operational and joint space tasks.

    Decouples task mathematics (Jacobians, errors, virtual forces) from the QP solver hierarchy.
    """

    def __init__(self, name: str, priority: int = 0, weight: float = 1.0) -> None:
        """
        Initialize the base task.

        Args:
            name: Human-readable task identifier.
            priority: Priority level index (0 = highest priority).
            weight: Task weighting multiplier.
        """
        self.name = name
        self.priority = priority
        self.weight = weight

    @abstractmethod
    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the task Jacobian matrix J_i and desired task force/torque reference f_i.

        Args:
            state: Robot state dictionary containing 'q', 'dq', 'J', 'B', etc.
            t: Current simulation time in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - J_i: Task Jacobian matrix of shape (m_i, n_dofs)
                - f_i: Task space virtual force (m_i,) or joint space torque (n_dofs,)
        """
        pass

    @abstractmethod
    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes the scalar tracking error norm for logging and evaluation.

        Args:
            state: Current robot state dictionary.

        Returns:
            float: Scalar tracking error norm.
        """
        pass
