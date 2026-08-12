"""
Abstract Base Class defining the unified interface for robot controllers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseController(ABC):
    """
    Abstract Base Class for modular manipulator controllers.

    Decouples control mathematics from physics engine rendering and simulation.
    All concrete controllers must implement the `compute_torques` method.
    """

    def __init__(self, n_dofs: int = 7) -> None:
        """
        Initialize the base controller.

        Args:
            n_dofs: Number of controllable joint degrees of freedom (default: 7 for Franka Panda).
        """
        self.n_dofs = n_dofs

    @abstractmethod
    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute control joint torques based on current robot state and desired task target.

        Args:
            state: Dictionary containing current robot state variables:
                - "q": Joint positions array of shape (n_dofs,)
                - "dq": Joint velocities array of shape (n_dofs,)
                - "J": End-effector / task Jacobian matrix of shape (m, n_dofs)
                - "B": Joint-space mass/inertia matrix of shape (n_dofs, n_dofs)
                - "ee_pos": Current end-effector 3D position (3,) [optional]
                - "ee_rot": Current end-effector 3x3 rotation matrix (3, 3) [optional]
            target: Dictionary containing target task references:
                - "pos": Desired Cartesian 3D position (3,)
                - "rot": Desired Cartesian 3x3 rotation matrix (3, 3)
                - "vel": Desired Cartesian 6D linear/angular velocity (6,) [optional]
                - "acc": Desired Cartesian 6D acceleration (6,) [optional]
                - "q_null": Desired secondary null-space joint posture (n_dofs,) [optional]

        Returns:
            np.ndarray: Array of commanded joint torques of shape (n_dofs,).
        """
        pass
