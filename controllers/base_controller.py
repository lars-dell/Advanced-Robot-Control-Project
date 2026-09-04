"""
Abstract Base Class defining the unified interface for robot controllers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional
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
        self.handles_inequalities: bool = False

        # Standard telemetry & feasibility tracking across all controllers
        self.violation_tol: float = 1e-4
        self.n_violations: int = 0
        self.n_solves: int = 0
        self.max_violation: float = 0.0
        self.unclipped_overshoot: float = 0.0
        self.priority_residuals: List[float] = []
        self.level_torques: List[np.ndarray] = []


    def _update_violation_telemetry(
        self,
        tau_cmd: np.ndarray,
        tau_min: np.ndarray,
        tau_max: np.ndarray,
        unclipped_tau: Optional[np.ndarray] = None
    ) -> float:
        """
        Record feasibility and boundary overshoot statistics.

        Args:
            tau_cmd: Commanded joint torques sent to actuators.
            tau_min: Minimum joint torque limit vector.
            tau_max: Maximum joint torque limit vector.
            unclipped_tau: Optional pre-clipped torque vector for saturated controllers.

        Returns:
            float: Maximum torque limit overshoot in N*m.
        """
        self.n_solves += 1
        eval_tau = unclipped_tau if unclipped_tau is not None else tau_cmd
        overshoot = float(np.max(np.maximum(tau_min - eval_tau, eval_tau - tau_max)))
        if unclipped_tau is not None:
            self.unclipped_overshoot = max(self.unclipped_overshoot, overshoot)
        if overshoot > self.violation_tol:
            self.n_violations += 1
            self.max_violation = max(self.max_violation, overshoot)
        return overshoot

    @abstractmethod
    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Union[Dict[str, Any], Any],
        t: float = 0.0
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
            target: Dictionary containing target task references OR a TaskStack instance.
            t: Simulation timestamp in seconds.

        Returns:
            np.ndarray: Array of commanded joint torques of shape (n_dofs,).
        """
        pass

