"""
Artificial Potential Field (APF) Repulsive Task for Reactive Obstacle Avoidance.

Computes operational space repulsive forces when an obstacle enters the influence margin.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from tasks.base_task import BaseTask


class APFRepulsiveTask(BaseTask):
    """
    Operational space Artificial Potential Field (APF) repulsive force task.

    Exerts a repulsive task force when distance d to obstacle is within influence radius d_0:
        f_rep = k_rep * (1/d - 1/d_0) * (1 / d^2) * (x_curr - x_obs) / d
    """

    def __init__(
        self,
        name: str = "apf_repulsion",
        priority: int = 0,
        obstacle_pos: Optional[np.ndarray] = None,
        obstacle_radius: float = 0.10,
        margin: float = 0.08,
        k_rep: float = 5.0,
        target_link: str = "hand"
    ) -> None:
        """
        Initialize APF Repulsive Task.

        Args:
            name: Task identifier string.
            priority: Priority level index (0 = highest priority).
            obstacle_pos: 3D coordinates [x, y, z] of obstacle center.
            obstacle_radius: Radius of the physical obstacle in meters.
            margin: Safety influence margin distance d_0 in meters.
            k_rep: Repulsive force gain multiplier.
            target_link: Robot link name to protect (default: 'hand').
        """
        super().__init__(name=name, priority=priority)
        self.obstacle_pos = np.array([0.45, 0.0, 0.45]) if obstacle_pos is None else np.asarray(obstacle_pos, dtype=np.float64)
        self.obstacle_radius = obstacle_radius
        self.margin = margin
        self.d0 = obstacle_radius + margin
        self.k_rep = k_rep
        self.target_link = target_link

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 3D linear Jacobian J_apf and repulsive force f_rep.

        Args:
            state: Robot state dictionary containing 'J', 'ee_pos', 'q', etc.
            t: Current simulation time in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (J_3d, f_rep)
        """
        J_full = state["J"]  # (6, n_dofs)
        J_3d = J_full[:3, :]  # (3, n_dofs)
        p_curr = state["ee_pos"]

        # Vector from obstacle center to current position
        vec_o = p_curr - self.obstacle_pos
        dist_o = np.linalg.norm(vec_o) + 1e-6

        if dist_o < self.d0:
            # Active repulsive force when inside influence region d_0
            dir_o = vec_o / dist_o
            magnitude = self.k_rep * (1.0 / dist_o - 1.0 / self.d0) * (1.0 / (dist_o ** 2))
            f_rep = magnitude * dir_o
        else:
            # Zero force outside influence region
            f_rep = np.zeros(3, dtype=np.float64)

        return J_3d, f_rep

    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes minimum clearance distance to obstacle boundary (negative if colliding).
        """
        p_curr = state["ee_pos"]
        dist_o = float(np.linalg.norm(p_curr - self.obstacle_pos))
        clearance = dist_o - self.obstacle_radius
        return clearance
