"""
Operational space force task and trajectory generation utilities.
"""

from typing import Dict, Tuple, Optional, Callable
import numpy as np

from tasks.base_task import BaseTask


class CartesianForceTask(BaseTask):
    """
    Operational space force control task.
    
    Exerts a specified target force vector (e.g. surface normal push force)
    in operational space.
    """

    def __init__(
        self,
        name: str = "cartesian_force",
        priority: int = 0,
        desired_force: float = -10.0,
        direction: str = "z",
        force_vector: Optional[np.ndarray] = None,
        target_pos: Optional[float] = None,
        kp: float = 0.0,
        kd: float = 0.0
    ) -> None:
        """
        Initialize Cartesian Force Task.

        Args:
            name: Human-readable task identifier.
            priority: Priority level index (0 = highest priority).
            desired_force: Magnitude of desired force in Newtons (negative = downward along Z).
            direction: Force axis direction ('x', 'y', 'z', or 'custom').
            force_vector: Explicit 3D force vector [Fx, Fy, Fz] if direction is 'custom'.
            target_pos: Optional target position along force axis (e.g. surface height z_surface = 0.40).
            kp: Position error gain for hybrid force/position control when approaching surface.
            kd: Damping velocity gain.
        """
        super().__init__(name=name, priority=priority)
        self.desired_force = desired_force
        self.direction = direction
        self.target_pos = target_pos
        self.kp = kp
        self.kd = kd

        if force_vector is not None:
            self.force_vector = np.asarray(force_vector, dtype=np.float64)
        elif direction == "z":
            self.force_vector = np.array([0.0, 0.0, desired_force], dtype=np.float64)
        elif direction == "y":
            self.force_vector = np.array([0.0, desired_force, 0.0], dtype=np.float64)
        elif direction == "x":
            self.force_vector = np.array([desired_force, 0.0, 0.0], dtype=np.float64)
        else:
            raise ValueError(f"Unknown direction '{direction}'. Choose 'x', 'y', 'z', or provide force_vector.")

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes task Jacobian J_force and desired task force f_force.

        Args:
            state: Robot state dictionary containing 'J', 'ee_pos', 'dq', etc.
            t: Current time in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (J_task, f_task)
        """
        J_full = state["J"]  # (6, n_dofs) or (3, n_dofs)
        p_curr = state.get("ee_pos", np.zeros(3))
        dq = state.get("dq", np.zeros(J_full.shape[1]))
        v_curr = J_full @ dq

        if self.direction == "z":
            J_task = J_full[2:3, :]  # (1, n_dofs)
            f_base = self.desired_force
            if self.target_pos is not None and self.kp > 0:
                e_p = self.target_pos - p_curr[2]
                e_v = 0.0 - v_curr[2]
                f_base += self.kp * e_p + self.kd * e_v
            f_task = np.array([f_base], dtype=np.float64)
        elif self.direction == "y":
            J_task = J_full[1:2, :]  # (1, n_dofs)
            f_base = self.desired_force
            if self.target_pos is not None and self.kp > 0:
                e_p = self.target_pos - p_curr[1]
                e_v = 0.0 - v_curr[1]
                f_base += self.kp * e_p + self.kd * e_v
            f_task = np.array([f_base], dtype=np.float64)
        elif self.direction == "x":
            J_task = J_full[0:1, :]  # (1, n_dofs)
            f_base = self.desired_force
            if self.target_pos is not None and self.kp > 0:
                e_p = self.target_pos - p_curr[0]
                e_v = 0.0 - v_curr[0]
                f_base += self.kp * e_p + self.kd * e_v
            f_task = np.array([f_base], dtype=np.float64)
        else:
            J_task = J_full[:3, :]  # (3, n_dofs)
            f_task = self.force_vector.copy()

        return J_task, f_task


    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes force tracking error against measured end-effector contact force if available.
        """
        measured_force = state.get("ee_force", np.zeros(3))
        if self.direction == "z":
            return float(abs(self.desired_force - measured_force[2]))
        elif self.direction == "y":
            return float(abs(self.desired_force - measured_force[1]))
        elif self.direction == "x":
            return float(abs(self.desired_force - measured_force[0]))
        else:
            return float(np.linalg.norm(self.force_vector - measured_force))


class CircularTrajectoryGenerator:
    """
    Generates 3D circular Cartesian trajectories on a specified plane.
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float = 0.08,
        period: float = 4.0,
        plane: str = "xy"
    ) -> None:
        """
        Initialize Circular Trajectory Generator.

        Args:
            center: 3D center coordinates of the circle [cx, cy, cz].
            radius: Circle radius in meters.
            period: Time in seconds for one complete circular revolution.
            plane: Trajectory plane ('xy', 'xz', 'yz').
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.period = period
        self.omega = 2.0 * np.pi / period
        self.plane = plane

    def __call__(self, t: float) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes trajectory state at time t.

        Args:
            t: Simulation timestamp in seconds.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                - pos_des: Desired 3D position [x, y, z]
                - rot_des: None (or 3x3 orientation if specified)
                - vel_des: Desired 3D velocity [vx, vy, vz]
        """
        angle = self.omega * t

        pos_des = self.center.copy()
        vel_des = np.zeros(3, dtype=np.float64)

        if self.plane == "xy":
            pos_des[0] += self.radius * np.cos(angle)
            pos_des[1] += self.radius * np.sin(angle)
            vel_des[0] = -self.radius * self.omega * np.sin(angle)
            vel_des[1] = self.radius * self.omega * np.cos(angle)
        elif self.plane == "xz":
            pos_des[0] += self.radius * np.cos(angle)
            pos_des[2] += self.radius * np.sin(angle)
            vel_des[0] = -self.radius * self.omega * np.sin(angle)
            vel_des[2] = self.radius * self.omega * np.cos(angle)
        elif self.plane == "yz":
            pos_des[1] += self.radius * np.cos(angle)
            pos_des[2] += self.radius * np.sin(angle)
            vel_des[1] = -self.radius * self.omega * np.sin(angle)
            vel_des[2] = self.radius * self.omega * np.cos(angle)

        return pos_des, None, vel_des
