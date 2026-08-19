"""
Cartesian space impedance control task implementation.
"""

from typing import Dict, Tuple, Optional, Callable, Any
import numpy as np

from tasks.base_task import BaseTask
from utils.math_utils import compute_orientation_error


class CartesianPoseTask(BaseTask):
    """
    Cartesian space end-effector position and orientation impedance task.
    """

    def __init__(
        self,
        name: str = "cartesian_pose",
        priority: int = 0,
        kp: float = 400.0,
        kd: float = 40.0,
        mode: str = "3d",
        use_full_impedance: bool = False,
        trajectory_fn: Optional[Callable[[float], Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    ) -> None:
        """
        Initialize Cartesian Pose Task.

        Args:
            name: Task identifier string.
            priority: Priority level index (0 = highest priority).
            kp: Proportional stiffness gain.
            kd: Derivative damping gain.
            mode: Task control dimension ('3d', '6d', 'xy', 'z').
            use_full_impedance: If True, uses Lambda mass matrix weighting; if False, uses VMC.
            trajectory_fn: Optional dynamic reference function t -> (pos_des, rot_des, vel_des).
        """
        super().__init__(name=name, priority=priority)
        self.kp = kp
        self.kd = kd
        self.mode = mode
        self.use_full_impedance = use_full_impedance
        self.trajectory_fn = trajectory_fn


    @property
    def dim(self) -> int:
        """Returns the task dimension corresponding to self.mode."""
        if self.mode == "6d":
            return 6
        elif self.mode == "xy":
            return 2
        elif self.mode == "z":
            return 1
        return 3

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Cartesian task Jacobian J_cart and task force f_cart.

        Args:
            state: Robot state dictionary containing 'q', 'dq', 'J', 'B', 'ee_pos', 'ee_rot'.
            t: Current time in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (J_cart, f_cart)
        """
        J_full = state["J"]  # (6, n_dofs)
        q = state["q"]
        dq = state["dq"]
        B = state["B"]
        p_curr = state["ee_pos"]
        R_curr = state.get("ee_rot", np.eye(3))

        # Retrieve setpoints
        if self.trajectory_fn is not None:
            p_des, R_des, v_des = self.trajectory_fn(t)
        else:
            p_des = state.get("target_pos", p_curr)
            R_des = state.get("target_rot", R_curr)
            v_des = state.get("target_vel", np.zeros(self.dim))

        if R_des is None:
            R_des = R_curr
        if v_des is None:
            v_des = np.zeros(self.dim)

        if self.mode == "6d":
            J_task = J_full  # (6, n_dofs)
            e_pos = p_des - p_curr
            e_rot = compute_orientation_error(R_curr, R_des)
            e_task = np.concatenate([e_pos, e_rot])
            v_curr = J_task @ dq
        elif self.mode == "xy":
            J_task = J_full[:2, :]  # (2, n_dofs)
            e_task = (p_des - p_curr)[:2]
            v_curr = J_task @ dq
            if len(v_des) >= 2:
                v_des = v_des[:2]
        elif self.mode == "z":
            J_task = J_full[2:3, :]  # (1, n_dofs)
            e_task = (p_des - p_curr)[2:3]
            v_curr = J_task @ dq
            if len(v_des) >= 3:
                v_des = v_des[2:3]
        else:
            J_task = J_full[:3, :]  # (3, n_dofs)
            e_task = p_des - p_curr
            v_curr = J_task @ dq

        v_err = v_des - v_curr

        if self.use_full_impedance:
            # Full Cartesian Impedance Law
            # Lambda = (J * B^(-1) * J^T)^(-1)
            B_inv = np.linalg.inv(B)
            Lambda_inv = J_task @ B_inv @ J_task.T
            # Damped pseudo-inverse near singularities
            reg = 1e-4 * np.eye(Lambda_inv.shape[0])
            Lambda = np.linalg.inv(Lambda_inv + reg)
            
            # Operational space acceleration feedforward and velocity drift compensation
            acc_des = state.get("target_acc", np.zeros(J_task.shape[0]))
            dJ_dq = state.get("dJ_dq", np.zeros(J_task.shape[0]))
            f_cart = Lambda @ (acc_des - dJ_dq) + self.kp * e_task + self.kd * v_err
        else:
            # Virtual Model Control (VMC) / Spring-Damper Law
            f_cart = self.kp * e_task + self.kd * v_err

        return J_task, f_cart

    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes tracking error norm matching the task mode dimension.
        """
        p_curr = state["ee_pos"]
        p_des = state.get("target_pos", p_curr)
        if self.mode == "xy":
            return float(np.linalg.norm((p_des - p_curr)[:2]))
        elif self.mode == "z":
            return float(abs(p_des[2] - p_curr[2]))
        return float(np.linalg.norm(p_des - p_curr))

