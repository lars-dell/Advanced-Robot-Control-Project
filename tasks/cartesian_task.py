"""
Cartesian space impedance control task implementation.
"""

from typing import Dict, Tuple, Optional, Callable, Any, Sequence, Union
import numpy as np

from tasks.base_task import BaseTask
from utils.math_utils import compute_orientation_error


class CartesianPoseTask(BaseTask):
    """
    Cartesian space end-effector position and orientation impedance task.
    
    Supports 3D position, 6D pose, planar (xy), or single-axis (x, y, z) sub-tasks.
    """

    def __init__(
        self,
        name: str = "cartesian_pose",
        priority: int = 0,
        kp: float = 400.0,
        kd: float = 40.0,
        mode: str = "3d",
        is_6d: Optional[bool] = None,
        use_full_impedance: bool = False,
        trajectory_fn: Optional[Callable[[float], Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]] = None,
        axes: Optional[Sequence[int]] = None
    ) -> None:
        """
        Initialize Cartesian Pose Task.

        Args:
            name: Task identifier string.
            priority: Priority level index (0 = highest priority).
            kp: Proportional stiffness gain.
            kd: Derivative damping gain.
            mode: Task control mode ('3d', '6d', 'xy', 'z').
            is_6d: Optional boolean flag for 6D pose (overrides mode if provided).
            use_full_impedance: If True, uses Lambda mass matrix weighting; if False, uses VMC.
            trajectory_fn: Optional dynamic reference function t -> (pos_des, rot_des, vel_des).
            axes: Optional subset of task-space rows this objective controls, e.g. [0] for x only,
                [1] for y only, [2] for z only, [0, 1] for xy.
        """
        super().__init__(name=name, priority=priority)
        self.kp = kp
        self.kd = kd
        self.mode = mode
        self.use_full_impedance = use_full_impedance
        self.trajectory_fn = trajectory_fn

        if is_6d is not None:
            self.is_6d = is_6d
        else:
            self.is_6d = (mode == "6d")

        if axes is not None:
            self.axes = np.asarray(axes, dtype=int)
        elif mode == "xy":
            self.axes = np.array([0, 1], dtype=int)
        elif mode == "z":
            self.axes = np.array([2], dtype=int)
        else:
            self.axes = None

    @property
    def dim(self) -> int:
        """Returns the task dimension."""
        if self.axes is not None:
            return len(self.axes)
        return 6 if self.is_6d else 3

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
            v_des = state.get("target_vel", np.zeros(6 if self.is_6d else 3))

        if R_des is None:
            R_des = R_curr
        if v_des is None:
            v_des = np.zeros(6 if self.is_6d else 3)

        if self.is_6d:
            J_task = J_full  # (6, n_dofs)
            e_pos = p_des - p_curr
            e_rot = compute_orientation_error(R_curr, R_des)
            e_task = np.concatenate([e_pos, e_rot])
            v_curr = J_task @ dq
        else:
            J_task = J_full[:3, :]  # (3, n_dofs)
            e_task = p_des - p_curr
            v_curr = J_task @ dq

        v_err = v_des - v_curr

        # Restrict to the controlled axes if requested
        if self.axes is not None:
            J_task = J_task[self.axes, :]
            e_task = e_task[self.axes]
            v_err = v_err[self.axes]

        if self.use_full_impedance:
            # Full Cartesian Impedance Law
            # Lambda = (J * B^(-1) * J^T)^(-1)
            B_inv = np.linalg.inv(B)
            Lambda_inv = J_task @ B_inv @ J_task.T
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
        Computes positional tracking error norm matching the controlled dimensions.
        """
        p_curr = state["ee_pos"]
        if self.trajectory_fn is not None:
            p_des = self.trajectory_fn(state.get("t", 0.0))[0]
        else:
            p_des = state.get("target_pos", p_curr)
        err = np.asarray(p_des) - np.asarray(p_curr)
        if self.axes is not None:
            # If axes are within position indices 0..2
            pos_axes = [a for a in self.axes if a < 3]
            if pos_axes:
                err = err[pos_axes]
        return float(np.linalg.norm(err))
