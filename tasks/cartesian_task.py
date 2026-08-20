"""
Cartesian space impedance control task implementation.
"""

from typing import Dict, Tuple, Optional, Callable, Any, Sequence
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
        is_6d: bool = False,
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
            is_6d: Whether to control 6D pose (True) or 3D position (False).
            use_full_impedance: If True, uses Lambda mass matrix weighting; if False, uses VMC.
            trajectory_fn: Optional dynamic reference function t -> (pos_des, rot_des, vel_des).
            axes: Subset of task-space rows this objective controls, e.g. [0] for x only or
                [2] for z only. None controls all of them (3 for position, 6 for pose).

                This is what lets a single Cartesian position be split into several independent
                objectives at different priorities, which is the decomposition Hoffman et al. use
                in section V-A: "reach 1.0 m along x", "reach 1.0 m along y" and "follow a sinusoid
                in z" are three one-dimensional tasks, not one three-dimensional one. Splitting
                them is what makes the priority ordering observable -- the axes then compete for
                the same joints and the hierarchy decides which one is sacrificed.
        """
        super().__init__(name=name, priority=priority)
        self.kp = kp
        self.kd = kd
        self.is_6d = is_6d
        self.use_full_impedance = use_full_impedance
        self.trajectory_fn = trajectory_fn
        self.axes = None if axes is None else np.asarray(axes, dtype=int)

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

        # Restrict to the controlled axes. Done after the full task-space velocity is formed, so
        # v_curr still comes from the complete Jacobian row set.
        if self.axes is not None:
            J_task = J_task[self.axes, :]
            e_task = e_task[self.axes]
            v_err = v_err[self.axes]

        if self.use_full_impedance:
            # Full Cartesian Impedance Law
            # Lambda = (J * B^(-1) * J^T)^(-1)
            B_inv = np.linalg.inv(B)
            Lambda_inv = J_task @ B_inv @ J_task.T
            # Damped pseudo-inverse near singularities
            reg = 1e-4 * np.eye(Lambda_inv.shape[0])
            Lambda = np.linalg.inv(Lambda_inv + reg)
            
            # Acceleration feedforward (default to zero if unmodeled)
            acc_des = state.get("target_acc", np.zeros(J_task.shape[0]))
            f_cart = Lambda @ acc_des + self.kp * e_task + self.kd * v_err
        else:
            # Virtual Model Control (VMC) / Spring-Damper Law
            f_cart = self.kp * e_task + self.kd * v_err

        return J_task, f_cart

    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes positional tracking error norm.
        """
        p_curr = state["ee_pos"]
        if self.trajectory_fn is not None:
            p_des = self.trajectory_fn(state.get("t", 0.0))[0]
        else:
            p_des = state.get("target_pos", p_curr)
        err = np.asarray(p_des) - np.asarray(p_curr)
        if self.axes is not None:
            err = err[self.axes]
        return float(np.linalg.norm(err))
