"""
Single-Level Weighted-Sum Quadratic Programming (QP) Cartesian Impedance Controller.

Solves a single QP optimization problem combining all prioritized tasks via soft weighting factors:
    min_tau  sum_i w_i * || J_i * B^(-1) * tau - J_i * B^(-1) * J_i^T * f_i ||^2 + eps * || tau ||^2
    subject to  tau_min - h <= tau <= tau_max - h
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import logging

from controllers.base_controller import BaseController
from tasks.base_task import BaseTask
from tasks.cartesian_task import CartesianPoseTask
from tasks.posture_task import JointPostureTask
from tasks.task_stack import TaskStack
from solvers.casadi_qp_solver import CasADiQPSolver

logger = logging.getLogger(__name__)


class WeightedQPController(BaseController):
    """
    Single-Level Weighted-Sum QP Cartesian Impedance Controller.

    Combines multiple task objectives into a single weighted cost function subject to hard joint-torque limits.
    Demonstrates priority bleed / cross-talk when tasks conflict compared to strict cascade QPs.
    """

    def __init__(
        self,
        n_dofs: int = 7,
        tau_min: Optional[np.ndarray] = None,
        tau_max: Optional[np.ndarray] = None,
        kp_cart: float = 400.0,
        kd_cart: float = 40.0,
        kp_null: float = 20.0,
        kd_null: float = 4.0,
        weights: Optional[List[float]] = None,
        solver_name: str = "daqp",
        reg_eps: float = 1e-4
    ) -> None:
        """
        Initialize the Weighted-Sum QP Controller.

        Args:
            n_dofs: Number of robot joints (default: 7).
            tau_min: Minimum joint torque limits (7,).
            tau_max: Maximum joint torque limits (7,).
            kp_cart: Cartesian stiffness gain.
            kd_cart: Cartesian damping gain.
            kp_null: Joint posture stiffness gain.
            kd_null: Joint posture damping gain.
            weights: List of weighting factors per task priority level (default: [1.0, 0.1, 0.01]).
            solver_name: Solver plugin name ("daqp", "osqp", "qpoases"). Defaults to "daqp".
            reg_eps: Regularization factor.
        """
        super().__init__(n_dofs=n_dofs)

        if tau_min is None:
            self.tau_min = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0], dtype=np.float64)
        else:
            self.tau_min = np.asarray(tau_min, dtype=np.float64)

        if tau_max is None:
            self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
        else:
            self.tau_max = np.asarray(tau_max, dtype=np.float64)

        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_null = kp_null
        self.kd_null = kd_null
        self.weights = weights if weights is not None else [1.0, 0.1, 0.01]
        self.solver_name = solver_name or "daqp"
        self.reg_eps = reg_eps

        # CasADi solver engine
        self.qp_solver = CasADiQPSolver(
            n_vars=self.n_dofs,
            solver_name=self.solver_name
        )

    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Union[Dict[str, Any], TaskStack],
        t: float = 0.0
    ) -> np.ndarray:
        """
        Computes joint torques via single-level weighted sum QP optimization.

        Args:
            state: Robot state dictionary containing 'q', 'dq', 'J', 'B', 'h'.
            target: TaskStack instance OR target dictionary.
            t: Simulation timestamp in seconds.

        Returns:
            np.ndarray: Commanded joint torques of shape (n_dofs,).
        """
        B = state["B"]
        h = state.get("h", np.zeros(self.n_dofs, dtype=np.float64))
        B_inv = np.linalg.inv(B)

        lb = self.tau_min - h
        ub = self.tau_max - h

        if isinstance(target, TaskStack):
            evaluated_tasks = target.evaluate_all(state, t)
        else:
            p_curr = state.get("ee_pos", np.zeros(3))
            R_curr = state.get("ee_rot", np.eye(3))
            p_des = target["pos"]
            R_des = target.get("rot", R_curr)
            v_des = target.get("vel", np.zeros(6 if state["J"].shape[0] == 6 else 3))

            mode = "6d" if state["J"].shape[0] == 6 else "3d"
            cart_task = CartesianPoseTask(
                name="cartesian_primary",
                priority=0,
                kp=self.kp_cart,
                kd=self.kd_cart,
                mode=mode
            )

            q_null_des = target.get("q_null", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))
            posture_task = JointPostureTask(
                name="posture_secondary",
                priority=1,
                kp=self.kp_null,
                kd=self.kd_null,
                q_des=q_null_des
            )

            state["target_pos"] = p_des
            state["target_rot"] = R_des
            state["target_vel"] = v_des
            state["q_null"] = q_null_des

            evaluated_tasks = [
                (cart_task, *cart_task.compute(state, t)),
                (posture_task, *posture_task.compute(state, t))
            ]

        # Accumulate weighted quadratic terms: H = sum w_i * M_i^T * M_i + eps * I
        H = self.reg_eps * np.eye(self.n_dofs)
        g = np.zeros(self.n_dofs, dtype=np.float64)

        for level, (_, J_i, f_i) in enumerate(evaluated_tasks):
            w_i = self.weights[level] if level < len(self.weights) else (self.weights[-1] * 0.1)
            f_vec = np.asarray(f_i, dtype=np.float64).flatten()

            if J_i.shape[0] == self.n_dofs:
                # Joint space task
                H += w_i * np.eye(self.n_dofs)
                g += - w_i * f_vec
            else:
                # Operational space task
                M_i = J_i @ B_inv  # (m_i, n_dofs)
                target_proj = M_i @ J_i.T @ f_vec
                H += w_i * (M_i.T @ M_i)
                g += - w_i * (M_i.T @ target_proj)

        # Solve single-level QP
        tau_opt = self.qp_solver.solve(H=H, g=g, lb=lb, ub=ub)

        # Total commanded torque
        tau_cmd = tau_opt + h
        tau_cmd = np.clip(tau_cmd, self.tau_min, self.tau_max)
        return tau_cmd
