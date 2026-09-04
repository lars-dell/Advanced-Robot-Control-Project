"""
Saturated Algebraic Null-Space Cartesian Impedance Controller.

Implements the classical dynamically consistent null-space projection law with post-hoc torque saturation based on:
    Equation (10) in "Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization"
    Enrico Mingo Hoffman et al. (IEEE ICRA 2018).
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import logging

from controllers.base_controller import BaseController
from tasks.base_task import BaseTask
from tasks.cartesian_task import CartesianPoseTask
from tasks.posture_task import JointPostureTask
from tasks.task_stack import TaskStack
from utils.math_utils import dynamically_consistent_pinv

logger = logging.getLogger(__name__)


class SaturatedAlgebraicController(BaseController):
    """
    Classical Dynamically-Consistent Null-Space Cartesian Impedance Controller with Naive Post-Hoc Saturation.

    Formula:
        tau_0 = J_0^T * f_0
        N_0 = I - J_0^T * (J_bar_0)^T
        tau_opt = tau_0 + N_0 * tau_1
        tau_cmd = clip(tau_opt + h(q, dq), tau_min, tau_max)
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
        reg_pinv: float = 1e-4
    ) -> None:
        """
        Initialize the Saturated Algebraic Null-Space Controller.

        Args:
            n_dofs: Number of robot joints (default: 7).
            tau_min: Minimum joint torque limits (7,).
            tau_max: Maximum joint torque limits (7,).
            kp_cart: Cartesian stiffness gain.
            kd_cart: Cartesian damping gain.
            kp_null: Joint posture stiffness gain.
            kd_null: Joint posture damping gain.
            reg_pinv: Regularization damping for dynamically consistent pseudo-inverse.
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
        self.reg_pinv = reg_pinv

    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Union[Dict[str, Any], TaskStack],
        t: float = 0.0
    ) -> np.ndarray:
        """
        Computes joint torques via algebraic dynamically consistent null-space projection with naive saturation clipping.

        Args:
            state: Robot state dictionary containing 'q', 'dq', 'J', 'B', 'h'.
            target: TaskStack instance OR target dictionary.
            t: Simulation timestamp in seconds.

        Returns:
            np.ndarray: Commanded joint torques of shape (n_dofs,).
        """
        state["handles_inequalities"] = False
        B = state["B"]
        h = state.get("h", np.zeros(self.n_dofs, dtype=np.float64))

        I_n = np.eye(self.n_dofs)

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

        # Multi-task cascading null-space projection:
        # tau_opt = sum_i N_{i-1} * tau_i
        tau_opt = np.zeros(self.n_dofs, dtype=np.float64)
        N_cum = I_n.copy()

        for level, (_, J_i, f_i) in enumerate(evaluated_tasks):
            f_vec = np.asarray(f_i, dtype=np.float64).flatten()
            if J_i.shape[0] == self.n_dofs:
                # Joint space task
                tau_i = f_vec
            else:
                # Operational space task
                tau_i = J_i.T @ f_vec

            # Add projected torque into cumulative null-space
            tau_opt += N_cum @ tau_i

            # Update null-space projector: N_cum = N_cum * (I - J_i^T * J_bar_i^T)
            if J_i.shape[0] < self.n_dofs:
                J_bar_i = dynamically_consistent_pinv(J_i, B, reg=self.reg_pinv)
                N_i = I_n - J_i.T @ J_bar_i.T
                N_cum = N_cum @ N_i

        tau_raw = tau_opt + h
        # Naive post-hoc torque saturation clipping
        tau_cmd = np.clip(tau_raw, self.tau_min, self.tau_max)

        # Record telemetry: unclipped_tau tracks instances where the controller required clipping to stay within limits
        self._update_violation_telemetry(tau_cmd=tau_cmd, tau_min=self.tau_min, tau_max=self.tau_max, unclipped_tau=tau_raw)
        return tau_cmd

