"""
Classical Jacobian Transpose Cartesian Impedance Controller (No Priority Hierarchy).

Implements the classical transpose control law based on:
    Equation (9) in "Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization"
    Enrico Mingo Hoffman et al. (IEEE ICRA 2018).
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from controllers.base_controller import BaseController
from tasks.cartesian_task import CartesianPoseTask
from tasks.posture_task import JointPostureTask
from tasks.task_stack import TaskStack

logger = logging.getLogger(__name__)


class ClassicalTransposeController(BaseController):
    """
    Classical Jacobian Transpose Cartesian Impedance Controller without task priority hierarchy.

    Directly maps virtual task wrenches to joint torques via the Jacobian transpose:
        tau = sum_i J_i^T * f_i + h(q, dq)
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
    ) -> None:
        """
        Initialize the Classical Transpose Controller.

        Args:
            n_dofs: Number of robot joints (default: 7).
            tau_min: Minimum joint torque limits (7,).
            tau_max: Maximum joint torque limits (7,).
            kp_cart: Cartesian stiffness gain.
            kd_cart: Cartesian damping gain.
            kp_null: Joint posture stiffness gain.
            kd_null: Joint posture damping gain.
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

    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Union[Dict[str, Any], TaskStack],
        t: float = 0.0
    ) -> np.ndarray:
        """
        Computes joint torques via simple Jacobian Transpose summation.

        Formula:
            tau_cmd = sum_i (J_i^T * f_i) + h(q, dq)

        Args:
            state: Robot state dictionary containing 'q', 'dq', 'J', 'B', 'h'.
            target: TaskStack instance OR target dictionary.
            t: Simulation timestamp in seconds.

        Returns:
            np.ndarray: Commanded joint torques of shape (n_dofs,).
        """
        state["handles_inequalities"] = False
        h = state.get("h", np.zeros(self.n_dofs, dtype=np.float64))
        tau_opt = np.zeros(self.n_dofs, dtype=np.float64)


        if isinstance(target, TaskStack):
            evaluated_tasks = target.evaluate_all(state, t)
            for _, J_i, f_i in evaluated_tasks:
                if J_i.shape[0] == self.n_dofs:
                    # Joint space task (e.g. posture)
                    tau_opt += np.asarray(f_i, dtype=np.float64).flatten()
                else:
                    # Operational space Cartesian task
                    tau_opt += J_i.T @ np.asarray(f_i, dtype=np.float64).flatten()
        else:
            # Construct default Cartesian + posture tasks from target dict
            p_curr = state.get("ee_pos", np.zeros(3))
            R_curr = state.get("ee_rot", np.eye(3))
            p_des = target["pos"]
            R_des = target.get("rot", R_curr)
            v_des = target.get("vel", np.zeros(6 if state["J"].shape[0] == 6 else 3))

            mode = "6d" if state["J"].shape[0] == 6 else "3d"
            cart_task = CartesianPoseTask(
                name="cartesian_task",
                priority=0,
                kp=self.kp_cart,
                kd=self.kd_cart,
                mode=mode
            )

            q_null_des = target.get("q_null", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))
            posture_task = JointPostureTask(
                name="posture_task",
                priority=1,
                kp=self.kp_null,
                kd=self.kd_null,
                q_des=q_null_des
            )

            state["target_pos"] = p_des
            state["target_rot"] = R_des
            state["target_vel"] = v_des
            state["q_null"] = q_null_des

            J_cart, f_cart = cart_task.compute(state, t)
            J_null, f_null = posture_task.compute(state, t)

            tau_opt += J_cart.T @ f_cart
            tau_opt += f_null

        # Total commanded torque with feedforward gravity/Coriolis compensation
        tau_cmd = tau_opt + h

        # Check torque limits (Classical transpose does not bound torques, records violations)
        self._update_violation_telemetry(tau_cmd=tau_cmd, tau_min=self.tau_min, tau_max=self.tau_max)
        return tau_cmd

