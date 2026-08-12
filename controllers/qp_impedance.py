"""
Multi-Priority Cartesian Impedance Controller using CasADi and qpOASES solver.

Implements the exact hierarchical QP formulation with strict priority equality constraints
and dynamic joint-torque limit bounds based on:
    "Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization"
    Enrico Mingo Hoffman et al. (IEEE ICRA 2018).
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import logging

from controllers.base_controller import BaseController
from tasks.base_task import BaseTask
from tasks.cartesian_task import CartesianPoseTask
from tasks.posture_task import JointPostureTask
from tasks.task_stack import TaskStack
from solvers.casadi_qp_solver import CasADiQPSolver

logger = logging.getLogger(__name__)


class QPImpedanceController(BaseController):
    """
    Hierarchical Quadratic Programming (QP) Multi-Priority Cartesian Impedance Controller.

    Solves a cascade of QPs to achieve multi-task tracking
    subject to strict task priority equality constraints and dynamic joint torque limits.
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
        use_qpoases: bool = True,
        reg_eps: float = 1e-4
    ) -> None:
        """
        Initialize the Hierarchical QP Impedance Controller.

        Args:
            n_dofs: Number of robot joints (default: 7).
            tau_min: Minimum joint torque limits of shape (n_dofs,).
            tau_max: Maximum joint torque limits of shape (n_dofs,).
            kp_cart: Cartesian proportional stiffness gain (default fallback task).
            kd_cart: Cartesian derivative damping gain (default fallback task).
            kp_null: Null-space posture proportional gain (default fallback task).
            kd_null: Null-space posture derivative gain (default fallback task).
            use_qpoases: Whether to configure CasADi solver with qpOASES plugin.
            reg_eps: Quadratic regularization weight epsilon for QP objective.
        """
        super().__init__(n_dofs=n_dofs)

        # Torque limits (default Franka Panda bounds if unspecified)
        if tau_min is None:
            self.tau_min = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0], dtype=np.float64)
        else:
            self.tau_min = np.asarray(tau_min, dtype=np.float64)

        if tau_max is None:
            self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
        else:
            self.tau_max = np.asarray(tau_max, dtype=np.float64)

        # Control gains for default tasks
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_null = kp_null
        self.kd_null = kd_null
        self.use_qpoases = use_qpoases
        self.reg_eps = reg_eps

        # Instantiate pre-compiled parametric CasADi QP solver module
        self.qp_solver = CasADiQPSolver(n_vars=self.n_dofs, use_qpoases=self.use_qpoases)

    def solve_single_qp(
        self,
        H: np.ndarray,
        g: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Delegates QP solving to the pre-compiled CasADi solver module.

        Args:
            H: Quadratic cost matrix (n_vars, n_vars).
            g: Linear cost vector (n_vars,).
            lb: Lower bounds vector (n_vars,).
            ub: Upper bounds vector (n_vars,).
            A_eq: Optional equality constraint matrix (n_eq, n_vars).
            b_eq: Optional equality constraint vector (n_eq,).

        Returns:
            np.ndarray: Optimal decision variable vector x of shape (n_vars,).
        """
        return self.qp_solver.solve(H=H, g=g, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq)

    def compute_torques(
        self,
        state: Dict[str, np.ndarray],
        target: Union[Dict[str, Any], TaskStack],
        t: float = 0.0
    ) -> np.ndarray:
        """
        Computes joint torques via Hierarchical QP Cascade Optimization.

        Steps:
            1. Extract state (q, dq, J, B, h).
            2. Compute dynamic bounds: lb = tau_min - h, ub = tau_max - h.
            3. Evaluate task stack or default targets.
            4. Level 0 QP (Primary Task):
               min_{tau_0} || J_0 B^-1 tau_0 - J_0 B^-1 J_0^T f_0 ||^2 + eps || tau_0 ||^2
               s.t. lb <= tau_0 <= ub
               Yields optimal primary torques tau_0^*.
            5. Level 1 QP (Secondary Task):
               min_{tau_1} || tau_1 - tau_null ||^2 + eps || tau_1 ||^2
               s.t. J_0 B^-1 tau_1 = J_0 B^-1 tau_0^*  (Priority Equality Constraint)
                    lb <= tau_1 <= ub
               Yields optimal secondary torques tau_1^*.
            6. Return final commanded torques: tau_cmd = tau_1^* + h.

        Args:
            state: Current robot state dictionary.
            target: Target dictionary OR a TaskStack instance.
            t: Simulation timestamp in seconds.

        Returns:
            np.ndarray: Desired joint torques of shape (n_dofs,).
        """
        q = state["q"]
        dq = state["dq"]
        B = state["B"]
        n_dofs = self.n_dofs

        # Extract dynamic bias vector h(q, dq) = C(q, dq)*dq + g(q)
        h = state.get("h", np.zeros(n_dofs, dtype=np.float64))

        # Dynamic bounds compensation: tau_min - h <= tau <= tau_max - h
        lb = self.tau_min - h
        ub = self.tau_max - h

        # Inverse of inertia matrix B
        B_inv = np.linalg.inv(B)

        # Build task list from TaskStack or target dict
        if isinstance(target, TaskStack):
            evaluated_tasks = target.evaluate_all(state, t)
        else:
            # Construct default 2-level task hierarchy from target dictionary
            p_curr = state.get("ee_pos", np.zeros(3))
            R_curr = state.get("ee_rot", np.eye(3))
            p_des = target["pos"]
            R_des = target.get("rot", R_curr)
            v_des = target.get("vel", np.zeros(6 if state["J"].shape[0] == 6 else 3))

            is_6d = (state["J"].shape[0] == 6)
            cart_task = CartesianPoseTask(
                name="cartesian_primary",
                priority=0,
                kp=self.kp_cart,
                kd=self.kd_cart,
                is_6d=is_6d,
                use_full_impedance=target.get("use_full_impedance", False)
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

        # Process Cascade QPs across priority levels
        prev_A_eq_list = []
        prev_b_eq_list = []
        tau_opt = np.zeros(n_dofs, dtype=np.float64)

        for level, (task, J_i, f_i) in enumerate(evaluated_tasks):
            # Compute mapping matrix M_i = J_i * B^(-1)
            M_i = J_i @ B_inv  # (m_i, n_dofs)

            if level == 0:
                # Level 0 QP: Primary Cartesian Task Optimization
                target_force_proj = M_i @ J_i.T @ f_i  # (m_i,)
                H0 = M_i.T @ M_i + self.reg_eps * np.eye(n_dofs)
                g0 = - M_i.T @ target_force_proj

                tau_opt = self.qp_solver.solve(
                    H=H0,
                    g=g0,
                    lb=lb,
                    ub=ub
                )

                # Store equality constraint for priority preservation in subsequent QPs
                prev_A_eq_list.append(M_i)
                prev_b_eq_list.append(M_i @ tau_opt)

            else:
                # Level 1+ QP: Secondary Task subject to Strict Priority Equality Constraints
                H_i = (1.0 + self.reg_eps) * np.eye(n_dofs)
                g_i = - f_i  # f_i is posture torque tau_null for JointPostureTask

                A_eq = np.vstack(prev_A_eq_list)
                b_eq = np.concatenate(prev_b_eq_list)

                tau_opt = self.qp_solver.solve(
                    H=H_i,
                    g=g_i,
                    lb=lb,
                    ub=ub,
                    A_eq=A_eq,
                    b_eq=b_eq
                )

                # Update equality constraints list for higher levels if any
                prev_A_eq_list.append(M_i)
                prev_b_eq_list.append(M_i @ tau_opt)

        # Final total joint torque command: tau_cmd = tau_opt + h(q, dq)
        tau_cmd = tau_opt + h

        # Enforce strict saturation clipping as safeguard
        tau_cmd = np.clip(tau_cmd, self.tau_min, self.tau_max)

        return tau_cmd
