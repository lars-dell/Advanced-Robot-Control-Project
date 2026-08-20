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

        # Feasibility accounting. The paper guarantees tau_cmd stays inside [tau_min, tau_max]
        # without post-hoc clipping; these counters are the evidence for that claim.
        self.violation_tol = 1e-6
        # Per-level diagnostics, refreshed each compute_torques() call. priority_residuals[k] is
        # || J_k B^-1 tau_final - J_k B^-1 tau_k* ||: how far the final solution drifted from what
        # priority level k had already decided. Strict priority means these sit at solver tolerance.
        self.level_torques: list = []
        self.priority_residuals: list = []
        self.n_violations = 0
        self.n_solves = 0
        self.max_violation = 0.0
        self.n_qp_failures = 0

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

    def _is_feasible(
        self,
        tau: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        tol: float = 1e-6
    ) -> bool:
        """
        Check a QP solution against its own bounds and equality constraints.

        The solver does not raise on failure, so this is the only way to know whether the returned
        vector is a solution or a fallback.
        """
        if tau is None or tau.shape != lb.shape or not np.all(np.isfinite(tau)):
            return False
        if np.any(tau < lb - tol) or np.any(tau > ub + tol):
            return False
        if A_eq is not None and np.linalg.norm(A_eq @ tau - b_eq) > 1e-4 * max(1.0, np.linalg.norm(b_eq)):
            return False
        return True

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

        # Dynamic bounds compensation (eq. 21): tau_min - h <= tau <= tau_max - h
        lb = self.tau_min - h
        ub = self.tau_max - h

        # If the feed-forward alone exceeds a joint's limit the box is empty and the QP is
        # infeasible. Collapse that joint's bounds to the achievable value rather than handing the
        # solver an impossible problem, and let the violation counter record the consequence.
        crossed = lb > ub
        if np.any(crossed):
            mid = 0.5 * (lb[crossed] + ub[crossed])
            lb[crossed] = mid
            ub[crossed] = mid

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
        level_torques: list = []
        tau_opt = np.zeros(n_dofs, dtype=np.float64)
        tau_fallback = np.clip(np.zeros(n_dofs, dtype=np.float64), lb, ub)

        for level, (task, J_i, f_i) in enumerate(evaluated_tasks):
            # Mapping matrix M_i = J_i * B^(-1), shape (m_i, n_dofs)
            M_i = J_i @ B_inv

            # Equation (18), applied identically at every priority level:
            #     min_tau || M_i tau - M_i J_i^T f_i ||^2 + eps ||tau||^2
            # Expanded to the QP's 1/2 tau^T H tau + g^T tau form.
            #
            # For a joint-space task (J_i = I) this reduces to || B^-1 (tau - f_i) ||^2, i.e.
            # inertia-weighted, which is what the paper specifies. An earlier version special-cased
            # level >= 1 as || tau - f_i ||^2 (identity-weighted); that was not the paper's cost and
            # it silently assumed f_i was a joint-torque vector, so any Cartesian task below level 0
            # failed on shape.
            target_force_proj = M_i @ J_i.T @ f_i          # (m_i,)
            H_i = M_i.T @ M_i + self.reg_eps * np.eye(n_dofs)
            g_i = -M_i.T @ target_force_proj               # (n_dofs,)

            # Strict priority: reproduce every higher-priority level's optimum exactly (eq. 17).
            if prev_A_eq_list:
                A_eq = np.vstack(prev_A_eq_list)
                b_eq = np.concatenate(prev_b_eq_list)
            else:
                A_eq = None
                b_eq = None

            tau_i = self.qp_solver.solve(H=H_i, g=g_i, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq)

            # Validate before trusting it. The solver is configured with error_on_fail=False and
            # returns zeros from its exception handler, and zero is NOT a feasible point when the
            # bounds are shifted far from the origin by a large feed-forward h. Accepting it
            # silently destroys both the torque guarantee and the priority ordering.
            #
            # tau_fallback is always feasible: at level 0 it is the origin projected into the box,
            # and afterwards it is the previous level's optimum, which satisfies this level's bounds
            # and every equality constraint carried into it by construction.
            if not self._is_feasible(tau_i, lb, ub, A_eq, b_eq):
                self.n_qp_failures += 1
                if self.n_qp_failures <= 5:
                    logger.warning(
                        f"QP at priority level {level} returned an infeasible point; "
                        f"falling back to the last feasible solution"
                    )
                tau_i = tau_fallback
            tau_opt = tau_i
            tau_fallback = tau_opt.copy()

            # Carry this level's optimum forward as a constraint on all lower levels.
            prev_A_eq_list.append(M_i)
            prev_b_eq_list.append(M_i @ tau_opt)
            level_torques.append(tau_opt.copy())

        # Priority check: did the final solution preserve every higher level's task-space outcome?
        self.level_torques = level_torques
        self.priority_residuals = [
            float(np.linalg.norm(A_k @ tau_opt - b_k))
            for A_k, b_k in zip(prev_A_eq_list, prev_b_eq_list)
        ]

        # Final total joint torque command: tau_cmd = tau_opt + h(q, dq)   (eq. 20)
        tau_cmd = tau_opt + h

        # No clipping. Equations (19)-(21) make the result provably feasible: the QP was bounded by
        # [tau_min - h, tau_max - h], so adding h back lands inside [tau_min, tau_max]. Clipping here
        # would silently discard the QP's optimality and priority ordering, and would mask any bug
        # that broke the guarantee. Measure it instead.
        overshoot = float(np.max(np.maximum(self.tau_min - tau_cmd, tau_cmd - self.tau_max)))
        self.n_solves += 1
        if overshoot > self.violation_tol:
            self.n_violations += 1
            self.max_violation = max(self.max_violation, overshoot)
            if self.n_violations <= 5:
                logger.warning(
                    f"torque bound violated by {overshoot:.4f} Nm - the eq.(19)-(21) guarantee "
                    f"does not hold, check that h and the QP bounds agree"
                )

        return tau_cmd
