"""
Pre-compiled parametric CasADi QP solver wrapper.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import numpy as np

logger = logging.getLogger(__name__)


class CasADiQPSolver:
    """
    Pre-compiled parametric Quadratic Programming (QP) solver engine using CasADi.

    Pre-compiles CasADi parametric C-functions for Level 0 (unconstrained equality)
    and Level 1+ (constrained priority equality) QPs during initialization.
    """

    def __init__(self, n_vars: int = 7, solver_name: str = "daqp") -> None:
        """
        Initialize the CasADi QP Solver Engine.

        Args:
            n_vars: Number of decision variables (default: 7 for robot joints).
            solver_name: Solver plugin name ("daqp", "osqp", "qpoases"). Defaults to "daqp".
        """
        self.n_vars = n_vars
        self.solver_name = solver_name or "daqp"

        # Registry for compiled parametric solver functions keyed by (eq_dim, ineq_dim)
        self._compiled_solvers: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Pre-compile standard solver shapes during instantiation
        self._compile_solver(eq_dim=0, ineq_dim=0)  # Unconstrained
        self._compile_solver(eq_dim=3, ineq_dim=0)  # 3D position priority equality
        self._compile_solver(eq_dim=6, ineq_dim=0)  # 6D pose priority equality
        self._compile_solver(eq_dim=0, ineq_dim=1)  # 1D inequality (e.g. height corridor)
        self._compile_solver(eq_dim=3, ineq_dim=1)  # 3D equality + 1D inequality
        self._compile_solver(eq_dim=6, ineq_dim=1)  # 6D equality + 1D inequality

    def _compile_solver(self, eq_dim: int, ineq_dim: int = 0) -> None:
        """
        Pre-compiles a fast parametric CasADi C-function for given equality and inequality constraint dimensions.

        Args:
            eq_dim: Number of linear equality constraints (0 for none).
            ineq_dim: Number of two-sided linear inequality constraints (0 for none).
        """
        opti = ca.Opti("conic")
        x = opti.variable(self.n_vars)

        # Declare symbolic parameters for fast numerical updates
        H_p = opti.parameter(self.n_vars, self.n_vars)
        g_p = opti.parameter(self.n_vars)
        lb_p = opti.parameter(self.n_vars)
        ub_p = opti.parameter(self.n_vars)

        # Ensure symmetric Hessian formulation
        H_sym = 0.5 * (H_p + H_p.T)

        # Quadratic objective function
        obj = 0.5 * ca.mtimes([x.T, H_sym, x]) + ca.dot(g_p, x)
        opti.minimize(obj)

        # Bound constraints: lb <= x <= ub
        opti.subject_to(opti.bounded(lb_p, x, ub_p))

        inputs = [H_p, g_p, lb_p, ub_p]

        # Linear equality constraints if present: A_eq * x == b_eq
        if eq_dim > 0:
            A_eq_p = opti.parameter(eq_dim, self.n_vars)
            b_eq_p = opti.parameter(eq_dim)
            opti.subject_to(ca.mtimes(A_eq_p, x) == b_eq_p)
            inputs.extend([A_eq_p, b_eq_p])

        # Linear inequality constraints if present: b_ineq_lb <= A_ineq * x <= b_ineq_ub
        if ineq_dim > 0:
            A_ineq_p = opti.parameter(ineq_dim, self.n_vars)
            b_ineq_lb_p = opti.parameter(ineq_dim)
            b_ineq_ub_p = opti.parameter(ineq_dim)
            opti.subject_to(opti.bounded(b_ineq_lb_p, ca.mtimes(A_ineq_p, x), b_ineq_ub_p))
            inputs.extend([A_ineq_p, b_ineq_lb_p, b_ineq_ub_p])

        # Configure solver backend (DAQP, qpOASES, or OSQP)
        solver_name = self.solver_name
        if solver_name == "qpoases":
            opts = {
                "printLevel": "none",
                "print_time": False,
                "error_on_fail": False,
            }
        elif solver_name == "daqp":
            opts = {
                "print_time": False,
                "error_on_fail": False,
            }
        else:
            # OSQP backend with high accuracy and active-set polishing
            opts = {
                "error_on_fail": False,
                "print_time": False,
                "osqp": {
                    "verbose": False,
                    "polish": True,
                    "eps_abs": 1e-7,
                    "eps_rel": 1e-7,
                }
            }

        key = (eq_dim, ineq_dim)
        try:
            opti.solver(solver_name, opts)
            solver_fn = opti.to_function(f"qp_solver_{eq_dim}_{ineq_dim}", inputs, [x])
            self._compiled_solvers[key] = {
                "fn": solver_fn,
                "eq_dim": eq_dim,
                "ineq_dim": ineq_dim
            }
        except Exception as e:
            logger.warning(f"Failed to pre-compile {solver_name} for (eq={eq_dim}, ineq={ineq_dim}): {e}. Retrying OSQP backend.")
            opts_fb = {
                "error_on_fail": False,
                "print_time": False,
                "osqp": {
                    "verbose": False,
                    "polish": True,
                    "eps_abs": 1e-7,
                    "eps_rel": 1e-7,
                }
            }
            opti.solver("osqp", opts_fb)
            solver_fn = opti.to_function(f"qp_solver_{eq_dim}_{ineq_dim}_osqp", inputs, [x])
            self._compiled_solvers[key] = {
                "fn": solver_fn,
                "eq_dim": eq_dim,
                "ineq_dim": ineq_dim
            }

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq_lb: Optional[np.ndarray] = None,
        b_ineq_ub: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solves a QP using pre-compiled parametric CasADi functions.

        Args:
            H: Quadratic cost matrix (n_vars, n_vars).
            g: Linear cost vector (n_vars,).
            lb: Lower bound vector (n_vars,).
            ub: Upper bound vector (n_vars,).
            A_eq: Optional linear equality matrix (eq_dim, n_vars).
            b_eq: Optional linear equality vector (eq_dim,).
            A_ineq: Optional linear inequality matrix (ineq_dim, n_vars).
            b_ineq_lb: Optional linear inequality lower bound vector (ineq_dim,).
            b_ineq_ub: Optional linear inequality upper bound vector (ineq_dim,).

        Returns:
            np.ndarray: Optimal decision variable array x of shape (n_vars,).
        """
        eq_dim = A_eq.shape[0] if (A_eq is not None and A_eq.size > 0) else 0
        ineq_dim = A_ineq.shape[0] if (A_ineq is not None and A_ineq.size > 0) else 0

        key = (eq_dim, ineq_dim)
        # Dynamically compile solver for new constraint dimensions if encountered
        if key not in self._compiled_solvers:
            logger.info(f"Compiling CasADi solver for constraint dimensions eq_dim={eq_dim}, ineq_dim={ineq_dim}...")
            self._compile_solver(eq_dim=eq_dim, ineq_dim=ineq_dim)

        compiled_info = self._compiled_solvers[key]
        solver_fn = compiled_info["fn"]

        call_args = [H, g, lb, ub]
        if eq_dim > 0:
            call_args.extend([A_eq, b_eq])
        if ineq_dim > 0:
            call_args.extend([A_ineq, b_ineq_lb, b_ineq_ub])

        try:
            res = solver_fn(*call_args)
            x_opt = np.array(res, dtype=np.float64).flatten()
            # Cleanly project minor floating-point solver tolerances (e.g. 1e-5) back onto box bounds
            if lb is not None and ub is not None:
                x_opt = np.clip(x_opt, lb, ub)
            return x_opt
        except Exception as e:
            logger.error(f"Pre-compiled QP solver evaluation failed for (eq_dim={eq_dim}, ineq_dim={ineq_dim}): {e}")
            return np.zeros(self.n_vars, dtype=np.float64)
