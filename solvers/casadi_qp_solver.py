"""
Pre-compiled parametric CasADi QP solver wrapper.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import casadi as ca
import logging

logger = logging.getLogger(__name__)


class CasADiQPSolver:
    """
    Pre-compiled parametric Quadratic Programming (QP) solver engine using CasADi.

    Pre-compiles CasADi parametric C-functions for Level 0 (unconstrained equality)
    and Level 1+ (constrained priority equality) QPs during initialization.
    """

    def __init__(self, n_vars: int = 7, use_qpoases: bool = True, solver_name: str = None) -> None:
        """
        Initialize the CasADi QP Solver Engine.

        Args:
            n_vars: Number of decision variables (default: 7 for robot joints).
            use_qpoases: Whether to use qpOASES solver plugin (default: True, fallback: OSQP).
        """
        self.n_vars = n_vars
        self.use_qpoases = use_qpoases
        # Explicit override wins; otherwise qpOASES (the paper's solver) or OSQP.
        # qpOASES by default -- the solver Hoffman et al. use.
        #
        # It is an active-set method that warm-starts from its previous working set, so calling it
        # twice with the SAME problem can return answers differing by ~1e-13. That does not occur
        # in a control loop, which presents a sequence of different problems: two runs replaying
        # the same sequence drive the internal state identically and produce bit-identical
        # trajectories. Verified across all three scenarios.
        #
        # Other CasADi conic plugins are selectable by name for comparison; DAQP is stateless and
        # was checked as an alternative, and gives the same trajectories as qpOASES here.
        self.solver_name = solver_name or ("qpoases" if use_qpoases else "osqp")

        # Registry for compiled parametric solver functions keyed by equality constraint dimension eq_dim
        self._compiled_solvers: Dict[int, Dict[str, Any]] = {}

        # Pre-compile standard solver shapes during instantiation
        self._compile_solver(eq_dim=0)  # Level 0 QP (no equality constraints)
        self._compile_solver(eq_dim=3)  # Level 1 QP (3D position priority equality)
        self._compile_solver(eq_dim=6)  # Level 1 QP (6D pose priority equality)

    def _compile_solver(self, eq_dim: int) -> None:
        """
        Pre-compiles a fast parametric CasADi C-function for a given equality constraint dimension.

        Args:
            eq_dim: Number of linear equality constraints (0 for unconstrained equality).
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

        # Linear equality constraints if present: A_eq * x == b_eq
        if eq_dim > 0:
            A_eq_p = opti.parameter(eq_dim, self.n_vars)
            b_eq_p = opti.parameter(eq_dim)
            opti.subject_to(ca.mtimes(A_eq_p, x) == b_eq_p)
            inputs = [H_p, g_p, lb_p, ub_p, A_eq_p, b_eq_p]
        else:
            inputs = [H_p, g_p, lb_p, ub_p]

        # Configure solver backend (qpOASES or OSQP)
        solver_name = self.solver_name
        opts = {
            "printLevel": "none",
            "print_time": False,
            "error_on_fail": False,
        } if solver_name == "qpoases" else {"print_time": False, "error_on_fail": False} if solver_name == "daqp" else {
            "error_on_fail": False,
            "print_time": False,
            "osqp": {"verbose": False}
        }

        try:
            opti.solver(solver_name, opts)
            solver_fn = opti.to_function(f"qp_solver_{eq_dim}", inputs, [x])
            self._compiled_solvers[eq_dim] = {
                "fn": solver_fn,
                "eq_dim": eq_dim
            }
        except Exception as e:
            logger.warning(f"Failed to pre-compile qpOASES for eq_dim={eq_dim}: {e}. Retrying OSQP backend.")
            opts_fb = {"error_on_fail": False, "print_time": False, "osqp": {"verbose": False}}
            opti.solver("osqp", opts_fb)
            solver_fn = opti.to_function(f"qp_solver_{eq_dim}_osqp", inputs, [x])
            self._compiled_solvers[eq_dim] = {
                "fn": solver_fn,
                "eq_dim": eq_dim
            }

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None
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

        Returns:
            np.ndarray: Optimal decision variable array x of shape (n_vars,).
        """
        eq_dim = A_eq.shape[0] if (A_eq is not None and A_eq.size > 0) else 0

        # Dynamically compile solver for new equality constraint dimension if encountered
        if eq_dim not in self._compiled_solvers:
            logger.info(f"Compiling CasADi solver for new equality constraint dimension eq_dim={eq_dim}...")
            self._compile_solver(eq_dim)

        compiled_info = self._compiled_solvers[eq_dim]
        solver_fn = compiled_info["fn"]

        try:
            if eq_dim > 0:
                res = solver_fn(H, g, lb, ub, A_eq, b_eq)
            else:
                res = solver_fn(H, g, lb, ub)
            return np.array(res, dtype=np.float64).flatten()
        except Exception as e:
            logger.error(f"Pre-compiled QP solver evaluation failed for eq_dim={eq_dim}: {e}")
            return np.zeros(self.n_vars, dtype=np.float64)
