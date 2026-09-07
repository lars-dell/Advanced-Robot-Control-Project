"""
Operational space height boundary (ceiling and floor) task implementation.

Enforces unilateral safety constraints on end-effector height:
    - Upper bound (Ceiling): z <= z_max
    - Lower bound (Floor):   z >= z_min
Formulated as strict QP inequality constraints b_l <= A_ineq * tau <= b_u based on
Hoffman et al. (ICRA 2018) Equation (18).
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from tasks.base_task import BaseTask


class ZBoundaryTask(BaseTask):
    """
    Operational space Z-coordinate height boundary and corridor task.

    Strictly guarantees that the end-effector remains below z_max and above z_min:
        - When inside corridor [z_min, z_max]: inequality constraint is inactive.
        - When approaching or reaching z_max: hard inequality J_z B^-1 tau <= b_u prevents penetration.
        - When approaching or reaching z_min: hard inequality J_z B^-1 tau >= b_l prevents penetration.
    """

    def __init__(
        self,
        name: str = "z_corridor_p0",
        priority: int = 0,
        z_min: float = 0.35,
        z_max: float = 0.55,
        nominal_z_fn: Optional[Callable[[float], Tuple[float, float]]] = None,
        as_inequality: bool = True,
        omega_n: float = 35.0,
        kp: float = 1200.0,
        kd: float = 80.0,
    ) -> None:
        """
        Initialize Z-Boundary / Corridor Task.

        Args:
            name: Task identifier string.
            priority: Priority level index (0 = highest priority).
            z_min: Floor boundary height in meters (default: 0.35m).
            z_max: Ceiling boundary height in meters (default: 0.55m).
            nominal_z_fn: Optional callable t -> (z_nom, vz_nom) providing unconstrained vertical reference.
            as_inequality: If True, enforces strict inequality constraints in QP without equality cost.
            omega_n: Control barrier function natural frequency for safe deceleration (rad/s).
            kp: Proportional stiffness gain along Z (fallback equality mode).
            kd: Derivative damping gain along Z (fallback equality mode).
        """
        super().__init__(name=name, priority=priority)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.nominal_z_fn = nominal_z_fn
        self.as_inequality = as_inequality
        self.omega_n = float(omega_n)
        self.kp = float(kp)
        self.kd = float(kd)

        # State diagnostics
        self.ceiling_active = False
        self.floor_active = False
        self.current_violation = 0.0

    def compute_inequality(
        self, state: Dict[str, np.ndarray], t: float = 0.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Computes the operational-space height corridor inequality constraint:
            b_l <= J_z B^-1 tau_qp <= b_u

        Enforces:
            z <= z_max (Ceiling limit)
            z >= z_min (Floor limit)
        using a second-order control barrier function and discrete viability horizon.
        """
        if not self.as_inequality:
            return None

        J_full = state["J"]  # (6, n_dofs)
        n_dofs = J_full.shape[1]
        p_curr = state["ee_pos"]
        dq = state.get("dq", np.zeros(n_dofs))
        B = state["B"]
        dt = float(state.get("dt", 0.005))

        z_curr = float(p_curr[2])
        J_z = J_full[2:3, :]  # (1, n_dofs)
        v_z = float((J_z @ dq)[0])

        B_inv = np.linalg.inv(B)
        A_ineq = J_z @ B_inv  # (1, n_dofs)

        # Retrieve nominal reference to update diagnostic flags
        if self.nominal_z_fn is not None:
            z_nom, _ = self.nominal_z_fn(t)
        else:
            z_nom = state.get("target_pos", p_curr)[2]

        self.ceiling_active = bool(z_nom > self.z_max or z_curr >= self.z_max - 0.001)
        self.floor_active = bool(z_nom < self.z_min or z_curr <= self.z_min + 0.001)
        self.current_violation = max(0.0, z_curr - self.z_max, self.z_min - z_curr)

        # Second-order viability and CBF limits on vertical acceleration:
        omega_n = self.omega_n

        # Upper bound (Ceiling: z <= z_max)
        d_ceil = self.z_max - z_curr
        a_max_cbf = omega_n * omega_n * d_ceil - 2.0 * omega_n * v_z
        a_max_discrete = (2.0 / (dt * dt)) * d_ceil - (2.0 / dt) * v_z
        bu_val = min(a_max_cbf, a_max_discrete)

        # Lower bound (Floor: z >= z_min)
        d_flr = self.z_min - z_curr
        a_min_cbf = omega_n * omega_n * d_flr - 2.0 * omega_n * v_z
        a_min_discrete = (2.0 / (dt * dt)) * d_flr - (2.0 / dt) * v_z
        bl_val = max(a_min_cbf, a_min_discrete)

        # Convective operational acceleration offset:
        # Since ddz = J_z ddq + dJ_z dq = J_z B^-1 tau + dJ_z dq,
        # the acceleration bounds b_l <= ddz <= b_u map to torque space as:
        #     b_l - dJ_z dq <= J_z B^-1 tau <= b_u - dJ_z dq
        dJ_dq = state.get("dJ_dq", None)
        if dJ_dq is not None and len(dJ_dq) > 2:
            dJ_dq_z = float(dJ_dq[2])
        else:
            dJ_dq_z = 0.0

        bu_val -= dJ_dq_z
        bl_val -= dJ_dq_z

        # Actuator feasibility and kinematic singularity safeguard:
        # When J_z -> 0 (loss of vertical control authority), A_ineq -> 0.
        # Ensure 0 <= bu and bl <= 0 so 0*tau remains feasible when authority is lost.
        norm_A = float(np.linalg.norm(A_ineq))
        if norm_A < 1e-4:
            bu_val = max(bu_val, 0.0)
            bl_val = min(bl_val, 0.0)

        # Numerical safeguard against numerical overlap
        if bl_val > bu_val:
            mid = 0.5 * (bl_val + bu_val)
            bl_val = mid
            bu_val = mid

        return A_ineq, np.array([bl_val], dtype=np.float64), np.array([bu_val], dtype=np.float64)

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes task objective.
        - For QP controllers supporting hard inequalities: returns an empty equality task so the
          corridor is handled via strict QP linear inequalities (b_l <= A_ineq * tau <= b_u).
        - For non-QP controllers (Transpose, Saturated Algebraic): automatically activates a unilateral
          virtual barrier spring-damper whenever the end-effector breaches or approaches the corridor.
        """
        J_full = state["J"]
        n_dofs = J_full.shape[1]

        controller_handles_ineq = state.get("handles_inequalities", False)

        if self.as_inequality and controller_handles_ineq:
            # Native inequality mode for QP controllers: corridor is an inequality constraint, not an equality cost
            return np.zeros((0, n_dofs), dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Unilateral barrier spring-damper mode (for non-QP controllers or when as_inequality=False)
        p_curr = state["ee_pos"]
        dq = state.get("dq", np.zeros(n_dofs))
        z_curr = float(p_curr[2])
        J_z = J_full[2:3, :]
        v_z = float((J_z @ dq)[0])

        f_z = 0.0
        active = False

        if z_curr > self.z_max:
            # Penetrating ceiling -> downward repulsive force
            penetration = z_curr - self.z_max
            f_z = -self.kp * penetration - self.kd * v_z
            self.ceiling_active = True
            self.floor_active = False
            active = True
        elif z_curr < self.z_min:
            # Penetrating floor -> upward repulsive force
            penetration = self.z_min - z_curr
            f_z = self.kp * penetration - self.kd * v_z
            self.ceiling_active = False
            self.floor_active = True
            active = True
        else:
            self.ceiling_active = False
            self.floor_active = False

        self.current_violation = max(0.0, z_curr - self.z_max, self.z_min - z_curr)

        if active:
            return J_z, np.array([f_z], dtype=np.float64)
        else:
            return np.zeros((0, n_dofs), dtype=np.float64), np.zeros(0, dtype=np.float64)


    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes boundary violation error in meters (0.0 if inside safe bounds [z_min, z_max]).
        """
        p_curr = state["ee_pos"]
        z = p_curr[2]
        return max(0.0, float(z - self.z_max), float(self.z_min - z))
