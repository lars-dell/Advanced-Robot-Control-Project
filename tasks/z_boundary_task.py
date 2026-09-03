"""
Operational space height boundary (ceiling and floor) task implementation.

Enforces unilateral safety constraints on end-effector height:
    - Upper bound (Ceiling): z <= z_max
    - Lower bound (Floor):   z >= z_min
While tracking nominal trajectory height within the admissible corridor [z_min, z_max].
"""

from typing import Dict, Tuple, Optional, Callable
import numpy as np

from tasks.base_task import BaseTask


class ZBoundaryTask(BaseTask):
    """
    Operational space Z-coordinate height boundary and corridor task.

    Guarantees that the end-effector strictly remains below z_max and above z_min:
        - When nominal z > z_max: clamps reference to z_max (active ceiling constraint)
        - When nominal z < z_min: clamps reference to z_min (active floor constraint)
        - When z_min <= nominal z <= z_max: tracks nominal vertical trajectory smoothly
    """

    def __init__(
        self,
        name: str = "z_corridor_p0",
        priority: int = 0,
        z_min: float = 0.35,
        z_max: float = 0.55,
        nominal_z_fn: Optional[Callable[[float], Tuple[float, float]]] = None,
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
            kp: Proportional stiffness gain along Z.
            kd: Derivative damping gain along Z.
        """
        super().__init__(name=name, priority=priority)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.nominal_z_fn = nominal_z_fn
        self.kp = float(kp)
        self.kd = float(kd)

        # State diagnostics
        self.ceiling_active = False
        self.floor_active = False
        self.current_violation = 0.0

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Z-axis Jacobian J_z and virtual impedance force f_z.

        Args:
            state: Robot state dictionary containing 'J', 'ee_pos', 'dq'.
            t: Simulation timestamp in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (J_z of shape (1, n_dofs), f_z of shape (1,))
        """
        J_full = state["J"]  # (6, n_dofs)
        n_dofs = J_full.shape[1]
        p_curr = state["ee_pos"]
        dq = state.get("dq", np.zeros(n_dofs))

        z_curr = p_curr[2]
        J_z = J_full[2:3, :]  # (1, n_dofs)
        v_z = float((J_z @ dq)[0])

        # Retrieve nominal vertical reference
        if self.nominal_z_fn is not None:
            z_nom, vz_nom = self.nominal_z_fn(t)
        else:
            z_nom = state.get("target_pos", p_curr)[2]
            vz_nom = state.get("target_vel", np.zeros(6))[2]

        # Enforce strict corridor projection:
        # Highest priority requirement: remain below z_max and above z_min
        if z_nom > self.z_max:
            # Ceiling limit active: clamp reference to ceiling
            z_target = self.z_max
            vz_target = 0.0
            self.ceiling_active = True
            self.floor_active = False
        elif z_nom < self.z_min:
            # Floor limit active: clamp reference to floor
            z_target = self.z_min
            vz_target = 0.0
            self.ceiling_active = False
            self.floor_active = True
        else:
            # Inside admissible vertical corridor
            z_target = z_nom
            vz_target = vz_nom
            self.ceiling_active = False
            self.floor_active = False

        # Virtual spring-damper impedance law along Z
        e_z = z_target - z_curr
        ve_z = vz_target - v_z
        f_z = self.kp * e_z + self.kd * ve_z

        # Strong barrier unilateral push if penetrating boundary
        if z_curr > self.z_max:
            # Must strictly push downwards
            f_z = min(f_z, -abs(self.kp * (z_curr - self.z_max)))
        elif z_curr < self.z_min:
            # Must strictly push upwards
            f_z = max(f_z, abs(self.kp * (self.z_min - z_curr)))

        # Violation diagnostics
        self.current_violation = max(0.0, z_curr - self.z_max, self.z_min - z_curr)

        return J_z, np.array([f_z], dtype=np.float64)

    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes boundary violation error in meters (0.0 if inside safe bounds [z_min, z_max]).
        """
        p_curr = state["ee_pos"]
        z = p_curr[2]
        return max(0.0, float(z - self.z_max), float(self.z_min - z))
