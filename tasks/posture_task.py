"""
Joint space posture control task implementation.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from tasks.base_task import BaseTask


class JointPostureTask(BaseTask):
    """
    Joint space posture stiffness task (Null-Space priority level task).

    Implements joint space spring-damper law:
        tau_p = Kp * (q_des - q) - Dp * dq
    """

    def __init__(
        self,
        name: str = "joint_posture",
        priority: int = 1,
        kp: float = 20.0,
        kd: float = 4.0,
        q_des: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize Joint Posture Task.

        Args:
            name: Task identifier string.
            priority: Priority level index (default: 1 for secondary task).
            kp: Joint proportional stiffness gain.
            kd: Joint derivative damping gain.
            q_des: Desired joint configuration array of shape (n_dofs,).
        """
        super().__init__(name=name, priority=priority)
        self.kp = kp
        self.kd = kd
        self.q_des = q_des

    def compute(self, state: Dict[str, np.ndarray], t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes joint space identity Jacobian J_posture and posture torques.

        Args:
            state: Robot state dictionary containing 'q', 'dq'.
            t: Current time in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (I_{n x n}, tau_p)
        """
        q = state["q"]
        dq = state["dq"]
        n_dofs = len(q)

        target_q = self.q_des if self.q_des is not None else state.get("q_null", q)

        # Joint space Jacobian is identity matrix
        J_posture = np.eye(n_dofs)

        # Virtual joint torque
        tau_posture = self.kp * (target_q - q) - self.kd * dq

        return J_posture, tau_posture

    def compute_error(self, state: Dict[str, np.ndarray]) -> float:
        """
        Computes joint configuration posture error norm.
        """
        q = state["q"]
        target_q = self.q_des if self.q_des is not None else state.get("q_null", q)
        return float(np.linalg.norm(target_q - q))
