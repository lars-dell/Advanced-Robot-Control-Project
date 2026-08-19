"""
Mathematical and kinematic utilities for multi-priority robot control.

Provides numerical tools for dynamically consistent pseudo-inverses, null-space projection
matrices, and Cartesian pose error calculations.
"""

from typing import Tuple, Optional
import numpy as np


def dynamically_consistent_pinv(
    J: np.ndarray,
    B: np.ndarray,
    reg: float = 1e-4
) -> np.ndarray:
    """
    Computes the dynamically consistent pseudo-inverse of a task Jacobian J.

    Formula:
        J_bar = B^(-1) * J^T * (J * B^(-1) * J^T + reg * I)^(-1)

    Args:
        J: Task Jacobian matrix of shape (m, n) where m is task dimension and n is joint DOFs.
        B: Joint space mass/inertia matrix of shape (n, n).
        reg: Regularization / damping factor for numerical stability near singularities.

    Returns:
        Dynamically consistent pseudo-inverse J_bar of shape (n, m).
    """
    m, n = J.shape
    B_inv = np.linalg.inv(B)
    J_BT = B_inv @ J.T  # (n, m)
    lambda_sq_I = reg * np.eye(m)
    
    # Lambda = J * B^(-1) * J^T
    Lambda_inv = np.linalg.inv(J @ J_BT + lambda_sq_I)
    
    J_bar = J_BT @ Lambda_inv  # (n, m)
    return J_bar


def null_space_projection(
    J: np.ndarray,
    pinv_J: Optional[np.ndarray] = None,
    dynamically_consistent: bool = False,
    B: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes the null-space projection matrix N for a task Jacobian J.

    Formula:
        Kinematic null-space: N = I - J^dagger * J
        Dynamically consistent null-space: N = I - J^T * (J_bar)^T

    Args:
        J: Task Jacobian matrix of shape (m, n).
        pinv_J: Precomputed pseudo-inverse of J of shape (n, m). If None, standard SVD/damped pinv is used.
        dynamically_consistent: If True, computes the dynamically consistent null-space projector using B.
        B: Joint space mass matrix (n, n), required if dynamically_consistent is True.

    Returns:
        Null-space projection matrix N of shape (n, n).
    """
    m, n = J.shape
    I_n = np.eye(n)

    if dynamically_consistent:
        if B is None:
            raise ValueError("Mass matrix B must be provided when dynamically_consistent is True.")
        J_bar = dynamically_consistent_pinv(J, B)
        # Dynamically consistent null-space projector transpose identity: N = I - J^T * (J_bar)^T
        N = I_n - J.T @ J_bar.T
    else:
        if pinv_J is None:
            pinv_J = np.linalg.pinv(J, rcond=1e-4)
        N = I_n - pinv_J @ J

    return N


def compute_orientation_error(
    R_curr: np.ndarray,
    R_des: np.ndarray
) -> np.ndarray:
    """
    Computes 3D orientation error between current and desired rotation matrices (target minus actual).

    Formula:
        e_rot = 0.5 * vee(R_des * R_curr^T - R_curr * R_des^T)

    Args:
        R_curr: Current 3x3 rotation matrix.
        R_des: Desired 3x3 rotation matrix.

    Returns:
        3D orientation error vector (target - actual).
    """
    R_err = R_des @ R_curr.T
    # Skew-symmetric part extraction (vee operator)
    e_rot = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    return e_rot


def compute_pose_error(
    p_curr: np.ndarray,
    R_curr: np.ndarray,
    p_des: np.ndarray,
    R_des: np.ndarray
) -> np.ndarray:
    """
    Computes 6D pose error vector (3D translational + 3D rotational error, target minus actual).

    Args:
        p_curr: Current position (3,).
        R_curr: Current orientation matrix (3, 3).
        p_des: Desired position (3,).
        R_des: Desired orientation matrix (3, 3).

    Returns:
        6D error vector e = [e_pos; e_rot] of shape (6,).
    """
    e_pos = p_des - p_curr
    e_rot = compute_orientation_error(R_curr, R_des)
    return np.concatenate([e_pos, e_rot])
