"""
Experiment 6: Kinematic Singularity & Workspace Boundary Tracking.

Demonstrates controller stability and joint torque boundedness when tracking target setpoints
near or beyond the manipulator's workspace boundary (R >= 0.85 m for Franka Panda).
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from controllers import make_controller
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger("Exp6_Singularity")


def run_experiment_6(
    sim_time: float = 6.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    save_plot: bool = True,
    target: Optional[np.ndarray] = None,
    controller_name: str = "hierarchical_qp",
    out_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 6: Workspace Boundary & Kinematic Singularity Tracking.

    Args:
        sim_time: Duration of experiment in seconds (default: 6.0s).
        dt: Simulation timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display interactive 3D visualizer.
        device: Computing backend ('cpu' or 'gpu').
        save_plot: Whether to save diagnostic plot figure to disk.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data arrays.
    """
    logger.info("[Exp 6] Setting up environment...")
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device
    )

    logger.info("[Exp 6] Initializing QP Impedance Controller with Regularization (eps=1e-4)...")
    controller = make_controller(
        controller_name,
        n_dofs=7,
        kp_cart=400.0,
        kd_cart=40.0,
        kp_null=20.0,
        kd_null=4.0,
        reg_eps=1e-4,
        solver_name="daqp"
    )

    state = sim.get_state()
    initial_ee_pos = state["ee_pos"].copy()

    # Commanded far enough along +x that the arm must fully extend to chase it, which is what
    # drives the position Jacobian toward rank deficiency.
    #
    # The previous target [0.82, 0, 0.40] was justified as being "near max workspace reach
    # (~0.85 m)". That figure is wrong for this model: 855 mm is the datasheet *horizontal flange*
    # reach, whereas the relevant quantity is the 3-D radius to the tool tip, measured at >= 1.267 m
    # (scripts/probe_genesis.py). At 0.91 m radius the old target left the arm comfortably inside
    # its workspace and sigma_min fell only from 0.284 to 0.204 -- no singularity was approached and
    # the experiment did not demonstrate what its name claims.
    singularity_target = (np.asarray(target, dtype=np.float64) if target is not None
                          else np.array([1.30, 0.0, 0.40]))

    # Build 2-Level Task Stack
    task_stack = TaskStack()

    cart_task = CartesianPoseTask(
        name="singularity_reach_p0",
        priority=0,
        kp=400.0,
        kd=40.0,
        mode="3d"
    )
    task_stack.add_task(cart_task)

    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(
        name="posture_p1",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    logger.info(f"[Exp 6] Starting simulation loop for {sim_time}s ({n_steps} steps)...")

    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_cart_error: List[float] = []
    log_posture_error: List[float] = []
    log_cond_num: List[float] = []
    log_min_singular: List[float] = []
    log_reach_radius: List[float] = []

    for step in range(n_steps):
        t_curr = step * dt

        # Smooth interpolation towards workspace boundary target
        alpha = min(1.0, t_curr / 3.0)
        p_des = (1.0 - alpha) * initial_ee_pos + alpha * singularity_target

        state = sim.get_state()
        state["target_pos"] = p_des

        # Jacobian singularity diagnostics
        J_3d = state["J"][:3, :]
        U, S, Vt = np.linalg.svd(J_3d)
        sigma_min = float(S[-1])
        cond_num = float(S[0] / (S[-1] + 1e-9))

        # Compute QP multi-priority torques
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques & step simulation
        sim.apply_torques(torques)
        sim.step()

        # Telemetry
        p_curr = state["ee_pos"]
        q_curr = state["q"]

        cart_err = float(np.linalg.norm(p_des - p_curr))
        posture_err = float(np.linalg.norm(q_curr - target_q_null))
        reach_radius = float(np.linalg.norm(p_curr))

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_ee_pos_des.append(p_des.copy())
        log_torques.append(torques.copy())
        log_cart_error.append(cart_err)
        log_posture_error.append(posture_err)
        log_cond_num.append(cond_num)
        log_min_singular.append(sigma_min)
        log_reach_radius.append(reach_radius)

        if step % 200 == 0:
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | "
                f"Reach Radius: {reach_radius:5.3f}m | Cond Num: {cond_num:6.1f} | "
                f"Min Singular: {sigma_min:6.4f} | Max Torque: {max_tau:5.2f}Nm"
            )

    logger.info("[Exp 6] Simulation completed successfully.")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "cart_error": np.array(log_cart_error),
        "posture_error": np.array(log_posture_error),
        "cond_num": np.array(log_cond_num),
        "min_singular": np.array(log_min_singular),
        "reach_radius": np.array(log_reach_radius),
        "target_pos": singularity_target,
        "tau_min": sim.tau_min,
        "tau_max": sim.tau_max,
        "n_violations": getattr(controller, "n_violations", -1),
        "n_solves": getattr(controller, "n_solves", -1),
        "controller": np.array(str(controller_name)),
    }

    if out_path:
        out = pathlib.Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **logs)
        logger.info(f"Logs written to {out}")

    if save_plot:
        plot_experiment_6(logs)

    return logs


def plot_experiment_6(logs: Dict[str, np.ndarray], output_path: str = "exp6_singularity_tracking.png") -> None:
    """
    Generates and saves detailed multi-panel diagnostic plots for Experiment 6.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    torques = logs["torques"]
    cart_err = logs["cart_error"]
    posture_err = logs["posture_error"]
    cond_num = logs["cond_num"]
    sigma_min = logs["min_singular"]
    reach_radius = logs["reach_radius"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 6: Kinematic Singularity & Workspace Boundary Task Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: EE Position Trajectories & Reach Radius
    ax1 = axs[0, 0]
    ax1.plot(t, p_act[:, 0], "r-", label="Actual X Position [m]")
    ax1.plot(t, p_des[:, 0], "r--", label="Target X Position [m]")
    ax1.plot(t, reach_radius, "k-", linewidth=1.5, label="Reach Radius ||p_ee|| [m]")
    ax1.axhline(0.855, color="gray", linestyle=":", label="Franka Max Reach Radius (0.855m)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position / Reach Radius [m]")
    ax1.set_title("1. Workspace Reaching & Radius Boundary")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: Jacobian Condition Number & Smallest Singular Value
    ax2 = axs[0, 1]
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(t, cond_num, "m-", label="Jacobian Condition Number κ(J)")
    l2 = ax2_twin.plot(t, sigma_min, "c-", label="Min Singular Value σ_min(J)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Condition Number κ(J)", color="m")
    ax2_twin.set_ylabel("Min Singular Value σ_min", color="c")
    ax2.set_title("2. Kinematic Singularity Metrics")
    ax2.grid(True)

    # Panel 3: Level 0 Cartesian Tracking Error Norm
    ax3 = axs[1, 0]
    ax3.plot(t, cart_err * 1000.0, "b-", linewidth=1.5, label="||e_cart|| Norm")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Tracking Error [mm]")
    ax3.set_title("3. Level 0 Task: Cartesian Tracking Error Norm")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Cartesian Component Error Decomposition
    ax4 = axs[1, 1]
    ax4.plot(t, (p_des[:, 0] - p_act[:, 0]) * 1000.0, "r-", label="e_x [mm]", alpha=0.8)
    ax4.plot(t, (p_des[:, 1] - p_act[:, 1]) * 1000.0, "g-", label="e_y [mm]", alpha=0.8)
    ax4.plot(t, (p_des[:, 2] - p_act[:, 2]) * 1000.0, "b-", label="e_z [mm]", alpha=0.8)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Error Component [mm]")
    ax4.set_title("4. Level 0 Task: XYZ Error Component Breakdown")
    ax4.grid(True)
    ax4.legend()

    # Panel 5: Level 1 Joint Posture Error Norm
    ax5 = axs[2, 0]
    ax5.plot(t, posture_err, "m-", linewidth=1.5, label="||q - q_null||")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Posture Error [rad]")
    ax5.set_title("5. Level 1 Task: Joint Posture Deflection")
    ax5.grid(True)
    ax5.legend()

    # Panel 6: Commanded Joint Torques
    ax6 = axs[2, 1]
    for i in range(torques.shape[1]):
        ax6.plot(t, torques[:, i], label=f"Joint {i+1}")
    ax6.axhline(87.0, color="red", linestyle="--", alpha=0.5, label="Torque Limit")
    ax6.axhline(-87.0, color="red", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Torque [Nm]")
    ax6.set_title("6. QP Commanded Joint Torques (Bounded Near Singularity)")
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    logger.info(f"[Exp 6] Saved diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 6: Kinematic Singularity Tracking")
    parser.add_argument("--time", type=float, default=6.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_6(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
