"""
Experiment 4: Multi-Link Disturbance Rejection & Hierarchy Preservation.

Demonstrates task priority preservation when strong external forces (40 N) are applied
sequentially to lower-body structure (elbow link4) versus operational space end-effector (hand link).
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask, CircularTrajectoryGenerator

logger = logging.getLogger("Exp4_MultiLinkPush")


def run_experiment_4(
    sim_time: float = 8.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 4: Multi-Link Disturbance Rejection.

    Args:
        sim_time: Duration of experiment in seconds (default: 8.0s).
        dt: Simulation timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display interactive 3D visualizer.
        device: Computing backend ('cpu' or 'gpu').
        save_plot: Whether to save diagnostic plot figure to disk.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data arrays.
    """
    logger.info("[Exp 4] Setting up environment...")
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device
    )

    logger.info("[Exp 4] Initializing Hierarchical QP Impedance Controller...")
    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=500.0,
        kd_cart=45.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    initial_ee_pos = state["ee_pos"].copy()
    circle_center = initial_ee_pos + np.array([0.05, 0.0, -0.05])
    radius = 0.08
    period = 4.0

    traj_gen = CircularTrajectoryGenerator(
        center=circle_center,
        radius=radius,
        period=period,
        plane="xy"
    )

    # Build 2-Level Task Stack
    task_stack = TaskStack()

    cart_task = CartesianPoseTask(
        name="circle_primary_p0",
        priority=0,
        kp=500.0,
        kd=45.0,
        mode="3d",
        trajectory_fn=traj_gen
    )
    task_stack.add_task(cart_task)

    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(
        name="posture_secondary_p1",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    logger.info(f"[Exp 4] Starting simulation loop for {sim_time}s ({n_steps} steps)...")

    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_elbow_force: List[np.ndarray] = []
    log_ee_push_force: List[np.ndarray] = []
    log_cart_error: List[float] = []
    log_posture_error: List[float] = []

    # Target links for external pushes
    elbow_link = "link4"
    hand_link = "hand"

    for step in range(n_steps):
        t_curr = step * dt

        state = sim.get_state()
        p_des, _, v_des = traj_gen(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        f_elbow_applied = np.zeros(3, dtype=np.float64)
        f_ee_applied = np.zeros(3, dtype=np.float64)

        # Disturbance Schedule:
        # Phase 1 (t in 2.0s .. 3.0s): 40 N force in +Y on Elbow (panda_link4)
        if 2.0 <= t_curr <= 3.0:
            f_elbow_applied = np.array([0.0, 40.0, 0.0])
            sim.apply_external_disturbance(force=f_elbow_applied, link_name=elbow_link)

        # Phase 2 (t in 5.0s .. 6.0s): 40 N force in +X on End-Effector (hand)
        if 5.0 <= t_curr <= 6.0:
            f_ee_applied = np.array([40.0, 0.0, 0.0])
            sim.apply_external_disturbance(force=f_ee_applied, link_name=hand_link)

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

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_ee_pos_des.append(p_des.copy())
        log_torques.append(torques.copy())
        log_elbow_force.append(f_elbow_applied)
        log_ee_push_force.append(f_ee_applied)
        log_cart_error.append(cart_err)
        log_posture_error.append(posture_err)

        if step % 200 == 0:
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | "
                f"EE Error: {cart_err*1000:5.2f}mm | Max Torque: {max_tau:5.2f}Nm"
            )

    logger.info("[Exp 4] Simulation completed successfully.")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "elbow_force": np.array(log_elbow_force),
        "ee_push_force": np.array(log_ee_push_force),
        "cart_error": np.array(log_cart_error),
        "posture_error": np.array(log_posture_error),
    }

    if save_plot:
        plot_experiment_4(logs)

    return logs


def plot_experiment_4(logs: Dict[str, np.ndarray], output_path: str = "exp4_multilink_push.png") -> None:
    """
    Generates and saves detailed multi-panel diagnostic plots for Experiment 4.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    torques = logs["torques"]
    f_elbow = logs["elbow_force"]
    f_ee = logs["ee_push_force"]
    cart_err = logs["cart_error"]
    posture_err = logs["posture_error"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 4: Multi-Link Disturbance Rejection Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: XY Trajectory Path
    ax1 = axs[0, 0]
    ax1.plot(p_des[:, 0], p_des[:, 1], "r--", label="Desired Circle Path", linewidth=2)
    ax1.plot(p_act[:, 0], p_act[:, 1], "b-", label="Actual EE Path", linewidth=1.5)
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.set_title("1. Planar End-Effector Path (XY)")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")

    # Panel 2: Applied External Disturbance Forces
    ax2 = axs[0, 1]
    ax2.plot(t, f_elbow[:, 1], "g-", linewidth=2, label="Elbow Push (+Y Force) [N]")
    ax2.plot(t, f_ee[:, 0], "r-", linewidth=2, label="EE Push (+X Force) [N]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Force Magnitude [N]")
    ax2.set_title("2. Scheduled External Disturbance Forces")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: Level 0 End-Effector Tracking Error Norm
    ax3 = axs[1, 0]
    ax3.plot(t, cart_err * 1000.0, "b-", linewidth=1.5, label="||e_cart|| Norm")
    ax3.axvspan(2.0, 3.0, color="green", alpha=0.15, label="Elbow Push Active")
    ax3.axvspan(5.0, 6.0, color="red", alpha=0.15, label="EE Push Active")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Tracking Error [mm]")
    ax3.set_title("3. Level 0 Task: EE Tracking Error Norm")
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
    ax5.axvspan(2.0, 3.0, color="green", alpha=0.15)
    ax5.axvspan(5.0, 6.0, color="red", alpha=0.15)
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
    ax6.set_title("6. QP Commanded Joint Torques")
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    logger.info(f"[Exp 4] Saved diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 4: Multi-Link Disturbance Rejection")
    parser.add_argument("--time", type=float, default=8.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_4(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
