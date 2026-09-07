"""
Experiment 5: Surface Wiping under Tightened Joint Torque Bounds.

Demonstrates Multi-Priority QP Impedance Control under tightened joint torque bounds
(+/-20 Nm on joints 1-4):

WARNING: the tightened bounds do NOT actually engage. Measured max|tau| over this run is
15.3 Nm against the 20 Nm limit, so no inequality constraint becomes active and this
experiment does not demonstrate what its original title claimed. See
docs/EVALUATION_FRAMEWORK.md section 2.4. Use the `conflict` scenario for an active
constraint set (>=1 joint on a bound in 92.5% of steps).

  - Priority 0 (Primary): Exert downward contact force (F_z = -15 N) onto table surface.
  - Priority 1 (Secondary): Follow a smooth S-curve wiping trajectory in XY plane.
  - Priority 2 (Tertiary): Maintain joint space posture.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.qp_impedance import QPImpedanceController
from envs.genesis_sim import GenesisSim
from tasks import CartesianForceTask, CartesianPoseTask, JointPostureTask, TaskStack

logger = logging.getLogger("Exp5_TorqueWipe")


def generate_scurve_trajectory(t: float, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates smooth S-curve trajectory in XY plane.
    """
    omega = 2.0 * np.pi / 4.0
    dx = 0.08 * np.sin(omega * t)
    dy = 0.06 * np.sin(2.0 * omega * t)
    pos_des = center + np.array([dx, dy, 0.0])

    vx = 0.08 * omega * np.cos(omega * t)
    vy = 0.06 * 2.0 * omega * np.cos(2.0 * omega * t)
    vel_des = np.array([vx, vy, 0.0])

    return pos_des, vel_des


def run_experiment_5(
    sim_time: float = 8.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    desired_force: float = -15.0,
    save_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 5: Torque-Constrained Surface Wiping.

    Args:
        sim_time: Duration of experiment in seconds (default: 8.0s).
        dt: Simulation timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display interactive 3D visualizer.
        device: Computing backend ('cpu' or 'gpu').
        desired_force: Downward surface normal force magnitude in Newtons (default: -15.0 N).
        save_plot: Whether to save diagnostic plot figure to disk.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data arrays.
    """
    logger.info("[Exp 5] Setting up environment with Surface Box...")
    surface_box_cfg = {
        "pos": (0.38, 0.0, 0.20),
        "size": (0.40, 0.50, 0.40)
    }

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        surface_box=surface_box_cfg
    )

    # Intentionally tight joint torque bounds (±20 Nm) to test active QP constraints
    tight_tau_min = np.array([-20.0, -20.0, -20.0, -20.0, -12.0, -12.0, -12.0], dtype=np.float64)
    tight_tau_max = np.array([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 12.0], dtype=np.float64)

    logger.info("[Exp 5] Initializing QP Impedance Controller with Tight Bounds (±20 Nm)...")
    controller = QPImpedanceController(
        n_dofs=7,
        tau_min=tight_tau_min,
        tau_max=tight_tau_max,
        kp_cart=500.0,
        kd_cart=45.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    box_surface_z = 0.40
    wipe_center = np.array([0.38, 0.0, box_surface_z])

    # Build 3-Level Task Hierarchy Stack
    task_stack = TaskStack()

    # Priority 0: Downward Normal Force Control (Highest Priority)
    force_task = CartesianForceTask(
        name="surface_force_p0",
        priority=0,
        desired_force=desired_force,
        direction="z",
        target_pos=box_surface_z,
        kp=300.0,
        kd=25.0
    )
    task_stack.add_task(force_task)

    # Priority 1: Planar S-Curve Trajectory Tracking in XY (Secondary Priority)
    cart_task = CartesianPoseTask(
        name="scurve_xy_p1",
        priority=1,
        kp=500.0,
        kd=45.0,
        mode="xy"
    )
    task_stack.add_task(cart_task)

    # Priority 2: Joint Posture Nullspace Task
    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(
        name="posture_nullspace_p2",
        priority=2,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    logger.info(f"[Exp 5] Starting simulation loop for {sim_time}s ({n_steps} steps)...")

    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_contact_force: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_xy_error: List[float] = []
    log_force_error: List[float] = []
    log_posture_error: List[float] = []

    for step in range(n_steps):
        t_curr = step * dt

        state = sim.get_state()
        p_des, v_des = generate_scurve_trajectory(t_curr, wipe_center)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # Compute QP multi-priority torques
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques & step simulation
        sim.apply_torques(torques)
        sim.step()

        # Telemetry
        p_curr = state["ee_pos"]
        q_curr = state["q"]
        f_meas = state.get("ee_force", np.zeros(3))

        xy_err = float(np.linalg.norm((p_des - p_curr)[:2]))
        force_err = float(abs(desired_force - f_meas[2]))
        posture_err = float(np.linalg.norm(q_curr - target_q_null))

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_ee_pos_des.append(p_des.copy())
        log_contact_force.append(f_meas.copy())
        log_torques.append(torques.copy())
        log_xy_error.append(xy_err)
        log_force_error.append(force_err)
        log_posture_error.append(posture_err)

        if step % 200 == 0:
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | "
                f"XY Error: {xy_err*1000:5.2f}mm | Z Force: {f_meas[2]:5.1f}N | "
                f"Max Torque: {max_tau:5.2f}Nm"
            )

    logger.info("[Exp 5] Simulation completed successfully.")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "contact_force": np.array(log_contact_force),
        "torques": np.array(log_torques),
        "xy_error": np.array(log_xy_error),
        "force_error": np.array(log_force_error),
        "posture_error": np.array(log_posture_error),
        "tight_tau_max": tight_tau_max,
        "tight_tau_min": tight_tau_min,
    }

    if save_plot:
        plot_experiment_5(logs, wipe_center)

    return logs


def plot_experiment_5(logs: Dict[str, np.ndarray], wipe_center: np.ndarray, output_path: str = "results/figures/exp5_torque_constrained_wipe.png") -> None:
    """
    Generates and saves detailed multi-panel diagnostic plots for Experiment 5.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    f_meas = logs["contact_force"]
    torques = logs["torques"]
    xy_err = logs["xy_error"]
    force_err = logs["force_error"]
    posture_err = logs["posture_error"]
    tight_tau_max = logs["tight_tau_max"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 5: Torque-Constrained Surface Wiping Task Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: XY S-Curve Wiping Path
    ax1 = axs[0, 0]
    ax1.plot(p_des[:, 0], p_des[:, 1], "r--", label="Desired S-Curve Path", linewidth=2)
    ax1.plot(p_act[:, 0], p_act[:, 1], "b-", label="Actual EE Path", linewidth=1.5)
    ax1.plot(wipe_center[0], wipe_center[1], "k+", markersize=10, label="Center")
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.set_title("1. Planar S-Curve Path (XY)")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")

    # Panel 2: Priority 0 Normal Force Tracking
    ax2 = axs[0, 1]
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(t, p_act[:, 2], "b-", label="EE Z Position [m]")
    l2 = ax2_twin.plot(t, f_meas[:, 2], "r-", label="Measured Z Force [N]")
    ax2.axhline(0.40, color="gray", linestyle=":", label="Surface Z = 0.40m")
    ax2_twin.axhline(-15.0, color="red", linestyle="--", label="Target Force (-15N)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Z Position [m]", color="b")
    ax2_twin.set_ylabel("Contact Force [N]", color="r")
    ax2.set_title("2. Level 0 Task: Z Surface Force Tracking")
    ax2.grid(True)

    # Panel 3: Priority 1 Trajectory Tracking Error
    ax3 = axs[1, 0]
    ax3.plot(t, xy_err * 1000.0, "g-", linewidth=1.5, label="||e_xy|| Norm")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Tracking Error [mm]")
    ax3.set_title("3. Level 1 Task: Planar Trajectory Error Norm")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Force Tracking Error
    ax4 = axs[1, 1]
    ax4.plot(t, force_err, "r-", linewidth=1.5, label="|F_z,des - F_z,meas|")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Force Error [N]")
    ax4.set_title("4. Level 0 Task: Normal Force Error")
    ax4.grid(True)
    ax4.legend()

    # Panel 5: Joint Posture Error
    ax5 = axs[2, 0]
    ax5.plot(t, posture_err, "m-", linewidth=1.5, label="||q - q_null||")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Posture Error [rad]")
    ax5.set_title("5. Level 2 Task: Joint Posture Error Norm")
    ax5.grid(True)
    ax5.legend()

    # Panel 6: Commanded Joint Torques with Tight Bounds (±20 Nm)
    ax6 = axs[2, 1]
    for i in range(torques.shape[1]):
        ax6.plot(t, torques[:, i], label=f"Joint {i+1}")
    ax6.axhline(20.0, color="red", linestyle="--", linewidth=2, label="Tight Saturation Bound (±20 Nm)")
    ax6.axhline(-20.0, color="red", linestyle="--", linewidth=2)
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Torque [Nm]")
    ax6.set_title("6. QP Commanded Joint Torques (Tight Bounds Saturated)")
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"[Exp 5] Saved diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 5: Torque-Constrained Surface Wiping")
    parser.add_argument("--time", type=float, default=8.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--force", type=float, default=-15.0, help="Desired surface normal force in Newtons")
    args = parser.parse_args()

    run_experiment_5(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        desired_force=args.force,
        save_plot=True
    )


if __name__ == "__main__":
    main()
