"""
Experiment 2: Circular Surface Trajectory Tracking with Blocking Obstacle Box.

Demonstrates Multi-Priority QP Impedance Control with Obstacle Constraints:
  - Priority 0 (Primary): Exert downward contact force (F_z = -10 N) onto top surface of a box.
  - Priority 1 (Secondary): Follow a circular trajectory (R = 0.08m, T = 4.0s) in XY plane.
  - Priority 2 (Tertiary): Maintain nullspace joint posture.
  - Environmental Setup: An obstacle box is placed on top of the surface box, blocking part of the circular path.
  - Behavior: Evaluates compliant impedance deflection and joint torque bounds upon obstacle collision.
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
from matplotlib.patches import Rectangle

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask, CartesianForceTask, CircularTrajectoryGenerator

logger = logging.getLogger("Exp2_BlockedCircle")


def run_experiment_2(
    sim_time: float = 8.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    desired_force: float = -10.0,
    save_plot: bool = True,
    kp_cart: float = 600.0,
    kd_cart: float = 50.0,
    out_path: Optional[str] = None,
    record_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 2: Surface Circle Trajectory with Blocking Obstacle Box.

    Args:
        sim_time: Duration of experiment in seconds (default: 8.0s ~ 2 full circles).
        dt: Control timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display interactive 3D visualizer.
        device: Computing backend ('cpu' or 'gpu').
        desired_force: Downward surface normal force magnitude in Newtons (default: -10.0 N).
        save_plot: Whether to save diagnostic plot figure to disk.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data arrays.
    """
    logger.info("[Exp 2] Setting up environment with Surface Box and Obstacle Box...")
    # Base Surface Box: center (0.38, 0.0, 0.20), size (0.40, 0.50, 0.40) -> top surface z = 0.40m
    surface_box_cfg = {
        "pos": (0.38, 0.0, 0.20),
        "size": (0.40, 0.50, 0.40)
    }

    # Obstacle Box placed on top of surface box at x = 0.44 (intercepting circle path at radius R = 0.06m)
    obstacle_box_cfg = {
        "pos": (0.44, 0.0, 0.425),
        "size": (0.05, 0.08, 0.05)
    }

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        surface_box=surface_box_cfg,
        obstacle_box=obstacle_box_cfg,
        record_path=record_path,
        # The default camera frames the whole workspace, which leaves this experiment's 5 cm
        # obstacle and the ~13 cm deflection around it too small to read. Move in close on the
        # contact region instead.
        camera_pos=(0.95, -0.70, 0.78),
        camera_lookat=(0.40, 0.0, 0.43),
    )

    logger.info("[Exp 2] Initializing Hierarchical QP Impedance Controller...")
    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=400.0,
        kd_cart=40.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    # State extraction & trajectory setup
    state = sim.get_state()
    box_surface_z = 0.40
    circle_center = np.array([0.38, 0.0, box_surface_z])
    radius = 0.06
    period = 4.0


    traj_gen = CircularTrajectoryGenerator(
        center=circle_center,
        radius=radius,
        period=period,
        plane="xy"
    )

    logger.info(
        f"[Exp 2] Circle Center: {circle_center}, Radius: {radius}m, "
        f"Obstacle at: {obstacle_box_cfg['pos']}, Surface Force: {desired_force}N"
    )

    # Build 3-Level Task Hierarchy Stack
    task_stack = TaskStack()

    # Priority 0: Planar Circular Trajectory Tracking in XY (Highest Priority)
    # kp_cart is the commanded Cartesian stiffness whose rendering we are measuring: when the
    # obstacle blocks the path, the steady interaction force should equal kp_cart times the
    # deflection the task sees (commanded minus achieved).
    cart_task = CartesianPoseTask(
        name="circle_xy_p0",
        priority=0,
        kp=kp_cart,
        kd=kd_cart,
        mode="xy",
        trajectory_fn=traj_gen
    )
    task_stack.add_task(cart_task)

    # Priority 1: Downward Surface Normal Force Control (Secondary Priority)
    force_task = CartesianForceTask(
        name="surface_force_p1",
        priority=1,
        desired_force=desired_force,
        direction="z",
        target_pos=box_surface_z,
        kp=300.0,
        kd=20.0
    )
    task_stack.add_task(force_task)

    # Priority 2: Joint Posture Nullspace Task (Strictly projected into nullspace of Priority 0 & 1 via QP equality constraints)
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
    logger.info(f"[Exp 2] Starting simulation loop for {sim_time}s ({n_steps} steps)...")

    # Log arrays
    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_contact_force: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    # Strict priority has only ever been measured in free space. Logging it here answers whether
    # the guarantee survives an external force the controller never modelled.
    log_priority_residuals: List[np.ndarray] = []
    log_xy_error: List[float] = []
    log_x_error: List[float] = []
    log_y_error: List[float] = []
    log_z_error: List[float] = []
    log_force_error: List[float] = []
    log_posture_error: List[float] = []

    sim.start_recording()

    for step in range(n_steps):
        t_curr = step * dt

        state = sim.get_state()
        p_des, _, v_des = traj_gen(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # Compute QP multi-priority torques
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques & step simulation
        sim.apply_torques(torques)
        sim.step()
        sim.record_frame()

        # Telemetry
        p_curr = state["ee_pos"]
        q_curr = state["q"]
        f_meas = state.get("ee_force", np.zeros(3))

        x_err = p_des[0] - p_curr[0]
        y_err = p_des[1] - p_curr[1]
        z_err = box_surface_z - p_curr[2]
        xy_err = np.sqrt(x_err**2 + y_err**2)
        force_err = abs(desired_force - f_meas[2])
        posture_err = float(np.linalg.norm(q_curr - target_q_null))

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_ee_pos_des.append(p_des.copy())
        log_contact_force.append(f_meas.copy())
        log_torques.append(torques.copy())
        log_priority_residuals.append(
            np.asarray(getattr(controller, "priority_residuals", []), dtype=float)
        )
        log_xy_error.append(xy_err)
        log_x_error.append(x_err)
        log_y_error.append(y_err)
        log_z_error.append(z_err)
        log_force_error.append(force_err)
        log_posture_error.append(posture_err)

        if step % 200 == 0:
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | "
                f"XY Error: {xy_err*1000:5.2f}mm | Z-Pos: {p_curr[2]:6.4f}m | "
                f"Max Torque: {max_tau:5.2f}Nm"
            )

    logger.info("[Exp 2] Simulation completed successfully.")

    # Package logs
    sim.stop_recording()
    if record_path:
        logger.info(f"Recording written to {record_path}")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "contact_force": np.array(log_contact_force),
        "torques": np.array(log_torques),
        "kp_cart": float(kp_cart),
        "kd_cart": float(kd_cart),
        "n_violations": getattr(controller, "n_violations", -1),
        "n_solves": getattr(controller, "n_solves", -1),
        "xy_error": np.array(log_xy_error),
        "x_error": np.array(log_x_error),
        "y_error": np.array(log_y_error),
        "z_error": np.array(log_z_error),
        "force_error": np.array(log_force_error),
        "posture_error": np.array(log_posture_error),
    }

    res = log_priority_residuals
    if res and len({r.shape[0] for r in res}) == 1:
        logs["priority_residuals"] = np.array(res)

    if out_path:
        out = pathlib.Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **logs)
        logger.info(f"Logs written to {out}")

    if save_plot:
        plot_experiment_2(logs, surface_box_cfg, obstacle_box_cfg, circle_center, radius)

    return logs


def plot_experiment_2(
    logs: Dict[str, np.ndarray],
    surface_box_cfg: Dict,
    obstacle_box_cfg: Dict,
    center: np.ndarray,
    radius: float,
    output_path: str = "exp2_blocked_circle.png"
) -> None:
    """
    Generates and saves detailed multi-panel diagnostic plots for Experiment 2 including Task Errors.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    f_meas = logs["contact_force"]
    torques = logs["torques"]
    xy_err = logs["xy_error"]
    x_err = logs["x_error"]
    y_err = logs["y_error"]
    z_err = logs["z_error"]
    force_err = logs["force_error"]
    posture_err = logs["posture_error"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 2: Blocked Circular Trajectory & Compliant Obstacle Task Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: XY Surface Circle Trajectory with Obstacle Overlay
    ax1 = axs[0, 0]
    ax1.plot(p_des[:, 0], p_des[:, 1], "r--", label="Desired Circle Path", linewidth=2)
    ax1.plot(p_act[:, 0], p_act[:, 1], "b-", label="Actual Compliant Path", linewidth=1.5)
    ax1.plot(center[0], center[1], "k+", markersize=10, label="Circle Center")

    # Draw Obstacle Box boundary
    obs_pos = obstacle_box_cfg["pos"]
    obs_size = obstacle_box_cfg["size"]
    rect = Rectangle(
        (obs_pos[0] - obs_size[0] / 2.0, obs_pos[1] - obs_size[1] / 2.0),
        obs_size[0],
        obs_size[1],
        linewidth=2,
        edgecolor="orange",
        facecolor="orange",
        alpha=0.4,
        label="Obstacle Box"
    )
    ax1.add_patch(rect)

    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.set_title("1. Planar Trajectory Path (XY & Obstacle)")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")

    # Panel 2: Priority 0 Force Task & Z Position
    ax2 = axs[0, 1]
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(t, p_act[:, 2], "b-", label="EE Z Position [m]")
    l2 = ax2_twin.plot(t, f_meas[:, 2], "r-", label="Measured Z Force [N]")
    ax2.axhline(0.40, color="gray", linestyle=":", label="Surface Z = 0.40m")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Z Position [m]", color="b")
    ax2_twin.set_ylabel("Contact Force [N]", color="r")
    ax2.set_title("2. Level 0 Task: Z Position & Contact Force")
    ax2.grid(True)

    # Panel 3: Priority 1 Trajectory Error Norm
    ax3 = axs[1, 0]
    ax3.plot(t, xy_err * 1000.0, "g-", linewidth=1.5, label="||e_xy|| Norm")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("XY Tracking Error [mm]")
    ax3.set_title("3. Level 1 Task: Planar Trajectory Error Norm")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Priority 1 Component Task Errors (Ex, Ey, Ez)
    ax4 = axs[1, 1]
    ax4.plot(t, x_err * 1000.0, "r-", label="X Error (e_x) [mm]", alpha=0.8)
    ax4.plot(t, y_err * 1000.0, "b-", label="Y Error (e_y) [mm]", alpha=0.8)
    ax4.plot(t, z_err * 1000.0, "k:", label="Z Height Error (e_z) [mm]", alpha=0.8)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Error [mm]")
    ax4.set_title("4. Level 1 Task: Cartesian Component Errors")
    ax4.grid(True)
    ax4.legend()

    # Panel 5: Priority 2 Joint Posture Task Error
    ax5 = axs[2, 0]
    ax5.plot(t, posture_err, "m-", linewidth=1.5, label="||q - q_null||")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Posture Error [rad]")
    ax5.set_title("5. Level 2 Task: Joint Posture Error Norm")
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
    logger.info(f"[Exp 2] Saved diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 2: Blocked Circular Surface Trajectory & Obstacle Interaction")
    parser.add_argument("--time", type=float, default=8.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode without viewer GUI")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--force", type=float, default=-10.0, help="Desired surface normal force in Newtons")
    args = parser.parse_args()

    run_experiment_2(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        desired_force=args.force,
        save_plot=True
    )


if __name__ == "__main__":
    main()
