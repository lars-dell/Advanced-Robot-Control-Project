"""
Experiment 3: Reactive APF Obstacle Avoidance via Hierarchical QP.

Demonstrates Multi-Priority QP Impedance Control with Repulsive Potential Field:
  - Priority 0 (Primary): APF Repulsive Task when entering obstacle safety region.
  - Priority 1 (Secondary): 3D Cartesian Goal Reaching Task.
  - Priority 2 (Tertiary): Nullspace Joint Posture Task.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask, APFRepulsiveTask

logger = logging.getLogger("Exp3_APFAvoidance")


def run_experiment_3(
    sim_time: float = 6.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 3: APF Obstacle Avoidance via Hierarchical QP.

    Args:
        sim_time: Experiment duration in seconds (default: 6.0s).
        dt: Simulation timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display interactive 3D visualizer.
        device: Computing backend ('cpu' or 'gpu').
        save_plot: Whether to save diagnostic plot figure to disk.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data arrays.
    """
    logger.info("[Exp 3] Setting up environment with Obstacle & Goal Spheres...")
    obstacle_sphere_cfg = {
        "pos": (0.42, 0.00, 0.45),
        "radius": 0.08
    }
    goal_sphere_cfg = {
        "pos": (0.55, 0.20, 0.45),
        "radius": 0.03
    }

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        obstacle_sphere=obstacle_sphere_cfg,
        goal_sphere=goal_sphere_cfg
    )

    logger.info("[Exp 3] Initializing Hierarchical QP Impedance Controller...")
    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=400.0,
        kd_cart=40.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    goal_pos = np.array(goal_sphere_cfg["pos"])
    obs_pos = np.array(obstacle_sphere_cfg["pos"])
    obs_radius = obstacle_sphere_cfg["radius"]

    # Build Task Hierarchy Stack
    task_stack = TaskStack()

    # Priority 0: APF Repulsive Task (Active when entering safety margin)
    apf_task = APFRepulsiveTask(
        name="apf_repulsion_p0",
        priority=0,
        obstacle_pos=obs_pos,
        obstacle_radius=obs_radius,
        margin=0.08,
        k_rep=10.0
    )
    task_stack.add_task(apf_task)

    # Priority 1: 3D Cartesian Goal Reaching Task
    goal_task = CartesianPoseTask(
        name="goal_reach_p1",
        priority=1,
        kp=450.0,
        kd=45.0,
        mode="3d"
    )
    task_stack.add_task(goal_task)

    # Priority 2: Joint Posture Null-Space Task
    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(
        name="posture_p2",
        priority=2,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    logger.info(f"[Exp 3] Starting simulation loop for {sim_time}s ({n_steps} steps)...")

    # Data logging arrays
    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_clearance: List[float] = []
    log_goal_dist: List[float] = []
    log_posture_err: List[float] = []

    for step in range(n_steps):
        t_curr = step * dt

        state = sim.get_state()
        state["target_pos"] = goal_pos
        state["q_null"] = target_q_null

        # Compute QP multi-priority torques
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques & step simulation
        sim.apply_torques(torques)
        sim.step()

        # Telemetry
        p_curr = state["ee_pos"]
        q_curr = state["q"]

        clearance = float(np.linalg.norm(p_curr - obs_pos) - obs_radius)
        goal_dist = float(np.linalg.norm(p_curr - goal_pos))
        posture_err = float(np.linalg.norm(q_curr - target_q_null))

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_torques.append(torques.copy())
        log_clearance.append(clearance)
        log_goal_dist.append(goal_dist)
        log_posture_err.append(posture_err)

        if step % 200 == 0:
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | "
                f"Goal Dist: {goal_dist*1000:5.1f}mm | Obs Clearance: {clearance*1000:5.1f}mm | "
                f"Max Torque: {max_tau:5.2f}Nm"
            )

    logger.info("[Exp 3] Simulation completed successfully.")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "torques": np.array(log_torques),
        "clearance": np.array(log_clearance),
        "goal_dist": np.array(log_goal_dist),
        "posture_error": np.array(log_posture_err),
        "obs_pos": obs_pos,
        "obs_radius": obs_radius,
        "goal_pos": goal_pos,
    }

    if save_plot:
        plot_experiment_3(logs)

    return logs


def plot_experiment_3(logs: Dict[str, np.ndarray], output_path: str = "exp3_apf_avoidance.png") -> None:
    """
    Generates and saves detailed multi-panel diagnostic plots for Experiment 3.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    torques = logs["torques"]
    clearance = logs["clearance"]
    goal_dist = logs["goal_dist"]
    posture_err = logs["posture_error"]
    obs_pos = logs["obs_pos"]
    obs_radius = logs["obs_radius"]
    goal_pos = logs["goal_pos"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 3: Reactive APF Obstacle Avoidance Task Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: Planar Trajectory Path (XY)
    ax1 = axs[0, 0]
    ax1.plot(p_act[:, 0], p_act[:, 1], "b-", label="EE Trajectory Path", linewidth=2)
    ax1.plot(p_act[0, 0], p_act[0, 1], "go", markersize=8, label="Start Position")
    ax1.plot(goal_pos[0], goal_pos[1], "r*", markersize=12, label="Goal Position")

    # Draw Obstacle Circle
    circ = Circle((obs_pos[0], obs_pos[1]), obs_radius, color="orange", alpha=0.5, label="Obstacle Boundary")
    circ_margin = Circle((obs_pos[0], obs_pos[1]), obs_radius + 0.08, color="orange", fill=False, linestyle="--", label="APF Margin (d0)")
    ax1.add_patch(circ)
    ax1.add_patch(circ_margin)
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.set_title("1. Planar Path & APF Obstacle Boundary (XY)")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")

    # Panel 2: 3D Trajectory Components
    ax2 = axs[0, 1]
    ax2.plot(t, p_act[:, 0], "r-", label="X Position [m]")
    ax2.plot(t, p_act[:, 1], "g-", label="Y Position [m]")
    ax2.plot(t, p_act[:, 2], "b-", label="Z Position [m]")
    ax2.axhline(goal_pos[0], color="r", linestyle="--", alpha=0.5)
    ax2.axhline(goal_pos[1], color="g", linestyle="--", alpha=0.5)
    ax2.axhline(goal_pos[2], color="b", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("2. Cartesian Position Trajectories (XYZ)")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: Obstacle Clearance Distance
    ax3 = axs[1, 0]
    ax3.plot(t, clearance * 1000.0, "orange", linewidth=2, label="Clearance (d - r_obs)")
    ax3.axhline(0.0, color="red", linestyle="--", label="Collision Boundary")
    ax3.axhline(80.0, color="gray", linestyle=":", label="APF Margin (80mm)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Clearance [mm]")
    ax3.set_title("3. Level 0 Task: Obstacle Clearance Distance")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Distance to Goal
    ax4 = axs[1, 1]
    ax4.plot(t, goal_dist * 1000.0, "r-", linewidth=2, label="Goal Distance Norm")
    ax4.axhline(30.0, color="green", linestyle=":", label="Success Threshold (30mm)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Distance [mm]")
    ax4.set_title("4. Level 1 Task: Distance to Goal")
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
    logger.info(f"[Exp 3] Saved diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 3: Reactive APF Obstacle Avoidance")
    parser.add_argument("--time", type=float, default=6.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_3(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
