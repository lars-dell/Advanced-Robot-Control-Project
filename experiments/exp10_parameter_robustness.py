"""
Experiment 10: Model Parameter Uncertainty, Sensor Noise, and Unmodeled Payload Robustness.

Stress tests the Hierarchical QP Impedance Controller under non-ideal physical conditions:
    1. Nominal Condition (Baseline): Ideal model parameters B(q), h(q, dq), clean sensors.
    2. Inertia Parameter Mismatch: Controller uses scaled inertia matrix B_est = 1.35 * B (+35% error).
    3. Unmodeled End-Effector Payload: Extra 1.5 kg mass on end-effector without updating gravity compensation.
    4. Sensor Velocity Noise: Gaussian noise N(0, sigma^2) added to joint velocity measurements (sigma = 0.03 rad/s).
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.qp_impedance import QPImpedanceController
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, CircularTrajectoryGenerator, JointPostureTask, TaskStack

logger = logging.getLogger("Exp10_ParamRobustness")


def run_robustness_condition(
    condition_name: str,
    sim_time: float = 6.0,
    dt: float = 0.005,
    device: str = "cpu",
    sim: Any = None
) -> Dict[str, np.ndarray]:
    """
    Runs a single simulation run under a specified uncertainty/disturbance condition.

    Args:
        condition_name: 'nominal', 'inertia_mismatch', 'unmodeled_payload', or 'sensor_noise'.
        sim_time: Simulation time in seconds.
        dt: Control timestep in seconds.
        device: 'cpu' or 'gpu'.
        sim: Optional existing GenesisSim instance to reuse.

    Returns:
        Dict[str, np.ndarray]: Telemetry log data.
    """
    if sim is None:
        sim = GenesisSim(
            model_xml="panda_cylinder.xml",
            show_viewer=False,
            dt=dt,
            device=device
        )
    else:
        sim.reset()

    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=600.0,
        kd_cart=50.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    circle_center = p_init + np.array([0.05, 0.0, -0.02])
    radius = 0.06
    period = 3.0

    traj_gen = CircularTrajectoryGenerator(
        center=circle_center,
        radius=radius,
        period=period,
        plane="xy"
    )

    task_stack = TaskStack()
    cart_task = CartesianPoseTask(
        name="circle_p0",
        priority=0,
        kp=600.0,
        kd=50.0,
        mode="3d",
        trajectory_fn=traj_gen
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
    log_t: List[float] = []
    log_p_act: List[np.ndarray] = []
    log_p_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_err: List[float] = []

    np.random.seed(42)

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        p_des, _, v_des = traj_gen(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # Inject uncertainty according to selected condition
        if condition_name == "inertia_mismatch":
            # Scale inertia matrix by +35%
            state["B"] = 1.35 * state["B"]

        elif condition_name == "unmodeled_payload":
            # Apply continuous downward gravitational force corresponding to +1.5 kg payload (1.5 * 9.81 = 14.7 N)
            sim.apply_external_disturbance(force=np.array([0.0, 0.0, -14.715]), link_name="hand")

        elif condition_name == "sensor_noise":
            # Inject Gaussian noise into dq measurements: sigma = 0.03 rad/s
            noise_dq = np.random.normal(0.0, 0.03, size=state["dq"].shape)
            state["dq"] = state["dq"] + noise_dq

        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)
        sim.apply_torques(torques)
        sim.step()

        p_curr = state["ee_pos"]
        err = float(np.linalg.norm(p_des - p_curr))

        log_t.append(t_curr)
        log_p_act.append(p_curr.copy())
        log_p_des.append(p_des.copy())
        log_torques.append(torques.copy())
        log_err.append(err)

    return {
        "time": np.array(log_t),
        "p_act": np.array(log_p_act),
        "p_des": np.array(log_p_des),
        "torques": np.array(log_torques),
        "error": np.array(log_err),
        "rmse": float(np.sqrt(np.mean(np.array(log_err)**2))),
        "max_err": float(np.max(np.array(log_err))),
        "max_tau": float(np.max(np.abs(np.array(log_torques))))
    }


def run_experiment_10(
    sim_time: float = 6.0,
    dt: float = 0.005,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Executes all 4 robustness stress test conditions.
    """
    logger.info("=================================================================")
    logger.info("Running Experiment 10: Model Parameter Uncertainty & Robustness")
    logger.info("=================================================================")

    conditions = [
        ("nominal", "Nominal Model (Ideal Reference)"),
        ("inertia_mismatch", "Inertia Matrix Mismatch (+35% B)"),
        ("unmodeled_payload", "Unmodeled 1.5kg Payload (+14.7N Grav.)"),
        ("sensor_noise", "Sensor Velocity Noise (sigma = 0.03 rad/s)")
    ]

    results: Dict[str, Any] = {}
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device
    )

    for key, desc in conditions:
        logger.info(f"Evaluating: {desc}...")
        res = run_robustness_condition(
            condition_name=key,
            sim_time=sim_time,
            dt=dt,
            device=device,
            sim=sim
        )
        results[key] = res
        logger.info(f"[{key.upper()}] RMSE: {res['rmse']*1000:.2f}mm | Max Error: {res['max_err']*1000:.2f}mm | Max Torque: {res['max_tau']:.2f}Nm")

    if save_plot:
        plot_experiment_10(results)

    logger.info("Experiment 10 completed successfully.")
    return results


def plot_experiment_10(
    results: Dict[str, Any],
    output_path: str = "results/figures/exp10_parameter_robustness.png"
) -> None:
    """
    Generates multi-panel comparative diagnostic plots across uncertainty conditions.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Experiment 10: Model Uncertainty, Payload & Sensor Noise Robustness", fontsize=14, fontweight="bold")

    styles = {
        "nominal": {"color": "black", "ls": "-", "label": "Nominal Baseline"},
        "inertia_mismatch": {"color": "blue", "ls": "--", "label": "Inertia Mismatch (+35% B)"},
        "unmodeled_payload": {"color": "red", "ls": "-.", "label": "Unmodeled 1.5kg Payload"},
        "sensor_noise": {"color": "green", "ls": ":", "label": "Sensor Noise (sigma=0.03)"}
    }

    # Panel 1: XY Trajectory Comparison
    ax1 = axs[0, 0]
    p_des = results["nominal"]["p_des"]
    ax1.plot(p_des[:, 0], p_des[:, 1], "k--", linewidth=2, label="Desired Path")
    for key, cfg in styles.items():
        p_act = results[key]["p_act"]
        ax1.plot(p_act[:, 0], p_act[:, 1], color=cfg["color"], linestyle=cfg["ls"], alpha=0.85, label=cfg["label"])
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.set_title("1. Planar Trajectory Path (XY Plane)")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend(loc="lower right", fontsize=8)

    # Panel 2: Tracking Error Norm Over Time
    ax2 = axs[0, 1]
    for key, cfg in styles.items():
        t = results[key]["time"]
        err = results[key]["error"] * 1000.0
        ax2.plot(t, err, color=cfg["color"], linestyle=cfg["ls"], alpha=0.85, label=f"{cfg['label']} (RMSE: {results[key]['rmse']*1000:.1f}mm)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Tracking Error Norm [mm]")
    ax2.set_title("2. Tracking Error Norm Comparison")
    ax2.grid(True)
    ax2.legend(fontsize=8)

    # Panel 3: Z-Axis Position & Steady-State Payload Droop
    ax3 = axs[1, 0]
    for key, cfg in styles.items():
        t = results[key]["time"]
        z_act = results[key]["p_act"][:, 2]
        ax3.plot(t, z_act, color=cfg["color"], linestyle=cfg["ls"], alpha=0.85, label=cfg["label"])
    ax3.axhline(p_des[0, 2], color="gray", linestyle=":", label="Nominal Z Height")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Z Position [m]")
    ax3.set_title("3. Z-Axis Elevation & Compliance Droop")
    ax3.grid(True)
    ax3.legend(fontsize=8)

    # Panel 4: Maximum Commanded Joint Torque Across Conditions
    ax4 = axs[1, 1]
    for key, cfg in styles.items():
        t = results[key]["time"]
        max_tau_t = np.max(np.abs(results[key]["torques"]), axis=1)
        ax4.plot(t, max_tau_t, color=cfg["color"], linestyle=cfg["ls"], alpha=0.85, label=cfg["label"])
    ax4.axhline(87.0, color="red", linestyle="--", alpha=0.5, label="Joint Torque Limit (87 Nm)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Max Joint Torque [Nm]")
    ax4.set_title("4. Peak Joint Torque Envelope")
    ax4.grid(True)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved parameter robustness diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 10: Model Parameter Robustness")
    parser.add_argument("--time", type=float, default=6.0, help="Simulation time per condition in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_10(
        sim_time=args.time,
        dt=args.dt,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
