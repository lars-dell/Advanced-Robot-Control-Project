"""
Experiment: Single-Objective Cartesian Reaching & Disturbance Recovery Scenario.

Evaluates baseline 3-D Cartesian impedance tracking for a reachable point
combined with a lower-priority joint-space posture task:
    Priority 0: Cartesian 3D Position Task (kp=400, kd=40)
    Priority 1: Joint Posture Null-Space Task (kp=20, kd=4)
"""

import argparse
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers import make_controller, BaseController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger("ExpReach")


def run_reach(
    sim_time: float = 5.0,
    dt: float = 0.005,
    show_viewer: bool = False,
    device: str = "cpu",
    out_path: str = "results/reach.npz",
    show_markers: bool = True,
    record_path: Optional[str] = None,
    save_plot: bool = True,
    solver: str = "daqp",
    controller: Union[str, BaseController] = "hierarchical_qp"
) -> Dict[str, np.ndarray]:
    """
    Executes the single-objective Cartesian reaching scenario.

    Args:
        sim_time: Simulation duration in seconds (default: 5.0s).
        dt: Control timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to launch interactive 3D viewer.
        device: 'cpu' or 'gpu'.
        out_path: Path to save telemetry .npz file.
        show_markers: Whether to render debug overlays.
        record_path: Optional path to save recording (e.g. docs/media/reach.gif).
        save_plot: Whether to generate diagnostic figures.
        solver: QP solver backend engine ("daqp", "osqp", "qpoases").
        controller: Controller name or BaseController instance (default: 'hierarchical_qp').

    Returns:
        Dict[str, np.ndarray]: Telemetry logs.
    """
    logger.info("=================================================================")
    logger.info(f"Running Scenario: Single-Objective Cartesian Reaching (Reach) [Solver: {solver}]")
    logger.info("=================================================================")

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        show_markers=show_markers,
        record_path=record_path
    )

    if isinstance(controller, str):
        controller = make_controller(
            controller,
            n_dofs=7,
            kp_cart=400.0,
            kd_cart=40.0,
            kp_null=20.0,
            kd_null=4.0,
            solver_name=solver,
            reg_eps=1e-4
        )

    initial_state = sim.get_state()
    initial_ee_pos = initial_state["ee_pos"].copy()
    target_pos = initial_ee_pos + np.array([0.15, 0.10, -0.05])
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def reference(t: float):
        return target_pos, None, np.zeros(3)

    stack = TaskStack()
    stack.add_task(CartesianPoseTask(
        name="cartesian_primary",
        priority=0,
        kp=400.0,
        kd=40.0,
        is_6d=False,
        trajectory_fn=reference
    ))
    stack.add_task(JointPostureTask(
        name="posture",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=q_home
    ))

    sim.set_goal(target_pos)
    sim.start_recording()

    n_steps = int(sim_time / dt)
    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_errors: List[float] = []
    log_task_errors: List[Dict[str, float]] = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["t"] = t_curr
        state["target_pos"] = target_pos

        torques = controller.compute_torques(state=state, target=stack, t=t_curr)
        sim.apply_torques(torques)

        # External disturbance pulse for 0.2 s
        f_dist = np.array([10.0, 0.0, 0.0]) if 2.0 <= t_curr <= 2.2 else np.zeros(3)
        sim.set_external_force(f_dist)

        sim.step()

        sim.set_goal(target_pos)
        sim.update_viz(tip_pos=state["ee_pos"], goal_pos=target_pos)
        sim.record_frame()

        err = float(np.linalg.norm(target_pos - state["ee_pos"]))
        log_time.append(t_curr)
        log_ee_pos.append(state["ee_pos"].copy())
        log_ee_pos_des.append(target_pos.copy())
        log_torques.append(torques.copy())
        log_errors.append(err)
        log_task_errors.append(stack.get_task_errors(state))

        if step % 100 == 0:
            errs = " ".join(f"{k}={v:.3f}" for k, v in log_task_errors[-1].items())
            logger.info(f"Step {step:4d}/{n_steps} | t={t_curr:5.2f}s | {errs} | "
                        f"max|tau|={np.max(np.abs(torques)):5.2f} Nm")

    sim.stop_recording()
    if record_path:
        logger.info(f"Recording written to {record_path}")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "errors": np.array(log_errors),
        "tau_min": sim.tau_min,
        "tau_max": sim.tau_max,
        **{f"err_{name}": np.array([e[name] for e in log_task_errors])
           for name in (log_task_errors[0] if log_task_errors else {})},
        "task_names": np.array([t.name for t in stack.tasks]),
        "n_violations": controller.n_violations,
        "n_solves": controller.n_solves,
        "max_violation": controller.max_violation,
    }

    if out_path:
        out = pathlib.Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **logs)
        logger.info(f"Logs written to {out}")

    if save_plot:
        plot_reach(logs, target_pos)

    return logs


def plot_reach(logs: Dict[str, np.ndarray], target_pos: np.ndarray, output_path: str = "exp_reach.png") -> None:
    """Generates a 4-panel diagnostic plot for the reaching scenario."""
    t = logs["time"]
    p_act = logs["ee_pos"]
    errors = logs["errors"]
    torques = logs["torques"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scenario: Single-Objective Cartesian Reaching & Disturbance Response", fontsize=14, fontweight="bold")

    # 1. Tracking Error Norm
    axs[0, 0].plot(t, errors * 1000.0, "b-", linewidth=2.0, label="Tracking Error (mm)")
    axs[0, 0].axvspan(2.0, 2.2, color="orange", alpha=0.3, label="10 N Disturbance Pulse")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Error [mm]")
    axs[0, 0].set_title("1. End-Effector Tracking Error")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Cartesian Coordinates
    axs[0, 1].plot(t, p_act[:, 0], "r-", label="x actual")
    axs[0, 1].plot(t, p_act[:, 1], "g-", label="y actual")
    axs[0, 1].plot(t, p_act[:, 2], "b-", label="z actual")
    axs[0, 1].axhline(target_pos[0], color="r", linestyle="--", alpha=0.6, label="x target")
    axs[0, 1].axhline(target_pos[1], color="g", linestyle="--", alpha=0.6, label="y target")
    axs[0, 1].axhline(target_pos[2], color="b", linestyle="--", alpha=0.6, label="z target")
    axs[0, 1].axvspan(2.0, 2.2, color="orange", alpha=0.3)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Position [m]")
    axs[0, 1].set_title("2. Cartesian Coordinates")
    axs[0, 1].grid(True)
    axs[0, 1].legend(ncol=2, fontsize=8)

    # 3. Commanded Joint Torques
    for i in range(torques.shape[1]):
        axs[1, 0].plot(t, torques[:, i], label=f"Joint {i+1}")
    axs[1, 0].axvspan(2.0, 2.2, color="orange", alpha=0.3)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Torque [Nm]")
    axs[1, 0].set_title("3. Commanded Joint Torques")
    axs[1, 0].grid(True)
    axs[1, 0].legend(ncol=4, fontsize=8)

    # 4. 3D Trajectory
    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax3d.plot(p_act[:, 0], p_act[:, 1], p_act[:, 2], "purple", linewidth=2, label="EE Path")
    ax3d.scatter(p_act[0, 0], p_act[0, 1], p_act[0, 2], color="green", s=50, label="Start")
    ax3d.scatter(target_pos[0], target_pos[1], target_pos[2], color="red", s=60, marker="*", label="Target")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("4. 3D End-Effector Trajectory")
    ax3d.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic plot to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Scenario: Single-Objective Cartesian Reaching")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run headless without viewer")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--out", type=str, default="results/reach.npz", help="Telemetry output path")
    args = parser.parse_args()

    run_reach(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        out_path=args.out,
        save_plot=True
    )


if __name__ == "__main__":
    main()
