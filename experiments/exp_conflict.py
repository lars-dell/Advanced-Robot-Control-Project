"""
Experiment: Conflicting Objectives & Task Priority Hierarchy Trade-off Scenario.

Reproduction of Section V-A from Hoffman et al. (IEEE ICRA 2018):
Target [1.30, 0.15, 0.75] m sits at a radius of 1.51 m from the base origin, against a measured
reachable radius of >= 1.267 m for this model (tool tip, 404 sampled configurations plus
hand-picked near-extension poses; see scripts/probe_genesis.py). The target is therefore genuinely
unreachable, while its Y and Z components individually are not.

NOTE: an earlier version of this docstring justified the conflict with a "~0.93 m maximum
reachable extension". That figure is wrong for this model - it is close to the 855 mm datasheet
*horizontal flange* reach, whereas the relevant quantity is the 3-D radius to the tool tip, which
includes the 0.333 m base height and the 0.21 m cylinder tool. The conclusion is unchanged; only
the stated reason was incorrect.

Demonstrates that the priority order decides which task is sacrificed:
    - Priority order x > y > z: X reach is prioritized; Z height sags to absorb the shortfall.
    - Priority order z > y > x: Z height is satisfied exactly; X reach absorbs the deficit.
"""

import argparse
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers import BaseController, make_controller
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, JointPostureTask, TaskStack

logger = logging.getLogger("ExpConflict")


def _stack_residuals(residuals: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Stacks the per-step priority-residual vectors into one (n_steps, n_levels) array.

    Returns an empty dict when nothing was recorded, or when the level count varied between steps
    (which would mean the task stack changed mid-run and the array would be meaningless).
    """
    if not residuals:
        return {}
    widths = {r.shape[0] for r in residuals}
    if len(widths) != 1:
        logger.warning(
            f"priority-residual level count varied across steps ({sorted(widths)}); not logging it"
        )
        return {}
    return {"priority_residuals": np.array(residuals)}


def run_conflict(
    priority_order: Sequence[str] = ("x", "y", "z"),
    sim_time: float = 7.0,
    dt: float = 0.005,
    show_viewer: bool = False,
    device: str = "cpu",
    out_path: Optional[str] = None,
    show_markers: bool = True,
    record_path: Optional[str] = None,
    save_plot: bool = True,
    solver: str = "daqp",
    controller: Union[str, BaseController] = "hierarchical_qp"
) -> Dict[str, np.ndarray]:
    """
    Executes the conflicting priority scenario under the specified hierarchy order.

    Args:
        priority_order: Sequence of axis strings ('x', 'y', 'z') from highest to lowest priority.
        sim_time: Duration in seconds (default: 7.0s).
        dt: Control timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to display 3D viewer GUI.
        device: 'cpu' or 'gpu'.
        out_path: Optional path for output .npz telemetry.
        show_markers: Whether to render goal/error debug overlays.
        record_path: Optional path to record GIF/video.
        save_plot: Whether to generate diagnostic plots.
        solver: QP solver backend engine ("daqp", "osqp", "qpoases").
        controller: Controller name or BaseController instance (default: 'hierarchical_qp').

    Returns:
        Dict[str, np.ndarray]: Recorded telemetry data.
    """
    order_str = "".join(priority_order)
    default_out = f"results/conflict_{order_str}.npz"
    out_file = out_path or default_out

    logger.info("=================================================================")
    logger.info(f"Running Scenario: Conflicting Objectives (Order: {' > '.join(priority_order)}) [Solver: {solver}]")
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
            kp_cart=300.0,
            kd_cart=60.0,
            kp_null=20.0,
            kd_null=10.0,
            solver_name=solver,
            reg_eps=1e-2
        )



    target_pos = np.array([1.30, 0.15, 0.75])
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def reference(t: float):
        return target_pos, None, np.zeros(3)

    stack = TaskStack()
    axis_of = {"x": 0, "y": 1, "z": 2}
    for level, axis_name in enumerate(priority_order):
        stack.add_task(CartesianPoseTask(
            name=f"{axis_name}_axis",
            priority=level,
            kp=300.0,
            kd=60.0,
            is_6d=False,
            axes=[axis_of[axis_name]],
            trajectory_fn=reference
        ))
    stack.add_task(JointPostureTask(
        name="posture",
        priority=3,
        kp=20.0,
        kd=10.0,
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
    # || J_k B^-1 (tau_final - tau_k*) || per priority level, every step. The controller recomputes
    # this each call and overwrites it, so without capturing it here only the final step survives.
    # This is the evidence that priority is enforced exactly rather than as a weighting.
    log_priority_residuals: List[np.ndarray] = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["t"] = t_curr
        state["target_pos"] = target_pos

        torques = controller.compute_torques(state=state, target=stack, t=t_curr)
        sim.apply_torques(torques)
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
        log_priority_residuals.append(
            np.asarray(getattr(controller, "priority_residuals", []), dtype=float)
        )

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
        "priority_order": np.array(list(priority_order)),
        "n_violations": controller.n_violations,
        "n_solves": controller.n_solves,
        "max_violation": controller.max_violation,
        # Provenance: a violation count is meaningless without the tolerance it was measured at,
        # and the solver changed under this project once already (qpOASES -> daqp, commit 5ac5d65).
        "violation_tol": getattr(controller, "violation_tol", float("nan")),
        "solver_name": np.array(str(getattr(controller, "solver_name", "unknown"))),
        "slack_weight": (float(controller.slack_weight)
                         if getattr(controller, "slack_weight", None) is not None
                         else float("nan")),
        "n_qp_failures": getattr(controller, "n_qp_failures", -1),
        **_stack_residuals(log_priority_residuals),
    }

    if out_file:
        out = pathlib.Path(out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **logs)
        logger.info(f"Logs written to {out}")

    if save_plot:
        plot_conflict(logs, target_pos,
                      output_path=f"results/figures/exp_conflict_{order_str}.png")

    return logs


def plot_conflict(logs: Dict[str, np.ndarray], target_pos: np.ndarray, output_path: str = "results/figures/exp_conflict.png") -> None:
    """Generates per-scenario diagnostic plots for the conflicting task hierarchy."""
    t = logs["time"]
    p_act = logs["ee_pos"]
    torques = logs["torques"]
    p_order = " > ".join(logs.get("priority_order", ["x", "y", "z"]))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Scenario: Conflicting Objectives (Priority Hierarchy: {p_order})", fontsize=14, fontweight="bold")

    # 1. Per-Axis Tracking Errors
    for ax_name, col in (("x", "red"), ("y", "green"), ("z", "blue")):
        k = f"err_{ax_name}_axis"
        if k in logs:
            axs[0, 0].plot(t, logs[k], label=f"{ax_name}-error ({k})", color=col, linewidth=1.8)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Error [m]")
    axs[0, 0].set_title("1. Individual Axis Errors (Evidence of Priority Trade-off)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Position Coordinates vs Target
    axs[0, 1].plot(t, p_act[:, 0], "r-", label="x actual")
    axs[0, 1].plot(t, p_act[:, 1], "g-", label="y actual")
    axs[0, 1].plot(t, p_act[:, 2], "b-", label="z actual")
    axs[0, 1].axhline(target_pos[0], color="r", linestyle="--", alpha=0.6, label="x target (1.30m - unreachable)")
    axs[0, 1].axhline(target_pos[1], color="g", linestyle="--", alpha=0.6, label="y target (0.15m)")
    axs[0, 1].axhline(target_pos[2], color="b", linestyle="--", alpha=0.6, label="z target (0.75m)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Position [m]")
    axs[0, 1].set_title("2. Cartesian Coordinates Evolution")
    axs[0, 1].grid(True)
    axs[0, 1].legend(fontsize=8)

    # 3. Commanded Torques & Limits
    for i in range(torques.shape[1]):
        axs[1, 0].plot(t, torques[:, i], label=f"Joint {i+1}")
    axs[1, 0].axhline(87.0, color="red", linestyle="--", alpha=0.5, label="Torque Bound")
    axs[1, 0].axhline(-87.0, color="red", linestyle="--", alpha=0.5)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Torque [Nm]")
    axs[1, 0].set_title(f"3. Commanded Joint Torques (Violations: {logs['n_violations']})")
    axs[1, 0].grid(True)
    axs[1, 0].legend(ncol=4, fontsize=8)

    # 4. 3D Trajectory
    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax3d.plot(p_act[:, 0], p_act[:, 1], p_act[:, 2], color="purple", linewidth=2, label="Actual EE Path")
    ax3d.scatter(p_act[0, 0], p_act[0, 1], p_act[0, 2], color="green", s=50, label="Start")
    ax3d.scatter(target_pos[0], target_pos[1], target_pos[2], color="red", marker="*", s=80, label="Unreachable Target")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("4. 3D Motion Path")
    ax3d.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic plot to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Scenario: Conflicting Objectives & Priority Hierarchy")
    parser.add_argument("--priority-order", type=str, default="xyz", help="Priority order (e.g. 'xyz' or 'zyx')")
    parser.add_argument("--time", type=float, default=7.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run headless without viewer")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--out", type=str, default=None, help="Telemetry output path")
    args = parser.parse_args()

    run_conflict(
        priority_order=tuple(args.priority_order),
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        out_path=args.out,
        save_plot=True
    )


if __name__ == "__main__":
    main()
