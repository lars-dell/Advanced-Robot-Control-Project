"""
Main entry point for multi-priority Cartesian impedance control simulation.

Initializes the Genesis simulator, instantiates the QP impedance controller,
builds the modular task hierarchy stack, executes the real-time simulation loop,
and handles telemetry data logging.
"""

import argparse
import logging
import pathlib
from typing import Dict, List
import numpy as np

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger(__name__)


def build_scenario(name: str, initial_ee_pos: np.ndarray,
                   priority_order: tuple = ("x", "y", "z")):
    """
    Build a prioritized task stack and its Cartesian reference.

    Returns:
        (TaskStack, reference_fn, reg_eps) where reference_fn(t) -> (pos_des, rot_des, vel_des)
        and reg_eps is the QP regularisation weight this scenario needs.

    On reg_eps: the eps*||tau||^2 term in eq. (18) trades task accuracy against robustness in
    near-singular configurations. A scenario that stays well inside the workspace wants it small
    (1e-4 tracks to ~0 error). A scenario deliberately driven to the workspace boundary needs it
    large: measured on 'conflict', eps=1e-4 leaves the arm unstable at 40 rad/s with 65 failed QPs,
    while eps=10 gives 1.95 rad/s and none. It is therefore a per-scenario parameter, not a global
    default.
    """
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    if name == "reach":
        # The original single-objective demo: move to a nearby reachable point and hold it.
        # One Cartesian objective plus a joint-space posture task, which cannot conflict with it
        # because the posture target is compatible with the Cartesian one.
        target = initial_ee_pos + np.array([0.15, 0.10, -0.05])

        def reference(t: float):
            return target, None, np.zeros(3)

        stack = TaskStack()
        stack.add_task(CartesianPoseTask(name="cartesian_primary", priority=0,
                                         kp=400.0, kd=40.0, is_6d=False,
                                         trajectory_fn=reference))
        stack.add_task(JointPostureTask(name="posture", priority=1,
                                        kp=20.0, kd=4.0, q_des=q_home))
        return stack, reference, 1e-4

    if name == "reach_split":
        # Same reachable target as 'reach', but the single 3-D Cartesian objective is SPLIT into
        # three independent 1-D objectives at descending priority (x > y > z). Every one of them is
        # satisfiable at once, so this isolates one thing: does decomposing a task into a priority
        # stack still reproduce the undecomposed result? If the cascade is correct it must, because
        # nothing is being traded off. It is the control case for the conflicting version.
        target = initial_ee_pos + np.array([0.15, 0.10, -0.05])

        def reference(t: float):
            return target, None, np.zeros(3)

        stack = TaskStack()
        for axis, (nm, pr) in enumerate((("x_axis", 0), ("y_axis", 1), ("z_axis", 2))):
            stack.add_task(CartesianPoseTask(name=nm, priority=pr, kp=400.0, kd=40.0,
                                             is_6d=False, axes=[axis], trajectory_fn=reference))
        stack.add_task(JointPostureTask(name="posture", priority=3,
                                        kp=20.0, kd=4.0, q_des=q_home))
        return stack, reference, 1e-4

    if name == "conflict":
        # One axis out of reach, the other two comfortably reachable.
        #
        # target = [1.30, 0.15, 0.75]. The arm can extend to about x = 0.93 while holding the y and
        # z components, so the x objective is permanently unsatisfiable while y and z are not. The
        # three objectives therefore compete for the same joints: serving x demands stretching out
        # along +x, which drags the tip away from the z it is asked to hold.
        #
        # The priority order decides who loses, and swapping it swaps the outcome:
        #     x > y > z   ->  x_err 0.375   z_err 0.307
        #     z > y > x   ->  x_err 0.461   z_err 0.000
        # Putting z on top satisfies it exactly and costs x; putting x on top recovers x at z's
        # expense. That trade is the observable consequence of the hierarchy, and it is the
        # evidence for requirement 7 ("explain how task priorities are handled").
        target = np.array([1.30, 0.15, 0.75])

        def reference(t: float):
            return target, None, np.zeros(3)

        stack = TaskStack()
        axis_of = {"x": 0, "y": 1, "z": 2}
        for level, axis_name in enumerate(priority_order):
            stack.add_task(CartesianPoseTask(name=f"{axis_name}_axis", priority=level,
                                             kp=300.0, kd=60.0, is_6d=False,
                                             axes=[axis_of[axis_name]], trajectory_fn=reference))
        stack.add_task(JointPostureTask(name="posture", priority=3,
                                        kp=20.0, kd=10.0, q_des=q_home))
        return stack, reference, 1e-2

    raise ValueError(f"unknown scenario {name!r}; expected 'reach' or 'conflict'")


def run_simulation(
    sim_time: float = 5.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "cpu",
    out_path: str = "results/run.npz",
    show_markers: bool = True,
    scenario: str = "reach",
    priority_order: tuple = ("x", "y", "z"),
    record_path: str = None
) -> Dict[str, np.ndarray]:
    """
    Executes the main control simulation loop using the modular task stack and paper-compliant QP controller.

    Args:
        sim_time: Total simulation duration in seconds (default: 5.0s).
        dt: Control timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to launch interactive 3D Genesis viewer.
        device: Computing backend ('cpu' or 'gpu').
        out_path: Destination .npz for the telemetry log.
        show_markers: Draw the goal sphere, error line, disturbance arrow and tip trail.

    Returns:
        Dict[str, np.ndarray]: Recorded log data arrays for analysis and plotting.
    """
    logger.info(f"[Main] Initializing Genesis simulation on backend '{device}'...")
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        show_markers=show_markers,
        record_path=record_path
    )

    logger.info("[Main] Instantiating Hierarchical QP Impedance Controller (CasADi / qpOASES)...")
    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=400.0,
        kd_cart=40.0,
        kp_null=20.0,
        kd_null=4.0,
        use_qpoases=True
    )

    # Initial state extraction
    initial_state = sim.get_state()
    initial_ee_pos = initial_state["ee_pos"].copy()
    logger.info(f"[Main] Robot initialized. Initial EE position: {initial_ee_pos}")

    # Build the prioritized task stack for the requested scenario
    task_stack, reference, reg_eps = build_scenario(scenario, initial_ee_pos, priority_order)
    controller.reg_eps = reg_eps
    logger.info(f"[Main] Scenario '{scenario}' with {len(task_stack.tasks)} priority levels: "
                + " > ".join(t.name for t in task_stack.tasks)
                + f"  (reg_eps={reg_eps:g})")

    sim.set_goal(reference(0.0)[0])

    sim.start_recording()

    n_steps = int(sim_time / dt)
    logger.info(f"[Main] Starting simulation control loop for {sim_time} seconds ({n_steps} steps)...")

    # Data logging containers
    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_errors: List[np.ndarray] = []

    log_task_errors: List[Dict[str, float]] = []

    for step in range(n_steps):
        t_curr = step * dt

        # Extract robot dynamics state
        state = sim.get_state()
        state["t"] = t_curr
        target_pos = reference(t_curr)[0]
        state["target_pos"] = target_pos

        # Compute commanded joint torques via Hierarchical QP Optimization
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques to robot joints
        sim.apply_torques(torques)

        # External disturbance for a 0.2 s window (injected as J^T f_ext)
        sim.set_external_force(
            np.array([10.0, 0.0, 0.0]) if 2.0 <= t_curr <= 2.2 else np.zeros(3)
        )

        # Step physics simulation engine
        sim.step()

        # Refresh viewer overlays (no-op when headless; internally throttled)
        sim.set_goal(target_pos)
        sim.update_viz(tip_pos=state["ee_pos"], goal_pos=target_pos)
        sim.record_frame()

        # Log telemetry data
        log_time.append(t_curr)
        log_ee_pos.append(state["ee_pos"].copy())
        log_ee_pos_des.append(target_pos.copy())
        log_torques.append(torques.copy())
        log_task_errors.append(task_stack.get_task_errors(state))
        log_errors.append(float(np.linalg.norm(target_pos - state["ee_pos"])))

        if step % 100 == 0:
            errs = " ".join(f"{k}={v:.3f}" for k, v in log_task_errors[-1].items())
            logger.info(f"Step {step:4d}/{n_steps} | t={t_curr:5.2f}s | {errs} | "
                        f"max|tau|={np.max(np.abs(torques)):5.2f} Nm")

    sim.stop_recording()
    if record_path:
        logger.info(f"[Main] Recording written to {record_path}")

    logger.info("[Main] Simulation completed successfully.")

    # Package log data
    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "errors": np.array(log_errors),
        "tau_min": sim.tau_min,
        "tau_max": sim.tau_max,
        # Feasibility evidence: the paper guarantees these stay at zero without any clipping.
        **{f"err_{name}": np.array([e[name] for e in log_task_errors])
           for name in (log_task_errors[0] if log_task_errors else {})},
        "task_names": np.array([t.name for t in task_stack.tasks]),
        "n_violations": controller.n_violations,
        "n_solves": controller.n_solves,
        "max_violation": controller.max_violation,
    }

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **logs)
    logger.info(f"[Main] Logs written to {out} ({len(logs['time'])} steps)")

    return logs


def main() -> None:
    """
    CLI parser and launcher.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Multi-Priority Cartesian Impedance Control (QP) in Genesis")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode without 3D viewer GUI")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend device")
    parser.add_argument("--out", type=str, default="results/run.npz", help="Where to write the telemetry .npz")
    parser.add_argument("--no-markers", action="store_true", help="Disable goal/error/trail overlays")
    parser.add_argument("--record", type=str, default=None,
                        help="Record the run to this file (e.g. docs/media/run.gif)")
    parser.add_argument("--priority-order", type=str, default="xyz",
                        help="Priority ranking of the Cartesian axes in 'conflict', highest first "
                             "(e.g. 'xyz' or 'zyx'). Swapping it swaps which objective is sacrificed.")
    parser.add_argument("--scenario", type=str, default="reach", choices=["reach", "reach_split", "conflict"],
                        help="'reach': one 3-D Cartesian objective. "
                             "'reach_split': the same target as three 1-D Cartesian objectives "
                             "at descending priority (reachable, so nothing is sacrificed). "
                             "'conflict': three 1-D Cartesian objectives competing for an "
                             "out-of-reach reference (Hoffman et al. sec. V-A)")
    args = parser.parse_args()

    logs = run_simulation(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        out_path=args.out,
        show_markers=not args.no_markers,
        scenario=args.scenario,
        priority_order=tuple(args.priority_order),
        record_path=args.record
    )

    tau = logs["torques"]
    logger.info(f"[Main] final EE error {logs['errors'][-1]:.4f} m | "
                f"max |tau| {np.abs(tau).max():.2f} Nm | "
                f"torque-limit violations: {logs['n_violations']}/{logs['n_solves']} steps "
                f"(worst {logs['max_violation']:.2e} Nm)")


if __name__ == "__main__":
    main()
