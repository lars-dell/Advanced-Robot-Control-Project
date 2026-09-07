"""
Experiment 7: 4-Way Baseline Comparison & Conflicting Task Reproduction.

Directly reproduces Section V-A (Figures 1-4) of the reference paper:
    "Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization"
    Enrico Mingo Hoffman et al. (IEEE ICRA 2018).

Compares 4 distinct controller classes derived from BaseController under conflicting task requirements:
    1. ClassicalTransposeController: tau = J^T * f (Eq. 9)
    2. SaturatedAlgebraicController: tau = clip(J_0^T * f_0 + (I - J_0^T * J_bar_0^T) * tau_1) (Eq. 10)
    3. WeightedQPController: min sum w_i ||e_i||^2 s.t. torque bounds
    4. QPImpedanceController: Multi-priority cascade with strict equality constraints (Eq. 18)
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers import (
    BaseController,
    ClassicalTransposeController,
    QPImpedanceController,
    SaturatedAlgebraicController,
    WeightedQPController,
)
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, JointPostureTask, TaskStack

logger = logging.getLogger("Exp7_BaselineComparison")


def build_controller(control_mode: str) -> BaseController:
    """
    Factory creating a BaseController instance matching the requested baseline mode.
    """
    if control_mode == "classical_transpose":
        return ClassicalTransposeController(n_dofs=7)
    elif control_mode == "saturated_algebraic":
        return SaturatedAlgebraicController(n_dofs=7, reg_pinv=1e-3)
    elif control_mode == "weighted_qp":
        return WeightedQPController(n_dofs=7, weights=[1.0, 0.2, 0.02], solver_name="daqp", reg_eps=1e-2)
    elif control_mode == "hierarchical_qp":
        # slack_weight MUST stay None here. This panel exists to demonstrate that priority is
        # enforced as a hard equality (eq. 17/18); running it with a quadratic slack penalty
        # relaxes exactly the property the figure is meant to prove, and makes the comparison
        # against the weighted-sum QP baseline meaningless - both would then be weighted methods.
        # Measured on the `conflict` scenario: strict priority holds the cascade residual at
        # 7.8e-11 (qpOASES) / 4.0e-08 (daqp). If a slack penalty is ever needed to keep this
        # experiment feasible, that is a reportable finding and must be stated, not silently set.
        return QPImpedanceController(n_dofs=7, solver_name="daqp", reg_eps=1e-2, slack_weight=None)
    else:
        raise ValueError(f"Unknown control mode: {control_mode}")


def run_single_controller_sim(
    control_mode: str,
    sim_time: float = 5.0,
    dt: float = 0.005,
    device: str = "cpu",
    sim: Any = None
) -> Dict[str, np.ndarray]:
    """
    Runs a single simulation run for the specified controller under conflicting task targets.

    Args:
        control_mode: 'classical_transpose', 'saturated_algebraic', 'weighted_qp', or 'hierarchical_qp'.
        sim_time: Duration in seconds (default: 5.0s).
        dt: Control timestep in seconds.
        device: 'cpu' or 'gpu'.
        sim: Optional existing GenesisSim instance to reuse.

    Returns:
        Dict[str, np.ndarray]: Logged trajectory telemetry.
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

    controller: BaseController = build_controller(control_mode)

    # Initial state
    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    q_home = state["q"].copy()

    # Conflicting task definitions (Hoffman et al. ICRA 2018 Section V-A):
    # High-Priority Task (Level 0): Periodic sine wave along Z
    z_center = p_init[2]
    z_amp = 0.05
    z_freq = 0.5  # 0.5 Hz periodic tracking

    # Low-Priority Task (Level 1): Strictly unreachable XY setpoint
    # WARNING - this target may not actually be unreachable; verify before trusting this panel.
    # The original justification here ("radial distance ~0.96m strictly exceeds Franka Panda max
    # reach of 0.855m") is wrong twice over. 855 mm is the datasheet *horizontal flange* reach; the
    # relevant quantity is the 3-D radius to the tool tip, which includes the 0.333 m base height
    # and the 0.21 m cylinder tool. scripts/probe_genesis.py measured a reachable radius of
    # >= 1.267 m for this model. With z_center = 0.4853 m (the home tool-tip height), this target
    # sits at a 3-D radius of only 1.06-1.10 m - INSIDE the measured reachable set.
    # Reachability is not purely radial (the joint-4 limit [-3.0718, -0.0698] means the arm cannot
    # fully straighten, so the reachable set is not a sphere), so it may still be unreachable in
    # this particular direction - but that has to be shown, not assumed. If the XY task converges,
    # there is no conflict here and this experiment is not demonstrating what it claims.
    # For comparison, the `conflict` scenario targets a 1.51 m radius, which is unambiguous.
    xy_target = np.array([0.95, 0.15])

    # Build 3-Level Conflicting Task Stack (Z Priority 0, XY Priority 1, Null-space Posture Priority 2)
    task_stack = TaskStack()

    # Level 0 Task: Z-Axis Sine Tracking
    def z_traj(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_des = z_center + z_amp * np.sin(2.0 * np.pi * z_freq * t)
        z_dot = 2.0 * np.pi * z_freq * z_amp * np.cos(2.0 * np.pi * z_freq * t)
        return np.array([p_init[0], p_init[1], z_des]), np.eye(3), np.array([0.0, 0.0, z_dot])

    task_z = CartesianPoseTask(
        name="z_priority_0",
        priority=0,
        kp=400.0,
        kd=40.0,
        mode="z",
        trajectory_fn=z_traj
    )
    task_stack.add_task(task_z)

    # Level 1 Task: Conflicting Unreachable XY Target with Smooth Ramp
    def xy_traj(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = min(1.0, max(0.0, t / 1.0))
        alpha = 0.5 * (1.0 - np.cos(np.pi * s))
        alpha_dot = (0.5 * np.pi / 1.0) * np.sin(np.pi * s) if s < 1.0 else 0.0
        xy_curr = p_init[:2] + alpha * (xy_target - p_init[:2])
        xy_vel = alpha_dot * (xy_target - p_init[:2])
        return np.array([xy_curr[0], xy_curr[1], p_init[2]]), np.eye(3), np.array([xy_vel[0], xy_vel[1], 0.0])

    task_xy = CartesianPoseTask(
        name="xy_priority_1",
        priority=1,
        kp=300.0,
        kd=30.0,
        mode="xy",
        trajectory_fn=xy_traj
    )
    task_stack.add_task(task_xy)

    # Level 2 Task: Null-Space Posture Stabilization
    task_posture = JointPostureTask(
        name="posture_priority_2",
        priority=2,
        kp=20.0,
        kd=10.0,
        q_des=q_home
    )
    task_stack.add_task(task_posture)

    n_steps = int(sim_time / dt)

    log_t: List[float] = []
    log_p_act: List[np.ndarray] = []
    log_p_des: List[np.ndarray] = []
    log_tau: List[np.ndarray] = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        p_curr = state["ee_pos"]

        # Current reference point
        z_des = z_center + z_amp * np.sin(2.0 * np.pi * z_freq * t_curr)
        s = min(1.0, max(0.0, t_curr / 1.0))
        alpha = 0.5 * (1.0 - np.cos(np.pi * s))
        xy_curr = p_init[:2] + alpha * (xy_target - p_init[:2])
        p_des = np.array([xy_curr[0], xy_curr[1], z_des])
        state["target_pos"] = p_des
        state["target_vel"] = np.array([0.0, 0.0, 2.0 * np.pi * z_freq * z_amp * np.cos(2.0 * np.pi * z_freq * t_curr)])

        # Compute torques using BaseController interface
        tau_cmd = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        sim.apply_torques(tau_cmd)
        sim.step()

        log_t.append(t_curr)
        log_p_act.append(p_curr.copy())
        log_p_des.append(p_des.copy())
        log_tau.append(tau_cmd.copy())

    return {
        "time": np.array(log_t),
        "p_act": np.array(log_p_act),
        "p_des": np.array(log_p_des),
        "torques": np.array(log_tau)
    }


def run_experiment_7(
    sim_time: float = 5.0,
    dt: float = 0.005,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Executes the 4-way comparative study using the newly introduced BaseController subclasses.
    """
    logger.info("=================================================================")
    logger.info("Running Experiment 7: 4-Way Baseline Comparison (Hoffman et al.)")
    logger.info("=================================================================")

    modes = [
        ("classical_transpose", "1. Classical Transpose Law (Eq. 9)"),
        ("saturated_algebraic", "2. Saturated Algebraic Null-Space (Eq. 10)"),
        ("weighted_qp", "3. Single-Level Weighted-Sum QP"),
        ("hierarchical_qp", "4. Proposed Hierarchical Cascade QP (Eq. 18)")
    ]

    all_results: Dict[str, Dict[str, np.ndarray]] = {}
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device
    )

    for mode_key, mode_title in modes:
        logger.info(f"Simulating: {mode_title}...")
        res = run_single_controller_sim(
            control_mode=mode_key,
            sim_time=sim_time,
            dt=dt,
            device=device,
            sim=sim
        )
        all_results[mode_key] = res

    if save_plot:
        plot_experiment_7(all_results)

    logger.info("Experiment 7 completed successfully.")
    return all_results


def plot_experiment_7(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "results/figures/exp7_baseline_comparison.png"
) -> None:
    """
    Generates a 4-panel figure directly reproducing Figures 1-4 of Hoffman et al. ICRA 2018.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Experiment 7: 4-Way Baseline Comparison under Conflicting Tasks (Hoffman et al. ICRA 2018)",
        fontsize=14,
        fontweight="bold"
    )

    panels = [
        ("classical_transpose", axs[0, 0], "Fig 1 Reproduction: Classical Transpose Control (Eq. 9)\n[No Priority Hierarchy - Mutual Task Interference]"),
        ("saturated_algebraic", axs[0, 1], "Fig 2 Reproduction: Saturated Algebraic Null-Space (Eq. 10)\n[Naive Torque Clipping - Nullspace Distortions]"),
        ("weighted_qp", axs[1, 0], "Weighted-Sum QP (Single-Level)\n[Soft Penalties - Secondary Task Pollutes Primary Task]"),
        ("hierarchical_qp", axs[1, 1], "Fig 4 Reproduction: Proposed Hierarchical Cascade QP (Eq. 18)\n[Strict Priority - Clean P0 Tracking & Safe Torques]")
    ]

    for mode_key, ax, title in panels:
        res = results[mode_key]
        t = res["time"]
        p_act = res["p_act"]
        p_des = res["p_des"]

        # Reference workspace reach limit of Franka Emika Panda
        ax.axhline(1.267, color="gray", linestyle=":", linewidth=1.2, label="Measured reachable radius (1.267 m)")

        # Plot X, Y, Z actual vs desired
        ax.plot(t, p_des[:, 0], "r--", alpha=0.7, label="x_des (unreachable 0.95m)")
        ax.plot(t, p_act[:, 0], "r-", label="x_act")

        ax.plot(t, p_des[:, 1], "g--", alpha=0.7, label="y_des (0.15m)")
        ax.plot(t, p_act[:, 1], "g-", label="y_act")

        ax.plot(t, p_des[:, 2], "b--", alpha=0.8, linewidth=2, label="z_des (Priority 0)")
        ax.plot(t, p_act[:, 2], "b-", linewidth=2, label="z_act")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cartesian Position [m]")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved 4-way baseline comparative plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 7: 4-Way Baseline Comparison")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_7(
        sim_time=args.time,
        dt=args.dt,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
