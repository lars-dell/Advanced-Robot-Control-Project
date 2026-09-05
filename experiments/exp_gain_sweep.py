"""
Experiment: Systematic Parameter & Gain Sensitivity Benchmark.

Evaluates the impact of:
    1. Cartesian Stiffness Gains Kp in [100, 400, 1000, 2000] N/m (with critical damping Kd = 2*sqrt(Kp)).
    2. Damping Ratios zeta in [0.5 (underdamped), 1.0 (critically damped), 2.0 (overdamped)].
    3. Cross-Controller Robustness across Gains (Hierarchical QP vs Weighted QP vs Saturated Algebraic vs Classical Transpose).
    4. Actuator Torque Limit Strictness (100%, 50%, 25% of nominal torque bounds).

Outputs high-resolution comparative diagnostic plots to 'exp_gain_sweep.png' and logs metrics.
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers import (
    BaseController,
    QPImpedanceController,
    ClassicalTransposeController,
    SaturatedAlgebraicController,
    WeightedQPController,
    make_controller,
)
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger("ExpGainSweep")


def run_reach_trial(
    controller_name: str = "hierarchical_qp",
    kp: float = 400.0,
    kd: float = 40.0,
    kp_null: float = 20.0,
    kd_null: float = 4.0,
    torque_scale: float = 1.0,
    reg_eps: float = 1e-4,
    sim_time: float = 4.0,
    dt: float = 0.005,
    device: str = "cpu",
    target_offset: Tuple[float, float, float] = (0.15, 0.10, -0.08),
    solver: str = "daqp"
) -> Dict[str, Any]:
    """
    Executes a single step reach simulation trial with specified control parameters.
    """
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device,
        show_markers=False
    )

    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    p_target = p_init + np.array(target_offset, dtype=np.float64)
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    # Scaled torque bounds
    nominal_min = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0])
    nominal_max = np.array([ 87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0])
    tau_min = nominal_min * torque_scale
    tau_max = nominal_max * torque_scale

    ctrl = make_controller(
        name=controller_name,
        n_dofs=7,
        tau_min=tau_min,
        tau_max=tau_max,
        kp_cart=kp,
        kd_cart=kd,
        kp_null=kp_null,
        kd_null=kd_null,
        solver_name=solver,
        reg_eps=reg_eps
    )

    task_stack = TaskStack()
    cart_task = CartesianPoseTask(
        name="reach_p0",
        priority=0,
        kp=kp,
        kd=kd,
        mode="3d",
        trajectory_fn=lambda t: (p_target, None, np.zeros(3))
    )
    task_stack.add_task(cart_task)

    posture_task = JointPostureTask(
        name="posture_p1",
        priority=1,
        kp=kp_null,
        kd=kd_null,
        q_des=q_home
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    log_t = []
    log_pos = []
    log_err = []
    log_torques = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["t"] = t_curr
        state["target_pos"] = p_target

        torques = ctrl.compute_torques(state=state, target=task_stack, t=t_curr)
        sim.apply_torques(torques)
        sim.step()

        p_curr = state["ee_pos"]
        err = float(np.linalg.norm(p_target - p_curr))

        log_t.append(t_curr)
        log_pos.append(p_curr.copy())
        log_err.append(err)
        log_torques.append(torques.copy())

    log_err_arr = np.array(log_err)
    log_torques_arr = np.array(log_torques)

    # Compute key quantitative metrics
    steady_state_err = float(np.mean(log_err_arr[-int(1.0 / dt):]))
    peak_torque = float(np.max(np.abs(log_torques_arr)))
    max_overshoot = float(np.max(np.maximum(0.0, np.maximum(tau_min - log_torques_arr, log_torques_arr - tau_max))))
    settling_idx = np.where(log_err_arr < 0.01)[0]
    settling_time = float(log_t[settling_idx[0]]) if len(settling_idx) > 0 else sim_time

    return {
        "time": np.array(log_t),
        "pos": np.array(log_pos),
        "target": p_target,
        "error": log_err_arr,
        "torques": log_torques_arr,
        "steady_state_error": steady_state_err,
        "peak_torque": peak_torque,
        "max_torque_overshoot": max_overshoot,
        "settling_time": settling_time,
        "n_violations": ctrl.n_violations,
        "kp": kp,
        "kd": kd,
        "controller": controller_name,
        "torque_scale": torque_scale
    }


def run_parameter_sensitivity_study(
    sim_time: float = 3.5,
    dt: float = 0.005,
    device: str = "cpu",
    output_path: str = "exp_gain_sweep.png"
) -> Dict[str, Any]:
    """
    Orchestrates the 4 systematic parameter variation benchmarks.
    """
    logger.info("=================================================================")
    logger.info("Starting Comprehensive Parameter & Gain Sensitivity Benchmark")
    logger.info("=================================================================")

    # 1. Stiffness Gain Sweep: Kp in [100, 400, 1000, 2500] with critically damped Kd
    kp_values = [100.0, 400.0, 1000.0, 2500.0]
    kp_results = []
    logger.info("--- Phase 1: Cartesian Stiffness Kp Sweep ---")
    for kp in kp_values:
        # Effective Cartesian mass is roughly ~2.5 kg, critical damping: kd = 2 * sqrt(m * kp)
        kd = 2.0 * np.sqrt(2.5 * kp)
        logger.info(f"Simulating Hierarchical QP with Kp={kp:.0f} N/m, Kd={kd:.1f} Ns/m...")
        res = run_reach_trial(
            controller_name="hierarchical_qp",
            kp=kp,
            kd=kd,
            sim_time=sim_time,
            dt=dt,
            device=device
        )
        kp_results.append(res)

    # 2. Damping Ratio Sweep at Kp = 600: underdamped (zeta=0.4), critically damped (zeta=1.0), overdamped (zeta=2.2)
    logger.info("--- Phase 2: Damping Ratio Sweep ---")
    base_kp = 600.0
    kd_crit = 2.0 * np.sqrt(2.5 * base_kp)  # approx 77.5
    zeta_dict = {
        "Underdamped (zeta=0.35)": 0.35 * kd_crit,
        "Critically Damped (zeta=1.0)": 1.0 * kd_crit,
        "Overdamped (zeta=2.2)": 2.2 * kd_crit,
    }
    damping_results = {}
    for name, kd in zeta_dict.items():
        logger.info(f"Simulating Damping: {name} (Kd={kd:.1f} Ns/m)...")
        damping_results[name] = run_reach_trial(
            controller_name="hierarchical_qp",
            kp=base_kp,
            kd=kd,
            sim_time=sim_time,
            dt=dt,
            device=device
        )

    # 3. Cross-Controller Comparison across Gain Levels: Kp = 100 vs Kp = 1000
    logger.info("--- Phase 3: Cross-Controller Comparison under Low vs High Gains ---")
    controllers = ["hierarchical_qp", "weighted_qp", "saturated_algebraic", "classical_transpose"]
    cross_results = {}
    for ctrl in controllers:
        cross_results[f"{ctrl}_low"] = run_reach_trial(
            controller_name=ctrl, kp=150.0, kd=35.0, sim_time=sim_time, dt=dt, device=device
        )
        cross_results[f"{ctrl}_high"] = run_reach_trial(
            controller_name=ctrl, kp=1200.0, kd=110.0, sim_time=sim_time, dt=dt, device=device
        )

    # 4. Torque Limit Scaling: 100% vs 50% vs 25% nominal torque limits
    logger.info("--- Phase 4: Actuator Torque Constraint Strictness Sweep ---")
    scales = [1.0, 0.50, 0.25]
    torque_scale_results = {}
    for s in scales:
        logger.info(f"Simulating Hierarchical QP under {int(s*100)}% Torque Limits...")
        torque_scale_results[f"{int(s*100)}%"] = run_reach_trial(
            controller_name="hierarchical_qp",
            kp=800.0,
            kd=85.0,
            torque_scale=s,
            sim_time=sim_time,
            dt=dt,
            device=device
        )

    # Plot comprehensive 4-panel figure
    plot_gain_sensitivity(
        kp_results=kp_results,
        damping_results=damping_results,
        cross_results=cross_results,
        torque_results=torque_scale_results,
        output_path=output_path
    )

    return {
        "kp_results": kp_results,
        "damping_results": damping_results,
        "cross_results": cross_results,
        "torque_results": torque_scale_results
    }


def plot_gain_sensitivity(
    kp_results: List[Dict[str, Any]],
    damping_results: Dict[str, Dict[str, Any]],
    cross_results: Dict[str, Dict[str, Any]],
    torque_results: Dict[str, Dict[str, Any]],
    output_path: str = "exp_gain_sweep.png"
) -> None:
    """
    Renders a 4-panel publication-grade figure summarizing the parameter sensitivity analysis.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Parametric Sensitivity & Controller Robustness Analysis (Hoffman et al. ICRA 2018)",
        fontsize=15,
        fontweight="bold"
    )

    # Panel 1: Cartesian Stiffness Kp Sweep
    ax1 = axs[0, 0]
    for r in kp_results:
        ax1.plot(r["time"], r["error"] * 1000.0, label=f"Kp = {r['kp']:.0f} N/m (settle: {r['settling_time']:.2f}s)", linewidth=2.0)
    ax1.set_title("Panel A: Cartesian Stiffness Kp Sweep (Critically Damped)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Tracking Error [mm]")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Damping Ratio Sweep (Zeta)
    ax2 = axs[0, 1]
    for name, r in damping_results.items():
        ax2.plot(r["time"], r["error"] * 1000.0, label=f"{name}", linewidth=2.0)
    ax2.set_title("Panel B: Damping Ratio Impact at Kp = 600 N/m", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Tracking Error [mm]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Cross-Controller Comparison under High Gain (Kp = 1200 N/m)
    ax3 = axs[1, 0]
    ctrl_names = {
        "hierarchical_qp_high": ("Hierarchical QP (Proposed)", "blue", "-"),
        "weighted_qp_high": ("Weighted-Sum QP", "green", "--"),
        "saturated_algebraic_high": ("Saturated Algebraic (Eq. 10)", "orange", "-."),
        "classical_transpose_high": ("Classical Transpose (Eq. 9)", "red", ":"),
    }
    for k, (label, color, ls) in ctrl_names.items():
        if k in cross_results:
            r = cross_results[k]
            tau_norm = np.linalg.norm(r["torques"], axis=1)
            ax3.plot(r["time"], tau_norm, label=label, color=color, linestyle=ls, linewidth=2.0)
    ax3.set_title("Panel C: Commanded Joint Torque Norm ||tau|| at High Gain (Kp=1200)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Total Torque Norm [N·m]")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=9)

    # Panel 4: Actuator Torque Constraint Strictness (100% vs 50% vs 25%)
    ax4 = axs[1, 1]
    colors = ["teal", "darkorange", "crimson"]
    for i, (scale_name, r) in enumerate(torque_results.items()):
        ax4.plot(r["time"], r["error"] * 1000.0, label=f"Torque Bounds: {scale_name} (Violations: {r['n_violations']})", color=colors[i], linewidth=2.0)
    ax4.set_title("Panel D: Performance under Strict Torque Limits (100%, 50%, 25%)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Tracking Error [mm]")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved publication-grade parameter sensitivity plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Systematic Parameter & Gain Sensitivity Benchmark")
    parser.add_argument("--sim-time", type=float, default=3.5, help="Simulation duration per trial in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Computing backend")
    parser.add_argument("--out", type=str, default="exp_gain_sweep.png", help="Output figure filename")
    args = parser.parse_args()

    run_parameter_sensitivity_study(
        sim_time=args.sim_time,
        dt=args.dt,
        device=args.device,
        output_path=args.out
    )


if __name__ == "__main__":
    main()
