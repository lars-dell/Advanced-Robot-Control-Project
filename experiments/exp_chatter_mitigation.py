"""
Experiment: Active-Set Torque Chattering Mitigation Benchmark.

Directly evaluates and compares four hierarchical QP configurations on the 'conflict' scenario
to demonstrate the elimination of boundary active-set limit cycles:
    1. Baseline Strict Hierarchical QP (Hoffman Eq. 18: hard equality, no rate penalty)
    2. Soft Equality Constraints Only (quadratic slack penalty rho = 2000.0)
    3. Torque Rate Penalty Only (w_dtau = 0.05 on ||tau_k - tau_{k-1}||^2)
    4. Combined (Soft Constraints rho = 2000.0 + Torque Rate Penalty w_dtau = 0.05)

Generates a dedicated 4-panel publication-grade comparative plot 'results/figures/exp_chatter_mitigation.png'.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.qp_impedance import QPImpedanceController
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, JointPostureTask, TaskStack

logger = logging.getLogger("ExpChatterMitigation")


def compute_chatter(torques: np.ndarray, dt: float) -> float:
    """
    Computes RMS torque rate-of-change (chatter metric): sqrt(mean(||tau_dot||^2)).
    """
    if len(torques) < 2:
        return 0.0
    tau_dot = np.diff(torques, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(tau_dot ** 2, axis=1))))


def run_chatter_trial(
    case_name: str,
    slack_weight: Any = None,
    torque_rate_weight: float = 0.0,
    max_torque_rate: Any = None,
    sim_time: float = 5.0,
    dt: float = 0.005,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Simulates a single run of the conflict scenario with specified QP smoothing options.
    """
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device,
        show_markers=False
    )

    ctrl = QPImpedanceController(
        n_dofs=7,
        kp_cart=300.0,
        kd_cart=60.0,
        kp_null=20.0,
        kd_null=10.0,
        solver_name="daqp",
        reg_eps=1e-2,
        slack_weight=slack_weight,
        torque_rate_weight=torque_rate_weight,
        max_torque_rate=max_torque_rate
    )

    target_pos = np.array([1.30, 0.15, 0.75])
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    stack = TaskStack()
    axis_of = {"x": 0, "y": 1, "z": 2}
    for level, ax in enumerate(["x", "y", "z"]):
        stack.add_task(CartesianPoseTask(
            name=f"{ax}_axis",
            priority=level,
            kp=300.0,
            kd=60.0,
            is_6d=False,
            axes=[axis_of[ax]],
            trajectory_fn=lambda t: (target_pos, None, np.zeros(3))
        ))
    stack.add_task(JointPostureTask(
        name="posture",
        priority=3,
        kp=20.0,
        kd=10.0,
        q_des=q_home
    ))

    n_steps = int(sim_time / dt)
    log_t = []
    log_pos = []
    log_torques = []
    log_residuals = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["t"] = t_curr
        state["target_pos"] = target_pos

        torques = ctrl.compute_torques(state=state, target=stack, t=t_curr)
        sim.apply_torques(torques)
        sim.step()

        log_t.append(t_curr)
        log_pos.append(state["ee_pos"].copy())
        log_torques.append(torques.copy())
        log_residuals.append(list(ctrl.priority_residuals))

    t_arr = np.array(log_t)
    pos_arr = np.array(log_pos)
    torques_arr = np.array(log_torques)

    # Metrics
    err_x = np.abs(pos_arr[:, 0] - target_pos[0])
    err_y = np.abs(pos_arr[:, 1] - target_pos[1])
    err_z = np.abs(pos_arr[:, 2] - target_pos[2])

    rmse_x = float(np.sqrt(np.mean(err_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(err_y ** 2)))
    rmse_z = float(np.sqrt(np.mean(err_z ** 2)))

    chatter = compute_chatter(torques_arr, dt)
    tau_min = ctrl.tau_min
    tau_max = ctrl.tau_max
    overshoot = float(np.max(np.maximum(0.0, np.maximum(tau_min - torques_arr, torques_arr - tau_max))))

    return {
        "case_name": case_name,
        "time": t_arr,
        "pos": pos_arr,
        "target": target_pos,
        "torques": torques_arr,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
        "chatter": chatter,
        "overshoot": overshoot,
        "n_violations": ctrl.n_violations,
    }


def run_chatter_mitigation_benchmark(
    sim_time: float = 5.0,
    dt: float = 0.005,
    device: str = "cpu",
    output_path: str = "results/figures/exp_chatter_mitigation.png"
) -> Dict[str, Any]:
    """
    Executes the 4-case comparison study and produces the comparative visualization.
    """
    logger.info("=================================================================")
    logger.info("Running Active-Set Torque Chattering Mitigation Benchmark")
    logger.info("=================================================================")

    cases_cfg = [
        ("1. Baseline Strict QP (Eq. 18)", None, 0.0, None),
        ("2. Soft Constraints Only (rho=2000)", 2000.0, 0.0, None),
        ("3. Torque Rate Penalty Only (w=0.05)", None, 0.05, None),
        ("4. Combined (Soft + Rate Penalty)", 2000.0, 0.05, None),
    ]

    all_results = {}
    for name, slack, rate_w, max_rate in cases_cfg:
        logger.info(f"Simulating: {name}...")
        res = run_chatter_trial(
            case_name=name,
            slack_weight=slack,
            torque_rate_weight=rate_w,
            max_torque_rate=max_rate,
            sim_time=sim_time,
            dt=dt,
            device=device
        )
        all_results[name] = res

    # Print summary table
    print_chatter_summary_table(all_results)

    # Plot 4-panel comparative figure
    plot_chatter_mitigation(all_results, output_path=output_path)

    return all_results


def print_chatter_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Prints a formatted summary comparison table.
    """
    header = (
        f"\n{'='*105}\n"
        f"  ACTIVE-SET CHATTERING MITIGATION BENCHMARK: CONFLICT SCENARIO (X > Y > Z)  \n"
        f"{'='*105}\n"
        f"{'Configuration':<42} | {'P0 (X) RMSE [m]':<15} | {'P2 (Z) RMSE [m]':<15} | {'Chatter [Nm/s]':<15} | {'Violations':<10}\n"
        f"{'-'*105}"
    )
    print(header)
    logger.info(header)

    baseline_chatter = None
    for name, r in results.items():
        if baseline_chatter is None:
            baseline_chatter = r["chatter"]
            chatter_str = f"{r['chatter']:10.1f} (100.0%)"
        else:
            pct = 100.0 * r["chatter"] / max(1e-6, baseline_chatter)
            chatter_str = f"{r['chatter']:10.1f} ({pct:5.1f}%)"

        line = (
            f"{name:<42} | "
            f"{r['rmse_x']:15.4f} | "
            f"{r['rmse_z']:15.4f} | "
            f"{chatter_str:<15} | "
            f"{r['n_violations']:10d}"
        )
        print(line)
        logger.info(line)
    print(f"{'='*105}\n")


def plot_chatter_mitigation(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/figures/exp_chatter_mitigation.png"
) -> None:
    """
    Generates a 4-panel high-resolution publication-grade figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Active-Set Torque Chattering Mitigation in Multi-Priority QP Impedance Control\n"
        "Conflict Scenario: X > Y > Z (Target [1.30, 0.15, 0.75] m - Out-of-Reach Boundary)",
        fontsize=14,
        fontweight="bold"
    )

    styles = {
        "1. Baseline Strict QP (Eq. 18)": ("#1f77b4", "-", 1.8),
        "2. Soft Constraints Only (rho=2000)": ("#ff7f0e", "--", 2.0),
        "3. Torque Rate Penalty Only (w=0.05)": ("#2ca02c", "-.", 2.0),
        "4. Combined (Soft + Rate Penalty)": ("#d62728", "-", 2.5),
    }

    # Panel 1: Joint 2 (Shoulder Pitch) Torque Profile
    ax1 = axs[0, 0]
    for name, r in results.items():
        color, ls, lw = styles.get(name, ("black", "-", 1.5))
        ax1.plot(r["time"], r["torques"][:, 1], label=name, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    ax1.axhline(87.0, color="darkred", linestyle=":", label="Max Limit (+87 Nm)")
    ax1.axhline(-87.0, color="darkred", linestyle=":", label="Min Limit (-87 Nm)")
    ax1.set_title("Panel A: Joint 2 (Shoulder Pitch) Commanded Torque", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Torque [N·m]")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Primary Task (X-axis) Tracking
    ax2 = axs[0, 1]
    for name, r in results.items():
        color, ls, lw = styles.get(name, ("black", "-", 1.5))
        ax2.plot(r["time"], r["pos"][:, 0], label=f"{name} (RMSE: {r['rmse_x']:.3f}m)", color=color, linestyle=ls, linewidth=lw)
    ax2.axhline(1.30, color="black", linestyle="--", label="Target X (1.30m - Unreachable)")
    ax2.set_title("Panel B: Primary Task Tracking (X Axis Reach)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("X Position [m]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)

    # Panel 3: Secondary Task (Z-axis) Tracking
    ax3 = axs[1, 0]
    for name, r in results.items():
        color, ls, lw = styles.get(name, ("black", "-", 1.5))
        ax3.plot(r["time"], r["pos"][:, 2], label=f"{name} (RMSE: {r['rmse_z']:.3f}m)", color=color, linestyle=ls, linewidth=lw)
    ax3.axhline(0.75, color="black", linestyle="--", label="Target Z (0.75m)")
    ax3.set_title("Panel C: Secondary Task Tracking (Z Axis Height)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Z Position [m]")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8)

    # Panel 4: Bar Chart - Chatter Metric vs Primary Tracking
    ax4 = axs[1, 1]
    case_labels = ["Baseline Strict", "Soft Only", "Rate Penalty Only", "Combined"]
    chatter_vals = [r["chatter"] for r in results.values()]
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax4.bar(case_labels, chatter_vals, color=bar_colors, width=0.55, edgecolor="black")
    ax4.set_title("Panel D: Actuator Chatter Metric RMS(tau_dot) [Lower is Better]", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Chatter [N·m / s]")
    ax4.grid(True, linestyle="--", alpha=0.6, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h + 200, f"{h:.0f} N·m/s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved active-set chatter mitigation figure to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Active-Set Torque Chattering Mitigation Benchmark")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation duration per trial in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Computing backend")
    parser.add_argument("--out", type=str, default="results/figures/exp_chatter_mitigation.png", help="Output figure filename")
    args = parser.parse_args()

    run_chatter_mitigation_benchmark(
        sim_time=args.time,
        dt=args.dt,
        device=args.device,
        output_path=args.out
    )


if __name__ == "__main__":
    main()
