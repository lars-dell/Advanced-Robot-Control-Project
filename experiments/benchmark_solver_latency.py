"""
Benchmark: Real-Time Computational Efficiency & Solver Latency Profiling ("Handiness").

Quantifies execution latency across thousands of control steps:
    - Level 0 QP Solve Time (Parametric qpOASES / OSQP, 7 decision variables)
    - Level 1 QP Solve Time (Constrained with Level 0 Priority Equalities)
    - Total Cascade Controller Latency per timestep (2-level hierarchy)
    - Classical Pseudo-Inverse J_bar Computation Time

Computes Mean, Median, 95th/99th Percentile Latency, Max Jitter, and Real-Time Feasibility Margin.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.qp_impedance import QPImpedanceController
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, CircularTrajectoryGenerator, JointPostureTask, TaskStack
from utils.math_utils import dynamically_consistent_pinv

logger = logging.getLogger("BenchmarkLatency")


def run_latency_benchmark(
    n_steps: int = 2000,
    dt: float = 0.005,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Executes high-precision timing benchmarks across simulation steps.
    """
    logger.info("=================================================================")
    logger.info(f"Running Real-Time Solver Latency Benchmark ({n_steps} steps)")
    logger.info("=================================================================")

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device
    )

    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=500.0,
        kd_cart=45.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    circle_center = p_init + np.array([0.05, 0.0, -0.02])
    traj_gen = CircularTrajectoryGenerator(center=circle_center, radius=0.06, period=3.0, plane="xy")

    task_stack = TaskStack()
    cart_task = CartesianPoseTask(name="cart_p0", priority=0, kp=500.0, kd=45.0, mode="3d", trajectory_fn=traj_gen)
    task_stack.add_task(cart_task)

    q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(name="posture_p1", priority=1, kp=20.0, kd=4.0, q_des=q_null)
    task_stack.add_task(posture_task)

    # Warmup runs to ensure JIT/cache compilation
    logger.info("Executing solver warmup steps...")
    for _ in range(50):
        s = sim.get_state()
        p_des, _, v_des = traj_gen(0.0)
        s["target_pos"] = p_des
        s["target_vel"] = v_des
        _ = controller.compute_torques(state=s, target=task_stack, t=0.0)
        sim.step()

    # Timing storage arrays (in microseconds)
    times_lvl0_us: List[float] = []
    times_lvl1_us: List[float] = []
    times_total_ctrl_us: List[float] = []
    times_pinv_us: List[float] = []

    logger.info(f"Benchmarking {n_steps} active control steps...")

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        p_des, _, v_des = traj_gen(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # 1. Measure Classical Pseudo-Inverse Computation J_bar
        t_pinv_start = time.perf_counter()
        _ = dynamically_consistent_pinv(state["J"][:3, :], state["B"])
        t_pinv_end = time.perf_counter()
        times_pinv_us.append((t_pinv_end - t_pinv_start) * 1e6)

        # 2. Measure Component QP Solves inside Cascade
        B_inv = np.linalg.inv(state["B"])
        h = state["h"]
        lb = controller.tau_min - h
        ub = controller.tau_max - h

        # Level 0 QP
        J0, f0 = cart_task.compute(state, t_curr)
        M0 = J0 @ B_inv
        H0 = M0.T @ M0 + controller.reg_eps * np.eye(7)
        g0 = - M0.T @ (M0 @ J0.T @ f0)

        t_l0_start = time.perf_counter()
        tau_0_opt = controller.qp_solver.solve(H=H0, g=g0, lb=lb, ub=ub)
        t_l0_end = time.perf_counter()
        times_lvl0_us.append((t_l0_end - t_l0_start) * 1e6)

        # Level 1 QP
        J1, f1 = posture_task.compute(state, t_curr)
        H1 = (1.0 + controller.reg_eps) * np.eye(7)
        g1 = - np.asarray(f1, dtype=np.float64).flatten()
        A_eq = M0
        b_eq = M0 @ tau_0_opt

        t_l1_start = time.perf_counter()
        tau_1_opt = controller.qp_solver.solve(H=H1, g=g1, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq)
        t_l1_end = time.perf_counter()
        times_lvl1_us.append((t_l1_end - t_l1_start) * 1e6)

        # 3. Measure Full Controller Function compute_torques()
        t_full_start = time.perf_counter()
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)
        t_full_end = time.perf_counter()
        times_total_ctrl_us.append((t_full_end - t_full_start) * 1e6)

        sim.apply_torques(torques)
        sim.step()

    # Convert to numpy arrays
    l0_arr = np.array(times_lvl0_us)
    l1_arr = np.array(times_lvl1_us)
    tot_arr = np.array(times_total_ctrl_us)
    pinv_arr = np.array(times_pinv_us)

    metrics = {
        "Level 0 QP (Bound-Constrained)": {
            "mean": float(np.mean(l0_arr)),
            "median": float(np.median(l0_arr)),
            "p95": float(np.percentile(l0_arr, 95)),
            "p99": float(np.percentile(l0_arr, 99)),
            "max": float(np.max(l0_arr)),
            "std": float(np.std(l0_arr)),
        },
        "Level 1 QP (Equality-Constrained)": {
            "mean": float(np.mean(l1_arr)),
            "median": float(np.median(l1_arr)),
            "p95": float(np.percentile(l1_arr, 95)),
            "p99": float(np.percentile(l1_arr, 99)),
            "max": float(np.max(l1_arr)),
            "std": float(np.std(l1_arr)),
        },
        "Total Cascade QP Controller": {
            "mean": float(np.mean(tot_arr)),
            "median": float(np.median(tot_arr)),
            "p95": float(np.percentile(tot_arr, 95)),
            "p99": float(np.percentile(tot_arr, 99)),
            "max": float(np.max(tot_arr)),
            "std": float(np.std(tot_arr)),
        },
        "Classical Pseudo-Inverse J_bar": {
            "mean": float(np.mean(pinv_arr)),
            "median": float(np.median(pinv_arr)),
            "p95": float(np.percentile(pinv_arr, 95)),
            "p99": float(np.percentile(pinv_arr, 99)),
            "max": float(np.max(pinv_arr)),
            "std": float(np.std(pinv_arr)),
        }
    }

    # Print summary table
    logger.info("-----------------------------------------------------------------------------------------")
    logger.info(f"{'Module / Routine':<38} | {'Mean (µs)':<10} | {'Median':<8} | {'P95':<8} | {'P99':<8} | {'Max (µs)':<8}")
    logger.info("-----------------------------------------------------------------------------------------")
    for name, m in metrics.items():
        logger.info(f"{name:<38} | {m['mean']:8.2f} µs | {m['median']:6.2f} µs | {m['p95']:6.2f} µs | {m['p99']:6.2f} µs | {m['max']:6.2f} µs")
    logger.info("-----------------------------------------------------------------------------------------")

    dt_us = dt * 1e6
    max_ctrl_us = metrics["Total Cascade QP Controller"]["max"]
    margin = (dt_us - max_ctrl_us) / dt_us * 100.0
    logger.info(f"Control Loop Timestep: {dt*1000:.1f} ms ({dt_us:.0f} µs)")
    logger.info(f"Max Measured Latency:  {max_ctrl_us:.2f} µs | Real-Time Headroom Margin: {margin:.1f}%")
    logger.info(f"Theoretical Max Frequency: {1e6 / metrics['Total Cascade QP Controller']['mean']:.0f} Hz (Mean)")

    benchmark_data = {
        "metrics": metrics,
        "l0_us": l0_arr,
        "l1_us": l1_arr,
        "total_us": tot_arr,
        "pinv_us": pinv_arr,
        "dt_us": dt_us
    }

    if save_plot:
        plot_benchmark_latency(benchmark_data)

    return benchmark_data


def plot_benchmark_latency(
    data: Dict[str, Any],
    output_path: str = "results/figures/benchmark_solver_latency.png"
) -> None:
    """
    Generates diagnostic histograms and latency distribution plots.
    """
    l0 = data["l0_us"]
    l1 = data["l1_us"]
    tot = data["total_us"]
    pinv = data["pinv_us"]
    m = data["metrics"]

    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Computational Efficiency & Real-Time Solver Latency Profiling", fontsize=14, fontweight="bold")

    # Panel 1: Histogram of Level 0 vs Level 1 QP Solve Times
    ax1 = axs[0, 0]
    ax1.hist(l0, bins=40, alpha=0.6, color="blue", label=f"Level 0 QP (Mean: {m['Level 0 QP (Bound-Constrained)']['mean']:.1f} µs)")
    ax1.hist(l1, bins=40, alpha=0.6, color="green", label=f"Level 1 QP (Mean: {m['Level 1 QP (Equality-Constrained)']['mean']:.1f} µs)")
    ax1.set_xlabel("Solve Latency [µs]")
    ax1.set_ylabel("Step Frequency Count")
    ax1.set_title("1. QP Solver Execution Latency Distribution")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: Total Cascade Controller Timestep Latency
    ax2 = axs[0, 1]
    ax2.plot(tot, color="navy", alpha=0.75, label="Total Controller Latency per Step")
    ax2.axhline(m["Total Cascade QP Controller"]["mean"], color="red", linestyle="-", label=f"Mean: {m['Total Cascade QP Controller']['mean']:.1f} µs")
    ax2.axhline(m["Total Cascade QP Controller"]["p99"], color="orange", linestyle="--", label=f"99th Percentile: {m['Total Cascade QP Controller']['p99']:.1f} µs")
    ax2.axhline(data["dt_us"], color="black", linestyle=":", label=f"Control Budget dt: {data['dt_us']:.0f} µs")
    ax2.set_xlabel("Simulation Step Index")
    ax2.set_ylabel("Execution Time [µs]")
    ax2.set_title("2. Step-by-Step Execution Time & Jitter Profile")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: Empirical Cumulative Distribution Function (ECDF)
    ax3 = axs[1, 0]
    sorted_tot = np.sort(tot)
    ecdf = np.arange(1, len(sorted_tot) + 1) / len(sorted_tot)
    ax3.plot(sorted_tot, ecdf, "b-", linewidth=2, label="Cascade Controller ECDF")
    ax3.axvline(m["Total Cascade QP Controller"]["p95"], color="orange", linestyle="--", label=f"P95: {m['Total Cascade QP Controller']['p95']:.1f} µs")
    ax3.axvline(m["Total Cascade QP Controller"]["p99"], color="red", linestyle="--", label=f"P99: {m['Total Cascade QP Controller']['p99']:.1f} µs")
    ax3.set_xlabel("Execution Time [µs]")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("3. Latency CDF & Determinism (99% Guarantee)")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Benchmark Comparison Bar Chart
    ax4 = axs[1, 1]
    labels = ["Lvl 0 QP", "Lvl 1 QP", "Total Cascade", "Classical J_bar"]
    means = [
        m["Level 0 QP (Bound-Constrained)"]["mean"],
        m["Level 1 QP (Equality-Constrained)"]["mean"],
        m["Total Cascade QP Controller"]["mean"],
        m["Classical Pseudo-Inverse J_bar"]["mean"]
    ]
    p99s = [
        m["Level 0 QP (Bound-Constrained)"]["p99"],
        m["Level 1 QP (Equality-Constrained)"]["p99"],
        m["Total Cascade QP Controller"]["p99"],
        m["Classical Pseudo-Inverse J_bar"]["p99"]
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax4.bar(x - width/2, means, width, label="Mean Latency (µs)", color="cornflowerblue")
    ax4.bar(x + width/2, p99s, width, label="99th Percentile (µs)", color="darkorange")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel("Time [µs]")
    ax4.set_title("4. Comparative Computational Cost Summary")
    ax4.grid(True, axis="y")
    ax4.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved benchmark latency plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark: Real-Time Solver Latency Profiling")
    parser.add_argument("--steps", type=int, default=2000, help="Number of benchmark steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_latency_benchmark(
        n_steps=args.steps,
        dt=args.dt,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
