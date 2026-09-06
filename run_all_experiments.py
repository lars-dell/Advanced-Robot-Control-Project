"""
Master Test Runner for the complete Multi-Priority QP Impedance Control Evaluation Suite.

Executes all 10 experiment benchmarks and real-time solver latency profiling sequentially,
generating all diagnostic plots and printing an executive summary table.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

from experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
    run_experiment_6,
    run_experiment_7,
    run_experiment_8,
    run_experiment_9,
    run_experiment_10,
    run_experiment_11,
    run_latency_benchmark,
)

logger = logging.getLogger("RunAllExperiments")


def run_experiment_sub(script_name: str, extra_args: List[str]) -> None:
    """
    Executes a single experiment benchmark in an isolated Python subprocess.
    Ensures a clean LLVM JIT / PyTorch runtime environment per simulation benchmark.
    """
    script_path = os.path.join(os.path.dirname(__file__), "experiments", script_name)
    cmd = [sys.executable, script_path] + extra_args
    subprocess.run(cmd, check=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Master Test Runner for Multi-Priority QP Impedance Evaluation Suite")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--quick", action="store_true", help="Run short versions of experiments for fast verification")
    parser.add_argument("--inprocess", action="store_true", help="Run all experiments inside the same Python process instead of isolated subprocesses")
    parser.add_argument("--vis", action="store_true", help="Show the 3D viewer for each experiment. The suite then runs one window at a time and is much slower; intended for watching behaviour, not for regenerating figures")
    args = parser.parse_args()

    device = args.device
    scale = 0.5 if args.quick else 1.0
    # In-process calls take a boolean; subprocess calls take the flag, so keep both forms.
    show_viewer = args.vis
    vis_args = [] if args.vis else ["--no-vis"]

    logger.info("================================================================================")
    logger.info("  STARTING FULL EVALUATION SUITE: MULTI-PRIORITY CARTESIAN IMPEDANCE CONTROL    ")
    logger.info("================================================================================")
    t_suite_start = time.perf_counter()

    steps = 500 if args.quick else 2000

    if args.inprocess:
        # In-process execution mode
        logger.info("\n>>> [1/11] Running Exp 1: Surface Circle & Force Exertion...")
        run_experiment_1(sim_time=8.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [2/11] Running Exp 2: Blocked Circle & Obstacle Compliance...")
        run_experiment_2(sim_time=8.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [3/11] Running Exp 3: Reactive APF Obstacle Avoidance...")
        run_experiment_3(sim_time=6.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [4/11] Running Exp 4: Multi-Link Disturbance Rejection...")
        run_experiment_4(sim_time=8.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [5/11] Running Exp 5: Torque-Constrained Surface Wiping...")
        run_experiment_5(sim_time=8.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [6/11] Running Exp 6: Kinematic Singularity Tracking...")
        run_experiment_6(sim_time=6.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [7/11] Running Exp 7: 4-Way Baseline Comparison (Paper Reproduction)...")
        run_experiment_7(sim_time=5.0 * scale, dt=0.005, device=device, save_plot=True)

        logger.info("\n>>> [8/11] Running Exp 8: Frequency Response & Bode Bandwidth...")
        freqs = [0.5, 1.0, 2.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 3.0]
        run_experiment_8(frequencies=freqs, sim_time=5.0 * scale, dt=0.005, device=device, save_plot=True)

        logger.info("\n>>> [9/11] Running Exp 9: Energy Tank & Passivity Profiling...")
        run_experiment_9(sim_time=6.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [10/12] Running Exp 10: Model Parameter Robustness...")
        run_experiment_10(sim_time=6.0 * scale, dt=0.005, device=device, save_plot=True)

        logger.info("\n>>> [11/12] Running Exp 11: Prioritized Z-Corridor & Out-of-Bounds Circle...")
        run_experiment_11(sim_time=8.0 * scale, dt=0.005, show_viewer=show_viewer, device=device, save_plot=True)

        logger.info("\n>>> [12/12] Running Benchmark: Real-Time Solver Latency Profiling...")
        run_latency_benchmark(n_steps=steps, dt=0.005, device=device, save_plot=True)
    else:
        # Isolated subprocess mode (default): completely isolates Genesis/LLVM runtimes
        logger.info("\n>>> [1/11] Running Exp 1: Surface Circle & Force Exertion...")
        run_experiment_sub("exp1_surface_circle.py", ["--time", str(8.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [2/11] Running Exp 2: Blocked Circle & Obstacle Compliance...")
        run_experiment_sub("exp2_blocked_circle.py", ["--time", str(8.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [3/11] Running Exp 3: Reactive APF Obstacle Avoidance...")
        run_experiment_sub("exp3_apf_avoidance.py", ["--time", str(6.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [4/11] Running Exp 4: Multi-Link Disturbance Rejection...")
        run_experiment_sub("exp4_multilink_push.py", ["--time", str(8.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [5/11] Running Exp 5: Torque-Constrained Surface Wiping...")
        run_experiment_sub("exp5_torque_constrained_wipe.py", ["--time", str(8.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [6/11] Running Exp 6: Kinematic Singularity Tracking...")
        run_experiment_sub("exp6_singularity_tracking.py", ["--time", str(6.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [7/11] Running Exp 7: 4-Way Baseline Comparison (Paper Reproduction)...")
        run_experiment_sub("exp7_baseline_comparison.py", ["--time", str(5.0 * scale), "--device", device])

        logger.info("\n>>> [8/11] Running Exp 8: Frequency Response & Bode Bandwidth...")
        exp8_args = (["--quick"] if args.quick else []) + ["--time", str(5.0 * scale), "--device", device]
        run_experiment_sub("exp8_frequency_bode_analysis.py", exp8_args)

        logger.info("\n>>> [9/11] Running Exp 9: Energy Tank & Passivity Profiling...")
        run_experiment_sub("exp9_passivity_energy_profiling.py", ["--time", str(6.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [10/12] Running Exp 10: Model Parameter Robustness...")
        run_experiment_sub("exp10_parameter_robustness.py", ["--time", str(6.0 * scale), "--device", device])

        logger.info("\n>>> [11/12] Running Exp 11: Prioritized Z-Corridor & Out-of-Bounds Circle...")
        run_experiment_sub("exp11_cartesian_corridor_circle.py", ["--time", str(8.0 * scale), "--device", device] + vis_args)

        logger.info("\n>>> [12/12] Running Benchmark: Real-Time Solver Latency Profiling...")
        run_experiment_sub("benchmark_solver_latency.py", ["--steps", str(steps), "--device", device])

    t_suite_total = time.perf_counter() - t_suite_start
    logger.info("\n================================================================================")
    logger.info(f"  ALL 12 EVALUATION EXPERIMENTS COMPLETED IN {t_suite_total:.1f}s")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
