"""
Master Test Runner for the complete Multi-Priority QP Impedance Control Evaluation Suite.

Executes all 10 experiment benchmarks and real-time solver latency profiling sequentially,
generating all diagnostic plots and printing an executive summary table.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any

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
    run_latency_benchmark,
)

logger = logging.getLogger("RunAllExperiments")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Master Test Runner for Multi-Priority QP Impedance Evaluation Suite")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--quick", action="store_true", help="Run short versions of experiments for fast verification")
    args = parser.parse_args()

    device = args.device
    scale = 0.5 if args.quick else 1.0

    logger.info("================================================================================")
    logger.info("  STARTING FULL EVALUATION SUITE: MULTI-PRIORITY CARTESIAN IMPEDANCE CONTROL    ")
    logger.info("================================================================================")
    t_suite_start = time.perf_counter()

    # Exp 1: Surface Circle + Normal Force
    logger.info("\n>>> [1/11] Running Exp 1: Surface Circle & Force Exertion...")
    run_experiment_1(sim_time=8.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 2: Blocked Circle & Obstacle Compliance
    logger.info("\n>>> [2/11] Running Exp 2: Blocked Circle & Obstacle Compliance...")
    run_experiment_2(sim_time=8.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 3: Reactive APF Avoidance
    logger.info("\n>>> [3/11] Running Exp 3: Reactive APF Obstacle Avoidance...")
    run_experiment_3(sim_time=6.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 4: Multi-Link Disturbance Rejection (Hierarchy Verification)
    logger.info("\n>>> [4/11] Running Exp 4: Multi-Link Disturbance Rejection...")
    run_experiment_4(sim_time=8.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 5: Torque Constrained Wiping (Active Inequality Bounds)
    logger.info("\n>>> [5/11] Running Exp 5: Torque-Constrained Surface Wiping...")
    run_experiment_5(sim_time=8.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 6: Kinematic Singularity Tracking
    logger.info("\n>>> [6/11] Running Exp 6: Kinematic Singularity Tracking...")
    run_experiment_6(sim_time=6.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 7: 4-Way Baseline Comparison (Hoffman et al. Section V-A Reproduction)
    logger.info("\n>>> [7/11] Running Exp 7: 4-Way Baseline Comparison (Paper Reproduction)...")
    run_experiment_7(sim_time=5.0 * scale, dt=0.005, device=device, save_plot=True)

    # Exp 8: Frequency Response & Bode Analysis
    logger.info("\n>>> [8/11] Running Exp 8: Frequency Response & Bode Bandwidth...")
    freqs = [0.5, 1.0, 2.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 3.0]
    run_experiment_8(frequencies=freqs, sim_time=5.0 * scale, dt=0.005, device=device, save_plot=True)

    # Exp 9: Energy Tank & Passivity Profiling
    logger.info("\n>>> [9/11] Running Exp 9: Energy Tank & Passivity Profiling...")
    run_experiment_9(sim_time=6.0 * scale, dt=0.005, show_viewer=False, device=device, save_plot=True)

    # Exp 10: Model Parameter Uncertainty & Robustness
    logger.info("\n>>> [10/11] Running Exp 10: Model Parameter Robustness...")
    run_experiment_10(sim_time=6.0 * scale, dt=0.005, device=device, save_plot=True)

    # Exp 11: Real-Time Solver Latency Benchmark
    logger.info("\n>>> [11/11] Running Benchmark: Real-Time Solver Latency Profiling...")
    steps = 500 if args.quick else 2000
    run_latency_benchmark(n_steps=steps, dt=0.005, device=device, save_plot=True)

    t_suite_total = time.perf_counter() - t_suite_start
    logger.info("\n================================================================================")
    logger.info(f"  ALL 11 EVALUATION EXPERIMENTS COMPLETED IN {t_suite_total:.1f}s")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
