"""
Main CLI entry point for Multi-Priority Cartesian Impedance Control (Genesis Simulation).

Provides command-line dispatching via --experiment / --exp for:
    - Priority Hierarchy Experiments: 'reach', 'reach_split', 'conflict'
    - Benchmark Evaluation Experiments: 'surface_circle', 'blocked_circle', 'apf_avoidance',
      'multilink_push', 'torque_wipe', 'singularity', 'baseline_comparison', 'bode',
      'passivity', 'robustness', 'benchmark', 'all'
"""

import argparse
import logging
import sys
import subprocess
from typing import Optional

from experiments import (
    run_reach,
    run_reach_split,
    run_conflict,
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

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and dispatch the selected scenario or benchmark experiment."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Multi-Priority Cartesian Impedance Control (QP in Genesis) - Scenarios & Benchmarks"
    )
    parser.add_argument("--time", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds (default: 0.005s / 200 Hz)")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode without 3D viewer GUI")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend device")
    parser.add_argument("--out", type=str, default=None, help="Destination file path for telemetry .npz")
    parser.add_argument("--no-markers", action="store_true", help="Disable visual goal/error/trail overlays")
    parser.add_argument("--record", type=str, default=None, help="Record simulation run to video/GIF (e.g. docs/media/run.gif)")
    parser.add_argument(
        "--priority-order",
        type=str,
        default="xyz",
        help="Priority ranking of Cartesian axes in 'conflict' scenario (e.g. 'xyz' or 'zyx')"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="daqp",
        choices=["daqp", "osqp", "qpoases"],
        help="QP solver backend engine plugin (default: 'daqp', choices: 'daqp', 'osqp', 'qpoases')"
    )
    parser.add_argument(
        "--experiment",
        "--exp",
        type=str,
        default="reach",
        choices=[
            "reach",
            "reach_split",
            "conflict",
            "surface_circle",
            "blocked_circle",
            "apf_avoidance",
            "multilink_push",
            "torque_wipe",
            "singularity",
            "baseline_comparison",
            "bode",
            "passivity",
            "robustness",
            "benchmark",
            "corridor",
            "z_bounds_circle",
            "exp11",
            "all",
        ],
        help="Experiment or priority scenario selection (alias: --exp, default: 'reach')",
    )
    parser.add_argument("--z-min", type=float, default=0.35, help="Lower height limit in meters (default: 0.35)")
    parser.add_argument("--z-max", type=float, default=0.55, help="Upper height limit in meters (default: 0.55)")
    parser.add_argument(
        "--scenario",
        dest="experiment",
        choices=["reach", "reach_split", "conflict"],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    show_viewer = not args.no_vis
    show_markers = not args.no_markers

    # Dispatch experiment or priority hierarchy scenario
    exp = args.experiment

    if exp == "reach":
        sim_time = args.time if args.time is not None else 5.0
        out_path = args.out or "results/reach.npz"
        run_reach(
            sim_time=sim_time,
            dt=args.dt,
            show_viewer=show_viewer,
            device=args.device,
            out_path=out_path,
            show_markers=show_markers,
            record_path=args.record,
            solver=args.solver,
        )
    elif exp == "reach_split":
        sim_time = args.time if args.time is not None else 5.0
        out_path = args.out or "results/reach_split.npz"
        run_reach_split(
            sim_time=sim_time,
            dt=args.dt,
            show_viewer=show_viewer,
            device=args.device,
            out_path=out_path,
            show_markers=show_markers,
            record_path=args.record,
            solver=args.solver,
        )
    elif exp == "conflict":
        sim_time = args.time if args.time is not None else 7.0
        order = tuple(args.priority_order)
        out_path = args.out or f"results/conflict_{''.join(order)}.npz"
        run_conflict(
            priority_order=order,
            sim_time=sim_time,
            dt=args.dt,
            show_viewer=show_viewer,
            device=args.device,
            out_path=out_path,
            show_markers=show_markers,
            record_path=args.record,
            solver=args.solver,
        )
    elif exp == "surface_circle":
        run_experiment_1(sim_time=args.time or 8.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "blocked_circle":
        run_experiment_2(sim_time=args.time or 8.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "apf_avoidance":
        run_experiment_3(sim_time=args.time or 6.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "multilink_push":
        run_experiment_4(sim_time=args.time or 8.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "torque_wipe":
        run_experiment_5(sim_time=args.time or 8.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "singularity":
        run_experiment_6(sim_time=args.time or 6.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "baseline_comparison":
        sim_time = args.time if args.time is not None else 5.0
        run_experiment_7(sim_time=sim_time, dt=args.dt, device=args.device)
    elif exp == "bode":
        sim_time = args.time if args.time is not None else 5.0
        run_experiment_8(sim_time=sim_time, dt=args.dt, device=args.device)
    elif exp == "passivity":
        run_experiment_9(sim_time=args.time or 6.0, dt=args.dt, show_viewer=show_viewer, device=args.device)
    elif exp == "robustness":
        run_experiment_10(sim_time=args.time or 6.0, dt=args.dt, device=args.device)
    elif exp == "benchmark":
        run_latency_benchmark(dt=args.dt, device=args.device)
    elif exp in ("corridor", "z_bounds_circle", "exp11"):
        sim_time = args.time if args.time is not None else 8.0
        out_path = args.out or "results/exp11_corridor_circle.npz"
        run_experiment_11(
            sim_time=sim_time,
            dt=args.dt,
            show_viewer=show_viewer,
            device=args.device,
            z_min=args.z_min,
            z_max=args.z_max,
            out_path=out_path,
            show_markers=show_markers,
            record_path=args.record,
            solver=args.solver,
        )
    elif exp == "all":
        cmd = [sys.executable, "run_all_experiments.py", "--device", args.device]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
