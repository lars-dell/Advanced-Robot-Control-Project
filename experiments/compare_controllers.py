"""
Multi-Controller Comparison Benchmark Runner.

Executes all four controller paradigms under the exact same simulation environment,
physics parameters, and TaskStack priority hierarchy to benchmark and demonstrate
the properties established in:
    "Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization"
    Enrico Mingo Hoffman et al. (IEEE ICRA 2018).

Controllers evaluated:
    1. 'hierarchical_qp': Proposed Multi-Priority Cascade QP with strict equality constraints (Eq. 18)
    2. 'weighted_qp': Single-Level Weighted-Sum QP (soft weighting, priority bleed)
    3. 'saturated_algebraic': Dynamically consistent null-space projection + naive post-hoc clipping (Eq. 10)
    4. 'classical_transpose': Classical Jacobian transpose law (Eq. 9, no hierarchy, torque explosion)
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple, Sequence, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers import make_controller, list_controllers, BaseController, CANONICAL_CONTROLLERS
from experiments.exp_conflict import run_conflict
from experiments.exp_reach import run_reach
from experiments.exp_reach_split import run_reach_split
from experiments.exp11_cartesian_corridor_circle import run_experiment_11

logger = logging.getLogger("CompareControllers")


CONTROLLER_DISPLAY_NAMES = {
    "hierarchical_qp": "Proposed Hierarchical QP (Hoffman Eq. 18)",
    "weighted_qp": "Single-Level Weighted QP",
    "saturated_algebraic": "Saturated Algebraic Null-Space (Eq. 10)",
    "classical_transpose": "Classical Transpose (Eq. 9)",
}


def compute_smoothness(torques: np.ndarray, dt: float) -> float:
    """
    Computes RMS torque rate-of-change (chatter metric): sqrt(mean(||tau_dot||^2)).
    """
    if len(torques) < 2:
        return 0.0
    tau_dot = np.diff(torques, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(tau_dot ** 2, axis=1))))


def compute_metrics(
    logs: Dict[str, Any],
    dt: float = 0.005,
    scenario: str = "conflict",
    priority_order: Sequence[str] = ("x", "y", "z"),
    activity_tol: float = 0.1
) -> Dict[str, float]:
    """
    Extracts quantitative evaluation metrics from simulation logs.

    Args:
        logs: Telemetry dictionary from a scenario run.
        dt: Control timestep, used for the torque-rate chatter metric.
        scenario: Scenario name (recorded for provenance).
        priority_order: Axis ranking, highest priority first, used to attribute per-level RMSE.
        activity_tol: Distance from a torque bound, in N*m, within which that bound counts as
            active. 0.1 N*m is ~0.8% of the 12 N*m wrist limit and ~0.1% of the 87 N*m proximal
            limit, so it registers genuine saturation without firing on ordinary large torques.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    torques = logs["torques"]
    tau_min = logs["tau_min"]
    tau_max = logs["tau_max"]

    errors = np.linalg.norm(p_act - p_des, axis=1)
    rmse_total = float(np.sqrt(np.mean(errors ** 2)))
    max_err_total = float(np.max(errors))

    # Priority-specific errors
    axis_map = {"x": 0, "y": 1, "z": 2}
    p0_axis = axis_map.get(priority_order[0], 0)
    p1_axis = axis_map.get(priority_order[1], 1)
    p2_axis = axis_map.get(priority_order[2], 2)

    err_p0 = np.abs(p_act[:, p0_axis] - p_des[:, p0_axis])
    err_p1 = np.abs(p_act[:, p1_axis] - p_des[:, p1_axis])
    err_p2 = np.abs(p_act[:, p2_axis] - p_des[:, p2_axis])

    rmse_p0 = float(np.sqrt(np.mean(err_p0 ** 2)))
    rmse_p1 = float(np.sqrt(np.mean(err_p1 ** 2)))
    rmse_p2 = float(np.sqrt(np.mean(err_p2 ** 2)))

    # Torque bounds and overshoot
    overshoot_lower = np.maximum(0.0, tau_min - torques)
    overshoot_upper = np.maximum(0.0, torques - tau_max)
    max_overshoot = float(np.max(np.maximum(overshoot_lower, overshoot_upper)))

    # Constraint ACTIVITY, which is independent of constraint violation: a run can score zero
    # violations while the bounds never engage at all. Requirement 5 of the brief asks for the
    # torque constraints to *become active*, so a torque plot that never touches a bound does not
    # demonstrate it, however clean it looks. Counted as the fraction of control steps in which at
    # least one joint sits within `activity_tol` of either of its bounds.
    near_upper = torques >= (tau_max - activity_tol)
    near_lower = torques <= (tau_min + activity_tol)
    active_steps = np.any(near_upper | near_lower, axis=1)
    constraint_activity_pct = 100.0 * float(np.mean(active_steps))
    # Which joints actually saturate matters: joint 7 has a zero lever arm to the tool tip and can
    # never be driven to its bound by a Cartesian position task, so it should never appear here.
    per_joint_activity_pct = (100.0 * np.mean(near_upper | near_lower, axis=0)).tolist()

    # Strict-priority evidence (Hoffman eq. 17/18). || J_k B^-1 (tau_final - tau_k*) || should sit
    # at solver tolerance when priority is enforced as a hard equality; it grows by orders of
    # magnitude when the cascade is relaxed with a slack penalty instead. This is the only metric
    # that separates the hierarchical controller from the weighted-sum baseline.
    residuals = logs.get("priority_residuals", None)
    if residuals is not None and np.size(residuals) > 0:
        max_priority_residual = float(np.max(np.abs(residuals)))
        mean_priority_residual = float(np.mean(np.abs(residuals)))
    else:
        max_priority_residual = float("nan")
        mean_priority_residual = float("nan")

    # Actuator effort: mean ||tau||
    effort = float(np.mean(np.linalg.norm(torques, axis=1)))
    smoothness = compute_smoothness(torques, dt)

    n_violations = int(logs.get("n_violations", 0))
    n_solves = int(logs.get("n_solves", len(t)))
    violation_pct = 100.0 * n_violations / max(1, n_solves)

    return {
        "rmse_total": rmse_total,
        "max_err_total": max_err_total,
        "rmse_p0": rmse_p0,
        "rmse_p1": rmse_p1,
        "rmse_p2": rmse_p2,
        "max_overshoot": max_overshoot,
        "n_violations": n_violations,
        "violation_pct": violation_pct,
        "mean_effort": effort,
        "smoothness_chatter": smoothness,
        "constraint_activity_pct": constraint_activity_pct,
        "per_joint_activity_pct": per_joint_activity_pct,
        "max_priority_residual": max_priority_residual,
        "mean_priority_residual": mean_priority_residual,
        # A violation count is uninterpretable without the tolerance that produced it.
        "violation_tol": float(logs.get("violation_tol", float("nan"))),
    }


def print_comparison_table(
    all_metrics: Dict[str, Dict[str, float]],
    scenario: str = "conflict",
    priority_order: Sequence[str] = ("x", "y", "z")
) -> None:
    """
    Prints a formatted executive summary comparison table to stdout and logger.
    """
    width = 132
    header = (
        f"\n{'='*width}\n"
        f"  MULTI-CONTROLLER BENCHMARK COMPARISON SUMMARY: Scenario [{scenario.upper()}]  \n"
        f"  Priority Order: {' > '.join(priority_order).upper()} | Evaluation of Hoffman et al. ICRA 2018  \n"
        f"{'='*width}\n"
        f"{'Controller':<38} | {'P0 RMSE [m]':<11} | {'P1 RMSE [m]':<11} | {'Violations':<10} | "
        f"{'Max Ov [Nm]':<11} | {'Chatter [Nm/s]':<14} | {'Bnd act [%]':<11} | {'Prio resid':<10}\n"
        f"{'-'*width}"
    )
    logger.info(header)
    print(header)

    for ctrl_key, m in all_metrics.items():
        name = CONTROLLER_DISPLAY_NAMES.get(ctrl_key, ctrl_key)
        resid = m.get("max_priority_residual", float("nan"))
        resid_str = "     n/a  " if not np.isfinite(resid) else f"{resid:10.2e}"
        line = (
            f"{name:<38} | "
            f"{m['rmse_p0']:11.4f} | "
            f"{m['rmse_p1']:11.4f} | "
            f"{int(m['n_violations']):4d} ({m['violation_pct']:4.1f}%) | "
            f"{m['max_overshoot']:11.2f} | "
            f"{m['smoothness_chatter']:14.2f} | "
            f"{m.get('constraint_activity_pct', float('nan')):11.1f} | "
            f"{resid_str}"
        )
        logger.info(line)
        print(line)

    # Both added columns are easy to misread, so state what they mean rather than relying on the
    # header abbreviations.
    tol = next((m.get("violation_tol") for m in all_metrics.values()
                if np.isfinite(m.get("violation_tol", float("nan")))), float("nan"))
    notes = (
        f"{'-'*width}\n"
        f"  Bnd act [%]: share of control steps with >=1 joint within 0.1 Nm of a torque bound.\n"
        f"               Independent of 'Violations': 0 violations with 0% activity means the\n"
        f"               constraints never engaged, so requirement 5 is NOT demonstrated.\n"
        f"  Prio resid : max || J_k B^-1 (tau_final - tau_k*) ||. ~1e-14 = strict priority held\n"
        f"               exactly; orders of magnitude larger = the cascade was relaxed by slack.\n"
        f"  Violations measured at violation_tol = {tol:.1e} Nm.\n"
        f"{'='*width}\n"
    )
    logger.info(notes)
    print(notes)


def plot_comparison(
    all_logs: Dict[str, Dict[str, Any]],
    all_metrics: Dict[str, Dict[str, float]],
    scenario: str = "conflict",
    priority_order: Sequence[str] = ("x", "y", "z"),
    output_path: str = "results/compare_conflict.png"
) -> None:
    """
    Generates a publication-quality comparative figure directly contrasting the 4 controllers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    controllers = list(all_logs.keys())
    n_ctrl = len(controllers)

    # Color palette
    colors = {
        "hierarchical_qp": "#1f77b4",     # Solid Blue (Hero)
        "weighted_qp": "#ff7f0e",         # Orange
        "saturated_algebraic": "#2ca02c", # Green
        "classical_transpose": "#d62728", # Red
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    p0_name = priority_order[0].upper()
    p1_name = priority_order[1].upper()
    p2_name = priority_order[2].upper()

    fig.suptitle(
        f"Multi-Controller Comparison: Scenario '{scenario.upper()}' (Priority: {p0_name} > {p1_name} > {p2_name})\n"
        f"Evaluating Proposed Hierarchical QP (Hoffman et al. ICRA 2018) against Baseline Formulations",
        fontsize=13,
        fontweight="bold"
    )

    axis_map = {"x": 0, "y": 1, "z": 2}
    p0_idx = axis_map.get(priority_order[0], 0)
    p1_idx = axis_map.get(priority_order[1], 1)

    # Panel 1: Primary Priority Task Tracking (P0)
    ax1 = axs[0, 0]
    for ctrl_key in controllers:
        logs = all_logs[ctrl_key]
        t = logs["time"]
        p_act = logs["ee_pos"][:, p0_idx]
        p_des = logs["ee_pos_des"][:, p0_idx]
        c = colors.get(ctrl_key, "black")
        lw = 2.4 if ctrl_key == "hierarchical_qp" else 1.5
        ax1.plot(t, p_act, color=c, linewidth=lw, label=f"{ctrl_key} (RMSE: {all_metrics[ctrl_key]['rmse_p0']:.3f}m)")
    # Plot desired
    ax1.plot(t, p_des, "k--", linewidth=1.5, alpha=0.7, label=f"Desired {p0_name} (Priority 0)")
    ax1.set_title(f"Primary Task Tracking ({p0_name} Axis) - Strict Priority Protection", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(f"{p0_name} Position [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # Panel 2: Secondary Priority Task Tracking (P1)
    ax2 = axs[0, 1]
    for ctrl_key in controllers:
        logs = all_logs[ctrl_key]
        t = logs["time"]
        p_act = logs["ee_pos"][:, p1_idx]
        p_des = logs["ee_pos_des"][:, p1_idx]
        c = colors.get(ctrl_key, "black")
        lw = 2.4 if ctrl_key == "hierarchical_qp" else 1.5
        ax2.plot(t, p_act, color=c, linewidth=lw, label=f"{ctrl_key} (RMSE: {all_metrics[ctrl_key]['rmse_p1']:.3f}m)")
    ax2.plot(t, p_des, "k--", linewidth=1.5, alpha=0.7, label=f"Desired {p1_name} (Priority 1)")
    ax2.set_title(f"Secondary Task Tracking ({p1_name} Axis) - Sacrifice / Trade-off", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(f"{p1_name} Position [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    # Panel 3: Shoulder Pitch Joint (Joint 2) Torque vs Hardware Limits
    ax3 = axs[1, 0]
    j_idx = 1  # Joint 2 (Shoulder Pitch - primary torque saturation joint)
    for ctrl_key in controllers:
        logs = all_logs[ctrl_key]
        t = logs["time"]
        tau_j = logs["torques"][:, j_idx]
        c = colors.get(ctrl_key, "black")
        lw = 2.2 if ctrl_key == "hierarchical_qp" else 1.4
        ax3.plot(t, tau_j, color=c, linewidth=lw, label=f"{ctrl_key}")
    tau_min = logs["tau_min"][j_idx]
    tau_max = logs["tau_max"][j_idx]
    ax3.axhline(tau_max, color="red", linestyle=":", linewidth=2, label=f"Max Limit ({tau_max:.0f} Nm)")
    ax3.axhline(tau_min, color="red", linestyle=":", linewidth=2, label=f"Min Limit ({tau_min:.0f} Nm)")
    ax3.set_title(f"Actuator Torque Feasibility: Joint {j_idx+1} (Shoulder Pitch)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Torque [N·m]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    # Panel 4: Quantitative Benchmark Summary (Bar Chart of P0 RMSE and Violations)
    ax4 = axs[1, 1]
    ctrl_labels = [c.replace("_", "\n") for c in controllers]
    x_pos = np.arange(len(controllers))
    p0_rmses = [all_metrics[c]["rmse_p0"] for c in controllers]
    bar_colors = [colors.get(c, "gray") for c in controllers]

    bars = ax4.bar(x_pos, p0_rmses, color=bar_colors, alpha=0.85, width=0.55)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(ctrl_labels, fontsize=9)
    ax4.set_ylabel(f"{p0_name} Tracking RMSE [m] (Lower is Better)")
    ax4.set_title("Priority 0 Accuracy vs Priority Bleed / Interference", fontsize=11, fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.3)

    for bar, c in zip(bars, controllers):
        val = all_metrics[c]["rmse_p0"]
        ov = all_metrics[c]["max_overshoot"]
        note = f"{val:.3f}m"
        if ov > 0.1:
            note += f"\n(Ov: {ov:.1f}Nm)"
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            note,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved publication comparison figure to {output_path}")
    print(f"Saved publication comparison figure to: {output_path}")
    plt.close()


def run_controller_comparison(
    scenario: str = "conflict",
    controllers: Optional[Sequence[str]] = None,
    sim_time: Optional[float] = None,
    dt: float = 0.005,
    device: str = "cpu",
    priority_order: Sequence[str] = ("x", "y", "z"),
    solver: str = "daqp",
    save_plot: bool = True,
    out_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the multi-controller comparison suite across all selected controllers.

    Args:
        scenario: Scenario name ('conflict', 'corridor', 'reach', 'reach_split').
        controllers: List of controller identifiers (defaults to all 4 canonical controllers).
        sim_time: Simulation duration in seconds (scenario default if None).
        dt: Control loop timestep in seconds (default: 0.005s / 200 Hz).
        device: Physics computing device ('cpu' or 'gpu').
        priority_order: Sequence of axes for priority hierarchy (e.g. ('z', 'y', 'x') or ('x', 'y', 'z')).
        solver: QP solver backend plugin ('daqp', 'osqp', 'qpoases').
        save_plot: Whether to generate comparative figure.
        out_path: Destination path for combined telemetry .npz file.

    Returns:
        Dict[str, Any]: Mapping of controller name -> recorded telemetry and metrics.
    """
    target_controllers = list(controllers) if controllers is not None else list(CANONICAL_CONTROLLERS)

    default_times = {
        "conflict": 7.0,
        "corridor": 8.0,
        "reach": 5.0,
        "reach_split": 5.0,
    }
    duration = sim_time if sim_time is not None else default_times.get(scenario, 5.0)

    logger.info("================================================================================")
    logger.info(f"  STARTING MULTI-CONTROLLER BENCHMARK COMPARISON")
    logger.info(f"  Scenario: {scenario.upper()} | Duration: {duration:.1f}s | Device: {device}")
    logger.info(f"  Controllers to evaluate: {', '.join(target_controllers)}")
    logger.info("================================================================================")

    all_logs: Dict[str, Dict[str, Any]] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    for idx, ctrl_name in enumerate(target_controllers, 1):
        display_name = CONTROLLER_DISPLAY_NAMES.get(ctrl_name, ctrl_name)
        logger.info(f"\n>>> [{idx}/{len(target_controllers)}] Evaluating: {display_name}...")

        if scenario == "conflict":
            logs = run_conflict(
                priority_order=priority_order,
                sim_time=duration,
                dt=dt,
                show_viewer=False,
                device=device,
                out_path=None,
                show_markers=False,
                save_plot=False,
                solver=solver,
                controller=ctrl_name
            )
        elif scenario == "corridor":
            logs = run_experiment_11(
                sim_time=duration,
                dt=dt,
                show_viewer=False,
                device=device,
                out_path=None,
                show_markers=False,
                save_plot=False,
                solver=solver,
                controller=ctrl_name
            )
        elif scenario == "reach":
            logs = run_reach(
                sim_time=duration,
                dt=dt,
                show_viewer=False,
                device=device,
                out_path=None,
                show_markers=False,
                save_plot=False,
                solver=solver,
                controller=ctrl_name
            )
        elif scenario == "reach_split":
            logs = run_reach_split(
                sim_time=duration,
                dt=dt,
                show_viewer=False,
                device=device,
                out_path=None,
                show_markers=False,
                save_plot=False,
                solver=solver,
                controller=ctrl_name
            )
        else:
            raise ValueError(f"Unknown comparison scenario: '{scenario}'. Choose from: 'conflict', 'corridor', 'reach', 'reach_split'")

        all_logs[ctrl_name] = logs
        all_metrics[ctrl_name] = compute_metrics(
            logs=logs,
            dt=dt,
            scenario=scenario,
            priority_order=priority_order
        )

    # Print comparative score matrix
    print_comparison_table(all_metrics, scenario=scenario, priority_order=priority_order)

    # Output paths
    default_out_dir = "results"
    os.makedirs(default_out_dir, exist_ok=True)
    plot_file = f"{default_out_dir}/compare_{scenario}.png"
    npz_file = out_path or f"{default_out_dir}/compare_{scenario}.npz"

    if save_plot:
        plot_comparison(
            all_logs=all_logs,
            all_metrics=all_metrics,
            scenario=scenario,
            priority_order=priority_order,
            output_path=plot_file
        )

    # Save combined telemetry archive
    save_dict: Dict[str, Any] = {
        "controllers": np.array(target_controllers),
        "scenario": np.array([scenario]),
        "priority_order": np.array(list(priority_order)),
    }
    for ctrl_name, logs in all_logs.items():
        for k, v in logs.items():
            if isinstance(v, np.ndarray):
                save_dict[f"{ctrl_name}_{k}"] = v
    np.savez_compressed(npz_file, **save_dict)
    logger.info(f"Saved combined comparison telemetry to {npz_file}")
    print(f"Saved combined comparison telemetry to: {npz_file}")

    return {
        "logs": all_logs,
        "metrics": all_metrics,
        "plot_path": plot_file,
        "npz_path": npz_file,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Multi-Controller Benchmark Comparison Runner")
    parser.add_argument("--scenario", type=str, default="conflict", choices=["conflict", "corridor", "reach", "reach_split"], help="Benchmark scenario")
    parser.add_argument("--controllers", nargs="+", default=CANONICAL_CONTROLLERS, help="Controllers to evaluate")
    parser.add_argument("--time", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--priority-order", type=str, default="xyz", help="Priority order of axes (e.g. 'xyz' or 'zyx')")
    parser.add_argument("--solver", type=str, default="daqp", help="QP solver backend plugin")
    parser.add_argument("--out", type=str, default=None, help="Output .npz path")
    args = parser.parse_args()

    order = tuple(args.priority_order)
    run_controller_comparison(
        scenario=args.scenario,
        controllers=args.controllers,
        sim_time=args.time,
        dt=args.dt,
        device=args.device,
        priority_order=order,
        solver=args.solver,
        save_plot=True,
        out_path=args.out
    )


if __name__ == "__main__":
    main()
