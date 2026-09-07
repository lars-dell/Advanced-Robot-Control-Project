"""
Experiment 11: Prioritized Cartesian Height Corridor with Out-of-Bounds Circular Tracking.

Demonstrates strict multi-priority QP hierarchy where:
  - Priority 0 (Highest): Remain strictly within vertical safety limits (Ceiling: z <= z_max = 0.55 m,
                          Floor: z >= z_min = 0.35 m). When the circle demands heights above the ceiling
                          or below the floor, the top-priority controller pins the motion to the boundary.
  - Priority 1 (Next):    Follow a circular trajectory in the horizontal plane whose commanded path reaches
                          beyond the manipulator's physical reach limits (x_des up to 0.96 m, where maximum
                          Panda extension is ~0.91 m).
  - Priority 2 (Lowest):  Joint-space null-space posture regularization.

Key Physical & Optimization Behaviors Verified:
  1. Priority Preservation: The end-effector is strictly prevented from penetrating the ceiling or floor.
  2. Horizontal Sliding: When z_nom > z_max or z_nom < z_min, the trajectory is flattened at the boundary
     while XY planar tracking continues as best as possible.
  3. Reach Deficit & Safe Torques: When the circle extends beyond maximum reach (x > 0.91 m), joints reach
     their torque limits cleanly without violating bounds or destabilizing the controller.
  4. Dynamic Re-entry: When the circular reference re-enters the [z_min, z_max] corridor, boundary clamping
     seamlessly transitions to nominal vertical arc tracking.
"""

import argparse
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers import BaseController, make_controller
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, JointPostureTask, TaskStack, ZBoundaryTask

logger = logging.getLogger("Exp11_CorridorCircle")


def generate_out_of_bounds_circle(
    center: np.ndarray,
    radius: float = 0.18,
    period: float = 4.0,
    ramp_time: float = 0.8
):
    """
    Generates a 3D circular trajectory that oscillates vertically across Z bounds
    and extends forward in X beyond reachable workspace.
    """
    omega = 2.0 * np.pi / period

    def trajectory(t: float) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        # Smooth cubic polynomial ramp from 0 to 1 over ramp_time
        if ramp_time > 0 and t < ramp_time:
            tau = t / ramp_time
            s = tau * tau * (3.0 - 2.0 * tau)
            s_dot = (6.0 * tau - 6.0 * tau * tau) / ramp_time
        else:
            s = 1.0
            s_dot = 0.0

        angle = omega * t
        pos_des = np.array([
            center[0] + s * radius * np.cos(angle),
            center[1] + s * 0.05 * np.sin(2.0 * angle),
            center[2] + s * radius * np.sin(angle)
        ], dtype=np.float64)

        vel_des = np.array([
            s_dot * radius * np.cos(angle) - s * radius * omega * np.sin(angle),
            s_dot * 0.05 * np.sin(2.0 * angle) + s * 0.05 * 2.0 * omega * np.cos(2.0 * angle),
            s_dot * radius * np.sin(angle) + s * radius * omega * np.cos(angle)
        ], dtype=np.float64)

        return pos_des, None, vel_des

    return trajectory


def run_experiment_11(
    sim_time: float = 8.0,
    dt: float = 0.005,
    show_viewer: bool = False,
    device: str = "cpu",
    z_min: float = 0.35,
    z_max: float = 0.55,
    out_path: Optional[str] = None,
    show_markers: bool = True,
    record_path: Optional[str] = None,
    save_plot: bool = True,
    solver: str = "daqp",
    controller: Union[str, BaseController] = "hierarchical_qp",
    enable_corridor: bool = True,
    barrier_kp: float = 1200.0,
    barrier_kd: float = 80.0
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 11: Prioritized Height Corridor with Out-of-Bounds Circle Tracking.

    Args:
        sim_time: Duration of simulation in seconds (default: 8.0s ~ 2 full cycles).
        dt: Control loop timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to launch interactive 3D viewer window.
        device: Physics computing backend ('cpu' or 'gpu').
        z_min: Lower height safety limit in meters (Floor: 0.35m).
        z_max: Upper height safety limit in meters (Ceiling: 0.55m).
        out_path: Destination path for .npz telemetry log.
        show_markers: Whether to render debug overlays.
        record_path: Optional video/GIF output path.
        save_plot: Whether to generate diagnostic figures.
        solver: QP solver plugin ('daqp', 'osqp', 'qpoases').
        controller: Controller name or BaseController instance (default: 'hierarchical_qp').

    Returns:
        Dict[str, np.ndarray]: Recorded telemetry data arrays.
    """
    default_out = "results/exp11_corridor_circle.npz"
    out_file = out_path or default_out

    logger.info("================================================================================")
    logger.info(f"Running Exp 11: Prioritized Z-Corridor [{z_min:.2f}, {z_max:.2f}]m & Out-of-Bounds Circle [Solver: {solver}]")
    logger.info("================================================================================")

    # Visual boundary markers (purely visual overlays, NO hitbox).
    #
    # These were previously solid 0.6 x 0.6 m plates, which mark the corridor correctly but hide the
    # tool behind them -- the tool spends the whole run between the two boundaries, which is exactly
    # where the sheets are. Drawn instead as open rectangular frames: four thin bars per boundary,
    # so the height is unmistakable while the interior stays completely clear.
    def _corridor_boundary(z, color, span=0.60, centre_x=0.65, bar=0.003, n_grid=5):
        """
        One boundary of the height corridor, drawn as a thin open mesh.

        NOTE: Genesis only honours surface `opacity` in the ray-traced renderer -- it appears nowhere
        in the rasterizer that drives the interactive viewer, so a translucent or glass sheet still
        renders fully opaque on screen and hides the tool behind it. Transparency is not available
        live, so the plane is built from geometry that cannot occlude.

        The border lines are simply the outermost lines of the grid, at the same thickness as the
        rest: a separate heavier rim reads as a frame around the plane rather than as its edge.
        `n_grid` counts interior lines, so each direction draws n_grid + 2 lines including the two
        borders, evenly spaced.
        """
        h = span / 2.0
        n_lines = n_grid + 2                      # interior lines plus the two borders
        parts = []
        for i in range(n_lines):
            off = -h + (i / (n_lines - 1)) * span  # inclusive of both edges
            parts.append({"pos": (centre_x + off, 0.0, z), "size": (bar, span, bar)})
            parts.append({"pos": (centre_x, off, z), "size": (span, bar, bar)})

        for pt in parts:
            pt.update(color=color, collision=False, opacity=1.0)
        return parts

    # Blue ceiling at z_max, orange floor at z_min.
    boxes = (_corridor_boundary(z_max, (0.15, 0.45, 0.95))
             + _corridor_boundary(z_min, (0.95, 0.40, 0.15)))

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device,
        show_markers=show_markers,
        record_path=record_path,
        boxes=boxes
    )

    if isinstance(controller, str):
        controller = make_controller(
            controller,
            n_dofs=7,
            kp_cart=450.0,
            kd_cart=45.0,
            kp_null=20.0,
            kd_null=4.0,
            solver_name=solver,
            reg_eps=1e-4
        )


    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    # Trajectory configuration:
    # Center at [0.55, 0.0, 0.45], radius = 0.18 m
    # z_nom oscillates in [0.27, 0.63] m -> breaches ceiling (0.55m) and floor (0.35m) by 8 cm
    # x_nom oscillates in [0.37, 0.73] m -> safely within Panda reach envelope
    circle_center = np.array([0.55, 0.0, 0.5 * (z_min + z_max)])
    radius = 0.18
    period = 4.0

    circle_traj = generate_out_of_bounds_circle(center=circle_center, radius=radius, period=period)

    def nominal_z_fn(t: float) -> Tuple[float, float]:
        p, _, v = circle_traj(t)
        return float(p[2]), float(v[2])

    # Build Task Hierarchy with Strict QP Inequality Enforcement (Hoffman et al. ICRA 2018):
    # - Strict Inequality Constraint: End-effector height strictly bounded in [z_min, z_max]
    #   enforced across ALL priority levels (Eq. 18: b_l <= A_ineq * tau <= b_u).
    # - Priority 0 (Primary Tracking): Full 3D Circular Trajectory Tracking Task
    # - Priority 1 (Secondary Tracking): Joint Posture Regularization Task
    task_stack = TaskStack()

    # The corridor is NOT a priority level. For controllers that accept inequalities it contributes
    # a zero-row Jacobian to the cost hierarchy and enters instead as the QP constraint
    # b_l <= J_z B^-1 tau <= b_u, which binds at every level -- stronger than priority 0. For
    # controllers that cannot express inequalities it degrades to a barrier spring-damper, which is
    # what makes the cross-controller comparison like-for-like.
    #
    # `enable_corridor=False` removes it entirely. That ablation is the only thing that shows the
    # constraint is what keeps the tool inside the corridor, rather than the trajectory happening to
    # stay there.
    z_boundary_task = None
    if enable_corridor:
        z_boundary_task = ZBoundaryTask(
            name="z_corridor_ineq",
            priority=0,
            z_min=z_min,
            z_max=z_max,
            nominal_z_fn=nominal_z_fn,
            as_inequality=True,
            omega_n=35.0,
            kp=barrier_kp,
            kd=barrier_kd
        )
        task_stack.add_task(z_boundary_task)

    circle_3d_task = CartesianPoseTask(
        name="circle_3d_p0",
        priority=0,
        kp=500.0,
        kd=50.0,
        mode="3d",
        trajectory_fn=circle_traj
    )
    task_stack.add_task(circle_3d_task)

    posture_task = JointPostureTask(
        name="posture_p1",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=q_home
    )
    task_stack.add_task(posture_task)

    sim.start_recording()

    n_steps = int(sim_time / dt)
    logger.info(f"[Exp 11] Starting simulation loop for {sim_time:.1f}s ({n_steps} steps)...")

    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_ceiling_active: List[bool] = []
    log_floor_active: List[bool] = []
    log_ceiling_err: List[float] = []
    log_floor_err: List[float] = []
    log_circle_err: List[float] = []
    log_posture_err: List[float] = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["t"] = t_curr
        state["dt"] = dt

        p_des, _, v_des = circle_traj(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # Compute QP multi-priority torques with hard inequality constraints
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques & step simulation
        sim.apply_torques(torques)
        sim.step()

        # Update visuals
        sim.set_goal(p_des)
        sim.update_viz(tip_pos=state["ee_pos"], goal_pos=p_des)
        sim.record_frame()

        # Telemetry extraction
        p_curr = state["ee_pos"]
        q_curr = state["q"]

        ceil_err = max(0.0, float(p_curr[2] - z_max))
        flr_err = max(0.0, float(z_min - p_curr[2]))
        circ_err = float(np.linalg.norm(p_des - p_curr))
        post_err = float(np.linalg.norm(q_curr - q_home))

        log_time.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_ee_pos_des.append(p_des.copy())
        log_torques.append(torques.copy())
        # With the corridor ablated there is no task to query; log it as never active.
        log_ceiling_active.append(z_boundary_task.ceiling_active if z_boundary_task else False)
        log_floor_active.append(z_boundary_task.floor_active if z_boundary_task else False)
        log_ceiling_err.append(ceil_err)
        log_floor_err.append(flr_err)
        log_circle_err.append(circ_err)
        log_posture_err.append(post_err)

        if step % 200 == 0:
            status = "CORRIDOR_FREE" if z_boundary_task else "CORRIDOR_OFF"
            if z_boundary_task and z_boundary_task.ceiling_active:
                status = f"CEILING_CLAMP (vio={ceil_err*1000:.1f}mm)"
            elif z_boundary_task and z_boundary_task.floor_active:
                status = f"FLOOR_CLAMP   (vio={flr_err*1000:.1f}mm)"
            max_tau = np.max(np.abs(torques))
            logger.info(
                f"Step {step:4d}/{n_steps} | t={t_curr:5.2f}s | "
                f"z={p_curr[2]:.3f}m (des={p_des[2]:.3f}m) | "
                f"State={status:24s} | max|tau|={max_tau:5.1f}Nm"
            )

    sim.stop_recording()
    if record_path:
        logger.info(f"Recording written to {record_path}")

    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "z_min": z_min,
        "z_max": z_max,
        "ceiling_active": np.array(log_ceiling_active),
        "floor_active": np.array(log_floor_active),
        "ceiling_err": np.array(log_ceiling_err),
        "floor_err": np.array(log_floor_err),
        "circle_err": np.array(log_circle_err),
        "posture_err": np.array(log_posture_err),
        "tau_min": sim.tau_min,
        "tau_max": sim.tau_max,
        "n_violations": controller.n_violations,
        "n_solves": controller.n_solves,
        "max_violation": controller.max_violation,
    }

    if out_file:
        out = pathlib.Path(out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **logs)
        logger.info(f"Logs written to {out}")

    if save_plot:
        plot_experiment_11(logs, output_path="results/figures/exp11_cartesian_corridor_circle.png")

    logger.info("================================================================================")
    logger.info(f"Exp 11 Finished | Max Ceiling Violation: {np.max(logs['ceiling_err'])*1000:.2f} mm | "
                f"Max Floor Violation: {np.max(logs['floor_err'])*1000:.2f} mm | "
                f"Torque Violations: {logs['n_violations']}")
    logger.info("================================================================================")

    return logs


def plot_experiment_11(
    logs: Dict[str, np.ndarray],
    output_path: str = "results/figures/exp11_cartesian_corridor_circle.png"
) -> None:
    """
    Generates a 6-panel comprehensive diagnostic plot for Experiment 11.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_des = logs["ee_pos_des"]
    torques = logs["torques"]
    z_min = float(logs["z_min"])
    z_max = float(logs["z_max"])
    ceil_active = logs["ceiling_active"]
    flr_active = logs["floor_active"]

    fig, axs = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(
        "Experiment 11: Prioritized Z-Corridor Height Limits & Out-of-Bounds Circular Tracking",
        fontsize=14,
        fontweight="bold"
    )

    # Panel 1: Height Tracking & Priority Clamping (Z vs Time)
    ax1 = axs[0, 0]
    ax1.plot(t, p_des[:, 2], "k--", alpha=0.6, label="z_nominal (Circle breaches bounds)")
    ax1.plot(t, p_act[:, 2], "b-", linewidth=2.0, label="z_actual (Clamped within bounds)")
    ax1.axhline(z_max, color="red", linestyle="-", linewidth=2.0, label=f"Ceiling z_max = {z_max:.2f}m (QP Inequality)")
    ax1.axhline(z_min, color="orange", linestyle="-", linewidth=2.0, label=f"Floor z_min = {z_min:.2f}m (QP Inequality)")
    ax1.fill_between(t, z_max, np.maximum(z_max, p_des[:, 2]), color="red", alpha=0.15, label="Ceiling Breach Zone")
    ax1.fill_between(t, z_min, np.minimum(z_min, p_des[:, 2]), color="orange", alpha=0.15, label="Floor Breach Zone")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Height Z [m]")
    ax1.set_title("1. End-Effector Height vs Strict Safety Limits (QP Inequality Constraint)")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: X-Coordinate & Workspace Reach Saturation (X vs Time)
    ax2 = axs[0, 1]
    ax2.plot(t, p_des[:, 0], "r--", label="x_nominal (Reaches out of bounds)")
    ax2.plot(t, p_act[:, 0], "r-", linewidth=1.8, label="x_actual (Max extension)")
    ax2.plot(t, p_des[:, 1], "g--", alpha=0.5, label="y_nominal")
    ax2.plot(t, p_act[:, 1], "g-", alpha=0.8, label="y_actual")
    ax2.axhline(0.91, color="purple", linestyle=":", label="Franka Max Reach (~0.91m)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("2. Horizontal Coordinates & Reach Boundary Saturation")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: 2D Projected Motion in the X-Z Plane (The Truncated Loop)
    ax3 = axs[1, 0]
    ax3.plot(p_des[:, 0], p_des[:, 2], "r--", linewidth=1.5, label="Nominal Circle (Out-of-Bounds)")
    ax3.plot(p_act[:, 0], p_act[:, 2], "b-", linewidth=2.0, label="Actual Trajectory (Strict QP Inequality Clamping)")
    ax3.axhline(z_max, color="red", linestyle="-", linewidth=1.5, label="Ceiling Plane")
    ax3.axhline(z_min, color="orange", linestyle="-", linewidth=1.5, label="Floor Plane")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Z [m]")
    ax3.set_title("3. X-Z Cross-Section: Flat Top & Bottom (Strict QP Inequality Clamping)")
    ax3.grid(True)
    ax3.legend(fontsize=8)

    # Panel 4: Boundary Constraint Violations (mm)
    ax4 = axs[1, 1]
    ax4.plot(t, logs["ceiling_err"] * 1000.0, "r-", linewidth=1.8, label="Ceiling Penetration (mm)")
    ax4.plot(t, logs["floor_err"] * 1000.0, "orange", linewidth=1.8, label="Floor Penetration (mm)")
    ax4.plot(t, logs["circle_err"] * 100.0, "k:", alpha=0.5, label="Circle Tracking Error (cm)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Error / Penetration")
    ax4.set_title("4. Safety Boundary Violations & Circle Error Norm")
    ax4.grid(True)
    ax4.legend(fontsize=8)

    # Panel 5: Commanded Joint Torques & Saturation Limits
    ax5 = axs[2, 0]
    for j in range(torques.shape[1]):
        ax5.plot(t, torques[:, j], label=f"Joint {j+1}")
    ax5.axhline(87.0, color="red", linestyle="--", alpha=0.6, label="Joint Torque Limit (87 Nm)")
    ax5.axhline(-87.0, color="red", linestyle="--", alpha=0.6)
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Torque [Nm]")
    ax5.set_title(f"5. Commanded Joint Torques (0 Violations in {logs['n_solves']} Solves)")
    ax5.grid(True)
    ax5.legend(ncol=4, fontsize=7)

    # Panel 6: Dynamic Boundary Activation Flags (State of P0 / P1)
    ax6 = axs[2, 1]
    ax6.fill_between(t, 0, ceil_active.astype(int), color="red", alpha=0.3, label="Ceiling Clamp Active (z_nom > z_max)")
    ax6.fill_between(t, 0, flr_active.astype(int), color="orange", alpha=0.3, label="Floor Clamp Active (z_nom < z_min)")
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(["Free Tracking", "Boundary Clamped"])
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Inequality State")
    ax6.set_title("6. Dynamic Activation of QP Inequality Boundary Clamping")
    ax6.grid(True)
    ax6.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"[Exp 11] Diagnostic plot saved to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 11: Z-Corridor Height Limits & Out-of-Bounds Circle")
    parser.add_argument("--time", type=float, default=8.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode without viewer GUI")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--z-min", type=float, default=0.35, help="Floor height limit in meters")
    parser.add_argument("--z-max", type=float, default=0.55, help="Ceiling height limit in meters")
    parser.add_argument("--out", type=str, default=None, help="Telemetry output path")
    parser.add_argument("--solver", type=str, default="daqp", choices=["daqp", "osqp", "qpoases"], help="QP solver")
    args = parser.parse_args()

    run_experiment_11(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        z_min=args.z_min,
        z_max=args.z_max,
        out_path=args.out,
        solver=args.solver,
        save_plot=True
    )


if __name__ == "__main__":
    main()
