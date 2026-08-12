"""
Main entry point for multi-priority Cartesian impedance control simulation.

Initializes the Genesis simulator, instantiates the QP impedance controller,
builds the modular task hierarchy stack, executes the real-time simulation loop,
and handles telemetry data logging.
"""

import argparse
import logging
import time
from typing import Dict, List
import numpy as np

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger(__name__)


def run_simulation(
    sim_time: float = 5.0,
    dt: float = 0.005,
    show_viewer: bool = True,
    device: str = "gpu"
) -> Dict[str, np.ndarray]:
    """
    Executes the main control simulation loop using the modular task stack and paper-compliant QP controller.

    Args:
        sim_time: Total simulation duration in seconds (default: 5.0s).
        dt: Control timestep in seconds (default: 0.005s / 200 Hz).
        show_viewer: Whether to launch interactive 3D Genesis viewer.
        device: Computing backend ('cpu' or 'gpu').

    Returns:
        Dict[str, np.ndarray]: Recorded log data arrays for analysis and plotting.
    """
    logger.info(f"[Main] Initializing Genesis simulation on backend '{device}'...")
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device
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

    # Define target reference setpoints
    target_ee_pos = initial_ee_pos + np.array([0.15, 0.10, -0.05])
    target_ee_rot = initial_state["ee_rot"].copy()
    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    # Construct modular task stack hierarchy
    task_stack = TaskStack()

    # Priority 0: Primary Cartesian 3D Position Impedance Task
    cart_task = CartesianPoseTask(
        name="cartesian_primary",
        priority=0,
        kp=400.0,
        kd=40.0,
        is_6d=False,
        use_full_impedance=False
    )
    task_stack.add_task(cart_task)

    # Priority 1: Secondary Joint Posture Null-Space Task
    posture_task = JointPostureTask(
        name="posture_secondary",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    logger.info(f"[Main] Starting simulation control loop for {sim_time} seconds ({n_steps} steps)...")

    # Data logging containers
    log_time: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_ee_pos_des: List[np.ndarray] = []
    log_torques: List[np.ndarray] = []
    log_errors: List[np.ndarray] = []

    for step in range(n_steps):
        t_curr = step * dt

        # Extract robot dynamics state
        state = sim.get_state()
        state["target_pos"] = target_ee_pos
        state["target_rot"] = target_ee_rot
        state["q_null"] = target_q_null

        # Compute commanded joint torques via Hierarchical QP Optimization
        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)

        # Apply torques to robot joints
        sim.apply_torques(torques)

        # Optional: Apply disturbance force halfway through simulation
        if 2.0 <= t_curr <= 2.2:
            sim.apply_external_disturbance(force=np.array([10.0, 0.0, 0.0]), link_name="hand")

        # Step physics simulation engine
        sim.step()

        # Log telemetry data
        log_time.append(t_curr)
        log_ee_pos.append(state["ee_pos"].copy())
        log_ee_pos_des.append(target_ee_pos.copy())
        log_torques.append(torques.copy())
        log_errors.append(cart_task.compute_error(state))

        if step % 100 == 0:
            err = cart_task.compute_error(state)
            max_tau = np.max(np.abs(torques))
            logger.info(f"Step {step:4d}/{n_steps} | Time: {t_curr:5.2f}s | EE Tracking Error: {err:6.4f}m | Max Torque: {max_tau:5.2f} Nm")

    logger.info("[Main] Simulation completed successfully.")

    # Package log data
    logs = {
        "time": np.array(log_time),
        "ee_pos": np.array(log_ee_pos),
        "ee_pos_des": np.array(log_ee_pos_des),
        "torques": np.array(log_torques),
        "errors": np.array(log_errors),
    }

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
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Physics backend device")
    args = parser.parse_args()

    run_simulation(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device
    )


if __name__ == "__main__":
    main()
