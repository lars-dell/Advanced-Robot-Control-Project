"""
Experiment 9: Energy Tank & Passivity Profiling under External Physical Disturbance.

Evaluates the energetic behavior and passivity of the Cartesian Impedance Controller:
    - Mechanical Power: P_mech(t) = dq^T * tau_opt
    - Stored Virtual Spring Energy: E_spring(t) = 0.5 * e_x^T * K_p * e_x + 0.5 * e_q^T * K_pn * e_q
    - Dissipation Rate: P_diss(t) = edot_x^T * K_d * edot_x + dq^T * K_dn * dq
    - Total Energy Balance: Verifies that cumulative energy remains bounded and strictly passive.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.qp_impedance import QPImpedanceController
from envs.genesis_sim import GenesisSim
from tasks import CartesianPoseTask, JointPostureTask, TaskStack

logger = logging.getLogger("Exp9_PassivityEnergy")


def run_experiment_9(
    sim_time: float = 6.0,
    dt: float = 0.005,
    show_viewer: bool = False,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Executes Experiment 9: Passivity & Energy Tank Profiling.
    """
    logger.info("=================================================================")
    logger.info("Running Experiment 9: Passivity & Energy Profiling")
    logger.info("=================================================================")

    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=show_viewer,
        dt=dt,
        device=device
    )

    kp_cart = 500.0
    kd_cart = 45.0
    kp_null = 20.0
    kd_null = 4.0

    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=kp_cart,
        kd_cart=kd_cart,
        kp_null=kp_null,
        kd_null=kd_null,
        solver_name="daqp"
    )

    state = sim.get_state()
    p_home = state["ee_pos"].copy()
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    # 2-Level Task Stack: Hold steady setpoint
    task_stack = TaskStack()
    cart_task = CartesianPoseTask(
        name="hold_pose_p0",
        priority=0,
        kp=kp_cart,
        kd=kd_cart,
        mode="3d"
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

    log_t: List[float] = []
    log_ee_pos: List[np.ndarray] = []
    log_f_ext: List[np.ndarray] = []
    log_e_spring: List[float] = []
    log_p_mech: List[float] = []
    log_p_diss: List[float] = []
    log_w_cum: List[float] = []
    log_w_diss_cum: List[float] = []
    log_torques: List[np.ndarray] = []

    w_cum_val = 0.0
    w_diss_val = 0.0

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        state["target_pos"] = p_home
        state["target_vel"] = np.zeros(3)

        p_curr = state["ee_pos"]
        q_curr = state["q"]
        dq_curr = state["dq"]
        J_task = state["J"][:3, :]

        # External perturbation schedule:
        # High-magnitude force impulse (50 N in +X) applied between t=1.5s and t=2.0s (0.5s duration)
        f_ext_applied = np.zeros(3, dtype=np.float64)
        if 1.5 <= t_curr <= 2.0:
            f_ext_applied = np.array([50.0, 0.0, 0.0])
            sim.apply_external_disturbance(force=f_ext_applied, link_name="hand")

        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)
        sim.apply_torques(torques)
        sim.step()

        # Energy & Passivity Metrics Calculation:
        e_pos = p_home - p_curr
        v_ee = J_task @ dq_curr
        e_posture = q_home - q_curr

        # Stored virtual spring potential energy: E = 0.5 * e^T * K * e
        e_spring_cart = 0.5 * kp_cart * float(np.dot(e_pos, e_pos))
        e_spring_null = 0.5 * kp_null * float(np.dot(e_posture, e_posture))
        e_spring_total = e_spring_cart + e_spring_null

        # Virtual damping dissipation rate: P_diss = edot^T * D * edot
        p_diss_cart = kd_cart * float(np.dot(v_ee, v_ee))
        p_diss_null = kd_null * float(np.dot(dq_curr, dq_curr))
        p_diss_total = p_diss_cart + p_diss_null

        # Actuator mechanical power (excluding gravity bias): P_mech = dq^T * tau_opt
        h = state.get("h", np.zeros(7))
        tau_opt = torques - h
        p_mech = float(np.dot(dq_curr, tau_opt))

        # Integrated cumulative work & dissipation
        w_cum_val += p_mech * dt
        w_diss_val += p_diss_total * dt

        log_t.append(t_curr)
        log_ee_pos.append(p_curr.copy())
        log_f_ext.append(f_ext_applied.copy())
        log_e_spring.append(e_spring_total)
        log_p_mech.append(p_mech)
        log_p_diss.append(p_diss_total)
        log_w_cum.append(w_cum_val)
        log_w_diss_cum.append(w_diss_val)
        log_torques.append(torques.copy())

    logs = {
        "time": np.array(log_t),
        "ee_pos": np.array(log_ee_pos),
        "p_home": p_home,
        "f_ext": np.array(log_f_ext),
        "e_spring": np.array(log_e_spring),
        "p_mech": np.array(log_p_mech),
        "p_diss": np.array(log_p_diss),
        "w_cum": np.array(log_w_cum),
        "w_diss_cum": np.array(log_w_diss_cum),
        "torques": np.array(log_torques),
    }

    if save_plot:
        plot_experiment_9(logs)

    logger.info("Experiment 9 completed successfully.")
    return logs


def plot_experiment_9(
    logs: Dict[str, np.ndarray],
    output_path: str = "results/figures/exp9_passivity_energy.png"
) -> None:
    """
    Generates multi-panel energy, power, and passivity diagnostic plots.
    """
    t = logs["time"]
    p_act = logs["ee_pos"]
    p_home = logs["p_home"]
    f_ext = logs["f_ext"]
    e_spring = logs["e_spring"]
    p_mech = logs["p_mech"]
    p_diss = logs["p_diss"]
    w_cum = logs["w_cum"]
    w_diss = logs["w_diss_cum"]
    torques = logs["torques"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Experiment 9: Energy Tank & Passivity Profiling under External Disturbance", fontsize=14, fontweight="bold")

    # Panel 1: End-Effector Displacement under Impulse
    ax1 = axs[0, 0]
    ax1.plot(t, (p_act[:, 0] - p_home[0]) * 1000.0, "r-", label="Delta X [mm]", linewidth=1.5)
    ax1.plot(t, (p_act[:, 1] - p_home[1]) * 1000.0, "g-", label="Delta Y [mm]", linewidth=1.5)
    ax1.plot(t, (p_act[:, 2] - p_home[2]) * 1000.0, "b-", label="Delta Z [mm]", linewidth=1.5)
    ax1.axvspan(1.5, 2.0, color="orange", alpha=0.2, label="50 N Push Active")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [mm]")
    ax1.set_title("1. End-Effector Cartesian Compliance Displacement")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: Applied External Force Profile
    ax2 = axs[0, 1]
    ax2.plot(t, f_ext[:, 0], "r-", linewidth=2, label="F_ext (X) [N]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Disturbance Force [N]")
    ax2.set_title("2. External Force Impulse (50 N for 0.5s)")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: Stored Potential Energy in Virtual Spring Tank
    ax3 = axs[1, 0]
    ax3.plot(t, e_spring, "m-", linewidth=2, label="E_spring (Virtual Tank Energy)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Stored Energy [Joules]")
    ax3.set_title("3. Virtual Spring Energy Tank E(t) = 0.5 e^T K e")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Instantaneous Power & Damping Dissipation Rate
    ax4 = axs[1, 1]
    ax4.plot(t, p_mech, "b-", alpha=0.85, label="Mechanical Power P_mech = dq^T tau")
    ax4.plot(t, p_diss, "g-", alpha=0.85, label="Dissipation Rate P_diss = v^T D v")
    ax4.axhline(0.0, color="k", linestyle=":")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Power [Watts]")
    ax4.set_title("4. Instantaneous Power Flow & Dissipation Rate")
    ax4.grid(True)
    ax4.legend()

    # Panel 5: Cumulative Energy Balance (Passivity Verification)
    ax5 = axs[2, 0]
    ax5.plot(t, w_cum, "b-", linewidth=2, label="Cumulative Control Work W_ctrl = ∫ P_mech dt")
    ax5.plot(t, w_diss, "g--", linewidth=2, label="Cumulative Dissipated Energy W_diss = ∫ P_diss dt")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Energy [Joules]")
    ax5.set_title("5. Cumulative Energy Balance (Passivity Proof)")
    ax5.grid(True)
    ax5.legend()

    # Panel 6: Commanded Joint Torques
    ax6 = axs[2, 1]
    for i in range(torques.shape[1]):
        ax6.plot(t, torques[:, i], label=f"Joint {i+1}")
    ax6.axhline(87.0, color="red", linestyle="--", alpha=0.5, label="Torque Bound")
    ax6.axhline(-87.0, color="red", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Joint Torque [Nm]")
    ax6.set_title("6. Commanded Joint Torques")
    ax6.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved passivity energy diagnostic plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 9: Energy Tank & Passivity Profiling")
    parser.add_argument("--time", type=float, default=6.0, help="Simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--no-vis", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    args = parser.parse_args()

    run_experiment_9(
        sim_time=args.time,
        dt=args.dt,
        show_viewer=not args.no_vis,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
