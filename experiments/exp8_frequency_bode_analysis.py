"""
Experiment 8: Tracking Bandwidth & Dynamic Frequency Response (Bode-Style Analysis).

Compares:
    - Virtual Model Control (VMC / Spring-Damper Law, Eq. 22)
    - Full Mass Matrix Cartesian Impedance Law (Lambda(q) Inertia Shaping, Eq. 23-24)

Evaluates tracking bandwidth, gain attenuation (dB), phase lag (deg), and RMSE
across multiple trajectory excitation frequencies f in [0.25, 0.5, 1.0, 2.0, 3.0] Hz.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Ensure root workspace directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.genesis_sim import GenesisSim
from controllers.qp_impedance import QPImpedanceController
from tasks import TaskStack, CartesianPoseTask, JointPostureTask

logger = logging.getLogger("Exp8_FrequencyBode")


def evaluate_frequency_response(
    freq: float,
    use_full_impedance: bool,
    sim_time: float = 6.0,
    dt: float = 0.005,
    device: str = "cpu",
    sim: Any = None
) -> Dict[str, Any]:
    """
    Executes a sinusoidal trajectory tracking test at a specific excitation frequency.

    Args:
        freq: Frequency in Hz.
        use_full_impedance: Whether to use Lambda mass matrix compensation.
        sim_time: Total simulation time in seconds.
        dt: Control timestep in seconds.
        device: 'cpu' or 'gpu'.
        sim: Optional existing GenesisSim instance to reuse.

    Returns:
        Dict[str, Any]: Telemetry and frequency response metrics.
    """
    if sim is None:
        sim = GenesisSim(
            model_xml="panda_cylinder.xml",
            show_viewer=False,
            dt=dt,
            device=device
        )
    else:
        sim.reset()

    controller = QPImpedanceController(
        n_dofs=7,
        kp_cart=600.0,
        kd_cart=50.0,
        kp_null=20.0,
        kd_null=4.0,
        solver_name="daqp"
    )

    state = sim.get_state()
    p_init = state["ee_pos"].copy()
    amplitude = 0.05  # 5 cm amplitude along Y axis

    # Trajectory function
    def traj_fn(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        omega = 2.0 * np.pi * freq
        p_des = p_init.copy()
        p_des[1] += amplitude * np.sin(omega * t)
        v_des = np.zeros(3)
        v_des[1] = amplitude * omega * np.cos(omega * t)
        return p_des, np.eye(3), v_des

    task_stack = TaskStack()
    cart_task = CartesianPoseTask(
        name="tracking_p0",
        priority=0,
        kp=600.0,
        kd=50.0,
        mode="3d",
        use_full_impedance=use_full_impedance,
        trajectory_fn=traj_fn
    )
    task_stack.add_task(cart_task)

    target_q_null = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    posture_task = JointPostureTask(
        name="posture_p1",
        priority=1,
        kp=20.0,
        kd=4.0,
        q_des=target_q_null
    )
    task_stack.add_task(posture_task)

    n_steps = int(sim_time / dt)
    log_t: List[float] = []
    log_y_des: List[float] = []
    log_y_act: List[float] = []

    for step in range(n_steps):
        t_curr = step * dt
        state = sim.get_state()
        p_des, _, v_des = traj_fn(t_curr)
        state["target_pos"] = p_des
        state["target_vel"] = v_des

        # Analytical acceleration feedforward for full impedance law
        omega = 2.0 * np.pi * freq
        a_des = np.zeros(3)
        a_des[1] = - amplitude * (omega**2) * np.sin(omega * t_curr)
        state["target_acc"] = a_des

        torques = controller.compute_torques(state=state, target=task_stack, t=t_curr)
        sim.apply_torques(torques)
        sim.step()

        log_t.append(t_curr)
        log_y_des.append(p_des[1])
        log_y_act.append(state["ee_pos"][1])

    t_arr = np.array(log_t)
    y_des = np.array(log_y_des)
    y_act = np.array(log_y_act)

    # Discard initial transient (first 2 seconds) for steady-state frequency analysis
    idx_ss = t_arr >= 2.0
    t_ss = t_arr[idx_ss]
    y_des_ss = y_des[idx_ss] - p_init[1]
    y_act_ss = y_act[idx_ss] - p_init[1]

    # Compute metrics: RMSE, Amplitude Ratio, and Phase Lag
    rmse = float(np.sqrt(np.mean((y_des_ss - y_act_ss)**2)))
    
    # Peak amplitude via standard deviation scaling
    amp_des = np.sqrt(2.0) * np.std(y_des_ss)
    amp_act = np.sqrt(2.0) * np.std(y_act_ss)
    gain_linear = amp_act / (amp_des + 1e-8)
    gain_db = 20.0 * np.log10(max(gain_linear, 1e-4))

    # Phase calculation via cross-correlation
    correlation = np.correlate(y_act_ss - np.mean(y_act_ss), y_des_ss - np.mean(y_des_ss), mode="full")
    lags = np.arange(-len(y_des_ss) + 1, len(y_des_ss))
    best_lag = lags[np.argmax(correlation)]
    time_delay = best_lag * dt
    phase_lag_deg = float((time_delay * freq * 360.0) % 360.0)
    if phase_lag_deg > 180.0:
        phase_lag_deg -= 360.0

    return {
        "freq": freq,
        "rmse": rmse,
        "gain_linear": gain_linear,
        "gain_db": gain_db,
        "phase_lag_deg": phase_lag_deg,
        "time": t_arr,
        "y_des": y_des,
        "y_act": y_act,
    }


def run_experiment_8(
    frequencies: Optional[List[float]] = None,
    sim_time: float = 6.0,
    dt: float = 0.005,
    device: str = "cpu",
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Runs the full frequency response sweep.
    """
    if frequencies is None:
        frequencies = [0.25, 0.5, 1.0, 2.0, 3.0]

    logger.info("=================================================================")
    logger.info("Running Experiment 8: Frequency Response & Bandwidth (Bode)")
    logger.info("=================================================================")

    results_vmc: List[Dict[str, Any]] = []
    results_full: List[Dict[str, Any]] = []
    sim = GenesisSim(
        model_xml="panda_cylinder.xml",
        show_viewer=False,
        dt=dt,
        device=device
    )

    for f in frequencies:
        logger.info(f"Testing Frequency f = {f:.2f} Hz | Virtual Model Control (VMC)...")
        res_v = evaluate_frequency_response(
            freq=f,
            use_full_impedance=False,
            sim_time=sim_time,
            dt=dt,
            device=device,
            sim=sim
        )
        results_vmc.append(res_v)

        logger.info(f"Testing Frequency f = {f:.2f} Hz | Full Mass Matrix Lambda(q)...")
        res_f = evaluate_frequency_response(
            freq=f,
            use_full_impedance=True,
            sim_time=sim_time,
            dt=dt,
            device=device,
            sim=sim
        )
        results_full.append(res_f)

    benchmark_data = {
        "frequencies": np.array(frequencies),
        "vmc": results_vmc,
        "full_impedance": results_full
    }

    if save_plot:
        plot_experiment_8(benchmark_data)

    logger.info("Experiment 8 completed successfully.")
    return benchmark_data


def plot_experiment_8(
    data: Dict[str, Any],
    output_path: str = "exp8_frequency_bode_analysis.png"
) -> None:
    """
    Generates multi-panel Bode plots and time-domain tracking comparisons.
    """
    freqs = data["frequencies"]
    vmc_res = data["vmc"]
    full_res = data["full_impedance"]

    vmc_gain = [r["gain_db"] for r in vmc_res]
    vmc_phase = [r["phase_lag_deg"] for r in vmc_res]
    vmc_rmse = [r["rmse"] * 1000.0 for r in vmc_res]

    full_gain = [r["gain_db"] for r in full_res]
    full_phase = [r["phase_lag_deg"] for r in full_res]
    full_rmse = [r["rmse"] * 1000.0 for r in full_res]

    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Experiment 8: Frequency Response & Dynamic Bandwidth (Bode Analysis)", fontsize=14, fontweight="bold")

    # Panel 1: Bode Magnitude Plot (Gain in dB)
    ax1 = axs[0, 0]
    ax1.plot(freqs, vmc_gain, "ro-", linewidth=2, label="Virtual Model Control (VMC)")
    ax1.plot(freqs, full_gain, "bs--", linewidth=2, label="Full Cartesian Impedance (Lambda)")
    ax1.axhline(0.0, color="gray", linestyle=":", label="0 dB (Ideal Tracking)")
    ax1.axhline(-3.0, color="k", linestyle="--", alpha=0.5, label="-3 dB Cutoff")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude Gain [dB]")
    ax1.set_title("1. Bode Magnitude Response (|X_act| / |X_des|)")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: Bode Phase Plot (Phase Lag in degrees)
    ax2 = axs[0, 1]
    ax2.plot(freqs, vmc_phase, "ro-", linewidth=2, label="Virtual Model Control (VMC)")
    ax2.plot(freqs, full_phase, "bs--", linewidth=2, label="Full Cartesian Impedance (Lambda)")
    ax2.axhline(0.0, color="gray", linestyle=":", label="0 deg (Zero Lag)")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase Shift [deg]")
    ax2.set_title("2. Bode Phase Response")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: RMSE Tracking Error vs Frequency
    ax3 = axs[1, 0]
    ax3.plot(freqs, vmc_rmse, "ro-", linewidth=2, label="VMC (Spring-Damper)")
    ax3.plot(freqs, full_rmse, "bs--", linewidth=2, label="Full Impedance (Lambda)")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Steady-State RMSE [mm]")
    ax3.set_title("3. Tracking Error vs Frequency")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: Time-Domain Trajectory Overlay at High Frequency (f = 2.0 Hz)
    ax4 = axs[1, 1]
    idx_high = -2 if len(freqs) >= 2 else 0
    t = vmc_res[idx_high]["time"]
    y_des = vmc_res[idx_high]["y_des"]
    y_vmc = vmc_res[idx_high]["y_act"]
    y_full = full_res[idx_high]["y_act"]

    # Show steady-state window t in [3.0, 5.0]s
    mask = (t >= 3.0) & (t <= 5.0)
    ax4.plot(t[mask], y_des[mask], "k--", linewidth=2, label=f"Desired (f={freqs[idx_high]:.1f}Hz)")
    ax4.plot(t[mask], y_vmc[mask], "r-", alpha=0.85, label="VMC (Notice Phase Lag)")
    ax4.plot(t[mask], y_full[mask], "b-", alpha=0.85, label="Full Impedance (Compensated)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Y Position [m]")
    ax4.set_title(f"4. High-Frequency Tracking Overlay (f = {freqs[idx_high]:.1f} Hz)")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    logger.info(f"Saved frequency response plot to {output_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Experiment 8: Frequency Response & Bode Analysis")
    parser.add_argument("--time", type=float, default=6.0, help="Simulation time per frequency in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Physics backend")
    parser.add_argument("--quick", action="store_true", help="Run reduced frequency set for rapid testing")
    args = parser.parse_args()

    freqs = [0.5, 1.0, 2.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 3.0]

    run_experiment_8(
        frequencies=freqs,
        sim_time=args.time,
        dt=args.dt,
        device=args.device,
        save_plot=True
    )


if __name__ == "__main__":
    main()
