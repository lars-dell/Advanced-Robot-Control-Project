"""
Experiments package for Multi-Priority Cartesian Impedance Control validation suite.
"""

# Base priority hierarchy scenarios
from experiments.exp_reach import run_reach
from experiments.exp_reach_split import run_reach_split
from experiments.exp_conflict import run_conflict

# Benchmark suite experiments (Task 2.2 & ICRA 2018 Reproduction)
from experiments.exp1_surface_circle import run_experiment_1
from experiments.exp2_blocked_circle import run_experiment_2
from experiments.exp3_apf_avoidance import run_experiment_3
from experiments.exp4_multilink_push import run_experiment_4
from experiments.exp5_torque_constrained_wipe import run_experiment_5
from experiments.exp6_singularity_tracking import run_experiment_6
from experiments.exp7_baseline_comparison import run_experiment_7
from experiments.exp8_frequency_bode_analysis import run_experiment_8
from experiments.exp9_passivity_energy_profiling import run_experiment_9
from experiments.exp10_parameter_robustness import run_experiment_10
from experiments.exp11_cartesian_corridor_circle import run_experiment_11, run_experiment_11 as run_corridor
from experiments.benchmark_solver_latency import run_latency_benchmark
from experiments.compare_controllers import run_controller_comparison
from experiments.exp_gain_sweep import run_parameter_sensitivity_study
from experiments.exp_chatter_mitigation import run_chatter_mitigation_benchmark

__all__ = [
    "run_reach",
    "run_reach_split",
    "run_conflict",
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "run_experiment_4",
    "run_experiment_5",
    "run_experiment_6",
    "run_experiment_7",
    "run_experiment_8",
    "run_experiment_9",
    "run_experiment_10",
    "run_experiment_11",
    "run_corridor",
    "run_latency_benchmark",
    "run_controller_comparison",
    "run_parameter_sensitivity_study",
    "run_chatter_mitigation_benchmark",
]

