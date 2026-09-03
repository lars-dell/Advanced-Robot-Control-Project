# Multi-Priority Cartesian Impedance Control (QP)

Implementation and evaluation of a Quadratic Programming (QP)-based hierarchical Cartesian impedance controller with joint-torque constraints for redundant manipulators.

Based on the paper:
> **Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization**  
> Enrico Mingo Hoffman, Arturo Laurenzi, Luca Muratore, Nikos G. Tsagarakis, and Darwin G. Caldwell  
> *IEEE International Conference on Robotics and Automation (ICRA), 2018*  
> [DOI: 10.1109/ICRA.2018.8462877](https://doi.org/10.1109/ICRA.2018.8462877)

---

## Key Features & Implementation Highlights

- **Hierarchical QP Structure:** Formulates multi-task priority levels as a cascade of Quadratic Programs operating directly in joint-torque space $\tau \in \mathbb{R}^n$, avoiding full Inverse Dynamics optimization over joint accelerations and contact force variables.
- **Pseudo-Inverse Free Formulation:** Eliminates expensive matrix pseudo-inversions ($\bar{J}$) within the QP objective functions by exploiting the unconstrained optimality condition $J B^{-1} \tau = J B^{-1} J^T f$.
- **Strict Joint Torque Saturation Bounds:** Enforces joint-torque saturation limits ($\tau_{\min} - h \le \tau \le \tau_{\max} - h$) directly inside the optimization problem to respect motor hardware limits.
- **Pre-Compiled CasADi Solver Engine:** Pre-compiles parametric CasADi C-functions using the `qpOASES` active-set solver (with automatic `OSQP` fallback) for real-time control capability.
- **Modular Task Hierarchy Stack:** Object-oriented task stack abstraction supporting arbitrary combinations of 3D/6D Cartesian pose impedance tasks, Virtual Model Control (VMC), full mass matrix ($\Lambda$) impedance, and joint space posture tasks.
- **Genesis Physics Integration:** Decoupled environment layer using the [Genesis physics engine](https://github.com/Genesis-Embodied-AI/Genesis) for a 7-DOF Franka Emika Panda manipulator (`panda_cylinder.xml`).

---

## Codebase Architecture

```
.
├── main.py                     # Entry point: simulation control loop, scenarios & experiment benchmarks
├── run_all_experiments.py      # Master benchmark runner executing all 10 evaluation experiments
├── pyproject.toml              # Project metadata & dependencies (genesis-world, casadi, qpsolvers, torch)
├── panda_cylinder.xml          # Franka Emika Panda 7-DOF MJCF robot model
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py      # Abstract base class interface for robot controllers
│   ├── qp_impedance.py         # Multi-Priority Hierarchical QP Cartesian Impedance Controller (Eq. 18)
│   ├── classical_transpose.py  # Classical Jacobian Transpose Impedance Controller (Eq. 9)
│   ├── saturated_algebraic.py  # Saturated Algebraic Null-Space Controller with Post-Hoc Clipping (Eq. 10)
│   └── weighted_qp.py          # Single-Level Weighted-Sum QP Controller
├── solvers/
│   ├── __init__.py
│   └── casadi_qp_solver.py     # Pre-compiled parametric CasADi QP solver module (qpOASES / OSQP / DAQP)
├── tasks/
│   ├── __init__.py
│   ├── base_task.py            # Abstract base class for prioritizable control tasks
│   ├── cartesian_task.py       # Cartesian 3D position / 6D pose impedance task (VMC & Full Impedance)
│   ├── posture_task.py         # Joint-space null-space posture stiffness task
│   ├── force_task.py           # Cartesian force control task & circular trajectory generator
│   ├── apf_task.py             # Artificial Potential Field (APF) obstacle repulsion task
│   └── task_stack.py           # Priority-ordered task stack manager
├── experiments/
│   ├── __init__.py
│   ├── exp1_surface_circle.py  # Exp 1: Surface circular tracking & normal force exertion
│   ├── exp2_blocked_circle.py  # Exp 2: Blocked circular tracking & compliant obstacle contact
│   ├── exp3_apf_avoidance.py   # Exp 3: Reactive APF obstacle avoidance
│   ├── exp4_multilink_push.py  # Exp 4: Multi-link external disturbance rejection
│   ├── exp5_torque_constrained_wipe.py # Exp 5: Torque-constrained surface wiping
│   ├── exp6_singularity_tracking.py    # Exp 6: Kinematic singularity tracking & damping
│   ├── exp7_baseline_comparison.py     # Exp 7: 4-Way baseline comparison (Hoffman et al. Figs 1-4)
│   ├── exp8_frequency_bode_analysis.py # Exp 8: Frequency response & Bode bandwidth
│   ├── exp9_passivity_energy_profiling.py # Exp 9: Energy tank & passivity profiling
│   ├── exp10_parameter_robustness.py   # Exp 10: Model parameter uncertainty robustness
│   └── benchmark_solver_latency.py     # Real-time solver latency profiling (< 0.5 ms @ 200 Hz)
├── utils/
│   ├── __init__.py
│   └── math_utils.py           # Dynamically consistent pseudo-inverses, null-space projectors, 6D pose error
├── scripts/
│   ├── plot_results.py         # Telemetry comparison plotting for conflict scenarios
│   └── probe_genesis.py        # Diagnostic probe for Genesis physics & MuJoCo shadow dynamics
└── envs/
    ├── __init__.py
    └── genesis_sim.py          # Genesis physics simulator wrapper with MuJoCo dynamics bridge
```

---

## Mathematical Formulation & Paper Alignment

### 1. Problem Definition & Operational Space Dynamics
For an $n$-DOF redundant manipulator with joint configuration vector $q \in \mathbb{R}^n$, task space coordinates $x = x(q) \in \mathbb{R}^m$ ($m < n$), and task Jacobian $J(q) = \frac{\partial x}{\partial q} \in \mathbb{R}^{m \times n}$, the joint-space dynamics in contact with the environment are:
$$B(q)\ddot{q} + h(q, \dot{q}) = \tau + J^T f_{\text{ext}}$$

where $B(q) \in \mathbb{R}^{n \times n}$ is the joint-space inertia matrix, $h(q, \dot{q}) = C(q, \dot{q})\dot{q} + g(q)$ is the vector of gravity and Coriolis/centrifugal forces, $\tau \in \mathbb{R}^n$ are joint torques, and $f_{\text{ext}}$ are external forces.

The relation between joint torques $\tau$ and task-space forces $f \in \mathbb{R}^m$ is:
$$\bar{J}^T \tau = f$$

where $\bar{J} = B^{-1} J^T (J B^{-1} J^T)^{-1}$ is the dynamically consistent pseudo-inverse.

### 2. Eliminating Matrix Pseudo-Inversion in QP Objectives
Classical formulations minimize $\|\bar{J}^T \tau - f\|^2$, requiring explicit evaluation of the dynamically consistent pseudo-inverse $\bar{J}$ inside the cost function.

By taking the derivative of the quadratic cost $F(\tau) = \frac{1}{2} \tau^T \bar{J} \bar{J}^T \tau - f^T \bar{J}^T \tau + \frac{1}{2} f^T f$, the optimality condition yields:
$$\frac{\partial F(\tau)}{\partial \tau} = \bar{J}\bar{J}^T \tau - \bar{J}f = 0 \implies J B^{-1} \tau = J B^{-1} J^T f$$

Thus, the single-task QP is reformulated without pseudo-inverses (Equation 16 in the paper):
$$\min_{\tau} \left\| J B^{-1} \tau - J B^{-1} J^T f \right\|^2$$

### 3. Hierarchical Prioritized QP Cascade
For multiple prioritized tasks ($0 = \text{highest priority}, 1, \dots$), the controller solves a cascade of QP optimization problems.

At priority level $i$, the optimal joint torques $\tau_i$ are obtained by solving (Equation 18 in the paper):
$$\arg\min_{\tau_i} \left\| J_i B^{-1} \tau_i - J_i B^{-1} J_i^T f_i \right\|^2 + \varepsilon \|\tau_i\|^2$$

$$\text{subject to:} \quad \tau_{\min} - h(q,\dot{q}) \le \tau_i \le \tau_{\max} - h(q,\dot{q})$$
$$J_{j} B^{-1} \tau_i = J_{j} B^{-1} \tau_j^* \quad \forall j \in \{0, \dots, i-1\}$$

where:
- $\varepsilon > 0$ is a regularization factor ensuring numeric stability near singularities.
- The linear equality constraints $J_j B^{-1} \tau_i = J_j B^{-1} \tau_j^*$ preserve the optimal task performance of higher-priority levels $j < i$.

### 4. Task Virtual Forces $f_i$
- **Virtual Model Control (VMC) / Simplified Spring-Damper (Equation 22):**
  $$f_{\text{cart}} = K_p (x_{\text{des}} - x) + K_d (\dot{x}_{\text{des}} - \dot{x})$$
- **Full Cartesian Impedance (Equation 23–24):**
  $$f_{\text{cart}} = \Lambda(q) \left(\ddot{x}_{\text{des}} - \dot{J}\dot{q}\right) + K_p (x_{\text{des}} - x) + K_d (\dot{x}_{\text{des}} - \dot{x})$$
  where $\Lambda(q) = (J B^{-1} J^T)^{-1}$ is the operational space mass matrix.
- **Null-Space Posture Task (Level 1, Equation 25):**
  $$\tau_{\text{null}} = K_{p,n} (q_{\text{null}} - q) - K_{d,n} \dot{q}$$

### 5. Feedforward Bias Compensation & Output Torques
The final output torque command sent to the actuators includes feedforward dynamics compensation (Equations 20–21):
$$\tau_{\text{cmd}} = \tau_{\text{opt}} + h(q, \dot{q})$$

---

## Setup & Execution

### 1. Prerequisites & Installation
This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/lars-dell/Advanced-Robot-Control-Project.git
cd Advanced-Robot-Control-Project

# Create virtual environment and install dependencies from pyproject.toml
uv sync

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Running the Simulation Loop
Run the control simulation with interactive 3D rendering or headless:

```bash
# Run simulation with 3D GUI viewer (default: 5.0 seconds on the CPU backend)
uv run python main.py

# Run headless simulation (no visualizer window)
uv run python main.py --no-vis

# Run for 60 seconds so the viewer window stays open long enough to watch
uv run python main.py --time 60

# Run with the viewer but without any markers
uv run python main.py --no-markers
```

### Scenarios

`--scenario` selects the priority hierarchy. All three use the same controller.

| Scenario | Priority levels | Reference | Purpose |
| :--- | :--- | :--- | :--- |
| `reach` | `cartesian (3-D) > posture` | reachable point | One Cartesian objective. Baseline. |
| `reach_split` | `x > y > z > posture` | the same reachable point | **Three Cartesian objectives at different priorities** (assignment requirement 2). Everything is satisfiable, so all three errors go to zero. |
| `conflict` | configurable, e.g. `x > y > z > posture` | `[1.30, 0.15, 0.75]` — **x out of reach**, y and z reachable | Objectives compete; the priority order decides who is sacrificed. |

`reach` and `reach_split` both converge to `0.0000 m`. That equivalence is the point of
`reach_split`: decomposing one 3-D objective into three ranked 1-D objectives reproduces the
undecomposed result exactly, which is only true if the cascade's priority constraints are correct.
Priority ordering has no *visible* effect there because nothing has to be given up — that requires
a conflicting reference.

#### Priority ordering decides who is sacrificed

In `conflict` the arm can extend to about `x = 0.93` while holding the commanded y and z, so the x
objective is permanently unsatisfiable while y and z are not. All three compete for one thing —
which direction the arm commits its reach to — and the ranking settles it:

| Priority order | x error | y error | z error | |
| :--- | ---: | ---: | ---: | :--- |
| `x > y > z` | **0.376** | 0.019 | 0.293 | x is protected, z absorbs the shortfall |
| `z > y > x` | 0.461 | 0.000 | **0.000** | z is satisfied exactly, x absorbs it instead |

<table>
<tr><th><code>--priority-order xyz</code></th><th><code>--priority-order zyx</code></th></tr>
<tr>
<td><img src="docs/media/conflict_xyz.gif" width="420" alt="x highest priority"></td>
<td><img src="docs/media/conflict_zyx.gif" width="420" alt="z highest priority"></td>
</tr>
<tr>
<td>Reach is prioritised: the arm commits further forward and lets the tip sag below the commanded height.</td>
<td>Height is prioritised: the tip holds the commanded z exactly and gives up reach instead.</td>
</tr>
</table>

Both clips are 7 s of the same scenario with the same controller, gains and target — only the
ranking differs. The green sphere is the (unreachable) target, the amber line is the tracking
error, and the cyan trail is the path of the tool tip.

![Per-axis tracking error under both priority orders](docs/media/conflict_task_errors.png)

The crossover between the first and third panels is the whole result. In **x**, blue (x on top)
settles *below* orange — 0.376 against 0.461. In **z** they invert: orange is flat at 0.000 while
blue oscillates around 0.293. The **y** panel shows both near zero, because y is satisfiable either
way and the ranking never has to decide anything.

Two honest caveats. Priority did not make x succeed — 0.376 is still a large error, because the
target is out of reach and no ranking can change physics. And the z trace under `x > y > z` never
settles; it oscillates for the full run rather than converging.

![Share of control steps each joint spends at its torque bound](docs/media/conflict_torques.png)

Every joint touches its bound at some point, so peak utilisation is 1.0 across the board and
carries no information; what varies is how long each stays there. Under `x > y > z` the shoulder
(J2) is pinned for **80%** of the run and the 12 N·m wrist joints J5/J6 for **53%** and **70%** —
the joints the lever-arm geometry predicts would saturate first. Every bar is far lower under
`z > y > x`: prioritising the unreachable objective costs much more actuator effort *and* gives a
worse overall outcome.

Across both runs there were **0 torque-limit violations in 2,800 control steps**, with no clipping
anywhere in the controller. That is the evidence for requirement 4 — feasibility comes from the QP
bounds of eq. (19)–(21), not from clamping the output.

Regenerate both figures from the logs with:

```bash
uv run python main.py --no-vis --time 7 --scenario conflict --priority-order xyz --out results/conflict_xyz.npz
uv run python main.py --no-vis --time 7 --scenario conflict --priority-order zyx --out results/conflict_zyx.npz
uv run python scripts/plot_results.py
```

```bash
uv run python main.py --scenario conflict --priority-order xyz
uv run python main.py --scenario conflict --priority-order zyx
```

Two things are worth reading off that table.

**Priority does not guarantee success, it guarantees non-interference.** At top priority x still
fails, with an error of 0.371 — the target is physically out of reach and no ranking can change
that. What the ranking bought is 0.371 instead of 0.461: nothing below x was permitted to make it
worse. A lower-priority task can never degrade a higher one.

**Ranking only matters between objectives that actually compete.** The y error is ~0 under both
orderings, because y = 0.15 is easy and there is enough freedom to satisfy it regardless of where
it sits in the stack.

A one-dimensional objective is expressed with `CartesianPoseTask(axes=[0])` (x only), `[1]` (y),
`[2]` (z). This is the decomposition Hoffman et al. use in section V-A.

### Visualization

When the viewer is open, the simulation draws what the controller is doing:

| Element | Appearance | Meaning |
| :--- | :--- | :--- |
| **Goal** | green glowing sphere (r = 2 cm) | Where the primary Cartesian task is aiming |
| **Error line** | amber line, tool tip → goal | **Its length _is_ the tracking error.** It shrinks to nothing as the arm converges |
| **Disturbance** | red arrow at the tool tip | The external force while it acts (2 cm of arrow per newton). Visible only during the t = 2.0–2.2 s push |
| **Trail** | cyan breadcrumbs, fading with age | Path the tool tip has travelled, including the kink where the disturbance knocked it off course |

Turn everything off with `--no-markers`.

Notes:
- Overlays are **viewer-only**. In a headless run (`--no-vis`) they are skipped entirely and cost
  nothing, so telemetry runs are unaffected.
- The goal marker is **massless and collision-free**, and the other three elements are debug-draw
  overlays rather than scene objects. None of them can perturb the physics — verified by comparing
  headless runs with markers on and off, which produce identical results.
- The overlays are redrawn every 5th control step (40 Hz at the default `dt`). Redrawing at the full
  200 Hz would take the viewer lock every step and measurably slow the simulation.

### CLI Command Line Arguments for `main.py`
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--time` | `float` | `5.0` | Total simulation duration in seconds. |
| `--dt` | `float` | `0.005` | Control loop timestep in seconds (200 Hz). |
| `--no-vis` | `flag` | `False` | Run in headless mode without 3D viewer window. |
| `--device` | `str` | `cpu` | Genesis physics backend (`cpu` or `gpu`). |
| `--no-markers` | `flag` | `False` | Disable the goal, error-line, disturbance and trail overlays. |
| `--out` | `str` | `results/run.npz` | Destination file for the telemetry log. |
| `--scenario` | `str` | `reach` | Task hierarchy to run (`reach`, `reach_split`, `conflict`). |
| `--priority-order` | `str` | `xyz` | Axis ranking in `conflict`, highest priority first (e.g. `xyz`, `zyx`). |
| `--record` | `str` | `None` | Record the run to a file, e.g. `docs/media/run.gif`. |
| `--experiment` | `str` | `default` | Run experiment benchmark (`surface_circle`, `blocked_circle`, `apf_avoidance`, `multilink_push`, `torque_wipe`, `singularity`, `baseline_comparison`, `bode`, `passivity`, `robustness`, `benchmark`, `all`). |

---

## Evaluation Benchmark Suite (Task 2.2 & ICRA 2018 Reproduction)

The repository provides a complete benchmark suite covering 10 validation experiments and real-time solver latency profiling:

```bash
# Run the entire evaluation benchmark suite sequentially (quick mode)
uv run python run_all_experiments.py --quick --device cpu

# Run specific experiment benchmarks individually:
uv run python main.py --experiment surface_circle       # Exp 1: Surface circular tracking & normal force
uv run python main.py --experiment blocked_circle       # Exp 2: Blocked circular tracking & obstacle compliance
uv run python main.py --experiment apf_avoidance        # Exp 3: Reactive APF obstacle avoidance
uv run python main.py --experiment multilink_push       # Exp 4: Multi-link disturbance rejection
uv run python main.py --experiment torque_wipe          # Exp 5: Active torque inequality constraint wiping
uv run python main.py --experiment singularity          # Exp 6: Kinematic singularity tracking & damping
uv run python main.py --experiment baseline_comparison  # Exp 7: 4-way baseline comparison (Figs 1-4)
uv run python main.py --experiment bode                 # Exp 8: Frequency response & Bode bandwidth
uv run python main.py --experiment passivity            # Exp 9: Energy tank & passivity profiling
uv run python main.py --experiment robustness           # Exp 10: Model parameter uncertainty robustness
uv run python main.py --experiment benchmark            # Solver latency profiling (< 0.5 ms)
```

---

## Telemetry & Verification

During simulation, `main.py` logs operational telemetry to the console:
- **Timestamp & Control Step Index**
- **End-Effector Tracking Error Norm ($[m]$)**
- **Maximum Commanded Joint Torque ($\max |\tau| \, [\text{N}\cdot\text{m}]$)**

At the end of the run a summary line reports the final tracking error, the peak commanded torque,
and a **torque-limit violation count**, and the full per-step telemetry is written to
`results/run.npz` (override with `--out`) for offline plotting.

### Verifying the two properties the paper claims

The controller applies **no post-hoc clipping** to its output. Equations (19)–(21) already guarantee
feasibility: the QP is bounded by $[\tau_{min} - h,\ \tau_{max} - h]$, so adding $h$ back lands
inside $[\tau_{min},\ \tau_{max}]$ by construction. Clipping would discard the QP's optimality and
priority ordering, and would mask any bug that broke the guarantee — so instead the controller
*measures* it, via `n_violations`, `n_solves` and `max_violation` (also saved into the `.npz`).

Strict priority is likewise measurable rather than assumed. After each solve,
`QPImpedanceController.priority_residuals[k]` holds

$$\left\| J_k B^{-1}\tau_{final} - J_k B^{-1}\tau_k^* \right\|$$

— how far the final solution drifted from what priority level $k$ had already decided. Strict
priority means these sit at solver tolerance.

Measured on the default scenario (Cartesian position at level 0, joint posture at level 1, 400
control steps):

| Property | Result |
| :--- | :--- |
| Torque-limit violations | **0 / 400 steps**, worst $0.0$ N·m |
| Priority-0 constraint residual | **max $1.3\times10^{-14}$**, mean $7.7\times10^{-15}$ |

The residual is at floating-point noise, so the lower-priority task provably never disturbs the
higher-priority one.

Between $t=2.0\text{s}$ and $t=2.2\text{s}$, an external force ($[10, 0, 0]\,\text{N}$) is applied
at the end-effector to demonstrate compliant interaction and recovery. Genesis links expose no
`apply_force()`, so the disturbance is injected as the equivalent joint torque $J_{lin}^T f_{ext}$;
it is drawn as a red arrow while it acts.
