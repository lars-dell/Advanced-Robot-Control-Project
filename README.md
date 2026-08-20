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
├── main.py                     # Entry point: simulation control loop, task hierarchy, telemetry logging
├── pyproject.toml              # Project metadata & dependencies (genesis-world, casadi, qpsolvers, torch)
├── panda_cylinder.xml          # Franka Emika Panda 7-DOF MJCF robot model
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py      # Abstract base class interface for robot controllers
│   └── qp_impedance.py         # Multi-Priority Hierarchical QP Cartesian Impedance Controller
├── solvers/
│   ├── __init__.py
│   └── casadi_qp_solver.py     # Pre-compiled parametric CasADi QP solver module (qpOASES / OSQP)
├── tasks/
│   ├── __init__.py
│   ├── base_task.py            # Abstract base class for prioritizable control tasks
│   ├── cartesian_task.py       # Cartesian 3D position / 6D pose impedance task (VMC & Full Impedance)
│   ├── posture_task.py         # Joint-space null-space posture stiffness task
│   └── task_stack.py           # Priority-ordered task stack manager
├── utils/
│   ├── __init__.py
│   └── math_utils.py           # Dynamically consistent pseudo-inverses, null-space projectors, 6D pose error
└── envs/
    ├── __init__.py
    └── genesis_sim.py          # Genesis physics simulator wrapper & PyTorch-NumPy state conversion
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
| `conflict` | `x > y > z > posture` | out-of-reach point + z sinusoid | Objectives compete; the hierarchy decides who is sacrificed. **Not yet tuned to a steady state.** |

`reach` and `reach_split` both converge to `0.0000 m`. That equivalence is the point of
`reach_split`: decomposing one 3-D objective into three ranked 1-D objectives reproduces the
undecomposed result exactly, which is only true if the cascade's priority constraints are correct.
Priority ordering has no *visible* effect there because nothing has to be given up — that requires
a conflicting reference.

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
