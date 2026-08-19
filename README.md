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
# Run simulation with 3D GUI viewer (default: 20.0 seconds on GPU backend)
uv run python main.py

# Run headless simulation (no visualizer window)
uv run python main.py --no-vis

# Run simulation on CPU backend for 25 seconds
uv run python main.py --device cpu --time 25.0
```

### CLI Command Line Arguments for `main.py`
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--time` | `float` | `5.0` | Total simulation duration in seconds. |
| `--dt` | `float` | `0.005` | Control loop timestep in seconds (200 Hz). |
| `--no-vis` | `flag` | `False` | Run in headless mode without 3D viewer window. |
| `--device` | `str` | `gpu` | Genesis physics backend (`cpu` or `gpu`). |

---

## Telemetry & Verification

During simulation, `main.py` logs operational telemetry to the console:
- **Timestamp & Control Step Index**
- **End-Effector Tracking Error Norm ($[m]$)**
- **Maximum Commanded Joint Torque ($\max |\tau| \, [\text{N}\cdot\text{m}]$)**

Between $t=8.0\text{s}$ and $t=9.0\text{s}$, an external 3D force perturbation ($[10, 0, 0]\,\text{N}$) is applied to the end-effector link to demonstrate compliant interaction and robust trajectory recovery under joint-torque constraints.
