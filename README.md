# Multi-Priority Cartesian Impedance Control (QP)

Implementation and evaluation of a Quadratic Programming (QP)-based hierarchical Cartesian impedance controller with joint-torque constraints for redundant manipulators.

Based on the paper:
> **Multi-Priority Cartesian Impedance Control Based on Quadratic Programming Optimization**  
> E. M. Hoffman, A. Laurenzi, L. Muratore, N. G. Tsagarakis, and D. G. Caldwell (*ICRA 2018*)

---

## Objectives & Requirements
* **Platform:** Redundant manipulator (Franka Emika Panda) in [Genesis](https://genesis-world.readthedocs.io/).
* **Method:** Formulate and solve hierarchical Quadratic Programs (QPs) for multiple Cartesian impedance tasks with priority ordering.
* **Constraints:** Enforce explicit joint-torque limit constraints in the controller.
* **Evaluation:** Assess task behavior, priority resolution, and commanded torques under conflicting objectives and torque saturation.

---

## Setup & Execution

### 1. Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install genesis-world torch numpy matplotlib qpsolvers
```

```
