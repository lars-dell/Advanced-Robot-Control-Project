"""
Controllers package for Multi-Priority Cartesian Impedance Control and baseline comparisons.
"""

import inspect
from typing import List, Optional, Union, Dict, Any

from controllers.base_controller import BaseController
from controllers.qp_impedance import QPImpedanceController
from controllers.classical_transpose import ClassicalTransposeController
from controllers.saturated_algebraic import SaturatedAlgebraicController
from controllers.weighted_qp import WeightedQPController

CONTROLLER_REGISTRY: Dict[str, Any] = {
    "hierarchical_qp": QPImpedanceController,
    "qp_impedance": QPImpedanceController,
    "hierarchical": QPImpedanceController,
    "hoffman": QPImpedanceController,
    "proposed": QPImpedanceController,
    "weighted_qp": WeightedQPController,
    "weighted": WeightedQPController,
    "saturated_algebraic": SaturatedAlgebraicController,
    "algebraic": SaturatedAlgebraicController,
    "nullspace": SaturatedAlgebraicController,
    "saturated": SaturatedAlgebraicController,
    "classical_transpose": ClassicalTransposeController,
    "transpose": ClassicalTransposeController,
    "classical": ClassicalTransposeController,
}

CANONICAL_CONTROLLERS: List[str] = [
    "hierarchical_qp",
    "weighted_qp",
    "saturated_algebraic",
    "classical_transpose",
]


def list_controllers() -> List[str]:
    """Return list of canonical controller names."""
    return list(CANONICAL_CONTROLLERS)


def make_controller(
    name: Union[str, BaseController] = "hierarchical_qp",
    n_dofs: int = 7,
    **kwargs: Any
) -> BaseController:
    """
    Factory creating a BaseController instance by name with keyword arguments.

    If an existing BaseController instance is passed, it is returned directly.

    Supported names/aliases:
        - 'hierarchical_qp' (default, 'qp_impedance', 'hoffman', 'proposed'):
          Multi-Priority Hierarchical QP Cartesian Impedance Controller (Hoffman et al. ICRA 2018 Eq. 18)
        - 'weighted_qp' ('weighted'):
          Single-Level Weighted-Sum QP Controller
        - 'saturated_algebraic' ('algebraic', 'nullspace'):
          Dynamically Consistent Null-Space Controller with Post-Hoc Saturation (Eq. 10)
        - 'classical_transpose' ('transpose', 'classical'):
          Classical Jacobian Transpose Impedance Controller (Eq. 9)
    """
    if isinstance(name, BaseController):
        return name

    key = str(name).strip().lower().replace("-", "_")
    if key not in CONTROLLER_REGISTRY:
        valid = ", ".join(CANONICAL_CONTROLLERS)
        raise ValueError(f"Unknown controller '{name}'. Valid canonical controllers: {valid}")

    cls = CONTROLLER_REGISTRY[key]

    # Map aliases in kwargs
    ctrl_kwargs = dict(kwargs)
    if "solver" in ctrl_kwargs and "solver_name" not in ctrl_kwargs:
        ctrl_kwargs["solver_name"] = ctrl_kwargs.pop("solver")

    # Filter arguments to match target controller __init__ signature
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in ctrl_kwargs.items() if k in valid_params}
    if "n_dofs" in valid_params and "n_dofs" not in filtered_kwargs:
        filtered_kwargs["n_dofs"] = n_dofs

    return cls(**filtered_kwargs)


__all__ = [
    "BaseController",
    "QPImpedanceController",
    "ClassicalTransposeController",
    "SaturatedAlgebraicController",
    "WeightedQPController",
    "make_controller",
    "list_controllers",
    "CANONICAL_CONTROLLERS",
]

