"""
Generates the report figures from saved comparison telemetry.

One panel per figure, sized for a single IEEE conference column (3.5 in wide). The combined 2x2
grid written by compare_controllers.py is fine on screen but illegible once scaled into a column,
so the report uses these instead.

Usage:
    uv run python scripts/make_report_figures.py [--npz results/compare_conflict.npz]
"""

import argparse
import pathlib
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE conference single column is 3.5 in; keep type at ~8 pt after placement.
COL_W = 3.5
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "figure.dpi": 300,
})

# Ordered worst-to-best so the proposed method draws on top.
CONTROLLERS = [
    ("classical_transpose", "Transpose, eq. (9)", "#c44e52", "-."),
    ("saturated_algebraic", "Null-space, eq. (10)", "#8172b2", ":"),
    ("weighted_qp",         "Weighted-sum QP",     "#ccb974", "--"),
    ("hierarchical_qp",     "Proposed, eq. (18)",  "#4c72b0", "-"),
]

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _load(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def figure_primary_task(data: Dict[str, np.ndarray], out: pathlib.Path, p0_axis: str) -> None:
    """Priority-0 axis tracking for every controller against the shared reference."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.30))
    idx = AXIS_INDEX[p0_axis]

    ref_key = "hierarchical_qp_ee_pos_des"
    t = data["hierarchical_qp_time"]
    ax.plot(t, data[ref_key][:, idx], color="k", lw=1.0, ls=(0, (4, 2)), label="Reference", zorder=1)

    for key, label, color, style in CONTROLLERS:
        ax.plot(data[f"{key}_time"], data[f"{key}_ee_pos"][:, idx],
                color=color, ls=style, label=label, zorder=2)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{p0_axis} position [m]")
    ax.grid(alpha=0.3, lw=0.4)
    # Legend above the axes: at column width every in-axes position covered real data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, columnspacing=1.1, handlelength=1.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_torque_bounds(data: Dict[str, np.ndarray], out: pathlib.Path,
                         window: int = 50) -> None:
    """
    Worst-joint torque utilisation over time, for every controller.

    A raw torque trace is unusable here: the cascade switches its active set every step, so the
    time series fills as a solid band and hides the very thing the figure has to show. Instead we
    plot the normalised worst-joint utilisation

        u(t) = max_j |tau_j(t)| / tau_max,j

    which puts all seven joints and all four controllers on one axis. u = 1 is the bound exactly:
    below is feasible, above is a violation, and u riding at 1 is the constraint being *active*
    rather than merely respected. The faint line is the raw per-step value and the bold line a
    rolling maximum over `window` steps, which is what makes the envelope legible through the
    chattering.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.30))

    for key, label, color, style in CONTROLLERS:
        tau = np.abs(data[f"{key}_torques"])
        lim = np.abs(data[f"{key}_tau_max"])
        u = np.max(tau / lim, axis=1)
        t = data[f"{key}_time"]

        ax.plot(t, u, color=color, lw=0.35, alpha=0.28, zorder=1)
        # Rolling maximum: the envelope of the chattering signal.
        pad = np.pad(u, (window - 1, 0), mode="edge")
        env = np.max(np.lib.stride_tricks.sliding_window_view(pad, window), axis=1)
        ax.plot(t, env, color=color, ls=style, label=label, zorder=2)

    ax.axhline(1.0, color="k", lw=1.0, ls="--", zorder=3)
    ax.text(0.99, 1.0, " torque limit", ha="right", va="bottom", fontsize=6.5,
            transform=ax.get_yaxis_transform())
    lo, hi = ax.get_ylim()
    if hi > 1.0:
        ax.axhspan(1.0, hi, color="red", alpha=0.07, zorder=0)
    ax.set_ylim(0.0, min(hi, 1.45))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\max_j |\tau_j| / \tau_{\max,j}$")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, columnspacing=1.1, handlelength=1.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/compare_conflict.npz",
                    help="Telemetry from `main.py --exp compare`")
    ap.add_argument("--out-dir", default="docs/report/figures")
    args = ap.parse_args()

    data = _load(args.npz)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p0_axis = str(data["priority_order"][0])

    figure_primary_task(data, out_dir / "fig_primary_task.png", p0_axis)
    figure_torque_bounds(data, out_dir / "fig_torque_bounds.png")


if __name__ == "__main__":
    main()
