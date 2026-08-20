"""
Figures for the conflict scenario: does the priority order decide who is sacrificed,
and do the torque constraints stay satisfied?

Usage:
    uv run python main.py --no-vis --time 7 --scenario conflict --priority-order xyz \
        --out results/conflict_xyz.npz
    uv run python main.py --no-vis --time 7 --scenario conflict --priority-order zyx \
        --out results/conflict_zyx.npz
    uv run python scripts/plot_results.py
"""

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of the reference palette, in fixed order. Two conditions are being
# compared (a priority ordering), so this is an identity encoding, not magnitude.
C_XYZ = "#2a78d6"   # blue   - x on top
C_ZYX = "#eb6834"   # orange - z on top

INK = "#1a1a19"
INK_MUTED = "#6b6a63"
GRID = "#e3e2dc"
LIMIT = "#b3261e"

AXES = ("x", "y", "z")


def _style(ax):
    """Recessive axes and grid: the data carries the ink, the frame does not."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def plot_task_errors(xyz, zyx, out):
    """One panel per axis, two lines per panel. The crossover between panels is the result."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    t = xyz["time"]

    for ax, axis_name in zip(axes, AXES):
        key = f"err_{axis_name}_axis"
        ax.plot(t, xyz[key], color=C_XYZ, linewidth=2.0, zorder=3)
        ax.plot(t, zyx[key], color=C_ZYX, linewidth=2.0, zorder=3)
        _style(ax)
        ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
        ax.set_title(f"{axis_name} objective", color=INK, fontsize=11, pad=8, loc="left")

        # Direct-label the steady-state value of each line. Two series, so label both -- but the
        # values can land on top of each other (they do in the y panel, where both are ~0), so
        # push the labels apart vertically when they are closer than a readable gap.
        tail = slice(int(0.75 * len(t)), None)
        vals = [(float(np.mean(xyz[key][tail])), C_XYZ),
                (float(np.mean(zyx[key][tail])), C_ZYX)]
        vals.sort(key=lambda vc: vc[0])
        # Stack upward, never downward: a value of 0.000 sits on the axis, and pushing its label
        # below would collide with the tick labels.
        offsets = [6, 19] if (vals[1][0] - vals[0][0]) < 0.06 else [6, 6]
        for (v, color), dy in zip(vals, offsets):
            ax.annotate(f"{v:.3f}", xy=(t[-1], v), xytext=(-6, dy),
                        textcoords="offset points", ha="right", va="bottom",
                        color=color, fontsize=9, fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))

    axes[0].set_ylabel("tracking error (m)", color=INK_MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=C_XYZ, linewidth=2.0, label="priority  x > y > z"),
               plt.Line2D([], [], color=C_ZYX, linewidth=2.0, label="priority  z > y > x")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK, ncols=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Priority order decides which objective absorbs the shortfall",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=1.04)
    fig.text(0.008, 0.955,
             "Target [1.30, 0.15, 0.75] m: x is out of reach, y and z are not. "
             "Same controller, gains and target in both runs.",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_torque_utilisation(runs, out):
    """
    Share of control steps each joint spends pinned at its torque bound.

    The claim being evidenced is "the commanded torque never exceeds a limit, with no clipping".
    That headline is a number, not a picture, so it lives in the title. The chart carries the
    supporting question -- how hard is each joint working -- as one bar per joint.

    Three forms were tried first and discarded: seven overlapping time series (unreadable, the
    torque chatters between zero and the bound); a joint-by-time heatmap (legible but answering a
    question nobody asked); and PEAK utilisation per joint, which turned out to be exactly 1.0 for
    all seven joints in both runs and therefore carried no information at all.

    Normalising by each joint's own limit puts the 87 N.m proximal joints and the 12 N.m wrist on
    one scale, so a single reference line at 1.0 is the bound for all of them.
    """
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    n_j = 7
    idx = np.arange(1, n_j + 1)
    width = 0.38

    for k, (label, d, color) in enumerate(runs):
        lim = np.maximum(np.abs(d["tau_min"]), np.abs(d["tau_max"]))
        util = np.abs(d["torques"]) / lim
        frac = (util >= 0.999).mean(axis=0) * 100.0     # % of control steps spent at the bound
        off = (k - 0.5) * width
        ax.bar(idx + off, frac, width * 0.92, color=color, zorder=3, label=label)

    # Direct-label only the extremes, not all fourteen bars.
    for k, (label, d, color) in enumerate(runs):
        lim = np.maximum(np.abs(d["tau_min"]), np.abs(d["tau_max"]))
        frac = ((np.abs(d["torques"]) / lim) >= 0.999).mean(axis=0) * 100.0
        j = int(np.argmax(frac))
        ax.annotate(f"{frac[j]:.0f}%", xy=(j + 1 + (k - 0.5) * width, frac[j]),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=color, fontsize=9, fontweight="bold", zorder=5)

    _style(ax)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"J{j}\n{int(l)} N·m" for j, l in
                        zip(idx, np.maximum(np.abs(runs[0][1]["tau_min"]),
                                            np.abs(runs[0][1]["tau_max"])))])
    ax.set_ylabel("% of control steps at the torque bound", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", ncols=2)

    v = sum(int(d["n_violations"]) for _, d, _ in runs)
    n = sum(int(d["n_solves"]) for _, d, _ in runs)
    fig.suptitle(f"Joints run at their limits — and cross them {v} times in {n:,} steps",
                 color=INK, fontsize=13, x=0.008, ha="left", y=1.05)
    fig.text(0.008, 0.955,
             "Every joint reaches its bound at some point, so peak utilisation is 1.0 everywhere; "
             "what differs is how long each spends there. No clipping is applied — feasibility "
             "comes from the QP constraints (eq. 19–21).",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", default="results/conflict_xyz.npz")
    ap.add_argument("--zyx", default="results/conflict_zyx.npz")
    ap.add_argument("--outdir", default="docs/media")
    a = ap.parse_args()

    xyz = np.load(a.xyz, allow_pickle=True)
    zyx = np.load(a.zyx, allow_pickle=True)
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(plot_task_errors(xyz, zyx, outdir / "conflict_task_errors.png"))
    print(plot_torque_utilisation(
        [("priority  x > y > z", xyz, C_XYZ), ("priority  z > y > x", zyx, C_ZYX)],
        outdir / "conflict_torques.png"))


if __name__ == "__main__":
    main()
