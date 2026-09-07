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
import sys
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The README figures follow the house style established in scripts/plot_results.py: recessive
# frame, conclusion as the title, setup as a subtitle, and steady-state values labelled directly
# on the traces. Reuse its palette rather than re-deriving one.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_results import INK, INK_MUTED, GRID, _style  # noqa: E402

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

# README figures are displayed at roughly 800 px rather than in a 3.5 in column.
README_W, README_H = 7.2, 3.0
README_RC = {
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
}


def _headline(fig, title: str, subtitle: str) -> None:
    """Conclusion as the title, setup underneath: the figure should read without the body text."""
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.008, ha="left", y=1.04)
    fig.text(0.008, 0.955, subtitle, color=INK_MUTED, fontsize=9, ha="left")


def _label_last(ax, x, y, color, fmt="{:.3f}", dy=6):
    """Direct-label a trace at its right-hand end, so no legend lookup is needed to read a value."""
    ax.annotate(fmt.format(y), xy=(x, y), xytext=(-6, dy), textcoords="offset points",
                ha="right", va="bottom", color=color, fontsize=9, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))


def _load(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def figure_primary_task(data: Dict[str, np.ndarray], out: pathlib.Path, p0_axis: str,
                        headline: bool = False) -> None:
    """
    Priority-0 axis tracking for every controller against the shared reference.

    `headline=True` produces the self-explaining README variant: conclusion as the title and values
    labelled on the traces. The report variant stays bare, because its LaTeX caption does that job
    and a title inside the image would duplicate it.
    """
    fig, ax = plt.subplots(figsize=(8.4, 3.6) if headline else (COL_W, 2.30))
    idx = AXIS_INDEX[p0_axis]

    ref_key = "hierarchical_qp_ee_pos_des"
    t = data["hierarchical_qp_time"]
    ax.plot(t, data[ref_key][:, idx], color="k", lw=1.0, ls=(0, (4, 2)), label="Reference", zorder=1)

    for key, label, color, style in CONTROLLERS:
        ax.plot(data[f"{key}_time"], data[f"{key}_ee_pos"][:, idx],
                color=color, ls=style, label=label, zorder=2)

    if headline:
        # Three of the four settle within ~2 cm of each other, so labelling all four just stacks
        # unreadable text. Label the best and worst only: the gap between them is the headline, and
        # the fact that the other two sit inside it is the supporting point.
        finals = [(float(np.mean(data[f"{key}_ee_pos"][int(0.75 * len(data[f"{key}_time"])):, idx])),
                   color) for key, _lbl, color, _st in CONTROLLERS]
        for v, color in (min(finals), max(finals)):
            _label_last(ax, data["hierarchical_qp_time"][-1], v, color, dy=4)
        _headline(fig,
                  "Tracking barely separates the controllers \u2014 feasibility is what does",
                  "Highest-priority axis against a reference outside the workspace. All four stall "
                  "short of it; the difference that matters is in the torque figure, not this one.")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{p0_axis} position [m]")
    ax.grid(alpha=0.3, lw=0.4)
    # Legend above the axes: at column width every in-axes position covered real data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, columnspacing=1.1, handlelength=1.9)
    fig.tight_layout(rect=(0, 0, 1, 0.86) if headline else None, pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_torque_bounds(data: Dict[str, np.ndarray], out: pathlib.Path,
                         window: int = 50, headline: bool = False) -> None:
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
    fig, ax = plt.subplots(figsize=(8.4, 3.6) if headline else (COL_W, 2.30))

    for key, label, color, style in CONTROLLERS:
        tau = np.abs(data[f"{key}_torques"])
        lim = np.abs(data[f"{key}_tau_max"])
        u = np.max(tau / lim, axis=1)
        t = data[f"{key}_time"]

        ax.plot(t, u, color=color, lw=0.35, alpha=0.28, zorder=1)
        # Rolling maximum: the envelope of the chattering signal.
        pad = np.pad(u, (window - 1, 0), mode="edge")
        env = np.max(np.lib.stride_tricks.sliding_window_view(pad, window), axis=1)
        ax.plot(t, env, color=color, ls=style, label=label, lw=2.0, zorder=3)

    ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(0.99, 1.0, " torque limit", ha="right", va="bottom", fontsize=6.5,
            transform=ax.get_yaxis_transform())
    lo, hi = ax.get_ylim()
    if hi > 1.0:
        ax.axhspan(1.0, hi, color="red", alpha=0.07, zorder=0)
    ax.set_ylim(0.0, min(hi, 1.45))

    if headline:
        _headline(fig,
                  "Only the QP rides the torque limit without crossing it",
                  "Worst-joint utilisation max|\u03c4|/\u03c4_max: 1.0 is the bound and the shaded "
                  "band is infeasible. Bold lines are a rolling maximum of the faint per-step signal.")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\max_j |\tau_j| / \tau_{\max,j}$")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, columnspacing=1.1, handlelength=1.9)
    fig.tight_layout(rect=(0, 0, 1, 0.86) if headline else None, pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_priority_residual(data: Dict[str, np.ndarray], out: pathlib.Path) -> None:
    """
    Priority residual over time for the cascade against a weighted-sum QP.

    The separation is far too large for a linear axis: the cascade sits near 1e-11 while the
    weighted controller, solving the identical tasks with the identical solver, sits near 1e0.
    """
    series = [
        ("weighted_qp", "single-level weighted-sum QP", "#eb6834"),
        ("hierarchical_qp", "proposed hierarchical cascade, eq. (18)", "#2a78d6"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for key, label, color in series:
        r = np.abs(data[f"{key}_priority_residuals"])
        if r.ndim == 2 and r.shape[1] > 0:
            r = r.max(axis=1)                      # worst level at each step
        r = np.maximum(r, 1e-18)                   # zeros must stay representable on a log axis
        t = data[f"{key}_time"]
        ax.semilogy(t, r, color=color, linewidth=1.7, zorder=3, label=label)
        med = float(np.median(r[len(r) // 4:]))    # steady-state median, past the transient
        _label_last(ax, t[-1], med, color, fmt="{:.1e}", dy=8)

    _style(ax)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(r"$\max_k\ \|J_k B^{-1}(\tau-\tau_k^*)\|$", color=INK_MUTED, fontsize=9)
    ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK)
    _headline(fig,
              "A hierarchy enforces priority exactly; a weighting only approximates it",
              "Same tasks, same solver, same scenario \u2014 only the formulation differs. "
              "Logarithmic axis: the two are eleven orders of magnitude apart.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def figure_corridor(out: pathlib.Path, results_dir: str = "results") -> None:
    """
    End-effector height against the commanded corridor.

    The ablated run is the control: without it a tool that stays inside proves nothing, since the
    trajectory might simply never have left.
    """
    runs = [
        ("fig_corridor_off.npz",     "constraint removed (control)", "#b3261e", "-."),
        ("fig_corridor_barrier.npz", "transpose, eq. (9) + barrier", "#eb6834", ":"),
        ("fig_corridor_qp.npz",      "hierarchical QP, hard inequality", "#2a78d6", "-"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    zmin = zmax = None
    excursions = []
    for fname, label, color, style in runs:
        d = _load(str(pathlib.Path(results_dir) / fname))
        t, z = d["time"], d["ee_pos"][:, 2]
        if zmin is None:
            zmin, zmax = float(d["z_min"]), float(d["z_max"])
            ax.axhspan(zmin, zmax, color="#2a78d6", alpha=0.07, zorder=0)
            for bnd in (zmin, zmax):
                ax.axhline(bnd, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=2)
            ax.plot(t, d["ee_pos_des"][:, 2], color=INK_MUTED, linewidth=1.3,
                    linestyle=(0, (4, 2)), zorder=2, label="commanded reference")
        ax.plot(t, z, color=color, linestyle=style, linewidth=1.9, zorder=3, label=label)
        # How far outside the corridor each run strayed -- the quantity in question. Collected and
        # drawn as a stacked block below: every trace ends near z = 0.45, so labelling each at its
        # own endpoint puts all three on top of each other.
        excursions.append((max(float(z.max() - zmax), float(zmin - z.min()), 0.0), color))

    for row, (exc, color) in enumerate(excursions):
        ax.annotate(f"{exc * 1000:.2f} mm outside", xy=(0.988, 0.20 - 0.075 * row),
                    xycoords="axes fraction", ha="right", va="center",
                    color=color, fontsize=9.5, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))

    _style(ax)
    ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("end-effector height z (m)", color=INK_MUTED, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=2, frameon=False,
              fontsize=9, labelcolor=INK)
    _headline(fig,
              "A constraint holds the corridor; a penalty only leans against it",
              "Shaded band is z \u2208 [0.35, 0.55] m, which the circular reference deliberately "
              "leaves. Removing the constraint entirely is the control.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def figure_blocked_circle(out: pathlib.Path,
                          npz: str = "results/fig_blocked_circle.npz",
                          window: int = 25) -> None:
    """
    Blocked deflection against the contact force it produces.

    An obstacle the controller knows nothing about holds the tool short of its commanded position.
    If the arm is rendering the impedance it was told to, the force it pushes with should rise and
    fall with that deflection -- which is the spring law, plotted directly rather than inferred.

    The force is smoothed over `window` steps because the simulator reports contact in bursts
    during sliding (non-zero on ~46% of steps); the raw trace is spiky but its average is correct.
    See the contact discussion in docs/EVALUATION_FRAMEWORK.md.
    """
    d = _load(npz)
    # Skip the approach: the arm starts 133 mm from the reference and lands with a ~90 N impact,
    # which compresses the sustained-contact behaviour the figure is about into the bottom fifth
    # of the axes. The transient is characterised in the text, not here.
    t_start = 0.8
    keep = np.asarray(d["time"]) >= t_start
    d = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v)
         for k, v in d.items()}
    t = d["time"]
    dx = (d["ee_pos_des"][:, 0] - d["ee_pos"][:, 0]) * 1000.0          # mm, task-space deflection
    fx = np.abs(d["contact_force"][:, 0])
    fx_s = np.convolve(fx, np.ones(window) / window, mode="same")      # ~125 ms moving average
    r = float(np.corrcoef(dx, fx_s)[0, 1])

    # Mark when the reference commands the tool past the obstacle. This is geometry, not a
    # threshold on a noisy signal: the obstacle's near face is at x = 0.415 m and the tool stalls
    # at 0.413 m, so a commanded x beyond that cannot be reached. Inside these windows the
    # deflection reaches 133 mm; outside them it never exceeds 24 mm.
    #
    # Note this is NOT obstacle detection or avoidance -- the controller has no model of the
    # obstacle and no avoidance term. It is commanded straight through and is simply blocked.
    x_block = 0.413
    engaged = d["ee_pos_des"][:, 0] > x_block

    c_dx, c_f = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    edges = np.diff(np.concatenate(([0], engaged.astype(int), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for i, (a, b) in enumerate(zip(starts, ends)):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=INK_MUTED, alpha=0.10, zorder=0,
                   label="reference commands past the obstacle" if i == 0 else None)
    ax.plot(t, dx, color=c_dx, linewidth=1.9, zorder=3, label="blocked deflection $\\Delta x$")
    ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("deflection $\\Delta x$ (mm)", color=c_dx, fontsize=9)
    ax.tick_params(axis="y", colors=c_dx)
    _style(ax)

    ax2 = ax.twinx()
    ax2.plot(t, fx_s, color=c_f, linewidth=1.9, linestyle="--", zorder=3,
             label="contact force $|F_x|$")
    ax2.set_ylabel("contact force $|F_x|$ (N)", color=c_f, fontsize=9)
    ax2.tick_params(axis="y", colors=c_f, labelsize=9, length=0)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(GRID)

    handles = [plt.Line2D([], [], color=c_dx, lw=1.9, label="blocked deflection $\\Delta x$"),
               plt.Line2D([], [], color=c_f, lw=1.9, ls="--", label="contact force $|F_x|$"),
               plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=0.18,
                             label="reference commands past the obstacle")]
    ax.legend(handles=handles, loc="upper center", ncols=3, frameon=False,
              fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.02))
    _headline(fig,
              "The force follows the deflection: the arm renders the spring it was commanded",
              f"Shaded: the reference commands past the obstacle face. The arm has no model "
              f"of it and no avoidance term \u2014 it is blocked, not evading. "
              f"Correlation {r:.2f}; force is a {window}-step moving average.")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def figure_singularity(out: pathlib.Path,
                       npz: str = "results/fig_singularity.npz",
                       sigma_near: float = 1e-2) -> None:
    """
    Jacobian conditioning against torque utilisation as the arm extends toward a singularity.

    Deliberately a single controller. Overlaying the classical laws would invite the reading that
    they blow up here, and they do not: the transpose law inverts nothing so it never approaches a
    singularity, and our null-space implementation is damped and clipped. They also end up in
    different configurations, so "at the same sigma_min" is not a well-defined comparison. The
    cross-controller numbers belong in prose, where that can be said. See EVALUATION_FRAMEWORK.md
    section 2.3.
    """
    d = _load(npz)
    t = d["time"]
    sm = np.asarray(d["min_singular"], dtype=float)
    u = (np.abs(d["torques"]) / np.abs(d["tau_max"])).max(axis=1)
    near = sm < sigma_near

    c_s, c_u = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    edges = np.diff(np.concatenate(([0], near.astype(int), [0])))
    for i, (a, b) in enumerate(zip(np.where(edges == 1)[0], np.where(edges == -1)[0])):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=INK_MUTED, alpha=0.12, zorder=0)

    ax.semilogy(t, np.maximum(sm, 1e-6), color=c_s, linewidth=1.9, zorder=3)
    ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(r"$\sigma_{\min}(J)$", color=c_s, fontsize=9)
    ax.tick_params(axis="y", colors=c_s)
    _style(ax)
    i_min = int(np.argmin(sm))
    ax.annotate(f"{sm[i_min]:.1e}", xy=(t[i_min], sm[i_min]), xytext=(4, -12),
                textcoords="offset points", color=c_s, fontsize=9, fontweight="bold", zorder=5)

    ax2 = ax.twinx()
    ax2.plot(t, u, color=c_u, linewidth=1.9, linestyle="--", zorder=3)
    ax2.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax2.set_ylabel(r"torque utilisation $\max_j|\tau_j|/\tau_{\max,j}$", color=c_u, fontsize=9)
    ax2.set_ylim(0, 1.25)
    ax2.tick_params(axis="y", colors=c_u, labelsize=9, length=0)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.annotate("torque limit", xy=(t[-1], 1.0), xytext=(-4, 4), textcoords="offset points",
                 ha="right", color=INK_MUTED, fontsize=8.5)

    handles = [plt.Line2D([], [], color=c_s, lw=1.9, label=r"$\sigma_{\min}(J)$"),
               plt.Line2D([], [], color=c_u, lw=1.9, ls="--", label="torque utilisation"),
               plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=0.2,
                             label=fr"near-singular ($\sigma_{{\min}}<10^{{-2}}$)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=3,
              frameon=False, fontsize=9, labelcolor=INK)
    _headline(fig,
              "The torque bounds hold as the arm drives into a near-singular configuration",
              f"\u03c3_min falls to {sm.min():.0e} and utilisation touches 1.0 without crossing it. "
              f"Only the QP is shown: the classical laws do not fail here.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/compare_conflict.npz",
                    help="Telemetry from `main.py --exp compare`")
    ap.add_argument("--out-dir", default="docs/report/figures")
    ap.add_argument("--readme", action="store_true",
                    help="Also write the two README figures into docs/media/")
    args = ap.parse_args()

    data = _load(args.npz)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p0_axis = str(data["priority_order"][0])

    figure_primary_task(data, out_dir / "fig_primary_task.png", p0_axis)
    figure_torque_bounds(data, out_dir / "fig_torque_bounds.png")

    if args.readme:
        media = pathlib.Path("docs/media")
        media.mkdir(parents=True, exist_ok=True)
        figure_priority_residual(data, media / "priority_residual.png")
        figure_corridor(media / "corridor_height.png")
        figure_blocked_circle(media / "blocked_circle_force.png")
        figure_singularity(media / "singularity.png")
        # Regenerate rather than copy: the README variants carry a conclusion title and direct
        # labels, which would duplicate the LaTeX caption in the report versions.
        figure_primary_task(data, media / "fig_primary_task.png", p0_axis, headline=True)
        figure_torque_bounds(data, media / "fig_torque_bounds.png", headline=True)


if __name__ == "__main__":
    main()
