"""
Diagnostic probe: establish Genesis API facts for this project by measurement, not assumption.

Answers the open questions listed in docs/PROJECT_OVERVIEW.md section 6, plus the empirical
workspace radius and wrist lever arm that docs/DESIGN_DECISIONS.md D2/D3 depend on.

Read-only with respect to the project: prints facts, changes nothing.
Run:  ./.venv/bin/python scripts/probe_genesis.py 2>/dev/null
"""

import inspect
import numpy as np
import torch
import genesis as gs

R = "RESULT"


def hdr(title):
    print(f"\n{R} {'=' * 66}")
    print(f"{R} {title}")
    print(f"{R} {'=' * 66}")


def np_(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x[0] if x.ndim >= 2 and x.shape[0] == 1 else x


gs.init(backend=gs.cpu, logging_level="warning")

# ---------------------------------------------------------------- Q: ViewerOptions field name
hdr("Q1  ViewerOptions: refresh_rate vs max_FPS")
for field in ("refresh_rate", "max_FPS"):
    try:
        gs.options.ViewerOptions(**{field: 200})
        print(f"{R}   {field:14s} ACCEPTED")
    except Exception as e:
        print(f"{R}   {field:14s} REJECTED  ({type(e).__name__}: {str(e)[:70]})")

# ---------------------------------------------------------------- Q0: model assets
hdr("Q0  panda_cylinder.xml mesh assets")
import os, pathlib, genesis as _gs
GA = pathlib.Path(_gs.__file__).parent / "assets" / "xml" / "franka_emika_panda" / "assets"
print(f"{R}   repo ./assets exists? {os.path.isdir('assets')}")
print(f"{R}   genesis bundled meshes: {GA}  (exists: {GA.is_dir()}, files: {len(list(GA.glob('*'))) if GA.is_dir() else 0})")
_src = pathlib.Path("panda_cylinder.xml").read_text()
_patched = _src.replace('meshdir="assets"', f'meshdir="{GA}"')
MODEL = pathlib.Path(os.environ["SCRATCH"]) / "panda_cylinder_patched.xml"
MODEL.write_text(_patched)
print(f"{R}   -> probing with meshdir repointed at the Genesis copy")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=2, gravity=(0.0, 0.0, -9.81)),
    rigid_options=gs.options.RigidOptions(dt=0.005, enable_collision=True, enable_joint_limit=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file=str(MODEL), pos=(0, 0, 0), quat=(1, 0, 0, 0)))
scene.build(n_envs=1)

# ---------------------------------------------------------------- Q: model structure
hdr("Q2  Model structure (panda_cylinder.xml)")
print(f"{R}   n_dofs  = {robot.n_dofs}")
print(f"{R}   n_links = {robot.n_links}")
print(f"{R}   links   = {[l.name for l in robot.links]}")
print(f"{R}   'hand' present? {'hand' in [l.name for l in robot.links]}")
print(f"{R}   links[-1] (what genesis_sim.py falls back to) = {robot.links[-1].name!r}")

# ---------------------------------------------------------------- Q: are MJCF actuator gains imported?
hdr("Q3  Internal PD gains as imported from MJCF  <-- blocker B2")
kp, kv = np_(robot.get_dofs_kp()), np_(robot.get_dofs_kv())
print(f"{R}   kp = {np.round(kp, 2)}")
print(f"{R}   kv = {np.round(kv, 2)}")
print(f"{R}   -> internal position servo ACTIVE? {bool(np.any(np.abs(kp) > 1e-9))}")
try:
    lo, hi = robot.get_dofs_force_range()
    print(f"{R}   force_range lower = {np.round(np_(lo), 2)}")
    print(f"{R}   force_range upper = {np.round(np_(hi), 2)}")
except Exception as e:
    print(f"{R}   get_dofs_force_range FAILED: {type(e).__name__}: {e}")
print(f"{R}   armature = {np.round(np_(robot.get_dofs_armature()), 4)}")
print(f"{R}   damping  = {np.round(np_(robot.get_dofs_damping()), 4)}")

# ---------------------------------------------------------------- Q: torque command entry point
hdr("Q4  Torque command API")
for name in ("control_dofs_force", "control_dofs_position", "control_dofs_velocity"):
    if hasattr(robot, name):
        try:
            sig = str(inspect.signature(getattr(robot, name)))
        except (TypeError, ValueError):
            sig = "(signature unavailable)"
        print(f"{R}   {name}{sig}")
    else:
        print(f"{R}   {name}: ABSENT")

# ---------------------------------------------------------------- Q: dynamics terms (bias / gravity)
hdr("Q5  Dynamics APIs: is there a bias term h = C(q,dq)dq + g(q)?  <-- blocker B3")
cands = sorted(m for m in dir(robot)
               if any(k in m.lower() for k in ("grav", "bias", "coriolis", "inverse_dyn", "dyn"))
               and not m.startswith("__"))
print(f"{R}   candidate members: {cands}")
for name in cands:
    try:
        attr = getattr(robot, name)
        if callable(attr):
            val = np_(attr())
            print(f"{R}   {name}() -> shape {np.shape(val)}, value {np.round(np.ravel(val)[:9], 4)}")
    except Exception as e:
        print(f"{R}   {name}() raised {type(e).__name__}: {str(e)[:70]}")

# ---------------------------------------------------------------- Q: mass matrix and Jacobian
hdr("Q6  Mass matrix and Jacobian")
B = np_(robot.get_mass_mat())
print(f"{R}   get_mass_mat shape = {B.shape}")
print(f"{R}   diag(B) = {np.round(np.diag(B), 4)}")
print(f"{R}   armature inside B? diag - armature = {np.round(np.diag(B) - np_(robot.get_dofs_armature()), 4)}")

ee = robot.get_link("tool_tip")
J = np_(robot.get_jacobian(link=ee))
print(f"{R}   get_jacobian(tool_tip) shape = {J.shape}")

# ---------------------------------------------------------------- Q: quaternion convention
hdr("Q7  Quaternion convention and quat_to_R")
quat = np_(ee.get_quat())
print(f"{R}   tool_tip get_quat() = {np.round(quat, 4)}  (norm {np.linalg.norm(quat):.4f})")
print(f"{R}   |first| = {abs(quat[0]):.4f}, |last| = {abs(quat[3]):.4f}  -> larger is likely the w component")
try:
    from genesis.utils import geom as ggeom
    fns = [f for f in dir(ggeom) if "quat" in f.lower() and ("R" in f or "mat" in f.lower())]
    print(f"{R}   genesis.utils.geom quat->matrix helpers: {fns}")
    print(f"{R}   gs.utils.geom reachable as attribute? {hasattr(gs, 'utils') and hasattr(gs.utils, 'geom')}")
except Exception as e:
    print(f"{R}   genesis.utils.geom import FAILED: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- Q: external force on a link
hdr("Q8  External disturbance API on a link")
print(f"{R}   link has apply_force? {hasattr(ee, 'apply_force')}")
print(f"{R}   force-ish link members: {[m for m in dir(ee) if 'force' in m.lower() and not m.startswith('__')]}")

# ---------------------------------------------------------------- workspace + lever arm
hdr("Q9  Empirical workspace radius and wrist lever arm  <-- decisions D2 / D3")
lo_q = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
hi_q = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
n_arm = 7

rng = np.random.default_rng(0)
samples = rng.uniform(lo_q, hi_q, size=(400, n_arm))
# hand-picked near-maximal-extension candidates
samples = np.vstack([samples,
                     np.array([0, 0, 0, -0.0698, 0, 1.5708, 0]),
                     np.array([0, -1.7628, 0, -0.0698, 0, 3.7525, 0]),
                     np.array([0, 1.7628, 0, -0.0698, 0, 0.0, 0]),
                     np.array([0, 0, 0, -0.0698, 0, 0.0, 0])])

best_r, best_q = 0.0, None
lever6 = []
for qs in samples:
    full = np.zeros(robot.n_dofs)
    full[:n_arm] = qs
    robot.set_qpos(torch.tensor(full[None, :], dtype=gs.tc_float, device=gs.device))
    p = np_(ee.get_pos())
    r = float(np.linalg.norm(p))
    if r > best_r:
        best_r, best_q = r, qs.copy()
    Jc = np_(robot.get_jacobian(link=ee))[:3, :n_arm]
    lever6.append(np.linalg.norm(Jc[:, 4:7], axis=0))  # joints 5,6,7

lever6 = np.array(lever6)
print(f"{R}   max ||p_tooltip|| over {len(samples)} configs = {best_r:.4f} m")
print(f"{R}     at q = {np.round(best_q, 4)}")
print(f"{R}   effective lever arm ||J_lin[:, j]|| for wrist joints (m):")
for k, j in enumerate((5, 6, 7)):
    col = lever6[:, k]
    print(f"{R}     joint{j}:  median {np.median(col):.4f}   max {col.max():.4f}")
worst = lever6.max()
print(f"{R}   -> max wrist lever arm = {worst:.4f} m")
print(f"{R}   -> tip force saturating a 12 Nm wrist joint = {12.0 / worst:.1f} N  (at that worst-case pose)")
print(f"{R}   -> position error saturating it at K=15000 N/m = {1000 * (12.0 / worst) / 15000:.2f} mm")
print(f"{R}   -> position error saturating it at K=400   N/m = {1000 * (12.0 / worst) / 400:.1f} mm")

hdr("PROBE COMPLETE")
