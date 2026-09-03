"""
Genesis Simulator Wrapper Environment.

Handles scene setup, robot loading (panda_cylinder.xml), physics stepping, external disturbance application,
and state variable extraction with automatic PyTorch tensor to NumPy array conversion for CasADi compatibility.
"""

from typing import Dict, Optional, Tuple, Any, List
from collections import deque
import os
import pathlib
import tempfile
import numpy as np
import torch
import genesis as gs
import mujoco


def resolve_model(model_xml: str) -> str:
    """
    Resolve an MJCF path to one whose meshes actually exist.

    `panda_cylinder.xml` declares meshdir="assets", but this repository ships no assets/ directory
    (the Panda meshes are ~31 MB and are deliberately not vendored). Genesis installs them with the
    package, so if there is no local assets/ we rewrite meshdir to point at the installed copy and
    hand Genesis the patched file.

    Raises instead of falling back to a different robot: silently swapping models produces results
    for a machine nobody chose.
    """
    src = pathlib.Path(model_xml)
    if not src.is_file():
        # Not a local file: assume it is a Genesis built-in asset path and let Genesis resolve it.
        return model_xml

    text = src.read_text()
    if 'meshdir="assets"' not in text:
        return str(src.resolve())

    if (src.parent / "assets").is_dir():
        return str(src.resolve())  # vendored meshes present, nothing to do

    bundled = pathlib.Path(gs.__file__).parent / "assets" / "xml" / "franka_emika_panda" / "assets"
    if not bundled.is_dir():
        raise FileNotFoundError(
            f"{src} needs meshes in '{src.parent / 'assets'}' but that directory does not exist, "
            f"and the Genesis-bundled copy was not found at '{bundled}' either. "
            f"Vendor the meshes into ./assets/ or reinstall genesis-world."
        )

    patched = pathlib.Path(tempfile.gettempdir()) / f"{src.stem}_resolved.xml"
    patched.write_text(text.replace('meshdir="assets"', f'meshdir="{bundled}"'))
    return str(patched)


def _tensor_to_numpy(x: Any) -> np.ndarray:
    """
    Converts a PyTorch tensor or nested tensor to a clean NumPy array.

    Removes any singleton environment dimensions if present.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        # Squeeze leading single-env dimension (1, N, ...) -> (N, ...)
        if x_np.ndim >= 2 and x_np.shape[0] == 1:
            x_np = np.squeeze(x_np, axis=0)
        return x_np
    elif isinstance(x, np.ndarray):
        if x.ndim >= 2 and x.shape[0] == 1:
            return np.squeeze(x, axis=0)
        return x
    return np.asarray(x)


class GenesisSim:
    """
    Wrapper class for the Genesis physics simulator.

    Manages scene lifecycle, loads Franka Emika Panda from MJCF, handles low-level torque application,
    applies external disturbance forces, and extracts robot dynamics state as NumPy arrays.
    """

    def __init__(
        self,
        model_xml: str = "panda_cylinder.xml",
        show_viewer: bool = True,
        dt: float = 0.005,
        device: str = "cpu",
        ee_link_name: str = "tool_tip",
        show_markers: bool = True,
        record_path: Optional[str] = None,
        record_fps: int = 12,
        boxes: Optional[List[Dict]] = None,
        spheres: Optional[List[Dict]] = None,
        goal_sphere_cfg: Optional[Dict] = None,
        surface_box: Optional[Dict[str, Any]] = None,
        obstacle_box: Optional[Dict[str, Any]] = None,
        obstacle_sphere: Optional[Dict[str, Any]] = None,
        goal_sphere: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Genesis simulation wrapper.

        Args:
            model_xml: Path to the MJCF robot XML file (default: 'panda_cylinder.xml').
            show_viewer: Whether to display interactive 3D viewer GUI window.
            dt: Simulation physics timestep in seconds (default: 0.005s / 200 Hz).
            device: Computing backend device ('cpu' or 'gpu').
            ee_link_name: Name of the robot end-effector link (default: 'tool_tip').
            show_markers: Whether to render debug overlays.
            record_path: Optional video/GIF output path.
            record_fps: Frame rate for video/GIF recording.
            boxes: Optional list of box dictionaries [{'pos': [x,y,z], 'size': [dx,dy,dz], 'color': [r,g,b]}].
            spheres: Optional list of sphere dictionaries [{'pos': [x,y,z], 'radius': r, 'color': [r,g,b]}].
            goal_sphere_cfg: Optional configuration dict for target goal marker.
            surface_box: Optional dict with 'pos', 'size' for spawning a base surface box.
            obstacle_box: Optional dict with 'pos', 'size' for spawning an obstacle box.
            obstacle_sphere: Optional dict with 'pos', 'radius' for spawning an obstacle sphere.
            goal_sphere: Optional alias for goal_sphere_cfg.
        """
        self.model_xml = model_xml
        self.ee_link_name = ee_link_name
        self.show_viewer = show_viewer
        self.dt = dt
        self.device = device
        self.n_envs = 1

        self.surface_box_cfg = surface_box
        self.obstacle_box_cfg = obstacle_box
        self.obstacle_sphere_cfg = obstacle_sphere
        self.goal_sphere_cfg = goal_sphere if goal_sphere is not None else goal_sphere_cfg

        # Combine explicit boxes list with optional convenience entities
        box_list = list(boxes) if boxes is not None else []
        if surface_box is not None:
            sb = dict(surface_box)
            sb.setdefault("color", (0.8, 0.8, 0.8))
            sb.setdefault("collision", True)
            box_list.append(sb)
        if obstacle_box is not None:
            ob = dict(obstacle_box)
            ob.setdefault("color", (0.9, 0.2, 0.2))
            ob.setdefault("collision", True)
            box_list.append(ob)
        self.boxes = box_list if box_list else None

        # Combine explicit spheres list with optional convenience entities
        sph_list = list(spheres) if spheres is not None else []
        if obstacle_sphere is not None:
            os_cfg = dict(obstacle_sphere)
            os_cfg.setdefault("color", (1.0, 0.4, 0.0))
            os_cfg.setdefault("collision", True)
            sph_list.append(os_cfg)
        self.spheres = sph_list if sph_list else None

        self.scene: Optional[gs.Scene] = None
        self.robot: Optional[gs.Entity] = None
        self.plane: Optional[gs.Entity] = None
        self.surface_box: Optional[gs.Entity] = None
        self.obstacle_box: Optional[gs.Entity] = None
        self.obstacle_sphere: Optional[gs.Entity] = None
        self.goal_sphere: Optional[gs.Entity] = None

        self._arm_dof_dim = 7
        self._f_ext = np.zeros(3, dtype=np.float64)  # pending external disturbance, world frame

        # ---- Visualization -------------------------------------------------------------------
        # Markers are viewer-only: the goal is a massless, collision-free entity, everything else
        # is debug-draw overlay. None of it touches the physics.
        self.show_markers = show_markers
        # Offscreen recording. The camera is created with debug=True so the marker overlays are
        # included in the render, not just the live viewer.
        self.record_path = record_path
        self.record_fps = record_fps
        self._camera = None
        self._frames: List[np.ndarray] = []
        self._record_every = max(1, int(round(1.0 / (record_fps * dt)))) if record_path else 0
        self._record_step = 0
        self._viz_every = 5          # redraw every Nth control step (dt=0.005 -> 40 Hz)
        self._trail_every = 10       # sample the tip trail every Nth step
        self._trail_max = 300        # breadcrumbs retained
        self._arrow_scale = 0.02     # metres of arrow per newton of disturbance
        self._viz_step = 0
        self._trail: deque = deque(maxlen=self._trail_max)
        self._dbg_line = None
        self._dbg_arrow = None
        self._dbg_trail = None
        self.goal_marker = None

        # Panda joint torque limits (MJCF forcerange); these are tau_min/tau_max for the QP.
        self.tau_min = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0])
        self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        # Initialize environment
        self.setup_environment()

    def setup_environment(self) -> None:
        """
        Initializes Genesis backend, creates scene, loads ground plane & robot, and builds the scene.
        """
        # Initialize Genesis engine
        if self.device == "gpu" and torch.cuda.is_available():
            backend = gs.gpu
        else:
            backend = gs.cpu
        gs.init(backend=backend, logging_level="warning")

        # Create scene with rigid options
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
                gravity=(0.0, 0.0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, -1.5, 1.2),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                refresh_rate=int(1.0 / self.dt),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
        )

        # Add ground plane entity
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add optional custom box obstacles / contact surfaces
        self.surface_box = None
        self.obstacle_box = None
        if self.boxes is not None:
            for b_cfg in self.boxes:
                b_pos = b_cfg.get("pos", (0.5, 0.0, 0.2))
                b_size = b_cfg.get("size", (0.5, 0.5, 0.4))
                b_col = b_cfg.get("color", (0.8, 0.8, 0.8))
                b_collision = b_cfg.get("collision", True)
                b_vis_contact = b_cfg.get("visualize_contact", b_collision)
                b_ent = self.scene.add_entity(
                    gs.morphs.Box(pos=b_pos, size=b_size, fixed=True, collision=b_collision),
                    surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=b_col)),
                    visualize_contact=b_vis_contact
                )
                if self.surface_box_cfg is not None and b_cfg.get("pos") == self.surface_box_cfg.get("pos"):
                    self.surface_box = b_ent
                elif self.obstacle_box_cfg is not None and b_cfg.get("pos") == self.obstacle_box_cfg.get("pos"):
                    self.obstacle_box = b_ent

        # Add optional custom spherical obstacles
        self.obstacle_sphere = None
        if self.spheres is not None:
            for s_cfg in self.spheres:
                sph_pos = s_cfg.get("pos", (0.45, 0.0, 0.45))
                sph_radius = s_cfg.get("radius", 0.08)
                sph_col = s_cfg.get("color", (1.0, 0.4, 0.0))
                sph_collision = s_cfg.get("collision", True)
                sph_vis_contact = s_cfg.get("visualize_contact", sph_collision)
                s_ent = self.scene.add_entity(
                    gs.morphs.Sphere(pos=sph_pos, radius=sph_radius, fixed=True, collision=sph_collision),
                    surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=sph_col)),
                    visualize_contact=sph_vis_contact
                )
                if self.obstacle_sphere_cfg is not None and s_cfg.get("pos") == self.obstacle_sphere_cfg.get("pos"):
                    self.obstacle_sphere = s_ent

        # Resolve robot XML so its meshes are findable, then load. No fallback: a missing model
        # must fail loudly rather than silently substituting a different robot.
        self._resolved_xml = resolve_model(self.model_xml)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self._resolved_xml,
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0)
            )
        )

        # Goal marker: emissive, massless, collision-free. Added after the robot so the robot's
        # link indices are untouched, and before build() as Genesis requires.
        if self.goal_sphere_cfg is not None:
            goal_pos = self.goal_sphere_cfg.get("pos", (0.55, 0.20, 0.45))
            goal_radius = self.goal_sphere_cfg.get("radius", 0.03)
            self.goal_marker = self.scene.add_entity(
                gs.morphs.Sphere(pos=goal_pos, radius=goal_radius, fixed=True, collision=False),
                surface=gs.surfaces.Emission(color=(0.0, 1.0, 0.0))
            )
            self.goal_sphere = self.goal_marker
        elif self.show_markers and (self.show_viewer or self.record_path is not None):
            self.goal_marker = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.02,
                    fixed=True,
                    collision=False,
                    batch_fixed_verts=True,
                ),
                surface=gs.surfaces.Emission(color=(0.1, 0.85, 0.25)),
            )
            self.goal_sphere = self.goal_marker
        else:
            self.goal_sphere = None

        # Recording camera, added before build() like every other entity.
        if self.record_path is not None:
            self._camera = self.scene.add_camera(
                res=(480, 360), pos=(1.6, -1.4, 1.2), lookat=(0.55, 0.1, 0.5),
                fov=45, GUI=False, debug=True,
            )

        # Build simulation scene
        self.scene.build(n_envs=self.n_envs)

        # ---- Torque-control setup -------------------------------------------------------------
        # Genesis imports the MJCF <general> actuator gains (kp up to 4500, kv up to 450), which is
        # a position servo. Left active it overwhelms any commanded torque, so it must be zeroed.
        zeros = torch.zeros(self.robot.n_dofs, dtype=gs.tc_float, device=gs.device)
        self.robot.set_dofs_kp(zeros)
        self.robot.set_dofs_kv(zeros)
        self.robot.set_dofs_force_range(
            torch.tensor(self.tau_min, dtype=gs.tc_float, device=gs.device),
            torch.tensor(self.tau_max, dtype=gs.tc_float, device=gs.device),
        )

        # ---- End-effector link ----------------------------------------------------------------
        link_names = [l.name for l in self.robot.links]
        if self.ee_link_name not in link_names:
            raise ValueError(
                f"end-effector link {self.ee_link_name!r} not present in model "
                f"{self.model_xml!r}. Available links: {link_names}"
            )
        self.ee_link = self.robot.get_link(self.ee_link_name)

        # ---- Shadow MuJoCo model for the dynamic bias term ------------------------------------
        # Genesis exposes no gravity or inverse-dynamics call, so h(q,dq) = C(q,dq)dq + g(q) is
        # taken from a MuJoCo model built from the same XML. mujoco ships as a Genesis dependency.
        self._mj_model = mujoco.MjModel.from_xml_path(self._resolved_xml)
        self._mj_data = mujoco.MjData(self._mj_model)
        if self._mj_model.nv < self._arm_dof_dim:
            raise RuntimeError(
                f"shadow MuJoCo model has nv={self._mj_model.nv}, expected at least "
                f"{self._arm_dof_dim}; it does not match the Genesis model."
            )
        self._mj_ee_body = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ee_link_name
        )
        if self._mj_ee_body < 0:
            raise RuntimeError(f"body {self.ee_link_name!r} not found in the shadow MuJoCo model")

        # Reset home position
        self.reset()

    def reset(self) -> None:
        """
        Resets the robot to home joint pose.
        """
        default_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        if self.robot.n_dofs > 7:
            # Add gripper default positions if 9 DOFs
            default_q += [0.04, 0.04]

        q_tensor = torch.tensor([default_q], dtype=gs.tc_float, device=gs.device)
        self.robot.set_qpos(q_tensor)
        self.robot.zero_all_dofs_velocity()
        self._f_ext[:] = 0.0
        # No settling steps: with the internal PD zeroed and no torque commanded yet, stepping here
        # would just let the arm fall. set_qpos already refreshes the kinematics.

    def step(self) -> None:
        """
        Steps physics engine forward by one control timestep dt.
        """
        self.scene.step()

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Extracts current robot state and dynamics matrices, converting all PyTorch tensors to NumPy arrays.

        Returns:
            Dict[str, np.ndarray]:
                - "q": Arm joint positions of shape (7,)
                - "dq": Arm joint velocities of shape (7,)
                - "J": End-effector task Jacobian matrix of shape (6, 7)
                - "B": Joint-space mass matrix of shape (7, 7)
                - "ee_pos": End-effector 3D position of shape (3,)
                - "ee_rot": End-effector 3x3 rotation matrix of shape (3, 3)
                - "h": Dynamic bias C(q, dq) * dq + g(q) of shape (7,)
        """
        # Joint positions and velocities (extract arm DOFs)
        q_raw = self.robot.get_dofs_position()
        dq_raw = self.robot.get_dofs_velocity()

        q_np = _tensor_to_numpy(q_raw)[:self._arm_dof_dim]
        dq_np = _tensor_to_numpy(dq_raw)[:self._arm_dof_dim]

        # End-effector Jacobian J(q)
        J_raw = self.robot.get_jacobian(link=self.ee_link)
        J_np = _tensor_to_numpy(J_raw)[:, :self._arm_dof_dim]  # (6, 7)

        # End-effector position from Genesis
        ee_pos_np = _tensor_to_numpy(self.ee_link.get_pos())

        # ---- Shadow MuJoCo: dynamic bias term and EE orientation ------------------------------
        # Sync the shadow model to the Genesis state, then read what Genesis does not expose.
        self._mj_data.qpos[:self._arm_dof_dim] = q_np
        self._mj_data.qvel[:self._arm_dof_dim] = dq_np
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # h(q, dq) = C(q, dq) * dq + g(q), exactly. Genesis has no equivalent call.
        h_np = np.asarray(self._mj_data.qfrc_bias[:self._arm_dof_dim], dtype=np.float64).copy()

        # Mass matrix from the same synced source. Genesis's get_mass_mat() is NOT refreshed by
        # set_qpos alone -- after a reset it still reflects the previous step's configuration
        # (measured: B differs by 2.47 while q, dq, J and ee_pos are all bit-identical). That
        # staleness made the whole simulation irreproducible, because a wrong B changes the QP's
        # H and g on the very first control step. Taking B from the shadow model guarantees it is
        # consistent with the q and dq that h was computed from.
        # data.M is a packed sparse triangle in MuJoCo 3.x, so build the dense matrix a column at
        # a time with mj_mulM (B e_i = i-th column). Seven columns, negligible cost, and stable
        # across MuJoCo versions.
        nv = self._mj_model.nv
        M_full = np.zeros((nv, nv), dtype=np.float64)
        e = np.zeros(nv, dtype=np.float64)
        col = np.zeros(nv, dtype=np.float64)
        for i in range(nv):
            e[:] = 0.0
            e[i] = 1.0
            mujoco.mj_mulM(self._mj_model, self._mj_data, col, e)
            M_full[:, i] = col
        M_np = M_full[:self._arm_dof_dim, :self._arm_dof_dim].copy()

        # Rotation matrix straight from MuJoCo's xmat, which sidesteps the unresolved question of
        # whether Genesis get_quat() is (w,x,y,z) or (x,y,z,w).
        ee_rot_np = np.asarray(
            self._mj_data.xmat[self._mj_ee_body], dtype=np.float64
        ).reshape(3, 3).copy()

        # End-effector contact force feedback
        ee_force_np = self.get_ee_contact_force()

        return {
            "q": q_np,
            "dq": dq_np,
            "J": J_np,
            "B": M_np,
            "h": h_np,
            "ee_pos": ee_pos_np,
            "ee_rot": ee_rot_np,
            "ee_force": ee_force_np,
        }

    def get_ee_contact_force(self) -> np.ndarray:
        """
        Extracts net contact force vector acting on the end-effector link in world frame.

        Returns:
            np.ndarray: 3D net contact force vector [Fx, Fy, Fz] in N.
        """
        if hasattr(self.robot, "get_links_net_contact_force"):
            try:
                F_links = _tensor_to_numpy(self.robot.get_links_net_contact_force())
                ee_idx = getattr(self.ee_link, "idx_local", -1)
                if ee_idx >= 0 and ee_idx < len(F_links):
                    return F_links[ee_idx]
            except Exception:
                pass
        return np.zeros(3, dtype=np.float64)

    def apply_external_disturbance(
        self,
        force: np.ndarray,
        link_name: str = "hand"
    ) -> None:
        """
        Applies external disturbance force vector to a specific link of the robot.

        Args:
            force: 3D force vector [Fx, Fy, Fz] in Newtons.
            link_name: Target link name (default: 'hand').
        """
        link = self.robot.get_link(link_name)
        if link is None and link_name in ["hand", "ee", "ee_link", "tool"]:
            link = self.robot.get_link(self.ee_link_name)
        if link is not None and hasattr(self.scene, "rigid_solver"):
            f_tensor = torch.zeros((self.scene.rigid_solver.n_links, 3), dtype=gs.tc_float, device=gs.device)
            f_tensor[link.idx] = torch.tensor(force, dtype=gs.tc_float, device=gs.device)
            self.scene.rigid_solver.apply_links_external_force(f_tensor)
        else:
            # Fallback to EE force injection
            self.set_external_force(force)

    def apply_torques(self, torques: np.ndarray) -> None:
        """
        Applies joint force/torque commands to the robot DOFs.

        Args:
            torques: Joint torque array of shape (7,) or matching n_dofs.
        """
        torques_flat = np.asarray(torques, dtype=np.float64).flatten()

        full_torques = np.zeros(self.robot.n_dofs, dtype=np.float64)
        full_torques[:self._arm_dof_dim] = torques_flat[:self._arm_dof_dim]

        # External disturbance. Genesis links expose no apply_force(), so a Cartesian push is
        # injected as its equivalent joint torque, tau_ext = J_lin^T f_ext.
        if np.any(self._f_ext):
            J_lin = _tensor_to_numpy(self.robot.get_jacobian(link=self.ee_link))[:3, :self._arm_dof_dim]
            full_torques[:self._arm_dof_dim] += J_lin.T @ self._f_ext

        # Convert to PyTorch tensor for Genesis
        tau_tensor = torch.tensor(full_torques, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
        
        # Apply joint forces in Genesis
        self.robot.control_dofs_force(tau_tensor)

    def set_external_force(self, force: np.ndarray) -> None:
        """
        Set a persistent external disturbance force applied at the end-effector.

        Genesis links have no apply_force() member, so the force is realised on the next
        apply_torques() call as the equivalent joint torque J_lin^T f_ext. Pass zeros to clear.

        Args:
            force: 3D force vector [Fx, Fy, Fz] in Newtons, world frame.
        """
        self._f_ext = np.asarray(force, dtype=np.float64).reshape(3).copy()

    # ---------------------------------------------------------------------- #
    #                              Visualization                             #
    # ---------------------------------------------------------------------- #

    def start_recording(self) -> None:
        """Begin capturing frames."""
        self._frames = []

    def record_frame(self) -> None:
        """Capture one frame, throttled to the configured fps."""
        if self._camera is None:
            return
        self._record_step += 1
        if self._record_step % self._record_every == 0:
            rgb = self._camera.render()
            if isinstance(rgb, tuple):
                rgb = rgb[0]
            self._frames.append(np.asarray(rgb, dtype=np.uint8).copy())

    def stop_recording(self) -> None:
        """
        Write the captured frames to disk.

        Genesis's own recorder is hard-wired to the libx264 codec and so only writes mp4. Frames
        are collected here instead and written with imageio, which allows GIF -- the format that
        renders inline in a GitHub README, and the one this repo's .gitignore does not exclude.
        """
        if self._camera is None or not self._frames:
            return
        import imageio.v2 as imageio
        out = pathlib.Path(self.record_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".gif":
            imageio.mimsave(out, self._frames, fps=self.record_fps, loop=0)
        else:
            imageio.mimsave(out, self._frames, fps=self.record_fps)
        self._frames = []

    def _viz_active(self) -> bool:
        """Markers are pure cost when there is no viewer to show them in."""
        return bool(self.show_markers and (self.show_viewer or self._camera is not None))

    def set_goal(self, pos: np.ndarray) -> None:
        """
        Place the goal marker at a Cartesian target.

        Args:
            pos: Desired end-effector position [x, y, z] in world frame.
        """
        if self.goal_marker is None:
            return
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        self.goal_marker.set_pos(
            torch.tensor(p[None, :], dtype=gs.tc_float, device=gs.device)
        )

    def update_viz(self, tip_pos: np.ndarray, goal_pos: np.ndarray) -> None:
        """
        Redraw the transient debug overlays: tip-to-goal error line, disturbance arrow, tip trail.

        Throttled to every `_viz_every` calls; each debug draw takes the visualizer lock, so doing
        this at the full control rate measurably slows the simulation. The pending external force
        is read from internal state, so callers do not have to pass it twice.

        Args:
            tip_pos:  Current end-effector position (3,).
            goal_pos: Current Cartesian target (3,).
        """
        if not self._viz_active():
            return

        self._viz_step += 1
        tip = np.asarray(tip_pos, dtype=np.float64).reshape(3)

        if self._viz_step % self._trail_every == 0:
            self._trail.append(tip.copy())

        if self._viz_step % self._viz_every != 0:
            return

        goal = np.asarray(goal_pos, dtype=np.float64).reshape(3)

        # --- error line: its length IS the tracking error --------------------------------------
        if self._dbg_line is not None:
            self.scene.clear_debug_object(self._dbg_line)
            self._dbg_line = None
        if np.linalg.norm(goal - tip) > 1e-4:
            self._dbg_line = self.scene.draw_debug_line(
                start=tip, end=goal, radius=0.004, color=(1.0, 0.75, 0.1, 0.9)
            )

        # --- disturbance arrow: only while a push is actually being applied ---------------------
        if self._dbg_arrow is not None:
            self.scene.clear_debug_object(self._dbg_arrow)
            self._dbg_arrow = None
        if np.any(self._f_ext):
            self._dbg_arrow = self.scene.draw_debug_arrow(
                pos=tip,
                vec=self._f_ext * self._arrow_scale,
                radius=0.008,
                color=(1.0, 0.15, 0.15, 1.0),
            )

        # --- trail: cyan breadcrumbs, older points more transparent -----------------------------
        if self._dbg_trail is not None:
            self.scene.clear_debug_object(self._dbg_trail)
            self._dbg_trail = None
        if len(self._trail) >= 2:
            poss = np.asarray(self._trail, dtype=np.float64)
            alpha = np.linspace(0.15, 0.9, len(poss))
            colors = np.column_stack([
                np.full(len(poss), 0.1),
                np.full(len(poss), 0.85),
                np.full(len(poss), 0.95),
                alpha,
            ])
            self._dbg_trail = self.scene.draw_debug_points(poss=poss, colors=colors)

    def clear_viz(self) -> None:
        """Remove all debug overlays this object owns and forget the trail."""
        for handle in (self._dbg_line, self._dbg_arrow, self._dbg_trail):
            if handle is not None:
                self.scene.clear_debug_object(handle)
        self._dbg_line = self._dbg_arrow = self._dbg_trail = None
        self._trail.clear()
