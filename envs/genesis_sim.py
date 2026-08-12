"""
Genesis Simulator Wrapper Environment.

Handles scene setup, robot loading (panda_cylinder.xml), physics stepping, external disturbance application,
and state variable extraction with automatic PyTorch tensor to NumPy array conversion for CasADi compatibility.
"""

from typing import Dict, Optional, Tuple, Any
import os
import numpy as np
import torch
import genesis as gs


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
        device: str = "cpu"
    ) -> None:
        """
        Initialize the Genesis simulation wrapper.

        Args:
            model_xml: Path to the MJCF robot XML file (default: 'panda_cylinder.xml').
            show_viewer: Whether to display interactive 3D viewer GUI window.
            dt: Simulation physics timestep in seconds (default: 0.005s / 200 Hz).
            device: Computing backend device ('cpu' or 'gpu').
        """
        self.model_xml = model_xml
        self.show_viewer = show_viewer
        self.dt = dt
        self.device = device
        self.n_envs = 1

        self.scene: Optional[gs.Scene] = None
        self.robot: Optional[gs.Entity] = None
        self.plane: Optional[gs.Entity] = None

        self._arm_dof_dim = 7

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

        # Resolve robot XML file path
        if os.path.exists(self.model_xml):
            xml_file = os.path.abspath(self.model_xml)
        else:
            # Pass as relative path string for Genesis built-in asset resolution (e.g. 'xml/franka_emika_panda/panda.xml')
            xml_file = self.model_xml

        # Load robot entity from MJCF
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=xml_file,
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0)
                )
            )
        except Exception:
            # Fallback to Genesis built-in Franka model if custom XML assets are missing
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file="xml/franka_emika_panda/panda.xml",
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0)
                )
            )

        # Build simulation scene
        self.scene.build(n_envs=self.n_envs)

        # Retrieve EE link handle
        self.ee_link = self.robot.get_link("hand") if "hand" in [l.name for l in self.robot.links] else self.robot.links[-1]

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

        # Step simulation to let state update
        for _ in range(10):
            self.scene.step()

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
        """
        # Joint positions and velocities (extract arm DOFs)
        q_raw = self.robot.get_dofs_position()
        dq_raw = self.robot.get_dofs_velocity()

        q_np = _tensor_to_numpy(q_raw)[:self._arm_dof_dim]
        dq_np = _tensor_to_numpy(dq_raw)[:self._arm_dof_dim]

        # Joint space mass matrix M(q) / B(q)
        M_raw = self.robot.get_mass_mat()
        M_np = _tensor_to_numpy(M_raw)[:self._arm_dof_dim, :self._arm_dof_dim]

        # End-effector Jacobian J(q)
        J_raw = self.robot.get_jacobian(link=self.ee_link)
        J_np = _tensor_to_numpy(J_raw)[:, :self._arm_dof_dim]  # (6, 7)

        # End-effector position and orientation matrix
        ee_pos_raw = self.ee_link.get_pos()
        ee_quat_raw = self.ee_link.get_quat()  # (w, x, y, z) or (x, y, z, w)

        ee_pos_np = _tensor_to_numpy(ee_pos_raw)
        
        # Convert quaternion to rotation matrix
        ee_quat_np = _tensor_to_numpy(ee_quat_raw)
        ee_rot_np = _tensor_to_numpy(gs.utils.geom.quat_to_R(ee_quat_np))

        # Dynamic bias vector h(q, dq) = C(q, dq) * dq + g(q) if available
        h_np = np.zeros(self._arm_dof_dim, dtype=np.float64)
        if hasattr(self.robot, "get_gravity_force"):
            try:
                g_raw = self.robot.get_gravity_force()
                h_np = _tensor_to_numpy(g_raw)[:self._arm_dof_dim]
            except Exception:
                pass

        return {
            "q": q_np,
            "dq": dq_np,
            "J": J_np,
            "B": M_np,
            "h": h_np,
            "ee_pos": ee_pos_np,
            "ee_rot": ee_rot_np,
        }

    def apply_torques(self, torques: np.ndarray) -> None:
        """
        Applies joint force/torque commands to the robot DOFs.

        Args:
            torques: Joint torque array of shape (7,) or matching n_dofs.
        """
        torques_flat = np.asarray(torques, dtype=np.float32).flatten()

        full_torques = np.zeros(self.robot.n_dofs, dtype=np.float32)
        full_torques[:self._arm_dof_dim] = torques_flat[:self._arm_dof_dim]

        # Convert to PyTorch tensor for Genesis
        tau_tensor = torch.tensor(full_torques, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
        
        # Apply joint forces in Genesis
        self.robot.control_dofs_force(tau_tensor)

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
        if link is not None and hasattr(link, "apply_force"):
            f_tensor = torch.tensor(force, dtype=gs.tc_float, device=gs.device)
            link.apply_force(f_tensor)

