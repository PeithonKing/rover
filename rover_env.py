"""
rover_env.py
============
TorchRL environment for the 6-wheel rocker-bogie rover in MuJoCo.
Subclasses `torchrl.envs.EnvBase` and natively produces `TensorDict` instances.

Observation Spec (Composite):
    cameras       → Bounded (12, 128, 128) uint8 [0, 255] (4×RGB 128×128 images stacked on channel axis)
    numeric       → UnboundedContinuous (13,) float32 [IMU quat(4) + IMU angvel(3) + 4 passive joints
                                                      + target_dx(local) + target_dy(local)]

Action Spec:
    Bounded (2,) float32 [-1.0, 1.0]
    [0]  → accel_cmd   : incremental acceleration of COM speed.
                         Clipped so speed stays in [-MAX_COM_SPEED, +MAX_COM_SPEED].
    [1]  → steer_cmd   : incremental steering delta, suppressed by STEER_RATE.
                         -1 = sharpest left, 0 = straight, +1 = sharpest right.
                         Accumulated each step as: steer += steer_cmd * STEER_RATE
                         Clipped to [-1.0, 1.0].

Reward Spec:
    UnboundedContinuous (1,) float32

Done Spec (Composite):
    done          → Binary (1,) bool
    terminated    → Binary (1,) bool
    truncated     → Binary (1,) bool

Drive strategy: True Ackermann — 4 corner steering servos lock to computed angles,
all 6 drive wheels spin at individually correct speeds so NO lateral drag occurs.
Wheel speeds are computed from each wheel's dynamic xanchor (body-frame), so the
rocker-bogie suspension geometry is respected frame-by-frame.

Episode:
    - Rover spawns at origin facing +X dropped from z=0.1m
    - Target is 5.0–5.1m ahead (semi-circle [-pi/2, +pi/2])
    - Terminated when within 0.5m of target, rover flips (tilt > 1.2 rad), or z > 1.0m
    - Truncated after MAX_STEPS steps (2000 steps = 40s)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# Enable headless EGL rendering if not explicitly overridden
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import mujoco.viewer
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import (
    Binary,
    Bounded,
    Composite,
    UnboundedContinuous,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE_XML = os.path.join(os.path.dirname(__file__), "3D_files/mujoco/scene.xml")
IMG_H, IMG_W = 128, 128
N_CAMERAS = 4
CAM_NAMES = ["cam_front_left", "cam_side_left", "cam_front_right", "cam_side_right"]
MAX_WHEEL_VEL = 29.24  # rad/s  (physical max of drive motors)
MAX_COM_SPEED = 2.77  # m/s    (= MAX_WHEEL_VEL * WHEEL_RADIUS, physical cap)
WHEEL_RADIUS = 0.095  # m      (190mm diameter wheels)
MAX_STEER_ANG = 0.7854  # rad    (±45° hard cap on servo angle)
STEER_RATE = 1.0 / 20.0  # steer accumulates at 1/20th per step
MAX_STEPS = 2000
SUCCESS_RADIUS = 0.5  # m
MAX_TILT = 1.2  # rad (~68.75°) — beyond this we treat it as flipped

# Numeric obs shape: IMU quat(4) + IMU angvel(3) + 4 passive joints + dx + dy = 13
N_NUMERIC = 13

# Drive joint names — [right_front, right_mid, right_back, left_front, left_mid, left_back]
# Signs: right side Y is negative in body frame, left side Y is positive
DRV_JOINTS = [
    ("drv_right_front_wheel", "srv_right_front_rotator", -1),
    ("drv_right_middle_wheel", None, -1),
    ("drv_right_back_wheel", "srv_right_back_rotator", -1),
    ("drv_left_front_wheel", "srv_left_front_rotator", +1),
    ("drv_left_middle_wheel", None, +1),
    ("drv_left_back_wheel", "srv_left_back_rotator", +1),
]

# Passive joint sensor names
PASSIVE_SENSORS = [
    "sensor_pass_left_rocker",
    "sensor_pass_right_rocker",
    "sensor_pass_left_rockerbogie",
    "sensor_pass_right_rockerbogie",
]

# Reward weights
R_SUCCESS = 100.0  # one-time bonus for reaching target
R_FLIP_PENALTY = -50.0  # one-time penalty for flipping over or sky-high launch


class RoverEnv(EnvBase):
    """
    TorchRL native environment for the 6-wheel rover.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        blind: bool = False,
        device: torch.device | str = "cpu",
    ):
        super().__init__(device=device)
        self.render_mode = render_mode
        self.blind = bool(blind)

        # Load MuJoCo model and instantiate simulation data
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        # Pre-cache actuator / sensor / joint IDs for fast lookup
        self._wheel_info: List[Tuple[int, Optional[int], int, int, str, Optional[str]]] = []
        for drv_name, srv_name, lat_sign in DRV_JOINTS:
            drv_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, drv_name
            )
            srv_id = (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, srv_name)
                if srv_name
                else None
            )
            drv_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, drv_name
            )
            self._wheel_info.append(
                (drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name)
            )

        # Cache steer joint IDs separately for xanchor queries
        self._srv_jnt_ids: Dict[str, int] = {}
        for _, srv_name, _ in DRV_JOINTS:
            if srv_name and srv_name not in self._srv_jnt_ids:
                self._srv_jnt_ids[srv_name] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, srv_name
                )

        self._sens_ids: Dict[str, int] = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in PASSIVE_SENSORS
        }

        # Offscreen renderer for camera images
        if not self.blind:
            self._renderer: Optional[mujoco.Renderer] = mujoco.Renderer(
                self.model, height=IMG_H, width=IMG_W
            )
        else:
            self._renderer = None

        # Viewer for human render mode
        self._viewer: Optional[mujoco.viewer.Handle] = None

        # Target position and telemetry tracking
        self._target: np.ndarray = np.zeros(2, dtype=np.float32)
        self._prev_dist: float = 0.0

        # Drive state (accumulated across steps)
        self._com_speed: float = 0.0
        self._current_steer: float = 0.0
        self._steps: int = 0
        self._last_info: Dict[str, Any] = {}

        # Random number generator
        self.np_random = np.random.RandomState()

        # Build TorchRL specs
        self._make_spec()

    def _make_spec(self) -> None:
        """Defines TorchRL observation, action, reward, and done specifications."""
        self.observation_spec = Composite(
            cameras=Bounded(
                shape=torch.Size([N_CAMERAS * 3, IMG_H, IMG_W]),
                dtype=torch.uint8,
                low=0,
                high=255,
                device=self.device,
            ),
            numeric=UnboundedContinuous(
                shape=torch.Size([N_NUMERIC]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        self.action_spec = Bounded(
            shape=torch.Size([2]),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            done=Binary(shape=torch.Size([1]), dtype=torch.bool, device=self.device),
            terminated=Binary(shape=torch.Size([1]), dtype=torch.bool, device=self.device),
            truncated=Binary(shape=torch.Size([1]), dtype=torch.bool, device=self.device),
            shape=torch.Size([]),
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int] = None) -> Optional[int]:
        """Seeds numpy and torch RNGs for deterministic episode generation."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            torch.manual_seed(seed)
        return seed

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDict:
        """
        Resets simulation physics, target position, and kinematics.
        Calls mj_forward to ensure xanchor, xpos, and sensors are properly computed.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Spawn rover dropped from z=0.1m at origin facing +X
        root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qpos_adr = self.model.jnt_qposadr[root_jnt_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = [0.0, 0.0, 0.1]
        self.data.qpos[qpos_adr + 3] = 1.0  # qw (upright)
        self.data.qpos[qpos_adr + 4 : qpos_adr + 7] = 0.0  # qx, qy, qz

        # CRITICAL BUG FIX: Forward kinematics must be updated after setting qpos
        mujoco.mj_forward(self.model, self.data)

        # Spawn target 5.0–5.1m away in a ±90° arc in front (+X)
        dist = float(self.np_random.uniform(5.0, 5.1))
        angle = float(self.np_random.uniform(-np.pi / 2, np.pi / 2))
        self._target = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)
        self._prev_dist = dist
        
        # Update yellow sphere marker position in MuJoCo scene
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
        if site_id != -1:
            self.model.site_pos[site_id][:2] = self._target

        # Reset drive state
        self._com_speed = 0.0
        self._current_steer = 0.0
        self._steps = 0
        self._last_info = {
            "target": self._target.copy(),
            "dist": dist,
            "tilt_rad": 0.0,
            "progress": 0.0,
        }

        obs_cameras, obs_numeric = self._get_obs()

        return TensorDict(
            {
                "cameras": torch.as_tensor(obs_cameras, dtype=torch.uint8, device=self.device),
                "numeric": torch.as_tensor(obs_numeric, dtype=torch.float32, device=self.device),
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        """
        Executes one control step (10 MuJoCo substeps) with True Ackermann steering.
        """
        action = tensordict.get("action")
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)

        action_np = np.clip(action_np.reshape(-1), -1.0, 1.0)
        accel_cmd = float(action_np[0])
        steer_cmd = float(action_np[1])

        # 1. Accumulate drive state (Original Momentum Physics)
        self._com_speed = float(
            np.clip(
                self._com_speed + accel_cmd * MAX_COM_SPEED * 0.1,
                -MAX_COM_SPEED,
                MAX_COM_SPEED,
            )
        )
        self._current_steer = float(
            np.clip(self._current_steer + steer_cmd * STEER_RATE, -1.0, 1.0)
        )

        # 2. Dynamic Ackermann geometry (body frame projection)
        xmat_inv = self.data.body("body").xmat.reshape(3, 3).T
        center_pos = self.data.body("body").xpos

        def get_local_pos(jnt_id: int) -> np.ndarray:
            """Returns [x_fwd, y_lat, z_up] of a joint anchor in rover body frame."""
            global_anchor = self.data.xanchor[jnt_id].copy()
            return np.dot(xmat_inv, global_anchor - center_pos)

        # 3. Compute per-wheel velocity and steering commands
        wheel_speeds: List[float] = []
        for drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name in self._wheel_info:
            if srv_id is not None:
                assert srv_name is not None
                srv_jnt_id = self._srv_jnt_ids[srv_name]
                lp = get_local_pos(srv_jnt_id)
            else:
                lp = get_local_pos(drv_jnt_id)

            X_i = lp[0]  # forward offset from chassis center
            Y_i = lp[1]  # lateral offset (+Y left, -Y right)

            W_track = abs(Y_i) * 2.0 if abs(Y_i) > 1e-4 else 0.3546
            omega = self._current_steer * (2.0 * MAX_COM_SPEED / W_track)

            Vx_i = self._com_speed - omega * Y_i
            Vy_i = omega * X_i

            # Compute steering angle via arctan2
            if srv_id is not None:
                steer_angle = float(
                    np.arctan2(Vy_i, max(abs(Vx_i), 1e-6) * np.sign(Vx_i + 1e-9))
                )
                steer_angle = float(np.clip(steer_angle, -MAX_STEER_ANG, MAX_STEER_ANG))
                self.data.ctrl[srv_id] = steer_angle

            # Resultant wheel hub velocity
            hub_speed = float(np.sqrt(Vx_i**2 + Vy_i**2)) * np.sign(Vx_i + 1e-9)
            wheel_speeds.append(hub_speed)

        # 4. Normalize wheel speeds so no motor exceeds physical speed limits
        max_hub = max(abs(s) for s in wheel_speeds) if wheel_speeds else 1.0
        scale = 1.0 if max_hub <= MAX_COM_SPEED else MAX_COM_SPEED / max_hub

        for i, (drv_id, _, _, _, _, _) in enumerate(self._wheel_info):
            motor_rads = (wheel_speeds[i] * scale) / WHEEL_RADIUS
            self.data.ctrl[drv_id] = float(
                np.clip(motor_rads, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            )

        # 5. Advance simulation by 10 substeps (dt=0.002s -> 0.02s per step)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs_cameras, obs_numeric = self._get_obs()
        reward_val, terminated, info = self._compute_reward()
        self._steps += 1
        truncated = self._steps >= MAX_STEPS
        done = terminated or truncated

        if self.render_mode == "human":
            self._render_human()

        self._last_info = info

        return TensorDict(
            {
                "cameras": torch.as_tensor(obs_cameras, dtype=torch.uint8, device=self.device),
                "numeric": torch.as_tensor(obs_numeric, dtype=torch.float32, device=self.device),
                "reward": torch.tensor([reward_val], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collects 4-camera visual observations and 13-dim numerical telemetry."""
        # 1. Camera frames (4x RGB 128x128 transposed to CHW stacked along channel axis -> 12x128x128)
        if self.blind or self._renderer is None:
            cam_obs = np.zeros((N_CAMERAS * 3, IMG_H, IMG_W), dtype=np.uint8)
        else:
            frames: List[np.ndarray] = []
            for cam_name in CAM_NAMES:
                self._renderer.update_scene(self.data, camera=cam_name)
                frame = self._renderer.render()  # (H, W, 3)
                frames.append(frame.transpose(2, 0, 1))  # (3, H, W)
            cam_obs = np.concatenate(frames, axis=0).astype(np.uint8)

        # 2. Numerical observations (13 dimensions)
        # IMU quaternion (4) + IMU angular velocity (3)
        imu_quat = self.data.sensor("imu_quat").data.copy().astype(np.float32)
        imu_angvel = self.data.sensor("imu_angvel").data.copy().astype(np.float32)

        # 4 Passive suspension joint angles (4)
        passive = np.array(
            [self.data.sensor(n).data[0] for n in PASSIVE_SENSORS], dtype=np.float32
        )

        # Relative target vector projected into rover body frame (2)
        rover_pos = self.data.body("body").xpos[:2]
        world_delta = (self._target - rover_pos).astype(np.float32)

        xmat = self.data.body("body").xmat.reshape(3, 3)
        forward = xmat[:2, 0]  # Local +X in world XY
        right = xmat[:2, 1]  # Local +Y in world XY

        local_dx = float(np.dot(world_delta, forward))
        local_dy = float(np.dot(world_delta, right))
        rel_target = np.array([local_dx, local_dy], dtype=np.float32)

        numeric = np.concatenate([imu_quat, imu_angvel, passive, rel_target], dtype=np.float32)

        return cam_obs, numeric

    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Computes dense progress reward, tilt penalty, terminal rewards, and telemetry flags."""
        rover_pos = self.data.body("body").xpos[:2]
        dist = float(np.linalg.norm(self._target - rover_pos))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        # Chassis tilt angle calculation using world-frame orientation
        xquat = self.data.body("body").xquat.copy()
        w, x, y, z = xquat
        body_z_world = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
            dtype=np.float32,
        )
        tilt = float(np.arccos(np.clip(body_z_world[2], -1.0, 1.0)))

        # Telemetry metrics
        info: Dict[str, Any] = {
            "dist": dist,
            "tilt_rad": tilt,
            "progress": progress,
            "success": False,
            "flipped": False,
            "sky_high": False,
        }

        # Time penalty scales with distance (closer = smaller penalty)
        reward = -0.1 * dist

        # Terminal conditions
        terminated = False
        if dist < SUCCESS_RADIUS:
            reward = 100.0  # Big reward for reaching goal
            terminated = True
            info["success"] = True
        elif tilt > MAX_TILT:
            reward = -100.0  # Penalty for flipping
            terminated = True
            info["flipped"] = True
        elif self.data.body("body").xpos[2] > 1.0:
            reward = -100.0
            terminated = True
            info["sky_high"] = True

        return reward, terminated, info

    @property
    def last_info(self) -> Dict[str, Any]:
        """Access latest physics telemetry and status flags."""
        return self._last_info

    def _render_human(self) -> None:
        """Synchronizes passive MuJoCo viewer for human render mode."""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def render(self) -> Optional[np.ndarray]:
        """Renders an RGB array for visual inspection."""
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self, *args, **kwargs) -> None:
        """Cleans up viewer and renderer resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
