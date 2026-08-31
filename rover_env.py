"""
rover_env.py
============
Gymnasium environment for the 6-wheel rocker-bogie rover in MuJoCo.

Observation Space (Dict):
    cameras       → (12, 128, 128) uint8   4×RGB 128×128 images stacked on channel axis
    numeric       → (13,) float32          [IMU quat(4) + IMU angvel(3) + 4 passive joints
                                            + target_dx(local) + target_dy(local)]

Action Space:
    Box(-1, 1, shape=(2,))
    [0]  → accel_cmd   : incremental acceleration of COM speed.
                         Clipped so speed stays in [-MAX_COM_SPEED, +MAX_COM_SPEED].
    [1]  → steer_cmd   : incremental steering delta, suppressed by STEER_RATE.
                         -1 = sharpest left, 0 = straight, +1 = sharpest right.
                         Accumulated each step as: steer += steer_cmd * STEER_RATE
                         Clipped to [-1, 1].

Drive strategy: True Ackermann — 4 corner steering servos lock to computed angles,
all 6 drive wheels spin at individually correct speeds so NO lateral drag occurs.
Wheel speeds are computed from each wheel's dynamic xanchor (body-frame), so the
rocker-bogie suspension geometry is respected frame-by-frame.

Episode:
    - Rover spawns at origin facing +X
    - Target is 5–5.1m ahead (semi-circle)
    - Terminated when within 0.5m of target or rover flips
    - Truncated after MAX_STEPS steps
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

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
MAX_TILT = 1.2  # rad (~70°) — beyond this we treat it as flipped

# Numeric obs shape: IMU quat(4) + IMU angvel(3) + 4 passive joints + dx + dy = 13
N_NUMERIC = 13

# Drive joint names — [right_front, right_mid, right_back, left_front, left_mid, left_back]
# Signs: right side Y is negative in body frame, left side Y is positive
DRV_JOINTS = [
    (
        "drv_right_front_wheel",
        "srv_right_front_rotator",
        -1,
    ),  # (drive, steer, lateral_sign)
    ("drv_right_middle_wheel", None, -1),  # middle has no servo
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
R_FLIP_PENALTY = -50.0  # one-time penalty for flipping over


class RoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, blind=False):
        super().__init__()
        self.render_mode = render_mode
        self.blind = blind

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        # Pre-cache actuator / sensor IDs for fast lookup
        # For each wheel: drive actuator ID, servo actuator ID (or None), lateral sign
        self._wheel_info = []
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
            # Store the joint ID for xanchor queries (drive joint anchor = wheel center)
            self._wheel_info.append(
                (drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name)
            )

        # For dynamic Ackermann geometry: cache steer joint IDs separately for xanchor
        self._srv_jnt_ids = {}
        for _, srv_name, _ in DRV_JOINTS:
            if srv_name and srv_name not in self._srv_jnt_ids:
                self._srv_jnt_ids[srv_name] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, srv_name
                )

        self._sens_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in PASSIVE_SENSORS
        }

        # Offscreen renderer for camera images
        if not self.blind:
            self._renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        else:
            self._renderer = None

        # Target position (world XY)
        self._target = np.zeros(2)
        self._prev_dist = 0.0

        # Drive state — these accumulate across steps
        self._com_speed = 0.0  # m/s, current COM speed (modified by accel_cmd)
        self._current_steer = (
            0.0  # [-1, 1], current steering state (modified by steer_cmd/20)
        )

        # --- Gym spaces ---
        self.observation_space = spaces.Dict(
            {
                "cameras": spaces.Box(
                    low=0, high=255, shape=(N_CAMERAS * 3, IMG_H, IMG_W), dtype=np.uint8
                ),
                "numeric": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_NUMERIC,), dtype=np.float32
                ),
            }
        )
        # 2 actions: [accel_cmd, steer_cmd]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Viewer for human render mode
        self._viewer = None

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Always drop from 50cm — rover settles naturally on its wheels
        # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
        root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qpos_adr = self.model.jnt_qposadr[root_jnt_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = [0.0, 0.0, 0.1]  # x, y, z
        self.data.qpos[qpos_adr + 3] = 1.0  # qw  (upright)
        self.data.qpos[qpos_adr + 4 : qpos_adr + 7] = 0.0  # qx, qy, qz

        # Spawn target 5–5.1m away in a ±90° arc in front (along +X)
        dist = self.np_random.uniform(5.0, 5.1)
        angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self._target = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        self._prev_dist = dist

        # Reset drive state
        self._com_speed = 0.0
        self._current_steer = 0.0
        self._steps = 0
        obs = self._get_obs()
        info = {"target": self._target.copy()}
        return obs, info

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        accel_cmd = float(action[0])
        steer_cmd = float(action[1])

        # --- 1. Accumulate drive state ---
        # Speed: accel_cmd directly shifts COM speed (m/s), then cap
        self._com_speed = float(
            np.clip(
                self._com_speed + accel_cmd * MAX_COM_SPEED * 0.1,
                -MAX_COM_SPEED,
                MAX_COM_SPEED,
            )
        )
        # Steer: incremental at 1/20th rate, cap to [-1, 1]
        self._current_steer = float(
            np.clip(self._current_steer + steer_cmd * STEER_RATE, -1.0, 1.0)
        )

        # --- 2. Dynamic Ackermann geometry (body frame) ---
        # Get chassis rotation matrix to project world coords → body frame
        xmat_inv = self.data.body("body").xmat.reshape(3, 3).T
        center_pos = self.data.body("body").xpos

        def get_local_pos(jnt_id):
            """Returns [x_fwd, y_lat, z_up] of a joint anchor in rover body frame."""
            global_anchor = self.data.xanchor[jnt_id].copy()
            return np.dot(xmat_inv, global_anchor - center_pos)

        # --- 3. Compute per-wheel commands ---
        wheel_speeds = []  # m/s at each wheel hub
        for (
            drv_id,
            srv_id,
            drv_jnt_id,
            lat_sign,
            drv_name,
            srv_name,
        ) in self._wheel_info:
            if srv_id is not None:
                # Corner wheel: steer joint anchor gives precise kinematic pivot
                srv_jnt_id = self._srv_jnt_ids[srv_name]
                lp = get_local_pos(srv_jnt_id)
            else:
                # Middle wheel: use drive joint anchor for X, lat_sign * track_half for Y
                lp = get_local_pos(drv_jnt_id)

            X_i = lp[0]  # forward offset from chassis center
            Y_i = lp[
                1
            ]  # lateral offset (positive = left, negative = right in body frame)

            # Yaw rate from steering: sharpest turn (steer=1) = point turn
            # W_track = 2 * abs(Y_i of any corner wheel). Use dynamic Y.
            # We clamp to avoid division issues at zero speed / zero steer.
            W_track = abs(Y_i) * 2.0 if abs(Y_i) > 1e-4 else 0.3546

            omega = self._current_steer * (2.0 * MAX_COM_SPEED / W_track)

            Vx_i = self._com_speed - omega * Y_i  # forward velocity at wheel i
            Vy_i = omega * X_i  # lateral velocity at wheel i

            # Servo angle (arctan2 gives the toe-in/out angle needed for this arc)
            if srv_id is not None:
                steer_angle = float(
                    np.arctan2(Vy_i, max(abs(Vx_i), 1e-6) * np.sign(Vx_i + 1e-9))
                )
                steer_angle = float(np.clip(steer_angle, -MAX_STEER_ANG, MAX_STEER_ANG))
                self.data.ctrl[srv_id] = steer_angle

            # Drive speed: resultant wheel hub speed → convert to motor rad/s
            hub_speed = float(np.sqrt(Vx_i**2 + Vy_i**2)) * np.sign(Vx_i + 1e-9)
            wheel_speeds.append(hub_speed)

        # --- 4. Normalize wheel speeds so no wheel exceeds MAX_WHEEL_VEL ---
        max_hub = max(abs(s) for s in wheel_speeds) if wheel_speeds else 1.0
        scale = 1.0 if max_hub <= MAX_COM_SPEED else MAX_COM_SPEED / max_hub

        for i, (drv_id, _, _, _, _, _) in enumerate(self._wheel_info):
            motor_rads = (wheel_speeds[i] * scale) / WHEEL_RADIUS
            self.data.ctrl[drv_id] = float(
                np.clip(motor_rads, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            )

        # --- 5. Advance physics ---
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()
        self._steps += 1
        truncated = self._steps >= MAX_STEPS

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------
    def _get_obs(self):
        # --- Cameras ---
        if self.blind:
            cam_obs = np.zeros((N_CAMERAS * 3, IMG_H, IMG_W), dtype=np.uint8)
        else:
            frames = []
            for cam_name in CAM_NAMES:
                self._renderer.update_scene(self.data, camera=cam_name)
                frames.append(self._renderer.render())  # (H, W, 3)
            # Stack to (N_CAMERAS*3, H, W)
            cam_obs = np.concatenate(
                [f.transpose(2, 0, 1) for f in frames], axis=0
            ).astype(np.uint8)

        # --- IMU ---
        # framequat sensor → 4 floats  |  frameangvel → 3 floats
        imu_quat = self.data.sensor("imu_quat").data.copy().astype(np.float32)
        imu_angvel = self.data.sensor("imu_angvel").data.copy().astype(np.float32)

        # --- Passive joint angles ---
        passive = np.array(
            [self.data.sensor(n).data[0] for n in PASSIVE_SENSORS], dtype=np.float32
        )

        # --- Relative target vector (projected to rover body frame) ---
        rover_pos = self.data.body("body").xpos[:2]
        world_delta = (self._target - rover_pos).astype(np.float32)

        # Get rover heading yaw from xmat
        xmat = self.data.body("body").xmat.reshape(3, 3)
        forward = xmat[:2, 0]  # Local X in world coords
        right = xmat[:2, 1]  # Local Y in world coords

        # Project delta into rover local frame
        local_dx = float(np.dot(world_delta, forward))
        local_dy = float(np.dot(world_delta, right))
        rel_target = np.array([local_dx, local_dy], dtype=np.float32)

        numeric = np.concatenate([imu_quat, imu_angvel, passive, rel_target])  # (13,)

        return {"cameras": cam_obs, "numeric": numeric}

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------
    def _compute_reward(self):
        rover_pos = self.data.body("body").xpos[:2]
        dist = float(np.linalg.norm(self._target - rover_pos))
        progress = self._prev_dist - dist  # positive when moving closer
        self._prev_dist = dist

        # Tilt angle: use body xquat directly (world frame, always correct)
        # MuJoCo xquat convention: [w, x, y, z]
        xquat = self.data.body("body").xquat.copy()
        w, x, y, z = xquat
        body_z_world = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        tilt = float(np.arccos(np.clip(body_z_world[2], -1.0, 1.0)))

        # Build reward
        reward = 0.0
        terminated = False
        info = {"dist": dist, "tilt_rad": tilt, "progress": progress}

        # Dense progress reward + small step cost to encourage speed
        reward = (progress * 20.0) - 0.01

        if tilt > 0.3:
            reward -= tilt * 0.5  # soft penalty for leaning

        if dist < SUCCESS_RADIUS:
            reward += R_SUCCESS
            terminated = True
            info["success"] = True
        elif tilt > MAX_TILT:
            reward += R_FLIP_PENALTY
            terminated = True
            info["flipped"] = True
        elif self.data.body("body").xpos[2] > 1.0:
            reward += R_FLIP_PENALTY
            terminated = True
            info["sky_high"] = True

        return reward, terminated, info

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------
    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
