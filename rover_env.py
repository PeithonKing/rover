"""
rover_env.py
============
Gymnasium environment for the 6-wheel rocker-bogie rover in MuJoCo.

Observation Space (Dict):
    cameras       → (12, 128, 128) uint8   4×RGB 128×128 images stacked on channel axis
    numeric       → (12,) float32          [IMU quat(4) + IMU angvel(3) + 4 passive joints
                                            + target_dx + target_dy]  → 4+3+4+1 = 12 (wait: 4+3=7, +4=11, +2=13... let me count)
                                            Actually: quat=4, angvel=3, passive_joints=4, rel_target=2 → 13

Action Space:
    Box(-1, 1, shape=(10,))
    [0:6]   → 6 drive wheel velocities   (scaled by MAX_WHEEL_VEL = 29.24 rad/s)
    [6:10]  → 4 steering angles          (scaled by MAX_STEER_ANG = ±45°)

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
SCENE_XML     = os.path.join(os.path.dirname(__file__), "3D_files/mujoco/scene.xml")
IMG_H, IMG_W  = 128, 128
N_CAMERAS     = 4
CAM_NAMES     = ["cam_front_left", "cam_side_left", "cam_front_right", "cam_side_right"]
MAX_WHEEL_VEL = 29.24   # rad/s
MAX_STEER_ANG = 0.7854  # rad (±45°)
MAX_STEPS     = 2000
SUCCESS_RADIUS = 0.5    # m
MAX_TILT      = 1.2     # rad (~70°) — beyond this we treat it as flipped

# Numeric obs shape: IMU quat(4) + IMU angvel(3) + 4 passive joints + dx + dy = 13
N_NUMERIC     = 13

# Drive actuator names (order must match action[0:6])
DRV_ACTUATORS = [
    "drv_right_front_wheel", "drv_right_middle_wheel", "drv_right_back_wheel",
    "drv_left_front_wheel",  "drv_left_middle_wheel",  "drv_left_back_wheel",
]
# Steering actuator names (order must match action[6:10])
SRV_ACTUATORS = [
    "srv_right_front_rotator", "srv_right_back_rotator",
    "srv_left_front_rotator",  "srv_left_back_rotator",
]
# Passive joint sensor names (same order as PASSIVE_SENSOR_JOINTS in scene_builder)
PASSIVE_SENSORS = [
    "sensor_pass_left_rocker", "sensor_pass_right_rocker",
    "sensor_pass_left_rockerbogie", "sensor_pass_right_rockerbogie",
]

# Reward weights — tweak these to shape behaviour
R_PROGRESS       =  10.0   # reward per metre of forward progress towards target
R_TILT_PENALTY   = -0.5    # per radian of dangerous tilt, per step
R_TIME_PENALTY   = -0.05   # existence penalty per step → encourages speed
R_SUCCESS        = 100.0   # one-time bonus for reaching target
R_FLIP_PENALTY   = -50.0   # one-time penalty for flipping over


class RoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, blind=False):
        super().__init__()
        self.render_mode = render_mode
        self.blind = blind

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        # Pre-cache actuator / sensor IDs for fast lookup
        self._drv_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in DRV_ACTUATORS]
        self._srv_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in SRV_ACTUATORS]
        self._sens_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in PASSIVE_SENSORS}

        # Offscreen renderer for camera images
        if not self.blind:
            self._renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        else:
            self._renderer = None

        # Target position (world XY)
        self._target = np.zeros(2)
        self._prev_dist = 0.0

        # --- Gym spaces ---
        self.observation_space = spaces.Dict({
            "cameras": spaces.Box(low=0, high=255, shape=(N_CAMERAS * 3, IMG_H, IMG_W), dtype=np.uint8),
            "numeric": spaces.Box(low=-np.inf, high=np.inf, shape=(N_NUMERIC,), dtype=np.float32),
        })
        # 10 actions: 6 drive + 4 steer, all in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        # Viewer for human render mode
        self._viewer = None

        # Contact evaluation setup
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        wheel_names = [
            "wheel_right_front geom", "wheel_right_middle geom", "wheel_right_back geom",
            "wheel_left_front geom", "wheel_left_middle geom", "wheel_left_back geom"
        ]
        self._wheel_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, w) for w in wheel_names]
        
        self._has_landed = False
        self._air_time_steps = 0

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Always drop from 50cm — rover settles naturally on its wheels
        # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
        root_jnt_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qpos_adr     = self.model.jnt_qposadr[root_jnt_id]
        self.data.qpos[qpos_adr:qpos_adr + 3] = [0.0, 0.0, 0.1]  # x, y, z
        self.data.qpos[qpos_adr + 3]          = 1.0               # qw  (upright)
        self.data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0           # qx, qy, qz

        # Spawn rover at origin. The scene.xml already puts body at world origin.
        # Spawn target 5–5.1m away in a ±90° arc in front (along +X)
        dist  = self.np_random.uniform(5.0, 5.1)
        angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self._target = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        self._prev_dist = dist

        self._steps = 0
        self._has_landed = False
        self._air_time_steps = 0
        obs  = self._get_obs()
        info = {"target": self._target.copy()}
        return obs, info

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Scale and apply drive velocities
        for i, aid in enumerate(self._drv_ids):
            self.data.ctrl[aid] = action[i] * MAX_WHEEL_VEL

        # Scale and apply steering angles
        for i, aid in enumerate(self._srv_ids):
            self.data.ctrl[aid] = action[6 + i] * MAX_STEER_ANG

        # Advance physics (10 sub-steps for stability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs  = self._get_obs()
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
                frames.append(self._renderer.render())          # (H, W, 3)
            # Stack to (N_CAMERAS*3, H, W)
            cam_obs = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0).astype(np.uint8)

        # --- IMU ---
        # framequat sensor → 4 floats  |  frameangvel → 3 floats
        imu_quat   = self.data.sensor("imu_quat").data.copy().astype(np.float32)
        imu_angvel = self.data.sensor("imu_angvel").data.copy().astype(np.float32)

        # --- Passive joint angles ---
        passive = np.array(
            [self.data.sensor(n).data[0] for n in PASSIVE_SENSORS],
            dtype=np.float32
        )

        # --- Relative target vector (world XY → rover body frame) ---
        rover_pos = self.data.body("body").xpos[:2]
        rel_target = (self._target - rover_pos).astype(np.float32)  # (dx, dy) in world frame

        numeric = np.concatenate([imu_quat, imu_angvel, passive, rel_target])  # (13,)

        return {"cameras": cam_obs, "numeric": numeric}

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------
    def _compute_reward(self):
        rover_pos = self.data.body("body").xpos[:2]
        dist      = float(np.linalg.norm(self._target - rover_pos))
        progress  = self._prev_dist - dist        # positive when moving closer
        self._prev_dist = dist

        # Tilt angle: use body xquat directly (world frame, always correct)
        # MuJoCo xquat convention: [w, x, y, z]
        xquat = self.data.body("body").xquat.copy()
        w, x, y, z = xquat
        body_z_world = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x*x + y*y)
        ])
        tilt = float(np.arccos(np.clip(body_z_world[2], -1.0, 1.0)))

        # Build reward
        reward     = 0.0
        terminated = False
        info       = {"dist": dist, "tilt_rad": tilt, "progress": progress}

        # --- THE CARROT & STICK ---
        # 1. Distance Bleed: Instead of rewarding 'progress', we penalise absolute distance every step.
        # If the target is 5m away, it bleeds -0.5/step. If it is 1m away, it bleeds -0.1/step.
        # This completely prevents circling/farming! It forces the AI to rush the target to stop the bleeding.
        reward -= (dist * 0.1)

        if tilt > 0.3:   
            reward -= tilt * 1.0  # soft penalty for leaning

        # Check wheel contacts
        touching_wheels = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self._floor_geom_id and contact.geom2 in self._wheel_geom_ids:
                touching_wheels.add(contact.geom2)
            elif contact.geom2 == self._floor_geom_id and contact.geom1 in self._wheel_geom_ids:
                touching_wheels.add(contact.geom1)
        
        n_touching = len(touching_wheels)
        
        if not self._has_landed and n_touching >= 1:
            self._has_landed = True
            
        if self._has_landed:
            if n_touching >= 2:
                # 2. Survival Bonus: Positive reinforcement for keeping wheels on the ground
                reward += 0.5
                self._air_time_steps = 0
            else:
                # Soft penalty for lifting wheels, but NOT terrifying enough to cause a suicide loop
                reward -= 1.0
                self._air_time_steps += 1
                
            if self._air_time_steps >= 20:
                reward += R_FLIP_PENALTY  # Final death penalty
                terminated = True
                info["airborne"] = True
                return reward, terminated, info

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
