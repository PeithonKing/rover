"""
components/controllers.py
=========================
Pluggable actuator controllers for 6-wheel rocker-bogie rover.

This module provides modular controller interfaces mapping continuous agent
action tensors to physical MuJoCo drive actuators and steering servos:
  - BaseController: Abstract base class specifying controller interface.
  - AckermannController: 2-action controller computing synchronized Ackermann
    steering angles and scaled hub velocities for 6-wheel rocker-bogie chassis.
  - DirectController: 10-action controller providing raw per-actuator control
    over all 6 drive wheels and 4 corner steering servos.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import mujoco
from torchrl.data import Bounded

# ---------------------------------------------------------------------------
# Physical constants and hardware bounds
# ---------------------------------------------------------------------------
MAX_WHEEL_VEL = 29.24   # rad/s, maximum angular velocity of drive motors
MAX_COM_SPEED = 2.77    # m/s, maximum center-of-mass linear speed (= MAX_WHEEL_VEL * WHEEL_RADIUS)
WHEEL_RADIUS = 0.095    # m, nominal wheel radius (190 mm wheel diameter)
MAX_STEER_ANG = 0.7854  # rad, maximum steering servo angle (+/- 45 degrees)
STEER_RATE = 1.0 / 20.0 # rad/step, steering accumulation rate per control step

# Standard 6-wheel drive actuator and steering joint layout:
# [right_front, right_middle, right_back, left_front, left_middle, left_back]
# Signs: right side Y is negative in body frame, left side Y is positive
DRV_JOINTS = [
    ("drv_right_front_wheel", "srv_right_front_rotator", -1),
    ("drv_right_middle_wheel", None, -1),
    ("drv_right_back_wheel", "srv_right_back_rotator", -1),
    ("drv_left_front_wheel", "srv_left_front_rotator", +1),
    ("drv_left_middle_wheel", None, +1),
    ("drv_left_back_wheel", "srv_left_back_rotator", +1),
]


class BaseController(ABC):
    """
    Abstract base class for pluggable rover actuator controllers.

    Subclasses must implement action_dim, get_action_spec, and apply_action.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the continuous control action space."""
        pass

    @abstractmethod
    def get_action_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """Returns the TorchRL Bounded action specification."""
        pass

    def reset(self) -> None:
        """Resets internal controller state (e.g., speed and steering accumulators)."""
        pass

    @abstractmethod
    def apply_action(self, env, action: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Translates raw policy actions and writes control commands directly to env.data.ctrl.

        Args:
            env: RoverEnv instance containing MuJoCo model, data, and wheel info.
            action: Action tensor or array of shape [action_dim] with values in [-1.0, 1.0].
        """
        pass

    def apply(self, env, action: Union[torch.Tensor, np.ndarray]) -> None:
        """Compatibility alias forwarding directly to apply_action."""
        self.apply_action(env, action)


class AckermannController(BaseController):
    """
    Two-dimensional continuous action controller using true Ackermann kinematics.

    Action definition:
      action[0]: Longitudinal acceleration command in [-1.0, 1.0].
                 Accumulates linear speed: com_speed += accel * MAX_COM_SPEED * 0.1,
                 clamped to [-MAX_COM_SPEED, MAX_COM_SPEED].
      action[1]: Steering rate command in [-1.0, 1.0].
                 Accumulates steering target: current_steer += steer * STEER_RATE,
                 clamped to [-1.0, 1.0].

    Kinematics formulation:
      1. Dynamic body frame projection:
         Transforms global joint anchor positions to rover body local coordinates
         via R_body_to_world.T (transpose of body rotation matrix data.body('body').xmat).
         lp = R_body_to_world.T * (xanchor - center_pos)
         X_i = lp[0] (longitudinal offset from chassis center)
         Y_i = lp[1] (lateral offset, positive left, negative right)

      2. Ackermann curvature and angular velocity:
         Track width: W_track = 2 * |Y_i| (nominal ~0.3546 m)
         Yaw rate: omega = current_steer * (2 * MAX_COM_SPEED / W_track)

      3. Per-wheel tangent velocity components:
         Vx_i = com_speed - omega * Y_i
         Vy_i = omega * X_i

      4. Steering servo angle calculation (4 corner wheels):
         steer_angle = arctan2(Vy_i, max(|Vx_i|, 1e-6) * sign(Vx_i))
         clamped to [-MAX_STEER_ANG, MAX_STEER_ANG].

      5. Wheel hub speed and motor angular velocity conversion:
         hub_speed_i = sqrt(Vx_i^2 + Vy_i^2) * sign(Vx_i)
         Normalized so no wheel exceeds MAX_COM_SPEED:
           scale = min(1.0, MAX_COM_SPEED / max(|hub_speed|))
         motor_angular_velocity = (hub_speed_i * scale) / WHEEL_RADIUS
         clamped to [-MAX_WHEEL_VEL, MAX_WHEEL_VEL].
    """

    def __init__(self) -> None:
        self._com_speed: float = 0.0
        self._current_steer: float = 0.0

    @property
    def action_dim(self) -> int:
        """2 continuous actions: [accel_cmd, steer_cmd]."""
        return 2

    @property
    def com_speed(self) -> float:
        """Current accumulated center-of-mass linear speed in m/s."""
        return self._com_speed

    @com_speed.setter
    def com_speed(self, value: float) -> None:
        self._com_speed = float(value)

    @property
    def current_steer(self) -> float:
        """Current accumulated steering demand in [-1.0, 1.0]."""
        return self._current_steer

    @current_steer.setter
    def current_steer(self, value: float) -> None:
        self._current_steer = float(value)

    def reset(self) -> None:
        """Resets drive momentum and accumulated steering angle to zero."""
        self._com_speed = 0.0
        self._current_steer = 0.0

    def get_action_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """Returns 2D Bounded spec in [-1.0, 1.0]."""
        return Bounded(
            shape=torch.Size([2]),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
            device=device,
        )

    def apply_action(self, env, action: Union[torch.Tensor, np.ndarray]) -> None:
        """Executes full Ackermann kinematic calculations and writes to env.data.ctrl."""
        if isinstance(action, torch.Tensor):
            act = action.detach().cpu().numpy().reshape(-1)
        else:
            act = np.asarray(action, dtype=np.float32).reshape(-1)

        act = np.clip(act, -1.0, 1.0)
        accel_cmd = float(act[0])
        steer_cmd = float(act[1])

        # 1. Accumulate drive momentum state
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

        # 2. Dynamic body frame projection matrix
        xmat_inv = env.data.body("body").xmat.reshape(3, 3).T
        center_pos = env.data.body("body").xpos

        def get_local_pos(jnt_id: int) -> np.ndarray:
            global_anchor = env.data.xanchor[jnt_id].copy()
            return np.dot(xmat_inv, global_anchor - center_pos)

        wheel_info = getattr(env, "_wheel_info", None)
        if wheel_info is None:
            # Fallback if env._wheel_info has not been constructed
            wheel_info = []
            for drv_name, srv_name, lat_sign in DRV_JOINTS:
                drv_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, drv_name)
                srv_id = (
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, srv_name)
                    if srv_name
                    else None
                )
                drv_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, drv_name)
                wheel_info.append((drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name))

        # 3. Compute per-wheel velocity and steering commands
        wheel_speeds: List[float] = []
        for drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name in wheel_info:
            if srv_id is not None:
                assert srv_name is not None
                if hasattr(env, "_srv_jnt_ids") and srv_name in env._srv_jnt_ids:
                    srv_jnt_id = env._srv_jnt_ids[srv_name]
                else:
                    srv_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, srv_name)
                lp = get_local_pos(srv_jnt_id)
            else:
                lp = get_local_pos(drv_jnt_id)

            X_i = lp[0]  # longitudinal coordinate (forward/backward)
            Y_i = lp[1]  # lateral coordinate (left/right)

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
                env.data.ctrl[srv_id] = steer_angle

            # Resultant wheel hub velocity
            hub_speed = float(np.sqrt(Vx_i**2 + Vy_i**2)) * np.sign(Vx_i + 1e-9)
            wheel_speeds.append(hub_speed)

        # 4. Normalize wheel speeds so no motor exceeds physical speed limits
        max_hub = max(abs(s) for s in wheel_speeds) if wheel_speeds else 1.0
        scale = 1.0 if max_hub <= MAX_COM_SPEED else MAX_COM_SPEED / max_hub

        for i, (drv_id, _, _, _, _, _) in enumerate(wheel_info):
            motor_rads = (wheel_speeds[i] * scale) / WHEEL_RADIUS
            env.data.ctrl[drv_id] = float(
                np.clip(motor_rads, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            )


class DirectController(BaseController):
    """
    Ten-dimensional continuous action controller actuating all motors and servos directly.

    Action mapping (inputs normalized to [-1.0, 1.0]):
      action[0:6]: Drive motor angular velocities (scaled to [-MAX_WHEEL_VEL, MAX_WHEEL_VEL] rad/s)
        - action[0]: drv_right_front_wheel
        - action[1]: drv_right_middle_wheel
        - action[2]: drv_right_back_wheel
        - action[3]: drv_left_front_wheel
        - action[4]: drv_left_middle_wheel
        - action[5]: drv_left_back_wheel
      action[6:10]: Steering servo angles (scaled to [-MAX_STEER_ANG, MAX_STEER_ANG] rad)
        - action[6]: srv_right_front_rotator
        - action[7]: srv_right_back_rotator
        - action[8]: srv_left_front_rotator
        - action[9]: srv_left_back_rotator
    """

    def __init__(self) -> None:
        self._com_speed: float = 0.0
        self._current_steer: float = 0.0

    @property
    def action_dim(self) -> int:
        """10 continuous actions: 6 drive wheel velocities + 4 steering angles."""
        return 10

    @property
    def com_speed(self) -> float:
        """Telemetry surrogate for center-of-mass linear speed in m/s."""
        return self._com_speed

    @com_speed.setter
    def com_speed(self, value: float) -> None:
        self._com_speed = float(value)

    @property
    def current_steer(self) -> float:
        """Telemetry surrogate for average steering servo angle command."""
        return self._current_steer

    @current_steer.setter
    def current_steer(self, value: float) -> None:
        self._current_steer = float(value)

    def reset(self) -> None:
        """Resets telemetry surrogate variables."""
        self._com_speed = 0.0
        self._current_steer = 0.0

    def get_action_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """Returns 10D Bounded spec in [-1.0, 1.0]."""
        return Bounded(
            shape=torch.Size([10]),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
            device=device,
        )

    def apply_action(self, env, action: Union[torch.Tensor, np.ndarray]) -> None:
        """Directly writes scaled drive velocities and servo angles to env.data.ctrl."""
        if isinstance(action, torch.Tensor):
            act = action.detach().cpu().numpy().reshape(-1)
        else:
            act = np.asarray(action, dtype=np.float32).reshape(-1)

        act = np.clip(act, -1.0, 1.0)
        if len(act) < 10:
            raise ValueError(f"DirectController requires 10 action values, received {len(act)}")

        wheel_info = getattr(env, "_wheel_info", None)
        if wheel_info is None:
            wheel_info = []
            for drv_name, srv_name, lat_sign in DRV_JOINTS:
                drv_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, drv_name)
                srv_id = (
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, srv_name)
                    if srv_name
                    else None
                )
                drv_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, drv_name)
                wheel_info.append((drv_id, srv_id, drv_jnt_id, lat_sign, drv_name, srv_name))

        # 6 Drive Motors: indices 0 to 5
        for i in range(6):
            drv_id = wheel_info[i][0]
            motor_vel = float(act[i] * MAX_WHEEL_VEL)
            env.data.ctrl[drv_id] = float(np.clip(motor_vel, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))

        # 4 Steering Servos: indices 6 to 9 (applied to wheels with steering servos)
        steer_idx = 6
        for i in range(6):
            srv_id = wheel_info[i][1]
            if srv_id is not None:
                steer_angle = float(act[steer_idx] * MAX_STEER_ANG)
                env.data.ctrl[srv_id] = float(np.clip(steer_angle, -MAX_STEER_ANG, MAX_STEER_ANG))
                steer_idx += 1

        # Update telemetry surrogate variables for backward compatibility
        self._com_speed = float(np.mean(act[:6]) * MAX_COM_SPEED)
        self._current_steer = float(np.mean(act[6:10])) if steer_idx > 6 else 0.0
