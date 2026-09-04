"""
components/rewards.py
=====================
Pluggable reward and termination functions for 6-wheel rover.

Defines:
- BaseReward: Abstract base class defining reset and compute interface.
- StandardReward: Distance-based dense progress reward, orientation tilt
  flip check, sky-high launch check, and waypoint goal success bonus.
- EnergyPenaltyReward: Inherits StandardReward and adds quadratic actuator
  effort penalty for regularizing continuous 10D action spaces.

All comments and strings strictly use 7-bit ASCII characters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Physical and simulation thresholds
SUCCESS_RADIUS = 0.5  # meters to waypoint target
MAX_TILT = 1.2  # radians (~68.75 degrees), beyond which robot is flipped
R_SUCCESS = 100.0  # terminal bonus for reaching target waypoint
R_FLIP_PENALTY = -100.0  # terminal penalty for flipping or sky-high launch


class BaseReward(ABC):
    """
    Abstract base class for environment reward and termination functions.
    """

    def reset(self, env: Any) -> None:
        """
        Optional reset callback invoked during env.reset().

        Parameters
        ----------
        env : Any
            The RoverEnv instance being reset.
        """
        pass

    @abstractmethod
    def compute(
        self, env: Any, action: Optional[Any] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Computes step reward, termination status, and telemetry dictionary.

        Parameters
        ----------
        env : Any
            The RoverEnv instance after physics step.
        action : Optional[Any]
            Optional action array/tensor passed for effort calculation.

        Returns
        -------
        Tuple[float, bool, Dict[str, Any]]
            reward : float
                Scalar reward for the current step.
            terminated : bool
                True if episode terminated due to goal success or failure.
            info : Dict[str, Any]
                Dictionary containing telemetry metrics and status flags.
        """
        pass


class StandardReward(BaseReward):
    """
    Standard dense distance-based reward with progress tracking,
    orientation tilt flip check, sky-high launch check, and goal success.
    """

    def __init__(
        self,
        distance_weight: float = 0.1,
        success_radius: float = SUCCESS_RADIUS,
        max_tilt: float = MAX_TILT,
        success_reward: float = R_SUCCESS,
        flip_penalty: float = R_FLIP_PENALTY,
        sky_penalty: float = R_FLIP_PENALTY,
    ) -> None:
        self.distance_weight = float(distance_weight)
        self.success_radius = float(success_radius)
        self.max_tilt = float(max_tilt)
        self.success_reward = float(success_reward)
        self.flip_penalty = float(flip_penalty)
        self.sky_penalty = float(sky_penalty)
        self._prev_dist: float = 0.0

    @property
    def prev_dist(self) -> float:
        """Cached previous Euclidean distance to target."""
        return self._prev_dist

    @prev_dist.setter
    def prev_dist(self, value: float) -> None:
        self._prev_dist = float(value)

    def reset(self, env: Any) -> None:
        """
        Initializes distance tracking from the rover body position to target.
        """
        rover_pos = env.data.body("body").xpos[:2]
        self._prev_dist = float(np.linalg.norm(env._target - rover_pos))
        if hasattr(env, "_prev_dist"):
            env._prev_dist = self._prev_dist

    def compute(
        self, env: Any, action: Optional[Any] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates distance, progress, tilt angle, terminal conditions,
        and telemetry info.
        """
        rover_pos = env.data.body("body").xpos[:2]
        dist = float(np.linalg.norm(env._target - rover_pos))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        if hasattr(env, "_prev_dist"):
            env._prev_dist = dist

        # Extract rover chassis orientation: body Z-axis in world frame
        # Quat format in MuJoCo: [w, x, y, z]
        xquat = env.data.body("body").xquat.copy()
        w, x, y, z = xquat
        body_z_world = np.array(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float32,
        )
        tilt = float(np.arccos(np.clip(body_z_world[2], -1.0, 1.0)))

        info: Dict[str, Any] = {
            "dist": dist,
            "tilt_rad": tilt,
            "progress": progress,
            "success": False,
            "flipped": False,
            "sky_high": False,
        }

        # Dense distance penalty: closer to target yields smaller penalty
        reward = -self.distance_weight * dist
        terminated = False

        if dist < self.success_radius:
            reward = self.success_reward
            terminated = True
            info["success"] = True
        elif tilt > self.max_tilt:
            reward = self.flip_penalty
            terminated = True
            info["flipped"] = True
        elif env.data.body("body").xpos[2] > 1.0:
            reward = self.sky_penalty
            terminated = True
            info["sky_high"] = True

        return reward, terminated, info


class EnergyPenaltyReward(StandardReward):
    """
    Reward module penalizing actuator energy usage sum(ctrl**2)
    in addition to the standard objective. Stabilizes the
    10-dimensional direct motor action space.
    """

    def __init__(
        self,
        energy_weight: float = 0.001,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.energy_weight = float(energy_weight)

    def compute(
        self, env: Any, action: Optional[Any] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates standard reward and subtracts actuator quadratic penalty.
        """
        reward, terminated, info = super().compute(env, action=action)
        actuator_energy = float(np.sum(np.square(env.data.ctrl)))
        penalty = self.energy_weight * actuator_energy
        reward -= penalty
        info["energy_penalty"] = penalty
        info["actuator_energy"] = actuator_energy
        return reward, terminated, info
