import numpy as np
from typing import Any

PASSIVE_SENSORS = [
    "sensor_pass_left_rocker",
    "sensor_pass_right_rocker",
    "sensor_pass_left_rockerbogie",
    "sensor_pass_right_rockerbogie",
]

class BaseNumericObservation:
    def __init__(self, dim: int):
        self.dim = dim

    def _get_base_state(self, env: Any) -> np.ndarray:
        imu_quat = env.data.sensor("imu_quat").data.copy().astype(np.float32)
        imu_angvel = env.data.sensor("imu_angvel").data.copy().astype(np.float32)
        passive = np.array(
            [env.data.sensor(n).data[0] for n in PASSIVE_SENSORS], dtype=np.float32
        )
        return np.concatenate([imu_quat, imu_angvel, passive], dtype=np.float32)

    def read(self, env: Any) -> np.ndarray:
        raise NotImplementedError

class TargetAwareObservation(BaseNumericObservation):
    """13-dimensional observation including the relative GPS target."""
    def __init__(self):
        super().__init__(dim=13)

    def read(self, env: Any) -> np.ndarray:
        base = self._get_base_state(env)
        rover_pos = env.data.body("body").xpos[:2]
        world_delta = (env._target - rover_pos).astype(np.float32)
        xmat = env.data.body("body").xmat.reshape(3, 3)
        forward = xmat[:2, 0]
        right = xmat[:2, 1]
        local_dx = float(np.dot(world_delta, forward))
        local_dy = float(np.dot(world_delta, right))
        rel_target = np.array([local_dx, local_dy], dtype=np.float32)
        return np.concatenate([base, rel_target], dtype=np.float32)

class TargetBlindObservation(BaseNumericObservation):
    """11-dimensional observation stripping the relative GPS target (for vision models)."""
    def __init__(self):
        super().__init__(dim=11)

    def read(self, env: Any) -> np.ndarray:
        return self._get_base_state(env)
