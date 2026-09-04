"""
components/eyes.py
==================
Pluggable visual perception components for the 6-wheel rocker-bogie rover.

Provides camera sensor abstractions for reinforcement learning and simulation:
- BaseEyes: Abstract base class defining observation spec and rendering interface.
- BlindEyes: Zero-overhead sensor returning empty array or dummy legacy buffers.
  Eliminates MuJoCo OpenGL/EGL context creation for fast kinematic training.
- DepthmapEyes: Offscreen depth buffer renderer utilizing MuJoCo depth mode
  (enable_depth_rendering) across 4 onboard cameras (stacked 4x128x128 float32).
- RGBEyes: Offscreen 3-channel color renderer across 4 onboard cameras
  (concatenated 12x128x128 uint8 in channel-first CHW format).

Camera layout matches 3D_files/mujoco/cameras.xml:
- cam_front_left: Front-facing left camera (points +X)
- cam_side_left: Side-facing left camera (points +Y)
- cam_front_right: Front-facing right camera (points +X)
- cam_side_right: Side-facing right camera (points -Y)
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import mujoco
from torchrl.data import Bounded

# Default camera dimensions and naming matching FreeCAD export
IMG_H: int = 128
IMG_W: int = 128
N_CAMERAS: int = 4
CAM_NAMES: List[str] = [
    "cam_front_left",
    "cam_side_left",
    "cam_front_right",
    "cam_side_right",
]


class BaseEyes(ABC):
    """
    Abstract interface for rover visual perception systems.
    """

    @abstractmethod
    def get_observation_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """
        Return the TorchRL Bounded specification for camera observations.
        """
        pass

    @abstractmethod
    def read(self, env: Any) -> np.ndarray:
        """
        Acquire visual observations from the simulation environment.
        """
        pass

    def close(self) -> None:
        """
        Clean up offscreen renderers and OpenGL/EGL resources.
        """
        pass

    def render_rgb(self, env: Any) -> Optional[np.ndarray]:
        """
        Render a single RGB frame for visualization or recording.
        Returns None if visual rendering is not supported.
        """
        return None


class BlindEyes(BaseEyes):
    """
    Zero-overhead sensor returning zeroed or empty observation arrays.
    Bypasses all OpenGL/EGL rendering calls, eliminating context overhead.
    """

    def __init__(
        self,
        empty_array: bool = True,
        height: int = IMG_H,
        width: int = IMG_W,
        **kwargs: Any,
    ) -> None:
        # Gracefully handle model or env passed as first positional arg
        if not isinstance(empty_array, bool):
            empty_array = not kwargs.get("legacy", False)
        if "legacy" in kwargs:
            empty_array = not kwargs["legacy"]

        self.empty_array: bool = bool(empty_array)
        self.height: int = int(height)
        self.width: int = int(width)

        if self.empty_array:
            self._shape: Tuple[int, ...] = (0,)
        else:
            self._shape = (len(CAM_NAMES) * 3, self.height, self.width)

    def get_observation_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """
        Observation spec for blind mode. Shape is (0,) for empty mode,
        or (12, H, W) for legacy test backward-compatibility mode.
        """
        return Bounded(
            shape=torch.Size(self._shape),
            dtype=torch.uint8,
            low=0,
            high=255,
            device=device,
        )

    def read(self, env: Any) -> np.ndarray:
        """
        Return pre-allocated zero array without simulation rendering.
        """
        return np.zeros(self._shape, dtype=np.uint8)

    def close(self) -> None:
        """
        No-op cleanup since no renderer is instantiated.
        """
        pass

    def render_rgb(self, env: Any) -> Optional[np.ndarray]:
        """
        Blind mode does not support RGB rendering.
        """
        return None


class DepthmapEyes(BaseEyes):
    """
    Offscreen depth sensor utilizing MuJoCo depth buffer rendering.
    Outputs stacked single-channel depth maps across all 4 cameras (4, H, W) in float32.
    """

    def __init__(
        self,
        model_or_env: Optional[Any] = None,
        height: int = IMG_H,
        width: int = IMG_W,
        cameras: Optional[List[str]] = None,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if model_or_env is None and model is not None:
            model_or_env = model
        elif model_or_env is None and "model" in kwargs:
            model_or_env = kwargs["model"]

        if "height" in kwargs:
            height = kwargs["height"]
        if "width" in kwargs:
            width = kwargs["width"]

        self.height: int = int(height)
        self.width: int = int(width)
        self.cameras: List[str] = list(cameras) if cameras is not None else list(CAM_NAMES)
        self.renderer: Optional[mujoco.Renderer] = None

        if model_or_env is not None:
            actual_model = model_or_env.model if hasattr(model_or_env, "model") else model_or_env
            self._init_renderer(actual_model)

    def _init_renderer(self, model: mujoco.MjModel) -> None:
        """
        Initialize offscreen renderer and enable depth rendering mode.
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            self.renderer.enable_depth_rendering()

    def get_observation_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """
        Observation spec for depth mode: shape (N_CAMERAS, H, W), float32 meters.
        """
        return Bounded(
            shape=torch.Size([len(self.cameras), self.height, self.width]),
            dtype=torch.float32,
            low=0.0,
            high=100.0,
            device=device,
        )

    def read(self, env: Any) -> np.ndarray:
        """
        Render depth map for each camera and stack along channel axis.
        Returns shape (N_CAMERAS, H, W) of float32 values.
        """
        if self.renderer is None:
            if hasattr(env, "model"):
                self._init_renderer(env.model)
            else:
                return np.zeros((len(self.cameras), self.height, self.width), dtype=np.float32)

        data = env.data if hasattr(env, "data") else env
        frames: List[np.ndarray] = []
        for cam_name in self.cameras:
            self.renderer.update_scene(data, camera=cam_name)
            depth_map = self.renderer.render()
            frames.append(depth_map)
        return np.stack(frames, axis=0).astype(np.float32)

    def close(self) -> None:
        """
        Safely clean up MuJoCo renderer resources.
        """
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render_rgb(self, env: Any) -> Optional[np.ndarray]:
        """
        Depth renderer does not produce RGB frames.
        """
        return None


class RGBEyes(BaseEyes):
    """
    Offscreen color sensor rendering 3-channel RGB across all 4 cameras.
    Outputs concatenated channel-first frames (12, H, W) in uint8.
    """

    def __init__(
        self,
        model_or_env: Optional[Any] = None,
        height: int = IMG_H,
        width: int = IMG_W,
        cameras: Optional[List[str]] = None,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if model_or_env is None and model is not None:
            model_or_env = model
        elif model_or_env is None and "model" in kwargs:
            model_or_env = kwargs["model"]

        if "height" in kwargs:
            height = kwargs["height"]
        if "width" in kwargs:
            width = kwargs["width"]

        self.height: int = int(height)
        self.width: int = int(width)
        self.cameras: List[str] = list(cameras) if cameras is not None else list(CAM_NAMES)
        self.renderer: Optional[mujoco.Renderer] = None

        if model_or_env is not None:
            actual_model = model_or_env.model if hasattr(model_or_env, "model") else model_or_env
            self._init_renderer(actual_model)

    def _init_renderer(self, model: mujoco.MjModel) -> None:
        """
        Initialize offscreen RGB renderer.
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def get_observation_spec(self, device: Union[torch.device, str] = "cpu") -> Bounded:
        """
        Observation spec for RGB mode: shape (N_CAMERAS * 3, H, W), uint8 [0, 255].
        """
        return Bounded(
            shape=torch.Size([len(self.cameras) * 3, self.height, self.width]),
            dtype=torch.uint8,
            low=0,
            high=255,
            device=device,
        )

    def read(self, env: Any) -> np.ndarray:
        """
        Render RGB frames from each camera, transpose to CHW, and concatenate.
        Returns shape (12, H, W) of uint8 values.
        """
        if self.renderer is None:
            if hasattr(env, "model"):
                self._init_renderer(env.model)
            else:
                return np.zeros(
                    (len(self.cameras) * 3, self.height, self.width), dtype=np.uint8
                )

        data = env.data if hasattr(env, "data") else env
        frames: List[np.ndarray] = []
        for cam_name in self.cameras:
            self.renderer.update_scene(data, camera=cam_name)
            frame = self.renderer.render()
            frames.append(frame.transpose(2, 0, 1))
        return np.concatenate(frames, axis=0).astype(np.uint8)

    def render_rgb(self, env: Any) -> Optional[np.ndarray]:
        """
        Render a single overview RGB frame using default free camera.
        """
        if self.renderer is None and hasattr(env, "model"):
            self._init_renderer(env.model)
        if self.renderer is not None:
            data = env.data if hasattr(env, "data") else env
            self.renderer.update_scene(data)
            return self.renderer.render()
        return None

    def close(self) -> None:
        """
        Safely clean up MuJoCo renderer resources.
        """
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
