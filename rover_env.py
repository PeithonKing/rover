import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
"""
rover_env.py
============
Modular TorchRL environment for the 6-wheel rocker-bogie rover in MuJoCo.
Subclasses torchrl.envs.EnvBase and natively produces TensorDict instances.

Hollow Motherboard Architecture:
--------------------------------
Extracts environment mechanics into pluggable components while isolating the
underlying MuJoCo physics engine:
- Controller: BaseController (AckermannController or DirectController)
- Vision / Eyes: BaseEyes (BlindEyes, DepthmapEyes, RGBEyes)
- Reward: BaseReward (StandardReward, EnergyPenaltyReward)
- Terrain: BaseTerrain (FlatTerrain)

Backward Compatibility:
-----------------------
Provides transparent property proxies for legacy attributes and properties:
- _current_steer (proxies to self.controller._current_steer)
- current_steer (alias for _current_steer)
- _com_speed (proxies to self.controller._com_speed)
- com_speed (alias for _com_speed)
- _prev_dist (proxies to self.reward.prev_dist)
- _wheel_info (cached 6-wheel actuator and joint telemetry)
- last_info (returns dictionary containing step telemetry)
- blind (indicates whether environment operates in blind mode)
- xml_path (proxies to self.terrain.xml_path)
- control_mode, vision_mode, terrain_mode, reward_mode (string mode inspectors)
- _compute_reward_done (compatibility method alias for test suites)

All strings and comments strictly use 7-bit ASCII characters.
Zero main() wrapper patterns or top-level execution constructs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Enable headless EGL rendering if not explicitly overridden
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

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

from components import (
    BaseController,
    AckermannController,
    DirectController,
    BaseEyes,
    BlindEyes,
    DepthmapEyes,
    RGBEyes,
    BaseReward,
    StandardReward,
    EnergyPenaltyReward,
    BaseTerrain,
    FlatTerrain,
    ComponentBundle,
    make_components,
    MAX_WHEEL_VEL,
    MAX_COM_SPEED,
    WHEEL_RADIUS,
    MAX_STEER_ANG,
    STEER_RATE,
    DRV_JOINTS,
    IMG_H,
    IMG_W,
    N_CAMERAS,
    CAM_NAMES,
    SUCCESS_RADIUS,
    MAX_TILT,
    R_SUCCESS,
    R_FLIP_PENALTY,
)

# ---------------------------------------------------------------------------
# Constants and Hardware Layout
# ---------------------------------------------------------------------------
SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D_files/mujoco/scene.xml")
MAX_STEPS = 2000
N_NUMERIC = 13

PASSIVE_SENSORS = [
    "sensor_pass_left_rocker",
    "sensor_pass_right_rocker",
    "sensor_pass_left_rockerbogie",
    "sensor_pass_right_rockerbogie",
]


class RoverEnv(EnvBase):
    """
    TorchRL native environment for the 6-wheel rover.
    Hollow motherboard orchestrating pluggable components with 100% legacy API compatibility.
    """

    def __init__(
        self,
        *args: Any,
        controller: Optional[BaseController] = None,
        eyes: Optional[BaseEyes] = None,
        reward: Optional[BaseReward] = None,
        terrain: Optional[Union[BaseTerrain, str]] = None,
        control_mode: Optional[Union[str, BaseController]] = None,
        vision_mode: Optional[Union[str, BaseEyes]] = None,
        terrain_mode: Optional[Union[str, BaseTerrain]] = None,
        reward_mode: Optional[Union[str, BaseReward]] = None,
        render_mode: Optional[str] = None,
        blind: Optional[bool] = None,
        device: Union[torch.device, str] = "cpu",
        xml_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)

        # Handle potential positional arguments for legacy signatures
        if len(args) > 0:
            if isinstance(args[0], ComponentBundle):
                controller = args[0].controller
                eyes = args[0].eyes
                reward = args[0].reward
                terrain = args[0].terrain
            elif isinstance(args[0], BaseController):
                controller = args[0]
                if len(args) > 1 and isinstance(args[1], BaseEyes):
                    eyes = args[1]
                if len(args) > 2 and isinstance(args[2], BaseReward):
                    reward = args[2]
                if len(args) > 3 and isinstance(args[3], BaseTerrain):
                    terrain = args[3]
            elif isinstance(args[0], str) or args[0] is None:
                render_mode = args[0]
                if len(args) > 1 and isinstance(args[1], bool):
                    blind = args[1]
                if len(args) > 2 and isinstance(args[2], (torch.device, str)):
                    self.device = torch.device(args[2])

        self.render_mode = render_mode

        # 1. Resolve Terrain
        actual_terrain = (
            terrain
            if terrain is not None
            else (terrain_mode if terrain_mode is not None else "flat")
        )
        if isinstance(actual_terrain, BaseTerrain):
            self.terrain = actual_terrain
        elif isinstance(actual_terrain, str):
            self.terrain = FlatTerrain(xml_path=xml_path)
        elif xml_path is not None:
            self.terrain = FlatTerrain(xml_path=xml_path)
        else:
            self.terrain = FlatTerrain()

        # Compile MuJoCo physics model and data from terrain XML path
        self.model = mujoco.MjModel.from_xml_path(self.terrain.xml_path)
        self.data = mujoco.MjData(self.model)

        # 2. Resolve Controller
        actual_controller = (
            controller
            if controller is not None
            else (control_mode if control_mode is not None else "ackermann")
        )

        # 3. Resolve Reward
        actual_reward = (
            reward
            if reward is not None
            else (reward_mode if reward_mode is not None else "standard")
        )

        # 4. Resolve Eyes (Vision)
        if eyes is not None:
            actual_eyes = eyes
        elif vision_mode is not None:
            actual_eyes = vision_mode
            if isinstance(actual_eyes, str) and actual_eyes.lower().strip() == "blind":
                # Default empty_array=True for CLI/modern vision_mode="blind" unless overridden
                kwargs.setdefault("empty_array", True)
        elif blind is not None:
            actual_eyes = "blind" if blind else "rgb"
            if blind:
                # Default empty_array=False for legacy RoverEnv(blind=True)
                kwargs.setdefault("empty_array", False)
        else:
            actual_eyes = "rgb" if render_mode == "rgb_array" else "blind"
            if actual_eyes == "blind":
                kwargs.setdefault("empty_array", False)

        # 5. Instantiate components via factory
        bundle: ComponentBundle = make_components(
            control_mode=actual_controller,
            vision_mode=actual_eyes,
            terrain=self.terrain,
            reward_mode=actual_reward,
            model=self.model,
            xml_path=xml_path,
            **kwargs,
        )

        self.controller: BaseController = bundle.controller
        self.eyes: BaseEyes = bundle.eyes
        self.reward: BaseReward = bundle.reward
        self.reward_fn: BaseReward = bundle.reward
        self.terrain: BaseTerrain = bundle.terrain

        # Ensure offscreen renderer initialized for DepthmapEyes and RGBEyes
        if hasattr(self.eyes, "renderer") and getattr(self.eyes, "renderer", None) is None:
            if hasattr(self.eyes, "_init_renderer"):
                self.eyes._init_renderer(self.model)

        self._blind: bool = isinstance(self.eyes, BlindEyes)

        # Pre-cache actuator, joint, and sensor IDs for fast kinematics lookup
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

        # Viewer and overview renderer handles
        self._viewer: Optional[mujoco.viewer.Handle] = None
        self._overview_renderer: Optional[mujoco.Renderer] = None

        # Episode state variables
        self._target: np.ndarray = np.zeros(2, dtype=np.float32)
        self._steps: int = 0
        self._last_info: Dict[str, Any] = {}
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.set_seed(seed)

        # Build dynamic TorchRL specifications
        self._make_spec()

    def _make_spec(self) -> None:
        """Defines dynamic TorchRL specs delegated to injected components."""
        cameras_spec = self.eyes.get_observation_spec(device=self.device)
        numeric_spec = UnboundedContinuous(
            shape=torch.Size([N_NUMERIC]),
            dtype=torch.float32,
            device=self.device,
        )
        self.observation_spec = Composite(
            cameras=cameras_spec,
            numeric=numeric_spec,
            shape=torch.Size([]),
            device=self.device,
        )

        self.action_spec = self.controller.get_action_spec(device=self.device)

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
        Resets simulation physics, target position, and component kinematics.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Spawn rover dropped from z=0.1m at origin facing +X
        root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qpos_adr = self.model.jnt_qposadr[root_jnt_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = [0.0, 0.0, 0.1]
        self.data.qpos[qpos_adr + 3] = 1.0  # qw (upright)
        self.data.qpos[qpos_adr + 4 : qpos_adr + 7] = 0.0  # qx, qy, qz

        mujoco.mj_forward(self.model, self.data)

        # Spawn target 5.0-5.1m away in a +/-90 degrees arc in front (+X)
        dist = float(self.np_random.uniform(5.0, 5.1))
        angle = float(self.np_random.uniform(-np.pi / 2, np.pi / 2))
        self._target = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)

        # Update yellow sphere marker position in MuJoCo scene
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
        if site_id != -1:
            self.model.site_pos[site_id][:2] = self._target

        # Reset components
        if hasattr(self.terrain, "randomize"):
            self.terrain.randomize(self)
        self.controller.reset()
        self.reward.reset(self)

        self._steps = 0
        self._last_info = {
            "target": self._target.copy(),
            "dist": dist,
            "tilt_rad": 0.0,
            "progress": 0.0,
            "success": False,
            "flipped": False,
            "sky_high": False,
        }

        obs_cameras, obs_numeric = self._get_obs()
        cam_dtype = self.observation_spec["cameras"].dtype

        return TensorDict(
            {
                "cameras": torch.as_tensor(obs_cameras, dtype=cam_dtype, device=self.device),
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
        Executes one control step (10 MuJoCo substeps) delegating to components.
        """
        action = tensordict.get("action")

        # 1. Apply action through Controller component
        self.controller.apply_action(self, action)

        # 2. Advance simulation by 10 substeps (dt=0.002s -> 0.02s per step)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # 3. Collect observations from Eyes and simulation state
        obs_cameras, obs_numeric = self._get_obs()

        # 4. Compute reward and termination through Reward component
        reward_val, terminated, info = self._compute_reward(action=action)
        self._steps += 1
        truncated = self._steps >= MAX_STEPS
        done = bool(terminated or truncated)

        if self.render_mode == "human":
            self._render_human()

        self._last_info = info
        cam_dtype = self.observation_spec["cameras"].dtype

        return TensorDict(
            {
                "cameras": torch.as_tensor(obs_cameras, dtype=cam_dtype, device=self.device),
                "numeric": torch.as_tensor(obs_numeric, dtype=torch.float32, device=self.device),
                "reward": torch.tensor([reward_val], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _get_numeric_obs(self) -> np.ndarray:
        """Collects 13-dimensional proprioception and local target vector."""
        imu_quat = self.data.sensor("imu_quat").data.copy().astype(np.float32)
        imu_angvel = self.data.sensor("imu_angvel").data.copy().astype(np.float32)

        passive = np.array(
            [self.data.sensor(n).data[0] for n in PASSIVE_SENSORS], dtype=np.float32
        )

        rover_pos = self.data.body("body").xpos[:2]
        world_delta = (self._target - rover_pos).astype(np.float32)

        xmat = self.data.body("body").xmat.reshape(3, 3)
        forward = xmat[:2, 0]  # Local +X in world XY
        right = xmat[:2, 1]    # Local +Y in world XY

        local_dx = float(np.dot(world_delta, forward))
        local_dy = float(np.dot(world_delta, right))
        rel_target = np.array([local_dx, local_dy], dtype=np.float32)

        return np.concatenate([imu_quat, imu_angvel, passive, rel_target], dtype=np.float32)

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collects camera perception from eyes and 13-dim numeric observation."""
        return self.eyes.read(self), self._get_numeric_obs()

    def _compute_reward(
        self, action: Optional[Any] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Delegates reward and termination evaluation to the injected Reward component."""
        reward_val, terminated, info = self.reward.compute(self, action=action)
        self._last_info = info
        return reward_val, terminated, info

    def _compute_reward_done(
        self, action: Optional[Any] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Compatibility method alias for test suites accessing _compute_reward_done."""
        return self._compute_reward(action=action)

    # -----------------------------------------------------------------------
    # Backward-Compatible Property Proxies
    # -----------------------------------------------------------------------
    @property
    def _current_steer(self) -> float:
        """Backward-compatible proxy returning current accumulated steering demand."""
        if hasattr(self, "controller"):
            if hasattr(self.controller, "_current_steer"):
                return float(self.controller._current_steer)
            if hasattr(self.controller, "current_steer"):
                return float(self.controller.current_steer)
        return getattr(self, "_legacy_current_steer", 0.0)

    @_current_steer.setter
    def _current_steer(self, value: float) -> None:
        """Backward-compatible setter for accumulated steering demand."""
        val = float(value)
        self._legacy_current_steer = val
        if hasattr(self, "controller"):
            if hasattr(self.controller, "_current_steer"):
                self.controller._current_steer = val
            elif hasattr(self.controller, "current_steer"):
                self.controller.current_steer = val

    @property
    def current_steer(self) -> float:
        """Alias for _current_steer."""
        return self._current_steer

    @current_steer.setter
    def current_steer(self, value: float) -> None:
        self._current_steer = value

    @property
    def _com_speed(self) -> float:
        """Backward-compatible proxy returning accumulated linear speed."""
        if hasattr(self, "controller"):
            if hasattr(self.controller, "_com_speed"):
                return float(self.controller._com_speed)
            if hasattr(self.controller, "com_speed"):
                return float(self.controller.com_speed)
        return getattr(self, "_legacy_com_speed", 0.0)

    @_com_speed.setter
    def _com_speed(self, value: float) -> None:
        """Backward-compatible setter for accumulated linear speed."""
        val = float(value)
        self._legacy_com_speed = val
        if hasattr(self, "controller"):
            if hasattr(self.controller, "_com_speed"):
                self.controller._com_speed = val
            elif hasattr(self.controller, "com_speed"):
                self.controller.com_speed = val

    @property
    def com_speed(self) -> float:
        """Alias for _com_speed."""
        return self._com_speed

    @com_speed.setter
    def com_speed(self, value: float) -> None:
        self._com_speed = value

    @property
    def _prev_dist(self) -> float:
        """Backward-compatible proxy returning previous distance to target."""
        if hasattr(self, "reward"):
            if hasattr(self.reward, "prev_dist"):
                return float(self.reward.prev_dist)
            if hasattr(self.reward, "_prev_dist"):
                return float(self.reward._prev_dist)
        return getattr(self, "_cached_prev_dist", 0.0)

    @_prev_dist.setter
    def _prev_dist(self, value: float) -> None:
        """Backward-compatible setter for previous distance to target."""
        val = float(value)
        self._cached_prev_dist = val
        if hasattr(self, "reward"):
            if hasattr(self.reward, "prev_dist"):
                self.reward.prev_dist = val
            elif hasattr(self.reward, "_prev_dist"):
                self.reward._prev_dist = val

    @property
    def last_info(self) -> Dict[str, Any]:
        """Access latest physics telemetry and status flags."""
        return self._last_info

    @property
    def blind(self) -> bool:
        """Indicates whether environment operates in blind mode."""
        if hasattr(self, "_blind") and self._blind is not None:
            return bool(self._blind)
        return isinstance(getattr(self, "eyes", None), BlindEyes)

    @blind.setter
    def blind(self, value: bool) -> None:
        self._blind = bool(value)

    @property
    def xml_path(self) -> str:
        """Absolute path to the loaded MuJoCo model XML file."""
        if hasattr(self, "terrain") and hasattr(self.terrain, "xml_path"):
            return self.terrain.xml_path
        return SCENE_XML

    @property
    def control_mode(self) -> str:
        """Identifier of active control scheme ('ackermann' or 'direct')."""
        if hasattr(self, "controller"):
            if isinstance(self.controller, AckermannController):
                return "ackermann"
            if isinstance(self.controller, DirectController):
                return "direct"
            return getattr(self.controller, "name", self.controller.__class__.__name__.lower())
        return "ackermann"

    @property
    def vision_mode(self) -> str:
        """Identifier of active perception scheme ('blind', 'depthmap', or 'rgb')."""
        if hasattr(self, "eyes"):
            if isinstance(self.eyes, BlindEyes):
                return "blind"
            if isinstance(self.eyes, DepthmapEyes):
                return "depthmap"
            if isinstance(self.eyes, RGBEyes):
                return "rgb"
            return getattr(self.eyes, "name", self.eyes.__class__.__name__.lower())
        return "blind" if self.blind else "rgb"

    @property
    def terrain_mode(self) -> str:
        """Identifier of active terrain environment ('flat')."""
        if hasattr(self, "terrain"):
            if isinstance(self.terrain, FlatTerrain):
                return "flat"
            return getattr(self.terrain, "name", self.terrain.__class__.__name__.lower())
        return "flat"

    @property
    def reward_mode(self) -> str:
        """Identifier of active reward objective ('standard' or 'energy')."""
        if hasattr(self, "reward"):
            if isinstance(self.reward, EnergyPenaltyReward):
                return "energy"
            if isinstance(self.reward, StandardReward):
                return "standard"
            return getattr(self.reward, "name", self.reward.__class__.__name__.lower())
        return "standard"

    @property
    def _renderer(self) -> Optional[mujoco.Renderer]:
        """Backward-compatible property accessing active camera or overview renderer."""
        if hasattr(self, "eyes") and hasattr(self.eyes, "renderer") and self.eyes.renderer is not None:
            return self.eyes.renderer
        return getattr(self, "_overview_renderer", None)

    @_renderer.setter
    def _renderer(self, renderer: Optional[mujoco.Renderer]) -> None:
        self._overview_renderer = renderer
        if hasattr(self, "eyes") and hasattr(self.eyes, "renderer"):
            self.eyes.renderer = renderer

    # -----------------------------------------------------------------------
    # Rendering and Resource Lifecycle
    # -----------------------------------------------------------------------
    def _render_human(self) -> None:
        """Synchronizes passive MuJoCo viewer for human render mode."""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def render(self) -> Optional[np.ndarray]:
        """
        Renders an RGB array for visual inspection.
        Delegates first to self.eyes.render_rgb(self).
        Falls back to lazy self._overview_renderer when self.eyes is non-RGB.
        """
        if self.render_mode == "rgb_array":
            if hasattr(self, "eyes") and hasattr(self.eyes, "render_rgb"):
                frame = self.eyes.render_rgb(self)
                if frame is not None:
                    return frame

            if self._overview_renderer is None and hasattr(self, "model"):
                self._overview_renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
            if self._overview_renderer is not None and hasattr(self, "data"):
                self._overview_renderer.update_scene(self.data)
                return self._overview_renderer.render()

        return None

    def close(self, *args: Any, **kwargs: Any) -> None:
        """Cleans up viewer, renderer, and sensor resources."""
        if getattr(self, "_viewer", None) is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

        if hasattr(self, "eyes") and hasattr(self.eyes, "close"):
            try:
                self.eyes.close()
            except Exception:
                pass

        if getattr(self, "_overview_renderer", None) is not None:
            try:
                self._overview_renderer.close()
            except Exception:
                pass
            self._overview_renderer = None

    def __del__(self) -> None:
        """Destructor ensuring clean release of graphical contexts."""
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "RoverEnv",
    "SCENE_XML",
    "IMG_H",
    "IMG_W",
    "N_CAMERAS",
    "CAM_NAMES",
    "MAX_WHEEL_VEL",
    "MAX_COM_SPEED",
    "WHEEL_RADIUS",
    "MAX_STEER_ANG",
    "STEER_RATE",
    "MAX_STEPS",
    "SUCCESS_RADIUS",
    "MAX_TILT",
    "N_NUMERIC",
    "DRV_JOINTS",
    "PASSIVE_SENSORS",
    "R_SUCCESS",
    "R_FLIP_PENALTY",
    "BaseController",
    "AckermannController",
    "DirectController",
    "BaseEyes",
    "BlindEyes",
    "DepthmapEyes",
    "RGBEyes",
    "BaseReward",
    "StandardReward",
    "EnergyPenaltyReward",
    "BaseTerrain",
    "FlatTerrain",
    "ComponentBundle",
    "make_components",
]
