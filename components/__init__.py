"""
components
==========
Pluggable component framework for 6-wheel rocker-bogie rover RL environment.

Exports:
- Controllers: BaseController, AckermannController, DirectController
- Sensors (Eyes): BaseEyes, BlindEyes, DepthmapEyes, RGBEyes
- Rewards: BaseReward, StandardReward, EnergyPenaltyReward
- Terrains: BaseTerrain, FlatTerrain
- Factory: make_components

All comments and strings strictly use 7-bit ASCII characters.
"""

from typing import Any, Dict, NamedTuple, Optional, Union
import mujoco

from .controllers import (
    BaseController,
    AckermannController,
    DirectController,
    MAX_WHEEL_VEL,
    MAX_COM_SPEED,
    WHEEL_RADIUS,
    MAX_STEER_ANG,
    STEER_RATE,
    DRV_JOINTS,
)
from .eyes import (
    BaseEyes,
    BlindEyes,
    DepthmapEyes,
    RGBEyes,
    IMG_H,
    IMG_W,
    N_CAMERAS,
    CAM_NAMES,
)
from .rewards import (
    BaseReward,
    StandardReward,
    EnergyPenaltyReward,
    SUCCESS_RADIUS,
    MAX_TILT,
    R_SUCCESS,
    R_FLIP_PENALTY,
)
from .terrains import (
    BaseTerrain,
    FlatTerrain,
)


class ComponentBundle(NamedTuple):
    """
    Container holding all 4 instantiated components for RoverEnv.
    Supports tuple unpacking and dictionary conversion.
    """

    controller: BaseController
    eyes: BaseEyes
    reward: BaseReward
    terrain: BaseTerrain

    def as_dict(self) -> Dict[str, Any]:
        """Convert component bundle to dictionary suitable for RoverEnv kwargs."""
        return {
            "controller": self.controller,
            "eyes": self.eyes,
            "reward": self.reward,
            "terrain": self.terrain,
        }


def make_components(
    control_mode: Union[str, BaseController] = "ackermann",
    vision_mode: Union[str, BaseEyes] = "blind",
    terrain_mode: Optional[Union[str, BaseTerrain]] = None,
    reward_mode: Union[str, BaseReward] = "standard",
    terrain: Optional[Union[str, BaseTerrain]] = None,
    model: Optional[Any] = None,
    **kwargs: Any,
) -> ComponentBundle:
    """
    Unified factory constructing the 4 pluggable components for RoverEnv.

    Parameters
    ----------
    control_mode : Union[str, BaseController]
        Control scheme: 'ackermann' (2-action) or 'direct' (10-action),
        or a pre-instantiated BaseController instance.
    vision_mode : Union[str, BaseEyes]
        Visual perception mode: 'blind' (no rendering overhead),
        'depth' / 'depthmap' (4-camera depth maps), or 'rgb' (4-camera RGB),
        or a pre-instantiated BaseEyes instance.
    terrain_mode : Optional[Union[str, BaseTerrain]]
        Terrain environment: 'flat' (loads world_flat.xml with fallback),
        or a pre-instantiated BaseTerrain instance.
    reward_mode : Union[str, BaseReward]
        Reward objective: 'standard' (dense distance progress + terminal bonus/penalty)
        or 'energy' / 'energy_penalty' (standard + quadratic actuator penalty),
        or a pre-instantiated BaseReward instance.
    terrain : Optional[Union[str, BaseTerrain]]
        Alias for terrain_mode to match CLI --terrain argument naming.
    model : Optional[Any]
        Optional compiled mujoco.MjModel instance. If required by visual sensors
        (depth/rgb) and not provided, it will be loaded automatically from the
        selected terrain XML path.
    **kwargs : Any
        Additional parameters forwarded to component constructors
        (e.g., energy_weight, distance_weight, empty_array, height, width, xml_path).

    Returns
    -------
    ComponentBundle
        Named tuple (controller, eyes, reward, terrain) supporting both positional
        unpacking and dictionary conversion via .as_dict() or ._asdict().
    """
    # 1. Resolve Terrain
    actual_terrain = (
        terrain
        if terrain is not None
        else (terrain_mode if terrain_mode is not None else "flat")
    )
    if isinstance(actual_terrain, BaseTerrain):
        terrain_obj = actual_terrain
    elif isinstance(actual_terrain, str):
        t_mode = actual_terrain.lower().strip()
        if t_mode == "flat":
            xml_path = kwargs.get("xml_path", None)
            terrain_obj = FlatTerrain(xml_path=xml_path)
        else:
            raise ValueError(
                f"Unknown terrain_mode: '{actual_terrain}'. Supported: 'flat'"
            )
    else:
        raise TypeError(
            f"terrain_mode must be str or BaseTerrain, got {type(actual_terrain)}"
        )

    # 2. Resolve Controller
    if isinstance(control_mode, BaseController):
        controller_obj = control_mode
    elif isinstance(control_mode, str):
        c_mode = control_mode.lower().strip()
        if c_mode == "ackermann":
            controller_obj = AckermannController()
        elif c_mode in ("direct", "10d"):
            controller_obj = DirectController()
        else:
            raise ValueError(
                f"Unknown control_mode: '{control_mode}'. Supported: 'ackermann', 'direct'"
            )
    else:
        raise TypeError(
            f"control_mode must be str or BaseController, got {type(control_mode)}"
        )

    # 3. Resolve Reward
    if isinstance(reward_mode, BaseReward):
        reward_obj = reward_mode
    elif isinstance(reward_mode, str):
        r_mode = reward_mode.lower().strip()
        if r_mode == "standard":
            reward_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "distance_weight",
                    "success_radius",
                    "max_tilt",
                    "success_reward",
                    "flip_penalty",
                    "sky_penalty",
                )
            }
            reward_obj = StandardReward(**reward_kwargs)
        elif r_mode in ("energy", "energy_penalty"):
            energy_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "energy_weight",
                    "distance_weight",
                    "success_radius",
                    "max_tilt",
                    "success_reward",
                    "flip_penalty",
                    "sky_penalty",
                )
            }
            reward_obj = EnergyPenaltyReward(**energy_kwargs)
        else:
            raise ValueError(
                f"Unknown reward_mode: '{reward_mode}'. Supported: 'standard', 'energy'"
            )
    else:
        raise TypeError(
            f"reward_mode must be str or BaseReward, got {type(reward_mode)}"
        )

    # 4. Resolve Eyes (Vision)
    if isinstance(vision_mode, BaseEyes):
        eyes_obj = vision_mode
    elif isinstance(vision_mode, str):
        v_mode = vision_mode.lower().strip()
        if v_mode == "blind":
            empty_array = kwargs.get("empty_array", True)
            eyes_obj = BlindEyes(empty_array=empty_array)
        elif v_mode in ("depth", "depthmap"):
            if model is None:
                model = mujoco.MjModel.from_xml_path(terrain_obj.xml_path)
            height = kwargs.get("height", IMG_H)
            width = kwargs.get("width", IMG_W)
            eyes_obj = DepthmapEyes(model=model, height=height, width=width)
        elif v_mode == "rgb":
            if model is None:
                model = mujoco.MjModel.from_xml_path(terrain_obj.xml_path)
            height = kwargs.get("height", IMG_H)
            width = kwargs.get("width", IMG_W)
            eyes_obj = RGBEyes(model=model, height=height, width=width)
        else:
            raise ValueError(
                f"Unknown vision_mode: '{vision_mode}'. Supported: 'blind', 'depth', 'rgb'"
            )
    else:
        raise TypeError(
            f"vision_mode must be str or BaseEyes, got {type(vision_mode)}"
        )

    return ComponentBundle(
        controller=controller_obj,
        eyes=eyes_obj,
        reward=reward_obj,
        terrain=terrain_obj,
    )


__all__ = [
    "BaseController",
    "AckermannController",
    "DirectController",
    "MAX_WHEEL_VEL",
    "MAX_COM_SPEED",
    "WHEEL_RADIUS",
    "MAX_STEER_ANG",
    "STEER_RATE",
    "DRV_JOINTS",
    "BaseEyes",
    "BlindEyes",
    "DepthmapEyes",
    "RGBEyes",
    "IMG_H",
    "IMG_W",
    "N_CAMERAS",
    "CAM_NAMES",
    "BaseReward",
    "StandardReward",
    "EnergyPenaltyReward",
    "SUCCESS_RADIUS",
    "MAX_TILT",
    "R_SUCCESS",
    "R_FLIP_PENALTY",
    "BaseTerrain",
    "FlatTerrain",
    "ComponentBundle",
    "make_components",
]
