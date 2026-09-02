"""
models.py
=========
Pure PyTorch Neural Network Architectures for 6-Wheel Rover SAC.
Provides feature extractors, policy actor, and critic Q-networks wrapped in TorchRL TensorDict modules.

Components:
- ConvBlock: Single isolated block (Conv2d -> Tanh -> MaxPool2d)
- MLP2: 2-layer Multi-Layer Perceptron (Linear -> Tanh -> Linear -> Tanh)
- CameraCNN: 7-stage ConvBlock CNN reducing (12, 128, 128) -> (1536, 1, 1) -> MLP2(1536, 128, 32)
- NumericMLP: MLP2(13, 16, 32) for kinematic, orientation, and suspension telemetry
- RoverFeaturesExtractor: Multi-modal fusion (CameraCNN + NumericMLP -> 64-dim embedding)
- BlindRoverFeaturesExtractor: Phase 1 numeric-only extractor (13 -> 16 -> 64)
- make_actor: ProbabilisticActor wrapped in TensorDictModule with TanhNormal distribution
- make_critic: TensorDictModule wrapped CriticNet for Double-Q SAC loss computation
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import Bounded
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal


class ConvBlock(nn.Module):
    """
    A single isolated convolutional block executing:
    Conv2d(in, out, kernel_size=3, stride=1, padding=1) -> Tanh -> MaxPool2d(kernel_size=2, stride=2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP2(nn.Module):
    """
    A 2-layer Multi-Layer Perceptron executing:
    Linear(in, hid) -> Tanh -> Linear(hid, out) -> Tanh
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CameraCNN(nn.Module):
    """
    Sub-model for camera frame processing.
    Executes 7 consecutive ConvBlocks reducing 128x128 -> 1x1,
    then passes the 1536-dim flattened output through MLP2(1536, 128, 32)
    to produce a 32-dim visual feature embedding.
    """

    def __init__(self, in_channels: int = 12):
        super().__init__()
        layers = []
        current_channels = in_channels

        for _ in range(7):
            out_channels = current_channels * 2
            layers.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels

        self.net = nn.Sequential(*layers)
        self.mlp = MLP2(input_dim=1536, hidden_dim=128, output_dim=32)
        self.out_dim = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        flat_x = x.flatten(start_dim=-3)
        return self.mlp(flat_x)


class NumericMLP(MLP2):
    """
    Sub-model for processing 13-dim numeric sensor telemetry:
    IMU quaternion (4) + IMU angular velocity (3) + passive suspension (4) + target delta (2).
    Linear(13, 16) -> Tanh -> Linear(16, 32) -> Tanh.
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 16, output_dim: int = 32):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


class RoverFeaturesExtractor(nn.Module):
    """
    Pure PyTorch multi-modal features extractor for Rover.
    Uses CameraCNN for camera frames (12x128x128 -> 32)
    and NumericMLP for numeric sensors (13 -> 16 -> 32),
    concatenating both along the feature dimension (32 + 32 = 64).
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        in_cam_channels: int = 12,
        num_dim: int = 13,
        out_dim: int = 64,
    ):
        super().__init__()
        if observation_space is not None:
            if hasattr(observation_space, "__getitem__"):
                if "cameras" in observation_space:
                    in_cam_channels = observation_space["cameras"].shape[0]
                if "numeric" in observation_space:
                    num_dim = int(np.prod(observation_space["numeric"].shape))

        self.cnn = CameraCNN(in_channels=in_cam_channels)
        self.numeric_mlp = NumericMLP(input_dim=num_dim, hidden_dim=16, output_dim=32)
        self.out_dim = out_dim

    def forward(
        self,
        cameras: Union[torch.Tensor, TensorDictBase, dict, None] = None,
        numeric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if numeric is None:
            if isinstance(cameras, (dict, TensorDictBase)):
                numeric = cameras["numeric"]
                cameras = cameras["cameras"]

        if cameras is None:
            raise ValueError("cameras tensor must be provided to RoverFeaturesExtractor")
        if numeric is None:
            raise ValueError("numeric tensor must be provided to RoverFeaturesExtractor")

        cam_float = cameras.float() / 255.0
        cam_feat = self.cnn(cam_float)
        num_feat = self.numeric_mlp(numeric.float())
        return torch.cat([cam_feat, num_feat], dim=-1)


class BlindRoverFeaturesExtractor(nn.Module):
    """
    Pure PyTorch blind features extractor for Rover (Phase 1 / Blind mode).
    Ignores camera feeds entirely, routing numeric inputs through MLP2 (13 -> 16 -> 64).
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        num_dim: int = 13,
        out_dim: int = 64,
    ):
        super().__init__()
        if observation_space is not None:
            if hasattr(observation_space, "__getitem__") and "numeric" in observation_space:
                num_dim = int(np.prod(observation_space["numeric"].shape))

        self.numeric_mlp = MLP2(input_dim=num_dim, hidden_dim=16, output_dim=out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        numeric: Union[torch.Tensor, TensorDictBase, dict],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(numeric, (dict, TensorDictBase)):
            numeric = numeric["numeric"]
        return self.numeric_mlp(numeric.float())


class ActorNet(nn.Module):
    """
    Policy trunk and distribution parameter head for Rover SAC Actor.
    Takes either raw observation keys or pre-extracted features, passes through
    [64, 64] ReLU layers, and produces loc and scale for TanhNormal distribution.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        blind: bool = False,
        feature_dim: int = 64,
        action_dim: int = 2,
    ):
        super().__init__()
        if feature_extractor is None:
            self.fe = BlindRoverFeaturesExtractor() if blind else RoverFeaturesExtractor()
        else:
            self.fe = feature_extractor
        self.blind = blind
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2),
            NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, (dict, TensorDictBase)):
                if "features" in arg0.keys():
                    feats = arg0["features"]
                elif self.blind:
                    feats = self.fe(arg0["numeric"])
                else:
                    feats = self.fe(arg0["cameras"], arg0["numeric"])
            elif isinstance(arg0, torch.Tensor):
                if arg0.shape[-1] == 64:
                    feats = arg0
                elif self.blind:
                    feats = self.fe(arg0)
                else:
                    feats = arg0
            else:
                feats = self.fe(arg0)
        elif len(args) >= 2:
            cameras, numeric = args[0], args[1]
            feats = self.fe(cameras, numeric)
        else:
            if "features" in kwargs:
                feats = kwargs["features"]
            elif self.blind:
                feats = self.fe(kwargs["numeric"])
            else:
                feats = self.fe(kwargs["cameras"], kwargs["numeric"])

        return self.trunk(feats)


class CriticNet(nn.Module):
    """
    Critic Q-network for Rover SAC.
    Takes either (cameras, numeric, action) or (numeric, action) or (features, action),
    concatenates (features, action), and evaluates through [64, 64] ReLU layers -> scalar Q-value.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        blind: bool = False,
        feature_dim: int = 64,
        action_dim: int = 2,
    ):
        super().__init__()
        if feature_extractor is None:
            self.fe = BlindRoverFeaturesExtractor() if blind else RoverFeaturesExtractor()
        else:
            self.fe = feature_extractor
        self.blind = blind
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) == 2:
            arg0, arg1 = args[0], args[1]
            if isinstance(arg0, torch.Tensor) and arg0.shape[-1] == 64:
                feats = arg0
                action = arg1
            elif self.blind:
                feats = self.fe(arg0)
                action = arg1
            else:
                feats = arg0
                action = arg1
        elif len(args) >= 3:
            cameras, numeric, action = args[0], args[1], args[2]
            feats = self.fe(cameras, numeric)
        elif len(args) == 1 and isinstance(args[0], (dict, TensorDictBase)):
            td = args[0]
            action = td["action"]
            if "features" in td.keys():
                feats = td["features"]
            elif self.blind:
                feats = self.fe(td["numeric"])
            else:
                feats = self.fe(td["cameras"], td["numeric"])
        else:
            action = kwargs.get("action")
            if "features" in kwargs:
                feats = kwargs["features"]
            elif self.blind:
                feats = self.fe(kwargs["numeric"])
            else:
                feats = self.fe(kwargs["cameras"], kwargs["numeric"])

        x = torch.cat([feats, action], dim=-1)
        return self.q(x)


def make_actor(
    feature_extractor: Optional[nn.Module] = None,
    action_spec: Optional[Bounded] = None,
    blind: bool = False,
) -> ProbabilisticActor:
    """
    Constructs a TorchRL ProbabilisticActor for Rover SAC.
    Maps ['cameras', 'numeric'] (or ['numeric'] if blind) to ['action'] in range [-1.0, 1.0]^2.
    """
    if action_spec is None:
        action_spec = Bounded(
            shape=torch.Size([2]),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
        )
    in_keys = ["numeric"] if blind else ["cameras", "numeric"]
    net = ActorNet(feature_extractor=feature_extractor, blind=blind)
    actor_module = TensorDictModule(
        net,
        in_keys=in_keys,
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        spec=action_spec,
    )
    return actor


def make_critic(
    feature_extractor: Optional[nn.Module] = None,
    blind: bool = False,
) -> TensorDictModule:
    """
    Constructs a TorchRL TensorDictModule critic for Rover SAC.
    Maps ['cameras', 'numeric', 'action'] (or ['numeric', 'action'] if blind) to ['state_action_value'].
    Compatible with TorchRL SACLoss (with num_qvalue_nets=2).
    """
    in_keys = ["numeric", "action"] if blind else ["cameras", "numeric", "action"]
    net = CriticNet(feature_extractor=feature_extractor, blind=blind)
    critic_module = TensorDictModule(
        net,
        in_keys=in_keys,
        out_keys=["state_action_value"],
    )
    return critic_module
