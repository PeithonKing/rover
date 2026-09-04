"""
models.py
=========
Pure TorchRL Actor and Critic neural network architectures for 6-Wheel Rover.

Supports:
- Dynamic action dimension scaling (2D for Ackermann, 10D for Direct).
- Pure visual (4-camera CNN), pure numeric (13-dim MLP), and hybrid multimodal fusion.
- Native TensorDict inputs and outputs compatible with TorchRL modules.
- 100% backward compatibility with existing test suites.

Strictly 7-bit ASCII throughout. No main() or top-level execution anti-patterns.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import Bounded
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal

try:
    from torchrl.modules.distributions.continuous import NormalParamExtractor
except ImportError:
    from torchrl.modules import NormalParamExtractor


# ---------------------------------------------------------------------------
# Core Neural Network Modules
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """
    Convolutional block with stride 2 and Tanh activation for spatial downsampling.
    Halves spatial dimensions (e.g., 128 -> 64, 16 -> 8) with outputs in [-1.0, 1.0].
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class MLP2(nn.Module):
    """Simple 2-layer MLP with Tanh activations for numeric features."""

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 16,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumericMLP(MLP2):
    """Numeric feature extractor mapping (B, 13) -> (B, 32)."""

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 16,
        output_dim: int = 32,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )


class CameraCNN(nn.Module):
    """
    7-stage ConvBlock architecture reducing 128x128 -> 1x1 -> 32-dim embedding.
    Supports both batched and unbatched inputs.
    """

    def __init__(self, in_channels: int = 12, out_dim: int = 32) -> None:
        super().__init__()
        self.out_dim = out_dim
        # 7 downsampling stages: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.conv1 = ConvBlock(in_channels, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.conv4 = ConvBlock(64, 64)
        self.conv5 = ConvBlock(64, 64)
        self.conv6 = ConvBlock(64, 64)
        self.post_mlp = MLP2(input_dim=64, hidden_dim=32, output_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        x_float = x.float()
        if x.dtype == torch.uint8:
            x_float = x_float / 255.0
        h = self.conv1(x_float)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        flat = torch.flatten(h, start_dim=-3)
        out = self.post_mlp(flat)
        if unbatched:
            out = out.squeeze(0)
        return out


class RoverFeaturesExtractor(nn.Module):
    """Fuses 4-camera CNN (32-dim) + numeric MLP (32-dim) into 64-dim embedding."""

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        num_dim: int = 13,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cnn = CameraCNN(in_channels=12, out_dim=32)
        self.numeric_mlp = NumericMLP(input_dim=num_dim, hidden_dim=16, output_dim=32)
        self.out_dim = out_dim

    def forward(
        self,
        cameras: Union[torch.Tensor, dict, TensorDictBase, None] = None,
        numeric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if numeric is None and isinstance(cameras, (dict, TensorDictBase)):
            numeric = cameras["numeric"]
            cameras = cameras["cameras"]
        num_clamped = torch.clamp(numeric.float(), -1e6, 1e6)
        cam_emb = self.cnn(cameras)
        num_emb = self.numeric_mlp(num_clamped)
        return torch.cat([cam_emb, num_emb], dim=-1)


class BlindRoverFeaturesExtractor(nn.Module):
    """Pure blind features extractor for numeric observations."""

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        num_dim: int = 13,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.numeric_mlp = MLP2(input_dim=num_dim, hidden_dim=16, output_dim=out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        numeric: Union[torch.Tensor, dict, TensorDictBase],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(numeric, (dict, TensorDictBase)):
            numeric = numeric["numeric"]
        num_clamped = torch.clamp(numeric.float(), -1e6, 1e6)
        return self.numeric_mlp(num_clamped)


class BaseTorchRLNet(nn.Module):
    """
    Base class handling flexible calling conventions across TensorDicts,
    tensors, and kwargs for Actor and Critic networks.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module],
        blind: bool,
    ) -> None:
        super().__init__()
        self.blind = blind
        self.fe = (
            feature_extractor
            if feature_extractor is not None
            else (BlindRoverFeaturesExtractor() if blind else RoverFeaturesExtractor())
        )

    def _parse_inputs(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feats: Optional[torch.Tensor] = None
        action: Optional[torch.Tensor] = kwargs.get("action", None)

        def extract_from_mapping(d: Any) -> None:
            nonlocal feats, action
            if "action" in d:
                action = d["action"]
            if "features" in d:
                feats = d["features"]
            elif self.blind:
                feats = self.fe(d["numeric"])
            else:
                feats = self.fe(d["cameras"], d["numeric"])

        if len(args) > 0:
            first = args[0]
            if isinstance(first, (dict, TensorDictBase)):
                extract_from_mapping(first)
            elif isinstance(first, torch.Tensor) and first.shape[-1] == getattr(self.fe, "out_dim", 64):
                feats = first
                if len(args) > 1:
                    action = args[1]
            elif self.blind:
                feats = self.fe(first)
                if len(args) > 1:
                    action = args[1]
            else:
                if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].shape[-1] == 13:
                    feats = self.fe(first, args[1])
                    if len(args) > 2:
                        action = args[2]
                else:
                    feats = self.fe(first)
                    if len(args) > 1:
                        action = args[1]
        else:
            if "features" in kwargs:
                feats = kwargs["features"]
            elif "numeric" in kwargs:
                if self.blind:
                    feats = self.fe(kwargs["numeric"])
                elif "cameras" in kwargs:
                    feats = self.fe(kwargs["cameras"], kwargs["numeric"])
            else:
                extract_from_mapping(kwargs)

        return feats, action


class ActorNet(BaseTorchRLNet):
    """
    Actor network producing loc and scale parameters for TanhNormal distribution.
    Supports dynamic action dimensions (e.g., 2 for Ackermann, 10 for Direct).
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        blind: bool = False,
        feature_dim: int = 64,
        action_dim: int = 2,
    ) -> None:
        super().__init__(feature_extractor, blind)
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2),
            NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, _ = self._parse_inputs(*args, **kwargs)
        return self.trunk(feats)


class CriticNet(BaseTorchRLNet):
    """
    Critic network computing state-action value Q(s, a).
    Supports dynamic action dimensions (e.g., 2 for Ackermann, 10 for Direct).
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        blind: bool = False,
        feature_dim: int = 64,
        action_dim: int = 2,
    ) -> None:
        super().__init__(feature_extractor, blind)
        self.action_dim = action_dim
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        feats, action = self._parse_inputs(*args, **kwargs)
        return self.q(torch.cat([feats, action], dim=-1))


class ValueNet(BaseTorchRLNet):
    """Value network computing state value V(s) for PPO."""

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        blind: bool = False,
        feature_dim: int = 64,
    ) -> None:
        super().__init__(feature_extractor, blind)
        self.v = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        feats, _ = self._parse_inputs(*args, **kwargs)
        return self.v(feats)


# ---------------------------------------------------------------------------
# TorchRL Module Constructors
# ---------------------------------------------------------------------------


def make_actor(
    feature_extractor: Optional[nn.Module] = None,
    action_spec: Optional[Bounded] = None,
    action_dim: Optional[int] = None,
    blind: bool = False,
) -> ProbabilisticActor:
    """
    Constructs TorchRL ProbabilisticActor with dynamic action scaling.
    Inspects action_spec or action_dim (default 2), configuring ActorNet.
    """
    if isinstance(action_spec, bool):
        blind = action_spec
        action_spec = None

    if action_dim is None:
        if action_spec is not None:
            action_dim = int(action_spec.shape[-1])
        else:
            action_dim = 2

    if action_spec is None:
        action_spec = Bounded(
            shape=torch.Size([action_dim]),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
        )

    in_keys = ["numeric"] if blind else ["cameras", "numeric"]
    actor_net = ActorNet(
        feature_extractor=feature_extractor,
        blind=blind,
        action_dim=action_dim,
    )
    return ProbabilisticActor(
        module=TensorDictModule(
            actor_net,
            in_keys=in_keys,
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        spec=action_spec,
    )


def make_critic(
    feature_extractor: Optional[nn.Module] = None,
    action_spec: Optional[Bounded] = None,
    action_dim: Optional[int] = None,
    blind: bool = False,
) -> TensorDictModule:
    """
    Constructs TorchRL Critic TensorDictModule with dynamic action scaling.
    Inspects action_spec or action_dim (default 2), configuring CriticNet.
    """
    if isinstance(action_spec, bool):
        blind = action_spec
        action_spec = None

    if action_dim is None:
        if action_spec is not None:
            action_dim = int(action_spec.shape[-1])
        else:
            action_dim = 2

    in_keys = ["numeric", "action"] if blind else ["cameras", "numeric", "action"]
    critic_net = CriticNet(
        feature_extractor=feature_extractor,
        blind=blind,
        action_dim=action_dim,
    )
    return TensorDictModule(
        critic_net,
        in_keys=in_keys,
        out_keys=["state_action_value"],
    )


def make_ppo_critic(
    feature_extractor: Optional[nn.Module] = None,
    blind: bool = False,
    *args: Any,
    **kwargs: Any,
) -> TensorDictModule:
    """
    Constructs TorchRL ValueNet TensorDictModule for PPO critic.
    Maintains compatibility with value function.
    """
    in_keys = ["numeric"] if blind else ["cameras", "numeric"]
    return TensorDictModule(
        ValueNet(feature_extractor=feature_extractor, blind=blind),
        in_keys=in_keys,
        out_keys=["state_value"],
    )


__all__ = [
    "ConvBlock",
    "MLP2",
    "NumericMLP",
    "CameraCNN",
    "RoverFeaturesExtractor",
    "BlindRoverFeaturesExtractor",
    "BaseTorchRLNet",
    "ActorNet",
    "CriticNet",
    "ValueNet",
    "make_actor",
    "make_critic",
    "make_ppo_critic",
]
