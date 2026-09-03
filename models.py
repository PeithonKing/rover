import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules.distributions import TanhNormal
try:
    from torchrl.modules.distributions.continuous import NormalParamExtractor
except ImportError:
    from torchrl.modules import NormalParamExtractor
from torchrl.data import Bounded

# ---------------------------------------------------------------------------
# Core Neural Network Modules (DRY)
# ---------------------------------------------------------------------------

class MLP2(nn.Module):
    """Simple 2-layer MLP for numeric features."""
    def __init__(self, input_dim: int = 13, hidden_dim: int = 16, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RoverFeaturesExtractor(nn.Module):
    """Fuses 4x 64x64 CNN + 13-dim numeric."""
    def __init__(self, observation_space: Optional[Any] = None, num_dim: int = 13, out_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
        )
        self.numeric_mlp = MLP2(input_dim=num_dim, hidden_dim=16, output_dim=64)
        self.out_dim = out_dim

    def forward(self, cameras: Union[torch.Tensor, dict, None] = None, numeric: Optional[torch.Tensor] = None) -> torch.Tensor:
        if numeric is None and isinstance(cameras, dict):
            numeric = cameras["numeric"]
            cameras = cameras["cameras"]
        cam_float = cameras.float() / 255.0
        return torch.cat([self.cnn(cam_float), self.numeric_mlp(numeric.float())], dim=-1)

class BlindRoverFeaturesExtractor(nn.Module):
    """Pure blind features extractor. Ignors cameras."""
    def __init__(self, observation_space: Optional[Any] = None, num_dim: int = 13, out_dim: int = 64):
        super().__init__()
        self.numeric_mlp = MLP2(input_dim=num_dim, hidden_dim=16, output_dim=out_dim)
        self.out_dim = out_dim

    def forward(self, numeric: Union[torch.Tensor, dict], *args, **kwargs) -> torch.Tensor:
        if isinstance(numeric, dict):
            numeric = numeric["numeric"]
        return self.numeric_mlp(numeric.float())

class BaseTorchRLNet(nn.Module):
    """
    DRY Base class that handles TorchRL's weird argument passing.
    Extracts features and optionally the action tensor.
    """
    def __init__(self, feature_extractor: Optional[nn.Module], blind: bool):
        super().__init__()
        self.blind = blind
        self.fe = feature_extractor if feature_extractor else (BlindRoverFeaturesExtractor() if blind else RoverFeaturesExtractor())

    def _parse_inputs(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feats, action = None, kwargs.get("action", None)
        
        # Helper to extract from dict
        def extract_from_dict(d):
            nonlocal feats, action
            if "action" in d: action = d["action"]
            if "features" in d: feats = d["features"]
            elif self.blind: feats = self.fe(d["numeric"])
            else: feats = self.fe(d["cameras"], d["numeric"])
            
        if len(args) > 0:
            if isinstance(args[0], (dict, TensorDictBase)):
                extract_from_dict(args[0])
            elif isinstance(args[0], torch.Tensor) and args[0].shape[-1] == getattr(self.fe, "out_dim", 64):
                feats = args[0]
                if len(args) > 1: action = args[1]
            elif self.blind:
                feats = self.fe(args[0])
                if len(args) > 1: action = args[1]
            else:
                if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                    feats = self.fe(args[0], args[1])
                    if len(args) > 2: action = args[2]
                else:
                    feats = self.fe(args[0])
        else:
            extract_from_dict(kwargs)
            
        return feats, action

class ActorNet(BaseTorchRLNet):
    def __init__(self, feature_extractor=None, blind=False, feature_dim=64, action_dim=2):
        super().__init__(feature_extractor, blind)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim * 2),
            NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
        )
    def forward(self, *args, **kwargs):
        feats, _ = self._parse_inputs(*args, **kwargs)
        return self.trunk(feats)

class CriticNet(BaseTorchRLNet):
    def __init__(self, feature_extractor=None, blind=False, feature_dim=64, action_dim=2):
        super().__init__(feature_extractor, blind)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, *args, **kwargs):
        feats, action = self._parse_inputs(*args, **kwargs)
        return self.q(torch.cat([feats, action], dim=-1))

class ValueNet(BaseTorchRLNet):
    def __init__(self, feature_extractor=None, blind=False, feature_dim=64):
        super().__init__(feature_extractor, blind)
        self.v = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, *args, **kwargs):
        feats, _ = self._parse_inputs(*args, **kwargs)
        return self.v(feats)

# ---------------------------------------------------------------------------
# TorchRL Module Constructors
# ---------------------------------------------------------------------------

def make_actor(feature_extractor=None, action_spec=None, blind=False):
    if action_spec is None:
        action_spec = Bounded(shape=torch.Size([2]), dtype=torch.float32, low=-1.0, high=1.0)
    in_keys = ["numeric"] if blind else ["cameras", "numeric"]
    return ProbabilisticActor(
        module=TensorDictModule(ActorNet(feature_extractor, blind), in_keys=in_keys, out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"], out_keys=["action"],
        distribution_class=TanhNormal, return_log_prob=True, spec=action_spec,
    )

def make_critic(feature_extractor=None, blind=False):
    in_keys = ["numeric", "action"] if blind else ["cameras", "numeric", "action"]
    return TensorDictModule(CriticNet(feature_extractor, blind), in_keys=in_keys, out_keys=["state_action_value"])

def make_ppo_critic(feature_extractor=None, blind=False):
    in_keys = ["numeric"] if blind else ["cameras", "numeric"]
    return TensorDictModule(ValueNet(feature_extractor, blind), in_keys=in_keys, out_keys=["state_value"])
