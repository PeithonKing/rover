import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class ConvBlock(nn.Module):
    """
    A single isolated block executing: Conv2d -> ReLU -> MaxPool2d
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)


class MLP2(nn.Module):
    """
    A simple 2-layer Multi-Layer Perceptron.
    Currently defined but unused.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  # Added to prevent linear collapse before SB3
        )

    def forward(self, x):
        return self.net(x)


class CameraCNN(nn.Module):
    """
    Sub-model for the camera input.
    Executes 7 consecutive ConvBlocks to reduce 128x128 -> 1x1.
    Passes the flattened output through an MLP2 to get a 32-dim embedding.
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
        
        # Pass 1536 flattened output through MLP2 down to 32
        self.mlp = MLP2(input_dim=1536, hidden_dim=128, output_dim=32)
        self.out_dim = 32

    def forward(self, x):
        x = self.net(x) 
        flat_x = x.flatten(start_dim=1)
        return self.mlp(flat_x)


class RoverFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor.
    Uses the CameraCNN for images, MLP2 for numeric data,
    and concatenates both (32+32=64) to pass directly to SB3.
    """
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self.cnn = CameraCNN(in_channels=observation_space["cameras"].shape[0])

        # Numeric processing (13 -> 16 -> 32)
        num_shape = observation_space["numeric"].shape
        num_flat_dim = int(np.prod(num_shape))
        self.numeric_mlp = MLP2(input_dim=num_flat_dim, hidden_dim=16, output_dim=32)

    def forward(self, observations):
        cam = observations["cameras"].float() / 255.0
        cam_feat = self.cnn(cam)  
        
        num_flat = observations["numeric"].float()
        num_feat = self.numeric_mlp(num_flat)

        # 32 (cam) + 32 (num) = 64. Return directly to SB3!
        return torch.cat([cam_feat, num_feat], dim=1)


class BlindRoverFeaturesExtractor(BaseFeaturesExtractor):
    """
    Phase 1 Extractor. Completely ignores cameras to speed up initial physical training.
    Routes the numeric features through a single MLP2 directly to the features_dim.
    """
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        num_shape = observation_space["numeric"].shape
        num_flat_dim = int(np.prod(num_shape))
        
        # 13 -> 16 -> 64
        self.numeric_mlp = MLP2(input_dim=num_flat_dim, hidden_dim=16, output_dim=features_dim)

    def forward(self, observations):
        # Ignore observations["cameras"] entirely
        num_flat = observations["numeric"].float()
        return self.numeric_mlp(num_flat)
