# Autonomous 6-Wheel Rover RL

A Reinforcement Learning pipeline for training a 6-wheel Rocker-Bogie rover in a MuJoCo physics simulation. Features a custom PyTorch feature extractor (CNN+MLP), True Ackermann kinematics, and a pure TorchRL architecture for maximum asynchronous hardware performance.

## Features
* **True Ackermann Steering**: The policy controls a 2D action space `[COM_Speed, Steer_Angle]`. The environment dynamically queries 3D wheel anchors to mathematically perfectly drive 6 wheels and angle 4 servos without lateral slip.
* **Pure TorchRL Architecture**: Completely native PyTorch integration using `EnvBase` and `TensorDict` objects. Zero-copy memory passing for massive speedups on multi-camera setups.
* **Async Producer-Consumer**: Utilizes `MultiaSyncDataCollector` and `MemmapStorage` to allow the CPU to simulate environments in parallel while the GPU continuously trains.
* **Weights & Biases Tracking**: Native cloud syncing of tensorboard metrics, model gradients, and custom Matplotlib reward graphs.

## Setup & Installation
```bash
# 1. Install dependencies
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Compile the MuJoCo physics scene from raw assets
python scene_builder.py

# 3. Authenticate with Weights & Biases (if you want cloud tracking)
wandb login
```

## Running the Training Pipeline
Train the Soft Actor-Critic (SAC) algorithm:
```bash
python train_sac.py
```

**Flags:**
* `--nolog` : Disable Weights & Biases tracking (runs strictly local).
* `--resume`: Resume training instantly from `latest_model.zip`.

### Headless Server Warning
If `BLIND_MODE = False` (meaning the AI uses camera feeds) and you are on a headless server without a monitor, you must tell MuJoCo to use software rendering or EGL:
```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl python train_sac.py
```

## Evaluation & Testing
Evaluate the trained AI (requires a local GUI display).
```bash
python evaluate.py
```

To test the physical suspension over the terrain without an AI driving:
```bash
python evaluate_passive.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
