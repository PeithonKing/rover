# Autonomous 6-Wheel Rover RL

A Reinforcement Learning pipeline for training a 6-wheel Rocker-Bogie rover in a MuJoCo physics simulation. Features a custom PyTorch feature extractor (CNN+MLP), True Ackermann kinematics, and support for both on-policy (PPO) and off-policy (SAC) learning.

## Features
* **True Ackermann Steering**: The policy controls a 2D action space `[COM_Speed, Steer_Angle]`. The environment dynamically queries 3D wheel anchors to mathematically perfectly drive 6 wheels and angle 4 servos without lateral slip.
* **Dual Algorithms**: Train using either standard PPO or highly sample-efficient Soft Actor-Critic (SAC).
* **Weights & Biases Tracking**: Native cloud syncing of tensorboard metrics, model gradients, and custom Matplotlib reward graphs.
* **Smart Checkpoints**: Automatically tracks and evaluates the `best_model.zip` and `latest_model.zip`.

## Setup & Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile the MuJoCo physics scene from raw assets
python scene_builder.py

# 3. Authenticate with Weights & Biases (if you want cloud tracking)
wandb login
```

## Running the Training Pipeline
We provide two distinct training algorithms. SAC is recommended for sample efficiency on this robotics task.

**Train with PPO:**
```bash
python train_ppo.py
```

**Train with SAC:**
```bash
python train_sac.py
```

**Flags for both scripts:**
* `--nolog` : Disable Weights & Biases tracking (runs strictly local).
* `--resume`: Resume training instantly from `latest_model.zip`.

### Headless Server Warning
If `BLIND_MODE = False` (meaning the AI uses camera feeds) and you are on a headless server without a monitor, you must tell MuJoCo to use software rendering or EGL:
```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl python train_sac.py
```

## Evaluation & Testing
Evaluate the trained AI (requires a local GUI display). The script intelligently looks inside the `.zip` archive to detect the algorithm used and automatically prioritizes `best_model.zip`:
```bash
python evaluate.py
```

To test the physical suspension over the terrain without an AI driving:
```bash
python evaluate_passive.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
