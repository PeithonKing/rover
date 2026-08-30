# Autonomous 6-Wheel Rover RL

A Reinforcement Learning pipeline for training a 6-wheel Rocker-Bogie rover in a MuJoCo physics simulation. The AI is trained using Proximal Policy Optimization (PPO) via Stable-Baselines3, utilizing a completely custom PyTorch feature extractor backbone.

## Setup & Installation
```bash
# 1. Install dependencies (includes SB3 [extra] for progress bars)
pip install -r requirements.txt

# 2. Compile the MuJoCo physics scene from raw assets
python scene_builder.py
```

## Running the Training Pipeline
How you launch the training script depends on your hardware and environment:

**Option A: Local Laptop (or Blind Mode)**
If you are training on a laptop, or if you have `BLIND_MODE = True` set in `train.py`, run normally:
```bash
python train.py
```

**Option B: Remote Headless Server (Vision Mode)**
If `BLIND_MODE = False` and you are on a headless server (no physical monitor), you must tell MuJoCo to use hardware-accelerated headless rendering (EGL) so it doesn't crash looking for an X11 display:
```bash
MUJOCO_GL=egl python train.py
```
*Note: If EGL throws a `Permission denied` error for `/dev/dri/cardX`, your Linux user does not have GPU access. You must ask a sysadmin to run `sudo usermod -aG render,video $USER`. If you lack sudo, your only workaround is to train using `BLIND_MODE = True`.*

## Evaluation
Watch the trained AI drive visually (requires a desktop environment / display):
```bash
python evaluate.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
