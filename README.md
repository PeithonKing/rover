# Autonomous 6-Wheel Rover RL

A Reinforcement Learning pipeline for training a 6-wheel Rocker-Bogie rover in a MuJoCo physics simulation. The AI is trained using Proximal Policy Optimization (PPO) via Stable-Baselines3, utilizing a completely custom PyTorch feature extractor backbone.

## Setup & Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile the MuJoCo physics scene from raw assets
python scene_builder.py

# 3. Train the model (Check train.py to toggle BLIND_MODE)
python train.py

# 4. Watch the trained AI drive visually
python evaluate.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
