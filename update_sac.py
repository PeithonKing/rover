import re

with open('train_sac.py', 'r') as f:
    content = f.read()

# 1. Imports
content = content.replace("from stable_baselines3 import PPO", "from stable_baselines3 import SAC")

# 2. Kwargs
ppo_kwargs_regex = r"PPO_KWARGS\s*=\s*dict\([^)]+\)"
sac_kwargs = """SAC_KWARGS = dict(
    buffer_size=25000,       # Lowered from 1M to prevent OOM with 4x cameras
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,            # Collect 1 step per env, then train
    gradient_steps=1,        # 1 gradient step per rollout
    ent_coef="auto",         # SAC automatically tunes entropy
)"""
content = re.sub(ppo_kwargs_regex, sac_kwargs, content)

# 3. Model init and calls
content = content.replace("PPO_KWARGS", "SAC_KWARGS")
content = content.replace("PPO.load", "SAC.load")
content = content.replace("model = PPO(", "model = SAC(")
content = content.replace('"algorithm": "PPO"', '"algorithm": "SAC"')
content = content.replace('zf.writestr("algo.txt", "PPO")', 'zf.writestr("algo.txt", "SAC")')
content = content.replace("rover_ppo_final.zip", "rover_sac_final.zip")

with open('train_sac.py', 'w') as f:
    f.write(content)

