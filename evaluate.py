"""
evaluate.py
===========
Load the best saved checkpoint and watch the rover drive live in MuJoCo.

Usage:
    python evaluate.py                        # loads latest checkpoint
    python evaluate.py --model checkpoints/rover_ppo_final.zip
    python evaluate.py --episodes 5
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from rover_env import RoverEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=None,  help="Path to .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=10,    help="Number of evaluation episodes")
    args = parser.parse_args()

    # --- Find model ---
    model_path = args.model
    if model_path is None:
        ckpts = sorted([
            f for f in os.listdir("checkpoints") if f.endswith(".zip")
        ])
        assert ckpts, "No checkpoints found in ./checkpoints/. Train first!"
        model_path = os.path.join("checkpoints", ckpts[-1])

    print(f"Loading: {model_path}")
    model = PPO.load(model_path, device="cpu")

    # --- Evaluate ---
    env = RoverEnv(render_mode="human")
    total_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        status = "SUCCESS" if info.get("success") else ("FLIPPED" if info.get("flipped") else "TIMEOUT")
        print(f"  Episode {ep+1:>2}: reward={total_reward:>8.1f}  dist={info['dist']:.2f}m  [{status}]")

    env.close()
    avg = np.mean(total_rewards)
    print(f"\nAverage reward over {args.episodes} episodes: {avg:.2f}")


if __name__ == "__main__":
    main()
