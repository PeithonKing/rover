"""
train.py
========
PPO training loop for the 6-wheel rover using:
  - Stable-Baselines3 for PPO algorithm
  - Our custom RoverFeaturesExtractor PyTorch model (see models.py)
  - 2 parallel SubprocVecEnv environments (scale up later when bottleneck known)

Usage:
    python train.py
    python train.py --resume   # continues from last checkpoint

Checkpoints saved to: checkpoints/
Reward plot saved to: rewards_plot.png  (updated every EVAL_FREQ steps)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from rover_env import RoverEnv
from models import RoverFeaturesExtractor, BlindRoverFeaturesExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ENVS         = 8                      # Scaled up for server CPU parallelism
TOTAL_STEPS    = 10_000_000             # 10M total env steps to train
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 10_000               # save every N steps
EVAL_FREQ      = 10_000
SEED           = 23
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BLIND_MODE     = True                   # Set to False to enable cameras and full CNN

FEModel = BlindRoverFeaturesExtractor if BLIND_MODE else RoverFeaturesExtractor

# PPO hyperparameters (sensible starting point for complex continuous control)
PPO_KWARGS = dict(
    learning_rate    = 3e-4,
    n_steps          = 2048,            # rollout buffer per env
    batch_size       = 64,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.01,            # exploration entropy bonus
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    verbose          = 1,
)

# Custom policy kwargs: plug our own CNN+MLP extractor in
POLICY_KWARGS = dict(
    features_extractor_class  = FEModel,
    features_extractor_kwargs = {"features_dim": 64},
    net_arch                  = [64, 64],   # MLP head after extractor
    activation_fn             = torch.nn.ReLU,
)


# ---------------------------------------------------------------------------
# Reward tracker callback (for the live plot)
# ---------------------------------------------------------------------------
class RewardPlotCallback(BaseCallback):
    def __init__(self, eval_freq: int, save_path: str = "rewards_plot.png"):
        super().__init__()
        self.eval_freq  = eval_freq
        self.save_path  = save_path
        self.ep_rewards = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for r, d in zip(rewards, dones):
            self._ep_reward += r
            if d:
                self.ep_rewards.append(self._ep_reward)
                self._ep_reward = 0.0

        if self.num_timesteps % self.eval_freq == 0 and self.ep_rewards:
            self._save_plot()
        return True

    def _save_plot(self):
        ep = self.ep_rewards
        window = min(100, len(ep))
        smoothed = np.convolve(ep, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(ep,       alpha=0.3, color="tab:blue", label="Episode reward")
        plt.plot(smoothed, color="tab:blue", label=f"Smoothed ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Rover RL — {self.num_timesteps:,} steps")
        plt.legend()
        plt.grid(True)
        
        # Dynamically scale Y-axis to the min/max of the smoothed curve (ignores extreme noise spikes)
        if len(smoothed) > 0:
            y_min, y_max = np.min(smoothed), np.max(smoothed)
            padding = max(0.1 * (y_max - y_min), 1.0) # Ensure at least some padding
            plt.ylim(y_min - padding, y_max + padding)

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_env(rank: int, seed: int):
    def _init():
        env = RoverEnv(render_mode=None, blind=BLIND_MODE)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize vectorized environment across multiple CPU cores
    vec_env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

    # Checkpoint callback
    ckpt_callback = CheckpointCallback(
        save_freq     = CHECKPOINT_FREQ // N_ENVS,  # per-env steps
        save_path     = CHECKPOINT_DIR,
        name_prefix   = "rover_ppo",
        save_replay_buffer = False,
    )
    plot_callback = RewardPlotCallback(eval_freq=EVAL_FREQ)

    # Look for a checkpoint to resume from
    resume_path = None
    if args.resume:
        ckpts = sorted([
            f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")
        ])
        if ckpts:
            resume_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
            print(f"Resuming from: {resume_path}")

    if resume_path:
        model = PPO.load(
            resume_path,
            env        = vec_env,
            device     = DEVICE,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs = POLICY_KWARGS,
            device        = DEVICE,
            seed          = SEED,
            **PPO_KWARGS,
        )

    print(f"\nTraining on {DEVICE.upper()} with {N_ENVS} parallel envs (Simulation strictly on CPU)...")
    print(f"Total steps: {TOTAL_STEPS:,}  |  Checkpoint every {CHECKPOINT_FREQ:,} steps\n")

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [ckpt_callback, plot_callback],
        reset_num_timesteps = (resume_path is None),
        progress_bar    = True,  # Enables SB3's native tqdm progress bar
    )

    # Final save
    model.save(os.path.join(CHECKPOINT_DIR, "rover_ppo_final"))
    print("\nTraining complete. Final model saved.")
    vec_env.close()


if __name__ == "__main__":
    main()
