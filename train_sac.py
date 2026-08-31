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

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rover_env import RoverEnv
from models import RoverFeaturesExtractor, BlindRoverFeaturesExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ENVS = 8  # Scaled up for server CPU parallelism
TOTAL_STEPS = 10_000_000  # 10M total env steps to train
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 10_000  # save every N steps
EVAL_FREQ = 10_000
SEED = 23
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIND_MODE = True  # Set to False to enable cameras and full CNN

FEModel = BlindRoverFeaturesExtractor if BLIND_MODE else RoverFeaturesExtractor

# PPO hyperparameters (sensible starting point for complex continuous control)
SAC_KWARGS = dict(
    buffer_size=25000,  # Lowered from 1M to prevent OOM with 4x cameras
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,  # Collect 1 step per env, then train
    gradient_steps=1,  # 1 gradient step per rollout
    ent_coef="auto",  # SAC automatically tunes entropy
)

# Custom policy kwargs: plug our own CNN+MLP extractor in
POLICY_KWARGS = dict(
    features_extractor_class=FEModel,
    features_extractor_kwargs={"features_dim": 64},
    net_arch=[64, 64],  # MLP head after extractor
    activation_fn=torch.nn.ReLU,
)


# ---------------------------------------------------------------------------
# Reward tracker callback (for the live plot)
# ---------------------------------------------------------------------------
class RewardPlotCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        save_path: str = "rewards_plot.png",
        ckpt_dir: str = "checkpoints",
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.ckpt_dir = ckpt_dir
        self.ep_rewards = []
        self._ep_reward = 0.0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for r, d in zip(rewards, dones):
            self._ep_reward += r
            if d:
                self.ep_rewards.append(self._ep_reward)
                self._ep_reward = 0.0

        if self.num_timesteps % self.eval_freq == 0 and self.ep_rewards:
            self._save_plot()
            self._save_models()

        return True

    def _save_models(self):
        import zipfile

        # 1. Always save the latest model
        latest_path = os.path.join(self.ckpt_dir, "latest_model.zip")
        self.model.save(latest_path)
        with zipfile.ZipFile(latest_path, "a") as zf:
            zf.writestr("algo.txt", "SAC")

        # 2. Check if this is the new best model
        window = min(100, len(self.ep_rewards))
        mean_reward = np.mean(self.ep_rewards[-window:])
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_path = os.path.join(self.ckpt_dir, "best_model.zip")
            self.model.save(best_path)
            with zipfile.ZipFile(best_path, "a") as zf:
                zf.writestr("algo.txt", "SAC")

    def _save_plot(self):
        ep = self.ep_rewards
        window = min(100, len(ep))
        smoothed = np.convolve(ep, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(ep, alpha=0.3, color="tab:blue", label="Episode reward")
        plt.plot(smoothed, color="tab:blue", label=f"Smoothed ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Rover RL — {self.num_timesteps:,} steps")
        plt.legend()
        plt.grid(True)

        if len(smoothed) > 0:
            y_min, y_max = np.min(smoothed), np.max(smoothed)
            padding = max(0.1 * (y_max - y_min), 1.0)
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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume", action="store_true", help="Resume from latest checkpoint"
)
parser.add_argument(
    "--nolog", action="store_true", help="Disable Weights & Biases logging"
)
args = parser.parse_args()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize wandb by default unless disabled
callbacks = []
if not args.nolog:
    run = wandb.init(
        project="rocker-bogie-rover",
        config={
            "algorithm": "SAC",
            "n_envs": N_ENVS,
            "total_steps": TOTAL_STEPS,
            **SAC_KWARGS,
        },
        sync_tensorboard=True,  # Automatically upload SB3's tensorboard metrics
        save_code=True,
    )
    callbacks.append(WandbCallback(gradient_save_freq=100, verbose=2))

# Initialize vectorized environment across multiple CPU cores
vec_env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

plot_callback = RewardPlotCallback(eval_freq=EVAL_FREQ, ckpt_dir=CHECKPOINT_DIR)
callbacks.append(plot_callback)

# Look for a checkpoint to resume from
resume_path = None
if args.resume:
    latest = os.path.join(CHECKPOINT_DIR, "latest_model.zip")
    if os.path.exists(latest):
        resume_path = latest
        print(f"Resuming from: {resume_path}")
    else:
        print("No latest_model.zip found to resume from. Starting fresh.")

if resume_path:
    model = SAC.load(
        resume_path,
        env=vec_env,
        device=DEVICE,
        tensorboard_log="runs" if not args.nolog else None,
    )
else:
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=POLICY_KWARGS,
        device=DEVICE,
        seed=SEED,
        tensorboard_log="runs" if not args.nolog else None,
        **SAC_KWARGS,
    )

print(
    f"\nTraining on {DEVICE.upper()} with {N_ENVS} parallel envs (Simulation strictly on CPU)..."
)
print(f"Total steps: {TOTAL_STEPS:,}\n")

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=callbacks,
    reset_num_timesteps=(resume_path is None),
    progress_bar=True,  # Enables SB3's native tqdm progress bar
)

# Final save
final_path = os.path.join(CHECKPOINT_DIR, "rover_sac_final.zip")
model.save(final_path)

# Inject the algorithm metadata into the same zip file
import zipfile

with zipfile.ZipFile(final_path, "a") as zf:
    zf.writestr("algo.txt", "SAC")

print("\nTraining complete. Final model saved.")
vec_env.close()
