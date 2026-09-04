"""
PPO Training script for the 6-Wheel Rover environment using pure TorchRL.
Analogous to train_sac.py, heavily optimized for parallel data collection.
"""
import os
import time
import argparse
import numpy as np
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvCreator, ParallelEnv, TransformedEnv, StepCounter
try:
    from torchrl.collectors import SyncDataCollector
except ImportError:
    from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import set_exploration_type, ExplorationType

import wandb
import matplotlib.pyplot as plt

from rover_env import RoverEnv
from components.observations import TargetAwareObservation, TargetBlindObservation
from models import RoverFeaturesExtractor, BlindRoverFeaturesExtractor, make_actor, make_ppo_critic

os.environ.setdefault("MUJOCO_GL", "egl")


def make_env_creator(
    control_mode: str = "ackermann",
    vision_mode: str = "blind",
    terrain: str = "flat",
    reward_mode: str = "standard",
    max_steps: int = 2000,
    device="cpu",
):
    def _create_env():
        render_mode = "rgb_array" if vision_mode != "blind" else None
        numeric_obs = TargetBlindObservation() if vision_mode != "blind" else TargetAwareObservation()
        env = RoverEnv(
            control_mode=control_mode,
            vision_mode=vision_mode,
            terrain_mode=terrain,
            device=device,
            numeric_obs=numeric_obs,
            reward_mode=reward_mode,
            render_mode=render_mode,
        )
        env = TransformedEnv(env, StepCounter(max_steps=max_steps))
        return env
    return _create_env


def save_plot(ep_rewards, step, save_path="rewards_plot_ppo.png"):
    if not ep_rewards:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(ep_rewards, alpha=0.3, label="Episode reward", color="C0")
    if len(ep_rewards) >= 100:
        smoothed = np.convolve(ep_rewards, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(ep_rewards)), smoothed, label="Smoothed (100)", color="C0")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Rover RL PPO (TorchRL) - {step:,} steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(path, actor, critic, optim, step, best_mean_reward, blind):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    state = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "step": step,
        "best_mean_reward": best_mean_reward,
        "blind": blind,
    }
    torch.save(state, path)


parser = argparse.ArgumentParser(description="Train 6-Wheel Rover PPO with pure TorchRL")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--total-timesteps", type=int, default=10_000_000)
parser.add_argument("--frames-per-batch", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--mini-batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--clip-epsilon", type=float, default=0.2)
parser.add_argument("--entropy-coef", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lmbda", type=float, default=0.95)
parser.add_argument(
    "--control-mode",
    type=str,
    choices=["ackermann", "direct"],
    default="ackermann",
    help="Control scheme ('ackermann' or 'direct')",
)
parser.add_argument(
    "--vision-mode",
    type=str,
    choices=["blind", "depth", "depthmap", "rgb"],
    default=None,
    help="Visual perception mode ('blind', 'depth', 'depthmap', 'rgb')",
)
parser.add_argument(
    "--terrain",
    type=str,
    choices=["flat"],
    default="flat",
    help="Terrain world environment ('flat')",
)
parser.add_argument(
    "--reward-mode",
    type=str,
    choices=["standard", "energy"],
    default="standard",
    help="Reward objective formulation ('standard', 'energy')",
)
parser.add_argument("--blind", action="store_true", help="Blind mode shortcut")
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
args = parser.parse_args()

# Resolve vision_mode and blind flag
if args.vision_mode is None:
    vision_mode = "blind" if args.blind else "rgb"
else:
    vision_mode = args.vision_mode
blind = (vision_mode == "blind")

args.checkpoint_dir = os.path.join(args.checkpoint_dir, "ppo")
os.makedirs(args.checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not args.no_wandb:
    wandb.init(
        project="rover_rl",
        name=f"ppo_{args.control_mode}_{vision_mode}",
        config=vars(args),
    )

# Env
env_creator = make_env_creator(
    control_mode=args.control_mode,
    vision_mode=vision_mode,
    terrain=args.terrain,
    reward_mode=args.reward_mode,
    device=device,
)
dummy_env = env_creator()
action_spec = dummy_env.action_spec
dummy_env.close()

env = ParallelEnv(args.workers, env_creator, mp_start_method="fork")

# Assemble feature extractor explicitly
num_dim = 11 if not blind else 13
feature_extractor = RoverFeaturesExtractor(num_dim=num_dim) if not blind else BlindRoverFeaturesExtractor(num_dim=num_dim)

# Networks
actor = make_actor(feature_extractor=feature_extractor, blind=blind, action_spec=action_spec).to(device)
critic = make_ppo_critic(feature_extractor=feature_extractor, blind=blind).to(device)

# Optimizer
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

# Loss & Advantage
adv_module = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=critic, average_gae=True)
loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=args.clip_epsilon,
    entropy_bonus=bool(args.entropy_coef > 0),
    entropy_coeff=args.entropy_coef,
    normalize_advantage=True,
).to(device)
try:
    loss_module.set_keys(value_target=adv_module.tensor_keys.value_target)
except Exception:
    pass

# Collector
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=args.frames_per_batch,
    total_frames=args.total_timesteps,
    device=device,
    storing_device=device,
)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=args.frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=args.mini_batch_size,
)

ep_rewards = []
current_ep_reward = 0.0
best_mean_reward = -float("inf")
total_collected = 0

print(f"Starting PPO training on {device} (Mode: {vision_mode}, Control: {args.control_mode})")

for i, tensordict_data in enumerate(collector):
    with torch.no_grad():
        adv_module(tensordict_data)
        
    replay_buffer.extend(tensordict_data.reshape(-1))
    
    # PPO Inner Epochs
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    
    for _ in range(args.epochs):
        for _ in range(args.frames_per_batch // args.mini_batch_size):
            subdata = replay_buffer.sample()
            loss_td = loss_module(subdata)
            
            loss_objective = loss_td["loss_objective"]
            loss_critic = loss_td["loss_critic"]
            loss_entropy = loss_td.get("loss_entropy", torch.tensor(0.0, device=device))
            
            loss = loss_objective + loss_critic + loss_entropy
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            optim.step()
            
            actor_losses.append(loss_objective.item())
            critic_losses.append(loss_critic.item())
            entropy_losses.append(loss_entropy.item())
    
    total_collected += tensordict_data.numel()
    
    # Track telemetry
    rewards = tensordict_data.get(("next", "reward"))
    dones = tensordict_data.get(("next", "done"))
    
    for r, d in zip(rewards.reshape(-1), dones.reshape(-1)):
        current_ep_reward += float(r.item())
        if d.item():
            ep_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            
            if len(ep_rewards) % 10 == 0:
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, f"rover_ep_{len(ep_rewards)}.pt"),
                    actor, critic, optim, total_collected, best_mean_reward, blind
                )
    
    # Logging
    mean_100 = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0
    print(f"Step: {total_collected:,} | Mean Rew (100): {mean_100:.2f} | Loss A: {np.mean(actor_losses):.3f} | Loss C: {np.mean(critic_losses):.3f}")
    
    if not args.no_wandb:
        wandb.log({
            "env/step": total_collected,
            "env/mean_ep_reward_100": mean_100,
            "train/loss_actor": np.mean(actor_losses),
            "train/loss_critic": np.mean(critic_losses),
            "train/loss_entropy": np.mean(entropy_losses),
        })
        
    if total_collected % (args.frames_per_batch * 5) < args.frames_per_batch:
        save_plot(ep_rewards, total_collected, save_path="rewards_plot_ppo.png")
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "latest_model.pt"),
            actor, critic, optim, total_collected, best_mean_reward, blind
        )

collector.shutdown()
save_plot(ep_rewards, total_collected, save_path="rewards_plot_ppo.png")
if not args.no_wandb:
    wandb.finish()
