"""
train_sac.py
============
Pure TorchRL Soft Actor-Critic (SAC) training pipeline for 6-Wheel Rover.

Features:
- Native TensorDict environment (RoverEnv)
- Asynchronous multi-worker data collection via MultiAsyncCollector / MultiSyncCollector / Collector
- LazyMemmapStorage disk-backed replay buffer for high-capacity memory efficiency
- SACLoss with double Q-networks and auto-tuned entropy temperature
- Polyak soft target updates via SoftUpdate
- Weights & Biases telemetry logging and live reward smoothing plot
- Periodic PyTorch state dict checkpointing (.pt)

Usage:
    python train_sac.py --total-timesteps 10000000 --frames-per-batch 64 --batch-size 256
    python train_sac.py --blind --no-wandb
    python train_sac.py --resume checkpoints/latest_model.pt
"""

import os
import sys
import time
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Set headless MuJoCo rendering backend before importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure fork start method is used on Linux for multiprocessing collector workers
import torch.multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector, MultiAsyncCollector, MultiSyncCollector
from torchrl.data import (
    Bounded,
    LazyMemmapStorage,
    LazyTensorStorage,
    RandomSampler,
    TensorDictReplayBuffer,
)
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.modules import ProbabilisticActor

import wandb

from rover_env import RoverEnv
from models import make_actor, make_critic
from sac_utils import (
    make_env_fn,
    build_sac_components,
    build_replay_buffer,
    build_collector,
    extract_telemetry,
    save_checkpoint,
    load_checkpoint,
    save_plot,
)

# ---------------------------------------------------------------------------
# CLI Argument Parser & Top-Level Execution Flow
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train 6-Wheel Rover SAC with pure TorchRL and MultiAsyncCollector"
)
parser.add_argument(
    "--total-timesteps",
    type=int,
    default=10_000_000,
    help="Total timesteps to collect and train",
)
parser.add_argument(
    "--frames-per-batch",
    type=int,
    default=64,
    help="Frames collected per rollout batch",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    help="Mini-batch size for SAC updates",
)
parser.add_argument(
    "--buffer-size",
    type=int,
    default=100_000,
    help="Replay buffer capacity (memmap frames)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=3e-4,
    help="Base learning rate for all optimizers",
)
parser.add_argument(
    "--lr-actor",
    type=float,
    default=None,
    help="Actor learning rate (defaults to --lr)",
)
parser.add_argument(
    "--lr-critic",
    type=float,
    default=None,
    help="Critic learning rate (defaults to --lr)",
)
parser.add_argument(
    "--lr-alpha",
    type=float,
    default=None,
    help="Alpha learning rate (defaults to --lr)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    help="Target network soft update rate (Polyak)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Discount factor",
)
parser.add_argument(
    "--learning-starts",
    type=int,
    default=1000,
    help="Number of transitions in buffer before gradient updates begin",
)
parser.add_argument(
    "--gradient-steps",
    type=int,
    default=4,
    help="Number of gradient steps per collector batch",
)
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of parallel collector worker processes",
)
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
parser.add_argument(
    "--blind",
    action="store_true",
    help="Blind mode shortcut (numeric observations only, bypasses camera rendering)",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=True,
    help="Enable Weights & Biases logging (default: True)",
)
parser.add_argument(
    "--no-wandb",
    action="store_false",
    dest="wandb",
    help="Disable Weights & Biases logging",
)
parser.add_argument(
    "--scratch-dir",
    type=str,
    default="./storage_scratch",
    help="Directory for LazyMemmapStorage disk backing",
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints",
    help="Directory for saving policy checkpoints",
)
parser.add_argument(
    "--eval-interval",
    type=int,
    default=10_000,
    help="Step interval for saving plots and evaluation checkpoints",
)
parser.add_argument(
    "--seed",
    type=int,
    default=23,
    help="Random seed for environment and networks",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device for training neural networks (cuda/cpu)",
)
parser.add_argument(
    "--sync",
    action="store_true",
    help="Force synchronous multi-worker collection (MultiSyncCollector)",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to .pt checkpoint file to resume training from",
)

args = parser.parse_args()

# Resolve vision_mode and blind flag
if args.vision_mode is None:
    vision_mode = "blind" if args.blind else "rgb"
else:
    vision_mode = args.vision_mode
blind = (vision_mode == "blind")

# Resolve learning rates
lr_act = args.lr_actor if args.lr_actor is not None else args.lr
lr_crt = args.lr_critic if args.lr_critic is not None else args.lr
# Lower the alpha learning rate by 10x by default to prolong the exploration phase!
lr_alp = args.lr_alpha if args.lr_alpha is not None else (args.lr / 10.0)

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Segregate checkpoints into algorithm folders
args.checkpoint_dir = os.path.join(args.checkpoint_dir, "sac")
os.makedirs(args.checkpoint_dir, exist_ok=True)
if args.scratch_dir:
    os.makedirs(args.scratch_dir, exist_ok=True)

# Build Neural Networks, SAC Loss, and Optimizers
actor, critic, loss_module, target_updater, optimizers = build_sac_components(
    blind=blind,
    device=args.device,
    lr_actor=lr_act,
    lr_critic=lr_crt,
    lr_alpha=lr_alp,
    tau=args.tau,
    control_mode=args.control_mode,
)

# Force the initial entropy temperature (alpha) to 1.0 for massive early exploration
loss_module.log_alpha.data = torch.tensor(1.0, dtype=torch.float32, device=args.device).log()

# Resume from checkpoint if requested
start_step = 0
best_mean_reward = -float("inf")
if args.resume:
    if os.path.exists(args.resume):
        start_step, best_mean_reward = load_checkpoint(
            args.resume,
            actor=actor,
            critic=critic,
            loss_module=loss_module,
            optimizers=optimizers,
            device=args.device,
        )
        print(f"Resumed from {args.resume} at step {start_step} (best mean reward: {best_mean_reward:.2f})")
    else:
        print(f"Resume path {args.resume} not found. Starting fresh from step 0.")

# Initialize Weights & Biases if enabled
if args.wandb:
    wandb.init(
        project="rocker-bogie-rover",
        config={
            "algorithm": "SAC_TorchRL",
            "blind": args.blind,
            "workers": args.workers,
            "total_timesteps": args.total_timesteps,
            "frames_per_batch": args.frames_per_batch,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "lr_actor": lr_act,
            "lr_critic": lr_crt,
            "lr_alpha": lr_alp,
            "tau": args.tau,
            "gamma": args.gamma,
            "seed": args.seed,
            "device": args.device,
        },
    )

# Build Memory-Mapped Replay Buffer
replay_buffer = build_replay_buffer(
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    scratch_dir=args.scratch_dir,
    device="cpu",
    use_memmap=True,
)

# Build MultiAsyncCollector / MultiSyncCollector / Collector
collector = build_collector(
    workers=args.workers,
    seed=args.seed,
    blind=blind,
    policy=actor,
    frames_per_batch=args.frames_per_batch,
    total_frames=args.total_timesteps,
    sync=args.sync,
    device="cpu",
    control_mode=args.control_mode,
    vision_mode=vision_mode,
    terrain_mode=args.terrain,
    reward_mode=args.reward_mode,
)

mode_name = vision_mode.capitalize()
sync_name = "Sync" if args.sync else "Async"
print("=" * 80)
print(f"STARTING TORCHRL SAC TRAINING: {args.total_timesteps:,} steps")
print(f"Device: {args.device.upper()} | Workers: {args.workers} ({sync_name}) | Control: {args.control_mode} | Mode: {mode_name}")
print(f"Scratch Dir: {args.scratch_dir} | Replay Capacity: {args.buffer_size:,}")
print("=" * 80)

total_collected = start_step
ep_rewards: List[float] = []
current_ep_reward = 0.0
last_eval_step = start_step
last_log_time = time.time()

pbar = tqdm(total=args.total_timesteps, desc="Training SAC")
try:
    for batch in collector:
        # Defensively flatten batch across batch dimensions before adding to replay buffer
        flat_batch = batch.reshape(-1)
        replay_buffer.extend(flat_batch.to("cpu"))

        num_collected = flat_batch.numel()
        total_collected += num_collected
        pbar.update(num_collected)

        # Extract episode rewards & telemetry
        next_td = flat_batch.get("next", flat_batch)
        rewards = next_td.get("reward", None)
        dones = next_td.get("done", None)
        if rewards is not None and dones is not None:
            r_np = rewards.detach().cpu().numpy().reshape(-1)
            d_np = dones.detach().cpu().numpy().reshape(-1)
            for r, d in zip(r_np, d_np):
                current_ep_reward += float(r)
                if d:
                    ep_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0
                    
                    # Save historical checkpoint every 10 episodes for timelapse playback
                    num_eps = len(ep_rewards)
                    if num_eps % 10 == 0:
                        ckpt_path = os.path.join(args.checkpoint_dir, f"rover_ep_{num_eps}.pt")
                        save_checkpoint(
                            ckpt_path,
                            actor=actor,
                            critic=critic,
                            loss_module=loss_module,
                            optimizers=optimizers,
                            step=total_collected,
                            best_mean_reward=best_mean_reward,
                            blind=blind,
                        )

        # Perform gradient updates once buffer reaches sufficient size
        min_start = max(args.batch_size, min(args.learning_starts, args.buffer_size))
        losses_summary: Dict[str, float] = {}

        if len(replay_buffer) >= min_start:
            for _ in range(args.gradient_steps):
                sample_td = replay_buffer.sample().to(args.device)
                loss_td = loss_module(sample_td)

                loss_critic = loss_td["loss_qvalue"]
                loss_actor = loss_td["loss_actor"]
                loss_alpha = loss_td["loss_alpha"]

                # 1. Zero all gradients
                optimizers["critic"].zero_grad()
                optimizers["actor"].zero_grad()
                optimizers["alpha"].zero_grad()

                # 2. Compute all backward passes
                loss_critic.backward()
                loss_actor.backward()
                loss_alpha.backward()

                # 3. Apply optimizer steps
                optimizers["critic"].step()
                optimizers["actor"].step()
                optimizers["alpha"].step()

                # 4. Target network soft update
                target_updater.step()

                losses_summary["train/loss_critic"] = float(loss_critic.detach().item())
                losses_summary["train/loss_actor"] = float(loss_actor.detach().item())
                losses_summary["train/loss_alpha"] = float(loss_alpha.detach().item())
                if hasattr(loss_module, "log_alpha"):
                    losses_summary["train/alpha"] = float(loss_module.log_alpha.exp().item())

            # Synchronize policy weights to asynchronous rollout workers
            if hasattr(collector, "update_policy_weights_"):
                collector.update_policy_weights_()

        # Telemetry logging & progress reporting
        now = time.time()
        if now - last_log_time >= 2.0 or total_collected >= args.total_timesteps:
            last_log_time = now
            telemetry = extract_telemetry(flat_batch)
            log_dict = {**telemetry, **losses_summary}
            if ep_rewards:
                window = min(100, len(ep_rewards))
                log_dict["env/mean_ep_reward_100"] = float(np.mean(ep_rewards[-window:]))
                log_dict["env/latest_ep_reward"] = float(ep_rewards[-1])

            if args.wandb and wandb.run is not None:
                wandb.log(log_dict, step=total_collected)

            mean_rew_str = f"{np.mean(ep_rewards[-20:]):.1f}" if ep_rewards else "N/A"
            dist_str = f"{telemetry.get('env/live_distance_to_target', 0.0):.2f}m"
            loss_c_str = f"{losses_summary.get('train/loss_critic', 0.0):.3f}"
            pbar.set_description(f"Rew: {mean_rew_str} | Dist: {dist_str} | L-Q: {loss_c_str}")

        # Periodic evaluation, plotting, and checkpointing
        if total_collected - last_eval_step >= args.eval_interval:
            last_eval_step = total_collected
            save_plot(ep_rewards, total_collected, save_path="rewards_plot.png")

            # Save latest checkpoint
            latest_ckpt = os.path.join(args.checkpoint_dir, "latest_model.pt")
            save_checkpoint(
                latest_ckpt,
                actor=actor,
                critic=critic,
                loss_module=loss_module,
                optimizers=optimizers,
                step=total_collected,
                best_mean_reward=best_mean_reward,
                blind=blind,
            )

            # Save best checkpoint if rolling mean reward improved
            if ep_rewards:
                window = min(100, len(ep_rewards))
                current_mean_100 = float(np.mean(ep_rewards[-window:]))
                if current_mean_100 > best_mean_reward:
                    best_mean_reward = current_mean_100
                    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
                    save_checkpoint(
                        best_ckpt,
                        actor=actor,
                        critic=critic,
                        loss_module=loss_module,
                        optimizers=optimizers,
                        step=total_collected,
                        best_mean_reward=best_mean_reward,
                        blind=blind,
                    )
                    print(f"*** New best model saved ({best_mean_reward:.2f}) -> {best_ckpt} ***")

finally:
    collector.shutdown()
    pbar.close()

# Final checkpoint save on completion
save_plot(ep_rewards, total_collected, save_path="rewards_plot.png")
final_ckpt = os.path.join(args.checkpoint_dir, "rover_sac_final.pt")
save_checkpoint(
    final_ckpt,
    actor=actor,
    critic=critic,
    loss_module=loss_module,
    optimizers=optimizers,
    step=total_collected,
    best_mean_reward=best_mean_reward,
    blind=blind,
)
print(f"Training finished. Final checkpoint saved -> {final_ckpt}")

if args.wandb and wandb.run is not None:
    wandb.finish()
