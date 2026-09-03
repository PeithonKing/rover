"""
sac_utils.py
============
Reusable helper components and utilities for Soft Actor-Critic (SAC) training.
Decoupled from train_sac.py executable script to allow clean, pythonic top-level
script execution without pseudo-main guards.
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

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

from rover_env import RoverEnv
from models import make_actor, make_critic


def make_env_fn(
    rank: int = 0,
    seed: Optional[int] = 23,
    blind: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Callable[[], RoverEnv]:
    """Factory returning an initializer for RoverEnv with isolated seed and device."""
    def _init() -> RoverEnv:
        env = RoverEnv(render_mode=None, blind=blind, device=device)
        if seed is not None:
            env.set_seed(seed + rank)
        return env

    return _init


def build_sac_components(
    blind: bool = False,
    device: Union[str, torch.device] = "cpu",
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    target_entropy: Optional[float] = None,
    tau: float = 0.005,
) -> Tuple[
    ProbabilisticActor,
    TensorDictModule,
    SACLoss,
    SoftUpdate,
    Dict[str, torch.optim.Optimizer],
]:
    """Constructs Actor, Critic, SACLoss, SoftUpdate, and Adam optimizers."""
    action_spec = Bounded(
        shape=torch.Size([2]),
        dtype=torch.float32,
        low=-1.0,
        high=1.0,
        device=device,
    )
    actor = make_actor(blind=blind, action_spec=action_spec).to(device)
    critic = make_critic(blind=blind).to(device)

    if target_entropy is None:
        target_entropy = -2.0

    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=critic,
        num_qvalue_nets=2,
        action_spec=action_spec,
        loss_function="smooth_l1",
        target_entropy=target_entropy,
    ).to(device)

    target_updater = SoftUpdate(loss_module, eps=1.0 - tau)

    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())

    optimizers = {
        "actor": torch.optim.Adam(actor_params, lr=lr_actor, eps=1e-8),
        "critic": torch.optim.Adam(critic_params, lr=lr_critic, eps=1e-8),
        "alpha": torch.optim.Adam([loss_module.log_alpha], lr=lr_alpha, eps=1e-8),
    }

    return actor, critic, loss_module, target_updater, optimizers


def build_replay_buffer(
    buffer_size: int = 100_000,
    batch_size: int = 256,
    scratch_dir: Optional[str] = "./storage_scratch",
    device: Union[str, torch.device] = "cpu",
    use_memmap: bool = True,
) -> TensorDictReplayBuffer:
    """Initializes disk-backed LazyMemmapStorage or in-memory LazyTensorStorage replay buffer."""
    if use_memmap:
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
        storage = LazyMemmapStorage(
            max_size=buffer_size,
            scratch_dir=scratch_dir,
            device=device,
            existsok=True,
        )
    else:
        storage = LazyTensorStorage(max_size=buffer_size, device=device)

    buffer = TensorDictReplayBuffer(
        storage=storage,
        batch_size=batch_size,
        sampler=RandomSampler(),
    )
    return buffer


def build_collector(
    workers: int = 2,
    seed: int = 23,
    blind: bool = False,
    policy: Optional[ProbabilisticActor] = None,
    frames_per_batch: int = 64,
    total_frames: int = 10_000_000,
    sync: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Union[MultiAsyncCollector, MultiSyncCollector, Collector]:
    """Constructs asynchronous or synchronous parallel rollout data collector."""
    if workers > 1:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        env_fns = [make_env_fn(i, seed=seed, blind=blind, device=device) for i in range(workers)]
        if sync:
            collector = MultiSyncCollector(
                create_env_fn=env_fns,
                policy=policy,
                frames_per_batch=frames_per_batch,
                total_frames=total_frames,
                device=device,
                storing_device=device,
                reset_at_each_iter=False,
            )
        else:
            collector = MultiAsyncCollector(
                create_env_fn=env_fns,
                policy=policy,
                frames_per_batch=frames_per_batch,
                total_frames=total_frames,
                device=device,
                storing_device=device,
                reset_at_each_iter=False,
            )
    else:
        env_fn = make_env_fn(0, seed=seed, blind=blind, device=device)
        collector = Collector(
            create_env_fn=env_fn,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=device,
            reset_at_each_iter=False,
        )
    return collector


def extract_telemetry(batch: TensorDictBase) -> Dict[str, float]:
    """Pulls custom physics telemetry metrics from rollout TensorDict."""
    next_td = batch.get("next", batch)
    telemetry: Dict[str, float] = {}

    if "dist" in next_td.keys():
        telemetry["env/live_distance_to_target"] = float(next_td["dist"].float().mean().item())
    if "tilt_rad" in next_td.keys():
        telemetry["env/live_tilt_radians"] = float(next_td["tilt_rad"].float().mean().item())
    if "progress" in next_td.keys():
        telemetry["env/live_step_progress"] = float(next_td["progress"].float().mean().item())
    if "flipped" in next_td.keys():
        telemetry["env/flips_in_batch"] = float(next_td["flipped"].float().sum().item())
    if "success" in next_td.keys():
        telemetry["env/successes_in_batch"] = float(next_td["success"].float().sum().item())
    if "reward" in next_td.keys():
        telemetry["env/batch_reward_mean"] = float(next_td["reward"].float().mean().item())
        telemetry["env/batch_reward_sum"] = float(next_td["reward"].float().sum().item())

    return telemetry


def save_checkpoint(
    path: str,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    loss_module: SACLoss,
    optimizers: Dict[str, torch.optim.Optimizer],
    step: int,
    best_mean_reward: float,
    blind: bool = False,
) -> None:
    """Persists neural network state dicts and optimizer states to disk (.pt)."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    state = {
        "algo": "SAC_TorchRL",
        "blind": blind,
        "step": step,
        "best_mean_reward": best_mean_reward,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "loss_module_state_dict": loss_module.state_dict(),
        "actor_opt": optimizers["actor"].state_dict(),
        "critic_opt": optimizers["critic"].state_dict(),
        "alpha_opt": optimizers["alpha"].state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    loss_module: SACLoss,
    optimizers: Dict[str, torch.optim.Optimizer],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[int, float]:
    """Restores model and optimizer states from a .pt checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    if "critic_state_dict" in ckpt:
        critic.load_state_dict(ckpt["critic_state_dict"])
    if "loss_module_state_dict" in ckpt:
        loss_module.load_state_dict(ckpt["loss_module_state_dict"], strict=False)
    if "actor_opt" in ckpt and "actor" in optimizers:
        optimizers["actor"].load_state_dict(ckpt["actor_opt"])
    if "critic_opt" in ckpt and "critic" in optimizers:
        optimizers["critic"].load_state_dict(ckpt["critic_opt"])
    if "alpha_opt" in ckpt and "alpha" in optimizers:
        optimizers["alpha"].load_state_dict(ckpt["alpha_opt"])
    step = ckpt.get("step", 0)
    best_mean_reward = ckpt.get("best_mean_reward", -float("inf"))
    return step, best_mean_reward


def save_plot(
    ep_rewards: List[float],
    total_steps: int,
    save_path: str = "rewards_plot.png",
) -> None:
    """Generates rolling smoothed episode rewards plot."""
    if not ep_rewards:
        return
    window = min(100, len(ep_rewards))
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(ep_rewards, alpha=0.3, color="tab:blue", label="Episode reward")
    plt.plot(
        np.arange(window - 1, len(ep_rewards)),
        smoothed,
        color="tab:blue",
        label=f"Smoothed ({window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Rover RL SAC (TorchRL) — {total_steps:,} steps")
    plt.legend()
    plt.grid(True)

    if len(smoothed) > 0:
        y_min, y_max = float(np.min(smoothed)), float(np.max(smoothed))
        padding = max(0.1 * (y_max - y_min), 1.0)
        plt.ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
