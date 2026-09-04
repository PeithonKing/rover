import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
"""
evaluate.py
===========
Load the best saved TorchRL SAC checkpoint and watch the rover drive in MuJoCo.

Usage:
    python evaluate.py                        # loads latest checkpoint from checkpoints/
    python evaluate.py --model checkpoints/best_model.pt
    python evaluate.py --episodes 5 --blind
    python evaluate.py --no-render            # Headless evaluation mode
"""

import os
import argparse
import numpy as np
import torch
from tensordict import TensorDict

from rover_env import RoverEnv
from models import make_actor


parser = argparse.ArgumentParser(description="Evaluate trained 6-Wheel Rover SAC policy")
parser.add_argument(
    "--model", "--model-path", type=str, default=None, help="Path to .pt policy checkpoint"
)
parser.add_argument(
    "--episodes", type=int, default=10, help="Number of evaluation episodes"
)
parser.add_argument(
    "--max-steps", type=int, default=2000, help="Maximum steps per episode"
)
parser.add_argument(
    "--control-mode",
    type=str,
    choices=["ackermann", "direct"],
    default=None,
    help="Control scheme ('ackermann' or 'direct'). Defaults to model checkpoint configuration.",
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
    "--blind", dest="blind", action="store_true", default=None, help="Force blind policy mode (numeric sensors only)"
)
parser.add_argument(
    "--visual", dest="blind", action="store_false", help="Force visual policy mode (4 cameras + numeric)"
)
parser.add_argument(
    "--deterministic", action="store_true", default=True, help="Use deterministic policy actions (loc mean)"
)
parser.add_argument(
    "--stochastic", action="store_false", dest="deterministic", help="Use stochastic sampled policy actions"
)
parser.add_argument(
    "--no-render", action="store_true", help="Disable human rendering for headless evaluation"
)
parser.add_argument(
    "--checkpoint-dir", type=str, default="checkpoints", help="Directory to search for checkpoints"
)

args = parser.parse_args()

# --- Find model checkpoint ---
model_path = args.model
if model_path is None:
    if os.path.exists(args.checkpoint_dir):
        ckpts = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
    else:
        ckpts = []

    if "best_model.pt" in ckpts:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    elif "latest_model.pt" in ckpts:
        model_path = os.path.join(args.checkpoint_dir, "latest_model.pt")
    elif "rover_sac_final.pt" in ckpts:
        model_path = os.path.join(args.checkpoint_dir, "rover_sac_final.pt")
    elif ckpts:
        model_path = os.path.join(args.checkpoint_dir, ckpts[-1])
    else:
        raise FileNotFoundError(
            f"No .pt checkpoints found in ./{args.checkpoint_dir}/. Train policy first or specify --model."
        )

print(f"Loading checkpoint: {model_path}")
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

# Extract state dict
if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
    actor_sd = ckpt["actor_state_dict"]
    step = ckpt.get("step", "N/A")
    best_mean_r = ckpt.get("best_mean_reward", "N/A")
    ckpt_blind = ckpt.get("blind", None)
    print(f"Loaded training checkpoint state (Step: {step}, Best Mean Reward: {best_mean_r})")
elif isinstance(ckpt, dict):
    actor_sd = ckpt
    ckpt_blind = None
    print("Loaded raw actor state dict.")
else:
    raise ValueError(f"Unrecognized checkpoint format in {model_path}")

# Determine model action dimension from checkpoint weights
model_action_dim = 2
for k, v in actor_sd.items():
    if "trunk" in k and k.endswith(".weight") and v.dim() == 2:
        model_action_dim = int(v.shape[0]) // 2

# Resolve control_mode: CLI override > auto-detection from checkpoint
if args.control_mode is not None:
    control_mode = args.control_mode
else:
    control_mode = "direct" if model_action_dim == 10 else "ackermann"

# Resolve vision_mode: CLI override > blind flag > checkpoint metadata > tensor shape heuristics
if args.vision_mode is not None:
    vision_mode = args.vision_mode
    is_blind = (vision_mode == "blind")
elif args.blind is not None:
    is_blind = args.blind
    vision_mode = "blind" if is_blind else "rgb"
elif ckpt_blind is not None:
    is_blind = bool(ckpt_blind)
    vision_mode = "blind" if is_blind else "rgb"
else:
    is_blind = not any("cnn" in k for k in actor_sd.keys())
    vision_mode = "blind" if is_blind else "rgb"

print(
    f"Policy configuration: Control={control_mode} (Model dim: {model_action_dim}) | "
    f"Mode={'Blind' if is_blind else vision_mode} | Deterministic={args.deterministic}"
)

# --- Load Actor Model ---
actor = make_actor(blind=is_blind, action_dim=model_action_dim)
actor.load_state_dict(actor_sd)
actor.eval()

# --- Evaluate Rollouts in RoverEnv ---
render_mode = None if args.no_render else "human"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = RoverEnv(
    device=device,
    render_mode=render_mode,
    blind=is_blind,
    control_mode=control_mode,
    vision_mode=vision_mode,
    terrain_mode=args.terrain,
    reward_mode=args.reward_mode,
    device="cpu",
)
total_rewards = []

print("=" * 80)
print(f"EVALUATING SAC POLICY ({args.episodes} episodes | Control: {control_mode} | Mode: {'Blind' if is_blind else vision_mode} | Deterministic: {args.deterministic})")
print("=" * 80)

for ep in range(args.episodes):
    td = env.reset()
    total_reward = 0.0
    done = False
    step_count = 0

    while not done and step_count < args.max_steps:
        with torch.no_grad():
            if args.deterministic:
                act_td = actor(td.clone())
                if "loc" in act_td.keys():
                    act_td["action"] = torch.tanh(act_td["loc"])
            else:
                act_td = actor(td.clone())

            # Dynamically adapt action dimensions if model and env differ
            if act_td["action"].shape[-1] == 2 and env.action_spec.shape[-1] == 10:
                speed = act_td["action"][..., 0:1].repeat_interleave(6, dim=-1)
                steer = act_td["action"][..., 1:2]
                steer_4 = torch.cat([steer, -steer, steer, -steer], dim=-1)
                act_td["action"] = torch.cat([speed, steer_4], dim=-1)
            elif act_td["action"].shape[-1] == 10 and env.action_spec.shape[-1] == 2:
                speed = act_td["action"][..., 0:6].mean(dim=-1, keepdim=True)
                steer = act_td["action"][..., 6:10].mean(dim=-1, keepdim=True)
                act_td["action"] = torch.cat([speed, steer], dim=-1)

            out_td = env.step(act_td)
            next_td = out_td["next"]
            reward = float(next_td["reward"].item())
            done = bool(next_td["done"].item())
            total_reward += reward
            step_count += 1
            td = next_td

    total_rewards.append(total_reward)
    info = env.last_info
    status = (
        "SUCCESS"
        if info.get("success")
        else ("FLIPPED" if info.get("flipped") else "TIMEOUT")
    )
    dist = info.get("dist", 0.0)
    tilt = info.get("tilt_rad", 0.0)
    print(
        f"  Episode {ep+1:>2}: reward={total_reward:>8.1f} | steps={step_count:>4} | dist={dist:.2f}m | tilt={tilt:.2f}rad | [{status}]"
    )

env.close()
avg_reward = float(np.mean(total_rewards))
print("=" * 80)
print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")
print("=" * 80)

import os
os._exit(0)

