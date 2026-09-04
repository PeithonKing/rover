"""
evaluate_passive.py
===================
A completely passive evaluation script. No AI model is loaded.
The rover is fed 0.0 across all active actuator channels.
Use this to visually verify gravity, collisions, and the rocker-bogie suspension joints in MuJoCo.

Usage:
    python evaluate_passive.py                # Runs interactive viewer
    python evaluate_passive.py --no-render    # Runs headless simulation
    python evaluate_passive.py --steps 300    # Run specific number of steps
    python evaluate_passive.py --control-mode direct --steps 10
"""

import os
import time
import argparse
import numpy as np
import torch
from tensordict import TensorDict

from rover_env import RoverEnv


parser = argparse.ArgumentParser(description="Run passive physics simulation for 6-Wheel Rover")
parser.add_argument(
    "--steps", type=int, default=1000, help="Maximum number of simulation steps (0 = infinite)"
)
parser.add_argument(
    "--no-render", action="store_true", help="Disable viewer rendering for headless testing"
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
    "--blind", action="store_true", default=True, help="Disable offscreen camera rendering for higher fps"
)
parser.add_argument(
    "--reset-interval", type=int, default=150, help="Reset interval in steps (drops rover from drop height)"
)

args = parser.parse_args()

# Resolve vision mode
if args.vision_mode is None:
    vision_mode = "blind" if args.blind else "rgb"
else:
    vision_mode = args.vision_mode
blind = (vision_mode == "blind")

render_mode = None if args.no_render else "human"
env = RoverEnv(
    render_mode=render_mode,
    blind=blind,
    control_mode=args.control_mode,
    vision_mode=vision_mode,
    terrain_mode=args.terrain,
    reward_mode=args.reward_mode,
    device="cpu",
)

print("=" * 80)
print(f"RUNNING PASSIVE ROVER SIMULATION (Control: {args.control_mode} | Zero Action Commands)")
print(f"Render Mode: {render_mode or 'headless'} | Steps: {args.steps or 'infinite'}")
print("=" * 80)

td = env.reset()
passive_action = torch.zeros(env.action_spec.shape[-1], dtype=torch.float32)
step_count = 0

try:
    while args.steps == 0 or step_count < args.steps:
        step_count += 1

        # Periodic drop reset
        if args.reset_interval > 0 and step_count % args.reset_interval == 0:
            print(f"Step {step_count}: Dropping rover via env.reset()!")
            td = env.reset()

        step_td = TensorDict({"action": passive_action}, batch_size=[])
        out_td = env.step(step_td)
        next_td = out_td["next"]

        if render_mode == "human":
            time.sleep(1.0 / 50.0)

        if bool(next_td["done"].item()):
            td = env.reset()
        else:
            td = next_td

    print(f"\nPassive simulation completed {step_count} steps.")

except KeyboardInterrupt:
    print("\nPassive test terminated by user.")
finally:
    env.close()

import os
os._exit(0)
