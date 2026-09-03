"""
evaluate_passive.py
===================
A completely passive evaluation script. No AI model is loaded.
The rover is fed 0.0 for acceleration and steering commands.
Use this to visually verify gravity, collisions, and the rocker-bogie suspension joints in MuJoCo.

Usage:
    python evaluate_passive.py                # Runs interactive viewer
    python evaluate_passive.py --no-render    # Runs headless simulation
    python evaluate_passive.py --steps 300    # Run specific number of steps
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
    "--blind", action="store_true", default=True, help="Disable offscreen camera rendering for higher fps"
)
parser.add_argument(
    "--reset-interval", type=int, default=150, help="Reset interval in steps (drops rover from drop height)"
)

args = parser.parse_args()

render_mode = None if args.no_render else "human"
env = RoverEnv(render_mode=render_mode, blind=args.blind, device="cpu")

print("=" * 80)
print("RUNNING PASSIVE ROVER SIMULATION (Zero Action Commands)")
print(f"Render Mode: {render_mode or 'headless'} | Steps: {args.steps or 'infinite'}")
print("=" * 80)

td = env.reset()
passive_action = torch.zeros(2, dtype=torch.float32)
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
