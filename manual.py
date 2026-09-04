"""
manual.py
=========
Interactive manual keyboard teleoperation for 6-Wheel Rover in MuJoCo.

Controls:
- Up Arrow: Accelerate forward (+0.2 speed)
- Down Arrow: Decelerate / reverse (-0.2 speed)
- Left Arrow: Steer left (+0.2 steer)
- Right Arrow: Steer right (-0.2 steer)

Supports:
- Ackermann 2D control and Direct 10D actuator mapping.
- Visual modes (RGB, Depth) and Blind mode.
- Top-level pythonic execution without wrappers.
- Strict 7-bit ASCII throughout.
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from rover_env import RoverEnv


parser = argparse.ArgumentParser(description="Manual keyboard teleoperation for 6-Wheel Rover")
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
    help="Run without camera rendering",
)
args = parser.parse_args()

# Resolve vision mode
if args.vision_mode is None:
    vision_mode = "blind" if args.blind else "rgb"
else:
    vision_mode = args.vision_mode
blind = (vision_mode == "blind")

env = RoverEnv(
    render_mode="human",
    blind=blind,
    control_mode=args.control_mode,
    vision_mode=vision_mode,
    terrain_mode=args.terrain,
    reward_mode=args.reward_mode,
)
obs = env.reset()

speed = 0.0
steer = 0.0

# Setup matplotlib for display and keyboard capture
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")
fig.canvas.manager.set_window_title("Rover Manual Control")
img_display = ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))


def on_press(event):
    global speed, steer
    if event.key == "up":
        speed = min(speed + 0.2, 1.0)
    elif event.key == "down":
        speed = max(speed - 0.2, -1.0)
    elif event.key == "left":
        steer = min(steer + 0.2, 1.0)
    elif event.key == "right":
        steer = max(steer - 0.2, -1.0)


def on_release(event):
    pass


fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)

print("Controls: Arrows for Speed/Steering. Close the window to exit.")

action_dim = env.action_spec.shape[-1]

while plt.fignum_exists(fig.number):
    # Friction on inputs (simulating a self-centering joystick)
    speed *= 0.8
    steer *= 0.8

    if action_dim == 10:
        # Map speed and steer across all 10 actuators: 6 drive motors + 4 steer rotators
        drive = [speed] * 6
        steer_rot = [steer, -steer, steer, -steer]
        action = torch.tensor(drive + steer_rot, dtype=torch.float32)
    else:
        action = torch.tensor([speed, steer], dtype=torch.float32)

    obs.set("action", action)
    obs = env.step(obs)

    if not blind and ("next", "cameras") in obs:
        cams_tensor = obs["next", "cameras"]
        if cams_tensor.numel() > 0:
            cams = cams_tensor.cpu().numpy()
            if cams.shape[0] == 12:
                # 4 RGB cameras: (12, 128, 128) -> 4 images of (128, 128, 3)
                cams_rgb = cams.reshape(4, 3, 128, 128).transpose(0, 2, 3, 1)
                top = np.hstack([cams_rgb[0], cams_rgb[1]])
                bottom = np.hstack([cams_rgb[2], cams_rgb[3]])
                grid = np.vstack([top, bottom])
                img_display.set_data(grid)
                fig.canvas.draw()
            elif cams.shape[0] == 4:
                # 4 depth cameras: (4, 128, 128)
                d_norm = (
                    (cams - cams.min()) / (cams.max() - cams.min() + 1e-6) * 255.0
                ).astype(np.uint8)
                d_rgb = np.stack([d_norm] * 3, axis=-1)
                top = np.hstack([d_rgb[0], d_rgb[1]])
                bottom = np.hstack([d_rgb[2], d_rgb[3]])
                grid = np.vstack([top, bottom])
                img_display.set_data(grid)
                fig.canvas.draw()

    fig.canvas.flush_events()

    if obs["next", "done"].item():
        print("Episode Finished! Resetting...")
        obs = env.reset()
        speed = 0.0
        steer = 0.0

    time.sleep(0.02)

env.close()
