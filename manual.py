import os
# os.environ["MUJOCO_GL"] = "glfw"  # Force GLFW software rendering for machines without NVIDIA GPUs

import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from rover_env import RoverEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", action="store_true", help="Run without laggy camera rendering")
    args = parser.parse_args()

    env = RoverEnv(render_mode="human", blind=args.blind)
    obs = env.reset()

    speed = 0.0
    steer = 0.0

    # Setup matplotlib for display and keyboard capture
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    fig.canvas.manager.set_window_title("Rover Manual Control")
    img_display = ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
    
    def on_press(event):
        nonlocal speed, steer
        if event.key == 'up': speed = min(speed + 0.2, 1.0)
        elif event.key == 'down': speed = max(speed - 0.2, -1.0)
        elif event.key == 'left': steer = min(steer + 0.2, 1.0)
        elif event.key == 'right': steer = max(steer - 0.2, -1.0)

    def on_release(event):
        pass

    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('key_release_event', on_release)
    
    print("Controls: Arrows for Speed/Steering. Close the window to exit.")

    while plt.fignum_exists(fig.number):
        # Friction on inputs (simulating a self-centering joystick)
        speed *= 0.8 
        steer *= 0.8
        
        action = torch.tensor([speed, steer], dtype=torch.float32)
        obs.set("action", action)
        obs = env.step(obs)
        
        if not args.blind:
            # Extract cameras (12, 128, 128) -> 4 images of (3, 128, 128)
            cams = obs["next", "cameras"].cpu().numpy()
            cams = cams.reshape(4, 3, 128, 128).transpose(0, 2, 3, 1)
            
            # Stitch 2x2 grid
            top = np.hstack([cams[0], cams[1]])
            bottom = np.hstack([cams[2], cams[3]])
            grid = np.vstack([top, bottom])
            
            img_display.set_data(grid)
            fig.canvas.draw()
            
        fig.canvas.flush_events()
        
        if obs["next", "done"].item():
            print("Episode Finished! Resetting...")
            obs = env.reset()
            speed = 0.0
            steer = 0.0
            
        time.sleep(0.02)  # Limit framerate

    env.close()

if __name__ == "__main__":
    main()
