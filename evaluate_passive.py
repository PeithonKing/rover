"""
evaluate_passive.py
===================
A completely passive evaluation script. No AI is loaded.
The rover is fed 0.0 for all motor and steering commands.
Use this to visually verify gravity, collisions, and the suspension joints.
"""

import numpy as np
import time
import mujoco
from rover_env import RoverEnv

def main():
    env = RoverEnv(render_mode="human", blind=True)
    obs, info = env.reset()
    
    passive_action = np.zeros(10, dtype=np.float32)
    step_count = 1
    
    print("Running passive simulation. Close the viewer window to exit.")
    try:
        while True:
            # Every 5 seconds (150 steps at 30 FPS), trigger a standard reset
            if step_count % 150 == 0:
                print("Dropping rover via reset()!")
                env.reset()
                
            obs, reward, terminated, truncated, info = env.step(passive_action)
            time.sleep(1.0 / 30.0)
            step_count += 1
            
    except KeyboardInterrupt:
        print("\nPassive test terminated by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
