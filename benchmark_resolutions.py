import os
import time
import torch
import numpy as np
from rover_env import RoverEnv
from components.controllers import AckermannController
from components.eyes import DepthmapEyes
from components.rewards import StandardReward
from components.terrains import FlatTerrain

def run_resolution_benchmark(res, steps=500, runs=5, device_str="cpu"):
    print(f"Testing Resolution: {res}x{res} on {device_str.upper()}")
    speeds = []
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    for run in range(runs):
        env = RoverEnv(
            controller=AckermannController(),
            eyes=DepthmapEyes(width=res, height=res),
            reward=StandardReward(),
            terrain=FlatTerrain(),
            # Passing device here controls where the TensorDict is created
            device=device
        )
        obs = env.reset()
        
        dummy_action = torch.tensor([0.5, 0.0], dtype=torch.float32, device=device)
        obs.set("action", dummy_action)
        
        start_time = time.time()
        for _ in range(steps):
            obs = env.step(obs)
            obs = obs.get("next")
            obs.set("action", dummy_action)
            
            if obs.get("done").item():
                obs = env.reset()
                obs.set("action", dummy_action)
                
        end_time = time.time()
        env.close()
        
        duration = end_time - start_time
        its = steps / duration
        speeds.append(its)
        print(f"  Run {run+1}/{runs}: {its:.2f} it/s")
        
    avg_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    print(f"-> AVERAGE for {res}x{res}: {avg_speed:.2f} it/s (+/- {std_speed:.2f})")
    print("-" * 40)


print("========================================")
print(f"   DEPTHMAP BENCHMARK")
print("========================================\n")

run_resolution_benchmark(8)
run_resolution_benchmark(16)
run_resolution_benchmark(32)
run_resolution_benchmark(64)
run_resolution_benchmark(128)
