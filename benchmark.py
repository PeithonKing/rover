import os
# Force CPU software rendering globally so it is hardware agnostic
os.environ["MUJOCO_GL"] = "osmesa"

import time
import torch
from torchrl.envs import ParallelEnv
from rover_env import RoverEnv

def make_env_creator(vision_mode):
    def _create_env():
        # Force CPU device for pure software benchmarking
        return RoverEnv(vision_mode=vision_mode, device="cpu")
    return _create_env

def run_parallel_benchmark(vision_mode, workers=12, steps_per_worker=250):
    print(f"Initializing {workers} workers for {vision_mode.upper()} mode...")
    
    env_creator = make_env_creator(vision_mode)
    env = ParallelEnv(workers, env_creator, mp_start_method="fork")
    
    obs = env.reset()
    
    # Create batched actions for all 12 workers
    dummy_action = torch.zeros((workers, 2), dtype=torch.float32, device="cpu")
    dummy_action[:, 0] = 0.5  # drive forward
    obs.set("action", dummy_action)
    
    print("Running...")
    start_time = time.time()
    
    for _ in range(steps_per_worker):
        obs = env.step(obs)
        obs = obs.get("next")
        obs.set("action", dummy_action)
        
        # If any environment in the batch finishes, reset the entire vector
        # to keep the benchmark simple and looping fast
        if obs.get("done").any():
            obs = env.reset()
            obs.set("action", dummy_action)
            
    end_time = time.time()
    env.close()
    
    duration = end_time - start_time
    total_steps = workers * steps_per_worker
    speed = total_steps / duration
    
    print("-" * 50)
    print(f"RESULTS FOR: {vision_mode.upper()}")
    print(f"Total Steps: {total_steps} ({workers} workers * {steps_per_worker} steps)")
    print(f"Total Time : {duration:.2f} seconds")
    print(f"Speed      : {speed:.2f} it/s")
    print("-" * 50 + "\n")


print("========================================")
print("   12-WORKER PARALLEL SPEED BENCHMARK   ")
print("========================================\n")

run_parallel_benchmark("blind", workers=12, steps_per_worker=250)
run_parallel_benchmark("depthmap", workers=12, steps_per_worker=250)
run_parallel_benchmark("rgb", workers=12, steps_per_worker=250)
