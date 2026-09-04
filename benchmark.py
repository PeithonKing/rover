import os
# Force CPU software rendering globally so it is hardware agnostic
os.environ["MUJOCO_GL"] = "egl"

import time
import torch
from torchrl.envs import ParallelEnv
from rover_env import RoverEnv
from components.observations import TargetAwareObservation, TargetBlindObservation

def make_env_creator(vision_mode):
    def _create_env():
        # Force CPU device for pure software benchmarking
        numeric_obs = TargetBlindObservation() if vision_mode != "blind" else TargetAwareObservation()
        return RoverEnv(vision_mode=vision_mode, device="cpu", numeric_obs=numeric_obs)
    return _create_env

def run_parallel_benchmark(vision_mode, workers=12, steps_per_worker=250):
    print(f"Initializing {workers} workers for {vision_mode.upper()} mode...")
    
    env_creator = make_env_creator(vision_mode)
    env = ParallelEnv(workers, env_creator, mp_start_method="spawn")
    
    obs = env.reset()
    
    # Create batched actions for all workers
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
        # print(_, end=" ", flush=True)
            
    end_time = time.time()
    env.close()
    
    duration = end_time - start_time
    total_steps = workers * steps_per_worker
    speed = total_steps / duration
    
    print()
    print("-" * 50)
    print(f"RESULTS FOR: {vision_mode.upper()}")
    print(f"Total Steps: {total_steps} ({workers} workers * {steps_per_worker} steps)")
    print(f"Total Time : {duration:.2f} seconds")
    print(f"Speed      : {speed:.2f} it/s")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    workers = 4
    print("========================================")
    print(f"   {workers}-WORKER PARALLEL SPEED BENCHMARK   ")
    print("========================================\n")
    
    run_parallel_benchmark("blind", workers=workers, steps_per_worker=1024*8)
    run_parallel_benchmark("depthmap", workers=workers, steps_per_worker=128)
    run_parallel_benchmark("rgb", workers=workers, steps_per_worker=128)
