# Project: 6-Wheel Rover SAC Migration to Pure TorchRL

## Architecture
The project migrates a 6-wheel rover reinforcement learning training pipeline from Stable-Baselines3 to pure TorchRL with native `TensorDict` integration, asynchronous environment data collection, and memory-mapped replay buffer storage.

### Component Diagram & Data Flow
```
[MuJoCo 3D Physics Simulation (50 Hz)]
                 │
                 ▼
[RoverEnv (torchrl.envs.EnvBase)] ──> yields TensorDicts (cameras: [12,128,128], numeric: [13])
                 │
                 ▼
[MultiAsyncCollector] ──> Async multi-worker rollout collection
                 │
                 ▼
[TensorDictReplayBuffer + LazyMemmapStorage] ──> Disk-backed 100k+ step buffer
                 │
                 ▼
[SAC Training Loop (train_sac.py)]
  ├── RoverFeaturesExtractor (Pure PyTorch CNN + MLP2)
  ├── Actor: ProbabilisticActor (TanhNormal continuous 2D action [-1, 1]^2)
  ├── Critic: TensorDictModule (Double Q-networks)
  ├── SACLoss + SoftUpdate (Target networks)
  └── Telemetry & Logging (Weights & Biases: distance, tilt, progress, flips, successes, loss)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Native EnvBase Environment | Migrate `rover_env.py` to `torchrl.envs.EnvBase` natively yielding `TensorDict`s with `CompositeSpec`, True Ackermann steering, 4 camera feeds, 13 numeric sensors | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Pure PyTorch Feature Extractor | Rewrite `models.py` with 7-stage ConvBlock `CameraCNN` + `MLP2` for numeric inputs -> 64-dim embedding, wrapped in `TensorDictModule` | M2 | ORIGINAL_REQUEST §R2 |
| 3 | Pure PyTorch Actor & Critic | Implement Actor (`ProbabilisticActor` with `TanhNormal`) and Critic (Double-Q networks) wrapped in `TensorDictModule` | M2 | ORIGINAL_REQUEST §R2 |
| 4 | Async Data Collection | Configure `MultiAsyncCollector` for multi-worker asynchronous parallel environment rollouts | M3 | ORIGINAL_REQUEST §R3 |
| 5 | Memmap Replay Buffer Storage | Implement `TensorDictReplayBuffer` with `LazyMemmapStorage` capable of 100,000+ steps with disk `scratch_dir` | M3 | ORIGINAL_REQUEST §R3 |
| 6 | SB3 Nuking & Cleanup | Delete `train_ppo.py`, remove all Stable-Baselines3 dependencies and imports across the codebase | M4 | ORIGINAL_REQUEST §R4 |
| 7 | Pure TorchRL SAC Training Loop | Rewrite `train_sac.py` using TorchRL `SACLoss`, `SoftUpdate`, optimizers, and policy sync | M4 | ORIGINAL_REQUEST §R4 |
| 8 | Telemetry & Custom Physics Parity | Implement W&B logging for live distance, tilt radians, step progress, batch flips, batch successes, rewards, and SAC training losses | M5 | ORIGINAL_REQUEST §R5 |
| 9 | End-to-End Test & Verification | Full verification of training loop execution, checkpointing, and evaluation script compatibility | M5 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Native TensorDict Environment | Rewrite `rover_env.py` to inherit from `torchrl.envs.EnvBase`, yield native `TensorDict`s, define specs, fix reset forward kinematics, verify with `check_env_specs` | none | DONE |
| M2 | Pure PyTorch Networks | Rewrite `models.py` feature extractors, Actor, and Critic as pure PyTorch modules wrapped in `TensorDictModule` | M1 | PLANNED |
| M3 | Async Data Collection & Storage | Create reusable collector and replay buffer factory utilities with `MultiAsyncCollector` and `LazyMemmapStorage` | M1, M2 | PLANNED |
| M4 | Pure TorchRL Training Loop & SB3 Cleanup | Delete `train_ppo.py`, rewrite `train_sac.py` with TorchRL training loop and loss updates, clean up dependencies | M1, M2, M3 | PLANNED |
| M5 | Telemetry Parity & E2E Verification | Integrate W&B telemetry, physics metrics, evaluate script parity, and run end-to-end integration tests | M4 | PLANNED |

## Interface Contracts

### M1: `rover_env.py` (EnvBase)
- Class: `RoverEnv(torchrl.envs.EnvBase)`
- `observation_spec`: `CompositeSpec(cameras=Bounded(shape=(12, 128, 128), dtype=torch.uint8, min=0, max=255), numeric=UnboundedContinuous(shape=(13,), dtype=torch.float32))`
- `action_spec`: `Bounded(shape=(2,), dtype=torch.float32, min=-1.0, max=1.0)`
- `reward_spec`: `UnboundedContinuous(shape=(1,), dtype=torch.float32)`
- `done_spec`: `CompositeSpec(done=Binary(shape=(1,), dtype=torch.bool), terminated=Binary(shape=(1,), dtype=torch.bool), truncated=Binary(shape=(1,), dtype=torch.bool))`
- `_reset()`: Returns `TensorDict` with `"cameras"`, `"numeric"`, `"done"`, `"terminated"`, `"truncated"`
- `_step(tensordict)`: Reads `tensordict["action"]`, executes 10 MuJoCo substeps with True Ackermann steering, returns `TensorDict` with `"cameras"`, `"numeric"`, `"reward"`, `"done"`, `"terminated"`, `"truncated"`, and optional physics telemetry in `"info"`/root.

### M2: `models.py`
- `RoverFeaturesExtractor(nn.Module)`: `forward(cameras, numeric) -> features (shape (B, 64))`
- `make_actor(feature_extractor=None) -> TensorDictSequential / ProbabilisticActor`: Inputs `["cameras", "numeric"]`, Outputs `["action"]`, `["loc"]`, `["scale"]`
- `make_critic(feature_extractor=None) -> TensorDictModule`: Inputs `["cameras", "numeric", "action"]`, Outputs `["state_action_value"]`

### M3: Data Collection & Storage
- Collector: `MultiAsyncCollector(create_env_fn, policy_module, total_frames=..., frames_per_batch=...)`
- Buffer: `TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=100_000, scratch_dir=...))`

### M4 & M5: Training Loop & Telemetry
- Script: `train_sac.py`
- Execution: Asynchronous batch collection -> replay buffer storage -> SAC mini-batch updates -> W&B metric logging -> policy weights sync -> periodic evaluation & model checkpointing.

## Code Layout
- `/extra/new_rover/rover_env.py` — Native TorchRL `EnvBase` environment
- `/extra/new_rover/models.py` — Pure PyTorch Actor, Critic, Feature Extractors wrapped in `TensorDictModule`
- `/extra/new_rover/train_sac.py` — Pure TorchRL training script with `MultiAsyncCollector`, `LazyMemmapStorage`, and W&B logging
- `/extra/new_rover/evaluate.py` — Evaluation script loading TorchRL policy weights
- `/extra/new_rover/3D_files/` — MuJoCo XML definitions, meshes, and assets (read-only)
- `/extra/new_rover/tests/` — Automated test suites for Env, Models, Storage, Collector, and E2E Pipeline
