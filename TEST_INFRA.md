# Test Infrastructure & Automated Verification Suite

## 1. Executive Summary

This document specifies the end-to-end automated testing infrastructure and validation matrix for the **6-Wheel Rover SAC Migration to Pure TorchRL**. The test suite guarantees behavioral parity with the original physical simulation while ensuring 100% compliance with TorchRL's `EnvBase`, `TensorDict`, `LazyMemmapStorage`, and `SACLoss` specifications.

---

## 2. Test Suite Architecture & Tiers

The test suite is structured into four rigorous verification tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        E2E Test Architecture                           │
├────────────────────────────────┬───────────────────────────────────────┤
│ Tier 1: Feature Coverage       │ Core specs, Ackermann kinematics,     │
│ (>= 5 tests per feature)       │ rendering, numeric state, rewards     │
├────────────────────────────────┼───────────────────────────────────────┤
│ Tier 2: Boundary & Corner      │ Clamping, max velocity/steer, tilt    │
│ (>= 5 tests per feature)       │ flips, sky-high z, step truncation   │
├────────────────────────────────┼───────────────────────────────────────┤
│ Tier 3: Cross-Feature          │ Steering + cameras, reset after flip, │
│ Combinations                   │ blind vs render parity, joint backprop│
├────────────────────────────────┼───────────────────────────────────────┤
│ Tier 4: Real-World Scenarios   │ Full trajectory rollouts, seed replay,│
│                                │ multi-worker stability, checkpoints   │
└────────────────────────────────┴───────────────────────────────────────┘
```

### Tier Breakdown & Scope Matrix

| Tier | Focus Area | Feature Categories | Total Tests |
|---|---|---|---|
| **Tier 1** | **Feature Coverage** | EnvBase Specs, Camera Observations, Numeric 13-dim State, Ackermann Kinematics, Reward Formulations, Tensor Shapes & Types, ConvBlock, MLP2, CameraCNN, RoverFeaturesExtractor, Actor/Critic Heads, Storage Setup, Loss Objectives | 70 |
| **Tier 2** | **Boundary & Corner Cases** | Max Accel & Steer Bounds, Extreme Tilt / Inverted Orientations, Sky-High Launch Z > 1.0m, Zero-Velocity Singularities, Max Episode Step Truncation (2000 steps), Target Proximity (5.0–5.1m spawn, 0.5m success), FIFO Replay Buffer Sizing | 38 |
| **Tier 3** | **Cross-Feature Interactions** | Steer + Offscreen RGB Rendering, Reset After Flip Cleanup, Extreme Command Action Clipping, Blind vs Render Mode Physics Parity, Buffer -> Critic Pipeline, Multi-Trajectory Batch Insertion, Policy Weight Target Smoothing | 12 |
| **Tier 4** | **Real-World Scenarios** | Complete Episode Trajectory Rollouts, Bit-for-Bit Deterministic Seed Replay, Multi-Worker Concurrent Simulators, Multi-Iteration Training Loop Steps, Checkpoint State-Dict Serialization & Inference | 10 |
| **Total** | **All Tiers** | **Comprehensive Full-Coverage Verification** | **130** |

---

## 3. Test File Directory & Responsibilities

All test implementations are co-located in `/extra/new_rover/tests/`:

```
/extra/new_rover/tests/
├── __init__.py                # Package initialization
├── run_all_tests.py           # Unified runner executing all test suites
├── test_env.py                # RoverEnv MuJoCo physics, kinematics & specs (66 tests)
├── test_models.py             # Pure PyTorch CNN, MLP2, Actor & Critic (35 tests)
├── test_storage.py            # LazyMemmapStorage & Replay Buffer (15 tests)
└── test_training_loop.py      # TorchRL Collectors, SACLoss & Telemetry (14 tests)
```

### Module Descriptions

1. **`tests/test_env.py` (66 Tests)**:
   - Verifies `observation_spec` (`cameras` `[12, 128, 128]` uint8, `numeric` `[13]` float32), `action_spec` (`[-1, 1]^2`), `reward_spec`, and composite `done_spec` (`done`, `terminated`, `truncated`).
   - Runs `check_env_specs(env)` compliance checks.
   - Tests True Ackermann steering differential angles and wheel speed scaling.
   - Tests dense progress reward, soft tilt penalties, terminal success bonus (+100.0), and flip penalties (-50.0).
   - Validates corner conditions: upside-down rover, sky-high launch, zero throttle singularity fallback, truncation at 2000 steps.

2. **`tests/test_models.py` (35 Tests)**:
   - Verifies 7-stage `ConvBlock` spatial reduction from $128\times 128 \rightarrow 1\times 1$.
   - Verifies `MLP2` non-linear projections and bounds.
   - Verifies `CameraCNN` 32-dim feature embedding.
   - Verifies multi-modal fusion ($32 \text{ cam} + 32 \text{ numeric} = 64 \text{ fused}$).
   - Validates `BlindRoverFeaturesExtractor` direct 13 -> 64-dim mapping.
   - Validates Actor continuous `TanhNormal` action generation and Critic double Q-value outputs.
   - Verifies gradient flow, parameter serialization, and extreme input stability.

3. **`tests/test_storage.py` (15 Tests)**:
   - Tests `TensorDictReplayBuffer` with `LazyMemmapStorage` targeting disk scratch directory.
   - Validates FIFO circular overwrite when buffer reaches `max_size`.
   - Tests batch insertion (`extend`) and random mini-batch sampling (`sample`).
   - Tests memory footprint safety and multi-trajectory persistence.

4. **`tests/test_training_loop.py` (14 Tests)**:
   - Tests `Collector` and `MultiSyncCollector` rollout generation.
   - Tests `SACLoss` objective calculation (`loss_actor`, `loss_qvalue`, `loss_alpha`).
   - Tests `SoftUpdate` Polyak target network parameter smoothing.
   - Validates the non-inplace backward pass sequence (`critic_loss.backward()`, `actor_loss.backward()`, `alpha_loss.backward()` before optimizer steps).
   - Tests high-resolution Weights & Biases telemetry dictionary generation.

5. **`tests/run_all_tests.py`**:
   - Master test runner script that discovers and executes all suites, formats clean output tables, and exits with standard exit codes (0 for pass, non-zero for fail).
   - Built cleanly without `if __name__ == '__main__': main()` anti-patterns.

---

## 4. Execution & Verification Instructions

### 4.1 Running the Full Test Suite

Execute the master runner with the project virtual environment:

```bash
.venv/bin/python tests/run_all_tests.py
```

Or using `pytest`:

```bash
.venv/bin/python -m pytest -p no:anyio tests/ -v
```

*(Alternatively with `uv`: `uv run python tests/run_all_tests.py` or `uv run pytest -p no:anyio tests/`)*

### 4.2 Running Specific Test Suites

```bash
# Test Environment & MuJoCo Physics
.venv/bin/python -m pytest -p no:anyio tests/test_env.py -v

# Test Neural Network Models & Extractors
.venv/bin/python -m pytest -p no:anyio tests/test_models.py -v

# Test Replay Buffer & Memmap Storage
.venv/bin/python -m pytest -p no:anyio tests/test_storage.py -v

# Test SAC Training Loop & Objectives
.venv/bin/python -m pytest -p no:anyio tests/test_training_loop.py -v
```

---

## 5. Authoritative Oracles & Expected Output Derivation

All expected values in the test suite are derived from:
1. **MuJoCo 3.12.0 Physics Solver**:
   - Exact numerical outputs from MuJoCo coordinate transformations, joint limits, and actuator force ranges.
2. **True Ackermann Kinematic Model**:
   - Dynamic track width calculation: $W_{\text{track}} = 2 |Y_i|$.
   - Per-wheel hub velocity: $V_{x,i} = v_{\text{COM}} - \omega Y_i$, $V_{y,i} = \omega X_i$.
   - Wheel steering angle: $\delta_i = \text{atan2}(V_{y,i}, |V_{x,i}| \cdot \text{sgn}(V_{x,i}))$.
3. **Formal Specification in `PROJECT.md` & `ORIGINAL_REQUEST.md`**:
   - 4-camera $128\times 128$ RGB stack ($12 \times 128 \times 128$).
   - 13-dim numeric observation vector.
   - Reward weights: $R_{\text{success}} = +100.0$, $R_{\text{flip}} = -50.0$, $R_{\text{dense}} = 20 \Delta d - 0.01$.

---

## 6. Latest Test Run Results

```
================================================================================
6-WHEEL ROVER TORCHRL SAC PIPELINE: FULL E2E TEST SUITE
================================================================================
Target test suites: tests/test_env.py, tests/test_models.py, tests/test_storage.py, tests/test_training_loop.py
Python interpreter: /extra/new_rover/.venv/bin/python
--------------------------------------------------------------------------------
tests/test_env.py:            66/66 PASSED [100%]
tests/test_models.py:         35/35 PASSED [100%]
tests/test_storage.py:        15/15 PASSED [100%]
tests/test_training_loop.py:  14/14 PASSED [100%]
================================================================================
ALL 130 TEST SUITES PASSED SUCCESSFULLY in 73.84s!
================================================================================
```
