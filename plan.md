# Rover RL Training Pipeline Plan

## Phase 1: MuJoCo Scene Compilation (`scene_builder.py`)
Since the raw FreeCAD exporter creates a static, rigid ghost of the rover, our first step is to write a Python compiler that mathematically bridges CAD to physical reality.
1. **Load Assets:** Read the raw `assembly.xml` and our generated `cameras.xml`.
2. **Fix Collisions:** Delete the default `contype="0"` tags so the wheels and chassis have physical substance.
3. **Weight Calculation:** Dynamically calculate the physical mass of the rover by parsing the mesh surface areas and applying the density of 2mm thick hollow aluminum. 
4. **Motor Replacement:** 
   - Strip all default FreeCAD positional servos.
   - Inject `<velocity>` actuators (Target Speed) for the 6 `drv_` wheels. **Max Speed:** 29.24 rad/s (calculated from 10km/h and 190mm wheel diameter).
   - Inject `<position>` actuators (Target Angle) for the 4 `srv_` steering rotators.
5. **Sensor Injection:** Add `<jointpos>` sensors for the passive Rocker-Bogie joints (`pass_diff`, etc.) acting as rotary encoders, and a simulated IMU sensor for the chassis tilt. Note: The differential is strictly passive now, mimicking real Mars rovers!
6. **Output:** Generate the final, battle-ready `scene.xml`.

## Phase 2: The Gymnasium Environment (`rover_env.py`)
Here we connect the MuJoCo physics engine to the AI's "Brain".
1. **Initialization:** Load `scene.xml` into a headless MuJoCo `MjData` instance.
2. **Action Space:** Map the AI's 10 raw outputs (scaled -1 to 1) to realistic physical limits:
   - 6 Drive speeds (0 to 29.24 rad/s).
   - 4 Steering angles (e.g., -45 to +45 degrees).
3. **Observation Space:** 
   - Render the 4 RGB cameras at 128x128 pixels.
   - Extract the 12 numeric variables (IMU Pitch/Roll, Passive Joint angles, Relative X/Y vector to the target destination).
4. **Reset Logic:** Reset the rover to the origin, flatten the joints, and spawn a new random Target Coordinate 5 meters ahead in a semi-circle.
5. **Reward Function:** 
   - We will implement a balanced initial reward function with variables for forward progress (`+`), excessive tilt (`-`), and a massive `+100` bonus for entering the 50cm success radius.

## Phase 3: The PyTorch Brain & Parallel Training (`train.py`)
We will use Stable-Baselines3 (SB3) to handle the PPO math, but we will write our own PyTorch Neural Network architecture.
1. **Custom PyTorch Architecture:** Write a custom `BaseFeaturesExtractor` in PyTorch that builds a CNN for the 4 cameras and an MLP for the numeric sensors, automatically gluing them into a 64-dimensional feature vector.
2. **Parallel Universes:** Wrap the `RoverEnv` in a `SubprocVecEnv` to spawn **2 parallel environments** initially. This ensures we don't starve PyTorch of CPU cores during the backward pass. We will monitor the bottleneck (Simulation vs. PyTorch) and scale up if there is CPU headroom.
3. **Training Loop:** Launch PPO with our custom PyTorch brain and save model checkpoints every 10,000 steps to a `checkpoints/` directory.

## Phase 4: Visual Evaluation (`evaluate.py`)
1. Write a script to load the best saved PyTorch model.
2. Run a single instance of `RoverEnv` but with the interactive `mujoco.viewer` GUI enabled.
3. Watch the fully trained AI drive the rover in real-time over the terrain!
