# Self-Correcting Autonomous Driving Agent (MetaDrive + SB3)

This project implements a Self-Correcting Autonomous Driving agent using MetaDrive and Stable-Baselines3, specifically tuned for macOS Arm64 compatibility and enhanced sensor fusion.

## 🚀 Key Features
- **Integration Layer**: Pinned dependencies and extensive monkey-patching to avoid macOS rendering/math conflicts (e.g., Panda3D shader and glTF loader errors).
- **Sensor Fusion**: Custom `SensorFusionEnv` combining **RGB Camera** (Visual) and **LiDAR** (Vector) data.
- **Enhanced Vehicle Detection**: Now tracks the **8 nearest vehicles** explicitly in the vector observation, improving traffic awareness.
- **Curriculum Training**: Automated Stage 1 (Straight Roads) and Stage 2 (Tough Intersections) pipeline.
- **Real-time Visualization**: Active LiDAR visualization in the 3D window to verify agent perception.

## 🛠 Setup Instructions

### 1. Prerequisites
Ensure you have `python3` (Python 3.9 recommended) installed.

### 2. Installation
Run the automated setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

## 🏃‍♂️ Usage

**Always use the virtual environment python:**
`./driving_env/bin/python3 <script_name>.py`

### 1. The Complete Pipeline (Recommended)
Train the agent for ~5 minutes then automatically launch the visual test:
```bash
./driving_env/bin/python3 run_full_pipeline.py
```

### 2. Manual Training & Testing
- **Headless Training**: `./driving_env/bin/python3 train.py` (Fastest)
- **Visual Training**: `./driving_env/bin/python3 train_visual.py` (Watch it learn)
- **Only Testing**: `./driving_env/bin/python3 test.py` (Visualizes the saved `final_model.zip`)

## 📂 File Structure
- `environment.py`: Custom environment with sensor fusion and macOS patches.
- `train.py`: Sequential training logic (Stage 1 -> Stage 2).
- `test.py`: Visual inference script.
- `run_full_pipeline.py`: Master orchestrator.
- `train_visual.py`: Training script with a real-time rendering callback.
- `requirements.txt`: Version-locked dependencies for stability.

## ⚠️ Notes for macOS Users
- The `environment.py` includes critical monkey-patches for `simplepbr`, `gltf`, and `metadrive` to work on Apple Silicon.
- If the simulation window doesn't appear, ensure you are not running over a non-GUI session (like standard SSH).
