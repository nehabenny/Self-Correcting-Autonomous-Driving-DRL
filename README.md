# 🏎️ Self-Correcting Autonomous Driving (DRL)

An advanced Reinforcement Learning pipeline for autonomous driving using **MetaDrive** and **Stable Baselines3**. This project focuses on **Vision-Only ("Tesla-style")** perception, modular architecture, and high-speed training optimization.

---

## 🚀 Key Features

### 1. 🏗️ Modular Architecture
The codebase is split into distinct, simulator-agnostic components:
- **`env_wrapper.py`**: Manages perception (64x64 Semantic-only) and safety logic.
- **`agent_logic.py`**: Handles PPO model initialization and hyperparameter tuning.
- **`curriculum_manager.py`**: Controls the transition from Straight Roads (Stage 1) to Complex S-Curves (Stage 2).
- **`metrics_logger.py`**: Human-readable terminal output and granular reward tracking.

### 2. ⚡ Turbo Mode (Fast-Track Training)
Optimized for rapid development and CPU-only training:
- **Resolution**: 64x64 (4x faster than HD).
- **Sensing**: **Semantic-Only Perception**. The agent sees a simplified map (Road vs. Non-Road), allowing for rapid learning of lane boundaries.
- **Speed**: Achieves **400+ FPS**. Stage 1 (20,000 steps) completes in **~50 seconds**.
- **Learning Rate**: Boosted to `1e-3` for faster philosophy updates.

### 3. 🛡️ Strict Safety Engine
Advanced failure conditions to ensure civil driving:
- **Yellow Line Termination**: Crossing the middle yellow line results in an **immediate failure (reset)** and a -20.0 penalty.
- **Vision-Only Boundaries**: Detects off-road and collision events using purely visual/semantic feedback.
- **Self-Correcting Rewards**: Penalties are weighted to prioritize "Not Crashing" over "Speeding."

### 4. 📊 Training Transparency
- **Single-Line Status**: Clean terminal output with real-time Score, Distance, Speed, and Safety metrics.
- **Evolution Tracker**: Use `progress.py` to analyze milestones (20%, 40%, etc.) and see exactly how the agent improves over time.
- **RL Glossary**: Included documentation explaining technical terms like `entropy_loss` and `explained_variance`.

---

## 🛠️ Getting Started

### Initial Setup
Ensure you have the environment ready:
```bash
bash setup_env.sh
```

### Start Fast-Track Training
Run the standard modular pipeline:
```bash
PYTHONUTF8=1 ./driving_env/bin/python3 train.py
```

### Visual Training (3D Window)
To see the agent learn in real-time:
```bash
PYTHONUTF8=1 ./driving_env/bin/python3 train_visual.py
```

### Analyze Learning Progress
Compare snapshots of the agent's brain across its lifetime:
```bash
./driving_env/bin/python3 progress.py
```

### Test Trained Agent
Run the final model in the visual simulator:
```bash
./driving_env/bin/python3 test.py
```

---

## 📈 Results (Stage 1)
| Milestone | Steps | Result |
| :--- | :--- | :--- |
| **0%** | 0 | Random steering, immediate off-road. |
| **20%** | 4,000 | Baseline exploration. |
| **60%** | 12,000 | **🌟 Breakthrough**: Holds lane consistently. |
| **100%** | 20,000 | Reliable lane-keeping on straight roads. |

---

## 📂 Project Structure
```text
.
├── agent_logic.py      # Brain (PPO Strategy)
├── env_wrapper.py      # Body (Sensors & Rewards)
├── train.py            # Headless Training Script
├── train_visual.py     # 3D Visual Training Script
├── test.py             # Inference/Demo Script
├── progress.py         # AI Evolution Analyzer
├── metrics_logger.py   # Terminal UI & Callbacks
└── models/             # Saved Brains & Milestones
```

---

## 🛑 Warning: Sensor Mismatch
This code uses **Vision-Only (Semantic)** sensing. If loading an old LiDAR-based model, use `test.py`'s built-in detection to troubleshoot or delete old `.zip` files in `models/`.
