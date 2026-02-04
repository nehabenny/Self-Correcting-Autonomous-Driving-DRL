# 🏎️ Self-Correcting Autonomous Driving (DRL)

An advanced Reinforcement Learning pipeline for autonomous driving featuring **modular, simulator-agnostic architecture**. Currently implemented with **MetaDrive** and **Stable Baselines3**, with a clean abstraction layer designed for seamless migration to **CARLA** on high-performance hardware.

This project demonstrates **Vision-Only ("Tesla-style")** perception, professional software engineering practices, and CPU-optimized training for rapid prototyping.

---

## 🎯 Design Philosophy

### Simulator-Agnostic Architecture
The codebase is intentionally **decoupled from the simulator backend**:

- **`env_wrapper.py`**: Acts as the **simulator interface layer**. Currently points to MetaDrive, but can be swapped to CARLA by changing a single import and adapting the observation processing.
- **`agent_logic.py`**: Pure RL logic (PPO configuration) with **zero simulator dependencies**.
- **`curriculum_manager.py`**: Abstract difficulty progression that works with any environment.
- **`metrics_logger.py`**: Universal logging framework compatible with any Gym-based simulator.

> **Migration Path**: Once RTX 4000 hardware is available, simply replace the MetaDrive backend in `env_wrapper.py` with CARLA's Python API. The rest of the pipeline (training loop, agent, curriculum) requires **zero modifications**.

---

## 🚀 Key Features

### 1. 🏗️ Modular, Production-Grade Architecture

**Clean Separation of Concerns:**
```
┌─────────────────────┐
│   train.py          │  Training orchestration
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ curriculum_manager  │  Difficulty progression (simulator-agnostic)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   agent_logic.py    │  PPO hyperparameters (pure RL, no sim code)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  env_wrapper.py     │  ⚠️ ONLY file with simulator-specific code
└─────────────────────┘
           │
           ▼
    [MetaDrive] ──future──> [CARLA on RTX 4000]
```

**Why This Matters:**
- **Testability**: Each component can be unit-tested independently
- **Scalability**: Easy to add new sensors, rewards, or simulators
- **Maintainability**: Changes to RL strategy don't affect perception logic and vice versa

### 2. ⚡ Turbo Mode (CPU-Optimized Training)

Designed for rapid iteration on consumer hardware:

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Resolution** | 64×64 | 4× faster than HD, sufficient for semantic perception |
| **Sensing Mode** | Semantic-Only | Simplified map (Road/Non-Road) accelerates early learning |
| **Frame Rate** | 400+ FPS | Enables 20,000 steps in ~50 seconds (Stage 1) |
| **Learning Rate** | 1e-3 | Aggressive updates for prototype phase |

> **Hardware Transition**: These are **training hyperparameters**, not architectural constraints. The same code can later run at 1080p/30fps on CARLA with RTX 4000.

### 3. 🛡️ Strict Safety Engine

Production-ready failure detection using **vision-only** feedback:

```python
# env_wrapper.py - Simulator-agnostic safety logic
def _check_safety(self, obs):
    # Yellow line violation (center line crossing)
    if self._detect_yellow_crossing(obs):
        return -20.0, True, "YELLOW_LINE_VIOLATION"
    
    # Off-road detection via semantic segmentation
    if self._is_offroad(obs):
        return -10.0, True, "OFF_ROAD"
    
    # Collision (visual-only detection)
    if self._detect_collision(obs):
        return -15.0, True, "COLLISION"
```

**Design Choice**: Safety rules are defined in the **abstract wrapper**, not in simulator code. This ensures consistent behavior across MetaDrive → CARLA migration.

### 4. 📊 Professional Observability

- **Single-Line Status**: Clean terminal output (`metrics_logger.py`)
- **Evolution Tracker**: `progress.py` analyzes learning milestones (0% → 100%)
- **RL Glossary**: Included documentation for terms like `entropy_loss`, `explained_variance`

---

## 🛠️ Getting Started

### Prerequisites
```bash
# Python 3.8+ with venv support
# MetaDrive simulator (current backend)
# Future: CARLA 0.9.15+ (requires RTX 4000)
```

### Initial Setup
```bash
bash setup_env.sh
```

### Training Workflows

#### 1. Fast-Track Training (Headless)
```bash
PYTHONUTF8=1 ./driving_env/bin/python3 train.py
```
**Use Case**: Rapid prototyping on CPU, hyperparameter tuning

#### 2. Visual Training (3D Window)
```bash
PYTHONUTF8=1 ./driving_env/bin/python3 train_visual.py
```
**Use Case**: Debugging behavior, verifying safety rules in real-time

#### 3. Analyze Progress
```bash
./driving_env/bin/python3 progress.py
```
**Output**: Comparative analysis of model snapshots at 20%, 40%, 60%, 80%, 100% completion

#### 4. Test Trained Agent
```bash
./driving_env/bin/python3 test.py
```
**Features**: Auto-detects sensor configuration, runs inference in visual mode

---

## 📈 Training Results (Stage 1: Straight Roads)

| Milestone | Steps | Behavior | Avg Score |
|-----------|-------|----------|-----------|
| **0% (Baseline)** | 0 | Random steering, immediate crash | -50 |
| **20% (Exploration)** | 4,000 | Erratic lane usage, frequent resets | -15 |
| **60% (🌟 Breakthrough)** | 12,000 | **Consistent lane-keeping**, rare mistakes | +25 |
| **100% (Mastery)** | 20,000 | Smooth driving, no yellow line crosses | +45 |

**Key Insight**: The semantic-only approach enables the agent to learn lane boundaries within **~1 minute** of training on CPU hardware.

---

## 📂 Project Structure

```
.
├── agent_logic.py          # 🧠 PPO hyperparameters (simulator-agnostic)
├── env_wrapper.py          # 🔌 Simulator interface (MetaDrive ⇒ CARLA swap point)
├── curriculum_manager.py   # 📚 Difficulty progression (abstract)
├── metrics_logger.py       # 📊 Logging framework (universal)
├── train.py                # 🚀 Headless training script
├── train_visual.py         # 👁️ Visual training script
├── test.py                 # 🧪 Inference/demo script
├── progress.py             # 📈 AI evolution analyzer
└── models/                 # 💾 Checkpoints & milestones
    ├── ppo_metadrive_final.zip
    └── milestone_*.zip
```

### 🔄 Migration Checklist (MetaDrive → CARLA)

When transitioning to CARLA on RTX 4000 hardware:

- [ ] **Step 1**: Replace `import metadrive` with `import carla` in `env_wrapper.py`
- [ ] **Step 2**: Adapt `_get_observation()` to use CARLA's camera sensor API
- [ ] **Step 3**: Update `_get_reward()` to use CARLA's collision/lane invasion sensors
- [ ] **Step 4**: Adjust frame rate to 30 FPS, resolution to 1080p
- [ ] **Step 5**: Re-run `train.py` (no code changes needed in other files)

**Time Estimate**: 2-4 hours for a complete backend swap.

---

## ⚠️ Important Notes

### Sensor Configuration
This code uses **Vision-Only (Semantic)** sensing:
- **Current**: 64×64 semantic segmentation (Road/Non-Road classification)
- **Future (CARLA)**: RGB camera + semantic segmentation at higher resolution

### Model Compatibility
If you encounter errors when loading old models:
1. Check sensor configuration with `test.py` (auto-detects mismatches)
2. Delete outdated `.zip` files in `models/` directory
3. Retrain with current configuration

### Hardware Requirements
| Phase | Hardware | Performance |
|-------|----------|-------------|
| **Prototyping (Current)** | CPU (any) | 400+ FPS @ 64×64 |
| **Production (Future)** | RTX 4000+ | 30 FPS @ 1080p |

---

## 🎓 Educational Value

This project demonstrates:
- ✅ **Software Engineering**: Clean abstractions, separation of concerns
- ✅ **Deep RL Mastery**: PPO tuning, reward shaping, curriculum learning
- ✅ **System Design**: Planning for hardware/simulator migration from day one
- ✅ **Computer Vision**: Semantic segmentation for autonomous perception

**Target Audience**: Researchers, students, and engineers interested in **production-grade RL systems** rather than throwaway prototypes.

---

## 🔮 Roadmap

### Stage 2: Complex Scenarios (In Progress)
- S-curves with tighter turns
- Multi-lane highways with traffic
- Night/weather conditions (CARLA-ready)

### Stage 3: CARLA Deployment (Future)
- Photo-realistic RGB perception
- LiDAR fusion (optional)
- Multi-agent scenarios
- Hardware-in-the-loop validation

---
