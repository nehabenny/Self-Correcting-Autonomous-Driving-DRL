# Self-Correcting Autonomous Driving Agent (DRL) 

An autonomous vehicle agent trained via **Deep Reinforcement Learning** in the **CARLA Simulator**. This project utilizes **Proximal Policy Optimization (PPO)** and a **Curriculum Learning** strategy to achieve stable navigation in complex urban environments.

## 🚀 Project Overview
* **Broad Area:** Machine Learning / Deep Reinforcement Learning
* **Core Technology:** PPO Algorithm & CARLA Simulator
* **Key Innovation:** A "Self-Correcting" curriculum that scales environment difficulty based on agent performance.

## 🏗️ System Architecture
The system follows a closed-loop Reinforcement Learning cycle:

1. **Perception:** Captures RGB Camera and Telemetry data.
2. **Decision:** PPO Policy determines Steering, Throttle, and Brake.
3. **Action:** Commands are executed in the high-fidelity CARLA physics engine.
4. **Correction:** A dynamic reward signal provides feedback for policy updates.


## 🛠️ Tech Stack
* **Simulator:** CARLA 0.9.13+
* **Frameworks:** PyTorch, Stable Baselines3, OpenAI Gym
* **Language:** Python 3.7+
* **Hardware Requirement:** NVIDIA GPU (8GB+ VRAM recommended)

## 🗺️ Roadmap & Work Division
- [ ] **Phase 1:** Environment Setup & Sensor Integration (Member 1)
- [ ] **Phase 2:** PPO Algorithm Implementation (Member 2)
- [ ] **Phase 3:** Curriculum Design & Reward Shaping (Member 3)
- [ ] **Phase 4:** Performance Analysis & Benchmarking (Member 4)

## 💻 How to Run (Coming Soon)
1. Install CARLA.
2. Clone this repo.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run `python src/main.py`.

## 📚 References
* Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms."
* Dosovitskiy, A., et al. (2017). "CARLA: An Open Urban Driving Simulator."