# Self-Correcting Autonomous Driving Agent (DRL) 

An autonomous vehicle agent trained via **Deep Reinforcement Learning** in the **CARLA Simulator**. This project utilizes **Proximal Policy Optimization (PPO)** and a **Curriculum Learning** strategy to achieve stable navigation in complex urban environments.

##  Project Overview
- [cite_start]**Broad Area:** Machine Learning / Deep Reinforcement Learning[cite: 98].
- [cite_start]**Core Technology:** PPO Algorithm & CARLA Simulator[cite: 113, 124].
- [cite_start]**Key Innovation:** A "Self-Correcting" curriculum that scales environment difficulty based on agent performance[cite: 100, 159].

##  System Architecture
The system follows a closed-loop Reinforcement Learning cycle:
1. [cite_start]**Perception:** Captures RGB Camera and Telemetry data[cite: 85, 122].
2. [cite_start]**Decision:** PPO Policy determines Steering, Throttle, and Brake[cite: 86, 122].
3. [cite_start]**Action:** Commands are executed in the high-fidelity CARLA physics engine[cite: 152].
4. [cite_start]**Correction:** A dynamic reward signal provides feedback for policy updates[cite: 86, 159].



##  Tech Stack
- [cite_start]**Simulator:** CARLA 0.9.13+[cite: 91, 150].
- [cite_start]**Frameworks:** PyTorch, Stable Baselines3, OpenAI Gym[cite: 150, 151].
- [cite_start]**Language:** Python 3.7+.
- [cite_start]**Hardware Requirement:** NVIDIA GPU (8GB+ VRAM recommended)[cite: 152].

##  Roadmap & Work Division
- [ ] [cite_start]**Phase 1:** Environment Setup & Sensor Integration (Member 1)[cite: 90, 146].
- [ ] [cite_start]**Phase 2:** PPO Algorithm Implementation (Member 2)[cite: 124, 146].
- [ ] [cite_start]**Phase 3:** Curriculum Design & Reward Shaping (Member 3)[cite: 123, 146].
- [ ] [cite_start]**Phase 4:** Performance Analysis & Benchmarking (Member 4)[cite: 145, 146].

## How to Run (Coming Soon)
1. Install CARLA.
2. Clone this repo.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run `python src/main.py`.

## 📚References
- Schulman, J., et al. (2017). [cite_start]"Proximal Policy Optimization Algorithms."[cite: 164, 166].
- Dosovitskiy, A., et al. (2017). [cite_start]"CARLA: An Open Urban Driving Simulator."[cite: 164].