import os
import torch
from agent_logic import get_ppo_agent
from curriculum_manager import RewardThresholdCallback, get_curriculum_config
from env_wrapper import make_env
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_logger import TransparencyCallback

def train():
    """
    Main training orchestrator.
    Now modularized to demonstrate the Phase 1 (MetaDrive) setup
    which will be migrated to Phase 2 (CARLA) with minimal changes.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    device = "cpu"
    print(f"=== Starting Modular Training Workflow (Device: {device}) ===")
    
    stages = get_curriculum_config()
    model = None

    try:
        for i, stage in enumerate(stages):
            stage_num = i + 1
            print(f"\n🚀 {stage['name']} (Map: {stage['map']})")
            
            # 1. Initialize/Switch Environment (Wrapped)
            env = make_env(render=False, map_type=stage['map'])
            
            # 2. Get/Update Agent (Simulator-Agnostic)
            if model is None:
                model = get_ppo_agent(env, device=device)
            else:
                model.set_env(env)
            
            # 3. Setup Callbacks
            checkpoint_path = "./models/milestones" if stage_num == 1 else f"./models/checkpoints_stage{stage_num}"
            checkpoint_callback = CheckpointCallback(
                save_freq=4000 if stage_num == 1 else 50000, 
                save_path=checkpoint_path,
                name_prefix=f"milestone" if stage_num == 1 else f"stage{stage_num}_model"
            )
            stop_callback = RewardThresholdCallback(threshold=stage['threshold'], verbose=1)
            transparency_callback = TransparencyCallback()
            
            # 4. Train
            print(f"Training Stage {stage_num} (Goal: {stage['threshold']} reward)...")
            model.learn(
                total_timesteps=20000 if stage_num == 1 else 50000, 
                callback=[checkpoint_callback, stop_callback, transparency_callback], 
                progress_bar=False,
                reset_num_timesteps=False # Maintain progress across stages
            )
            
            # 5. Save Progress
            model.save(f"models/stage{stage_num}_final")
            env.close()
            print(f"✅ Stage {stage_num} Complete.")

        model.save("models/final_model")
        print("\n🏁 Curriculum training complete. Final model saved in ./models/final_model")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        if model is not None:
            save_path = "models/interrupted_model"
            model.save(save_path)
            print(f"💾 Progress saved to {save_path}.zip")
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    train()
