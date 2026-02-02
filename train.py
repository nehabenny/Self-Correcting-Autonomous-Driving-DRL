import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from environment import make_env

class RewardThresholdCallback(BaseCallback):
    """
    Stop training if mean reward reaches threshold.
    """
    def __init__(self, threshold, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.threshold = threshold

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = sum([info['r'] for info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            if mean_reward >= self.threshold:
                if self.verbose > 0:
                    print(f"Stopping training: reward {mean_reward} reached threshold {self.threshold}")
                return False
        return True

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Stage 1: Straight Roads
    print("=== Stage 1: Training on Straight Roads (Map: S) ===")
    env = make_env(render=False, map_type="S")
    
    # Simple checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path='./models/checkpoints_stage1',
        name_prefix='stage1_model'
    )
    stop_callback = RewardThresholdCallback(threshold=35.0, verbose=1)
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/training"
    )
    
    print("Training Stage 1 (50,000 steps - ~1 min)...")
    model.learn(total_timesteps=50000, callback=[checkpoint_callback, stop_callback], progress_bar=True)
    model.save("models/stage1_final")
    env.close()

    # Stage 2: Tough Map (Intersections/Curves)
    print("\n=== Stage 2: Training on Tough Map (Map: SCX) ===")
    env = make_env(render=False, map_type="SCX")
    
    checkpoint_callback2 = CheckpointCallback(
        save_freq=25000, 
        save_path='./models/checkpoints_stage2',
        name_prefix='final_model'
    )
    stop_callback2 = RewardThresholdCallback(threshold=40.0, verbose=1)
    
    model.set_env(env)
    print("Training Stage 2 (100,000 steps - ~4 mins)...")
    model.learn(total_timesteps=100000, callback=[checkpoint_callback2, stop_callback2], progress_bar=True)
    model.save("models/final_model")
    env.close()
    
    print("Training complete. Models saved in ./models/")

if __name__ == "__main__":
    train()
