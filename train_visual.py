import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import make_env

class RenderingCallback(BaseCallback):
    """
    Callback for rendering the environment during training.
    """
    def __init__(self, render_freq=10, verbose=0):
        super(RenderingCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                # Access the original environment
                self.training_env.render()
            except Exception:
                pass
        return True

def train_visual():
    os.makedirs("models", exist_ok=True)
    
    # Callback instance
    render_callback = RenderingCallback()
    
    # Stage 1: Straight Roads
    print("=== Stage 1: Visual Training on Straight Roads (Map: S) ===")
    print("Map 'S' is loading... Window should appear shortly.")
    
    # Create env with render=True
    env_stage1 = make_env(render=True, map_type="S") 
    
    model = PPO(
        "MultiInputPolicy", 
        env_stage1, 
        verbose=1,
        tensorboard_log="./logs/visual_train"
    )
    
    print("Training Stage 1... (Close the window to stop early, or wait for steps to complete)")
    try:
        model.learn(total_timesteps=5000, callback=render_callback, progress_bar=True)
        model.save("models/stage1_straight_visual")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env_stage1.close()
    
    # Stage 2: Intersections / Tough Map
    print("\n=== Stage 2: Visual Training on Tough Map (Map: SCX) ===")
    print("Map 'SCX' is loading...")
    
    try:
        env_stage2 = make_env(render=True, map_type="SCX")
        model.set_env(env_stage2)
        
        print("Training Stage 2...")
        model.learn(total_timesteps=5000, callback=render_callback, progress_bar=True)
        model.save("models/final_model_visual")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if 'env_stage2' in locals():
            env_stage2.close()
    
    print("Visual Training complete. Models saved in ./models/")

if __name__ == "__main__":
    train_visual()
