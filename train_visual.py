import os
import torch
from agent_logic import get_ppo_agent
from curriculum_manager import get_curriculum_config
from env_wrapper import make_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from metrics_logger import TransparencyCallback

class RenderingCallback(BaseCallback):
    """
    Modular callback for rendering the environment during training.
    """
    def __init__(self, render_freq=1, verbose=0):
        super(RenderingCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                self.training_env.render()
            except Exception:
                pass
        return True

def train_visual():
    """
    Visual training loop to demonstrate the 'Self-Correcting' behavior.
    """
    os.makedirs("models", exist_ok=True)
    
    # Use CPU for maximum stability on Mac during visual rendering
    # unless MPS is specifically requested.
    device = "cpu" 
    print(f"=== Starting Visual Training (Device: {device}) ===")
    
    stages = get_curriculum_config()
    render_callback = RenderingCallback()
    model = None

    for i, stage in enumerate(stages):
        stage_num = i + 1
        print(f"\n📺 Stage {stage_num}: {stage['name']} (Render Mode active)")
        
        try:
            # 1. Create Env with Rendering
            env = make_env(render=True, map_type=stage['map'])
            
            # 2. Setup Agent
            if model is None:
                model = get_ppo_agent(env, device=device)
            else:
                model.set_env(env)
            
            # 3. Learn (Short bursts for visual demo)
            print(f"Watch the engine learn in the 3D window...")
            transparency_callback = TransparencyCallback()
            callbacks = CallbackList([render_callback, transparency_callback])
            model.learn(total_timesteps=10000, callback=callbacks, progress_bar=False)
            
            # 4. Save
            model.save(f"models/stage{stage_num}_visual")
            env.close()
        except KeyboardInterrupt:
            print("\n🛑 Visual training stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error during visual training: {e}")
            if 'env' in locals(): env.close()

    print("\n🏁 Visual demo training complete.")

if __name__ == "__main__":
    train_visual()
