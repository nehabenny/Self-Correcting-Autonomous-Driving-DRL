from stable_baselines3 import PPO
import torch

def get_ppo_agent(env, device="cpu", tensorboard_log="./logs/training"):
    """
    Initializes the PPO agent with a MultiInputPolicy (Sensor Fusion).
    This logic is simulator-agnostic and will remain the same for CARLA.
    """
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3, # Turbo: Faster philosophy update
        device=device,
        stats_window_size=1, # Quicker reward reporting
        tensorboard_log=tensorboard_log
    )
    return model

def load_agent(path, env=None, device="cpu"):
    """
    Loads a trained PPO model.
    """
    if env:
        return PPO.load(path, env=env, device=device)
    return PPO.load(path, device=device)
