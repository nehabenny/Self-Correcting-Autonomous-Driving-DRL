import os
import time
from stable_baselines3 import PPO
from environment import make_env

def test():
    model_path = "models/final_model.zip"
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print("Model not found. Please run train.py first.")
        return

    # Load the final model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    # We use the tough map to verify performance
    print("Creating environment with rendering (Map: SCX)...")
    env = make_env(render=True, map_type="SCX")
    
    print("Starting simulation. Press Ctrl+C to stop.")
    obs, info = env.reset()
    
    try:
        # Run loop
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render() 
            
            # Sleep removed to run as fast as possible
            
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopping test.")
    finally:
        env.close()

if __name__ == "__main__":
    test()
