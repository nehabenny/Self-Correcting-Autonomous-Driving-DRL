from environment import make_env
import numpy as np

def debug_obs():
    print("Initializing environment...")
    env = make_env(render=False, map_type="S")
    obs, info = env.reset()
    
    print("\n--- Observation Space ---")
    print(f"Keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Vector shape: {obs['vector'].shape}")
    
    # Analyze vector content
    # MetaDrive vector obs usually has:
    # 0-3: state (heading, steering, velocity, etc)
    # The rest: LiDAR and potentially navigation
    vec = obs['vector']
    print("\n--- Vector Observation Sample ---")
    print(f"Vector Mean: {np.mean(vec)}")
    print(f"Vector Max: {np.max(vec)}")
    print(f"Vector Min: {np.min(vec)}")
    
    print("\nFirst 10 values (State?):", vec[:10])
    
    # Try a few steps to see if vector changes
    print("\nTaking 10 steps...")
    for i in range(10):
        # Action: [steering, throttle] -> (0, 1) means go straight fast
        obs, reward, terminated, truncated, info = env.step([0, 1])
        if i % 5 == 0:
            print(f"Step {i}, Reward: {reward}, Velocity: {info.get('velocity', 'N/A')}")
    
    env.close()

if __name__ == "__main__":
    debug_obs()
