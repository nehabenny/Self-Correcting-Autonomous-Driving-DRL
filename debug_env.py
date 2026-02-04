from environment import make_env
import numpy as np

def debug():
    print("Creating env...")
    env = make_env(render=False, map_type="S")
    print(f"Observation Space: {env.observation_space}")
    
    print("Resetting...")
    ret = env.reset()
    if isinstance(ret, tuple):
        obs = ret[0]
    else:
        obs = ret
        
    print(f"Observation keys: {obs.keys()}")
    for k, v in obs.items():
        print(f"  - {k}: shape {v.shape}, dtype {v.dtype}")

    print("\nStepping...")
    try:
        action = env.action_space.sample()
        ret = env.step(action)
        obs = ret[0]
        print(f"Step Observation keys: {obs.keys()}")
        for k, v in obs.items():
            print(f"  - {k}: shape {v.shape}, dtype {v.dtype}")
    except Exception as e:
        print(f"Step failed: {e}")
        
    env.close()

if __name__ == "__main__":
    debug()
