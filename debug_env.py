from environment import make_env
import numpy as np

def debug():
    print("Creating env...")
    env = make_env(render=False, map_type="S")
    print(f"Observation Space: {env.observation_space}")
    
    print("Resetting...")
    ret = env.reset()
    print(f"Reset returned type: {type(ret)}")
    if isinstance(ret, tuple):
        print(f"Reset returned {len(ret)} values")
        obs = ret[0]
    else:
        obs = ret
        
    print(f"Obs type: {type(obs)}")
    if isinstance(obs, dict):
        print("Obs keys:", obs.keys())
        for k, v in obs.items():
            print(f"Key {k}: Type {type(v)}, Shape {getattr(v, 'shape', 'N/A')}")
    else:
        print("Obs is not a dict:", obs)

    print("Stepping...")
    try:
        action = env.action_space.sample()
        ret = env.step(action)
        print(f"Step returned {len(ret)} values")
        obs = ret[0]
        print(f"Step Obs type: {type(obs)}")
    except Exception as e:
        print(f"Step failed: {e}")
        
    env.close()

if __name__ == "__main__":
    debug()
