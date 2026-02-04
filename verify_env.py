from environment import make_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
import shimmy

try:
    print("Creating environment...")
    env = make_env(render=False, map_type="S")
    
    # Check if we need to wrap it
    # SB3 check_env 
    print("Checking environment with SB3...")
    # We might need to wrap it if it is a legacy gym env
    # MetaDrive is legacy gym (0.21 usually)
    
    # Try checking directly
    # formatting output to ensure we see errors
    check_env(env)
    print("Environment is compatible!")
except Exception as e:
    print(f"Environment check failed: {e}")
    # Suggest wrapper
    import gymnasium
    print("Attempting to wrap with GymV21CompatibilityV0...")
