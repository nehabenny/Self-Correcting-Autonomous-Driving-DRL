import os
from agent_logic import load_agent
from env_wrapper import make_env

def test():
    """
    Visual inference script using the modular codebase.
    """
    # Priority: Latest interrupted model -> stage2 final -> final model
    potential_models = [
        "models/interrupted_model.zip",
        "models/stage2_final.zip",
        "models/final_model.zip",
        "models/stage1_final.zip"
    ]
    
    model_path = None
    for path in potential_models:
        if os.path.exists(path):
            model_path = path
            break

    if not model_path:
        print("❌ No trained model found in ./models/. Please run train.py first.")
        return

    print(f"📡 Loading modular agent from {model_path}...")
    try:
        model = load_agent(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("🌍 Creating environment (Map: SCX)...")
    env = make_env(render=True, map_type="SCX")
    
    print("▶️ Starting simulation. Press Ctrl+C to stop.")
    obs, info = env.reset()
    
    try:
        while True:
            try:
                action, _states = model.predict(obs, deterministic=True)
            except ValueError as e:
                print("\n❌ SENSOR MISMATCH ERROR:")
                print(f"Details: {e}")
                print("\n💡 POSSIBLE FIXES:")
                print("1. You are trying to load an OLD model (trained with LiDAR) into the NEW vision-only environment.")
                print("2. Delete your old models: 'rm models/*.zip'")
                print("3. Re-run training: './driving_env/bin/python3 train.py'")
                break

            obs, reward, terminated, truncated, info = env.step(action)
            env.render() 
            
            if terminated or truncated:
                print("🔄 Episode finished. Resetting.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n🛑 Stopping test.")
    finally:
        env.close()

if __name__ == "__main__":
    test()
