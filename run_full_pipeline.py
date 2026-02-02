import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from train import train
from test import test

def run_pipeline():
    print("🚀 Starting Full Autonomous Driving Pipeline")
    
    # 1. Training (Headless)
    print("\n--- Phase 1: Training ---")
    try:
        train()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # 2. Testing (Visual)
    print("\n--- Phase 2: Visualization ---")
    print("Launching simulation window...")
    try:
        test()
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

if __name__ == "__main__":
    run_pipeline()
