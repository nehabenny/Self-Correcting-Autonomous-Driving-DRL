import os
import glob
from agent_logic import load_agent
from env_wrapper import make_env
from metrics_logger import print_episode_summary, TransparencyCallback

def evaluate_milestone(model_path, env, num_episodes=5):
    """
    Runs evaluation for a specific milestone and returns aggregated stats.
    """
    model = load_agent(model_path)
    
    # We use a dict to accumulate results across episodes
    all_episode_stats = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        # Create an empty stats dict for this episode
        ep_stats = TransparencyCallback._get_empty_stats()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record granular rewards from the 'info' dict (populated by env_wrapper)
            ep_stats["speed"].append(info.get("reward_speed", 0.0))
            ep_stats["lateral"].append(info.get("reward_lateral", 0.0))
            ep_stats["route"].append(info.get("reward_route", 0.0))
            ep_stats["collision"].append(info.get("penalty_collision", 0.0))
            ep_stats["offroad"].append(info.get("penalty_offroad", 0.0))
            ep_stats["yellow_line"].append(info.get("penalty_yellow_line", 0.0))
            ep_stats["success"].append(info.get("reward_success", 0.0))
        
        all_episode_stats.append(ep_stats)
    
    # Aggregate stats over all episodes
    final_stats = TransparencyCallback._get_empty_stats()
    for ep_s in all_episode_stats:
        for key in final_stats:
            final_stats[key].extend(ep_s[key])
            
    # Normalize by number of episodes for the summary
    for key in final_stats:
        final_stats[key] = [sum(final_stats[key]) / num_episodes]
        
    return final_stats

def run_progress_analysis():
    print("\n" + "="*50)
    print("📈 AI EVOLUTION TRACKER: MILESTONE ANALYSIS")
    print("="*50 + "\n")

    milestone_files = sorted(glob.glob("models/milestones/milestone_*_steps.zip"))
    
    if not milestone_files:
        print("❌ No milestones found in models/milestones/")
        print("💡 Ensure you have started training with Stage 1.")
        return

    env = make_env(render=False, map_type="S") # Evaluate on Straight road (Stage 1)
    
    prev_total = None
    
    for file in milestone_files:
        steps = int(file.split("_")[-2])
        percent = (steps / 20000) * 100
        
        print(f"\n🔍 Analyzing Milestone: {percent:.0f}% ({steps} steps)")
        stats = evaluate_milestone(file, env)
        summary = print_episode_summary(stats, title=f"MILESTONE {percent:.0f}% PERFORMANCE")
        
        total = sum(summary.values())
        if prev_total is not None:
            diff = total - prev_total
            trend = "📈 IMPROVEMENT" if diff > 0 else "📉 REGRESSION"
            print(f"📊 TREND: {trend} ({diff:+.2f} points vs previous milestone)")
        
        prev_total = total

    env.close()
    print("\n🏁 Progress analysis complete.")

if __name__ == "__main__":
    run_progress_analysis()
