from stable_baselines3.common.callbacks import BaseCallback

class TransparencyCallback(BaseCallback):
    """
    Consolidated, single-line logger for cleaner training output.
    """
    def __init__(self, verbose=1):
        super(TransparencyCallback, self).__init__(verbose)
        self.stats = self._get_empty_stats()
        self.episode_count = 0

    @staticmethod
    def _get_empty_stats():
        return {
            "speed": [], "lateral": [], "route": [],
            "collision": [], "offroad": [], "yellow_line": [], "success": []
        }

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.stats["speed"].append(info.get("reward_speed", 0.0))
            self.stats["lateral"].append(info.get("reward_lateral", 0.0))
            self.stats["route"].append(info.get("reward_route", 0.0))
            self.stats["collision"].append(info.get("penalty_collision", 0.0))
            self.stats["offroad"].append(info.get("penalty_offroad", 0.0))
            self.stats["yellow_line"].append(info.get("penalty_yellow_line", 0.0))
            self.stats["success"].append(info.get("reward_success", 0.0))
            
            if self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                print_episode_summary(self.stats, count=self.episode_count)
                self.stats = self._get_empty_stats()
        return True

def print_episode_summary(stats, title="EPISODE", count=None):
    summary = {k: sum(v) for k, v in stats.items()}
    total = sum(summary.values())
    
    # Concise Single-Line Format
    count_str = f"#{count}" if count else ""
    line = (f"🚗 {title} {count_str:4} | Reward: {total:7.2f} | "
            f"Dist: {summary['route']:5.1f} | Spd: {summary['speed']:5.1f} | "
            f"Align: {summary['lateral']:5.1f} | Safety: {summary['collision']+summary['offroad']+summary['yellow_line']:5.1f}")
    
    print(line)
    return summary
