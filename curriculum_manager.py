from stable_baselines3.common.callbacks import BaseCallback

class RewardThresholdCallback(BaseCallback):
    """
    Stop training if mean reward reaches threshold. 
    This is used to trigger the transition between curriculum stages.
    """
    def __init__(self, threshold, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.threshold = threshold

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            # Calculate mean reward from the recent episodes
            mean_reward = sum([info['r'] for info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            if mean_reward >= self.threshold:
                if self.verbose > 0:
                    print(f"\n[Curriculum] Threshold reached: {mean_reward:.2f} >= {self.threshold}")
                    print(f"[Curriculum] Transitioning to next stage...")
                return False
        return True

def get_curriculum_config():
    """
    Defines the curriculum stages.
    """
    return [
        {"name": "Stage 1: Straight Roads", "map": "S", "threshold": 50.0},
        {"name": "Stage 2: Complex Scenarios", "map": "SCX", "threshold": 50.0}
    ]
