import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import csv

class AlertLoggingCallback(BaseCallback):
    """This callback logs in when the alerts are issued"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.when_alerted = []
        self.streaks = []
        self.current_streak = None
        self.last_alert = None
        self.num_over_budget = 0
        self.num_alerts = 0
        self.num_steps = 0

    def _on_step(self) -> bool:
        self.n_envs = len(self.training_env.envs)
        if self.current_streak is None:
            self.last_alert = np.zeros(self.n_envs, dtype=int)
            self.current_streak = np.zeros(self.n_envs, dtype=int)
            self.rolled_rewards = np.zeros(self.n_envs, dtype=float)
            self.a_50 = np.full(self.n_envs, np.nan)
            self.a_80 = np.full(self.n_envs, np.nan)
            self.a_100 = np.full(self.n_envs, np.nan)

        for i, env in enumerate(self.training_env.envs):
            self.num_steps += 1

            if env.penalize:
                self.num_over_budget += 1

            if env.attempted_alert_buffer:
                prev_alert = self.last_alert[i]
                this_alert = env.attempted_alert_buffer[-1]
                if this_alert:  # alert issued
                    self.when_alerted.append(env.t)
                    self.num_alerts += 1
                    self.current_streak[i] += 1
                elif prev_alert:  # end streak
                    self.streaks.append(self.current_streak[i])
                    self.current_streak[i] = 0
                self.last_alert[i] = this_alert
            
            if env.t == env.n_days - 2: # if done on last day, cum_reward will already have been reset to 0
                self.rolled_rewards[i] += env.cum_reward
                s = sum(env.allowed_alert_buffer)
                if s > 0:
                    fracs = np.cumsum(env.allowed_alert_buffer)/s
                    for k in range(0,len(fracs)):
                        if np.isnan(self.a_100[i]) and fracs[k] == 1:
                            self.a_100[i] = k
                        if np.isnan(self.a_80[i]) and fracs[k] >= 0.8:
                            self.a_80[i] = k
                        if np.isnan(self.a_50[i]) and fracs[k] >= 0.5:
                            self.a_50[i] = k

        return True

    def _on_rollout_end(self):
        # Log the metrics to TensorBoard
        summary = {
            "training_rewards": np.mean(self.rolled_rewards),
            "over_budget_freq": self.num_over_budget / self.num_steps,
            "alerts_freq": self.num_alerts / self.num_steps,
            "average_t_alerts": np.mean(self.when_alerted) if self.when_alerted else 0,
            "stdev_t_alerts": np.std(self.when_alerted) if self.when_alerted else 0,
            "average_streak": np.mean(self.streaks) if self.streaks else 0,
            "stdev_streak": np.std(self.streaks) if self.streaks else 0,
            "alert_t_50%": np.nanmean(self.a_50),
            "alert_t_80%": np.nanmean(self.a_80),
            "alert_t_100%": np.nanmean(self.a_100),
        }

        for k, v in summary.items():
            self.logger.record(f"custom/{k}", v)

        # Reset counters
        self.when_alerted = []
        self.streaks = []
        self.current_streak = None
        self.last_alert = None
        self.num_over_budget = 0
        self.num_alerts = 0
        self.num_steps = 0
        self.rolled_rewards = np.zeros(self.n_envs, dtype=float)


class FinalEvalCallback(BaseCallback):
    """This callback logs the year, budget, alerts, and rewards"""
    def __init__(self, verbose=0, filename="test_log"):
        super().__init__(verbose)
        self.filename=filename
        self.year = 0
        self.budget = 0
        self.alerts = []
        self.sum_alerts = 0
        self.reward = 0
        self.when_alerted = []
        self.streaks = []
        self.current_streak = 0
        self.last_alert = 0
    def __call__(self, L = None, G = None):
        self.L = L
        self.year = 0
        self.budget = 0
        self.alerts = []
        self.sum_alerts = 0
        self.reward = 0
        self.when_alerted = []
        self.streaks = []
        self.current_streak = 0
        self.last_alert = 0
        self.data = []
    def _on_step(self) -> bool:
        env = self.L["eval_env"] #self.training_env
        prev_alert = self.last_alert
        this_alert = env.allowed_alert_buffer[-1]
        if this_alert: # alert issued
            self.when_alerted.append(env.t)
            self.current_streak += 1
        elif prev_alert:  # end streak
            self.streaks.append(self.current_streak)
            self.current_streak = 0
        self.last_alert = this_alert
        if env.t == env.n_days - 2:
            self.year = env.other_data["y"][env.feature_ep_index, self.env.t]
            self.budget = env.other_data["budget"][env.feature_ep_index, self.env.t]
            self.alerts = env.allowed_alert_buffer
            self.sum_alerts = sum(self.alerts)
            self.reward = env.cum_reward
        return True
    def _on_rollout_end(self):
        self.data.append({
            "year": self.year, 
            "alert_budget": self.budget,
            "sum_alerts": self.sum_alerts,
            "reward": self.reward,
            "average_t_alerts": np.mean(self.when_alerted) if self.when_alerted else 0,
            "stdev_t_alerts": np.std(self.when_alerted) if self.when_alerted else 0,
            "average_streak": np.mean(self.streaks) if self.streaks else 0,
            "stdev_streak": np.std(self.streaks) if self.streaks else 0,
            "alerts": self.alerts,
        })
        # Reset:
        self.when_alerted = []
        self.streaks = []
        self.current_streak = 0
        self.last_alert = 0
    def _on_training_end(self):
        with open(self.filename, 'w', newline='') as csvfile:
            fieldnames = list(self.data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)