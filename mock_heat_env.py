"""
Mock Heat Alert Environment for testing DDTS baselines without network dependency.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple


class MockHeatAlertEnv(gym.Env):
    """Mock version of HeatAlertEnv for testing DDTS baselines."""
    
    def __init__(self, season_length: int = 153, budget: int = 20):
        super().__init__()
        
        self.season_length = season_length
        self.initial_budget = budget
        self.budget = budget
        self.current_step = 0
        
        # Observation space: [heat_index, remaining_budget, alert_lag1, alert_2wks, alert_streak]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, max(1.0, float(budget)), 1.0, 14.0, max(1.0, float(season_length))], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Action space: 0 = no alert, 1 = alert
        self.action_space = spaces.Discrete(2)
        
        # Internal state
        self.heat_index_series = None
        self.alert_history = []
        self.alert_streak = 0
        self.rng = np.random.RandomState(42)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.current_step = 0
        self.budget = self.initial_budget
        self.alert_history = []
        self.alert_streak = 0
        
        # Generate synthetic heat index series
        # Start with base temperature and add daily variations
        base_temp = 0.3 + 0.4 * self.rng.random()  # Random baseline
        daily_variations = self.rng.normal(0, 0.1, self.season_length)
        trend = np.linspace(0, 0.2, self.season_length)  # Seasonal warming
        heat_waves = self._generate_heat_waves()
        
        self.heat_index_series = np.clip(
            base_temp + daily_variations + trend + heat_waves,
            0.0, 1.0
        )
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _generate_heat_waves(self):
        """Generate synthetic heat wave events."""
        heat_waves = np.zeros(self.season_length)
        
        # Only add heat waves if season is long enough
        if self.season_length <= 10:
            return heat_waves
        
        # Add 2-4 heat wave events
        num_events = self.rng.randint(2, 5)
        
        for _ in range(num_events):
            max_start = max(0, self.season_length - 10)
            start = self.rng.randint(0, max_start + 1) if max_start > 0 else 0
            duration = min(self.rng.randint(3, 8), self.season_length - start)
            intensity = 0.2 + 0.3 * self.rng.random()
            
            for i in range(duration):
                if start + i < self.season_length:
                    heat_waves[start + i] += intensity * np.exp(-i / 3.0)
        
        return heat_waves
    
    def _get_observation(self):
        """Get current observation."""
        # Handle case where episode is done
        step_idx = min(self.current_step, self.season_length - 1)
        heat_index = self.heat_index_series[step_idx]
        remaining_budget = float(self.budget)
        alert_lag1 = float(self.alert_history[-1] if self.alert_history else 0)
        alert_2wks = float(sum(self.alert_history[-14:]))
        alert_streak = float(self.alert_streak)
        
        return np.array([heat_index, remaining_budget, alert_lag1, alert_2wks, alert_streak], 
                       dtype=np.float32)
    
    def _get_info(self):
        """Get info dictionary."""
        step_idx = min(self.current_step, self.season_length - 1)
        return {
            'remaining_budget': self.budget,
            'at_budget': self.budget <= 0,
            'current_step': self.current_step,
            'season_length': self.season_length,
            'heat_index': self.heat_index_series[step_idx]
        }
    
    def step(self, action: int):
        """Take a step in the environment."""
        action = int(action)
        
        # Check if action is valid (budget constraint)
        if action == 1 and self.budget <= 0:
            # Invalid action - can't alert without budget
            actual_action = 0
            penalty = -1.0  # Penalty for invalid action
        else:
            actual_action = action
            penalty = 0.0
        
        # Update alert history and budget
        self.alert_history.append(actual_action)
        if actual_action == 1:
            self.budget -= 1
            self.alert_streak += 1
        else:
            self.alert_streak = 0
        
        # Calculate reward based on heat index and action
        step_idx = min(self.current_step, self.season_length - 1)
        heat_index = self.heat_index_series[step_idx]
        
        # Reward function: negative hospitalizations prevented by alerts
        # Higher heat index means more potential hospitalizations
        baseline_hospitalizations = heat_index * 100  # Scale for demonstration
        
        if actual_action == 1:
            # Alert issued - reduce hospitalizations by effectiveness
            effectiveness = 0.3  # 30% effectiveness
            prevented = baseline_hospitalizations * effectiveness
            reward = prevented - 5.0  # Cost of issuing alert
        else:
            # No alert - full hospitalizations occur
            reward = -baseline_hospitalizations
        
        reward += penalty
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.season_length
        
        observation = self._get_observation() if not done else self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        if self.current_step < len(self.heat_index_series):
            heat_index = self.heat_index_series[self.current_step]
            print(f"Step {self.current_step}: Heat Index = {heat_index:.3f}, "
                  f"Budget = {self.budget}, Streak = {self.alert_streak}")