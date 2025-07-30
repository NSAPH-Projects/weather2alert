#!/usr/bin/env python3
"""
DDTS Gym Runner for Weather2Alert

This script runs DDTS (Decision-Theoretic Time Series) baselines for various numeric environments
including Heat Alerts, Uganda, Mimic, and Bin Packing environments using Gymnasium.

DDTS baselines provide simple heuristic policies that can serve as benchmarks for RL agents.
These baselines originated from decision-theoretic approaches to sequential decision making
under uncertainty.
"""

import argparse
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Callable
import warnings
warnings.filterwarnings("ignore")

# Import weather2alert environment
try:
    import weather2alert.env
    WEATHER2ALERT_AVAILABLE = True
except ImportError:
    WEATHER2ALERT_AVAILABLE = False
    print("Warning: weather2alert environment not available")

# Import mock environment for testing
try:
    from mock_heat_env import MockHeatAlertEnv
    MOCK_ENV_AVAILABLE = True
except ImportError:
    MOCK_ENV_AVAILABLE = False

# Import numeric environments integration
try:
    from numeric_envs import create_numeric_environment, get_environment_baselines, ALL_NUMERIC_BASELINES
    NUMERIC_ENVS_AVAILABLE = True
except ImportError:
    NUMERIC_ENVS_AVAILABLE = False
    ALL_NUMERIC_BASELINES = {}


class DDTSBaseline:
    """Base class for DDTS baseline policies."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset baseline state for new episode."""
        pass
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Get action from baseline policy."""
        raise NotImplementedError


class ThresholdBaseline(DDTSBaseline):
    """Threshold-based baseline policy for heat alerts."""
    
    def __init__(self, threshold: float = 0.75, name: str = "threshold"):
        super().__init__(name)
        self.threshold = threshold
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Issue alert if heat index exceeds threshold and budget allows."""
        # Get heat index from observation (assuming it's the first feature)
        heat_index = observation[0] if len(observation) > 0 else 0.0
        
        # Check if we have budget remaining
        remaining_budget = info.get('remaining_budget', 0)
        
        # Issue alert if heat index exceeds threshold and we have budget
        if heat_index >= self.threshold and remaining_budget > 0:
            return 1  # Issue alert
        return 0  # No alert


class ConservativeBaseline(DDTSBaseline):
    """Conservative baseline that rarely issues alerts."""
    
    def __init__(self, threshold: float = 0.9, name: str = "conservative"):
        super().__init__(name)
        self.threshold = threshold
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Issue alert only for very high heat index values."""
        heat_index = observation[0] if len(observation) > 0 else 0.0
        remaining_budget = info.get('remaining_budget', 0)
        
        if heat_index >= self.threshold and remaining_budget > 0:
            return 1
        return 0


class AggressiveBaseline(DDTSBaseline):
    """Aggressive baseline that issues many alerts."""
    
    def __init__(self, threshold: float = 0.5, name: str = "aggressive"):
        super().__init__(name)
        self.threshold = threshold
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Issue alert for moderate heat index values."""
        heat_index = observation[0] if len(observation) > 0 else 0.0
        remaining_budget = info.get('remaining_budget', 0)
        
        if heat_index >= self.threshold and remaining_budget > 0:
            return 1
        return 0


class BudgetAwareBaseline(DDTSBaseline):
    """Budget-aware baseline that conserves budget for later in season."""
    
    def __init__(self, threshold: float = 0.75, budget_factor: float = 0.5, name: str = "budget_aware"):
        super().__init__(name)
        self.threshold = threshold
        self.budget_factor = budget_factor
        self.total_steps = 0
        self.current_step = 0
    
    def reset(self):
        """Reset for new episode."""
        self.total_steps = 0
        self.current_step = 0
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Issue alert based on heat index and remaining season time."""
        self.current_step += 1
        
        heat_index = observation[0] if len(observation) > 0 else 0.0
        remaining_budget = info.get('remaining_budget', 0)
        
        # Estimate if we're in the latter part of the season
        # If we don't know total steps, be more conservative early
        if self.total_steps == 0:
            # Assume 153 days as typical season length
            progress = self.current_step / 153.0
        else:
            progress = self.current_step / self.total_steps
        
        # Adjust threshold based on season progress and budget
        adjusted_threshold = self.threshold - (progress * self.budget_factor * 0.2)
        
        if heat_index >= adjusted_threshold and remaining_budget > 0:
            return 1
        return 0


class RandomBaseline(DDTSBaseline):
    """Random baseline for comparison."""
    
    def __init__(self, alert_probability: float = 0.1, name: str = "random"):
        super().__init__(name)
        self.alert_probability = alert_probability
        self.rng = np.random.RandomState(42)
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Randomly issue alerts."""
        remaining_budget = info.get('remaining_budget', 0)
        
        if remaining_budget > 0 and self.rng.random() < self.alert_probability:
            return 1
        return 0


# DDTS baseline registry (includes both general and environment-specific baselines)
DDTS_BASELINES = {
    'threshold': ThresholdBaseline,
    'conservative': ConservativeBaseline,
    'aggressive': AggressiveBaseline,
    'budget_aware': BudgetAwareBaseline,
    'random': RandomBaseline,
    **ALL_NUMERIC_BASELINES,  # Add environment-specific baselines
}


def run_baseline_episode(env: gym.Env, baseline: DDTSBaseline, 
                        max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
    """Run a single episode with a baseline policy."""
    baseline.reset()
    observation, info = env.reset()
    
    total_reward = 0.0
    steps = 0
    actions_taken = []
    rewards = []
    
    done = False
    while not done and steps < max_steps:
        if render:
            env.render()
        
        action = baseline.get_action(observation, info)
        actions_taken.append(action)
        
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        steps += 1
        
        if truncated:
            done = True
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'actions': actions_taken,
        'rewards': rewards,
        'final_info': info
    }


def run_baseline_evaluation(env: gym.Env, baseline: DDTSBaseline, 
                          num_episodes: int = 10) -> Dict[str, Any]:
    """Evaluate a baseline over multiple episodes."""
    episode_rewards = []
    episode_lengths = []
    total_alerts = []
    
    for episode in range(num_episodes):
        result = run_baseline_episode(env, baseline)
        
        episode_rewards.append(result['total_reward'])
        episode_lengths.append(result['steps'])
        total_alerts.append(sum(result['actions']))
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {result['total_reward']:.2f}, "
                  f"Alerts = {sum(result['actions'])}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_alerts': np.mean(total_alerts),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'total_alerts': total_alerts
    }


def create_environment(env_name: str, **kwargs) -> gym.Env:
    """Create and return the specified environment."""
    if env_name.lower() in ['heat', 'heatalert', 'heat_alert']:
        if WEATHER2ALERT_AVAILABLE:
            try:
                return weather2alert.env.HeatAlertEnv(**kwargs)
            except Exception as e:
                print(f"Failed to create real HeatAlertEnv: {e}")
                if MOCK_ENV_AVAILABLE:
                    print("Using mock environment instead")
                    return MockHeatAlertEnv(**kwargs)
                raise
        elif MOCK_ENV_AVAILABLE:
            print("Using mock HeatAlertEnv for testing")
            return MockHeatAlertEnv(**kwargs)
        else:
            raise ImportError("Neither real nor mock HeatAlertEnv available")
    
    elif NUMERIC_ENVS_AVAILABLE:
        # Try to create using numeric environments integration
        env = create_numeric_environment(env_name, **kwargs)
        if env is not None:
            return env
        
        # If not found, show available options
        from numeric_envs import list_available_environments
        available = list_available_environments()
        print(f"Environment '{env_name}' not found.")
        print(f"Available numeric environments: {available}")
        print("To add support for a new environment, see numeric_envs/ folder")
        return None
    
    else:
        # Try to create as standard Gym environment
        try:
            return gym.make(env_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Unknown environment: {env_name}. Error: {e}")


def main():
    """Main function to run DDTS baselines."""
    parser = argparse.ArgumentParser(description="Run DDTS baselines on numeric environments")
    parser.add_argument('--env', type=str, default='heat', 
                       help='Environment to run (heat, uganda, mimic, binpacking)')
    parser.add_argument('--baseline', type=str, default='threshold',
                       choices=list(DDTS_BASELINES.keys()),
                       help='Baseline policy to use')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Running DDTS baseline '{args.baseline}' on environment '{args.env}'")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print("-" * 50)
    
    try:
        # Create environment
        env = create_environment(args.env)
        if env is None:
            print(f"Environment '{args.env}' is not available")
            return
        
        # Create baseline
        baseline_class = DDTS_BASELINES[args.baseline]
        baseline = baseline_class()
        
        # Run evaluation
        results = run_baseline_evaluation(env, baseline, args.episodes)
        
        # Print results
        print(f"\nResults for {args.baseline} baseline:")
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean episode length: {results['mean_length']:.1f}")
        print(f"Mean alerts per episode: {results['mean_alerts']:.1f}")
        
        if args.verbose:
            print(f"\nDetailed results:")
            for i, (reward, alerts) in enumerate(zip(results['episode_rewards'], 
                                                   results['total_alerts'])):
                print(f"Episode {i+1}: Reward = {reward:.2f}, Alerts = {alerts}")
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()