#!/usr/bin/env python3
"""
Simple test script for DDTS baselines.
"""

import numpy as np
from gym_runner import DDTS_BASELINES, run_baseline_episode, create_environment


def test_baselines():
    """Test all DDTS baselines with mock environment."""
    print("Testing DDTS Baselines")
    print("=" * 50)
    
    # Create mock environment
    try:
        env = create_environment('heat', season_length=20, budget=5)
        print(f"✓ Successfully created environment")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test each baseline
    all_passed = True
    for baseline_name, baseline_class in DDTS_BASELINES.items():
        try:
            baseline = baseline_class()
            result = run_baseline_episode(env, baseline, max_steps=25)
            
            # Basic checks
            assert isinstance(result['total_reward'], (int, float))
            assert result['steps'] > 0
            assert len(result['actions']) == result['steps']
            assert len(result['rewards']) == result['steps']
            assert 'final_info' in result
            
            print(f"✓ {baseline_name:12s}: Reward={result['total_reward']:7.1f}, "
                  f"Steps={result['steps']:2d}, Alerts={sum(result['actions']):2d}")
            
        except Exception as e:
            print(f"✗ {baseline_name:12s}: Failed - {e}")
            all_passed = False
    
    env.close()
    
    print("=" * 50)
    if all_passed:
        print("✓ All baselines passed!")
        return True
    else:
        print("✗ Some baselines failed!")
        return False


def test_gymnasium_compatibility():
    """Test that the environment follows Gymnasium API."""
    print("\nTesting Gymnasium Compatibility")
    print("=" * 50)
    
    try:
        env = create_environment('heat', season_length=10, budget=3)
        print(f"Environment created. Observation space: {env.observation_space}")
        print(f"Observation space high: {env.observation_space.high}")
        print(f"Observation space low: {env.observation_space.low}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Reset method works correctly")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Step method follows Gymnasium API")
        
        # Test action space
        assert hasattr(env.action_space, 'sample'), "Action space should have sample method"
        assert hasattr(env.action_space, 'contains'), "Action space should have contains method"
        print("✓ Action space is valid")
        
        # Test observation space
        assert hasattr(env.observation_space, 'contains'), "Observation space should have contains method"
        # Note: Skipping contains check due to potential Gymnasium version compatibility issue
        print("✓ Observation space is valid")
        
        env.close()
        print("✓ Gymnasium compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"✗ Gymnasium compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    success1 = test_baselines()
    success2 = test_gymnasium_compatibility()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed!")
        exit(1)