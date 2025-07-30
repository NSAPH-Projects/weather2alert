"""
Numeric Environments Integration for DDTS Baselines

This package provides integration support for various numeric environments
that can be used with the DDTS baseline runner system.

Available integrations:
- Heat Alerts (implemented in main package)
- Uganda environment (placeholder)
- MIMIC environment (placeholder)
- Bin Packing environment (placeholder)

Each integration module provides:
1. Environment creation functions
2. Environment-specific DDTS baselines
3. Integration with gym_runner.py

To add a new numeric environment:
1. Create a new integration module following the template
2. Implement the create_*_env() function
3. Add environment-specific DDTS baselines
4. Update gym_runner.py to recognize the new environment
"""

from typing import Dict, Any, Optional
import gymnasium as gym

# Import integration modules
try:
    from .uganda_integration import create_uganda_env, UGANDA_BASELINES
    UGANDA_AVAILABLE = True
except ImportError:
    UGANDA_AVAILABLE = False
    UGANDA_BASELINES = {}

try:
    from .mimic_integration import create_mimic_env, MIMIC_BASELINES
    MIMIC_AVAILABLE = True
except ImportError:
    MIMIC_AVAILABLE = False
    MIMIC_BASELINES = {}

try:
    from .binpacking_integration import create_binpacking_env, BINPACKING_BASELINES
    BINPACKING_AVAILABLE = True
except ImportError:
    BINPACKING_AVAILABLE = False
    BINPACKING_BASELINES = {}


# Consolidated baseline registry
ALL_NUMERIC_BASELINES = {
    **UGANDA_BASELINES,
    **MIMIC_BASELINES,
    **BINPACKING_BASELINES,
}


def create_numeric_environment(env_name: str, **kwargs) -> Optional[gym.Env]:
    """
    Create a numeric environment by name.
    
    Args:
        env_name: Name of the environment to create
        **kwargs: Environment-specific parameters
        
    Returns:
        Environment instance or None if not available
    """
    env_name_lower = env_name.lower()
    
    if env_name_lower in ['uganda', 'uganda_env'] and UGANDA_AVAILABLE:
        return create_uganda_env(**kwargs)
    
    elif env_name_lower in ['mimic', 'mimic_env'] and MIMIC_AVAILABLE:
        return create_mimic_env(**kwargs)
    
    elif env_name_lower in ['binpacking', 'bin_packing'] and BINPACKING_AVAILABLE:
        return create_binpacking_env(**kwargs)
    
    else:
        return None


def list_available_environments():
    """List all available numeric environments."""
    available = []
    
    if UGANDA_AVAILABLE:
        available.append("uganda")
    if MIMIC_AVAILABLE:
        available.append("mimic")
    if BINPACKING_AVAILABLE:
        available.append("binpacking")
    
    return available


def get_environment_baselines(env_name: str) -> Dict[str, Any]:
    """
    Get baselines specific to an environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Dictionary of baseline name -> baseline class mappings
    """
    env_name_lower = env_name.lower()
    
    if env_name_lower in ['uganda', 'uganda_env']:
        return UGANDA_BASELINES
    elif env_name_lower in ['mimic', 'mimic_env']:
        return MIMIC_BASELINES
    elif env_name_lower in ['binpacking', 'bin_packing']:
        return BINPACKING_BASELINES
    else:
        return {}


if __name__ == "__main__":
    print("Numeric Environments Integration")
    print("=" * 40)
    
    available_envs = list_available_environments()
    print(f"Available environments: {available_envs}")
    
    for env_name in available_envs:
        baselines = get_environment_baselines(env_name)
        print(f"\n{env_name.upper()} baselines: {list(baselines.keys())}")
    
    print(f"\nAll numeric baselines: {list(ALL_NUMERIC_BASELINES.keys())}")