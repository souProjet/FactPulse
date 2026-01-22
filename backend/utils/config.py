"""
Configuration loader for FactPulse.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to backend/config.yaml
        
    Returns:
        Configuration dictionary
    """
    global _config
    
    if config_path is None:
        # Default path relative to this file
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    
    return _config


def get_config() -> Dict[str, Any]:
    """Get loaded configuration, loading it if necessary."""
    global _config
    if _config is None:
        load_config()
    return _config


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    config = get_config()
    return config.get("models", {}).get(model_name, {})


def get_threshold(threshold_name: str, sub_key: str) -> float:
    """Get a specific threshold value."""
    config = get_config()
    return config.get("thresholds", {}).get(threshold_name, {}).get(sub_key, 0.5)


def get_data_path(path_key: str) -> Path:
    """Get a data file path from config."""
    config = get_config()
    relative_path = config.get("data", {}).get(path_key, "")
    # Resolve relative to project root
    project_root = Path(__file__).parent.parent.parent
    return project_root / relative_path
