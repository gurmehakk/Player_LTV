"""
Configuration management for pLTV prediction system
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    """Configuration class for pLTV prediction system"""
    
    # BigQuery Configuration
    project_id: str = "gc-forecasting-dev"
    dataset_id: str = "test_data"
    table_id: str = "app_data"
    credentials_path: Optional[str] = None
    
    # Data Parameters
    days_lookback: int = 90
    prediction_window: int = 30
    min_sessions_threshold: int = 1
    
    # Model Parameters
    test_size: float = 0.2
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 6
    random_state: int = 42
    
    # Segmentation Parameters
    n_clusters: int = 5
    
    # Output Configuration
    output_dir: str = "output"
    save_plots: bool = True
    save_enhanced_data: bool = True
    
    # Feature Engineering Parameters
    revenue_features: bool = True
    behavioral_features: bool = True
    engagement_features: bool = True
    attribution_features: bool = True
    session_features: bool = True
    progression_features: bool = True
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """Save current configuration to JSON file"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    @property
    def config(self):
        """Return configuration as dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @property
    def full_table_id(self):
        """Get full BigQuery table ID"""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"