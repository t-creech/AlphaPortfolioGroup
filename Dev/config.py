# config.py
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

class Config:
    """Configuration class for AlphaPortfolio."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config from YAML file.
        
        Args:
            config_path: Path to config YAML file
        """
        logging.info(f"Loading configuration from {config_path}")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set experiment ID if not provided
        if not self.config.get("experiment_id"):
            self.config["experiment_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self._create_directories()
        
        logging.info(f"Configuration loaded with experiment ID: {self.config['experiment_id']}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config["paths"]["output_dir"],
            self.config["paths"]["model_dir"],
            self.config["paths"]["log_dir"],
            self.config["paths"]["plot_dir"]
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
    
    def get_all_cycles(self) -> List[Dict[str, Any]]:
        """Get parameters for all training cycles."""
        return self.config.get("cycles", [])
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for search."""
        return self.config.get("hyperparameters", {})