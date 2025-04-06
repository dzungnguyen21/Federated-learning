import os
import yaml
import numpy as np

class Path:
    """
    Class to load configuration settings from a YAML file.
    """
    def __init__(self):
        # Get the root directory of the project
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_path = os.path.join(self.root_dir,'config.yaml')
        # Load configuration
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    