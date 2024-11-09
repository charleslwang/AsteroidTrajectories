import numpy as np
import json
from pathlib import Path


class SatelliteManager:
    def __init__(self, config_path=None):
        self.satellites = {}
        if config_path:
            self.load_config(config_path)

    def add_satellite(self, name, position, orbit_type, parameters=None):
        """Add a satellite with position and orbital parameters"""
        self.satellites[name] = {
            'position': np.array(position),
            'orbit_type': orbit_type,
            'parameters': parameters or {}
        }

    def remove_satellite(self, name):
        """Remove a satellite by name"""
        if name in self.satellites:
            del self.satellites[name]

    def get_all_positions(self):
        """Get positions of all satellites"""
        return np.array([sat['position'] for sat in self.satellites.values()])

    def save_config(self, filepath):
        """Save satellite configuration to JSON"""
        config = {name: {
            'position': sat['position'].tolist(),
            'orbit_type': sat['orbit_type'],
            'parameters': sat['parameters']
        } for name, sat in self.satellites.items()}

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config(self, filepath):
        """Load satellite configuration from JSON"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        for name, data in config.items():
            self.add_satellite(
                name,
                np.array(data['position']),
                data['orbit_type'],
                data['parameters']
            )