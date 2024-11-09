import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from src.models.satellite import SatelliteManager


class TestSatelliteManager(unittest.TestCase):
    def setUp(self):
        self.manager = SatelliteManager()

    def test_add_satellite(self):
        """Test adding a satellite"""
        self.manager.add_satellite(
            "TEST-SAT",
            [42164000, 0, 0],
            "GEO",
            {"inclination": 0}
        )

        self.assertIn("TEST-SAT", self.manager.satellites)

    def test_remove_satellite(self):
        """Test removing a satellite"""
        self.manager.add_satellite("TEST-SAT", [0, 0, 0], "LEO")
        self.manager.remove_satellite("TEST-SAT")

        self.assertNotIn("TEST-SAT", self.manager.satellites)

    def test_get_all_positions(self):
        """Test getting all satellite positions"""
        self.manager.add_satellite("SAT1", [1, 0, 0], "LEO")
        self.manager.add_satellite("SAT2", [0, 1, 0], "GEO")

        positions = self.manager.get_all_positions()
        self.assertEqual(len(positions), 2)

    def test_save_load_config(self):
        """Test saving and loading configuration"""
        self.manager.add_satellite("SAT1", [1, 0, 0], "LEO")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.manager.save_config(tmp.name)

            new_manager = SatelliteManager()
            new_manager.load_config(tmp.name)

            self.assertEqual(
                len(self.manager.satellites),
                len(new_manager.satellites)
            )

        Path(tmp.name).unlink()


if __name__ == '__main__':
    unittest.main()