import unittest
import numpy as np
import pandas as pd
from src.models.predictor import AsteroidTrajectoryPredictor


class TestAsteroidTrajectoryPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = AsteroidTrajectoryPredictor()

    def test_initialization(self):
        """Test proper initialization of predictor"""
        self.assertIsNotNone(self.predictor.ml_model)
        self.assertIsNotNone(self.predictor.kf)

    def test_generate_measurements(self):
        """Test measurement generation with noise"""
        true_state = np.array([1000, 0, 0, 100, 0, 0])
        measurement = self.predictor.generate_measurements(true_state)
        self.assertEqual(len(measurement), 3)

    def test_kalman_prediction(self):
        """Test Kalman filter prediction"""
        initial_state = np.array([1000, 0, 0, 100, 0, 0])
        t_span = 3600 * 24  # 1 day

        t, states, uncertainties = self.predictor.predict_with_kalman(initial_state, t_span)

        self.assertTrue(len(t) > 0)
        self.assertEqual(states.shape[1], 6)
        self.assertEqual(uncertainties.shape[1], 6)

    def test_collision_risk(self):
        """Test collision risk assessment"""
        trajectory = np.array([[1e7, 0, 0, 0, 0, 0],
                               [5e7, 0, 0, 0, 0, 0]])
        earth_position = np.zeros(3)
        satellite_positions = np.array([[2e7, 0, 0]])

        risks = self.predictor.assess_collision_risk(
            trajectory, earth_position, satellite_positions)

        self.assertIn('earth_collision_risk', risks)
        self.assertIn('satellite_collision_risks', risks)


if __name__ == '__main__':
    unittest.main()