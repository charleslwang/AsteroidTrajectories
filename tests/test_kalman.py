import unittest
import numpy as np
from src.models.kalman import KalmanFilter


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

    def test_initialization(self):
        """Test proper initialization of Kalman filter"""
        self.assertEqual(self.kf.dim_x, 6)
        self.assertEqual(self.kf.dim_z, 3)
        self.assertTrue(np.array_equal(self.kf.x, np.zeros(6)))
        self.assertTrue(np.array_equal(self.kf.F, np.eye(6)))

    def test_prediction(self):
        """Test state prediction"""
        initial_state = np.array([1000, 0, 0, 100, 0, 0])
        self.kf.x = initial_state
        dt = 1.0

        predicted_state, _ = self.kf.predict(dt)

        # Check if position is updated by velocity
        self.assertEqual(predicted_state[0], initial_state[0] + initial_state[3] * dt)

    def test_update(self):
        """Test measurement update"""
        self.kf.x = np.array([1000, 0, 0, 100, 0, 0])
        self.kf.H[0:3, 0:3] = np.eye(3)

        measurement = np.array([1100, 50, 25])
        updated_state, _ = self.kf.update(measurement)

        # Check if state is pulled toward measurement
        self.assertTrue(np.all(updated_state[:3] >= self.kf.x[:3]))


if __name__ == '__main__':
    unittest.main()