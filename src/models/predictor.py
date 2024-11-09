import numpy as np
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .kalman import KalmanFilter
from ..utils.physics import orbital_equations


class AsteroidTrajectoryPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Initialize Kalman filter for 6D state (position and velocity)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.setup_kalman_filter()

    def setup_kalman_filter(self):
        """Configure Kalman filter parameters"""
        # Measurement matrix (we only measure position)
        self.kf.H[0:3, 0:3] = np.eye(3)

        # Process noise
        q = 1e-6  # Small process noise for space environment
        self.kf.Q = np.eye(6) * q

        # Measurement noise
        r = 1000  # 1km uncertainty in position measurements
        self.kf.R = np.eye(3) * r

    def generate_measurements(self, true_state, noise_std=1000):
        """Generate noisy position measurements"""
        position = true_state[:3]
        noise = np.random.normal(0, noise_std, 3)
        return position + noise

    def predict_with_kalman(self, initial_state, t_span, dt=3600):
        """Predict trajectory using Kalman filter"""
        self.kf.x = initial_state
        t = np.arange(0, t_span, dt)
        n_steps = len(t)

        states = np.zeros((n_steps, 6))
        uncertainties = np.zeros((n_steps, 6))

        for i in range(n_steps):
            state_pred, cov_pred = self.kf.predict(dt)
            true_state = odeint(orbital_equations, state_pred, [0, dt])[-1]
            measurement = self.generate_measurements(true_state)
            state_updated, cov_updated = self.kf.update(measurement)

            states[i] = state_updated
            uncertainties[i] = np.sqrt(np.diag(cov_updated))

        return t, states, uncertainties

    def train_ml_model(self, orbital_elements_df, historic_positions):
        """Train the ML model on historical data"""
        features = self.scaler.fit_transform(orbital_elements_df)
        self.ml_model.fit(features, historic_positions)
        return self.ml_model.score(features, historic_positions)

    def hybrid_prediction(self, initial_state, orbital_elements, t_span, dt=3600):
        """Combine Kalman filter, physics-based, and ML predictions"""
        # Get Kalman filter prediction
        t, kf_states, kf_uncertainties = self.predict_with_kalman(initial_state, t_span, dt)

        # Get physics-based prediction
        t_points = np.arange(0, t_span, dt)
        physics_pred = np.array([
            odeint(orbital_equations, initial_state, [0, t])[-1]
            for t in t_points
        ])

        # ML prediction
        ml_features = self.scaler.transform(orbital_elements.reshape(1, -1))
        ml_pred = np.tile(self.ml_model.predict(ml_features), (len(t_points), 1))

        # Weighted combination
        weights = {'kalman': 0.5, 'physics': 0.3, 'ml': 0.2}

        hybrid_prediction = (
                weights['kalman'] * kf_states +
                weights['physics'] * physics_pred +
                weights['ml'] * ml_pred
        )

        return t, hybrid_prediction, kf_uncertainties

    def assess_collision_risk(self, trajectory, earth_position, satellite_positions):
        """Calculate collision probabilities with Earth and satellites"""
        earth_threshold = 50000000  # 50,000 km
        satellite_threshold = 1000  # 1 km

        earth_distances = np.sqrt(np.sum((trajectory[:, :3] - earth_position) ** 2, axis=1))
        earth_risk = np.min(earth_distances) < earth_threshold

        satellite_risks = []
        min_distances = []
        for sat_pos in satellite_positions:
            distances = np.sqrt(np.sum((trajectory[:, :3] - sat_pos) ** 2, axis=1))
            satellite_risks.append(np.min(distances) < satellite_threshold)
            min_distances.append(np.min(distances))

        return {
            'earth_collision_risk': earth_risk,
            'min_earth_distance': np.min(earth_distances),
            'satellite_collision_risks': satellite_risks,
            'min_satellite_distances': min_distances
        }