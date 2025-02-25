# Asteroid Trajectory Prediction System

## Overview

The Asteroid Trajectory Prediction System is a sophisticated application designed to predict and visualize the trajectory of asteroids in Earth's vicinity, assess potential collision risks with Earth and satellites, and provide detailed analysis of orbital dynamics. This tool combines physics-based modeling, Kalman filtering, and machine learning to deliver accurate trajectory predictions with uncertainty quantification.

## Features

- **Advanced Trajectory Prediction**: Combines Kalman filtering, physics-based orbital mechanics, and machine learning for robust predictions
- **Interactive 3D Visualization**: View asteroid trajectories and satellite positions in an interactive 3D environment
- **Satellite Management**: Add, remove, and configure satellites with custom positions and orbit types
- **Collision Risk Assessment**: Calculate and display collision probabilities with Earth and satellites
- **Distance Analysis**: Plot distance vs. time for Earth and all tracked satellites

## System Components

- **AsteroidTrajectoryPredictor**: Core prediction engine combining multiple prediction methods
- **KalmanFilter**: Implementation of the Kalman filter algorithm for state estimation
- **SatelliteManager**: Handles satellite data and configuration
- **Orbital Equations**: Physics-based orbital mechanics calculations
- **Streamlit Interface**: Interactive web application for user interaction

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/asteroid-trajectory-predictor.git
   cd asteroid-trajectory-predictor
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Configure initial state parameters in the sidebar:
   - Initial position (X, Y, Z in km)
   - Initial velocity (VX, VY, VZ in km/s)
   - Prediction period (days)

3. Add satellites to track:
   - Enter satellite name
   - Set position (X, Y, Z in km)
   - Select orbit type (LEO, MEO, GEO, or Other)

4. Click "Run Prediction" to generate trajectory predictions and visualizations

## Prediction Methods

The system uses three complementary prediction methods:

1. **Kalman Filter**: For state estimation with measurement noise handling
2. **Physics-based Prediction**: Implements orbital equations for accurate trajectory modeling
3. **Machine Learning**: Uses Random Forest to predict based on learned patterns from historical data

## Visualization Features

- **3D Trajectory Plot**: Interactive 3D visualization of asteroid path and satellites
- **Uncertainty Cloud**: Visual representation of prediction uncertainty
- **Earth Representation**: 3D model of Earth for reference
- **Distance Plot**: Time-series plot of distances between the asteroid and Earth/satellites

## File Structure

```
asteroid-trajectory-predictor/
├── app.py                 # Main Streamlit application
├── src/
│   ├── models/
│   │   ├── predictor.py   # AsteroidTrajectoryPredictor class
│   │   ├── kalman.py      # KalmanFilter implementation
│   │   └── satellite.py   # SatelliteManager class
│   └── utils/
│       ├── physics.py     # Orbital mechanics equations
│       └── visualization.py # Visualization functions
├── data/                  # Sample data and configurations
└── requirements.txt       # Project dependencies
```

## Technical Details

- **Kalman Filter**: 6D state vector (position and velocity) with position measurements
- **Orbital Mechanics**: Implements standard gravitational equations with perturbation capabilities
- **Machine Learning**: Random Forest regressor trained on historical data
- **Risk Assessment**: Proximity-based collision risk calculations

## Requirements

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- Plotly
- Streamlit
- Pandas

## Future Development

- Integration with asteroid databases (JPL, NASA)
- Improved perturbation models for higher accuracy
- Cloud deployment for public access
- Time-series analysis of close approaches
- Enhanced visualization options
