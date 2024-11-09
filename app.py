import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

from src.models.predictor import AsteroidTrajectoryPredictor
from src.models.satellite import SatelliteManager
from src.utils.visualization import create_3d_visualization, create_distance_plot

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = AsteroidTrajectoryPredictor()
if 'satellite_manager' not in st.session_state:
    st.session_state.satellite_manager = SatelliteManager()


def display_current_satellites():
    """Displays a list of currently added satellites"""
    st.subheader("Current Satellites")
    if st.session_state.satellite_manager.satellites:
        for name, data in st.session_state.satellite_manager.satellites.items():
            position = data['position']
            orbit_type = data['orbit_type']
            st.write(
                f"- **{name}**: Position (km) = [{position[0] / 1000:.2f}, {position[1] / 1000:.2f}, {position[2] / 1000:.2f}], Orbit Type = {orbit_type}")
    else:
        st.write("No satellites added.")


def main():
    st.title("Asteroid Trajectory Prediction System")

    # Display current satellites
    display_current_satellites()

    # Option to clear all satellites
    if st.button("Clear All Satellites"):
        st.session_state.satellite_manager.satellites.clear()
        st.success("All satellites cleared.")
        st.experimental_rerun()

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")

    # Initial state input
    st.sidebar.subheader("Initial State Vector")
    x = st.sidebar.number_input("X Position (km)", value=384400.0) * 1000  # Convert to meters
    y = st.sidebar.number_input("Y Position (km)", value=0.0) * 1000
    z = st.sidebar.number_input("Z Position (km)", value=0.0) * 1000
    vx = st.sidebar.number_input("X Velocity (km/s)", value=0.0) * 1000
    vy = st.sidebar.number_input("Y Velocity (km/s)", value=1.0) * 1000
    vz = st.sidebar.number_input("Z Velocity (km/s)", value=0.0) * 1000

    initial_state = np.array([x, y, z, vx, vy, vz])

    # Prediction parameters
    st.sidebar.subheader("Prediction Parameters")
    prediction_days = st.sidebar.slider("Prediction Period (days)", 1, 30, 7)
    t_span = prediction_days * 24 * 3600  # Convert to seconds

    # Satellite management
    st.sidebar.subheader("Satellite Management")

    # Add custom satellite form
    with st.sidebar.form("add_satellite_form"):
        st.write("Add a Custom Satellite")
        name = st.text_input("Satellite Name", "TestSat")
        x_pos = st.number_input("X Position (km)", value=10000) * 1000  # Convert to meters
        y_pos = st.number_input("Y Position (km)", value=20000) * 1000
        z_pos = st.number_input("Z Position (km)", value=30000) * 1000
        orbit_type = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO", "Other"])

        if st.form_submit_button("Add Satellite"):
            st.session_state.satellite_manager.add_satellite(
                name,
                np.array([x_pos, y_pos, z_pos]),
                orbit_type
            )
            st.success(f"Satellite '{name}' added!")
            st.experimental_rerun()  # Refresh to show updated satellite list

    # Main content
    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            # Get predictions
            t, states, uncertainties = st.session_state.predictor.predict_with_kalman(
                initial_state, t_span)

            # Create visualizations
            fig_3d = create_3d_visualization(
                states,
                st.session_state.satellite_manager.satellites,
                uncertainties
            )

            fig_distances = create_distance_plot(
                t,
                states,
                np.zeros(3),  # Earth at origin
                st.session_state.satellite_manager.satellites
            )

            # Display visualizations
            st.plotly_chart(fig_3d, use_container_width=True)
            st.plotly_chart(fig_distances, use_container_width=True)

            # Calculate collision risks
            risks = st.session_state.predictor.assess_collision_risk(
                states,
                np.zeros(3),
                st.session_state.satellite_manager.get_all_positions()
            )

            # Display risk assessment
            st.subheader("Risk Assessment")
            st.write(f"Earth Collision Risk: {'High' if risks['earth_collision_risk'] else 'Low'}")
            st.write(f"Minimum Distance to Earth: {risks['min_earth_distance'] / 1000:.0f} km")

            if len(risks['satellite_collision_risks']) > 0:
                st.write("Satellite Collision Risks:")
                for i, (risk, dist) in enumerate(zip(
                        risks['satellite_collision_risks'],
                        risks['min_satellite_distances']
                )):
                    sat_name = list(st.session_state.satellite_manager.satellites.keys())[i]
                    st.write(f"- {sat_name}: {'High' if risk else 'Low'} (Min. Distance: {dist / 1000:.0f} km)")


if __name__ == "__main__":
    main()
