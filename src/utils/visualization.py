import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


def create_3d_visualization(trajectory, satellites, uncertainties=None):
    """Create interactive 3D visualization using plotly"""
    fig = go.Figure()

    # Add asteroid trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0] / 1e6,
        y=trajectory[:, 1] / 1e6,
        z=trajectory[:, 2] / 1e6,
        mode='lines',
        name='Asteroid Trajectory',
        line=dict(color='blue', width=2)
    ))

    # Add uncertainty cloud
    if uncertainties is not None:
        for i in range(0, len(trajectory), 10):
            fig.add_trace(go.Scatter3d(
                x=[trajectory[i, 0] / 1e6],
                y=[trajectory[i, 1] / 1e6],
                z=[trajectory[i, 2] / 1e6],
                mode='markers',
                marker=dict(
                    size=uncertainties[i, :3].mean() / 1e5,
                    color='blue',
                    opacity=0.1
                ),
                showlegend=False
            ))

    # Add Earth
    earth_radius = 6371  # km
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        name='Earth',
        colorscale='Viridis',
        showscale=False
    ))

    # Add satellites
    for name, sat_data in satellites.items():
        pos = sat_data['position']
        orbit_colors = {
            'GEO': 'red',
            'LEO': 'green',
            'MEO': 'yellow'
        }
        fig.add_trace(go.Scatter3d(
            x=[pos[0] / 1e6],
            y=[pos[1] / 1e6],
            z=[pos[2] / 1e6],
            mode='markers+text',
            name=name,
            text=[name],
            marker=dict(
                size=10,
                symbol='diamond',
                color=orbit_colors.get(sat_data['orbit_type'], 'white')
            )
        ))

    fig.update_layout(
        title='Asteroid Trajectory with Satellites',
        scene=dict(
            xaxis_title='X (1000 km)',
            yaxis_title='Y (1000 km)',
            zaxis_title='Z (1000 km)',
            aspectmode='data'
        ),
        width=800,
        height=800
    )

    return fig


def create_distance_plot(t, trajectory, earth_position, satellites):
    """Create distance vs time plot"""
    fig = go.Figure()

    # Earth distance
    earth_distances = np.sqrt(np.sum((trajectory[:, :3] - earth_position) ** 2, axis=1)) / 1e3
    fig.add_trace(go.Scatter(
        x=[datetime.now() + timedelta(seconds=int(ts)) for ts in t],  # Convert ts to int
        y=earth_distances,
        name='Distance to Earth',
        mode='lines'
    ))

    # Satellite distances
    for name, sat_data in satellites.items():
        distances = np.sqrt(np.sum((trajectory[:, :3] - sat_data['position']) ** 2, axis=1)) / 1e3
        fig.add_trace(go.Scatter(
            x=[datetime.now() + timedelta(seconds=int(ts)) for ts in t],  # Convert ts to int
            y=distances,
            name=f'Distance to {name}',
            mode='lines'
        ))

    fig.update_layout(
        title='Distance vs Time',
        xaxis_title='Time',
        yaxis_title='Distance (km)',
        yaxis_type='log'
    )

    return fig
