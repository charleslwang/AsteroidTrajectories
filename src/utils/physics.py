import numpy as np
from scipy.constants import G


def orbital_equations(state, t, mu=G * 5.972e24):  # mu for Earth
    """
    Implements the equations of motion for an object in orbit.

    Args:
        state (array): 6D state vector [x, y, z, vx, vy, vz]
        t (float): Time (unused, but required for scipy.integrate.odeint)
        mu (float): Standard gravitational parameter (GM) of the central body
                   Defaults to Earth's GM

    Returns:
        array: State derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    # Unpack the state vector
    x, y, z, vx, vy, vz = state

    # Calculate radius
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Check for division by zero
    if r < 1e-10:
        r = 1e-10

    # Gravitational acceleration components
    ax = -mu * x / r ** 3
    ay = -mu * y / r ** 3
    az = -mu * z / r ** 3

    # Optional: Add perturbation forces here
    # Examples could include:
    # - Solar radiation pressure
    # - Third body effects (Moon, Sun)
    # - Non-spherical Earth (J2 effect)

    return np.array([vx, vy, vz, ax, ay, az])


def calculate_orbital_elements(state):
    """
    Calculate orbital elements from state vector.

    Args:
        state (array): 6D state vector [x, y, z, vx, vy, vz]

    Returns:
        dict: Orbital elements (a, e, i, Ω, ω, ν)
    """
    mu = G * 5.972e24  # Earth's gravitational parameter

    # Position and velocity vectors
    r = state[:3]
    v = state[3:]

    # Magnitude of position and velocity
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Angular momentum vector
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector (points toward ascending node)
    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e = ((v_mag ** 2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    ecc = np.linalg.norm(e)

    # Specific energy
    energy = v_mag ** 2 / 2 - mu / r_mag

    # Semi-major axis
    sma = -mu / (2 * energy) if abs(ecc - 1.0) > 1e-10 else float('inf')

    # Inclination
    inc = np.arccos(h[2] / h_mag)

    # Right ascension of ascending node
    raan = np.arccos(n[0] / n_mag) if n[1] >= 0 else 2 * np.pi - np.arccos(n[0] / n_mag)

    # Argument of periapsis
    argp = np.arccos(np.dot(n, e) / (n_mag * ecc)) if e[2] >= 0 else \
        2 * np.pi - np.arccos(np.dot(n, e) / (n_mag * ecc))

    # True anomaly
    true_anom = np.arccos(np.dot(e, r) / (ecc * r_mag)) if np.dot(r, v) >= 0 else \
        2 * np.pi - np.arccos(np.dot(e, r) / (ecc * r_mag))

    return {
        'semi_major_axis': sma,
        'eccentricity': ecc,
        'inclination': np.degrees(inc),
        'raan': np.degrees(raan),
        'arg_periapsis': np.degrees(argp),
        'true_anomaly': np.degrees(true_anom)
    }


def state_to_elements(state):
    """
    Convert state vector to orbital elements suitable for ML model input.

    Args:
        state (array): 6D state vector [x, y, z, vx, vy, vz]

    Returns:
        array: Flattened array of relevant orbital parameters
    """
    elements = calculate_orbital_elements(state)

    return np.array([
        elements['semi_major_axis'],
        elements['eccentricity'],
        elements['inclination'],
        elements['raan'],
        elements['arg_periapsis'],
        elements['true_anomaly'],
        np.linalg.norm(state[:3]),  # Current radius
        np.linalg.norm(state[3:])  # Current velocity
    ])


def elements_to_state(elements):
    """
    Convert orbital elements to state vector.

    Args:
        elements (array): Orbital elements [a, e, i, Ω, ω, ν, r, v]

    Returns:
        array: 6D state vector [x, y, z, vx, vy, vz]
    """
    mu = G * 5.972e24

    # Unpack elements
    sma, ecc, inc, raan, argp, true_anom, _, _ = elements

    # Convert angles to radians
    inc = np.radians(inc)
    raan = np.radians(raan)
    argp = np.radians(argp)
    true_anom = np.radians(true_anom)

    # Calculate radius
    p = sma * (1 - ecc ** 2)
    r = p / (1 + ecc * np.cos(true_anom))

    # Position in orbital plane
    x_orbit = r * np.cos(true_anom)
    y_orbit = r * np.sin(true_anom)

    # Velocity in orbital plane
    h = np.sqrt(mu * p)
    v_radial = (h * ecc / p) * np.sin(true_anom)
    v_transverse = h / r

    vx_orbit = v_radial * np.cos(true_anom) - v_transverse * np.sin(true_anom)
    vy_orbit = v_radial * np.sin(true_anom) + v_transverse * np.cos(true_anom)

    # Rotation matrices
    R_w = np.array([[np.cos(argp), -np.sin(argp), 0],
                    [np.sin(argp), np.cos(argp), 0],
                    [0, 0, 1]])

    R_i = np.array([[1, 0, 0],
                    [0, np.cos(inc), -np.sin(inc)],
                    [0, np.sin(inc), np.cos(inc)]])

    R_W = np.array([[np.cos(raan), -np.sin(raan), 0],
                    [np.sin(raan), np.cos(raan), 0],
                    [0, 0, 1]])

    # Transform to inertial frame
    R = R_W @ R_i @ R_w

    r_vec = R @ np.array([x_orbit, y_orbit, 0])
    v_vec = R @ np.array([vx_orbit, vy_orbit, 0])

    return np.concatenate([r_vec, v_vec])