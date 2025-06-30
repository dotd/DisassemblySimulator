"""
Utility Functions for MuJoCo Simulations

This module provides utility functions for:
- Mathematical operations
- Data processing and analysis
- Visualization helpers
- File I/O operations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import json
import pickle
from pathlib import Path
import time


# Mathematical Utilities
def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(rotation_matrix)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
        
    return np.array([w, x, y, z])


def euler_to_quaternion(euler_angles: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler_angles: Euler angles [x, y, z] in radians
        order: Rotation order ('xyz', 'zyx', etc.)
        
    Returns:
        Quaternion [w, x, y, z]
    """
    x, y, z = euler_angles / 2
    
    if order == 'xyz':
        qx = np.array([np.cos(x), np.sin(x), 0, 0])
        qy = np.array([np.cos(y), 0, np.sin(y), 0])
        qz = np.array([np.cos(z), 0, 0, np.sin(z)])
    elif order == 'zyx':
        qx = np.array([np.cos(x), np.sin(x), 0, 0])
        qy = np.array([np.cos(y), 0, np.sin(y), 0])
        qz = np.array([np.cos(z), 0, 0, np.sin(z)])
    else:
        raise ValueError(f"Unsupported rotation order: {order}")
        
    # Quaternion multiplication
    q = quaternion_multiply(quaternion_multiply(qz, qy), qx)
    return q


def quaternion_to_euler(quaternion: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """
    Convert quaternion to Euler angles.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        order: Rotation order ('xyz', 'zyx', etc.)
        
    Returns:
        Euler angles [x, y, z] in radians
    """
    w, x, y, z = quaternion
    
    if order == 'xyz':
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    elif order == 'zyx':
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    else:
        raise ValueError(f"Unsupported rotation order: {order}")
        
    return np.array([roll, pitch, yaw])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion [w, x, y, z]
    """
    norm = np.linalg.norm(quaternion)
    if norm > 0:
        return quaternion / norm
    else:
        return np.array([1, 0, 0, 0])


def interpolate_quaternions(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated quaternion [w, x, y, z]
    """
    # Ensure shortest path
    if np.dot(q1, q2) < 0:
        q2 = -q2
        
    # Spherical linear interpolation
    dot = np.clip(np.dot(q1, q2), -1, 1)
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        return q1
    
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


# Trajectory Generation
def generate_circular_trajectory(center: np.ndarray, radius: float, 
                                num_points: int = 100) -> np.ndarray:
    """
    Generate a circular trajectory.
    
    Args:
        center: Center point [x, y, z]
        radius: Circle radius
        num_points: Number of points in trajectory
        
    Returns:
        Array of positions [num_points, 3]
    """
    angles = np.linspace(0, 2*np.pi, num_points)
    trajectory = []
    
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        trajectory.append([x, y, z])
        
    return np.array(trajectory)


def generate_line_trajectory(start: np.ndarray, end: np.ndarray, 
                           num_points: int = 100) -> np.ndarray:
    """
    Generate a linear trajectory.
    
    Args:
        start: Start point [x, y, z]
        end: End point [x, y, z]
        num_points: Number of points in trajectory
        
    Returns:
        Array of positions [num_points, 3]
    """
    t = np.linspace(0, 1, num_points)
    trajectory = []
    
    for ti in t:
        pos = start + ti * (end - start)
        trajectory.append(pos)
        
    return np.array(trajectory)


def generate_smooth_trajectory(waypoints: List[np.ndarray], 
                             num_points_per_segment: int = 50) -> np.ndarray:
    """
    Generate a smooth trajectory through waypoints using cubic splines.
    
    Args:
        waypoints: List of waypoints [x, y, z]
        num_points_per_segment: Number of points per segment
        
    Returns:
        Array of positions
    """
    if len(waypoints) < 2:
        return np.array(waypoints)
        
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        # Generate intermediate points
        t = np.linspace(0, 1, num_points_per_segment)
        segment = []
        
        for ti in t:
            # Cubic interpolation
            pos = (1 - ti)**3 * start + 3*(1 - ti)**2 * ti * start + \
                  3*(1 - ti) * ti**2 * end + ti**3 * end
            segment.append(pos)
            
        trajectory.extend(segment)
        
    return np.array(trajectory)


# Data Processing
def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth data using moving average.
    
    Args:
        data: Input data array
        window_size: Size of smoothing window
        
    Returns:
        Smoothed data array
    """
    if window_size < 2:
        return data
        
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


def compute_derivative(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute numerical derivative of data.
    
    Args:
        data: Input data array
        dt: Time step
        
    Returns:
        Derivative array
    """
    derivative = np.zeros_like(data)
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    derivative[0] = (data[1] - data[0]) / dt
    derivative[-1] = (data[-1] - data[-2]) / dt
    
    return derivative


def compute_integral(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute numerical integral of data.
    
    Args:
        data: Input data array
        dt: Time step
        
    Returns:
        Integral array
    """
    return np.cumsum(data) * dt


# Visualization
def plot_trajectory(trajectory: np.ndarray, title: str = "Trajectory", 
                   labels: List[str] = None):
    """
    Plot a 3D trajectory.
    
    Args:
        trajectory: Array of positions [n_points, 3]
        title: Plot title
        labels: Axis labels
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               c='g', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
               c='r', s=100, label='End')
    
    if labels is None:
        labels = ['X', 'Y', 'Z']
        
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.legend()
    
    plt.show()


def plot_joint_data(time_data: np.ndarray, joint_data: np.ndarray, 
                   joint_names: List[str] = None, title: str = "Joint Data"):
    """
    Plot joint data over time.
    
    Args:
        time_data: Time array
        joint_data: Joint data array [n_joints, n_time_steps]
        joint_names: Names of joints
        title: Plot title
    """
    n_joints = joint_data.shape[0]
    
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(n_joints)]
        
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints))
    if n_joints == 1:
        axes = [axes]
        
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(time_data, joint_data[i], 'b-', linewidth=2)
        ax.set_ylabel(name)
        ax.grid(True)
        
        if i == n_joints - 1:
            ax.set_xlabel('Time (s)')
            
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_force_data(time_data: np.ndarray, force_data: np.ndarray, 
                   title: str = "Force Data"):
    """
    Plot force data over time.
    
    Args:
        time_data: Time array
        force_data: Force data array [n_forces, n_time_steps]
        title: Plot title
    """
    n_forces = force_data.shape[0]
    force_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n_forces, 6)):
        ax = axes[i]
        ax.plot(time_data, force_data[i], 'r-', linewidth=2)
        ax.set_ylabel(force_names[i])
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# File I/O
def save_simulation_data(data: Dict[str, Any], filename: str):
    """
    Save simulation data to file.
    
    Args:
        data: Simulation data dictionary
        filename: Output filename
    """
    filepath = Path(filename)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_simulation_data(filename: str) -> Dict[str, Any]:
    """
    Load simulation data from file.
    
    Args:
        filename: Input filename
        
    Returns:
        Simulation data dictionary
    """
    filepath = Path(filename)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_trajectory(trajectory: np.ndarray, filename: str):
    """
    Save trajectory to file.
    
    Args:
        trajectory: Trajectory array
        filename: Output filename
    """
    np.save(filename, trajectory)


def load_trajectory(filename: str) -> np.ndarray:
    """
    Load trajectory from file.
    
    Args:
        filename: Input filename
        
    Returns:
        Trajectory array
    """
    return np.load(filename)


# Performance Utilities
class Timer:
    """Simple timer utility for performance measurement."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
            
        self.elapsed_time = time.time() - self.start_time
        self.start_time = None
        return self.elapsed_time
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        print(f"{self.name}: {self.elapsed_time:.4f} seconds")


def benchmark_function(func, *args, num_runs: int = 100, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of runs
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for _ in range(num_runs):
        with Timer() as timer:
            result = func(*args, **kwargs)
        times.append(timer.elapsed_time)
        
    times = np.array(times)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    } 