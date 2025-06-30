"""
Main MuJoCo Simulator Class

This module provides the core MuJoCo simulation functionality.
"""

import mujoco
import mujoco_viewer
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class MuJoCoSimulator:
    """
    Main MuJoCo simulator class for running physics simulations.
    
    This class provides a high-level interface to MuJoCo for:
    - Loading and managing simulation models
    - Running simulation steps
    - Controlling robots and objects
    - Rendering and visualization
    """
    
    def __init__(self, model_path: str, dt: float = 0.01):
        """
        Initialize the MuJoCo simulator.
        
        Args:
            model_path: Path to the MuJoCo XML model file
            dt: Simulation time step (default: 0.01 seconds)
        """
        self.model_path = Path(model_path)
        self.dt = dt
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        
        # Set simulation parameters
        self.model.opt.timestep = self.dt
        
        # Viewer and rendering
        self.viewer = None
        self.render_enabled = False
        
        # Simulation state
        self.is_running = False
        self.simulation_time = 0.0
        
        # Callbacks
        self.step_callbacks = []
        self.reset_callbacks = []
        
    def add_step_callback(self, callback):
        """Add a callback function to be called after each simulation step."""
        self.step_callbacks.append(callback)
        
    def add_reset_callback(self, callback):
        """Add a callback function to be called when simulation is reset."""
        self.reset_callbacks.append(callback)
        
    def start_viewer(self, show_contact_points: bool = True, show_contact_forces: bool = True):
        """
        Start the MuJoCo viewer for visualization.
        
        Args:
            show_contact_points: Whether to show contact points
            show_contact_forces: Whether to show contact forces
        """
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model, 
                self.data,
                show_contact_points=show_contact_points,
                show_contact_forces=show_contact_forces
            )
            self.render_enabled = True
            
    def stop_viewer(self):
        """Stop the MuJoCo viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            self.render_enabled = False
            
    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        """
        Run simulation for a specified number of steps.
        
        Args:
            n_steps: Number of simulation steps to run
            
        Returns:
            Dictionary containing simulation state information
        """
        for _ in range(n_steps):
            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update simulation time
            self.simulation_time += self.dt
            
            # Call step callbacks
            for callback in self.step_callbacks:
                callback(self)
                
            # Render if viewer is active
            if self.render_enabled and self.viewer is not None:
                self.viewer.render()
                
        return self.get_state()
        
    def reset(self):
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.simulation_time = 0.0
        self.is_running = False
        
        # Call reset callbacks
        for callback in self.reset_callbacks:
            callback(self)
            
    def get_state(self) -> Dict[str, Any]:
        """
        Get current simulation state.
        
        Returns:
            Dictionary containing current simulation state
        """
        return {
            'time': self.simulation_time,
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'act': self.data.act.copy() if self.data.act is not None else None,
            'ctrl': self.data.ctrl.copy(),
            'qfrc_applied': self.data.qfrc_applied.copy(),
            'xfrc_applied': self.data.xfrc_applied.copy(),
            'contact': self.data.contact.copy() if len(self.data.contact) > 0 else None,
        }
        
    def set_state(self, state: Dict[str, Any]):
        """
        Set simulation state.
        
        Args:
            state: Dictionary containing simulation state
        """
        if 'qpos' in state:
            self.data.qpos[:] = state['qpos']
        if 'qvel' in state:
            self.data.qvel[:] = state['qvel']
        if 'act' in state and state['act'] is not None:
            self.data.act[:] = state['act']
        if 'ctrl' in state:
            self.data.ctrl[:] = state['ctrl']
        if 'qfrc_applied' in state:
            self.data.qfrc_applied[:] = state['qfrc_applied']
        if 'xfrc_applied' in state:
            self.data.xfrc_applied[:] = state['xfrc_applied']
            
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.data.qpos.copy()
        
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions."""
        self.data.qpos[:] = positions
        mujoco.mj_forward(self.model, self.data)
        
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        return self.data.qvel.copy()
        
    def set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities."""
        self.data.qvel[:] = velocities
        
    def get_joint_forces(self) -> np.ndarray:
        """Get current joint forces/torques."""
        return self.data.qfrc_applied.copy()
        
    def set_joint_forces(self, forces: np.ndarray):
        """Set joint forces/torques."""
        self.data.qfrc_applied[:] = forces
        
    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get position of a specific body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()
        
    def get_body_quaternion(self, body_name: str) -> np.ndarray:
        """Get quaternion orientation of a specific body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xquat[body_id].copy()
        
    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and orientation of a specific body."""
        return self.get_body_position(body_name), self.get_body_quaternion(body_name)
        
    def apply_force_to_body(self, body_name: str, force: np.ndarray, point: np.ndarray = None):
        """
        Apply force to a specific body.
        
        Args:
            body_name: Name of the body
            force: 3D force vector
            point: Point of application (in body frame, optional)
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if point is None:
            point = np.zeros(3)
            
        # Apply force and torque
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = np.cross(point, force)
        
    def run_simulation(self, duration: float, render: bool = True):
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation duration in seconds
            render: Whether to render the simulation
        """
        if render and self.viewer is None:
            self.start_viewer()
            
        n_steps = int(duration / self.dt)
        self.is_running = True
        
        try:
            for _ in range(n_steps):
                self.step()
                if render and self.viewer is not None:
                    time.sleep(self.dt)  # Real-time simulation
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        finally:
            self.is_running = False
            
    def close(self):
        """Clean up resources."""
        self.stop_viewer()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 