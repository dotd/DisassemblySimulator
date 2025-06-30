"""
Environment Management for MuJoCo Simulations

This module provides classes for managing simulation environments,
including scene setup, object placement, and environment configuration.
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class Environment:
    """
    Environment class for managing MuJoCo simulation environments.
    
    This class provides functionality for:
    - Setting up simulation scenes
    - Managing objects and obstacles
    - Configuring environment parameters
    - Handling environment-specific logic
    """
    
    def __init__(self, simulator):
        """
        Initialize the environment.
        
        Args:
            simulator: MuJoCoSimulator instance
        """
        self.simulator = simulator
        self.model = simulator.model
        self.data = simulator.data
        
        # Environment objects
        self.objects = {}
        self.obstacles = {}
        self.targets = {}
        
        # Environment parameters
        self.gravity = np.array([0, 0, -9.81])
        self.wind = np.zeros(3)
        self.ambient_temperature = 293.15  # Kelvin
        
        # Contact settings
        self.contact_settings = {
            'friction': 0.8,
            'restitution': 0.5,
            'margin': 0.001
        }
        
    def set_gravity(self, gravity: np.ndarray):
        """Set gravity vector."""
        self.gravity = np.array(gravity)
        self.model.opt.gravity[:] = self.gravity
        
    def set_wind(self, wind: np.ndarray):
        """Set wind vector."""
        self.wind = np.array(wind)
        
    def add_box(self, name: str, size: np.ndarray, position: np.ndarray, 
                quaternion: np.ndarray = None, mass: float = 1.0, 
                color: np.ndarray = None) -> str:
        """
        Add a box object to the environment.
        
        Args:
            name: Name of the box
            size: Box dimensions [x, y, z]
            position: Position in world coordinates
            quaternion: Orientation quaternion
            mass: Mass of the box
            color: RGB color [r, g, b]
            
        Returns:
            Body name of the created box
        """
        if quaternion is None:
            quaternion = np.array([1, 0, 0, 0])
        if color is None:
            color = np.array([0.7, 0.7, 0.7])
            
        # Create body
        body_name = f"box_{name}"
        
        # Add body to model (this is a simplified approach)
        # In practice, you would modify the XML or use MuJoCo's API
        self.objects[name] = {
            'type': 'box',
            'body_name': body_name,
            'size': size,
            'position': position,
            'quaternion': quaternion,
            'mass': mass,
            'color': color
        }
        
        return body_name
        
    def add_sphere(self, name: str, radius: float, position: np.ndarray,
                   quaternion: np.ndarray = None, mass: float = 1.0,
                   color: np.ndarray = None) -> str:
        """
        Add a sphere object to the environment.
        
        Args:
            name: Name of the sphere
            radius: Radius of the sphere
            position: Position in world coordinates
            quaternion: Orientation quaternion
            mass: Mass of the sphere
            color: RGB color [r, g, b]
            
        Returns:
            Body name of the created sphere
        """
        if quaternion is None:
            quaternion = np.array([1, 0, 0, 0])
        if color is None:
            color = np.array([0.7, 0.7, 0.7])
            
        body_name = f"sphere_{name}"
        
        self.objects[name] = {
            'type': 'sphere',
            'body_name': body_name,
            'radius': radius,
            'position': position,
            'quaternion': quaternion,
            'mass': mass,
            'color': color
        }
        
        return body_name
        
    def add_cylinder(self, name: str, radius: float, height: float, 
                     position: np.ndarray, quaternion: np.ndarray = None,
                     mass: float = 1.0, color: np.ndarray = None) -> str:
        """
        Add a cylinder object to the environment.
        
        Args:
            name: Name of the cylinder
            radius: Radius of the cylinder
            height: Height of the cylinder
            position: Position in world coordinates
            quaternion: Orientation quaternion
            mass: Mass of the cylinder
            color: RGB color [r, g, b]
            
        Returns:
            Body name of the created cylinder
        """
        if quaternion is None:
            quaternion = np.array([1, 0, 0, 0])
        if color is None:
            color = np.array([0.7, 0.7, 0.7])
            
        body_name = f"cylinder_{name}"
        
        self.objects[name] = {
            'type': 'cylinder',
            'body_name': body_name,
            'radius': radius,
            'height': height,
            'position': position,
            'quaternion': quaternion,
            'mass': mass,
            'color': color
        }
        
        return body_name
        
    def add_obstacle(self, name: str, geometry_type: str, **kwargs) -> str:
        """
        Add an obstacle to the environment.
        
        Args:
            name: Name of the obstacle
            geometry_type: Type of geometry ('box', 'sphere', 'cylinder')
            **kwargs: Geometry-specific parameters
            
        Returns:
            Body name of the created obstacle
        """
        if geometry_type == 'box':
            return self.add_box(name, **kwargs)
        elif geometry_type == 'sphere':
            return self.add_sphere(name, **kwargs)
        elif geometry_type == 'cylinder':
            return self.add_cylinder(name, **kwargs)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
            
    def add_target(self, name: str, position: np.ndarray, 
                   tolerance: float = 0.1, color: np.ndarray = None) -> str:
        """
        Add a target position to the environment.
        
        Args:
            name: Name of the target
            position: Target position in world coordinates
            tolerance: Tolerance for reaching the target
            color: RGB color [r, g, b]
            
        Returns:
            Target name
        """
        if color is None:
            color = np.array([0, 1, 0])  # Green
            
        self.targets[name] = {
            'position': position,
            'tolerance': tolerance,
            'color': color,
            'reached': False
        }
        
        return name
        
    def remove_object(self, name: str):
        """Remove an object from the environment."""
        if name in self.objects:
            del self.objects[name]
        elif name in self.obstacles:
            del self.obstacles[name]
        elif name in self.targets:
            del self.targets[name]
        else:
            raise ValueError(f"Object '{name}' not found")
            
    def get_object_position(self, name: str) -> np.ndarray:
        """Get position of an object."""
        if name in self.objects:
            return self.simulator.get_body_position(self.objects[name]['body_name'])
        elif name in self.obstacles:
            return self.simulator.get_body_position(self.obstacles[name]['body_name'])
        else:
            raise ValueError(f"Object '{name}' not found")
            
    def set_object_position(self, name: str, position: np.ndarray):
        """Set position of an object."""
        if name in self.objects:
            body_name = self.objects[name]['body_name']
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.data.xpos[body_id] = position
        elif name in self.obstacles:
            body_name = self.obstacles[name]['body_name']
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.data.xpos[body_id] = position
        else:
            raise ValueError(f"Object '{name}' not found")
            
    def check_target_reached(self, robot_body: str, target_name: str) -> bool:
        """
        Check if a robot has reached a target.
        
        Args:
            robot_body: Name of the robot body
            target_name: Name of the target
            
        Returns:
            True if target is reached, False otherwise
        """
        if target_name not in self.targets:
            raise ValueError(f"Target '{target_name}' not found")
            
        robot_pos = self.simulator.get_body_position(robot_body)
        target_pos = self.targets[target_name]['position']
        tolerance = self.targets[target_name]['tolerance']
        
        distance = np.linalg.norm(robot_pos - target_pos)
        reached = distance <= tolerance
        
        if reached and not self.targets[target_name]['reached']:
            self.targets[target_name]['reached'] = True
            print(f"Target '{target_name}' reached!")
            
        return reached
        
    def reset_targets(self):
        """Reset all targets to unreached state."""
        for target in self.targets.values():
            target['reached'] = False
            
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Returns:
            Dictionary containing environment information
        """
        return {
            'gravity': self.gravity,
            'wind': self.wind,
            'ambient_temperature': self.ambient_temperature,
            'num_objects': len(self.objects),
            'num_obstacles': len(self.obstacles),
            'num_targets': len(self.targets),
            'objects': list(self.objects.keys()),
            'obstacles': list(self.obstacles.keys()),
            'targets': list(self.targets.keys())
        }
        
    def create_simple_environment(self):
        """Create a simple test environment with basic objects."""
        # Add a ground plane (assuming it exists in the model)
        
        # Add some test objects
        self.add_box("test_box", 
                     size=np.array([0.1, 0.1, 0.1]),
                     position=np.array([0.5, 0, 0.05]),
                     mass=1.0,
                     color=np.array([1, 0, 0]))
                     
        self.add_sphere("test_sphere",
                        radius=0.05,
                        position=np.array([-0.5, 0, 0.05]),
                        mass=0.5,
                        color=np.array([0, 0, 1]))
                        
        # Add a target
        self.add_target("goal",
                        position=np.array([0, 0.5, 0.1]),
                        tolerance=0.1,
                        color=np.array([0, 1, 0]))
                        
    def create_cluttered_environment(self, num_obstacles: int = 10):
        """
        Create a cluttered environment with random obstacles.
        
        Args:
            num_obstacles: Number of obstacles to create
        """
        for i in range(num_obstacles):
            # Random position
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(0.05, 0.5)
            
            # Random geometry
            geom_type = np.random.choice(['box', 'sphere', 'cylinder'])
            
            if geom_type == 'box':
                size = np.random.uniform(0.05, 0.2, 3)
                self.add_box(f"obstacle_{i}", size, np.array([x, y, z]))
            elif geom_type == 'sphere':
                radius = np.random.uniform(0.05, 0.1)
                self.add_sphere(f"obstacle_{i}", radius, np.array([x, y, z]))
            else:  # cylinder
                radius = np.random.uniform(0.05, 0.1)
                height = np.random.uniform(0.1, 0.3)
                self.add_cylinder(f"obstacle_{i}", radius, height, np.array([x, y, z])) 