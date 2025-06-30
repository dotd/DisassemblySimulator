"""
Robot Management for MuJoCo Simulations

This module provides classes for managing robots in MuJoCo simulations,
including robot control, kinematics, and dynamics.
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod


class Robot(ABC):
    """
    Abstract base class for robots in MuJoCo simulations.
    
    This class provides common functionality for:
    - Robot state management
    - Joint control
    - Kinematics and dynamics
    - Robot-specific behaviors
    """
    
    def __init__(self, simulator, robot_body_name: str):
        """
        Initialize the robot.
        
        Args:
            simulator: MuJoCoSimulator instance
            robot_body_name: Name of the robot's main body
        """
        self.simulator = simulator
        self.model = simulator.model
        self.data = simulator.data
        self.robot_body_name = robot_body_name
        
        # Get robot body ID
        self.robot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name
        )
        
        # Robot state
        self.joint_names = []
        self.joint_ids = []
        self.actuator_names = []
        self.actuator_ids = []
        
        # Control parameters
        self.control_mode = 'position'  # 'position', 'velocity', 'torque'
        self.control_gains = {
            'kp': 100.0,
            'kd': 10.0,
            'ki': 0.0
        }
        
        # Trajectory tracking
        self.target_positions = None
        self.target_velocities = None
        self.target_torques = None
        
        # Initialize robot-specific components
        self._initialize_robot()
        
    def _initialize_robot(self):
        """Initialize robot-specific components. Override in subclasses."""
        # Get joint information
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self.joint_names.append(joint_name)
                self.joint_ids.append(i)
                
        # Get actuator information
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                self.actuator_names.append(actuator_name)
                self.actuator_ids.append(i)
                
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.simulator.get_joint_positions()
        
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions."""
        self.simulator.set_joint_positions(positions)
        
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        return self.simulator.get_joint_velocities()
        
    def set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities."""
        self.simulator.set_joint_velocities(velocities)
        
    def get_joint_torques(self) -> np.ndarray:
        """Get current joint torques."""
        return self.simulator.get_joint_forces()
        
    def set_joint_torques(self, torques: np.ndarray):
        """Set joint torques."""
        self.simulator.set_joint_forces(torques)
        
    def get_end_effector_position(self) -> np.ndarray:
        """Get end effector position."""
        return self.simulator.get_body_position(self.robot_body_name)
        
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector position and orientation."""
        return self.simulator.get_body_pose(self.robot_body_name)
        
    def set_control_mode(self, mode: str):
        """
        Set control mode.
        
        Args:
            mode: Control mode ('position', 'velocity', 'torque')
        """
        if mode not in ['position', 'velocity', 'torque']:
            raise ValueError(f"Invalid control mode: {mode}")
        self.control_mode = mode
        
    def set_control_gains(self, kp: float = None, kd: float = None, ki: float = None):
        """
        Set control gains.
        
        Args:
            kp: Proportional gain
            kd: Derivative gain
            ki: Integral gain
        """
        if kp is not None:
            self.control_gains['kp'] = kp
        if kd is not None:
            self.control_gains['kd'] = kd
        if ki is not None:
            self.control_gains['ki'] = ki
            
    def set_target_positions(self, positions: np.ndarray):
        """Set target joint positions."""
        self.target_positions = np.array(positions)
        
    def set_target_velocities(self, velocities: np.ndarray):
        """Set target joint velocities."""
        self.target_velocities = np.array(velocities)
        
    def set_target_torques(self, torques: np.ndarray):
        """Set target joint torques."""
        self.target_torques = np.array(torques)
        
    def compute_control(self) -> np.ndarray:
        """
        Compute control output based on current control mode.
        
        Returns:
            Control output (positions, velocities, or torques)
        """
        if self.control_mode == 'position':
            return self._compute_position_control()
        elif self.control_mode == 'velocity':
            return self._compute_velocity_control()
        elif self.control_mode == 'torque':
            return self._compute_torque_control()
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
            
    def _compute_position_control(self) -> np.ndarray:
        """Compute position control output."""
        if self.target_positions is None:
            return np.zeros(len(self.joint_names))
            
        current_positions = self.get_joint_positions()
        current_velocities = self.get_joint_velocities()
        
        # PID control
        position_error = self.target_positions - current_positions
        velocity_error = np.zeros_like(current_velocities)
        if self.target_velocities is not None:
            velocity_error = self.target_velocities - current_velocities
            
        control_output = (
            self.control_gains['kp'] * position_error +
            self.control_gains['kd'] * velocity_error
        )
        
        return control_output
        
    def _compute_velocity_control(self) -> np.ndarray:
        """Compute velocity control output."""
        if self.target_velocities is None:
            return np.zeros(len(self.joint_names))
            
        current_velocities = self.get_joint_velocities()
        velocity_error = self.target_velocities - current_velocities
        
        control_output = self.control_gains['kp'] * velocity_error
        return control_output
        
    def _compute_torque_control(self) -> np.ndarray:
        """Compute torque control output."""
        if self.target_torques is None:
            return np.zeros(len(self.joint_names))
            
        return self.target_torques
        
    def apply_control(self):
        """Apply computed control to the robot."""
        control_output = self.compute_control()
        
        if self.control_mode == 'position':
            # For position control, we set the control signal directly
            self.data.ctrl[:] = control_output
        elif self.control_mode == 'velocity':
            # For velocity control, we set the control signal directly
            self.data.ctrl[:] = control_output
        elif self.control_mode == 'torque':
            # For torque control, we apply forces
            self.set_joint_torques(control_output)
            
    def move_to_position(self, target_position: np.ndarray, duration: float = 1.0):
        """
        Move robot to a target position over a specified duration.
        
        Args:
            target_position: Target joint positions
            duration: Movement duration in seconds
        """
        self.set_target_positions(target_position)
        
        # Simple trajectory generation (linear interpolation)
        start_positions = self.get_joint_positions()
        n_steps = int(duration / self.simulator.dt)
        
        for i in range(n_steps):
            t = i / n_steps
            current_target = start_positions + t * (target_position - start_positions)
            self.set_target_positions(current_target)
            self.apply_control()
            self.simulator.step()
            
    def follow_trajectory(self, trajectory: np.ndarray, duration: float = None):
        """
        Follow a joint trajectory.
        
        Args:
            trajectory: Array of joint positions [n_steps, n_joints]
            duration: Total duration (if None, uses simulator dt)
        """
        n_steps = len(trajectory)
        
        if duration is None:
            duration = n_steps * self.simulator.dt
            
        dt_step = duration / n_steps
        
        for i, target_pos in enumerate(trajectory):
            self.set_target_positions(target_pos)
            self.apply_control()
            
            # Step simulation
            step_steps = int(dt_step / self.simulator.dt)
            self.simulator.step(step_steps)
            
    def get_robot_info(self) -> Dict[str, Any]:
        """
        Get information about the robot.
        
        Returns:
            Dictionary containing robot information
        """
        return {
            'robot_body_name': self.robot_body_name,
            'robot_body_id': self.robot_body_id,
            'num_joints': len(self.joint_names),
            'num_actuators': len(self.actuator_names),
            'joint_names': self.joint_names,
            'actuator_names': self.actuator_names,
            'control_mode': self.control_mode,
            'control_gains': self.control_gains,
            'current_positions': self.get_joint_positions().tolist(),
            'current_velocities': self.get_joint_velocities().tolist(),
            'end_effector_position': self.get_end_effector_position().tolist()
        }
        
    @abstractmethod
    def get_robot_type(self) -> str:
        """Get the type of robot. Override in subclasses."""
        pass
        
    def reset(self):
        """Reset robot to initial state."""
        # Reset targets
        self.target_positions = None
        self.target_velocities = None
        self.target_torques = None
        
        # Reset to initial joint positions
        initial_positions = np.zeros(len(self.joint_names))
        self.set_joint_positions(initial_positions)
        self.set_joint_velocities(initial_positions)


class SimpleRobot(Robot):
    """
    Simple robot implementation for basic simulations.
    """
    
    def __init__(self, simulator, robot_body_name: str):
        super().__init__(simulator, robot_body_name)
        
    def get_robot_type(self) -> str:
        return "SimpleRobot"
        
    def simple_movement_test(self):
        """Perform a simple movement test."""
        print("Starting simple movement test...")
        
        # Get current position
        current_pos = self.get_joint_positions()
        print(f"Current position: {current_pos}")
        
        # Move to a simple target
        target_pos = current_pos + np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[:len(current_pos)]
        print(f"Moving to target: {target_pos}")
        
        self.move_to_position(target_pos, duration=2.0)
        
        print("Movement test completed!")


class ManipulatorRobot(Robot):
    """
    Robot manipulator implementation for arm-like robots.
    """
    
    def __init__(self, simulator, robot_body_name: str, end_effector_body: str = None):
        super().__init__(simulator, robot_body_name)
        self.end_effector_body = end_effector_body or robot_body_name
        
    def get_robot_type(self) -> str:
        return "ManipulatorRobot"
        
    def get_end_effector_position(self) -> np.ndarray:
        """Get end effector position."""
        return self.simulator.get_body_position(self.end_effector_body)
        
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector position and orientation."""
        return self.simulator.get_body_pose(self.end_effector_body)
        
    def move_end_effector_to(self, target_position: np.ndarray, duration: float = 2.0):
        """
        Move end effector to a target position.
        
        Note: This is a simplified implementation. In practice, you would need
        inverse kinematics to convert end effector position to joint positions.
        
        Args:
            target_position: Target end effector position
            duration: Movement duration
        """
        print(f"Moving end effector to: {target_position}")
        
        # For now, we'll just move the joints slightly
        # In a real implementation, you would use inverse kinematics
        current_pos = self.get_joint_positions()
        target_joint_pos = current_pos + np.random.uniform(-0.1, 0.1, len(current_pos))
        
        self.move_to_position(target_joint_pos, duration)
        
        final_pos = self.get_end_effector_position()
        print(f"End effector final position: {final_pos}")


class MobileRobot(Robot):
    """
    Mobile robot implementation for wheeled or legged robots.
    """
    
    def __init__(self, simulator, robot_body_name: str):
        super().__init__(simulator, robot_body_name)
        self.base_velocity = np.zeros(3)
        self.base_angular_velocity = np.zeros(3)
        
    def get_robot_type(self) -> str:
        return "MobileRobot"
        
    def set_base_velocity(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray = None):
        """
        Set base velocity for mobile robot.
        
        Args:
            linear_velocity: Linear velocity [vx, vy, vz]
            angular_velocity: Angular velocity [wx, wy, wz]
        """
        self.base_velocity = np.array(linear_velocity)
        if angular_velocity is not None:
            self.base_angular_velocity = np.array(angular_velocity)
        else:
            self.base_angular_velocity = np.zeros(3)
            
    def move_forward(self, velocity: float, duration: float = 1.0):
        """
        Move robot forward.
        
        Args:
            velocity: Forward velocity
            duration: Movement duration
        """
        self.set_base_velocity([velocity, 0, 0])
        
        n_steps = int(duration / self.simulator.dt)
        for _ in range(n_steps):
            # Apply velocity control
            self.apply_control()
            self.simulator.step()
            
    def turn(self, angular_velocity: float, duration: float = 1.0):
        """
        Turn robot.
        
        Args:
            angular_velocity: Angular velocity (positive for left turn)
            duration: Turn duration
        """
        self.set_base_velocity([0, 0, 0], [0, 0, angular_velocity])
        
        n_steps = int(duration / self.simulator.dt)
        for _ in range(n_steps):
            # Apply velocity control
            self.apply_control()
            self.simulator.step() 