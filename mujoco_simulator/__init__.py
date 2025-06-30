"""
MuJoCo Simulator Package

A Python-based MuJoCo simulation environment for robotics and physics simulation.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .simulator import MuJoCoSimulator
from .environment import Environment
from .robot import Robot
from .utils import *

__all__ = [
    "MuJoCoSimulator",
    "Environment", 
    "Robot",
] 