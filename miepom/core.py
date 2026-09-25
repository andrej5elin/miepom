"""Core module for the MIEPOM library.

This module provides convenient access to the main components of the MIEPOM library, including particles, simulators, microscopes, sensors, and backend management.
"""

from .particles import *
from .sim import *
from .pom import *
from .sensor import  *
from .backend import *

from .config import get_array