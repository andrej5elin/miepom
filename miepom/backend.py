"""Backend switching for the MIEPOM library.

Backend switching allows the user to change the underlying computational library (e.g., NumPy, CuPy) used by the MIEPOM library.
Use the `set_backend(xp)` function to switch the backend, where `xp` is the desired computational library (e.g., `numpy`, `cupy`).
"""

from .sim import set_backend as _set_simulator_backend
from .linalg import set_backend as _set_linalg_backend
from .sensor import set_backend as _set_sensor_backend
from .pom import set_backend as _set_microscope_backend
from .config import set_backend as _set_config_backend

import numpy as np

globals()["xp"] = np

def set_backend(xp):
    """Set the linear algebra backend for the MIEPOM library."""
    _set_linalg_backend(xp)
    _set_sensor_backend(xp)
    _set_microscope_backend(xp)
    _set_simulator_backend(xp)
    _set_config_backend(xp)
    globals()["xp"] = xp
    print(f"MIEPOM backend set to {xp.__name__}")

def get_backend():
    """Get the name of the current linear algebra backend for the MIEPOM library."""
    return globals()["xp"]

def get_backend_name():
    """Get the name of the current linear algebra backend for the MIEPOM library."""
    return globals()["xp"].__name__
