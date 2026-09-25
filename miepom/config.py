"""Configuration settings for the MIEPOM library."""

import numpy as np
import os, time
from .utils import print_line

globals()["xp"] = np

def set_backend(xp):
    globals()["xp"] = xp
try:
    import scipy as sp
    USESCIPY = True
except ImportError:
    USESCIPY = False
try:
    import numba as nb
    USENUMBA = True
except ImportError:
    USENUMBA = False

try:
    import mlx.core as mx
    USEMLX = True
except ImportError:
    USEMLX = False

try:
    import cupy as cp
    USECUPY = True
except ImportError:
    USECUPY = False

VERBOSE = False if os.environ.get("MIEPOM_VERBOSE", "0") == "0" else True

# always use numpy as the default backend for the core object, but allow switching to other backends if available
DEFAULT_BACKEND = "numpy"

#: Computation precision. Note that if MLX (or CuPy) is used, it is always single precision
PRECISION = "single" if os.environ.get("MIEPOM_USE_DOUBLE", "0") == "0" else "double" # 'single' or 'double'

#: whether to use fast math (numba)
NBFASTMATH = False if os.environ.get("MIEPOM_FASTMATH", "0") == "0" else True

#: numba's compile target argument for vectorize and guvectorize functions.
NBTARGET = os.environ.get("MIEPOM_TARGET", "parallel")
if NBTARGET not in ["cpu", "parallel", "cuda"]:
    raise ValueError(f"Invalid MIEPOM_TARGET value: {NBTARGET}. Must be one of 'cpu', 'parallel', or 'cuda'.")

#: if set to True, it assumes a vectorized version of the miepython library
VECTORIZED_MIE = False if os.environ.get("MIEPOM_VECTORIZED_MIE", "0") == "0" else True

# Computed constants below. Do not modify!
#-----------------------------------------

if USENUMBA:
    import numba as nb

class nbtype():
    def __getitem__(self, key):
        pass

try:
    import mlx.core as mx
    mx_compile = mx.compile
    mx_eval = mx.eval
except ImportError:
    mx_compile = lambda f: f
    mx_eval = lambda *args, **kwargs: None

def mx_compile_out(func):
    """A decorator to compile a function with mlx.core.compile. This is used to compile the sensor_image function with mlx.
    It also handles the 'out' keyword argument for in-place computation.
    """
    func = mx_compile(func)
    def wrapper(*args, **kwargs):
        if 'out' in kwargs:
            out = kwargs.pop('out')
        else:
            out = None
        if out is not None:
            out[:] = func(*args, **kwargs)[:]
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper

if USENUMBA:
    vectorize = nb.vectorize
    guvectorize = nb.guvectorize
    jit = nb.jit
    njit = nb.njit
else:
    vectorize = lambda f,*args,**kwargs: f
    guvectorize = lambda f,*args,**kwargs: f
    jit = lambda f,*args,**kwargs: f
    njit = lambda f,*args,**kwargs: f
    NBFLOAT = nbtype()
    NBCOMPLEX = nbtype()
    NBUINT = nbtype()
    NBBOOL = nbtype()

    nbfloat = nbtype()
    nbcomplex = nbtype()
    nbuint = nbtype()
    nbbool = nbtype()

os.environ["MIEPYTHON_CACHE"] = "0"
os.environ["MIEPYTHON_FASTMATH"] = "1"
os.environ["MIEPYTHON_TARGET"] = NBTARGET

if PRECISION == "single":
    os.environ["MIEPYTHON_USE_DOUBLE"] = "0"
    if USENUMBA:
        NBFLOAT = nb.float32
        NBCOMPLEX = nb.complex64
        NBUINT = nb.uint16
        NBBOOL = nb.boolean

        nbfloat = nb.float32
        nbcomplex = nb.complex64
        nbuint = nb.uint16
        nbbool = nb.boolean

    CDTYPE = np.complex64
    FDTYPE = np.float32
    UDTYPE = "uint16"

    cdtype = lambda: xp.complex64
    fdtype = lambda: xp.float32
    udtype = lambda: xp.uint16

elif PRECISION == "double":
    os.environ["MIEPYTHON_USE_DOUBLE"] = "1"
    if USENUMBA:
        NBFLOAT = nb.float64
        NBCOMPLEX = nb.complex128
        NBUINT = nb.uint16
        NBBOOL = nb.boolean

        nbfloat = nb.float64
        nbcomplex = nb.complex128
        nbuint = nb.uint16
        nbbool = nb.boolean

    CDTYPE = np.complex128
    FDTYPE = np.float64
    UDTYPE = "uint16"

    cdtype = lambda: xp.complex128
    fdtype = lambda: xp.float64
    udtype = lambda: xp.uint16
else:
    raise ValueError("Invalid precision name")

import miepython as mie

def print_settings():
    print("Library settings")
    print_line("-")
    
    print(f"Using Numba:         {USENUMBA}")
    if USENUMBA:
        print(f"Using fastmath:      {NBFASTMATH}")
        print(f"Numba target:        '{NBTARGET}'")
        print(f"Num threads:         {nb.get_num_threads()}")
    print(f"Precission:          '{PRECISION}'")
    print(f"Using mlx:            {USEMLX}")
    print(f"Using cupy:           {USECUPY}")
    print_line("=")

if VERBOSE:
    print_settings()

    print("numpy", np.__version__)
    print("miepython", mie.__version__)
    if USESCIPY:
        print("scipy", sp.__version__)
    if USECUPY:
        print("cupy", cp.__version__)
    if USEMLX:
        print("mlx", mx.__version__)   
    if USENUMBA:
        print("numba", nb.__version__)

    print("-----------------------------------------------------------------------")

