"""Linear algebra utilities for the PPOM library."""

# Compiled numba code
from .config import USENUMBA, nbfloat,nbcomplex, nbuint, nbbool, NBFASTMATH, NBTARGET, vectorize, guvectorize, jit, njit, mx_compile, mx_compile_out, mx_compile
import numpy as np

# Set the default linear algebra backend to numpy so that we can compile and run numba functions correctly.
globals()["xp"] = np

def backend_name():
    """Return the name of the current linear algebra backend."""
    return globals()["xp"].__name__

def set_backend(xp):
    globals()["xp"] = xp

# data packing and unpacking functions for boolean masks

# reference implementation of the pack function using numpy advanced indexing. 
def _pack_np(data,mask):
    return data[...,mask]

# numba implementation of the pack function, requires a dummy argument so that we can determine the output shape at compile time. This is a hack, but it works.
@guvectorize([(nbcomplex[:,:],nbbool[:,:],nbcomplex[:],nbcomplex[:])],
                 "(m,n),(m,n),(l)->(l)",boundscheck = False, target = NBTARGET, fastmath = NBFASTMATH)
def __pack_nb(a, mask, _dummy, out):
    n = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == True:
                if n < len(out):
                    out[n] = a[i,j]
                    n+=1

# numba wrapper to remove the dummy argument
def _pack_nb(a,mask, out = None):
    if out is None:
        out = np.empty(a.shape[:-2] + (np.sum(mask),), dtype = a.dtype)
    return __pack_nb(a,mask,out,out = out) 

# generic xp implementation that calls the numpy version and adds support for output array
def _pack_xp(data,mask, out = None):
    if out is None:
        return _pack_np(data,mask)
    else:   
        out[:] = _pack_np(data,mask)
        return out

# boolean indices are not supported in mlx, so we need to implement a custom pack function for mlx
def _pack_mx(data,mask, out = None):
    if out is None:
        #we convert to numpy array first, because mlx does not support boolean indexing
        return xp.array(np.asarray(data)[...,mask])
    else:
        out[:] = np.asarray(data)[...,mask]
        return out

def _unpack_np(mask,data, out = None):
    if out is None:
        out = xp.zeros(data.shape[:-1] + mask.shape, dtype = data.dtype)
    out[...] = 0.
    out[...,mask] = data
    return out

_unpack_xp = _unpack_np

# the mx version cannot use vectorize version because it does not support boolean indexing, so we need to implement a custom unpack function for mlx
def __unpack_mx(mask,data, out = None):
    if out is None:
        out = xp.zeros(mask.shape, dtype = data.dtype)
    out[:] = 0.
    out[mask] = data
    return out

# the vectorized version of the mlx unpack function is used when the data is 2D, so that we can use vmap to speed up the computation.
def __unpack_mx_vec(mask,data, out = None):
    if out is None:
        return xp.vmap(__unpack_mx, in_axes=(None, 0), out_axes=0)(mask,data)
    else:
        out[:] = 0.
        tmp = xp.vmap(__unpack_mx, in_axes=(None, 0), out_axes=0)(mask,data)
        out[:] = tmp
        return out

def _unpack_mx(mask,data, out = None):
    if data.ndim == 2:
        return __unpack_mx_vec(mask,data, out = out)
    else:   
        return __unpack_mx(mask,data, out = out)

@jit([(nbbool[:,:],nbcomplex[:],nbcomplex[:,:])])
def __unpack_nb(mask,data,out):
    k = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if mask[i,j] == True:
                out[i,j] = data[k]
                k+=1
            else:
                out[i,j] = 0.

@guvectorize([(nbbool[:,:],nbcomplex[:],nbcomplex[:,:])], "(n,m),(k)->(n,m)", target = NBTARGET, fastmath = NBFASTMATH)
def _unpack_nb(mask,data,out):
    __unpack_nb(mask,data,out)

def unpack(mask, data, out = None):
    """Unpack a 1D array into a 2D array at the locations specified by a boolean mask."""
    if backend_name() == "mlx.core":   
        return _unpack_mx(mask,data, out = out)
    elif backend_name() == "numba":
        return _unpack_nb(mask,data, out = out)
    else:
        return _unpack_xp(mask,data, out)

def pack(data, mask, out = None):
    """Pack a 2D array into a 1D array at the locations specified by a boolean mask."""
    if backend_name() == "mlx.core":
        return _pack_mx(data, mask, out)
    elif backend_name() == "numba":
        return _pack_nb(data, mask, out)
    else:
        return _pack_xp(data, mask, out)

def _abs2_np(a):
    return a.real**2 + a.imag**2

_abs2_nb = vectorize([nbfloat(nbcomplex)],target = NBTARGET,fastmath = NBFASTMATH)(_abs2_np)

def _abs2_mx(a, out = None):
    if out is None:
        return mx_compile(_abs2_np)(a)
    else:
        out[:] = mx_compile(_abs2_np)(a)
        return out

def _abs2_xp(a, out = None):
    x2 = np.real(a)**2
    y = np.imag(a)
    y2 = np.multiply(y, y, out = out)
    return np.add(x2, y2, out = y2)

def abs2(a, out = None):
    if backend_name() == "mlx.core":
        return _abs2_mx(a, out)
    elif backend_name() == "numba":
        return _abs2_nb(a, out)
    else:
        return _abs2_xp(a, out)

_vecdot_xp = np.vecdot

def _vecdot_mx(a,b, out = None):
    if out is None:
        return mx_compile(_vecdot_xp)(a,b)
    else:
        out[:] = mx_compile(_vecdot_xp)(a,b)
        return out  

@guvectorize([(nbfloat[:],nbfloat[:],nbfloat[:])], "(n),(n)->()", 
                fastmath = NBFASTMATH, cache = True, target = NBTARGET)
def _vecdot_nb(a,b, out):
    tmp = 0
    for i in range(3):
        tmp += a[i] * b[i]
    out[0] = tmp

def vecdot(a,b, out = None):
    if backend_name() == "mlx.core":
        return _vecdot_mx(a,b, out)
    elif backend_name() == "numba":
        return _vecdot_nb(a,b, out)
    else:
        return _vecdot_xp(a,b, out)

def _exp_xp(x):
    return xp.exp(x)

def _exp_mx(x, out = None):
    if out is None:
        return mx_compile(_exp_xp)(x)
    else:
        out[:] = mx_compile(_exp_xp)(x)
        return out

_exp_nb = vectorize([nbcomplex(nbcomplex)], target = NBTARGET, fastmath = NBFASTMATH)(_exp_xp)

def exp(x):
    """Compute the complex exponential of a real array."""
    if backend_name() == "mlx.core":
        return _exp_mx(x)
    elif backend_name() == "numba":
        return _exp_nb(x)
    else:
        return _exp_xp(x)

