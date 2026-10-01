from . import linalg as linalg
import numpy as np
from .config import mx_compile, fdtype, cdtype, get_array

# Set the default linear algebra backend to numpy so that we can compile and run numba functions correctly.
globals()["xp"] = np

def set_backend(xp):
    name = xp.__name__
    globals()["xp"] = xp

def _propagate_modes_xp(modes,xi,xj,z,k,ki,kj,kz, sigma_q, sigma_k):    
    q2 = kj**2 + ki**2
    qz = q2/2/k
    qz2 = qz*qz
    decay = 0.5*z**2*(sigma_q**2*q2 + sigma_k**2*qz2)
    phase = linalg.exp(1j*kz*z - 1j*k*z - decay + 1j*ki*xi +1j*kj*xj) 
    modes = modes * phase
    return modes

_propagate_modes_mx = mx_compile(_propagate_modes_xp)
    
def propagate_modes_mx(modes,xi,xj,z,k,ki,kj,kz,dq,dk):
    f = xp.vmap(_propagate_modes_mx,in_axes=(0,0,0,0,None,None,None,None,None,None))         
    out = f(modes,xi,xj,z,k,ki,kj,kz,dq,dk)
    out = out.sum(axis = 0)
    return out

def propagate_modes_xp(modes,xi,xj,z,k,ki,kj,kz,dq,dk):
    out = _propagate_modes_xp(modes,xi[:,None],xj[:,None],z[:,None],k,ki,kj,kz,dq,dk).sum(axis = 0)
    return out

def propagate_modes(modes,xi,xj,z,k,ki,kj,kz,dq,dk):
    if xp.__name__ == 'mlx.core':
        return propagate_modes_mx(modes,xi,xj,z,k,ki,kj,kz,dq,dk)
    else:
        return propagate_modes_xp(modes,xi,xj,z,k,ki,kj,kz,dq,dk)

def propagate_modes_mono_mx(modes,xi,xj,z,k,ki,kj,kz,dq,dk):
    f = xp.vmap(_propagate_modes_mx,in_axes=(None,0,0,0,None,None,None,None,None,None))         
    out = f(modes,xi,xj,z,k,ki,kj,kz,dq,dk)
    out = out.sum(axis = 0)
    return out

def propagate_modes_mono(modes,xi,xj,z,k,ki,kj,kz,dq,dk):
    if xp.__name__ == 'mlx.core':
        return propagate_modes_mono_mx(modes,xi,xj,z,k,ki,kj,kz,dq,dk)
    else:
        return propagate_modes_xp(modes,xi,xj,z,k,ki,kj,kz,dq,dk)

def compute_e_field(mask, modes, out = None):
    out = linalg.unpack(mask,modes)
    out[0,0] += xp.sqrt(mask.shape[0]*mask.shape[1])
    return out

class ParticleSimulator:
    def __init__(self, microscope, particles, simulator_numerical_aperture = None, focal_plane = 0, simulator_dof = np.inf, monodisperse_particles = False):
        self.particles = particles
        self.microscope = microscope
        self.reciprocal_plane = microscope.get_reciprocal_plane()
        self.wavelength = self.reciprocal_plane.wavelength
        self.field_plane = microscope.get_object_plane()
        simulator_na = self.particles.n_medium if simulator_numerical_aperture is None else simulator_numerical_aperture
        self.compute_simulator_mask(simulator_na)
        self.compute_scattering_vectors()
        self.monodisperse_particles = monodisperse_particles
        
        self.object_plane_intensity = xp.array(self.microscope.light_source.intensity * self.microscope.magnification**2)

        self.kx = xp.array((self.scattering_vectors[...,0])[self.simulator_mask], dtype=fdtype())
        self.ky = xp.array((self.scattering_vectors[...,1])[self.simulator_mask], dtype=fdtype())
        self.kz = xp.array((self.scattering_vectors[...,2])[self.simulator_mask], dtype=fdtype())
        self.k = xp.array(self.reciprocal_plane.wavenumber * self.particles.n_medium, dtype=fdtype())

        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

        self.modes = None
    
        # adjustment of focal plane also computes the simulator depth of field mask
        self.set_focal_plane(focal_plane, dof = simulator_dof)

    @property
    def sigma_k(self):
        return xp.array(self.microscope.light_source.sigma/self.microscope.light_source.wavenumber, dtype=fdtype())

    @property
    def sigma_q(self):
        na = self.microscope.condenser_numerical_aperture
        refind = self.particles.n_medium
        sigma = na / refind / 2
        return xp.array(sigma, dtype=fdtype())

    @property
    def x(self):
        return xp.array(self.particles.coordinates[:,0] -  self.x0, dtype=fdtype()) 
    
    @property
    def x_dof(self):
        if self.simulator_dof_mask is not None:
            return xp.array(self.particles.coordinates[:,0][self.simulator_dof_mask] -  self.x0, dtype=fdtype())
        else:
            return self.x

    @property
    def y(self):
        return xp.array(self.particles.coordinates[:,1] -  self.y0, dtype=fdtype()) 

    @property
    def y_dof(self):
        if self.simulator_dof_mask is not None:
            return xp.array(self.particles.coordinates[:,1][self.simulator_dof_mask] -  self.y0, dtype=fdtype())
        else:
            return self.y

    @property
    def z(self):
        return xp.array(self.particles.coordinates[:,2] -  self.z0, dtype=fdtype()) 
    
    @property
    def z_dof(self):
        if self.simulator_dof_mask is not None:
            return xp.array(self.particles.coordinates[:,2][self.simulator_dof_mask] -  self.z0, dtype=fdtype())
        else:
            return self.z
        
    def set_simulator_dof(self, dof = np.inf):
        if self.modes is not None:
            if dof == np.inf:
                self.modes_dof = self.modes
                self.simulator_dof_mask = None
            else:
                mask = xp.abs(self.z) <= dof/2.
                modes_np = get_array(self.modes)
                self.modes_dof = xp.array(modes_np[mask])
                self.simulator_dof_mask = mask
        else:
            self.simulator_dof_mask = None
            self.modes_dof = None
        self.simulator_dof = dof  

    def get_simulator_dof(self):
        return self.simulator_dof

    def set_focal_point(self, x =0, y = 0, z = 0, dof = None):
        self.x0 = x
        self.y0 = y
        self.z0 = z
        if dof is not None:
            self.set_simulator_dof(dof)
        else:
            self.set_simulator_dof(self.simulator_dof)

    def get_focal_point(self):
        return self.x0, self.y0, self.z0

    def set_focal_plane(self, focal_plane, dof = None):
        self.z0 = focal_plane
        if dof is not None:
            self.set_simulator_dof(dof)
        else:
            self.set_simulator_dof(self.simulator_dof)

    def get_focal_plane(self):
        return self.z0
        
    def compute_scattering_vectors(self): 
        self.scattering_vectors = self.reciprocal_plane.get_wavevectors(n_medium = self.particles.n_medium)

    def compute_scattering_coefficients(self):
        self.scattering_coefficients = self.particles.compute_scattering_coefficients(sf = self.scattering_vectors[self.simulator_mask], wavelength = self.wavelength, monodisperse = self.monodisperse_particles)
        return self.scattering_coefficients

    def compute_simulator_mask(self, simulator_numerical_aperture):
        # must be a numpy array to be used as a mask for the scattering vectors, which are also numpy arrays
        self.simulator_mask = np.array(self.reciprocal_plane.get_aperture(simulator_numerical_aperture))
        return self.simulator_mask
    
    def compute_modal_coefficient(self):
        self.modes = xp.array(self.scattering_coefficients / self.particles.n_medium * self.reciprocal_plane.pixel_size / self.field_plane.pixel_size, dtype=cdtype())
        self.set_simulator_dof(self.simulator_dof)
        return self.modes

    def compute_lut(self):
        self.compute_scattering_coefficients()
        self.compute_modal_coefficient()
        return self.simulator_mask, self.modes

    def compute_reciprocal_field(self):
        if self.monodisperse_particles:
            modes = propagate_modes_mono(self.modes_dof,self.x_dof,self.y_dof,self.z_dof,self.k,self.kx,self.ky,self.kz,self.sigma_q,self.sigma_k)
        else:
            modes = propagate_modes(self.modes_dof,self.x_dof,self.y_dof,self.z_dof,self.k,self.kx,self.ky,self.kz,self.sigma_q,self.sigma_k)
        field = compute_e_field(self.simulator_mask,modes) * self.microscope.light_source.intensity**0.5
        return field

    def compute_field(self):
        field = self.compute_reciprocal_field()
        return self.microscope.compute_field(field, fft_input = True)

    def compute_intensity(self):
        field = self.compute_reciprocal_field()
        return self.microscope.compute_intensity(field, fft_input = True)

    def compute_image(self):
        field = self.compute_reciprocal_field()
        return self.microscope.compute_image(field, fft_input = True)

def create_simulator(microscope, particles, simulator_numerical_aperture = None, focal_plane = 200e-6, simulator_dof = np.inf, monodisperse_particles = False):
    return ParticleSimulator(microscope=microscope, particles=particles, simulator_numerical_aperture=simulator_numerical_aperture, focal_plane=focal_plane, simulator_dof=simulator_dof, monodisperse_particles=monodisperse_particles)    

__all__ = ["ParticleSimulator", "create_simulator"]