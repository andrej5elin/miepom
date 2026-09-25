"""This module contains classes and functions for simulating Mie scattering by spherical particles.
"""

import miepython as mie
import numpy as np
from .config import FDTYPE, CDTYPE, VECTORIZED_MIE
from .utils import print_progress

DEFAULT_PARTICLE_DIAMETER = 1e-6
DEFAULT_PARTICLE_REFRACTIVE_INDEX = 1.5 
DEFAULT_MEDIUM_REFRACTIVE_INDEX = 1.33

DEFAULT_SAMPLE_THICKNESS = 100e-6
DEFAULT_SAMPLE_HEIGHT = 100e-6
DEFAULT_SAMPLE_WIDTH = 100e-6

ez = np.array([0,0,1], FDTYPE)

def compute_scattering_coefficient(diameter, n_particle = 1.0, wavelength = 550e-9, si = ez, sf = ez, n_medium = DEFAULT_MEDIUM_REFRACTIVE_INDEX):
    # convert scattering and incident directions to arrays
    sf = np.asarray(sf, FDTYPE)
    si = np.asarray(si, FDTYPE)
    # normalize the scattering and incident direction vectors
    sf /= np.linalg.norm(sf,axis = -1, keepdims = True)
    si /= np.linalg.norm(si,axis = -1, keepdims = True)
    # extract particle properties
    d = np.asarray(diameter, FDTYPE)
    n = np.asarray(n_particle, CDTYPE)
    # the cosine of the scattering angle
    mu = np.vecdot(si ,sf)
    mu = np.asarray(mu,FDTYPE)
    # environment wavelength
    wn = wavelength / n_medium 
    # size parameter for Mie scattering
    x = np.pi * d / wn 
    x = np.asarray(x,FDTYPE)
    # relative refractive index
    m = n / n_medium 
    m = np.asarray(m, CDTYPE)
    # computes mie coefficients 
    s1, s2 = mie.S1_S2(m, x, mu, norm = "qsca")

    ssum = (s1+s2)/2.

    # cosine correction factor for input wave
    fcos0 = 1/np.sqrt(np.vecdot(si,ez))
    
    # cosine correction factor for output wave
    fcos1 = 1/np.sqrt(np.vecdot(sf,ez))

    particle_area = np.pi * (d/2)**2

    fact = -1*np.asarray(np.sqrt(particle_area))

    if fact.ndim == 0:
        scale = fcos1*fact[None]
    else:
        scale = fcos1*fact[:,None]

    out = np.multiply(fcos0,scale, out = s2)
    out = np.multiply(out, ssum, out = out)    

    return out

class MieParticles:
    def __init__(self, count, diameter = DEFAULT_PARTICLE_DIAMETER, n_particle = DEFAULT_PARTICLE_REFRACTIVE_INDEX, n_medium = DEFAULT_MEDIUM_REFRACTIVE_INDEX, coordinates = None):
        self.diameter = np.asarray(diameter, FDTYPE)
        self.n_particle = np.asarray(n_particle, CDTYPE)
        self.n_medium = n_medium
        self.count = count
        if coordinates is not None:
            self.set_coordinates(coordinates)
        if coordinates is None:
            self.coordinates = np.zeros((count,3,), FDTYPE)
        if len(self.diameter * self.n_particle * self.coordinates[...,0]) != count:
            raise ValueError("The length of the diameter, n_particle, and coordinates arrays must match the count of particles.")

    def set_coordinates(self, coordinates):
        if len(coordinates) != self.count:
            raise ValueError("The length of the coordinates array must match the count of particles.")
        self.coordinates = np.asarray(coordinates, FDTYPE)
        if self.coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be a 2D array with shape (count, 3).")

    def compute_scattering_coefficients(self, wavelength = 550e-9, si = ez, sf = ez, monodisperse = False):
        """Compute the scattering coefficients for the particles.

        Parameters
        ----------
        wavelength : float
            Wavelength of the incident light in meters.
        si : array_like
            Incident wave vector direction.
        sf : array_like
            Scattered wave vector direction.
        monodisperse : bool
            If True, use the mean diameter and refractive index for all particles.

        Returns
        -------
        out : array_like
            Scattering coefficients for the particles.
        """
        diameter = self.diameter
        n = self.n_particle
        n_medium = self.n_medium

        if monodisperse :
            diameter = np.mean(diameter)
            n = np.mean(n)
            out = compute_scattering_coefficient(diameter = diameter, n_particle= n, wavelength = wavelength, si = si, sf = sf, n_medium = n_medium)

        else:
            if VECTORIZED_MIE:
                out = compute_scattering_coefficient(diameter = diameter, n_particle= n, wavelength = wavelength, si = si, sf = sf, n_medium = n_medium)
            else:
                out = []
                for i, (d, ni) in enumerate(zip(diameter, n)):
                    print_progress(i, self.count)
                    out.append(compute_scattering_coefficient(diameter = d, n_particle= ni, wavelength = wavelength, si = si, sf = sf, n_medium = n_medium))
                print_progress(self.count, self.count)
                out = np.array(out)
        return out
        
    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "diameter": self.diameter[index],
            "n_particle": self.n_particle[index],
            "n_medium": self.n_medium,
            "coordinates": self.coordinates[index]
        }

    def plot_distribution(self, ax_diameter = None, ax_refractive_index = None):
        if ax_diameter is None and ax_refractive_index is None:
            import matplotlib.pyplot as plt
            fig, (ax_diameter, ax_refractive_index) = plt.subplots(1, 2, figsize=(12, 5))
        self.plot_diameter_distribution(ax = ax_diameter)
        self.plot_refractive_index_distribution(ax = ax_refractive_index)

    def plot_diameter_distribution(self,ax = None):
        if self.diameter.ndim == 0 or len(self.diameter) == 0:
            print("No particles in the sample.")
            return
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.diameter, bins = 20)
        ax.set_xlabel("Diameter (m)")
        ax.set_ylabel("Count")
        ax.set_title("Particle diameter distribution")
        if ax is None:
            plt.show()

    def plot_refractive_index_distribution(self, ax = None):
        if self.n_particle.ndim == 0 or len(self.n_particle) == 0:
            print("No particles in the sample.")
            return

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.n_particle.real, bins = 20)
        ax.set_xlabel("Refractive Index")
        ax.set_ylabel("Count")
        ax.set_title("Particle refractive index distribution")
        if ax is None:
            plt.show()

    def plot_coordinates(self, ax = None):
        # Convert from meters to micrometers for readability
        coords_um = self.coordinates * 1e6

        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            coords_um[:, 0],
            coords_um[:, 1],
            coords_um[:, 2],
            c=coords_um[:, 2],
            cmap="viridis",
            s=6,
            alpha=0.7
        )

        ax.set_title("3D Distribution of Particle Coordinates")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_zlabel("z (µm)")
        fig.colorbar(scatter, ax=ax, label="z (µm)")



def create_mie_particles(count, diameter = DEFAULT_PARTICLE_DIAMETER, n_particle = DEFAULT_PARTICLE_REFRACTIVE_INDEX, n_medium = DEFAULT_MEDIUM_REFRACTIVE_INDEX, coordinates = None):
    particles = MieParticles(count = count, diameter = diameter, n_particle = n_particle, n_medium = n_medium)
    if coordinates is not None:
        particles.set_coordinates(coordinates)
    return particles

__all__ = ["MieParticles", "create_mie_particles"]