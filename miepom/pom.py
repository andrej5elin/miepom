"""This module contains classes and functions for simulating a non-polarizing microscope system, including light source and sensor configurations.

In the future, this module may be extended to include additional microscope types, such as polarizing microscopes"""

import numpy as np

from .linalg import abs2
from .config import  USENUMBA, fdtype, cdtype
from .sensor import Sensor

DEFAULT_OBJECTIVE_MAGNIFICATION = 10
DEFAULT_OBJECTIVE_NUMERICAL_APERTURE = 0.25
DEFAULT_TUBE_LENS_FOCAL_LENGTH = 200e-3
DEFAULT_CONDENSER_NUMERICAL_APERTURE = 0.25
DEFAULT_LIGHT_SOURCE_WAVELENGTH = 550e-9
DEFAULT_LIGHT_SOURCE_BANDWIDTH = 20e-9
DEFAULT_LIGHT_SOURCE_POWER = 1.0

# Set the default linear algebra backend to numpy so that we can compile and run numba functions correctly.
globals()["xp"] = np
globals()["usenumba"] = USENUMBA

def set_backend(xp):
    name = xp.__name__
    globals()["xp"] = xp
    
def enable_numba():
    old = globals()["usenumba"]
    globals()["usenumba"] = True
    return old

def disable_numba():
    old = globals()["usenumba"]
    globals()["usenumba"] = False
    return old

def backend_name():
    """Return the name of the current sensor backend."""
    name = globals()["xp"].__name__
    if name == "numpy" and globals()["usenumba"]:
        return "numba"
    return name

def lens_transform(a, input_pixel_size = 1, output_pixel_size = 1, fft_input = False):
    """Apply a lens transform to a 2D array."""
    scale = -1j * input_pixel_size / output_pixel_size
    a = xp.asarray(a)
    if fft_input:
        return a * scale
    else:
        return scale * xp.fft.fft2(a, norm = "ortho")

class LightSource:
    def __init__(self, wavelength = 0.5e-6, bandwidth = 28e-9, intensity = 1.0):
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.intensity = intensity  

    @property
    def sigma(self):
        return (2*np.pi/(self.wavelength - self.bandwidth/2) - 2*np.pi/(self.wavelength + self.bandwidth/2))/(np.sqrt(8 * np.log(2)))

    @property
    def frequency(self):
        return 3e8 / self.wavelength    

    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    def print_info(self):
        print("Light source settings")
        print("----------------------")
        print(f"Intensity at object plane:  {self.intensity:.2f}")
        print(f"Wavelength:                 {self.wavelength*1e9:.2f} nm")
        print(f"Bandwidth (FWHM):           {self.bandwidth*1e9:.2f} nm")
        print(f"Frequency:                  {self.frequency:.2e} Hz")
        print(f"Wavenumber:                 {self.wavenumber:.2e} 1/m")
        print(f"Wavenumber spread (sigma):  {self.sigma:.2e} 1/m")

    def plot_spectrum(self,ax=None):
        """Plots the spectrum of the light source"""
        from matplotlib import pyplot as plt

        sigma = self.sigma
        k0 = self.wavenumber
        k0s = np.linspace(k0 - 5*sigma, k0 + 5*sigma, 1000)

        k_weights = np.exp(-(k0s-k0)**2/2/sigma**2)
        k_weights /= k_weights.sum()

        if ax is None:
            ax = plt.gca()

        ax.plot(np.pi*2/k0s*1e9, k_weights)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Light source spectrum")
        ax.grid()
        if ax is None:
            plt.show()

def create_light_source(wavelength = 0.5e-6, bandwidth = 28e-9, intensity = 1.0):
    return LightSource(wavelength = wavelength, bandwidth = bandwidth, intensity = intensity)

class NonpolarizingMicroscope:
    def __init__(self,light_source : LightSource, sensor : Sensor, magnification = 60.0, tube_lens_focal_length = 200e-3, objective_numerical_aperture = 0.8, condenser_numerical_aperture = 0.2, immersion_medium_refractive_index = 1.):
        self.light_source = light_source
        self.magnification = magnification
        self.tube_lens_focal_length = tube_lens_focal_length
        self.condenser_numerical_aperture = condenser_numerical_aperture
        self.objective_numerical_aperture = objective_numerical_aperture
        self.immersion_medium_refractive_index = immersion_medium_refractive_index
        self.sensor = sensor
        self.set_pupil_function(self.get_effective_aperture())
        
        self.object_plane_pixel_size = self.get_object_plane().pixel_size
        self.fourier_plane_pixel_size = self.get_fourier_plane().pixel_size
        self.image_plane_pixel_size = self.get_image_plane().pixel_size

    def get_objective_aperture(self):
        reciprocal_plane = self.get_reciprocal_plane()
        return reciprocal_plane.get_aperture(na = self.objective_numerical_aperture)

    def get_condenser_aperture(self):
        reciprocal_plane = self.get_reciprocal_plane()
        return reciprocal_plane.get_aperture(na = self.condenser_numerical_aperture)

    def get_effective_aperture(self):
        """Returns the effective aperture of the microscope system, which is a convolution of the objective and condenser apertures."""
        objective_aperture = self.get_objective_aperture()
        condenser_aperture = self.get_condenser_aperture()
        if condenser_aperture.sum() > 0:  
            # perform convolution using fft.     
            a = np.fft.irfft2(np.fft.rfft2(condenser_aperture)*np.fft.rfft2(objective_aperture))
            a = a / condenser_aperture.sum()
            return xp.asarray(a, dtype=fdtype())
        else:
            return xp.asarray(objective_aperture, dtype=fdtype())

    def get_fft_mask(self):
        """Returns the FFT mask corresponding to the pupil function"""
        reciprocal_plane = self.get_reciprocal_plane()
        return reciprocal_plane.get_aperture(na = self.immersion_medium_refractive_index)

    def set_pupil_function(self, pupil_function):
        """Sets the pupil function of the microscope system. The pupil function is a complex-valued function that describes the amplitude and phase of the light passing through the objective lens."""
        self.pupil_function = xp.asarray(pupil_function, dtype=fdtype())

    def get_pupil_function(self):
        """Returns the pupil function of the microscope system"""
        return self.pupil_function
        
    @property
    def objective_focal_length(self):
        return self.tube_lens_focal_length / self.magnification

    def get_fourier_plane(self):
        """Returns a FieldPlane object representing the Fourier plane of the microscope system"""
        return self.sensor.get_image_plane().fourier_plane(focal_length = self.tube_lens_focal_length, wavelength = self.light_source.wavelength)

    def get_object_plane(self):
        """Returns a FieldPlane object representing the object plane of the microscope system"""
        return self.get_fourier_plane().fourier_plane(focal_length = self.objective_focal_length, wavelength = self.light_source.wavelength)

    def get_image_plane(self):
        """Returns a FieldPlane object representing the image plane of the microscope system"""
        return self.sensor.get_image_plane()

    def get_reciprocal_plane(self):
        """Returns a FieldPlane object representing the reciprocal plane of the microscope system"""
        return self.get_object_plane().reciprocal_plane(wavelength = self.light_source.wavelength)

    def print_info(self, full = True):
        print("Microscope settings")
        print("----------------------")
        print(f"Magnification:                     {self.magnification}")
        print(f"Tube lens focal length:            {self.tube_lens_focal_length:.2e} m")
        print(f"Immersion medium refractive index: {self.immersion_medium_refractive_index}")
        print(f"Objective numerical aperture:      {self.objective_numerical_aperture}")
        print(f"Condenser numerical aperture:      {self.condenser_numerical_aperture}")
        print()
        if full:
            print("Light source info:")
            self.light_source.print_info()
            print()
            print("Sensor info:")
            self.sensor.print_info()

    def compute_field(self, object_field, fft_input = False):
        """Transfers input field through the microscope system and returns the resulting field in the image plane."""
        # apply pupil function to the scene
        fourier_field = lens_transform(object_field, input_pixel_size = self.object_plane_pixel_size, output_pixel_size = self.fourier_plane_pixel_size, fft_input = fft_input)
        fourier_field = fourier_field * self.pupil_function
        image_field = lens_transform(fourier_field, input_pixel_size = self.fourier_plane_pixel_size, output_pixel_size = self.image_plane_pixel_size, fft_input = False)
        return image_field

    def compute_intensity(self, object_field, fft_input = False):
        """Transfers input field through the microscope system and returns the resulting intensity in the image plane."""
        image_field = self.compute_field(object_field, fft_input = fft_input)
        return abs2(image_field)

    def compute_image(self, object_field, fft_input = False):
        """Transfers input field through the microscope system and returns the image captured by the sensor."""
        intensity = self.compute_intensity(object_field, fft_input = fft_input)
        return self.sensor.capture(intensity)

def create_microscope(light_source : LightSource, sensor : Sensor, magnification = 60.0, tube_lens_focal_length = 200e-3, objective_numerical_aperture = 0.8, condenser_numerical_aperture = 0.2, immersion_medium_refractive_index = 1.):
    return NonpolarizingMicroscope(light_source = light_source, sensor = sensor, magnification = magnification, tube_lens_focal_length = tube_lens_focal_length, objective_numerical_aperture = objective_numerical_aperture, condenser_numerical_aperture = condenser_numerical_aperture, immersion_medium_refractive_index = immersion_medium_refractive_index)

__all__ = ["NonpolarizingMicroscope", "create_microscope", "LightSource", "create_light_source", "lens_transform"]