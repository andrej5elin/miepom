"""This module contains classes and functions for handling coordinate systems and plotting for field and reciprocal planes in optical simulations.
"""

import numpy as np

class Coordinates:    
    def __init__(self, height = 10e-6, width = 10e-6, shape = (256,256)):
        self.height = height
        self.width = width
        self.shape = shape

    def cartesian_function(self, f):
        """A decorator function for functions expressed in cartesian coordinates"""
        x,y = self.get_cartesian_meshgrid()
        def _f(*args,**kwargs):
            return f(x,y,*args,**kwargs)
        return _f

    def polar_function(self, f):
        """A decorator function for functions expressed in polar coordinates"""
        r,phi = self.get_polar_meshgrid()
        def _f(*args,**kwargs):
            return f(r,phi,*args,**kwargs)
        return _f
                   
    @property
    def pixel_size(self):
        return (self.width*self.height/self.shape[1]/self.shape[0])**0.5

    @property
    def pixel_aspect(self):
        return self.width / self.height * self.shape[0] / self.shape[1]
    
    def print_info(self):
        print(f"Height : {self.height*1000:.2f} mm @ {self.shape[0]} pixels.")
        print(f"Width : {self.width*1000:.2f} mm @ {self.shape[1]} pixels.")
        print(f"Pixel size : {self.pixel_size*1000000:.2f} um")
        print(f"Pixel aspect ratio : {self.pixel_aspect}")
        
    def _get_x_coordinates(self):
        pixel_width = self.width/self.shape[1]
        return np.fft.fftfreq(self.shape[1],1/self.shape[1]) * pixel_width

    def _get_y_coordinates(self):
        pixel_height = self.height/self.shape[0]
        return np.fft.fftfreq(self.shape[0],1/self.shape[0]) * pixel_height

    def get_cartesian_meshgrid(self):
        x,y = np.meshgrid(self._get_x_coordinates(),self._get_y_coordinates(), indexing = "xy")
        return x,y

    def get_polar_meshgrid(self):
        x,y = self.get_cartesian_meshgrid()
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        return r, phi
        
class PlotCoordinates(Coordinates):
    def _prepare_data(self,data):
        if data.shape != self.shape:
            raise ValueError("Invalid data shape.")
        return np.fft.fftshift(data)
    
    def get_x_ticks(self,n, fmt = "{:.2e}"):
        xt = np.linspace(-0.5, 2*(self.shape[1]//2)+0.5, n)
        xl = np.linspace(-0.5*self.width,0.5*self.width, n)
        xl = [fmt.format(x*1000) for x in xl]
        return xt,xl
    
    def get_y_ticks(self,n, fmt = "{:.2e}"):
        yt = np.linspace(-0.5, 2*(self.shape[0]//2)+0.5, n)
        yl = np.linspace(-0.5*self.height,0.5*self.height, n)
        yl = [fmt.format(y*1000) for y in yl]
        return yt,yl
    
    def get_imshow_kwargs(self):
        kwargs = {}
        kwargs.setdefault("aspect", 1/self.pixel_aspect)
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("interpolation", "none")
        return kwargs
    
    def set_xlim(self,ax, xlim = None):
        if xlim:
            x1,x2 = xlim
            x1 = (self.width/2 + x1)/self.width * self.shape[1]
            x2 = (self.width/2 + x2)/self.width * self.shape[1]
            ax.set_xlim((x1,x2))
            
    def set_ylim(self,ax, ylim = None):
        if ylim:
            y1,y2 = ylim
            y1 = (self.height/2 + y1)/self.height * self.shape[0]
            y2 = (self.height/2 + y2)/self.height * self.shape[0]
            ax.set_ylim((y1,y2))        
        
    def _set_labels(self,ax):
        
        ax.set_xlabel("$x [mm]$")
        ax.set_ylabel("$y [mm]$")   

    def imshow(self,data, ax = None, nticks = (5,5), xlim = None, ylim = None, 
               **kwargs):
        
        data = self._prepare_data(data)
        
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            
        d = self.get_imshow_kwargs()
        d.update(kwargs)

        ax.set_xticks(*self.get_x_ticks(nticks[0]))
        ax.set_yticks(*self.get_y_ticks(nticks[1]))

        self._set_labels(ax)
        
        im = ax.imshow(data, **d)
        
        self.set_xlim(ax,xlim)
        self.set_ylim(ax,ylim)

        return im
    
class FieldPlane(PlotCoordinates):
    def get_r_phi(self):
        return self.get_polar_meshgrid()
    
    def get_x_y(self):
        return self.get_cartesian_meshgrid()
    
    def reciprocal_plane(self, wavelength):
        """Return the corresponding reciprocal plane."""
        return ReciprocalPlane(wavelength*self.shape[0]/self.height,wavelength*self.shape[1]/self.width,self.shape, wavelength = wavelength)

    def fourier_plane(self, wavelength, focal_length):
        """Return the lens Fourier plane corresponding to this real plane."""
        return FieldPlane(focal_length*wavelength*self.shape[0]/self.height,focal_length*wavelength*self.shape[1]/self.width,self.shape)
    
class ReciprocalPlane(PlotCoordinates):
    def __init__(self, height = 10e-6, width = 10e-6, shape = (256,256), wavelength = 1e-6):
        super().__init__(height = height, width = width, shape = shape)
        self.wavelength = wavelength

    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    def field_plane(self):
        return FieldPlane(self.wavelength*self.shape[0]/self.height,self.wavelength*self.shape[1]/self.width,self.shape)
    
    def get_beta_phi(self):
        return self.get_polar_meshgrid()
    
    def get_betax_betay(self):
        return self.get_cartesian_meshgrid()

    def get_x_ticks(self,n, fmt = "{:.2f}"):
        xt = np.linspace(-0.5, 2*(self.shape[1]//2)+0.5, n)
        xl = np.linspace(-self.width/2,self.width/2, n)
        xl = [fmt.format(x) for x in xl]
        return xt,xl
    
    def get_y_ticks(self,n, fmt = "{:.2f}"):
        yt = np.linspace(-0.5, 2*(self.shape[0]//2)+0.5, n)
        yl = np.linspace(-self.height/2,self.height/2, n)
        yl = [fmt.format(y) for y in yl]
        return yt,yl
    
    def _set_labels(self,ax):
        ax.set_xlabel(r"$\beta_x$")
        ax.set_ylabel(r"$\beta_y$")  
        
    def get_mask(self, betamax = 1):
        r,phi = self.get_polar_meshgrid()
        return r < betamax

    def get_aperture(self, na = 1.0):
        """Return the aperture function for a given numerical aperture and medium refractive index."""
        return self.get_mask(betamax = na)

    def get_wavevectors(self, n_medium = 1.0):
        beta, phi = self.get_polar_meshgrid()
        
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        sintheta = beta/n_medium

        mask = sintheta > 1
        sintheta[mask] = np.nan
        
        theta = np.arcsin(sintheta)

        svec = np.zeros(shape = beta.shape + (3,), dtype = np.float32)
        svec[...,0] = cosphi * sintheta
        svec[...,1] = sinphi * sintheta
        svec[...,2] = np.cos(theta)
        svec *= (self.wavenumber * n_medium)
        return svec 

    __all__ = ["FieldPlane", "ReciprocalPlane"]
