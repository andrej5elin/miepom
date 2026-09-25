"""This module contains classes and functions for simulating image formation on a digital sensor, including ADC conversion and noise models.
In the future, this module may be extended to include additional sensor types, such as color sensors and polarizing sensors."""


from .config import USENUMBA, nbfloat, nbcomplex, nbuint, NBFASTMATH, NBTARGET, fdtype, vectorize, guvectorize, jit, njit, mx_compile, mx_compile_out
import numpy as np
from .utils import print_line
from .plane import FieldPlane

# Set the default linear algebra backend to numpy so that we can compile and run numba functions correctly.
globals()["xp"] = np
globals()["usenumba"] = USENUMBA

def set_backend(xp):
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

DEFAULT_ADCBIT = 12
DEFAULT_FULL_WELL_CAPACITY = 2**15
DEFAULT_SENSOR_SHAPE = (512,512)
DEFAULT_PIXEL_SIZE = 6e-6  # in meters, typical for modern image sensors

SENSOR_DISABLE_ADC = 1
SENSOR_DISABLE_SHOT_NOISE = 2
SENSOR_NORMAL_NOISE_MODEL = 4

# make a reference to the original numpy clip function
clip_np = np.clip

# numba does not support np.clip in vectorized functions, but we need it in sensor_image function
@jit([nbfloat(nbfloat,nbfloat,nbfloat)])
def clip_nb(a,low,high):
    if a < low:
        return low
    if a > high:
        return high
    else:
        return a

# temporely override xp.clip with the numba-compatible clip function so that we can compile the sensor_image function with numba. This is a hack, but it works.
xp.clip = clip_nb

#pure python implementation, suitable for all backends, but slow. This is used a a source for the numba compiled version
def _sensor_image_py(im, transmittance = 1.0, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, flags = 0):
    im = im * transmittance
    if not (flags & SENSOR_DISABLE_SHOT_NOISE):
        if flags & SENSOR_NORMAL_NOISE_MODEL:
            scale = (im / full_well_capacity)**0.5
            nim = xp.random.normal(loc = im, scale = scale) 
            nim *= full_well_capacity
        else:
            lam = im * full_well_capacity
            nim = xp.random.poisson(lam)
        im = xp.round(nim)

    if not (flags & SENSOR_DISABLE_ADC):
        # ADC conversion. A full well with signal of 1.0 gives max byte value  
        # first scale and clip
        norm = (2**adcbit-1)
        scale = norm / full_well_capacity
        im = xp.clip(im * scale, -0.5, norm+0.5)
        # create digital image by rounding then convert back to float for float representation of the uint image
        im = xp.round(im)
        im/=norm
    else:
        im/= full_well_capacity
    return im

# make a numba-optimized function.
_sensor_image_nb = vectorize([nbfloat(nbfloat,nbfloat,nbfloat,nbfloat,nbuint)], target = NBTARGET, fastmath = NBFASTMATH)(_sensor_image_py)

# ok, we have compuledn the numba version of the sensor_image function, now we can restore the original numpy clip function to xp.clip
xp.clip = clip_np

def __sensor_image_mx(im, transmittance = 1.0, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, flags = 0):
    # compute the sensor image using mlx.core as the backend. This function is compiled with mlx.core.compile to optimize performance.
    # difference is that we suply shape argument to the random function, which is required to work correctly with mlx.core's random number generation.
    im = im * transmittance
    if not (flags & SENSOR_DISABLE_SHOT_NOISE):
        scale = (im / full_well_capacity)**0.5
        nim = xp.random.normal(shape = im.shape,loc = im, scale = scale) 
        nim *= full_well_capacity
        im = xp.round(nim)

    if not (flags & SENSOR_DISABLE_ADC):
        # ADC conversion. A full well with signal of 1.0 gives max byte value  
        # first scale and clip
        norm = (2**adcbit-1)
        scale = norm / full_well_capacity
        im = xp.clip(im * scale, -0.5, norm+0.5)
        # create digital image by rounding then convert back to float for float representation of the uint image
        im = xp.round(im)
        im/=norm
    else:
        im/= full_well_capacity
    return im

_sensor_image_mx = mx_compile_out(__sensor_image_mx)

def _sensor_image_xp(im, transmittance = 1.0, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, flags = 0, out = None):
    im = np.multiply(im, transmittance, out = out)
    if not (flags & SENSOR_DISABLE_SHOT_NOISE):
        if flags & SENSOR_NORMAL_NOISE_MODEL:
            scale = (im / full_well_capacity)
            scale = np.sqrt(scale, out = scale)
            nim = xp.random.normal(loc = im, scale = scale) 
            nim *= full_well_capacity
        else:
            lam = im * full_well_capacity
            nim = xp.random.poisson(lam)
        im = xp.round(nim, out = im)

    if not (flags & SENSOR_DISABLE_ADC):
        # ADC conversion. A full well with signal of 1.0 gives max byte value  
        # first scale and clip
        norm = (2**adcbit-1)
        scale = norm / full_well_capacity
        im = xp.clip(im * scale, -0.5, norm+0.5, out = im)
        # create digital image by rounding then convert back to float for float representation of the uint image
        im = xp.round(im, out = im)
        im/=norm
    else:
        im/= full_well_capacity
    return im

def sensor_image(im, transmittance = 1.0, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, flags = 0, out = None):
    """Computes sensor repsonse from the expected relative image value and sensor window function"""
    if backend_name() == "numba":
        return _sensor_image_nb(im, transmittance, adcbit , full_well_capacity, flags, out = out)
    elif xp.__name__ == "mlx.core":
        return _sensor_image_mx(im, transmittance, adcbit , full_well_capacity, flags, out = out)
    else:
        # numpy, cupy, custom backends
        return _sensor_image_xp(im, transmittance, adcbit , full_well_capacity, flags, out = out)

class Sensor():
    """A class representing a sensor with specific settings. It can capture images of a scene using the sensor settings."""
    def __init__(self, shape = DEFAULT_SENSOR_SHAPE, pixel_size = DEFAULT_PIXEL_SIZE, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, transmittance = 1.0, pixel_aspect = 1.0, flags = 0):
        self.shape = shape
        self.pixel_size = pixel_size
        self.pixel_aspect = pixel_aspect
        self.adcbit = adcbit
        self.full_well_capacity = full_well_capacity
        self.transmittance = transmittance
        self.flags = flags
        self.set_exposure(0.5)

    @property
    def pixel_height(self):
        return (self.pixel_size**2/self.pixel_aspect)**0.5

    @property
    def pixel_width(self):
        return (self.pixel_size**2*self.pixel_aspect)**0.5

    @property
    def height(self):
        return self.shape[0] * self.pixel_height

    @property
    def width(self):
        return self.shape[1] * self.pixel_width

    @property
    def diagonal(self):
        return (self.height**2 + self.width**2)**0.5
    
    def capture(self, scene, out = None):
        """Captures an image of the given scene using the sensor settings"""
        return sensor_image(scene, transmittance = self.multiplier, adcbit=self.adcbit, full_well_capacity=self.full_well_capacity, flags=self.flags, out=out)
    
    def set_exposure(self, exposure):
        """Sets the exposure level for the sensor"""
        self.exposure = exposure
        self.multiplier = exposure * self.transmittance

    def get_exposure(self):
        """Returns the current exposure level of the sensor"""
        return self.exposure

    def print_info(self):
        print("Sensor settings")
        print_line("-")
        print(f"Sensor shape:               {self.shape}")
        print(f"Sensor height:              {self.height*1e3:.2f} mm")
        print(f"Sensor width:               {self.width*1e3:.2f} mm")
        print(f"Sensor diagonal:            {self.diagonal*1e3:.2f} mm")
        print(f"Pixel size:                 {self.pixel_size*1e6:.2f} um")
        print(f"Pixel aspect:               {self.pixel_aspect}")
        print(f"Exposure level:             {self.exposure:.2f}")
        print(f"Full well capacity:         {self.full_well_capacity} e-")
        print(f"ADC applied:                {'No' if self.flags & SENSOR_DISABLE_ADC else 'Yes'}")
        if not (self.flags & SENSOR_DISABLE_ADC):
            print(f"ADC bit depth:              {self.adcbit} bits")
        print(f"Shot noise applied:         {'No' if self.flags & SENSOR_DISABLE_SHOT_NOISE else 'Yes'}")
        if not (self.flags & SENSOR_DISABLE_SHOT_NOISE):
            print(f"Shot noise model:           {'Normal' if self.flags & SENSOR_NORMAL_NOISE_MODEL else 'Poisson'}")
        print()
        print_line("=")

    def get_image_plane(self):
        """Returns a FieldPlane object representing the image plane of the sensor"""
        return FieldPlane(height=self.height, width=self.width, shape=self.shape)


class MultiSensor():
    """A class representing a multi-sensor setup, where multiple sensors can capture images of a scene simultaneously."""
    def __init__(self, sensors):
        self.sensors = sensors

    def set_exposure(self, exposure):
        """Sets the exposure level for all sensors in the multi-sensor setup"""
        for sensor in self.sensors:
            sensor.set_exposure(exposure)

    def capture(self, scene, out = None):
        """Captures images of the given scene using all the sensors in the setup"""
        results = []
        for sensor in self.sensors:
            results.append(sensor.capture(scene, out = out))
        return results

def create_sensor(shape, pixel_size = DEFAULT_PIXEL_SIZE, adcbit = DEFAULT_ADCBIT, full_well_capacity = DEFAULT_FULL_WELL_CAPACITY, transmittance = 1.0, pixel_aspect = 1.0, flags = 0):
    height, width = shape
    return Sensor(shape=shape, pixel_size=pixel_size, pixel_aspect=pixel_aspect, transmittance=transmittance, full_well_capacity=full_well_capacity, adcbit=adcbit, flags=flags)

def spot(shape, center = (0,0), diameter = 4, attenuation = 0.1):
    h,w = shape 
    yn = np.fft.fftfreq(h,d = 1/h)
    xn = np.fft.fftfreq(w,d = 1/w)
    y,x = np.meshgrid(yn,xn, indexing = 'ij')
    y0,x0 = center
    sigma = diameter / (8*np.log(2))**0.5 #sigma from full width half maximum (diameter)
    window = 1. - attenuation*np.exp(-((x-x0)**2 + (y-y0)**2)/2/sigma**2)
    return xp.array(window, dtype=fdtype())

def blackman(shape):
    """Returns a Blackman window function"""
    h,w = shape
    
    yn = np.fft.fftfreq(h,d = 1/h)
    xn = np.fft.fftfreq(w,d = 1/w)
    y,x = np.meshgrid(yn,xn, indexing = 'ij')

    phase = np.sqrt((2*np.pi*x/w)**2 + (2*np.pi*y/h)**2)
    mask = np.abs(phase) > np.pi
    window = (np.cos(phase)+1)/2
    window[mask] = 0.
    window = np.fft.fftshift(window)
    return xp.array(window, dtype=fdtype())

def ones(shape):
    """Returns a noisless bright sensor image aray"""
    return xp.ones(shape, dtype=fdtype())

def zeros(shape):
    """Returns a noisless dark sensor image aray"""
    return xp.zeros(shape, dtype=fdtype())

def create_dust_image(shape, n = 50, attenuation_min = 0.4, attenuation_max = 0.8, diameter_min = 1, diameter_max = 6):
    dust = 1.
    for i in range(n):
        a = attenuation_min + np.random.rand()*(attenuation_max - attenuation_min)
        d = np.random.rand()*(diameter_max-diameter_min)+ diameter_min
        
        dust = dust * spot(shape = shape,center =
        (np.random.randint(-shape[0]//2+diameter_max,shape[0]//2-diameter_max),
         np.random.randint(-shape[1]//2+diameter_max,shape[1]//2-diameter_max)), 
                           attenuation = a, 
                           diameter = d)   
    return xp.array(dust, dtype=fdtype())

__all__ = ["Sensor", "create_sensor", "spot", "blackman", "ones", "zeros", "create_dust_image"]