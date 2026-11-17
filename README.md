[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24091326.svg)](https://doi.org/10.5281/zenodo.24091326)



# miepom
A python library for simulating Polarizing Optical Microscopy of Mie scatterers (spherical particles)

Currently, the package supports unpolarized-light microscopy of weakly
scattering particles.

The package is a backend that is being used to study particle dynamics in digital Fourier
microscopy. See [mie_particle_motion](https://github.com/andrej5elin/mie_particle_motion)
for use case and additional examples beyond the quickstart example that is delivered in this
repository.

## Installation

Requires Python 3.9 or later. Install miepom and its required dependencies from
PyPI:

```bash
python -m pip install miepom
```

Optional backend dependencies can be installed with extras, for example:

```bash
python -m pip install 'miepom[numba]'
```

Available extras are `cupy`, `mlx`, `numba`, and `scipy`.
