[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24091326.svg)](https://doi.org/10.5281/zenodo.24091326)



# miepom
A python library for simulating Polarizing Optical Microscopy of Mie scatterers (spherical particles)

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
