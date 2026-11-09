# miepom
A python library for simulating Polarizing Optical Microscopy of Mie scatterers (spherical particles)

## Installation

Requires Python 3.9 or later. Install miepom and its required dependencies from
the repository:

```bash
git clone https://github.com/andrej5elin/miepom.git
cd miepom
python -m pip install .
```

Optional backend dependencies can be installed with extras, for example:

```bash
python -m pip install '.[numba]'
```

Available extras are `cupy`, `mlx`, `numba`, and `scipy`.
