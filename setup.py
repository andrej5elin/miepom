from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="miepom",
    version="0.1.0",
    description="A python library for simulating Polarizing Optical Microscopy of Mie scatterers",
    long_description=README,
    long_description_content_type="text/markdown",
    author="andrej5elin",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "miepython",
        "numpy",
    ],
    extras_require={
        "cupy": ["cupy"],
        "mlx": ["mlx"],
        "numba": ["numba"],
        "scipy": ["scipy"],
    },
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
