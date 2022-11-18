from numpy.distutils.core import setup, Extension
import sys

def readme():
    with open("README.md") as f:
        return f.read()



setup( name = "py-electrostatic",
       version = "0.1",
       description = "Python utility to remove long-range electrostatic Coulomb interactions from atomic forces",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/py-electristatic",
       packages = ["pyelectrostatic"],
       package_data={"": ["*.jl"]},
       package_dir = {"pyelectrostatic": "modules"},
       install_requires = ["numpy", "ase", "scipy", "cellconstructor"],
       license = "GPLv3",
       scripts = [],
       ext_modules = [])
