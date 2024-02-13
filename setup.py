from setuptools import setup
import sys

def readme():
    with open("README.md") as f:
        return f.read()



setup( name = "py-electrostatic",
       version = "0.1",
       description = "Force-field to calculate long-range electrostatic interactions in polar materials",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/py-electristatic",
       packages = ["pyelectrostatic"],
       package_data={"": ["*.jl"]},
       package_dir = {"pyelectrostatic": "modules"},
       install_requires = ["numpy", "ase", "scipy", "cellconstructor", "julia"],
       license = "GPLv3",
       scripts = [],
       ext_modules = [])
