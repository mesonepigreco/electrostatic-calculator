import pyelectrostatic, pyelectrostatic.calculator as calc
import sys, os

def test_unit_cell():
    """
    Simple test to check if forces are derivatives of the energy
    """

    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    

