import pyelectrostatic, pyelectrostatic.fourier_calculator as calc
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.io


def test_remove():
    # Go in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    atms = ase.io.read("rhombo-002.pwo")

    BaTiO3 = CC.Phonons.Phonons("../BaTiO3_")
    calculator = calc.FourierCalculator()
    calculator.init_from_dyn(BaTiO3)

    atms.set_calculator(calculator)
    print(atms.get_total_energy())
    #print(atms.get_forces())

if __name__ == "__main__":
    test_remove()
