import torch
from mace.calculators import MACECalculator
from ase import Atoms
import numpy as np

calc_orig = MACECalculator(model_paths="MACE_model_original_stagetwo.model", device='cuda')
calc_mod = MACECalculator(model_paths="MACE_model_electrostatics_stagetwo.model", device='cuda')


atoms = Atoms('AuO', positions=[[0, 0, 0], [0, 0, 8.0]])
atoms.center(vacuum=10.0)
atoms.pbc = False 

atoms.info['charge'] = 1.0 

print(f"--- Long Range Force Test (Separation: {atoms.get_distance(0,1):.2f} A) ---")

atoms.calc = calc_orig
atoms.calc.r_max = 15.0 
forces_orig = atoms.get_forces()

print("\n[ ORIGINAL MACE ]")
print(f"Force on O (Z): {forces_orig[1, 2]:.8f} eV/A")
print(f"Charges:        N/A (Standard MACE does not predict charges)")

atoms.calc = calc_mod
atoms.calc.r_max = 15.0 
forces_mod = atoms.get_forces()
predicted_charges = atoms.calc.results.get('charges', np.array([0,0]))

print("\n[ MODIFIED 4G MACE ]")
print(f"Force on O (Z): {forces_mod[1, 2]:.8f} eV/A")
print(f"Predicted Q:    {predicted_charges.flatten()}")

print("\n--- Physical Comparison ---")
diff = abs(forces_mod[1, 2]) - abs(forces_orig[1, 2])
if abs(forces_orig[1, 2]) < 1e-7 and abs(forces_mod[1, 2]) > 1e-7:
    print("RESULT: SUCCESS. The original model is blind at this range.")
    print("The modified model correctly captures long-range electrostatics.")
else:
    print(f"Difference in force: {diff:.8f} eV/A")
