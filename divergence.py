import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from mace.calculators import MACECalculator
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
PATH_ORIG = "MACE_model_original_stagetwo.model"
PATH_4G   = "MACE_model_electrostatics_stagetwo.model"
DEVICE    = "cuda"

print("==================================================")
print("       4G MACE MECHANISM VERIFICATION SCAN        ")
print("==================================================")

# ==========================================
# 2. MODEL FORENSICS (Safety Check)
# ==========================================
print("\n[STEP 1] Inspecting Model Files...")

try:
    # Load raw models to check internal flags
    # weights_only=False is required for custom MACE classes
    raw_orig = torch.load(PATH_ORIG, map_location='cpu', weights_only=False)
    raw_4g   = torch.load(PATH_4G,   map_location='cpu', weights_only=False)
    
    # Check Electrostatics Flag
    flag_orig = getattr(raw_orig, 'use_electrostatics', False)
    flag_4g   = getattr(raw_4g,   'use_electrostatics', False)
    
    print(f"   > Original Model: Electrostatics = {flag_orig}")
    print(f"   > 4G Model:       Electrostatics = {flag_4g}")
    
    if flag_orig:
        print("\n[CRITICAL ERROR] The Original Model has Electrostatics ENABLED!")
        print("   This is why your graphs overlapped. It is running the same physics as the 4G model.")
        print("   FIX: Retrain 'MACE_model_original' with 'use_electrostatics: False' in config.")
        sys.exit(1)
        
    if not flag_4g:
        print("\n[CRITICAL ERROR] The 4G Model has Electrostatics DISABLED!")
        sys.exit(1)
        
    print("[PASS] Models are correctly configured.")

except Exception as e:
    print(f"[WARNING] Could not run forensics (likely old PyTorch version). Proceeding with scan.\nError: {e}")

# ==========================================
# 3. LOAD CALCULATORS
# ==========================================
print("\n[STEP 2] Loading Calculators to GPU...")
calc_orig = MACECalculator(model_paths=PATH_ORIG, device=DEVICE)
calc_mod  = MACECalculator(model_paths=PATH_4G,   device=DEVICE)

# ==========================================
# 4. RUN DISTANCE SCAN (Au - O Dimer)
# ==========================================
print("\n[STEP 3] Running Distance Scan (2.0 A -> 15.0 A)...")

distances = np.linspace(2.0, 15.0, 50)
forces_orig = []
forces_mod  = []
charges_4g  = []

# Setup Atoms: Au at Origin, Oxygen moving along Z
# We use a net charge of +1.0 so the QEq solver has something to solve
atoms = Atoms('AuO', positions=[[0,0,0], [0,0,2.0]])
atoms.pbc = False
atoms.info['charge'] = 1.0 

print(f"\n{'Dist (A)':<10} | {'Orig Force':<12} | {'4G Force':<12} | {'Status'}")
print("-" * 55)

for d in distances:
    # Update Position
    atoms.positions[1] = [0, 0, d]
    
    # --- ORIGINAL ---
    atoms.calc = calc_orig
    f_orig = abs(atoms.get_forces()[1, 2]) # Force on Oxygen (Z-axis)
    forces_orig.append(f_orig)
    
    # --- 4G MODIFIED ---
    atoms.calc = calc_mod
    f_mod = abs(atoms.get_forces()[1, 2])
    forces_mod.append(f_mod)
    
    # Capture Charge (if available)
    q = atoms.calc.results.get('charges', np.array([0.0, 0.0]))[1]
    charges_4g.append(q)
    
    # Live Print (Sampled)
    if d in distances[::10] or d == distances[-1]:
        # "Blind" means force is essentially zero (machine precision)
        status = "BLIND" if f_orig < 1e-8 else "ACTIVE"
        print(f"{d:<10.2f} | {f_orig:<12.6e} | {f_mod:<12.6e} | {status}")

# ==========================================
# 5. GENERATE REPORT PLOT
# ==========================================
print("\n[STEP 4] Generating Plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: Force Magnitude (Log Scale)
ax1.plot(distances, forces_orig, 'r--', linewidth=2, label='Original MACE (Local)')
ax1.plot(distances, forces_mod,  'b-',  linewidth=2, label='4G MACE (Non-Local)')
ax1.axvline(x=5.0, color='gray', linestyle=':', label='Cutoff (5.0 A)')

ax1.set_yscale('log')
ax1.set_xlabel('Separation Distance (Å)')
ax1.set_ylabel('Force Magnitude (eV/Å)')
ax1.set_title('Proof of Long-Range Interaction: Au-O Dimer')
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# Plot 2: Charges
ax2.plot(distances, charges_4g, 'g-', label='Oxygen Charge (QEq)')
ax2.set_xlabel('Separation Distance (Å)')
ax2.set_ylabel('Charge (e)')
ax2.set_title('QEq Charge Response')
ax2.axhline(0, color='k', linewidth=0.5)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("final_proof_AuO.png")

print("\n[SUCCESS] Plot saved to 'final_proof_AuO.png'")
print("If the Red line drops to ~1e-16 after 5.0A and the Blue line stays up,")
print("you have definitive proof that your implementation works.")