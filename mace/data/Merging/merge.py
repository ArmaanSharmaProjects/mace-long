import numpy as np
from ase.io import read, write

# ================= CONFIGURATION =================
XYZ_FILE = "training.xyz"          # Your structure file
CHARGE_FILE = "charges.dat"     # Your charge file
OUTPUT_FILE = "train_final.xyz" # The file you will feed to MACE

# Define the number of valence electrons for the PPs used in your DFT.
# BASED ON YOUR DATA: Au=11, O=6, Li=1, H=1
VALENCE_COUNTS = {
    "Au": 11,
    "O": 6,
    "Li": 1,
    "H": 1, 
    # Add other elements if they appear in your dataset
}
# =================================================

def parse_charge_file(filepath):
    """
    Parses the specific formatted text file provided.
    Returns a list of lists (one list of values per frame).
    """
    all_frames_populations = []
    current_frame_pops = []
    
    in_table = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect start of a block (using the dashed line or header)
            if "X" in line and "Y" in line and "CHARGE" in line:
                continue # Header line
            
            if "--------" in line:
                # This delimiter appears before AND after the table
                if in_table: 
                    # End of table
                    in_table = False
                    if current_frame_pops:
                        all_frames_populations.append(np.array(current_frame_pops))
                        current_frame_pops = []
                else:
                    # Start of table
                    in_table = True
                continue

            # Data parsing
            if in_table:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # The charge/population is usually the 5th column (index 4)
                        # Col 1: ID, Col 2-4: Pos, Col 5: CHARGE
                        pop = float(parts[4])
                        current_frame_pops.append(pop)
                    except ValueError:
                        pass # specific footer lines inside the block?

    return all_frames_populations

def merge_data():
    print(f"Reading structures from {XYZ_FILE}...")
    atoms_list = read(XYZ_FILE, index=":")
    
    print(f"Parsing populations from {CHARGE_FILE}...")
    populations_list = parse_charge_file(CHARGE_FILE)

    # Sanity Check
    if len(atoms_list) != len(populations_list):
        raise ValueError(f"Mismatch! XYZ has {len(atoms_list)} frames but Charge file has {len(populations_list)} frames.")

    print("Merging and converting to Net Charge...")
    
    for i, (atoms, pops) in enumerate(zip(atoms_list, populations_list)):
        if len(atoms) != len(pops):
            raise ValueError(f"Atom count mismatch in frame {i}. XYZ: {len(atoms)}, ChargeFile: {len(pops)}")
        
        # 1. Calculate Net Charges (Valence - Population)
        net_charges = []
        symbols = atoms.get_chemical_symbols()
        
        for symbol, electron_pop in zip(symbols, pops):
            if symbol not in VALENCE_COUNTS:
                raise KeyError(f"Element {symbol} found in XYZ but not defined in VALENCE_COUNTS dictionary in script.")
            
            # CALCULATION: q = Z_valence - N_population
            q = VALENCE_COUNTS[symbol] - electron_pop
            net_charges.append(q)
            
        # 2. Assign to Arrays (This is what MACE looks for)
        atoms.arrays['charges'] = np.array(net_charges)
        
        # 3. Ensure Total Charge is in info (Important for QEq solver)
        # We sum the calculated charges to see if it's 0.0 or something else
        total_q = sum(net_charges)
        # Round to nearest integer to avoid float drift (e.g. 1.0000004 -> 1.0)
        atoms.info['charge'] = round(total_q) 

    print(f"Writing combined data to {OUTPUT_FILE}...")
    write(OUTPUT_FILE, atoms_list)
    print("Done! Use this new file for MACE training.")

if __name__ == "__main__":
    merge_data()