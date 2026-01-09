from ase.io import read, write
import random

# Load your merged data
all_atoms = read("train_final.xyz", index=":")
random.shuffle(all_atoms)

# 80% Train, 10% Val, 10% Test
n = len(all_atoms)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_atoms = all_atoms[:n_train]
val_atoms = all_atoms[n_train:n_train+n_val]
test_atoms = all_atoms[n_train+n_val:]

write("mace_train.xyz", train_atoms)
write("mace_val.xyz", val_atoms)
write("mace_test.xyz", test_atoms)