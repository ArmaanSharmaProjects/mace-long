#!/bin/bash
#SBATCH -J mace_long_test
#SBATCH -o mace_long_test_%j.out
#SBATCH -e mace_long_test_%j.err
#SBATCH -p gpu-a100-dev                    # RTX partition
#SBATCH -N 1                      # Number of nodes
#SBATCH -n 1                      # Number of tasks
#SBATCH -t 02:00:00               # Time limit
#SBATCH -A DMR24028
# Load modules
module load python3

# Activate your virtual environment
cd /work/10869/armaansharma547/ls6/mace-main
source mace-long/bin/activate

# Run training with direct CLI args
python3 mace/cli/run_train.py --config="config.yaml"

