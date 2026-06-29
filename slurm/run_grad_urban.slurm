#!/bin/bash
#SBATCH --job-name=grad_urban
#SBATCH --partition=serc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

# I-FDG gradiometry on das_ani VSGs (CPU-only; the whole workflow is FFTs
# and elementwise math -- Davis et al. 2026 report 49 OBN VSGs in < 8 min
# on one 24-core node, and the fiber case is lighter still).
#
# Prerequisite: a finished das_ani run whose stacks live at the
# paths.vsg_root configured in configs/urban_grad.yaml.

# Always start from repo root
cd /home/groups/biondo/spoobua/das_grad

# Load environment
source sherlock_setup.sh

# Run workload
python3 -m src.grad --config configs/urban_grad.yaml --verbose
