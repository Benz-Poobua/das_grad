#!/bin/bash
#SBATCH --job-name=grad_eval
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Synthetic-recovery benchmark: analytic single-mode VSGs with a known
# dispersion curve c(f) -> full I-FDG pipeline -> recovery error report
# (with and without the fiber curvature term).
#
# Outputs: benchmark_results.csv + run_manifest.json under --outdir.

# Always start from repo root
cd /home/groups/biondo/spoobua/das_grad

# Load environment
source sherlock_setup.sh

# Run workload
python3 -m src.eval \
  --outdir data/benchmarks/synth \
  --n_vsg 9 \
  --nch 400 \
  --dx 8.0 \
  --fs 50.0 \
  --nlag 2001 \
  --f_min 1.0 \
  --f_max 8.0 \
  --noise_rel 0.01
