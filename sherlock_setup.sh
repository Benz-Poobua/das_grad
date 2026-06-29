#!/usr/bin/env bash
# sherlock_setup.sh  (das_grad)
# Usage:
#   source sherlock_setup.sh
# Must be sourced (not executed).

echo "========================================"
echo " Setting up Sherlock DAS GRAD environment"
echo "========================================"

# -------------------------
# Load Modules (numpy/scipy/pandas stack -- no torch needed here)
# -------------------------
module reset
module load devel math
module load python/3.12.1
module load py-pandas/2.2.1_py312
module load py-scipy/1.12.0_py312

# -------------------------
# Set Project Root (Dynamic)
# -------------------------
PROJ="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJ"

# Avoid duplicate PYTHONPATH entries
case ":${PYTHONPATH-}:" in
  *":$PROJ:"*) ;;
  *) export PYTHONPATH="$PROJ:${PYTHONPATH-}" ;;
esac

# -------------------------
# Print Environment Info
# -------------------------
echo
echo "Project directory: $PWD"
echo "PYTHONPATH: $PYTHONPATH"
echo
echo "Loaded modules:"
module list
echo

# -------------------------
# Verify Python Environment
# -------------------------
python3 - << 'EOF'
import sys
import numpy as np
import scipy
import pandas as pd
import src

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Pandas:", pd.__version__)
print("src imported from:", src.__file__)
EOF

echo "OK: Sherlock das_grad environment ready."
