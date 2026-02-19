#!/bin/bash
set -e

TASK=$1
REQ=$2
VENV="venvs/$TASK"

echo "Setting up venv for $TASK"

# Only load modules on cluster
if command -v module >/dev/null 2>&1; then
    module purge
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
fi

if [ ! -d "$VENV" ]; then
    python3 -m venv $VENV
fi

source $VENV/bin/activate

pip install --upgrade pip
pip install -r $REQ

echo "Venv ready at $VENV"
