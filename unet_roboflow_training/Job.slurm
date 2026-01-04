#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=studiegrupper-vortex
#SBATCH --job-name=vortex-unet-train
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu40g|gpu80g|gpu32g"
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Helps fragmentation for some training runs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

module purge
module --ignore_cache load foss/2022a
module --ignore_cache load Python/3.10.4-GCCcore-11.3.0

# Put venv on node-local scratch if available (faster + avoids shared FS contention)
VENV_BASE="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="$(mktemp -d -p "$VENV_BASE" venv-unet-XXXXXXXX)"

cleanup() {
  # Don't fail cleanup if deactivate isn't defined
  deactivate 2>/dev/null || true
  rm -rf "$VENV_DIR"
}
trap cleanup EXIT

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install deps without pip cache (faster/cleaner on clusters)
python -m pip install --no-cache-dir -r requirements.txt

# Run under Slurm step (better accounting / env handling)
srun python train.py