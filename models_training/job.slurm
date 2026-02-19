#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --account=studiegrupper-vortex
#SBATCH --partition=GPUQ
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu40g|gpu80g|gpu32g"
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

set -e

echo "Running on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install -r requirements-obb.txt

python train.py --config config.yaml
