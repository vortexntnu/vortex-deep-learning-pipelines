#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH --constraint="{constraint}"
#SBATCH --mem={mem}
#SBATCH --output=%x_%j.out

set -e

echo "Running on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "Installing requirements"
python3 -m pip install -r {requirements}

echo "Starting training"
python3 {train_script} --config {config}
