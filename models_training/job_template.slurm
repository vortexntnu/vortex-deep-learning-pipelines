#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}

set -e

echo "Running on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

source {venv_path}/bin/activate

pip install -r {requirements}

python {train_script} --config {config}
