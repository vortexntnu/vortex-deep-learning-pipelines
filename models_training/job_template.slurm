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

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source {venv_path}/bin/activate

python {train_script} --config {config}
