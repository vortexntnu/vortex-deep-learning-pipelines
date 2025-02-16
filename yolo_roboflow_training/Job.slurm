#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=studiegrupper-vortex
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint="gpu40g|gpu80g|gpu32g"
#SBATCH --job-name="vortex-img-process"
#SBATCH --output=vortex_img_process_log.out
#SBATCH --mem=32G

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/apps/eb/software/Python/3.10.4-GCCcore-11.3.0/lib/

set -e

module purge
module --ignore_cache load foss/2022a
module --ignore_cache load Python/3.10.4-GCCcore-11.3.0

pip cache purge

# makes sure that the pip is up to date
python3 -m pip install --upgrade pip

# Create a temporary virtual environment
VENV_DIR=$(mktemp -d -t env-repaint-XXXXXXXXXX)
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --upgrade pip 

# install the required packages
pip install -r requirements.txt
#pip install pyyaml # used to read the configuration file
#pip install blobfile # install blobfile to download the dataset
#pip install kagglehub # install kagglehub to download the dataset
pip install --force-reinstall torch -U 
pip install torchvision torchaudio 
#pip install diffusers transformers accelerate --user

# Mixing expandable_segments:True with max_split_size doesn't make sense because the expandable segment is the size of RAM and so it could never be split with max_split_size.
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 train.py

# Deactivate and remove the virtual environment
deactivate
rm -rf $VENV_DIR
