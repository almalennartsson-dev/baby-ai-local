#!/bin/bash
#SBATCH -A berzelius-2025-224
#SBATCH --gpus=1
#SBATCH -t 40:00:00
#SBATCH -J train_unet

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate pytorch-2.5.0-python-3.10

python main.py


