#!/bin/bash
#
#SBATCH --job-name=train_pep_flows
#SBATCH --output=train_pep_flows_%j.out
#SBATCH --error=train_pep_flows_%j.err
#
#SBATCH --partition=gpu_batch   # Using the GPU partition
#SBATCH --gres=gpu:1            # Requesting 1 GPUs
#SBATCH --cpus-per-gpu=8        # Allocating 8 CPUs per GPU
#SBATCH --mem-per-gpu=80G       # Allocating 80 GB memory per GPU
#SBATCH --time=4-00:00:00       # Max run time of 1 days
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user="fang.wu@arcinstitute.org"

# Activate your conda environment if needed
source /opt/conda/etc/profile.d/conda.sh
conda activate multiflow   # Replace with your actual conda environment

# Run the Python script
python -W ignore multiflow/experiments/train_pep_flows.py
