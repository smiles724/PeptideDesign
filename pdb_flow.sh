#!/bin/bash
#
#SBATCH --job-name=train_se3_flows
#SBATCH --output=train_se3_flows_%j.out
#SBATCH --error=train_se3_flows_%j.err
#
#SBATCH --partition=gpu_batch   # Using the GPU partition
#SBATCH --gres=gpu:4            # Requesting 4 GPUs
#SBATCH --cpus-per-gpu=8        # Allocating 8 CPUs per GPU
#SBATCH --mem-per-gpu=80G       # Allocating 80 GB memory per GPU
#SBATCH --time=3-00:00:00       # Max run time of 3 days
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user="fang.wu@arcinstitute.org"

# Activate your conda environment if needed
source /opt/conda/etc/profile.d/conda.sh
conda activate dflow   # Replace with your actual conda environment

# Run the Python script
python -W ignore dflow/experiments/train_se3_flows.py -cn pdb_codesign
