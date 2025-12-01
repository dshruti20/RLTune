#!/bin/bash
#SBATCH --job-name=test_samenode1
#SBATCH --output=test_samenode1.out
#SBATCH --nodelist=p100-control
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --partition=debug

echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

nvidia-smi
