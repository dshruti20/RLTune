#!/bin/bash
#SBATCH --job-name=job2
#SBATCH --output=job2.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=debug

# Load conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate env_slurm

# Set CUDA paths (optional here, but safe to include)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

# Move to DeepSpeed inference directory
cd /home/cc/MultiSourceML_Project/Training_Inference_Run

# Run DeepSpeed inference
python Training_LogCollection.py --arch resnet50 --epochs 4

