#!/bin/bash
#SBATCH --job-name=opt125_node2
#SBATCH --output=now.out
#SBATCH --nodelist=p100-20april-3
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --partition=debug

# Load conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate env_slurm

# Set CUDA paths (optional here, but safe to include)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

# Move to DeepSpeed inference directory
cd /home/cc/DS-examples/inference/huggingface/text-generation/slurm

# Run DeepSpeed inference
deepspeed --num_gpus 2 inference-test.py --name facebook/opt-125m
