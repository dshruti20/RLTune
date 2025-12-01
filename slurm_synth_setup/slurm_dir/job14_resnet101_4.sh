#!/bin/bash
#SBATCH --job-name=job14_resnet101_4
#SBATCH --output=job14_resnet101_4.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=debug

# Load conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate env_slurm

# Set CUDA paths (optional)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

# Setup logging directory
LOGDIR=~/job_logs/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

# Start CPU & memory logging
vmstat 1 > "$LOGDIR/cpu_mem.log" &
VMSTAT_PID=$!

# Start GPU logging
nvidia-smi   --query-gpu=index,utilization.gpu,utilization.memory,memory.used   --format=csv -lms 1000   > "$LOGDIR/gpu_usage.log" &
GPU_PID=$!

cd /home/cc/MultiSourceML_Project/Training_Inference_Run
torchrun --nproc_per_node=4 Training_LogCollection.py --arch resnet101 --num-gpus=4 --mode ddp --epochs 2 --batch-size 32

# Stop logging
kill $VMSTAT_PID $GPU_PID
