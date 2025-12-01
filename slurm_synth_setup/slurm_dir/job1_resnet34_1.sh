#!/bin/bash
#SBATCH --job-name=job1_resnet34_1
#SBATCH --output=job1_resnet34_1.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
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
python Training_LogCollection.py --arch resnet34 --epochs 1 --batch-size 32 --dataset cifar100

# Stop logging
kill $VMSTAT_PID $GPU_PID
