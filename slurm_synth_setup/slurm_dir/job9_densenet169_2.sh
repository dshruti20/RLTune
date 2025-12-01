#!/bin/bash
#SBATCH --job-name=job9_densenet169_2
#SBATCH --output=job9_densenet169_2.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
python Training_LogCollection.py --arch densenet169 --num-gpus=2 --mode dp --epochs 1 --batch-size 64

# Stop logging
kill $VMSTAT_PID $GPU_PID
