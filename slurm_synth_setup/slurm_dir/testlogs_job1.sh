#!/bin/bash
#SBATCH --job-name=test1_logcollect
#SBATCH --output=test1_logcollect.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=debug

# ——————————————————————————
# 0) Setup logging directory
LOGDIR=~/job_metrics/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

# 1) Gather Slurm metadata (including submit time)
SUBMIT_TIME=$(
  scontrol show job "${SLURM_JOB_ID}" \
    | sed -n 's/.*SubmitTime=\([^ ]*\).*/\1/p'
)
echo "JOB_ID,JOB_NAME,NUM_GPUS,NODE,SUBMIT_TIME" > "$LOGDIR/metadata.csv"
echo "${SLURM_JOB_ID},${SLURM_JOB_NAME},${SLURM_GPUS_ON_NODE},${SLURMD_NODENAME},${SUBMIT_TIME}" \
  >> "$LOGDIR/metadata.csv"

# 2) Start GPU sampler in the background
nvidia-smi \
  --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
  --format=csv -lms 1000 \
  > "$LOGDIR/gpu_usage.csv" &
GPU_LOGGER_PID=$!

# 3) Record start time
echo "START_TIME,$(date '+%Y-%m-%d %H:%M:%S')" > "$LOGDIR/timestamps.csv"

# ——————————————————————————
# 4) Load your environment & run the job under `time -v`
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate env_slurm

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

cd /home/cc/MultiSourceML_Project/Training_Inference_Run

# the stderr from `time -v` goes into time.log
/usr/bin/time -v \
  python Training_LogCollection.py \
    --arch resnet152 \
    --epochs 2 \
    --batch_size 64 \
    --dataset mnist \
  2> "$LOGDIR/time.log"
EXIT_CODE=$?

# 5) Record end time
echo "END_TIME,$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGDIR/timestamps.csv"

# 6) Append job status
if [ $EXIT_CODE -eq 0 ]; then
  STATUS="COMPLETED"
else
  STATUS="FAILED"
fi
echo "JOB_STATUS,${STATUS}" >> "$LOGDIR/metadata.csv"

# 7) Tear down GPU sampler
kill $GPU_LOGGER_PID || true

# 8) Exit with the original job’s return code
exit $EXIT_CODE
