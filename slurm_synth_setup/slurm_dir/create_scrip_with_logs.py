import pandas as pd
import os

# == Configuration ==
input_csv = '/home/cc/slurm_dir/balanced_16.csv'
llm_dir    = '/home/cc/DS-examples/inference/huggingface/text-generation/slurm'
vision_dir = '/home/cc/MultiSourceML_Project/Training_Inference_Run'
conda_env  = 'env_slurm'
llm_models = [
    'facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
    'facebook/opt-2.7b', 'bigscience/bloom-3b'
]

# Output log base directory
default_log_base = '~/job_logs'

# Read workloads
df = pd.read_csv(input_csv)

# == Script Generation ==
for idx, row in df.iterrows():
    model    = row['model']
    num_gpus = int(row['Num of GPUs'])
    mode     = row.get('mode', '')  # '' / 'dp' / 'ddp'

    # decide Slurm ntasks & cpus-per-task
    if mode == 'ddp':
        ntasks        = num_gpus
        cpus_per_task = 4
    else:
        ntasks        = 1
        cpus_per_task = 4 * num_gpus

    # sanitize model name for script
    safe_model = model.replace('/', '_').replace('.', '_')
    job_name   = f"job{idx+1}_{safe_model}_{num_gpus}"

    header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time=02:00:00
#SBATCH --partition=debug

# Load conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate {conda_env}

# Set CUDA paths (optional)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

# Setup logging directory
LOGDIR={default_log_base}/${{SLURM_JOB_ID}}
mkdir -p "$LOGDIR"

# Start CPU & memory logging
vmstat 1 > "$LOGDIR/cpu_mem.log" &
VMSTAT_PID=$!

# Start GPU logging
nvidia-smi \
  --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
  --format=csv -lms 1000 \
  > "$LOGDIR/gpu_usage.log" &
GPU_PID=$!

"""

    lines = [header]

    # LLM inference
    if model in llm_models:
        lines.append(f"cd {llm_dir}\n")
        lines.append(
            f"deepspeed inference-test.py --name {model} --batch_size {int(row['batch_size'])}\n"
        )

    # Vision training
    else:
        lines.append(f"cd {vision_dir}\n")
        bs = int(row['batch_size'])
        ds = row.get('dataset', '')
        ep = int(row['epochs']) if pd.notna(row['epochs']) else ''
        if num_gpus == 1:
            lines.append(
                f"python Training_LogCollection.py --arch {model} "
                f"--epochs {ep} --batch-size {bs} --dataset {ds}\n"
            )
        elif mode == 'dp':
            lines.append(
                f"python Training_LogCollection.py --arch {model} "
                f"--num-gpus={num_gpus} --mode dp "
                f"--epochs {ep} --batch-size {bs}\n"
            )
        else:  # ddp
            lines.append(
                f"torchrun --nproc_per_node={num_gpus} Training_LogCollection.py "
                f"--arch {model} --num-gpus={num_gpus} --mode ddp "
                f"--epochs {ep} --batch-size {bs}\n"
            )

    # Tear down logging after workload
    teardown = """
# Stop logging
kill $VMSTAT_PID $GPU_PID
"""
    lines.append(teardown)

    # Write out and make executable
    script_file = f"{job_name}.sh"
    with open(script_file, 'w') as fp:
        fp.writelines(lines)
    os.chmod(script_file, 0o755)

print(f"Generated {len(df)} SLURM scripts with logging (*.sh).")
