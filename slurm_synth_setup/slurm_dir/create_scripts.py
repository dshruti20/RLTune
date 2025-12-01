import pandas as pd
import os

# == Configuration ==
# Path to your input CSV file (mixed 16 or 32 jobs)
input_csv = '/home/cc/slurm_dir/Mixed_P100_Workloads__16_jobs_.csv'  # ← adjust as needed
# Directories for LLM inference and vision training
llm_dir    = '/home/cc/DS-examples/inference/huggingface/text-generation/slurm'
vision_dir = '/home/cc/MultiSourceML_Project/Training_Inference_Run'
# Environment
conda_env  = 'env_slurm'
# List of LLM model names for detection
llm_models = [
    'facebook/opt125m', 'facebook/opt350m', 'facebook/opt1.3b',
    'facebook/opt2.7b', 'bigscience/bloom-3b'
]

# == Read the workloads CSV ==
df = pd.read_csv(input_csv)

# == Bash Header Template ==
header = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=debug

# Load conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate {conda_env}

# Set CUDA paths (optional)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu

'''

# == Script Generation ==
for idx, row in df.iterrows():
    model      = row['model']
    num_gpus   = int(row['Num of GPUs'])
    batch_size = int(row['batch_size']) if pd.notna(row['batch_size']) else ''
    dataset    = row.get('dataset', '')
    epochs     = int(row['epochs']) if pd.notna(row['epochs']) else ''
    mode       = row.get('mode', '')

    # Sanitize model name for filenames
    safe_model = model.replace('/', '_').replace('.', '_')
    job_name   = f"job{idx+1}_{safe_model}_{num_gpus}"

    # Start writing lines
    lines = [header.format(job_name=job_name, num_gpus=num_gpus, conda_env=conda_env)]

    # LLM inference scripts
    if model in llm_models:
        lines.append(f"cd {llm_dir}\n")
        lines.append(f"deepspeed inference-test.py --name {model} --batch_size {batch_size}\n")

    # Vision training scripts
    else:
        lines.append(f"cd {vision_dir}\n")
        # Single-GPU case
        if num_gpus == 1:
            lines.append(
                f"python Training_LogCollection.py --arch {model} --epochs {epochs} "
                f"--batch_size {batch_size} --dataset {dataset}\n"
            )
        # DataParallel
        elif mode == 'dp':
            lines.append(
                f"python Training_LogCollection.py --arch {model} --num-gpus={num_gpus} "
                f"--mode dp --epochs {epochs} --batch_size {batch_size}\n"
            )
        # Distributed DataParallel
        elif mode == 'ddp':
            lines.append(
                f"torchrun --nproc_per_node={num_gpus} Training_LogCollection.py "
                f"--arch {model} --num-gpus={num_gpus} --mode ddp "
                f"--epochs {epochs} --batch_size {batch_size}\n"
            )

    # Write out the script file
    script_file = f"{job_name}.sh"
    with open(script_file, 'w') as fp:
        fp.writelines(lines)
    os.chmod(script_file, 0o755)

print(f"Generated {len(df)} SLURM scripts (*.sh) in the current directory.")
