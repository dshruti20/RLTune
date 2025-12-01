import csv
import os

# == Configuration ==
# Path to your input CSV file containing workloads
input_csv = '/home/cc/slurm_dir/Mixed_P100_Workloads__16_jobs_.csv'  # adjust path
# Output submission driver script
submit_script = 'submit_all_jobs.sh'

# Read CSV and generate submission script
with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    with open(submit_script, 'w') as out:
        out.write("#!/bin/bash\n")
        out.write("# This script will submit all generated job scripts to Slurm\n\n")
        for idx, row in enumerate(reader, start=1):
            model     = row['model']
            num_gpus  = row['Num of GPUs']
            # sanitize model name for filename
            safe_model = model.replace('/', '_').replace('.', '_')
            # construct job script name
            job_script = f"job{idx}_{safe_model}_{num_gpus}.sh"
            out.write(f"sbatch {job_script}\n")

# Make the submission script executable
os.chmod(submit_script, 0o755)

print(f"Generated '{submit_script}' that will submit all {idx} jobs.")
