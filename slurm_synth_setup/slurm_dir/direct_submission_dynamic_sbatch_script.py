#!/usr/bin/env python3
import csv
import os
import time
import random

# == Configuration ==
input_csv = '/home/cc/slurm_dir/Mixed_P100_Workloads__16_jobs_.csv'  # adjust path as needed

# == Read all jobs from the CSV ==
with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

num_jobs = len(rows)
if num_jobs == 0:
    print("No jobs found in CSV.")
    exit(1)

# == Generate random schedule within 15 minutes (900s) ==
# Ensure first job at t=0, remaining at random times <=900s
schedule = [0.0] + sorted(random.uniform(0, 900) for _ in range(num_jobs - 1))
prev_time = 0.0

print(f"Submitting {num_jobs} jobs over a random schedule up to 15 minutes...")
for idx, (row, t_target) in enumerate(zip(rows, schedule), start=1):
    # Wait until target time offset
    sleep_time = t_target - prev_time
    if sleep_time > 0:
        print(f"Sleeping for {sleep_time:.1f}s before submitting job {idx}...")
        time.sleep(sleep_time)
    # Build job script name
    model = row['model']
    num_gpus = row['Num of GPUs']
    safe_model = model.replace('/', '_').replace('.', '_')
    job_script = f"job{idx}_{safe_model}_{num_gpus}.sh"

    # Submit to Slurm
    print(f"[{time.strftime('%H:%M:%S')}] sbatch {job_script}")
    os.system(f"sbatch {job_script}")

    prev_time = t_target

print("All jobs submitted within 15-minute window.")
