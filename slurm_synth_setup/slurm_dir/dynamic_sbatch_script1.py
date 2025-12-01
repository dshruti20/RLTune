#!/usr/bin/env python3
import csv
import os
import random

# == Configuration ==
input_csv      = '/home/cc/slurm_dir/balanced_16.csv'  # path to your CSV
dynamic_submit = 'dynamic_submit_all_jobs.sh'

# 1) Read CSV and collect job-script names
job_scripts = []
with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader, start=1):
        model     = row['model']
        num_gpus  = row['Num of GPUs']
        safe_model = model.replace('/', '_').replace('.', '_')
        job_scripts.append(f"job{idx}_{safe_model}_{num_gpus}.sh")

n = len(job_scripts)
if n == 0:
    raise SystemExit("No jobs found in CSV.")

# 2) Pick n random submission times in [0,900], sort them, then convert to sleep intervals
window = 900.0  # seconds
times = sorted(random.uniform(0, window) for _ in range(n))
intervals = [times[0]] + [times[i] - times[i-1] for i in range(1, n)]

# 3) Write the dynamic-submit script
with open(dynamic_submit, 'w') as out:
    out.write("#!/bin/bash\n")
    out.write("# Submit each job at a random time within the next 15 minutes\n\n")
    for sleep_t, js in zip(intervals, job_scripts):
        out.write(f"sleep {sleep_t:.2f}\n")
        out.write(f"sbatch {js}\n\n")

# 4) Make it executable
os.chmod(dynamic_submit, 0o755)

print(f"Generated '{dynamic_submit}' with {n} jobs scheduled over {window:.0f}s.")
