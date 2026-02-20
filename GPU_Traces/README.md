# GPU Traces

This folder contains job trace CSV files for GPU cluster scheduling experiments.

## Trace Files

| File | Description |
|------|-------------|
| `Helios_Formatted_withGPUType.csv` | Helios cluster trace |
| `Philly_Formatted_withGPUType.csv` | Microsoft Philly cluster trace |
| `Alibaba_Formatted_withGPUType.csv` | Alibaba GPU cluster trace |
| `synthetic_trace.csv` | Synthetic workload for testing |

## CSV Format

Each trace follows the same 19-column format. A value of `-1` indicates the field is not available.

| Column | Field | Description |
|--------|-------|-------------|
| 0 | job_id | Unique job identifier |
| 1 | submit_time | Job submission timestamp (seconds) |
| 2 | wait_time | Time spent waiting in queue |
| 3 | run_time | Actual job runtime (seconds) |
| 4 | num_gpus | Number of GPUs allocated |
| 5 | cpu_time | Average CPU time used |
| 6 | used_memory | Memory used |
| 7 | request_cpus | Number of CPUs requested |
| 8 | request_time | Requested runtime |
| 9 | request_memory | Memory requested |
| 10 | status | Job status |
| 11 | user_id | User identifier |
| 12 | group_id | Group identifier |
| 13 | executable_number | Executable identifier |
| 14 | queue_number | Queue identifier |
| 15 | partition_number | Partition identifier |
| 16 | proceeding_job_number | Preceding job in workflow |
| 17 | vc_id | Virtual cluster ID (-1 for Alibaba) |
| 18 | gpu_type | GPU type encoding (see below) |

## GPU Type Encoding

| Code | GPU Type |
|------|----------|
| 0 | T4 |
| 1 | MISC |
| 2 | P100 |
| 3 | V100_16GB |
| 4 | V100_32GB |

## Example (Alibaba Trace)

```csv
c936346f45eccd34bf748541,2693235,-1,2612,1,-1,-1,-1,-1,29.296875,1,-1,-1,-1,0,-1,-1,-1,1
455c3dec270f4777ad67721c,3399583,-1,149,1,-1,-1,-1,-1,29.296875,1,-1,-1,-1,0,-1,-1,-1,1
ba64aa2f0feff18428923e92,2152271,-1,5942,1,-1,-1,-1,-1,29.296875,1,-1,-1,-1,0,-1,-1,-1,1
```

**Breakdown of first row:**
- Job ID: `c936346f45eccd34bf748541`
- Submit time: `2693235` seconds
- Run time: `2612` seconds
- GPUs requested: `1`
- VC ID: `-1` (not applicable for Alibaba)
- GPU type: `1` (MISC)
