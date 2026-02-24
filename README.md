# RLTune

RLTune is a reinforcement learning–driven scheduler for heterogeneous GPU clusters that combines RL-based prioritization with MILP-based allocation strategies to optimize queue delay, GPU utilization, and fragmentation.

📄 Published at ACM SoCC 2025  
Paper: [Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters](https://dl.acm.org/doi/10.1145/3772052.3772257)

## Overview

RLTune implements RL+MILP-based job scheduling for GPU clusters. The framework supports multiple cluster traces:

- **Helios** - `GPU_Traces/Helios_Formatted_withGPUType.csv`
- **Philly** - `GPU_Traces/Philly_Formatted_withGPUType.csv`
- **Alibaba** - `GPU_Traces/Alibaba_Formatted_withGPUType.csv`
- **Synthetic** - `GPU_Traces/synthetic_trace.csv`

Each cluster implementation includes:
- `*_env_base_v2.py` - Environment with tier1/tier2 allocation
- `*_train_v2.py` - PPO-based RL training
- `*_eval_v2.py` - Evaluation comparing RL vs baseline policies

## Installation

```bash
pip install -r requirements.txt
```

Requires TensorFlow and [Spinning Up](https://github.com/openai/spinningup) for PPO utilities.

## Quick Start (Helios Example)

### Training

```bash
python helios_train_v2.py --epochs 100 --trajs 100 --sched_algo 0 --exp_name helios_FCFS_BSLD_batch256_epochs100 --workload GPU_Traces/Helios_Formatted_withGPUType.csv --score_type 0
```

This trains a PPO agent for 100 epochs with 100 trajectories per epoch, using FCFS as the baseline.

**Output:** Model checkpoints saved to `data/logs/helios_v2/helios_v2_s0/`

### Evaluation

```bash
python helios_eval_v2.py --rlmodel data/logs/helios_v2/helios_v2_s0 --sched_algo 0 --len 256 --iter 10 --score_type 0 
```

Evaluates the trained model on 10 random batches of 256 jobs, comparing RL vs baseline.

**Output:** Performance comparison and plot saved to `new_exp_figure/philly/test.png`

## Command Line Options

### Training (`*_train_v2.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--workload` | `GPU_Traces/<cluster>_Formatted_withGPUType.csv` | Path to job trace CSV |
| `--epochs` | 1 | Number of training epochs |
| `--trajs` | 100 | Trajectories per epoch |
| `--sched_algo` | 0 | Baseline algorithm (FCFS, SJF, F1, Slurm) |
| `--exp_name` | `<cluster>_v2` | Experiment name for logging |
| `--pre_trained` | 0 | Load pre-trained model (0=no, 1=yes) |
| `--trained_model` | `./data/logs/...` | Path to pre-trained model |
| `--attn` | 0 | Use attention mechanism (0=no, 1=yes) |
| `--backfil` | 0 | Enable backfill scheduling |
| `--score_type` | 0 | Job scoring metric: 0=bounded slowdown, 1=wait time, 2=turnaround time, 3=utilization, 4=slowdown |
| `--use_milp_allocation` | 0 | Use MILP solver for allocation (0=tier1+lex, 1=MILP) |

### Evaluation (`*_eval_v2.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--rlmodel` | `data/logs/<cluster>_v2/<cluster>_v2_s0` | Path to trained model |
| `--workload` | `GPU_Traces/<cluster>_Formatted_withGPUType.csv` | Path to job trace CSV |
| `--len` | 256 | Number of jobs per evaluation batch |
| `--iter` | 10 | Number of evaluation iterations |
| `--sched_algo` | 0 | Baseline algorithm for comparison |
| `--job_score_type` | 0 | Job scoring metric: 0=bounded slowdown, 1=wait time, 2=turnaround time, 3=utilization, 4=slowdown |

## Project Structure

```
RLTune/
├── helios_env_base_v2.py      # Helios environment
├── helios_train_v2.py         # Helios training
├── helios_eval_v2.py          # Helios evaluation
├── philly_env_base_v2.py      # Philly environment
├── philly_train_v2.py         # Philly training
├── philly_eval_v2.py          # Philly evaluation
├── alibaba_env_base_v2.py     # Alibaba environment
├── alibaba_train_v2.py        # Alibaba training
├── alibaba_eval_v2.py         # Alibaba evaluation
├── allocation_score.py        # Tier1/Tier2 allocation logic
├── GPU_Traces/                # Job trace CSV files
├── legacy/                    # Previous implementations
└── requirements.txt
```



