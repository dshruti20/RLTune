"""
Eval script for Philly v2 env and v2-trained models.
Uses philly_env_base_v2 (PhillyEnvSkip, two pools 2-GPU/8-GPU, no mixing).
Loads a policy from a Philly v2 training run and compares RL vs baseline on random 256-job batches.
"""
import time
import os
import os.path as osp
import math
import numpy as np
import sys
from statistics import mean

import tensorflow as tf
from spinup.utils.logx import restore_tf_graph

from helios_env_base_v2 import MAX_QUEUE_SIZE, JOB_FEATURES
from philly_env_base_v2 import PhillyEnvSkip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcdefaults()


def load_policy(model_path, itr='last'):
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save' + itr))
    pi = model['pi']
    out = model['out']
    get_out = lambda x, y: sess.run(out, feed_dict={
        model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES),
        model['mask']: y.reshape(-1, MAX_QUEUE_SIZE),
    })
    get_probs = lambda x, y: sess.run(pi, feed_dict={
        model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES),
        model['mask']: y.reshape(-1, MAX_QUEUE_SIZE),
    })
    return get_probs, get_out


def action_from_obs(o):
    """Fallback when policy confidence is 0: pick valid slot with smallest first feature (normalized wait time)."""
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
            pass
        elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
            pass
        else:
            lst.append((o[i], math.floor(i / JOB_FEATURES)))
    if not lst:
        return 0
    min_val = min(x[0] for x in lst)
    result = [x[1] for x in lst if x[0] == min_val]
    return result[0]


def run_policy(env, get_probs, get_out, nums, iters, score_type, sched_algo):
    rl_r = []
    sjf_r = []

    for iter_num in range(iters):
        start = env.np_random.randint(0, max(1, env.loads.size() - nums))
        env.reset_for_test(nums, start)
        schedule_logs = env.schedule_curr_sequence_reset(env.schedule_algos[sched_algo])
        sjf_r.append(abs(sum(schedule_logs.values())))

        o = env.build_observation()
        print("schedule: ", end="")
        rl = 0
        total_decisions = 0
        rl_decisions = 0
        while True:
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)

            out = get_out(o, np.array(lst))
            softmax_out = np.exp(out - np.max(out))
            softmax_out = softmax_out / softmax_out.sum()
            confidence = float(np.max(softmax_out))
            total_decisions += 1.0
            if confidence > 0:
                pi = get_probs(o, np.array(lst))
                a = pi[0]
                rl_decisions += 1.0
            else:
                a = action_from_obs(o)

            o, r, d, _ = env.step_for_test_picktype(a)
            rl += r
            if d:
                print("Sequence Length:", total_decisions)
                break
        rl_r.append(abs(rl))
        print("")

    all_data = [sjf_r, rl_r]
    print("Mean baseline (sched_algo=%d):" % sched_algo, mean(sjf_r))
    print("Mean RL:", mean(rl_r))
    if mean(sjf_r) != 0:
        print("% improvement:", (mean(sjf_r) - mean(rl_r)) / mean(sjf_r))

    plt.rc("font", size=33)
    plt.figure(figsize=(5, 7))
    axes = plt.axes()
    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', linewidth=1, color='black')
    plt.plot(xticks[1:2], all_data[1:2], 'o', linewidth=1, color='black')
    plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True, widths=0.5,
                medianprops={"linewidth": 0}, meanprops={"color": "black", "linewidth": 4, "linestyle": "solid"})
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['Baseline', 'RL']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))], xticklabels=xticklabels)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='minor', labelsize=30)
    plt.tight_layout(pad=0.5)
    os.makedirs("new_exp_figure/philly", exist_ok=True)
    plt.savefig("new_exp_figure/philly/test.png")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Eval Philly v2-trained policy on Philly v2 env')
    parser.add_argument('--rlmodel', type=str, default='data/logs/philly_v2/philly_v2_s0')
    parser.add_argument('--workload', type=str, default='GPU_Traces/Philly_Formatted_withGPUType.csv')
    parser.add_argument('--cluster', type=str, default='philly', choices=['philly'])
    parser.add_argument('--len', '-l', type=int, default=256, help='Batch size (num jobs per eval trajectory)')
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--job_score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--sched_algo', type=int, default=0, help='Baseline: 0=FCFS, 4=SJF, etc.')
    parser.add_argument('--use_milp_allocation', type=int, default=0, choices=[0, 1],
                        help='1 = use MILP solver for RL allocation; 0 = tier1+lex')
    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)
    print("model file:", model_file)
    get_probs, get_out = load_policy(model_file, 'last')

    env = PhillyEnvSkip(
        shuffle=bool(args.shuffle),
        backfil=bool(args.backfil),
        skip=bool(args.skip),
        job_score_type=args.job_score_type,
        batch_job_slice=args.batch_job_slice,
        build_sjf=False,
        sched_algo=args.sched_algo,
        use_milp_allocation=bool(args.use_milp_allocation),
    )
    env.my_init(workload_file=workload_file, cluster_name=args.cluster)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, get_probs, get_out, args.len, args.iter, args.job_score_type, args.sched_algo)
    print("time elapsed: {}".format(time.time() - start))
