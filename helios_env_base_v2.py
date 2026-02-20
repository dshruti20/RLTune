import os
import math
import json
import time
import sys
import random
from random import shuffle
import csv
import re
import numpy as np
import heapdict
import copy
import cvxpy as cp

import numpy as np
import tensorflow as tf
import scipy.signal

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

from allocation_score import find_gpu_assignments

MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours
MAX_SKIP_TIME = 12*60 * 60

JOB_FEATURES = 7
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 600  # skip 60 seconds


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Job_helios:
    def __init__(self, line):
      line = line.strip()
      s_array = re.split(',',line)
      #print(s_array)
      self.job_id = (s_array[0])
      self.submit_time = int(s_array[1])
      self.wait_time = int(s_array[2])
      self.run_time = int(s_array[3])

      self.skip_time = 0

      if self.run_time == -1:
        self.run_time = 10
      # Column 18 = VC (0-4 for 5 VCs), Column 19 = GPU type (0 for single type)
      self.vc_id = int(s_array[18])
      self.gpu_type_id = int(s_array[19]) if len(s_array) > 19 else 0
      self.number_of_allocated_gpus = int(s_array[4])
      self.average_cpu_time_used = float(s_array[5])
      self.used_memory = int(s_array[6])
      self.request_number_of_cpus = int(s_array[7])
      #self.number_of_allocated_gpus = max(self.number_of_allocated_gpus, self.request_number_of_gpus)
      self.request_number_of_gpus = self.number_of_allocated_gpus
      self.request_number_of_nodes = -1
      self.request_time = int(s_array[8])
      if self.request_time == -1:
        self.request_time = self.run_time

      self.request_time = self.run_time

      self.request_memory = float(s_array[9])
      self.status = int(s_array[10])
      self.user_id = int(s_array[11])
      self.group_id = int(s_array[12])
      self.executable_number = int(s_array[13])
      self.queue_number = int(s_array[14])

      try:
        self.partition_number = int(s_array[15])
      except ValueError:
              self.partition_number = 0

      self.proceeding_job_number = int(s_array[16])
      self.think_time_from_proceeding_job = int(s_array[17])

      self.random_id = self.submit_time

      self.scheduled_time = -1

      self.allocated_machines = None

      self.slurm_in_queue_time = 0
      self.slurm_age = 0
      self.slurm_job_size = 0.0
      self.slurm_fair = 0.0
      self.slurm_partition = 0
      self.slurm_qos = 0
      self.slurm_tres_cpu = 0.0

    def __eq__(self, other):
      return self.job_id == other.job_id

    def __lt__(self, other):
      return self.job_id < other.job_id

    def __hash__(self):
      return hash(self.job_id)

    def __str__(self):
      return "J["+str(self.job_id)+"]-["+str(self.request_number_of_gpus)+"]-["+str(self.submit_time)+"]-["+str(self.request_time)+"]"
    def __feature__(self):
      return [self.submit_time, self.request_number_of_gpus, self.request_time,
                  self.user_id, self.group_id, self.executable_number, self.queue_number]

# Helios cluster: 5 VCs, each with different node count, V100 8 GPUs per node
HELIOS_VC_NODE_COUNTS = [6, 4, 3, 2, 2]
HELIOS_GPUS_PER_NODE = 8

def get_cluster_config(cluster_name):
    """Return (vc_node_counts, gpus_per_node) for the given cluster."""
    if cluster_name == 'helios':
        return HELIOS_VC_NODE_COUNTS, HELIOS_GPUS_PER_NODE
    raise ValueError(f"Unknown cluster: {cluster_name}")

class Workload_helios:
    def __init__(self, path, cluster_name='helios'):
      self.all_jobs = []
      self.max = 0
      self.max_exec_time = 0
      self.min_exec_time = sys.maxsize
      self.max_job_id = 0

      self.max_requested_memory = 0
      self.max_user_id = 0
      self.max_group_id = 0
      self.max_executable_number = 0
      self.max_job_id = 0
      self.which_node = 0
      self.max_nodes = 0
      self.max_gpu = 0
      rd = np.random.RandomState(0)
      n = 0

      vc_node_counts, gpus_per_node = get_cluster_config(cluster_name)
      vc1 = [(i+1, gpus_per_node) for i in range(vc_node_counts[0])]
      vc2 = [(i+1, gpus_per_node) for i in range(vc_node_counts[1])]
      vc3 = [(i+1, gpus_per_node) for i in range(vc_node_counts[2])]
      vc4 = [(i+1, gpus_per_node) for i in range(vc_node_counts[3])]
      vc5 = [(i+1, gpus_per_node) for i in range(vc_node_counts[4])]

      vc1_full = []
      vc2_full = []
      vc3_full = []
      vc4_full = []
      vc5_full = []

      with open(path) as fp:
        for line in fp:
          if line.startswith(";"):
            if line.startswith("; MaxNodes:"):
              self.max_nodes = int(line.split(":")[1].strip())
            if line.startswith("; MaxProcs:"):
              self.max_gpu = int(line.split(":")[1].strip())
            continue

          j = Job_helios(line)

          if j.run_time > self.max_exec_time:
              self.max_exec_time = j.run_time
          if j.run_time < self.min_exec_time:
              self.min_exec_time = j.run_time
          if j.request_memory > self.max_requested_memory:
              self.max_requested_memory = j.request_memory
          if j.user_id > self.max_user_id:
              self.max_user_id = j.user_id
          if j.group_id > self.max_group_id:
              self.max_group_id = j.group_id
          if j.executable_number > self.max_executable_number:
              self.max_executable_number = j.executable_number

          self.all_jobs.append(j)

          if j.request_number_of_gpus > self.max:
              self.max = j.request_number_of_gpus

      #This need to be parced in automatic way when testing with other datasets. Following values are for Alibaba Trace.
      #max_nodes = [[(0,2),(1,2),...(320,2)], [(0,8),(1,8),...,(320,8)], []]
      self.cluster_status = [vc1, vc2, vc3, vc4, vc5]
      #print("printing heapdicts:", self.max_nodes[0].popitem(), len(self.max_nodes[1]))
      self.total_nodes_count = [len(vc1), len(vc2), len(vc3), len(vc4), len(vc5)]
      #P8_touple_full =
      self.max_gpu = gpus_per_node
      self.assigned_nodes = [vc1_full, vc2_full, vc3_full, vc4_full, vc5_full]

      print ("Max Allocated gpus:", str(self.max), "initial cluster status:", self.cluster_status,
                "max gpus:", self.max_gpu,
                "max execution time:", self.max_exec_time,
            "max user id:", self.max_user_id)
      #print('see here',self.max_nodes[])
      print("total number of jobs:",len(self.all_jobs))

      self.all_jobs.sort(key=lambda job: job.submit_time)

    def size(self):
      return len(self.all_jobs)

    def reset(self):    #recheck this fuction once completed with allocation
      for job in self.all_jobs:
        job.scheduled_time = -1
        job.skip_time = 0

    def __getitem__(self, item):
      return self.all_jobs[item]

class Machine:
    def __init__(self, id, strg, gpu_count):
        self.id = f'{strg} {id}'                     # nachine ids = P8 0, P8 1, P8 2, .....
        self.running_job_id = list()
        self.gpu_status = gpu_count                  # originally 8.
        #selected_nodes[id] = gpu_count
        #self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id, gpu_count):
        if self.gpu_status!= 0:
            self.running_job_id.append({job_id:gpu_count})
            self.gpu_status = self.gpu_status - gpu_count
            #self.is_free = False
            self.job_history.append({job_id:gpu_count})
            return self.gpu_status
        else:
          print("gpu status is full")
          return []

    def machine_release(self, job_release):
        incremental_val = 0
        for i in self.running_job_id:
          for k,v in i.items():
            if k == job_release.job_id:
              self.running_job_id.remove(i)
              self.gpu_status = self.gpu_status +  v
              #print("Printing inside machine_release function i, v, gpu_status:", i,v,self.gpu_status)
              incremental_val = v
        return self.gpu_status, incremental_val

    def dummy_machine_release(self, job_release):
        incremental_val = 0
        for i in self.running_job_id:
          for k,v in i.items():
            if k == job_release.job_id:
              #self.running_job_id.remove(i)
              incremental_val = v
        return incremental_val

    def reset(self,gpu_count):
        #self.is_free = True
        self.running_job_id = list()
        self.gpu_status = gpu_count
        #selected_nodes[id] = gpu_count
        #self.is_free = True
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "

class Cluster:
    def __init__(self, cluster_name, cluster_status, gpu_per_node, assigned_nodes, total_nodes_count):
        # num_procs_per_node
        self.name = cluster_name
        self.cluster_status = cluster_status    # self.total_node has list of GPUs [vc1,vc2,vc3,vc4,vc5]
        self.free_node = []
        self.used_node = assigned_nodes   # another list of heapdicts of all GPUs with '0' available GPUs
        self.unchanges_used_node = []
        self.num_gpu_per_node = gpu_per_node    # 8
        self.total_nodes_every_gpu = total_nodes_count      #[22, 18, 18, 8, 8]
        self.all_nodes = []
        self.temp_list = []
        self.selected_machines =  heapdict.heapdict()     #or simply self.selected_machines=[]
        self.dummy_cluster_status = []

        print("printing inside cluster init self.total_nodes_count:", self.total_nodes_every_gpu)
        print("printing inside cluster init cluster status:", self.cluster_status)
        print("printing inside cluster init used_nodes", self.used_node)

        for i in range(len(self.cluster_status[0])):
            self.all_nodes.append(Machine(i+1,'vc1',int(self.num_gpu_per_node)))
        for i in range(len(self.cluster_status[1])):
            self.all_nodes.append(Machine(i+1,'vc2',int(self.num_gpu_per_node)))
        for i in range(len(self.cluster_status[2])):
            self.all_nodes.append(Machine(i+1,'vc3',int(self.num_gpu_per_node)))
        for i in range(len(self.cluster_status[3])):
            self.all_nodes.append(Machine(i+1,'vc4',int(self.num_gpu_per_node)))
        for i in range(len(self.cluster_status[4])):
            self.all_nodes.append(Machine(i+1,'vc5',int(self.num_gpu_per_node)))
        print("printing all_nodes inside cluster init:", self.all_nodes[0], self.all_nodes[1], len(self.all_nodes))
        #output : [<Backfill_check_env.Machine object at 0x7fa20e3f83d0>, <Backfill_check_env.Machine object at 0x7fa1f4f1a050>,.......] M[T4 0] 49


    # all_nodes should have now machines which has ids as T4 0, T4 1, Misc 0, P100 0, etc

    def feature(self):
        return [self.free_node]

    def get_allocation_options(self, job, use_dummy=False, allow_relax_min_nodes=False):
        """Read-only: return find_gpu_assignments result for this job's VC pool. Does not set selected_machines.
        allow_relax_min_nodes=True includes tier2 (min_nodes+1) options for MILP choice."""
        vc_id = job.vc_id
        pool = self.dummy_cluster_status[vc_id] if use_dummy else self.cluster_status[vc_id]
        if not pool:
            return None
        free_nodes = list(pool)
        result = find_gpu_assignments(
            free_nodes=free_nodes,
            request_gpus=job.request_number_of_gpus,
            gpus_per_node=self.num_gpu_per_node,
            allow_relax_min_nodes=allow_relax_min_nodes,
        )
        if not result["assignments"]:
            return None
        return result

    def _pick_greedy_assignment(self, result):
        """From find_gpu_assignments result, pick tier1 assignment by lexicographic order (deterministic)."""
        tier1 = result["tier1"]["all"]
        if not tier1:
            return None
        def key_fn(a):
            return tuple(sorted(a["assignment"], key=lambda x: (x[0], x[1])))
        chosen = min(tier1, key=key_fn)
        return chosen["assignment"]

    def can_allocated(self, job):
        """Greedy allocation: tier1 only, pick by lexicographic order of assignment."""
        self.selected_machines = []
        result = self.get_allocation_options(job, use_dummy=False)
        if result is None:
            return False
        assignment = self._pick_greedy_assignment(result)
        if assignment is None:
            return False
        self.selected_machines = list(assignment)
        return True

    def dummy_can_allocated(self, job):
        """Feasibility check only on dummy cluster state (for backfill)."""
        result = self.get_allocation_options(job, use_dummy=True)
        if result is None:
            return False
        return len(result["tier1"]["all"]) > 0


    def allocate(self, job, job_id, request_num_gpus):
        allocated_nodes = []
        req_num_of_gpus = request_num_gpus
        gpu_index = [node for node, gpu in self.selected_machines]
        gpu_count = [gpu for node, gpu in self.selected_machines]
        assert sum(gpu_count) == req_num_of_gpus

        vc_id = job.vc_id
        vc_names = ['vc1', 'vc2', 'vc3', 'vc4', 'vc5']
        gpu_name = vc_names[vc_id]
        all_gpu = [f'{gpu_name} {gpu_index[i]}' for i in range(len(gpu_index))]

        # Find all machines first without modifying state (avoid corrupting cluster_status on failure)
        for i in range(len(all_gpu)):
            found = False
            for m in self.all_nodes:
                if m.id == all_gpu[i]:
                    allocated_nodes.append((m, gpu_count[i]))
                    found = True
                    break
            if not found:
                print("Error in allocation: machine id %r not found in all_nodes (vc_id=%s, gpu_index=%s)"
                      % (all_gpu[i], vc_id, gpu_index))
                return []

        if len(allocated_nodes) != len(gpu_index):
            return []

        # Apply allocation: update cluster_status and machines
        for i in range(len(gpu_index)):
            self.cluster_status[vc_id] = [t for t in self.cluster_status[vc_id] if t[0] != gpu_index[i]]
        result_machines = []
        for i, (m, gpu_cnt) in enumerate(allocated_nodes):
            node_id = gpu_index[i]
            current_state = m.taken_by_job(job_id, gpu_cnt)
            result_machines.append(m)
            if current_state == 0:
                self.used_node[vc_id].append((node_id, current_state))
            else:
                self.cluster_status[vc_id].append((node_id, current_state))
        self.selected_machines = []
        return result_machines

    def release(self,job_release, machine_releases):
        #print("In release, printing before release clusture_status:", self.cluster_status[job_release.vc_id])
        #print("printing before release used_node:", self.used_node[job_release.vc_id])
        #print("Printing machines to be released here:", machine_releases)
        for m in machine_releases:
            gpu_status,v =  m.machine_release(job_release)
            #print("gpu_status:", gpu_status)
            gpu , index = m.id.split()
            #print(gpu , index)
            if gpu == 'vc1':
              found_in_cluster = False
              if self.cluster_status[0]:
                for i, (node_id, gpus) in enumerate(self.cluster_status[0]):
                  if node_id == int(index):
                    self.cluster_status[0][i] = (node_id, gpu_status)  # Update GPU status
                    found_in_cluster = True
                    break
              if not found_in_cluster:
                for i, (node_id, gpus) in enumerate(self.used_node[0]):
                  if node_id == int(index):
                    # Remove from used_node and add to cluster_status
                    self.used_node[0].pop(i)
                    self.cluster_status[0].append((node_id, gpu_status))
                    break

            if gpu == 'vc2':
              found_in_cluster = False
              if self.cluster_status[1]:
                for i, (node_id, gpus) in enumerate(self.cluster_status[1]):
                  if node_id == int(index):
                    self.cluster_status[1][i] = (node_id, gpu_status)  # Update GPU status
                    found_in_cluster = True
                    break
              if not found_in_cluster:
                for i, (node_id, gpus) in enumerate(self.used_node[1]):
                  if node_id == int(index):
                    # Remove from used_node and add to cluster_status
                    self.used_node[1].pop(i)
                    self.cluster_status[1].append((node_id, gpu_status))
                    break

            if gpu == 'vc3':
              found_in_cluster = False
              if self.cluster_status[2]:
                for i, (node_id, gpus) in enumerate(self.cluster_status[2]):
                  if node_id == int(index):
                    self.cluster_status[2][i] = (node_id, gpu_status)  # Update GPU status
                    found_in_cluster = True
                    break
              if not found_in_cluster:
                for i, (node_id, gpus) in enumerate(self.used_node[2]):
                  if node_id == int(index):
                    # Remove from used_node and add to cluster_status
                    self.used_node[2].pop(i)
                    self.cluster_status[2].append((node_id, gpu_status))
                    break

            if gpu == 'vc4':
              found_in_cluster = False
              if self.cluster_status[3]:
                for i, (node_id, gpus) in enumerate(self.cluster_status[3]):
                  if node_id == int(index):
                    self.cluster_status[3][i] = (node_id, gpu_status)  # Update GPU status
                    found_in_cluster = True
                    break
              if not found_in_cluster:
                for i, (node_id, gpus) in enumerate(self.used_node[3]):
                  if node_id == int(index):
                    # Remove from used_node and add to cluster_status
                    self.used_node[3].pop(i)
                    self.cluster_status[3].append((node_id, gpu_status))
                    break

            if gpu == 'vc5':
              found_in_cluster = False
              if self.cluster_status[4]:
                for i, (node_id, gpus) in enumerate(self.cluster_status[4]):
                  if node_id == int(index):
                    self.cluster_status[4][i] = (node_id, gpu_status)  # Update GPU status
                    found_in_cluster = True
                    break
              if not found_in_cluster:
                for i, (node_id, gpus) in enumerate(self.used_node[4]):
                  if node_id == int(index):
                    # Remove from used_node and add to cluster_status
                    self.used_node[4].pop(i)
                    self.cluster_status[4].append((node_id, gpu_status))
                    break
        #print("printing after release clusture_status:", self.cluster_status[job_release.vc_id])
        #print("printing after release used_node:", self.used_node[job_release.vc_id])


    def dummy_release(self, job_release, machine_releases):
        for m in machine_releases:
            v = m.dummy_machine_release(job_release)
            gpu, index_str = m.id.split()
            node_idx = int(index_str)
            if gpu == 'vc1':
              found_in_cluster = False
              for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[0]):
                if node_id == node_idx:
                  self.dummy_cluster_status[0][i] = (node_id, gpus + v)
                  found_in_cluster = True
                  break
              if not found_in_cluster:
                self.dummy_cluster_status[0].append((node_idx, v))

            if gpu == 'vc2':
              found_in_cluster = False
              for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[1]):
                if node_id == node_idx:
                  self.dummy_cluster_status[1][i] = (node_id, gpus + v)
                  found_in_cluster = True
                  break
              if not found_in_cluster:
                self.dummy_cluster_status[1].append((node_idx, v))

            if gpu == 'vc3':
              found_in_cluster = False
              for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[2]):
                if node_id == node_idx:
                  self.dummy_cluster_status[2][i] = (node_id, gpus + v)
                  found_in_cluster = True
                  break
              if not found_in_cluster:
                self.dummy_cluster_status[2].append((node_idx, v))

            if gpu == 'vc4':
              found_in_cluster = False
              for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[3]):
                if node_id == node_idx:
                  self.dummy_cluster_status[3][i] = (node_id, gpus + v)
                  found_in_cluster = True
                  break
              if not found_in_cluster:
                self.dummy_cluster_status[3].append((node_idx, v))

            if gpu == 'vc5':
              found_in_cluster = False
              for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[4]):
                if node_id == node_idx:
                  self.dummy_cluster_status[4][i] = (node_id, gpus + v)
                  found_in_cluster = True
                  break
              if not found_in_cluster:
                self.dummy_cluster_status[4].append((node_idx, v))


    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):   # Restore cluster_status to match initial shape (same as all_nodes)
        self.cluster_status = [
            [(i+1, self.num_gpu_per_node) for i in range(self.total_nodes_every_gpu[vc_id])]
            for vc_id in range(len(self.total_nodes_every_gpu))
        ]
        self.used_node = [[] for _ in range(len(self.total_nodes_every_gpu))]
        for m in self.all_nodes:
            m.reset(self.num_gpu_per_node)

# --- Philly-style allocation choice: top-K jobs + MILP-like scoring (fragmentation vs load balance, future slack) ---

def _apply_assignment_to_pool(pool, assignment, gpus_per_node):
    """Return per-node free GPUs after applying assignment. pool and result are list of (node_id, free)."""
    assign_dict = dict(assignment)
    return [(node_id, free - assign_dict.get(node_id, 0)) for (node_id, free) in pool]


def _fragmentation_count(pool_state, gpus_per_node):
    """Number of nodes with 0 < free < gpus_per_node."""
    return sum(1 for (_, f) in pool_state if 0 < f < gpus_per_node)


def _load_balance_score(pool_state, gpus_per_node):
    """Variance of utilization (1 - free/gpus_per_node) across nodes. Lower is more balanced."""
    if not pool_state:
        return 0.0
    utils = [1.0 - (f / gpus_per_node) for (_, f) in pool_state]
    n = len(utils)
    mean_u = sum(utils) / n
    return sum((u - mean_u) ** 2 for u in utils) / n


def _future_slack_single_node(pool_state, request_gpus):
    """1 if at least one node has free >= request_gpus (next job can fit on one node), else 0."""
    return 1 if any(f >= request_gpus for (_, f) in pool_state) else 0


def _collect_allocation_ways(alloc_result):
    """Collect distinct allocation options from tier1/tier2 defrag and load_balance. Returns list of (assignment, label)."""
    seen = set()
    ways = []
    for (label, key) in [
        ("tier1_defrag", "defrag_sorted"),
        ("tier1_lb", "load_balance_sorted"),
        ("tier2_defrag", "defrag_sorted"),
        ("tier2_lb", "load_balance_sorted"),
    ]:
        tier = "tier1" if "tier1" in label else "tier2"
        lst = alloc_result.get(tier, {}).get(key, [])
        if not lst:
            continue
        a = lst[0]
        assignment = tuple(sorted(a["assignment"], key=lambda x: (x[0], x[1])))
        if assignment not in seen:
            seen.add(assignment)
            ways.append((list(a["assignment"]), label))
    return ways


def choose_allocation_philly_style(pool, alloc_result, gpus_per_node, top_k_jobs,
                                   alpha=1.0, beta=0.5, gamma=0.5):
    """
    Choose among allocation options (tier1/tier2, defrag vs load-balance) using future top-K job requirements.
    Equivalent to MILP: min sum_w x_w * score_w, sum x_w = 1, x binary; solved by argmin.
    pool: list of (node_id, free_gpus) for the job's VC.
    top_k_jobs: list of up to K jobs (including current) for future demand.
    Returns best assignment (list of (node_id, gpus)) or None.
    """
    ways = _collect_allocation_ways(alloc_result)
    if not ways:
        return None
    if len(ways) == 1:
        return ways[0][0]

    next_job_gpus = top_k_jobs[1].request_number_of_gpus if len(top_k_jobs) > 1 else top_k_jobs[0].request_number_of_gpus

    best_score = float("inf")
    best_assignment = ways[0][0]
    for assignment, label in ways:
        after = _apply_assignment_to_pool(pool, assignment, gpus_per_node)
        frag = _fragmentation_count(after, gpus_per_node)
        lb = _load_balance_score(after, gpus_per_node)
        slack = _future_slack_single_node(after, next_job_gpus)
        score = alpha * frag + beta * lb - gamma * slack
        if score < best_score:
            best_score = score
            best_assignment = assignment
    return best_assignment


# Unlimited capacity when CPU/memory not provided (-1 or None)
_UNLIMITED = 1e9


def _pool_to_3d(pool_gpu_only):
    """Convert pool list of (node_id, free_gpus) to 3D: (node_id, free_gpus, free_cpus, free_memory).
    When CPU/memory not tracked, use _UNLIMITED so fit uses only GPU."""
    return [(nid, g, _UNLIMITED, _UNLIMITED) for (nid, g) in pool_gpu_only]


def _apply_assignment_to_pool_3d(pool_3d, assignment, gpus_per_node):
    """Return 3D state after assignment: list of (node_id, free_gpus, free_cpus, free_mem).
    CPU/memory: we don't deduct (no per-node usage yet); pass through or use _UNLIMITED."""
    assign_dict = dict(assignment)
    out = []
    for (node_id, free_g, free_c, free_m) in pool_3d:
        used_g = assign_dict.get(node_id, 0)
        new_g = free_g - used_g
        # CPU/memory not deducted per node yet; keep as-is
        out.append((node_id, new_g, free_c, free_m))
    return out


def _job_fit_uses_cpu_mem(job):
    """True if we should use CPU and memory for fit (values given, not -1)."""
    c = getattr(job, "request_number_of_cpus", -1)
    m = getattr(job, "request_memory", -1.0)
    return (c is not None and c >= 0) and (m is not None and m >= 0)


def choose_allocation_milp_solver(pool_3d, alloc_result, gpus_per_node, top_k_jobs,
                                  alpha=1.0, beta=0.3, gamma=0.5):
    """
    Choose allocation via MILP (cvxpy + GLPK_MI). 3D resources (GPU, CPU, memory).
    When CPU/memory are -1 or not given, fit uses only GPU. Soft objective for next-job fit.
    pool_3d: list of (node_id, free_gpus, free_cpus, free_memory); use _UNLIMITED when not tracked.
    Returns best assignment (list of (node_id, gpus)) or None; falls back to first way if solve fails.
    """
    ways = _collect_allocation_ways(alloc_result)
    if not ways:
        return None
    if len(ways) == 1:
        return ways[0][0]

    W = len(ways)
    N = len(pool_3d)
    G = gpus_per_node
    M = G + 10  # big-M for fragmentation and fit

    # Next job (first in top-K after current)
    next_job = top_k_jobs[1] if len(top_k_jobs) > 1 else top_k_jobs[0]
    G_next = next_job.request_number_of_gpus
    C_next = getattr(next_job, "request_number_of_cpus", -1)
    M_next = getattr(next_job, "request_memory", -1.0)
    if C_next is None or C_next < 0:
        C_next = 0
    if M_next is None or M_next < 0:
        M_next = 0.0
    use_cpu_mem = _job_fit_uses_cpu_mem(next_job)

    # Precompute per-way per-node free (GPU, CPU, mem) after placement
    g_wn = np.zeros((W, N))
    c_wn = np.zeros((W, N))
    m_wn = np.zeros((W, N))
    node_ids = [pool_3d[i][0] for i in range(N)]
    for w, (assignment, _) in enumerate(ways):
        after = _apply_assignment_to_pool_3d(pool_3d, assignment, G)
        for i in range(N):
            g_wn[w, i] = after[i][1]
            c_wn[w, i] = after[i][2]
            m_wn[w, i] = after[i][3]

    # Decision variables
    x = cp.Variable(W, boolean=True)
    f = cp.Variable(N, boolean=True)
    y = cp.Variable(N, boolean=True)
    free_max = cp.Variable()
    free_min = cp.Variable()

    # free_gpus_n = sum_w x[w] * g_wn[w,n]
    free_gpus = cp.reshape(cp.matmul(cp.reshape(x, (1, W)), g_wn), (N,))
    free_cpus = cp.reshape(cp.matmul(cp.reshape(x, (1, W)), c_wn), (N,))
    free_mem = cp.reshape(cp.matmul(cp.reshape(x, (1, W)), m_wn), (N,))

    constraints = [
        cp.sum(x) == 1,
        free_max >= 0,
        free_min >= 0,
    ]
    # free_min <= free_gpus_n <= free_max for range (load balance)
    for n in range(N):
        constraints.append(free_gpus[n] <= free_max)
        constraints.append(free_gpus[n] >= free_min)
    # Fragmentation: f_n = 1 iff 1 <= free_gpus_n <= G-1
    for n in range(N):
        constraints.append(free_gpus[n] >= 1 - M * (1 - f[n]))
        constraints.append(free_gpus[n] <= (G - 1) + M * (1 - f[n]))
    # Next job fit (soft): y_n = 1 only when node n has enough; we reward sum(y) in objective
    for n in range(N):
        constraints.append(free_gpus[n] >= G_next * y[n])
        if use_cpu_mem:
            constraints.append(free_cpus[n] >= C_next * y[n])
            constraints.append(free_mem[n] >= M_next * y[n])

    # Objective: minimize fragmentation + load imbalance - reward next-job fit
    objective = cp.Minimize(
        alpha * cp.sum(f) + beta * (free_max - free_min) - gamma * cp.sum(y)
    )
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return ways[0][0]
        w_chosen = int(np.argmax(x.value))
        return ways[w_chosen][0]
    except Exception:
        return ways[0][0]


class HPCEnvSkip(gym.Env):
    # Top-K jobs considered for Philly-style MILP allocation choice (future demand).
    TOP_K_MILP = 5

    def __init__(self, shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0,
                 build_sjf=False, sched_algo=4, use_milp_allocation=False):  # do nothing and return. A workaround for passing parameters to the environment
        super(HPCEnvSkip, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE) # 128
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * MAX_QUEUE_SIZE,),      #JOB_FEATURES = 7 , MAX_QUEUE_SIZE=128
                                            dtype=np.float32)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.vector_score_dict = {}

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_preworkloads = False
        self.pre_workloads = []

        self.shuffle = shuffle
        self.backfil = backfil
        self.skip = skip
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type
        self.batch_job_slice = batch_job_slice

        self.build_sjf = build_sjf
        self.sjf_scores = []

        self.base_score_list_normalized = []
        self.rl_score_normalized = []
        self.makespan_base = []
        self.makespan_rl = []
        self.rewards = []

        self.schedule_algos = [self.fcfs_score, self.lcfs_score, self.smallest_score, self.largest_score, self.sjf_score, self.lpf_score, self.saf_score,
                self.laf_score, self.sexp_score, self.lexp_score, self.srf_score, self.lrf_score, self.multifactor_score, self.f1_score, self.wfp_score, self.uni_score]
        assert 0<= sched_algo < len(self.schedule_algos)
        self.schedule_algo = self.schedule_algos[sched_algo]
        self.use_milp_allocation = use_milp_allocation

    def _get_top_k_jobs(self, k):
        """Top K jobs from queue (by schedule_algo order). Used for MILP allocation choice (future demand)."""
        return sorted(self.job_queue, key=self.schedule_algo)[:k]

    def _get_pool_id(self, job):
        """Return index into cluster_status for this job's pool. VC-only (e.g. Helios) -> vc_id; GPU-type-only -> gpu_type_id; both -> use vc_id as primary (cluster_status indexed by VC)."""
        return job.vc_id

    def _same_pool(self, job, other):
        """True if both jobs use the same resource pool (same VC and same GPU type)."""
        return job.vc_id == other.vc_id and job.gpu_type_id == other.gpu_type_id

    def my_init(self, workload_file='', cluster_name='helios'):
        print("loading workloads from dataset:", workload_file, "cluster:", cluster_name)
        self.loads = Workload_helios(workload_file, cluster_name=cluster_name)
        self.cluster = Cluster("Cluster", self.loads.cluster_status, self.loads.max_gpu, self.loads.assigned_nodes, self.loads.total_nodes_count)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.build_sjf:  # this is for trajectory filtering.
            # calculate SJF scores for all sample sequence and save them here
            index = 0
            if self.batch_job_slice == 0:
                max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1
            else:
                max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1
            print("max index... initializing SJF Score Array", max_index)

            while index <= max_index:
                index += 1
                if index % 100 == 0:
                    print("index", index)

                self.cluster.reset()
                self.loads.reset()

                self.job_queue = []
                self.running_jobs = []
                self.visible_jobs = []
                self.pairs = []

                self.current_timestamp = 0
                self.start = 0
                self.next_arriving_job_idx = 0
                self.last_job_in_batch = 0
                self.num_job_in_batch = 0
                self.scheduled_rl = {}
                self.penalty = 0
                self.pivot_job = False
                self.scheduled_scores = []

                job_sequence_size = JOB_SEQUENCE_SIZE
                self.pre_workloads = []

                self.start = index;
                self.start_idx_last_reset = self.start
                self.num_job_in_batch = job_sequence_size
                self.last_job_in_batch = self.start + self.num_job_in_batch
                self.current_timestamp = self.loads[self.start].submit_time
                self.job_queue.append(self.loads[self.start])
                self.next_arriving_job_idx = self.start + 1

                if self.enable_preworkloads:
                    self.gen_preworkloads(job_sequence_size + self.np_random.randint(job_sequence_size))

                self.sjf_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))

            #print(self.sjf_scores)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_gpus
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(
            submit_time if submit_time > 0 else 0.1))

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f3: r * n + 6860000 * log10(s)
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, job):
        # run_time = job.run_time
        request_time = job.request_time
        request_processors = job.request_number_of_gpus
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return (request_time, request_processors)

    def smallest_score(self, job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return (request_processors, submit_time)

    def largest_score(self, job):
        return -job.request_number_of_processors

    def wfp_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_gpus
        request_time = job.request_time
        waiting_time = job.scheduled_time - job.submit_time
        return -np.power(float(waiting_time) / request_time, 3) * request_processors

    def uni_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time - job.submit_time

        return -(waiting_time + 1e-15) / (np.log2(request_processors + 1e-15) * request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def lcfs_score(self, job):
        return -job.submit_time

    def lpf_score(self, job):
        return -job.request_time

    def saf_score(self, job):
        return job.request_time * job.request_number_of_processors

    def laf_score(self, job):
        return -self.saf_score(job)

    def sexp_score(self, job):
        waiting_time = job.scheduled_time - job.submit_time
        request_time = job.request_time
        request_processors = job.request_number_of_processors
        return (waiting_time + request_time)/request_processors

    def lexp_score(self, job):
        return -self.sexp_score(job)

    def srf_score(self, job):
        return job.request_time / job.request_number_of_processors

    def lrf_score(self, job):
        return -self.srf_score(job)

    def multifactor_score(self, job, PriorityWeightAge=1000, PriorityWeightJobSize=1000):
        """
        ===========================
        Age Factor
        Association Factor
        Job Size Factor
        Nice Factor
        Partition Factor
        Quality of Service (QOS) Factor
        Site Factor
        TRES Factors
        Fair-share Factor
        ===========================
        Job_priority =
                site_factor +
                (PriorityWeightAge) * (age_factor) +
                (PriorityWeightAssoc) * (assoc_factor) +
                (PriorityWeightFairshare) * (fair-share_factor) +
                (PriorityWeightJobSize) * (job_size_factor) +
                (PriorityWeightPartition) * (partition_factor) +
                (PriorityWeightQOS) * (QOS_factor) +
                SUM(TRES_weight_cpu * TRES_factor_cpu,
                    TRES_weight_<type> * TRES_factor_<type>,
                    ...)
                - nice_factor
        """
        part1 = PriorityWeightAge * (job.wait_time/(MAX_WAIT_TIME))
        part2 = PriorityWeightJobSize * (1 - job.request_number_of_gpus/160)
        job_priority = part1 + part2

        # Larger job_priority will be scheduled sooner, but smaller score means sooner, so set -job_priority as score.
        return -job_priority

    def gen_preworkloads(self, size):
        # Generate some running jobs to randomly fill the cluster.
        # size = self.np_random.randint(2 * job_sequence_size)
        running_job_size = size
        for i in range(running_job_size):
            _job = self.loads[self.start - i - 1]
            req_num_of_processors = _job.request_number_of_processors
            runtime_of_job = _job.request_time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i)  # to be different from the normal jobs; normal jobs have a job_id >= 0
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                job_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, max(runtime_of_job, 1))))
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
                self.pre_workloads.append(job_tmp)
            else:
                break

    def refill_preworkloads(self):
        for _job in self.pre_workloads:
            self.running_jobs.append(_job)
            _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)



    def reset(self, full_workload=False):
        self.cluster.reset() #this is not fixed yet
        self.loads.reset()
        #print("printing inside resetTotal_nodes",self.cluster.total_node[0], self.cluster.total_node[1], list(self.cluster.total_node[0].items()),list(self.cluster.total_node[1].items()), list(self.cluster.total_node[2].items()), list(self.cluster.total_node[3].items()), list(self.cluster.total_node[4].items()))
        #print("printing inside reset used_nodes", self.cluster.used_node[0], self.cluster.used_node[1], list(self.cluster.used_node[0].items()),list(self.cluster.used_node[1].items()), list(self.cluster.used_node[2].items()), list(self.cluster.used_node[3].items()), list(self.cluster.used_node[4].items()))

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.vector_score_dict = {}

        if full_workload:
            job_sequence_size = self.loads.size()
            self.start = 0
        else:
            job_sequence_size = JOB_SEQUENCE_SIZE
            assert self.batch_job_slice == 0 or self.batch_job_slice >= job_sequence_size
            if self.build_sjf:
                done = False
                while not done:
                    if self.batch_job_slice == 0:
                        self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
                    else:
                        self.start = self.np_random.randint(job_sequence_size,
                                                            (self.batch_job_slice - job_sequence_size - 1))
                    if self.sjf_scores[self.start] > 10 and self.sjf_scores[self.start] < 150:
                        done = True
            else:
                if self.batch_job_slice == 0:
                    self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
                else:
                    self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.pre_workloads = []

        self.schedule_time_list = []
        self.rl_schedule_time_list=[]
        self.makespan = 0

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        #print("printing self.num_job_in_batch here:", self.num_job_in_batch)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        #print("printing self.last_job_in_batch here:", self.last_job_in_batch)
        self.current_timestamp = self.loads[self.start].submit_time
        #print("printing self.current_timestamp here:", self.current_timestamp)
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1
        #print("printing self.next_arriving_job_idx here:", self.next_arriving_job_idx)

        if self.enable_preworkloads:
            self.gen_preworkloads(job_sequence_size + self.np_random.randint(job_sequence_size))

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.schedule_algo).values())) #now here

        #self.build_observation()
        #self.build_critic_observation()
        #print("batch start here ------------------------------------------------------------------------")
        #print("Printing scheduled_scores here:", self.scheduled_scores)
        #print("Printing mean scheduled_scores here:", self.scheduled_scores[0])
        #print("first job in batch scheduled at timestamp:", min(self.schedule_time_list))
        #print("last job in batch scheduled at timestamp:", max(self.schedule_time_list))
        #print("makespan = time required for scheduling 128 jobs(last job scheduled-first job scheduled)=", self.makespan)
        self.base_score_list_normalized.append(self.scheduled_scores)
        self.makespan_base.append(self.makespan)

        #self.build_observation()
        #self.build_critic_observation()
        return self.build_observation(), self.build_critic_observation()

        '''
        if (np.mean(self.scheduled_scores) > 5):
            return self.build_observation()
        else:
            return self.reset()
        '''

    def reset_for_test(self, num, start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        self.schedule_time_list = []
        self.rl_schedule_time_list=[]
        self.makespan = 0

        job_sequence_size = num
        assert self.batch_job_slice == 0 or self.batch_job_slice >= job_sequence_size
        #if self.batch_job_slice == 0:
        #    self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        #else:
        #    self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))
        self.start = start
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

    def skip_for_resources_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            if not self.running_jobs:
                return  # nothing to wait for; caller must handle (skip this job)
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            next_resource_release_job = self.running_jobs[0]

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.


    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)
        #print("Inside backfill function with job:",job)
        #print("Printing which gpu type is asked and how many gpus:", job.vc_id, job.request_number_of_gpus)
        earliest_start_time = self.current_timestamp
        running_jobs_vc_wise = []
        #job_queue_gpuwise = []
        # sort all running jobs by estimated finish time
        running_job_id = []
        for i in self.running_jobs:
          running_job_id.append(i.job_id)
        #print("Inside backfill function printing running jobs with job ids:", running_job_id)
        for i in self.running_jobs:
          if i.vc_id == job.vc_id:
            running_jobs_vc_wise.append(i)
        running_jobs_vc_wise.sort(key=lambda current_running_job: (current_running_job.scheduled_time + current_running_job.request_time))

        #print("printing inside backfill, total nodes before copying in dummy:",self.cluster.total_node[0], self.cluster.total_node[1], list(self.cluster.total_node[0].items()),list(self.cluster.total_node[1].items()), list(self.cluster.total_node[2].items()), list(self.cluster.total_node[3].items()), list(self.cluster.total_node[4].items()))
        self.cluster.dummy_cluster_status = [[(n, g) for (n, g) in L] for L in self.cluster.cluster_status]

        #free_cluster_status_gpuwise = self.cluster.total_node[gpu_type_here]
        for running_job in running_jobs_vc_wise:
            dummy_machines_to_release = running_job.allocated_machines
            #print("Printing inside backfill allocated machines for running job:", dummy_machines_to_release)
            self.cluster.dummy_release(running_job, dummy_machines_to_release)
            # call dummy_release to release machines held by running_job and make change in dummy_cluster with released machines.
            # write dummy_can_allocate function to check if we can allocate the upcomming job or not.
            # keep the variables intact untill can_allocate writes true. These functions will have all the temp variables
            # which will not impact actual state of the cluster.
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if self.cluster.dummy_can_allocated(job):
                break
        #print("Printing inside backfill earliest_start_time:", earliest_start_time)

        while not self.cluster.can_allocated(job):
            job_queue_vc_wise = []
            for i in self.job_queue:
              if i.vc_id == job.vc_id:
                job_queue_vc_wise.append(i)
            # try to backfill as many jobs as possible. Use FCFS
            #print("Printing earliest_start_time here:", earliest_start_time)
            job_queue_vc_wise.sort(key=lambda _j: self.fcfs_score(_j))
            #job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_vc_wise:
                #print("Printing self.current_timestamp + _j.request_time here:",self.current_timestamp + _j.request_time,self.job_queue)
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    #print("Printing self.current_timestamp + _j.request_time here:",self.current_timestamp + _j.request_time,self.job_queue)
                    if self.cluster.can_allocated(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        #print("Printing inside backfill scheduled time of current job:", _j.scheduled_time)
                        _j.allocated_machines = self.cluster.allocate(_j, _j.job_id, _j.request_number_of_gpus)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue
                        job_queue_vc_wise.remove(_j)

            # move to the next timestamp
            #print("Printing inside backfill_greedy running jobs:", self.running_jobs, len(self.running_jobs))
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            next_resource_release_job = self.running_jobs[0]

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job


    def post_process_score(self, scheduled_logs):
        scheduled_logs_len = len(scheduled_logs)
        if self.job_score_type == 0:
            # bsld
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 1:
            # wait time
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 2:
            # turnaround time
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 3:
            # Utilization: denominator = cluster GPU-seconds in this window (cluster-dependent)
            end_time = max(self.current_timestamp, *[i.scheduled_time+i.run_time for i in self.running_jobs])
            wall_seconds = end_time - self.loads[self.start].submit_time
            total_cluster_gpus = sum(self.cluster.total_nodes_every_gpu) * self.cluster.num_gpu_per_node
            total_gpu_seconds = max(1, wall_seconds * total_cluster_gpus)
            for i in scheduled_logs:
                scheduled_logs[i] /= total_gpu_seconds
        elif self.job_score_type == 4:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        else:
            raise NotImplementedError


    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm.
        scheduled_logs = {}
        # f = False
        # if score_fn.__name__ == "sjf_score":
        #     f = True
        #     num_total = 0
        # start_time = time.time()
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            # if f:
            #     num_total += 1
            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            #print("Printing inside schedule_curr_sequence_reset job:",job_for_scheduling.job_id, "Current timestamp:",self.current_timestamp, "how many gpus:",job_for_scheduling.request_number_of_gpus)
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfil:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    if self.running_jobs:
                        self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
                    if not self.cluster.can_allocated(job_for_scheduling):
                        # Still cannot schedule (no running jobs to wait for, or released all and still no fit). Skip this job.
                        self.job_queue.remove(job_for_scheduling)
                        not_empty = self.moveforward_for_job()
                        if not not_empty:
                            break
                        continue

            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            self.schedule_time_list.append(job_for_scheduling.scheduled_time)
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling, job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_gpus)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)
            not_empty = self.moveforward_for_job()
            #print("printing after moveforward job_queue and eun_queue:", len(self.job_queue), len(self.running_jobs))
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        # Debug: print cluster state after base run (before we reset for RL). Shows that allocations did happen.
        _total_free = sum(
            sum(g for (_, g) in self.cluster.cluster_status[vc_id])
            for vc_id in range(len(self.cluster.cluster_status))
        )
        _total_cap = sum(
            self.loads.total_nodes_count[vc_id] * self.cluster.num_gpu_per_node
            for vc_id in range(len(self.loads.total_nodes_count))
        )
        _jobs_scheduled = len(scheduled_logs)
        print("[base run] Jobs scheduled in this batch: %d | Cluster: free GPUs = %d / total capacity = %d (used = %d)"
              % (_jobs_scheduled, _total_free, _total_cap, _total_cap - _total_free))
        # reset again so RL episode starts from clean cluster
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.refill_preworkloads()
        list_job_id = []
        list_sched_score = []
        #print("printing scheduled_logs here:", scheduled_logs)
        self.makespan = max(self.schedule_time_list) - min(self.schedule_time_list)
        for key, val in scheduled_logs.items():
            list_job_id.append(key)
            list_sched_score.append(val)
            #print("printing job id:", key, "printing job score:", val)
        #print("printing job id:", list_job_id)
        #print("printing sched_score list:",list_sched_score)
        #print("Printing running job and job queue variable here schedule_curr_sequence_reset:", self.running_jobs, self.job_queue, len(self.running_jobs), len(self.job_queue))
        return scheduled_logs

    def build_observation_featureSampling(self, current_job):
        job = current_job
        job_feature_vector = []
        submit_time = job.submit_time
        request_gpus = job.request_number_of_gpus
        request_time = job.request_time
        # run_time = job.run_time
        wait_time = self.current_timestamp - submit_time

        # make sure that larger value is better.
        normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
        normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
        normalized_request_nodes = min(math.ceil(float(request_gpus) / float(self.loads.max_gpu)), 1.0 - 1e-5)

        vc_id = job.vc_id
        normalized_free_nodes = min(float(len(self.cluster.cluster_status[vc_id])) / max(1, float(self.loads.total_nodes_count[vc_id])), 1.0 - 1e-5)

        if self.cluster.can_allocated(job):
          can_schedule_now = 1.0 - 1e-5
        else:
          can_schedule_now = 1e-5

        normalized_skip_time = min(job.skip_time / float(MAX_SKIP_TIME), 1.0)
        delta_bsld = sum((float(SKIP_TIME) / max(i.request_time, 10)) for i in self.job_queue)
        normalized_delta_bsld = min(delta_bsld / float(SKIP_TIME/10 * JOB_SEQUENCE_SIZE), 1.0 - 1e-5)
        #wait_queue_time = sum(float(SKIP_TIME) for i in self.job_queue)
        #normalized_delta_bsld = min(wait_queue_time / float(SKIP_TIME * JOB_SEQUENCE_SIZE), 1.0 - 1e-5)

        job_feature_vector = [job.job_id, normalized_wait_time, normalized_run_time,
                        normalized_request_nodes, normalized_free_nodes, normalized_skip_time, normalized_delta_bsld, can_schedule_now]

        return job_feature_vector

    def build_observation(self):
        """Philly-style observation: 7 features per job, all metrics computed within job's pool (VC and/or GPU type)."""
        vector = np.zeros((MAX_QUEUE_SIZE) * JOB_FEATURES, dtype=float)
        self.job_queue.sort(key=lambda job: self.schedule_algo(job))
        self.pairs = []

        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            for i in range(0, len(self.job_queue)):
                self.visible_jobs.append(self.job_queue[i])
        else:
            for i in range(0, MAX_QUEUE_SIZE):
                self.visible_jobs.append(self.job_queue[i])

        gpus_per_node = self.cluster.num_gpu_per_node

        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < MAX_QUEUE_SIZE:
                job = self.visible_jobs[i]
                pool_id = self._get_pool_id(job)
                pool_status = self.cluster.cluster_status[pool_id]
                total_nodes_pool = self.loads.total_nodes_count[pool_id]
                total_pool_capacity = total_nodes_pool * gpus_per_node

                # Pool-specific: free GPUs and fragmentation (only within this VC/GPU-type pool)
                free_gpu_pool = sum(num_gpus for (node_id, num_gpus) in pool_status)
                num_partial = len([g for (_, g) in pool_status if 0 < g < gpus_per_node])
                cluster_fragmentation_factor = num_partial / max(len(pool_status), 1e-5)

                # Future GPU demand from visible jobs in the SAME pool only
                future_gpu_demand = sum(
                    j.request_number_of_gpus for j in self.visible_jobs if self._same_pool(job, j)
                ) - job.request_number_of_gpus

                submit_time = job.submit_time
                request_gpus = job.request_number_of_gpus
                request_time = job.request_time
                wait_time = self.current_timestamp - submit_time

                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_gpus) / float(self.loads.max_gpu), 1.0 - 1e-5)

                # Feasibility and ways-to-schedule from allocation_score (read-only; do not mutate selected_machines)
                # Option B: tier1 multiple -> 2, tier1 single -> 1; tier2 has ways -> +1. Total in {0, 1, 2, 3}.
                alloc_result = self.cluster.get_allocation_options(job, use_dummy=False)
                can_schedule = alloc_result is not None and len(alloc_result["tier1"]["all"]) > 0
                n_tier1 = len(alloc_result["tier1"]["all"]) if alloc_result else 0
                n_tier2 = len(alloc_result["tier2"]["all"]) if alloc_result else 0
                tier1_contribution = 2 if n_tier1 > 1 else (1 if n_tier1 == 1 else 0)
                tier2_contribution = 1 if n_tier2 > 0 else 0
                num_ways = tier1_contribution + tier2_contribution  # 0, 1, 2, or 3
                if can_schedule:
                    can_schedule_now = (1.0 - 1e-5) * 1.5 if num_ways >= 2 else (1.0 - 1e-5)
                else:
                    can_schedule_now = 1e-5
                # Normalize num_ways to [0, 1]: 0 -> ~0, 1 -> 1/3, 2 -> 2/3, 3 -> 1
                normalized_num_ways_to_schedule = min(num_ways / 3.0, 1.0 - 1e-5) if num_ways > 0 else 1e-5

                demand_supply_ratio = request_gpus / max(free_gpu_pool, 1e-5)
                normalized_demand_supply_ratio = min(demand_supply_ratio / max(total_pool_capacity, 1), 1.0 - 1e-5)

                near_future_availability = (free_gpu_pool - request_gpus) - future_gpu_demand
                # Normalize to [0, 1]: map [-total_cap, total_cap] -> [0, 1]
                normalized_near_future_availability = (near_future_availability + total_pool_capacity) / (2.0 * total_pool_capacity + 1e-5)
                normalized_near_future_availability = max(0.0, min(1.0 - 1e-5, normalized_near_future_availability))

                self.pairs.append([job, normalized_wait_time, normalized_request_nodes,
                                  normalized_demand_supply_ratio, normalized_near_future_availability,
                                  cluster_fragmentation_factor, normalized_num_ways_to_schedule, can_schedule_now])
            elif self.pivot_job:
                self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1])
            else:
                self.pairs.append([None, 0, 1, 1, 1, 1, 1, 0])

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i * JOB_FEATURES:(i + 1) * JOB_FEATURES] = self.pairs[i][1:]
        return vector


    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3, dtype=float)          # critic vector size is 128 * 3
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        pairs = []
        # Batch job indices are [start_idx_last_reset, last_job_in_batch); last_job_in_batch is exclusive
        for i in range(self.start_idx_last_reset, self.last_job_in_batch):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_gpus = job.request_number_of_gpus
            request_time = job.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(math.ceil(float(request_gpus) / float(self.loads.max_gpu)), 1.0 - 1e-5)

            pairs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        # Pad if batch has fewer jobs than JOB_SEQUENCE_SIZE (e.g. full_workload with small CSV)
        while len(pairs) < JOB_SEQUENCE_SIZE:
            pairs.append([0.0, 0.0, 0.0])
        for i in range(JOB_SEQUENCE_SIZE):
            vector[i * 3:(i + 1) * 3] = pairs[i]
        #print("Printing critic vector size here:",vector.shape)
        return vector



    def moveforward_for_resources_backfill(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)
        #print("Inside backfill function with job:",job)

        earliest_start_time = self.current_timestamp
        running_jobs_vc_wise = []
        #job_queue_gpuwise = []
        # sort all running jobs by estimated finish time
        for i in self.running_jobs:
          if i.vc_id == job.vc_id:
            running_jobs_vc_wise.append(i)
        #running_jobs_gpuwise = []
        #job_queue_gpuwise = []
        # sort all running jobs by estimated finish time
        running_jobs_vc_wise.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        self.cluster.dummy_cluster_status = [[(n, g) for (n, g) in L] for L in self.cluster.cluster_status]

        #free_cluster_status_gpuwise = self.cluster.total_node[gpu_type_here]
        for running_job in running_jobs_vc_wise:
            dummy_machines_to_release = running_job.allocated_machines
            self.cluster.dummy_release(running_job, dummy_machines_to_release)
            # call dummy_release to release machines held by running_job and make change in dummy_cluster with released machines.
            # write dummy_can_allocate function to check if we can allocate the upcomming job or not.
            # keep the variables intact intill can_allocate writes true. These functions will all the temp variables
            # which will not impact actuall state of the cluster.
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if self.cluster.dummy_can_allocated(job):
                break

        while not self.cluster.can_allocated(job):

            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            #job_queue_iter_copy = list(self.job_queue)
            for _j in self.job_queue:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j, _j.job_id, _j.request_number_of_gpus)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        self.scheduled_rl[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            next_resource_release_job = self.running_jobs[0]

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
            #print("Printing inside backfill running jobs after else:", self.running_jobs, len(self.running_jobs))

    def skip_for_resources(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            if not self.running_jobs:
                return  # nothing to wait for; caller must handle (drop this job)
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            next_resource_release_job = self.running_jobs[0]

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def moveforward_for_job(self):
        if self.job_queue:                      # I need to add jobs in queue if any job has submit_time<self.current_timestamp
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            #print("Printing inside moveforward for job running jobs:", self.running_jobs, len(self.running_jobs))
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines
                next_resource_release_job = self.running_jobs[0]

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True  # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == 0:
            # bsld
            _tmp = max(1.0, (float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                             /
                             max(job_for_scheduling.run_time, 10)))
        elif self.job_score_type == 1:
            # wait time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
        elif self.job_score_type == 2:
            # turnaround time
            _tmp = float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif self.job_score_type == 3:
            # utilization
            #free_gpu = sum(num_gpus for node_id, num_gpus in self.cluster.cluster_status)
            _tmp = -float(job_for_scheduling.run_time * job_for_scheduling.request_number_of_gpus)
        elif self.job_score_type == 4:
            # sld
            _tmp = float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time) \
                   / job_for_scheduling.run_time
        else:
            raise NotImplementedError

            # Weight larger jobs.
        # _tmp = _tmp * (job_for_scheduling.run_time * job_for_scheduling.request_number_of_processors)
        return _tmp

    def has_only_one_job(self):
        if len(self.job_queue) == 1:
            return True
        else:
            return False

    def skip_schedule(self):
        # schedule nothing, just move forward to next timestamp. It should 1) add a new job; 2) finish a running job; 3) reach skip time
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        #print("Printing inside skip_schedule:", self.running_jobs, len(self.running_jobs))
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            next_resource_release_job = self.running_jobs[0]

        if self.next_arriving_job_idx >= self.last_job_in_batch and not self.running_jobs:
            if not self.pivot_job:
                self.pivot_job = True
                return False, 0
            else:
                return False, 0

        if next_time_after_skip < min(self.loads[self.next_arriving_job_idx].submit_time, next_resource_release_time):
            self.current_timestamp = next_time_after_skip
            return False, 0

        if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
            self.next_arriving_job_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_job, next_resource_release_machines)
            self.running_jobs.pop(0)  # remove the first running job.
        return False, 0

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        self.scheduled_time_list = [] #maybe this should not be here
        if not self.cluster.can_allocated(job_for_scheduling):
            if self.backfil:
                self.moveforward_for_resources_backfill(job_for_scheduling)
            else:
                self.skip_for_resources(job_for_scheduling)
            if not self.cluster.can_allocated(job_for_scheduling):
                # Still cannot schedule (no running jobs to wait for). Drop this job and continue.
                self.job_queue.remove(job_for_scheduling)
                self.moveforward_for_job()
                return False  # not done; caller will get new obs and take another step

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        # Philly-style: choose among allocation options (tier1/tier2, defrag vs load-balance) using top-K future jobs
        if self.use_milp_allocation:
            top_k = self._get_top_k_jobs(self.TOP_K_MILP)
            alloc_result = self.cluster.get_allocation_options(job_for_scheduling, use_dummy=False, allow_relax_min_nodes=True)
            if alloc_result and len(_collect_allocation_ways(alloc_result)) > 1:
                pool = list(self.cluster.cluster_status[job_for_scheduling.vc_id])
                pool_3d = _pool_to_3d(pool)
                chosen = choose_allocation_milp_solver(pool_3d, alloc_result, self.cluster.num_gpu_per_node, top_k)
                if chosen is not None:
                    self.cluster.selected_machines = chosen
        job_for_scheduling.scheduled_time = self.current_timestamp
        self.rl_schedule_time_list.append(job_for_scheduling.scheduled_time)
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling, job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_gpus)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.score_rl = score
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue
        #print("Printing inside schedule running jobs:", self.running_jobs, len(self.running_jobs))

        # after scheduling, check if job queue is empty, try to add jobs.
        #print("Inside Schedule function printing job queue:", self.job_queue, len(self.job_queue))
        not_empty = self.moveforward_for_job()
        #print("Inside Schedule function printing value for not_empty:", not_empty)

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]


    def step(self, a):
        #print("inside step function printing a[0]=", a)
        #will_skip = a
        #job_for_scheduling = min(self.job_queue, key=self.schedule_algo)
        job_for_scheduling = self.pairs[a][0]
        self.last_action_scheduled = (job_for_scheduling is not None)  # for per-step reward in improved train
        #print("Printing inside step function job id of job for scheduling:",job_for_scheduling.job_id )
        '''
        running_job_id=[]
        for i in self.running_jobs:
            running_job_id.append(i.job_id)
        '''
        #print("Printing job ids of running job inside step function:", running_job_id)
        #print("Printing inside step function running jobs and job_queue:",  len(self.running_jobs), len(self.job_queue))
        if not job_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)
            #job_vector = self.build_observation_featureSampling(job_for_scheduling)
            #self.vector_score_dict[job_for_scheduling] = [job_vector, self.score_rl]

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0, 0, 0]
        else:
            list_job_id = []
            list_rl_score = []
            #print("printing scheduled_logs here:", scheduled_logs)
            for key, val in self.scheduled_rl.items():
              list_job_id.append(key)
              list_rl_score.append(val)
            #print("printing job id list:", list_job_id)
            #print("printing rl_score list:",list_rl_score)
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            #print("printing rl_total:",rl_total)
            self.rl_score_normalized.append(rl_total)
            best_total = min(self.scheduled_scores)
            #print("printing self.scheduled_scores:",self.scheduled_scores)
            #sjf = self.scheduled_scores[1]
            #f1 = self.scheduled_scores[2]
            sched_algo_reward = self.scheduled_scores[0]
            rwd2 = (sched_algo_reward - rl_total)
            rwd = (sched_algo_reward - rl_total)
            #print("Printing inside step function sched_algo_reward, rl_total:", sched_algo_reward, rl_total)
            rwd = (sched_algo_reward - rl_total)/max(sched_algo_reward, rl_total,1)
            self.makespan = max(self.rl_schedule_time_list) - min(self.rl_schedule_time_list)
            self.makespan_rl.append(self.makespan)
            #rwd = (- rl_total)

            #if rwd < 0:
            #    rwd = -1
            #elif rwd == 0:
            #    rwd = 0
            #else:
            #    rwd = 1
            #print("printing len of dict in step:",len(self.vector_score_dict))
            list_256_feature_score = []
            for key,val in self.vector_score_dict.items():
              list_256_feature_score.append(val)
            #print("Printing list_256_feature_score:",list_256_feature_score)
            return [None, rwd, True, rwd2, sched_algo_reward, rl_total]

    def step_for_test_skiptype(self, a):
        will_skip = a
        job_for_scheduling = min(self.job_queue, key=self.schedule_algo)

        if will_skip==1:
            # print("SKIP", end=" ")
            job_for_scheduling.skip_time += SKIP_TIME
            done, _ = self.skip_schedule()
        else:
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]

    def step_for_test_picktype(self, a):
        job_for_scheduling = self.pairs[a][0]

        if not job_for_scheduling:
            # print("SKIP", end=" ")
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnvSkip(batch_job_slice=0, sched_algo=0 ,job_score_type=1)
    env.seed(0)
    env.my_init(workload_file=workload_file)
