"""
Philly trace v2: same flow as Helios v2 (tier1/tier2, defrag/load_balance, lex base, MILP for RL).
Cluster: 16 x 2-GPU nodes (P100), 20 x 8-GPU nodes (P100). No VCs; column 18 = -1, column 19 = 3.
No mixing: each job uses either only 2-GPU nodes or only 8-GPU nodes.
- request <= 4 GPUs: can use 2-GPU pool or 8-GPU pool (tier1/tier2 per pool, merge and pick by defrag/lb + lex).
- request > 4 GPUs: only 8-GPU pool.
- No tier2 when request == 1 GPU.
"""
import os
import re
import sys
import math
import copy
import numpy as np
import heapdict

from helios_env_base_v2 import (
    HPCEnvSkip,
    MAX_QUEUE_SIZE,
    JOB_FEATURES,
    JOB_SEQUENCE_SIZE,
    MAX_WAIT_TIME,
    SKIP_TIME,
    Machine,
    _collect_allocation_ways,
    _pool_to_3d,
    choose_allocation_milp_solver,
)
from allocation_score import find_gpu_assignments

# Philly: 16 nodes x 2 GPUs (node_id 0-15), 20 nodes x 8 GPUs (node_id 16-35)
PHILLY_NODES_2GPU = 16
PHILLY_NODES_8GPU = 20
PHILLY_TOTAL_NODES = PHILLY_NODES_2GPU + PHILLY_NODES_8GPU
PHILLY_GPUS_PER_NODE_SMALL = 2
PHILLY_GPUS_PER_NODE_LARGE = 8


class Job_philly:
    """Same CSV columns as Helios; col 18 = VC (-1 for Philly), col 19 = GPU type (3 = P100)."""
    def __init__(self, line):
        line = line.strip()
        s_array = re.split(',', line)
        self.job_id = s_array[0]
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.skip_time = 0
        if self.run_time == -1:
            self.run_time = 10
        self.vc_id = int(s_array[18]) if len(s_array) > 18 else -1
        self.gpu_type_id = int(s_array[19]) if len(s_array) > 19 else 3
        self.number_of_allocated_gpus = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])
        self.request_number_of_cpus = int(s_array[7])
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
        self.think_time_from_proceeding_job = int(s_array[17]) if len(s_array) > 17 else 0
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
        return "J[" + str(self.job_id) + "]-[" + str(self.request_number_of_gpus) + "]-[" + str(self.submit_time) + "]-[" + str(self.request_time) + "]"


class Workload_philly:
    """Load Philly CSV. cluster_status[0] = 2-GPU nodes (0..15), cluster_status[1] = 8-GPU nodes (16..35)."""
    def __init__(self, path, cluster_name='philly'):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_gpu = 0

        # pool 0: 2-GPU nodes (node_id 0..15), pool 1: 8-GPU nodes (16..35)
        pool_2 = [(i, PHILLY_GPUS_PER_NODE_SMALL) for i in range(PHILLY_NODES_2GPU)]
        pool_8 = [(i, PHILLY_GPUS_PER_NODE_LARGE) for i in range(PHILLY_NODES_2GPU, PHILLY_TOTAL_NODES)]
        self.cluster_status = [pool_2, pool_8]
        self.total_nodes_count = [PHILLY_NODES_2GPU, PHILLY_NODES_8GPU]
        self.assigned_nodes = [[], []]
        self.max_gpu = PHILLY_GPUS_PER_NODE_LARGE  # for normalization

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    continue
                j = Job_philly(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_number_of_gpus > self.max:
                    self.max = j.request_number_of_gpus
                self.all_jobs.append(j)

        self.all_jobs.sort(key=lambda job: job.submit_time)
        print("Philly workload: max GPUs", self.max, "jobs", len(self.all_jobs),
              "cluster_status pools", [len(p) for p in self.cluster_status])

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1
            job.skip_time = 0

    def __getitem__(self, item):
        return self.all_jobs[item]


def _merge_philly_allocation_results(r2, r8, request_gpus):
    """
    Merge tier1/tier2 from 2-GPU pool and 8-GPU pool. No tier2 when request_gpus <= 1.
    Re-sort combined tier1 and tier2 by defrag and load_balance (same keys as Helios).
    """
    allow_tier2 = request_gpus > 1
    tier1_all = (r2.get("tier1", {}).get("all", []) or []) + (r8.get("tier1", {}).get("all", []) or [])
    tier2_all = []
    if allow_tier2:
        tier2_all = (r2.get("tier2", {}).get("all", []) or []) + (r8.get("tier2", {}).get("all", []) or [])

    min_nodes_theoretical = min(
        r2.get("min_nodes_theoretical", 999),
        r8.get("min_nodes_theoretical", 999),
    ) if (r2.get("assignments") or r8.get("assignments")) else 0

    def sort_defrag(L):
        return sorted(
            L,
            key=lambda a: (
                a["frag_score"][0],
                a["frag_score"][1],
                a["lb_score"],
                a["nodes_used"],
                [x for (x, _) in sorted(a["assignment"], key=lambda x: (x[0], x[1]))],
            ),
        )

    def sort_lb(L):
        return sorted(
            L,
            key=lambda a: (
                a["lb_score"],
                a["frag_score"][0],
                a["frag_score"][1],
                a["nodes_used"],
                [x for (x, _) in sorted(a["assignment"], key=lambda x: (x[0], x[1]))],
            ),
        )

    return {
        "min_nodes_theoretical": min_nodes_theoretical,
        "assignments": (r2.get("assignments") or []) + (r8.get("assignments") or []),
        "tier1": {
            "all": tier1_all,
            "defrag_sorted": sort_defrag(tier1_all),
            "load_balance_sorted": sort_lb(tier1_all),
        },
        "tier2": {
            "all": tier2_all,
            "defrag_sorted": sort_defrag(tier2_all),
            "load_balance_sorted": sort_lb(tier2_all),
        },
    }


class Cluster_philly:
    """
    Two pools: cluster_status[0] = 2-GPU nodes (node_id 0..15), cluster_status[1] = 8-GPU nodes (16..35).
    get_allocation_options merges results from both pools when request <= 4; >4 uses only pool 1.
    """
    def __init__(self, cluster_name, cluster_status, total_nodes_count, assigned_nodes):
        self.name = cluster_name
        self.cluster_status = cluster_status
        self.used_node = assigned_nodes
        self.total_nodes_every_gpu = total_nodes_count
        self.num_gpu_per_node = PHILLY_GPUS_PER_NODE_LARGE  # for normalization / MILP
        self.selected_machines = []
        self.dummy_cluster_status = []

        self.all_nodes = []
        for i in range(PHILLY_NODES_2GPU):
            self.all_nodes.append(Machine(i, 'pool0', PHILLY_GPUS_PER_NODE_SMALL))
        for i in range(PHILLY_NODES_2GPU, PHILLY_TOTAL_NODES):
            self.all_nodes.append(Machine(i, 'pool1', PHILLY_GPUS_PER_NODE_LARGE))

    def get_allocation_options(self, job, use_dummy=False, allow_relax_min_nodes=False):
        req = job.request_number_of_gpus
        if req <= 0:
            return None
        # No tier2 when just 1 GPU
        allow_tier2 = allow_relax_min_nodes and req > 1

        src = self.dummy_cluster_status if use_dummy else self.cluster_status
        pool_2 = list(src[0])
        pool_8 = list(src[1])

        if req > 4:
            r8 = find_gpu_assignments(pool_8, req, PHILLY_GPUS_PER_NODE_LARGE, allow_relax_min_nodes=allow_tier2)
            return r8 if r8.get("assignments") else None
        else:
            r2 = find_gpu_assignments(pool_2, req, PHILLY_GPUS_PER_NODE_SMALL, allow_relax_min_nodes=allow_tier2)
            r8 = find_gpu_assignments(pool_8, req, PHILLY_GPUS_PER_NODE_LARGE, allow_relax_min_nodes=allow_tier2)
            merged = _merge_philly_allocation_results(r2, r8, req)
            return merged if merged["assignments"] else None

    def _pick_greedy_assignment(self, result):
        """Lex order on combined tier1 (same as Helios)."""
        tier1 = result.get("tier1", {}).get("all", [])
        if not tier1:
            return None
        def key_fn(a):
            return tuple(sorted(a["assignment"], key=lambda x: (x[0], x[1])))
        chosen = min(tier1, key=key_fn)
        return chosen["assignment"]

    def can_allocated(self, job):
        self.selected_machines = []
        result = self.get_allocation_options(job, use_dummy=False, allow_relax_min_nodes=True)
        if result is None:
            return False
        assignment = self._pick_greedy_assignment(result)
        if assignment is None:
            return False
        self.selected_machines = list(assignment)
        return True

    def dummy_can_allocated(self, job):
        result = self.get_allocation_options(job, use_dummy=True, allow_relax_min_nodes=True)
        if result is None:
            return False
        return len(result.get("tier1", {}).get("all", [])) > 0

    def allocate(self, job, job_id, request_num_gpus):
        allocated_nodes = []
        gpu_index = [node for node, gpu in self.selected_machines]
        gpu_count = [gpu for node, gpu in self.selected_machines]
        assert sum(gpu_count) == request_num_gpus

        for i in range(len(gpu_index)):
            nid = gpu_index[i]
            allocated_nodes.append((self.all_nodes[nid], gpu_count[i]))

        for i in range(len(gpu_index)):
            nid = gpu_index[i]
            gpu_cnt = gpu_count[i]
            pool_id = 0 if nid < PHILLY_NODES_2GPU else 1
            new_list = []
            for (n, f) in self.cluster_status[pool_id]:
                if n == nid:
                    new_free = f - gpu_cnt
                    if new_free == 0:
                        self.used_node[pool_id].append((nid, 0))
                    else:
                        new_list.append((nid, new_free))
                else:
                    new_list.append((n, f))
            self.cluster_status[pool_id] = new_list

        result_machines = []
        for m, gpu_cnt in allocated_nodes:
            m.taken_by_job(job_id, gpu_cnt)
            result_machines.append(m)
        self.selected_machines = []
        return result_machines

    def release(self, job_release, machine_releases):
        for m in machine_releases:
            gpu_status, _ = m.machine_release(job_release)
            pool_name, idx_str = m.id.split()
            nid = int(idx_str)
            pool_id = 0 if pool_name == 'pool0' else 1
            found = False
            for i, (node_id, gpus) in enumerate(self.cluster_status[pool_id]):
                if node_id == nid:
                    self.cluster_status[pool_id][i] = (nid, gpu_status)
                    found = True
                    break
            if not found:
                for i, (node_id, gpus) in enumerate(self.used_node[pool_id]):
                    if node_id == nid:
                        self.used_node[pool_id].pop(i)
                        self.cluster_status[pool_id].append((nid, gpu_status))
                        break

    def dummy_release(self, job_release, machine_releases):
        for m in machine_releases:
            v = m.dummy_machine_release(job_release)
            pool_name, idx_str = m.id.split()
            nid = int(idx_str)
            pool_id = 0 if pool_name == 'pool0' else 1
            found = False
            for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[pool_id]):
                if node_id == nid:
                    self.dummy_cluster_status[pool_id][i] = (nid, gpus + v)
                    found = True
                    break
            if not found:
                self.dummy_cluster_status[pool_id].append((nid, v))

    def get_combined_pool(self, use_dummy=False):
        """Combined list (node_id, free_gpus) for both pools; for MILP / observation."""
        src = self.dummy_cluster_status if use_dummy else self.cluster_status
        return list(src[0]) + list(src[1])

    def reset(self):
        self.cluster_status[0] = [(i, PHILLY_GPUS_PER_NODE_SMALL) for i in range(PHILLY_NODES_2GPU)]
        self.cluster_status[1] = [(i, PHILLY_GPUS_PER_NODE_LARGE) for i in range(PHILLY_NODES_2GPU, PHILLY_TOTAL_NODES)]
        self.used_node[0] = []
        self.used_node[1] = []
        for m in self.all_nodes:
            cap = PHILLY_GPUS_PER_NODE_SMALL if m.id.startswith('pool0') else PHILLY_GPUS_PER_NODE_LARGE
            m.reset(cap)


class PhillyEnvSkip(HPCEnvSkip):
    """Philly v2 env: two pools (2-GPU / 8-GPU), merged allocation, combined pool for observation."""

    def _get_pool_id(self, job):
        """Single logical pool (all P100) for observation."""
        return 0

    def _same_pool(self, job, other):
        """All jobs share the same logical pool."""
        return True

    def my_init(self, workload_file='', cluster_name='philly'):
        print("loading workloads from dataset:", workload_file, "cluster:", cluster_name)
        self.loads = Workload_philly(workload_file, cluster_name=cluster_name)
        self.cluster = Cluster_philly(
            "Philly",
            copy.deepcopy(self.loads.cluster_status),
            self.loads.total_nodes_count,
            copy.deepcopy(self.loads.assigned_nodes),
        )
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.batch_job_slice == 0:
            max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1
        else:
            max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1
        if max_index <= 0:
            raise ValueError("Philly workload too small for batch_job_slice / JOB_SEQUENCE_SIZE")
        self.loads.reset()

    def build_observation(self):
        """Observation: use combined pool free GPUs and total capacity (one logical pool)."""
        self.job_queue.sort(key=lambda job: self.schedule_algo(job))
        self.visible_jobs = []
        for i in range(min(len(self.job_queue), MAX_QUEUE_SIZE)):
            if i < len(self.job_queue):
                self.visible_jobs.append(self.job_queue[i])

        # Combined pool stats for Philly
        free_gpu_pool = sum(g for (_, g) in self.cluster.cluster_status[0]) + sum(g for (_, g) in self.cluster.cluster_status[1])
        total_pool_capacity = PHILLY_NODES_2GPU * PHILLY_GPUS_PER_NODE_SMALL + PHILLY_NODES_8GPU * PHILLY_GPUS_PER_NODE_LARGE
        gpus_per_node = self.cluster.num_gpu_per_node  # 8 for normalization

        self.pairs = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < MAX_QUEUE_SIZE:
                job = self.visible_jobs[i]
                num_partial = len([g for (_, g) in self.cluster.cluster_status[0] if 0 < g < PHILLY_GPUS_PER_NODE_SMALL])
                num_partial += len([g for (_, g) in self.cluster.cluster_status[1] if 0 < g < PHILLY_GPUS_PER_NODE_LARGE])
                cluster_fragmentation_factor = num_partial / max(PHILLY_TOTAL_NODES, 1e-5)

                future_gpu_demand = sum(j.request_number_of_gpus for j in self.visible_jobs) - job.request_number_of_gpus

                submit_time = job.submit_time
                request_gpus = job.request_number_of_gpus
                wait_time = self.current_timestamp - submit_time
                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_gpus) / float(self.loads.max_gpu), 1.0 - 1e-5)

                alloc_result = self.cluster.get_allocation_options(job, use_dummy=False, allow_relax_min_nodes=True)
                can_schedule = alloc_result is not None and len(alloc_result.get("tier1", {}).get("all", [])) > 0
                n_tier1 = len(alloc_result["tier1"]["all"]) if alloc_result else 0
                n_tier2 = len(alloc_result["tier2"]["all"]) if alloc_result else 0
                tier1_contribution = 2 if n_tier1 > 1 else (1 if n_tier1 == 1 else 0)
                tier2_contribution = 1 if n_tier2 > 0 else 0
                num_ways = tier1_contribution + tier2_contribution
                if can_schedule:
                    can_schedule_now = (1.0 - 1e-5) * 1.5 if num_ways >= 2 else (1.0 - 1e-5)
                else:
                    can_schedule_now = 1e-5
                normalized_num_ways_to_schedule = min(num_ways / 3.0, 1.0 - 1e-5) if num_ways > 0 else 1e-5

                demand_supply_ratio = request_gpus / max(free_gpu_pool, 1e-5)
                normalized_demand_supply_ratio = min(demand_supply_ratio / max(total_pool_capacity, 1), 1.0 - 1e-5)
                near_future_availability = (free_gpu_pool - request_gpus) - future_gpu_demand
                normalized_near_future_availability = (near_future_availability + total_pool_capacity) / (2.0 * total_pool_capacity + 1e-5)
                normalized_near_future_availability = max(0.0, min(1.0 - 1e-5, normalized_near_future_availability))

                normalized_free_nodes = min(float(free_gpu_pool) / max(total_pool_capacity, 1e-5), 1.0 - 1e-5)

                self.pairs.append([job, normalized_wait_time, normalized_request_nodes,
                                  normalized_demand_supply_ratio, normalized_near_future_availability,
                                  cluster_fragmentation_factor, normalized_num_ways_to_schedule, can_schedule_now])
            elif self.pivot_job:
                self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1])
            else:
                self.pairs.append([None, 0, 1, 1, 1, 1, 1, 0])

        vector = np.zeros(MAX_QUEUE_SIZE * JOB_FEATURES, dtype=float)
        for i in range(0, MAX_QUEUE_SIZE):
            vector[i * JOB_FEATURES:(i + 1) * JOB_FEATURES] = self.pairs[i][1:]
        return vector

    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        assert not self.cluster.can_allocated(job)
        earliest_start_time = self.current_timestamp
        running_jobs_list = list(self.running_jobs)
        running_jobs_list.sort(key=lambda r: (r.scheduled_time + r.request_time))
        self.cluster.dummy_cluster_status = [[(n, g) for (n, g) in L] for L in self.cluster.cluster_status]

        for running_job in running_jobs_list:
            self.cluster.dummy_release(running_job, running_job.allocated_machines)
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if self.cluster.dummy_can_allocated(job):
                break

        while not self.cluster.can_allocated(job):
            job_queue_sorted = list(self.job_queue)
            job_queue_sorted.sort(key=lambda _j: self.fcfs_score(_j))
            for _j in job_queue_sorted:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        assert _j.scheduled_time == -1
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j, _j.job_id, _j.request_number_of_gpus)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)
                        self.scheduled_rl[_j.job_id] = score
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)

            assert self.running_jobs
            self.running_jobs.sort(key=lambda r: (r.scheduled_time + r.run_time))
            next_resource_release_time = self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time
            next_resource_release_job = self.running_jobs[0]

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_job, next_resource_release_job.allocated_machines)
                self.running_jobs.pop(0)

    def schedule(self, job_for_scheduling):
        """RL step: same as Helios; MILP uses combined pool and pool_3d. One job per call."""
        if not self.cluster.can_allocated(job_for_scheduling):
            if self.backfil:
                self.moveforward_for_resources_backfill_greedy(job_for_scheduling, {})
            else:
                self.skip_for_resources_greedy(job_for_scheduling, {})
            if not self.cluster.can_allocated(job_for_scheduling):
                self.job_queue.remove(job_for_scheduling)
                self.moveforward_for_job()
                return False

        if self.use_milp_allocation:
            top_k = self._get_top_k_jobs(self.TOP_K_MILP)
            alloc_result = self.cluster.get_allocation_options(job_for_scheduling, use_dummy=False, allow_relax_min_nodes=True)
            if alloc_result and len(_collect_allocation_ways(alloc_result)) > 1:
                pool = self.cluster.get_combined_pool(use_dummy=False)
                pool_3d = _pool_to_3d(pool)
                chosen = choose_allocation_milp_solver(pool_3d, alloc_result, self.cluster.num_gpu_per_node, top_k)
                if chosen is not None:
                    self.cluster.selected_machines = chosen

        assert job_for_scheduling.scheduled_time == -1
        job_for_scheduling.scheduled_time = self.current_timestamp
        self.rl_schedule_time_list.append(job_for_scheduling.scheduled_time)
        job_for_scheduling.allocated_machines = self.cluster.allocate(
            job_for_scheduling, job_for_scheduling.job_id, job_for_scheduling.request_number_of_gpus
        )
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)
        self.score_rl = score
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)

        not_empty = self.moveforward_for_job()
        return not not_empty

    def post_process_score(self, scheduled_logs):
        """Override for utilization (job_score_type==3): Philly total GPU count is 16*2+20*8."""
        scheduled_logs_len = len(scheduled_logs)
        if self.job_score_type == 0:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 1:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 2:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        elif self.job_score_type == 3:
            end_time = max(self.current_timestamp, *[r.scheduled_time + r.run_time for r in self.running_jobs]) if self.running_jobs else self.current_timestamp
            wall_seconds = end_time - self.loads[self.start].submit_time
            total_cluster_gpus = PHILLY_NODES_2GPU * PHILLY_GPUS_PER_NODE_SMALL + PHILLY_NODES_8GPU * PHILLY_GPUS_PER_NODE_LARGE
            total_gpu_seconds = max(1, wall_seconds * total_cluster_gpus)
            for i in scheduled_logs:
                scheduled_logs[i] /= total_gpu_seconds
        elif self.job_score_type == 4:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        else:
            raise NotImplementedError
