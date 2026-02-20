"""
Alibaba trace v2: same logic as Helios v2 (tier1/tier2, defrag/load_balance, lex base, MILP for RL).
No VCs: column 18 = -1, column 19 = GPU type (0-4).
5 GPU types: 0=T4 (8 nodes x 2 GPUs), 1=MISC (8 x 8), 2=P100 (15 x 2), 3=V100_16GB (10 x 8), 4=V100_32GB (8 x 8).
Each job is bound to one GPU type (pool). Per-pool tier1/tier2 same as Helios.
"""
import re
import sys
import copy
import numpy as np

from helios_env_base_v2 import (
    HPCEnvSkip,
    MAX_QUEUE_SIZE,
    JOB_FEATURES,
    JOB_SEQUENCE_SIZE,
    MAX_WAIT_TIME,
    Machine,
    _collect_allocation_ways,
    _pool_to_3d,
    choose_allocation_milp_solver,
)
from allocation_score import find_gpu_assignments

# Alibaba: 5 GPU types, (node_count, gpus_per_node) per type
# 0=T4, 1=MISC, 2=P100, 3=V100_16GB, 4=V100_32GB
ALIBABA_NODE_COUNTS = [8, 8, 15, 10, 8]
ALIBABA_GPUS_PER_NODE = [2, 8, 2, 8, 8]
ALIBABA_POOL_NAMES = ['gpu0', 'gpu1', 'gpu2', 'gpu3', 'gpu4']
NUM_POOLS = 5


class Job_alibaba:
    """CSV: 20 columns. Column 18 = VC (-1), Column 19 = GPU type (0-4)."""
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
        # Column 18 (1-based) = VC (-1), column 19 (1-based) = GPU type (0-4); 0-based indices 17, 18
        self.vc_id = int(s_array[17]) if len(s_array) > 17 else -1
        self.gpu_type_id = int(s_array[18]) if len(s_array) > 18 else 0
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


class Workload_alibaba:
    """Load Alibaba CSV. 5 pools by GPU type; each pool has (node_id, gpus_per_node) per node."""
    def __init__(self, path, cluster_name='alibaba'):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_gpu = max(ALIBABA_GPUS_PER_NODE)

        self.cluster_status = []
        self.total_nodes_count = list(ALIBABA_NODE_COUNTS)
        self.assigned_nodes = [[] for _ in range(NUM_POOLS)]

        for pool_id in range(NUM_POOLS):
            n_nodes = ALIBABA_NODE_COUNTS[pool_id]
            gpu_per_node = ALIBABA_GPUS_PER_NODE[pool_id]
            self.cluster_status.append([(i + 1, gpu_per_node) for i in range(n_nodes)])

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    continue
                j = Job_alibaba(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_number_of_gpus > self.max:
                    self.max = j.request_number_of_gpus
                self.all_jobs.append(j)

        self.all_jobs.sort(key=lambda job: job.submit_time)
        print("Alibaba workload: max GPUs", self.max, "jobs", len(self.all_jobs),
              "pools", NUM_POOLS, "node_counts", self.total_nodes_count)

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1
            job.skip_time = 0

    def __getitem__(self, item):
        return self.all_jobs[item]


class Cluster_alibaba:
    """
    5 pools by GPU type. Per-pool gpus_per_node: [2, 8, 2, 8, 8].
    Pool index = job.gpu_type_id. Same tier1/tier2 logic as Helios via find_gpu_assignments.
    """
    def __init__(self, cluster_name, cluster_status, gpus_per_node_list, assigned_nodes, total_nodes_count):
        self.name = cluster_name
        self.cluster_status = cluster_status
        self.used_node = assigned_nodes
        self.total_nodes_every_gpu = total_nodes_count
        self.num_gpu_per_node = gpus_per_node_list  # list of 5
        self.selected_machines = []
        self.dummy_cluster_status = []
        self.all_nodes = []

        for pool_id in range(NUM_POOLS):
            gpu_per_node = gpus_per_node_list[pool_id]
            pool_name = ALIBABA_POOL_NAMES[pool_id]
            for i in range(len(cluster_status[pool_id])):
                self.all_nodes.append(Machine(i + 1, pool_name, gpu_per_node))

    def get_allocation_options(self, job, use_dummy=False, allow_relax_min_nodes=False):
        pool_id = job.gpu_type_id
        pool = self.dummy_cluster_status[pool_id] if use_dummy else self.cluster_status[pool_id]
        if not pool:
            return None
        free_nodes = list(pool)
        gpus_per_node = self.num_gpu_per_node[pool_id]
        result = find_gpu_assignments(
            free_nodes=free_nodes,
            request_gpus=job.request_number_of_gpus,
            gpus_per_node=gpus_per_node,
            allow_relax_min_nodes=allow_relax_min_nodes,
        )
        if not result["assignments"]:
            return None
        return result

    def _pick_greedy_assignment(self, result):
        tier1 = result["tier1"]["all"]
        if not tier1:
            return None
        def key_fn(a):
            return tuple(sorted(a["assignment"], key=lambda x: (x[0], x[1])))
        chosen = min(tier1, key=key_fn)
        return chosen["assignment"]

    def can_allocated(self, job):
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
        result = self.get_allocation_options(job, use_dummy=True)
        if result is None:
            return False
        return len(result["tier1"]["all"]) > 0

    def allocate(self, job, job_id, request_num_gpus):
        allocated_nodes = []
        gpu_index = [node for node, gpu in self.selected_machines]
        gpu_count = [gpu for node, gpu in self.selected_machines]
        assert sum(gpu_count) == request_num_gpus

        pool_id = job.gpu_type_id
        pool_name = ALIBABA_POOL_NAMES[pool_id]
        all_gpu = [f'{pool_name} {gpu_index[i]}' for i in range(len(gpu_index))]

        for i in range(len(all_gpu)):
            found = False
            for m in self.all_nodes:
                if m.id == all_gpu[i]:
                    allocated_nodes.append((m, gpu_count[i]))
                    found = True
                    break
            if not found:
                return []

        for i in range(len(gpu_index)):
            node_id = gpu_index[i]
            self.cluster_status[pool_id] = [t for t in self.cluster_status[pool_id] if t[0] != node_id]
        result_machines = []
        for i, (m, gpu_cnt) in enumerate(allocated_nodes):
            node_id = gpu_index[i]
            current_state = m.taken_by_job(job_id, gpu_cnt)
            result_machines.append(m)
            if current_state == 0:
                self.used_node[pool_id].append((node_id, current_state))
            else:
                self.cluster_status[pool_id].append((node_id, current_state))
        self.selected_machines = []
        return result_machines

    def release(self, job_release, machine_releases):
        pool_id = job_release.gpu_type_id
        for m in machine_releases:
            gpu_status, _ = m.machine_release(job_release)
            gpu_name, index_str = m.id.split()
            node_idx = int(index_str)
            pid = ALIBABA_POOL_NAMES.index(gpu_name)
            found = False
            for i, (node_id, gpus) in enumerate(self.cluster_status[pid]):
                if node_id == node_idx:
                    self.cluster_status[pid][i] = (node_id, gpu_status)
                    found = True
                    break
            if not found:
                for i, (node_id, gpus) in enumerate(self.used_node[pid]):
                    if node_id == node_idx:
                        self.used_node[pid].pop(i)
                        self.cluster_status[pid].append((node_idx, gpu_status))
                        break

    def dummy_release(self, job_release, machine_releases):
        for m in machine_releases:
            v = m.dummy_machine_release(job_release)
            gpu_name, index_str = m.id.split()
            node_idx = int(index_str)
            pid = ALIBABA_POOL_NAMES.index(gpu_name)
            found = False
            for i, (node_id, gpus) in enumerate(self.dummy_cluster_status[pid]):
                if node_id == node_idx:
                    self.dummy_cluster_status[pid][i] = (node_id, gpus + v)
                    found = True
                    break
            if not found:
                self.dummy_cluster_status[pid].append((node_idx, v))

    def reset(self):
        for pool_id in range(NUM_POOLS):
            gpu_per_node = self.num_gpu_per_node[pool_id]
            n_nodes = self.total_nodes_every_gpu[pool_id]
            self.cluster_status[pool_id] = [(i + 1, gpu_per_node) for i in range(n_nodes)]
            self.used_node[pool_id] = []
        for m in self.all_nodes:
            gpu_name = m.id.split()[0]
            pool_id = ALIBABA_POOL_NAMES.index(gpu_name)
            m.reset(self.num_gpu_per_node[pool_id])


class AlibabaEnvSkip(HPCEnvSkip):
    """Alibaba v2: 5 pools by GPU type; pool_id = gpu_type_id. Rest same as Helios."""

    def _get_pool_id(self, job):
        return job.gpu_type_id

    def _same_pool(self, job, other):
        return job.gpu_type_id == other.gpu_type_id

    def my_init(self, workload_file='', cluster_name='alibaba'):
        print("loading workloads from dataset:", workload_file, "cluster:", cluster_name)
        self.loads = Workload_alibaba(workload_file, cluster_name=cluster_name)
        self.cluster = Cluster_alibaba(
            "Alibaba",
            copy.deepcopy(self.loads.cluster_status),
            list(ALIBABA_GPUS_PER_NODE),
            copy.deepcopy(self.loads.assigned_nodes),
            self.loads.total_nodes_count,
        )
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.batch_job_slice == 0:
            max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1
        else:
            max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1
        if max_index <= 0:
            raise ValueError("Alibaba workload too small for batch_job_slice / JOB_SEQUENCE_SIZE")
        self.loads.reset()

    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        """Backfill: filter by gpu_type_id (same pool) instead of vc_id."""
        assert not self.cluster.can_allocated(job)
        earliest_start_time = self.current_timestamp
        running_jobs_pool = [i for i in self.running_jobs if i.gpu_type_id == job.gpu_type_id]
        running_jobs_pool.sort(key=lambda r: (r.scheduled_time + r.request_time))
        self.cluster.dummy_cluster_status = [[(n, g) for (n, g) in L] for L in self.cluster.cluster_status]
        for running_job in running_jobs_pool:
            self.cluster.dummy_release(running_job, running_job.allocated_machines)
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if self.cluster.dummy_can_allocated(job):
                break
        while not self.cluster.can_allocated(job):
            job_queue_pool = [i for i in self.job_queue if i.gpu_type_id == job.gpu_type_id]
            job_queue_pool.sort(key=lambda _j: self.fcfs_score(_j))
            for _j in job_queue_pool:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        assert _j.scheduled_time == -1
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j, _j.job_id, _j.request_number_of_gpus)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)
                        job_queue_pool = [i for i in self.job_queue if i.gpu_type_id == job.gpu_type_id]
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

    def build_observation(self):
        """Same as base but gpus_per_node is per-pool (cluster.num_gpu_per_node[pool_id])."""
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

        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < MAX_QUEUE_SIZE:
                job = self.visible_jobs[i]
                pool_id = self._get_pool_id(job)
                pool_status = self.cluster.cluster_status[pool_id]
                total_nodes_pool = self.loads.total_nodes_count[pool_id]
                gpus_per_node = self.cluster.num_gpu_per_node[pool_id]
                total_pool_capacity = total_nodes_pool * gpus_per_node

                free_gpu_pool = sum(num_gpus for (node_id, num_gpus) in pool_status)
                num_partial = len([g for (_, g) in pool_status if 0 < g < gpus_per_node])
                cluster_fragmentation_factor = num_partial / max(len(pool_status), 1e-5)

                future_gpu_demand = sum(
                    j.request_number_of_gpus for j in self.visible_jobs if self._same_pool(job, j)
                ) - job.request_number_of_gpus

                submit_time = job.submit_time
                request_gpus = job.request_number_of_gpus
                request_time = job.request_time
                wait_time = self.current_timestamp - submit_time

                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_gpus) / float(self.loads.max_gpu), 1.0 - 1e-5)

                alloc_result = self.cluster.get_allocation_options(job, use_dummy=False)
                can_schedule = alloc_result is not None and len(alloc_result["tier1"]["all"]) > 0
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

    def post_process_score(self, scheduled_logs):
        """Override utilization (job_score_type==3): total GPUs = sum over pools of nodes * gpus_per_node."""
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
            total_cluster_gpus = sum(
                self.loads.total_nodes_count[i] * self.cluster.num_gpu_per_node[i]
                for i in range(NUM_POOLS)
            )
            total_gpu_seconds = max(1, wall_seconds * total_cluster_gpus)
            for i in scheduled_logs:
                scheduled_logs[i] /= total_gpu_seconds
        elif self.job_score_type == 4:
            for i in scheduled_logs:
                scheduled_logs[i] /= scheduled_logs_len
        else:
            raise NotImplementedError

    def schedule(self, job_for_scheduling):
        """Override so MILP uses job's pool and that pool's gpus_per_node."""
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
                pool_id = job_for_scheduling.gpu_type_id
                pool = list(self.cluster.cluster_status[pool_id])
                pool_3d = _pool_to_3d(pool)
                gpus_per_node = self.cluster.num_gpu_per_node[pool_id]
                chosen = choose_allocation_milp_solver(pool_3d, alloc_result, gpus_per_node, top_k)
                if chosen is not None:
                    self.cluster.selected_machines = chosen

        job_for_scheduling.scheduled_time = self.current_timestamp
        self.rl_schedule_time_list.append(job_for_scheduling.scheduled_time)
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling, job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_gpus)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)
        self.score_rl = score
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)

        not_empty = self.moveforward_for_job()
        return not not_empty

    def schedule_curr_sequence_reset(self, score_fn):
        """Override to fix total_cap print for per-pool gpus_per_node."""
        scheduled_logs = {}
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfil:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    if self.running_jobs:
                        self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
                    if not self.cluster.can_allocated(job_for_scheduling):
                        self.job_queue.remove(job_for_scheduling)
                        not_empty = self.moveforward_for_job()
                        if not not_empty:
                            break
                        continue

            assert job_for_scheduling.scheduled_time == -1
            job_for_scheduling.scheduled_time = self.current_timestamp
            self.schedule_time_list.append(job_for_scheduling.scheduled_time)
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling, job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_gpus)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)
            not_empty = self.moveforward_for_job()
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        _total_free = sum(
            sum(g for (_, g) in self.cluster.cluster_status[vc_id])
            for vc_id in range(len(self.cluster.cluster_status))
        )
        _total_cap = sum(
            self.loads.total_nodes_count[vc_id] * self.cluster.num_gpu_per_node[vc_id]
            for vc_id in range(len(self.loads.total_nodes_count))
        )
        _jobs_scheduled = len(scheduled_logs)
        print("[base run] Jobs scheduled in this batch: %d | Cluster: free GPUs = %d / total capacity = %d (used = %d)"
              % (_jobs_scheduled, _total_free, _total_cap, _total_cap - _total_free))
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
        self.makespan = max(self.schedule_time_list) - min(self.schedule_time_list)
        return scheduled_logs
