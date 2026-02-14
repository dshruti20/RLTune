from math import ceil
from statistics import mean
from collections import defaultdict
from itertools import combinations_with_replacement

cluster_structure = {
    "cluster_name": "Cluster_trail",
    "virtual_clusters": [
        {
            "vc_name": "VC1",
            "gpu_types": [
                {"gpu_name": "M40", "nodes": [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]},
                {"gpu_name": "T4", "nodes": [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]},
                {"gpu_name": "P100", "nodes": [(0, 1), (1, 2), (2, 3), (3, 4)]},
                {"gpu_name": "V100", "nodes": [(0, 4), (1, 4), (2, 8), (3, 8)]},
            ],
        }
    ],
}

gpu_capacities = {"M40": 1, "T4": 2, "P100": 4, "V100": 8}


def get_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")


def group_nodes_by_free_gpus(free_nodes):
    """Group nodes by their free GPU count.
    
    Returns:
        dict: {free_gpu_count: [list of node indices]}
    """
    groups = defaultdict(list)
    for node_idx, free_gpus in free_nodes:
        groups[free_gpus].append(node_idx)
    return dict(groups)


def generate_distributions(total_gpus, num_nodes, max_per_node):
    """Generate all unique ways to distribute total_gpus across num_nodes.
    
    Each node can get at most max_per_node GPUs.
    Returns distributions in sorted (descending) order to avoid duplicates.
    """
    if num_nodes == 0:
        return [] if total_gpus == 0 else []
    if num_nodes == 1:
        return [[total_gpus]] if total_gpus <= max_per_node else []
    
    distributions = []
    
    def backtrack(remaining, nodes_left, current, min_value):
        if nodes_left == 0:
            if remaining == 0:
                distributions.append(current[:])
            return
        
        # To avoid duplicates, ensure descending order
        # Next value must be <= min_value and <= remaining
        max_next = min(min_value, remaining, max_per_node)
        
        # Try assigning different amounts to the next node
        for amount in range(max_next, 0, -1):
            current.append(amount)
            backtrack(remaining - amount, nodes_left - 1, current, amount)
            current.pop()
    
    backtrack(total_gpus, num_nodes, [], max_per_node)
    return distributions


def generate_allocation_patterns(groups, request_gpus, gpus_per_node, min_nodes, max_nodes):
    """Generate unique allocation patterns across node groups.
    
    Returns:
        list of patterns, where each pattern is:
        {
            'group_allocations': {free_count: {'nodes_used': int, 'distribution': [gpus_per_node]}},
            'total_nodes': int
        }
    """
    patterns = []
    
    # Get sorted group keys (free GPU counts)
    group_keys = sorted(groups.keys(), reverse=True)
    
    def backtrack_groups(group_idx, remaining_gpus, nodes_used, group_allocations):
        # Base case: all GPUs allocated
        if remaining_gpus == 0:
            if min_nodes <= nodes_used <= max_nodes:
                patterns.append({
                    'group_allocations': dict(group_allocations),
                    'total_nodes': nodes_used
                })
            return
        
        # Base case: no more groups to try
        if group_idx >= len(group_keys):
            return
        
        free_count = group_keys[group_idx]
        available_nodes_in_group = len(groups[free_count])
        
        # Skip this group (don't use any nodes from it)
        backtrack_groups(group_idx + 1, remaining_gpus, nodes_used, group_allocations)
        
        # Try using different numbers of nodes from this group
        max_nodes_from_group = min(
            available_nodes_in_group,
            max_nodes - nodes_used,
            ceil(remaining_gpus / 1)  # At least 1 GPU per node
        )
        
        for num_nodes_from_group in range(1, max_nodes_from_group + 1):
            if nodes_used + num_nodes_from_group > max_nodes:
                break
            
            # Maximum GPUs we can take from this group
            max_gpus_from_group = min(
                num_nodes_from_group * min(free_count, gpus_per_node),
                remaining_gpus
            )
            
            # Generate all distributions for these nodes
            distributions = generate_distributions(
                max_gpus_from_group,
                num_nodes_from_group,
                min(free_count, gpus_per_node)
            )
            
            for distribution in distributions:
                total_in_dist = sum(distribution)
                if total_in_dist == 0:
                    continue
                
                group_allocations[free_count] = {
                    'nodes_used': num_nodes_from_group,
                    'distribution': distribution
                }
                
                backtrack_groups(
                    group_idx + 1,
                    remaining_gpus - total_in_dist,
                    nodes_used + num_nodes_from_group,
                    group_allocations
                )
                
                del group_allocations[free_count]
    
    backtrack_groups(0, request_gpus, 0, {})
    return patterns


def pattern_to_assignment(pattern, groups):
    """Convert a pattern to a concrete node assignment.
    
    Picks the lowest node indices from each group.
    """
    assignment = []
    
    for free_count, allocation_info in pattern['group_allocations'].items():
        nodes_from_group = sorted(groups[free_count])[:allocation_info['nodes_used']]
        distribution = allocation_info['distribution']
        
        for node_idx, gpus_assigned in zip(nodes_from_group, distribution):
            assignment.append((node_idx, gpus_assigned))
    
    return sorted(assignment, key=lambda x: x[0])


def find_gpu_assignments(free_nodes, request_gpus, gpus_per_node, allow_relax_min_nodes=True):
    if request_gpus <= 0:
        return {
            "assignments": [],
            "tier1": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "tier2": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "min_nodes_theoretical": 0,
        }

    min_nodes = ceil(request_gpus / gpus_per_node)
    max_nodes = min_nodes
    if allow_relax_min_nodes:
        max_nodes = min(min_nodes + 1, len(free_nodes))

    # Group nodes by free GPU count
    groups = group_nodes_by_free_gpus(free_nodes)
    
    # Store original state
    free_before = {node: free for node, free in free_nodes}

    # Generate unique allocation patterns
    patterns = generate_allocation_patterns(groups, request_gpus, gpus_per_node, min_nodes, max_nodes)

    if not patterns:
        return {
            "assignments": [],
            "tier1": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "tier2": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "min_nodes_theoretical": min_nodes,
        }

    # FIXED: Fragmentation metrics
    def frag_nodes(free_map):
        return sum(1 for f in free_map.values() if 0 < f < gpus_per_node)

    def waste(free_map):
        """FIXED: waste = gpus_per_node - f for partial nodes"""
        w = 0
        for f in free_map.values():
            if 0 < f < gpus_per_node:
                w += (gpus_per_node - f)
        return w

    def score_assignment(assignment):
        free_after = free_before.copy()
        for node_idx, assigned in assignment:
            if node_idx not in free_after:
                return (float("inf"), float("inf")), float("inf")
            free_after[node_idx] = free_after[node_idx] - assigned
            if free_after[node_idx] < 0:
                return (float("inf"), float("inf")), float("inf")

        # Fragmentation score
        frag_delta = frag_nodes(free_after) - frag_nodes(free_before)
        waste_delta = waste(free_after) - waste(free_before)
        frag_score = (frag_delta, waste_delta)

        # FIXED: Load balance score - global variance delta across ALL nodes of this GPU type
        utils_before = []
        utils_after = []
        for node_idx in free_before.keys():
            # Before state
            f_before = free_before[node_idx]
            util_before = 1.0 - (f_before / gpus_per_node)
            utils_before.append(util_before)
            
            # After state
            f_after = free_after[node_idx]
            util_after = 1.0 - (f_after / gpus_per_node)
            utils_after.append(util_after)

        if len(utils_before) <= 1:
            lb_score = 0.0
        else:
            # Calculate variance before and after
            m_before = mean(utils_before)
            variance_before = sum((u - m_before) ** 2 for u in utils_before) / len(utils_before)
            
            m_after = mean(utils_after)
            variance_after = sum((u - m_after) ** 2 for u in utils_after) / len(utils_after)
            
            # lb_score = change in variance (negative = improved balance, positive = worsened balance)
            lb_score = variance_after - variance_before

        return frag_score, lb_score

    # Convert patterns to assignments and score them
    all_assignment_objs = []
    for pattern in patterns:
        assignment = pattern_to_assignment(pattern, groups)
        frag_score, lb_score = score_assignment(assignment)
        
        # Skip invalid assignments
        if frag_score[0] == float("inf"):
            continue
        
        nodes_used = len({node_idx for node_idx, _ in assignment})
        all_assignment_objs.append(
            {
                "assignment": assignment,
                "frag_score": frag_score,
                "lb_score": lb_score,
                "nodes_used": nodes_used,
                "pattern": pattern  # Keep pattern info for debugging
            }
        )

    if not all_assignment_objs:
        return {
            "assignments": [],
            "tier1": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "tier2": {"all": [], "defrag_sorted": [], "load_balance_sorted": []},
            "min_nodes_theoretical": min_nodes,
        }

    # Separate into tiers
    tier1_all = [a for a in all_assignment_objs if a["nodes_used"] == min_nodes]
    # Tier 2: Only include if using min_nodes+1 AND frag_delta <= 0 (reduces or maintains fragmentation)
    tier2_all = [
        a for a in all_assignment_objs 
        if allow_relax_min_nodes 
        and a["nodes_used"] == min_nodes + 1
        and a["frag_score"][0] <= 0  # Only if frag_delta <= 0
    ]

    def sort_for_defrag(assignments_list):
        return sorted(
            assignments_list,
            key=lambda a: (
                a["frag_score"][0],  # frag_delta
                a["frag_score"][1],  # waste_delta
                a["lb_score"],
                a["nodes_used"],
                [n for (n, _) in sorted(a["assignment"], key=lambda x: x[0])],
            ),
        )

    def sort_for_load_balance(assignments_list):
        return sorted(
            assignments_list,
            key=lambda a: (
                a["lb_score"],
                a["frag_score"][0],
                a["frag_score"][1],
                a["nodes_used"],
                [n for (n, _) in sorted(a["assignment"], key=lambda x: x[0])],
            ),
        )

    return {
        "min_nodes_theoretical": min_nodes,
        "assignments": all_assignment_objs,
        "tier1": {
            "all": tier1_all,
            "defrag_sorted": sort_for_defrag(tier1_all),
            "load_balance_sorted": sort_for_load_balance(tier1_all),
        },
        "tier2": {
            "all": tier2_all,
            "defrag_sorted": sort_for_defrag(tier2_all),
            "load_balance_sorted": sort_for_load_balance(tier2_all),
        },
    }


def get_gpu_pool_from_cluster(cluster_structure, vc_name, gpu_name):
    if gpu_name not in gpu_capacities:
        raise ValueError(
            f"No capacity info found for GPU type '{gpu_name}'. Allowed: {sorted(gpu_capacities.keys())}"
        )

    gpus_per_node = gpu_capacities[gpu_name]

    for vc in cluster_structure["virtual_clusters"]:
        if vc["vc_name"] != vc_name:
            continue
        for gpu_type in vc["gpu_types"]:
            if gpu_type["gpu_name"] != gpu_name:
                continue
            free_nodes = list(gpu_type["nodes"])

            for node_idx, free in free_nodes:
                if free < 0 or free > gpus_per_node:
                    raise ValueError(
                        f"Invalid free GPUs for node {node_idx}: {free}. Must be in [0, {gpus_per_node}]."
                    )

            return gpus_per_node, free_nodes

    raise ValueError(f"Could not find VC '{vc_name}' with GPU type '{gpu_name}' in cluster.")


def main():
    print("Current cluster structure:")
    print(cluster_structure)

    print("\n--- Job Request ---")
    vc_name = input("Enter virtual cluster name (e.g., VC1): ").strip()
    gpu_name = input("Enter GPU type (M40, T4, P100, V100): ").strip()
    request_gpus = get_int("Enter number of GPUs requested: ")

    try:
        gpus_per_node, free_nodes = get_gpu_pool_from_cluster(cluster_structure, vc_name, gpu_name)
    except ValueError as e:
        print("Error:", e)
        return

    print(f"\nUsing VC '{vc_name}', GPU type '{gpu_name}'")
    print(f"GPUs per node (capacity): {gpus_per_node}")
    print(f"Current free GPUs per node: {free_nodes}")

    # Show grouping
    groups = group_nodes_by_free_gpus(free_nodes)
    print(f"\nNode groups by free GPU count: {dict(groups)}")

    result = find_gpu_assignments(
        free_nodes=free_nodes,
        request_gpus=request_gpus,
        gpus_per_node=gpus_per_node,
        allow_relax_min_nodes=True,
    )

    print("\nTheoretical minimum nodes needed (capacity-based):", result["min_nodes_theoretical"])

    if not result["assignments"]:
        print("No feasible assignment found with current free GPUs.")
        return

    print(f"\nTotal unique patterns generated: {len(result['assignments'])}")

    def fmt_defrag(a):
        frag_delta, waste_delta = a["frag_score"]
        return (a["assignment"], frag_delta, waste_delta, a["lb_score"])

    def fmt_lb(a):
        frag_delta, waste_delta = a["frag_score"]
        return (a["assignment"], a["lb_score"], frag_delta, waste_delta)

    print("\n=== Tier 1: DE-FRAGMENTATION (best → worst) ===")
    for i, a in enumerate(result["tier1"]["defrag_sorted"][:5]):  # Show top 5
        print(f"{i+1}. {fmt_defrag(a)}")
    if len(result["tier1"]["defrag_sorted"]) > 5:
        print(f"... and {len(result['tier1']['defrag_sorted']) - 5} more")

    print("\n=== Tier 1: LOAD BALANCING (best → worst) ===")
    for i, a in enumerate(result["tier1"]["load_balance_sorted"][:5]):  # Show top 5
        print(f"{i+1}. {fmt_lb(a)}")
    if len(result["tier1"]["load_balance_sorted"]) > 5:
        print(f"... and {len(result['tier1']['load_balance_sorted']) - 5} more")

    print("\n=== Tier 2: DE-FRAGMENTATION (best → worst) ===")
    for i, a in enumerate(result["tier2"]["defrag_sorted"][:5]):  # Show top 5
        print(f"{i+1}. {fmt_defrag(a)}")
    if len(result["tier2"]["defrag_sorted"]) > 5:
        print(f"... and {len(result['tier2']['defrag_sorted']) - 5} more")

    print("\n=== Tier 2: LOAD BALANCING (best → worst) ===")
    for i, a in enumerate(result["tier2"]["load_balance_sorted"][:5]):  # Show top 5
        print(f"{i+1}. {fmt_lb(a)}")
    if len(result["tier2"]["load_balance_sorted"]) > 5:
        print(f"... and {len(result['tier2']['load_balance_sorted']) - 5} more")


if __name__ == "__main__":
    main()
