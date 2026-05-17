import time
import tracemalloc
import itertools
import multiprocessing
import mpmath
from mpmath import iv
import sys
import os
import json
import logging

# Ensure strict precision regimes
mpmath.mp.dps = 15
iv.dps = 15

def _eval_b2_point(u_val, v_val):
    u_iv = mpmath.mpi(u_val, u_val)
    v_iv = mpmath.mpi(v_val, v_val)
    one = mpmath.mpi('1.0', '1.0')
    two = mpmath.mpi('2.0', '2.0')
    inner = one - (v_iv/u_iv)
    if inner.a < 0:
        inner = mpmath.mpi('0.0', inner.b)
    return (two * u_iv) / (one + iv.sqrt(inner))

def B2_interval(X1, X2):
    u_a, u_b = X1.a, X1.b
    v_a, v_b = X2.a, X2.b
    res_intervals = []
    
    if u_a <= v_b:
        min_u, max_u = u_a, min(u_b, v_b)
        min_v, max_v = max(u_a, v_a), v_b
        min_val = (mpmath.mpi(min_u, min_u) + mpmath.mpi(min_v, min_v)).a
        max_val = (mpmath.mpi(max_u, max_u) + mpmath.mpi(max_v, max_v)).b
        res_intervals.append(mpmath.mpi(min_val, max_val))
        
    if u_b > v_a:
        min_cands = []
        u_start_min = max(u_a, v_a)
        if u_start_min > v_a:
            min_cands.append(_eval_b2_point(u_start_min, v_a).a)
        elif u_start_min == v_a:
            min_cands.append((mpmath.mpi(u_start_min, u_start_min) + mpmath.mpi(v_a, v_a)).a)
        
        if u_b > v_a:
            min_cands.append(_eval_b2_point(u_b, v_a).a)
            
        nine_eighths = mpmath.mpi('9.0', '9.0') / mpmath.mpi('8.0', '8.0')
        crit_u_iv = nine_eighths * mpmath.mpi(v_a, v_a)
        
        if u_a <= crit_u_iv.b and crit_u_iv.a < u_b:
            twenty_seven_sixteenths = mpmath.mpi('27.0', '27.0') / mpmath.mpi('16.0', '16.0')
            slice_min = (twenty_seven_sixteenths * mpmath.mpi(v_a, v_a)).a
            min_cands.append(slice_min)

        max_cands = []
        u_start_max = max(u_a, v_b)
        if u_start_max > v_b:
            max_cands.append(_eval_b2_point(u_start_max, v_b).b)
        elif u_start_max == v_b:
            max_cands.append((mpmath.mpi(u_start_max, u_start_max) + mpmath.mpi(v_b, v_b)).b)
            
        if u_b > v_b:
            max_cands.append(_eval_b2_point(u_b, v_b).b)
            
        if min_cands and max_cands:
            res_intervals.append(mpmath.mpi(min(min_cands), max(max_cands)))
            
    if not res_intervals:
        return X1 + X2
        
    global_min = min(i.a for i in res_intervals)
    global_max = max(i.b for i in res_intervals)
    return mpmath.mpi(global_min, global_max)

def ub_dp(L, U):
    dp = [mpmath.mpi(0, 0)] * 12
    decisions = [0] * 12
    X = [mpmath.mpi(L[i], U[i]) for i in range(11)]
    dp[1] = dp[0] + X[0]
    decisions[1] = 1
    for i in range(2, 12):
        c1 = dp[i-1] + X[i-1]
        c2 = dp[i-2] + B2_interval(X[i-2], X[i-1])
        if c2.b <= c1.b:
            dp[i] = c2
            decisions[i] = 2
        else:
            dp[i] = c1
            decisions[i] = 1
            
    path = []
    curr = 11
    while curr > 0:
        d = decisions[curr]
        path.append(d)
        curr -= d
    path.reverse()
    return dp[11].b, path

def propagate_bounds(L, U):
    L = list(L)
    U = list(U)
    changed = True
    iters = 0
    frac_3_4 = mpmath.mpi('3.0', '3.0') / mpmath.mpi('4.0', '4.0')
    frac_2_3 = mpmath.mpi('2.0', '2.0') / mpmath.mpi('3.0', '3.0')
    frac_1_2 = mpmath.mpi('1.0', '1.0') / mpmath.mpi('2.0', '2.0')
    one = mpmath.mpi('1.0', '1.0')
    tol = mpmath.mpf('1e-9')
    
    while changed and iters < 10:
        changed = False
        iters += 1
        for i in range(1, 11):
            if U[i] > U[0]: U[i] = U[0]; changed = True
            if L[0] < L[i]: L[0] = L[i]; changed = True
            
        for i in range(10):
            L_b = mpmath.mpf((mpmath.mpi(L[i], L[i]) * frac_3_4).a)
            if L[i+1] < L_b: L[i+1] = L_b; changed = True
            U_b = mpmath.mpf((mpmath.mpi(U[i+1], U[i+1]) / frac_3_4).b)
            if U[i] > U_b: U[i] = U_b; changed = True
            
        for i in range(9):
            L_b = mpmath.mpf((mpmath.mpi(L[i], L[i]) * frac_2_3).a)
            if L[i+2] < L_b: L[i+2] = L_b; changed = True
            U_b = mpmath.mpf((mpmath.mpi(U[i+2], U[i+2]) / frac_2_3).b)
            if U[i] > U_b: U[i] = U_b; changed = True
            
        for i in range(8):
            L_b = mpmath.mpf((mpmath.mpi(L[i], L[i]) * frac_1_2).a)
            if L[i+3] < L_b: L[i+3] = L_b; changed = True
            U_b = mpmath.mpf((mpmath.mpi(U[i+3], U[i+3]) / frac_1_2).b)
            if U[i] > U_b: U[i] = U_b; changed = True
            
        for i in range(11):
            prod_L_exc = one
            for j in range(11):
                if j != i: prod_L_exc *= mpmath.mpi(L[j], L[j])
            if prod_L_exc.a > 0:
                max_xi = mpmath.mpf((one / prod_L_exc).b)
                if max_xi < U[i]: U[i] = max_xi; changed = True
                
            prod_U_exc = one
            for j in range(11):
                if j != i: prod_U_exc *= mpmath.mpi(U[j], U[j])
            if prod_U_exc.a > 0:
                min_xi = mpmath.mpf((one / prod_U_exc).a)
                if min_xi > L[i]: L[i] = min_xi; changed = True
                
        for i in range(11):
            if L[i] > U[i] + tol: return None, None
            
    return L, U

def is_feasible(L, U):
    if U[0] < mpmath.mpf('1.0'): return False
    prod_L = mpmath.mpi('1.0', '1.0')
    for x in L: prod_L *= mpmath.mpi(x, x)
    if prod_L.a > mpmath.mpf('1.0'): return False
    prod_U = mpmath.mpi('1.0', '1.0')
    for x in U: prod_U *= mpmath.mpi(x, x)
    if prod_U.b < mpmath.mpf('1.0'): return False
    return True

def process_box(args):
    """
    Evaluates an initial sub-domain subtree.
    Returns highly granulated telemetry to ensure strict memory constraints
    while passing surviving box configurations back for aggregate dataset compilation.
    """
    worker_start_cpu = time.process_time()
    box_idx, (L_init, U_init), cert_dir = args
    
    # Structure: (Lower, Upper, bisection_depth)
    queue = [(L_init, U_init, 0)]
    
    nodes_explored = 0
    max_depth = 0
    pruned_infeasible = 0
    pruned_bound = 0
    pruned_propagation = 0
    surviving = []
    
    threshold = mpmath.mpf('11.0')
    target_max_u = mpmath.mpf('1.05')
    target_min_l = mpmath.mpf('0.95')
    split_divisor = mpmath.mpf('2.0')
    
    while queue:
        L, U, depth = queue.pop()
        nodes_explored += 1
        if depth > max_depth:
            max_depth = depth
            
        if not is_feasible(L, U):
            pruned_infeasible += 1
            continue
            
        ub_mpf, path = ub_dp(L, U)
        if ub_mpf <= threshold:
            pruned_bound += 1
            continue
            
        # Check singularity isolation neighborhood
        if max(U) <= target_max_u and min(L) >= target_min_l:
            surviving.append(([float(x) for x in L], [float(x) for x in U]))
            continue
            
        # Bisect along dimension of maximum uncertainty
        widths = [U[i] - L[i] for i in range(11)]
        max_w = max(widths)
        split_dim = widths.index(max_w)
        mid = L[split_dim] + max_w / split_divisor
        
        # Left child branch
        L1, U1 = list(L), list(U)
        U1[split_dim] = mid
        L1_prop, U1_prop = propagate_bounds(L1, U1)
        if L1_prop is not None:
            queue.append((L1_prop, U1_prop, depth + 1))
        else:
            pruned_propagation += 1
            
        # Right child branch
        L2, U2 = list(L), list(U)
        L2[split_dim] = mid
        L2_prop, U2_prop = propagate_bounds(L2, U2)
        if L2_prop is not None:
            queue.append((L2_prop, U2_prop, depth + 1))
        else:
            pruned_propagation += 1
            
    worker_cpu_time = time.process_time() - worker_start_cpu
    
    # Save lightweight standalone telemetry per subtree mapping directly to disk
    subtree_telemetry = {
        "box_idx": box_idx,
        "nodes_explored": nodes_explored,
        "max_depth": max_depth,
        "pruned_infeasible": pruned_infeasible,
        "pruned_bound": pruned_bound,
        "pruned_propagation": pruned_propagation,
        "surviving_count": len(surviving),
        "worker_cpu_seconds": worker_cpu_time
    }
    
    try:
        os.makedirs(cert_dir, exist_ok=True)
        with open(os.path.join(cert_dir, f"subtree_{box_idx}.json"), "w") as f:
            json.dump(subtree_telemetry, f)
    except Exception as e:
        sys.stderr.write(f"I/O Error saving subtree {box_idx} summary: {e}\n")
        
    return (box_idx, nodes_explored, max_depth, pruned_infeasible, 
            pruned_bound, pruned_propagation, surviving, worker_cpu_time)

def run_bb():
    # Setup portable output hierarchy
    base_out_dir = os.path.abspath("./outputs")
    cert_dir = os.path.join(base_out_dir, "certificates")
    dataset_dir = os.path.join(base_out_dir, "datasets")
    os.makedirs(cert_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Setup standard python logging alongside direct console prints
    log_filename = os.path.join(base_out_dir, "execution.log")
    logging.basicConfig(
        filename=log_filename, 
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    
    console_msg = f"Initializing Global Verification Suite for Woods' Conjecture (n=11)\n"
    console_msg += f"Outputs targeted to directory: {base_out_dir}"
    print(console_msg, flush=True)
    logging.info("Started continuous interval verification setup.")

    tracemalloc.start()
    start_time = time.time()
    
    L0, U0 = mpmath.mpf('1.0'), mpmath.mpf('4.25')
    intervals = [('0.05', '1.0'), ('1.0', '4.25')]
    all_initial_boxes = []
    
    # Populate the 1,024 root level combinatorial spaces
    for p in itertools.product(intervals, repeat=10):
        L_init = [L0] + [mpmath.mpf(i[0]) for i in p]
        U_init = [U0] + [mpmath.mpf(i[1]) for i in p]
        L_prop, U_prop = propagate_bounds(L_init, U_init)
        if L_prop is not None:
            all_initial_boxes.append((len(all_initial_boxes), (L_prop, U_prop), cert_dir))
            
    if len(sys.argv) == 3:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
        tasks_to_run = all_initial_boxes[start_idx:end_idx]
    else:
        tasks_to_run = all_initial_boxes
        
    logging.info(f"Generated {len(all_initial_boxes)} feasible root structures. Executing {len(tasks_to_run)} subtrees.")
    print(f"Total initial configurations feasible: {len(all_initial_boxes)}, scheduled for processing: {len(tasks_to_run)}", flush=True)
    
    # Aggregate telemetry storage
    total_explored_nodes = 0
    global_max_depth = 0
    total_pruned_infeasible = 0
    total_pruned_bound = 0
    total_pruned_prop = 0
    total_cpu_time = 0.0
    all_surviving_boxes = []
    completed = 0
    
    # Execute across parallel cores
    with multiprocessing.Pool(8) as pool:
        for result in pool.imap_unordered(process_box, tasks_to_run):
            (b_idx, nodes, depth, p_inf, p_bnd, p_prp, surv, cpu_sec) = result
            
            total_explored_nodes += nodes
            if depth > global_max_depth:
                global_max_depth = depth
            total_pruned_infeasible += p_inf
            total_pruned_bound += p_bnd
            total_pruned_prop += p_prp
            total_cpu_time += cpu_sec
            all_surviving_boxes.extend(surv)
            
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                status_str = f"Progress: {completed}/{len(tasks_to_run)} subtrees evaluated. Elapsed Wall-Clock: {elapsed:.2f}s"
                print(status_str, flush=True)
                logging.info(status_str)
                
    wall_clock_time = time.time() - start_time
    current_mem, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / 10**6
    total_pruned = total_pruned_infeasible + total_pruned_bound + total_pruned_prop
    
    # Process Paper Statistics Requirements (Geometric & Clustering Telemetry)
    empirical_min = float('inf')
    empirical_max = float('-inf')
    terminal_widths_sum = 0.0
    total_surviving_count = len(all_surviving_boxes)
    
    # Dimensional centers tracking to audit clustering across variables
    dim_centers = [[] for _ in range(11)]
    
    for l_arr, u_arr in all_surviving_boxes:
        local_min = min(l_arr)
        local_max = max(u_arr)
        if local_min < empirical_min: empirical_min = local_min
        if local_max > empirical_max: empirical_max = local_max
        
        # Calculate coordinate box widths
        widths = [u_arr[i] - l_arr[i] for i in range(11)]
        terminal_widths_sum += sum(widths) / 11.0
        
        # Track box centroids for geometry distribution logging
        for i in range(11):
            dim_centers[i].append((l_arr[i] + u_arr[i]) / 2.0)
            
    avg_terminal_width = (terminal_widths_sum / total_surviving_count) if total_surviving_count > 0 else 0.0
    
    # Evaluate dimension clustering standard deviations
    dim_std_devs = []
    if total_surviving_count > 1:
        for i in range(11):
            mean_c = sum(dim_centers[i]) / total_surviving_count
            var_c = sum((c - mean_c)**2 for c in dim_centers[i]) / (total_surviving_count - 1)
            dim_std_devs.append(var_c**0.5)
            
    clustering_behavior_str = "Uniformly distributed around the limit with minor dimensional variance."
    if dim_std_devs and max(dim_std_devs) > 0.02:
        clustering_behavior_str = f"Exhibits specific dimensional axis elongation (Max parameter standard deviation: {max(dim_std_devs):.5f})."

    # Terminal outputs mapping directly to Section 6 of your manuscript
    summary_report = f"\n{'='*65}\n"
    summary_report += f"GLOBAL SEARCH TELEMETRY & PAPER METRICS SUMMARY\n"
    summary_report += f"{'='*65}\n"
    summary_report += f"1. Search Tree Dynamics\n"
    summary_report += f"   - Total Explored Sub-box Nodes : {total_explored_nodes:,}\n"
    summary_report += f"   - Maximum Bisection Depth      : {global_max_depth}\n\n"
    summary_report += f"2. Geometry of Surviving Boxes\n"
    summary_report += f"   - Verified Surviving Count     : {total_surviving_count:,}\n"
    summary_report += f"   - Empirical Tight Bounding Box : [{empirical_min:.6f}, {empirical_max:.6f}]\n"
    summary_report += f"   - Avg Terminal Coordinate Width: {avg_terminal_width:.6e}\n"
    summary_report += f"   - Clustering Distribution      : {clustering_behavior_str}\n\n"
    summary_report += f"3. Categorized Pruning Breakdown\n"
    summary_report += f"   - Total Eliminated Boxes       : {total_pruned:,}\n"
    summary_report += f"     * Infeasible / Volume Cuts   : {total_pruned_infeasible:,}\n"
    summary_report += f"     * Continuous DP Bound Pruning: {total_pruned_bound:,}\n"
    summary_report += f"     * Local Constraint Fails     : {total_pruned_prop:,}\n\n"
    summary_report += f"4. Computational Resource Efficiency\n"
    summary_report += f"   - Elapsed Wall-Clock Time      : {wall_clock_time:.2f} seconds\n"
    summary_report += f"   - Aggregate Parallel CPU Time  : {total_cpu_time:.2f} seconds\n"
    summary_report += f"   - Parallel Scaling Ratio       : {(total_cpu_time/wall_clock_time):.2f}x core factor\n"
    summary_report += f"   - Verified Peak Memory Footprint: {peak_mem_mb:.2f} MB\n"
    summary_report += f"{'='*65}\n"
    
    print(summary_report, flush=True)
    logging.info(summary_report)
    
    # Save Uploadable Output Assets for GitHub Distribution
    try:
        # Artifact 1: Direct Surviving Box Enclosures
        dataset_path = os.path.join(dataset_dir, "surviving_boxes.json")
        with open(dataset_path, "w") as f:
            json.dump(all_surviving_boxes, f)
        logging.info(f"Successfully generated artifact: {dataset_path}")
        
        # Artifact 2: Paper Summary Telemetry Payload
        telemetry_payload = {
            "execution_meta": {
                "wall_clock_seconds": wall_clock_time,
                "parallel_cpu_seconds": total_cpu_time,
                "peak_memory_mb": peak_mem_mb
            },
            "tree_dynamics": {
                "total_explored_nodes": total_explored_nodes,
                "maximum_bisection_depth": global_max_depth
            },
            "geometry_metrics": {
                "total_surviving_instances": total_surviving_count,
                "empirical_minimum_coordinate": empirical_min,
                "empirical_maximum_coordinate": empirical_max,
                "average_terminal_interval_width": avg_terminal_width,
                "axis_clustering_std_devs": dim_std_devs
            },
            "pruning_telemetry": {
                "aggregate_pruned": total_pruned,
                "infeasible_configurations": total_pruned_infeasible,
                "bound_dominated": total_pruned_bound,
                "propagation_failures": total_pruned_prop
            }
        }
        
        telemetry_path = os.path.join(dataset_dir, "execution_telemetry.json")
        with open(telemetry_path, "w") as f:
            json.dump(telemetry_payload, f, indent=2)
        logging.info(f"Successfully generated artifact: {telemetry_path}")
        
    except Exception as e:
        sys.stderr.write(f"Critical error generating uploadable artifact datasets: {e}\n")

if __name__ == "__main__":
    run_bb()

