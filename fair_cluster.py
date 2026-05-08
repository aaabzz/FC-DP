import numpy as np
import itertools

def min_max_scaler(data):

    # data: n*d matrix,type -> numpy.ndarray or list

    N = len(data)
    min_component = np.min(data, axis=0)  # 1*d
    max_component = np.max(data, axis=0)  # 1*d
    max_min = max_component - min_component
    min_component = np.tile(min_component, (N, 1))
    max_min = np.tile(max_min, (N, 1))
    datanorm = (data - min_component) / max_min

    return datanorm

def cal_total_ratio(groups):

    group_indices, counts = np.unique(groups, return_counts=True)
    total_ratio = counts / len(groups)

    return total_ratio, group_indices

def cal_cluster_balance(c, total_ratio, group_indices, groups):

    group_c = groups[c]

    group_count = np.array([np.sum(group_c == g) for g in group_indices])

    c_ratio = group_count / len(c)

    balance_per_group = np.minimum(total_ratio / c_ratio, c_ratio / total_ratio)
    c_balance = np.min(balance_per_group)

    return c_balance, group_count

def get_worse_balance_c(clusters_balance, beta=0.05):

    clusters_balance = np.array(clusters_balance)
    threshold = 1 - beta
    c_indices = np.where(clusters_balance < threshold)[0]

    if c_indices.size == 0:

        return np.arange(len(clusters_balance), dtype=int)

    else:

        return c_indices

def select_next_group(c_indices, clusters_count, group_indices, total_ratio, clusters_balance, beta=0.05):


    threshold = 1 - beta
    num_clusters = len(c_indices)
    num_groups = len(group_indices)

    c_count_matrix = np.array([clusters_count[i] for i in c_indices], dtype=float)

    temp_counts = np.repeat(c_count_matrix[:, :, None], num_groups, axis=2)
    temp_counts[np.arange(num_clusters)[:, None], np.arange(num_groups), np.arange(num_groups)] += 1

    temp_sums = np.sum(temp_counts, axis=1, keepdims=True)
    c_ratios = temp_counts / temp_sums

    total_ratio_arr = np.array(total_ratio, dtype=float)
    total_ratio_exp = total_ratio_arr[None, :, None]
    balance_per_group = np.minimum(total_ratio_exp / c_ratios, c_ratios / total_ratio_exp)
    next_balance = np.min(balance_per_group, axis=1)

    cluster_balance_selected = np.array([clusters_balance[i] for i in c_indices])
    cluster_balance_selected = cluster_balance_selected[:, None]

    balance_flag = np.where(next_balance >= cluster_balance_selected, 1, -1)

    inThreshold = [group_indices[next_balance[i] >= threshold].tolist() for i in range(num_clusters)]

    group_priority = np.argsort(-next_balance, axis=1)

    return inThreshold, group_priority, balance_flag

def cluster_knn_neighborhood(distmat, c, next_points, k, groups):

    c = np.array(c)
    s = np.array(next_points)

    dist_sub = distmat[np.ix_(c, s)].copy()
    groups_equal = groups[c][:, None] == groups[s][None, :]
    dist_sub[~groups_equal] =  np.inf
    k_c = min(k, len(s))
    knn_local = np.argpartition(dist_sub, k_c-1, axis=1)[:, :k_c]
    knn_idx = s[knn_local]
    knn_dist = dist_sub[np.arange(len(c))[:, None], knn_local]

    valid_mask = knn_dist != np.inf
    valid_indices = knn_idx[valid_mask]

    neighbors = np.unique(valid_indices)

    dist_matrix =  distmat[np.ix_(c, neighbors)].copy()

    groups_equal = groups[c][:, None] == groups[neighbors][None, :]
    dist_matrix[~groups_equal] = np.inf
    min_dists = np.min(dist_matrix, axis=0)

    return neighbors, min_dists

def fair_clustering(c_indices, inThreshold, group_priority, balance_flag, group_indices, groups, clusters, visited, distmat, k=7, size=False):


    num_clusters = len(clusters)
    other_points = np.where(visited != 1)[0]
    next_c = [clusters[i] for i in c_indices]

    total_count = np.array([np.sum(groups == g) for g in group_indices])
    visited_count = np.array([np.sum((groups == g) & (visited == 1)) for g in group_indices])
    all_visited = (visited_count >= total_count)

    next_c_points = []
    next_c_flag = [1]

    for i in range(len(group_priority)):

        c_inThreshold = inThreshold[i]
        priority = group_priority[i]
        flag = balance_flag[i]

        arrays = [other_points[groups[other_points] == j] for j in c_inThreshold]

        if len(arrays) == 0:

            temp_points = np.array([], dtype=int)

        else:

            temp_points = np.concatenate(arrays)

        if len(temp_points) > 0:

            next_c_points.append(temp_points)

        else:

            for j in priority:

                temp = other_points[groups[other_points] == group_indices[j]]

                if len(temp) > 0:

                    next_c_points.append(temp)
                    next_c_flag.append(flag[j])
                    break

    if np.any(all_visited) and np.any(next_c_flag != 1):

        mask = next_c_flag[1:] == 1

        if np.any(mask):

            c_indices = [idx for idx, m in zip(c_indices, mask) if m]
            next_c = [clusters[i] for i in c_indices]
            next_c_points = [p for p, m in zip(next_c_points, mask) if m]

        else:

            c_indices = np.arange(num_clusters, dtype=int)
            next_c = clusters
            next_c_points = [other_points.copy() for _ in range(num_clusters)]

    costs = []
    points = []

    cluster_flat = np.concatenate([np.array(cl) for cl in clusters])
    visited_mask = np.zeros(len(cluster_flat), dtype=bool)

    for cl in clusters:

        visited_mask[np.isin(cluster_flat, cl)] = False

    for i, c in enumerate(next_c):

        next_points = next_c_points[i]

        points_c, dists_c = cluster_knn_neighborhood(distmat, c, next_points, k, groups)
        points_c = np.array(points_c)
        dists_c = np.array(dists_c)

        other_clusters = np.concatenate([np.array(cl) for idx, cl in enumerate(clusters) if idx != c_indices[i]])
        other_groups = groups[other_clusters]

        dist_matrix = distmat[np.ix_(points_c, other_clusters)]
        mask_same_group = (groups[points_c][:, None] == other_groups[None, :])
        dist_matrix[~mask_same_group] = np.inf
        dist_other = np.min(dist_matrix, axis=1)

        val = np.full_like(dists_c, 100, dtype=float)
        np.divide(dist_other, dists_c, out=val, where=dists_c != 0)

        if size:

            val = val / len(c)

        costs.append(val)
        points.append(points_c)

    max_vals = np.array([np.max(cost) for cost in costs])
    max_idx = np.array([np.argmax(cost) for cost in costs])

    idc = np.argmax(max_vals)
    id_point = max_idx[idc]

    visited[points[idc][id_point]] = 1
    next_c[idc].append(points[idc][id_point])

    return clusters, visited, costs, points

def brute_force_matching(dist_matrix, initial):

    n = len(initial[0])
    k = len(initial)

    best_cost = float('inf')
    best_match = None

    base = initial[0]

    perms = [list(itertools.permutations(m)) for m in initial[1:]]

    for perm_comb in itertools.product(*perms):

        current_cost = 0
        match = []

        for i in range(n):

            tuple_points = [base[i]]

            for p in perm_comb:
                tuple_points.append(p[i])

            for u in range(len(tuple_points)):
                for v in range(u+1,len(tuple_points)):
                    current_cost += dist_matrix[tuple_points[u], tuple_points[v]]

            match.append(tuple(tuple_points))

        if current_cost < best_cost:
            best_cost = current_cost
            best_match = match

    return best_match, best_cost

def most_disperse_points(dist_matrix, candidates, k):

    best_points = None
    best_score = -float('inf')

    for comb in itertools.combinations(candidates, k):

        score = 0

        for i in range(k):
            for j in range(i+1, k):
                score += dist_matrix[comb[i], comb[j]]

        if score > best_score:
            best_score = score
            best_points = comb

    return best_points, best_score

def compute_sse(X, labels):

    sse = 0.0

    unique_labels = np.unique(labels)

    for k in unique_labels:

        cluster_points = X[labels == k]
        centroid = cluster_points.mean(axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)

    return sse

