import numpy as np
from sklearn.metrics import silhouette_score
from fair_cluster import *
from density_peak import DensityPeak
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist, squareform

def load_file(filename, delimiter=','):

    parent = 'datasets'
    data_file = filename + '.txt'
    group_file = filename + '_color.txt'

    data = os.path.join(parent, data_file)
    groups = os.path.join(parent, group_file)

    data = np.loadtxt(data, delimiter=delimiter, dtype=np.float32)
    groups = np.loadtxt(groups, delimiter=delimiter, dtype=int)

    return data, groups

def main(filename, L, n_clusters, k=10, beta=0.2, delimiter=','):

    data, groups = load_file(filename, delimiter=delimiter)

    datanorm = min_max_scaler(data)

    distance_matrix = squareform(pdist(datanorm, metric='euclidean'))

    g_unique = np.unique(groups)

    initial_points = []

    for g in g_unique:

        g_set = np.where(groups == g)[0]
        sub_distance = distance_matrix[g_set, :][:, g_set]
        dp = DensityPeak(n_peaks=L, k=k, distance_matrix=sub_distance)
        dp.fit()

        candidate_points = dp.peaks
        initial, cost = most_disperse_points(sub_distance, candidate_points, n_clusters)
        g_initial = g_set[list(initial)]
        initial_points.append(g_initial)

    matching, cost = brute_force_matching(distance_matrix, initial_points)

    visited = np.ones(len(data), dtype=int) * -1

    total_ratio, group_indices = cal_total_ratio(groups)

    clusters = []

    for c in matching:

        clusters.append(list(c))

    for c in clusters:

        visited[c] = 1

    num_remain = len(visited[visited != 1])

    for i in range(num_remain):

        clusters_balance = []
        clusters_count = []

        for c in clusters:
            c_balance, group_count = cal_cluster_balance(c, total_ratio, group_indices, groups)
            clusters_balance.append(c_balance)
            clusters_count.append(group_count)

        c_indices = get_worse_balance_c(clusters_balance, beta=beta)
        inThreshold, group_priority, balance_flag = select_next_group(c_indices, clusters_count, group_indices,
                                                                      total_ratio, clusters_balance, beta=beta)

        clusters, visited, costs, points = fair_clustering(c_indices, inThreshold, group_priority, balance_flag,
                                                           group_indices, groups, clusters, visited, distance_matrix,
                                                           k=k, size=True)

    label = np.ones(len(data), dtype=int)

    for i in range(len(clusters)):

        c = clusters[i]
        label[c] = i

    sc = silhouette_score(data, label)

    clusters_balance = []
    clusters_count = []

    for c in clusters:
        c_balance, group_count = cal_cluster_balance(c, total_ratio, group_indices, groups)
        clusters_balance.append(c_balance)
        clusters_count.append(group_count)

    balance = min(clusters_balance)
    sse = compute_sse(data, label)

    return label, balance, sc, sse

if __name__ == '__main__':

    label, balance, sc, sse = main('DS577', L=4, n_clusters=2)

    print(balance, sc, sse, sep='\n')



