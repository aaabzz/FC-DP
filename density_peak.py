import numpy as np

def knn_search(distance_matrix, k):

    num = len(distance_matrix)
    dist_copy = distance_matrix.copy()

    for n in range(num):

        dist_copy[n, n] = np.inf

    dist_knn_set = np.sort(dist_copy, axis=1)

    return dist_knn_set[:, :k]


class DensityPeak(object):

    def __init__(self,
                 n_peaks=None,
                 k=10,
                 distance_matrix=None):

        self.n_peaks = n_peaks
        self.k = k
        self.distance_matrix = distance_matrix

    def local_density(self):

        if self.distance_matrix is None or not isinstance(self.distance_matrix, np.ndarray):

            raise ValueError('Please set a correct distance matrix')

        else:

            dist_knn_set = knn_search(self.distance_matrix, self.k)
            dist_knn_set = np.exp(- dist_knn_set / 2)
            rho = np.sum(dist_knn_set, axis=1)

            return rho

    def min_neighbor_and_distance(self):

        sort_rho_idx = np.argsort(-self.rho)
        delta = [np.max(self.distance_matrix)] * self.n
        nneigh = [0] * self.n

        delta[sort_rho_idx[0]] = -1.

        for i in range(self.n):

            for j in range(0, i):

                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]

                if self.distance_matrix[old_i, old_j] < delta[old_i]:

                    delta[old_i] = self.distance_matrix[old_i, old_j]
                    nneigh[old_i] = old_j

        delta[sort_rho_idx[0]] = np.max(delta)

        return np.array(delta, dtype=np.float32), np.array(nneigh, dtype=np.float32)

    def pick_peaks(self):

        gamma = self.rho * self.delta
        peaks = np.argpartition(-gamma, self.n_peaks - 1)[:self.n_peaks]

        return peaks

    def fit(self):

        if self.n_peaks is None:

            raise ValueError('Please set n_peaks')

        self.n = len(self.distance_matrix)

        self.rho = self.local_density()

        self.delta, self.nneigh = self.min_neighbor_and_distance()

        self.peaks = self.pick_peaks()

