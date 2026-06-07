from knn import run_knn
from scipy.sparse import csr_matrix
import hdbscan
import numpy as np
import os, shutil


# Sparse distance matrix

def knn_to_sparse_distance_matrix(neighbors: np.ndarray, distances: np.ndarray) -> csr_matrix:
    """
    Convert the (n, k) KNN neighbour graph produced by knn.run_knn() into a
    symmetric sparse CSR distance matrix for HDBSCAN(metric='precomputed').

    The KNN graph is directed (i→j does not imply j→i), so we symmetrize by
    keeping the *minimum* distance observed for each pair.  Structural zeros
    mean *unknown distance* — not zero — which is the correct semantics for a
    sparse precomputed matrix in HDBSCAN.
    """
    n = neighbors.shape[0]
    best: dict[tuple[int, int], float] = {}

    for i, (nn_idx, nn_dist) in enumerate(zip(neighbors, distances)):
        for j, d in zip(nn_idx.tolist(), nn_dist.tolist()):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in best or d < best[key]:
                best[key] = d

    ii, jj, dd = [], [], []
    for (i, j), d in best.items():
        ii += [i, j]
        jj += [j, i]
        dd += [d, d]

    return csr_matrix((dd, (ii, jj)), shape=(n, n), dtype=np.float64)


# Noise reassignment

def assign_noise_to_nearest_cluster(
    labels: np.ndarray, neighbors: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """
    Give every noise point (label -1) the cluster of its nearest already-clustered
    KNN neighbour, so HDBSCAN's noise is folded back into the clustering.

    pynndescent returns neighbours sorted by ascending distance, so the first
    non-noise entry in neighbors[i] is the closest clustered point to i. The pass
    is repeated: once a noise point is labelled it can in turn label its own noise
    neighbours, letting labels propagate along chains. Points whose whole KNN
    neighbourhood stays noise (genuinely disconnected) keep label -1.

    Args:
        labels    : (n,) HDBSCAN labels (-1 = noise), modified on a copy.
        neighbors : (n, k) KNN indices from knn.run_knn(), sorted by distance.
        max_iter  : safety cap on propagation passes.

    Returns:
        (n,) int array of labels with (almost) no -1 remaining.
    """
    labels = labels.copy()
    for _ in range(max_iter):
        noise_idx = np.where(labels == -1)[0]
        if noise_idx.size == 0:
            break
        changed = False
        for i in noise_idx:
            for j in neighbors[i]:
                if j != i and labels[j] != -1:
                    labels[i] = labels[j]
                    changed = True
                    break
        if not changed:
            break
    return labels


# Main

def run_hdbscan(
    n: int,
    n_neighbors: int = 20,
    min_cluster_size: int = 5,
    max_cluster_size: int = 200,
    min_samples: int = 1,
    assign_noise: bool = False,
    seed: int = 67,
):
    """
    Cluster n randomly selected images with HDBSCAN and save results to:
        ./outputs/hdbscan_classes/class0/       for images in cluster 0
        ./outputs/hdbscan_classes/class1/       for images in cluster 1
        ...
        ./outputs/hdbscan_classes/class_noise/  for leftover noise points

    Uses the approximate KNN graph.

    Args:
        n                : number of images to sample from the dataset
        n_neighbors      : neighbours per image in the KNN graph; a denser graph
                           (higher k) yields fewer noise points
        min_cluster_size : smallest cluster HDBSCAN will form
        max_cluster_size : maximum number of images per cluster; oversized clusters
                           are split into sub-clusters (or noise) before reassignment.
                           None = no limit.
        min_samples      : core-point threshold (1 = least conservative)
        assign_noise     : if True, fold HDBSCAN noise back into clusters by
                           assigning each noise point to its nearest clustered
                           neighbour (see assign_noise_to_nearest_cluster)
        seed             : random seed for reproducibility

    Returns:
        labels : (n,) int array of cluster labels (-1 = noise)
        paths  : list of n selected image paths
    """
    # Reuse knn.py
    neighbors, distances, paths = run_knn(n, n_neighbors=n_neighbors, seed=seed)

    print("Building sparse distance matrix from KNN graph…")
    D_sparse = knn_to_sparse_distance_matrix(neighbors, distances)

    # Clusters
    print("Running HDBSCAN…")
    hdbscan_kwargs = dict(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,            # key noise-reduction lever
        cluster_selection_method="eom",     # "eom" absorbs more points than "leaf"
    )
    if max_cluster_size is not None:
        hdbscan_kwargs["max_cluster_size"] = max_cluster_size
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    labels = clusterer.fit_predict(D_sparse)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise    = int((labels == -1).sum())
    print(f"→ {n_clusters} cluster(s) found, {n_noise} noise point(s) before reassignment.")

    if assign_noise and n_noise > 0:
        labels = assign_noise_to_nearest_cluster(labels, neighbors)
        n_noise = int((labels == -1).sum())
        print(f"→ {n_noise} noise point(s) remaining after nearest-cluster reassignment.")

    # Saving into folders
    os.makedirs("./outputs/hdbscan_classes", exist_ok=True)
    for label in set(labels):
        folder = f"./outputs/hdbscan_classes/class{label}" if label >= 0 else "./outputs/hdbscan_classes/class_noise"
        os.makedirs(folder, exist_ok=True)

    for path, label in zip(paths, labels):
        folder = f"./outputs/hdbscan_classes/class{label}" if label >= 0 else "./outputs/hdbscan_classes/class_noise"
        shutil.copy2(path, os.path.join(folder, os.path.basename(path)))

    print("Done. Images saved to ./outputs/hdbscan_classes/class*/")
    return labels, paths


if __name__ == "__main__":
    run_hdbscan(n=500, n_neighbors=10, min_cluster_size=5, max_cluster_size=50, min_samples=1)
