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


# Main

def run_hdbscan(
    n: int,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    max_cluster_size: int = 40,
    min_samples: int = 1,
    seed: int = 42,
):
    """
    Cluster n randomly selected images with HDBSCAN and save results to:
        ./outputs/class0/       for images in cluster 0
        ./outputs/class1/       for images in cluster 1
        ...
        ./outputs/class_noise/  for noise points (HDBSCAN label = -1)

    Uses the approximate KNN graph

    Args:
        n                : number of images to sample from the dataset
        n_neighbors      : neighbours per image in the KNN graph;
                           should be ≥ min_cluster_size
        min_cluster_size : smallest cluster HDBSCAN will form
        min_samples      : core-point threshold (1 = least conservative)
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
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_samples=min_samples,          # key noise-reduction lever
        cluster_selection_method="leaf",   # "leaf" or "eom"
    )
    labels = clusterer.fit_predict(D_sparse)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise    = int((labels == -1).sum())
    print(f"→ {n_clusters} cluster(s) found, {n_noise} noise point(s).")

    # Saving into folders
    os.makedirs("./outputs", exist_ok=True)
    for label in set(labels):
        folder = f"./outputs/class{label}" if label >= 0 else "./outputs/class_noise"
        os.makedirs(folder, exist_ok=True)

    for path, label in zip(paths, labels):
        folder = f"./outputs/class{label}" if label >= 0 else "./outputs/class_noise"
        shutil.copy2(path, os.path.join(folder, os.path.basename(path)))

    print("Done. Images saved to ./outputs/class*/")
    return labels, paths


if __name__ == "__main__":
    run_hdbscan(n=500, n_neighbors=15, min_cluster_size=5, min_samples=1)
