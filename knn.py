from features_comparator import find_paths
from features_extractor import load_model, extract_features
from preprocessing import imread_safe
from pynndescent import NNDescent
import numpy as np
import numba
import random


# Compact descriptors
# Each activation (1, H, W, C) is reduced to its global-average-pool (C,) and
# the 5 layers are concatenated into a single 3904-dim float32 vector:
#   [0    : 64  ]  conv1_relu        (w = 1.0)
#   [64   : 320 ]  conv2_block3_out  (w = 0.1)
#   [320  : 832 ]  conv3_block4_out  (w = 0.1)
#   [832  : 1856]  conv4_block6_out  (w = 1.0)
#   [1856 : 3904]  conv5_block3_out  (w = 1.0)

def extract_gap_vector(features):
    """Reduce each activation to its GAP and concatenate -> (3904,) float32."""
    return np.concatenate(
        [f[0].mean(axis=(0, 1)) for f in features]
    ).astype(np.float32)


# Distance
# Mirrors features_comparator.distance() but on GAP vectors instead of full
# spatial activations. pynndescent requires a numba-jit'd metric.

@numba.njit
def my_distance(a, b):
    """Weighted sum of layer-wise cosine distances on the 3904-dim GAP vector."""
    total = 0.0

    # Layer 0: conv1_relu  (w = 1.0, indices 0..63)
    dot = na = nb = 0.0
    for i in range(0, 64):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    total += (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0

    # Layer 1: conv2_block3_out  (w = 0.1, indices 64..319)
    dot = na = nb = 0.0
    for i in range(64, 320):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    total += 0.1 * (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0

    # Layer 2: conv3_block4_out  (w = 0.1, indices 320..831)
    dot = na = nb = 0.0
    for i in range(320, 832):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    total += 0.1 * (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0

    # Layer 3: conv4_block6_out  (w = 1.0, indices 832..1855)
    dot = na = nb = 0.0
    for i in range(832, 1856):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    total += (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0

    # Layer 4: conv5_block3_out  (w = 1.0, indices 1856..3903)
    dot = na = nb = 0.0
    for i in range(1856, 3904):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    total += (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0

    return total


# Main

def run_knn(n: int, n_neighbors: int = 5, seed: int = 42):
    """
    Approximate KNN for n randomly selected images using pynndescent.

    Args:
        n            : number of images to sample from the dataset
        n_neighbors  : number of nearest neighbors to retrieve per image
        seed         : random seed for reproducibility

    Returns:
        neighbors  : (n, n_neighbors) int array   — neighbor indices in paths
        distances  : (n, n_neighbors) float array — corresponding distances
        paths      : list of n selected image paths
    """
    random.seed(seed)
    _, feature_model = load_model()

    all_paths = find_paths(14553)
    paths = random.sample(all_paths, n)

    print(f"Extracting features for {n} images…")
    features_list = [extract_features(imread_safe(p), feature_model) for p in paths]

    # Build compact (n, 3904) feature matrix
    X = np.array([extract_gap_vector(f) for f in features_list])

    print("Building approximate KNN graph (pynndescent)…")
    index = NNDescent(X, metric=my_distance, n_neighbors=n_neighbors)
    neighbors, distances = index.neighbor_graph   # (n, k), (n, k)

    print("\nResults:")
    for i, (nn_idx, nn_dist) in enumerate(zip(neighbors, distances)):
        print(f"\n{paths[i].split('/')[-1]}:")
        for rank, (j, d) in enumerate(zip(nn_idx, nn_dist), 1):
            print(f"  {rank}. {paths[j].split('/')[-1]}  (d = {d:.4f})")

    return neighbors, distances, paths


if __name__ == "__main__":
    run_knn(n=200, n_neighbors=5)
