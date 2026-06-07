from features_comparator import find_paths
from features_extractor import load_model, extract_all_features
from preprocessing import imread_safe
from pynndescent import NNDescent
import numpy as np
import numba
import random


# Compact descriptors
# Each activation (1, H, W, C) is reduced to its global-average-pool (C,) and
# the 10 layers are concatenated into a single 7808-dim float32 vector:
#   indices 0..3903    full COLOUR image     (conv1_relu .. conv5_block3_out)
#   indices 3904..7807 uniform GRAYSCALE crop (same layers, texture), offset 3904

def extract_gap_vector(features):
    """Reduce each activation to its GAP and concatenate -> (7808,) float32."""
    return np.concatenate(
        [f[0].mean(axis=(0, 1)) for f in features]
    ).astype(np.float32)


# Used AI for _LAYER_SPANS to convert from using 5 to 10 activations in hdbscan

# Distance
# Mirrors features_comparator.distance() but on GAP vectors instead of full spatial activations. pynndescent requires a numba-jit'd metric.

# Half-open [start, end) channel ranges for all 10 activations in the 7808-dim GAP vector, with the matching per-layer weight (must mirror LAYER_WEIGHTS in features_comparator). Indices 0-4 = full COLOUR image; 5-9 = grayscale crop.
_LAYER_SPANS = (
    (0, 64, 1.0), # 0 conv1_relu full image (colour)
    (64, 320, 1.0), # 1 conv2_block3_out full image (colour)
    (320, 832, 0.1), # 2 conv3_block4_out full image (colour)
    (832, 1856, 0.1), # 3 conv4_block6_out full image (colour)
    (1856, 3904, 0.1), # 4 conv5_block3_out full image (colour)
    (3904, 3968, 1.0), # 5 conv1_relu uniform crop (grayscale texture)
    (3968, 4224, 1.0), # 6 conv2_block3_out uniform crop (grayscale texture)
    (4224, 4736, 1.0), # 7 conv3_block4_out uniform crop (grayscale texture)
    (4736, 5760, 0.1), # 8 conv4_block6_out uniform crop (grayscale texture)
    (5760, 7808, 0.1), # 9 conv5_block3_out uniform crop (grayscale texture)
)


@numba.njit
def _cos_block(a, b, start, end):
    """Cosine distance in [0, 1] over the channel range [start, end)."""
    dot = na = nb = 0.0
    for i in range(start, end):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    return (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0


@numba.njit
def my_distance(a, b):
    """Weighted sum of layer-wise cosine distances on the 7808-dim GAP vector."""
    total = 0.0
    for start, end, w in _LAYER_SPANS:
        total += w * _cos_block(a, b, start, end)
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
    features_list = [extract_all_features(imread_safe(p), feature_model) for p in paths]

    # Build compact (n, 7808) feature matrix
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
