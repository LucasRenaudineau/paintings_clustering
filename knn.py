from features_comparator import find_paths
from features_extractor import load_model, extract_all_features
from preprocessing import imread_safe
from pynndescent import NNDescent
import numpy as np
import numba
import random

def extract_gap_vector(features): # Cossine dissimilarity measure between raw activations was highly too time-consuming so we used gap
    """Reduce each activation to its GAP and concatenate"""
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
    (832, 1856, 0.2), # 3 conv4_block6_out full image (colour)
    (1856, 3904, 0.8), # 4 conv5_block3_out full image (colour)
    (3904, 3968, 0.2), # 5 conv1_relu uniform crop (grayscale texture)
    (3968, 4224, 0.2), # 6 conv2_block3_out uniform crop (grayscale texture)
    (4224, 4736, 0.1), # 7 conv3_block4_out uniform crop (grayscale texture)
    (4736, 5760, 0.1), # 8 conv4_block6_out uniform crop (grayscale texture)
    (5760, 7808, 0.2), # 9 conv5_block3_out uniform crop (grayscale texture)
)


@numba.njit
def _cos_block(a, b, start, end):
    """Cosine distance in [0, 1] over the channel range from start to end)."""
    dot = na = nb = 0.0
    for i in range(start, end):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    return (1.0 - dot / (na ** 0.5 * nb ** 0.5)) / 2.0


@numba.njit
def my_distance(a, b):
    """Weighted sum of layer-wise cosine distances on the GAP vector."""
    total = 0.0
    for start, end, w in _LAYER_SPANS:
        total += w * _cos_block(a, b, start, end)
    return total


# Main

def run_knn(n: int, n_neighbors: int = 5, seed: int = 42):
    """
    Returns:
        neighbors : (n, n_neighbors) int array, neighbor indices in paths
        distances : (n, n_neighbors) float array, corresponding distances
        paths : list of n selected image paths
    """
    random.seed(seed)
    _, feature_model = load_model()

    all_paths = find_paths(14553)
    paths = random.sample(all_paths, n)

    counter = 0
    print(f"Extracting features for {n} images...")
    X = np.empty((n, 7808), dtype=np.float32)
    for i, p in enumerate(paths):
        features = extract_all_features(imread_safe(p), feature_model)
        X[i] = extract_gap_vector(features)
        if (counter%100==0):
            print(f"Extracting : {counter} / {n}")
        counter +=1

    print("Building approximate KNN graph (pynndescent)...")
    index = NNDescent(X, metric=my_distance, n_neighbors=n_neighbors)
    neighbors, distances = index.neighbor_graph # (n, k), (n, k)

    print("\nResults:")
    for i, (nn_idx, nn_dist) in enumerate(zip(neighbors, distances)):
        print(f"\n{paths[i].split('/')[-1]}:")
        for rank, (j, d) in enumerate(zip(nn_idx, nn_dist), 1):
            print(f"  {rank}. {paths[j].split('/')[-1]}  (d = {d:.4f})")

    return neighbors, distances, paths


if __name__ == "__main__":
    run_knn(n=200, n_neighbors=5)
