from pynndescent import NNDescent
import numpy as np

def my_distance(a, b):
    return ...

index = NNDescent(
    X,
    metric=my_distance,
    n_neighbors=15
)

neighbors, distances = index.neighbor_graph
