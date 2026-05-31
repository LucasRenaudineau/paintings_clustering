import hdbscan

clusterer = hdbscan.HDBSCAN(
    metric=my_distance,
    min_cluster_size=10
)

labels = clusterer.fit_predict(X)
