import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from preprocessing import *
from glob import glob
import gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from archetype_visualisation import *
from sklearn.manifold import TSNE 
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as hcluster
paths = [
        path for path in glob("archetypes/archetype_*") if path.split("/")[1][0] in ["a", "b", "c", "d", "e", "f", "g"]
    ]
x = np.array([imread_safe(path) for path in paths])
N = len(x)
for i in range(N):
    print(x[i].shape)

print(x.shape)
x_flattened = x.reshape(N, -1)
pca = PCA(n_components=15)
x_pca = pca.fit_transform(x_flattened)

plt.Figure()
plt.scatter(x_pca[:,0], x_pca[:,1])
plt.savefig(f"archetypes/embedded_archetypes.png", bbox_inches="tight", dpi=300)
plt.show()

print("Clustering...")
# Use of AI
Z = hcluster.linkage(x_pca, method='ward')
best_k = 2
best_score = -1
for k in range(2, N):
    labels_test = hcluster.fcluster(Z, t=k, criterion='maxclust')
    # On calcule la pertinence de ce découpage
    score = silhouette_score(x_pca, labels_test) 
    
    if score > best_score:
        best_score = score
        best_k = k
clusters = hcluster.fcluster(Z, t=best_k, criterion='maxclust')

print(clusters)

plt.Figure()
plt.scatter(x_pca[:,0], x_pca[:,1], c=clusters)
plt.savefig(f"archetypes/clustered_archetypes.png", bbox_inches="tight", dpi=300)
plt.show()


