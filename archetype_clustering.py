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
from sklearn.metrics import silhouette_score, pairwise_distances
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
print(f"x_pca shape : {x_pca.shape}")

class DeepClusterer:
    def __init__(self, model, data, dist=np.linalg.norm, epochs=100):
        self.x = data
        self.N = len(self.data)
        self.model = model
        self.dist = dist
        self.labels = np.array([np.arange(self.N)]) # Initialization Everything in the same cluster
        self.lambd = 0.5
        self.K = len(self.labels)
    
    def D(self, k1, k2):
        for i in range(len(np.where(self.labels==k1)[0])):
            for j in range(len(np.where(self.labels==k2)[0])):
                ind_i = np.where(self.labels==k1)[0][i]
                ind_j = np.where(self.labels==k2)[0][j]
                SP += self.dist(self.x[ind_i], self.x[ind_j])

    def compactness(self):
        return np.sum([self.D(k, k, self.dist, self.labels) for k in range(self.K)])
    def separation(self):
        return np.sum([[self.D(k1, k2, self.dist, self.labels) for k1 in range(self.K) if k1!=k2]for k2 in range(K)])
    def loss(self):
        return self.compactness(self.K)-(self.lambd/self.K) * self.separation(self.K)
    
    def get_split_threshold(self):
        def JS_div(k1, k2):
            return
        return self.lambd / (2*self.K * (self.lambd + self.K + 1)) * np.sum([[JS_div(k1,k2) for k1 in range(self.K) if k1!=k2]for k2 in range(self.K)])
    
    def cluster_split(self):
        return
    
    def cluster_merge(self):
        return
    

