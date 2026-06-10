import numpy as np
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from preprocessing import *
from glob import glob
import gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from archetype_visualisation import *
from sklearn.cluster import k_means
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
pca = PCA(n_components=32)
x_pca = pca.fit_transform(x_flattened)
print(f"x_pca shape : {x_pca.shape}")

# Deep Learning Part -------------------------
class Clustering_end:
    def __init__(self, input, K):
        super().__init__()
        self.fc = nn.Linear(input, K, bias=False)
    
    def forward(self, x):
        return nn.functional.softmax(self.fc(x))

class Network:
    def __init__(self, model, mid_dim, K):
        self.main_nn = model
        self.end = Clustering_end(mid_dim, K) 
    def forward(self, x):
        torch.no_grad()
        y = nn.functional.normalize(self.main_nn(x))
        z = self.end(y)
        return z
# -----------------------------------------------

class DeepClusterer:
    """This class implements the Clustering With unkown number of clusters method proposed in the paper :\\
    **Deep Plug-and-Play Clustering with Unknown Number of Clusters** by *An Xiao et al.*"""
    def __init__(self, model, data, dist=np.linalg.norm, epochs=100):
        self.x = data
        self.N = len(self.x)
        self.model = model
        self.dist = dist
        self.labels = np.array([np.arange(self.N)]) # Initialization Everything in the same cluster
        self.lambd = 0.5
        self.K = len(self.labels)
        self.probs = np.array([[1/self.K for i in range(self.K)] for j in range(self.N)]).T
        self.n_eps = epochs
    
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
    
    def JS_div(self, p, q):
        """
        Calculates the JS divergence where p and q are probabilities of shape (N,)
        """
        def D_KL(P, Q):
            eps = 1e-14
            if len(P)!=len(Q):
                raise TabError("P and Q have not the same lenght.")
            return np.sum([P[i]*np.log((P(i)+eps)/(Q(i)+eps)) for i in range(len(P))])
        M = (p+q)/2
        return (D_KL(p, M)+D_KL(q, M))/2
    
    def JS_div_clusters(self, k1, k2):
        """Calculates the JS divergence between two clusters ``k1`` and ``k2``"""
        p = self.probs[:, k1]
        q = self.probs[:, k2]
        if p.sum()!=0:
            p = p / p.sum() 
        if q.sum()!=0:
            q = q / q.sum()
        return self.JS_div(p, q)

    def get_split_threshold(self):
        return self.lambd / (2*self.K * (self.lambd + self.K + 1)) * np.sum([[self.JS_div(k1,k2) for k1 in range(self.K) if k1!=k2]for k2 in range(self.K)])
    
    def get_merge_threshold(self, merged_probs):
        final_sum = 0
        for k1 in range(self.K-2):
            p = self.probs[:, k1]
            if p.sum()!=0:
                p = p / p.sum() 
            final_sum += self.JS_div(p, merged_probs)
        return (self.lambd)/(2*(self.K-1)*(self.lambd+self.K))*final_sum

    def cluster_split(self, k, k1, k2):
        labs = self.labels
        labs[k]=k1
        labs.insert(k+1, k2)
        self.labels = labs
        self.K += 1   
    
    def cluster_merge(self, k1,k2):
        i = min(k1, k2)
        j = max(k1, k2)
        labs = self.labels
        labs[i]= np.concatenate(labs[i], labs[j])
        labs.pop(j)
        self.labels = labs
        self.K -= 1
    
    def clusterize(self):
        for epoch in range(self.n_eps):
            # Apply A to training the network N with current number of cluster K*
            # ...
            for k in range(self.K):
                # Using A, split cluster into two.
                cluster_k = self.labels[k]
                k1, k2 = cluster_k[:len(cluster_k)//2], cluster_k[len(cluster_k)//2:]# à changer 
                J_div = self.JS_div(k1, k2)
                Ts = self.get_split_threshold()
                if J_div > Ts:
                    self.cluster_split(k, k1, k2)
            # Apply A to training the network N with current number of cluster K* and equ 7
            # ...
            all_divs = np.array([[self.JS_div(k1, k2) for k1 in range(self.K)]for k2 in range(self.K)])
            mask = ~np.eye(self.K, dtype=bool) 
            cand_k2, cand_k1 = np.unravel_index(np.argmin(all_divs[mask]), (self.K, self.K))
            J_div = self.JS_div(cand_k1, cand_k2)
            merged_probs = (self.probs[k1]+self.probs[k2])/2
            Tm = self.get_merge_threshold(merged_probs)
            if J_div < Tm:
                self.cluster_merge(cand_k1, cand_k2)
            # Apply A to training network N with K*.
            # ...
        return

clusterer=DeepClusterer(None,x)
centroids, clusters=clusterer.k_means(3)
print(centroids)
print(clusterer)
            
            
