# %%


# %%


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# %%
iris = load_iris()
X = iris.data
y = iris.target

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
n_clusters = 3  # Known from the Iris dataset
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = model.fit_predict(X_scaled)

# %%
silhouette = silhouette_score(X_scaled, clusters)
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)


# %%
print("Hierarchical Clustering Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Linkage method: ward")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# %%
unique, counts = np.unique(clusters, return_counts=True)
print("\nCluster Distribution:")
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} samples")

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# %%
plt.figure(figsize=(8, 5))
for cluster in range(n_clusters):
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], label=f'Cluster {cluster}')
plt.title('Hierarchical Clustering (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# %%
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.title('Dendrogram (Ward Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Hierarchical Clustering Results:
# Number of clusters: 3
# Linkage method: ward
# Silhouette Score: 0.4467
# Adjusted Rand Index (ARI): 0.6153
# Normalized Mutual Information (NMI): 0.6755

# Cluster Distribution:
# Cluster 0: 71 samples
# Cluster 1: 49 samples
# Cluster 2: 30 samples
