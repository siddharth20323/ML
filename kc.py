# %%
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a simple dataset
 

# %%

np.random.seed(42)
X = np.random.rand(100, 2)


# %%
X

# %%
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(X)

# %%
labels=kmeans.labels_

# %%
centers=kmeans.cluster_centers_

# %%

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()



